# -*- coding: utf-8 -*-
"""Render por poses: texturas de un playblast sin tocar su movimiento.

Un playblast de animacion suele ir a dos: muchas parejas de fotogramas son la misma pose.
Se pinta UNA imagen por pose y luego se monta con la cadencia exacta del playblast, asi que el
tiempo no lo interpreta ningun modelo. Lo que no se mueve queda en una placa fija, sin parpadeo.

Las mascaras y las sombras se calculan a 540 de alto y se escalan al final: los umbrales significan
lo mismo con cualquier playblast y el montaje no se dispara con la resolucion de salida.
"""

import json
import logging

import numpy as np
import torch
import torch.nn.functional as F
from comfy_api.latest import io
from scipy.ndimage import binary_closing, binary_opening, distance_transform_edt, gaussian_filter, label

ALTO_TRABAJO = 540
ALTO_DETECCION = 270
TOPE_COBERTURA = 0.6  # por encima, casi todo cambia: camara movil, la placa no tiene sentido
RANGO_ESTATICO = 6.0
TOLERANCIA_TELON = 10.0
UMBRAL_REFLEJO = 12.0
MAX_POSES_PLACA = 40  # la mediana no gana nada con mas, y con 300 poses seria muy lenta
AREA_MINIMA = 60  # px a 960x540: quita motas, deja el trozo pequeno que vuela


def _a_escala(frames, alto):
    """[N,H,W,C] 0-1 -> numpy [N,h,w,C] 0-255 a ese alto, promediando (sin escalonado)."""
    n, h, w, _ = frames.shape
    ancho = max(1, int(round(w * alto / h)))
    salida = np.empty((n, alto, ancho, 3), np.float32)
    for i in range(n):
        x = frames[i:i + 1, ..., :3].movedim(-1, 1).float()
        x = F.interpolate(x, size=(alto, ancho), mode="area")
        salida[i] = x[0].movedim(0, -1).cpu().numpy() * 255.0
    return salida


def _ensanchar(mascara, radio):
    """Dilatacion redonda; con la distancia cuesta lo mismo con cualquier radio."""
    if not mascara.any():
        return mascara.copy()
    return distance_transform_edt(~mascara) <= radio


def _reducir(imagen, alto, ancho):
    """[H,W,C] 0-1 -> numpy [alto,ancho,3] 0-255."""
    x = imagen[..., :3].movedim(-1, 0)[None].float()
    return F.interpolate(x, size=(alto, ancho), mode="area")[0].movedim(0, -1).cpu().numpy() * 255.0


def _subir(mapa, alto, ancho):
    """Mapa [h,w] float -> tensor [alto,ancho] con interpolacion suave."""
    x = torch.from_numpy(np.asarray(mapa, np.float32))[None, None]
    return F.interpolate(x, size=(alto, ancho), mode="bicubic", align_corners=False)[0, 0].clamp(0.0, 1.0)


def detectar_bloques(frames, umbral):
    """Bloques [(ini, fin)] de fotogramas que son la misma pose. Cada fotograma se compara con el
    primero de su bloque, no con el anterior: un movimiento lento acaba abriendo pose nueva."""
    bloques, ini = [], 0
    for t in range(1, len(frames)):
        if float(np.abs(frames[t] - frames[ini]).mean()) > umbral:
            bloques.append((ini, t - 1))
            ini = t
    bloques.append((ini, len(frames) - 1))
    return bloques


def telon_liso(imagen, quieto):
    """Si lo que nunca se mueve es un telon suave (el degradado de Maya), su modelo; si no, None.
    Polinomio de grado 3 por canal, descartando en cada vuelta lo que no encaja: objetos quietos."""
    h, w = quieto.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    yy, xx = yy / h, xx / w
    base = np.stack([xx ** i * yy ** j for i in range(4) for j in range(4 - i)], axis=-1)
    paso = max(1, int(np.sqrt(h * w / 40000)))
    a = base[::paso, ::paso].reshape(-1, base.shape[-1])
    b = imagen[::paso, ::paso].reshape(-1, 3)
    usar = quieto[::paso, ::paso].ravel()
    for _ in range(4):
        if usar.sum() < 0.3 * usar.size:
            return None
        coef = np.linalg.lstsq(a[usar], b[usar], rcond=None)[0]
        resto = np.abs(a @ coef - b).max(axis=-1)
        usar = quieto[::paso, ::paso].ravel() & (resto < TOLERANCIA_TELON)
    if usar.sum() < 0.3 * usar.size or float(np.median(resto[usar])) > TOLERANCIA_TELON / 3:
        return None
    return base @ coef


def _valor_mas_repetido(valores):
    """[B,P,3] -> [P,3]: el valor que mas poses comparten en cada pixel (lo tapado cambia, lo quieto
    se repite), en grupos de 16 niveles por canal."""
    grupos = (valores / 16.0).astype(np.int32).clip(0, 15)
    codigos = grupos[..., 0] * 256 + grupos[..., 1] * 16 + grupos[..., 2]
    salida = np.empty(valores.shape[1:], np.float32)
    for p in range(valores.shape[1]):
        cuenta = np.bincount(codigos[:, p], minlength=4096)
        salida[p] = valores[codigos[:, p] == cuenta.argmax(), p].mean(axis=0)
    return salida


def placa_playblast(poses):
    """Fondo del playblast sin rastro de lo que se mueve, o None si casi nada esta quieto.

    La mediana no basta: donde el personaje pasa mas de la mitad del plano deja su rastro. Donde hay
    un telon liso, con que se vea unas pocas poses ya se sabe lo que es, aunque el personaje este ahi
    casi todo el plano. El resto (la bolsa tapada a ratos, un decorado) es el valor mas repetido."""
    rango = (poses.max(axis=0) - poses.min(axis=0)).max(axis=-1)
    quieto = rango < RANGO_ESTATICO
    if quieto.mean() < 1.0 - TOPE_COBERTURA:
        return None
    placa = poses.mean(axis=0)
    if quieto.all():
        return placa
    ys, xs = np.where(~quieto)
    valores = poses[:, ys, xs]  # [B,P,3]
    resuelto = np.zeros(len(ys), bool)
    telon = telon_liso(placa, quieto)
    if telon is not None:
        es_telon = np.abs(valores - telon[ys, xs]).max(axis=-1) < TOLERANCIA_TELON
        vistas = es_telon.sum(axis=0)
        resuelto = vistas >= max(2, int(np.ceil(0.05 * len(poses))))
        placa[ys[resuelto], xs[resuelto]] = ((valores * es_telon[..., None]).sum(axis=0)[resuelto]
                                             / vistas[resuelto][..., None])
    if not resuelto.all():
        placa[ys[~resuelto], xs[~resuelto]] = _valor_mas_repetido(valores[:, ~resuelto])
    return placa


def mascara_pose(pose, placa, umbral):
    """Lo que en esta pose se aparta de la placa: el personaje y lo que vuela, sin motas."""
    m = np.abs(pose - placa).max(axis=-1) > umbral
    m = binary_opening(binary_closing(m, iterations=2), iterations=1)
    lab, n = label(m)
    if n == 0:
        return m
    area = np.bincount(lab.ravel())
    area[0] = 0
    minimo = AREA_MINIMA * m.size / (960 * 540)
    return np.isin(lab, np.where(area > minimo)[0])


def zona_de_sombra(m):
    """Alrededor de la parte baja del personaje, que es donde puede tocar el suelo."""
    ys = np.where(m.any(axis=1))[0]
    if len(ys) == 0:
        return np.zeros(m.shape, np.float32)
    baja = m.copy()
    baja[: int(ys[0] + 0.66 * (ys[-1] - ys[0]))] = False
    return gaussian_filter(_ensanchar(baja, 35).astype(np.float32), 11.0)


def _mapa_sombra(pintada, placa, m):
    """Solo oscurece, solo cerca de los pies y solo con lo que la pintura dejo mas oscuro."""
    gris = gaussian_filter(pintada.mean(axis=-1), 4.5) / np.maximum(gaussian_filter(placa.mean(axis=-1), 4.5), 1.0)
    gris = np.clip(gris, 0.4, 1.0)
    gris = np.where(gris > 0.92, 1.0, gris)
    return 1.0 - (1.0 - gaussian_filter(gris, 2.0)) * zona_de_sombra(m)


def _mediana_libre(reducidas, mascaras):
    """Mediana de las poses donde no esta el personaje; donde nunca se vio el fondo, se rellena
    desde lo que rodea a ese hueco, abriendo el radio solo donde hace falta."""
    libres = ~np.stack([_ensanchar(m, 16) for m in mascaras])
    pila = np.where(libres[..., None], reducidas, np.nan)
    # nanmedian avisa en cada pixel sin datos; esos se rellenan justo despues.
    visto = libres.any(axis=0)
    mediana = np.zeros(reducidas.shape[1:], np.float32)
    mediana[visto] = np.nanmedian(pila[:, visto], axis=0)
    if not visto.all():
        pendiente = ~visto
        for sigma in (15.0, 45.0, 135.0, 405.0):
            peso = gaussian_filter(visto.astype(np.float32), sigma)
            fiable = pendiente & (peso > 0.05)
            for c in range(3):
                mediana[..., c][fiable] = gaussian_filter(mediana[..., c] * visto, sigma)[fiable] / peso[fiable]
            pendiente &= ~fiable
    return mediana


def _placa_pintada(pintadas, mascaras):
    """Placa a la resolucion de salida: la ultima pose, que conserva el detalle de lo quieto. Se
    cambia por la mediana de las poses donde tapa el personaje y, en la zona que este barre, donde
    esa pose se aparta de lo habitual: Klein pinta reflejos y sombras de un momento que no pueden
    quedarse en todo el plano."""
    n, h, w, _ = pintadas.shape
    ah, aw = mascaras[0].shape
    elegidas = np.unique(np.linspace(0, n - 1, min(n, MAX_POSES_PLACA)).round().astype(int))
    reducidas = np.stack([_reducir(pintadas[i], ah, aw) for i in elegidas])
    mediana = _mediana_libre(reducidas, [mascaras[i] for i in elegidas])
    ultima = reducidas[-1]
    aparte = np.abs(gaussian_filter(ultima, (2, 2, 0)) - gaussian_filter(mediana, (2, 2, 0))).max(axis=-1) > UMBRAL_REFLEJO
    barrido = _ensanchar(np.any(mascaras, axis=0), 60)
    tapado = _ensanchar(mascaras[-1], 12) | (aparte & barrido)
    a = _subir(gaussian_filter(tapado.astype(np.float32), 4.0), h, w)[..., None]
    fondo = F.interpolate(torch.from_numpy(mediana / 255.0).movedim(-1, 0)[None], size=(h, w), mode="bicubic", align_corners=False)
    return pintadas[-1, ..., :3].cpu() * (1.0 - a) + fondo[0].movedim(0, -1).clamp(0.0, 1.0) * a


def montar(pintadas, playblast, bloques, fijar_fondo=True, umbral_mascara=18.0, sombras=True):
    """Devuelve (fotogramas [N,H,W,3], placa o None, cobertura media de la mascara)."""
    n_poses, h, w, _ = pintadas.shape
    idx = torch.zeros(sum(b - a + 1 for a, b in bloques), dtype=torch.long)
    for i, (a, b) in enumerate(bloques):
        idx[a:b + 1] = i
    pintadas = pintadas[..., :3].cpu()
    if not fijar_fondo:
        return pintadas[idx], None, 0.0

    primeros = torch.tensor([a for a, _ in bloques])
    poses_pb = _a_escala(playblast.index_select(0, primeros.to(playblast.device)), ALTO_TRABAJO)
    placa_pb = placa_playblast(poses_pb)
    if placa_pb is None:
        logging.warning("[CP_PosesMontar] Casi nada esta quieto (camara movil): se monta sin placa fija.")
        return pintadas[idx], None, 1.0
    mascaras = [mascara_pose(p, placa_pb, umbral_mascara) for p in poses_pb]
    cobertura = float(np.mean([m.mean() for m in mascaras]))
    if cobertura > TOPE_COBERTURA:
        logging.warning("[CP_PosesMontar] La mascara cubre el %.0f %% (camara movil): se monta sin placa fija.", cobertura * 100)
        return pintadas[idx], None, cobertura

    placa = _placa_pintada(pintadas, mascaras)
    ah, aw = mascaras[0].shape
    placa_w = _reducir(placa, ah, aw)
    montadas = torch.empty_like(pintadas)
    for i in range(n_poses):
        al = _subir(gaussian_filter(_ensanchar(mascaras[i], 3).astype(np.float32), 0.8), h, w)[..., None]
        fondo = placa
        if sombras:
            fondo = placa * _subir(_mapa_sombra(_reducir(pintadas[i], ah, aw), placa_w, mascaras[i]), h, w)[..., None]
        montadas[i] = pintadas[i] * al + fondo * (1.0 - al)
    return montadas[idx], placa, cobertura


def leer_mapa(mapa, n_poses, n_fotogramas):
    try:
        bloques = [tuple(b) for b in json.loads(mapa)["bloques"]]
    except (ValueError, KeyError, TypeError) as e:
        raise ValueError(f"[CP_PosesMontar] El mapa de poses no se puede leer: {e}") from e
    if len(bloques) != n_poses:
        raise ValueError(f"[CP_PosesMontar] Llegaron {n_poses} poses pintadas y el mapa tiene {len(bloques)}.")
    if not bloques or bloques[0][0] != 0 or bloques[-1][1] != n_fotogramas - 1:
        raise ValueError(f"[CP_PosesMontar] El mapa no cubre los {n_fotogramas} fotogramas del playblast.")
    return bloques


def factor_intermedios(bloques):
    """Intermedios por salto: 12 reparte exacto saltos de 2, 3, 4, 6 y 12 fotogramas. Con saltos mayores
    sube hasta el tope del interpolador; por encima de 16 alguna espera se queda sin mover."""
    inicios = [a for a, _ in bloques]
    mayor = max((b - a for a, b in zip(inicios, inicios[1:])), default=1)
    return 12 if mayor <= 12 else min(16, mayor)


def indices_sin_esperas(bloques, factor, subpasos=1):
    """Para cada fotograma de salida, que fotograma interpolado le toca: cada pose cae en su fotograma original y
    el camino hasta la siguiente se reparte entre medias. La ultima pose se mantiene. Con subpasos=2 sale al doble
    de fps (medio fotograma del blocking por paso): LTX mantiene solido un giro rapido que a 24 fps emborrona."""
    inicios = [a for a, _ in bloques]
    total = bloques[-1][1] + 1
    salida = []
    k = 0
    for s in range((total - 1) * subpasos + 1):
        t = s / subpasos
        while k + 1 < len(inicios) and inicios[k + 1] <= t:
            k += 1
        if k == len(inicios) - 1:
            salida.append(factor * k)
        else:
            salida.append(factor * k + round(factor * (t - inicios[k]) / (inicios[k + 1] - inicios[k])))
    return salida


def fotogramas_ltx(n):
    """LTX trabaja con 8k+1 fotogramas y redondea hacia abajo: se pide el primero que no corte el plano."""
    return ((n - 1 + 7) // 8) * 8 + 1


class PosesDetectar(io.ComfyNode):
    """Saca del playblast una imagen por pose y el mapa para volver a montarlo con su cadencia."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CP_PosesDetectar",
            display_name="Poses · Detectar",
            category="character-pipeline/poses",
            description="Una imagen por pose del playblast y el mapa de su cadencia.",
            inputs=[
                io.Image.Input("imagenes"),
                io.Float.Input(
                    "umbral", default=0.5, min=0.05, max=20.0, step=0.05,
                    tooltip="Diferencia media (0-255) con la que se abre pose nueva. Mas bajo, mas poses.",
                ),
                io.Int.Input(
                    "max_poses", default=300, min=1, max=4096,
                    tooltip="Tope de seguridad: cada pose es una imagen que se paga.",
                ),
            ],
            outputs=[
                io.Image.Output(display_name="poses"),
                io.String.Output(display_name="mapa"),
                io.Int.Output(display_name="n_poses"),
                io.Int.Output(display_name="factor"),
            ],
        )

    @classmethod
    def execute(cls, imagenes, umbral, max_poses) -> io.NodeOutput:
        if imagenes is None or imagenes.numel() == 0:
            raise ValueError("[CP_PosesDetectar] El playblast esta vacio.")
        bloques = detectar_bloques(_a_escala(imagenes, ALTO_DETECCION), umbral)
        if len(bloques) > max_poses:
            raise ValueError(
                f"[CP_PosesDetectar] Salen {len(bloques)} poses y el tope es {max_poses}. "
                "Si el playblast lleva HUD o ruido, exportalo limpio; si es asi, sube el tope."
            )
        logging.info("[CP_PosesDetectar] %d poses en %d fotogramas.", len(bloques), len(imagenes))
        primeros = torch.tensor([a for a, _ in bloques], device=imagenes.device)
        mapa = json.dumps({"bloques": bloques})
        return io.NodeOutput(imagenes.index_select(0, primeros), mapa, len(bloques), factor_intermedios(bloques))


class PosesMontar(io.ComfyNode):
    """Monta las poses pintadas con la cadencia del playblast, sobre una placa fija."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CP_PosesMontar",
            display_name="Poses · Montar",
            category="character-pipeline/poses",
            description="Poses pintadas + cadencia del playblast, con el fondo fijo.",
            inputs=[
                io.Image.Input("pintadas"),
                io.Image.Input("playblast"),
                io.String.Input("mapa", force_input=True),
                io.Boolean.Input(
                    "fijar_fondo", default=True,
                    tooltip="Solo con camara fija. Con camara movil se apaga sola y se montan las poses tal cual.",
                ),
                io.Boolean.Input("sombras", default=True, tooltip="Conserva las sombras de contacto de la pintura."),
                io.Float.Input(
                    "umbral_mascara", default=18.0, min=1.0, max=128.0, step=1.0,
                    tooltip="Diferencia (0-255) con el fondo a partir de la que un pixel se mueve.",
                ),
            ],
            outputs=[
                io.Image.Output(display_name="imagenes"),
            ],
        )

    @classmethod
    def execute(cls, pintadas, playblast, mapa, fijar_fondo, sombras, umbral_mascara) -> io.NodeOutput:
        bloques = leer_mapa(mapa, len(pintadas), len(playblast))
        imagenes, _, cobertura = montar(pintadas, playblast, bloques, fijar_fondo, umbral_mascara, sombras)
        logging.info("[CP_PosesMontar] %d poses, %d fotogramas, mascara media %.1f %%.", len(pintadas), len(imagenes), cobertura * 100)
        return io.NodeOutput(imagenes)


class PosesSinEsperas(io.ComfyNode):
    """Coloca los intermedios en los tiempos del blocking: el guia de movimiento sin esperas."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CP_PosesSinEsperas",
            display_name="Poses · Guia sin esperas",
            category="character-pipeline/poses",
            description="Recoloca una secuencia interpolada entre poses en los tiempos del blocking y la rellena a 8k+1 para LTX.",
            inputs=[
                io.Image.Input("intermedios", tooltip="Las poses de CP_PosesDetectar ya interpoladas con su factor."),
                io.String.Input("mapa", force_input=True),
                io.Int.Input("factor", default=12, min=2, max=16, force_input=True),
                io.Int.Input("subpasos", default=2, min=1, max=4, tooltip="2 = guia al doble de fps: el giro rapido sale solido."),
                io.Float.Input("fps", default=24.0, min=1.0, max=120.0, tooltip="Los del blocking; sale multiplicado por subpasos."),
            ],
            outputs=[
                io.Image.Output(display_name="guia"),
                io.Int.Output(display_name="fotogramas", tooltip="Los del video generado antes de rellenar: (blocking - 1) x subpasos + 1."),
                io.Int.Output(display_name="fotogramas_ltx"),
                io.Float.Output(display_name="fps_guia"),
            ],
        )

    @classmethod
    def execute(cls, intermedios, mapa, factor, subpasos=1, fps=24.0) -> io.NodeOutput:
        try:
            bloques = [tuple(b) for b in json.loads(mapa)["bloques"]]
        except (ValueError, KeyError, TypeError) as e:
            raise ValueError(f"[CP_PosesSinEsperas] El mapa de poses no se puede leer: {e}") from e
        esperados = factor * (len(bloques) - 1) + 1
        if len(intermedios) != esperados:
            raise ValueError(f"[CP_PosesSinEsperas] Llegaron {len(intermedios)} intermedios y con {len(bloques)} poses a x{factor} son {esperados}.")
        indices = indices_sin_esperas(bloques, factor, subpasos)
        n_ltx = fotogramas_ltx(len(indices))
        # La ultima pose se repite: LTX necesita 8k+1 y al final se recorta a los fotogramas del blocking.
        indices = indices + [indices[-1]] * (n_ltx - len(indices))
        guia = intermedios.index_select(0, torch.tensor(indices, device=intermedios.device))
        return io.NodeOutput(guia, len(indices), n_ltx, float(fps) * subpasos)
