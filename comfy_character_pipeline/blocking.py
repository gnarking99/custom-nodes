# -*- coding: utf-8 -*-
"""Blocking a animacion: las piezas que no da ComfyUI para convertir un blocking a saltos en animacion suave con LTX.

Medido el 05-11-2026 con un blocking de 25 poses en 77 f:
- Con las esperas del blocking como guia, los modelos copian los saltos. Hay que quitarlas: una imagen por pose,
  intermedios con RIFE y recolocados en los tiempos del blocking (CP_PosesSinEsperas).
- Fijar unas pocas poses pintadas con la LoRA del personaje acerca el render al personaje; 25 no mejoran a 6 y
  tardan el doble. Se pintan sobre el telon liso del playblast (un decorado inventado cambia en cada pose) y se
  componen sobre una placa unica.
- El mate deja alrededor del personaje un reborde claro (restos del telon) y LTX lo copia a todo el plano: donde ese
  reborde aclara la placa se vuelve a poner la placa (quitar_reborde, 10-11-2026).
"""

import json
import logging

import numpy as np
import torch
import torch.nn.functional as F
from comfy_api.latest import io
from scipy.ndimage import binary_dilation, binary_opening, gaussian_filter, label, shift

ALTO_TRABAJO = 540
TOLERANCIA_TELON = 14.0  # 0-255: lo que se aparta mas que esto del telon ajustado es personaje
RAMPA_MATE = 16.0  # 0-255: anchura de la transicion del mate
AREA_MINIMA = 40  # px a 540 de alto: motas fuera, la pieza pequena que vuela se queda

# Reborde claro del compositor. Las medidas en px estan tomadas con imagenes de 1088 de alto y se escalan con el alto real.
ALTO_REBORDE = 1088
UMBRAL_SILUETA = 30.0 / 255  # 0-1: lo que se aparta mas que esto del telon es personaje seguro (el mate empieza antes)
CIERRE_SILUETA = 2  # px: junta las rendijas finas de la silueta
AREA_SILUETA = 200  # px: las piezas sueltas mas pequenas no cuentan como silueta
ANCHO_REBORDE = 9  # px: anillo por fuera de la silueta donde se busca el reborde
MARGEN_REBORDE = 4.0 / 255  # 0-1: solo se corrige lo que aclara la placa mas que esto
SUAVIZADO_REBORDE = 1.2  # px: gaussiana de la mezcla con la placa


def leer_bloques(mapa, nodo):
    try:
        bloques = [tuple(b) for b in json.loads(mapa)["bloques"]]
    except (ValueError, KeyError, TypeError) as e:
        raise ValueError(f"[{nodo}] El mapa de poses no se puede leer: {e}") from e
    if not bloques:
        raise ValueError(f"[{nodo}] El mapa de poses esta vacio.")
    return bloques


def elegir_claves(bloques, n):
    """Indices de las poses que se fijan: la primera, la ultima y las mas cercanas a un reparto uniforme en el tiempo."""
    inicios = [a for a, _ in bloques]
    if n >= len(inicios):
        return list(range(len(inicios)))
    objetivos = np.linspace(0, inicios[-1], n)
    elegidas = {0, len(inicios) - 1}
    for o in objetivos[1:-1]:
        libres = [i for i in range(len(inicios)) if i not in elegidas]
        elegidas.add(min(libres, key=lambda i: abs(inicios[i] - o)))
    return sorted(elegidas)


def _a_trabajo(img):
    """[H,W,3] 0-1 -> numpy 0-255 a 540 de alto."""
    t = img.permute(2, 0, 1)[None].float()
    h, w = t.shape[-2:]
    if h != ALTO_TRABAJO:
        t = F.interpolate(t, size=(ALTO_TRABAJO, max(1, round(w * ALTO_TRABAJO / h))), mode="area")
    return t[0].permute(1, 2, 0).cpu().numpy() * 255.0


def ajustar_telon(img):
    """Telon suave (polinomio de grado 3 por canal) ajustado descartando lo que no encaja. Devuelve (telon, fraccion)."""
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    yy, xx = yy / h, xx / w
    base = np.stack([xx ** i * yy ** j for i in range(4) for j in range(4 - i)], axis=-1)
    paso = max(1, int(np.sqrt(h * w / 40000)))
    a = base[::paso, ::paso].reshape(-1, base.shape[-1])
    b = img[::paso, ::paso].reshape(-1, 3)
    usar = np.ones(len(a), bool)
    for _ in range(6):
        coef = np.linalg.lstsq(a[usar], b[usar], rcond=None)[0]
        usar = np.abs(a @ coef - b).max(axis=-1) < TOLERANCIA_TELON
    return base @ coef, float(usar.mean())


def mate_sobre_telon(img, ajuste=None):
    """Alfa 0-1 del personaje pintado sobre un telon liso, a la resolucion de img (0-255).
    ajuste = lo que devuelve ajustar_telon(img), si ya esta calculado."""
    telon, cubre = ajustar_telon(img) if ajuste is None else ajuste
    d = np.abs(img - telon).max(axis=-1)
    alfa = np.clip((d - TOLERANCIA_TELON) / RAMPA_MATE, 0, 1)
    duro = binary_opening(alfa > 0.5, iterations=1)
    et, n = label(duro)
    if n:
        areas = np.bincount(et.ravel())[1:]
        duro = np.isin(et, 1 + np.flatnonzero(areas >= AREA_MINIMA))
    alfa = alfa * binary_dilation(duro, iterations=3)
    return np.clip(gaussian_filter(alfa, 0.8), 0, 1), cubre


def sombra_contacto(alfa):
    """Oscurecimiento 0-1 bajo el personaje, corto y suave: solo para que la pose no parezca pegada. La pose 0 entra en
    LTX como arte, y una sombra ancha y oscura se quedaba en el suelo cuando el personaje ya se habia ido (05-11-2026)."""
    ancha = gaussian_filter(shift(alfa, (8, 0), order=1), 10)
    corta = gaussian_filter(shift(alfa, (3, 0), order=1), 3)
    return np.clip(0.12 * ancha + 0.22 * corta, 0, 0.3)


def _cubrir(img, alto, ancho):
    """[1,H,W,C] escalado para cubrir alto x ancho y recortado al centro."""
    h, w = img.shape[1:3]
    s = max(alto / h, ancho / w)
    t = F.interpolate(img.permute(0, 3, 1, 2), size=(max(alto, round(h * s)), max(ancho, round(w * s))), mode="bicubic", align_corners=False)
    y0, x0 = (t.shape[2] - alto) // 2, (t.shape[3] - ancho) // 2
    return t[:, :, y0:y0 + alto, x0:x0 + ancho].permute(0, 2, 3, 1).clamp(0, 1)


def medidas_reborde(alto):
    """Radios, area y suavizado del quitado de reborde para una imagen de este alto (medidos a 1088 de alto)."""
    e = alto / ALTO_REBORDE

    def px(v):
        return max(1, int(v * e + 0.5))

    return {"cierre": px(CIERRE_SILUETA), "area": AREA_SILUETA * e * e, "anillo": px(ANCHO_REBORDE), "suavizado": SUAVIZADO_REBORDE * e}


def quitar_reborde(compuesta, pintada, fondo, telon=None):
    """Quita el reborde claro que deja el mate alrededor del personaje: en un anillo fino por fuera de la silueta, donde la
    compuesta sale MAS CLARA que la placa, se vuelve a poner la placa. Lo que oscurece (sombras) no se toca, y del personaje
    solo el filo: la mezcla es suave y entra unos 3 px (a 1088 de alto) en la silueta.
    Todo numpy [H,W,3] en 0-1 y del mismo tamano: la compuesta, la pose pintada sobre su telon y la placa ya encajada.
    telon = el telon liso ajustado a la pintada (0-1); si falta, se ajusta aqui a resolucion completa.
    Devuelve (corregida, pixeles del anillo devueltos a la placa)."""
    from scipy.ndimage import binary_closing, binary_fill_holes

    if telon is None:
        telon = ajustar_telon(pintada * 255.0)[0] / 255.0
    m = medidas_reborde(compuesta.shape[0])
    lejos = np.abs(pintada - telon).max(axis=-1) > UMBRAL_SILUETA
    # Con margen: binary_closing erosiona contra el borde del cuadro y sin el se come el filo del personaje que lo toca.
    k = m["cierre"]
    silueta = binary_fill_holes(binary_closing(np.pad(lejos, k), iterations=k)[k:-k, k:-k])
    et, n = label(silueta)
    if n:
        areas = np.bincount(et.ravel())[1:]
        silueta = np.isin(et, 1 + np.flatnonzero(areas >= m["area"]))
    anillo = binary_dilation(silueta, iterations=m["anillo"]) & ~silueta
    aclara = anillo & (compuesta.mean(axis=-1) > fondo.mean(axis=-1) + MARGEN_REBORDE)
    corregidos = int(aclara.sum())
    if not corregidos:
        return compuesta, 0
    peso = gaussian_filter(aclara.astype(np.float32), m["suavizado"])[..., None]
    return compuesta * (1 - peso) + fondo * peso, corregidos


def componer(pintadas, placa, sin_halo=True):
    """Cada pose pintada sobre telon liso, recortada y puesta sobre la placa con sombra de contacto.
    Con sin_halo, el reborde claro que deja el mate alrededor del personaje se cambia por la placa (quitar_reborde)."""
    n, alto, ancho = pintadas.shape[:3]
    fondo = _cubrir(placa[:1].float(), alto, ancho)[0]
    fondo_np = fondo.cpu().numpy() if sin_halo else None
    salida, mascaras, corregidos = [], [], 0
    for i in range(n):
        trabajo = _a_trabajo(pintadas[i])
        ajuste = ajustar_telon(trabajo)
        alfa, cubre = mate_sobre_telon(trabajo, ajuste)
        if cubre < 0.5:
            logging.warning("[CP_PosesComponer] La pose %d no parece pintada sobre un telon liso (%.0f %% de fondo): el recorte puede fallar.", i, 100 * cubre)
        osc = sombra_contacto(alfa)
        a = F.interpolate(torch.from_numpy(alfa)[None, None].float(), size=(alto, ancho), mode="bilinear", align_corners=False)[0, 0]
        o = F.interpolate(torch.from_numpy(osc)[None, None].float(), size=(alto, ancho), mode="bilinear", align_corners=False)[0, 0]
        a, o = a.to(pintadas.device), o.to(pintadas.device)
        f = fondo.to(pintadas.device) * (1 - o)[..., None]
        compuesta = f * (1 - a)[..., None] + pintadas[i].float() * a[..., None]
        if sin_halo:
            # El telon del mate (ajustado a 540 de alto) subido al tamano de salida: es liso, el bilineal no lo cambia.
            telon = F.interpolate(torch.from_numpy(ajuste[0]).permute(2, 0, 1)[None].float(), size=(alto, ancho), mode="bilinear", align_corners=False)
            telon = telon[0].permute(1, 2, 0).numpy() / 255.0
            limpia, k = quitar_reborde(compuesta.cpu().numpy(), pintadas[i].float().cpu().numpy(), fondo_np, telon)
            compuesta = torch.from_numpy(limpia).to(pintadas.device)
            corregidos += k
        salida.append(compuesta)
        mascaras.append(a)
    if sin_halo:
        logging.info("[CP_PosesComponer] Reborde claro: %d pixeles devueltos a la placa en %d poses.", corregidos, n)
    return torch.stack(salida).clamp(0, 1), torch.stack(mascaras)


class PosesClaves(io.ComfyNode):
    """Elige las poses que se fijan en LTX y su fotograma en el video que se genera."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CP_PosesClaves",
            display_name="Poses · Claves para fijar",
            category="character-pipeline/poses",
            description="Unas pocas poses repartidas en el tiempo, con su fotograma en el video de salida.",
            inputs=[
                io.Image.Input("poses", tooltip="Una imagen por pose, de CP_PosesDetectar."),
                io.String.Input("mapa", force_input=True),
                io.Int.Input("n_claves", default=6, min=2, max=32, tooltip="6 bastan: 25 no mejoran y tardan el doble."),
                io.Int.Input("subpasos", default=2, min=1, max=4, tooltip="Fotogramas generados por fotograma del blocking (2 = se genera al doble de fps)."),
            ],
            outputs=[
                io.Image.Output(display_name="claves"),
                io.String.Output(display_name="fotogramas"),
                io.Int.Output(display_name="n_claves"),
            ],
        )

    @classmethod
    def execute(cls, poses, mapa, n_claves, subpasos) -> io.NodeOutput:
        bloques = leer_bloques(mapa, "CP_PosesClaves")
        if len(bloques) != len(poses):
            raise ValueError(f"[CP_PosesClaves] Llegaron {len(poses)} poses y el mapa tiene {len(bloques)}.")
        idx = elegir_claves(bloques, n_claves)
        fotogramas = [bloques[i][0] * subpasos for i in idx]
        logging.info("[CP_PosesClaves] Poses %s en los fotogramas %s.", idx, fotogramas)
        return io.NodeOutput(poses.index_select(0, torch.tensor(idx, device=poses.device)), json.dumps(fotogramas), len(idx))


class PosesComponer(io.ComfyNode):
    """Pone las poses pintadas sobre telon liso encima de una placa unica, con sombra de contacto."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CP_PosesComponer",
            display_name="Poses · Componer sobre la placa",
            category="character-pipeline/poses",
            description="Recorta cada pose pintada de su telon liso y la pone sobre la placa: el mismo decorado en todas.",
            inputs=[
                io.Image.Input("pintadas", tooltip="Poses pintadas sobre el telon liso del playblast, todas del mismo tamano."),
                io.Image.Input("placa", tooltip="El decorado vacio. Se escala para cubrir y se recorta al centro."),
                io.Boolean.Input("sin_halo", default=True, optional=True,
                                 tooltip="Quita el reborde claro (restos del telon) que queda alrededor del personaje: donde aclara la placa, vuelve la placa."),
            ],
            outputs=[
                io.Image.Output(display_name="compuestas"),
                io.Mask.Output(display_name="mascaras"),
            ],
        )

    @classmethod
    def execute(cls, pintadas, placa, sin_halo=True) -> io.NodeOutput:
        # Opcional: un grafo guardado antes de v0.13 no trae sin_halo y entonces vale el valor por defecto.
        salida, mascaras = componer(pintadas, placa, sin_halo=True if sin_halo is None else bool(sin_halo))
        return io.NodeOutput(salida, mascaras)


class LTXGuiasPoses(io.ComfyNode):
    """Fija cada pose en su fotograma con la guia IC-LoRA de LTX, en un bucle (el numero de poses no es fijo)."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CP_LTXGuiasPoses",
            display_name="LTX · Fijar poses",
            category="character-pipeline/ltx",
            description="LTXAddVideoICLoRAGuide para cada imagen en su fotograma, recorte al centro.",
            inputs=[
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Vae.Input("vae"),
                io.Latent.Input("latent"),
                io.Image.Input("imagenes"),
                io.String.Input("fotogramas", force_input=True, tooltip="Lista JSON con un fotograma por imagen."),
                io.Float.Input("latent_downscale_factor", default=1.0, min=1.0, max=8.0, tooltip="El de la IC-LoRA en el pase 1; 1 en el pase 2."),
                io.Float.Input("fuerza", default=1.0, min=0.0, max=1.0, step=0.05),
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive"),
                io.Conditioning.Output(display_name="negative"),
                io.Latent.Output(display_name="latent"),
            ],
        )

    @classmethod
    def execute(cls, positive, negative, vae, latent, imagenes, fotogramas, latent_downscale_factor, fuerza) -> io.NodeOutput:
        try:
            frames = [int(f) for f in json.loads(fotogramas)]
        except (ValueError, TypeError) as e:
            raise ValueError(f"[CP_LTXGuiasPoses] Los fotogramas no se pueden leer: {e}") from e
        if len(frames) != len(imagenes):
            raise ValueError(f"[CP_LTXGuiasPoses] Llegaron {len(imagenes)} imagenes y {len(frames)} fotogramas.")
        guia = _nodo_guia()
        for i, f in enumerate(frames):
            r = guia.execute(positive, negative, vae, latent, imagenes[i:i + 1], f, fuerza, latent_downscale_factor,
                             "center", False, 256, 64, False, 0)
            positive, negative, latent = r.args[:3] if hasattr(r, "args") else r[:3]
        return io.NodeOutput(positive, negative, latent)


def _nodo_guia():
    import nodes
    guia = nodes.NODE_CLASS_MAPPINGS.get("LTXAddVideoICLoRAGuide")
    if guia is None:
        raise RuntimeError("[CP_LTXGuiasPoses] Falta LTXAddVideoICLoRAGuide: instala ComfyUI-LTXVideo.")
    return guia
