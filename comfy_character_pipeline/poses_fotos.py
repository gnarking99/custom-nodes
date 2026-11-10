# -*- coding: utf-8 -*-
"""Poses desde fotos: el blocking llega como una imagen por pose, en orden y sin tiempos.

Sustituye a la carga del video + CP_PosesDetectar: da lo mismo que ese nodo (el lote "poses" y el "mapa"
{"bloques": [[primer_fotograma, ultimo_fotograma], ...]}), asi que CP_LentaPlan y lo que viene detras no cambian.
Como las fotos no traen tiempos, cada pose dura fotogramas_por_pose y la ultima se queda 1 + espera_final.
La salida "transicion" (= fotogramas_por_pose) va al transicion_max de CP_LentaPlan: el movimiento ocupa todo el
salto y no queda una espera quieta en medio.

Una casilla de imagen por pose (io.Combo con subida de imagen) porque es lo que sabe rellenar un panel de campos: un
campo de imagen por archivo. Las casillas que no se usan llevan MARCADOR_VACIO y no cuentan; los huecos entre
casillas rellenas tampoco (casillas 1, 2 y 5 = 3 poses).
"""

import hashlib
import inspect
import json
import logging
import os
import re

import numpy as np
import torch
import torch.nn.functional as F
from comfy_api.latest import io

NODO = "CP_PosesDesdeFotos"
CASILLAS = 24  # cuantas casillas de imagen tiene el nodo; es lo unico que hay que tocar para tener mas
FPS = 24.0  # con fotos no hay video del que sacar los fps: se fijan y la tarjeta los lleva a la salida de video
GRIS_ALFA = 0.5  # lo transparente de una foto con alfa se pone sobre este gris liso (0-1)
TOLERANCIA_PROPORCION = 0.01  # diferencia relativa de proporcion desde la que se avisa de que se recorta
# Tope de fotogramas_por_pose. Con 32, cada salto lleva 32 lentos (lo validado en GPU; con mas saldrian 40, 48...) y
# con las 24 casillas llenas y la espera final al maximo salen 23 x 32 + 1 + 240 = 977 fotogramas: con ritmo 1 nunca se
# pasa de los 1000 fotogramas de salida que CP_LentaPlan admite como mucho cuando hay ritmo (por memoria).
MAX_FOTOGRAMAS_POR_POSE = 32
MAX_ESPERA_FINAL = 240
# Lado largo del lote de poses como mucho: la primera foto fija el tamano de todo el lote y una de 6000 px haria un
# lote de varios GB (24 x 6000 x 4000 x 3 en float32 = 7 GB). Por encima se reduce, sin deformar, y se avisa.
MAX_LADO = 2048

# Valor de una casilla sin rellenar. Por que este:
# - empieza por PON-AQUI: los paneles de campos tratan como vacio lo que empieza por "PON-AQUI" o "example." (el mismo
#   convenio que el resto de marcadores del pack), asi que la casilla sale como "sin rellenar" y no como un archivo;
# - NO acaba en una extension de imagen o video: lo que lleva el prompt a la nube solo sube los valores que acaban en
#   .png/.jpg/.webp/.mp4... y existen en la carpeta de entrada; sin extension ni siquiera se busca como archivo;
# - sin extension tampoco puede ser una imagen de la carpeta de entrada (la lista de la casilla solo lleva imagenes), y
#   el nodo lo reconoce como vacio antes de mirar el disco.
MARCADOR_VACIO = "PON-AQUI-UNA-POSE"
VACIO = re.compile(r"^(example\.|PON-AQUI)", re.IGNORECASE)
ORDENES = ["casilla", "nombre"]


def nombre_casilla(i):
    """1 -> 'pose_01'. El ancho crece solo si un dia hay mas de 99 casillas."""
    return f"pose_{i:0{max(2, len(str(CASILLAS)))}d}"


NOMBRES_CASILLAS = [nombre_casilla(i) for i in range(1, CASILLAS + 1)]


# ---------------------------------------------------------------- piezas puras

def es_vacia(valor):
    """Casilla sin rellenar: nada, solo espacios, o un nombre (sin carpeta) que empieza por PON-AQUI o example."""
    if valor is None:
        return True
    s = str(valor).strip()
    if not s:
        return True
    return bool(VACIO.match(s.replace("\\", "/").split("/")[-1]))


def clave_natural(nombre):
    """pose_2 antes que pose_10, sin distinguir mayusculas; la carpeta no cuenta (la de la anotacion [input] tampoco)."""
    base = re.sub(r"\s*\[(input|output|temp)\]$", "", str(nombre)).replace("\\", "/").split("/")[-1].lower()
    return [(0, int(t), "") if t.isdigit() else (1, 0, t) for t in re.split(r"(\d+)", base) if t != ""]


def casillas_rellenas(valores, orden="casilla"):
    """{nombre_casilla: valor} -> [(casilla, valor)] de las rellenas, en el orden pedido.
    'casilla': pose_01, pose_02...; 'nombre': orden natural del nombre del archivo (si empatan, el de las casillas)."""
    if orden not in ORDENES:
        raise ValueError(f"[{NODO}] Orden desconocido: {orden!r}. Tiene que ser {' o '.join(ORDENES)}.")
    rellenas = [(c, str(valores[c]).strip()) for c in NOMBRES_CASILLAS if c in valores and not es_vacia(valores[c])]
    if orden == "nombre":
        rellenas.sort(key=lambda cv: clave_natural(cv[1]))  # sort es estable: los empates quedan por casilla
    return rellenas


def mapa_bloques(n_poses, fotogramas_por_pose, espera_final=0):
    """Bloques [[ini, fin]] seguidos desde 0: cada pose dura fotogramas_por_pose y la ultima 1 + espera_final."""
    d, extra = int(fotogramas_por_pose), int(espera_final)
    if n_poses < 2:
        raise ValueError(f"[{NODO}] Sube al menos dos poses: con {n_poses} no hay nada que animar.")
    if d < 1 or extra < 0:
        raise ValueError(f"[{NODO}] fotogramas_por_pose tiene que ser al menos 1 y espera_final no puede ser negativa.")
    bloques = [[i * d, i * d + d - 1] for i in range(n_poses - 1)]
    ultimo = (n_poses - 1) * d
    bloques.append([ultimo, ultimo + extra])
    return bloques


def total_fotogramas(n_poses, fotogramas_por_pose, espera_final=0):
    return (int(n_poses) - 1) * int(fotogramas_por_pose) + 1 + int(espera_final)


def a_tensor(img, que="la imagen"):
    """PIL -> tensor [1,H,W,3] float32 0-1, como LoadImage: orientacion EXIF aplicada y RGB. Si trae alfa, lo
    transparente se aplana sobre GRIS_ALFA (y se avisa). Devuelve (tensor, tenia_alfa)."""
    from PIL import ImageOps

    img = ImageOps.exif_transpose(img)
    ancho, alto = img.size
    if ancho == 0 or alto == 0:
        raise ValueError(f"[{NODO}] {que} mide {ancho}x{alto} pixeles: esta vacia.")
    if img.mode.startswith("I") or img.mode == "F":  # gris de 16 bits o flotante
        x = np.asarray(img, np.float32)
        escala = 65535.0 if (img.mode.startswith("I;16") or float(x.max(initial=0)) > 255.0) else 255.0
        if img.mode == "F":
            escala = 1.0 if float(x.max(initial=0)) <= 1.0 else 255.0
        x = np.clip(x / escala, 0.0, 1.0)
        return torch.from_numpy(np.ascontiguousarray(np.repeat(x[..., None], 3, axis=-1)))[None], False
    con_alfa = "A" in img.getbands() or "transparency" in img.info
    if con_alfa:
        rgba = np.asarray(img.convert("RGBA"), np.float32) / 255.0
        a = rgba[..., 3:4]
        con_alfa = bool((a < 1.0).any())
        rgb = rgba[..., :3] * a + GRIS_ALFA * (1.0 - a)
    else:
        rgb = np.asarray(img.convert("RGB"), np.float32) / 255.0
    return torch.from_numpy(np.ascontiguousarray(rgb, dtype=np.float32))[None], con_alfa


def cubrir_y_recortar(x, alto, ancho):
    """[1,h,w,3] -> [1,alto,ancho,3]: escala para cubrir (sin deformar) y recorta al centro."""
    _, h, w, _ = x.shape
    if (h, w) == (alto, ancho):
        return x
    escala = max(ancho / w, alto / h)
    nw, nh = max(ancho, round(w * escala)), max(alto, round(h * escala))
    y = F.interpolate(x.movedim(-1, 1), size=(nh, nw), mode="bicubic", antialias=True, align_corners=False)
    t, l = (nh - alto) // 2, (nw - ancho) // 2
    return y[:, :, t:t + alto, l:l + ancho].clamp(0.0, 1.0).movedim(1, -1)


def tamano_lote(alto, ancho, max_lado=MAX_LADO):
    """Tamano del lote a partir del de la primera foto: el mismo, o reducido sin deformar si el lado largo pasa de
    max_lado. Devuelve (alto, ancho, aviso o None)."""
    largo = max(alto, ancho)
    if largo <= max_lado:
        return alto, ancho, None
    escala = max_lado / largo
    nuevo_alto, nuevo_ancho = max(1, round(alto * escala)), max(1, round(ancho * escala))
    aviso = (f"la primera foto mide {ancho}x{alto} y el lado largo pasa de {max_lado} px: todo el lote se reduce a "
             f"{nuevo_ancho}x{nuevo_alto}, sin deformar.")
    return nuevo_alto, nuevo_ancho, aviso


def ajustar(x, alto, ancho, et):
    """[1,h,w,3] al tamano del lote. Devuelve (tensor, aviso o None si la proporcion es la del lote)."""
    _, h, w, _ = x.shape
    aviso = None
    if abs((w / h) / (ancho / alto) - 1.0) > TOLERANCIA_PROPORCION:
        aviso = f"{et} es {w}x{h} y el lote {ancho}x{alto}: otra proporcion, se recorta al centro."
    return cubrir_y_recortar(x.float(), alto, ancho), aviso


def igualar(imagenes, etiquetas=None, max_lado=MAX_LADO):
    """Lista de [1,h,w,3] -> lote [N,H,W,3] al tamano de la PRIMERA (reducido si su lado largo pasa de max_lado).
    Devuelve (lote, avisos)."""
    _, alto, ancho, _ = imagenes[0].shape
    alto, ancho, aviso = tamano_lote(alto, ancho, max_lado)
    etiquetas = etiquetas or [f"imagen {i + 1}" for i in range(len(imagenes))]
    salida, avisos = [], [aviso] if aviso else []
    for x, et in zip(imagenes, etiquetas):
        y, aviso = ajustar(x, alto, ancho, et)
        salida.append(y)
        if aviso:
            avisos.append(aviso)
    return torch.cat(salida, dim=0), avisos


def parecen_marcador(valores):
    """Casillas [(casilla, valor)] que cuentan como vacias por el nombre (empieza por example. o PON-AQUI) sin ser el
    marcador del nodo: si hay un archivo de verdad con ese nombre, el artista lo subio y se descarta (se avisa)."""
    return [
        (c, str(valores[c]).strip()) for c in NOMBRES_CASILLAS
        if c in valores and valores[c] is not None and str(valores[c]).strip()
        and str(valores[c]).strip() != MARCADOR_VACIO and es_vacia(valores[c])
    ]


def sin_ruta(texto, ruta, valor):
    """El texto de un error sin la ruta del trabajador: la ruta (y la real, si es otra) se cambia por el valor."""
    texto = str(texto)
    for r in sorted({ruta, os.path.realpath(ruta), os.path.abspath(ruta)}, key=len, reverse=True):
        if r:
            texto = texto.replace(r, valor)
    return texto


def huella_archivo(ruta):
    """sha256 del contenido, como LoadImage: otra foto subida con el mismo nombre cambia la huella."""
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for trozo in iter(lambda: f.read(1 << 20), b""):
            h.update(trozo)
    return h.hexdigest()


# ---------------------------------------------------------------- carpeta de entrada

def _imagenes_de_entrada():
    """Las imagenes de la carpeta de entrada, como las lista LoadImage (solo el primer nivel). Un valor con carpeta
    (lo que se sube a una subcarpeta) no sale en la lista pero vale igual: validate_inputs no exige la lista."""
    import folder_paths

    carpeta = folder_paths.get_input_directory()
    try:
        archivos = [f for f in os.listdir(carpeta) if os.path.isfile(os.path.join(carpeta, f))]
    except OSError:
        return []
    return sorted(folder_paths.filter_files_content_types(archivos, ["image"]))


def ruta_de(valor):
    """Ruta de un valor de casilla en la carpeta de entrada (admite 'carpeta/nombre.png' y la anotacion [input]),
    o None si no hay un archivo ahi o el nombre se sale de la carpeta."""
    import folder_paths

    try:
        if not folder_paths.exists_annotated_filepath(valor):
            return None
        ruta = folder_paths.get_annotated_filepath(valor)
    except ValueError:
        return None
    return ruta if os.path.isfile(ruta) else None


def _con_firma_de_casillas(funcion):
    """El motor solo se salta la comprobacion de "Value not in list" (y la de min/max) de las entradas que nombra la
    firma de validate_inputs; con **kwargs se la saltaria a TODAS. Aqui la firma nombra solo las casillas, asi que
    los enteros y el orden se siguen comprobando, y la lista de casillas sale de CASILLAS."""
    P = inspect.Parameter
    funcion.__signature__ = inspect.Signature(
        [P("cls", P.POSITIONAL_OR_KEYWORD)] + [P(c, P.POSITIONAL_OR_KEYWORD, default=None) for c in NOMBRES_CASILLAS]
    )
    return funcion


def _validate_inputs(cls, **casillas):
    # Siempre True: el marcador y una foto recien subida (que no estaba en la lista al cargar el nodo) tienen que
    # pasar. La foto que falte se dice en execute, con su casilla: un texto devuelto aqui el motor lo repetiria una
    # vez por cada casilla.
    return True


# ---------------------------------------------------------------- el nodo

class PosesDesdeFotos(io.ComfyNode):
    """Una foto por pose -> el lote de poses y el mapa de tiempos que esperan CP_LentaPlan y CP_LentaElegir."""

    @classmethod
    def define_schema(cls):
        opciones = [MARCADOR_VACIO] + _imagenes_de_entrada()
        casillas = [
            io.Combo.Input(
                c, options=opciones, default=MARCADOR_VACIO, upload=io.UploadType.image, optional=True,
                tooltip="Una pose del blocking. Las casillas sin rellenar no cuentan.",
            )
            for c in NOMBRES_CASILLAS
        ]
        return io.Schema(
            node_id=NODO,
            display_name="Poses · Desde fotos",
            category="character-pipeline/poses",
            description="Blocking en fotos (una imagen por pose, en orden): el lote de poses y el mapa de tiempos de "
                        "CP_PosesDetectar, con cada pose durando fotogramas_por_pose.",
            inputs=[
                *casillas,
                io.Int.Input(
                    "fotogramas_por_pose", default=12, min=2, max=MAX_FOTOGRAMAS_POR_POSE,
                    tooltip="Fotogramas de cada salto entre una pose y la siguiente (a 24 fps, 12 = medio segundo). "
                            f"Hasta {MAX_FOTOGRAMAS_POR_POSE}.",
                ),
                io.Int.Input(
                    "espera_final", default=0, min=0, max=MAX_ESPERA_FINAL,
                    tooltip="Fotogramas que se queda quieta la ultima pose al final, ademas del suyo.",
                ),
                io.Combo.Input(
                    "orden", options=ORDENES, default="casilla",
                    tooltip="casilla: pose_01, pose_02... nombre: por el nombre del archivo (pose_2 antes que pose_10).",
                ),
            ],
            outputs=[
                io.Image.Output(display_name="poses"),
                io.String.Output(display_name="mapa"),
                io.Int.Output(display_name="n_poses"),
                io.Int.Output(display_name="transicion", tooltip="= fotogramas_por_pose: para el transicion_max de CP_LentaPlan."),
                io.Int.Output(display_name="fotogramas", tooltip="Los de la animacion: (poses - 1) x fotogramas_por_pose + 1 + espera_final."),
                io.Float.Output(display_name="fps"),
            ],
        )

    validate_inputs = classmethod(_con_firma_de_casillas(_validate_inputs))

    @classmethod
    def fingerprint_inputs(cls, **kwargs):
        partes = []
        for c in NOMBRES_CASILLAS:
            v = kwargs.get(c)
            ruta = None if es_vacia(v) else ruta_de(str(v).strip())
            try:
                huella = huella_archivo(ruta) if ruta else "-"
            except OSError:
                huella = "?"
            partes.append(f"{c}={v}:{huella}")
        return "|".join(partes)

    @classmethod
    def execute(cls, fotogramas_por_pose=12, espera_final=0, orden="casilla", **casillas) -> io.NodeOutput:
        from PIL import Image

        for c, valor in parecen_marcador(casillas):
            if ruta_de(valor) is not None:
                logging.warning(
                    "[%s] %s: la foto %r no se usa: los nombres que empiezan por 'example.' o 'PON-AQUI' cuentan como "
                    "casilla vacia. Renombrala y vuelve a subirla si es una pose.", NODO, c, valor,
                )
        rellenas = casillas_rellenas(casillas, orden)
        if len(rellenas) < 2:
            raise ValueError(f"[{NODO}] Sube al menos dos poses: hay {len(rellenas)} casilla(s) con foto.")
        lote, alto, ancho = [], None, None
        for c, valor in rellenas:
            ruta = ruta_de(valor)
            if ruta is None:
                raise FileNotFoundError(f"[{NODO}] {c}: no se encuentra la foto {valor!r} en la carpeta de entrada. Vuelve a subirla.")
            et = f"{c} ({valor})"
            try:
                img = Image.open(ruta)
                img.load()
            except Exception as e:
                detalle = sin_ruta(e, ruta, valor)
                raise ValueError(f"[{NODO}] {c}: no se puede leer {valor!r} como imagen ({type(e).__name__}: {detalle}).") from None
            if getattr(img, "n_frames", 1) > 1:
                logging.warning("[%s] %s tiene %d fotogramas: se usa el primero.", NODO, et, img.n_frames)
            x, tenia_alfa = a_tensor(img, et)
            if tenia_alfa:
                logging.warning("[%s] %s tiene transparencia: se aplana sobre gris liso (%.2f).", NODO, et, GRIS_ALFA)
            img.close()
            # Cada foto se lleva al tamano del lote nada mas leerla: asi nunca hay 24 fotos enormes a la vez en memoria.
            if alto is None:
                alto, ancho, aviso = tamano_lote(x.shape[1], x.shape[2])
                if aviso:
                    logging.warning("[%s] %s: %s", NODO, et, aviso)
            x, aviso = ajustar(x, alto, ancho, et)
            if aviso:
                logging.warning("[%s] %s", NODO, aviso)
            lote.append(x)
        poses = torch.cat(lote, dim=0)
        n = len(lote)
        bloques = mapa_bloques(n, fotogramas_por_pose, espera_final)
        total = total_fotogramas(n, fotogramas_por_pose, espera_final)
        logging.info(
            "[%s] %d poses a %dx%d (orden por %s), %d fotogramas por pose, %d en total.",
            NODO, n, poses.shape[2], poses.shape[1], orden, int(fotogramas_por_pose), total,
        )
        return io.NodeOutput(poses, json.dumps({"bloques": bloques}), n, int(fotogramas_por_pose), total, FPS)
