# -*- coding: utf-8 -*-
"""Geometria: resolucion y troceado temporal por familia de modelo."""

import math
import comfy.utils
from comfy_api.latest import io

# multiplo espacial exigido por cada familia
SPATIAL = {
    "flux2 (x32)": 32,
    "ltx-2.3 (x64)": 64,
    "wan-2.2 (x16)": 16,
    "sdxl (x8)": 8,
}
# regla temporal: n_frames = step*k + 1
TEMPORAL = {
    "ltx-2.3 (8n+1)": 8,
    "wan-2.2 (4n+1)": 4,
    "ninguna": 1,
}


def _snap(v, mult):
    return max(mult, int(round(v / mult)) * mult)


class SnapResolution(io.ComfyNode):
    """Escala una imagen a N megapixeles y alinea las dimensiones al multiplo que
    exige el modelo, en un solo nodo.

    Sustituye dos chapuzas que veniamos arrastrando: el round-trip VAE
    encode->decode que se usaba solo para obtener medidas legales, y la aritmetica
    mental de acordarse de si toca multiplo de 16, 32 o 64 segun el flujo.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CP_SnapResolution",
            display_name="Snap Resolution",
            category="character-pipeline/geometry",
            description="Escala a megapixeles y alinea al multiplo correcto del modelo.",
            inputs=[
                io.Image.Input("image"),
                io.Combo.Input("model_family", options=list(SPATIAL.keys()), default="flux2 (x32)"),
                io.Float.Input(
                    "megapixels", default=1.0, min=0.0, max=64.0, step=0.05,
                    tooltip="0 = conservar el tamano actual y solo alinear.",
                ),
                io.Combo.Input(
                    "upscale_method",
                    options=["lanczos", "bicubic", "bilinear", "area", "nearest-exact"],
                    default="lanczos",
                ),
            ],
            outputs=[
                io.Image.Output(display_name="image"),
                io.Int.Output(display_name="width"),
                io.Int.Output(display_name="height"),
            ],
        )

    @classmethod
    def execute(cls, image, model_family, megapixels, upscale_method) -> io.NodeOutput:
        mult = SPATIAL[model_family]
        _, h, w, _ = image.shape

        if megapixels > 0:
            scale = math.sqrt((megapixels * 1_000_000.0) / (w * h))
            w, h = w * scale, h * scale

        tw, th = _snap(w, mult), _snap(h, mult)

        samples = image.movedim(-1, 1)
        samples = comfy.utils.common_upscale(samples, tw, th, upscale_method, "disabled")
        out = samples.movedim(1, -1)
        return io.NodeOutput(out, tw, th)


class VideoChunk(io.ComfyNode):
    """Trocea un lote de frames en ventanas validas para el modelo, con solape,
    y devuelve la longitud REAL del trozo.

    Origen real: el flujo pedia 97 frames pero el clip tenia 49. El troceador
    devolvia 49 imagenes mientras el latente seguia generando 97, y el pipeline
    reventaba tres nodos mas abajo. La causa de fondo es que el numero de frames
    se fijaba a mano en dos sitios distintos.

    Aqui la longitud sale de UNA sola fuente: el trozo real, ya redondeado a la
    regla del modelo. Conectala tanto al latente de video como al de audio.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CP_VideoChunk",
            display_name="Video Chunk",
            category="character-pipeline/geometry",
            description="Trocea frames respetando la regla 8n+1 / 4n+1 y expone la longitud real.",
            inputs=[
                io.Image.Input("images"),
                io.Combo.Input("model_family", options=list(TEMPORAL.keys()), default="ltx-2.3 (8n+1)"),
                io.Int.Input("chunk_index", default=0, min=0, max=4096,
                             tooltip="0 = primer trozo. Sube de uno en uno para recorrer el clip."),
                io.Int.Input("chunk_length", default=97, min=1, max=4096,
                             tooltip="Longitud deseada. Se redondea hacia abajo a la regla del modelo."),
                io.Int.Input("overlap", default=8, min=0, max=256,
                             tooltip="Frames compartidos con el trozo anterior. Evita el salto en la costura."),
            ],
            outputs=[
                io.Image.Output(display_name="images"),
                io.Int.Output(display_name="length"),
                io.Int.Output(display_name="total_chunks"),
                io.Int.Output(display_name="start_index"),
            ],
        )

    @classmethod
    def execute(cls, images, model_family, chunk_index, chunk_length, overlap) -> io.NodeOutput:
        step = TEMPORAL[model_family]
        total = int(images.shape[0])

        def valid(n):
            if step <= 1:
                return max(1, n)
            if n < 1:
                return 1
            return max(1, ((n - 1) // step) * step + 1)

        want = valid(chunk_length)
        stride = max(1, want - overlap)
        total_chunks = max(1, math.ceil(max(0, total - overlap) / stride))

        start = min(chunk_index * stride, max(0, total - 1))
        piece = images[start:start + want]
        length = valid(int(piece.shape[0]))
        piece = piece[:length]

        if int(piece.shape[0]) != length:
            raise ValueError(
                f"[VideoChunk] Inconsistencia interna: {piece.shape[0]} frames vs longitud {length}."
            )
        return io.NodeOutput(piece, length, total_chunks, start)
