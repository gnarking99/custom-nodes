# -*- coding: utf-8 -*-
"""Medir si una etapa del flujo esta cambiando algo de verdad."""

import torch
import comfy.utils
from comfy_api.latest import io


class ABDiff(io.ComfyNode):
    """Compara dos imagenes y dice, con numeros, cuanto han cambiado.

    "No noto diferencia" es un sintoma, no un diagnostico. Puede significar tres
    cosas muy distintas:

    1. La etapa no se esta ejecutando (algo en bypass, algo mal conectado).
    2. Se ejecuta pero converge a la entrada (referencia demasiado fuerte,
       sigmas demasiado cortas, denoise demasiado bajo).
    3. Se ejecuta y cambia, pero el cambio es sutil y a tamano de reproduccion
       no se aprecia.

    Los tres se ven igual en el reproductor y se distinguen al instante con un
    numero. `mean_diff` por debajo de ~0.002 es practicamente un no-op; entre
    0.005 y 0.03 es un refinado normal; por encima de 0.08 el modelo esta
    reinterpretando, no restaurando.

    La salida `visual` amplifica la diferencia para que se vea DONDE cambia:
    si solo se ilumina el borde del encuadre, es reescalado; si se ilumina la
    textura de la piel y la ropa, esta refinando de verdad.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CP_ABDiff",
            display_name="A/B Diff",
            category="character-pipeline/guards",
            description="Mide cuanto ha cambiado una etapa. Convierte 'no noto diferencia' en un numero.",
            inputs=[
                io.Image.Input("a", tooltip="Antes (por ejemplo el clip original)."),
                io.Image.Input("b", tooltip="Despues (la salida de la etapa)."),
                io.Float.Input(
                    "amplificacion", default=12.0, min=1.0, max=60.0, step=1.0,
                    tooltip="Multiplica la diferencia para que sea visible. No afecta a los numeros.",
                ),
                io.Combo.Input(
                    "igualar_tamano", options=["b (el resultado)", "a (el original)"],
                    default="b (el resultado)",
                    tooltip="A que tamano se comparan. Se reescala la otra.",
                ),
            ],
            outputs=[
                io.Image.Output(display_name="visual"),
                io.Float.Output(display_name="mean_diff"),
                io.Float.Output(display_name="max_diff"),
                io.String.Output(display_name="veredicto"),
            ],
        )

    @classmethod
    def execute(cls, a, b, amplificacion, igualar_tamano) -> io.NodeOutput:
        if a is None or b is None or a.shape[0] == 0 or b.shape[0] == 0:
            raise ValueError("[ABDiff] Falta una de las dos imagenes.")

        n = min(int(a.shape[0]), int(b.shape[0]))
        a, b = a[:n].float(), b[:n].float()

        if igualar_tamano.startswith("b"):
            th, tw = int(b.shape[1]), int(b.shape[2])
            src, keep = a, b
        else:
            th, tw = int(a.shape[1]), int(a.shape[2])
            src, keep = b, a

        if (int(src.shape[1]), int(src.shape[2])) != (th, tw):
            src = comfy.utils.common_upscale(
                src.movedim(-1, 1), tw, th, "lanczos", "disabled").movedim(1, -1)

        aa, bb = (src, keep) if igualar_tamano.startswith("b") else (keep, src)

        d = (aa - bb).abs()
        mean_diff = float(d.mean().item())
        max_diff = float(d.max().item())
        visual = (d * float(amplificacion)).clamp(0.0, 1.0)

        if mean_diff < 0.002:
            v = ("NO-OP. La etapa practicamente no ha tocado la imagen.\n"
                 "  Revisa: algun nodo en bypass, la referencia mandando demasiado, "
                 "o las sigmas empezando demasiado abajo.")
        elif mean_diff < 0.005:
            v = ("CAMBIO MINIMO. Se ejecuta, pero apenas modifica.\n"
                 "  Sube el sigma inicial de la etapa o baja la fuerza de la referencia.")
        elif mean_diff < 0.03:
            v = "REFINADO NORMAL. Es el rango sano para una etapa de detalle."
        elif mean_diff < 0.08:
            v = "CAMBIO FUERTE. Mira que no se hayan movido rasgos del personaje."
        else:
            v = ("REINTERPRETACION. El modelo esta rehaciendo, no restaurando.\n"
                 "  Baja el sigma inicial de la etapa.")

        veredicto = (
            f"mean_diff {mean_diff:.5f}   max_diff {max_diff:.4f}   frames {n}\n"
            f"{aa.shape[2]}x{aa.shape[1]}\n\n{v}"
        )
        return io.NodeOutput(visual, mean_diff, max_diff, veredicto)
