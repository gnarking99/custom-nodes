# -*- coding: utf-8 -*-
"""Guardas: nodos que convierten fallos silenciosos en fallos ruidosos."""

import torch
from comfy_api.latest import io


class MaskGuard(io.ComfyNode):
    """Detiene el desastre clasico: una mascara vacia que aguas abajo se invierte
    y acaba pegando la imagen original encima de todo el trabajo.

    Origen real: `LoadImage` devuelve una mascara 64x64 de ceros cuando el PNG no
    tiene canal alfa. Esa mascara pasa por `InvertMask` y sale toda blanca, y el
    `ImageCompositeMasked` tapa el resultado al 100%. El flujo no falla: entrega
    silenciosamente la imagen de entrada.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CP_MaskGuard",
            display_name="Mask Guard",
            category="character-pipeline/guards",
            description=(
                "Comprueba que una mascara tiene contenido util antes de dejarla "
                "pasar. Evita el fallo silencioso de mascara vacia."
            ),
            inputs=[
                io.Mask.Input("mask"),
                io.Combo.Input(
                    "on_empty",
                    options=["error", "passthrough", "force_full", "force_none"],
                    default="error",
                    tooltip=(
                        "error: detiene la ejecucion con un mensaje claro. "
                        "passthrough: la deja pasar tal cual. "
                        "force_full: la sustituye por todo blanco. "
                        "force_none: la sustituye por todo negro."
                    ),
                ),
                io.Float.Input(
                    "min_coverage", default=0.001, min=0.0, max=1.0, step=0.001,
                    tooltip="Fraccion minima de pixeles activos para considerarla valida.",
                ),
                io.Float.Input(
                    "max_coverage", default=0.999, min=0.0, max=1.0, step=0.001,
                    tooltip="Por encima de esto la mascara cubre casi todo: suele ser una inversion accidental.",
                ),
            ],
            outputs=[
                io.Mask.Output(display_name="mask"),
                io.Float.Output(display_name="coverage"),
                io.Boolean.Output(display_name="is_suspicious"),
            ],
        )

    @classmethod
    def execute(cls, mask, on_empty, min_coverage, max_coverage) -> io.NodeOutput:
        coverage = float(mask.float().mean().item())
        h, w = int(mask.shape[-2]), int(mask.shape[-1])

        too_low = coverage < min_coverage
        too_high = coverage > max_coverage
        suspicious = too_low or too_high

        if suspicious and on_empty == "error":
            hint = ""
            if (h, w) == (64, 64) and coverage == 0.0:
                hint = (
                    "\n  PISTA: 64x64 y totalmente vacia es exactamente lo que devuelve "
                    "`LoadImage` cuando el archivo NO tiene canal alfa. La mascara del "
                    "MaskEditor no llego al nodo. Comprueba que el widget diga "
                    "`clipspace/clipspace-mask-....png [input]`, o usa `LoadImageMask` "
                    "con un PNG blanco/negro aparte."
                )
            elif too_high:
                hint = (
                    "\n  PISTA: la mascara cubre casi todo el encuadre. Suele significar "
                    "que se ha invertido por accidente (InvertMask sobre una mascara vacia)."
                )
            raise ValueError(
                f"[MaskGuard] Mascara sospechosa: cobertura={coverage:.5f} "
                f"(rango valido {min_coverage}-{max_coverage}), tamano={w}x{h}.{hint}"
            )

        out = mask
        if suspicious:
            if on_empty == "force_full":
                out = torch.ones_like(mask)
            elif on_empty == "force_none":
                out = torch.zeros_like(mask)

        return io.NodeOutput(out, coverage, suspicious)


class BatchMatch(io.ComfyNode):
    """Iguala el numero de frames de dos lotes de imagenes.

    Origen real: el crash `IndexError: index 49 is out of bounds for dimension 0
    with size 49` en `ColorMatch`, porque la referencia tenia 49 frames y el
    resultado 97. Cualquier nodo que recorra el lote asumiendo que ambos miden
    igual revienta.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CP_BatchMatch",
            display_name="Batch Match",
            category="character-pipeline/guards",
            description="Recorta ambos lotes al minimo comun de frames y avisa si no coincidian.",
            inputs=[
                io.Image.Input("images_a"),
                io.Image.Input("images_b"),
                io.Combo.Input(
                    "mode", options=["truncate", "error"], default="truncate",
                    tooltip="truncate: recorta al menor. error: detiene si no coinciden.",
                ),
            ],
            outputs=[
                io.Image.Output(display_name="images_a"),
                io.Image.Output(display_name="images_b"),
                io.Int.Output(display_name="count"),
            ],
        )

    @classmethod
    def execute(cls, images_a, images_b, mode) -> io.NodeOutput:
        na, nb = int(images_a.shape[0]), int(images_b.shape[0])
        if na != nb and mode == "error":
            raise ValueError(
                f"[BatchMatch] Los lotes no coinciden: A={na} frames, B={nb} frames. "
                "Normalmente significa que el numero de frames del latente se fijo a mano "
                "en vez de derivarse del clip real."
            )
        n = min(na, nb)
        return io.NodeOutput(images_a[:n], images_b[:n], n)
