# -*- coding: utf-8 -*-
"""Guardas: nodos que convierten fallos silenciosos en fallos ruidosos."""

import torch
from comfy_api.latest import io


def _normalize_mask(mask):
    """Devuelve la mascara como [B,H,W] float en 0..1."""
    if mask is None or mask.numel() == 0:
        raise ValueError(
            "[MaskGuard] La mascara esta vacia (0 elementos). "
            "Suele significar que el nodo de origen no produjo nada."
        )
    m = mask.float()
    if m.dim() == 2:
        m = m.unsqueeze(0)
    while m.dim() > 3:
        m = m.squeeze(-1) if m.shape[-1] == 1 else m.reshape(-1, m.shape[-2], m.shape[-1])
    return m.clamp(0.0, 1.0)


class MaskGuard(io.ComfyNode):
    """Detiene el desastre clasico: una mascara vacia que aguas abajo se invierte
    y acaba pegando la imagen original encima de todo el trabajo.

    Origen real: `LoadImage` devuelve una mascara 64x64 de ceros cuando el PNG no
    tiene canal alfa. Esa mascara pasa por `InvertMask`, sale toda blanca, y el
    `ImageCompositeMasked` tapa el resultado al 100%. El flujo no falla: entrega
    silenciosamente la imagen de entrada.

    En video comprueba **cada frame por separado**: una media global esconde un
    frame vacio entre 49 buenos, que es exactamente donde aparecen los parpadeos.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CP_MaskGuard",
            display_name="Mask Guard",
            category="character-pipeline/guards",
            description=(
                "Comprueba que una mascara tiene contenido util antes de dejarla pasar. "
                "Evita el fallo silencioso de mascara vacia. Analiza frame a frame."
            ),
            inputs=[
                io.Mask.Input("mask"),
                io.Boolean.Input(
                    "enabled", default=True,
                    tooltip=(
                        "Apagalo cuando NO estes usando mascara. Deja pasar todo sin "
                        "comprobar nada, pero sigue reportando la cobertura para el panel "
                        "de estado. Es lo que hay que usar cuando el modo mascara esta "
                        "desactivado: sin esto el guard se queja de una mascara vacia que "
                        "en realidad nadie va a usar."
                    ),
                ),
                io.Combo.Input(
                    "on_empty",
                    options=["warn", "error", "passthrough", "force_full", "force_none"],
                    default="warn",
                    tooltip=(
                        "warn: avisa por consola y deja pasar (recomendado si activas y "
                        "desactivas la mascara a menudo; el panel de estado lo reporta igual). "
                        "error: detiene la ejecucion con un diagnostico. "
                        "passthrough: la deja pasar en silencio. "
                        "force_full / force_none: la sustituye por todo blanco o todo negro."
                    ),
                ),
                io.Float.Input(
                    "min_coverage", default=0.001, min=0.0, max=1.0, step=0.001,
                    tooltip="Fraccion minima de pixeles activos para considerarla valida.",
                ),
                io.Float.Input(
                    "max_coverage", default=0.999, min=0.0, max=1.0, step=0.001,
                    tooltip="Por encima de esto cubre casi todo: suele ser una inversion accidental.",
                ),
                io.Boolean.Input(
                    "check_every_frame", default=True,
                    tooltip=(
                        "Analiza cada frame por separado en vez de la media del lote. "
                        "Desactivalo solo si trabajas con mascaras animadas que empiezan vacias a proposito."
                    ),
                ),
            ],
            outputs=[
                io.Mask.Output(display_name="mask"),
                io.Float.Output(display_name="coverage"),
                io.Boolean.Output(display_name="is_suspicious"),
                io.Float.Output(display_name="coverage_min"),
                io.Int.Output(display_name="bad_frames"),
            ],
        )

    @classmethod
    def execute(cls, mask, enabled, on_empty, min_coverage, max_coverage,
                check_every_frame) -> io.NodeOutput:
        m = _normalize_mask(mask)
        n, h, w = m.shape

        if not enabled:
            # Modo "no estoy usando mascara": reporta pero no juzga.
            per_frame = m.reshape(n, -1).mean(dim=1)
            return io.NodeOutput(m, float(per_frame.mean().item()), False,
                                 float(per_frame.min().item()), 0)

        per_frame = m.reshape(n, -1).mean(dim=1)
        coverage = float(per_frame.mean().item())
        coverage_min = float(per_frame.min().item())
        coverage_max = float(per_frame.max().item())

        if check_every_frame:
            bad = ((per_frame < min_coverage) | (per_frame > max_coverage))
        else:
            flag = (coverage < min_coverage) or (coverage > max_coverage)
            bad = torch.full((n,), bool(flag), dtype=torch.bool)
        bad_frames = int(bad.sum().item())
        suspicious = bad_frames > 0

        if suspicious and on_empty in ("error", "warn"):
            hint = ""
            if (h, w) == (64, 64) and coverage_max == 0.0:
                hint = (
                    "\n  PISTA: 64x64 y totalmente vacia es exactamente lo que devuelve "
                    "`LoadImage` cuando el archivo NO tiene canal alfa. La mascara del "
                    "MaskEditor no llego al nodo. Comprueba que el widget diga "
                    "`clipspace/clipspace-mask-....png [input]`, o usa `LoadImageMask` "
                    "con un PNG blanco/negro aparte."
                )
            elif coverage_min > max_coverage:
                hint = (
                    "\n  PISTA: la mascara cubre casi todo el encuadre. Casi siempre es "
                    "una inversion accidental (InvertMask sobre una mascara vacia)."
                )
            elif n > 1 and bad_frames < n:
                idx = [str(i) for i in torch.nonzero(bad).flatten().tolist()[:12]]
                hint = (
                    f"\n  PISTA: solo fallan {bad_frames} de {n} frames (indices: {', '.join(idx)}"
                    f"{'...' if bad_frames > 12 else ''}). Una mascara animada o mal generada "
                    "en parte del clip produce parpadeo justo en esos frames."
                )
            msg = (
                f"[MaskGuard] Mascara sospechosa. cobertura media={coverage:.5f}, "
                f"minima={coverage_min:.5f}, maxima={coverage_max:.5f}, tamano={w}x{h}, "
                f"frames={n}, frames malos={bad_frames}. "
                f"Rango valido: {min_coverage}-{max_coverage}.{hint}"
            )
            if on_empty == "error":
                raise ValueError(msg)
            print("AVISO " + msg)
            print("  (modo `warn`: se deja pasar. Si no estas usando mascara, apaga "
                  "`enabled` en este nodo o metelo en el interruptor de la mascara.)")

        out = m
        if suspicious:
            if on_empty == "force_full":
                out = torch.ones_like(m)
            elif on_empty == "force_none":
                out = torch.zeros_like(m)

        return io.NodeOutput(out, coverage, suspicious, coverage_min, bad_frames)


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
                io.Boolean.Output(display_name="did_match"),
            ],
        )

    @classmethod
    def execute(cls, images_a, images_b, mode) -> io.NodeOutput:
        na, nb = int(images_a.shape[0]), int(images_b.shape[0])

        if na == 0 or nb == 0:
            raise ValueError(
                f"[BatchMatch] Un lote llega vacio: A={na} frames, B={nb} frames. "
                "Recortar no arregla esto; revisa el nodo que produce el lote vacio."
            )

        if na != nb and mode == "error":
            raise ValueError(
                f"[BatchMatch] Los lotes no coinciden: A={na} frames, B={nb} frames.\n"
                "  PISTA: normalmente significa que el numero de frames del latente se fijo "
                "a mano en vez de derivarse del clip real. Usa `Video Chunk` y conecta su "
                "salida `length` a TODOS los nodos que necesiten la longitud."
            )

        n = min(na, nb)
        return io.NodeOutput(images_a[:n], images_b[:n], n, na == nb)
