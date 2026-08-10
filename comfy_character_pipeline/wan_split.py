# -*- coding: utf-8 -*-
"""El corte entre el experto high-noise y el low-noise de Wan 2.2, por sigma."""

import torch

from comfy_api.latest import io

# Sigma de traspaso que la comunidad reporta como correcta por tipo de modelo.
TARGETS = {
    "i2v  ·  sigma 0.900": 0.900,
    "t2v  ·  sigma 0.875": 0.875,
    "manual": None,
}


class WanSplit(io.ComfyNode):
    """Parte una curva de sigmas en el punto donde Wan 2.2 debe cambiar de experto.

    Wan 2.2 es MoE: un experto para ruido alto (composicion y movimiento) y otro
    para ruido bajo (detalle e identidad). El traspaso **no** va por la mitad de
    los pasos: va a una **sigma** concreta — 0.9 en I2V, 0.875 en T2V.

    El problema: el paso en el que cae esa sigma depende del `shift` del
    `ModelSamplingSD3`, del scheduler y del numero de pasos. Cambias el shift de
    5 a 8 y el `step` correcto se mueve, pero `SplitSigmas` sigue cortando donde
    lo dejaste. No da error: te quedas muestreando con el experto equivocado en
    parte de la curva, y el sintoma — movimiento pobre o identidad que se va —
    parece un problema de LoRA.

    Este nodo busca el corte por ti: recibe la curva ya calculada y devuelve las
    dos mitades, mas el `step` real y la sigma real donde ha cortado, para que
    puedas verlo antes de lanzar.

    El reparto de pasos que sale **no es 50/50**, y eso es correcto. Con shift 5
    y 20 pasos la sigma 0.9 cae muy pronto: pocos pasos arriba y muchos abajo.
    Ese desequilibrio es justo lo que la comunidad describe cuando dice que "el
    50/50 de manual no funciona bien".

    Se corta igual que `SplitSigmas` del core — `[:step+1]` y `[step:]` — asi que
    el paso del corte se ejecuta en los dos lados y no se pierde ninguno.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CP_WanSplit",
            display_name="Wan High/Low Split",
            category="character-pipeline/sampling",
            description="Parte la curva de sigmas en la sigma de traspaso de Wan 2.2 (0.9 i2v / 0.875 t2v) en vez de en un paso fijo.",
            inputs=[
                io.Sigmas.Input("sigmas"),
                io.Combo.Input("objetivo", options=list(TARGETS.keys()),
                               default="i2v  ·  sigma 0.900"),
                io.Float.Input(
                    "sigma_manual", default=0.900, min=0.0, max=1.0, step=0.001,
                    tooltip="Solo se usa con objetivo='manual'. Mas alta = menos pasos en el experto high.",
                ),
                io.Int.Input(
                    "min_pasos_por_lado", default=1, min=0, max=100,
                    tooltip=(
                        "Se detiene si un lado se queda con menos pasos que esto. "
                        "Un lado con 0 pasos significa que ese experto no llega a ejecutarse."
                    ),
                ),
            ],
            outputs=[
                io.Sigmas.Output(display_name="high_sigmas"),
                io.Sigmas.Output(display_name="low_sigmas"),
                io.Int.Output(display_name="step"),
                io.Float.Output(display_name="sigma_real"),
                io.Int.Output(display_name="pasos_high"),
                io.Int.Output(display_name="pasos_low"),
                io.String.Output(display_name="resumen"),
            ],
        )

    @classmethod
    def execute(cls, sigmas, objetivo, sigma_manual, min_pasos_por_lado) -> io.NodeOutput:
        if sigmas is None or sigmas.shape[-1] < 3:
            n = 0 if sigmas is None else int(sigmas.shape[-1])
            raise ValueError(
                f"[WanSplit] La curva tiene {n} valor(es); hacen falta al menos 3 para poder "
                "partirla en dos tramos con un paso cada uno. Sube `steps` en el scheduler."
            )

        objetivo_val = TARGETS[objetivo]
        if objetivo_val is None:
            objetivo_val = float(sigma_manual)

        vals = [float(v) for v in sigmas.tolist()]
        total = len(vals) - 1

        # Primer indice cuya sigma ya esta por debajo del objetivo: ahi es donde
        # el experto high deja de tener sentido. Nos quedamos con el mas cercano
        # entre ese y el anterior.
        idx = next((i for i, v in enumerate(vals) if v <= objetivo_val), total)
        if idx > 0 and abs(vals[idx - 1] - objetivo_val) < abs(vals[idx] - objetivo_val):
            idx -= 1
        step = min(max(idx, 1), total - 1)

        high = sigmas[:step + 1].clone()
        low = sigmas[step:].clone()
        pasos_high = int(high.shape[-1]) - 1
        pasos_low = int(low.shape[-1]) - 1
        sigma_real = float(vals[step])

        if min_pasos_por_lado > 0 and min(pasos_high, pasos_low) < min_pasos_por_lado:
            raise ValueError(
                f"[WanSplit] El corte deja {pasos_high} paso(s) en high y {pasos_low} en low, "
                f"por debajo de min_pasos_por_lado={min_pasos_por_lado}.\n"
                f"  La sigma {objetivo_val:.4f} cae en el paso {step} de {total}.\n"
                f"  ARREGLO: sube `steps` en el scheduler, o mueve el `shift` del "
                f"ModelSamplingSD3 (mas shift = la curva baja mas tarde = mas pasos arriba)."
            )

        curva = ", ".join(f"{v:.4f}" for v in vals[:12]) + (" ..." if len(vals) > 12 else "")
        resumen = "\n".join([
            f"objetivo     {objetivo_val:.4f}",
            f"corte        paso {step} de {total}   (sigma real {sigma_real:.4f})",
            f"high-noise   {pasos_high} paso(s)",
            f"low-noise    {pasos_low} paso(s)",
            "",
            f"curva        [{curva}]",
            "",
            ("El reparto no es 50/50 y eso es lo correcto: el traspaso va por sigma, no por pasos."
             if pasos_high != pasos_low else
             "El corte ha caido justo por la mitad. Coincidencia, no objetivo."),
        ])

        return io.NodeOutput(high, low, step, sigma_real, pasos_high, pasos_low, resumen)
