# -*- coding: utf-8 -*-
"""Muestreo: control de denoise sobre cualquier curva de sigmas."""

from comfy_api.latest import io


class SigmasDenoise(io.ComfyNode):
    """Aplica un denoise parcial a una lista de sigmas ya calculada.

    Origen real: `Flux2Scheduler` calcula el shift en funcion de la resolucion
    (que es lo bueno) pero NO tiene entrada de denoise: siempre devuelve la curva
    completa. Eso obliga a elegir entre shift correcto o control de denoise.

    Este nodo elimina la disyuntiva: deja que Flux2Scheduler (o el que sea)
    calcule la curva y quedate solo con la cola. Funciona con cualquier scheduler
    porque opera sobre el tensor SIGMAS, no sobre el modelo.

    OJO, DIFERENCIA IMPORTANTE CON `BasicScheduler`
    -----------------------------------------------
    `BasicScheduler` con denoise 0.42 y steps 12 **sigue ejecutando 12 pasos**:
    calcula internamente una curva mas larga y se queda con la cola. Aqui la curva
    ya viene dada, asi que 12 sigmas a denoise 0.42 dejan **5 pasos reales**.

    Si quieres N pasos reales, pon en el scheduler de arriba `round(N / denoise)`.
    Para 8 pasos reales a denoise 0.42 -> 19 steps arriba.

    El parametro `min_steps` existe para que nunca te pase en silencio: si el
    recorte deja menos pasos de los que consideras aceptables, se detiene.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CP_SigmasDenoise",
            display_name="Sigmas Denoise",
            category="character-pipeline/sampling",
            description="Recorta una curva de sigmas para conseguir denoise parcial con cualquier scheduler.",
            inputs=[
                io.Sigmas.Input("sigmas"),
                io.Float.Input(
                    "denoise", default=0.42, min=0.0, max=1.0, step=0.01,
                    tooltip=(
                        "Fraccion de la curva que se ejecuta. "
                        "0.25-0.32 apenas mueve nada · 0.38-0.45 detalle con deriva minima · "
                        "0.60+ empieza a cambiar pose y expresion · 1.0 regenera desde ruido."
                    ),
                ),
                io.Int.Input(
                    "min_steps", default=3, min=0, max=1000,
                    tooltip=(
                        "Se detiene si el recorte deja menos pasos que esto. "
                        "Evita muestrear con 1-2 pasos sin darte cuenta. 0 = desactivado."
                    ),
                ),
            ],
            outputs=[
                io.Sigmas.Output(display_name="sigmas"),
                io.Int.Output(display_name="steps"),
                io.Float.Output(display_name="sigma_start"),
                io.Int.Output(display_name="steps_in"),
            ],
        )

    @classmethod
    def execute(cls, sigmas, denoise, min_steps) -> io.NodeOutput:
        if sigmas is None or sigmas.shape[0] < 2:
            raise ValueError(
                f"[SigmasDenoise] La curva de sigmas tiene {0 if sigmas is None else int(sigmas.shape[0])} "
                "valores; hacen falta al menos 2 (un paso). Revisa el scheduler de arriba."
            )

        total_steps = int(sigmas.shape[0]) - 1

        if denoise >= 1.0:
            out = sigmas.clone()
            keep = total_steps
        elif denoise <= 0.0:
            out = sigmas[-1:].clone()
            keep = 0
        else:
            keep = max(1, int(round(total_steps * float(denoise))))
            out = sigmas[-(keep + 1):].clone()

        if min_steps > 0 and keep < min_steps:
            need = math_ceil_div(min_steps, denoise) if denoise > 0 else 0
            raise ValueError(
                f"[SigmasDenoise] El recorte deja {keep} paso(s) reales, por debajo de "
                f"min_steps={min_steps}.\n"
                f"  La curva de entrada tiene {total_steps} pasos y denoise={denoise}.\n"
                f"  ARREGLO: sube los steps del scheduler de arriba a ~{need} "
                f"(round({min_steps} / {denoise})), o baja min_steps si de verdad quieres "
                f"muestrear con tan pocos pasos."
            )

        sigma_start = float(out[0].item())
        return io.NodeOutput(out, keep, sigma_start, total_steps)


def math_ceil_div(target_steps, denoise):
    import math
    return int(math.ceil(target_steps / max(1e-6, float(denoise))))
