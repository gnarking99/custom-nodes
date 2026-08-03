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

    denoise 1.0 -> curva entera (genera desde ruido)
    denoise 0.42 -> el ultimo 42% (respeta la estructura de la imagen de entrada)
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
            ],
            outputs=[
                io.Sigmas.Output(display_name="sigmas"),
                io.Int.Output(display_name="steps"),
            ],
        )

    @classmethod
    def execute(cls, sigmas, denoise) -> io.NodeOutput:
        if denoise >= 1.0:
            return io.NodeOutput(sigmas, max(0, int(sigmas.shape[0]) - 1))
        if denoise <= 0.0:
            return io.NodeOutput(sigmas[-1:], 0)

        total_steps = int(sigmas.shape[0]) - 1
        keep = max(1, int(round(total_steps * float(denoise))))
        out = sigmas[-(keep + 1):]
        return io.NodeOutput(out, keep)
