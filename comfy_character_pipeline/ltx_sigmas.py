# -*- coding: utf-8 -*-
"""Curvas de sigmas para LTX-2.3, por etapa y por perfil.

LTX-2.3 no usa `steps` + `denoise`: usa una lista explicita de sigmas. Eso es
una ventaja — puedes esculpir la curva — y una trampa: las listas viajan por los
flujos de la comunidad como cadenas de texto sueltas, sin decir a que etapa
pertenecen ni de donde salieron. Este nodo las guarda con su procedencia.
"""

import torch

from comfy_api.latest import io

# --------------------------------------------------------------------------
# Las curvas. Cada una con su origen, porque la diferencia entre ellas no es
# cosmetica: cambia cuanto movimiento y cuanta identidad sobreviven.
# --------------------------------------------------------------------------
PRESETS = {
    # --- ETAPA 1 ----------------------------------------------------------
    "1 · oficial 8 pasos": (
        [1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0],
        "Lightricks / Comfy-Org. La curva de las plantillas oficiales de LTX-2.3 "
        "(blueprint t2v/i2v y ejemplos del repo de Lightricks). Cinco pasos casi "
        "planos arriba y luego la caida. Es la referencia: si algo no te cuadra, "
        "vuelve aqui antes de tocar nada mas.",
    ),
    "1 · movimiento + LoRA 8 pasos": (
        [1.0, 0.82, 0.76, 0.38, 0.32, 0.26, 0.22, 0.18, 0.0],
        "Comunidad (hilo 'Character LoRA severely reduces motion', HF Lightricks/LTX-2.3 #36). "
        "Cae en picado en el paso 3. La tesis: LTX resuelve el video en 2-3 pasos y "
        "todo el denoise alto posterior aplana el movimiento y diluye el LoRA de "
        "personaje. Usala cuando el personaje sale bien pero se queda quieto. "
        "OJO: la lista original acaba en 0.18; aqui se le anade el 0.0 final que "
        "el sampler necesita para cerrar.",
    ),
    "1 · dev sin destilar 20 pasos": (
        None,  # se genera lineal
        "Para `ltx-2.3-22b-dev` SIN el LoRA de destilacion y con CFG 3-5. Curva "
        "lineal de 20 pasos. Mas lenta y con prompt negativo de verdad, pero sin "
        "el LoRA destilado pierdes el atajo de 8 pasos.",
    ),

    # --- ETAPA 2 y 3 ------------------------------------------------------
    "2 · oficial 3 pasos": (
        [0.85, 0.7250, 0.4219, 0.0],
        "Lightricks / Comfy-Org. La cola que va DESPUES del upscaler latente x2. "
        "Empieza en 0.85, no en 1.0: el latente ya trae la estructura, esto solo "
        "reconstruye detalle espacial.",
    ),
    "2 · fiel 3 pasos": (
        [0.421875, 0.2109375, 0.0],
        "Preset de ComfyUI_WLTX_nodes. Arranca mucho mas abajo: casi no reinterpreta. "
        "Es la opcion cuando la etapa 2 te esta cambiando la cara del personaje.",
    ),
    "2 · invencion 2 pasos": (
        [0.77, 0.44, 0.0],
        "Comunidad (mismo hilo #36). Dos pasos altos. Mete mas detalle inventado y "
        "cuesta un paso menos. Sube el riesgo de deriva de identidad.",
    ),
    "3 · remate 1 paso": (
        [0.65, 0.0],
        "Un solo paso para una tercera etapa x4. Mas que esto en la etapa mas cara "
        "del flujo no se amortiza: a partir de x2 el upscaler espacial afila, no "
        "reconstruye.",
    ),
    "3 · remate 2 pasos": (
        [0.525, 0.421875, 0.0],
        "Preset de ComfyUI_WLTX_nodes para una tercera pasada algo mas ambiciosa.",
    ),

    "personalizado": (None, "Escribe la lista a mano en `custom`."),
}


def _parse(txt):
    vals = []
    for chunk in txt.replace(";", ",").split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        vals.append(float(chunk))
    return vals


class LTXSigmas(io.ComfyNode):
    """Un desplegable en vez de una cadena de numeros copiada de un foro.

    El problema real: en un flujo de tres etapas hay tres `ManualSigmas` con
    listas parecidas, y no hay forma de mirar el canvas y saber cual es la de la
    etapa 1. Cuando pruebas una variante acabas cambiando la equivocada.

    Aqui la etapa va en el nombre del preset, la salida `procedencia` dice de
    donde sale la curva, y `pasos` te dice cuantos pasos reales vas a ejecutar
    antes de lanzar.

    `inicio` reescala la curva entera para que empiece donde tu digas,
    manteniendo su forma y el 0.0 final. Es la palanca de fidelidad de las etapas
    2 y 3: bajarlo hace que la etapa toque menos, subirlo la deja inventar mas.
    Es preferible a cambiar de sampler — un solver de segundo orden en la segunda
    pasada suele salir demasiado duro, y la comunidad coincide en que lo correcto
    es bajar la sigma de arranque, no cambiar el solver.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CP_LTXSigmas",
            display_name="LTX Sigmas",
            category="character-pipeline/sampling",
            description="Curvas de sigmas de LTX-2.3 por etapa, con su procedencia y el numero de pasos reales.",
            inputs=[
                io.Combo.Input("preset", options=list(PRESETS.keys()), default="1 · oficial 8 pasos"),
                io.Float.Input(
                    "inicio", default=0.0, min=0.0, max=1.0, step=0.005,
                    tooltip=(
                        "0 = usar el arranque del preset. Cualquier otro valor reescala "
                        "la curva para empezar ahi conservando su forma. Etapa 2: baja a "
                        "0.5-0.6 si te cambia la cara; sube a 0.9 si quieres mas detalle nuevo."
                    ),
                ),
                io.String.Input(
                    "custom", default="", multiline=True,
                    tooltip="Lista separada por comas. Solo se usa con el preset 'personalizado'.",
                ),
                io.Int.Input(
                    "pasos_dev", default=20, min=1, max=100,
                    tooltip="Solo para el preset 'dev sin destilar': cuantos pasos lineales generar.",
                ),
            ],
            outputs=[
                io.Sigmas.Output(display_name="sigmas"),
                io.Int.Output(display_name="pasos"),
                io.Float.Output(display_name="sigma_inicial"),
                io.String.Output(display_name="procedencia"),
            ],
        )

    @classmethod
    def execute(cls, preset, inicio, custom="", pasos_dev=20) -> io.NodeOutput:
        vals, origen = PRESETS[preset]

        if preset == "personalizado":
            vals = _parse(custom or "")
            if len(vals) < 2:
                raise ValueError(
                    "[LTXSigmas] El preset es 'personalizado' pero `custom` tiene "
                    f"{len(vals)} valor(es). Hacen falta al menos 2 (un paso), y el "
                    "ultimo debe ser 0.0."
                )
        elif vals is None:  # dev lineal
            vals = [1.0 - i / float(pasos_dev) for i in range(pasos_dev)] + [0.0]

        vals = [float(v) for v in vals]

        if vals[-1] != 0.0:
            vals = vals + [0.0]

        for a, b in zip(vals, vals[1:]):
            if b >= a:
                raise ValueError(
                    f"[LTXSigmas] La curva no es estrictamente decreciente: {a} -> {b}. "
                    "Un sampler de flow matching recorre las sigmas de mayor a menor; "
                    "un valor que sube da salidas ruidosas sin ningun mensaje de error."
                )

        if inicio > 0.0:
            k = inicio / vals[0]
            vals = [v * k for v in vals]
            origen += (
                f"\n\nReescalada: arranque {inicio:.4f} en vez de {inicio / k:.4f} "
                f"(factor {k:.4f}). La forma de la curva no cambia."
            )

        pasos = len(vals) - 1
        if pasos < 2:
            origen += "\n\nAVISO: 1 solo paso. Vale para un remate, no para generar."

        sig = torch.tensor(vals, dtype=torch.float32)
        texto = (
            f"{preset}\n"
            f"{pasos} pasos · arranque {vals[0]:.4f}\n"
            f"[{', '.join(f'{v:g}' for v in vals)}]\n\n{origen}"
        )
        return io.NodeOutput(sig, pasos, float(vals[0]), texto)
