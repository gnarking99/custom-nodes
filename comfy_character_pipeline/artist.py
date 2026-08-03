# -*- coding: utf-8 -*-
"""Nodos pensados para artistas: un desplegable en vez de seis numeros."""

from comfy_api.latest import io

# steps del scheduler, denoise, MP base, MP refinado, MP referencia
PRESETS = {
    "flux2-klein": {
        "Borrador (rapido)": (8, 0.32, 1.0, 1.5, 1.0),
        "Normal":            (14, 0.42, 1.5, 3.0, 1.0),
        "Maximo (lento)":    (22, 0.48, 2.0, 4.0, 1.0),
        "Solo limpiar":      (10, 0.22, 1.5, 3.0, 1.0),
    },
    "ltx-2.3 (video)": {
        "Borrador (rapido)": (8, 1.00, 0.35, 0.9, 1.0),
        "Normal":            (8, 1.00, 0.59, 2.4, 1.0),
        "Maximo (lento)":    (12, 1.00, 0.90, 3.5, 1.0),
        "Solo limpiar":      (8, 1.00, 0.59, 2.4, 1.0),
    },
    "wan-2.2 (video)": {
        "Borrador (rapido)": (12, 0.18, 0.50, 1.2, 1.0),
        "Normal":            (16, 0.25, 0.74, 1.7, 1.0),
        "Maximo (lento)":    (20, 0.32, 1.00, 2.4, 1.0),
        "Solo limpiar":      (16, 0.15, 0.74, 1.7, 1.0),
    },
}

EXPLAIN = {
    "Borrador (rapido)": "Para decidir encuadre y prompt. No lo uses para entregar.",
    "Normal": "El que debes usar el 90% de las veces.",
    "Maximo (lento)": "Solo para la toma final. Tarda bastante mas y cambia mas la imagen.",
    "Solo limpiar": "Denoise muy bajo: recupera nitidez sin tocar practicamente nada.",
}


class QualityPreset(io.ComfyNode):
    """Un desplegable en vez de seis numeros sueltos.

    Tus artistas no tienen por que saber que `steps 14 x denoise 0.42` da 6 pasos
    reales, ni que Wan quiere 0.15-0.25 mientras Flux aguanta 0.42. Eligen
    **Borrador / Normal / Maximo** y el nodo reparte los numeros correctos para el
    flujo en el que estan.

    La salida `resumen` esta escrita para leerse en un PreviewAny: dice en
    castellano que va a pasar y cuanto va a tardar aproximadamente.
    """

    @classmethod
    def define_schema(cls):
        first = PRESETS["flux2-klein"]
        return io.Schema(
            node_id="CP_QualityPreset",
            display_name="Quality Preset",
            category="character-pipeline/artist",
            description="Un desplegable de calidad que reparte steps, denoise y megapixeles.",
            inputs=[
                io.Combo.Input("flujo", options=list(PRESETS.keys()), default="flux2-klein"),
                io.Combo.Input("calidad", options=list(first.keys()), default="Normal"),
                io.Float.Input(
                    "ajuste_denoise", default=0.0, min=-0.20, max=0.20, step=0.01,
                    tooltip=(
                        "Retoque fino sobre el preset. Negativo = respeta mas el original. "
                        "Positivo = mas detalle nuevo pero mas riesgo de que cambie la cara."
                    ),
                ),
            ],
            outputs=[
                io.Int.Output(display_name="steps"),
                io.Float.Output(display_name="denoise"),
                io.Float.Output(display_name="mp_base"),
                io.Float.Output(display_name="mp_refinado"),
                io.Float.Output(display_name="mp_referencia"),
                io.String.Output(display_name="resumen"),
            ],
        )

    @classmethod
    def execute(cls, flujo, calidad, ajuste_denoise) -> io.NodeOutput:
        table = PRESETS.get(flujo)
        if table is None:
            raise ValueError(f"[QualityPreset] Flujo desconocido: {flujo}")
        if calidad not in table:
            raise ValueError(
                f"[QualityPreset] La calidad '{calidad}' no existe para {flujo}. "
                f"Opciones: {', '.join(table)}"
            )

        steps, denoise, mp_base, mp_ref, mp_reference = table[calidad]
        denoise = round(min(1.0, max(0.0, denoise + float(ajuste_denoise))), 3)
        real_steps = steps if denoise >= 1.0 else max(1, int(round(steps * denoise)))

        aviso = ""
        if denoise >= 0.55 and "video" not in flujo:
            aviso = ("\nAVISO: por encima de 0.55 el modelo empieza a cambiar la pose y la "
                     "expresion del personaje. Si no quieres eso, baja el ajuste.")
        if real_steps < 4 and denoise < 1.0:
            aviso += ("\nAVISO: quedan muy pocos pasos reales. Sube la calidad o el "
                      "ajuste de denoise.")

        resumen = (
            f"{flujo}  ·  {calidad}\n"
            f"{EXPLAIN.get(calidad, '')}\n\n"
            f"denoise {denoise}   ({steps} steps -> {real_steps} pasos reales)\n"
            f"resolucion base {mp_base} MP  ·  refinado {mp_ref} MP\n"
            f"referencia {mp_reference} MP"
            f"{aviso}"
        )
        return io.NodeOutput(steps, denoise, mp_base, mp_ref, mp_reference, resumen)


class PipelineStatus(io.ComfyNode):
    """Un semaforo en castellano en vez de leer trazas de Python.

    Recoge lo que reportan las guardas del pack y lo convierte en un parrafo que
    un artista puede leer sin saber que es un tensor. Conectalo a un `PreviewAny`
    y ponlo bien grande arriba del flujo.

    Todas las entradas son opcionales: conecta solo las que tenga tu flujo.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CP_PipelineStatus",
            display_name="Pipeline Status",
            category="character-pipeline/artist",
            description="Convierte las salidas de las guardas en un informe legible.",
            inputs=[
                io.String.Input("titulo", default="Estado de la ejecucion"),
                io.Float.Input("mask_coverage", default=-1.0, optional=True, force_input=True),
                io.Int.Input("mask_bad_frames", default=-1, optional=True, force_input=True),
                io.Boolean.Input("batch_did_match", default=True, optional=True, force_input=True),
                io.Int.Input("chunk_length", default=-1, optional=True, force_input=True),
                io.Int.Input("chunk_total", default=-1, optional=True, force_input=True),
                io.Boolean.Input("chunk_is_last", default=True, optional=True, force_input=True),
                io.Int.Input("sampler_steps", default=-1, optional=True, force_input=True),
                io.String.Input("trigger", default="", optional=True, force_input=True),
                io.String.Input("prompt_unresolved", default="", optional=True, force_input=True),
                io.Boolean.Input("vlm_has_changes", default=True, optional=True, force_input=True),
            ],
            outputs=[
                io.String.Output(display_name="informe"),
                io.Boolean.Output(display_name="todo_ok"),
            ],
        )

    @classmethod
    def execute(cls, titulo, mask_coverage=-1.0, mask_bad_frames=-1, batch_did_match=True,
                chunk_length=-1, chunk_total=-1, chunk_is_last=True, sampler_steps=-1,
                trigger="", prompt_unresolved="", vlm_has_changes=True) -> io.NodeOutput:
        # Un nodo aguas arriba en bypass entrega None por su salida. Eso NO es un
        # fallo: significa "esa parte del flujo esta apagada". Se trata como
        # "sin dato" en vez de reventar.
        def v(x, d):
            return d if x is None else x

        mask_coverage = v(mask_coverage, -1.0)
        mask_bad_frames = v(mask_bad_frames, -1)
        batch_did_match = v(batch_did_match, True)
        chunk_length = v(chunk_length, -1)
        chunk_total = v(chunk_total, -1)
        chunk_is_last = v(chunk_is_last, True)
        sampler_steps = v(sampler_steps, -1)
        trigger = v(trigger, "")
        prompt_unresolved = v(prompt_unresolved, "")
        vlm_has_changes = v(vlm_has_changes, True)

        ok, lines = True, [titulo, "=" * len(titulo), ""]

        if trigger:
            lines.append(f"[OK] Trigger del personaje: '{trigger}'")

        if mask_coverage >= 0:
            pct = mask_coverage * 100
            if pct < 0.1 and mask_bad_frames <= 0:
                lines.append("[i] Sin mascara. El flujo edita la imagen entera: es lo "
                             "normal si el interruptor de mascara esta apagado.")
            elif mask_bad_frames and mask_bad_frames > 0:
                ok = False
                lines.append(f"[!!] Mascara: {mask_bad_frames} frame(s) vacios o invertidos. "
                             "Repinta la mascara o usa un PNG blanco y negro aparte.")
            elif pct < 3:
                lines.append(f"[?] Mascara muy pequena ({pct:.1f}% del encuadre). "
                             "Comprueba que cubre al personaje entero.")
            elif pct > 90:
                lines.append(f"[?] Mascara casi total ({pct:.1f}%). Si querias editar solo "
                             "al personaje, algo la invirtio.")
            else:
                lines.append(f"[OK] Mascara: cubre el {pct:.1f}% del encuadre.")

        if batch_did_match is False:
            ok = False
            lines.append("[!!] Los lotes de imagenes no tenian el mismo numero de frames. "
                         "Se han recortado al menor: revisa de donde sale la longitud.")

        if chunk_length > 0:
            txt = f"[OK] Trozo de {chunk_length} frames"
            if chunk_total > 0:
                txt += f" de {chunk_total} trozo(s)"
            if chunk_total > 1 and not chunk_is_last:
                txt += ".  FALTAN TROZOS: sube `chunk_index` en 1 y vuelve a lanzar."
            elif chunk_total > 1:
                txt += ".  Este es el ULTIMO trozo del clip."
            lines.append(txt)

        if sampler_steps >= 0:
            if sampler_steps < 4:
                ok = False
                lines.append(f"[!!] Solo {sampler_steps} paso(s) de muestreo. Es muy poco: "
                             "sube los steps del scheduler o la calidad del preset.")
            else:
                lines.append(f"[OK] {sampler_steps} pasos reales de muestreo.")

        if prompt_unresolved:
            ok = False
            lines.append(f"[!!] Hay huecos sin rellenar en el prompt: {prompt_unresolved}. "
                         "Revisa que los nombres del PromptTemplate coincidan con la plantilla.")

        if vlm_has_changes is False:
            lines.append("[i] El analizador no encontro nada que corregir en el personaje. "
                         "El pase de edicion no va a cambiar gran cosa: es normal.")

        lines.append("")
        lines.append("TODO CORRECTO — puedes lanzar." if ok
                     else "HAY AVISOS — lee los [!!] de arriba antes de dar por buena la salida.")
        return io.NodeOutput("\n".join(lines), ok)
