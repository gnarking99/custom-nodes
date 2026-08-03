# -*- coding: utf-8 -*-
"""Lectura de metadatos de LoRA: trigger word y compatibilidad de arquitectura."""

import json
import os
import struct

import folder_paths
from comfy_api.latest import io

# Claves explicitas de trigger, por orden de fiabilidad
EXPLICIT_KEYS = [
    "modelspec.trigger_phrase",
    "ss_activation_tags",
    "ss_trigger_words",
    "trigger_words",
    "activation_text",
]

ARCH_HINTS = {
    "flux": ("flux",),
    "sd15": ("sd1", "stable-diffusion-v1"),
    "sdxl": ("sdxl", "stable-diffusion-xl"),
    "sd3": ("sd3",),
    "ltx": ("ltx",),
    "wan": ("wan",),
    "qwen": ("qwen",),
    "hunyuan": ("hunyuan",),
}

# palabras que aparecen en casi todos los datasets y nunca son el trigger
STOPWORDS = {
    "1girl", "1boy", "solo", "looking at viewer", "simple background",
    "white background", "highres", "best quality", "masterpiece", "absurdres",
    "photo", "realistic", "3d", "render", "man", "woman", "person", "character",
}


def _read_header(path):
    """Lee solo la cabecera JSON de un .safetensors. No carga pesos."""
    with open(path, "rb") as fh:
        raw = fh.read(8)
        if len(raw) < 8:
            raise ValueError("archivo demasiado corto para ser un safetensors")
        n = struct.unpack("<Q", raw)[0]
        if n <= 0 or n > 200 * 1024 * 1024:
            raise ValueError("cabecera de tamano imposible (%d bytes)" % n)
        return json.loads(fh.read(n).decode("utf-8"))


def _tag_frequency_trigger(meta, min_ratio=0.8):
    """Deduce el trigger del histograma de etiquetas del dataset.

    Un trigger word aparece en practicamente TODAS las imagenes de entrenamiento;
    el resto de etiquetas describen imagenes concretas. Buscamos la etiqueta cuya
    frecuencia se acerca al total de imagenes.
    """
    raw = meta.get("ss_tag_frequency")
    if not raw:
        return None, []
    try:
        data = json.loads(raw) if isinstance(raw, str) else raw
    except Exception:
        return None, []

    counts = {}
    for _dir, tags in (data or {}).items():
        if not isinstance(tags, dict):
            continue
        for tag, c in tags.items():
            t = str(tag).strip()
            if t:
                counts[t] = counts.get(t, 0) + int(c)
    if not counts:
        return None, []

    top = max(counts.values())
    ranked = sorted(counts.items(), key=lambda kv: -kv[1])
    candidates = [t for t, c in ranked
                  if c >= top * min_ratio and t.lower() not in STOPWORDS]
    return (candidates[0] if candidates else None), ranked[:12]


def _detect_arch(meta, keys):
    blob = " ".join([
        str(meta.get("modelspec.architecture", "")),
        str(meta.get("ss_base_model_version", "")),
        str(meta.get("ss_sd_model_name", "")),
        str(meta.get("modelspec.title", "")),
    ]).lower()
    for name, hints in ARCH_HINTS.items():
        if any(h in blob for h in hints):
            return name
    # sin metadatos: deducir por los nombres de las capas
    joined = " ".join(list(keys)[:40]).lower()
    for name, hints in ARCH_HINTS.items():
        if any(h in joined for h in hints):
            return name
    if "double_blocks" in joined or "single_blocks" in joined:
        return "flux"
    return "desconocida"


class LoraTriggerReader(io.ComfyNode):
    """Saca la trigger word del propio archivo del LoRA. Sin preguntarle a nadie.

    El problema real: cada vez que llega un LoRA nuevo hay que escribirle a quien
    lo entreno para preguntarle la palabra de disparo. Y esa informacion **ya
    esta dentro del archivo**: quien lo entreno la dejo ahi sin saberlo.

    Un `.safetensors` empieza por 8 bytes con el tamano de la cabecera y despues
    un JSON con `__metadata__`. Este nodo lee **solo esa cabecera** — no carga los
    pesos, asi que tarda milisegundos aunque el LoRA pese 2 GB.

    Busca por orden:
    1. Claves explicitas: `modelspec.trigger_phrase`, `ss_activation_tags`...
    2. `ss_tag_frequency`: el histograma de etiquetas del dataset. Un trigger
       aparece en practicamente todas las imagenes; las demas etiquetas describen
       imagenes sueltas. La que roza el 100% es el trigger.
    3. El nombre del archivo, como ultimo recurso.

    Siempre dice **de donde** lo saco, para que sepas cuanto fiarte.

    Ademas detecta la arquitectura y te avisa si el LoRA no es de la familia que
    esperas — el fallo de cargar un LoRA de Flux sobre LTX o Wan no da error, solo
    no aplica ninguna clave y te vuelves loco buscando por que no se nota.

    Nota honesta: si quien entreno el LoRA desactivo el guardado de metadatos, no
    hay nada que leer. En ese caso el nodo te lo dice claramente en vez de
    inventarse una respuesta.
    """

    @classmethod
    def define_schema(cls):
        try:
            loras = folder_paths.get_filename_list("loras")
        except Exception:
            loras = []
        return io.Schema(
            node_id="CP_LoraTriggerReader",
            display_name="LoRA Trigger Reader",
            category="character-pipeline/organization",
            description="Lee la trigger word y la arquitectura desde los metadatos del .safetensors.",
            inputs=[
                io.Combo.Input("lora_name", options=loras or ["<no hay loras>"]),
                io.Combo.Input(
                    "expected_arch",
                    options=["cualquiera", "flux", "ltx", "wan", "sdxl", "sd15", "sd3", "qwen"],
                    default="cualquiera",
                    tooltip="Si no coincide con la del LoRA, se detiene. Evita cargar un LoRA de otra familia.",
                ),
                io.String.Input(
                    "fallback_trigger", default="",
                    tooltip="Se usa solo si el archivo no trae ningun metadato aprovechable.",
                ),
                io.Float.Input(
                    "frequency_ratio", default=0.8, min=0.1, max=1.0, step=0.05,
                    tooltip=(
                        "Que porcentaje del maximo debe alcanzar una etiqueta para considerarla "
                        "trigger. 0.8 = aparece en al menos el 80% de las veces que aparece la mas comun."
                    ),
                ),
            ],
            outputs=[
                io.String.Output(display_name="trigger"),
                io.String.Output(display_name="source"),
                io.String.Output(display_name="architecture"),
                io.String.Output(display_name="report"),
                io.String.Output(display_name="top_tags"),
                io.String.Output(display_name="lora_name"),
            ],
        )

    @classmethod
    def execute(cls, lora_name, expected_arch, fallback_trigger, frequency_ratio) -> io.NodeOutput:
        if not lora_name or lora_name.startswith("<"):
            raise ValueError("[LoraTriggerReader] No hay LoRAs en la carpeta `models/loras`.")

        path = folder_paths.get_full_path("loras", lora_name)
        if not path or not os.path.isfile(path):
            raise ValueError(f"[LoraTriggerReader] No encuentro el archivo del LoRA '{lora_name}'.")

        if not path.lower().endswith(".safetensors"):
            trig = fallback_trigger.strip() or os.path.splitext(os.path.basename(lora_name))[0]
            return io.NodeOutput(trig, "nombre de archivo (no es .safetensors)",
                                 "desconocida", "Formato sin metadatos legibles.", "", lora_name)

        try:
            header = _read_header(path)
        except Exception as exc:
            trig = fallback_trigger.strip() or os.path.splitext(os.path.basename(lora_name))[0]
            return io.NodeOutput(trig, f"nombre de archivo (cabecera ilegible: {exc})",
                                 "desconocida", f"No se pudo leer la cabecera: {exc}", "", lora_name)

        meta = header.get("__metadata__", {}) or {}
        keys = [k for k in header.keys() if k != "__metadata__"]

        trigger, source = None, None
        for key in EXPLICIT_KEYS:
            val = meta.get(key)
            if val and str(val).strip():
                trigger = str(val).strip().strip('"').strip()
                source = f"metadato explicito `{key}`"
                break

        ranked = []
        if not trigger:
            trigger, ranked = _tag_frequency_trigger(meta, frequency_ratio)
            if trigger:
                source = "deducido de `ss_tag_frequency` (etiqueta mas constante del dataset)"

        if not trigger:
            trigger = fallback_trigger.strip() or os.path.splitext(os.path.basename(lora_name))[0]
            source = ("fallback manual" if fallback_trigger.strip()
                      else "nombre del archivo — el LoRA no trae metadatos de entrenamiento")

        arch = _detect_arch(meta, keys)
        if expected_arch != "cualquiera" and arch != "desconocida" and arch != expected_arch:
            raise ValueError(
                f"[LoraTriggerReader] Este LoRA es de arquitectura **{arch}** y esperabas "
                f"**{expected_arch}**.\n"
                f"  Archivo: {lora_name}\n"
                f"  Cargarlo igualmente no daria error: simplemente no aplicaria ninguna clave "
                f"y pensarias que el LoRA 'no se nota'. Los LoRA no cruzan de arquitectura."
            )

        def g(k, d="?"):
            return meta.get(k, d)

        report = "\n".join([
            f"LoRA        : {lora_name}",
            f"TRIGGER     : {trigger}",
            f"origen      : {source}",
            f"arquitectura: {arch}",
            f"base model  : {g('ss_base_model_version', g('ss_sd_model_name'))}",
            f"imagenes    : {g('ss_num_train_images')}   epochs: {g('ss_epoch')}   "
            f"pasos: {g('ss_steps')}",
            f"dim / alpha : {g('ss_network_dim')} / {g('ss_network_alpha')}",
            f"titulo      : {g('modelspec.title', '(sin titulo)')}",
            "",
            ("AVISO: sin metadatos de entrenamiento. El trigger es una suposicion."
             if "nombre del archivo" in source else
             "Metadatos leidos correctamente."),
        ])
        top = ", ".join(f"{t} ({c})" for t, c in ranked) if ranked else ""

        return io.NodeOutput(trigger, source, arch, report, top, lora_name)
