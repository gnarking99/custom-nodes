# -*- coding: utf-8 -*-
"""Canon: construir el panel comparativo y montar el prompt correctivo."""

import re

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from comfy_api.latest import io

VLM_INSTRUCTION = """You are a character continuity supervisor.

The image contains {n} labelled panels side by side. Panels labelled CANON are independent samples of the canonical character. The panel labelled TARGET is the render that must be corrected.

Treat a feature as canonical ONLY if it appears consistently in EVERY canon panel. If a feature differs between canon panels, it is not reliable: ignore it completely.

Compare ONLY physical, permanent character features: number of fingers, number of limbs, moles, scars, markings, tattoos, accessories, clothing items, props, logos, and the colours of those items.

Do NOT mention pose, camera angle, facial expression, lighting, background, framing, mood or composition. Those are allowed to differ and must never be corrected.

Output ONE short English line listing ONLY what is MISSING or WRONG in the TARGET panel, as a comma-separated list of corrective noun phrases. Do not add any preamble or explanation. If nothing reliable is missing, output exactly: no changes"""

FONT_CANDIDATES = [
    "DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "arialbd.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
    "seguisb.ttf",
]

NO_CHANGE_PHRASES = {
    "no changes", "no change", "none", "nothing", "no changes needed",
    "nothing missing", "no differences", "no corrections", "n a",
}

PREAMBLE_PREFIXES = (
    "output:", "answer:", "result:", "corrections:", "response:", "list:",
)


def _load_font(px):
    for name in FONT_CANDIDATES:
        try:
            return ImageFont.truetype(name, px)
        except Exception:
            continue
    try:
        return ImageFont.load_default(size=px)      # Pillow >= 10.1
    except TypeError:
        return ImageFont.load_default()


def _to_pil(t):
    arr = (t.detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr).convert("RGB")


def _from_pil(p):
    return torch.from_numpy(np.array(p.convert("RGB")).astype(np.float32) / 255.0).unsqueeze(0)


class CanonPanelBuilder(io.ComfyNode):
    """Monta [CANON 1 | CANON 2 | ... | TARGET] en una sola imagen con etiquetas
    quemadas encima de cada panel, y devuelve la instruccion para el VLM.

    Dos razones para que esto sea un nodo y no una cadena de `ImageConcanate`:

    1. Casi ningun nodo VLM acepta varias imagenes separadas. Pegandolas en una
       sola funciona con cualquiera. Pero concatenar a ciegas deja al modelo
       adivinando cual es cual; con las etiquetas quemadas puede referirse a
       ellas sin ambiguedad.
    2. La instruccion del VLM y el numero de paneles tienen que ir sincronizados.
       Si subes el batch de canon a 4 y el texto sigue diciendo "dos paneles", el
       filtro anti-alucinacion deja de funcionar en silencio.

    El filtro anti-alucinacion es el motivo de usar varias muestras de canon: una
    generacion es una muestra, no una especificacion. Si una muestra saca cuatro
    dedos por accidente, exigir coincidencia entre todas lo descarta.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CP_CanonPanelBuilder",
            display_name="Canon Panel Builder",
            category="character-pipeline/canon",
            description="Panel etiquetado CANON|TARGET para VLM, con instruccion sincronizada.",
            inputs=[
                io.Image.Input("canon", tooltip="Lote de muestras canonicas (batch 2-4 recomendado)."),
                io.Image.Input("target", tooltip="El render actual a corregir."),
                io.Int.Input("label_height", default=48, min=0, max=256,
                             tooltip="Alto de la banda de etiqueta. 0 = sin etiquetas."),
                io.Int.Input("panel_height", default=768, min=64, max=4096,
                             tooltip="Todos los paneles se escalan a este alto antes de unirse."),
                io.Int.Input(
                    "max_total_width", default=4096, min=512, max=16384,
                    tooltip=(
                        "Si la tira supera este ancho se reescala entera. Los VLM reducen la "
                        "imagen de entrada: una tira demasiado ancha llega ilegible y el "
                        "modelo empieza a inventar diferencias."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output(display_name="panel"),
                io.String.Output(display_name="vlm_instruction"),
                io.Int.Output(display_name="canon_count"),
                io.Int.Output(display_name="panel_width"),
            ],
        )

    @classmethod
    def execute(cls, canon, target, label_height, panel_height, max_total_width) -> io.NodeOutput:
        if canon is None or canon.shape[0] == 0:
            raise ValueError(
                "[CanonPanelBuilder] No llega ninguna muestra de canon. "
                "Con menos de 2 muestras el filtro anti-alucinacion no funciona: "
                "una sola generacion es una muestra, no una especificacion."
            )
        if target is None or target.shape[0] == 0:
            raise ValueError("[CanonPanelBuilder] No llega imagen target.")

        n_canon = int(canon.shape[0])
        tiles = [_to_pil(canon[i]) for i in range(n_canon)]
        labels = [f"CANON {i + 1}" for i in range(n_canon)]
        tiles.append(_to_pil(target[0]))
        labels.append("TARGET")

        resized = []
        for im in tiles:
            w = max(1, int(round(im.width * (panel_height / im.height))))
            resized.append(im.resize((w, panel_height), Image.LANCZOS))

        total_w = sum(im.width for im in resized)
        sheet = Image.new("RGB", (total_w, panel_height + label_height), (18, 18, 18))

        font = _load_font(max(12, label_height - 18)) if label_height > 0 else None
        draw = ImageDraw.Draw(sheet)

        x = 0
        for im, text in zip(resized, labels):
            sheet.paste(im, (x, label_height))
            if label_height > 0:
                colour = (235, 90, 90) if text == "TARGET" else (90, 205, 120)
                draw.rectangle([x, 0, x + im.width, label_height], fill=(18, 18, 18))
                draw.text((x + 10, max(0, (label_height - 26) // 2)), text, fill=colour, font=font)
                draw.line([(x, 0), (x, panel_height + label_height)], fill=(70, 70, 70), width=2)
            x += im.width

        if sheet.width > max_total_width:
            k = max_total_width / sheet.width
            sheet = sheet.resize(
                (max_total_width, max(1, int(round(sheet.height * k)))), Image.LANCZOS
            )

        instruction = VLM_INSTRUCTION.format(n=len(labels))
        return io.NodeOutput(_from_pil(sheet), instruction, n_canon, int(sheet.width))


def _sanitize(text):
    """Limpia la respuesta cruda de un VLM.

    Itera porque las dos suciedades se anidan en cualquier orden: un VLM puede
    devolver `Output: "texto"` y tambien `"Output: texto"`. Una sola pasada deja
    la mitad del ruido dentro.
    """
    t = " ".join((text or "").strip().split())
    for _ in range(4):
        before = t
        for q in ('"', "'", "“", "‘", "`"):
            if len(t) > 1 and t[0] == q and t[-1] in ('"', "'", "”", "’", "`"):
                t = t[1:-1].strip()
                break
        low = t.lower()
        for pref in PREAMBLE_PREFIXES:
            if low.startswith(pref):
                t = t[len(pref):].strip()
                break
        if t == before:
            break
    return t.strip()


def _is_no_change(text):
    norm = re.sub(r"[^a-z ]+", " ", (text or "").lower())
    norm = " ".join(norm.split())
    return norm in NO_CHANGE_PHRASES or norm == ""


class PromptTemplate(io.ComfyNode):
    """Plantilla con variables con nombre: `{diff}`, `{trigger}`, `{lock}`...

    `StringReplace` del core sustituye UNA variable. En cuanto quieres montar
    "trigger + diferencias + candado de identidad" acabas encadenando tres nodos y
    perdiendo de vista el prompt real. Aqui se ve entero en un widget.

    Si una variable no se conecta, su marcador se elimina limpiamente junto con la
    coma o el espacio sobrante. Y cualquier marcador `{...}` que quede sin
    resolver (por una errata en el nombre) se elimina tambien Y se reporta en la
    salida `unresolved`: un `{diff}` literal colandose dentro del prompt es
    exactamente el tipo de fallo silencioso que este pack existe para matar.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CP_PromptTemplate",
            display_name="Prompt Template",
            category="character-pipeline/text",
            description="Plantilla de prompt con variables con nombre, limpieza de huecos y aviso de erratas.",
            inputs=[
                io.String.Input(
                    "template", multiline=True,
                    default="{trigger}, {diff}. {lock}",
                    tooltip="Usa {nombre} como marcador. Los que no reciban valor se eliminan.",
                ),
                io.String.Input("name_1", default="trigger"),
                io.String.Input("value_1", default="", optional=True, force_input=True),
                io.String.Input("name_2", default="diff"),
                io.String.Input("value_2", default="", optional=True, force_input=True),
                io.String.Input("name_3", default="lock"),
                io.String.Input("value_3", default="", optional=True, force_input=True),
                io.String.Input("name_4", default="extra"),
                io.String.Input("value_4", default="", optional=True, force_input=True),
                io.Boolean.Input(
                    "drop_on_no_changes", default=True,
                    tooltip="Si el VLM responde 'no changes' (o variantes), elimina esa variable.",
                ),
                io.Boolean.Input(
                    "sanitize_values", default=True,
                    tooltip=(
                        "Limpia las respuestas del VLM: quita comillas envolventes, colapsa "
                        "saltos de linea y elimina preambulos tipo 'Output:'."
                    ),
                ),
                io.Boolean.Input(
                    "error_on_unresolved", default=False,
                    tooltip=(
                        "Detiene la ejecucion si queda algun {marcador} sin resolver. "
                        "Actívalo en produccion; en pruebas deja que se limpie solo."
                    ),
                ),
            ],
            outputs=[
                io.String.Output(display_name="prompt"),
                io.Boolean.Output(display_name="has_changes"),
                io.String.Output(display_name="unresolved"),
            ],
        )

    @classmethod
    def execute(cls, template, name_1, name_2, name_3, name_4,
                drop_on_no_changes, sanitize_values, error_on_unresolved,
                value_1=None, value_2=None, value_3=None, value_4=None) -> io.NodeOutput:
        pairs = [(name_1, value_1), (name_2, value_2), (name_3, value_3), (name_4, value_4)]
        has_changes = True
        out = template or ""

        for name, value in pairs:
            if not name or not str(name).strip():
                continue
            token = "{" + str(name).strip() + "}"
            text = _sanitize(value) if sanitize_values else (value or "").strip()

            if drop_on_no_changes and _is_no_change(text):
                if text:
                    has_changes = False
                text = ""

            if text:
                out = out.replace(token, text)
            else:
                out = out.replace(", " + token, "").replace(token + ", ", "")
                out = out.replace(" " + token, "").replace(token, "")

        # marcadores huerfanos: erratas en el nombre, o variables que nadie definio
        leftovers = re.findall(r"\{[^{}\s]{1,64}\}", out)
        if leftovers:
            unresolved = ", ".join(sorted(set(leftovers)))
            if error_on_unresolved:
                raise ValueError(
                    f"[PromptTemplate] Quedan marcadores sin resolver: {unresolved}.\n"
                    "  Suele ser una errata entre el nombre del marcador en la plantilla y "
                    "el widget name_N, o una variable que no conectaste."
                )
            for tok in set(leftovers):
                out = out.replace(", " + tok, "").replace(tok + ", ", "")
                out = out.replace(" " + tok, "").replace(tok, "")
        else:
            unresolved = ""

        out = " ".join(out.split())
        out = out.replace(" ,", ",").replace(",,", ",").replace(" .", ".").replace("..", ".")
        out = out.replace(", .", ".").replace(",.", ".")
        # quita separadores huerfanos en los extremos, pero respeta el punto final
        return io.NodeOutput(out.strip().strip(",").strip(), has_changes, unresolved)
