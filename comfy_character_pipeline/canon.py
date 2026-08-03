# -*- coding: utf-8 -*-
"""Canon: construir el panel comparativo y montar el prompt correctivo."""

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from comfy_api.latest import io

VLM_INSTRUCTION = """You are a character continuity supervisor.

The image contains {n} labelled panels side by side. Panels labelled CANON are independent samples of the canonical character. The panel labelled TARGET is the render that must be corrected.

Treat a feature as canonical ONLY if it appears consistently in EVERY canon panel. If a feature differs between canon panels, it is not reliable: ignore it completely.

Compare ONLY physical, permanent character features: number of fingers, number of limbs, moles, scars, markings, tattoos, accessories, clothing items, props, logos, and the colours of those items.

Do NOT mention pose, camera angle, facial expression, lighting, background, framing, mood or composition. Those are allowed to differ and must never be corrected.

Output ONE short English line listing ONLY what is MISSING or WRONG in the TARGET panel, as a comma-separated list of corrective noun phrases. If nothing reliable is missing, output exactly: no changes."""


def _to_pil(t):
    return Image.fromarray((t.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8))


def _from_pil(p):
    return torch.from_numpy(np.array(p).astype(np.float32) / 255.0).unsqueeze(0)


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
       filtro anti-alucinacion deja de funcionar. Aqui el texto se genera solo.

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
            ],
            outputs=[
                io.Image.Output(display_name="panel"),
                io.String.Output(display_name="vlm_instruction"),
                io.Int.Output(display_name="canon_count"),
            ],
        )

    @classmethod
    def execute(cls, canon, target, label_height, panel_height) -> io.NodeOutput:
        tiles, labels = [], []
        for i in range(int(canon.shape[0])):
            tiles.append(_to_pil(canon[i]))
            labels.append(f"CANON {i + 1}")
        tiles.append(_to_pil(target[0]))
        labels.append("TARGET")

        resized = []
        for im in tiles:
            w = max(1, int(im.width * (panel_height / im.height)))
            resized.append(im.resize((w, panel_height), Image.LANCZOS))

        total_w = sum(im.width for im in resized)
        sheet = Image.new("RGB", (total_w, panel_height + label_height), (18, 18, 18))

        try:
            font = ImageFont.truetype("DejaVuSans-Bold.ttf", max(12, label_height - 18))
        except Exception:
            font = ImageFont.load_default()

        draw = ImageDraw.Draw(sheet)
        x = 0
        for im, text in zip(resized, labels):
            sheet.paste(im, (x, label_height))
            if label_height > 0:
                colour = (230, 90, 90) if text == "TARGET" else (90, 200, 120)
                draw.rectangle([x, 0, x + im.width, label_height], fill=(18, 18, 18))
                draw.text((x + 10, max(0, (label_height - 24) // 2)), text, fill=colour, font=font)
                draw.line([(x, 0), (x, panel_height + label_height)], fill=(70, 70, 70), width=2)
            x += im.width

        instruction = VLM_INSTRUCTION.format(n=len(labels))
        return io.NodeOutput(_from_pil(sheet), instruction, int(canon.shape[0]))


class PromptTemplate(io.ComfyNode):
    """Plantilla con variables con nombre: `{diff}`, `{trigger}`, `{lock}`...

    `StringReplace` del core sustituye UNA variable. En cuanto quieres montar
    "trigger + diferencias + candado de identidad" acabas encadenando tres nodos y
    perdiendo de vista el prompt real. Aqui se ve entero en un widget.

    Si una variable no se conecta, su marcador se elimina limpiamente junto con la
    coma o el espacio sobrante: el prompt no se queda con `{diff}` literal dentro,
    que es un fallo que se cuela sin avisar.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CP_PromptTemplate",
            display_name="Prompt Template",
            category="character-pipeline/text",
            description="Plantilla de prompt con variables con nombre y limpieza de huecos.",
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
                    tooltip="Si el VLM responde 'no changes', elimina esa variable en vez de escribirla.",
                ),
            ],
            outputs=[
                io.String.Output(display_name="prompt"),
                io.Boolean.Output(display_name="has_changes"),
            ],
        )

    @classmethod
    def execute(cls, template, name_1, name_2, name_3, name_4,
                drop_on_no_changes,
                value_1=None, value_2=None, value_3=None, value_4=None) -> io.NodeOutput:
        pairs = [(name_1, value_1), (name_2, value_2), (name_3, value_3), (name_4, value_4)]
        has_changes = True
        out = template

        for name, value in pairs:
            if not name:
                continue
            token = "{" + name.strip() + "}"
            text = (value or "").strip()
            if drop_on_no_changes and text.lower().rstrip(" .") == "no changes":
                text = ""
                has_changes = False
            if text:
                out = out.replace(token, text)
            else:
                # elimina el marcador y la puntuacion huerfana que deja detras
                out = out.replace(", " + token, "").replace(token + ", ", "")
                out = out.replace(" " + token, "").replace(token, "")

        out = " ".join(out.split())
        out = out.replace(" ,", ",").replace(",,", ",").replace(" .", ".").replace("..", ".")
        # quita separadores huerfanos en los extremos, pero respeta el punto final
        return io.NodeOutput(out.strip().strip(",").strip(), has_changes)
