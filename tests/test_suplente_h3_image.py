# -*- coding: utf-8 -*-
"""Prueba el suplente de MiniMaxH3ImageConditioning (suplente_h3_image.py) sin ComfyUI: el esquema que declara (nombres,
orden, tipos, rangos y el Autogrow de imagenes) frente al del nucleo de la v0.45.6 guardado en
tests/datos/minimax_h3_image_conditioning_v0456.json, y que ejecutarlo falla con su mensaje.
La comprobacion dentro de un ComfyUI real (quien gana el nombre en 0.42.1 y en 0.45.6) es tools/comprobar_suplente_h3_image.py.
Se ejecuta con: python tests/test_suplente_h3_image.py"""
import json
import os
import sys
import types

# ---- stub minimo de comfy_api.latest: cada puerto guarda sus argumentos y su tipo ----
mod = types.ModuleType("comfy_api")
latest = types.ModuleType("comfy_api.latest")


class _Port:
    def __init__(self, *a, **k):
        self.a, self.k = a, k


def _mk(name):
    return type(name, (), {"Input": type(name + "Input", (_Port,), {}), "Output": type(name + "Output", (_Port,), {})})


class _Autogrow:
    Input = type("AutogrowInput", (_Port,), {})
    TemplatePrefix = type("TemplatePrefix", (_Port,), {})


class _IO:
    Schema = staticmethod(lambda **k: k)
    NodeOutput = object
    ComfyNode = type("ComfyNode", (), {})
    Autogrow = _Autogrow


for _n in ("Image", "Int", "String", "Clip", "Vae", "Conditioning", "Latent"):
    setattr(_IO, _n, _mk(_n))

latest.io = _IO
mod.latest = latest
sys.modules["comfy_api"] = mod
sys.modules["comfy_api.latest"] = latest

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(AQUI, "..", "comfy_character_pipeline"))
import suplente_h3_image as S  # noqa: E402

REF = json.load(open(os.path.join(AQUI, "datos", "minimax_h3_image_conditioning_v0456.json"), encoding="utf-8"))
TIPOS = {"ClipInput": "CLIP", "VaeInput": "VAE", "StringInput": "STRING", "IntInput": "INT", "AutogrowInput": "COMFY_AUTOGROW_V3"}

esquema = S.SuplenteMiniMaxH3ImageConditioning.define_schema()
assert esquema["node_id"] == "MiniMaxH3ImageConditioning" == REF["name"]
assert esquema["display_name"] == REF["display_name"]
assert esquema["category"] == REF["category"]
assert esquema["description"] == REF["description"]

# Entradas: mismo orden que input_order de la nube (obligatorias y luego opcionales), mismos tipos y opciones.
entradas = {p.a[0]: p for p in esquema["inputs"]}
orden = [p.a[0] for p in esquema["inputs"]]
obligatorias = [n for n in orden if not entradas[n].k.get("optional")]
opcionales = [n for n in orden if entradas[n].k.get("optional")]
assert obligatorias == REF["input_order"]["required"], obligatorias
assert opcionales == REF["input_order"]["optional"], opcionales
for nombre, p in entradas.items():
    tipo = TIPOS[type(p).__name__]
    seccion = "optional" if p.k.get("optional") else "required"
    tipo_ref, opts = REF["input"][seccion][nombre]
    assert tipo == tipo_ref, (nombre, tipo, tipo_ref)
    if tipo == "INT":
        for k in ("default", "min", "max", "step", "tooltip"):
            assert p.k[k] == opts[k], (nombre, k, p.k[k], opts[k])
    elif tipo == "STRING":
        assert p.k["multiline"] is opts["multiline"] is True and p.k["dynamic_prompts"] is opts["dynamicPrompts"] is True
    elif tipo == "VAE":
        assert p.k["tooltip"] == opts["tooltip"]
    elif tipo == "COMFY_AUTOGROW_V3":
        plantilla = p.k["template"]
        t = opts["template"]
        assert plantilla.k["prefix"] == t["prefix"] == "image_"
        assert (plantilla.k["min"], plantilla.k["max"]) == (t["min"], t["max"]) == (0, 100)
        imagen = plantilla.k["input"]
        assert type(imagen).__name__ == "ImageInput" and imagen.a[0] == "image"
        assert imagen.k["tooltip"] == t["input"]["required"]["image"][1]["tooltip"]

# Salidas: CONDITIONING «positive» y LATENT.
salidas = esquema["outputs"]
assert [type(s).__name__ for s in salidas] == ["ConditioningOutput", "LatentOutput"]
assert salidas[0].k.get("display_name") == REF["output_name"][0] == "positive"
assert REF["output"] == ["CONDITIONING", "LATENT"]

# En local no se ejecuta: falla con un mensaje que dice que es de la nube.
try:
    S.SuplenteMiniMaxH3ImageConditioning.execute(clip=None, prompt="x", width=0, height=0)
    raise AssertionError("el suplente no debe ejecutarse")
except RuntimeError as e:
    assert "solo funciona en la nube" in str(e) and "MINIMAX H3 IMAGE" in str(e), e

print("test_suplente_h3_image: OK (esquema igual al de la v0.45.6 y no se ejecuta en local)")
