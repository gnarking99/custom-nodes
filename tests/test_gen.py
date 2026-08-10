# -*- coding: utf-8 -*-
"""Pruebas de CP_LTXGenPlan y CP_LTXSigmas. Se ejecutan sin ComfyUI."""

import sys, types, os

mod = types.ModuleType("comfy_api"); latest = types.ModuleType("comfy_api.latest")
class _P:
    def __init__(self, *a, **k): pass
def _mk(n): return type(n, (), {"Input": staticmethod(_P), "Output": staticmethod(_P)})
class _IO:
    Schema = staticmethod(lambda **k: k)
    NodeOutput = staticmethod(lambda *a, **k: a)
    ComfyNode = type("ComfyNode", (), {})
    for _n in ("Image","Mask","Latent","Int","Float","String","Boolean","Combo","Model",
               "Conditioning","Sigmas","VAE","CLIP"):
        locals()[_n] = _mk(_n)
latest.io = _IO; latest.ComfyExtension = type("ComfyExtension", (), {})
mod.latest = latest
sys.modules["comfy_api"] = mod; sys.modules["comfy_api.latest"] = latest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from comfy_character_pipeline.ltx_gen import LTXGenPlan  # noqa
from comfy_character_pipeline.ltx_sigmas import LTXSigmas  # noqa

class Img:
    def __init__(self, w, h): self.shape = (1, h, w, 3)
    def dim(self): return 4

FAILS = []
def check(label, got, exp):
    ok = got == exp
    print(("  OK   " if ok else "  FALLO") + " " + label)
    if not ok:
        print("         esperado:", exp); print("         obtenido:", got)
        FAILS.append(label)

# --------------------------------------------------------------------------
print("GenPlan · i2v vertical 1080x1920, x2, 5 s a 24 fps")
r = LTXGenPlan.execute("i2v · imagen inicial", "vertical 1080x1920 (9:16)",
                       "x2 · dos pasadas", "64 (recomendado con LoRA)",
                       5.0, 24, 0.60, Img(1080, 1920))
(base_w, base_h, x2w, x2h, x4w, x4h, fw, fh,
 frames, fps, fps_f, usa_img, byp_img, usa_ref, usa_last, res) = r
check("A. base multiplo de 64", (base_w % 64, base_h % 64), (0, 0))
check("A. x2 = base*2", (x2w, x2h), (base_w * 2, base_h * 2))
check("A. entrega 1080x1920", (fw, fh), (1080, 1920))
check("A. frames 8n+1", (frames - 1) % 8, 0)
check("A. frames = 121 (5 s a 24 fps)", frames, 121)
check("A. usa_imagen", usa_img, True)
check("A. bypass_imagen es el inverso", byp_img, False)
check("A. no usa referencia", usa_ref, False)
check("A. base bajo el techo de MP", base_w * base_h / 1e6 <= 0.61, True)
check("A. fps_f es float", (fps, fps_f), (24, 24.0))
print("   ", " | ".join(res.split("\n")[1:6]))

print("\nGenPlan · t2v horizontal 1920x1080")
r = LTXGenPlan.execute("t2v · solo texto", "horizontal 1920x1080 (16:9)",
                       "x2 · dos pasadas", "64 (recomendado con LoRA)",
                       5.0, 24, 0.60, None)
check("B. usa_imagen False", r[11], False)
check("B. bypass_imagen True", r[12], True)
check("B. x2 alcanza la entrega", r[2] >= r[6] and r[3] >= r[7], True)

print("\nGenPlan · ref2v avisa de la IC-LoRA")
r = LTXGenPlan.execute("ref2v · hoja de referencia (IC-LoRA)", "vertical 1080x1920 (9:16)",
                       "x2 · dos pasadas", "64 (recomendado con LoRA)",
                       5.0, 24, 0.60, None)
check("C. usar_referencia", r[13], True)
check("C. avisa de la IC-LoRA", "IC-LoRA" in r[15], True)

print("\nGenPlan · duracion que no cae en 8n+1")
r = LTXGenPlan.execute("t2v · solo texto", "horizontal 1920x1080 (16:9)",
                       "x2 · dos pasadas", "32 (minimo del VAE)",
                       4.0, 25, 0.60, None)
check("D. 4 s x 25 fps -> 8n+1", (r[8] - 1) % 8, 0)
check("D. avisa del ajuste", "8n+1" in r[15], True)

print("\nGenPlan · aspecto de imagen distinto al de la entrega")
r = LTXGenPlan.execute("i2v · imagen inicial", "horizontal 1920x1080 (16:9)",
                       "x2 · dos pasadas", "64 (recomendado con LoRA)",
                       5.0, 24, 0.60, Img(1080, 1920))
check("E. avisa del encuadre", "estirada" in r[15], True)

print("\nGenPlan · x4 avisa de salirse del rango nativo")
r = LTXGenPlan.execute("t2v · solo texto", "horizontal 2560x1440 (16:9)",
                       "x4 · tres pasadas", "64 (recomendado con LoRA)",
                       5.0, 24, 2.0, None)
check("F. x4 = base*4", (r[4], r[5]), (r[0] * 4, r[1] * 4))

# --------------------------------------------------------------------------
print("\nSigmas · preset oficial etapa 1")
sig, pasos, s0, txt = LTXSigmas.execute("1 · oficial 8 pasos", 0.0)
check("G. 8 pasos", pasos, 8)
check("G. arranca en 1.0", round(s0, 4), 1.0)
check("G. acaba en 0", float(sig[-1]), 0.0)
check("G. estrictamente decreciente", all(sig[i] > sig[i+1] for i in range(len(sig)-1)), True)

print("\nSigmas · movimiento + LoRA (se le anade el 0.0 que falta)")
sig, pasos, s0, txt = LTXSigmas.execute("1 · movimiento + LoRA 8 pasos", 0.0)
check("H. 8 pasos tras cerrar la curva", pasos, 8)
check("H. acaba en 0", float(sig[-1]), 0.0)

print("\nSigmas · reescalado de arranque")
sig, pasos, s0, txt = LTXSigmas.execute("2 · oficial 3 pasos", 0.60)
check("I. arranca donde se pide", round(s0, 4), 0.60)
check("I. mantiene el numero de pasos", pasos, 3)
check("I. sigue acabando en 0", float(sig[-1]), 0.0)
check("I. conserva la forma", round(float(sig[1]) / float(sig[0]), 5),
      round(0.7250 / 0.85, 5))

print("\nSigmas · dev lineal")
sig, pasos, s0, txt = LTXSigmas.execute("1 · dev sin destilar 20 pasos", 0.0, "", 20)
check("J. 20 pasos", pasos, 20)

print("\nSigmas · personalizado invalido debe romper")
try:
    LTXSigmas.execute("personalizado", 0.0, "0.5, 0.7, 0.0")
    check("K. curva creciente detectada", False, True)
except ValueError as e:
    check("K. curva creciente detectada", "decreciente" in str(e), True)

try:
    LTXSigmas.execute("personalizado", 0.0, "0.5")
    check("L. lista demasiado corta detectada", False, True)
except ValueError as e:
    check("L. lista demasiado corta detectada", "al menos 2" in str(e), True)

print("\n" + ("TODO OK" if not FAILS else "%d FALLOS: %s" % (len(FAILS), FAILS)))
sys.exit(1 if FAILS else 0)
