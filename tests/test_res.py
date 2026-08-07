# -*- coding: utf-8 -*-
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
from comfy_character_pipeline.ltx_res import LTXResolution  # noqa

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

print("Video 1280x720 (16:9), entrega UHD 4K, x4, multiplo 64")
r = LTXResolution.execute(Img(1280, 720), "UHD 4K 3840x2160", "x4 (dos etapas)",
                          "64 (recomendado con LoRA)", True, 1.2)
base_w, base_h, x2w, x2h, x4w, x4h, fw, fh, res = r
check("A. base multiplo de 64", (base_w % 64, base_h % 64), (0, 0))
check("A. x4 = base*4", (x4w, x4h), (base_w * 4, base_h * 4))
check("A. entrega mantiene 16:9", round(fw / fh, 2), 1.78)
check("A. x4 alcanza o supera la entrega", x4w >= fw and x4h >= fh, True)
print("   ", "  ".join(res.split("\n")[:5]))

print("\nVideo vertical 1080x1920, entrega UHD 4K, x4")
r = LTXResolution.execute(Img(1080, 1920), "UHD 4K 3840x2160", "x4 (dos etapas)",
                          "64 (recomendado con LoRA)", True, 1.2)
check("B. respeta vertical", r[0] < r[1], True)
check("B. aspecto ~0.5625", round(r[6] / r[7], 3), 0.562)

print("\nLimite de megapixeles: UHD con UNA sola etapa")
r = LTXResolution.execute(Img(1280, 720), "UHD 4K 3840x2160", "x2 (una etapa)",
                          "64 (recomendado con LoRA)", True, 1.2)
check("C. base recortada por el limite", r[0] * r[1] / 1e6 <= 1.25, True)
check("C. avisa del recorte", "AVISO" in r[8], True)

print("\nMultiplo 32")
r = LTXResolution.execute(Img(1280, 720), "1080p 1920x1080", "x2 (una etapa)",
                          "32 (VAE de LTX)", True, 1.2)
check("D. multiplo de 32", (r[0] % 32, r[1] % 32), (0, 0))
check("D. x2 alcanza 1080p", r[2] >= r[6] and r[3] >= r[7], True)

print("\nAuto: entrega = tamano del video")
r = LTXResolution.execute(Img(1280, 720), "auto (igual que el video)", "x2 (una etapa)",
                          "64 (recomendado con LoRA)", True, 1.2)
check("E. entrega = origen", (r[6], r[7]), (1280, 720))

print("\nSin respetar aspecto: fuerza el de la entrega")
r = LTXResolution.execute(Img(1080, 1920), "UHD 4K 3840x2160", "x4 (dos etapas)",
                          "64 (recomendado con LoRA)", False, 1.2)
check("F. fuerza 3840x2160", (r[6], r[7]), (3840, 2160))

print("\n" + ("TODO OK" if not FAILS else "%d FALLOS: %s" % (len(FAILS), FAILS)))
sys.exit(1 if FAILS else 0)
