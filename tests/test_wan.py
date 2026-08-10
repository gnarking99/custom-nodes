# -*- coding: utf-8 -*-
"""Pruebas de CP_WanGenPlan y CP_WanSplit. Se ejecutan sin ComfyUI."""

import sys, types, os
import torch

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
from comfy_character_pipeline.wan_gen import WanGenPlan  # noqa
from comfy_character_pipeline.wan_split import WanSplit  # noqa

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

def plan(**kw):
    a = dict(modo="i2v · imagen inicial", entrega="vertical 720x1280 (9:16)",
             duracion_total_s=5.0, fps=16, chunk_index=0, overlap_frames=16,
             max_frames_tanda=81, imagen=None)
    a.update(kw)
    return WanGenPlan.execute(**a)

# --------------------------------------------------------------------------
print("WanGenPlan · 5 s a 16 fps -> una sola tanda")
r = plan()
(w, h, length, fps, fps_f, tandas, start, ov, ultima, usa_img, usa_ref, res) = r
check("A. resolucion multiplo de 16", (w % 16, h % 16), (0, 0))
check("A. 720x1280", (w, h), (720, 1280))
check("A. length 4n+1", (length - 1) % 4, 0)
check("A. 5 s a 16 fps = 81 frames", length, 81)
check("A. una tanda", tandas, 1)
check("A. es la ultima", ultima, True)
check("A. solape 0 en la primera", ov, 0)
check("A. empieza en 0", start, 0)
check("A. usa imagen", usa_img, True)
print("   ", " | ".join(res.split("\n")[1:6]))

print("\nWanGenPlan · 20 s -> varias tandas")
r = plan(duracion_total_s=20.0)
check("B. mas de una tanda", r[5] > 1, True)
check("B. length tapado a 81", r[2] <= 81, True)
tandas = r[5]
# recorre todas las tandas y comprueba el avance
starts = []
for i in range(tandas):
    ri = plan(duracion_total_s=20.0, chunk_index=i, modo="continuacion · extiende el clip anterior")
    starts.append(ri[6])
    if i > 0:
        check(f"B. tanda {i} solapa 16", ri[7], 16)
check("B. los inicios avanzan de 65 en 65", [starts[i+1]-starts[i] for i in range(len(starts)-1)],
      [65] * (len(starts) - 1))
check("B. la ultima se marca", plan(duracion_total_s=20.0, chunk_index=tandas-1,
                                    modo="continuacion · extiende el clip anterior")[8], True)

print("\nWanGenPlan · pasarse de tandas debe romper")
try:
    plan(duracion_total_s=20.0, chunk_index=99)
    check("C. chunk_index fuera de rango detectado", False, True)
except ValueError as e:
    check("C. chunk_index fuera de rango detectado", "Has terminado el clip" in str(e), True)

print("\nWanGenPlan · solape >= longitud debe romper")
try:
    plan(overlap_frames=81)
    check("D. solape invalido detectado", False, True)
except ValueError as e:
    check("D. solape invalido detectado", "menor que la longitud" in str(e), True)

print("\nWanGenPlan · avisos")
r = plan(entrega="horizontal 1536x864 (16:9, fuera de rango)")
check("E. avisa de pasarse de 720p", "rango nativo" in r[11], True)
r = plan(fps=24)
check("F. avisa de fps distinto de 16", "se entreno a 16" in r[11], True)
r = plan(modo="ref2v · imagenes de referencia (Phantom)")
check("G. ref2v avisa de Phantom", r[10] is True and "Phantom" in r[11], True)
r = plan(entrega="horizontal 1280x704 (16:9)", imagen=Img(720, 1280))
check("H. avisa del encuadre", "estirada" in r[11], True)

# --------------------------------------------------------------------------
def curva(steps, shift=5.0):
    """Curva flow-match como la que da BasicScheduler con ModelSamplingSD3."""
    ts = [i / steps for i in range(steps + 1)]
    return torch.tensor([(shift * (1 - t)) / (1 + (shift - 1) * (1 - t)) for t in ts],
                        dtype=torch.float32)

print("\nWanSplit · corte en la sigma de i2v")
sig = curva(20)
high, low, step, sreal, ph, pl, res = WanSplit.execute(sig, "i2v  ·  sigma 0.900", 0.9, 1)
check("I. los dos tramos suman los pasos", ph + pl, 20)
check("I. comparten el paso del corte", (float(high[-1]), float(low[0])), (float(sig[step]), float(sig[step])))
check("I. sigma real cerca de 0.9", abs(sreal - 0.9) < 0.06, True)
check("I. high arranca en el maximo", float(high[0]), float(sig[0]))
check("I. low acaba en 0", float(low[-1]), float(sig[-1]))
print("   ", " | ".join(res.split("\n")[:4]))

print("\nWanSplit · t2v corta mas abajo que i2v")
_, _, step_i, _, _, _, _ = WanSplit.execute(sig, "i2v  ·  sigma 0.900", 0.9, 1)
_, _, step_t, _, _, _, _ = WanSplit.execute(sig, "t2v  ·  sigma 0.875", 0.875, 1)
check("J. t2v corta igual o mas tarde", step_t >= step_i, True)

print("\nWanSplit · el shift mueve el corte")
_, _, s5, _, _, _, _ = WanSplit.execute(curva(20, 5.0), "i2v  ·  sigma 0.900", 0.9, 1)
_, _, s8, _, _, _, _ = WanSplit.execute(curva(20, 8.0), "i2v  ·  sigma 0.900", 0.9, 1)
check("K. mas shift = mas pasos arriba", s8 > s5, True)

print("\nWanSplit · curva demasiado corta debe romper")
try:
    WanSplit.execute(torch.tensor([1.0, 0.0]), "i2v  ·  sigma 0.900", 0.9, 1)
    check("L. curva corta detectada", False, True)
except ValueError as e:
    check("L. curva corta detectada", "al menos 3" in str(e), True)

print("\nWanSplit · min_pasos_por_lado protege")
try:
    WanSplit.execute(curva(4, 1.0), "manual", 0.999, 2)
    check("M. lado sin pasos detectado", False, True)
except ValueError as e:
    check("M. lado sin pasos detectado", "min_pasos_por_lado" in str(e), True)

print("\n" + ("TODO OK" if not FAILS else "%d FALLOS: %s" % (len(FAILS), FAILS)))
sys.exit(1 if FAILS else 0)
