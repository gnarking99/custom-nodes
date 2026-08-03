# -*- coding: utf-8 -*-
"""Prueba la logica pura de los nodos sin necesitar ComfyUI.
Importa los modulos reales stubbeando comfy_api para que se puedan cargar."""
import os
import sys
import types

# ---- stub minimo de comfy_api.latest / comfy.utils ----
mod = types.ModuleType("comfy_api")
latest = types.ModuleType("comfy_api.latest")


class _Port:
    def __init__(self, *a, **k):
        pass


def _mk(name):
    return type(name, (), {"Input": staticmethod(_Port), "Output": staticmethod(_Port)})


class _IO:
    Schema = staticmethod(lambda **k: k)
    NodeOutput = staticmethod(lambda *a, **k: a)
    ComfyNode = type("ComfyNode", (), {})
    for _n in ("Image", "Mask", "Latent", "Int", "Float", "String", "Boolean",
               "Combo", "Model", "Conditioning", "Sigmas", "VAE", "CLIP"):
        locals()[_n] = _mk(_n)


latest.io = _IO
latest.ComfyExtension = type("ComfyExtension", (), {})
latest.ui = types.ModuleType("ui")
mod.latest = latest
sys.modules["comfy_api"] = mod
sys.modules["comfy_api.latest"] = latest
cu = types.ModuleType("comfy.utils")
cu.common_upscale = lambda s, w, h, m, c: s
comfy = types.ModuleType("comfy")
comfy.utils = cu
sys.modules["comfy"] = comfy
sys.modules["comfy.utils"] = cu

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from comfy_character_pipeline.canon import PromptTemplate, _sanitize, _is_no_change  # noqa: E402
from comfy_character_pipeline.geometry import VideoChunk, SPATIAL, _snap             # noqa: E402

FAILS = []


def check(label, got, expected):
    ok = got == expected
    print(("  OK   " if ok else "  FALLO") + f" {label}")
    if not ok:
        print(f"         esperado: {expected!r}")
        print(f"         obtenido: {got!r}")
        FAILS.append(label)


T = "{trigger}, {diff}. {lock}"
LOCK = "Keep the character identity, pose and composition exactly as in the reference. Change nothing else."

print("PromptTemplate")
p, hc, un = PromptTemplate.execute(
    T, "trigger", "diff", "lock", "extra", True, True, False,
    value_1="clayantboss", value_2="exactly three fingers on each hand", value_3=LOCK)
check("A. con cambios", p, "clayantboss, exactly three fingers on each hand. " + LOCK)
check("A. has_changes", hc, True)
check("A. sin marcadores sueltos", "{" in p, False)

p, hc, un = PromptTemplate.execute(
    T, "trigger", "diff", "lock", "extra", True, True, False,
    value_1="clayantboss", value_2="No changes.", value_3=LOCK)
check("B. 'No changes.' se elimina", p, "clayantboss. " + LOCK)
check("B. has_changes False", hc, False)

p, hc, un = PromptTemplate.execute(
    "{trigger}, {difff}. {lock}", "trigger", "diff", "lock", "extra", True, True, False,
    value_1="clayantboss", value_2="three fingers", value_3=LOCK)
check("C. errata {difff} detectada", un, "{difff}")
check("C. errata limpiada del prompt", "{" in p, False)

p, hc, un = PromptTemplate.execute(
    T, "trigger", "diff", "lock", "extra", True, True, False,
    value_1="clayantboss", value_2='Output: "a striped tie on the chest"', value_3=LOCK)
check("D. preambulo y comillas limpiados", p, "clayantboss, a striped tie on the chest. " + LOCK)

check("E. sanitize multilinea", _sanitize("  a,\n  b  "), "a, b")
check("F. no-change variantes", [_is_no_change(x) for x in
      ["no changes", "No changes.", "NONE", "nothing missing", "three fingers"]],
      [True, True, True, True, False])

try:
    PromptTemplate.execute("{a} {b}", "a", "zz", "c", "d", True, True, True, value_1="x")
    check("G. error_on_unresolved dispara", False, True)
except ValueError as e:
    check("G. error_on_unresolved dispara", "{b}" in str(e), True)

print("\nVideoChunk")
img = types.SimpleNamespace(shape=(50,))


class Fake:
    def __init__(self, n):
        self.n = n
        self.shape = (n,)

    def __getitem__(self, s):
        start = s.start or 0
        stop = min(self.n, s.stop if s.stop is not None else self.n)
        return Fake(max(0, stop - start))


r = VideoChunk.execute(Fake(50), "ltx-2.3 (8n+1)", 0, 97, 8, "error")
check("H. 50 frames, pide 97 -> 49", (r[1], r[2], r[3], r[4]), (49, 1, 0, True))
r = VideoChunk.execute(Fake(50), "wan-2.2 (4n+1)", 0, 97, 8, "error")
check("I. wan 4n+1 -> 49", r[1], 49)
r = VideoChunk.execute(Fake(300), "ltx-2.3 (8n+1)", 1, 97, 8, "error")
check("J. chunk 1 arranca en 89", (r[1], r[3]), (97, 89))
try:
    VideoChunk.execute(Fake(50), "ltx-2.3 (8n+1)", 5, 97, 8, "error")
    check("K. chunk fuera de rango avisa", False, True)
except ValueError as e:
    check("K. chunk fuera de rango avisa", "solo hay 1 trozo" in str(e), True)
try:
    VideoChunk.execute(Fake(50), "ltx-2.3 (8n+1)", 0, 17, 20, "error")
    check("L. overlap >= longitud avisa", False, True)
except ValueError as e:
    check("L. overlap >= longitud avisa", "menor que la longitud" in str(e), True)

print("\nSnapResolution (aritmetica)")
check("M. 1000x1500 @1MP x32", (_snap(816.5, 32), _snap(1224.7, 32)), (832, 1216))
check("N. multiplos correctos", (832 % 32, 1216 % 32), (0, 0))
check("O. ltx x64", (_snap(816.5, 64) % 64, _snap(1224.7, 64) % 64), (0, 0))

print("\n" + ("TODO OK" if not FAILS else f"{len(FAILS)} FALLOS: {FAILS}"))
sys.exit(1 if FAILS else 0)

