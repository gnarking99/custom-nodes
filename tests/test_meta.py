# -*- coding: utf-8 -*-
"""Prueba la deteccion de trigger contra un .safetensors sintetico real."""
import json
import os
import struct
import sys
import tempfile
import types

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

mod = types.ModuleType("comfy_api")
latest = types.ModuleType("comfy_api.latest")


class _P:
    def __init__(self, *a, **k):
        pass


def _mk(n):
    return type(n, (), {"Input": staticmethod(_P), "Output": staticmethod(_P)})


class _IO:
    Schema = staticmethod(lambda **k: k)
    NodeOutput = staticmethod(lambda *a, **k: a)
    ComfyNode = type("ComfyNode", (), {})
    for _n in ("Image", "Mask", "Latent", "Int", "Float", "String", "Boolean",
               "Combo", "Model", "Conditioning", "Sigmas", "VAE", "CLIP"):
        locals()[_n] = _mk(_n)


latest.io = _IO
latest.ComfyExtension = type("ComfyExtension", (), {})
mod.latest = latest
sys.modules["comfy_api"] = mod
sys.modules["comfy_api.latest"] = latest

TMP = tempfile.mkdtemp()
fp = types.ModuleType("folder_paths")
fp.get_filename_list = lambda k: sorted(os.listdir(TMP))
fp.get_full_path = lambda k, n: os.path.join(TMP, n)
sys.modules["folder_paths"] = fp

sys.path.insert(0, REPO)
from comfy_character_pipeline.lora_meta import LoraTriggerReader, _read_header  # noqa: E402

FAILS = []


def check(label, got, exp):
    ok = got == exp
    print(("  OK   " if ok else "  FALLO") + " " + label)
    if not ok:
        print("         esperado: %r" % (exp,))
        print("         obtenido: %r" % (got,))
        FAILS.append(label)


def write_lora(name, metadata, keys=("lora_unet_double_blocks_0_img_attn.lora_down.weight",)):
    header = {k: {"dtype": "F16", "shape": [4, 4], "data_offsets": [0, 32]} for k in keys}
    header["__metadata__"] = metadata
    blob = json.dumps(header).encode("utf-8")
    path = os.path.join(TMP, name)
    with open(path, "wb") as fh:
        fh.write(struct.pack("<Q", len(blob)))
        fh.write(blob)
        fh.write(b"\0" * 32)
    return path


print("Lectura de cabecera safetensors")
p = write_lora("a.safetensors", {"modelspec.trigger_phrase": "clayantboss"})
check("cabecera legible", "__metadata__" in _read_header(p), True)

print("\nDeteccion de trigger")
r = LoraTriggerReader.execute("a.safetensors", "cualquiera", "", 0.8)
check("A. clave explicita modelspec", r[0], "clayantboss")
check("A. reporta el origen", "explicito" in r[1], True)

freq = {"10_clayantboss": {"clayantboss": 42, "1girl": 41, "smiling": 12, "outdoors": 5}}
write_lora("b.safetensors", {
    "ss_tag_frequency": json.dumps(freq),
    "ss_num_train_images": "42", "ss_network_dim": "32", "ss_network_alpha": "16",
    "ss_base_model_version": "flux2_klein",
})
r = LoraTriggerReader.execute("b.safetensors", "cualquiera", "", 0.8)
check("B. deducido del histograma", r[0], "clayantboss")
check("B. descarta stopword 1girl", "1girl" not in r[0], True)
check("B. detecta arquitectura flux", r[2], "flux")
check("B. reporta imagenes en el informe", "42" in r[3], True)

write_lora("c.safetensors", {})
r = LoraTriggerReader.execute("c.safetensors", "cualquiera", "", 0.8)
check("C. sin metadatos -> nombre archivo", r[0], "c")
check("C. avisa de que es suposicion", "AVISO" in r[3], True)

r = LoraTriggerReader.execute("c.safetensors", "cualquiera", "mitrigger", 0.8)
check("D. fallback manual gana al nombre", r[0], "mitrigger")

print("\nGuarda de arquitectura")
try:
    LoraTriggerReader.execute("b.safetensors", "ltx", "", 0.8)
    check("E. bloquea LoRA de otra familia", False, True)
except ValueError as e:
    check("E. bloquea LoRA de otra familia", "arquitectura" in str(e).lower(), True)

r = LoraTriggerReader.execute("b.safetensors", "flux", "", 0.8)
check("F. deja pasar la familia correcta", r[0], "clayantboss")

print("\nQualityPreset")
from comfy_character_pipeline.artist import QualityPreset, PipelineStatus  # noqa: E402
q = QualityPreset.execute("flux2-klein", "Normal", 0.0)
check("G. flux normal = 14 steps / 0.42", (q[0], q[1]), (14, 0.42))
check("H. resumen indica 6 pasos reales", "6 pasos reales" in q[5], True)
q = QualityPreset.execute("wan-2.2 (video)", "Normal", 0.0)
check("I. wan normal denoise 0.25", q[1], 0.25)
q = QualityPreset.execute("flux2-klein", "Normal", 0.15)
check("J. ajuste sube el denoise y avisa", (q[1], "AVISO" in q[5]), (0.57, True))

print("\nPipelineStatus")
s = PipelineStatus.execute("Estado", mask_coverage=0.35, mask_bad_frames=0,
                           sampler_steps=6, trigger="clayantboss")
check("K. todo ok", s[1], True)
s = PipelineStatus.execute("Estado", mask_coverage=0.0, mask_bad_frames=3)
check("L. detecta frames malos", s[1], False)
check("L. mensaje legible", "frame(s) vacios" in s[0], True)
s = PipelineStatus.execute("Estado", chunk_length=97, chunk_total=3, chunk_is_last=False)
check("M. avisa de trozos pendientes", "FALTAN TROZOS" in s[0], True)

print("\n" + ("TODO OK" if not FAILS else "%d FALLOS: %s" % (len(FAILS), FAILS)))
sys.exit(1 if FAILS else 0)

