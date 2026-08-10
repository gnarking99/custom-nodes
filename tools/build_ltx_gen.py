# -*- coding: utf-8 -*-
"""Genera el flujo `LTX23 GEN t2v-i2v-ref2v.json`.

Por que un generador y no editar el JSON a mano: un flujo de ComfyUI guarda cada
enlace TRES veces (en `links`, en `outputs[].links` del origen y en
`inputs[].link` del destino). Editar a mano deja incoherencias que el frontend
no siempre repara, y el fallo aparece a mitad de una cola de 20 minutos.

Las firmas de nodo (que entradas tiene cada tipo y en que orden) no estan
inventadas: salen de `ltx_node_shapes.json`, extraido de las plantillas
oficiales de Comfy-Org y de Lightricks. Si ComfyUI cambia un nodo, se vuelve a
extraer y se regenera.
"""

import json
import os
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
SHAPES = json.load(open(os.path.join(HERE, "ltx_node_shapes.json"), encoding="utf-8"))
DST = r"D:\Users\crist\Documents\GitHub\workflows\LTX23 GEN t2v-i2v-ref2v.json"

# --------------------------------------------------------------------------
# Modelos. Nombres tal y como aparecen en los desplegables; si tu carpeta usa
# subdirectorios, ComfyUI los muestra con la ruta relativa incluida.
# --------------------------------------------------------------------------
CKPT = "ltx-2.3-22b-dev-fp8.safetensors"
TEXT_ENCODER = "gemma_3_12B_it_fp4_mixed.safetensors"
UPSCALER = "ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
LORA_DISTILL = "ltx_2.3_22b_distilled_1.1_lora_dynamic_fro09_avg_rank_111_bf16.safetensors"
LORA_ICLORA = "ltx-2.3-22b-ic-lora-ingredients-0.9.safetensors"
LORA_CHAR = "Duck_ltx_2_3_H100_animation_safe_v01_000003500.safetensors"

PROMPT_POS = (
    "Scene: a stylized 3D animated character, Pixar-like shading, soft key light "
    "from the left with a warm rim light, subsurface scattering on the skin, clean "
    "cinematic depth of field, shallow background bokeh, crisp micro-detail on "
    "fabric and hair.\n\n"
    "Action: the character turns slowly toward the camera and smiles; the camera "
    "does a slow dolly-in. Natural weight and follow-through in the motion.\n\n"
    "Music: none."
)
PROMPT_NEG = (
    "pc game, console game, video game, cartoon, childish, ugly, plastic skin, "
    "flat lighting, blurry, out of focus, jagged edges, flickering, deformed hands, "
    "extra limbs, warped face, text, watermark"
)

SEED = 815253765879496

# --------------------------------------------------------------------------
# Constructor de grafo
# --------------------------------------------------------------------------

# Firmas de los nodos propios (no salen de las plantillas: son nuestros).
SHAPES["CP_LTXGenPlan"] = {
    "inputs": [{"localized_name": "imagen", "name": "imagen", "type": "IMAGE", "link": None, "shape": 7}],
    "outputs": [{"localized_name": n, "name": n, "type": t, "links": []} for n, t in [
        ("base_w", "INT"), ("base_h", "INT"), ("x2_w", "INT"), ("x2_h", "INT"),
        ("x4_w", "INT"), ("x4_h", "INT"), ("final_w", "INT"), ("final_h", "INT"),
        ("frames", "INT"), ("fps", "INT"), ("fps_f", "FLOAT"),
        ("usar_imagen", "BOOLEAN"), ("bypass_imagen", "BOOLEAN"),
        ("usar_referencia", "BOOLEAN"), ("usar_ultimo_frame", "BOOLEAN"),
        ("resumen", "STRING"),
    ]],
    "properties": {"Node name for S&R": "CP_LTXGenPlan"},
    "size": [400, 240],
    "widgets_values": None,
}
SHAPES["CP_LTXSigmas"] = {
    "inputs": [],
    "outputs": [{"localized_name": n, "name": n, "type": t, "links": []} for n, t in [
        ("sigmas", "SIGMAS"), ("pasos", "INT"),
        ("sigma_inicial", "FLOAT"), ("procedencia", "STRING"),
    ]],
    "properties": {"Node name for S&R": "CP_LTXSigmas"},
    "size": [380, 180],
    "widgets_values": None,
}
SHAPES["MarkdownNote"] = {
    "inputs": [], "outputs": [],
    "properties": {}, "size": [520, 300], "widgets_values": [""],
}

# Entradas opcionales que las plantillas de origen serializan como obligatorias.
# `shape: 7` es como ComfyUI marca "opcional" en el JSON; sin esto el validador
# de abajo se queja de cables que a proposito no van a ninguna parte.
OPTIONAL = {
    ("Power Lora Loader (rgthree)", "clip"),
    ("LTXVAddGuide", "attention_mask"),
    ("LTX2SamplingPreviewOverride", "latent_upscale_model"),
    ("LTX2SamplingPreviewOverride", "vae"),
    ("LTX2_NAG", "nag_cond_video"),
    ("LTX2_NAG", "nag_cond_audio"),
    ("SeedVR2LoadDiTModel", "torch_compile_args"),
    ("SeedVR2LoadVAEModel", "torch_compile_args"),
    ("ImageResizeKJv2", "mask"),
    ("VHS_VideoCombine", "audio"),
    ("VHS_VideoCombine", "meta_batch"),
    ("VHS_VideoCombine", "vae"),
    ("CreateVideo", "audio"),
}
for _t, _shape in SHAPES.items():
    for _i in _shape["inputs"]:
        if (_t, _i["name"]) in OPTIONAL:
            _i["shape"] = 7

nodes = []
links = []
_state = {"nid": 0, "lid": 0}
STAGE_OF = {}


def add(ntype, widgets=None, title=None, mode=0, stage="misc", size=None):
    shape = SHAPES[ntype]
    _state["nid"] += 1
    nid = _state["nid"]
    n = {
        "id": nid,
        "type": ntype,
        "pos": [0, 0],
        "size": list(size or shape["size"]),
        "flags": {},
        "order": nid,
        "mode": mode,
        "inputs": json.loads(json.dumps(shape["inputs"])),
        "outputs": json.loads(json.dumps(shape["outputs"])),
        "properties": dict(shape.get("properties") or {}),
        "widgets_values": widgets if widgets is not None else shape.get("widgets_values"),
    }
    n["properties"].setdefault("Node name for S&R", ntype)
    if title:
        n["title"] = title
    nodes.append(n)
    STAGE_OF[nid] = stage
    return nid


BY_ID = {}


def _n(nid):
    if not BY_ID:
        BY_ID.update({x["id"]: x for x in nodes})
    return BY_ID.get(nid) or next(x for x in nodes if x["id"] == nid)


def si(nid, name):
    for i, inp in enumerate(_n(nid)["inputs"]):
        if inp["name"] == name:
            return i
    raise KeyError(f"{_n(nid)['type']} no tiene entrada '{name}'. "
                   f"Tiene: {[i['name'] for i in _n(nid)['inputs']]}")


def so(nid, name):
    for i, out in enumerate(_n(nid)["outputs"]):
        if out["name"] == name:
            return i
    raise KeyError(f"{_n(nid)['type']} no tiene salida '{name}'. "
                   f"Tiene: {[o['name'] for o in _n(nid)['outputs']]}")


def wire(src, src_name, dst, dst_name):
    s, d = _n(src), _n(dst)
    oi, ii = so(src, src_name), si(dst, dst_name)
    ot, it = s["outputs"][oi]["type"], d["inputs"][ii]["type"]
    if ot != it and it not in ("*",) and ot not in ("*",) and it.split(",")[0] != ot:
        raise TypeError(f"tipo incompatible: {s['type']}.{src_name}:{ot} -> "
                        f"{d['type']}.{dst_name}:{it}")
    _state["lid"] += 1
    lid = _state["lid"]
    s["outputs"][oi].setdefault("links", [])
    s["outputs"][oi]["links"].append(lid)
    d["inputs"][ii]["link"] = lid
    links.append([lid, src, oi, dst, ii, ot])
    return lid


def note(text, title, stage="notas", size=(540, 300)):
    return add("MarkdownNote", [text], title=title, stage=stage, size=list(size))


BYPASS = 4

# ==========================================================================
# 0 · PANEL
# ==========================================================================
plan = add("CP_LTXGenPlan",
           ["i2v · imagen inicial", "vertical 1080x1920 (9:16)", "x2 · dos pasadas",
            "64 (recomendado con LoRA)", 5.0, 24, 0.60],
           title="PLAN — empieza por aqui", stage="panel", size=[420, 260])
plan_prev = add("PreviewAny", [None, None, None], title="plan · resumen", stage="panel",
                size=[420, 260])
wire(plan, "resumen", plan_prev, "source")

sig1 = add("CP_LTXSigmas", ["1 · oficial 8 pasos", 0.0, "", 20],
           title="SIGMAS etapa 1", stage="panel", size=[400, 180])
sig2 = add("CP_LTXSigmas", ["2 · oficial 3 pasos", 0.0, "", 20],
           title="SIGMAS etapa 2", stage="panel", size=[400, 180])
sig_prev = add("PreviewAny", [None, None, None], title="sigmas etapa 1 · de donde sale",
               stage="panel", size=[400, 300])
wire(sig1, "procedencia", sig_prev, "source")

# ==========================================================================
# 1 · CARGA
# ==========================================================================
ckpt = add("CheckpointLoaderSimple", [CKPT], title="LTX-2.3 22B", stage="carga")
tenc = add("LTXAVTextEncoderLoader", [TEXT_ENCODER, CKPT, "default"],
           title="Gemma-3 12B (fp4 mixed)", stage="carga")
avae = add("LTXVAudioVAELoader", [CKPT], title="VAE de audio", stage="carga")
upsl = add("LatentUpscaleModelLoader", [UPSCALER], title="upscaler LATENTE x2",
           stage="carga")

# ==========================================================================
# 2 · MODELO: LoRAs y parches
# ==========================================================================
iclora = add("LoraLoaderModelOnly", [LORA_ICLORA, 1.0],
             title="IC-LoRA Ingredients  ·  SOLO ref2v", mode=BYPASS, stage="modelo")
wire(ckpt, "MODEL", iclora, "model")

icpar = add("GetICLoRAParameters", [], title="parametros IC-LoRA", mode=BYPASS,
            stage="modelo")
wire(iclora, "MODEL", icpar, "iclora_model")

power = add("Power Lora Loader (rgthree)",
            [{}, {"type": "PowerLoraLoaderHeaderWidget"},
             {"on": True, "lora": LORA_CHAR, "strength": 0.7, "strengthTwo": None},
             {"on": False, "lora": LORA_CHAR, "strength": 0.6, "strengthTwo": None},
             {}, ""],
            title="LoRAs DE PERSONAJE", stage="modelo", size=[420, 160])
wire(iclora, "MODEL", power, "model")

distill = add("LoraLoaderModelOnly", [LORA_DISTILL, 0.5],
              title="LoRA de destilacion (8 pasos)", stage="modelo")
wire(power, "MODEL", distill, "model")

sage_kj = add("PathchSageAttentionKJ", ["auto", False],
              title="SageAttention  ·  elige UNA de las dos", mode=BYPASS, stage="modelo")
wire(distill, "MODEL", sage_kj, "model")

sage_mem = add("LTX2MemoryEfficientSageAttentionPatch", [True],
               title="SageAttention memoria-eficiente (LTX2)", mode=BYPASS, stage="modelo")
wire(sage_kj, "MODEL", sage_mem, "model")

chunkff = add("LTXVChunkFeedForward", [2, 4096],
              title="Chunk FeedForward  ·  solo si falta VRAM", mode=BYPASS, stage="modelo")
wire(sage_mem, "model", chunkff, "model")

tuner = add("LTX2AttentionTunerPatch", ["", 1, 1, 1, 1, True],
            title="Attention Tuner (video/audio)", stage="modelo")
wire(chunkff, "model", tuner, "model")

# ==========================================================================
# 3 · PROMPT
# ==========================================================================
pstr = add("PrimitiveStringMultiline", [PROMPT_POS], title="PROMPT  ·  Scene / Action / Music",
           stage="prompt", size=[480, 320])
penc = add("CLIPTextEncode", [""], title="positivo", stage="prompt", size=[400, 100])
wire(tenc, "CLIP", penc, "clip")
wire(pstr, "STRING", penc, "text")

nenc = add("CLIPTextEncode", [PROMPT_NEG], title="negativo  ·  lo usa NAG, no el CFG",
           stage="prompt", size=[400, 160])
wire(tenc, "CLIP", nenc, "clip")

cond = add("LTXVConditioning", [24], title="conditioning + frame rate", stage="prompt")
wire(penc, "CONDITIONING", cond, "positive")
wire(nenc, "CONDITIONING", cond, "negative")
wire(plan, "fps_f", cond, "frame_rate")

nag = add("LTX2_NAG", [11, 0.25, 2.5, True],
          title="NAG  ·  negativo de verdad con CFG 1", stage="modelo")
wire(tuner, "model", nag, "model")
wire(nenc, "CONDITIONING", nag, "nag_cond_video")
wire(nenc, "CONDITIONING", nag, "nag_cond_audio")

prev_ovr = add("LTX2SamplingPreviewOverride", [8],
               title="PREVIEW en vivo durante el muestreo", stage="modelo")
wire(nag, "model", prev_ovr, "model")
wire(upsl, "LATENT_UPSCALE_MODEL", prev_ovr, "latent_upscale_model")
wire(ckpt, "VAE", prev_ovr, "vae")

MODEL = (prev_ovr, "MODEL")

# ==========================================================================
# 4 · ENTRADAS
# ==========================================================================
img = add("LoadImage", ["example.png", "image"], title="PRIMER FRAME  ·  i2v",
          stage="entrada", size=[320, 320])
img_rs = add("ImageResizeKJv2",
             [1080, 1920, "lanczos", "crop", "0, 0, 0", "center", 64, "cpu"],
             title="encuadre a la entrega", stage="entrada")
wire(img, "IMAGE", img_rs, "image")
wire(plan, "final_w", img_rs, "width")
wire(plan, "final_h", img_rs, "height")

img_le = add("ResizeImagesByLongerEdge", [1536], title="techo de 1536 px", stage="entrada")
wire(img_rs, "IMAGE", img_le, "images")

img_pre = add("LTXVPreprocess", [18], title="compresion 18  ·  el life hack", stage="entrada")
wire(img_le, "images", img_pre, "image")

ref = add("LoadImage", ["example.png", "image"], title="HOJA DE REFERENCIA  ·  ref2v",
          stage="entrada", size=[320, 320])
ref_rs = add("ImageResizeKJv2",
             [1536, 1024, "lanczos", "pad", "255,255,255", "center", 32, "cpu"],
             title="hoja a 1536x1024", stage="entrada")
wire(ref, "IMAGE", ref_rs, "image")

# ==========================================================================
# 5 · ETAPA 1  (base, mitad de resolucion)
# ==========================================================================
lat = add("EmptyLTXVLatentVideo", [768, 512, 121, 1], title="lienzo BASE", stage="etapa1")
wire(plan, "base_w", lat, "width")
wire(plan, "base_h", lat, "height")
wire(plan, "frames", lat, "length")

alat = add("LTXVEmptyLatentAudio", [121, 24, 1], title="lienzo de audio", stage="etapa1")
wire(avae, "Audio VAE", alat, "audio_vae")
wire(plan, "frames", alat, "frames_number")
wire(plan, "fps", alat, "frame_rate")

inj1 = add("LTXVImgToVideoInplace", [0.7, False],
           title="inyecta el primer frame  ·  0.7", stage="etapa1")
wire(ckpt, "VAE", inj1, "vae")
wire(img_pre, "output_image", inj1, "image")
wire(lat, "LATENT", inj1, "latent")
wire(plan, "bypass_imagen", inj1, "bypass")

guide = add("LTXVAddGuide", [0, 1.0],
            title="GUIA de referencia  ·  SOLO ref2v", mode=BYPASS, stage="etapa1",
            size=[380, 200])
wire(cond, "positive", guide, "positive")
wire(cond, "negative", guide, "negative")
wire(ckpt, "VAE", guide, "vae")
wire(inj1, "latent", guide, "latent")
wire(ref_rs, "IMAGE", guide, "image")
wire(icpar, "iclora_parameters", guide, "iclora_parameters")

cat1 = add("LTXVConcatAVLatent", None, title="video + audio", stage="etapa1")
wire(guide, "latent", cat1, "video_latent")
wire(alat, "Latent", cat1, "audio_latent")

noise1 = add("RandomNoise", [SEED, "randomize"], title="ruido etapa 1", stage="etapa1")
ks1 = add("KSamplerSelect", ["euler_ancestral_cfg_pp"], title="sampler etapa 1", stage="etapa1")
g1 = add("CFGGuider", [1], title="guider etapa 1  ·  cfg 1", stage="etapa1")
wire(prev_ovr, "MODEL", g1, "model")
wire(guide, "positive", g1, "positive")
wire(guide, "negative", g1, "negative")

s1 = add("SamplerCustomAdvanced", None, title="ETAPA 1", stage="etapa1")
wire(noise1, "NOISE", s1, "noise")
wire(g1, "GUIDER", s1, "guider")
wire(ks1, "SAMPLER", s1, "sampler")
wire(sig1, "sigmas", s1, "sigmas")
wire(cat1, "latent", s1, "latent_image")

sep1 = add("LTXVSeparateAVLatent", None, title="separa v/a", stage="etapa1")
wire(s1, "output", sep1, "av_latent")

# ==========================================================================
# 6 · ETAPA 2  (upscale latente x2 + cola de sigmas)
# ==========================================================================
crop = add("LTXVCropGuides", None, title="quita los frames guia", stage="etapa2")
wire(guide, "positive", crop, "positive")
wire(guide, "negative", crop, "negative")
wire(sep1, "video_latent", crop, "latent")

ups = add("LTXVLatentUpsampler", None, title="UPSCALE LATENTE x2", stage="etapa2")
wire(crop, "latent", ups, "samples")
wire(upsl, "LATENT_UPSCALE_MODEL", ups, "upscale_model")
wire(ckpt, "VAE", ups, "vae")

inj2 = add("LTXVImgToVideoInplace", [1.0, False],
           title="reinyecta el primer frame  ·  1.0", stage="etapa2")
wire(ckpt, "VAE", inj2, "vae")
wire(img_pre, "output_image", inj2, "image")
wire(ups, "LATENT", inj2, "latent")
wire(plan, "bypass_imagen", inj2, "bypass")

cat2 = add("LTXVConcatAVLatent", None, title="video + audio", stage="etapa2")
wire(inj2, "latent", cat2, "video_latent")
wire(sep1, "audio_latent", cat2, "audio_latent")

noise2 = add("RandomNoise", [42, "fixed"], title="ruido etapa 2  ·  fijo", stage="etapa2")
ks2 = add("KSamplerSelect", ["euler_cfg_pp"], title="sampler etapa 2", stage="etapa2")
g2 = add("CFGGuider", [1], title="guider etapa 2", stage="etapa2")
wire(prev_ovr, "MODEL", g2, "model")
wire(crop, "positive", g2, "positive")
wire(crop, "negative", g2, "negative")

s2 = add("SamplerCustomAdvanced", None, title="ETAPA 2", stage="etapa2")
wire(noise2, "NOISE", s2, "noise")
wire(g2, "GUIDER", s2, "guider")
wire(ks2, "SAMPLER", s2, "sampler")
wire(sig2, "sigmas", s2, "sigmas")
wire(cat2, "latent", s2, "latent_image")

sep2 = add("LTXVSeparateAVLatent", None, title="separa v/a", stage="etapa2")
wire(s2, "output", sep2, "av_latent")

# ==========================================================================
# 7 · DECODE
# ==========================================================================
dec = add("LTXVTiledVAEDecode", [2, 2, 6, False, "auto", "auto"],
          title="decode por tiles", stage="salida")
wire(ckpt, "VAE", dec, "vae")
wire(sep2, "video_latent", dec, "latents")

adec = add("LTXVAudioVAEDecode", None, title="decode de audio", stage="salida")
wire(sep2, "audio_latent", adec, "samples")
wire(avae, "Audio VAE", adec, "audio_vae")

# ==========================================================================
# 8 · POST  ·  SeedVR2 (bypass por defecto)
# ==========================================================================
tcs = add("SeedVR2TorchCompileSettings", ["inductor", "default", False, False, 64, 128],
          title="torch.compile (opcional)", mode=BYPASS, stage="post")
sdit = add("SeedVR2LoadDiTModel",
           ["seedvr2_ema_3b_fp16.safetensors", "cuda:0", 0, False, "cpu", True, "sdpa"],
           title="SeedVR2 DiT 3B fp16", mode=BYPASS, stage="post")
wire(tcs, "TORCH_COMPILE_ARGS", sdit, "torch_compile_args")
svae = add("SeedVR2LoadVAEModel",
           ["ema_vae_fp16.safetensors", "cuda:0", False, 1024, 128, False, 768, 128,
            "false", "cpu", True],
           title="SeedVR2 VAE", mode=BYPASS, stage="post")
wire(tcs, "TORCH_COMPILE_ARGS", svae, "torch_compile_args")

sup = add("SeedVR2VideoUpscaler",
          [42, "fixed", 1920, 0, 49, True, "lab", 5, 0, 0, 0, "cpu", False],
          title="SeedVR2  ·  el salto final de resolucion", mode=BYPASS, stage="post")
wire(dec, "image", sup, "image")
wire(sdit, "SEEDVR2_DIT", sup, "dit")
wire(svae, "SEEDVR2_VAE", sup, "vae")

gis = add("GetImageSizeAndCount", [], title="medida y numero de frames", stage="post")
wire(sup, "IMAGE", gis, "image")

vid = add("CreateVideo", [24], title="monta video + audio", stage="post")
wire(gis, "image", vid, "images")
wire(adec, "Audio", vid, "audio")
wire(plan, "fps_f", vid, "fps")

save = add("SaveVideo", ["video/LTX23-GEN", "auto", "auto"], title="GUARDAR", stage="post")
wire(vid, "VIDEO", save, "video")

vhs = add("VHS_VideoCombine",
          {"frame_rate": 24, "loop_count": 0, "filename_prefix": "LTX23-GEN",
           "format": "video/h264-mp4", "pix_fmt": "yuv420p", "crf": 19,
           "save_metadata": True, "trim_to_audio": False, "pingpong": False,
           "save_output": False},
          title="previsualizacion con audio", stage="post", size=[420, 420])
wire(gis, "image", vhs, "images")
wire(adec, "Audio", vhs, "audio")
wire(plan, "fps_f", vhs, "frame_rate")

# ==========================================================================
# NOTAS
# ==========================================================================
note("""# Como se usa

## 1. Elige el modo en **PLAN**
| modo | que conectas | que bypass tocas |
|---|---|---|
| **t2v** | nada | ninguno: `bypass_imagen` ya lo apaga solo |
| **i2v** | `PRIMER FRAME` | ninguno |
| **ref2v** | `HOJA DE REFERENCIA` | enciende el grupo `IC-LoRA` (3 nodos) |
| **i2v + ref2v** | las dos | igual que ref2v |

El nodo `PLAN` reparte medidas, frames y booleanas. **No hay ningun numero de
resolucion escrito a mano en el grafo.**

## 2. Lee el resumen antes de lanzar
`plan · resumen` te dice la resolucion real de cada etapa, los frames exactos y
los avisos. Un aviso ahi cuesta 3 segundos; el mismo fallo en la cola cuesta la
ejecucion entera.

## 3. Lanza
La primera vez, con el LoRA de personaje a 0.7 y las sigmas oficiales. Solo
cuando eso funcione empieza a mover cosas — y mueve **una**.""",
     "LEEME — arranque rapido", size=(560, 620))

note("""# La estructura de dos etapas

```
lienzo BASE (mitad de la entrega)
        |
   ETAPA 1  ·  8 pasos  ·  euler_ancestral_cfg_pp  ·  cfg 1
        |
   UPSCALE LATENTE x2   <- sin decodificar a pixeles
        |
   ETAPA 2  ·  3 pasos  ·  euler_cfg_pp  ·  cfg 1
        |
   decode por tiles  ->  video + audio
```

**La etapa 1 genera a la MITAD de la entrega, no a la entrega.** Es el error mas
caro que se puede cometer aqui: poner 1080x1920 en el lienzo base cuadruplica el
coste del paso lento y encima saca al upscaler de su rango.

**El upscale es latente**, no pixel space. No hay round-trip de VAE entre las dos
etapas — sobre 121 frames eso es la diferencia entre iterar y esperar.

**La etapa 2 no es un refinado independiente**: es la cola de la misma curva de
sigmas, aplicada al latente ya duplicado. Por eso encaja sin costura.

## Por que no hay etapa x4
La atencion es O(n^2) sobre los tokens. Una tercera etapa tiene 4 veces los
pixeles de la segunda, o sea **~16 veces su coste**, para darte 2-3 pasos de
refinado sobre algo ya refinado. Y a partir de x2 el upscaler espacial **afila,
no reconstruye**. Ese salto lo hace mejor SeedVR2 al final del flujo.""",
     "Arquitectura — por que dos etapas y no tres", size=(560, 660))

note("""# Sigmas: la palanca que mas cambia el resultado

LTX-2.3 no usa `steps` + `denoise`. Usa una **lista explicita**. El nodo
`SIGMAS` trae las curvas con su procedencia.

## Etapa 1
- **`1 · oficial 8 pasos`** — la de las plantillas de Comfy-Org y Lightricks.
  Empieza aqui siempre.
- **`1 · movimiento + LoRA 8 pasos`** — cae en picado en el paso 3. La tesis de
  la comunidad: *LTX cocina el video en 2-3 pasos, y todo el denoise alto
  posterior aplana el movimiento y diluye el LoRA de personaje*.

  **Este es el preset para el sintoma "mi personaje sale perfecto pero no se
  mueve".** Es el problema numero uno de los LoRA de personaje en LTX-2.3.

## Etapa 2
- **`2 · oficial 3 pasos`** (0.85 -> 0) por defecto.
- Si la etapa 2 te **cambia la cara**, no cambies de sampler: baja `inicio` a
  0.5-0.6. La comunidad coincide en que un solver de 2o orden en la segunda
  pasada sale duro, y lo correcto es bajar la sigma de arranque.
- Si quieres **mas detalle inventado**, sube `inicio` a 0.9.""",
     "Sigmas — donde se gana y se pierde el movimiento", size=(560, 640))

note("""# LoRA de personaje

## Fuerza
**0.55-0.70.** Por debajo de 0.55 la identidad no aguanta; por encima de 0.8 el
personaje se congela y deja de obedecer al prompt de accion.

Empieza en **0.7** y baja si pierdes movimiento. Si a 0.55 sigues sin movimiento,
el problema no es la fuerza: son las sigmas (cambia al preset de movimiento).

## Orden de la cadena
```
checkpoint -> IC-LoRA -> PERSONAJE -> destilacion -> parches -> NAG
```
IC-LoRA primero porque define el *modo de trabajo* del modelo; el personaje
despues porque aporta identidad; la destilacion al final porque solo cambia
cuantos pasos hacen falta.

## Multiplo de 64
Con LoRA de personaje, las medidas que no son multiplo de 64 pierden movimiento.
1280x704 si, 1280x720 no. El nodo `PLAN` ya lo fuerza.

## Palanca fina que casi nadie usa
`LTX2LoRALoaderAdvanced` (KJNodes) aplica el LoRA **por modalidad**: puedes
ponerlo a 1.0 en las capas de video y a 0 en las de audio y en las cruzadas
audio<->video. Un LoRA de personaje entrenado solo con imagenes no tiene nada
que decir sobre el audio, y dejarlo actuar ahi es parte de por que el
movimiento se apaga. Sustituye el `Power Lora Loader` por ese nodo cuando
quieras exprimirlo.""",
     "LoRA de personaje — fuerza, orden y el ajuste fino", size=(560, 700))

note("""# NAG: prompt negativo de verdad con CFG 1

El modelo destilado **no responde al CFG negativo**. Con `cfg = 1` el negativo
simplemente no se aplica: el nodo esta ahi, el texto esta escrito, y no hace
nada. Es el fallo silencioso mas comun de todo el flujo.

**NAG** (Normalized Attention Guidance) lo arregla por otra via: en vez de tocar
la guia, parchea la **atencion**. Por eso funciona a CFG 1.

- `nag_scale` **11** = equilibrado. 6 = sutil. 25 = maximo (empieza a deformar).
- `nag_alpha` 0.25 · `nag_tau` 2.5 — dejalos.
- Las dos entradas de conditioning van al **negativo**, antes de
  `LTXVConditioning`.

Si pones `nag_scale = 0` el nodo se desactiva solo, sin bypass.""",
     "NAG — por que el negativo no funcionaba", size=(540, 460))

note("""# `LTXVPreprocess` a 18 — el life hack que no es obvio

Mete compresion a la imagen de entrada **a proposito**.

LTX se entreno con video comprimido. Una referencia demasiado limpia — y un
render 3D lo es — le dice al modelo *"esto ya esta bien"*, y se queda quieto en
vez de sintetizar. Es comportamiento documentado de la familia LTX.

- **18** es el valor de las plantillas oficiales.
- Sube a **25-35** si el resultado sale demasiado pegado a tu imagen.
- Baja a **5-10** si se inventa demasiado.

Si tus fuentes son renders limpios de Pixar-style, este nodo es lo que
desbloquea que el modelo se atreva.""",
     "El preprocess — compresion a proposito", size=(540, 420))

note("""# ref2v: hoja de referencia con IC-LoRA Ingredients

Enciende los tres nodos en bypass: `IC-LoRA Ingredients`, `parametros IC-LoRA` y
`GUIA de referencia`.

## Como funciona
La hoja **no entra por el latente**: entra por la atencion, como frames guia que
`LTXVCropGuides` retira despues. Por eso puedes usar una hoja de 1536x1024 con
un video vertical de 1080x1920 sin deformar nada.

## Como se monta la hoja
4-6 vistas del personaje en **una sola imagen**: cara en primer plano, cuerpo
entero de frente, de perfil, de espaldas, y los props que salgan en el plano.
Fondo blanco. El resize del flujo ya la pone a 1536x1024 con relleno blanco.

## Alternativas al Ingredients, por caso
| quieres | LoRA |
|---|---|
| varios personajes + fondo en el mismo plano | **Licon-MSR V2** (`ComfyUI-Licon-MSR`) |
| identidad de cara con precision | **Best-Face-ID** (`Alissonerdx/LTX-Best-Face-ID`) — el prompt DEBE empezar por `ref_t2v:` |
| hoja de personaje + ropa | **Best-Face-ID CharacterSheet** — la hoja debe ser exactamente 1536x1024 |
| pose o camara de un video de referencia | **IC-LoRA Union Control** / **IC-Cameraman** |

## Y el LoRA entrenado, ¿sigue haciendo falta?
Si. La hoja da parecido; el LoRA da **identidad estable a lo largo de 121
frames**. Se apilan: es el uso previsto, no un hack, y las IC-LoRA cuestan menos
del 10% de computo.""",
     "ref2v — hoja de referencia y sus variantes", size=(560, 720))

note("""# Prompt: Scene / Action / Music

Gemma-3 12B es un LLM, no un tokenizador de etiquetas. **Frases completas.**

Estructurar el prompt en tres secciones explicitas recupera buena parte de la
obediencia que se pierde al meter un LoRA de personaje.

- **Scene** — materiales, luz, nitidez, lente. Aqui va el 90% del esfuerzo.
- **Action** — que hace el personaje y que hace la camara. Una accion, no tres.
- **Music** — `none` si no quieres que se invente audio. LTX-2.3 genera audio y
  video **en la misma pasada**; si no lo dices, se inventa una banda sonora.

**La palabra de disparo del LoRA va al principio de Scene.** El nodo
`LoRA Trigger Reader` del pack la saca del propio `.safetensors`.

## Atajo
El nodo core `TextGenerateLTX2Prompt` genera el prompt **a partir de tu imagen
de referencia** usando el mismo Gemma que ya tienes cargado. Para una hoja de
personaje describe ropa, colores y props mejor de lo que los escribirias tu.""",
     "Prompt — estructura y trigger word", size=(540, 560))

note("""# Velocidad y VRAM en tu hardware

## RTX PRO 6000 Blackwell (96 GB) — tu mejor opcion
- **NVFP4**: existe `Lightricks/LTX-2.3-nvfp4`. Hasta ~2x sobre fp8, **pero solo
  si PyTorch esta compilado con CUDA 13.0**. Con cu12 el mismo modelo puede ir
  hasta 2x MAS LENTO que fp8. Comprueba `torch.version.cuda` antes de cambiar.
- Con 96 GB no necesitas `Chunk FeedForward` ni block swap. Dejalos en bypass:
  el chunking cambia ligeramente la salida.
- El encoder Gemma en `fp4_mixed` ya esta puesto.

## SageAttention
Enciende **una sola** de las dos:
- `PathchSageAttentionKJ` — la general.
- `LTX2MemoryEfficientSageAttentionPatch` — especifica de LTX2, reduce el pico
  de VRAM. Requiere sageattention reciente y Triton.

En Blackwell (sm_120) la instalacion es la parte fragil: PyTorch estable tardo
en dar soporte y hay reportes de ruido con builds antiguas. Si ves ruido, lo
primero es actualizar ComfyUI, no bajar la calidad.

## H100 / L40S / A100
- **H100**: fp8 nativo + SageAttention 2. Va muy bien, sin NVFP4.
- **L40S** (48 GB): suficiente para este flujo tal cual.
- **A100**: **no tiene unidades fp8**. El checkpoint `-fp8` se dequantiza a bf16
  y pierdes la ventaja. Si acabas ahi, usa el bf16 completo y baja la base.

## Sobre `Nvidia_RTX_Nodes_ComfyUI`
El nodo *RTX Video Super Resolution* que preguntabas: hay un issue **abierto y
sin respuesta** de salida en negro precisamente en **RTX PRO 6000 Blackwell**
(driver abierto, CUDA 13.1). No lo pongas como pieza critica del flujo. SeedVR2
hace ese trabajo y esta probado en tu tarjeta.""",
     "Hardware — NVFP4, SageAttention y el nodo de NVIDIA", size=(560, 780))

note("""# Salida: por que SeedVR2 y no una tercera etapa latente

El grupo de SeedVR2 esta en bypass. Enciendelo cuando quieras pasar de la
entrega nativa (2x la base) a 4K.

- **3B fp16, no el 7B.** Contraintuitivo con 96 GB, pero es lo que reportan los
  usuarios con mas horas: el 3B da mejor calidad. Y fp16 en vez de GGUF porque
  te sobra VRAM.
- **`blocks_to_swap = 0`** — el block swap es para tarjetas pequenas; aqui es
  puro coste.
- **`cache_model = True`** en DiT y VAE: los deja residentes entre ejecuciones.
  Es la mayor ganancia de tiempo del grupo.
- **color correction `lab`**.

## El truco contraintuitivo
Bajar el material antes de subirlo funciona mejor que estirarlo. Con menos
pixeles de entrada el restaurador **reconstruye** en vez de afilar. Si vas a 4K,
prueba a alimentar SeedVR2 con la salida de la etapa 1 en vez de la etapa 2 y
compara.

## Interpolacion
Va **al final**, nunca antes. `ComfyUI-Frame-Interpolation` -> RIFE o GIMM-VFI,
de 24 a 48 fps. Interpolar antes de mejorar duplica el coste y propaga errores.""",
     "SeedVR2 e interpolacion — el remate", size=(560, 620))

note("""# Nodos que este flujo necesita

| pack | para que |
|---|---|
| **ComfyUI** al dia | casi todo lo de `LTXV*` es nativo desde 0.22 |
| **Lightricks/ComfyUI-LTXVideo** | `LTXVTiledVAEDecode` (el decode del final) |
| **ComfyUI-KJNodes** | NAG, Attention Tuner, preview en vivo, resize, chunk FF |
| **rgthree-comfy** | Power Lora Loader |
| **ComfyUI-VideoHelperSuite** | previsualizacion con audio |
| **ComfyUI-SeedVR2_VideoUpscaler** | el remate a 4K |
| **comfyui-character-pipeline** (el tuyo) | `PLAN` y `SIGMAS` |

> Si prefieres no depender del pack de Lightricks, sustituye `LTXVTiledVAEDecode`
> por el `VAEDecodeTiled` de core con `768, 64, 4096, 4`. Es lo que usa el
> blueprint oficial de ComfyUI.

## Lo que ese mismo pack trae y aqui no se usa
STG (guia espacio-temporal), sampler con tiles, sampler en bucle para clips
largos, `LTXVImgToVideoConditionOnly` — alternativa a `Inplace` que **no clava**
el primer frame en el latente y suele dar mas movimiento; si tu i2v sale rigido,
es lo primero que hay que probar — y el catalogo entero de IC-LoRA.

## Packs que NO usa este flujo pero que deberias conocer

- **`TenStrip/10S-Comfy-nodes`** — el mas infravalorado. *LTX Tiled Sampler*
  muestrea la pasada de upscale por tiles **al numero de tokens con el que se
  entreno el modelo**; es la forma correcta de pasar de 2K sin que aparezcan
  texturas repetidas. Ademas *Face Identity Reinforcer* y *Latent Anchor*.
- **`Winnougan/ComfyUI_WLTX_nodes`** — presets de sigmas y, sobre todo, **cache
  de conditioning**: guarda el encoding de Gemma en disco y lo recarga sin
  arrancar el encoder. Iterando sobre el mismo prompt es tiempo y VRAM gratis.
- **`liconstudio/ComfyUI-Licon-MSR`** — multi-personaje con fondo.
- **`alisson-anjos/ComfyUI-BFSNodes`** — identidad por overlap + ArcFace.
- **`WhatDreamsCost-ComfyUI`** — *LTX Director*: linea de tiempo con keyframes,
  audio y extension de clip dentro de un solo nodo.
- **`Deno2026/comfyui-deno-custom-nodes`** — upscaler espacial latente **por
  tiles** y wrapper de NVIDIA VFX.
- **`Sunwood-ai-labs/ComfyUI-LTXLongAudio`** — clips largos con audio.""",
     "El ecosistema — que instalar y que existe ahi fuera", size=(580, 780))

note("""# Que NO he verificado

- **No he ejecutado nada.** No tengo las GPUs ni los modelos aqui. Lo que si
  esta verificado: la topologia sale de las plantillas oficiales
  (`Comfy-Org/workflow_templates` blueprints `text_to_video_ltx_2_3` /
  `image_to_video_ltx_2_3` y `Lightricks/ComfyUI-LTXVideo`
  `LTX-2.3_T2V_I2V_Two_Stage_Distilled`), y el grafo se genera con las firmas de
  nodo extraidas de esos mismos JSON, con validacion de tipos en cada enlace.

- **Los nombres de archivo de modelo y LoRA son desplegables.** Si tu carpeta no
  tiene exactamente ese nombre, ComfyUI marcara el nodo en rojo: selecciona el
  que tengas. `Duck_ltx_2_3_H100_animation_safe_v01_000003500` es el LoRA que ya
  aparecia en tus flujos de enhance.

- **La curva `movimiento + LoRA`** es de un hilo de HuggingFace, no una medicion
  mia. Ademas la lista original acababa en 0.18 sin cerrar en 0.0; se le anade
  el 0.0 porque el sampler lo necesita, y eso la cambia ligeramente respecto a
  lo que escribio su autor.

- **`nag_scale = 11`** es el valor por defecto del nodo y el que usan los flujos
  de la comunidad. No lo he calibrado sobre tu material.

- **NVFP4 en tu tarjeta**: los reportes de velocidad se contradicen entre si
  segun el stack. Mide tu caso antes de migrar el flujo entero.

- **`LTXVAddGuide` con `iclora_parameters`**: cableado como en la plantilla
  oficial de Ingredients. Los tres nodos van juntos en bypass; encender solo uno
  da un fallo confuso.""",
     "Honestidad — lo que no esta comprobado", size=(560, 700))

# ==========================================================================
# LAYOUT
# ==========================================================================
GRAPH = [n for n in nodes if n["type"] != "MarkdownNote"]
NOTES = [n for n in nodes if n["type"] == "MarkdownNote"]
GIDS = {n["id"] for n in GRAPH}

# Cada etapa ocupa su propia banda horizontal. Colocar por profundidad
# topologica global mezcla etapas en la misma columna y los grupos acaban
# solapandose unos con otros, que es peor que un cable largo.
STAGE_ORDER = [
    ("panel", "0 · PANEL DEL ARTISTA — empieza aqui", "#2a6"),
    ("carga", "1 · CARGA", "#3f789e"),
    ("prompt", "2 · PROMPT", "#a1309b"),
    ("modelo", "3 · MODELO · LoRAs, parches, NAG", "#b58b2a"),
    ("entrada", "4 · ENTRADAS · primer frame / hoja de referencia", "#8a5a3c"),
    ("etapa1", "5 · ETAPA 1 · base", "#2a6"),
    ("etapa2", "6 · ETAPA 2 · upscale latente x2", "#A88"),
    ("salida", "7 · DECODE", "#8A8"),
    ("post", "8 · POST · SeedVR2 y entrega", "#555"),
]

preds = defaultdict(set)
for lid, on, osl, tn, tsl, ty in links:
    if on in GIDS and tn in GIDS and on != tn:
        preds[tn].add(on)

COL_GAP, ROW_GAP, STAGE_GAP, PAD = 110, 46, 240, 55
ROW_MAX_X = 6200          # ancho maximo de una fila de etapas antes de bajar
bounds = {}
groups = []
x_cursor = 0
y_cursor = 0
row_h = 0

for stage, title, color in STAGE_ORDER:
    members = [n for n in GRAPH if STAGE_OF[n["id"]] == stage]
    if not members:
        continue
    mids = {n["id"] for n in members}

    local = {}

    def ldepth(nid, stack):
        if nid in local:
            return local[nid]
        if nid in stack:
            return 0
        stack.add(nid)
        p = [q for q in preds.get(nid, ()) if q in mids]
        local[nid] = 0 if not p else 1 + max(ldepth(q, stack) for q in p)
        stack.discard(nid)
        return local[nid]

    for n in members:
        ldepth(n["id"], set())

    cols = defaultdict(list)
    for n in members:
        cols[local[n["id"]]].append(n)

    stage_w = sum(max(n["size"][0] for n in cols[c]) + COL_GAP for c in cols)
    if x_cursor > 0 and x_cursor + stage_w > ROW_MAX_X:
        x_cursor = 0
        y_cursor += row_h + STAGE_GAP + 90
        row_h = 0

    stage_x0 = x_cursor
    x = x_cursor
    for c in sorted(cols):
        def key(n):
            p = [q for q in preds.get(n["id"], ()) if q in bounds]
            return sum(bounds[q][1] for q in p) / len(p) if p else 0
        y = y_cursor
        for n in sorted(cols[c], key=key):
            n["pos"] = [x, y]
            bounds[n["id"]] = (x, y, n["size"][0], n["size"][1])
            y += n["size"][1] + ROW_GAP
        x += max(n["size"][0] for n in cols[c]) + COL_GAP
        row_h = max(row_h, y - y_cursor)

    x1 = stage_x0
    x2 = max(n["pos"][0] + n["size"][0] for n in members)
    y1 = min(n["pos"][1] for n in members)
    y2 = max(n["pos"][1] + n["size"][1] for n in members)
    groups.append({
        "id": len(groups) + 1, "title": title,
        "bounding": [x1 - PAD, y1 - PAD - 32, x2 - x1 + PAD * 2, y2 - y1 + PAD * 2],
        "color": color, "font_size": 26, "flags": {},
    })
    x_cursor = x2 + STAGE_GAP

# notas en una columna a la derecha de todo
note_x = max(g["bounding"][0] + g["bounding"][2] for g in groups) + 200
y = 0
for n in NOTES:
    n["pos"] = [note_x, y]
    y += n["size"][1] + ROW_GAP
notes_h = y
groups.append({
    "id": len(groups) + 1, "title": "DOCUMENTACION",
    "bounding": [note_x - PAD, -PAD - 32,
                 max(n["size"][0] for n in NOTES) + PAD * 2, notes_h + PAD],
    "color": "#444", "font_size": 26, "flags": {},
})

# orden de ejecucion sugerido: por etapa y, dentro de la etapa, por profundidad
STAGE_RANK = {s: i for i, (s, _, _) in enumerate(STAGE_ORDER)}
for n in GRAPH:
    n["order"] = STAGE_RANK.get(STAGE_OF[n["id"]], 99) * 1000 + n["id"]

wf = {
    "id": "ltx23-gen-t2v-i2v-ref2v",
    "revision": 0,
    "last_node_id": _state["nid"],
    "last_link_id": _state["lid"],
    "nodes": nodes,
    "links": links,
    "groups": groups,
    "config": {},
    "extra": {
        "ds": {"scale": 0.5, "offset": [120, 140]},   # abre encima del PANEL
        "groupNodes": {},
        "VHS_latentpreview": True,
        "VHS_latentpreviewrate": 8,
        "VHS_KeepIntermediate": True,
    },
    "version": 0.4,
}

# ==========================================================================
# VALIDACION
# ==========================================================================
errs = []
by = {n["id"]: n for n in nodes}
seen = set()
for lid, on, osl, tn, tsl, ty in links:
    if lid in seen:
        errs.append(f"link {lid} duplicado")
    seen.add(lid)
    o, t = by[on], by[tn]
    if osl >= len(o["outputs"]):
        errs.append(f"link {lid}: slot de salida {osl} fuera de rango en {o['type']}")
        continue
    if tsl >= len(t["inputs"]):
        errs.append(f"link {lid}: slot de entrada {tsl} fuera de rango en {t['type']}")
        continue
    if lid not in (o["outputs"][osl].get("links") or []):
        errs.append(f"link {lid}: no aparece en outputs de {o['type']}")
    if t["inputs"][tsl].get("link") != lid:
        errs.append(f"link {lid}: no aparece en inputs de {t['type']}")
    if o["outputs"][osl]["type"] != ty:
        errs.append(f"link {lid}: tipo {ty} != {o['outputs'][osl]['type']}")

# entradas obligatorias sin conectar (las que no tienen widget detras)
for n in GRAPH:
    for inp in n["inputs"]:
        if inp.get("link") is None and "widget" not in inp and inp.get("shape") != 7:
            errs.append(f"{n['type']} [{n['id']}]: entrada '{inp['name']}' sin conectar")

overlap = 0
items = list(bounds.items())
for i in range(len(items)):
    _, (x1, y1, w1, h1) = items[i]
    for j in range(i + 1, len(items)):
        _, (x2, y2, w2, h2) = items[j]
        if x1 < x2 + w2 and x2 < x1 + w1 and y1 < y2 + h2 and y2 < y1 + h1:
            overlap += 1

json.dump(wf, open(DST, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

print(f"nodos: {len(nodes)}  (grafo {len(GRAPH)}, notas {len(NOTES)})")
print(f"enlaces: {len(links)}   grupos: {len(groups)}")
print(f"solapes: {overlap}")
print(f"ERRORES: {len(errs)}")
for e in errs[:20]:
    print("   -", e)
print("->", DST)
