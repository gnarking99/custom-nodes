# -*- coding: utf-8 -*-
"""Genera el flujo `WAN22 GEN t2v-i2v-ref2v.json`.

Mismo criterio que `build_ltx_gen.py`: las firmas de nodo salen de
`wan_node_shapes.json`, extraido de los blueprints oficiales de Comfy-Org
(`text_to_video_wan_2_2`, `image_to_video_wan_2_2`, `video_inpaint_wan2_1_vace`)
y de los propios flujos del repo, y el script valida tipos, slots y entradas
obligatorias antes de escribir nada.
"""

import json
import os
import re
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
SHAPES = json.load(open(os.path.join(HERE, "wan_node_shapes.json"), encoding="utf-8"))
DST = r"D:\Users\crist\Documents\GitHub\workflows\WAN22 GEN t2v-i2v-ref2v.json"

# --------------------------------------------------------------------------
# Modelos
# --------------------------------------------------------------------------
UNET_HIGH = "wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors"
UNET_LOW = "wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors"
CLIP_NAME = "umt5_xxl_fp8_e4m3fn_scaled.safetensors"
VAE_NAME = "wan_2.1_vae.safetensors"
# El "viejo" 2.1: lo que la comunidad sigue recomendando en LOW, y a fuerza 3 en HIGH
SPEED_21 = "lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors"
# El 1030, el mas recomendado especificamente para 2.2 I2V HIGH
SPEED_1030 = "Wan_2_2_I2V_A14B_HIGH_lightx2v_4step_lora_v1030_rank_64_bf16.safetensors"
LORA_CHAR_HIGH = "TU_PERSONAJE_wan22_high.safetensors"
LORA_CHAR_LOW = "TU_PERSONAJE_wan22_low.safetensors"

PROMPT_POS = (
    "A stylized 3D animated character in a Pixar-like feature film. Soft key light "
    "from the left with a warm rim light, subsurface scattering on the skin, fine "
    "fabric weave and stitching, crisp silhouette edges, shallow cinematic depth of "
    "field. The character turns slowly toward the camera and smiles, then the camera "
    "does a slow dolly-in. Natural weight, overlapping action and follow-through."
)
# El negativo oficial de Wan, en chino, tal cual viene en las plantillas de Comfy-Org.
PROMPT_NEG = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
    "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，"
    "画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，"
    "杂乱的背景，三条腿，背景人很多，倒着走"
)

SEED = 421337

# --------------------------------------------------------------------------
OPTIONAL = {
    ("Power Lora Loader (rgthree)", "clip"),
    ("WanImageToVideo", "clip_vision_output"),
    ("WanImageToVideo", "start_image"),
    ("WanPhantomSubjectToVideo", "images"),
    ("ImageResizeKJv2", "mask"),
    ("VHS_VideoCombine", "audio"),
    ("VHS_VideoCombine", "meta_batch"),
    ("VHS_VideoCombine", "vae"),
    ("CreateVideo", "audio"),
    ("SeedVR2LoadDiTModel", "torch_compile_args"),
    ("SeedVR2LoadVAEModel", "torch_compile_args"),
    ("ComfyMathExpression", "values.b"),
    ("ComfyMathExpression", "values.c"),
}
for _t, _s in SHAPES.items():
    for _i in _s["inputs"]:
        if (_t, _i["name"]) in OPTIONAL:
            _i["shape"] = 7

nodes, links = [], []
_state = {"nid": 0, "lid": 0}
STAGE_OF = {}
BYPASS = 4


def add(ntype, widgets=None, title=None, mode=0, stage="misc", size=None):
    shape = SHAPES[ntype]
    _state["nid"] += 1
    nid = _state["nid"]
    n = {
        "id": nid, "type": ntype, "pos": [0, 0],
        "size": list(size or shape["size"]), "flags": {}, "order": nid, "mode": mode,
        "inputs": json.loads(json.dumps(shape["inputs"])),
        "outputs": json.loads(json.dumps(shape["outputs"])),
        "properties": dict(shape.get("properties") or {}),
    }
    wv = widgets if widgets is not None else shape.get("widgets_values")
    if wv is not None:
        n["widgets_values"] = wv
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
    raise KeyError(f"{_n(nid)['type']} no tiene entrada '{name}': {[i['name'] for i in _n(nid)['inputs']]}")


def so(nid, name):
    for i, out in enumerate(_n(nid)["outputs"]):
        if out["name"] == name:
            return i
    raise KeyError(f"{_n(nid)['type']} no tiene salida '{name}': {[o['name'] for o in _n(nid)['outputs']]}")


def wire(src, src_name, dst, dst_name):
    s, d = _n(src), _n(dst)
    oi, ii = so(src, src_name), si(dst, dst_name)
    ot, it = s["outputs"][oi]["type"], d["inputs"][ii]["type"]
    ok = (ot == it or "*" in (ot, it) or it.split(",")[0] == ot
          or ot in it.split(",") or (ot, it) == ("INT", "FLOAT,INT"))
    if not ok:
        raise TypeError(f"tipo incompatible: {s['type']}.{src_name}:{ot} -> {d['type']}.{dst_name}:{it}")
    _state["lid"] += 1
    lid = _state["lid"]
    s["outputs"][oi].setdefault("links", []).append(lid)
    d["inputs"][ii]["link"] = lid
    links.append([lid, src, oi, dst, ii, ot])
    return lid


def retype(nid, tipo, slots=("on_false", "on_true", "output")):
    """`ComfySwitchNode` es de tipo dinamico: sus slots adoptan el tipo de lo que
    le conectas. La firma extraida trae el tipo del flujo de origen (LATENT), asi
    que hay que fijarlo al que toque antes de cablear."""
    n = _n(nid)
    for coll in ("inputs", "outputs"):
        for s in n[coll]:
            if s["name"] in slots:
                s["type"] = tipo


def note(text, title, stage="notas", size=(560, 320)):
    return add("MarkdownNote", [text], title=title, stage=stage, size=list(size))


# ==========================================================================
# 0 · PANEL
# ==========================================================================
plan = add("CP_WanGenPlan",
           ["i2v · imagen inicial", "vertical 720x1280 (9:16)", 5.0, 16, 0, 16, 81],
           title="PLAN — modo, entrega y tandas de 81", stage="panel", size=[430, 300])
plan_prev = add("PreviewAny", [], title="plan · resumen", stage="panel", size=[430, 300])
wire(plan, "resumen", plan_prev, "source")

# ==========================================================================
# 1 · CARGA
# ==========================================================================
unet_h = add("UNETLoader", [UNET_HIGH, "default"], title="Wan 2.2 · experto HIGH NOISE", stage="carga")
unet_l = add("UNETLoader", [UNET_LOW, "default"], title="Wan 2.2 · experto LOW NOISE", stage="carga")
clip = add("CLIPLoader", [CLIP_NAME, "wan", "default"], title="umT5-XXL", stage="carga")
vae = add("VAELoader", [VAE_NAME], title="VAE de Wan 2.1", stage="carga")

# ==========================================================================
# 2 · MODELO HIGH
# ==========================================================================
speed_h = add("LoraLoaderModelOnly", [SPEED_21, 3.0],
              title="velocidad HIGH · 2.1 a fuerza 3", mode=BYPASS, stage="modelo_h")
wire(unet_h, "MODEL", speed_h, "model")
char_h = add("Power Lora Loader (rgthree)",
             [{}, {"type": "PowerLoraLoaderHeaderWidget"},
              {"on": True, "lora": LORA_CHAR_HIGH, "strength": 0.85, "strengthTwo": None},
              {}, ""],
             title="LoRA de personaje · HIGH", stage="modelo_h", size=[420, 140])
wire(speed_h, "MODEL", char_h, "model")
ms_h = add("ModelSamplingSD3", [5.0], title="shift HIGH", stage="modelo_h")
wire(char_h, "MODEL", ms_h, "model")
sage_h = add("PathchSageAttentionKJ", ["auto", False], title="SageAttention HIGH",
             mode=BYPASS, stage="modelo_h")
wire(ms_h, "MODEL", sage_h, "model")

# ==========================================================================
# 3 · MODELO LOW
# ==========================================================================
speed_l = add("LoraLoaderModelOnly", [SPEED_21, 1.0],
              title="velocidad LOW · 2.1 a fuerza 1", stage="modelo_l")
wire(unet_l, "MODEL", speed_l, "model")
char_l = add("Power Lora Loader (rgthree)",
             [{}, {"type": "PowerLoraLoaderHeaderWidget"},
              {"on": True, "lora": LORA_CHAR_LOW, "strength": 0.85, "strengthTwo": None},
              {}, ""],
             title="LoRA de personaje · LOW", stage="modelo_l", size=[420, 140])
wire(speed_l, "MODEL", char_l, "model")
ms_l = add("ModelSamplingSD3", [5.0], title="shift LOW", stage="modelo_l")
wire(char_l, "MODEL", ms_l, "model")
sage_l = add("PathchSageAttentionKJ", ["auto", False], title="SageAttention LOW",
             mode=BYPASS, stage="modelo_l")
wire(ms_l, "MODEL", sage_l, "model")

# ==========================================================================
# 4 · PROMPT
# ==========================================================================
pstr = add("PrimitiveStringMultiline", [PROMPT_POS], title="PROMPT", stage="prompt", size=[480, 300])
penc = add("CLIPTextEncode", [""], title="positivo", stage="prompt", size=[400, 90])
wire(clip, "CLIP", penc, "clip")
wire(pstr, "STRING", penc, "text")
nenc = add("CLIPTextEncode", [PROMPT_NEG], title="negativo oficial de Wan", stage="prompt", size=[400, 200])
wire(clip, "CLIP", nenc, "clip")

# ==========================================================================
# 5 · ENTRADAS
# ==========================================================================
img = add("LoadImage", ["example.png", "image"], title="PRIMER FRAME · i2v",
          stage="entrada", size=[320, 320])

prev_vid = add("LoadVideo", ["tanda_anterior.mp4", "video"], title="TANDA ANTERIOR · continuacion",
               stage="entrada", size=[320, 300])
prev_comp = add("GetVideoComponents", None, title="frames de la tanda anterior", stage="entrada")
wire(prev_vid, "VIDEO", prev_comp, "video")
prev_count = add("GetImageSizeAndCount", [], title="cuantos frames trae", stage="entrada")
wire(prev_comp, "images", prev_count, "image")
tail_idx = add("ComfyMathExpression", ["a - b"], title="indice de la cola", stage="entrada")
wire(prev_count, "count", tail_idx, "values.a")
wire(plan, "overlap", tail_idx, "values.b")
tail = add("ImageFromBatch", [0, 16], title="ultimos N frames", stage="entrada")
wire(prev_comp, "images", tail, "image")
wire(tail_idx, "INT", tail, "batch_index")
wire(plan, "overlap", tail, "length")

sw_start = add("ComfySwitchNode", [False], title="¿primera tanda o continuacion?", stage="entrada")
retype(sw_start, "IMAGE")
wire(img, "IMAGE", sw_start, "on_false")
wire(tail, "IMAGE", sw_start, "on_true")

start_rs = add("ImageResizeKJv2",
               [720, 1280, "lanczos", "crop", "0, 0, 0", "center", 16, "cpu"],
               title="encuadre a la resolucion de trabajo", stage="entrada")
wire(sw_start, "output", start_rs, "image")
wire(plan, "width", start_rs, "width")
wire(plan, "height", start_rs, "height")

ref = add("LoadImage", ["example.png", "image"], title="REFERENCIAS · ref2v (Phantom)",
          stage="entrada", size=[320, 320])
ref_rs = add("ImageResizeKJv2",
             [720, 1280, "lanczos", "pad", "255,255,255", "center", 16, "cpu"],
             title="referencias sobre blanco", stage="entrada")
wire(ref, "IMAGE", ref_rs, "image")
wire(plan, "width", ref_rs, "width")
wire(plan, "height", ref_rs, "height")

# ==========================================================================
# 6 · CONDICIONAMIENTO
# ==========================================================================
i2v = add("WanImageToVideo", [720, 1280, 81, 1],
          title="t2v / i2v / continuacion  ·  el mismo nodo", stage="cond", size=[380, 200])
wire(penc, "CONDITIONING", i2v, "positive")
wire(nenc, "CONDITIONING", i2v, "negative")
wire(vae, "VAE", i2v, "vae")
wire(start_rs, "IMAGE", i2v, "start_image")
wire(plan, "width", i2v, "width")
wire(plan, "height", i2v, "height")
wire(plan, "length", i2v, "length")

phantom = add("WanPhantomSubjectToVideo", [720, 1280, 81, 1],
              title="ref2v · Phantom  (salidas SIN conectar, lee la nota)",
              mode=BYPASS, stage="cond", size=[380, 200])
wire(penc, "CONDITIONING", phantom, "positive")
wire(nenc, "CONDITIONING", phantom, "negative")
wire(vae, "VAE", phantom, "vae")
wire(ref_rs, "IMAGE", phantom, "images")
wire(plan, "width", phantom, "width")
wire(plan, "height", phantom, "height")
wire(plan, "length", phantom, "length")

# ==========================================================================
# 7 · MUESTREO
# ==========================================================================
sched = add("BasicScheduler", ["normal", 20, 1.0], title="curva de sigmas", stage="sampler")
wire(ms_h, "MODEL", sched, "model")

split = add("CP_WanSplit", ["i2v  ·  sigma 0.900", 0.9, 1],
            title="CORTE high/low por SIGMA, no por paso", stage="sampler", size=[400, 200])
wire(sched, "SIGMAS", split, "sigmas")
split_prev = add("PreviewAny", [], title="corte · donde cae y con cuantos pasos",
                 stage="sampler", size=[400, 260])
wire(split, "resumen", split_prev, "source")

ks_h = add("KSamplerSelect", ["euler"], title="sampler HIGH", stage="sampler")
ks_l = add("KSamplerSelect", ["euler"], title="sampler LOW", stage="sampler")
noise = add("RandomNoise", [SEED, "randomize"], title="ruido", stage="sampler")
dis = add("DisableNoise", None, title="sin ruido nuevo en LOW", stage="sampler")

g_h = add("CFGGuider", [3.5], title="guider HIGH · cfg 3.5", stage="sampler")
wire(sage_h, "MODEL", g_h, "model")
wire(i2v, "positive", g_h, "positive")
wire(i2v, "negative", g_h, "negative")

s_h = add("SamplerCustomAdvanced", None, title="PASO HIGH · movimiento y composicion", stage="sampler")
wire(noise, "NOISE", s_h, "noise")
wire(g_h, "GUIDER", s_h, "guider")
wire(ks_h, "SAMPLER", s_h, "sampler")
wire(split, "high_sigmas", s_h, "sigmas")
wire(i2v, "latent", s_h, "latent_image")

g_l = add("CFGGuider", [1.0], title="guider LOW · cfg 1", stage="sampler")
wire(sage_l, "MODEL", g_l, "model")
wire(i2v, "positive", g_l, "positive")
wire(i2v, "negative", g_l, "negative")

s_l = add("SamplerCustomAdvanced", None, title="PASO LOW · detalle e identidad", stage="sampler")
wire(dis, "NOISE", s_l, "noise")
wire(g_l, "GUIDER", s_l, "guider")
wire(ks_l, "SAMPLER", s_l, "sampler")
wire(split, "low_sigmas", s_l, "sigmas")
wire(s_h, "output", s_l, "latent_image")

# ==========================================================================
# 8 · SALIDA
# ==========================================================================
dec = add("VAEDecode", None, title="decode", stage="salida")
wire(s_l, "output", dec, "samples")
wire(vae, "VAE", dec, "vae")

# NO va en bypass, va a fuerza 0. Si se pusiera en bypass, ComfyUI mapea la
# salida IMAGE a la PRIMERA entrada IMAGE del nodo, que es `image_ref` — o sea,
# devolveria la imagen de partida en vez del video, en silencio. A fuerza 0 el
# nodo es un paso a traves de verdad.
cmatch = add("ColorMatch", ["hm-mkl-hm", 0.0, True],
             title="ancla el color al frame de partida  ·  subelo a 0.6 al encadenar",
             stage="salida")
wire(start_rs, "IMAGE", cmatch, "image_ref")
wire(dec, "IMAGE", cmatch, "image_target")

sup_tcs = add("SeedVR2TorchCompileSettings", ["inductor", "default", False, False, 64, 128],
              title="torch.compile (opcional)", mode=BYPASS, stage="post")
sup_dit = add("SeedVR2LoadDiTModel",
              ["seedvr2_ema_3b_fp16.safetensors", "cuda:0", 0, False, "cpu", True, "sdpa"],
              title="SeedVR2 DiT 3B fp16", mode=BYPASS, stage="post")
wire(sup_tcs, "TORCH_COMPILE_ARGS", sup_dit, "torch_compile_args")
sup_vae = add("SeedVR2LoadVAEModel",
              ["ema_vae_fp16.safetensors", "cuda:0", False, 1024, 128, False, 768, 128,
               "false", "cpu", True],
              title="SeedVR2 VAE", mode=BYPASS, stage="post")
wire(sup_tcs, "TORCH_COMPILE_ARGS", sup_vae, "torch_compile_args")
sup = add("SeedVR2VideoUpscaler",
          [42, "fixed", 1440, 0, 81, True, "lab", 5, 0, 0, 0, "cpu", False],
          title="SeedVR2 · el salto de resolucion", mode=BYPASS, stage="post")
wire(cmatch, "image", sup, "image")
wire(sup_dit, "SEEDVR2_DIT", sup, "dit")
wire(sup_vae, "SEEDVR2_VAE", sup, "vae")

gis = add("GetImageSizeAndCount", [], title="medida y frames de salida", stage="post")
wire(sup, "IMAGE", gis, "image")

vid = add("CreateVideo", [16], title="monta el video", stage="post")
wire(gis, "image", vid, "images")
wire(plan, "fps_f", vid, "fps")
save = add("SaveVideo", ["video/WAN22-GEN", "auto", "auto"], title="GUARDAR la tanda", stage="post")
wire(vid, "VIDEO", save, "video")

vhs = add("VHS_VideoCombine",
          {"frame_rate": 16, "loop_count": 0, "filename_prefix": "WAN22-GEN",
           "format": "video/h264-mp4", "pix_fmt": "yuv420p", "crf": 19,
           "save_metadata": True, "trim_to_audio": False, "pingpong": False,
           "save_output": False},
          title="previsualizacion", stage="post", size=[420, 400])
wire(gis, "image", vhs, "images")
wire(plan, "fps_f", vhs, "frame_rate")

# ==========================================================================
# NOTAS
# ==========================================================================
note("""# Como se usa

## 1. Elige el modo en **PLAN**
| modo | que conectas | interruptor |
|---|---|---|
| **t2v** | nada | pon el switch de entrada a `false` y **quita** el cable `start_image` |
| **i2v** | `PRIMER FRAME` | switch a `false` |
| **continuacion** | `TANDA ANTERIOR` (el mp4 de la tanda previa) | switch a `true` |
| **ref2v** | `REFERENCIAS` | lee la nota de Phantom |

## 2. Para un clip largo, tanda a tanda
1. `chunk_index = 0`, modo **i2v**, switch `false`. Lanza. Guarda el mp4.
2. `chunk_index = 1`, modo **continuacion**, switch `true`, carga ese mp4 en
   `TANDA ANTERIOR`. Enciende `ColorMatch`. Lanza.
3. Repite. El resumen del PLAN te dice cuantas tandas quedan y cuando es la ultima.

**El plan no se mueve entre tandas.** Duracion total, resolucion y solape se
escriben una vez; lo unico que cambia es `chunk_index`.

## 3. Une las tandas fuera de ComfyUI
Con solape de 16 frames, cada tanda repite el ultimo segundo de la anterior.
Al concatenar, recorta esos 16 frames del principio de cada tanda menos la
primera. `Image Batch Extend With Overlap` de KJNodes lo hace dentro de Comfy
si prefieres no salir.""",
     "LEEME — arranque y clips largos", size=(580, 640))

note("""# Por que 81 frames y no mas

Wan 2.2 se entreno con clips de **81 frames a 16 fps** = 5.06 s. Por encima de
eso **no da error**: el video empieza a repetirse en bucle, o la coherencia se
degrada poco a poco. Es el limite que mas gente descubre tarde.

## Las tres salidas, y cual es cual

| via | que hace | veredicto |
|---|---|---|
| **Tandas encadenadas** (este flujo) | generas 81, coges los ultimos N frames, los metes como imagen inicial de la siguiente | La que funciona hoy sin nada extra. Wan 2.2 I2V admite varios frames iniciales sin ningun LoRA — Wan 2.1 no podia |
| **Context Windows** | una sola pasada con ventana deslizante | Existe (`WAN Context Windows (Manual)` nativo, `WanVideo Context Options` en el wrapper) pero con I2V esta reportado como *fragil*: cada ventana usa la imagen de entrada como punto de partida y el fondo se desplaza |
| **SVI 2.0 Pro** | LoRAs que anclan cada tanda a un frame de referencia | Lo mejor contra la degradacion. Ancla la camara y cuesta soltarla, y necesita nodos extra |

## El precio del solape
Con tandas de 81 y solape 16, cada tanda aporta **65 frames nuevos**, no 81.
Cuatro tandas no son 20 s, son 16.3 s. El PLAN ya cuenta bien; no lo hagas a mano.

## Los dos problemas del encadenado
- **Flashes** en los primeros frames de cada tanda: cambio brusco de brillo o un
  patron de ruido. Duran pocos frames.
- **Burn-out**: el contraste sube y los colores derivan con cada tanda.

Contra el segundo, el `ColorMatch` del final anclado a la imagen de partida. Es
lo mismo que hace la comunidad con SVI Pro para tapar la deriva residual.""",
     "El limite de 81 frames — lo que preguntabas", size=(580, 720))

note("""# El corte high/low: por sigma, no por pasos

Wan 2.2 es **MoE**: un experto para ruido alto (composicion y movimiento) y otro
para ruido bajo (detalle e identidad). El traspaso no va por la mitad de los
pasos: va a una **sigma** concreta.

| modelo | sigma de traspaso |
|---|---|
| **I2V** | **0.900** |
| **T2V** | **0.875** |

El paso en el que cae esa sigma **depende del shift y del numero de pasos**.
Cambias el shift de 5 a 8 y el paso correcto se mueve. `SplitSigmas` del core
corta donde tu le digas y no se entera; el nodo `CORTE` de aqui lo busca solo y
te ensena donde ha cortado.

**El reparto no sale 50/50, y eso es lo correcto.** Con shift 5 y 20 pasos, la
sigma 0.9 cae hacia el paso 7: 7 arriba y 13 abajo. Ese desequilibrio es
exactamente lo que la comunidad describe cuando dice que *"el 50/50 de manual no
funciona bien"*.

## Si quieres mas pasos arriba
Sube el `shift` del `ModelSamplingSD3` **de los dos** modelos. Mas shift = la
curva baja mas tarde = mas pasos por encima de la sigma de corte.""",
     "El corte high/low — la palanca de calidad numero 1", size=(560, 620))

note("""# Calidad vs velocidad: los LoRA de destilacion

El flujo viene en **modo calidad**: el LoRA de velocidad del HIGH esta en
**bypass** y el del LOW encendido. Es la receta que la comunidad reporta como la
de mejor movimiento:

> mucho mejor movimiento sin lightx2v en el modelo High;
> 10 pasos cfg 3.5 para high-noise y 4 para low-noise (con lightx2v)

## Los tres perfiles que merece la pena tener a mano

| perfil | HIGH | LOW | cfg high | pasos |
|---|---|---|---|---|
| **calidad** (puesto) | sin LoRA | 2.1 @1.0 | 3.5 | 20 |
| **equilibrio** | 2.1 @3.0 | 2.1 @1.0 | 2.0 solo el primer paso | 6 |
| **borrador** | 1030 @1.0 | 2.1 @1.0 | 1.0 | 4 |

## Cual LoRA de velocidad
- **`lightx2v_I2V_14B_480p_cfg_step_distill_rank64`** (el "viejo" de 2.1) sigue
  siendo el mas recomendado en **LOW** a 1.0-1.5, y funciona en **HIGH** a
  fuerza **3.0**. *"nada le gana todavia al viejo 2.1 lightx2v"*.
- **`Wan_2_2_I2V_A14B_HIGH_lightx2v_4step_v1030`** es el mas recomendado
  especificamente para HIGH de 2.2: mejor obediencia al prompt, pero movimiento
  humano mas basico, y hay quien reporta desplazamientos de color.

## El truco del cfg en un solo paso
Un paso con cfg > 1 cuesta el doble (dos pases del modelo). Por eso mucha gente
hace **solo el primer paso con cfg** y el resto a 1. *"al menos un paso con cfg
marca muchisima diferencia"*. Aqui se aproxima poniendo cfg 3.5 en todo el tramo
high, que son pocos pasos.""",
     "LoRAs de velocidad — calidad, equilibrio y borrador", size=(580, 720))

note("""# LoRA de personaje: dos, no uno

Wan 2.2 es MoE, asi que un personaje son **dos LoRA**: uno entrenado sobre el
experto `high_noise` y otro sobre el `low_noise`. Por eso hay dos
`Power Lora Loader` en el flujo.

- **El LOW es el que manda** en textura e identidad de cerca. Si solo vas a
  entrenar uno, entrena ese.
- **El HIGH** aporta silueta y como se mueve el personaje. Sin el, la identidad
  aparece pero el cuerpo se mueve como un generico.
- Fuerza de partida **0.85** en los dos.
- `musubi-tuner` es el estandar y acepta dataset de imagenes sueltas.

## Y los de Flux y los de LTX no valen
Un LoRA son deltas sobre matrices concretas de una arquitectura. ComfyUI carga
el archivo, no aplica **ninguna** clave y no dice nada. El nodo
`LoRA Trigger Reader` del pack detecta la arquitectura y puede detener la
ejecucion si no coincide — usalo antes de volverte loco buscando por que "no se
nota".""",
     "LoRA de personaje — por que son dos", size=(560, 540))

note("""# ref2v con Phantom — leelo antes de encender el nodo

`WanPhantomSubjectToVideo` es **nativo** y acepta hasta **4 imagenes de
referencia** (una por sujeto). Sus salidas estan **sin conectar a proposito**.

## Por que no esta cableado
Phantom devuelve **tres** conditionings (`positive`, `negative_text`,
`negative_img_text`) y su propio latente, y no encajan uno a uno con los de
`WanImageToVideo`. Un switch automatico aqui seria mentir sobre lo que hace.

## Para usarlo
1. Quita el bypass del nodo Phantom.
2. Lleva `positive` -> `guider LOW.positive` y `negative_img_text` ->
   `guider LOW.negative`.
3. Lleva `latent` -> `PASO HIGH.latent_image`.

## Y lo que de verdad tienes que saber
> Phantom **solo funciona en la parte LOW**. O te saltas el paso HIGH entero
> — que es perfectamente valido — o haces que la salida del HIGH se parezca
> bastante a tus referencias para que Phantom pueda rematar.

Ademas Phantom es un finetune de **Wan 2.1 T2V**: necesitas ese checkpoint, no
te vale el 2.2 que hay cargado arriba.

## Alternativas, por caso
| quieres | usa |
|---|---|
| identidad de TU personaje | **tu LoRA entrenado**. Es el mecanismo principal; Phantom es complementario |
| sujetos/objetos de referencia sueltos | **Phantom** |
| meter tu personaje en un video existente | **Wan Animate 2** (`WanAnimateToVideo`, nativo) — el paper de Animate dice que VACE es inestable para tareas de personaje |
| control de pose/profundidad | **VACE 2.2 Fun** — pero *"no sera bueno para personaje salvo que la referencia este casi donde va a estar en el primer frame"* |
| referencia + clips largos | **SVI 2.0 Pro**: el `anchor_image` hace de referencia y ademas encadena tandas |""",
     "ref2v — Phantom, Animate, VACE y cual toca", size=(580, 720))

note("""# Ajustes que no se ven pero se notan

## Resolucion
Multiplo de **16**. Las que la comunidad reporta como bien portadas:
`1280x704`, `1024x576`, `832x480`, `1536x864`, y sus verticales. Wan 2.2 es
nativo a ~720p; por encima aparecen repeticiones y deriva. Sube al final con
SeedVR2, no generando mas grande.

## fps
Wan 2.2 es nativo a **16 fps**. Generar a otro frame rate cambia la velocidad
percibida del movimiento. Genera a 16 e **interpola al final** (RIFE o GIMM-VFI).
Y siempre en ese orden: primero upscale, luego interpolacion.

## El VAE y el parpadeo
Si activas decode por tiles para ahorrar VRAM, pon `temporal_size` **por encima
del numero de frames** — con 81 frames, 84. Si lo dejas por debajo, el VAE
tileado introduce parpadeo. Con 96 GB no necesitas tiles: el `VAEDecode` normal
del flujo es lo correcto.

## Rejillas de ruido en la salida
Si aparece un patron de rejilla o moteado: no es el VAE, son los latentes
generados demasiado borrosos, que caen fuera de la distribucion del decoder.
Arreglos por orden: mas pasos, encender el LoRA de velocidad en LOW, desactivar
SageAttention, o una segunda pasada con el modelo LOW a denoise ~0.2.

## Precision
Wan corre en **fp16**, no bf16: es mas rapido con acumulacion rapida y mejor
cuando funciona. Y el orden de calidad entre cuantizaciones que reporta la
comunidad es **GGUF Q8 > INT8 ConvRot > MXFP8 > FP8 >= INT8**. Con 96 GB, fp16
directamente.

## Parentesis en el prompt
Escribir `(16:9)` o similar dentro del prompt puede **romper** la generacion en
Wan. Sin parentesis.""",
     "Los detalles que rompen el resultado en silencio", size=(580, 720))

note("""# Nodos que este flujo necesita

| pack | para que |
|---|---|
| **ComfyUI** al dia | `WanImageToVideo`, `WanPhantomSubjectToVideo`, samplers, switch |
| **ComfyUI-KJNodes** | resize, ColorMatch, SageAttention, contador de frames |
| **rgthree-comfy** | Power Lora Loader |
| **ComfyUI-VideoHelperSuite** | previsualizacion |
| **ComfyUI-SeedVR2_VideoUpscaler** | el remate |
| **comfyui-character-pipeline** (el tuyo) | `PLAN` y `CORTE` |

## El ecosistema de Wan, que es enorme y esta mal indexado

- **[wanx-troopers.github.io](https://wanx-troopers.github.io/)** — no es un
  pack, es *la* base de conocimiento de la comunidad Wan: 72 documentos con lo
  que funciona y lo que no, incluido un `hidden-knowledge.md` de 28 KB. La mitad
  de las cifras de estas notas salen de ahi.
- **`kijai/ComfyUI-WanVideoWrapper`** — 146 nodos. Es el camino de los que
  exprimen el modelo: block swap asincrono, context windows, radial attention,
  cache de encoder de texto en disco, RoPE scaling para resoluciones grandes.
  Este flujo es **nativo** a proposito, que es lo que ComfyDeploy digiere mejor.
- **`drozbay/ComfyUI-WanVaceAdvanced`** y **`drozbay/WanExperiments`** — el que
  mas empuja los limites de Wan ahora mismo. Embeds a medida, VACE avanzado, y
  un fork de Impact Pack con detailer SEGS para video.
- **`ckinpdx/ComfyUI-WanKeyframeBuilder`** — `Wan Keyframe Builder (Continuation)`:
  prepara imagenes y mascaras para encadenar tandas, con modo especifico para SVI.
- **`WASasquatch/WAS_Extras`** — `WASWanExposureStabilizer`, contra el burn-out
  del encadenado.
- **`Granddyser/wan-video-extender`** — extension por lotes con solape.
- **`vrgamegirl19/comfyui-vrgamedevgirl`** — FMML (first-middle-middle-last).
- **`Extraltodeus/Skimmed_CFG`** — cfg alto para obedecer el prompt sin quemar.
- **`Azornes/Comfyui-Resolution-Master`**, **`kijai/ComfyUI-MemoryVisualization`**,
  **`kijai/ComfyUI-GIMM-VFI`** — utilidad pura.""",
     "El ecosistema — que instalar y donde esta el conocimiento", size=(580, 740))

note("""# Que NO he verificado

- **No he ejecutado nada.** Lo que si esta verificado: la topologia sale de los
  blueprints oficiales `text_to_video_wan_2_2` / `image_to_video_wan_2_2` de
  Comfy-Org, y el grafo lo genera un script con las firmas de nodo extraidas de
  esos mismos JSON, validando tipos y slots. 0 errores.

- **Los nombres de LoRA de personaje son marcadores**
  (`TU_PERSONAJE_wan22_high/low`). Los tuyos de Flux y LTX **no valen aqui**.

- **Las cifras de LoRA de velocidad, sigmas de traspaso y resoluciones** vienen
  de la base de conocimiento de la comunidad y de hilos citados, no de
  mediciones mias sobre tu material.

- **`cfg 3.5` en el tramo high con 20 pasos** es un punto de partida razonable
  segun lo publicado, no un valor calibrado. Si sale quemado, baja a 2.5 o mete
  Skimmed CFG.

- **Phantom** esta cableado a medias **a proposito** y necesita el checkpoint de
  Wan 2.1 Phantom. No lo he probado.

- **El `ComfySwitchNode`** para elegir entre primer frame y cola de la tanda
  anterior: el nodo es de core y generico, pero si tu version de ComfyUI es
  antigua puede no existir. Sustitucion: conecta a mano el que toque.

- **`WanImageToVideo` sin `start_image`** se comporta como t2v — eso si lo he
  comprobado leyendo el codigo: sin imagen no anade conditioning de concat. Pero
  entonces hay que cargar los UNET **t2v**, no los i2v que vienen puestos.""",
     "Honestidad — lo que no esta comprobado", size=(580, 660))

# ==========================================================================
# LAYOUT
# ==========================================================================
GRAPH = [n for n in nodes if n["type"] != "MarkdownNote"]
NOTES = [n for n in nodes if n["type"] == "MarkdownNote"]
GIDS = {n["id"] for n in GRAPH}

STAGE_ORDER = [
    ("panel", "0 · PANEL — empieza aqui", "#2a6"),
    ("carga", "1 · CARGA", "#3f789e"),
    ("prompt", "2 · PROMPT", "#a1309b"),
    ("modelo_h", "3 · EXPERTO HIGH NOISE", "#b58b2a"),
    ("modelo_l", "4 · EXPERTO LOW NOISE", "#8a5a3c"),
    ("entrada", "5 · ENTRADAS · primer frame / tanda anterior / referencias", "#557"),
    ("cond", "6 · CONDICIONAMIENTO", "#A88"),
    ("sampler", "7 · MUESTREO · corte por sigma", "#2a6"),
    ("salida", "8 · DECODE Y COLOR", "#8A8"),
    ("post", "9 · POST Y ENTREGA", "#555"),
]

preds = defaultdict(set)
for lid, on, osl, tn, tsl, ty in links:
    if on in GIDS and tn in GIDS and on != tn:
        preds[tn].add(on)

COL_GAP, ROW_GAP, STAGE_GAP, PAD = 110, 46, 240, 55
ROW_MAX_X = 6200
bounds, groups = {}, []
x_cursor = y_cursor = row_h = 0

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

    stage_x0 = x = x_cursor
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

    x2 = max(n["pos"][0] + n["size"][0] for n in members)
    y1 = min(n["pos"][1] for n in members)
    y2 = max(n["pos"][1] + n["size"][1] for n in members)
    groups.append({"id": len(groups) + 1, "title": title,
                   "bounding": [stage_x0 - PAD, y1 - PAD - 32, x2 - stage_x0 + PAD * 2, y2 - y1 + PAD * 2],
                   "color": color, "font_size": 26, "flags": {}})
    x_cursor = x2 + STAGE_GAP

note_x = max(g["bounding"][0] + g["bounding"][2] for g in groups) + 200
y = 0
for n in NOTES:
    n["pos"] = [note_x, y]
    y += n["size"][1] + ROW_GAP
groups.append({"id": len(groups) + 1, "title": "DOCUMENTACION",
               "bounding": [note_x - PAD, -PAD - 32,
                            max(n["size"][0] for n in NOTES) + PAD * 2, y + PAD],
               "color": "#444", "font_size": 26, "flags": {}})

STAGE_RANK = {s: i for i, (s, _, _) in enumerate(STAGE_ORDER)}
for n in GRAPH:
    n["order"] = STAGE_RANK.get(STAGE_OF[n["id"]], 99) * 1000 + n["id"]

for n in nodes:
    n["pos"] = [round(v) for v in n["pos"]]
    n["size"] = [round(v) for v in n["size"]]
for g in groups:
    g["bounding"] = [round(v) for v in g["bounding"]]

wf = {
    # UUID fijo: el esquema zod lo exige y ComfyDeploy lo usa como identidad
    # del flujo entre despliegues.
    "id": "3c1f0a7e-5d84-4b62-9f30-8a6e2d17bb54",
    "revision": 0,
    "last_node_id": _state["nid"],
    "last_link_id": _state["lid"],
    "nodes": nodes,
    "links": links,
    "groups": groups,
    "config": {},
    "extra": {
        "ds": {"scale": 0.5, "offset": [120.0, 140.0]},
        "frontendVersion": "1.44.19",
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
if not re.match(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", str(wf["id"]), re.I):
    errs.append(f"el `id` del flujo no es un UUID: {wf['id']!r}")
for n in nodes:
    if "widgets_values" in n and n["widgets_values"] is None:
        errs.append(f"{n['type']} [{n['id']}]: widgets_values = null")
    for o in n["outputs"]:
        if o.get("links") is None:
            o["links"] = []

by = {n["id"]: n for n in nodes}
seen = set()
for lid, on, osl, tn, tsl, ty in links:
    if lid in seen:
        errs.append(f"link {lid} duplicado")
    seen.add(lid)
    o, t = by[on], by[tn]
    if osl >= len(o["outputs"]) or tsl >= len(t["inputs"]):
        errs.append(f"link {lid}: slot fuera de rango")
        continue
    if lid not in (o["outputs"][osl].get("links") or []):
        errs.append(f"link {lid}: no aparece en outputs de {o['type']}")
    if t["inputs"][tsl].get("link") != lid:
        errs.append(f"link {lid}: no aparece en inputs de {t['type']}")

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
