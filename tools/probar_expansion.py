# -*- coding: utf-8 -*-
"""Prueba del nodo expandible CP_BlockingLento DENTRO del motor real de ComfyUI, sin GPU y sin modelos.

Carga el motor (comfy_extras, el pack ComfyUI-LTXVideo y este pack) en CPU, en este mismo proceso, y ejecuta prompts de
API con execution.PromptExecutor, igual que el hilo de trabajo de main.py. Los nodos pesados se cambian por DOBLES que
heredan de la clase real (mismo esquema de entradas y salidas: el motor no ve la diferencia) y solo comprueban y
registran lo que reciben:
- LTXVAddGuide, LTXVCropGuides, LTXVLatentUpsampler, CFGGuider, SamplerCustomAdvanced, LTXAddVideoICLoRAGuide y
  LTXVTiledVAEDecode;
- las fuentes (modelos, vae, escalador, conditioning, factor, poses, arte) son nodos dobles con su etiqueta.
Son de verdad: EmptyLTXVLatentVideo, ImageFromBatch, ImageScale, KSamplerSelect, ManualSigmas, RandomNoise,
PrimitiveString, CP_PosesDesdeFotos, CP_LentaPlan, CP_BlockingLento y CP_LentaElegir.

El decodificado doble devuelve n fotogramas pequenos con un cuadrado que se mueve (para que el flujo optico mida algo)
y una marca exacta en el pixel de abajo a la izquierda: canal 0 = indice lento / 1000, canal 1 = codigo del nodo de
decodificado / 1000. Asi se sabe de donde sale cada fotograma de la animacion y que ancla recibe cada ventana.

Lo que se comprueba (aserciones):
1. el prompt raiz valida (execution.validate_prompt) y se ejecuta entero; animacion y playblast tienen 'total'
   fotogramas y el mismo tamano;
2. raw_link: los nodos de origen se ejecutan una sola vez y el subgrafo lleva sus enlaces tal cual;
3. cableado: etapa 1 con modelo y conditioning suaves, etapa 2 con los de render y el factor; frame_idx = pos del plan;
   la guia de la etapa 2 tiene n fotogramas; el ancla de w>0 es el ultimo fotograma del render anterior; el orden de
   ejecucion respeta la cadena; todos los nodos efimeros pasan execution.validate_inputs del motor;
4. cache: el mismo prompt otra vez no ejecuta ningun doble; cambiando solo la semilla se vuelve a ejecutar lo que
   depende del ruido y nada de lo que no;
5. mensajes 'executing' y 'progress_state' de los nodos efimeros (que node y display_node ve el cliente);
6. planes invalidos: error legible en espanol como execution_error del nodo;
7. solo_animacion: con True no se ejecuta nada de la etapa 2 y las dos salidas son el playblast elegido; sin la clave
   (como la envia la tarjeta publicada) y con False, el mismo subgrafo y los mismos dobles que en la seccion 1;
8. fotos: CP_PosesDesdeFotos de verdad (fotos PNG escritas en la carpeta de entrada de la carpeta de trabajo) ->
   CP_LentaPlan con transicion_max enlazado a su salida "transicion" -> CP_BlockingLento con solo_animacion True y
   False: salen los fotogramas que anuncia el nodo de fotos, cada pose en su fotograma (k x d) y su guia en el lento
   que dice el plan, y entre pose y pose se mueve en todos los fotogramas;
   La tarjeta de fotos lleva mas_lentos=True en CP_LentaPlan y aqui tambien: 32 lentos por salto con 12 por pose;
9. ritmo 2 con el mapa de video de la seccion 1: el doble de fotogramas, cada pose en el doble de su inicio;
10. la tarjeta publicada: el plano de control (25 poses, 77 fotogramas) con las entradas que envia su prompt a
   CP_LentaPlan y CP_BlockingLento (sin ritmo, mas_lentos ni solo_animacion); el plan (huella sha256 del JSON) y el
   subgrafo expandido por el motor (huella con el prefijo cambiado por '@') tienen que ser los que da la v0.14, que
   estan en tests/datos/compat_v014.json (los genera tools/generar_compat_v014.py con una copia de la v0.14).
Antes de nada (seccion 0) se comprueba que el pack publica sus nodos de camara lenta y de fotos con su esquema.

Uso (con el Python del motor; no arranca servidor ni escribe en la carpeta del motor):
    python3 -B tools/probar_expansion.py <carpeta de ComfyUI> <carpeta de trabajo vacia o desechable>
La carpeta de trabajo hace de base_directory (input, output, temp, user) y de cache de numba.
Sale con codigo 0 si todo se cumple.
"""

import asyncio
import json
import os
import sys
import time

sys.dont_write_bytecode = True

if len(sys.argv) != 3:
    print(__doc__)
    sys.exit(2)

MOTOR = os.path.abspath(sys.argv[1])
TRABAJO = os.path.abspath(sys.argv[2])
PACK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.makedirs(TRABAJO, exist_ok=True)
os.environ.setdefault("NUMBA_CACHE_DIR", os.path.join(TRABAJO, "numba"))
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.path.insert(0, MOTOR)
os.chdir(TRABAJO)

from comfy.cli_args import args  # noqa: E402

args.cpu = True
args.base_directory = TRABAJO
args.disable_all_custom_nodes = True

import torch  # noqa: E402

import folder_paths  # noqa: E402
import nodes  # noqa: E402
import execution  # noqa: E402
import comfy.model_management  # noqa: E402
from comfy_api.latest import io  # noqa: E402
from comfy_execution.utils import get_executing_context  # noqa: E402

assert folder_paths.get_output_directory().startswith(TRABAJO), "la salida tiene que ir a la carpeta de trabajo"

T0 = time.time()


async def _cargar():
    await nodes.init_extra_nodes(init_custom_nodes=False, init_api_nodes=False)
    assert await nodes.load_custom_node(os.path.join(MOTOR, "custom_nodes", "ComfyUI-LTXVideo")), "no carga ComfyUI-LTXVideo"
    assert await nodes.load_custom_node(PACK), "no carga el pack"


asyncio.run(_cargar())
print(f"motor y packs cargados en {time.time() - T0:.1f} s")

M = nodes.NODE_CLASS_MAPPINGS
for _n in ("CP_PosesDesdeFotos", "CP_LentaPlan", "CP_BlockingLento", "CP_LentaElegir", "EmptyLTXVLatentVideo",
           "ImageFromBatch", "ImageScale", "KSamplerSelect", "ManualSigmas", "RandomNoise", "PrimitiveString"):
    assert _n in M, f"falta el nodo real {_n}"

# ------------------------------------------------------------------------------------------------ 0: esquemas
# lo que veria GET /object_info/<nodo> (server.py llama a GET_NODE_INFO_V1 en los nodos V3)
print("\n== 0. el pack publica los nodos con su esquema ==")
INFO = {n: M[n].GET_NODE_INFO_V1() for n in ("CP_PosesDesdeFotos", "CP_LentaPlan", "CP_BlockingLento", "CP_LentaElegir")}
for _n, _i in INFO.items():
    _ent = list(_i["input_order"].get("required", [])) + list(_i["input_order"].get("optional", []))
    print(f"{_n}: {len(_ent)} entradas ({len(_i['input_order'].get('optional', []))} opcionales), salidas {_i['output_name']}")
_fo = INFO["CP_PosesDesdeFotos"]
_casillas = [f"pose_{i:02d}" for i in range(1, 25)]
_okfotos = (_fo["input_order"]["optional"] == _casillas
            and _fo["input_order"]["required"] == ["fotogramas_por_pose", "espera_final", "orden"]
            and all(_fo["input"]["optional"][c][1].get("image_upload") is True for c in _casillas)
            and _fo["input"]["required"]["fotogramas_por_pose"][1]["max"] == 32
            and list(_fo["output_name"]) == ["poses", "mapa", "n_poses", "transicion", "fotogramas", "fps"])
_lp, _bl = INFO["CP_LentaPlan"], INFO["CP_BlockingLento"]
_okplan = (_lp["input_order"]["optional"] == ["ritmo", "mas_lentos"] and _lp["input"]["optional"]["ritmo"][1]["default"] == 1
           and _lp["input"]["optional"]["mas_lentos"][0] == "BOOLEAN" and _lp["input"]["optional"]["mas_lentos"][1]["default"] is False)
_okbl = _bl["input_order"]["optional"] == ["solo_animacion"] and _bl["input"]["optional"]["solo_animacion"][1]["default"] is False
print("FALLO esquemas" if not (_okfotos and _okplan and _okbl) else "ok   esquemas: 24 casillas con subida, ritmo, mas_lentos y solo_animacion opcionales")
assert _okfotos and _okplan and _okbl, (_fo["input_order"], _lp["input_order"], _bl["input_order"])

# ------------------------------------------------------------------------------------------------ registro y objetos
REGISTRO = []  # (clase, id del nodo, datos) en orden de ejecucion
CODIGOS = {}  # id del nodo de decodificado -> codigo de su marca
RECOGIDO = {}


def _yo():
    ctx = get_executing_context()
    return ctx.node_id if ctx else None


def _apuntar(clase, **datos):
    REGISTRO.append((clase, _yo(), datos))


class Etq:
    """Objeto etiquetado: lo que pasa por MODEL, VAE, LATENT_UPSCALE_MODEL."""

    def __init__(self, tipo, etiqueta):
        self.tipo, self.etiqueta = tipo, etiqueta

    def __repr__(self):
        return f"{self.tipo}:{self.etiqueta}"


class Cond:
    """CONDITIONING doble: etiqueta del texto y las guias que lleva encima."""

    def __init__(self, etiqueta, guias=()):
        self.etiqueta, self.guias = etiqueta, tuple(guias)

    def con(self, guia):
        return Cond(self.etiqueta, self.guias + (guia,))

    def __repr__(self):
        return f"cond:{self.etiqueta}+{len(self.guias)}"


class Guider:
    def __init__(self, modelo, positivo, negativo, cfg):
        self.modelo, self.positivo, self.negativo, self.cfg = modelo, positivo, negativo, cfg


def _fotogramas(latente):
    return (latente["samples"].shape[2] - 1) * 8 + 1


def _lat(latente, **cambios):
    d = dict(latente.get("doble", {"guias": 0, "muestreado": None, "subido": False}))
    d.update(cambios)
    return {"samples": latente["samples"], "doble": d}


def _marca(imagen, f=0):
    """(codigo, indice) de la marca de un fotograma, o el valor de la banda de pose de una imagen de pose."""
    px = imagen[f, -1, 0]
    return int(round(float(px[1]) * 1000)), int(round(float(px[0]) * 1000))


def _pose(imagen):
    """Indice de pose leido de la banda de arriba (valor (k+1)/20), tambien despues de reescalar."""
    h, w = imagen.shape[1], imagen.shape[2]
    return int(round(float(imagen[0, max(1, h // 30), w // 2, 2]) * 20)) - 1


# ------------------------------------------------------------------------------------------------ fuentes dobles
def _fuente(nombre, tipo, construir, entradas=None):
    def f(self, **kw):
        _apuntar(nombre, **kw)
        return (construir(**kw),)

    return type(nombre, (), {
        "INPUT_TYPES": classmethod(lambda cls: {"required": entradas or {}}),
        "RETURN_TYPES": (tipo,), "FUNCTION": "f", "CATEGORY": "pruebas", "f": f})


def _poses(n):
    """n poses de 96x54: fondo gris, banda arriba con el indice de pose y un cuadrado que salta."""
    xs = [6, 7, 40, 41, 74, 75, 76, 77][:n]
    im = torch.full((n, 54, 96, 3), 0.25)
    for k in range(n):
        im[k, :6, :, :] = (k + 1) / 20
        im[k, 24:38, xs[k]:xs[k] + 14, :] = 0.9
    return im


def _arte():
    im = torch.full((1, 54, 96, 3), 0.5)
    im[0, -1, 0, 0], im[0, -1, 0, 1] = 0.0, 0.999  # codigo 999 = el arte
    return im


DOBLES_FUENTE = {
    "Doble_Modelo": _fuente("Doble_Modelo", "MODEL", lambda etiqueta: Etq("modelo", etiqueta), {"etiqueta": ("STRING", {})}),
    "Doble_VAE": _fuente("Doble_VAE", "VAE", lambda: Etq("vae", "ltx")),
    "Doble_Escalador": _fuente("Doble_Escalador", "LATENT_UPSCALE_MODEL", lambda: Etq("escalador", "x2")),
    "Doble_Cond": _fuente("Doble_Cond", "CONDITIONING", lambda etiqueta: Cond(etiqueta), {"etiqueta": ("STRING", {})}),
    "Doble_Factor": _fuente("Doble_Factor", "FLOAT", lambda valor: float(valor), {"valor": ("FLOAT", {"default": 2.0})}),
    "Doble_Poses": _fuente("Doble_Poses", "IMAGE", lambda n: _poses(n), {"n": ("INT", {"default": 6, "min": 1, "max": 8})}),
    "Doble_Arte": _fuente("Doble_Arte", "IMAGE", lambda: _arte()),
}


class Doble_Recoger:
    """Nodo de salida: guarda en memoria lo que recibe (nada en disco)."""
    OUTPUT_NODE = True
    RETURN_TYPES = ()
    FUNCTION = "f"
    CATEGORY = "pruebas"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"animacion": ("IMAGE",), "playblast": ("IMAGE",)}}

    def f(self, animacion, playblast):
        _apuntar("Doble_Recoger")
        RECOGIDO.update(animacion=animacion, playblast=playblast)
        return {"ui": {"forma": [list(animacion.shape), list(playblast.shape)]}}


# ------------------------------------------------------------------------------------------------ dobles pesados
def _doble_v3(nombre, ejecutar):
    real = M[nombre]
    return type(nombre, (real,), {"execute": classmethod(lambda cls, **kw: ejecutar(**kw))})


def _add_guide(positive, negative, vae, latent, image, frame_idx, strength):
    assert isinstance(vae, Etq) and vae.tipo == "vae", vae
    assert isinstance(positive, Cond) and isinstance(negative, Cond)
    n = _fotogramas(latent)
    h, w = latent["samples"].shape[3] * 32, latent["samples"].shape[4] * 32
    assert tuple(image.shape) == (1, h, w, 3), f"guia de {tuple(image.shape)} sobre un latente de {w}x{h}"
    assert isinstance(frame_idx, int) and frame_idx % 8 == 0 and 0 <= frame_idx <= n - 1, (frame_idx, n)
    assert strength == 1.0 and isinstance(strength, float), strength
    guia = ("pose", _pose(image), frame_idx, (h, w))
    _apuntar("LTXVAddGuide", modelo=None, positivo=positive.etiqueta, negativo=negative.etiqueta, pose=_pose(image),
             frame_idx=frame_idx, tamano=(w, h), n=n, subido=latent.get("doble", {}).get("subido", False))
    lat = _lat(latent)
    lat["doble"]["guias"] += 1
    return io.NodeOutput(positive.con(guia), negative.con(guia), lat)


def _crop(positive, negative, latent):
    g = latent["doble"]["guias"]
    assert g > 0 and len(positive.guias) == g == len(negative.guias), \
        f"LTXVCropGuides tiene que recibir el conditioning CON las guias ({len(positive.guias)}) y el latente ({g})"
    _apuntar("LTXVCropGuides", guias=g, positivo=positive.etiqueta)
    return io.NodeOutput(Cond(positive.etiqueta), Cond(negative.etiqueta), _lat(latent, guias=0))


def _subir(samples, upscale_model, vae):
    assert isinstance(upscale_model, Etq) and upscale_model.tipo == "escalador"
    d = samples["doble"]
    assert d["guias"] == 0 and d["muestreado"] == "suave" and not d["subido"], d
    s = samples["samples"]
    _apuntar("LTXVLatentUpsampler", n=_fotogramas(samples))
    nuevo = torch.zeros(s.shape[0], s.shape[1], s.shape[2], s.shape[3] * 2, s.shape[4] * 2)
    return io.NodeOutput({"samples": nuevo, "doble": dict(d, subido=True, muestreado=None)})


def _guider(model, positive, negative, cfg):
    assert isinstance(model, Etq) and model.tipo == "modelo"
    assert cfg == 1.0, cfg
    _apuntar("CFGGuider", modelo=model.etiqueta)
    return io.NodeOutput(Guider(model, positive, negative, cfg))


def _muestrear(noise, guider, sampler, sigmas, latent_image):
    assert hasattr(noise, "seed"), "el ruido tiene que ser el RandomNoise real"
    assert isinstance(guider, Guider)
    assert isinstance(sigmas, torch.Tensor)
    tipos = {g[0] for g in guider.positivo.guias}
    assert len(guider.positivo.guias) == latent_image["doble"]["guias"] > 0
    _apuntar("SamplerCustomAdvanced", modelo=guider.modelo.etiqueta, positivo=guider.positivo.etiqueta,
             negativo=guider.negativo.etiqueta, semilla=noise.seed, sigmas=[round(float(x), 6) for x in sigmas],
             muestreador=getattr(sampler, "sampler_function", None).__name__ if hasattr(sampler, "sampler_function") else repr(sampler),
             n=_fotogramas(latent_image), tipos_guia=sorted(tipos), guias=len(guider.positivo.guias),
             subido=latent_image["doble"]["subido"])
    familia = "suave" if "pose" in tipos else "render"
    salida = _lat(latent_image, muestreado=familia)
    return io.NodeOutput(salida, salida)


def _iclora(positive, negative, vae, latent, image, frame_idx, strength, latent_downscale_factor, crop,
            use_tiled_encode, tile_size, tile_overlap, use_streaming, tile_frames):
    assert isinstance(vae, Etq)
    assert strength == 1.0 and use_tiled_encode is False and use_streaming is False
    assert (tile_size, tile_overlap, tile_frames) == (256, 64, 97)
    n = _fotogramas(latent)
    codigo, indice = _marca(image, 0)
    ultimo = _marca(image, image.shape[0] - 1)
    guia = ("iclora", frame_idx, image.shape[0])
    _apuntar("LTXAddVideoICLoRAGuide", positivo=positive.etiqueta, negativo=negative.etiqueta, frame_idx=frame_idx,
             crop=crop, factor=latent_downscale_factor, fotogramas_imagen=image.shape[0], n=n, marca=(codigo, indice),
             marca_ultima=ultimo)
    lat = _lat(latent)
    lat["doble"]["guias"] += 1
    return io.NodeOutput(positive.con(guia), negative.con(guia), lat)


class DobleDecode(M["LTXVTiledVAEDecode"]):
    """Mismo INPUT_TYPES que el real. Devuelve n fotogramas a un cuarto del tamano del latente decodificado."""

    def decode(self, vae, latents, horizontal_tiles, vertical_tiles, overlap, last_frame_fix,
               working_device="auto", working_dtype="auto"):
        assert isinstance(vae, Etq)
        assert (horizontal_tiles, vertical_tiles, overlap, last_frame_fix) == (2, 2, 4, False)
        assert (working_device, working_dtype) == ("auto", "auto")
        d = latents["doble"]
        assert d["guias"] == 0 and d["muestreado"] is not None, d
        yo = _yo()
        codigo = CODIGOS.setdefault(yo, len(CODIGOS) + 1)
        s = latents["samples"]
        n = _fotogramas(latents)
        h, w = s.shape[3] * 32 // 4, s.shape[4] * 32 // 4
        im = torch.full((n, h, w, 3), 0.2)
        for f in range(n):
            x = int(round(2 + f * (w - 10) / max(1, n - 1)))
            im[f, h // 3:h // 3 + 6, x:x + 6, :] = 0.9
            im[f, -1, 0, 0], im[f, -1, 0, 1], im[f, -1, 0, 2] = f / 1000, codigo / 1000, 0.0
        _apuntar("LTXVTiledVAEDecode", familia=d["muestreado"], n=n, codigo=codigo, tamano=(w, h))
        return (im,)


REALES = {}


def instalar_dobles():
    for nombre, f in (("LTXVAddGuide", _add_guide), ("LTXVCropGuides", _crop), ("LTXVLatentUpsampler", _subir),
                      ("CFGGuider", _guider), ("SamplerCustomAdvanced", _muestrear), ("LTXAddVideoICLoRAGuide", _iclora)):
        REALES[nombre] = M[nombre]
        M[nombre] = _doble_v3(nombre, f)
    REALES["LTXVTiledVAEDecode"] = M["LTXVTiledVAEDecode"]
    M["LTXVTiledVAEDecode"] = DobleDecode
    M.update(DOBLES_FUENTE)
    M["Doble_Recoger"] = Doble_Recoger


instalar_dobles()
DOBLES = set(REALES) | set(DOBLES_FUENTE) | {"Doble_Recoger"}

# ------------------------------------------------------------------------------------------------ motor
DINAMICOS = []


class _DynamicPrompt(execution.DynamicPrompt):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        DINAMICOS.append(self)


execution.DynamicPrompt = _DynamicPrompt


class Servidor:
    """Lo que el motor usa del servidor (comfy_execution/server_protocol.py), guardando los mensajes."""

    def __init__(self):
        self.client_id, self.last_node_id, self.sockets_metadata = None, None, {}
        self.mensajes = []

    def send_sync(self, event, data, sid=None):
        self.mensajes.append((event, data))

    def queue_updated(self):
        pass


SERVIDOR = Servidor()
# la misma cache que main.py sin opciones (RAM_PRESSURE), que es lo que usa el trabajador por defecto
_ram = min(10.0, max(2.0, comfy.model_management.total_ram * 0.10 / 1024.0))
EJECUTOR = execution.PromptExecutor(SERVIDOR, cache_type=execution.CacheType.RAM_PRESSURE,
                                    cache_args={"lru": 0, "ram": _ram, "ram_inactive": min(128.0, comfy.model_management.total_ram / 1024.0)})


def ejecutar(prompt, nombre):
    """validate_prompt + PromptExecutor.execute, como server.py y main.py. Devuelve (registro, mensajes, valido)."""
    valido = asyncio.run(execution.validate_prompt(nombre, prompt, None))
    assert valido[0], f"{nombre}: el prompt no valida: {valido[1]} {valido[3]}"
    del REGISTRO[:]
    del SERVIDOR.mensajes[:]
    RECOGIDO.clear()
    EJECUTOR.execute(prompt, nombre, {"client_id": "prueba"}, valido[2])
    # main.py manda el executing con nodo nulo al terminar: el backend de la app espera ese mensaje
    SERVIDOR.send_sync("executing", {"node": None, "prompt_id": nombre}, "prueba")
    return list(REGISTRO), list(SERVIDOR.mensajes), valido


# ------------------------------------------------------------------------------------------------ prompts
# 6 poses; la tercera es una espera larga (20 fotogramas > transicion_max) y la ultima dura 8 fotogramas
MAPA = {"bloques": [[0, 3], [4, 7], [8, 27], [28, 31], [32, 35], [36, 43]]}
ANCHO, ALTO, FACTOR = 256, 128, 2.0


def prompt_raiz(semilla=42, mapa=MAPA, n_poses=6, plan_literal=None):
    p = {
        "1": {"class_type": "Doble_Poses", "inputs": {"n": n_poses}},
        "2": {"class_type": "PrimitiveString", "inputs": {"value": json.dumps(mapa)}},
        "3": {"class_type": "CP_LentaPlan", "inputs": {"poses": ["1", 0], "mapa": ["2", 0], "umbral": 9.0,
                                                       "max_ventana": 49, "transicion_max": 12, "max_total": 1601}},
        "4": {"class_type": "RandomNoise", "inputs": {"noise_seed": semilla}},
        "5": {"class_type": "Doble_Arte", "inputs": {}},
        "6": {"class_type": "Doble_Modelo", "inputs": {"etiqueta": "suave"}},
        "7": {"class_type": "Doble_Modelo", "inputs": {"etiqueta": "render"}},
        "8": {"class_type": "Doble_Factor", "inputs": {"valor": FACTOR}},
        "9": {"class_type": "Doble_VAE", "inputs": {}},
        "10": {"class_type": "Doble_Escalador", "inputs": {}},
        "11": {"class_type": "Doble_Cond", "inputs": {"etiqueta": "suave+"}},
        "12": {"class_type": "Doble_Cond", "inputs": {"etiqueta": "suave-"}},
        "13": {"class_type": "Doble_Cond", "inputs": {"etiqueta": "render+"}},
        "14": {"class_type": "Doble_Cond", "inputs": {"etiqueta": "render-"}},
        "20": {"class_type": "CP_BlockingLento", "inputs": {
            "plan": ["3", 0], "ancho": ANCHO, "alto": ALTO, "ruido": ["4", 0], "poses": ["1", 0], "arte": ["5", 0],
            "modelo_suave": ["6", 0], "modelo_render": ["7", 0], "factor_guia": ["8", 0], "vae": ["9", 0],
            "escalador": ["10", 0], "positivo_suave": ["11", 0], "negativo_suave": ["12", 0],
            "positivo_render": ["13", 0], "negativo_render": ["14", 0]}},
        "30": {"class_type": "Doble_Recoger", "inputs": {"animacion": ["20", 0], "playblast": ["20", 1]}},
    }
    if plan_literal is not None:
        p["2b"] = {"class_type": "PrimitiveString", "inputs": {"value": plan_literal}}
        p["20"]["inputs"]["plan"] = ["2b", 0]
    return p


FUENTES = {"1": "Doble_Poses", "5": "Doble_Arte", "6": "Doble_Modelo", "7": "Doble_Modelo", "8": "Doble_Factor",
           "9": "Doble_VAE", "10": "Doble_Escalador", "11": "Doble_Cond", "12": "Doble_Cond", "13": "Doble_Cond",
           "14": "Doble_Cond"}

FALLOS = []


def comprobar(cond, texto):
    print(("ok   " if cond else "FALLO ") + texto)
    if not cond:
        FALLOS.append(texto)


def ventana_de(nodo):
    """w del id efimero (prefijo.wN.sK.nombre)."""
    for parte in nodo.split("."):
        if parte.startswith("w") and parte[1:].isdigit():
            return int(parte[1:])
    return None


def etapa_de(nodo):
    return 1 if ".s1." in nodo else 2 if ".s2." in nodo else None


def ejecutados(mensajes):
    return [d["node"] for e, d in mensajes if e == "executing" and d.get("node") is not None]


# ================================================================================================ 1-3: primera ejecucion
print("\n== 1. el prompt raiz valida y se ejecuta entero ==")
reg, mens, valido = ejecutar(prompt_raiz(), "p1")
comprobar(EJECUTOR.success, "execution_success: " + str([e for e, _ in mens if e.startswith("execution_")]))
dyn = DINAMICOS[-1]
plan = json.loads(next(d["inputs"]["plan"] for i, d in dyn.ephemeral_prompt.items() if d["class_type"] == "CP_LentaElegir"))
V = plan["ventanas"]
print(f"plan: total={plan['total']} lentos={plan['lentos']} saltos_px={[round(x, 1) for x in plan['saltos_px']]}")
print(f"ventanas: {[(v['p0'], v['p1'], v['n']) for v in V]}")
comprobar(len(V) >= 3, f"al menos 3 ventanas ({len(V)})")
anim, play = RECOGIDO.get("animacion"), RECOGIDO.get("playblast")
comprobar(anim is not None and anim.shape[0] == plan["total"], f"animacion con 'total' fotogramas: {None if anim is None else tuple(anim.shape)}")
comprobar(play is not None and tuple(play.shape) == tuple(anim.shape), f"playblast igual: {None if play is None else tuple(play.shape)}")
efimeros = dyn.ephemeral_prompt
print(f"nodos efimeros: {len(efimeros)}; clases: " + ", ".join(
    f"{c}={sum(1 for d in efimeros.values() if d['class_type'] == c)}" for c in sorted({d['class_type'] for d in efimeros.values()})))

print("\n== 2. raw_link ==")
cuentas = {}
for clase, nodo, _ in reg:
    cuentas[nodo] = cuentas.get(nodo, 0) + 1
for nid, clase in FUENTES.items():
    comprobar(cuentas.get(nid) == 1, f"la fuente {nid} ({clase}) se ejecuta una vez ({cuentas.get(nid, 0)})")
comprobar(all(v == 1 for v in cuentas.values()), "ningun nodo se ejecuta dos veces en la misma pasada")
por_raw = {"vae": "9", "upscale_model": "10", "noise": "4", "latent_downscale_factor": "8"}
mal = [(i, k, d["inputs"][k]) for i, d in efimeros.items() for k, src in por_raw.items()
       if k in d["inputs"] and d["inputs"][k] != [src, 0]]
comprobar(not mal, f"el subgrafo lleva los enlaces de origen tal cual (vae, escalador, ruido, factor) {mal[:3]}")
modelos = {tuple(d["inputs"]["model"]) for d in efimeros.values() if d["class_type"] == "CFGGuider"}
comprobar(modelos == {("6", 0), ("7", 0)}, f"los CFGGuider apuntan a los dos modelos raiz {sorted(modelos)}")

print("\n== 3. cableado semantico ==")
# los nodos efimeros, validados con el validador del propio motor (nadie los valida al ejecutarse)
completo = dict(dyn.original_prompt)
completo.update(efimeros)
validados, errores = {}, []
for i in efimeros:
    r = asyncio.run(execution.validate_inputs("p1", completo, i, validados))
    if not r[0]:
        errores.append((i, r[1]))
comprobar(not errores, f"los {len(efimeros)} nodos efimeros pasan execution.validate_inputs {errores[:2]}")

muestreos = [(n, d) for c, n, d in reg if c == "SamplerCustomAdvanced"]
s1 = [(n, d) for n, d in muestreos if "pose" in d["tipos_guia"]]
s2 = [(n, d) for n, d in muestreos if "iclora" in d["tipos_guia"]]
comprobar(len(s1) == 2 * len(V) and len(s2) == len(V), f"{len(s1)} muestreos de la etapa 1 y {len(s2)} de la 2")
comprobar(all((d["modelo"], d["positivo"], d["negativo"]) == ("suave", "suave+", "suave-") for _, d in s1),
          "etapa 1: modelo suave y conditioning suave en todos los muestreos")
comprobar(all((d["modelo"], d["positivo"], d["negativo"]) == ("render", "render+", "render-") for _, d in s2),
          "etapa 2: modelo render y conditioning render")
comprobar(all(d["semilla"] == 42 for _, d in muestreos), "todos los muestreos usan el ruido del panel (semilla 42)")
s8 = [1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0]
comprobar(all(d["sigmas"] == (s8[5:] if d["subido"] else s8) for _, d in muestreos),
          "sigmas: 8 pasos en el primer pase y en el render, 3 en el refinado")
comprobar(all("euler_ancestral" in d["muestreador"] for _, d in muestreos), f"muestreador euler_ancestral ({muestreos[0][1]['muestreador']})")
ic = [(n, d) for c, n, d in reg if c == "LTXAddVideoICLoRAGuide"]
comprobar(all(d["factor"] == FACTOR for _, d in ic), "el factor de la guia llega a las tres guias IC-LoRA de cada ventana")
guias_s1 = [(n, d) for c, n, d in reg if c == "LTXVAddGuide"]
dec = {n: d for c, n, d in reg if c == "LTXVTiledVAEDecode"}
for w, v in enumerate(V):
    esperado = {k: v["pos"][j] for j, k in enumerate(range(v["p0"], v["p1"] + 1))}
    for subido in (False, True):
        visto = {d["pose"]: d["frame_idx"] for n, d in guias_s1 if ventana_de(n) == w and d["subido"] == subido}
        tam = {d["tamano"] for n, d in guias_s1 if ventana_de(n) == w and d["subido"] == subido}
        comprobar(visto == esperado, f"w{w} {'2.o' if subido else '1.er'} pase: pose->frame_idx {visto} = pos del plan")
        comprobar(tam == {(ANCHO, ALTO) if subido else (ANCHO // 2, ALTO // 2)}, f"w{w} tamano de las guias {tam}")
    de_w = [d for n, d in ic if ventana_de(n) == w]
    g_guia = [d for d in de_w if d["frame_idx"] == 0 and d["crop"] == "disabled"]
    anclas = [d for d in de_w if d["crop"] == "center"]
    comprobar(len(g_guia) == 1 and g_guia[0]["fotogramas_imagen"] == v["n"] == g_guia[0]["n"],
              f"w{w} la guia de la etapa 2 tiene n={v['n']} fotogramas")
    cod_guia = [d["codigo"] for n, d in dec.items() if ventana_de(n) == w and d["familia"] == "suave"]
    comprobar(g_guia and cod_guia == [g_guia[0]["marca"][0]], f"w{w} la guia de la etapa 2 es el playblast de la misma ventana")
    if w == 0:
        esperada = (999, 0)
    else:
        previo = [d["codigo"] for n, d in dec.items() if ventana_de(n) == w - 1 and d["familia"] == "render"]
        esperada = (previo[0], V[w - 1]["n"] - 1)
    comprobar(sorted(d["frame_idx"] for d in anclas) == [-1, 0] and all(d["marca"] == esperada and d["fotogramas_imagen"] == 1 for d in anclas),
              f"w{w} ancla (arte -1 y pose 0) = {'el arte' if w == 0 else 'ultimo fotograma del render de w%d' % (w - 1)} {[d['marca'] for d in anclas]}")

orden = ejecutados(mens)
pos_de = {n: i for i, n in enumerate(orden)}


def primero(pred):
    return min((i for i, n in enumerate(orden) if pred(n)), default=None)


def ultimo(pred):
    return max((i for i, n in enumerate(orden) if pred(n)), default=None)


cadena_ok, ventana_a_ventana = True, True
for w in range(1, len(V)):
    rend_prev = ultimo(lambda n, w=w: ventana_de(n) == w - 1 and n.endswith(".s2.render"))
    ancla = pos_de.get(next(n for n in orden if ventana_de(n) == w and n.endswith(".ancla")))
    rend = ultimo(lambda n, w=w: ventana_de(n) == w and n.endswith(".s2.render"))
    eleg_prev = ultimo(lambda n, w=w: ventana_de(n) == w - 1 and n.endswith(".elegir"))
    eleg = ultimo(lambda n, w=w: ventana_de(n) == w and n.endswith(".elegir"))
    cadena_ok &= rend_prev < ancla < rend and eleg_prev < eleg
    ventana_a_ventana &= eleg_prev < primero(lambda n, w=w: ventana_de(n) == w)
comprobar(cadena_ok, "orden: render w-1 < ancla w < render w y elegir w-1 < elegir w")
comprobar(ventana_a_ventana, "cada ventana empieza despues de terminar la anterior (elegir w-1 antes del primer nodo de w)")
print("orden de ventanas en la ejecucion: " + "".join(str(ventana_de(n)) if ventana_de(n) is not None else "." for n in orden))

# de donde sale cada fotograma: (ventana, indice lento) igual en animacion y playblast, avanzando
cod_ventana = {d["codigo"]: (ventana_de(n), d["familia"]) for n, d in dec.items()}
origen_a = [(cod_ventana[_marca(anim, f)[0]], _marca(anim, f)[1]) for f in range(anim.shape[0])]
origen_p = [(cod_ventana[_marca(play, f)[0]], _marca(play, f)[1]) for f in range(play.shape[0])]
comprobar(all(a[0][1] == "render" for a in origen_a) and all(p[0][1] == "suave" for p in origen_p),
          "la animacion sale de los render y el playblast de las guias")
comprobar([(a[0][0], a[1]) for a in origen_a] == [(p[0][0], p[1]) for p in origen_p],
          "animacion y playblast eligen los mismos (ventana, indice lento)")
seq = [(a[0][0], a[1]) for a in origen_a]
comprobar(all(b >= a for a, b in zip(seq, seq[1:])), "los fotogramas elegidos avanzan siempre")
ini = plan["inicios"]
comprobar(all(seq[t] == (len(V) - 1, V[-1]["n"] - 1) for t in range(ini[-1], plan["total"])),
          f"la ultima pose se mantiene sus {plan['total'] - ini[-1]} fotogramas")
k_espera = 2
w_esp = next(w for w, v in enumerate(V) if v["p0"] <= k_espera < v["p1"])
quieto = ini[k_espera + 1] - ini[k_espera] - plan["transicion_max"]
comprobar(all(seq[t] == (w_esp, V[w_esp]["pos"][k_espera - V[w_esp]["p0"]]) for t in range(ini[k_espera], ini[k_espera] + quieto)),
          f"la espera larga de la pose {k_espera} se queda quieta {quieto} fotogramas")
print("elegidos (ventana, indice): " + " ".join(f"{w}:{i}" for w, i in seq))

print("\n== 4. cache ==")
reg2, mens2, _ = ejecutar(prompt_raiz(), "p2")
comprobar(EJECUTOR.success and not reg2, f"el mismo prompt otra vez: ningun doble se ejecuta ({len(reg2)})")
cacheados = next((d["nodes"] for e, d in mens2 if e == "execution_cached"), [])
print(f"execution_cached: {sorted(cacheados, key=str)}; executing: {ejecutados(mens2)}")
comprobar(RECOGIDO == {}, "Doble_Recoger no se vuelve a ejecutar")

reg3, mens3, _ = ejecutar(prompt_raiz(semilla=7), "p3")
comprobar(EJECUTOR.success, "con otra semilla se ejecuta entero")
cuenta1, cuenta3 = {}, {}
for c, n, d in reg:
    cuenta1[c] = cuenta1.get(c, 0) + 1
for c, n, d in reg3:
    cuenta3[c] = cuenta3.get(c, 0) + 1
print("dobles ejecutados (1.a pasada -> semilla nueva): " + ", ".join(f"{c} {cuenta1.get(c, 0)}->{cuenta3.get(c, 0)}" for c in sorted(cuenta1)))
dependen = ("SamplerCustomAdvanced", "LTXVCropGuides", "LTXVLatentUpsampler", "LTXVTiledVAEDecode", "LTXAddVideoICLoRAGuide",
            "Doble_Recoger")
comprobar(all(cuenta3.get(c) == cuenta1[c] for c in dependen), "se vuelve a ejecutar todo lo que depende del ruido")
comprobar(all(d["semilla"] == 7 for c, n, d in reg3 if c == "SamplerCustomAdvanced"), "los muestreos usan la semilla nueva")
comprobar(not any(c in DOBLES_FUENTE for c, _, _ in reg3), "las fuentes (modelos, vae, textos, poses, arte) no se repiten")
ej3 = ejecutados(mens3)
comprobar("3" not in ej3, "CP_LentaPlan no se repite (no depende del ruido)")
no_dep = [(c, n) for c, n, d in reg3 if c == "LTXVAddGuide" and not d["subido"]]
print(f"guias del primer pase de la etapa 1 (no dependen del ruido) que se repiten: {len(no_dep)} de "
      f"{sum(1 for c, n, d in reg if c == 'LTXVAddGuide' and not d['subido'])}")
print(f"CFGGuider repetidos: {cuenta3.get('CFGGuider', 0)} de {cuenta1['CFGGuider']}")

print("\n== 5. mensajes para el cliente ==")
ex = [(d["node"], d.get("display_node")) for e, d in mens if e == "executing"]
ef = [(n, dn) for n, dn in ex if n is not None and n in efimeros]
comprobar(ef and all(dn == "20" for _, dn in ef), f"executing de los efimeros: node = id efimero, display_node = '20' ({len(ef)} mensajes)")
print("ejemplos executing: " + "; ".join(f"node={n} display_node={dn}" for n, dn in ex[:3] + ef[:3] + ex[-2:]))
cuantos20 = sum(1 for n, _ in ex if n == "20")
print(f"executing con node='20' (el nodo expandible): {cuantos20} (se expande y despues resuelve sus salidas)")
ps = [d for e, d in mens if e == "progress_state"]
if ps:
    ult = ps[-1]["nodes"]
    claves = ["20"] + [k for k in sorted(ult) if k in efimeros][:1]
    print(f"progress_state: {len(ps)} mensajes; el ultimo con {len(ult)} nodos; ejemplo "
          + json.dumps({k: {kk: ult[k].get(kk) for kk in ('state', 'display_node_id', 'parent_node_id', 'real_node_id')} for k in claves}))
otros = sorted({e for e, _ in mens})
print(f"eventos emitidos: {otros}")
print(f"ultimo mensaje: {mens[-1]}")
comprobar(mens[-1] == ("executing", {"node": None, "prompt_id": "p1"}), "el ultimo executing es el de nodo nulo (fin del trabajo)")

print("\n== 6. planes invalidos ==")
for nombre, prompt, nodo_err, prefijo in (
        ("texto roto", prompt_raiz(plan_literal="{roto"), "20", "[CP_BlockingLento]"),
        ("plan de una pose", prompt_raiz(plan_literal=json.dumps({"version": 1, "inicios": [0], "total": 8, "saltos_px": [],
                                                                    "lentos": [], "transicion_max": 12, "ventanas": []})),
         "20", "[CP_BlockingLento]"),
        ("blocking de una sola pose", prompt_raiz(mapa={"bloques": [[0, 9]]}, n_poses=1), "3", "[CP_LentaPlan]")):
    regx, mensx, _ = ejecutar(prompt, "p_" + nombre.replace(" ", "_"))
    err = next((d for e, d in mensx if e == "execution_error"), None)
    comprobar(not EJECUTOR.success and err is not None and err["node_id"] == nodo_err and err["exception_message"].startswith(prefijo),
              f"{nombre}: execution_error en el nodo {err and err['node_id']} ({err and err['node_type']}): "
              f"{err and err['exception_message'].strip()}")
    pesados = [c for c, _, _ in regx if c in REALES]
    comprobar(not pesados, f"{nombre}: no se ejecuta ningun nodo pesado ({pesados})")

print("\n== 7. solo animacion ==")


def nuevo_ejecutor():
    """Cache vacia: para ver que se ejecuta de verdad, sin nada guardado de las secciones anteriores."""
    global EJECUTOR
    EJECUTOR = execution.PromptExecutor(SERVIDOR, cache_type=execution.CacheType.RAM_PRESSURE,
                                        cache_args={"lru": 0, "ram": _ram, "ram_inactive": min(128.0, comfy.model_management.total_ram / 1024.0)})


def por_clase(registro):
    cuenta = {}
    for c, _, _ in registro:
        cuenta[c] = cuenta.get(c, 0) + 1
    return cuenta


comprobar("solo_animacion" not in prompt_raiz()["20"]["inputs"],
          "el prompt de las secciones 1-6 no lleva la clave solo_animacion, como el de la tarjeta publicada")
nuevo_ejecutor()
p_solo = prompt_raiz()
p_solo["20"]["inputs"]["solo_animacion"] = True
reg7, mens7, valido7 = ejecutar(p_solo, "p7")
comprobar(EJECUTOR.success, "solo_animacion=True: execution_success " + str([e for e, _ in mens7 if e.startswith("execution_")]))
comprobar(valido7[2] == ["30"] and p_solo["20"]["inputs"]["solo_animacion"] is True, "valida con el BOOLEAN tal cual")
dyn7 = DINAMICOS[-1]
ef7 = dyn7.ephemeral_prompt
print(f"nodos efimeros: {len(ef7)} (con render: {len(efimeros)}); clases: " + ", ".join(
    f"{c}={sum(1 for d in ef7.values() if d['class_type'] == c)}" for c in sorted({d['class_type'] for d in ef7.values()})))
comprobar(not any(etapa_de(i) == 2 or i.endswith(".ancla") for i in ef7), "ningun nodo de la etapa 2 ni ancla en el subgrafo")
comprobar(all(d["inputs"]["guia"] == d["inputs"]["render"] for d in ef7.values() if d["class_type"] == "CP_LentaElegir"),
          "cada CP_LentaElegir recibe el playblast como guia y como render")
c7 = por_clase(reg7)
print("dobles ejecutados: " + ", ".join(f"{c} {n}" for c, n in sorted(c7.items())))
comprobar("LTXAddVideoICLoRAGuide" not in c7, "LTXAddVideoICLoRAGuide no se ejecuta")
m7 = [d for c, n, d in reg7 if c == "SamplerCustomAdvanced"]
comprobar(len(m7) == 2 * len(V) and all("pose" in d["tipos_guia"] and d["modelo"] == "suave" for d in m7),
          f"solo los {2 * len(V)} muestreos de la etapa 1, con el modelo suave ({len(m7)})")
comprobar(all(d["modelo"] == "suave" for c, n, d in reg7 if c == "CFGGuider"), "ningun CFGGuider con el modelo de render")
dec7 = {n: d for c, n, d in reg7 if c == "LTXVTiledVAEDecode"}
comprobar(len(dec7) == len(V) and all(d["familia"] == "suave" for d in dec7.values()), "solo se decodifica el playblast de cada ventana")
anim7, play7 = RECOGIDO.get("animacion"), RECOGIDO.get("playblast")
comprobar(anim7 is not None and anim7.shape[0] == plan["total"] and play7 is not None and tuple(play7.shape) == tuple(anim7.shape),
          f"animacion y playblast con 'total'={plan['total']} fotogramas: {None if anim7 is None else tuple(anim7.shape)}, "
          f"{None if play7 is None else tuple(play7.shape)}")
comprobar(anim7 is not None and torch.equal(anim7, play7), "animacion y playblast son la misma secuencia")
cod7 = {d["codigo"]: ventana_de(n) for n, d in dec7.items()}
seq7 = [(cod7[_marca(play7, f)[0]], _marca(play7, f)[1]) for f in range(play7.shape[0])] if play7 is not None else []
comprobar(seq7 == seq, "eligen los mismos (ventana, indice lento) que con render (el recorrido se mide en el playblast)")
de_arriba = {"5": "arte", "7": "modelo_render", "8": "factor_guia", "13": "positivo_render", "14": "negativo_render"}
corridos7 = {n for _, n, _ in reg7}
print("fuentes de arriba del render que se ejecutan con solo_animacion=True: "
      + ", ".join(f"{nid} {papel}: {'SI' if nid in corridos7 else 'no'}" for nid, papel in de_arriba.items()))
comprobar(all(nid in corridos7 for nid in ("1", "6", "9", "10", "11", "12")), "las fuentes de la etapa 1 se ejecutan")

# sin la clave (la tarjeta publicada) y con False: lo de la seccion 1, con la cache vacia
for nombre, prompt in (("sin la clave", prompt_raiz()), ("con False", dict(prompt_raiz()))):
    if nombre == "con False":
        prompt["20"] = json.loads(json.dumps(prompt["20"]))
        prompt["20"]["inputs"]["solo_animacion"] = False
    nuevo_ejecutor()
    regx, mensx, _ = ejecutar(prompt, "p8_" + nombre.replace(" ", "_"))
    efx = DINAMICOS[-1].ephemeral_prompt
    comprobar(EJECUTOR.success, f"{nombre}: execution_success")
    comprobar(efx == efimeros, f"{nombre}: el subgrafo es el de la seccion 1, nodo a nodo ({len(efx)} nodos)")
    comprobar(por_clase(regx) == cuenta1, f"{nombre}: los mismos dobles y las mismas veces que en la seccion 1")
    ax, px = RECOGIDO.get("animacion"), RECOGIDO.get("playblast")
    comprobar(ax is not None and torch.equal(ax, anim) and torch.equal(px, play), f"{nombre}: la misma animacion y el mismo playblast")

# ================================================================================================ 8: fotos
print("\n== 8. blocking en fotos ==")
from PIL import Image  # noqa: E402

ENTRADA = os.path.join(TRABAJO, "input")
folder_paths.set_input_directory(ENTRADA)  # la carpeta de trabajo, nunca la del motor
os.makedirs(ENTRADA, exist_ok=True)
comprobar(os.path.realpath(folder_paths.get_input_directory()).startswith(os.path.realpath(TRABAJO)),
          "la carpeta de entrada esta en la de trabajo")
N_FOTOS, D_FOTOS, ESPERA_FOTOS = 6, 12, 5


def foto(nombre, k):
    """Como _poses(): banda de arriba con (k+1)/20 (la lee _pose) y un cuadrado que salta; fondo liso."""
    xs = [6, 40, 74, 7, 41, 75]
    im = Image.new("RGB", (96, 54), (64, 64, 64))
    im.paste((int(round((k + 1) / 20 * 255)),) * 3, (0, 0, 96, 6))
    im.paste((230, 230, 230), (xs[k], 24, xs[k] + 14, 38))
    im.save(os.path.join(ENTRADA, nombre))
    return nombre


# casillas 1-4 y 7-8 con foto (los huecos no cuentan) y la 5 con el marcador de vacio
CASILLAS_FOTOS = dict(zip(["pose_01", "pose_02", "pose_03", "pose_04", "pose_07", "pose_08"],
                          [foto(f"prueba_pose_{k}.png", k) for k in range(N_FOTOS)]))
CASILLAS_FOTOS["pose_05"] = "PON-AQUI-UNA-POSE"


class Doble_Numero:
    """Nodo de salida: guarda un INT (la salida 'fotogramas' del nodo de fotos)."""
    OUTPUT_NODE = True
    RETURN_TYPES = ()
    FUNCTION = "f"
    CATEGORY = "pruebas"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"valor": ("INT", {"forceInput": True})}}

    def f(self, valor):
        RECOGIDO["fotogramas"] = valor
        return {}


M["Doble_Numero"] = Doble_Numero


def prompt_fotos(solo):
    p = prompt_raiz()
    del p["2"]
    p["1"] = {"class_type": "CP_PosesDesdeFotos", "inputs": dict(CASILLAS_FOTOS, fotogramas_por_pose=D_FOTOS,
                                                                  espera_final=ESPERA_FOTOS, orden="casilla")}
    p["3"]["inputs"].update(mapa=["1", 1], transicion_max=["1", 3], mas_lentos=True)  # como la tarjeta de fotos
    if solo is not None:
        p["20"]["inputs"]["solo_animacion"] = solo
    p["31"] = {"class_type": "Doble_Numero", "inputs": {"valor": ["1", 4]}}
    return p


esperados = (N_FOTOS - 1) * D_FOTOS + 1 + ESPERA_FOTOS
secuencias = {}
for solo in (False, True):
    nuevo_ejecutor()
    reg8, mens8, _ = ejecutar(prompt_fotos(solo), f"p9_{solo}")
    nombre = f"solo_animacion={solo}"
    comprobar(EJECUTOR.success, f"{nombre}: execution_success " + str([(e, d.get('exception_message', '')[:200]) for e, d in mens8 if e.startswith('execution_e')]))
    if not EJECUTOR.success:
        continue
    dyn8 = DINAMICOS[-1]
    plan8 = json.loads(next(d["inputs"]["plan"] for d in dyn8.ephemeral_prompt.values() if d["class_type"] == "CP_LentaElegir"))
    V8 = plan8["ventanas"]
    print(f"{nombre}: plan inicios={plan8['inicios']} total={plan8['total']} transicion_max={plan8['transicion_max']} "
          f"lentos={plan8['lentos']} ventanas={[(v['p0'], v['p1'], v['n']) for v in V8]}")
    comprobar(RECOGIDO.get("fotogramas") == esperados, f"{nombre}: el nodo de fotos anuncia {RECOGIDO.get('fotogramas')} fotogramas ({esperados})")
    comprobar(plan8["inicios"] == [k * D_FOTOS for k in range(N_FOTOS)] and plan8["total"] == esperados
              and plan8["transicion_max"] == D_FOTOS, f"{nombre}: el plan sale del mapa de las fotos con transicion_max = {D_FOTOS}")
    comprobar(plan8.get("mas_lentos") is True and "ritmo" not in plan8 and all(x == 32 for x in plan8["lentos"]),
              f"{nombre}: con mas_lentos cada salto de {D_FOTOS} fotogramas lleva 32 lentos")
    an8, pl8 = RECOGIDO.get("animacion"), RECOGIDO.get("playblast")
    comprobar(an8 is not None and an8.shape[0] == esperados and tuple(pl8.shape) == tuple(an8.shape),
              f"{nombre}: animacion y playblast con {esperados} fotogramas: {None if an8 is None else tuple(an8.shape)}")
    if an8 is None:
        continue
    dec8 = {n: d for c, n, d in reg8 if c == "LTXVTiledVAEDecode"}
    cod8 = {d["codigo"]: (ventana_de(n), d["familia"]) for n, d in dec8.items()}
    origen = [cod8[_marca(an8, f)[0]] + (_marca(an8, f)[1],) for f in range(an8.shape[0])]
    familia = "suave" if solo else "render"
    comprobar(all(o[1] == familia for o in origen), f"{nombre}: la animacion sale {'del playblast' if solo else 'de los render'}")
    if solo:
        comprobar(torch.equal(an8, pl8) and "LTXAddVideoICLoRAGuide" not in por_clase(reg8),
                  f"{nombre}: animacion = playblast y nada de la etapa 2")
    seq8 = [(o[0], o[2]) for o in origen]
    secuencias[solo] = seq8
    # cada pose en su fotograma: el lento del plan, y la guia de ese lento es la foto de esa pose
    guias8 = [(n, d) for c, n, d in reg8 if c == "LTXVAddGuide" and not d["subido"]]
    en_su_sitio = True
    for k in range(N_FOTOS):
        w = max(i for i, v in enumerate(V8) if v["p0"] <= k)
        lento = V8[w]["pos"][k - V8[w]["p0"]]
        guia = {d["pose"] for n, d in guias8 if ventana_de(n) == w and d["frame_idx"] == lento}
        en_su_sitio &= seq8[k * D_FOTOS] == (w, lento) and guia == {k}
    comprobar(en_su_sitio, f"{nombre}: cada pose k cae en el fotograma k x {D_FOTOS} y su guia es la foto k")
    hasta = (N_FOTOS - 1) * D_FOTOS
    comprobar(all(b > a for a, b in zip(seq8[:hasta], seq8[1:hasta + 1])),
              f"{nombre}: de la primera pose a la ultima se mueve en todos los fotogramas")
    comprobar(seq8[hasta:] == [(len(V8) - 1, V8[-1]["n"] - 1)] * (1 + ESPERA_FOTOS),
              f"{nombre}: la ultima pose se queda {1 + ESPERA_FOTOS} fotogramas")
    comprobar(ejecutados(mens8).count("1") == 1, f"{nombre}: CP_PosesDesdeFotos (el de verdad) se ejecuta una vez")
comprobar(secuencias.get(True) == secuencias.get(False), "con y sin render se eligen los mismos (ventana, indice lento)")

# ================================================================================================ 9: ritmo 2
print("\n== 9. ritmo 2 con el mapa de video ==")
nuevo_ejecutor()
p_ritmo = prompt_raiz()
p_ritmo["3"]["inputs"]["ritmo"] = 2
p_ritmo["20"]["inputs"]["solo_animacion"] = True
reg9, mens9, _ = ejecutar(p_ritmo, "p10")
comprobar(EJECUTOR.success, "ritmo 2: execution_success")
plan9 = json.loads(next(d["inputs"]["plan"] for d in DINAMICOS[-1].ephemeral_prompt.values() if d["class_type"] == "CP_LentaElegir"))
V9 = plan9["ventanas"]
print(f"plan: inicios={plan9['inicios']} total={plan9['total']} transicion_max={plan9['transicion_max']} lentos={plan9['lentos']} "
      f"ventanas={[(v['p0'], v['p1'], v['n']) for v in V9]}")
comprobar(plan9.get("ritmo") == 2 and plan9["inicios"] == [2 * x for x in plan["inicios"]] and plan9["total"] == 2 * plan["total"]
          and plan9["transicion_max"] == 2 * plan["transicion_max"], "el plan lleva inicios, total y transicion_max al doble")
an9 = RECOGIDO.get("animacion")
comprobar(an9 is not None and an9.shape[0] == 2 * plan["total"], f"la animacion tiene {2 * plan['total']} fotogramas: "
          f"{None if an9 is None else tuple(an9.shape)}")
if an9 is not None:
    dec9 = {n: d for c, n, d in reg9 if c == "LTXVTiledVAEDecode"}
    cod9 = {d["codigo"]: ventana_de(n) for n, d in dec9.items()}
    seq9 = [(cod9[_marca(an9, f)[0]], _marca(an9, f)[1]) for f in range(an9.shape[0])]
    ini9 = plan9["inicios"]
    comprobar(all(seq9[ini9[k]] == (max(i for i, v in enumerate(V9) if v["p0"] <= k),
                                    V9[max(i for i, v in enumerate(V9) if v["p0"] <= k)]["pos"][k - V9[max(i for i, v in enumerate(V9) if v["p0"] <= k)]["p0"]])
                  for k in range(len(ini9))), "cada pose cae en el doble de su inicio")
    comprobar(all(b >= a for a, b in zip(seq9, seq9[1:])), "los fotogramas elegidos avanzan siempre")
    quieto9 = ini9[3] - ini9[2] - plan9["transicion_max"]
    tramo9 = seq9[ini9[2] + quieto9:ini9[3] + 1]
    comprobar(quieto9 > 0 and len(set(seq9[ini9[2]:ini9[2] + quieto9 + 1])) == 1 and all(b > a for a, b in zip(tramo9, tramo9[1:])),
              f"la espera larga sigue quieta {quieto9} fotogramas y el salto ocupa los {plan9['transicion_max']} ultimos")

# ================================================================================================ 10: tarjeta publicada
print("\n== 10. la tarjeta publicada (sin ritmo, mas_lentos ni solo_animacion) da el plan y el subgrafo de la v0.14 ==")
import hashlib  # noqa: E402

with open(os.path.join(PACK, "tests", "datos", "compat_v014.json"), encoding="utf-8") as _f:
    V014 = json.load(_f)["tarjeta_publicada"]
with open(os.path.join(PACK, "tests", "datos", "lenta_0003.json"), encoding="utf-8") as _f:
    CONTROL = json.load(_f)
N_CONTROL = len(CONTROL["inicios"])


def _poses_control():
    im = torch.full((N_CONTROL, 54, 96, 3), 0.25)
    for k in range(N_CONTROL):
        im[k, :6, :, :] = (k + 1) / 40
    return im


class Doble_Texto:
    """Nodo de salida: guarda el texto del plan tal cual sale de CP_LentaPlan."""
    OUTPUT_NODE = True
    RETURN_TYPES = ()
    FUNCTION = "f"
    CATEGORY = "pruebas"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"texto": ("STRING", {"forceInput": True})}}

    def f(self, texto):
        RECOGIDO["plan"] = texto
        return {}


M["Doble_Poses_Control"] = _fuente("Doble_Poses_Control", "IMAGE", lambda: _poses_control())
M["Doble_Texto"] = Doble_Texto
p10 = prompt_raiz(mapa={"bloques": CONTROL["bloques"]})
p10["1"] = {"class_type": "Doble_Poses_Control", "inputs": {}}
p10["3"]["inputs"].update(V014["plan_entradas"])  # umbral, max_ventana, transicion_max y max_total de la tarjeta
p10["32"] = {"class_type": "Doble_Texto", "inputs": {"texto": ["3", 0]}}
g10 = V014["grafos"]["prueba"]
p10["20"]["inputs"].update(ancho=g10["ancho"], alto=g10["alto"])
enlaces10 = {k: p10["20"]["inputs"][k] for k in g10["enlaces"]}
comprobar(enlaces10 == g10["enlaces"], "el prompt lleva los enlaces con los que se genero la huella del subgrafo")
comprobar(sorted(p10["3"]["inputs"]) == ["mapa", "max_total", "max_ventana", "poses", "transicion_max", "umbral"]
          and "solo_animacion" not in p10["20"]["inputs"], "CP_LentaPlan sin ritmo ni mas_lentos y CP_BlockingLento sin solo_animacion")
# la medida de los saltos es la del plano de control (las poses de la prueba son sinteticas)
_mod_plan = sys.modules[M["CP_LentaPlan"].__module__]
_medir = _mod_plan.medir_saltos
_mod_plan.medir_saltos = lambda poses, nodo="": list(CONTROL["saltos_dis_480"])
try:
    nuevo_ejecutor()
    reg10, mens10, _ = ejecutar(p10, "p11")
finally:
    _mod_plan.medir_saltos = _medir
comprobar(EJECUTOR.success, "tarjeta publicada: execution_success " + str([(e, d.get('exception_message', '')[:200]) for e, d in mens10 if e.startswith('execution_e')]))
texto10 = RECOGIDO.get("plan") or "{}"
plan10 = json.loads(texto10)
print(f"plan: lentos={plan10.get('lentos')} ventanas={[(v['p0'], v['p1'], v['n']) for v in plan10.get('ventanas', [])]} "
      f"generados={sum(v['n'] for v in plan10.get('ventanas', []))}")
comprobar(hashlib.sha256(texto10.encode("utf-8")).hexdigest() == V014["plan"]["sha256"] and "ritmo" not in plan10 and "mas_lentos" not in plan10,
          "el plan es el de la v0.14 byte a byte (huella del JSON)")
ef10 = DINAMICOS[-1].ephemeral_prompt
_final = [i for i in ef10 if i.endswith("w0.elegir")]
prefijo10 = _final[0][:-len("w0.elegir")] if len(_final) == 1 else None


def _sin_prefijo(v):
    if isinstance(v, str) and prefijo10 and v.startswith(prefijo10):
        return "@" + v[len(prefijo10):]
    if isinstance(v, list) and len(v) == 2 and isinstance(v[1], int):
        return [_sin_prefijo(v[0]), v[1]]
    return v


limpio10 = {_sin_prefijo(i): {"class_type": d["class_type"], "inputs": {k: _sin_prefijo(x) for k, x in d["inputs"].items()}}
            for i, d in ef10.items()}
huella10 = hashlib.sha256(json.dumps(limpio10, sort_keys=True).encode("utf-8")).hexdigest()
print(f"subgrafo: {len(ef10)} nodos (v0.14: {g10['nodos']}), prefijo {prefijo10!r}")
comprobar(prefijo10 is not None and len(ef10) == g10["nodos"] and huella10 == g10["sha256"],
          "el subgrafo que expande el motor es el de la v0.14, nodo a nodo (huella)")
an10 = RECOGIDO.get("animacion")
comprobar(an10 is not None and an10.shape[0] == CONTROL["bloques"][-1][1] + 1 and len([1 for c, _, _ in reg10 if c == "LTXAddVideoICLoRAGuide"]) > 0,
          f"la animacion tiene los {CONTROL['bloques'][-1][1] + 1} fotogramas del blocking y pasa por el render")

# ------------------------------------------------------------------------------------------------ cierre
fuera = [os.path.join(r, f) for r, _, fs in os.walk(folder_paths.get_output_directory()) for f in fs]
comprobar(not fuera, f"nada escrito en output/ ({fuera[:3]})")
print(f"\n{len(FALLOS)} fallos en {time.time() - T0:.1f} s")
for f in FALLOS:
    print("  FALLO " + f)
sys.exit(1 if FALLOS else 0)
