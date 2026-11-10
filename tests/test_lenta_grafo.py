# -*- coding: utf-8 -*-
"""Prueba el subgrafo de camara lenta (CP_BlockingLento) sin ComfyUI: lo compara nodo a nodo con los grafos con los que
se valido la receta (tests/datos/) y comprueba el encadenado entre ventanas.
Se ejecuta con: python tests/test_lenta_grafo.py"""
import copy
import hashlib
import json
import os
import re
import sys
import types
from collections import Counter

# ---- doble minimo de comfy_api.latest ----
mod = types.ModuleType("comfy_api")
latest = types.ModuleType("comfy_api.latest")


class _Port:
    tipo = None

    def __init__(self, *a, **k):
        self.id = a[0] if a else None
        self.k = k


def _mk(name):
    return type(name, (), {"Input": type("Input", (_Port,), {"tipo": name}), "Output": type("Output", (_Port,), {"tipo": name})})


class _Salida:
    def __init__(self, *a, **k):
        self.args = a
        self.expand = k.get("expand")


class _IO:
    Schema = staticmethod(lambda **k: k)
    NodeOutput = _Salida
    ComfyNode = type("ComfyNode", (), {})


for _n in ("Image", "Int", "String", "Float", "Noise", "Model", "Vae", "LatentUpscaleModel", "Conditioning"):
    setattr(_IO, _n, _mk(_n))

latest.io = _IO
mod.latest = latest
sys.modules["comfy_api"] = mod
sys.modules["comfy_api.latest"] = latest


# ---- doble de comfy_execution.graph_utils: el reparto de prefijos, copiado de ComfyUI 0.42.1 ----
class GraphBuilder:
    _default_prefix_root = ""
    _default_prefix_call_index = 0
    _default_prefix_graph_index = 0

    @classmethod
    def set_default_prefix(cls, prefix_root, call_index, graph_index=0):
        cls._default_prefix_root = prefix_root
        cls._default_prefix_call_index = call_index
        cls._default_prefix_graph_index = graph_index

    @classmethod
    def alloc_prefix(cls, root=None, call_index=None, graph_index=None):
        root = GraphBuilder._default_prefix_root if root is None else root
        call_index = GraphBuilder._default_prefix_call_index if call_index is None else call_index
        graph_index = GraphBuilder._default_prefix_graph_index if graph_index is None else graph_index
        GraphBuilder._default_prefix_graph_index += 1
        return f"{root}.{call_index}.{graph_index}."


_ejec = types.ModuleType("comfy_execution")
_utiles = types.ModuleType("comfy_execution.graph_utils")
_utiles.GraphBuilder = GraphBuilder
_ejec.graph_utils = _utiles
sys.modules["comfy_execution"] = _ejec
sys.modules["comfy_execution.graph_utils"] = _utiles

RAIZ = os.path.join(os.path.dirname(__file__), "..", "comfy_character_pipeline")
sys.path.insert(0, RAIZ)
import lenta_grafo as L  # noqa: E402

DATOS = os.path.join(os.path.dirname(__file__), "datos")


def _json(nombre):
    with open(os.path.join(DATOS, nombre), encoding="utf-8") as f:
        return json.load(f)


SUAVE = _json("grafo_suave_p13-20.json")  # etapa 1 validada: poses 13..20
RENDER = _json("grafo_render_un_pase.json")  # etapa 2 validada: un pase a tamano completo
CONTROL = _json("lenta_0003.json")


def _plan(ventanas, lentos, inicios=None):
    """Un plan con el formato de CP_LentaPlan a partir de ventanas (p0, p1) y lentos por salto."""
    vs = []
    for p0, p1 in ventanas:
        pos = [0]
        for x in lentos[p0:p1]:
            pos.append(pos[-1] + x)
        vs.append({"p0": p0, "p1": p1, "pos": pos, "n": pos[-1] + 1})
    inicios = list(inicios) if inicios else list(range(0, 4 * (len(lentos) + 1), 4))
    return {"version": 1, "inicios": inicios, "total": inicios[-1] + 1, "saltos_px": [10.0] * len(lentos),
            "lentos": list(lentos), "transicion_max": 12, "ventanas": vs}


# Cada papel con su enlace propio: si el subgrafo cruza dos, se nota.
ENLACES = {
    "ruido": ["semilla", 0], "poses": ["poses", 0], "arte": ["arte", 0],
    "modelo_suave": ["sage_suave", 0], "modelo_render": ["sage_render", 0], "factor_guia": ["ic_lora", 1],
    "vae": ["vae", 0], "escalador": ["escalador", 0],
    "positivo_suave": ["cond_suave", 0], "negativo_suave": ["cond_suave", 1],
    "positivo_render": ["cond_render", 0], "negativo_render": ["cond_render", 1],
}
TIPO_ENLACE = {
    "ruido": "NOISE", "poses": "IMAGE", "arte": "IMAGE", "modelo_suave": "MODEL", "modelo_render": "MODEL",
    "factor_guia": "FLOAT", "vae": "VAE", "escalador": "LATENT_UPSCALE_MODEL", "positivo_suave": "CONDITIONING",
    "negativo_suave": "CONDITIONING", "positivo_render": "CONDITIONING", "negativo_render": "CONDITIONING",
}
PRE = "12.0.0."  # como los reparte el motor: <id del nodo>.<llamada>.<grafo>.
A, H = 1920, 1088
# Las ventanas y los lentos que se usaron a mano en la version validada: 4 ventanas; la 2 son las poses 13..20.
PLAN_V8 = _plan(CONTROL["ventanas_v8"], CONTROL["lentos_v8"], CONTROL["inicios"])
GRAFO, ANIMACION, PLAYBLAST = L.grafo_lento(PLAN_V8, ENLACES, A, H, PRE)
W_SUAVE = 2

# Lo que el motor NO comprueba en una expansion, copiado de los esquemas de ComfyUI 0.42.1 y de ComfyUI-LTXVideo.
ENTRADAS = {
    "EmptyLTXVLatentVideo": {"width": "INT", "height": "INT", "length": "INT", "batch_size": "INT"},
    "ImageFromBatch": {"image": "IMAGE", "batch_index": "INT", "length": "INT"},
    "ImageScale": {"image": "IMAGE", "upscale_method": ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"],
                   "width": "INT", "height": "INT", "crop": ["disabled", "center"]},
    "LTXVAddGuide": {"positive": "CONDITIONING", "negative": "CONDITIONING", "vae": "VAE", "latent": "LATENT",
                     "image": "IMAGE", "frame_idx": "INT", "strength": "FLOAT"},
    "CFGGuider": {"model": "MODEL", "positive": "CONDITIONING", "negative": "CONDITIONING", "cfg": "FLOAT"},
    "KSamplerSelect": {"sampler_name": ["euler", "euler_ancestral"]},  # de la lista larga del motor, lo que hace falta
    "ManualSigmas": {"sigmas": "STRING"},
    "SamplerCustomAdvanced": {"noise": "NOISE", "guider": "GUIDER", "sampler": "SAMPLER", "sigmas": "SIGMAS",
                              "latent_image": "LATENT"},
    "LTXVCropGuides": {"positive": "CONDITIONING", "negative": "CONDITIONING", "latent": "LATENT"},
    "LTXVLatentUpsampler": {"samples": "LATENT", "upscale_model": "LATENT_UPSCALE_MODEL", "vae": "VAE"},
    "LTXVTiledVAEDecode": {"vae": "VAE", "latents": "LATENT", "horizontal_tiles": "INT", "vertical_tiles": "INT",
                           "overlap": "INT", "last_frame_fix": "BOOLEAN", "working_device": ["cpu", "auto"],
                           "working_dtype": ["float16", "float32", "auto"]},
    "LTXAddVideoICLoRAGuide": {"positive": "CONDITIONING", "negative": "CONDITIONING", "vae": "VAE", "latent": "LATENT",
                               "image": "IMAGE", "frame_idx": "INT", "strength": "FLOAT",
                               "latent_downscale_factor": "FLOAT", "crop": ["disabled", "center"],
                               "use_tiled_encode": "BOOLEAN", "tile_size": "INT", "tile_overlap": "INT",
                               "use_streaming": "BOOLEAN", "tile_frames": "INT"},
    "CP_LentaElegir": {"guia": "IMAGE", "render": "IMAGE", "plan": "STRING", "ventana": "INT",
                       "anterior_render": "IMAGE", "anterior_guia": "IMAGE"},
}
OPCIONALES = {"CP_LentaElegir": {"anterior_render", "anterior_guia"}}
SALIDAS = {
    "EmptyLTXVLatentVideo": ["LATENT"], "ImageFromBatch": ["IMAGE"], "ImageScale": ["IMAGE"],
    "LTXVAddGuide": ["CONDITIONING", "CONDITIONING", "LATENT"], "CFGGuider": ["GUIDER"], "KSamplerSelect": ["SAMPLER"],
    "ManualSigmas": ["SIGMAS"], "SamplerCustomAdvanced": ["LATENT", "LATENT"],
    "LTXVCropGuides": ["CONDITIONING", "CONDITIONING", "LATENT"], "LTXVLatentUpsampler": ["LATENT"],
    "LTXVTiledVAEDecode": ["IMAGE"], "LTXAddVideoICLoRAGuide": ["CONDITIONING", "CONDITIONING", "LATENT"],
    "CP_LentaElegir": ["IMAGE", "IMAGE"],
}
PYTHON = {"INT": int, "FLOAT": float, "BOOLEAN": bool, "STRING": str}


def _enlace(v):
    return isinstance(v, list) and len(v) == 2 and isinstance(v[0], str) and isinstance(v[1], int) and not isinstance(v[1], bool)


# ---- canonizacion ----
class Canon:
    """Firma recursiva de cada nodo: su clase y TODAS sus entradas con su nombre. Un enlace a un nodo frontera se cambia
    por su PAPEL (modelo, vae, pose 13...); cualquier otro enlace, por la firma del nodo de origen y su indice de
    salida: dos nodos firman igual solo si todo lo que tienen por debajo, hasta la frontera, es igual. Los literales
    entran tal cual; los numeros, por su valor (1 y 1.0 firman igual: el motor los convertia al validar los grafos de
    control; los tipos exactos del subgrafo se prueban aparte). `medidas` da papel a un literal de UN nodo y solo si
    vale lo esperado: es la unica diferencia admitida que toca un valor (el tamano del latente de la etapa 2)."""

    def __init__(self, grafo, papeles, medidas=None):
        self.grafo, self.papeles, self.medidas = grafo, papeles, medidas or {}
        self.firmas, self.sueltos = {}, set()

    def valor(self, ident, nombre, v):
        if _enlace(v):
            if (v[0], v[1]) in self.papeles:
                return ["papel", self.papeles[(v[0], v[1])]]
            if v[0] in self.grafo:
                return ["nodo", self.firma(v[0]), v[1]]
            self.sueltos.add((v[0], v[1]))
            return ["suelto", v[0], v[1]]
        if (ident, nombre) in self.medidas:
            esperado, papel = self.medidas[(ident, nombre)]
            if type(v) is int and v == esperado:
                return ["papel", papel]
        if isinstance(v, bool):
            return ["si o no", v]
        if isinstance(v, (int, float)):
            return ["numero", repr(float(v))]
        if isinstance(v, str):
            return ["texto", v]
        raise AssertionError(f"valor que no se sabe canonizar en {ident}.{nombre}: {v!r}")

    def firma(self, ident):
        if ident not in self.firmas:
            n = self.grafo[ident]
            forma = [n["class_type"], sorted([k, self.valor(ident, k, v)] for k, v in n["inputs"].items())]
            self.firmas[ident] = hashlib.sha256(json.dumps(forma, ensure_ascii=False).encode()).hexdigest()
        return self.firmas[ident]

    def debajo(self, raiz):
        """Firma de cada nodo del que depende raiz sin cruzar la frontera."""
        vistos, pila = {}, [raiz]
        while pila:
            i = pila.pop()
            if i in vistos:
                continue
            vistos[i] = self.firma(i)
            pila += [v[0] for v in self.grafo[i]["inputs"].values()
                     if _enlace(v) and (v[0], v[1]) not in self.papeles and v[0] in self.grafo]
        return vistos


def _diferencia(a, ia, b, ib, camino="decodificado"):
    """None si los dos nodos firman igual; si no, la primera diferencia bajando por los enlaces."""
    if a.firma(ia) == b.firma(ib):
        return None
    na, nb = a.grafo[ia], b.grafo[ib]
    if na["class_type"] != nb["class_type"]:
        return f"{camino}: {na['class_type']} ({ia}) frente a {nb['class_type']} ({ib})"
    if set(na["inputs"]) != set(nb["inputs"]):
        return f"{camino}: entradas {sorted(na['inputs'])} ({ia}) frente a {sorted(nb['inputs'])} ({ib})"
    for nombre in sorted(na["inputs"]):
        va, vb = a.valor(ia, nombre, na["inputs"][nombre]), b.valor(ib, nombre, nb["inputs"][nombre])
        if va == vb:
            continue
        if va[0] == vb[0] == "nodo" and va[2] == vb[2]:
            return _diferencia(a, na["inputs"][nombre][0], b, nb["inputs"][nombre][0], f"{camino} <- {nombre}")
        return f"{camino} <- {nombre}: {va} ({ia}) frente a {vb} ({ib})"
    raise AssertionError("firmas distintas sin diferencia visible")


def _clases(grafo, esperadas):
    for ident, clase in esperadas.items():
        assert grafo[ident]["class_type"] == clase, (ident, grafo[ident]["class_type"])


def _canon_suave_control(grafo=SUAVE):
    _clases(grafo, {"59": "PathchSageAttentionKJ", "61": "VAELoader", "62": "LatentUpscaleModelLoader",
                    "69": "LTXVConditioning", "89": "LTXVTiledVAEDecode"})
    papeles = {("59", 0): "modelo", ("61", 0): "vae", ("62", 0): "escalador", ("69", 0): "positivo", ("69", 1): "negativo"}
    for ident, n in grafo.items():
        if n["class_type"] == "RandomNoise":  # sueltos y con la misma semilla: equivalen a un unico ruido
            assert n["inputs"] == {"noise_seed": 42}, n
            papeles[(ident, 0)] = "ruido"
        if n["class_type"] == "LoadImage":
            papeles[(ident, 0)] = "pose %d" % int(re.fullmatch(r"bitt_mcp/pose_(\d+)\.png", n["inputs"]["image"]).group(1))
    return Canon(grafo, papeles)


def _canon_suave(grafo):
    papeles = {tuple(ENLACES[k]): p for k, p in (("modelo_suave", "modelo"), ("vae", "vae"), ("escalador", "escalador"),
                                                 ("positivo_suave", "positivo"), ("negativo_suave", "negativo"),
                                                 ("ruido", "ruido"))}
    for ident, n in grafo.items():
        e = n["inputs"]
        # una pose es EXACTAMENTE una imagen del lote de poses: el papel sale de lo que dice el nodo
        if n["class_type"] == "ImageFromBatch" and e.get("image") == ENLACES["poses"] and e.get("length") == 1 \
                and type(e.get("batch_index")) is int and set(e) == {"image", "batch_index", "length"}:
            papeles[(ident, 0)] = "pose %d" % e["batch_index"]
    return Canon(grafo, papeles)


def _canon_render_control(grafo=RENDER):
    _clases(grafo, {"59": "PathchSageAttentionKJ", "61": "VAELoader", "69": "LTXVConditioning", "9": "RandomNoise",
                    "58": "LTXICLoRALoaderModelOnly", "g2": "GetVideoComponents", "a0": "LoadImage",
                    "20": "GetImageSize", "89": "LTXVTiledVAEDecode"})
    assert grafo["20"]["inputs"] == {"image": ["g2", 0]}  # el latente mide lo que la guia
    return Canon(grafo, {("59", 0): "modelo", ("61", 0): "vae", ("69", 0): "positivo", ("69", 1): "negativo",
                         ("58", 1): "factor", ("9", 0): "ruido", ("g2", 0): "guia", ("a0", 0): "ancla",
                         ("20", 0): "tamano: ancho", ("20", 1): "tamano: alto", ("20", 2): "tamano: fotogramas"})


def _canon_render(grafo, w):
    pre = f"{PRE}w{w}."
    ancla = ENLACES["arte"] if w == 0 else [pre + "ancla", 0]
    papeles = {tuple(ENLACES[k]): p for k, p in (("modelo_render", "modelo"), ("vae", "vae"), ("positivo_render", "positivo"),
                                                 ("negativo_render", "negativo"), ("factor_guia", "factor"), ("ruido", "ruido"))}
    papeles[(pre + "s1.guia", 0)] = "guia"  # la de SU ventana
    papeles[tuple(ancla)] = "ancla"
    medidas = {(pre + "s2.latente", "width"): (A, "tamano: ancho"), (pre + "s2.latente", "height"): (H, "tamano: alto"),
               (pre + "s2.latente", "length"): (PLAN_V8["ventanas"][w]["n"], "tamano: fotogramas")}
    return Canon(grafo, papeles, medidas)


def _paridad_suave(grafo, control=SUAVE):
    return _diferencia(_canon_suave(grafo), f"{PRE}w{W_SUAVE}.s1.guia", _canon_suave_control(control), "89")


def _paridad_render(grafo, w=W_SUAVE, control=RENDER):
    return _diferencia(_canon_render(grafo, w), f"{PRE}w{w}.s2.render", _canon_render_control(control), "89")


def _cambiado(ident, nombre, valor, grafo=GRAFO):
    """Copia del grafo con UNA entrada cambiada."""
    g = copy.deepcopy(grafo)
    antes = g[ident]["inputs"][nombre]
    assert antes != valor or type(antes) is not type(valor), (ident, nombre, valor)
    g[ident]["inputs"][nombre] = valor
    return g


def _otro(v):
    """Un valor distinto del mismo tipo."""
    if isinstance(v, bool):
        return not v
    if isinstance(v, int):
        return v + 8
    if isinstance(v, float):
        return v + 0.25
    return v + "x"


def _reordenar(grafo, cadena, orden):
    """La misma cadena de guias en otro orden: cada una cuelga de la anterior y lo demas, de la que queda la ultima."""
    g = copy.deepcopy(grafo)
    assert sorted(cadena) == sorted(orden) and cadena != orden
    anterior = {k: g[cadena[0]]["inputs"][k] for k in ("positive", "negative", "latent")}
    for ident in orden:
        g[ident]["inputs"].update(copy.deepcopy(anterior))
        anterior = {"positive": [ident, 0], "negative": [ident, 1], "latent": [ident, 2]}
    for ident, n in g.items():
        if ident not in cadena:
            for nombre, v in n["inputs"].items():
                if _enlace(v) and v[0] == cadena[-1]:
                    n["inputs"][nombre] = [orden[-1], v[1]]
    return g


# ---- paridad con los grafos validados ----
def test_paridad_etapa1_con_el_grafo_validado():
    v = PLAN_V8["ventanas"][W_SUAVE]
    assert (v["p0"], v["p1"]) == (13, 20) and PLAN_V8["lentos"][13:20] == [16, 32, 32, 16, 16, 16, 16]
    assert v["pos"] == [0, 16, 48, 80, 96, 112, 128, 144] and v["n"] == 145
    mio, control = _canon_suave(GRAFO), _canon_suave_control()
    dif = _diferencia(mio, f"{PRE}w2.s1.guia", control, "89")
    assert dif is None, dif
    assert not mio.sueltos and not control.sueltos, (mio.sueltos, control.sueltos)
    # no solo el arbol: los mismos nodos y compartidos igual (un solo KSamplerSelect para los dos pases, etc.)
    debajo_mio, debajo_control = mio.debajo(f"{PRE}w2.s1.guia"), control.debajo("89")
    assert Counter(debajo_mio.values()) == Counter(debajo_control.values())
    assert len(debajo_mio) == len(debajo_control) == 44
    # y en la etapa no sobra nada: todo alimenta al decodificado
    poses = {i for i, _ in mio.papeles if i in GRAFO}
    assert set(debajo_mio) | {i for i in poses if i.startswith(f"{PRE}w2.s1.")} == {i for i in GRAFO if i.startswith(f"{PRE}w2.s1.")}
    assert sorted(p for (i, _), p in mio.papeles.items() if i.startswith(f"{PRE}w2.s1.")) == sorted(f"pose {k}" for k in range(13, 21))


def test_paridad_etapa1_falla_con_cualquier_cambio():
    assert _paridad_suave(GRAFO) is None
    s1 = f"{PRE}w2.s1."
    cambios = [
        (s1 + "K15", "frame_idx", 40), (s1 + "k14", "frame_idx", 24), (s1 + "k16", "strength", 0.9),
        (s1 + "e13", "crop", "disabled"), (s1 + "E20", "upscale_method", "bilinear"), (s1 + "E13", "width", 960),
        (s1 + "latente", "width", 1920), (s1 + "latente", "length", 137),
        (s1 + "guider1", "model", ENLACES["modelo_render"]), (s1 + "guider2", "cfg", 1.5),
        (s1 + "sigmas1", "sigmas", L.SIGMAS_3), (s1 + "sigmas2", "sigmas", "0.909375, 0.725, 0.0"),
        (s1 + "sampler", "sampler_name", "euler"),
        (s1 + "muestreo2", "sigmas", [s1 + "sigmas1", 0]), (s1 + "muestreo1", "noise", ["otra_semilla", 0]),
        (s1 + "k13", "positive", ENLACES["positivo_render"]),  # el texto de la otra etapa
        (s1 + "K13", "positive", [s1 + "k20", 0]),  # el segundo pase NO parte del conditioning limpio
        (s1 + "K13", "latent", [s1 + "recorte1", 2]),  # sin subir el latente
        (s1 + "K14", "image", [s1 + "e14", 0]),  # la pose a mitad de tamano en el segundo pase
        (s1 + "k14", "image", [s1 + "e15", 0]),  # otra pose en ese fotograma
        (s1 + "i14", "batch_index", 15), (s1 + "i14", "length", 2), (s1 + "i14", "image", ENLACES["arte"]),
        (s1 + "recorte1", "positive", ENLACES["positivo_suave"]),  # recorte con el conditioning sin guias
        (s1 + "recorte2", "latent", [s1 + "muestreo2", 1]),  # la otra salida del muestreador
        (s1 + "subir", "samples", [s1 + "muestreo1", 0]),  # sin recortar las guias
        (s1 + "subir", "upscale_model", ENLACES["vae"]),
        (s1 + "guia", "latents", [s1 + "recorte1", 2]),  # decodificar el pase a mitad
        (s1 + "guia", "horizontal_tiles", 1), (s1 + "guia", "overlap", 1), (s1 + "guia", "last_frame_fix", True),
    ]
    for ident, nombre, valor in cambios:
        assert _paridad_suave(_cambiado(ident, nombre, valor)) is not None, (ident, nombre, valor)
    # el mismo conjunto de guias en otro orden tampoco pasa
    cadena = [s1 + f"K{k}" for k in range(13, 21)]
    assert _paridad_suave(_reordenar(GRAFO, cadena, cadena[:1] + [cadena[2], cadena[1]] + cadena[3:])) is not None
    assert _paridad_suave(_reordenar(GRAFO, cadena, cadena[:6] + [cadena[7], cadena[6]])) is not None
    # y del otro lado igual: un cambio en el grafo de control se ve
    assert _paridad_suave(GRAFO, _cambiado("K15", "frame_idx", 56, SUAVE)) is not None
    assert _paridad_suave(GRAFO, _cambiado("88", "latent", ["87", 1], SUAVE)) is not None


def test_paridad_etapa2_con_el_grafo_validado():
    control = _canon_render_control()
    for w, v in enumerate(PLAN_V8["ventanas"]):
        mio = _canon_render(GRAFO, w)
        dif = _diferencia(mio, f"{PRE}w{w}.s2.render", control, "89")
        assert dif is None, (w, dif)
        assert not mio.sueltos and not control.sueltos, (mio.sueltos, control.sueltos)
        debajo_mio, debajo_control = mio.debajo(f"{PRE}w{w}.s2.render"), control.debajo("89")
        assert Counter(debajo_mio.values()) == Counter(debajo_control.values())
        assert len(debajo_mio) == len(debajo_control) == 10
        assert set(debajo_mio) == {i for i in GRAFO if i.startswith(f"{PRE}w{w}.s2.")}
        # la diferencia admitida, a la vista: el latente mide ancho x alto x n en vez de medir la guia
        assert GRAFO[f"{PRE}w{w}.s2.latente"]["inputs"] == {"width": A, "height": H, "length": v["n"], "batch_size": 1}


def test_paridad_etapa2_falla_con_cualquier_cambio():
    assert _paridad_render(GRAFO) is None
    s2, s1 = f"{PRE}w2.s2.", f"{PRE}w2.s1."
    cambios = [
        (s2 + "g_guia", "crop", "center"), (s2 + "g_guia", "frame_idx", 8), (s2 + "g_guia", "strength", 0.5),
        (s2 + "g_guia", "latent_downscale_factor", 1.0), (s2 + "g_guia", "use_streaming", True),
        (s2 + "g_guia", "tile_frames", 0), (s2 + "g_guia", "image", [f"{PRE}w1.s1.guia", 0]),  # la guia de otra ventana
        (s2 + "g_guia", "positive", ENLACES["positivo_suave"]), (s2 + "g_guia", "latent", [s1 + "latente", 0]),
        (s2 + "g_arte", "frame_idx", 0), (s2 + "g_arte", "crop", "disabled"), (s2 + "g_arte", "tile_size", 512),
        (s2 + "g_arte", "tile_overlap", 32), (s2 + "g_arte", "image", ENLACES["arte"]),  # el ancla de otra ventana
        (s2 + "g_pose", "frame_idx", -1), (s2 + "g_pose", "use_tiled_encode", True),
        (s2 + "g_pose", "image", [s1 + "guia", 0]),
        (s2 + "guider", "model", ENLACES["modelo_suave"]), (s2 + "guider", "positive", ENLACES["positivo_render"]),
        (s2 + "guider", "cfg", 3.0), (s2 + "sigmas", "sigmas", L.SIGMAS_3), (s2 + "sampler", "sampler_name", "euler"),
        (s2 + "latente", "width", 960), (s2 + "latente", "height", 544), (s2 + "latente", "length", 137),
        (s2 + "muestreo", "latent_image", [s2 + "latente", 0]),  # muestrear sin las guias en el latente
        (s2 + "muestreo", "noise", ["otra_semilla", 0]),
        (s2 + "recorte", "positive", ENLACES["positivo_render"]),  # recorte con el conditioning sin guias
        (s2 + "recorte", "latent", [s2 + "muestreo", 1]),
        (s2 + "render", "latents", [s2 + "muestreo", 0]),  # decodificar sin recortar las guias
        (s2 + "render", "vertical_tiles", 1), (s2 + "render", "working_dtype", "float16"),
    ]
    for ident, nombre, valor in cambios:
        assert _paridad_render(_cambiado(ident, nombre, valor)) is not None, (ident, nombre, valor)
    cadena = [s2 + "g_guia", s2 + "g_arte", s2 + "g_pose"]
    assert _paridad_render(_reordenar(GRAFO, cadena, [cadena[0], cadena[2], cadena[1]])) is not None  # el ancla fijada antes que el arte
    assert _paridad_render(_reordenar(GRAFO, cadena, [cadena[1], cadena[2], cadena[0]])) is not None  # la guia al final
    assert _paridad_render(GRAFO, control=_cambiado("72", "frame_idx", 0, RENDER)) is not None
    assert _paridad_render(GRAFO, control=_cambiado("71", "crop", "center", RENDER)) is not None


def test_la_canonizacion_ve_cada_entrada_de_cada_nodo():
    """Barrido: se cambia UNA entrada cada vez (el valor, el indice del enlace o su origen) en todos los nodos de las dos
    etapas de la ventana 2, y la paridad de esa etapa tiene que fallar siempre."""
    probados = 0
    for etapa, paridad in (("s1.", _paridad_suave), ("s2.", _paridad_render)):
        for ident in [i for i in GRAFO if i.startswith(f"{PRE}w2.{etapa}")]:
            for nombre, v in GRAFO[ident]["inputs"].items():
                variantes = [[v[0], v[1] + 1], ["desconocido", v[1]]] if _enlace(v) else [_otro(v)]
                for valor in variantes:
                    assert paridad(_cambiado(ident, nombre, valor)) is not None, (ident, nombre, valor)
                    probados += 1
            sin_una = copy.deepcopy(GRAFO)
            sin_una[ident]["inputs"].popitem()
            assert paridad(sin_una) is not None, ident
    assert probados > 400, probados


# ---- encadenado ----
def test_encadenado_ancla_y_acumulado():
    ventanas = PLAN_V8["ventanas"]
    assert len(ventanas) == 4
    texto = json.dumps(PLAN_V8)
    for w, v in enumerate(ventanas):
        pre = f"{PRE}w{w}."
        if w == 0:
            assert pre + "ancla" not in GRAFO
            ancla = ENLACES["arte"]
        else:
            anterior = f"{PRE}w{w - 1}."
            assert GRAFO[pre + "ancla"] == {"class_type": "ImageFromBatch", "inputs": {
                "image": [anterior + "s2.render", 0], "batch_index": ventanas[w - 1]["n"] - 1, "length": 1}}
            ancla = [pre + "ancla", 0]
        assert GRAFO[pre + "s2.g_arte"]["inputs"]["image"] == ancla and GRAFO[pre + "s2.g_arte"]["inputs"]["frame_idx"] == -1
        assert GRAFO[pre + "s2.g_pose"]["inputs"]["image"] == ancla and GRAFO[pre + "s2.g_pose"]["inputs"]["frame_idx"] == 0
        assert GRAFO[pre + "s2.g_guia"]["inputs"]["image"] == [pre + "s1.guia", 0]
        assert GRAFO[pre + "s1.guia"]["class_type"] == GRAFO[pre + "s2.render"]["class_type"] == "LTXVTiledVAEDecode"
        elegir = GRAFO[pre + "elegir"]
        esperado = {"guia": [pre + "s1.guia", 0], "render": [pre + "s2.render", 0], "plan": texto, "ventana": w}
        if w:
            esperado.update(anterior_render=[f"{PRE}w{w - 1}.elegir", 0], anterior_guia=[f"{PRE}w{w - 1}.elegir", 1])
        assert elegir == {"class_type": "CP_LentaElegir", "inputs": esperado}
        assert json.loads(elegir["inputs"]["plan"]) == PLAN_V8
    assert "anterior_render" not in GRAFO[f"{PRE}w0.elegir"]["inputs"] and "anterior_guia" not in GRAFO[f"{PRE}w0.elegir"]["inputs"]
    assert ANIMACION == [f"{PRE}w3.elegir", 0] and PLAYBLAST == [f"{PRE}w3.elegir", 1]


def test_una_sola_ventana_no_lleva_ancla_ni_acumulado():
    plan = _plan([(0, 2)], [16, 32])
    g, animacion, playblast = L.grafo_lento(plan, ENLACES, A, H, "")
    assert len(g) == 23 + 5 * 3 and "w0.ancla" not in g and "w1.elegir" not in g
    assert list(g["w0.elegir"]["inputs"]) == ["guia", "render", "plan", "ventana"]
    assert g["w0.s2.g_arte"]["inputs"]["image"] == g["w0.s2.g_pose"]["inputs"]["image"] == ENLACES["arte"]
    assert (animacion, playblast) == (["w0.elegir", 0], ["w0.elegir", 1])


# ---- cableado de cada etapa ----
def _enlaces_de_fuera(nodos):
    nombres = {tuple(v): k for k, v in ENLACES.items()}
    return {nombres[tuple(v)] for n in nodos for v in n["inputs"].values() if _enlace(v) and tuple(v) in nombres}


def test_cada_etapa_usa_su_modelo_y_sus_textos():
    e = ENLACES
    for w, v in enumerate(PLAN_V8["ventanas"]):
        s1, s2 = f"{PRE}w{w}.s1.", f"{PRE}w{w}.s2."
        p0, p1 = v["p0"], v["p1"]
        etapa1 = [n for i, n in GRAFO.items() if i.startswith(s1)]
        etapa2 = [n for i, n in GRAFO.items() if i.startswith(s2)]
        assert _enlaces_de_fuera(etapa1) == {"ruido", "poses", "modelo_suave", "vae", "escalador", "positivo_suave", "negativo_suave"}
        assert _enlaces_de_fuera(etapa2) == {"ruido", "modelo_render", "factor_guia", "vae", "positivo_render",
                                             "negativo_render"} | ({"arte"} if w == 0 else set())
        # etapa 1: los dos pases con el modelo suave y el mismo ruido
        assert GRAFO[s1 + "guider1"]["inputs"]["model"] == GRAFO[s1 + "guider2"]["inputs"]["model"] == e["modelo_suave"]
        assert GRAFO[s1 + "muestreo1"]["inputs"]["noise"] == GRAFO[s1 + "muestreo2"]["inputs"]["noise"] == e["ruido"]
        primera = GRAFO[s1 + f"k{p0}"]["inputs"]
        assert (primera["positive"], primera["negative"], primera["latent"]) == (e["positivo_suave"], e["negativo_suave"], [s1 + "latente", 0])
        # el segundo pase parte del conditioning LIMPIO y del latente subido
        assert GRAFO[s1 + "subir"]["inputs"] == {"samples": [s1 + "recorte1", 2], "upscale_model": e["escalador"], "vae": e["vae"]}
        segunda = GRAFO[s1 + f"K{p0}"]["inputs"]
        assert (segunda["positive"], segunda["negative"], segunda["latent"]) == (e["positivo_suave"], e["negativo_suave"], [s1 + "subir", 0])
        # LTXVCropGuides recibe el conditioning CON las guias (el de la ultima) y la salida 0 del muestreador
        for recorte, ultima, muestreo in ((s1 + "recorte1", s1 + f"k{p1}", s1 + "muestreo1"), (s1 + "recorte2", s1 + f"K{p1}", s1 + "muestreo2"),
                                          (s2 + "recorte", s2 + "g_pose", s2 + "muestreo")):
            assert GRAFO[recorte] == {"class_type": "LTXVCropGuides", "inputs": {"positive": [ultima, 0], "negative": [ultima, 1], "latent": [muestreo, 0]}}
            guider = GRAFO[GRAFO[muestreo]["inputs"]["guider"][0]]["inputs"]
            assert (guider["positive"], guider["negative"], guider["cfg"]) == ([ultima, 0], [ultima, 1], 1.0)
            assert GRAFO[muestreo]["inputs"]["latent_image"] == [ultima, 2]
        assert GRAFO[s1 + "guia"]["inputs"]["latents"] == [s1 + "recorte2", 2] and GRAFO[s2 + "render"]["inputs"]["latents"] == [s2 + "recorte", 2]
        # etapa 2: modelo de render, sus textos y el factor de la IC-LoRA en las tres guias
        assert GRAFO[s2 + "guider"]["inputs"]["model"] == e["modelo_render"] and GRAFO[s2 + "muestreo"]["inputs"]["noise"] == e["ruido"]
        g_guia = GRAFO[s2 + "g_guia"]["inputs"]
        assert (g_guia["positive"], g_guia["negative"], g_guia["latent"]) == (e["positivo_render"], e["negativo_render"], [s2 + "latente", 0])
        assert GRAFO[s2 + "g_arte"]["inputs"]["latent"] == [s2 + "g_guia", 2] and GRAFO[s2 + "g_pose"]["inputs"]["latent"] == [s2 + "g_arte", 2]
        for g in ("g_guia", "g_arte", "g_pose"):
            guia = GRAFO[s2 + g]["inputs"]
            assert guia["latent_downscale_factor"] == e["factor_guia"]
            assert guia["use_streaming"] is False and guia["use_tiled_encode"] is False
    # fuerza 1.0 exacta en todas las guias de las dos etapas
    guias = [n for n in GRAFO.values() if n["class_type"] in ("LTXVAddGuide", "LTXAddVideoICLoRAGuide")]
    assert len(guias) == 56 + 12 and all(type(n["inputs"]["strength"]) is float and n["inputs"]["strength"] == 1.0 for n in guias)
    assert {n["inputs"]["sigmas"] for n in GRAFO.values() if n["class_type"] == "ManualSigmas"} == {L.SIGMAS_8, L.SIGMAS_3}


def test_primer_pase_de_la_etapa1_a_mitad_de_tamano():
    for ancho, alto in ((1920, 1088), (1280, 704)):
        g, _, _ = L.grafo_lento(PLAN_V8, ENLACES, ancho, alto, PRE)
        for w, v in enumerate(PLAN_V8["ventanas"]):
            s1, s2 = f"{PRE}w{w}.s1.", f"{PRE}w{w}.s2."
            assert g[s1 + "latente"]["inputs"] == {"width": ancho // 2, "height": alto // 2, "length": v["n"], "batch_size": 1}
            assert g[s2 + "latente"]["inputs"] == {"width": ancho, "height": alto, "length": v["n"], "batch_size": 1}
            for k in range(v["p0"], v["p1"] + 1):
                escala = {"image": [s1 + f"i{k}", 0], "upscale_method": "lanczos", "crop": "center"}
                assert g[s1 + f"e{k}"]["inputs"] == dict(escala, width=ancho // 2, height=alto // 2)
                assert g[s1 + f"E{k}"]["inputs"] == dict(escala, width=ancho, height=alto)
                assert g[s1 + f"k{k}"]["inputs"]["image"] == [s1 + f"e{k}", 0] and g[s1 + f"K{k}"]["inputs"]["image"] == [s1 + f"E{k}", 0]


# ---- lo que el motor no valida en una expansion ----
def test_sin_guardados_enlaces_en_su_sitio_e_ids_con_prefijo():
    assert {n["class_type"] for n in GRAFO.values()} == set(ENTRADAS)
    assert not any(p in n["class_type"] for n in GRAFO.values() for p in ("Save", "Preview", "CreateVideo", "Load"))
    de_fuera = {tuple(v): TIPO_ENLACE[k] for k, v in ENLACES.items()}
    for ident, n in GRAFO.items():
        assert ident.startswith(PRE) and set(n) == {"class_type", "inputs"}, ident
        esquema = ENTRADAS[n["class_type"]]
        faltan = set(esquema) - set(n["inputs"]) - OPCIONALES.get(n["class_type"], set())
        assert not faltan and set(n["inputs"]) <= set(esquema), (ident, faltan, set(n["inputs"]) - set(esquema))
        for nombre, v in n["inputs"].items():
            tipo = esquema[nombre]
            if isinstance(v, list):
                assert _enlace(v), (ident, nombre, v)
                if v[0] in GRAFO:
                    salidas = SALIDAS[GRAFO[v[0]]["class_type"]]
                    assert v[1] < len(salidas) and salidas[v[1]] == tipo, (ident, nombre, v)
                else:
                    assert de_fuera.get(tuple(v)) == tipo, (ident, nombre, v)
            elif isinstance(tipo, list):
                assert v in tipo, (ident, nombre, v)
            else:
                # tipo exacto: aqui nadie convierte un 1 en 1.0 ni un 960.0 en 960
                assert type(v) is PYTHON[tipo], (ident, nombre, v)
    # cada entrada con su propia lista y sin tocar lo que se le paso
    listas = [id(v) for n in GRAFO.values() for v in n["inputs"].values() if isinstance(v, list)]
    assert len(listas) == len(set(listas))
    assert not any(v is e for n in GRAFO.values() for v in n["inputs"].values() for e in ENLACES.values())
    plan, enlaces = copy.deepcopy(PLAN_V8), copy.deepcopy(ENLACES)
    L.grafo_lento(plan, enlaces, A, H, PRE)
    assert plan == PLAN_V8 and enlaces == ENLACES


def test_ids_son_prefijo_mas_nombre():
    otro, animacion, playblast = L.grafo_lento(PLAN_V8, ENLACES, A, H, "7.3.1.")
    assert len(otro) == len(GRAFO) and animacion == ["7.3.1.w3.elegir", 0] and playblast == ["7.3.1.w3.elegir", 1]
    assert json.loads(json.dumps(otro).replace('"7.3.1.', '"' + PRE)) == GRAFO
    nombres = sorted(i[len(PRE):] for i in GRAFO if i.startswith(f"{PRE}w2.") and not re.search(r"\.[iekEK]\d+$", i))
    assert nombres == sorted([
        "w2.ancla", "w2.elegir", "w2.s1.latente", "w2.s1.sampler", "w2.s1.sigmas1", "w2.s1.sigmas2", "w2.s1.guider1",
        "w2.s1.muestreo1", "w2.s1.recorte1", "w2.s1.subir", "w2.s1.guider2", "w2.s1.muestreo2", "w2.s1.recorte2",
        "w2.s1.guia", "w2.s2.latente", "w2.s2.sampler", "w2.s2.sigmas", "w2.s2.g_guia", "w2.s2.g_arte", "w2.s2.g_pose",
        "w2.s2.guider", "w2.s2.muestreo", "w2.s2.recorte", "w2.s2.render"]), nombres
    assert all(f"{PRE}w2.s1.{letra}{k}" in GRAFO for letra in "iekEK" for k in range(13, 21))


def test_plan_de_cuatro_ventanas_nodos_y_fotogramas():
    ventanas = PLAN_V8["ventanas"]
    assert [(v["p0"], v["p1"], v["n"]) for v in ventanas] == [(0, 6, 177), (6, 13, 161), (13, 20, 145), (20, 24, 65)]
    poses = [v["p1"] - v["p0"] + 1 for v in ventanas]  # 7, 8, 8 y 5: la pose frontera entra en las dos ventanas
    # por ventana: etapa 1 = 12 + 5 por pose; etapa 2 = 10; CP_LentaElegir; y el ancla desde la segunda
    assert len(GRAFO) == sum(12 + 5 * p + 10 + 1 for p in poses) + (len(ventanas) - 1) == 235
    assert Counter(n["class_type"] for n in GRAFO.values()) == {
        "ImageFromBatch": 28 + 3, "ImageScale": 56, "LTXVAddGuide": 56, "EmptyLTXVLatentVideo": 8, "CFGGuider": 12,
        "KSamplerSelect": 8, "ManualSigmas": 12, "SamplerCustomAdvanced": 12, "LTXVCropGuides": 12,
        "LTXVLatentUpsampler": 4, "LTXVTiledVAEDecode": 8, "LTXAddVideoICLoRAGuide": 12, "CP_LentaElegir": 4}
    for w, v in enumerate(ventanas):
        s1 = f"{PRE}w{w}.s1."
        for j, k in enumerate(range(v["p0"], v["p1"] + 1)):
            assert GRAFO[s1 + f"i{k}"]["inputs"] == {"image": ENLACES["poses"], "batch_index": k, "length": 1}
            assert GRAFO[s1 + f"k{k}"]["inputs"]["frame_idx"] == GRAFO[s1 + f"K{k}"]["inputs"]["frame_idx"] == v["pos"][j]
            assert v["pos"][j] % 8 == 0 and v["pos"][j] <= v["n"] - 1
        # las cadenas van en el orden de las poses
        for a, b in zip(range(v["p0"], v["p1"]), range(v["p0"] + 1, v["p1"] + 1)):
            for letra in "kK":
                sig = GRAFO[s1 + f"{letra}{b}"]["inputs"]
                assert (sig["positive"], sig["negative"], sig["latent"]) == ([s1 + f"{letra}{a}", 0], [s1 + f"{letra}{a}", 1], [s1 + f"{letra}{a}", 2])


# ---- orden de ejecucion ----
def _orden_del_motor(grafo, salidas):
    """El orden en que ComfyUI 0.42.1 ejecuta una expansion. Port de comfy_execution/graph.py: TopologicalSort.add_node y
    add_strong_link dan de alta desde las salidas del nodo expandido, con una pila (las entradas de cada nodo, de la
    ultima a la primera), y ExecutionList elige el primer nodo listo por orden de alta (en la expansion no hay nodos
    de salida ni asincronos, que irian antes). Da el mismo orden que el planificador real del motor con este grafo."""
    pendientes, cuenta, bloquea = {}, {}, {}

    def alta(ident):
        pila, enlaces = [ident], []
        while pila:
            u = pila.pop()
            if u in pendientes:
                continue
            pendientes[u], cuenta[u], bloquea[u] = True, 0, {}
            for v in grafo[u]["inputs"].values():
                if _enlace(v):
                    if v[0] in grafo:  # lo de fuera ya esta calculado cuando el nodo se expande
                        pila.append(v[0])
                    enlaces.append((v[0], u))
        for de, a in enlaces:
            fuerte(de, a)

    def fuerte(de, a):
        if de in grafo:
            alta(de)
            if a not in bloquea[de]:
                bloquea[de][a] = True
                if a in cuenta:
                    cuenta[a] += 1

    for de, _ in salidas:
        fuerte(de, "<el nodo expandido>")
    orden = []
    while pendientes:
        listo = next(i for i in pendientes if cuenta[i] == 0)
        del pendientes[listo]
        for a in bloquea.pop(listo):
            if a in cuenta:
                cuenta[a] -= 1
        orden.append(listo)
    return orden


def _pesados(grafo, orden):
    return [i[len(PRE):] for i in orden if i.endswith(".ancla")
            or grafo[i]["class_type"] in ("SamplerCustomAdvanced", "LTXVTiledVAEDecode", "CP_LentaElegir")]


def test_el_motor_acaba_cada_ventana_antes_de_empezar_la_siguiente():
    """Cada lote de imagenes de una ventana pesa gigas y sigue en memoria mientras le quede un consumidor: el playblast
    lento, hasta su render y su CP_LentaElegir; el render, hasta su CP_LentaElegir y el ancla de la ventana siguiente."""
    orden = _orden_del_motor(GRAFO, [ANIMACION, PLAYBLAST])
    assert sorted(orden) == sorted(GRAFO)
    ventanas = [int(i[len(PRE) + 1:].split(".")[0]) for i in orden]
    assert ventanas == sorted(ventanas), "se mezclan nodos de ventanas distintas"
    esperado = []
    for w in range(4):
        esperado += ([f"w{w}.ancla"] if w else []) + [f"w{w}.s1.muestreo1", f"w{w}.s1.muestreo2", f"w{w}.s1.guia",
                                                      f"w{w}.s2.muestreo", f"w{w}.s2.render", f"w{w}.elegir"]
    assert _pesados(GRAFO, orden) == esperado
    # no es casualidad: con lo acumulado por delante del render, el motor deja todos los CP_LentaElegir para el final
    mal = copy.deepcopy(GRAFO)
    for n in mal.values():
        if n["class_type"] == "CP_LentaElegir":
            al_reves = ("anterior_render", "anterior_guia", "guia", "plan", "ventana", "render")
            n["inputs"] = {k: n["inputs"][k] for k in al_reves if k in n["inputs"]}
    assert _pesados(mal, _orden_del_motor(mal, [ANIMACION, PLAYBLAST]))[-4:] == [f"w{w}.elegir" for w in range(4)]


# ---- validacion ----
def _falla(texto, f, *a):
    try:
        f(*a)
    except ValueError as e:
        assert str(e).startswith("[CP_BlockingLento] ") and texto in str(e), str(e)
    else:
        raise AssertionError(f"tenia que fallar: {texto}")


def test_validacion_del_plan_el_tamano_y_los_enlaces():
    _falla("no se puede leer", L.leer_plan, "esto no es un plan")
    _falla("no se puede leer", L.leer_plan, "[1, 2]")
    _falla("no trae ventanas", L.grafo_lento, {"version": 1}, ENLACES, A, H, PRE)
    _falla("no trae ventanas", L.grafo_lento, {"ventanas": []}, ENLACES, A, H, PRE)
    _falla("multiplos de 64", L.grafo_lento, PLAN_V8, ENLACES, 1900, H, PRE)
    _falla("multiplos de 64", L.grafo_lento, PLAN_V8, ENLACES, A, 1080, PRE)
    _falla("multiplos de 64", L.grafo_lento, PLAN_V8, ENLACES, 64, 64, PRE)
    _falla("multiplos de 64", L.grafo_lento, PLAN_V8, ENLACES, 1920.0, H, PRE)

    def con(w, **cambios):
        plan = copy.deepcopy(PLAN_V8)
        plan["ventanas"][w].update(cambios)
        return plan

    _falla("multiplos de 8", L.grafo_lento, con(1, pos=[0, 16, 44, 64, 80, 112, 144, 160]), ENLACES, A, H, PRE)
    _falla("8k+1", L.grafo_lento, con(3, n=64), ENLACES, A, H, PRE)
    _falla("8k+1", L.grafo_lento, con(3, n=73), ENLACES, A, H, PRE)
    _falla("le falta", L.grafo_lento, {"ventanas": [{"p0": 0, "p1": 1, "pos": [0, 16]}]}, ENLACES, A, H, PRE)
    _falla("enteros", L.grafo_lento, con(0, n=177.0), ENLACES, A, H, PRE)
    _falla("posiciones", L.grafo_lento, con(0, pos=[0, 32, 64, 96, 112, 144]), ENLACES, A, H, PRE)
    _falla("empezar en 0 y crecer", L.grafo_lento, con(3, pos=[0, 16, 16, 48, 64]), ENLACES, A, H, PRE)
    _falla("pose posterior", L.grafo_lento, con(0, p1=0), ENLACES, A, H, PRE)
    _falla("comparte su primera pose", L.grafo_lento, con(2, p0=14, pos=[0, 32, 64, 80, 96, 112, 128], n=129), ENLACES, A, H, PRE)
    _falla("comparte su primera pose", L.grafo_lento, _plan([(1, 3)], [16, 16, 16]), ENLACES, A, H, PRE)
    _falla("no salen de los lentos", L.grafo_lento, con(3, pos=[0, 16, 32, 48, 72], n=73), ENLACES, A, H, PRE)
    _falla("no cubre el plano entero", L.grafo_lento, _plan([(0, 2)], [16, 16, 16]), ENLACES, A, H, PRE)
    _falla("el tope es", L.grafo_lento, _plan([(k, k + 1) for k in range(L.MAX_VENTANAS + 1)], [16] * (L.MAX_VENTANAS + 1)), ENLACES, A, H, PRE)
    for nombre in ("vae", "poses", "ruido"):
        _falla(f"'{nombre}' tiene que venir conectada", L.grafo_lento, PLAN_V8, {k: v for k, v in ENLACES.items() if k != nombre}, A, H, PRE)
        _falla(f"'{nombre}' tiene que venir conectada", L.grafo_lento, PLAN_V8, dict(ENLACES, **{nombre: None}), A, H, PRE)
    # el factor de la guia puede llegar escrito a mano: entra como numero en las tres guias de cada ventana
    g, _, _ = L.grafo_lento(PLAN_V8, dict(ENLACES, factor_guia=2), A, H, PRE)
    factores = [n["inputs"]["latent_downscale_factor"] for n in g.values() if n["class_type"] == "LTXAddVideoICLoRAGuide"]
    assert len(factores) == 12 and all(type(f) is float and f == 2.0 for f in factores)
    # un plan minimo, sin inicios ni lentos, vale: lo demas es cosa de CP_LentaElegir
    g, _, _ = L.grafo_lento({"ventanas": [{"p0": 0, "p1": 1, "pos": [0, 16], "n": 17}]}, ENLACES, A, H, PRE)
    assert len(g) == 23 + 5 * 2


# ---- el nodo ----
def test_nodo_se_declara_expandible_y_con_enlaces_crudos():
    esquema = L.BlockingLento.define_schema()
    assert esquema["node_id"] == "CP_BlockingLento" and esquema["enable_expand"] is True
    assert esquema["category"] == "character-pipeline/poses"
    entradas = {p.id: p for p in esquema["inputs"]}
    assert list(entradas) == ["plan", "ancho", "alto", *L.ENLACES] and set(L.ENLACES) == set(ENLACES)
    assert {k: p.tipo for k, p in entradas.items()} == {
        "plan": "String", "ancho": "Int", "alto": "Int", "ruido": "Noise", "poses": "Image", "arte": "Image",
        "modelo_suave": "Model", "modelo_render": "Model", "factor_guia": "Float", "vae": "Vae",
        "escalador": "LatentUpscaleModel", "positivo_suave": "Conditioning", "negativo_suave": "Conditioning",
        "positivo_render": "Conditioning", "negativo_render": "Conditioning"}
    for nombre, p in entradas.items():
        assert bool(p.k.get("raw_link")) == (nombre in L.ENLACES), nombre
    assert entradas["plan"].k.get("force_input") is True and entradas["factor_guia"].k.get("force_input") is True
    assert (entradas["ancho"].k["default"], entradas["alto"].k["default"]) == (1920, 1088)
    assert [(p.tipo, p.k.get("display_name")) for p in esquema["outputs"]] == [("Image", "animacion"), ("Image", "playblast")]


def test_nodo_devuelve_los_enlaces_y_el_subgrafo():
    GraphBuilder.set_default_prefix("12", 0, 0)  # lo hace el motor antes de llamar al nodo
    salida = L.BlockingLento.execute(json.dumps(PLAN_V8), A, H, **ENLACES)
    assert salida.args == (["12.0.0.w3.elegir", 0], ["12.0.0.w3.elegir", 1])
    assert salida.expand == GRAFO
    # las dos salidas son enlaces a nodos del subgrafo: el motor los resuelve cuando termina la expansion
    assert all(_enlace(a) and a[0] in salida.expand for a in salida.args)
    otra = L.BlockingLento.execute(json.dumps(PLAN_V8), A, H, **ENLACES)
    assert otra.args[0] == ["12.0.1.w3.elegir", 0] and not set(otra.expand) & set(salida.expand)
    _falla("no se puede leer", L.BlockingLento.execute, "{", A, H, *[ENLACES[k] for k in L.ENLACES])
    _falla("multiplos de 64", L.BlockingLento.execute, json.dumps(PLAN_V8), 1000, H, *[ENLACES[k] for k in L.ENLACES])
    _falla("'arte' tiene que venir conectada", L.BlockingLento.execute, json.dumps(PLAN_V8), A, H, *[None if k == "arte" else ENLACES[k] for k in L.ENLACES])
    # el nodo lee el plan con el mismo lector que CP_LentaElegir: lo que fallaria en la primera eleccion falla aqui
    sin_version = {k: v for k, v in PLAN_V8.items() if k != "version"}
    _falla("version", L.BlockingLento.execute, json.dumps(sin_version), A, H, *[ENLACES[k] for k in L.ENLACES])
    corto = dict(PLAN_V8, total=PLAN_V8["inicios"][-1])
    _falla("fotogramas", L.BlockingLento.execute, json.dumps(corto), A, H, *[ENLACES[k] for k in L.ENLACES])
    sin_transicion = {k: v for k, v in PLAN_V8.items() if k != "transicion_max"}
    _falla("le falta", L.BlockingLento.execute, json.dumps(sin_transicion), A, H, *[ENLACES[k] for k in L.ENLACES])


if __name__ == "__main__":
    pruebas = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in pruebas:
        t()
        print("ok", t.__name__)
    print(len(pruebas), "pruebas")
