# -*- coding: utf-8 -*-
"""Genera tests/datos/compat_v015.json: huellas de lo que hace la v0.15 (el commit bafe3eb) en camara lenta y fotos.

tests/test_lenta_compat_v015.py recalcula lo mismo con el codigo actual y lo compara, caso a caso. Cubre lo que la
v0.14 no tenia (y por eso test_lenta_compat_v014.py no mira):
- planes: planificar y el nodo CP_LentaPlan con ritmo 1 a 8 y mas_lentos True/False, sobre mapas al azar, mapas de
  fotos y casos de borde (errores incluidos: tope de salida con ritmo, max_total, una pose, ritmos que no valen);
- eleccion: elegir_indices con recorridos de muchas formas (empates, ceros, picos), indices_ventana y montar_ventana
  sobre planes de verdad;
- subgrafo de CP_BlockingLento con solo_animacion False y True (con grafo_lento y con el nodo);
- fotos: mapa_bloques, total_fotogramas y CP_PosesDesdeFotos (mapa, poses, transicion, fotogramas, fps).
De cada resultado se guarda una huella (los 16 primeros caracteres del sha256 de su JSON), o la del error con su
tipo y su texto: un cambio de un solo byte en un plan, unos indices, un subgrafo o un mensaje cambia la huella.

Para regenerarlo hace falta una copia de la v0.15 (el commit bafe3eb):

    mkdir -p /tmp/v015 && git archive bafe3eb | tar -x -C /tmp/v015
    python3 tools/generar_compat_v015.py /tmp/v015

Con un segundo argumento escribe en otro sitio; con --casos N --semilla S cambia cuantos mapas al azar y cuales (los
de por defecto son los del fichero del repo). No hace falta ComfyUI: si no esta, se usan dobles minimos de
comfy_api, folder_paths y comfy_execution. Los recorridos de la eleccion se rehacen desde su semilla solo con
random.Random(semilla).random(), que Python garantiza igual entre versiones; aun asi se guarda la huella de cada
recorrido para distinguir un cambio de entrada de un cambio de comportamiento.
"""

import argparse
import contextlib
import hashlib
import importlib
import importlib.util
import json
import math
import os
import random
import shutil
import sys
import tempfile
import types

AQUI = os.path.dirname(os.path.abspath(__file__))
CASOS = 150
SEMILLA = 15
RITMOS = range(1, 9)
VARIANTES = [(r, m) for r in RITMOS for m in (False, True)]
PREFIJO = "@"  # el prefijo de los ids del subgrafo
CORTA = 16  # caracteres de cada huella


# ---------------------------------------------------------------- dobles y carga

class _Salida:
    def __init__(self, *a, **k):
        self.args, self.k = a, k


def _doble_comfy_api():
    """comfy_api.latest.io lo justo para importar los modulos y llamar a execute sin ComfyUI."""
    try:
        import comfy_api.latest  # noqa: F401
        return
    except ImportError:
        pass

    class _Port:
        def __init__(self, *a, **k):
            self.a, self.k = a, k

    class _IO:
        Schema = staticmethod(lambda **k: k)
        NodeOutput = _Salida
        ComfyNode = type("ComfyNode", (), {})
        UploadType = types.SimpleNamespace(image="image_upload")

    for n in ("Image", "Float", "Int", "String", "Boolean", "Combo", "Mask"):
        setattr(_IO, n, type(n, (), {"Input": staticmethod(_Port), "Output": staticmethod(_Port)}))
    mod, latest = types.ModuleType("comfy_api"), types.ModuleType("comfy_api.latest")
    latest.io, mod.latest = _IO, latest
    sys.modules["comfy_api"], sys.modules["comfy_api.latest"] = mod, latest


@contextlib.contextmanager
def _modulos(**dobles):
    """Pone unos modulos en sys.modules mientras dura el bloque y deja despues los que habia."""
    antes = {n: sys.modules.get(n) for n in dobles}
    sys.modules.update(dobles)
    try:
        yield
    finally:
        for n, m in antes.items():
            if m is None:
                sys.modules.pop(n, None)
            else:
                sys.modules[n] = m


def _doble_graph_utils():
    gu = types.ModuleType("comfy_execution.graph_utils")
    gu.GraphBuilder = type("GraphBuilder", (), {"alloc_prefix": staticmethod(lambda: PREFIJO)})
    ce = types.ModuleType("comfy_execution")
    ce.graph_utils = gu
    return {"comfy_execution": ce, "comfy_execution.graph_utils": gu}


def _doble_folder_paths(carpeta):
    fp = types.ModuleType("folder_paths")

    def ruta(nombre):
        r = os.path.abspath(os.path.join(carpeta, nombre))
        if not r.startswith(carpeta + os.sep):
            raise ValueError(f"Invalid file path: {nombre!r}")
        return r

    def existe(nombre):
        try:
            return os.path.exists(ruta(nombre))
        except ValueError:
            return False

    fp.get_input_directory = lambda: carpeta
    fp.get_annotated_filepath = ruta
    fp.exists_annotated_filepath = existe
    fp.filter_files_content_types = lambda archivos, tipos: [f for f in archivos if f.lower().endswith(".png")]
    return {"folder_paths": fp}


def cargar_pack(carpeta_pack, nombre):
    """(lenta, lenta_grafo, poses_fotos) de un pack (la carpeta que tiene comfy_character_pipeline/), cargado como
    paquete con ese nombre: asi dos versiones conviven en el mismo proceso sin pisarse."""
    _doble_comfy_api()
    carpeta = os.path.join(carpeta_pack, "comfy_character_pipeline")
    spec = importlib.util.spec_from_file_location(nombre, os.path.join(carpeta, "__init__.py"),
                                                  submodule_search_locations=[carpeta])
    m = importlib.util.module_from_spec(spec)
    sys.modules[nombre] = m
    spec.loader.exec_module(m)
    return tuple(importlib.import_module(f"{nombre}.{x}") for x in ("lenta", "lenta_grafo", "poses_fotos"))


# ---------------------------------------------------------------- huellas

def huella(texto):
    return hashlib.sha256(texto.encode("utf-8")).hexdigest()[:CORTA]


def huella_de(valor):
    """Huella del JSON de un valor (claves ordenadas)."""
    return huella(json.dumps(valor, sort_keys=True, ensure_ascii=True))


def huella_error(e):
    return huella("ERROR " + type(e).__name__ + ": " + str(e))


def _intentar(funcion):
    """La huella de lo que devuelve funcion() (ya convertido a algo JSON) o la de su error, con tipo y texto."""
    try:
        return huella_de(funcion())
    except Exception as e:  # noqa: BLE001 -- el tipo del error es parte de lo que se compara
        return huella_error(e)


def huella_grafo(grafo):
    limpio = {i: {"class_type": d["class_type"], "inputs": d["inputs"]} for i, d in grafo.items()}
    return huella(json.dumps(limpio, sort_keys=True))


def _args(salida):
    return tuple(getattr(salida, "args", None) or tuple(getattr(salida, "result", ())))


def _expand(salida):
    e = getattr(salida, "expand", None)
    return e if e is not None else getattr(salida, "k", {}).get("expand")


# ---------------------------------------------------------------- planes

def _saltos(caso):
    return [float(x) for x in caso["saltos"]]


def mapa_de(caso):
    """El mapa {"bloques"} que da esos inicios y ese total (None si no se puede: el nodo no lo leeria)."""
    ini, total = caso["inicios"], caso["total"]
    if len(ini) < 1 or ini[0] != 0 or any(b <= a for a, b in zip(ini, ini[1:])) or total <= ini[-1]:
        return None
    bloques = [[a, b - 1] for a, b in zip(ini, ini[1:])] + [[ini[-1], total - 1]]
    return json.dumps({"bloques": bloques})


class _Poses:
    """Lo unico que mira CP_LentaPlan.execute de las poses antes de medir: que haya y cuantas son."""

    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n


def plan_de(L, caso, ritmo, mas_lentos):
    return L.planificar(caso["inicios"], caso["total"], _saltos(caso), caso["umbral"], caso["max_ventana"],
                        caso["transicion_max"], caso["max_total"], ritmo=ritmo, mas_lentos=mas_lentos)


def huella_planificar(L, caso, ritmo, mas_lentos):
    return _intentar(lambda: json.dumps(plan_de(L, caso, ritmo, mas_lentos)))


def huella_nodo(L, caso, ritmo, mas_lentos):
    """CP_LentaPlan.execute con la medida de los saltos sustituida por la del caso: plan, n_ventanas,
    fotogramas_lentos e info. None si el caso no tiene un mapa que el nodo pueda leer."""
    mapa = mapa_de(caso)
    if mapa is None:
        return None
    medir = L.medir_saltos
    L.medir_saltos = lambda poses, nodo="": _saltos(caso)
    try:
        return _intentar(lambda: list(_args(L.LentaPlan.execute(
            _Poses(len(caso["inicios"])), mapa, caso["umbral"], caso["max_ventana"], caso["transicion_max"],
            caso["max_total"], ritmo, mas_lentos))))
    finally:
        L.medir_saltos = medir


def variantes_con_error(L, caso):
    """Las variantes en las que planificar da error (para leer el fichero y para saber que hay de los dos)."""
    sal = []
    for r, m in VARIANTES:
        try:
            plan_de(L, caso, r, m)
        except Exception:  # noqa: BLE001
            sal.append(clave(r, m))
    return sal


def clave(ritmo, mas_lentos):
    return f"{ritmo}{'+' if mas_lentos else '-'}"


def mapa_al_azar(rnd):
    """2-12 poses (a veces 13-30), saltos sobre todo de 1-12 fotogramas (a veces 13-48), cola de 1-24: blockings
    cortos, para que con ritmo alto haya planes y no solo el error del tope de salida. Medidas de 0-40 px con bordes
    del umbral, transicion_max 2-64 y a veces un max_total bajo (para que salga su error)."""
    n = rnd.randint(2, 12) if rnd.random() < 0.7 else rnd.randint(13, 30)
    dur = [rnd.randint(1, 12) if rnd.random() < 0.75 else rnd.randint(13, 48) for _ in range(n - 1)]
    inicios = [0]
    for x in dur:
        inicios.append(inicios[-1] + x)
    total = inicios[-1] + rnd.randint(1, 24)
    saltos = []
    for _ in range(n - 1):
        if rnd.random() < 0.1:
            saltos.append(rnd.choice([0.0, 8.9995, 8.9994, 9.0, 40.0]))
        else:
            saltos.append(round(rnd.uniform(0.0, 40.0), rnd.choice([2, 3, 3, 4])))
    return {
        "inicios": inicios, "total": total, "saltos": saltos,
        "umbral": rnd.choice([0.0, 9.0, round(rnd.uniform(0.0, 40.0), 2)]),
        "max_ventana": rnd.randint(33, 513), "transicion_max": rnd.randint(2, 64),
        "max_total": rnd.randint(65, 8001) if rnd.random() < 0.8 else rnd.randint(65, 400),
    }


def casos_fijos(control):
    """El plano de control, los bordes de los topes y de la regla de mas lentos, y los errores."""
    c = []
    d = dict(umbral=9.0, max_ventana=177, transicion_max=12, max_total=1601)
    c.append(dict(d, que="plano de control", inicios=control["inicios"], total=control["bloques"][-1][1] + 1,
                  saltos=control["saltos_dis_480"]))
    c.append(dict(d, que="salida 1000 con ritmo 2", inicios=[0, 250], total=500, saltos=[5.0]))
    c.append(dict(d, que="salida 1002 con ritmo 2", inicios=[0, 250], total=501, saltos=[5.0]))
    c.append(dict(d, que="salida 1000 con ritmo 8", inicios=[0, 60], total=125, saltos=[5.0]))
    c.append(dict(d, que="salida 1008 con ritmo 8", inicios=[0, 60], total=126, saltos=[5.0]))
    c.append(dict(d, que="tramo corto: m 8 y 9", inicios=[0, 8, 17, 26], total=30, saltos=[1.0, 1.0, 1.0]))
    c.append(dict(d, que="m 16 y 17", inicios=[0, 16, 33], total=40, saltos=[1.0, 1.0], transicion_max=64))
    c.append(dict(d, que="espera larga", inicios=[0, 4, 68, 72], total=73, saltos=[1.0, 30.0, 1.0], max_ventana=33,
                  transicion_max=64))
    c.append(dict(d, que="umbral redondeado", inicios=[0, 4, 8, 12], total=13, saltos=[8.9995, 8.9994, 9.0]))
    c.append(dict(d, que="max_total justo", inicios=[i * 4 for i in range(30)], total=117, saltos=[30.0] * 29,
                  max_total=934))
    c.append(dict(d, que="max_total pasado", inicios=[i * 4 for i in range(30)], total=117, saltos=[30.0] * 29,
                  max_total=933))
    c.append(dict(d, que="max_total con ritmo", inicios=[i * 10 for i in range(40)], total=400, saltos=[1.0] * 39,
                  max_total=1000))
    c.append(dict(d, que="una pose", inicios=[0], total=12, saltos=[]))
    c.append(dict(d, que="saltos que no cuadran", inicios=[0, 4, 8], total=9, saltos=[1.0]))
    c.append(dict(d, que="inicios repetidos", inicios=[0, 4, 4], total=9, saltos=[1.0, 1.0]))
    c.append(dict(d, que="no empieza en 0", inicios=[2, 4, 8], total=9, saltos=[1.0, 1.0]))
    c.append(dict(d, que="total corto", inicios=[0, 4, 8], total=8, saltos=[1.0, 1.0]))
    c.append(dict(d, que="medida que no es numero", inicios=[0, 4, 8], total=9, saltos=[1.0, "nan"]))
    c.append(dict(d, que="muchas ventanas", inicios=[i * 2 for i in range(132)], total=264, saltos=[1.0] * 131,
                  max_ventana=33, max_total=8001))
    return c


def casos_fotos(P, rnd):
    """Los mapas que da CP_PosesDesdeFotos (con mapa_bloques de la v0.15) y lo que pone la tarjeta de fotos en
    CP_LentaPlan: transicion_max = fotogramas_por_pose."""
    c = []
    for n in (2, 5, 12, 24):
        for fpp in (2, 8, 9, 12, 16, 32):
            for espera in (0, 240):
                bloques = P.mapa_bloques(n, fpp, espera)
                c.append({"que": f"fotos {n}x{fpp}+{espera}", "inicios": [a for a, _ in bloques],
                          "total": bloques[-1][1] + 1, "saltos": [round(rnd.uniform(0.0, 30.0), 3) for _ in range(n - 1)],
                          "umbral": 9.0, "max_ventana": 177, "transicion_max": fpp, "max_total": 1601})
    return c


def ritmos_raros(L, caso):
    """Valores de ritmo y mas_lentos que no son los de la lista: lo que hace la v0.15 con ellos (error o plan)."""
    sal = {}
    for r in (0, 9, -1, True, 2.5, 2.0, "2", None):
        sal[f"planificar {r!r}"] = _intentar(lambda r=r: json.dumps(plan_de(L, caso, r, False)))
        sal[f"nodo {r!r}"] = huella_nodo(L, caso, r, False)
    for m in (0, 1, "", None):
        sal[f"nodo mas_lentos {m!r}"] = huella_nodo(L, caso, 1, m)
    return sal


def generar_planes(L, P, control, casos, semilla):
    rnd = random.Random(semilla)
    lista = casos_fijos(control) + casos_fotos(P, rnd) + [mapa_al_azar(rnd) for _ in range(casos)]
    for caso in lista:
        caso["saltos"] = [x if isinstance(x, str) else round(x, 4) for x in caso["saltos"]]
        caso["planificar"] = {clave(r, m): huella_planificar(L, caso, r, m) for r, m in VARIANTES}
        caso["con_error"] = variantes_con_error(L, caso)
        nodo = {clave(r, m): huella_nodo(L, caso, r, m) for r, m in VARIANTES}
        if all(v is not None for v in nodo.values()):
            caso["nodo"] = nodo
    return lista


# ---------------------------------------------------------------- eleccion

FORMAS = ("uniforme", "constante", "ceros", "minimo", "huecos", "pico", "escalon", "redondo")


def recorrido_de(semilla, n, forma):
    """n valores de recorrido rehechos desde la semilla (solo con random(), estable entre versiones de Python)."""
    r = random.Random(semilla)
    if forma == "uniforme":
        return [round(r.random() * 4, 4) for _ in range(n)]
    if forma == "constante":
        return [1.0] * n
    if forma == "ceros":
        return [0.0] * n
    if forma == "minimo":
        return [1e-8] * n
    if forma == "huecos":
        return [0.0 if r.random() < 0.4 else round(r.random() * 5, 3) for _ in range(n)]
    if forma == "pico":
        v = [0.0] * n
        if n:
            v[int(r.random() * n)] = 10.0
        return v
    if forma == "escalon":
        return [0.5 if (i // max(1, n // 4)) % 2 else 2.0 for i in range(n)]
    if forma == "redondo":  # enteros pequenos: muchos empates exactos al buscar el mas cercano
        return [float(int(r.random() * 3)) for _ in range(n)]
    raise ValueError(forma)


def eleccion_al_azar(rnd, i):
    """Un caso de elegir_indices: 1-6 saltos con lentos de 1-64 (no solo multiplos de 8), d de 1-80, transicion_max
    2-64. A veces lentos < m: es el error que la v0.15 da."""
    saltos = rnd.randint(1, 6)
    tm = rnd.randint(2, 64)
    d = [rnd.randint(1, 80) if rnd.random() < 0.5 else rnd.randint(1, 16) for _ in range(saltos)]
    lentos = []
    for x in d:
        m = min(x, tm)
        lentos.append(rnd.randint(max(1, m), max(m, 64)) if rnd.random() < 0.95 else max(1, m - 1))
    return {"lentos": lentos, "d": d, "transicion_max": tm, "forma": FORMAS[i % len(FORMAS)],
            "semilla": rnd.randint(0, 2 ** 31)}


def elecciones_fijas():
    """Los bordes: empates de redondeo con recorrido constante, sin movimiento, d < m y los errores."""
    c = []
    for lentos, d, tm in (([3], [2], 12), ([16], [12], 12), ([16], [6], 12), ([32], [12], 12), ([24], [9], 12),
                          ([5], [4], 4), ([16, 32], [40, 12], 12), ([16], [1], 12), ([8, 8], [3, 5], 2)):
        for forma in ("constante", "ceros", "minimo", "redondo"):
            c.append({"lentos": lentos, "d": d, "transicion_max": tm, "forma": forma, "semilla": 1})
    c.append({"lentos": [16], "d": [0], "transicion_max": 12, "forma": "uniforme", "semilla": 2})  # m < 1
    c.append({"lentos": [8], "d": [12], "transicion_max": 12, "forma": "uniforme", "semilla": 3})  # lentos < m
    c.append({"lentos": [16], "d": [12], "transicion_max": 12, "forma": "uniforme", "semilla": 4, "sobra": 1})
    c.append({"lentos": [16], "d": [12, 4], "transicion_max": 12, "forma": "uniforme", "semilla": 5})
    c.append({"lentos": [16], "d": [12], "transicion_max": 12, "forma": "uniforme", "semilla": 6, "raro": "negativo"})
    c.append({"lentos": [16], "d": [12], "transicion_max": 12, "forma": "uniforme", "semilla": 7, "raro": "nan"})
    return c


def recorrido_caso(caso):
    v = recorrido_de(caso["semilla"], sum(caso["lentos"]) + caso.get("sobra", 0), caso["forma"])
    if caso.get("raro") == "negativo":
        v[0] = -0.5
    elif caso.get("raro") == "nan":
        v[0] = math.nan
    return v


def huella_eleccion(L, caso):
    return _intentar(lambda: L.elegir_indices(recorrido_caso(caso), caso["lentos"], caso["d"], caso["transicion_max"]))


def huella_recorrido(caso):
    return huella(repr(recorrido_caso(caso)))


def variante_ventanas(i):
    """La variante de ritmo y mas_lentos con la que se eligen los indices del plan i (asi se reparten todas)."""
    return VARIANTES[(i * 5) % len(VARIANTES)]


def _medidos(plan, semilla):
    """Un recorrido por ventana del plan, rehecho desde la semilla."""
    formas = ("uniforme", "huecos", "redondo", "pico")
    sal = []
    for w, v in enumerate(plan["ventanas"]):
        sal.append(recorrido_de(semilla * 1000 + w, v["n"] - 1, formas[(semilla + w) % len(formas)]))
    return sal


def indices_plan(L, plan, semilla):
    """Los indices de indices_ventana de todas las ventanas del plan, con los recorridos de su semilla."""
    return [L.indices_ventana(plan, w, m) for w, m in enumerate(_medidos(plan, semilla))]


def montar_plan(L, plan, semilla):
    """montar_ventana encadenado por todas las ventanas con lotes cuyo fotograma i vale i (render) y 10000 + i (guia):
    lo que sale dice de que lento viene cada fotograma. Devuelve (render, guia) como listas de enteros."""
    import torch

    anterior_render = anterior_guia = None
    for w, medido in enumerate(_medidos(plan, semilla)):
        n = plan["ventanas"][w]["n"]
        base = torch.arange(n, dtype=torch.float32).view(n, 1, 1, 1) + 100000 * w
        render, guia = base.clone(), base + 10000
        anterior_render, anterior_guia = L.montar_ventana(guia, render, plan, w, anterior_render, anterior_guia,
                                                          medido=medido)
    return [int(x) for x in anterior_render.view(-1).tolist()], [int(x) for x in anterior_guia.view(-1).tolist()]


def errores_montar(L, plan):
    """Los errores de montar_ventana (ventana que no existe, lotes que no son los de la ventana, lo acumulado mal)."""
    import torch

    n0 = plan["ventanas"][0]["n"]
    bien = torch.zeros(n0, 1, 1, 1)
    sal = {}
    pruebas = {
        "ventana -1": lambda: L.montar_ventana(bien, bien, plan, -1, medido=[0.0] * (n0 - 1)),
        "ventana fuera": lambda: L.montar_ventana(bien, bien, plan, len(plan["ventanas"]), medido=[0.0] * (n0 - 1)),
        "guia corta": lambda: L.montar_ventana(bien[:-1], bien, plan, 0, medido=[0.0] * (n0 - 1)),
        "render nada": lambda: L.montar_ventana(bien, None, plan, 0, medido=[0.0] * (n0 - 1)),
        "acumulado de mas": lambda: L.montar_ventana(bien, bien, plan, 0, bien, bien, medido=[0.0] * (n0 - 1)),
    }
    if len(plan["ventanas"]) > 1:
        n1 = plan["ventanas"][1]["n"]
        otra = torch.zeros(n1, 1, 1, 1)
        previos = plan["inicios"][plan["ventanas"][1]["p0"]]
        pruebas["sin acumulado"] = lambda: L.montar_ventana(otra, otra, plan, 1, medido=[0.0] * (n1 - 1))
        pruebas["acumulado de otro tamano"] = lambda: L.montar_ventana(
            otra, otra, plan, 1, torch.zeros(previos, 2, 1, 1), torch.zeros(previos, 2, 1, 1), medido=[0.0] * (n1 - 1))
    for k, f in pruebas.items():
        sal[k] = _intentar(lambda f=f: [list(x.shape) for x in f()])
    return sal


def generar_eleccion(L, planes, semilla):
    rnd = random.Random(semilla + 1)
    casos = elecciones_fijas() + [eleccion_al_azar(rnd, i) for i in range(160)]
    for caso in casos:
        caso["recorrido"] = huella_recorrido(caso)
        caso["indices"] = huella_eleccion(L, caso)
    ventanas, montajes = [], []
    for i, caso in enumerate(planes):
        r, m = variante_ventanas(i)
        try:
            plan = plan_de(L, caso, r, m)
        except ValueError:
            continue
        ventanas.append({"plan": i, "variante": clave(r, m), "indices": _intentar(lambda: indices_plan(L, plan, i))})
        if plan["total"] <= 400 and len(montajes) < 40:
            montajes.append({"plan": i, "variante": clave(r, m), "montado": _intentar(lambda: montar_plan(L, plan, i))})
    control = plan_de(L, planes[0], 1, False)
    return {"elegir": casos, "ventanas": ventanas, "montar": montajes,
            "errores_montar": {"plan": 0, "variante": clave(1, False), "huellas": errores_montar(L, control)}}


# ---------------------------------------------------------------- subgrafo

# La tarjeta publicada de VIDEO (ids de su prompt, numeros de nodo y nada mas) y tools/probar_expansion.py.
TARJETA_ENLACES = {
    "ruido": ["9", 0], "poses": ["18", 0], "arte": ["51", 0], "modelo_suave": ["53", 0], "modelo_render": ["55", 0],
    "factor_guia": ["54", 1], "vae": ["57", 0], "escalador": ["58", 0], "positivo_suave": ["66", 0],
    "negativo_suave": ["66", 1], "positivo_render": ["65", 0], "negativo_render": ["65", 1],
}
PRUEBA_ENLACES = {
    "ruido": ["4", 0], "poses": ["1", 0], "arte": ["5", 0], "modelo_suave": ["6", 0], "modelo_render": ["7", 0],
    "factor_guia": ["8", 0], "vae": ["9", 0], "escalador": ["10", 0], "positivo_suave": ["11", 0],
    "negativo_suave": ["12", 0], "positivo_render": ["13", 0], "negativo_render": ["14", 0],
}
ENLACES = {"tarjeta": (TARJETA_ENLACES, 1920, 1088), "prueba": (PRUEBA_ENLACES, 256, 128)}


def planes_subgrafo(planes):
    """(nombre, indice del caso de planes, ritmo, mas_lentos): el plano de control y mapas de fotos."""
    def buscar(que):
        return next(i for i, c in enumerate(planes) if c.get("que") == que)

    sal = [("control", 0, 1, False), ("control+", 0, 1, True), ("control r2", 0, 2, False), ("control r3+", 0, 3, True)]
    for que, ritmos in (("fotos 5x12+0", (1, 2)), ("fotos 12x9+0", (1, 2)), ("fotos 12x16+0", (1, 3)),
                        ("fotos 24x32+240", (1,))):
        sal += [(f"{que} r{r}", buscar(que), r, True) for r in ritmos]
    return sal


def subgrafo(LG, plan, enlaces, ancho, alto, solo):
    grafo, animacion, playblast = LG.grafo_lento(plan, enlaces, ancho, alto, PREFIJO, solo_animacion=solo)
    return {"sha256": huella_grafo(grafo), "nodos": len(grafo), "salidas": [animacion, playblast]}


def nodo_lento(LG, plan, enlaces, ancho, alto, **extra):
    """CP_BlockingLento.execute con el prefijo fijo: huella de la expansion y las dos salidas."""
    with _modulos(**_doble_graph_utils()):
        salida = LG.BlockingLento.execute(json.dumps(plan), ancho, alto, **enlaces, **extra)
    return {"sha256": huella_grafo(_expand(salida)), "salidas": [list(x) for x in _args(salida)]}


def generar_subgrafos(L, LG, planes):
    grafos = []
    for nombre, i, r, m in planes_subgrafo(planes):
        plan = plan_de(L, planes[i], r, m)
        for enl_nombre, (enlaces, ancho, alto) in ENLACES.items():
            for solo in (False, True):
                g = {"que": nombre, "plan": i, "variante": clave(r, m), "enlaces": enl_nombre, "solo": solo}
                g.update(subgrafo(LG, plan, enlaces, ancho, alto, solo))
                if solo:  # sin los enlaces del render da lo mismo
                    suave = {k: v for k, v in enlaces.items() if k in LG.ENLACES_SUAVE}
                    g["solo_sin_render"] = subgrafo(LG, plan, suave, ancho, alto, True)["sha256"]
                grafos.append(g)
    control = plan_de(L, planes[0], 1, False)
    nodo = {}
    for nombre, extra in (("sin solo", {}), ("solo False", {"solo_animacion": False}),
                          ("solo True", {"solo_animacion": True}), ("solo None", {"solo_animacion": None})):
        nodo[nombre] = _intentar(lambda extra=extra: nodo_lento(LG, control, TARJETA_ENLACES, 1920, 1088, **extra))
    muchas = next(i for i, c in enumerate(planes) if c.get("que") == "muchas ventanas")
    errores = {
        "solo no booleano": _intentar(lambda: subgrafo(LG, control, TARJETA_ENLACES, 1920, 1088, "si")),
        "tamano": _intentar(lambda: subgrafo(LG, control, TARJETA_ENLACES, 1900, 1088, False)),
        "falta arte sin solo": _intentar(lambda: subgrafo(
            LG, control, {k: v for k, v in TARJETA_ENLACES.items() if k != "arte"}, 1920, 1088, False)),
        "falta ruido con solo": _intentar(lambda: subgrafo(
            LG, control, {k: v for k, v in TARJETA_ENLACES.items() if k != "ruido"}, 1920, 1088, True)),
        "mas de MAX_VENTANAS": _intentar(lambda: subgrafo(LG, plan_de(L, planes[muchas], 1, False), TARJETA_ENLACES,
                                                          1920, 1088, True)),
        "nodo plan roto": _intentar(lambda: nodo_lento(LG, {"version": 2}, TARJETA_ENLACES, 1920, 1088)),
    }
    return {"grafos": grafos, "nodo": nodo, "errores": errores}


# ---------------------------------------------------------------- fotos

FOTOS_POSES = (-1, 0, 1, 2, 3, 5, 12, 24, 25)
FOTOS_FPP = (-1, 0, 1, 2, 3, 7, 8, 9, 12, 13, 31, 32, 33)
FOTOS_ESPERA = (-1, 0, 1, 5, 239, 240, 241)


def huella_tiempos(P, fpp, espera):
    """mapa_bloques y total_fotogramas para todos los numeros de poses, con un fotogramas_por_pose y una espera."""
    sal = []
    for n in FOTOS_POSES:
        sal.append([_intentar(lambda n=n: P.mapa_bloques(n, fpp, espera)),
                    _intentar(lambda n=n: P.total_fotogramas(n, fpp, espera))])
    return huella_de(sal)


def _pngs(carpeta, n):
    """n fotos pequenas y distintas (8x6, un color liso por pose) en la carpeta: pose_01.png..."""
    from PIL import Image

    nombres = []
    for i in range(n):
        nombre = f"pose_{i + 1:02d}.png"
        Image.new("RGB", (8, 6), ((i * 37) % 256, (i * 91) % 256, (255 - i * 13) % 256)).save(os.path.join(carpeta, nombre))
        nombres.append(nombre)
    return nombres


def huella_desde_fotos(P, n, fpp, espera, orden="casilla", hueco=False, falta=False, al_reves=False):
    """CP_PosesDesdeFotos.execute con n fotos sinteticas: mapa, n_poses, transicion, fotogramas, fps y la forma y el
    sha256 de las poses. hueco deja vacia la casilla 2 (las fotos van de la 1 y de la 3 en adelante); al_reves pone
    la ultima foto (por nombre) en la primera casilla."""
    carpeta = os.path.realpath(tempfile.mkdtemp(prefix="compat_v015_fotos_"))
    try:
        nombres = _pngs(carpeta, n)
        casillas = {c: P.MARCADOR_VACIO for c in P.NOMBRES_CASILLAS}
        huecos = [c for c in P.NOMBRES_CASILLAS if not (hueco and c == P.NOMBRES_CASILLAS[1])]
        for c, nombre in zip(huecos, nombres[::-1] if al_reves else nombres):
            casillas[c] = nombre
        if falta:
            casillas[huecos[0]] = "no_esta.png"

        def correr():
            with _modulos(**_doble_folder_paths(carpeta)):
                a = _args(P.PosesDesdeFotos.execute(fpp, espera, orden, **casillas))
            poses = a[0].contiguous().numpy()
            return [list(poses.shape), hashlib.sha256(poses.tobytes()).hexdigest()] + list(a[1:])

        return _intentar(correr)
    finally:
        shutil.rmtree(carpeta, ignore_errors=True)


FOTOS_NODO = [(0, 12, 0), (1, 12, 0), (2, 12, 0), (2, 2, 0), (3, 9, 5), (5, 32, 240), (24, 12, 0), (24, 2, 1)]


def generar_fotos(P):
    tiempos = {f"{fpp}/{espera}": huella_tiempos(P, fpp, espera) for fpp in FOTOS_FPP for espera in FOTOS_ESPERA}
    nodo = {f"{n}/{fpp}/{espera}": huella_desde_fotos(P, n, fpp, espera) for n, fpp, espera in FOTOS_NODO}
    nodo["hueco 4/12/0"] = huella_desde_fotos(P, 4, 12, 0, hueco=True)
    nodo["al reves 4/12/0"] = huella_desde_fotos(P, 4, 12, 0, al_reves=True)
    nodo["al reves por nombre 4/12/0"] = huella_desde_fotos(P, 4, 12, 0, orden="nombre", al_reves=True)
    nodo["falta 3/12/0"] = huella_desde_fotos(P, 3, 12, 0, falta=True)
    constantes = huella_de([P.CASILLAS, P.FPS, P.MAX_FOTOGRAMAS_POR_POSE, P.MAX_ESPERA_FINAL, P.MARCADOR_VACIO,
                            P.NOMBRES_CASILLAS])
    return {"tiempos": tiempos, "nodo": nodo, "constantes": constantes}


# ---------------------------------------------------------------- todo

def constantes_lenta(L, LG):
    return huella_de([L.VERSION_PLAN, L.ANCHO_MEDIDA, L.PARTE_MOVIL, L.LENTOS_CORTO, L.LENTOS_LARGO, L.UMBRAL_SALTO,
                      L.MAX_VENTANA, L.TRANSICION_MAX, L.MAX_TOTAL, L.MAX_SALIDA, L.MAX_RITMO, L.TRAMO_CORTO,
                      L.SIN_MOVIMIENTO, LG.NODO, LG.MUESTREADOR, LG.MAX_VENTANAS, list(LG.ENLACES),
                      list(LG.ENLACES_RENDER)])


def generar(carpeta_pack, nombre, control, casos=CASOS, semilla=SEMILLA):
    L, LG, P = cargar_pack(carpeta_pack, nombre)
    planes = generar_planes(L, P, control, casos, semilla)
    return {
        "semilla": semilla,
        "constantes": {"lenta": constantes_lenta(L, LG)},
        "ritmos_raros": ritmos_raros(L, planes[0]),
        "planes": planes,
        "eleccion": generar_eleccion(L, planes, semilla),
        "subgrafo": generar_subgrafos(L, LG, planes),
        "fotos": generar_fotos(P),
    }


def leer_control():
    with open(os.path.join(AQUI, "..", "tests", "datos", "lenta_0003.json"), encoding="utf-8") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("v015", help="carpeta con la copia de la v0.15 (git archive bafe3eb)")
    ap.add_argument("salida", nargs="?", default=os.path.join(AQUI, "..", "tests", "datos", "compat_v015.json"))
    ap.add_argument("--casos", type=int, default=CASOS)
    ap.add_argument("--semilla", type=int, default=SEMILLA)
    a = ap.parse_args()
    datos = {"que": "Huellas de la v0.15 (bafe3eb) en camara lenta y fotos, por tools/generar_compat_v015.py"}
    datos.update(generar(a.v015, "pack_v015", leer_control(), a.casos, a.semilla))
    with open(a.salida, "w", encoding="utf-8") as f:
        json.dump(datos, f, ensure_ascii=True, separators=(",", ":"))
        f.write("\n")
    planes = datos["planes"]
    errores = sum(len(c["con_error"]) for c in planes)
    print(f"{len(planes)} mapas x {len(VARIANTES)} variantes ({errores} con error), "
          f"{len(datos['eleccion']['elegir'])} elecciones, "
          f"{len(datos['eleccion']['ventanas'])} planes elegidos, {len(datos['subgrafo']['grafos'])} subgrafos, "
          f"{len(datos['fotos']['tiempos'])} tiempos de fotos en {os.path.normpath(a.salida)}")

if __name__ == "__main__":
    main()
