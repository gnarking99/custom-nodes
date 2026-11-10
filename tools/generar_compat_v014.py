# -*- coding: utf-8 -*-
"""Genera tests/datos/compat_v014.json: mapas de blocking y el plan (o el error) que les da la v0.14 de lenta.py.

tests/test_lenta_compat_v014.py comprueba con ese fichero que la version actual, sin "ritmo" ni "mas_lentos", da el
mismo plan byte a byte (o el mismo error con el mismo texto) que la v0.14, en planificar y en el nodo CP_LentaPlan.
De cada plan se guarda el sha256 de su JSON (la comparacion byte a byte) y sus lentos y cortes (para leer un fallo).

Para regenerarlo hace falta una copia de la v0.14 (el commit 917bb3d):

    mkdir -p /tmp/v014 && git archive 917bb3d | tar -x -C /tmp/v014
    python3 tools/generar_compat_v014.py /tmp/v014

Con un segundo argumento escribe en otro sitio; con --casos N --semilla S cambia cuantos y cuales (los de por
defecto son los del fichero del repo). No hace falta ComfyUI: si no esta, se usa un doble minimo de comfy_api.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import random
import sys
import types

AQUI = os.path.dirname(os.path.abspath(__file__))
CASOS = 300
SEMILLA = 14


def _doble_comfy_api():
    """comfy_api.latest.io lo justo para importar lenta.py y llamar a execute sin ComfyUI."""
    try:
        import comfy_api.latest  # noqa: F401
        return
    except ImportError:
        pass

    class _Port:
        def __init__(self, *a, **k):
            self.a, self.k = a, k

    class _Salida:
        def __init__(self, *a, **k):
            self.args = a

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


def cargar_pack(carpeta_pack, nombre):
    """(lenta, lenta_grafo) de un pack (la carpeta que tiene comfy_character_pipeline/), cargado como paquete con ese
    nombre: asi dos versiones conviven en el mismo proceso sin pisarse."""
    _doble_comfy_api()
    carpeta = os.path.join(carpeta_pack, "comfy_character_pipeline")
    spec = importlib.util.spec_from_file_location(nombre, os.path.join(carpeta, "__init__.py"),
                                                  submodule_search_locations=[carpeta])
    m = importlib.util.module_from_spec(spec)
    sys.modules[nombre] = m
    spec.loader.exec_module(m)
    return importlib.import_module(nombre + ".lenta"), importlib.import_module(nombre + ".lenta_grafo")


def cargar_lenta(carpeta_pack, nombre):
    return cargar_pack(carpeta_pack, nombre)[0]


def mapa_al_azar(rnd):
    """Un caso: 2-60 poses, bloques de 1-80, ultimo de 1-60, saltos de 0-40 px, umbral 0/9/al azar, max_ventana
    33-513, transicion_max 2-64, max_total 65-8001."""
    n = rnd.randint(2, 60)
    dur = [rnd.randint(1, 80) if rnd.random() < 0.5 else rnd.randint(1, 16) for _ in range(n - 1)]
    inicios = [0]
    for x in dur:
        inicios.append(inicios[-1] + x)
    total = inicios[-1] + rnd.randint(1, 60)
    saltos = []
    for _ in range(n - 1):
        u = rnd.random()
        if u < 0.1:
            saltos.append(rnd.choice([0.0, 8.9995, 8.9994, 9.0, 40.0]))
        else:
            saltos.append(round(rnd.uniform(0.0, 40.0), rnd.choice([2, 3, 3, 4])))
    umbral = rnd.choice([0.0, 9.0, round(rnd.uniform(0.0, 40.0), 2)])
    return {
        "inicios": inicios, "total": total, "saltos": saltos, "umbral": umbral,
        "max_ventana": rnd.randint(33, 513), "transicion_max": rnd.randint(2, 64), "max_total": rnd.randint(65, 8001),
    }


def casos_fijos():
    """Los de los revisores y los bordes (errores incluidos)."""
    c = []
    d = dict(umbral=9.0, max_ventana=177, transicion_max=12, max_total=1601)
    ini = [i * 10 for i in range(12)]
    c.append(dict(d, inicios=ini, total=120, saltos=[5.0] * 11))  # 12 poses cada 10, saltos de 5 px
    ini = [i * 40 for i in range(30)]
    c.append(dict(d, inicios=ini, total=1200, saltos=[5.0] * 29))  # 1200 fotogramas
    ini = [i * 25 for i in range(41)]
    c.append(dict(d, inicios=ini, total=1025, saltos=[20.0] * 40))  # 1025 fotogramas, saltos grandes
    c.append(dict(d, inicios=[0, 6, 26, 36, 40, 60], total=72, saltos=[4.0, 5.0, 3.0, 6.0, 2.0]))
    c.append(dict(d, inicios=[0, 40], total=41, saltos=[2.0]))
    c.append(dict(d, inicios=[0, 20, 60, 61], total=62, saltos=[1.0, 50.0, 1.0], transicion_max=40))
    c.append(dict(d, inicios=[0, 4, 68, 72], total=73, saltos=[1.0, 30.0, 1.0], max_ventana=33, transicion_max=64))
    # errores: una pose, saltos que no cuadran, tiempos rotos, medida que no es numero, demasiados lentos
    c.append(dict(d, inicios=[0], total=12, saltos=[]))
    c.append(dict(d, inicios=[0, 4, 8], total=9, saltos=[1.0]))
    c.append(dict(d, inicios=[0, 4, 4], total=9, saltos=[1.0, 1.0]))
    c.append(dict(d, inicios=[2, 4, 8], total=9, saltos=[1.0, 1.0]))
    c.append(dict(d, inicios=[0, 4, 8], total=8, saltos=[1.0, 1.0]))
    c.append(dict(d, inicios=[0, 4, 8], total=9, saltos=[1.0, "nan"]))
    c.append(dict(d, inicios=[i * 4 for i in range(30)], total=117, saltos=[30.0] * 29, max_total=933))
    return c


def _saltos(caso):
    return [float(x) for x in caso["saltos"]]


def mapa_de(caso):
    """El mapa {"bloques"} que da esos inicios y ese total (None si no se puede: el nodo no lo leeria)."""
    ini, total = caso["inicios"], caso["total"]
    if len(ini) < 1 or ini[0] != 0 or any(b <= a for a, b in zip(ini, ini[1:])) or total <= ini[-1]:
        return None
    bloques = [[a, b - 1] for a, b in zip(ini, ini[1:])] + [[ini[-1], total - 1]]
    return json.dumps({"bloques": bloques})


def huella(texto):
    """sha256 del JSON del plan tal cual sale (byte a byte)."""
    return hashlib.sha256(texto.encode("utf-8")).hexdigest()


def resumen(texto):
    """Lo que se guarda de un plan: su huella (para la comparacion byte a byte) y, para leer un fallo, los lentos y
    la pose final de cada ventana."""
    plan = json.loads(texto)
    return {"sha256": huella(texto), "lentos": plan["lentos"], "cortes": [v["p1"] for v in plan["ventanas"]]}


def resultado_plan(L, caso):
    try:
        plan = L.planificar(caso["inicios"], caso["total"], _saltos(caso), caso["umbral"], caso["max_ventana"],
                            caso["transicion_max"], caso["max_total"])
    except ValueError as e:
        return {"error": str(e)}
    return {"plan": json.dumps(plan)}


class _Poses:
    """Lo unico que mira CP_LentaPlan.execute de las poses antes de medir: que haya y cuantas son."""

    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n


def resultado_nodo(L, caso):
    """CP_LentaPlan.execute con la medida de los saltos sustituida por la del caso. Sin info (la v0.15 le anade la
    linea de salida): plan, n_ventanas y fotogramas_lentos."""
    mapa = mapa_de(caso)
    if mapa is None:
        return None
    medir = L.medir_saltos
    L.medir_saltos = lambda poses, nodo="": _saltos(caso)
    try:
        salida = L.LentaPlan.execute(_Poses(len(caso["inicios"])), mapa, caso["umbral"], caso["max_ventana"],
                                     caso["transicion_max"], caso["max_total"])
    except ValueError as e:
        return {"error": str(e)}
    finally:
        L.medir_saltos = medir
    args = getattr(salida, "args", None) or tuple(getattr(salida, "result", ()))
    return {"plan": args[0], "n_ventanas": args[1], "fotogramas_lentos": args[2]}


# La tarjeta publicada (VIDEO/LTX25 BLOCKING A ANIMACION): lo que su prompt envia a CP_LentaPlan y CP_BlockingLento,
# sin ritmo, mas_lentos ni solo_animacion. Los enlaces son los ids de su prompt (numeros de nodo, nada mas).
TARJETA_PLAN = {"umbral": 9, "max_ventana": 177, "transicion_max": 12, "max_total": 1601}
TARJETA_ENLACES = {
    "ruido": ["9", 0], "poses": ["18", 0], "arte": ["51", 0], "modelo_suave": ["53", 0], "modelo_render": ["55", 0],
    "factor_guia": ["54", 1], "vae": ["57", 0], "escalador": ["58", 0], "positivo_suave": ["66", 0],
    "negativo_suave": ["66", 1], "positivo_render": ["65", 0], "negativo_render": ["65", 1],
}
# Los de tools/probar_expansion.py (seccion 10), que la ejecuta en el motor con sus dobles a 256x128.
PRUEBA_ENLACES = {
    "ruido": ["4", 0], "poses": ["1", 0], "arte": ["5", 0], "modelo_suave": ["6", 0], "modelo_render": ["7", 0],
    "factor_guia": ["8", 0], "vae": ["9", 0], "escalador": ["10", 0], "positivo_suave": ["11", 0],
    "negativo_suave": ["12", 0], "positivo_render": ["13", 0], "negativo_render": ["14", 0],
}
PREFIJO = "@"  # el prefijo de los ids del subgrafo; quien compare cambia el suyo por este


def huella_grafo(grafo):
    """sha256 del subgrafo ({id: {class_type, inputs}}) en JSON con las claves ordenadas."""
    limpio = {i: {"class_type": d["class_type"], "inputs": d["inputs"]} for i, d in grafo.items()}
    return huella(json.dumps(limpio, sort_keys=True))


def tarjeta_publicada(L, LG, control):
    """El plano de control (25 poses, 77 fotogramas, tests/datos/lenta_0003.json) con lo que envia la tarjeta
    publicada: huella del plan y del subgrafo a 1920x1088 con sus enlaces y a 256x128 con los de probar_expansion."""
    inicios, total, saltos = control["inicios"], control["bloques"][-1][1] + 1, control["saltos_dis_480"]
    plan = L.planificar(inicios, total, saltos, TARJETA_PLAN["umbral"], TARJETA_PLAN["max_ventana"],
                        TARJETA_PLAN["transicion_max"], TARJETA_PLAN["max_total"])
    texto = json.dumps(plan)
    grafos = {}
    for nombre, enlaces, ancho, alto in (("tarjeta", TARJETA_ENLACES, 1920, 1088), ("prueba", PRUEBA_ENLACES, 256, 128)):
        grafo, animacion, playblast = LG.grafo_lento(plan, enlaces, ancho, alto, PREFIJO)
        grafos[nombre] = {"enlaces": enlaces, "ancho": ancho, "alto": alto, "sha256": huella_grafo(grafo),
                          "nodos": len(grafo), "salidas": [animacion, playblast]}
    return {"plan_entradas": TARJETA_PLAN, "plan": resumen(texto), "grafos": grafos}


def generar(L, casos=CASOS, semilla=SEMILLA):
    rnd = random.Random(semilla)
    lista = casos_fijos() + [mapa_al_azar(rnd) for _ in range(casos)]
    for caso in lista:
        caso["saltos"] = [x if isinstance(x, str) else round(x, 4) for x in caso["saltos"]]
        caso["esperado"] = resultado_plan(L, caso)
        nodo = resultado_nodo(L, caso)
        if nodo is not None:
            if "plan" in nodo:
                nodo["plan"] = {"sha256": huella(nodo["plan"])}
            caso["esperado_nodo"] = nodo
        if "plan" in caso["esperado"]:
            caso["esperado"]["plan"] = resumen(caso["esperado"]["plan"])
    return lista


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("v014", help="carpeta con la copia de la v0.14 (git archive 917bb3d)")
    ap.add_argument("salida", nargs="?", default=os.path.join(AQUI, "..", "tests", "datos", "compat_v014.json"))
    ap.add_argument("--casos", type=int, default=CASOS)
    ap.add_argument("--semilla", type=int, default=SEMILLA)
    a = ap.parse_args()
    L, LG = cargar_pack(a.v014, "pack_v014")
    lista = generar(L, a.casos, a.semilla)
    with open(os.path.join(AQUI, "..", "tests", "datos", "lenta_0003.json"), encoding="utf-8") as f:
        control = json.load(f)
    datos = {
        "que": "Planes de CP_LentaPlan generados con la v0.14 (917bb3d) por tools/generar_compat_v014.py",
        "semilla": a.semilla,
        "tarjeta_publicada": tarjeta_publicada(L, LG, control),
        "casos": lista,
    }
    with open(a.salida, "w", encoding="utf-8") as f:
        json.dump(datos, f, ensure_ascii=True, separators=(",", ":"))
        f.write("\n")
    errores = sum(1 for c in lista if "error" in c["esperado"])
    print(f"{len(lista)} casos ({errores} con error) en {os.path.normpath(a.salida)}")


if __name__ == "__main__":
    main()
