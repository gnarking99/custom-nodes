# -*- coding: utf-8 -*-
"""Compatibilidad dura con la v0.15: el codigo actual hace lo mismo que la v0.15 (el commit bafe3eb) en camara lenta
y fotos, caso a caso, para todo lo de tests/datos/compat_v015.json (huellas generadas con la v0.15 de verdad):
- planificar y CP_LentaPlan con ritmo 1 a 8 y mas_lentos True/False (el mismo plan byte a byte o el mismo error con
  el mismo texto), y lo que hacen con ritmos y mas_lentos que no son de la lista;
- los indices elegidos: elegir_indices, indices_ventana y montar_ventana;
- el subgrafo de CP_BlockingLento con solo_animacion False y True;
- los tiempos de las fotos: mapa_bloques, total_fotogramas y CP_PosesDesdeFotos.
Asi un grafo guardado con la v0.15 (lo que envian las tarjetas publicadas) se ejecuta igual. Lo que ya existia en
la v0.14 lo mira ademas test_lenta_compat_v014.py. El fichero se regenera con tools/generar_compat_v015.py (lo
explica su cabecera); no hace falta ComfyUI ni git para esta prueba.
Se ejecuta con: python tests/test_lenta_compat_v015.py"""
import json
import os
import sys
import types

# ---- stub minimo de comfy_api.latest ----
mod = types.ModuleType("comfy_api")
latest = types.ModuleType("comfy_api.latest")


class _Port:
    def __init__(self, *a, **k):
        self.a, self.k = a, k


class _Salida:
    def __init__(self, *a, **k):
        self.args, self.k = a, k


class _IO:
    Schema = staticmethod(lambda **k: k)
    NodeOutput = _Salida
    ComfyNode = type("ComfyNode", (), {})
    UploadType = types.SimpleNamespace(image="image_upload")


for _n in ("Image", "Float", "Int", "String", "Boolean", "Combo", "Mask"):
    setattr(_IO, _n, type(_n, (), {"Input": staticmethod(_Port), "Output": staticmethod(_Port)}))

latest.io = _IO
mod.latest = latest
sys.modules["comfy_api"] = mod
sys.modules["comfy_api.latest"] = latest

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.normpath(os.path.join(AQUI, ".."))
sys.path.insert(0, os.path.join(RAIZ, "tools"))
import generar_compat_v015 as G  # noqa: E402

L, LG, P = G.cargar_pack(RAIZ, "pack_actual")

with open(os.path.join(AQUI, "datos", "compat_v015.json"), encoding="utf-8") as _f:
    DATOS = json.load(_f)
PLANES = DATOS["planes"]
ELECCION = DATOS["eleccion"]
SUBGRAFO = DATOS["subgrafo"]
FOTOS = DATOS["fotos"]
CONTROL = G.leer_control()
VARIANTE = {G.clave(r, m): (r, m) for r, m in G.VARIANTES}  # "2+" -> (2, True)


def _comparar(que, pares):
    """pares: (donde, esperado, obtenido). Falla con el recuento y los primeros distintos."""
    distintos = [(d, e, o) for d, e, o in pares if e != o]
    if distintos:
        muestra = "; ".join(f"{d}: v0.15 {e} y ahora {o}" for d, e, o in distintos[:5])
        raise AssertionError(f"{que}: {len(distintos)} de {len(pares)} distintos de la v0.15. {muestra}")
    return len(pares)


def _nombre(i, caso):
    return f"mapa {i}" + (f" ({caso['que']})" if "que" in caso else "") + f" inicios {caso['inicios'][:5]}"


def test_el_fichero_trae_cientos_de_casos_de_cada_familia():
    assert len(PLANES) >= 200, len(PLANES)
    for r, m in G.VARIANTES:
        k = G.clave(r, m)
        con_error = sum(1 for c in PLANES if k in c["con_error"])
        assert con_error >= 10 and len(PLANES) - con_error >= 50, (k, con_error)
    assert sum(1 for c in PLANES if "nodo" in c) >= 200
    assert PLANES[0]["que"] == "plano de control" and PLANES[0]["inicios"] == CONTROL["inicios"]
    assert len(ELECCION["elegir"]) >= 150 and len(ELECCION["ventanas"]) >= 100 and len(ELECCION["montar"]) >= 30
    assert len(SUBGRAFO["grafos"]) >= 40 and {g["solo"] for g in SUBGRAFO["grafos"]} == {False, True}
    assert len(FOTOS["tiempos"]) == len(G.FOTOS_FPP) * len(G.FOTOS_ESPERA)
    # el subgrafo del plano de control con lo que envia la tarjeta publicada es el de la v0.14 (mismo prefijo de huella)
    with open(os.path.join(AQUI, "datos", "compat_v014.json"), encoding="utf-8") as f:
        v014 = json.load(f)["tarjeta_publicada"]["grafos"]
    for nombre in ("tarjeta", "prueba"):
        g = next(g for g in SUBGRAFO["grafos"] if g["que"] == "control" and g["enlaces"] == nombre and not g["solo"])
        assert v014[nombre]["sha256"].startswith(g["sha256"]) and v014[nombre]["nodos"] == g["nodos"], nombre


def test_constantes_de_lenta_y_del_subgrafo():
    assert G.constantes_lenta(L, LG) == DATOS["constantes"]["lenta"], "cambio alguna constante de lenta.py o lenta_grafo.py"


def test_planificar_con_ritmo_y_mas_lentos_es_la_v015():
    assert all(list(c["planificar"]) == list(VARIANTE) for c in PLANES)
    pares = [(f"{_nombre(i, c)} ritmo/mas_lentos {k}", h, G.huella_planificar(L, c, *VARIANTE[k]))
             for i, c in enumerate(PLANES) for k, h in c["planificar"].items()]
    _comparar("planificar", pares)
    _comparar("variantes con error", [(_nombre(i, c), c["con_error"], G.variantes_con_error(L, c))
                                      for i, c in enumerate(PLANES)])


def test_nodo_lenta_plan_con_ritmo_y_mas_lentos_es_la_v015():
    pares = []
    for i, c in enumerate(PLANES):
        if "nodo" not in c:
            continue
        for k, h in c["nodo"].items():
            pares.append((f"{_nombre(i, c)} ritmo/mas_lentos {k}", h, G.huella_nodo(L, c, *VARIANTE[k])))
    assert len(pares) >= 200 * len(G.VARIANTES)
    _comparar("CP_LentaPlan (plan, n_ventanas, fotogramas_lentos e info)", pares)


def test_ritmos_y_mas_lentos_que_no_son_de_la_lista():
    obtenido = G.ritmos_raros(L, PLANES[0])
    _comparar("ritmos raros", [(k, v, obtenido.get(k)) for k, v in DATOS["ritmos_raros"].items()])


def test_elegir_indices_es_la_v015():
    casos = ELECCION["elegir"]
    _comparar("recorridos rehechos desde su semilla (entrada, no comportamiento)",
              [(f"eleccion {i}", c["recorrido"], G.huella_recorrido(c)) for i, c in enumerate(casos)])
    _comparar("elegir_indices", [(f"eleccion {i} lentos {c['lentos']} d {c['d']} tm {c['transicion_max']} {c['forma']}",
                                  c["indices"], G.huella_eleccion(L, c)) for i, c in enumerate(casos)])


def _plan(e):
    return G.plan_de(L, PLANES[e["plan"]], *VARIANTE[e["variante"]])


def test_indices_ventana_es_la_v015():
    pares = [(f"plan {e['plan']} {e['variante']}", e["indices"], G._intentar(lambda e=e: G.indices_plan(L, _plan(e), e["plan"])))
             for e in ELECCION["ventanas"]]
    _comparar("indices_ventana", pares)


def test_montar_ventana_es_la_v015():
    pares = [(f"plan {e['plan']} {e['variante']}", e["montado"], G._intentar(lambda e=e: G.montar_plan(L, _plan(e), e["plan"])))
             for e in ELECCION["montar"]]
    _comparar("montar_ventana", pares)
    em = ELECCION["errores_montar"]
    obtenido = G.errores_montar(L, _plan(em))
    _comparar("errores de montar_ventana", [(k, v, obtenido.get(k)) for k, v in em["huellas"].items()])


def test_subgrafo_con_y_sin_solo_animacion_es_el_de_la_v015():
    pares = []
    for g in SUBGRAFO["grafos"]:
        enlaces, ancho, alto = G.ENLACES[g["enlaces"]]
        plan = _plan(g)
        donde = f"{g['que']} {g['enlaces']} solo_animacion={g['solo']}"
        s = G.subgrafo(LG, plan, enlaces, ancho, alto, g["solo"])
        pares.append((donde, [g["sha256"], g["nodos"], g["salidas"]], [s["sha256"], s["nodos"], s["salidas"]]))
        if g["solo"]:
            suave = {k: v for k, v in enlaces.items() if k in LG.ENLACES_SUAVE}
            pares.append((donde + " sin enlaces de render", g["solo_sin_render"],
                          G.subgrafo(LG, plan, suave, ancho, alto, True)["sha256"]))
    _comparar("grafo_lento", pares)
    control = G.plan_de(L, PLANES[0], 1, False)
    nodo = {}
    for nombre, extra in (("sin solo", {}), ("solo False", {"solo_animacion": False}),
                          ("solo True", {"solo_animacion": True}), ("solo None", {"solo_animacion": None})):
        nodo[nombre] = G._intentar(lambda extra=extra: G.nodo_lento(LG, control, G.TARJETA_ENLACES, 1920, 1088, **extra))
    _comparar("CP_BlockingLento.execute", [(k, v, nodo.get(k)) for k, v in SUBGRAFO["nodo"].items()])
    # y la prueba distingue: con y sin solo_animacion el subgrafo no es el mismo
    assert SUBGRAFO["nodo"]["solo True"] != SUBGRAFO["nodo"]["solo False"] == SUBGRAFO["nodo"]["sin solo"]


def test_errores_del_subgrafo_son_los_de_la_v015():
    obtenido = G.generar_subgrafos(L, LG, PLANES)["errores"]
    _comparar("errores de grafo_lento y CP_BlockingLento", [(k, v, obtenido.get(k)) for k, v in SUBGRAFO["errores"].items()])


def test_tiempos_de_las_fotos_son_los_de_la_v015():
    assert G.huella_de([P.CASILLAS, P.FPS, P.MAX_FOTOGRAMAS_POR_POSE, P.MAX_ESPERA_FINAL, P.MARCADOR_VACIO,
                        P.NOMBRES_CASILLAS]) == FOTOS["constantes"], "cambio alguna constante de poses_fotos.py"
    pares = []
    for k, h in FOTOS["tiempos"].items():
        fpp, espera = (int(x) for x in k.split("/"))
        pares.append((f"fotogramas_por_pose {fpp}, espera_final {espera}", h, G.huella_tiempos(P, fpp, espera)))
    _comparar("mapa_bloques y total_fotogramas", pares)


def test_nodo_desde_fotos_es_el_de_la_v015():
    obtenido = G.generar_fotos(P)["nodo"]
    _comparar("CP_PosesDesdeFotos (poses, mapa, n_poses, transicion, fotogramas, fps)",
              [(k, v, obtenido.get(k)) for k, v in FOTOS["nodo"].items()])


def recuento():
    planes = sum(len(c["planificar"]) for c in PLANES)
    nodo = sum(len(c.get("nodo", {})) for c in PLANES)
    return (f"{len(PLANES)} mapas ({planes} planes de planificar y {nodo} del nodo), {len(DATOS['ritmos_raros'])} ritmos "
            f"raros, {len(ELECCION['elegir'])} elecciones, {len(ELECCION['ventanas'])} planes elegidos por ventana, "
            f"{len(ELECCION['montar'])} montados, {len(SUBGRAFO['grafos'])} subgrafos, {len(FOTOS['tiempos'])} tiempos "
            f"de fotos y {len(FOTOS['nodo'])} del nodo de fotos")


if __name__ == "__main__":
    pruebas = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    fallos = []
    for t in pruebas:
        try:
            t()
        except Exception as e:  # se siguen las demas: asi se ve de una vez todo lo que falla
            fallos.append(t.__name__)
            print("FALLA", t.__name__, repr(e)[:600])
            continue
        print("ok", t.__name__)
    print(len(pruebas), "pruebas,", recuento(), "de la v0.15", f"y {len(fallos)} fallan: {fallos}" if fallos else "")
    sys.exit(1 if fallos else 0)
