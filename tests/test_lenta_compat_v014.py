# -*- coding: utf-8 -*-
"""Compatibilidad dura con la v0.14: sin "ritmo" ni "mas_lentos" (o con ritmo 1 y mas_lentos False), CP_LentaPlan da
el mismo plan byte a byte, o el mismo error con el mismo texto, que la v0.14 para cada caso de
tests/datos/compat_v014.json (planes generados con la v0.14 de verdad, el commit 917bb3d).

Asi un grafo guardado con la v0.14 (lo que envia una tarjeta publicada) se ejecuta igual. El fichero se regenera con
tools/generar_compat_v014.py (lo explica su cabecera); no hace falta ComfyUI ni git para esta prueba.
Se ejecuta con: python tests/test_lenta_compat_v014.py"""
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
        self.args = a


class _IO:
    Schema = staticmethod(lambda **k: k)
    NodeOutput = _Salida
    ComfyNode = type("ComfyNode", (), {})


for _n in ("Image", "Float", "Int", "String", "Boolean"):
    setattr(_IO, _n, type(_n, (), {"Input": staticmethod(_Port), "Output": staticmethod(_Port)}))

latest.io = _IO
mod.latest = latest
sys.modules["comfy_api"] = mod
sys.modules["comfy_api.latest"] = latest

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.normpath(os.path.join(AQUI, ".."))
sys.path.insert(0, os.path.join(RAIZ, "tools"))
import generar_compat_v014 as G  # noqa: E402

L, LG = G.cargar_pack(RAIZ, "pack_actual")

with open(os.path.join(AQUI, "datos", "compat_v014.json"), encoding="utf-8") as _f:
    DATOS = json.load(_f)
CASOS = DATOS["casos"]


def _igual(esperado, obtenido, caso, donde):
    if "error" in esperado:
        assert obtenido.get("error") == esperado["error"], (donde, caso["inicios"][:6], esperado["error"], obtenido)
        return
    assert "plan" in obtenido, (donde, caso["inicios"][:6], obtenido.get("error"))
    huella = G.huella(obtenido["plan"])
    if huella != esperado["plan"]["sha256"]:
        plan = json.loads(obtenido["plan"])
        raise AssertionError(
            f"{donde}: plan distinto del de la v0.14 (inicios {caso['inicios'][:6]}...): lentos esperados "
            f"{esperado['plan'].get('lentos')} y salen {plan['lentos']}; claves {sorted(plan)}"
        )


def test_el_fichero_trae_cientos_de_casos_con_errores_y_planes():
    assert len(CASOS) >= 300, len(CASOS)
    errores = sum(1 for c in CASOS if "error" in c["esperado"])
    assert 10 <= errores < len(CASOS) - 200, errores
    assert sum(1 for c in CASOS if "esperado_nodo" in c) >= 300
    # los casos de los revisores estan: 12 poses cada 10 fotogramas (16 lentos) y 1200 fotogramas (sin error)
    a = CASOS[0]
    assert a["inicios"] == [i * 10 for i in range(12)] and set(a["esperado"]["plan"]["lentos"]) == {16}
    b = CASOS[1]
    assert b["total"] == 1200 and "plan" in b["esperado"]


def test_planificar_sin_ritmo_ni_mas_lentos_es_la_v014():
    for caso in CASOS:
        _igual(caso["esperado"], G.resultado_plan(L, caso), caso, "planificar")


def test_planificar_con_ritmo_1_y_mas_lentos_false_es_la_v014():
    for caso in CASOS:
        try:
            plan = L.planificar(caso["inicios"], caso["total"], G._saltos(caso), caso["umbral"], caso["max_ventana"],
                                caso["transicion_max"], caso["max_total"], ritmo=1, mas_lentos=False)
            obtenido = {"plan": json.dumps(plan)}
        except ValueError as e:
            obtenido = {"error": str(e)}
        _igual(caso["esperado"], obtenido, caso, "planificar ritmo=1 mas_lentos=False")


def test_nodo_sin_ritmo_ni_mas_lentos_es_la_v014():
    vistos = 0
    for caso in CASOS:
        if "esperado_nodo" not in caso:
            continue
        esperado, obtenido = caso["esperado_nodo"], G.resultado_nodo(L, caso)
        _igual(esperado, obtenido, caso, "CP_LentaPlan")
        if "plan" in esperado:
            assert (obtenido["n_ventanas"], obtenido["fotogramas_lentos"]) == (esperado["n_ventanas"], esperado["fotogramas_lentos"])
        vistos += 1
    assert vistos >= 300


def test_nodo_con_ritmo_1_y_mas_lentos_false_explicitos_es_la_v014():
    medir = L.medir_saltos
    try:
        for caso in CASOS:
            if "esperado_nodo" not in caso:
                continue
            L.medir_saltos = lambda poses, nodo="", c=caso: G._saltos(c)
            try:
                s = L.LentaPlan.execute(G._Poses(len(caso["inicios"])), G.mapa_de(caso), caso["umbral"], caso["max_ventana"],
                                        caso["transicion_max"], caso["max_total"], 1, False).args
                obtenido = {"plan": s[0], "n_ventanas": s[1], "fotogramas_lentos": s[2]}
            except ValueError as e:
                obtenido = {"error": str(e)}
            _igual(caso["esperado_nodo"], obtenido, caso, "CP_LentaPlan ritmo=1 mas_lentos=False")
    finally:
        L.medir_saltos = medir


def test_tarjeta_publicada_plan_y_subgrafo_de_la_v014():
    """El plano de control con lo que envia la tarjeta publicada (sin ritmo, mas_lentos ni solo_animacion): el plan y el
    subgrafo de CP_BlockingLento son, byte a byte, los de la v0.14."""
    t = DATOS["tarjeta_publicada"]
    with open(os.path.join(AQUI, "datos", "lenta_0003.json"), encoding="utf-8") as f:
        control = json.load(f)
    e = t["plan_entradas"]
    plan = L.planificar(control["inicios"], control["bloques"][-1][1] + 1, control["saltos_dis_480"], e["umbral"],
                        e["max_ventana"], e["transicion_max"], e["max_total"])
    assert G.huella(json.dumps(plan)) == t["plan"]["sha256"], plan["lentos"]
    assert plan["lentos"] == t["plan"]["lentos"] and "ritmo" not in plan and "mas_lentos" not in plan
    for nombre, g in t["grafos"].items():
        for extra in ({}, {"solo_animacion": False}):
            grafo, animacion, playblast = LG.grafo_lento(plan, g["enlaces"], g["ancho"], g["alto"], G.PREFIJO, **extra)
            assert len(grafo) == g["nodos"] and [animacion, playblast] == g["salidas"], (nombre, len(grafo))
            assert G.huella_grafo(grafo) == g["sha256"], (nombre, extra, "subgrafo distinto del de la v0.14")
    # y con solo_animacion True ya no es el mismo (la prueba sabe ver la diferencia)
    g = t["grafos"]["tarjeta"]
    grafo, _, _ = LG.grafo_lento(plan, g["enlaces"], g["ancho"], g["alto"], G.PREFIJO, solo_animacion=True)
    assert G.huella_grafo(grafo) != g["sha256"]


if __name__ == "__main__":
    pruebas = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in pruebas:
        t()
        print("ok", t.__name__)
    print(len(pruebas), "pruebas,", len(CASOS), "casos de la v0.14")
