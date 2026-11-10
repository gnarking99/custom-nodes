# -*- coding: utf-8 -*-
"""Prueba el plan y la eleccion de fotogramas de la camara lenta, sin ComfyUI.
Los datos de control (tests/datos/lenta_0003.json) son de un plano real de 77 fotogramas y 25 poses.
Se ejecuta con: python tests/test_lenta.py"""
import json
import os
import sys
import types

import numpy as np
import torch

# ---- stub minimo de comfy_api.latest ----
mod = types.ModuleType("comfy_api")
latest = types.ModuleType("comfy_api.latest")


class _Port:
    def __init__(self, *a, **k):
        self.a, self.k = a, k


def _mk(name):
    return type(name, (), {"Input": staticmethod(_Port), "Output": staticmethod(_Port)})


class _Salida:
    def __init__(self, *a, **k):
        self.args = a


class _IO:
    Schema = staticmethod(lambda **k: k)
    NodeOutput = _Salida
    ComfyNode = type("ComfyNode", (), {})


for _n in ("Image", "Float", "Int", "String", "Boolean"):
    setattr(_IO, _n, _mk(_n))

latest.io = _IO
mod.latest = latest
sys.modules["comfy_api"] = mod
sys.modules["comfy_api.latest"] = latest

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(AQUI, "..", "comfy_character_pipeline"))
import lenta as L  # noqa: E402

with open(os.path.join(AQUI, "datos", "lenta_0003.json"), encoding="utf-8") as _f:
    DATOS = json.load(_f)
INICIOS = DATOS["inicios"]
LENTOS_V8 = DATOS["lentos_v8"]


def _falla(fn, *trozos, error=ValueError):
    """La llamada tiene que fallar con ese error y con esos trozos en el mensaje."""
    try:
        fn()
    except error as e:
        for t in trozos:
            assert t in str(e), (t, str(e))
        return str(e)
    raise AssertionError(f"tenia que fallar con {error.__name__} {trozos}")


def _hay_cv2(prueba):
    try:
        import cv2  # noqa: F401
    except ImportError:
        print(f"AVISO: no hay cv2, se salta {prueba}")
        return False
    return True


def _plan_v8():
    """El plan de la version validada, con sus ventanas y sus lentos puestos A MANO (no salen del repartidor)."""
    ventanas = []
    for p0, p1 in DATOS["ventanas_v8"]:
        pos = [0]
        for k in range(p0, p1):
            pos.append(pos[-1] + LENTOS_V8[k])
        ventanas.append({"p0": p0, "p1": p1, "pos": pos, "n": pos[-1] + 1})
    plan = {"version": 1, "inicios": INICIOS, "total": 77, "saltos_px": DATOS["saltos_dis_480"], "lentos": LENTOS_V8,
            "transicion_max": 12, "ventanas": ventanas}
    return L.leer_plan(json.dumps(plan))


def _referencia(m, lentos_ventana, d_ventana):
    """El guion con el que se valido la receta, con la curva lineal: copia literal de su bucle."""
    m = np.asarray(m, np.float64)
    salida, base = [], 0
    for lentos, d in zip(lentos_ventana, d_ventana):
        s = np.concatenate([[0.0], np.cumsum(m[base:base + lentos])])
        total = max(s[-1], 1e-6)
        idx = [0]
        for i in range(1, d):
            obj = i / d * total
            c = int(np.searchsorted(s, obj))
            c = c if c >= lentos or abs(s[c] - obj) <= abs(s[c - 1] - obj) else c - 1
            idx.append(min(max(c, idx[-1] + 1), lentos - (d - i)))
        salida += [base + c for c in idx]
        base += lentos
    return salida


# ---------------------------------------------------------------- dorada

def test_dorada_elegir_reproduce_los_77_fotogramas_validados():
    assert DATOS["bloques"][-1][1] + 1 == 77 and [b[0] for b in DATOS["bloques"]] == INICIOS
    plan = _plan_v8()
    assert [v["n"] for v in plan["ventanas"]] == [177, 161, 145, 65]
    salida = []
    for w, v in enumerate(plan["ventanas"]):
        p0, p1 = v["p0"], v["p1"]
        medido = DATOS["recorrido_por_ventana"][w]
        assert len(medido) == v["n"] - 1
        d = [INICIOS[k + 1] - INICIOS[k] for k in range(p0, p1)]
        idx = L.elegir_indices(medido, LENTOS_V8[p0:p1], d, plan["transicion_max"])
        assert len(salida) == INICIOS[p0] and len(idx) == INICIOS[p1] - INICIOS[p0]
        # cada pose de la ventana cae en su fotograma del blocking
        for j, k in enumerate(range(p0, p1)):
            assert idx[INICIOS[k] - INICIOS[p0]] == v["pos"][j], (w, k)
        completos = L.indices_ventana(plan, w, medido)
        ultima = w == len(plan["ventanas"]) - 1
        assert completos == idx + ([v["n"] - 1] if ultima else []), w
        salida += [[f"w{w}", i] for i in completos]
    assert len(salida) == 77
    distintos = [(f, a, b) for f, (a, b) in enumerate(zip(salida, DATOS["elegidos_v8"])) if a != b]
    assert not distintos, distintos
    assert salida[-1] == ["w3", 64]  # la pose final solo sale de la ultima ventana


def test_dorada_no_cambia_con_transicion_max_si_los_saltos_caben():
    # El salto mayor del plano es de 5 fotogramas: con cualquier transicion_max >= 5 sale lo mismo.
    for w, (p0, p1) in enumerate(DATOS["ventanas_v8"]):
        d = [INICIOS[k + 1] - INICIOS[k] for k in range(p0, p1)]
        medido = DATOS["recorrido_por_ventana"][w]
        base = L.elegir_indices(medido, LENTOS_V8[p0:p1], d, 12)
        assert base == _referencia(medido, LENTOS_V8[p0:p1], d)
        assert L.elegir_indices(medido, LENTOS_V8[p0:p1], d, 5) == base
        assert L.elegir_indices(medido, LENTOS_V8[p0:p1], d, 64) == base


# ---------------------------------------------------------------- reparto en ventanas

def _tamanos(lentos, cortes):
    return [sum(lentos[a:b]) + 1 for a, b in cortes]


def _reparto_a_lo_bruto(lentos, tope):
    """Todas las particiones: menos ventanas, despues menor suma de n^2, despues cortes mas tempranos."""
    n, mejor = len(lentos), None
    for mascara in range(2 ** (n - 1)):
        cortes, a = [], 0
        for k in range(1, n):
            if mascara >> (k - 1) & 1:
                cortes.append((a, k))
                a = k
        cortes.append((a, n))
        tam = _tamanos(lentos, cortes)
        if any(t > tope and b - a > 1 for t, (a, b) in zip(tam, cortes)):
            continue
        clave = (len(cortes), sum(t * t for t in tam), cortes)
        if mejor is None or clave < mejor:
            mejor = clave
    return mejor[2]


def test_reparto_del_plano_de_control():
    cortes = L.repartir_ventanas(LENTOS_V8, 177)
    assert cortes == [(0, 5), (5, 11), (11, 16), (16, 24)], cortes
    assert _tamanos(LENTOS_V8, cortes) == [145, 145, 129, 129]
    # Las ventanas a mano de la version validada caben en el tope, pero no son el reparto mas parejo.
    assert max(_tamanos(LENTOS_V8, [tuple(v) for v in DATOS["ventanas_v8"]])) == 177


def test_reparto_casos_limite():
    assert L.repartir_ventanas([16], 177) == [(0, 1)]
    assert L.repartir_ventanas([64], 33) == [(0, 1)]  # un salto solo que no cabe: su propia ventana
    assert L.repartir_ventanas([16, 16, 32], 177) == [(0, 3)]  # todo cabe en una
    assert L.repartir_ventanas([16, 64, 16], 33) == [(0, 1), (1, 2), (2, 3)]
    assert L.repartir_ventanas([16, 16, 64, 16, 16], 33) == [(0, 2), (2, 3), (3, 5)]
    assert L.repartir_ventanas([32] * 5, 161) == [(0, 5)]  # n = 161 justo en el tope
    assert L.repartir_ventanas([32] * 5, 160) == [(0, 2), (2, 5)]  # empate 65+97 / 97+65: el corte mas temprano
    assert L.repartir_ventanas([16] * 6, 49) == [(0, 3), (3, 6)]
    _falla(lambda: L.repartir_ventanas([], 177), "No hay saltos")


def test_reparto_coincide_con_probar_todas_las_particiones():
    rng = np.random.default_rng(7)
    for _ in range(400):
        lentos = [int(x) for x in rng.choice([16, 24, 32, 40, 64], size=int(rng.integers(1, 11)))]
        tope = int(rng.choice([33, 49, 65, 97, 129, 177]))
        cortes = L.repartir_ventanas(lentos, tope)
        assert cortes == _reparto_a_lo_bruto(lentos, tope), (lentos, tope, cortes)
        assert cortes[0][0] == 0 and cortes[-1][1] == len(lentos)
        assert all(a[1] == b[0] for a, b in zip(cortes, cortes[1:]))


# ---------------------------------------------------------------- plan

def test_plan_lentos_16_o_32_segun_el_umbral():
    inicios, total = [0, 4, 8, 12, 16], 17
    plan = L.planificar(inicios, total, [8.999, 9.0, 30.0, 0.0])
    assert plan["lentos"] == [16, 32, 32, 16]  # 9.0 justo ya es salto grande
    assert plan["saltos_px"] == [8.999, 9.0, 30.0, 0.0] and plan["inicios"] == inicios and plan["total"] == total
    assert plan["version"] == 1 and plan["transicion_max"] == 12
    assert plan["ventanas"] == [{"p0": 0, "p1": 4, "pos": [0, 16, 48, 80, 96], "n": 97}]
    assert L.planificar(inicios, total, [8.999, 9.0, 30.0, 0.0], umbral=31.0)["lentos"] == [16] * 4
    assert L.planificar(inicios, total, [8.999, 9.0, 30.0, 0.0], umbral=0.0)["lentos"] == [32] * 4


def test_plan_del_plano_de_control_con_las_medidas_guardadas():
    plan = L.planificar(INICIOS, 77, DATOS["saltos_dis_480"])
    assert plan["lentos"] == [32 if x >= 9.0 else 16 for x in DATOS["saltos_dis_480"]]
    assert all(v["n"] <= 177 and v["n"] % 8 == 1 for v in plan["ventanas"])
    assert sum(v["n"] for v in plan["ventanas"]) == sum(plan["lentos"]) + len(plan["ventanas"])
    info = L.resumir_plan(plan)
    assert info.startswith("25 poses en 77 fotogramas") and "AVISO" not in info and "\n" not in info, info


def test_plan_sube_los_lentos_al_multiplo_de_8_que_hace_falta():
    inicios, total = [0, 20, 60, 61], 62
    # transicion_max 40: de los saltos hay que sacar 20, 40 y 1 fotogramas
    plan = L.planificar(inicios, total, [1.0, 50.0, 1.0], transicion_max=40)
    assert plan["lentos"] == [24, 40, 16], plan["lentos"]
    # con el valor por defecto se sacan 12, 12 y 1: mandan el 16 y el 32
    assert L.planificar(inicios, total, [1.0, 50.0, 1.0])["lentos"] == [16, 32, 16]
    assert L.lentos_por_salto([1.0, 1.0, 1.0, 50.0], [16, 17, 33, 33], 9.0, 64) == [16, 24, 40, 40]


def test_plan_espera_larga_se_queda_quieta_y_el_salto_ocupa_el_final():
    plan = L.planificar([0, 40], 41, [2.0])
    assert plan["lentos"] == [16] and plan["transicion_max"] == 12  # de 40 fotogramas solo 12 recorren el salto
    idx = L.elegir_indices(np.ones(16), [16], [40], plan["transicion_max"])
    assert len(idx) == 40 and idx[:28] == [0] * 28, idx  # 28 quietos
    assert idx[28:] == [0, 1, 3, 4, 5, 7, 8, 9, 11, 12, 13, 15], idx[28:]  # y 12 que se mueven, desde la pose
    assert "Esperas de mas de 12 fotogramas: 1" in L.resumir_plan(plan)
    # con transicion_max 40 el salto ocupa los 40 fotogramas y hacen falta 40 lentos
    assert L.planificar([0, 40], 41, [2.0], transicion_max=40)["lentos"] == [40]


def test_plan_tope_de_seguridad_max_total():
    inicios = list(range(0, 120, 4))  # 30 poses, 29 saltos grandes: 29 x 32 + 6 ventanas = 934
    saltos = [20.0] * 29
    plan = L.planificar(inicios, 117, saltos, max_total=934)
    assert sum(v["n"] for v in plan["ventanas"]) == 934 and len(plan["ventanas"]) == 6
    _falla(lambda: L.planificar(inicios, 117, saltos, max_total=933), "[CP_LentaPlan]", "934", "933", "demasiado largo", "planos mas cortos")


def test_plan_una_sola_pose_es_un_error_claro():
    _falla(lambda: L.planificar([0], 12, []), "[CP_LentaPlan]", "una sola pose", "nada que animar")
    # y desde el nodo, sin llegar a medir nada
    poses = torch.rand(1, 8, 12, 3)
    _falla(lambda: L.LentaPlan.execute(poses, json.dumps({"bloques": [[0, 11]]})), "[CP_LentaPlan]", "una sola pose")


def test_plan_entradas_que_no_cuadran():
    _falla(lambda: L.planificar([0, 4, 8], 9, [1.0]), "2 saltos", "llegaron 1")
    _falla(lambda: L.planificar([0, 4, 4], 9, [1.0, 1.0]), "crecientes")
    _falla(lambda: L.planificar([2, 4, 8], 9, [1.0, 1.0]), "empezar en 0")
    _falla(lambda: L.planificar([0, 4, 8], 8, [1.0, 1.0]), "8 fotogramas")
    _falla(lambda: L.planificar([0, 4, 8], 9, [1.0, float("nan")]), "no es un numero")


def test_plan_pasa_por_json_y_leer_plan_lo_acepta():
    plan = L.planificar(INICIOS, 77, DATOS["saltos_dis_480"])
    texto = json.dumps(plan)
    leido = L.leer_plan(texto)
    assert leido == plan and L.leer_plan(leido) == plan
    assert set(plan) == {"version", "inicios", "total", "saltos_px", "lentos", "transicion_max", "ventanas"}
    assert len(plan["lentos"]) == len(plan["saltos_px"]) == len(INICIOS) - 1
    for v in plan["ventanas"]:
        assert set(v) == {"p0", "p1", "pos", "n"}
        assert v["n"] == v["pos"][-1] + 1 and v["n"] % 8 == 1 and all(x % 8 == 0 for x in v["pos"])
        assert len(v["pos"]) == v["p1"] - v["p0"] + 1


def test_plan_con_un_salto_que_no_cabe_avisa():
    plan = L.planificar([0, 4, 68, 72], 73, [1.0, 30.0, 1.0], max_ventana=33, transicion_max=64)
    assert plan["lentos"] == [16, 64, 16]
    assert [(v["p0"], v["p1"], v["n"]) for v in plan["ventanas"]] == [(0, 1, 17), (1, 2, 65), (2, 3, 17)]
    info = L.resumir_plan(plan, 33)
    assert "AVISO: la ventana 1 mide 65" in info and "tope es 33" in info and info.count("AVISO") == 1, info
    assert "AVISO" not in L.resumir_plan(plan, 65)


def _plan_chico():
    return L.armar_plan([0, 3, 5, 9], 12, [1.0, 2.0, 3.0], [16, 16, 16], [(0, 2), (2, 3)], 12)


def test_leer_plan_valida_version_longitudes_y_posiciones():
    bueno = _plan_chico()
    assert L.leer_plan(json.dumps(bueno)) == bueno
    assert bueno["ventanas"] == [{"p0": 0, "p1": 2, "pos": [0, 16, 32], "n": 33}, {"p0": 2, "p1": 3, "pos": [0, 16], "n": 17}]

    def roto(cambio):
        plan = json.loads(json.dumps(bueno))
        cambio(plan)
        return lambda: L.leer_plan(json.dumps(plan))

    _falla(lambda: L.leer_plan("esto no es json"), "[CP_LentaElegir]", "no se puede leer")
    _falla(lambda: L.leer_plan("[1, 2]"), "no se puede leer")
    _falla(lambda: L.leer_plan(None), "no se puede leer")
    _falla(roto(lambda p: p.update(version=2)), "version 2", "CP_LentaPlan")
    _falla(roto(lambda p: p.pop("version")), "version")
    _falla(roto(lambda p: p.pop("lentos")), "le falta", "lentos")
    _falla(roto(lambda p: p.pop("transicion_max")), "le falta", "transicion_max")
    _falla(roto(lambda p: p["lentos"].append(16)), "no cuadra", "4 lentos")
    _falla(roto(lambda p: p["saltos_px"].pop()), "no cuadra", "2 medidas")
    _falla(roto(lambda p: p["lentos"].__setitem__(0, 12)), "salto 0", "multiplo de 8")
    _falla(roto(lambda p: p["lentos"].__setitem__(0, 16.5)), "no es un entero")
    _falla(roto(lambda p: p["ventanas"][0].update(pos=[0, 12, 32])), "ventana 0", "multiplo de 8")
    _falla(roto(lambda p: p["ventanas"][0].update(n=34)), "ventana 0", "tocan 33")
    _falla(roto(lambda p: p["ventanas"][0].update(pos=[0, 16, 36], n=37)), "ventana 0", "8k+1")
    _falla(roto(lambda p: p["ventanas"][0].update(pos=[0, 8, 32])), "ventana 0", "no salen de los lentos")
    _falla(roto(lambda p: p["ventanas"][1].update(pos=[0, 8, 16])), "ventana 1", "3 posiciones")
    _falla(roto(lambda p: p["ventanas"][1].update(p0=1)), "no cubren")
    _falla(roto(lambda p: p["ventanas"].pop()), "no cubren")
    _falla(roto(lambda p: p.update(ventanas=[])), "no cubren")
    _falla(roto(lambda p: p.update(total=9)), "9 fotogramas")
    _falla(roto(lambda p: p.update(inicios=[0, 5, 3, 9])), "crecientes")
    _falla(roto(lambda p: p.update(inicios=[0, 3, 5, 45], total=46, transicion_max=40)), "salto 2", "sacar 40")
    _falla(roto(lambda p: p.update(transicion_max=0)), "transicion_max")
    _falla(roto(lambda p: p.update(inicios=[0], lentos=[], saltos_px=[])), "una sola pose")


def test_leer_mapa_de_poses():
    mapa = json.dumps({"bloques": DATOS["bloques"]})
    assert L.leer_mapa(mapa, 25) == (INICIOS, 77)
    assert L.leer_mapa(json.dumps({"bloques": [[0, 3], [4, 11]]}), 2) == ([0, 4], 12)  # la ultima pose dura 8
    _falla(lambda: L.leer_mapa(mapa, 24), "[CP_LentaPlan]", "24 poses", "25")
    _falla(lambda: L.leer_mapa("{}", 2), "no se puede leer")
    _falla(lambda: L.leer_mapa("nada", 2), "no se puede leer")
    _falla(lambda: L.leer_mapa(json.dumps({"bloques": []}), 0), "vacio")
    _falla(lambda: L.leer_mapa(json.dumps({"bloques": [[1, 3], [4, 7]]}), 2), "desde el fotograma 0")
    _falla(lambda: L.leer_mapa(json.dumps({"bloques": [[0, 3], [5, 7]]}), 2), "cadencia seguida")


# ---------------------------------------------------------------- elegir

def test_elegir_movimiento_uniforme_es_el_reparto_por_indice():
    assert L.elegir_indices(np.ones(64), [32, 32], [4, 2], 12) == [0, 8, 16, 24, 32, 48]
    assert L.elegir_indices(np.full(16, 0.37), [16], [1], 12) == [0]  # salto de un fotograma: solo la pose


def test_elegir_reparte_por_recorrido_y_no_por_indice():
    # Todo el movimiento al principio del salto: por indice saldria [0, 8, 16, 24], con un tiron y un paron.
    cargado = np.r_[np.full(8, 4.0), np.full(24, 0.5)]
    assert L.elegir_indices(cargado, [32], [4], 12) == [0, 3, 6, 10]
    # y al final (el objetivo 22 queda justo entre dos lentos, a 20 y a 24: en el empate se toma el de delante)
    assert L.elegir_indices(cargado[::-1], [32], [4], 12) == [0, 22, 27, 29]
    assert _referencia(cargado, [32], [4]) == [0, 3, 6, 10] and _referencia(cargado[::-1], [32], [4]) == [0, 22, 27, 29]


def test_elegir_recorrido_todo_ceros_reparte_uniforme_por_indice():
    assert L.elegir_indices(np.zeros(16), [16], [4], 12) == [0, 4, 8, 12]
    assert L.elegir_indices(np.zeros(32), [32], [5], 12) == [0, 6, 13, 19, 26]
    assert L.elegir_indices(np.zeros(16), [16], [16], 16) == list(range(16))
    # un salto quieto entre dos que se mueven: cada uno con lo suyo
    medido = np.r_[np.ones(16), np.zeros(16), np.ones(16)]
    assert L.elegir_indices(medido, [16, 16, 16], [2, 4, 2], 12) == [0, 8, 16, 20, 24, 28, 32, 40]


def test_elegir_indices_siempre_crecientes_y_dentro_del_salto():
    rng = np.random.default_rng(11)
    for caso in range(600):
        saltos = int(rng.integers(1, 6))
        transicion = int(rng.choice([1, 2, 5, 12, 40]))
        d = [int(rng.integers(1, 50)) for _ in range(saltos)]
        lentos = [int(max(rng.choice([16, 32]), 8 * -(-min(x, transicion) // 8))) for x in d]
        forma = caso % 4
        if forma == 0:
            medido = rng.random(sum(lentos)) * 5.0
        elif forma == 1:  # casi todo quieto y un tiron
            medido = np.zeros(sum(lentos))
            medido[rng.integers(0, len(medido))] = 30.0
        elif forma == 2:
            medido = np.zeros(sum(lentos))
        else:  # todo el movimiento en el ultimo paso de cada salto
            medido = np.zeros(sum(lentos))
            medido[np.cumsum(lentos) - 1] = 3.0
        idx = L.elegir_indices(medido, lentos, d, transicion)
        assert len(idx) == sum(d)
        base, f = 0, 0
        for lento, dur in zip(lentos, d):
            m = min(dur, transicion)
            tramo = idx[f:f + dur]
            assert tramo[:dur - m + 1] == [base] * (dur - m + 1), (caso, tramo)  # la espera y la pose
            mueve = tramo[dur - m:]
            assert all(b > a for a, b in zip(mueve, mueve[1:])), (caso, mueve)
            assert base <= min(tramo) and max(tramo) <= base + lento - 1, (caso, tramo, base, lento)
            base, f = base + lento, f + dur
        if forma in (0, 1, 3) and max(d) <= transicion and all(
                medido[a:b].sum() > 1e-6 for a, b in zip(np.r_[0, np.cumsum(lentos)[:-1]], np.cumsum(lentos))):
            assert idx == _referencia(medido, lentos, d), caso  # sin esperas largas es el guion validado


def test_elegir_rechaza_lo_que_no_cuadra():
    _falla(lambda: L.elegir_indices(np.ones(15), [16], [4], 12), "[CP_LentaElegir]", "15 valores", "16 pasos")
    _falla(lambda: L.elegir_indices(np.ones(16), [16], [4, 2], 12), "1 saltos", "2 con duracion")
    _falla(lambda: L.elegir_indices(np.ones(16), [16], [20], 20), "no se pueden sacar 20")
    _falla(lambda: L.elegir_indices(np.ones(16), [16], [0], 12), "no se pueden sacar 0")
    _falla(lambda: L.elegir_indices(-np.ones(16), [16], [4], 12), "negativos")
    _falla(lambda: L.elegir_indices(np.full(16, np.nan), [16], [4], 12), "no son numeros")


# ---------------------------------------------------------------- montaje

def _lote(n, alto, ancho, desfase=0.0):
    """Cada fotograma lleva su indice en el valor: se ve de que lento sale cada fotograma elegido."""
    return (torch.arange(n, dtype=torch.float32).reshape(n, 1, 1, 1) / 1000.0 + desfase).repeat(1, alto, ancho, 3)


def _propio(lote):
    """El tensor ocupa solo lo suyo: no es una vista de un lote mayor, que seguiria vivo en la cache."""
    return lote.untyped_storage().nbytes() == lote.numel() * lote.element_size()


def _de_donde(lote, desfase=0.0):
    assert float((lote - lote[:, :1, :1, :1]).abs().max()) < 1e-6  # cada fotograma sigue siendo liso
    return [int(round((float(x) - desfase) * 1000.0)) for x in lote[:, 0, 0, 0]]


def test_montaje_de_dos_ventanas_encadenadas():
    plan = L.leer_plan(_plan_chico())  # poses en 0, 3, 5 y 9; el blocking tiene 12: la ultima pose dura 3
    guia0, render0 = _lote(33, 8, 12), _lote(33, 16, 24)
    guia1, render1 = _lote(17, 8, 12, 0.5), _lote(17, 16, 24, 0.5)

    r0, g0 = L.montar_ventana(guia0, render0, plan, 0, medido=np.ones(32))
    assert r0.shape == (5, 16, 24, 3) and g0.shape == (5, 16, 24, 3)  # la guia sale al tamano del render
    assert r0.is_contiguous() and g0.is_contiguous()
    assert _de_donde(r0) == [0, 5, 11, 16, 24] and _de_donde(g0) == [0, 5, 11, 16, 24]
    assert 32 not in _de_donde(r0)  # la pose final de la ventana 0 es el lento 0 de la ventana 1

    r1, g1 = L.montar_ventana(guia1, render1, plan, 1, r0, g0, medido=np.ones(16))
    assert r1.shape == (12, 16, 24, 3) and g1.shape == (12, 16, 24, 3)  # el total del blocking
    assert torch.equal(r1[:5], r0) and torch.equal(g1[:5], g0)
    # la pose final (lento 16) solo al final, y repetida porque el ultimo bloque dura 3 fotogramas
    assert _de_donde(r1[5:], 0.5) == [0, 4, 8, 12, 16, 16, 16] and _de_donde(g1[5:], 0.5) == [0, 4, 8, 12, 16, 16, 16]

    # nada comparte memoria con los lotes lentos ni con lo acumulado
    for salida in (r0, g0, r1, g1):
        for entrada in (guia0, render0, guia1, render1):
            assert not np.shares_memory(salida.numpy(), entrada.numpy())
    assert not np.shares_memory(r1.numpy(), r0.numpy()) and not np.shares_memory(g1.numpy(), g0.numpy())
    assert all(_propio(t) for t in (r0, g0, r1, g1))
    antes = r1.clone()
    render0.zero_(), render1.zero_(), r0.zero_()
    assert torch.equal(r1, antes)


def test_montaje_sin_relleno_y_con_la_guia_al_tamano_del_render():
    # El blocking acaba en la ultima pose: un solo fotograma final. La guia ya mide lo que el render: se copia igual.
    plan = L.leer_plan(L.armar_plan([0, 3, 5, 9], 10, [1.0, 2.0, 3.0], [16, 16, 16], [(0, 3)], 12))
    guia, render = _lote(49, 6, 10, 0.2), _lote(49, 6, 10)
    r, g = L.montar_ventana(guia, render, plan, 0, medido=np.ones(48))
    assert len(r) == 10 and _de_donde(r) == [0, 5, 11, 16, 24, 32, 36, 40, 44, 48]
    assert _de_donde(g, 0.2) == _de_donde(r) and g.shape == (10, 6, 10, 3)
    assert not np.shares_memory(r.numpy(), render.numpy()) and not np.shares_memory(g.numpy(), guia.numpy())
    assert _propio(r) and _propio(g) and not _propio(render[:10])


def test_montaje_rechaza_lotes_que_no_son_de_la_ventana():
    plan = L.leer_plan(_plan_chico())
    guia0, render0 = _lote(33, 4, 6), _lote(33, 4, 6)
    unos = np.ones(32)
    _falla(lambda: L.montar_ventana(guia0[:32], render0, plan, 0, medido=unos), "[CP_LentaElegir]", "33 fotogramas lentos", "la guia trae 32")
    _falla(lambda: L.montar_ventana(guia0, render0[:17], plan, 0, medido=unos), "33 fotogramas lentos", "el render trae 17")
    _falla(lambda: L.montar_ventana(guia0, None, plan, 0, medido=unos), "el render trae nada")
    _falla(lambda: L.montar_ventana(guia0, render0, plan, 2, medido=unos), "ventana 2 no existe", "tiene 2")
    _falla(lambda: L.montar_ventana(guia0, render0, plan, -1, medido=unos), "no existe")
    r0, g0 = L.montar_ventana(guia0, render0, plan, 0, medido=unos)
    guia1, render1 = _lote(17, 4, 6), _lote(17, 4, 6)
    unos = np.ones(16)
    # la ventana 1 necesita lo acumulado de la 0, entero y del mismo tamano
    _falla(lambda: L.montar_ventana(guia1, render1, plan, 1, medido=unos), "van 5 fotogramas", "anterior_render trae 0")
    _falla(lambda: L.montar_ventana(guia1, render1, plan, 1, r0, None, medido=unos), "anterior_guia trae 0")
    _falla(lambda: L.montar_ventana(guia1, render1, plan, 1, r0[:4], g0, medido=unos), "anterior_render trae 4")
    _falla(lambda: L.montar_ventana(guia0, render0, plan, 0, r0, g0, medido=np.ones(32)), "van 0 fotogramas", "trae 5", "es la primera")
    _falla(lambda: L.montar_ventana(guia1, render1, plan, 1, r0[:, :2], g0, medido=unos), "anterior_render mide 6x2", "6x4", "mismo tamano")
    _falla(lambda: L.montar_ventana(guia1, render1, plan, 1, r0, g0[:, :, :3], medido=unos), "anterior_guia mide 3x4", "6x4")
    assert len(L.montar_ventana(guia1, render1, plan, 1, r0, g0, medido=unos)[0]) == 12


# ---------------------------------------------------------------- flujo optico (necesita cv2)

def _escena(dx, alto=270, ancho=480, dy=0):
    """Telon degradado y una forma con textura suave desplazada dx, dy pixeles (medidos a 480 de ancho)."""
    f = ancho // 480
    g = np.linspace(0.75, 0.25, alto, dtype=np.float32)[:, None, None]
    img = np.broadcast_to(g * np.array([0.55, 0.65, 0.85], np.float32), (alto, ancho, 3)).copy()
    yy, xx = np.mgrid[0:120 * f, 0:150 * f].astype(np.float32) / f
    tex = 0.5 + 0.25 * np.sin(xx / 7.0) * np.cos(yy / 5.0) + 0.2 * np.sin((xx + 2 * yy) / 11.0)
    y0, x0 = (70 + dy) * f, (100 + dx) * f
    img[y0:y0 + 120 * f, x0:x0 + 150 * f] = np.stack([tex, 0.6 * tex + 0.1, 0.3 * tex], axis=-1)
    return img


def test_dis_mide_mas_cuanto_mas_se_desplaza_la_forma():
    if not _hay_cv2("la medida de saltos con DIS"):
        return
    desplazamientos = [0, 2, 4, 8, 16, 32, 64]
    poses = torch.from_numpy(np.stack([_escena(0)] + [_escena(d) for d in desplazamientos]))
    # cada pareja (pose 0, pose desplazada d): se miden de dos en dos para no encadenar
    medidas = [L.medir_saltos(torch.stack([poses[0], poses[1 + i]]))[0] for i in range(len(desplazamientos))]
    assert medidas[0] < 0.05, medidas  # nada se mueve
    assert all(b > a for a, b in zip(medidas, medidas[1:])), medidas
    for d, m in zip(desplazamientos[1:], medidas[1:]):
        assert 0.5 * d < m < 1.5 * d, (d, m)  # en pixeles a 480 de ancho
    # una pose detras de otra: tantos saltos como poses menos una
    seguidas = L.medir_saltos(torch.from_numpy(np.stack([_escena(d) for d in (0, 2, 42, 45)])))
    assert len(seguidas) == 3 and seguidas[1] > 9.0 > max(seguidas[0], seguidas[2]), seguidas
    # la medida no depende de la resolucion de las poses: el mismo salto al doble de tamano
    grande = L.medir_saltos(torch.from_numpy(np.stack([_escena(0, 540, 960), _escena(16, 540, 960)])))[0]
    assert abs(grande - medidas[4]) < 0.15 * medidas[4], (grande, medidas[4])
    # y acepta numpy uint8 igual que los tensores 0-1
    en_bytes = L.medir_saltos((np.stack([_escena(0), _escena(16)]) * 255).round().astype(np.uint8))[0]
    assert abs(en_bytes - medidas[4]) < 1e-3, (en_bytes, medidas[4])
    # el alto se reduce en proporcion: en un plano mas alto que ancho, bajar 16 px tambien mide 16
    vertical = L.medir_saltos(torch.from_numpy(np.stack([_escena(0, 540, 480), _escena(0, 540, 480, dy=16)])))[0]
    assert 0.85 * 16 < vertical < 1.15 * 16, vertical


def test_la_medida_se_hace_a_480_de_ancho_y_con_los_parametros_de_la_receta():
    if not _hay_cv2("los parametros del flujo optico"):
        return
    import cv2
    # 960x540 -> 480x270 (asi se midieron los saltos de control) y 1920x1088 -> 480x272 (asi se midio el recorrido)
    assert L._gris(torch.rand(540, 960, 3), cv2).shape == (270, 480)
    assert L._gris(torch.rand(1088, 1920, 3), cv2).shape == (272, 480)
    assert L._gris(np.zeros((270, 480, 4), np.uint8), cv2).dtype == np.uint8
    vistos = []
    dis, farneback = cv2.DISOpticalFlow_create, cv2.calcOpticalFlowFarneback
    cv2.DISOpticalFlow_create = lambda *a: vistos.append(("dis", a)) or dis(*a)
    cv2.calcOpticalFlowFarneback = lambda a, b, flujo, *resto: vistos.append(("farneback", (flujo,) + resto)) or farneback(a, b, flujo, *resto)
    try:
        par = torch.from_numpy(np.stack([_escena(0), _escena(3)]))
        L.medir_saltos(par)
        L.recorrido(par)
    finally:
        cv2.DISOpticalFlow_create, cv2.calcOpticalFlowFarneback = dis, farneback
    assert vistos == [("dis", (cv2.DISOPTICAL_FLOW_PRESET_MEDIUM,)), ("farneback", (None, 0.5, 4, 21, 3, 5, 1.1, 0))], vistos


def test_farneback_mide_el_recorrido_de_cada_paso():
    if not _hay_cv2("el recorrido con Farneback"):
        return
    pasos = [1, 1, 2, 2, 4, 4, 0, 0, 3]
    x = np.r_[0, np.cumsum(pasos)]
    guia = torch.from_numpy(np.stack([_escena(int(v), 272, 480) for v in x]))
    medido = L.recorrido(guia)
    assert medido.shape == (len(pasos),) and medido.dtype == np.float64
    for p, m in zip(pasos, medido):
        assert (m < 0.05) if p == 0 else (0.5 * p < m < 1.5 * p), (pasos, medido.tolist())
    assert medido[0] < medido[2] < medido[4] and medido[8] > medido[2]
    # un paso largo, de 24 px: sin la piramide de 4 niveles y la ventana de 21 la medida se dispara
    largo = L.recorrido(torch.from_numpy(np.stack([_escena(0, 272, 480), _escena(24, 272, 480)])))[0]
    assert 0.9 * 24 < largo < 1.1 * 24, largo
    assert len(L.recorrido(guia[:1])) == 0  # un solo fotograma: nada que medir


def test_nodo_elegir_mide_en_la_guia_y_corta_el_render():
    if not _hay_cv2("el nodo CP_LentaElegir entero"):
        return
    # La forma corre al principio del salto (5 px por lento) y frena (1 px por lento): por indice saldria [0, 4, 8, 12].
    x = np.r_[0, np.cumsum([5] * 4 + [1] * 12)]
    guia = torch.from_numpy(np.stack([_escena(int(v)) for v in x]))
    render = _lote(17, 8, 12)
    plan = json.dumps(L.planificar([0, 4], 5, [2.0]))
    elegido, elegida = L.LentaElegir.execute(guia, render, plan, 0).args
    assert _de_donde(elegido) == [0, 2, 3, 8, 16], _de_donde(elegido)
    assert elegido.shape == (5, 8, 12, 3) and elegida.shape == (5, 8, 12, 3)
    assert 0.0 <= float(elegida.min()) and float(elegida.max()) <= 1.0
    _falla(lambda: L.LentaElegir.execute(guia[:9], render, plan, 0), "[CP_LentaElegir]", "la guia trae 9")
    _falla(lambda: L.LentaElegir.execute(guia, render, "{}", 0), "[CP_LentaElegir]", "version")


def test_nodo_plan_mide_las_poses_y_devuelve_el_plan():
    if not _hay_cv2("el nodo CP_LentaPlan entero"):
        return
    poses = torch.from_numpy(np.stack([_escena(d) for d in (0, 2, 42, 45)]))
    mapa = json.dumps({"bloques": [[0, 3], [4, 7], [8, 11], [12, 12]]})
    texto, n_ventanas, lentos_totales, info = L.LentaPlan.execute(poses, mapa).args
    plan = L.leer_plan(texto)
    assert plan["inicios"] == [0, 4, 8, 12] and plan["total"] == 13
    assert plan["lentos"] == [16, 32, 16] and (n_ventanas, lentos_totales) == (1, 65)
    assert len(plan["saltos_px"]) == 3 and plan["saltos_px"][1] > 9.0
    assert info.startswith("4 poses en 13 fotogramas: 1 ventana de 65"), info
    # con las entradas del panel: con el tope en 33 cada salto va en su ventana (16 + 32 + 1 ya no cabe)
    texto, n_ventanas, lentos_totales, _ = L.LentaPlan.execute(poses, mapa, 9.0, 33, 12, 1601).args
    assert (n_ventanas, lentos_totales) == (3, 67) and [v["n"] for v in json.loads(texto)["ventanas"]] == [17, 33, 17]
    # y con el umbral por encima del salto grande, todos a 16 y en una ventana de 49
    texto, n_ventanas, lentos_totales, _ = L.LentaPlan.execute(poses, mapa, 60.0, 49, 12, 1601).args
    assert (n_ventanas, lentos_totales) == (1, 49) and json.loads(texto)["lentos"] == [16, 16, 16]
    _falla(lambda: L.LentaPlan.execute(poses[:3], mapa), "[CP_LentaPlan]", "3 poses", "4")
    _falla(lambda: L.LentaPlan.execute(poses, mapa, 9.0, 177, 12, 65 - 1), "demasiado largo")


def test_sin_cv2_el_error_dice_que_falta():
    guardado = sys.modules.get("cv2", "no estaba")
    sys.modules["cv2"] = None  # asi el import falla aunque este instalado
    try:
        _falla(lambda: L.medir_saltos(torch.rand(2, 8, 12, 3)), "[CP_LentaPlan]", "cv2", error=RuntimeError)
        _falla(lambda: L.recorrido(torch.rand(2, 8, 12, 3)), "[CP_LentaElegir]", "cv2", error=RuntimeError)
    finally:
        if guardado == "no estaba":
            del sys.modules["cv2"]
        else:
            sys.modules["cv2"] = guardado


# ---------------------------------------------------------------- esquemas

def test_esquemas_con_las_entradas_y_salidas_del_contrato():
    plan = L.LentaPlan.define_schema()
    assert plan["node_id"] == "CP_LentaPlan" and plan["category"] == "character-pipeline/poses"
    assert plan["display_name"] == "C\u00e1mara lenta \u00b7 Plan"
    assert [p.a[0] for p in plan["inputs"]] == ["poses", "mapa", "umbral", "max_ventana", "transicion_max", "max_total"]
    assert [p.k["display_name"] for p in plan["outputs"]] == ["plan", "n_ventanas", "fotogramas_lentos", "info"]
    por_nombre = {p.a[0]: p.k for p in plan["inputs"]}
    assert por_nombre["mapa"]["force_input"] is True
    assert [(por_nombre[n]["default"], por_nombre[n]["min"], por_nombre[n]["max"])
            for n in ("umbral", "max_ventana", "transicion_max", "max_total")] == [(9.0, 0.0, 100.0), (177, 33, 513), (12, 2, 64), (1601, 65, 8001)]

    elegir = L.LentaElegir.define_schema()
    assert elegir["node_id"] == "CP_LentaElegir" and elegir["category"] == "character-pipeline/poses"
    assert elegir["display_name"] == "C\u00e1mara lenta \u00b7 Elegir fotogramas"
    assert [p.a[0] for p in elegir["inputs"]] == ["guia", "render", "plan", "ventana", "anterior_render", "anterior_guia"]
    assert [p.k["display_name"] for p in elegir["outputs"]] == ["render", "guia"]  # salida 0 = render, salida 1 = guia
    por_nombre = {p.a[0]: p.k for p in elegir["inputs"]}
    assert por_nombre["plan"]["force_input"] is True
    assert por_nombre["anterior_render"]["optional"] is True and por_nombre["anterior_guia"]["optional"] is True
    assert not any(por_nombre[n].get("optional") for n in ("guia", "render", "plan", "ventana"))


if __name__ == "__main__":
    pruebas = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in pruebas:
        t()
        print("ok", t.__name__)
    print(len(pruebas), "pruebas")
