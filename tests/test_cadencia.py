# -*- coding: utf-8 -*-
"""Prueba CP_Cadencia (la animacion a doses o a treses, como stop motion), sin ComfyUI.
Compara la regla con una implementacion de referencia escrita de otra forma (fotograma a fotograma) en miles de planes
al azar de CP_LentaPlan (ritmo 1 a 8, mas_lentos, mapas de video y de fotos) y para cada cadencia de 1 a MAX_CADENCIA.
Los datos de control (tests/datos/lenta_0003.json) son los de un plano de 77 fotogramas y 25 poses.
Se ejecuta con: python tests/test_cadencia.py"""
import json
import os
import sys
import types

import numpy as np
import torch

# ---- stub minimo de comfy_api.latest (el de test_lenta.py, con lo que necesita poses_fotos) ----
mod = types.ModuleType("comfy_api")
latest = types.ModuleType("comfy_api.latest")


class _Port:
    def __init__(self, *a, **k):
        self.a, self.k = a, k


def _mk(name):
    # el puerto lleva el tipo en el nombre de su clase (IntInput...): asi el esquema se puede comprobar por tipo
    return type(name, (), {"Input": type(name + "Input", (_Port,), {}), "Output": type(name + "Output", (_Port,), {})})


class _Salida:
    def __init__(self, *a, **k):
        self.args = a


class _IO:
    Schema = staticmethod(lambda **k: k)
    NodeOutput = _Salida
    ComfyNode = type("ComfyNode", (), {})
    UploadType = types.SimpleNamespace(image="image_upload")


for _n in ("Image", "Float", "Int", "String", "Boolean", "Combo", "Mask"):
    setattr(_IO, _n, _mk(_n))

latest.io = _IO
mod.latest = latest
sys.modules["comfy_api"] = mod
sys.modules["comfy_api.latest"] = latest

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(AQUI, "..", "comfy_character_pipeline"))
import cadencia as C  # noqa: E402
import lenta as L  # noqa: E402
import poses_fotos as P  # noqa: E402

with open(os.path.join(AQUI, "datos", "lenta_0003.json"), encoding="utf-8") as _f:
    DATOS = json.load(_f)
INICIOS = DATOS["inicios"]
CADENCIAS = range(1, C.MAX_CADENCIA + 1)


def _falla(fn, *trozos, error=ValueError):
    """La llamada tiene que fallar con ese error y con esos trozos en el mensaje."""
    try:
        fn()
    except error as e:
        for t in trozos:
            assert t in str(e), (t, str(e))
        return str(e)
    raise AssertionError(f"tenia que fallar con {error.__name__} {trozos}")


# ---------------------------------------------------------------- referencia independiente

def _referencia(inicios, total, transicion_max, c):
    """La regla escrita de otra forma: para cada fotograma de salida, por separado, busca su salto, su sitio dentro
    del movimiento y el bloque que le toca contando fotogramas, sin listas de bloques ni rebanadas."""
    salida = []
    for f in range(total):
        k = None
        for j in range(len(inicios) - 1):
            if inicios[j] <= f < inicios[j + 1]:
                k = j
        if k is None or c == 1:  # la cola de la ultima pose (o a unos): tal cual
            salida.append(f)
            continue
        d = inicios[k + 1] - inicios[k]
        m = d if d < transicion_max else transicion_max
        empieza = inicios[k + 1] - m  # primer fotograma del movimiento
        if f < empieza:  # espera quieta
            salida.append(f)
            continue
        dibujos = m // c if m // c > 0 else 1
        largos = m % dibujos  # cuantos bloques llevan un fotograma de mas
        corto = m // dibujos
        # recorre los bloques sumando su largo hasta pasar por f
        inicio_bloque, b = empieza, 0
        while True:
            largo = corto + 1 if b < largos else corto
            if f < inicio_bloque + largo:
                break
            inicio_bloque += largo
            b += 1
        salida.append(inicio_bloque)
    return salida


def _tramos(seq):
    """[(valor, inicio, largo)] de los tramos de valores iguales seguidos."""
    tramos = []
    for i, v in enumerate(seq):
        if tramos and tramos[-1][0] == v:
            tramos[-1][2] += 1
        else:
            tramos.append([v, i, 1])
    return [tuple(t) for t in tramos]


def _comprobar_propiedades(inicios, total, tm, c, idx, caso=None):
    assert len(idx) == total, caso
    assert all(isinstance(x, int) for x in idx), caso
    assert all(0 <= x <= i for i, x in enumerate(idx)), caso  # nunca mira hacia delante
    assert all(b >= a for a, b in zip(idx, idx[1:])), caso  # no decrece
    assert all(idx[a] == a for a in inicios), caso  # cada pose en su fotograma
    if c == 1:
        assert idx == list(range(total)), caso
    assert idx[inicios[-1]:] == list(range(inicios[-1], total)), caso  # la cola no se toca
    dibujos = 1
    for a, b in zip(inicios, inicios[1:]):
        d = b - a
        m = min(d, tm)
        p = a + d - m
        assert idx[a:p] == list(range(a, p)), (caso, a, b)  # la espera quieta no se toca
        tramos = _tramos(idx[p:b])
        assert tramos[0][0] == p, (caso, a, b)  # el primer dibujo del salto es la pose
        assert all(v == p + ini for v, ini, _ in tramos), (caso, a, b)  # cada bloque ensena su primer fotograma
        largos = [n for _, _, n in tramos]
        if c > 1:
            if m >= c:
                assert all(c <= n <= 2 * c - 1 for n in largos), (caso, a, b, c, largos)
                assert largos == sorted(largos, reverse=True), (caso, largos)  # los largos primero
                assert len(largos) == m // c, (caso, largos)
            else:
                assert largos == [m], (caso, a, b, c, largos)
        dibujos += len(tramos) if c > 1 else m
    return dibujos


# ---------------------------------------------------------------- planes al azar

def _mapa_video(rng):
    n = int(rng.integers(2, 26))
    d_max = int(rng.choice([3, 5, 8, 12, 20, 40]))
    inicios = [0]
    for _ in range(n - 1):
        inicios.append(inicios[-1] + int(rng.integers(1, d_max + 1)))
    return inicios, inicios[-1] + int(rng.integers(1, 12)), [float(x) for x in rng.random(n - 1) * 25.0]


def _plan_al_azar(rng, caso):
    """Un plan de verdad de L.planificar: mapas de video o de fotos (P.mapa_bloques), ritmo 1 a 8 y mas_lentos."""
    ritmo = int(rng.integers(1, L.MAX_RITMO + 1)) if caso % 3 else 1
    if caso % 2:
        n, d, espera = int(rng.integers(2, P.CASILLAS + 1)), int(rng.integers(2, P.MAX_FOTOGRAMAS_POR_POSE + 1)), int(rng.integers(0, 30))
        bloques = P.mapa_bloques(n, d, espera)
        inicios, total = L.leer_mapa(json.dumps({"bloques": bloques}), n)
        assert total == P.total_fotogramas(n, d, espera)
        saltos = [float(x) for x in rng.random(n - 1) * 25.0]
        tm, mas = d, True  # como la tarjeta de fotos
    else:
        inicios, total, saltos = _mapa_video(rng)
        tm, mas = int(rng.choice([2, 5, 8, 12, 16, 40])), bool(rng.integers(0, 2))
    if ritmo > 1 and total * ritmo > L.MAX_SALIDA:
        ritmo = max(1, L.MAX_SALIDA // total)
    return L.planificar(inicios, total, saltos, transicion_max=tm, max_total=8001, ritmo=ritmo, mas_lentos=mas)


def test_igual_que_la_referencia_y_propiedades_en_planes_al_azar():
    rng = np.random.default_rng(16)
    casos = 0
    ritmos, fotos, mas = set(), 0, 0
    for caso in range(1500):
        plan = _plan_al_azar(rng, caso)
        ritmos.add(plan.get("ritmo", 1))
        fotos += caso % 2
        mas += bool(plan.get("mas_lentos"))
        ini, total, tm = plan["inicios"], plan["total"], plan["transicion_max"]
        for c in CADENCIAS:
            idx = C.indices_cadencia(ini, total, tm, c)
            assert idx == _referencia(ini, total, tm, c), (caso, c, ini, total, tm)
            dibujos = _comprobar_propiedades(ini, total, tm, c, idx, (caso, c))
            assert C.resumen_cadencia(ini, total, tm, c).startswith(f"{C.NOMBRES[c]}: {dibujos} dibujos en {total} fotogramas."), caso
            casos += 1
    assert ritmos == set(range(1, L.MAX_RITMO + 1)), ritmos
    assert casos == 1500 * C.MAX_CADENCIA and fotos == 750 and mas > 750
    print(f"   {casos} casos iguales a la referencia (ritmos {sorted(ritmos)}, {fotos} planes de fotos, {mas} con mas_lentos)")


def test_referencia_en_saltos_pequenos_exhaustivos():
    """Todo salto de 1 a 40 fotogramas con transicion_max de 1 a 40 y espera final de 0 a 3, en las cuatro cadencias."""
    casos = 0
    for d in range(1, 41):
        for tm in range(1, 41):
            for cola in range(1, 4):
                ini, total = [0, d, 2 * d], 2 * d + cola
                for c in CADENCIAS:
                    idx = C.indices_cadencia(ini, total, tm, c)
                    assert idx == _referencia(ini, total, tm, c), (d, tm, c)
                    _comprobar_propiedades(ini, total, tm, c, idx, (d, tm, c))
                    casos += 1
    assert casos == 40 * 40 * 3 * C.MAX_CADENCIA


# ---------------------------------------------------------------- casos fijos

def test_a_unos_es_la_identidad():
    for ini, total, tm in (([0, 4, 8, 13], 20, 12), ([0, 1], 1 + 1, 12), ([0, 20, 32], 40, 12)):
        assert C.indices_cadencia(ini, total, tm, 1) == list(range(total))


def test_fotos_doce_por_pose_a_doses_y_a_treses():
    ini, total = [0, 12, 24], 25
    assert C.indices_cadencia(ini, total, 12, 2) == [0, 0, 2, 2, 4, 4, 6, 6, 8, 8, 10, 10] * 1 + \
        [12, 12, 14, 14, 16, 16, 18, 18, 20, 20, 22, 22, 24]
    assert C.indices_cadencia(ini, total, 12, 3) == [0, 0, 0, 3, 3, 3, 6, 6, 6, 9, 9, 9, 12, 12, 12, 15, 15, 15, 18, 18, 18, 21, 21, 21, 24]
    assert C.indices_cadencia(ini, total, 12, 4) == [0] * 4 + [4] * 4 + [8] * 4 + [12] * 4 + [16] * 4 + [20] * 4 + [24]


def test_bloques_desiguales_con_los_largos_primero():
    # d = 13 a doses: 6 dibujos, 3 + 2 + 2 + 2 + 2 + 2
    idx = C.indices_cadencia([0, 13], 14, 13, 2)
    assert idx == [0, 0, 0, 3, 3, 5, 5, 7, 7, 9, 9, 11, 11, 13]
    # d = 11 a treses: 3 dibujos, 4 + 4 + 3
    assert C.indices_cadencia([0, 11], 12, 12, 3) == [0] * 4 + [4] * 4 + [8] * 3 + [11]
    # d = 7 a cuatros: 1 dibujo de 7 (7 // 4 = 1), dentro de 4..7
    assert C.indices_cadencia([0, 7], 8, 12, 4) == [0] * 7 + [7]


def test_espera_larga_de_video_no_se_toca():
    # d = 20 con transicion_max 12: los 8 primeros son la espera quieta y siguen igual
    idx = C.indices_cadencia([0, 20], 21, 12, 2)
    assert idx[:8] == list(range(8))
    assert idx[8:20] == [8, 8, 10, 10, 12, 12, 14, 14, 16, 16, 18, 18] and idx[20] == 20
    assert C.indices_cadencia([0, 20], 21, 12, 3)[8:20] == [8] * 3 + [11] * 3 + [14] * 3 + [17] * 3


def test_saltos_mas_cortos_que_la_cadencia_quedan_en_la_pose():
    assert C.indices_cadencia([0, 2, 4], 5, 12, 3) == [0, 0, 2, 2, 4]
    assert C.indices_cadencia([0, 5, 10], 11, 12, 3) == [0] * 5 + [5] * 5 + [10]  # 5 // 3 = 1 dibujo
    assert C.indices_cadencia([0, 3], 4, 12, 4) == [0, 0, 0, 3]
    assert C.indices_cadencia([0, 2], 3, 12, 2) == [0, 0, 2]  # 2 a doses: un dibujo de 2


def test_salto_de_un_fotograma_no_cambia():
    for c in CADENCIAS:
        assert C.indices_cadencia([0, 1, 2, 3], 4, 12, c) == [0, 1, 2, 3]
        assert C.indices_cadencia([0, 1, 5], 6, 12, c)[:2] == [0, 1]


def test_la_cola_de_la_ultima_pose_no_se_toca():
    bloques = P.mapa_bloques(3, 6, espera_final=7)
    ini, total = L.leer_mapa(json.dumps({"bloques": bloques}), 3)
    assert (ini, total) == ([0, 6, 12], 20)
    for c in CADENCIAS:
        idx = C.indices_cadencia(ini, total, 6, c)
        assert idx[12:] == list(range(12, 20)), (c, idx)


def test_ritmo_cuenta_en_fotogramas_de_salida_sin_volver_a_multiplicar():
    plan = L.planificar([0, 4, 8], 9, [3.0, 3.0], transicion_max=12, ritmo=2)
    assert plan["inicios"] == [0, 8, 16] and plan["total"] == 18 and plan["transicion_max"] == 24
    idx = C.indices_cadencia(plan["inicios"], plan["total"], plan["transicion_max"], 2)
    assert idx == [0, 0, 2, 2, 4, 4, 6, 6, 8, 8, 10, 10, 12, 12, 14, 14, 16, 17]


def test_plano_de_control_a_doses_38_dibujos_y_a_treses_25():
    plan = L.planificar(INICIOS, 77, DATOS["saltos_dis_480"])
    assert plan["total"] == 77 and len(plan["inicios"]) == 25 and plan["transicion_max"] == 12
    ini = plan["inicios"]
    a2 = C.indices_cadencia(ini, 77, 12, 2)
    a3 = C.indices_cadencia(ini, 77, 12, 3)
    assert len(set(a2)) == 38 and len(set(a3)) == 25
    assert sorted(set(a3)) == ini  # a treses solo quedan las poses: el blocking
    r1 = C.resumen_cadencia(ini, 77, 12, 1)
    r2 = C.resumen_cadencia(ini, 77, 12, 2)
    r3 = C.resumen_cadencia(ini, 77, 12, 3)
    assert r1 == "A unos: 77 dibujos en 77 fotogramas. La animacion sale tal cual de CP_BlockingLento.", r1
    assert r2.startswith("A doses: 38 dibujos en 77 fotogramas.") and "AVISO" not in r2, r2
    saltos_cortos = sum(1 for a, b in zip(ini, ini[1:]) if b - a < 4)
    assert saltos_cortos > 1, saltos_cortos
    assert f" {saltos_cortos} de 24 saltos se mueven en menos de 4 fotogramas y se quedan enteros en la pose." in r2, r2
    assert r3.startswith("A treses: 25 dibujos en 77 fotogramas. AVISO: los 24 saltos se mueven en menos de 6 fotogramas"), r3
    assert "la animacion queda como el blocking" in r3, r3
    assert "(mas ritmo o mas fotogramas por pose)" in r3, r3
    assert C.resumen_cadencia([0, 2], 3, 12, 3).startswith("A treses: 2 dibujos en 3 fotogramas. AVISO: el salto se mueve en menos de 6")


def test_resumen_en_singular_con_un_solo_salto_parcial():
    assert C.resumen_cadencia([0, 2, 30], 31, 12, 3) == (
        "A treses: 6 dibujos en 31 fotogramas. 1 de 2 saltos se mueve en menos de 6 fotogramas y se queda entero en la pose.")
    assert C.resumen_cadencia([0, 2, 4, 30], 31, 12, 3) == (
        "A treses: 7 dibujos en 31 fotogramas. 2 de 3 saltos se mueven en menos de 6 fotogramas y se quedan enteros en la pose.")


def test_resumen_de_un_salto_largo_con_transicion_max_corto_habla_del_movimiento():
    # d = 20 y transicion_max 3: el salto es largo, pero solo se mueve en 3 fotogramas
    texto = C.resumen_cadencia([0, 20], 21, 3, 2)
    assert texto == (
        "A doses: 2 dibujos en 21 fotogramas. AVISO: el salto se mueve en menos de 4 fotogramas y se queda entero en la "
        "pose: la animacion queda como el blocking, a saltos. Luce con mas fotogramas entre pose y pose (mas ritmo o mas "
        "fotogramas por pose) o con una cadencia menor."), texto
    assert "es de menos" not in texto and "son de menos" not in texto
    assert C.resumen_cadencia([0, 20], 21, 3, 4).startswith("A cuatros: 2 dibujos en 21 fotogramas. AVISO: el salto se mueve en menos de 8")


def test_resumen_no_avisa_de_saltos_sin_movimiento():
    # transicion_max 1: ningun salto se mueve y la cadencia no cambia nada; no es culpa de la cadencia
    assert C.resumen_cadencia([0, 3, 6], 7, 1, 3) == "A treses: 3 dibujos en 7 fotogramas."
    assert C.indices_cadencia([0, 3, 6], 7, 1, 3) == list(range(7))


def test_errores_de_cadencia_y_de_tiempos():
    for malo in (0, C.MAX_CADENCIA + 1, -1, True, False, 2.5, 2.0, None, "2", [2]):
        _falla(lambda: C.indices_cadencia([0, 4], 5, 12, malo), "[CP_Cadencia] La cadencia tiene que ser un entero de 1 a 4",
               repr(malo))
        _falla(lambda: C.resumen_cadencia([0, 4], 5, 12, malo), "[CP_Cadencia] La cadencia")
    assert C.indices_cadencia([0, 4], 5, 12, np.int64(2)) == [0, 0, 2, 2, 4]  # un entero de numpy vale
    _falla(lambda: C.indices_cadencia([1, 4], 5, 12, 2), "[CP_Cadencia] Los inicios de las poses tienen que empezar en 0")
    _falla(lambda: C.indices_cadencia([0, 4, 4], 5, 12, 2), "crecientes")
    _falla(lambda: C.indices_cadencia([0, 4], 4, 12, 2), "[CP_Cadencia] El plan dice tener 4 fotogramas y la ultima pose empieza en el 4.")
    _falla(lambda: C.indices_cadencia([0, 4], 5, 0, 2), "[CP_Cadencia] transicion_max tiene que ser al menos 1")
    _falla(lambda: C.indices_cadencia([0, 4], 5, 12, 9, nodo="otro"), "[otro] La cadencia")


def test_tiempos_que_no_son_enteros_se_rechazan_sin_truncar():
    # int() los truncaria en silencio: 12.7 -> 12, "12" -> 12, True -> 1, 4.9 -> 4
    for fn in (C.indices_cadencia, C.resumen_cadencia):
        for malo in (12.7, "12", True, 12.0, None):
            _falla(lambda: fn([0, 4], 5, malo, 2), "[CP_Cadencia] transicion_max tiene que ser un entero", repr(malo))
            _falla(lambda: fn([0, 4], malo, 12, 2), "[CP_Cadencia] El total de fotogramas tiene que ser un entero", repr(malo))
        for malo in (4.9, "4", True, 4.0, None):
            _falla(lambda: fn([0, malo], 5, 12, 2), "[CP_Cadencia] Los inicios de las poses tienen que ser enteros", repr(malo))
        _falla(lambda: fn([False, 4], 5, 12, 2), "[CP_Cadencia] Los inicios de las poses tienen que ser enteros", "False")
        _falla(lambda: fn([0, 4], 5, np.bool_(True), 2), "[CP_Cadencia] transicion_max tiene que ser un entero")
    _falla(lambda: C.indices_cadencia([0, 4.9], 5, 12, 2, nodo="otro"), "[otro] Los inicios")
    # los enteros de numpy siguen valiendo, sueltos o en un array
    ini = np.array([0, 4], dtype=np.int64)
    assert C.indices_cadencia(ini, np.int64(5), np.int32(12), 2) == [0, 0, 2, 2, 4]
    assert C.indices_cadencia([np.int64(0), np.uint8(4)], 5, 12, 2) == [0, 0, 2, 2, 4]
    assert C.resumen_cadencia(ini, np.int64(5), np.int32(12), 2) == C.resumen_cadencia([0, 4], 5, 12, 2)
    # y el plan real llega con enteros: el nodo sigue igual (leer_plan los valida antes)
    plan = L.leer_plan(_plan_control())
    assert all(type(x) is int for x in plan["inicios"]) and type(plan["total"]) is int and type(plan["transicion_max"]) is int


# ---------------------------------------------------------------- nodo

def _numerado(total, alto=4, ancho=6):
    """Lote de total fotogramas: el fotograma f vale f / 1000 en todos sus pixeles."""
    return (torch.arange(total, dtype=torch.float32) / 1000).view(total, 1, 1, 1).expand(total, alto, ancho, 3).contiguous()


def _plan_control():
    return json.dumps(L.planificar(INICIOS, 77, DATOS["saltos_dis_480"]))


def test_esquema():
    esquema = C.Cadencia.define_schema()
    assert (esquema["node_id"], esquema["display_name"], esquema["category"]) == (
        "CP_Cadencia", "Cámara lenta · Cadencia", "character-pipeline/poses")
    entradas = esquema["inputs"]
    assert [p.a[0] for p in entradas] == ["imagenes", "plan", "cadencia"]
    assert [type(p).__name__ for p in entradas] == ["ImageInput", "StringInput", "IntInput"]
    assert entradas[1].k.get("force_input") is True
    cad = entradas[2].k
    assert (cad["default"], cad["min"], cad["max"]) == (1, 1, C.MAX_CADENCIA) and C.MAX_CADENCIA == 4
    assert not any(p.k.get("optional") for p in entradas)
    assert [p.k["display_name"] for p in esquema["outputs"]] == ["imagenes", "info"]
    assert [type(p).__name__ for p in esquema["outputs"]] == ["ImageOutput", "StringOutput"]
    assert C.NODO == "CP_Cadencia"


def test_nodo_a_unos_devuelve_el_mismo_tensor():
    plan = _plan_control()
    im = _numerado(77)
    antes = im.clone()
    salida, info = C.Cadencia.execute(im, plan, 1).args
    assert salida is im and torch.equal(im, antes)
    assert info.startswith("A unos: 77 dibujos")
    # con un plan en el que la cadencia no cambia nada (saltos de 1 fotograma) tambien es el mismo tensor
    uno = json.dumps(L.planificar([0, 1, 2], 3, [1.0, 1.0]))
    im3 = _numerado(3)
    assert C.Cadencia.execute(im3, uno, 3).args[0] is im3


def test_nodo_a_doses_y_treses_saca_los_fotogramas_de_la_regla_sin_tocar_la_entrada():
    plan = _plan_control()
    datos = L.leer_plan(plan)
    for c in (2, 3, 4):
        im = _numerado(77)
        antes = im.clone()
        salida, info = C.Cadencia.execute(im, plan, c).args
        assert torch.equal(im, antes), c  # la entrada (de la cache del motor) no se toca
        assert salida is not im and salida.data_ptr() != im.data_ptr(), c
        assert tuple(salida.shape) == tuple(im.shape) and salida.dtype == im.dtype and salida.device == im.device
        idx = C.indices_cadencia(datos["inicios"], 77, datos["transicion_max"], c)
        assert torch.equal(salida, im[idx]), c
        leidos = [int(round(float(salida[f, 0, 0, 0]) * 1000)) for f in range(77)]
        assert leidos == idx, c
        assert info == C.resumen_cadencia(datos["inicios"], 77, datos["transicion_max"], c)
        salida[:] = 0  # escribir en la salida tampoco toca la entrada: es una copia
        assert torch.equal(im, antes)
    # otro dtype se conserva
    im16 = _numerado(77).half()
    assert C.Cadencia.execute(im16, plan, 2).args[0].dtype == torch.float16


def test_nodo_con_un_plan_de_fotos_y_ritmo():
    plan = L.planificar(*L.leer_mapa(json.dumps({"bloques": P.mapa_bloques(4, 6, 3)}), 4), [5.0, 5.0, 5.0],
                        transicion_max=6, ritmo=2, mas_lentos=True)
    im = _numerado(plan["total"])
    salida = C.Cadencia.execute(im, json.dumps(plan), 3).args[0]
    idx = C.indices_cadencia(plan["inicios"], plan["total"], plan["transicion_max"], 3)
    assert plan["total"] == 2 * (3 * 6 + 1 + 3) and torch.equal(salida, im[idx])
    assert idx[:12] == [0] * 3 + [3] * 3 + [6] * 3 + [9] * 3


def _salto_menor_igual_y_mayor(inicios, tm):
    saltos = [b - a for a, b in zip(inicios, inicios[1:])]
    return any(1 < d < tm for d in saltos) and tm in saltos and any(d > tm for d in saltos)


def test_nodo_con_ritmo_3_usa_el_transicion_max_del_plan_sin_volver_a_multiplicar():
    # ritmo 3: transicion_max 36 en fotogramas de salida y saltos de 12 (menor), 36 (igual) y 60 (mayor que 36)
    plan = L.planificar([0, 4, 8, 20, 40], 41, [3.0, 3.0, 3.0, 3.0], transicion_max=12, ritmo=3)
    ini, total, tm = plan["inicios"], plan["total"], plan["transicion_max"]
    assert plan.get("ritmo") == 3 and tm == 36 and ini == [0, 12, 24, 60, 120] and total == 123, plan
    assert _salto_menor_igual_y_mayor(ini, tm)
    texto = json.dumps(plan)
    for c in CADENCIAS:
        im = _numerado(total)
        antes = im.clone()
        salida, info = C.Cadencia.execute(im, texto, c).args
        idx = C.indices_cadencia(ini, total, tm, c)
        assert idx == _referencia(ini, total, tm, c), c
        assert torch.equal(salida, im[idx]) and torch.equal(im, antes), c
        assert [int(round(float(salida[f, 0, 0, 0]) * 1000)) for f in range(total)] == idx, c
        assert info == C.resumen_cadencia(ini, total, tm, c), (c, info)
        if c > 1:  # con otro transicion_max (12 o 36 x 3) saldrian otros fotogramas y otro info
            for otro in (12, tm * 3):
                assert idx != C.indices_cadencia(ini, total, otro, c), (c, otro)
                assert info != C.resumen_cadencia(ini, total, otro, c), (c, otro)
    assert C.Cadencia.execute(_numerado(total), texto, 2).args[1] == "A doses: 49 dibujos en 123 fotogramas."


def test_nodo_con_un_plan_de_fotos_de_20_fotogramas_por_pose():
    bloques = P.mapa_bloques(5, 20, 4)
    plan = L.planificar(*L.leer_mapa(json.dumps({"bloques": bloques}), 5), [5.0, 8.0, 5.0, 8.0],
                        transicion_max=20, mas_lentos=True)
    ini, total, tm = plan["inicios"], plan["total"], plan["transicion_max"]
    assert tm == 20 and ini == [0, 20, 40, 60, 80] and total == P.total_fotogramas(5, 20, 4), plan
    texto = json.dumps(plan)
    for c in CADENCIAS:
        im = _numerado(total)
        salida, info = C.Cadencia.execute(im, texto, c).args
        idx = C.indices_cadencia(ini, total, tm, c)
        assert idx == _referencia(ini, total, tm, c), c
        assert torch.equal(salida, im[idx]), c
        assert info == C.resumen_cadencia(ini, total, tm, c), (c, info)
        if c > 1:  # con transicion_max 12 saldrian otros fotogramas y otro info
            assert idx != C.indices_cadencia(ini, total, 12, c) and info != C.resumen_cadencia(ini, total, 12, c), c
    # a treses: 20 // 3 = 6 dibujos por salto, 4 + 4 + 3 + 3 + 3 + 3
    assert C.indices_cadencia(ini, total, tm, 3)[:20] == [0] * 4 + [4] * 4 + [8] * 3 + [11] * 3 + [14] * 3 + [17] * 3


def test_nodo_errores():
    plan = _plan_control()
    im = _numerado(77)
    _falla(lambda: C.Cadencia.execute(_numerado(76), plan, 2),
           "[CP_Cadencia] El plan es de 77 fotogramas y llegaron 76", "CP_BlockingLento")
    _falla(lambda: C.Cadencia.execute(_numerado(78), plan, 1), "El plan es de 77 fotogramas y llegaron 78")
    _falla(lambda: C.Cadencia.execute(None, plan, 2), "[CP_Cadencia] El plan es de 77 fotogramas y llegaron nada")
    _falla(lambda: C.Cadencia.execute(im[:, :, :, 0], plan, 2), "[CP_Cadencia] El plan es de 77 fotogramas",
           "no es un lote de imagenes (forma (77, 4, 6))")
    for malo in (0, 5, True, 2.5, None):
        _falla(lambda: C.Cadencia.execute(im, plan, malo), "[CP_Cadencia] La cadencia tiene que ser un entero de 1 a 4")
    _falla(lambda: C.Cadencia.execute(im, "{roto", 2), "[CP_Cadencia] El plan no se puede leer")
    viejo = json.loads(plan)
    viejo["version"] = 2
    _falla(lambda: C.Cadencia.execute(im, json.dumps(viejo), 2), "[CP_Cadencia] El plan es de la version 2")
    una = json.loads(plan)
    una["inicios"] = [0]
    _falla(lambda: C.Cadencia.execute(im, json.dumps(una), 2), "[CP_Cadencia]")
    assert torch.equal(im, _numerado(77))


if __name__ == "__main__":
    pruebas = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    fallos = []
    for t in pruebas:
        try:
            t()
        except Exception as e:  # se siguen las demas: asi se ve de una vez todo lo que falla
            fallos.append(t.__name__)
            print("FALLA", t.__name__, repr(e)[:300])
            continue
        print("ok", t.__name__)
    print(len(pruebas), "pruebas", f"y {len(fallos)} fallan: {fallos}" if fallos else "")
    sys.exit(1 if fallos else 0)
