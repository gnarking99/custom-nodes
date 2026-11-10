# -*- coding: utf-8 -*-
"""Prueba de integracion del blocking en FOTOS, sin ComfyUI: fotos sinteticas en una carpeta de entrada temporal ->
CP_PosesDesdeFotos -> CP_LentaPlan (transicion_max = la salida "transicion") -> CP_BlockingLento con y sin
solo_animacion -> el interprete de tests/test_lenta_integracion.py (dobles de LTX, CP_LentaElegir de verdad).

Se comprueba que salen los fotogramas que anuncia el nodo de fotos, que cada pose cae en su fotograma (k x d), que
entre una pose y la siguiente la imagen cambia en todos los fotogramas y que la ultima se queda 1 + espera_final.
Se ejecuta con: python tests/test_lenta_fotos_integracion.py"""
import json
import os
import shutil
import sys
import tempfile
import types

import torch
from PIL import Image

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
# los dobles de comfy_api y comfy_execution, el interprete y los enlaces de la prueba de integracion de video
import test_lenta_integracion as TI  # noqa: E402

LE, LG = TI.LE, TI.LG
_io = sys.modules["comfy_api.latest"].io
for _n in ("Combo", "Mask"):
    if not hasattr(_io, _n):
        setattr(_io, _n, TI._mk(_n))
if not hasattr(_io, "UploadType"):
    _io.UploadType = types.SimpleNamespace(image="image_upload")

# ---- doble de folder_paths: la carpeta de entrada es una carpeta temporal ----
ENTRADA = tempfile.mkdtemp(prefix="lenta_fotos_")
_fp = types.ModuleType("folder_paths")
_fp.get_input_directory = lambda: ENTRADA


def _ruta(nombre):
    ruta = os.path.abspath(os.path.join(ENTRADA, nombre))
    if not ruta.startswith(ENTRADA + os.sep):
        raise ValueError(f"Invalid file path: {nombre!r}")
    return ruta


def _existe(nombre):
    try:
        return os.path.exists(_ruta(nombre))
    except ValueError:
        return False


_fp.get_annotated_filepath = _ruta
_fp.exists_annotated_filepath = _existe
_fp.filter_files_content_types = lambda archivos, tipos: [f for f in archivos if f.lower().endswith(".png")]
sys.modules["folder_paths"] = _fp

import poses_fotos as P  # noqa: E402  (la ruta del pack ya la puso test_lenta_integracion)


def _foto(nombre, k):
    """Pose k: fondo liso como el de un playblast, con el canal verde = k/100 (lo lee _pose_de del interprete) y un
    cuadrado que cambia de sitio (para que el flujo optico mida algo)."""
    im = Image.new("RGB", (96, 54), (110, int(round(k / 100 * 255)), 110))
    x = 6 + (k * 11) % 70
    im.paste((110, int(round(k / 100 * 255)), 230), (x, 20, x + 14, 34))
    im.save(os.path.join(ENTRADA, nombre))
    return nombre


def _casillas(nombres):
    v = {c: P.MARCADOR_VACIO for c in P.NOMBRES_CASILLAS}
    v.update(nombres)
    return v


def _cadena(casillas, d, espera, solo_animacion, ritmo=None, mas_lentos=True):
    """CP_PosesDesdeFotos -> CP_LentaPlan -> CP_BlockingLento -> interprete. Devuelve lo que hace falta comprobar.
    mas_lentos=True como en la tarjeta de fotos (lo lleva cableado en su motor)."""
    poses, mapa, n, transicion, fotogramas, fps = P.PosesDesdeFotos.execute(
        fotogramas_por_pose=d, espera_final=espera, orden="casilla", **_casillas(casillas)).args
    assert transicion == d and fps == 24.0 and len(poses) == n
    extra = {} if ritmo is None else {"ritmo": ritmo}
    if mas_lentos is not None:
        extra["mas_lentos"] = mas_lentos
    plan_txt, n_ventanas, lentos, info = LE.LentaPlan.execute(poses, mapa, transicion_max=transicion, **extra).args
    plan = json.loads(plan_txt)
    TI.GraphBuilder.set_default_prefix("40", 0, 0)
    entradas = dict(TI.ENLACES)
    if solo_animacion is not None:
        entradas["solo_animacion"] = solo_animacion
    salida = LG.BlockingLento.execute(plan_txt, TI.A, TI.H, **entradas)
    fuera = TI._valores_fuera(n)
    fuera["poses"] = (poses,)  # las de las fotos, no las de la prueba de video
    interprete = TI.Interprete(salida.expand, fuera)
    animacion = interprete.valor(salida.args[0])
    playblast = interprete.valor(salida.args[1])
    assert len(interprete.hecho) == len(salida.expand), "hay nodos del subgrafo que no alimentan ninguna salida"
    return n, fotogramas, plan, salida.expand, interprete, animacion, playblast, info


def _comprobar(n, d, espera, fotogramas, plan, grafo, interprete, animacion, playblast, solo, ritmo=1):
    ventanas = plan["ventanas"]
    assert plan["inicios"] == [k * d * ritmo for k in range(n)] and plan["transicion_max"] == d * ritmo
    assert fotogramas == (n - 1) * d + 1 + espera and plan["total"] == fotogramas * ritmo
    assert animacion.shape == (fotogramas * ritmo, TI.H, TI.A, 3) and playblast.shape == animacion.shape
    marcas = [TI.leer_marca(f) for f in animacion]
    if solo:
        assert torch.equal(animacion, playblast), "con solo_animacion las dos salidas son el playblast"
        assert all(e == 0 for e, _, _ in marcas), "con solo_animacion todo sale de la etapa 1"
        assert not any(".s2." in i or i.endswith(".ancla") for i in grafo), "sin etapa 2 ni ancla"
        assert not any(r[0] == "guia2" for r in interprete.registro)
        assert len([r for r in interprete.registro if r[0] == "muestreo"]) == 2 * len(ventanas)
    else:
        assert all(e == 1 for e, _, _ in marcas) and all(TI.leer_marca(f)[0] == 0 for f in playblast)
        assert [m[1:] for m in marcas] == [TI.leer_marca(f)[1:] for f in playblast]
        assert len([r for r in interprete.registro if r[0] == "muestreo"]) == 3 * len(ventanas)
    # las fotos llegan a las guias de la etapa 1 en su orden y en el fotograma lento que dice el plan
    for w, v in enumerate(ventanas):
        guias = [r[2] for r in interprete.registro if r[0] == "guia1" and TI._ventana_etapa(r[1])[0] == w]
        esperadas = {(v["pos"][j], k) for j, k in enumerate(range(v["p0"], v["p1"] + 1))}
        assert {(g[1], g[2]) for g in guias} == esperadas, (w, sorted({(g[1], g[2]) for g in guias}), sorted(esperadas))
    # cada pose cae en su fotograma (k x d x ritmo), con el lento que le da el plan
    for k in range(n):
        w = max(i for i, v in enumerate(ventanas) if v["p0"] <= k)
        assert marcas[k * d * ritmo][1:] == (w, ventanas[w]["pos"][k - ventanas[w]["p0"]]), (k, marcas[k * d * ritmo])
    # de la primera pose a la ultima no hay ningun fotograma repetido: se mueve en todos
    hasta = (n - 1) * d * ritmo
    assert all(b[1:] > a[1:] for a, b in zip(marcas[:hasta], marcas[1:hasta + 1])), marcas[:hasta + 1]
    # y la ultima pose se queda 1 + espera_final (por el ritmo)
    assert marcas[hasta:] == [(marcas[hasta][0], len(ventanas) - 1, ventanas[-1]["n"] - 1)] * ((1 + espera) * ritmo)
    return marcas


# ---------------------------------------------------------------- pruebas

def test_fotos_de_principio_a_fin_con_y_sin_render():
    for n, d, espera in ((2, 8, 0), (5, 12, 6), (9, 16, 0), (13, 24, 12)):
        casillas = {P.NOMBRES_CASILLAS[k]: _foto(f"pose_{n}_{k}.png", k) for k in range(n)}
        marcas = {}
        for solo in (None, False, True):
            r = _cadena(casillas, d, espera, solo)
            marcas[solo] = _comprobar(r[0], d, espera, r[1], r[2], r[3], r[4], r[5], r[6], solo is True)
            assert "AVISO" not in r[7], r[7]
            # mas_lentos (la tarjeta de fotos): al menos dos lentos por fotograma de salida hasta 16 por pose
            assert r[2]["mas_lentos"] is True and "Regla de mas lentos activa (mas_lentos)" in r[7]
            assert all(x >= min(2 * d, 32) for x in r[2]["lentos"]), (d, r[2]["lentos"])
        # sin la clave y con False sale lo mismo; solo_animacion elige los mismos lentos (etapa 0 en vez de 1)
        assert marcas[None] == marcas[False]
        assert [m[1:] for m in marcas[True]] == [m[1:] for m in marcas[False]]


def test_casillas_con_huecos_y_veinticuatro_poses():
    # casillas 1, 2 y 5: tres poses, en ese orden
    huecos = {"pose_01": _foto("h0.png", 0), "pose_02": _foto("h1.png", 1), "pose_05": _foto("h2.png", 2)}
    r = _cadena(huecos, 12, 0, True)
    assert r[0] == 3
    _comprobar(r[0], 12, 0, r[1], r[2], r[3], r[4], r[5], r[6], True)
    # las 24 casillas llenas a 24 fotogramas por pose: 553 fotogramas de salida
    llenas = {c: _foto(f"llena_{k}.png", k) for k, c in enumerate(P.NOMBRES_CASILLAS)}
    r = _cadena(llenas, 24, 0, True)
    assert r[0] == 24 and r[1] == 553
    _comprobar(r[0], 24, 0, r[1], r[2], r[3], r[4], r[5], r[6], True)


def test_ritmo_con_fotos():
    casillas = {P.NOMBRES_CASILLAS[k]: _foto(f"ritmo_{k}.png", k) for k in range(4)}
    r1 = _cadena(casillas, 8, 0, True, ritmo=1)
    r0 = _cadena(casillas, 8, 0, True)
    assert r1[2] == r0[2] and "ritmo" not in r1[2], "ritmo 1 es el plan sin ritmo"
    r2 = _cadena(casillas, 8, 4, True, ritmo=2)
    _comprobar(r2[0], 8, 4, r2[1], r2[2], r2[3], r2[4], r2[5], r2[6], True, ritmo=2)
    assert r2[2]["ritmo"] == 2 and len(r2[5]) == 2 * r2[1]


def test_fotos_sin_mas_lentos_es_la_regla_de_la_v014():
    # 12 por pose: con mas_lentos los saltos pequenos llevan 32; sin el (lo que hacia la v0.14) 16 o 32 segun el umbral
    casillas = {P.NOMBRES_CASILLAS[k]: _foto(f"sin_{k}.png", k) for k in range(5)}
    con = _cadena(casillas, 12, 0, True)
    sin = _cadena(casillas, 12, 0, True, mas_lentos=None)
    assert "mas_lentos" not in sin[2] and con[2]["mas_lentos"] is True
    assert sin[2]["lentos"] == [32 if px >= LE.UMBRAL_SALTO else 16 for px in sin[2]["saltos_px"]], sin[2]
    assert con[2]["lentos"] == [32] * 4 and con[2]["saltos_px"] == sin[2]["saltos_px"]
    _comprobar(con[0], 12, 0, con[1], con[2], con[3], con[4], con[5], con[6], True)


if __name__ == "__main__":
    try:
        pruebas = [v for k, v in sorted(globals().items()) if k.startswith("test_") and v.__module__ == __name__]
        for t in pruebas:
            t()
            print("ok", t.__name__)
        print(len(pruebas), "pruebas")
    finally:
        shutil.rmtree(ENTRADA, ignore_errors=True)
