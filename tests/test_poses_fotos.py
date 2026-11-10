# -*- coding: utf-8 -*-
"""Prueba CP_PosesDesdeFotos (blocking en fotos sueltas) con imagenes sinteticas, sin ComfyUI.
Se ejecuta con: python tests/test_poses_fotos.py"""
import inspect
import json
import os
import shutil
import sys
import tempfile
import types

import numpy as np
import torch
from PIL import Image

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
    UploadType = types.SimpleNamespace(image="image_upload")


for _n in ("Image", "Float", "Int", "String", "Boolean", "Combo", "Mask"):
    setattr(_IO, _n, _mk(_n))

latest.io = _IO
mod.latest = latest
sys.modules["comfy_api"] = mod
sys.modules["comfy_api.latest"] = latest

# ---- doble de folder_paths: la carpeta de entrada es una carpeta temporal ----
ENTRADA = tempfile.mkdtemp(prefix="poses_fotos_")
fp = types.ModuleType("folder_paths")
fp.get_input_directory = lambda: ENTRADA


def _anotada(nombre):
    for a in (" [input]",):
        if nombre.endswith(a):
            return nombre[: -len(a)]
    return nombre


def _ruta(nombre):
    ruta = os.path.abspath(os.path.join(ENTRADA, _anotada(nombre)))
    if not ruta.startswith(ENTRADA + os.sep):
        raise ValueError(f"Invalid file path: {nombre!r}")
    return ruta


def _existe(nombre):
    try:
        return os.path.exists(_ruta(nombre))
    except ValueError:
        return False


fp.get_annotated_filepath = _ruta
fp.exists_annotated_filepath = _existe
fp.filter_files_content_types = lambda archivos, tipos: [f for f in archivos if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))]
sys.modules["folder_paths"] = fp

RAIZ = os.path.join(os.path.dirname(__file__), "..", "comfy_character_pipeline")
sys.path.insert(0, RAIZ)
import poses_fotos as P  # noqa: E402

N_CASILLAS = P.CASILLAS


def _guardar(nombre, img, **k):
    ruta = os.path.join(ENTRADA, nombre)
    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    img.save(ruta, **k)
    return nombre


def _color(i):
    return (20 + 9 * i, 200 - 7 * i, (60 * i) % 256)


def _foto(nombre, i, tam=(64, 36)):
    return _guardar(nombre, Image.new("RGB", tam, _color(i)))


def _casillas(**rellenas):
    v = {c: P.MARCADOR_VACIO for c in P.NOMBRES_CASILLAS}
    v.update(rellenas)
    return v


def _ejecutar(d=12, espera=0, orden="casilla", **rellenas):
    return P.PosesDesdeFotos.execute(fotogramas_por_pose=d, espera_final=espera, orden=orden, **_casillas(**rellenas)).args


def _centro(poses, k):
    return tuple(int(round(float(v) * 255)) for v in poses[k, poses.shape[1] // 2, poses.shape[2] // 2])


def _falla(fn, texto, tipo=Exception):
    try:
        fn()
    except tipo as e:
        assert texto in str(e), f"esperaba {texto!r} en {e}"
        assert str(e).startswith("[CP_PosesDesdeFotos]"), str(e)
        return str(e)
    raise AssertionError(f"tenia que fallar con {texto!r}")


# ---------------------------------------------------------------- mapa

def test_mapa_contiguo_desde_cero_y_total():
    for n in (2, 3, 13, 24):
        for d in (8, 12, 16):
            for espera in (0, 1, 12):
                bloques = P.mapa_bloques(n, d, espera)
                total = P.total_fotogramas(n, d, espera)
                assert len(bloques) == n and bloques[0][0] == 0
                assert all(c == b + 1 for (_, b), (c, _) in zip(bloques, bloques[1:]))
                assert all(b - a + 1 == d for a, b in bloques[:-1])
                assert bloques[-1] == [(n - 1) * d, (n - 1) * d + espera]
                assert bloques[-1][1] + 1 == total == (n - 1) * d + 1 + espera


def test_mapa_ejemplo_literal():
    assert P.mapa_bloques(3, 8, 0) == [[0, 7], [8, 15], [16, 16]]
    assert P.mapa_bloques(2, 12, 5) == [[0, 11], [12, 17]]


def test_mapa_con_menos_de_dos_poses_falla():
    _falla(lambda: P.mapa_bloques(1, 12, 0), "al menos dos poses", ValueError)


# ---------------------------------------------------------------- casillas y orden

def test_vacias():
    for v in (None, "", "   ", P.MARCADOR_VACIO, "PON-AQUI-algo.png", "pon-aqui", "example.png", "sub/example.png",
              "subidas/PON-AQUI-UNA-POSE"):
        assert P.es_vacia(v), v
    for v in ("pose_1.png", "ejemplo.png", "sub/pose.png", "pose.png [input]"):
        assert not P.es_vacia(v), v
    assert not P.MARCADOR_VACIO.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".mp4", ".mov"))


def test_huecos_entre_casillas_no_cuentan():
    r = P.casillas_rellenas(_casillas(pose_01="a.png", pose_02="b.png", pose_05="c.png"))
    assert r == [("pose_01", "a.png"), ("pose_02", "b.png"), ("pose_05", "c.png")]


def test_orden_por_casilla_y_por_nombre_natural():
    v = _casillas(pose_01="pose_10.png", pose_02="Pose_2.PNG", pose_03="subidas/pose_1.png", pose_07="pose_02b.jpg")
    assert [x for _, x in P.casillas_rellenas(v, "casilla")] == ["pose_10.png", "Pose_2.PNG", "subidas/pose_1.png", "pose_02b.jpg"]
    assert [x for _, x in P.casillas_rellenas(v, "nombre")] == ["subidas/pose_1.png", "Pose_2.PNG", "pose_02b.jpg", "pose_10.png"]
    _falla(lambda: P.casillas_rellenas(v, "fecha"), "Orden desconocido", ValueError)


def test_nodo_ordena_por_nombre_y_salta_huecos():
    a, b, c = _foto("ord_10.png", 1), _foto("ord_2.png", 2), _foto("ord_1.png", 3)
    poses, mapa, n, tr, total, fps = _ejecutar(d=8, orden="nombre", pose_01=a, pose_03=b, pose_09=c)
    assert n == 3 and tuple(poses.shape) == (3, 36, 64, 3)
    assert [_centro(poses, k) for k in range(3)] == [_color(3), _color(2), _color(1)]
    poses, *_ = _ejecutar(d=8, orden="casilla", pose_01=a, pose_03=b, pose_09=c)
    assert [_centro(poses, k) for k in range(3)] == [_color(1), _color(2), _color(3)]


# ---------------------------------------------------------------- salidas del nodo

def test_salidas_del_nodo():
    nombres = [_foto(f"sal_{i}.png", i) for i in range(1, 14)]
    for d, espera in ((8, 0), (12, 0), (16, 24)):
        rell = {P.NOMBRES_CASILLAS[i]: x for i, x in enumerate(nombres)}
        poses, mapa, n, tr, total, fps = _ejecutar(d=d, espera=espera, **rell)
        assert poses.dtype == torch.float32 and tuple(poses.shape) == (13, 36, 64, 3)
        assert float(poses.min()) >= 0.0 and float(poses.max()) <= 1.0
        assert json.loads(mapa) == {"bloques": P.mapa_bloques(13, d, espera)}
        assert (n, tr, total) == (13, d, 12 * d + 1 + espera)
        assert isinstance(fps, float) and fps == 24.0


def test_veinticuatro_casillas_llenas():
    rell = {c: _foto(f"llena_{i}.png", i) for i, c in enumerate(P.NOMBRES_CASILLAS, start=1)}
    poses, mapa, n, tr, total, fps = _ejecutar(d=16, **rell)
    assert n == N_CASILLAS == 24 and len(json.loads(mapa)["bloques"]) == 24 and total == 23 * 16 + 1


def test_tamanos_distintos_al_de_la_primera():
    a = _guardar("tam_a.png", Image.new("RGB", (192, 108), (200, 30, 30)))
    # vertical: al cubrir y recortar al centro solo queda su franja central (verde); los extremos (azul) se pierden
    vert = Image.new("RGB", (108, 192), (30, 30, 200))
    vert.paste((30, 200, 30), (0, 56, 108, 136))  # 80 px: un poco mas que la franja que queda
    b = _guardar("tam_b.png", vert)
    c = _guardar("tam_c.png", Image.new("RGB", (960, 540), (250, 250, 0)))  # misma proporcion, mas grande
    poses, *_ = _ejecutar(pose_01=a, pose_02=b, pose_03=c)
    assert tuple(poses.shape) == (3, 108, 192, 3)
    g = (poses[1] * 255).round()
    assert float(g[..., 1].mean()) > 190 and float(g[..., 2].max()) < 60, "la vertical tenia que recortarse al centro"
    assert _centro(poses, 2) == (250, 250, 0)
    _, avisos = P.igualar([P.a_tensor(Image.open(os.path.join(ENTRADA, x)))[0] for x in (a, b, c)])
    assert len(avisos) == 1 and "108x192" in avisos[0]


def test_alfa_se_aplana_sobre_gris():
    img = Image.new("RGBA", (64, 36), (0, 0, 0, 0))
    img.paste((255, 0, 0, 255), (16, 8, 48, 28))
    a = _guardar("alfa.png", img)
    b = _foto("alfa_b.png", 2)
    poses, *_ = _ejecutar(pose_01=a, pose_02=b)
    gris = int(round(P.GRIS_ALFA * 255))
    assert _centro(poses, 0) == (255, 0, 0)
    assert tuple(int(round(float(v) * 255)) for v in poses[0, 0, 0]) == (gris, gris, gris)
    x, tenia = P.a_tensor(Image.open(os.path.join(ENTRADA, a)))
    assert tenia
    _, tenia = P.a_tensor(Image.new("RGBA", (8, 8), (1, 2, 3, 255)))
    assert not tenia, "un alfa opaco no es transparencia"


def test_exif_se_respeta():
    img = Image.new("RGB", (80, 40), (0, 0, 255))
    img.paste((255, 0, 0), (0, 0, 40, 40))  # mitad izquierda roja
    exif = img.getexif()
    exif[0x0112] = 6  # girar 90 grados en el sentido del reloj al mostrar
    a = _guardar("exif.jpg", img, exif=exif, quality=95)
    x, _ = P.a_tensor(Image.open(os.path.join(ENTRADA, a)))
    assert tuple(x.shape) == (1, 80, 40, 3), tuple(x.shape)
    arriba, abajo = x[0, 10, 20], x[0, 70, 20]
    assert arriba[0] > 0.8 and arriba[2] < 0.2 and abajo[2] > 0.8, "la mitad roja tiene que quedar arriba"


def test_gris_de_16_bits():
    x, _ = P.a_tensor(Image.fromarray(np.full((10, 12), 65535, np.uint16)))
    assert tuple(x.shape) == (1, 10, 12, 3) and abs(float(x.mean()) - 1.0) < 1e-6


# ---------------------------------------------------------------- errores

def test_cero_o_una_foto_falla():
    a = _foto("sola.png", 1)
    _falla(lambda: _ejecutar(), "al menos dos poses", ValueError)
    _falla(lambda: _ejecutar(pose_04=a), "al menos dos poses", ValueError)


def test_archivo_que_no_existe_dice_casilla_y_nombre():
    a = _foto("existe.png", 1)
    e = _falla(lambda: _ejecutar(pose_01=a, pose_03="no_esta.png"), "pose_03", FileNotFoundError)
    assert "no_esta.png" in e
    _falla(lambda: _ejecutar(pose_01=a, pose_02="../fuera.png"), "pose_02", FileNotFoundError)


def test_archivo_que_no_es_imagen():
    a = _foto("buena.png", 1)
    with open(os.path.join(ENTRADA, "rota.png"), "wb") as f:
        f.write(b"no soy un png")
    e = _falla(lambda: _ejecutar(pose_01=a, pose_02="rota.png"), "no se puede leer", ValueError)
    assert "pose_02" in e and "rota.png" in e
    # sin la ruta del trabajador: PIL la pone en su mensaje y aqui se cambia por el nombre de la casilla
    assert ENTRADA not in e and os.path.realpath(ENTRADA) not in e, e
    try:
        _ejecutar(pose_01=a, pose_02="rota.png")
    except ValueError as err:
        assert err.__cause__ is None and err.__suppress_context__, "el error de PIL (con la ruta) no se encadena"


def test_sin_ruta_cambia_la_ruta_por_el_valor():
    ruta = os.path.join(ENTRADA, "sub", "x.png")
    assert P.sin_ruta(f"cannot identify image file '{ruta}'", ruta, "sub/x.png") == "cannot identify image file 'sub/x.png'"
    assert P.sin_ruta("sin ruta", ruta, "x.png") == "sin ruta"


class _Avisos:
    """Recoge los logging.warning del nodo mientras dura el with."""

    def __enter__(self):
        import logging

        self.textos = []
        avisos = self

        class _H(logging.Handler):
            def emit(self, r):
                avisos.textos.append(r.getMessage())

        self.h = _H(level=logging.WARNING)
        logging.getLogger().addHandler(self.h)
        return self

    def __exit__(self, *a):
        import logging

        logging.getLogger().removeHandler(self.h)


def test_archivo_real_con_nombre_de_marcador_avisa():
    a, b, c = _foto("marc_a.png", 1), _foto("marc_b.png", 2), _foto("example.pose.png", 3)
    with _Avisos() as av:
        poses, mapa, n, *_ = _ejecutar(pose_01=a, pose_02="example.pose.png", pose_03=b)
    assert n == 2, "un nombre que empieza por example. sigue contando como casilla vacia (convenio de marcadores)"
    assert any("pose_02" in t and "example.pose.png" in t and "Renombrala" in t for t in av.textos), av.textos
    # el marcador del nodo, un example que no existe y las casillas vacias no avisan
    with _Avisos() as av:
        _ejecutar(pose_01=a, pose_02=P.MARCADOR_VACIO, pose_03="example.png", pose_04="  ", pose_05=b)
    assert not any("Renombrala" in t for t in av.textos), av.textos
    assert P.parecen_marcador({"pose_01": "PON-AQUI-OTRA", "pose_02": P.MARCADOR_VACIO, "pose_03": "x.png"}) == [("pose_01", "PON-AQUI-OTRA")]


def test_primera_foto_enorme_reduce_el_lote_sin_deformar():
    assert P.tamano_lote(1088, 1920) == (1088, 1920, None)
    assert P.tamano_lote(2048, 1000)[:2] == (2048, 1000)
    alto, ancho, aviso = P.tamano_lote(4000, 6000)
    assert (alto, ancho) == (1365, 2048) and "6000x4000" in aviso and "2048x1365" in aviso
    alto, ancho, _ = P.tamano_lote(6000, 3000)
    assert (alto, ancho) == (2048, 1024)
    grande = _guardar("enorme.png", Image.new("RGB", (3000, 1500), (10, 200, 10)))
    otra = _guardar("normal.png", Image.new("RGB", (192, 96), (200, 10, 10)))
    with _Avisos() as av:
        poses, *_ = _ejecutar(pose_01=grande, pose_02=otra)
    assert tuple(poses.shape) == (2, 1024, 2048, 3), tuple(poses.shape)
    assert _centro(poses, 0) == (10, 200, 10) and _centro(poses, 1) == (200, 10, 10)
    assert any("3000x1500" in t and "2048x1024" in t for t in av.textos), av.textos
    # igualar aplica el mismo tope
    lote, avisos = P.igualar([torch.zeros(1, 1500, 3000, 3), torch.zeros(1, 96, 192, 3)])
    assert tuple(lote.shape) == (2, 1024, 2048, 3) and len(avisos) == 1 and "2048" in avisos[0]


def test_imagen_de_cero_pixeles():
    _falla(lambda: P.a_tensor(Image.new("RGB", (0, 0))), "0x0", ValueError)


# ---------------------------------------------------------------- esquema, validacion y huella

def test_esquema():
    s = P.PosesDesdeFotos.define_schema()
    assert s["node_id"] == "CP_PosesDesdeFotos" and s["category"] == "character-pipeline/poses"
    entradas = s["inputs"]
    casillas = entradas[:N_CASILLAS]
    assert [x.a[0] for x in casillas] == [f"pose_{i:02d}" for i in range(1, 25)]
    for x in casillas:
        assert x.k["upload"] == "image_upload" and x.k["default"] == P.MARCADOR_VACIO and x.k["optional"] is True
        assert x.k["options"][0] == P.MARCADOR_VACIO
    resto = {x.a[0]: x.k for x in entradas[N_CASILLAS:]}
    assert resto["fotogramas_por_pose"]["default"] == 12 and (resto["fotogramas_por_pose"]["min"], resto["fotogramas_por_pose"]["max"]) == (2, 32)
    assert resto["espera_final"]["default"] == 0 and (resto["espera_final"]["min"], resto["espera_final"]["max"]) == (0, 240)
    assert resto["orden"]["options"] == ["casilla", "nombre"] and resto["orden"]["default"] == "casilla"
    assert [x.k["display_name"] for x in s["outputs"]] == ["poses", "mapa", "n_poses", "transicion", "fotogramas", "fps"]


def test_lista_de_la_casilla_trae_las_imagenes_de_la_carpeta():
    _foto("lista.png", 1)
    with open(os.path.join(ENTRADA, "notas.txt"), "w") as f:
        f.write("x")
    opciones = P.PosesDesdeFotos.define_schema()["inputs"][0].k["options"]
    assert "lista.png" in opciones and "notas.txt" not in opciones and P.MARCADOR_VACIO == opciones[0]


def test_validate_inputs_nombra_solo_las_casillas():
    spec = inspect.getfullargspec(P.PosesDesdeFotos.validate_inputs)
    assert spec.args == ["cls"] + [f"pose_{i:02d}" for i in range(1, 25)] and spec.varkw is None
    assert P.PosesDesdeFotos.validate_inputs(pose_01="recien_subida.png", pose_02=P.MARCADOR_VACIO) is True


def test_huella_cambia_si_se_sube_otra_foto_con_el_mismo_nombre():
    nombre = _foto("misma.png", 1)
    kw = _casillas(pose_01=nombre, pose_02=_foto("otra.png", 2))
    h1 = P.PosesDesdeFotos.fingerprint_inputs(fotogramas_por_pose=12, **kw)
    assert h1 == P.PosesDesdeFotos.fingerprint_inputs(fotogramas_por_pose=12, **kw)
    _foto("misma.png", 5)
    h2 = P.PosesDesdeFotos.fingerprint_inputs(fotogramas_por_pose=12, **kw)
    assert h1 != h2
    kw["pose_03"] = "no_esta.png"
    P.PosesDesdeFotos.fingerprint_inputs(**kw)  # no revienta: el error lo da execute


# ---------------------------------------------------------------- encaje con CP_LentaPlan

def test_encaje_con_el_plan_de_camara_lenta():
    try:
        import lenta as L
    except Exception as e:  # lenta.py lo esta cambiando otra persona
        raise AssertionError(f"no se pudo importar lenta.py (cambio a medias?): {type(e).__name__}: {e}") from e
    for n in (2, 3, 13, 24):
        for d in (8, 12, 16):
            for espera in (0, 12):
                mapa = json.dumps({"bloques": P.mapa_bloques(n, d, espera)})
                inicios, total = L.leer_mapa(mapa, n)
                assert inicios == [i * d for i in range(n)] and total == P.total_fotogramas(n, d, espera)
                # como la tarjeta de fotos: mas_lentos encendido
                plan = L.planificar(inicios, total, [5.0] * (n - 1), transicion_max=d, mas_lentos=True)
                assert plan["total"] == P.total_fotogramas(n, d, espera)
                assert all(x >= d for x in plan["lentos"]), "cada salto tiene que dar para d fotogramas de movimiento"


def test_los_topes_del_nodo_nunca_pasan_los_del_plan():
    """Con los valores maximos del nodo (24 casillas, fotogramas_por_pose y espera_final al tope) el plan sale con
    ritmo 1, sin pasar del tope de salida y con 32 lentos por salto como mucho (lo validado en GPU)."""
    import lenta as L
    n, d, espera = P.CASILLAS, P.MAX_FOTOGRAMAS_POR_POSE, P.MAX_ESPERA_FINAL
    inicios, total = L.leer_mapa(json.dumps({"bloques": P.mapa_bloques(n, d, espera)}), n)
    assert total == P.total_fotogramas(n, d, espera) <= L.MAX_SALIDA, total
    plan = L.planificar(inicios, total, [20.0] * (n - 1), transicion_max=d, mas_lentos=True)
    assert plan["total"] == total and max(plan["lentos"]) == L.LENTOS_LARGO, plan["lentos"]
    assert "AVISO" not in L.resumir_salida(plan)
    # uno mas por pose ya pediria 40 lentos por salto, que no estan validados
    assert L.lentos_por_salto([1.0], [d + 1], L.UMBRAL_SALTO, d + 1, mas_margen=True) == [40]


if __name__ == "__main__":
    try:
        pruebas = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
        for t in pruebas:
            t()
            print("ok", t.__name__)
        print(len(pruebas), "pruebas")
    finally:
        shutil.rmtree(ENTRADA, ignore_errors=True)
