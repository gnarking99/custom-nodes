# -*- coding: utf-8 -*-
"""Prueba la lectura tolerante de etiquetas de video sin PyAV real.
Se ejecuta con: python tests/test_compat_av.py"""
import os
import sys
import types

RAIZ = os.path.join(os.path.dirname(__file__), "..", "comfy_character_pipeline")
sys.path.insert(0, RAIZ)
import compat_av as C  # noqa: E402


def _av_antiguo():
    """Como PyAV <= 17: acepta metadata_errors y, si no se lo dan, decodifica estricto y falla."""
    av = types.ModuleType("av")
    llamadas = []

    def open(file, mode="r", metadata_errors="strict", **k):  # noqa: A001
        llamadas.append({"file": file, "mode": mode, "metadata_errors": metadata_errors, **k})
        if metadata_errors == "strict":
            b"Gestor de contenido de v\xeddeo Apple".decode("utf-8")
        return ("contenedor", file, mode)

    av.open = open
    av.container = types.SimpleNamespace(open=open)
    av.llamadas = llamadas
    return av


def _av_nuevo():
    """Como PyAV 19: ya no existe metadata_errors."""
    av = types.ModuleType("av")
    llamadas = []

    def open(file, mode="r", **k):  # noqa: A001
        av.intentos += 1
        if "metadata_errors" in k:
            raise TypeError("open() got an unexpected keyword argument 'metadata_errors'")
        llamadas.append({"file": file, "mode": mode, **k})
        return ("contenedor", file, mode)

    av.open = open
    av.llamadas = llamadas
    av.intentos = 0
    return av


def test_con_pyav_antiguo_el_video_abre_y_sin_el_arreglo_no():
    av = _av_antiguo()
    try:
        av.open("blocking.mov")
        raise AssertionError("el doble tenia que fallar como PyAV 17")
    except UnicodeDecodeError as e:
        assert "0xed in position 24" in str(e), e
    assert C.tolerar_metadatos_de_video(av) == "puesto"
    assert av.open("blocking.mov", mode="r") == ("contenedor", "blocking.mov", "r")
    assert av.llamadas[-1]["metadata_errors"] == "replace"
    assert av.container.open is av.open  # los dos nombres apuntan al mismo sitio


def test_quien_pide_su_propio_modo_lo_conserva():
    av = _av_antiguo()
    C.tolerar_metadatos_de_video(av)
    av.open("x.mp4", metadata_errors="ignore")
    assert av.llamadas[-1]["metadata_errors"] == "ignore"
    try:
        av.open("blocking.mov", metadata_errors="strict")
        raise AssertionError("pedir strict tiene que seguir fallando")
    except UnicodeDecodeError:
        pass


def test_con_pyav_nuevo_no_cambia_nada_y_solo_prueba_una_vez():
    av = _av_nuevo()
    assert C.tolerar_metadatos_de_video(av) == "puesto"
    assert av.open("a.mp4") == ("contenedor", "a.mp4", "r")
    assert av.open("b.mp4", mode="w", format="mp4") == ("contenedor", "b.mp4", "w")
    assert av.llamadas == [{"file": "a.mp4", "mode": "r"}, {"file": "b.mp4", "mode": "w", "format": "mp4"}]
    assert av.intentos == 3, av.intentos  # el sondeo con metadata_errors y las dos llamadas de verdad: no vuelve a sondear


def test_indicar_el_modo_no_apaga_la_proteccion():
    """Ni por nombre ni por posicion (8.o argumento): la llamada siguiente sigue pidiendo 'replace'."""
    av = _av_antiguo()
    orig = av.open

    def open(file, mode="r", format=None, options=None, container_options=None, stream_options=None,  # noqa: A001, A002
             metadata_encoding="utf-8", metadata_errors="strict", **k):
        return orig(file, mode=mode, metadata_errors=metadata_errors, **k)

    av.open = open
    av.container = types.SimpleNamespace(open=open, core=types.SimpleNamespace(open=open))
    C.tolerar_metadatos_de_video(av)
    assert av.container.core.open is av.open
    av.open("x.mp4", "r", None, None, None, None, "utf-8", "ignore")
    assert av.llamadas[-1]["metadata_errors"] == "ignore"
    av.open("y.mp4", metadata_errors="ignore")
    av.open("blocking.mov")
    assert av.llamadas[-1]["metadata_errors"] == "replace"


def test_un_typeerror_ajeno_no_apaga_la_proteccion():
    av = _av_antiguo()
    orig = av.open

    def open(file, mode="r", metadata_errors="strict", **k):  # noqa: A001
        if file is None:
            raise TypeError("open() got multiple values for keyword argument 'metadata_errors'")
        return orig(file, mode=mode, metadata_errors=metadata_errors, **k)

    av.open = open
    C.tolerar_metadatos_de_video(av)
    try:
        av.open(None)
        raise AssertionError("tenia que salir el TypeError tal cual")
    except TypeError as e:
        assert "multiple values" in str(e)
    av.open("blocking.mov")
    assert av.llamadas[-1]["metadata_errors"] == "replace"


def test_no_se_envuelve_dos_veces():
    av = _av_antiguo()
    assert C.tolerar_metadatos_de_video(av) == "puesto"
    primero = av.open
    assert C.tolerar_metadatos_de_video(av) == "ya estaba"
    assert av.open is primero


def test_otros_errores_pasan_tal_cual():
    av = types.ModuleType("av")

    def open(file, mode="r", metadata_errors="strict", **k):  # noqa: A001
        if file is None:
            raise TypeError("file must be str, not None")
        raise FileNotFoundError(file)

    av.open = open
    C.tolerar_metadatos_de_video(av)
    for arg, tipo in ((None, TypeError), ("no_existe.mov", FileNotFoundError)):
        try:
            av.open(arg)
            raise AssertionError("tenia que fallar")
        except tipo:
            pass


def test_sin_pyav_el_pack_carga_igual():
    roto = types.SimpleNamespace()  # sin .open
    assert C.tolerar_metadatos_de_video(roto) == "sin av"


if __name__ == "__main__":
    n = 0
    for nombre, f in sorted(globals().items()):
        if nombre.startswith("test_") and callable(f):
            f()
            print("ok", nombre)
            n += 1
    print(n, "pruebas")
