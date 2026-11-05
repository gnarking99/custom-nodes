# -*- coding: utf-8 -*-
"""Prueba el render por poses con secuencias sinteticas de movimiento conocido, sin ComfyUI.
Se ejecuta con: python tests/test_poses.py"""
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
        pass


def _mk(name):
    return type(name, (), {"Input": staticmethod(_Port), "Output": staticmethod(_Port)})


class _IO:
    Schema = staticmethod(lambda **k: k)
    NodeOutput = staticmethod(lambda *a, **k: a)
    ComfyNode = type("ComfyNode", (), {})
    Image = _mk("Image")
    Float = _mk("Float")
    Int = _mk("Int")
    String = _mk("String")
    Boolean = _mk("Boolean")


latest.io = _IO
mod.latest = latest
sys.modules["comfy_api"] = mod
sys.modules["comfy_api.latest"] = latest

RAIZ = os.path.join(os.path.dirname(__file__), "..", "comfy_character_pipeline")
sys.path.insert(0, RAIZ)
import poses as P  # noqa: E402

H, W = 270, 480
ORANGE = (0.9, 0.45, 0.1)
ROJO = (0.5, 0.05, 0.05)


def _fondo():
    grad = np.linspace(0.65, 0.1, H, dtype=np.float32)[:, None, None]
    f = np.broadcast_to(grad, (H, W, 3)).copy()
    f[200:240, 340:400] = 0.02  # un objeto quieto: la bolsa
    return f


def _cuadro(x, y, s=40, color=ORANGE, fondo=None):
    f = (_fondo() if fondo is None else fondo).copy()
    f[y:y + s, x:x + s] = color
    return f


def _clip(posiciones):
    return torch.from_numpy(np.stack([_cuadro(x, y) for x, y in posiciones]))


def test_detecta_animacion_a_dos():
    pos = []
    for k in range(6):
        pos += [(60 + k * 20, 100)] * 2
    pos.append((180, 100))  # ultima pose suelta
    b = P.PosesDetectar.execute(_clip(pos), 0.5, 300)
    mapa = json.loads(b[1])["bloques"]
    assert b[2] == 7 and b[0].shape[0] == 7, b[2]
    assert mapa == [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 12]], mapa


def test_movimiento_lento_abre_pose_aunque_cada_paso_sea_pequeno():
    # Cada fotograma se mueve poco: comparando con el anterior nunca se abriria pose nueva.
    frames = np.stack([_cuadro(60 + t // 3, 100) for t in range(90)], axis=0) * 255.0
    b = P.detectar_bloques(frames.astype(np.float32), 0.5)
    assert len(b) > 1, b


def test_tope_de_poses_da_error_claro():
    pos = [(30 + k * 40, 100) for k in range(10)]
    try:
        P.PosesDetectar.execute(_clip(pos), 0.5, 5)
    except ValueError as e:
        assert "10 poses" in str(e)
    else:
        raise AssertionError("tenia que fallar")


def test_placa_sin_rastro_aunque_el_personaje_se_quede_mas_de_la_mitad():
    # El personaje pasa el 70 % de las poses en el mismo sitio: la mediana lo dejaria en la placa.
    pos = [(150, 100)] * 7 + [(30, 60), (250, 60), (300, 150)]
    poses = np.stack([_cuadro(x, y) for x, y in pos]).astype(np.float32) * 255.0
    placa = P.placa_playblast(poses)
    assert np.abs(placa - _fondo() * 255.0).max() < 3.0, np.abs(placa - _fondo() * 255.0).max()


def test_placa_sin_rastro_con_un_barrido_grande():
    # Un personaje enorme: en el centro de lo que barre no hay ningun pixel quieto cerca, y un
    # personaje con partes oscuras no puede colarse en la placa por parecerse a una estimacion mala.
    poses = []
    for x, y in [(60, 40), (100, 50), (140, 60), (180, 70), (220, 60), (140, 40), (60, 70), (240, 40)]:
        f = _fondo()
        f[y:y + 150, x:x + 150] = (0.03, 0.03, 0.03)
        f[y + 50:y + 100, x + 50:x + 100] = ORANGE
        poses.append(f)
    poses = np.stack(poses).astype(np.float32) * 255.0
    placa = P.placa_playblast(poses)
    fondo = _fondo() * 255.0
    visto = (np.abs(poses - fondo).max(axis=-1) < 1).sum(axis=0) >= 2
    error = np.abs(placa - fondo).max(axis=-1)[visto]
    assert error.max() < 3.0, error.max()


def test_placa_de_un_decorado_con_textura():
    # Sin telon liso: el valor mas repetido, aunque el personaje este en muchas poses.
    rng = np.random.default_rng(3)
    fondo = rng.random((H // 4, W // 4, 3)).astype(np.float32).repeat(4, axis=0).repeat(4, axis=1)
    poses = np.stack([_cuadro(60 + k * 25, 100, fondo=fondo) for k in range(10)]).astype(np.float32) * 255.0
    placa = P.placa_playblast(poses)
    assert P.telon_liso(poses.mean(axis=0), (poses.max(axis=0) - poses.min(axis=0)).max(axis=-1) < 6) is None
    error = np.abs(placa - fondo * 255.0).max(axis=-1)
    assert np.percentile(error, 99.5) < 16.0, np.percentile(error, 99.5)


def test_mascara_quita_motas_y_deja_el_trozo_que_vuela():
    fondo = _fondo() * 255.0
    p = fondo.copy()
    p[100:140, 100:140] = 200  # personaje
    p[30:40, 300:310] = 200    # lo que vuela: 100 px a esta escala
    p[60, 60] = 250            # mota
    m = P.mascara_pose(p.astype(np.float32), fondo.astype(np.float32), 18.0)
    assert m[120, 120] and m[35, 305] and not m[60, 60]


def _pintar(poses_tensor, semilla=0):
    """Hace de Klein: colores nuevos, y el fondo y la bolsa cambian un poco en cada pose."""
    rng = np.random.default_rng(semilla)
    pintadas = []
    for i in range(len(poses_tensor)):
        f = poses_tensor[i].numpy().copy()
        es_personaje = (f[..., 0] > 0.8) & (f[..., 2] < 0.2)
        f = f * (0.97 + 0.06 * rng.random())
        f[es_personaje] = ROJO
        pintadas.append(f)
    return torch.from_numpy(np.stack(pintadas).astype(np.float32))


def test_montar_conserva_cadencia_y_fija_el_fondo():
    pos = []
    for k in range(6):
        pos += [(60 + k * 30, 100)] * 2
    clip = _clip(pos)
    salida = P.PosesDetectar.execute(clip, 0.5, 300)
    pintadas = _pintar(salida[0])
    img = P.PosesMontar.execute(pintadas, clip, salida[1], True, False, 18.0)[0]
    assert img.shape == clip.shape, img.shape
    a = img.numpy()
    # A dos: cada pareja identica, y el personaje donde lo puso el playblast, con el color pintado.
    for k in range(6):
        assert np.abs(a[2 * k] - a[2 * k + 1]).max() == 0
        x = 60 + k * 30
        assert np.abs(a[2 * k][120, x + 20] - np.array(ROJO)).max() < 0.02
    # El fondo no parpadea aunque la pintura cambie en cada pose.
    assert np.abs(a[0][20:60, 400:470] - a[10][20:60, 400:470]).max() < 1e-4


def test_sombras_solo_oscurecen_y_junto_a_los_pies():
    pos = [(60, 100), (60, 100), (200, 100), (200, 100), (340 - 100, 150), (340 - 100, 150)]
    clip = _clip(pos)
    salida = P.PosesDetectar.execute(clip, 0.5, 300)
    pintadas = _pintar(salida[0]).numpy()
    pintadas[1][140:170, 190:250] *= 0.5  # sombra bajo la pose del centro
    pintadas[1][5:40, 190:250] *= 0.5     # mancha arriba, lejos de los pies: no debe pasar
    pintadas = torch.from_numpy(pintadas)
    sin = P.PosesMontar.execute(pintadas, clip, salida[1], True, False, 18.0)[0].numpy()
    con = P.PosesMontar.execute(pintadas, clip, salida[1], True, True, 18.0)[0].numpy()
    assert (con <= sin + 1e-3).all()
    assert con[2][150, 220].mean() < sin[2][150, 220].mean() * 0.85
    assert abs(con[2][20, 220].mean() - sin[2][20, 220].mean()) < 0.01


def test_un_reflejo_de_la_ultima_pose_no_se_queda_en_todo_el_plano():
    pos = [(60 + k * 30, 100) for k in range(6) for _ in range(2)]
    clip = _clip(pos)
    salida = P.PosesDetectar.execute(clip, 0.5, 300)
    pintadas = _pintar(salida[0]).numpy()
    # Junto a los pies de la ultima pose, fuera de su mascara: un reflejo rojizo que solo pinta ella.
    pintadas[-1][144:170, 220:250] = (0.6, 0.2, 0.15)
    img = P.PosesMontar.execute(torch.from_numpy(pintadas), clip, salida[1], True, False, 18.0)[0].numpy()
    fondo = _fondo()
    assert np.abs(img[0][155, 235] - fondo[155, 235]).max() < 0.06, img[0][155, 235]


def test_sin_fondo_fijo_son_las_poses_tal_cual():
    pos = [(60, 100)] * 2 + [(120, 100)] * 3
    clip = _clip(pos)
    salida = P.PosesDetectar.execute(clip, 0.5, 300)
    pintadas = _pintar(salida[0])
    img = P.PosesMontar.execute(pintadas, clip, salida[1], False, True, 18.0)[0]
    assert img.shape[0] == 5
    assert torch.equal(img[0], pintadas[0]) and torch.equal(img[4], pintadas[1])


def test_camara_movil_no_se_rompe_y_no_fija_el_fondo():
    rng = np.random.default_rng(1)
    base = rng.random((H, W + 200, 3)).astype(np.float32)
    clip = torch.from_numpy(np.stack([base[:, k * 40:k * 40 + W] for k in range(5)]))
    salida = P.PosesDetectar.execute(clip, 0.5, 300)
    assert salida[2] == 5
    pintadas = _pintar(salida[0])
    img = P.PosesMontar.execute(pintadas, clip, salida[1], True, True, 18.0)[0]
    assert torch.equal(img, pintadas)


def test_todo_quieto_da_una_sola_pose_y_no_falla():
    clip = _clip([(60, 100)] * 8)
    salida = P.PosesDetectar.execute(clip, 0.5, 300)
    assert salida[2] == 1
    img = P.PosesMontar.execute(_pintar(salida[0]), clip, salida[1], True, True, 18.0)[0]
    assert img.shape[0] == 8


def test_salida_a_mas_resolucion_que_el_playblast():
    pos = [(60 + k * 30, 100) for k in range(4) for _ in range(2)]
    clip = _clip(pos)
    salida = P.PosesDetectar.execute(clip, 0.5, 300)
    grande = torch.nn.functional.interpolate(_pintar(salida[0]).movedim(-1, 1), size=(608, 1080), mode="bilinear").movedim(1, -1)
    img = P.PosesMontar.execute(grande, clip, salida[1], True, True, 18.0)[0]
    assert img.shape == (8, 608, 1080, 3), img.shape
    assert float(img.min()) >= 0.0 and float(img.max()) <= 1.0


def test_mapa_malo_da_error_claro():
    clip = _clip([(60, 100)] * 4)
    pintadas = _pintar(clip[:1])
    for mapa, texto in (("no es json", "no se puede leer"),
                        (json.dumps({"bloques": [[0, 1], [2, 3]]}), "2"),
                        (json.dumps({"bloques": [[0, 1]]}), "no cubre")):
        try:
            P.PosesMontar.execute(pintadas, clip, mapa, True, True, 18.0)
        except ValueError as e:
            assert texto in str(e), (texto, str(e))
        else:
            raise AssertionError("tenia que fallar: " + mapa)


def test_guia_sin_esperas_cae_en_cada_pose_y_no_repite():
    # Los tiempos reales de 0003: cada pose en su fotograma y ningun fotograma quieto entre P.
    inicios = [0, 4, 8, 13, 16, 18, 20, 24, 28, 32, 34, 36, 40, 44, 48, 50, 54, 56, 60, 62, 64, 68, 70, 72, 76]
    bloques = [(a, b - 1) for a, b in zip(inicios, inicios[1:])] + [(76, 76)]
    factor = P.factor_intermedios(bloques)
    assert factor == 12
    idx = P.indices_sin_esperas(bloques, factor)
    assert len(idx) == 77
    for k, a in enumerate(inicios):
        assert idx[a] == 12 * k, (k, a, idx[a])
    assert all(y > x for x, y in zip(idx, idx[1:])), idx


def test_saltos_largos_suben_el_factor_sin_pasar_del_tope():
    assert P.factor_intermedios([(0, 13), (14, 20)]) == 14
    assert P.factor_intermedios([(0, 39), (40, 50)]) == 16
    assert P.factor_intermedios([(0, 50)]) == 12


def test_fotogramas_ltx_no_corta_el_plano():
    assert P.fotogramas_ltx(77) == 81
    assert P.fotogramas_ltx(81) == 81
    assert P.fotogramas_ltx(82) == 89
    assert P.fotogramas_ltx(1) == 1


def test_nodo_guia_rellena_y_avisa_si_no_cuadra():
    bloques = [(0, 3), (4, 8)]
    mapa = json.dumps({"bloques": bloques})
    inter = torch.arange(13, dtype=torch.float32).reshape(13, 1, 1, 1).repeat(1, 2, 2, 3)
    guia, n, n_ltx, _ = P.PosesSinEsperas.execute(inter, mapa, 12)
    assert (n, n_ltx) == (9, 9) and len(guia) == 9
    assert [int(g[0, 0, 0]) for g in guia] == [0, 3, 6, 9, 12, 12, 12, 12, 12]
    try:
        P.PosesSinEsperas.execute(inter[:5], mapa, 12)
    except ValueError as e:
        assert "son 13" in str(e)
    else:
        raise AssertionError("tenia que fallar con menos intermedios")


if __name__ == "__main__":
    pruebas = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in pruebas:
        t()
        print("ok", t.__name__)
    print(len(pruebas), "pruebas")
