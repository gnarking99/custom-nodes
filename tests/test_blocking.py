# -*- coding: utf-8 -*-
"""Prueba las piezas de blocking a animacion con datos sinteticos, sin ComfyUI.
Se ejecuta con: python tests/test_blocking.py"""
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


class _Salida:
    def __init__(self, *a, **k):
        self.args = a


class _IO:
    Schema = staticmethod(lambda **k: k)
    NodeOutput = _Salida
    ComfyNode = type("ComfyNode", (), {})


for _n in ("Image", "Float", "Int", "String", "Boolean", "Mask", "Conditioning", "Vae", "Latent"):
    setattr(_IO, _n, _mk(_n))

latest.io = _IO
mod.latest = latest
sys.modules["comfy_api"] = mod
sys.modules["comfy_api.latest"] = latest

RAIZ = os.path.join(os.path.dirname(__file__), "..", "comfy_character_pipeline")
sys.path.insert(0, RAIZ)
import blocking as B  # noqa: E402
import poses as P  # noqa: E402

INICIOS_0003 = [0, 4, 8, 13, 16, 18, 20, 24, 28, 32, 34, 36, 40, 44, 48, 50, 54, 56, 60, 62, 64, 68, 70, 72, 76]
BLOQUES_0003 = [(a, b - 1) for a, b in zip(INICIOS_0003, INICIOS_0003[1:])] + [(76, 76)]


def test_claves_reparte_y_guarda_primera_y_ultima():
    idx = B.elegir_claves(BLOQUES_0003, 6)
    assert len(idx) == 6 and idx == sorted(set(idx)), idx
    assert idx[0] == 0 and idx[-1] == len(INICIOS_0003) - 1
    fot = [INICIOS_0003[i] for i in idx]
    huecos = np.diff(fot)
    assert huecos.max() <= 20, fot  # repartidas: ningun hueco mayor que el doble del ideal (15)
    assert B.elegir_claves(BLOQUES_0003, 40) == list(range(25))


def test_nodo_claves_da_fotogramas_al_doble_de_fps():
    poses = torch.rand(25, 8, 8, 3)
    claves, fot, n = B.PosesClaves.execute(poses, json.dumps({"bloques": BLOQUES_0003}), 6, 2).args
    fot = json.loads(fot)
    assert n == 6 and len(claves) == 6 and fot[0] == 0 and fot[-1] == 152, fot
    try:
        B.PosesClaves.execute(poses[:3], json.dumps({"bloques": BLOQUES_0003}), 6, 2)
    except ValueError as e:
        assert "25" in str(e)
    else:
        raise AssertionError("tenia que fallar con menos poses que el mapa")


def _telon(h, w):
    g = np.linspace(0.75, 0.15, h, dtype=np.float32)[:, None, None]
    return np.broadcast_to(g * np.array([0.55, 0.65, 0.85], np.float32), (h, w, 3)).copy()


def test_componer_pone_el_personaje_sobre_la_placa_y_quita_el_telon():
    h, w = 270, 480
    img = _telon(h, w)
    img[100:180, 200:260] = (0.8, 0.4, 0.2)  # el personaje
    placa = np.zeros((h, w, 3), np.float32)
    placa[..., 1] = 0.6  # decorado verde
    sal, masc = B.componer(torch.from_numpy(img)[None], torch.from_numpy(placa)[None])
    s = sal[0].numpy()
    assert np.allclose(s[140, 230], (0.8, 0.4, 0.2), atol=0.03), s[140, 230]
    assert abs(s[20, 20, 1] - 0.6) < 0.02 and s[20, 20, 2] < 0.02  # lejos: placa intacta, nada del telon azul
    assert 0.45 < s[183, 230, 1] < 0.59  # bajo los pies: sombra de contacto, suave
    assert masc[0, 140, 230] > 0.95 and masc[0, 20, 20] < 0.01


def test_componer_escala_la_placa_a_las_poses():
    img = torch.from_numpy(_telon(64, 96))[None].repeat(2, 1, 1, 1)
    placa = torch.rand(1, 200, 200, 3)
    sal, masc = B.componer(img, placa)
    assert sal.shape == (2, 64, 96, 3) and masc.shape == (2, 64, 96)


def test_guias_llama_una_vez_por_pose_en_orden():
    llamadas = []

    class Falsa:
        @classmethod
        def execute(cls, positive, negative, vae, latent, image, frame_idx, strength, factor, crop, *resto):
            llamadas.append((frame_idx, strength, factor, crop, tuple(image.shape)))
            return _Salida(positive + [frame_idx], negative, {"n": latent["n"] + 1})

    nodes = types.ModuleType("nodes")
    nodes.NODE_CLASS_MAPPINGS = {"LTXAddVideoICLoRAGuide": Falsa}
    sys.modules["nodes"] = nodes
    imgs = torch.rand(3, 4, 4, 3)
    pos, neg, lat = B.LTXGuiasPoses.execute([], [], None, {"n": 0}, imgs, "[0, 64, 152]", 2.0, 1.0).args
    assert pos == [0, 64, 152] and lat == {"n": 3}
    assert [c[0] for c in llamadas] == [0, 64, 152] and all(c[2] == 2.0 and c[3] == "center" for c in llamadas)
    assert all(c[4] == (1, 4, 4, 3) for c in llamadas)
    try:
        B.LTXGuiasPoses.execute([], [], None, {"n": 0}, imgs, "[0, 64]", 1.0, 1.0)
    except ValueError as e:
        assert "3 imagenes" in str(e)
    else:
        raise AssertionError("tenia que fallar si no cuadran imagenes y fotogramas")


def test_guia_al_doble_de_fps_cae_en_cada_pose():
    idx = P.indices_sin_esperas(BLOQUES_0003, 12, 2)
    assert len(idx) == 153
    for k, a in enumerate(INICIOS_0003):
        assert idx[2 * a] == 12 * k
    assert all(y > x for x, y in zip(idx, idx[1:])), "con saltos de hasta 5 fotogramas no se repite ninguno"
    inter = torch.arange(13, dtype=torch.float32).reshape(13, 1, 1, 1).repeat(1, 2, 2, 3)
    guia, n, n_ltx, fps = P.PosesSinEsperas.execute(inter, json.dumps({"bloques": [(0, 3), (4, 8)]}), 12, 2, 24.0).args
    assert fps == 48.0
    assert (n, n_ltx, len(guia)) == (17, 17, 17)
    assert int(guia[8, 0, 0, 0]) == 12 and int(guia[4, 0, 0, 0]) == 6


if __name__ == "__main__":
    pruebas = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in pruebas:
        t()
        print("ok", t.__name__)
    print(len(pruebas), "pruebas")
