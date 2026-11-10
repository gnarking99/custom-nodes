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


# ---- reborde claro del compositor (sin_halo) ----

def _escena(h, placa=0.12, sombra=False, cx=0.5, cy=0.5):
    """Figura redonda de borde suavizado pintada sobre un telon degradado y claro, y una placa lisa con algo de degradado.
    Devuelve la pintada, la placa, la mascara dura de la figura y la sombra pintada (0-1, o None)."""
    w = h * 3 // 2
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    img = np.linspace(0.62, 0.36, h, dtype=np.float32)[:, None, None] * np.array([0.8, 0.88, 1.0], np.float32) * np.ones((1, w, 1), np.float32)
    s = None
    if sombra:  # mancha oscura y blanda bajo la figura, como la sombra que pinta el modelo sobre el telon
        s = np.exp(-(((xx - cx * w) / (0.25 * h)) ** 2 + ((yy - (cy + 0.21) * h) / (0.035 * h)) ** 2))
        img = img * (1 - 0.9 * s[..., None])
    alfa = np.clip(0.2 * h - np.hypot(xx - cx * w, yy - cy * h) + 0.5, 0, 1)[..., None]
    img = img * (1 - alfa) + np.array([0.75, 0.35, 0.2], np.float32) * alfa
    pl = np.full((h, w, 3), placa, np.float32) * (0.8 + 0.4 * xx / w)[..., None]
    return img.astype(np.float32), pl.astype(np.float32), alfa[..., 0] >= 0.5, s


def _con_y_sin(img, pl):
    P, L = torch.from_numpy(img)[None], torch.from_numpy(pl)[None]
    return B.componer(P, L, sin_halo=True)[0][0].numpy(), B.componer(P, L, sin_halo=False)[0][0].numpy()


def _distancias(figura):
    """Distancia en px de cada pixel a la figura (por fuera) y a su contorno (por dentro)."""
    from scipy.ndimage import distance_transform_edt
    return distance_transform_edt(~figura), distance_transform_edt(figura)


def _luz_de_mas(img, pl):
    return np.clip(img.mean(-1) - pl.mean(-1), 0, None)


def _componer_de_antes(pintadas, placa):
    """El compositor tal como era antes de quitar el reborde (v0.12), con las piezas del modulo."""
    import torch.nn.functional as F
    n, alto, ancho = pintadas.shape[:3]
    fondo = B._cubrir(placa[:1].float(), alto, ancho)[0]
    salida, mascaras = [], []
    for i in range(n):
        alfa, _ = B.mate_sobre_telon(B._a_trabajo(pintadas[i]))
        osc = B.sombra_contacto(alfa)
        a = F.interpolate(torch.from_numpy(alfa)[None, None].float(), size=(alto, ancho), mode="bilinear", align_corners=False)[0, 0]
        o = F.interpolate(torch.from_numpy(osc)[None, None].float(), size=(alto, ancho), mode="bilinear", align_corners=False)[0, 0]
        f = fondo * (1 - o)[..., None]
        salida.append(f * (1 - a)[..., None] + pintadas[i].float() * a[..., None])
        mascaras.append(a)
    return torch.stack(salida).clamp(0, 1), torch.stack(mascaras)


def _receta_validada(pintada, compuesta, fondo):
    """La receta que se valido a mano, con sus numeros tal cual (medidos a 1088 de alto; umbrales pasados a 0-1)."""
    from scipy.ndimage import binary_closing, binary_dilation, binary_fill_holes, gaussian_filter, label
    telon = B.ajustar_telon(pintada * 255.0)[0] / 255.0
    d = np.abs(pintada - telon).max(axis=-1)
    silueta = binary_fill_holes(binary_closing(d > 30 / 255, iterations=2))
    et, n = label(silueta)
    if n:
        areas = np.bincount(et.ravel())[1:]
        silueta = np.isin(et, 1 + np.flatnonzero(areas >= 200))
    anillo = binary_dilation(silueta, iterations=9) & ~silueta
    aclara = compuesta.mean(axis=-1) > fondo.mean(axis=-1) + 4 / 255
    peso = gaussian_filter((anillo & aclara).astype(np.float32), 1.2)[..., None]
    return compuesta * (1 - peso) + fondo * peso, int((anillo & aclara).sum())


def test_medidas_del_reborde_escalan_con_el_alto():
    assert B.medidas_reborde(1088) == {"cierre": 2, "area": 200.0, "anillo": 9, "suavizado": 1.2}
    assert B.medidas_reborde(544) == {"cierre": 1, "area": 50.0, "anillo": 5, "suavizado": 0.6}
    assert B.medidas_reborde(2176) == {"cierre": 4, "area": 800.0, "anillo": 18, "suavizado": 2.4}
    chica = B.medidas_reborde(64)
    assert chica["cierre"] == 1 and chica["anillo"] == 1, "nunca cero vueltas: scipy las repetiria hasta el final"


def test_quitar_reborde_a_1088_es_la_receta_validada():
    h = 1088
    img, pl, _, _ = _escena(h)
    w = img.shape[1]
    yy, xx = np.mgrid[0:h, 0:w]
    telon = _escena(h, cx=-9)[0]  # la misma escena con la figura fuera de cuadro: solo el telon
    hueco = np.hypot(xx - 0.5 * w, yy - 0.42 * h) < 0.04 * h  # telon visto a traves de la figura: es interior, no se toca
    img[hueco] = telon[hueco]
    img[int(0.5 * h) - 4:int(0.5 * h) + 4, int(0.5 * w + 0.2 * h) + 6:int(0.5 * w + 0.2 * h) + 14] = (0.75, 0.35, 0.2)  # mota de 64 px pegada: no es silueta
    img[100:130, 100:130] = (0.75, 0.35, 0.2)  # pieza suelta de 900 px: si es silueta, con su propio anillo
    P, L = torch.from_numpy(img)[None], torch.from_numpy(pl)[None]
    comp = B.componer(P, L, sin_halo=False)[0][0].numpy()
    esperada, n_esperados = _receta_validada(img, comp, pl)
    limpia, n = B.quitar_reborde(comp, img, pl)
    assert n == n_esperados and n > 2000, (n, n_esperados)
    assert np.array_equal(limpia, esperada), float(np.abs(limpia - esperada).max())
    # el compositor usa el telon que ya ajusto a 540 de alto: sale lo mismo salvo algun pixel suelto del borde del anillo
    del_nodo = B.componer(P, L, sin_halo=True)[0][0].numpy()
    dif = np.abs(del_nodo - esperada).max(-1)
    assert (dif > 2 / 255).mean() < 2e-4 and dif.mean() < 1e-4, ((dif > 2 / 255).sum(), float(dif.mean()))


def test_sin_halo_quita_el_reborde_claro_y_no_toca_el_interior():
    cambiados = {}
    for h in (544, 1088):
        e = h / 1088
        img, pl, figura, _ = _escena(h)
        con, sin = _con_y_sin(img, pl)
        fuera, dentro = _distancias(figura)
        pegado, anillo = (fuera > 0) & (fuera <= 2 * e), (fuera > 2 * e) & (fuera <= 9 * e)
        antes, despues = _luz_de_mas(sin, pl), _luz_de_mas(con, pl)
        assert antes[pegado].mean() > 0.1 and antes[anillo].max() > 0.1, (h, "la escena tiene que traer reborde")
        # de 2 px (a 1088) hacia fuera el anillo es la placa; pegado a la silueta queda la mezcla suave de la receta
        assert despues[anillo].max() < 0.05, (h, float(despues[anillo].max()))
        assert despues[pegado].mean() < 0.5 * antes[pegado].mean(), (h, float(despues[pegado].mean()), float(antes[pegado].mean()))
        todo = pegado | anillo
        assert despues[todo].sum() < 0.45 * antes[todo].sum(), (h, float(despues[todo].sum() / antes[todo].sum()))
        # lo que cambia va hacia la placa, mas oscura: nada se aclara (salvo 1-2 niveles donde la mezcla pisa la sombra de contacto)
        assert (con <= sin + 0.01).all(), (h, float((con - sin).max()))
        cambio = np.abs(con - sin).max(-1)
        assert cambio[dentro > 8 * e].max() == 0, (h, "el interior de la figura no cambia")
        assert cambio[fuera > 18 * e].max() == 0, (h, "lejos de la figura no cambia nada")
        cambiados[h] = int((cambio > 0).sum())
    # los radios van con el alto: al doble de alto, la misma correccion ocupa unas 4 veces mas pixeles
    assert 3.0 < cambiados[1088] / cambiados[544] < 5.5, cambiados


def test_sin_halo_no_toca_nada_si_la_placa_es_mas_clara_que_el_reborde():
    img, pl, _, _ = _escena(544, placa=0.8)
    P, L = torch.from_numpy(img)[None], torch.from_numpy(pl)[None]
    con, sin = B.componer(P, L, sin_halo=True), B.componer(P, L, sin_halo=False)
    assert torch.equal(con[0], sin[0]) and torch.equal(con[1], sin[1])


def test_sin_halo_no_borra_las_sombras():
    for h in (544, 1088):
        e = h / 1088
        img, pl, figura, sombra = _escena(h, placa=0.25, sombra=True)
        con, sin = _con_y_sin(img, pl)
        fuera, _ = _distancias(figura)
        cambio = np.abs(con - sin).max(-1)
        # la sombra pintada bajo la figura llega a la compuesta mas oscura que la placa, y ahi se queda
        pintada = (sombra > 0.6) & (fuera > 8 * e)
        assert pintada.sum() > 10000 * e * e and (sin.mean(-1)[pintada] < pl.mean(-1)[pintada] - 0.03).all(), h
        assert cambio[pintada].max() == 0, (h, float(cambio[pintada].max()))
        # nada de lo que ya era mas oscuro que la placa (sombra pintada o de contacto) se aclara
        oscuro = sin.mean(-1) < pl.mean(-1) - 0.03
        assert oscuro.sum() > 20000 * e * e and (con.mean(-1) - sin.mean(-1))[oscuro].max() < 0.01, h
        # y en la misma imagen el reborde claro (arriba, donde no hay sombra) si se quita
        arriba = (fuera > 0) & (fuera <= 9 * e) & (np.mgrid[0:h, 0:img.shape[1]][0] < 0.5 * h)
        antes, despues = _luz_de_mas(sin, pl)[arriba], _luz_de_mas(con, pl)[arriba]
        assert antes.max() > 0.1 and despues.sum() < 0.55 * antes.sum(), (h, float(despues.sum() / antes.sum()))


def test_sin_halo_apagado_es_el_compositor_de_antes():
    img = np.stack([_escena(272, cx=0.4)[0], _escena(272, cx=0.6, sombra=True)[0]])
    P, L = torch.from_numpy(img), torch.rand(1, 200, 320, 3, generator=torch.Generator().manual_seed(3))
    sal, masc = B.componer(P, L, sin_halo=False)
    sal0, masc0 = _componer_de_antes(P, L)
    assert torch.equal(sal, sal0) and torch.equal(masc, masc0)
    sal1, masc1 = B.componer(P, L)  # por defecto quita el reborde: cambia la imagen, no el mate
    assert torch.equal(masc1, masc0) and not torch.equal(sal1, sal0)


def test_sin_halo_en_un_lote_de_varias_poses():
    h = 544
    escenas = [_escena(h, cx=cx) for cx in (0.3, 0.5, 0.7)]
    P, L = torch.from_numpy(np.stack([x[0] for x in escenas])), torch.from_numpy(escenas[0][1])[None]
    con, masc = B.componer(P, L, sin_halo=True)
    sin, masc0 = B.componer(P, L, sin_halo=False)
    assert con.shape == P.shape and masc.shape == P.shape[:3] and torch.equal(masc, masc0)
    pl = escenas[0][1]
    for i, (_, _, figura, _) in enumerate(escenas):
        sola = B.componer(P[i:i + 1], L, sin_halo=True)[0][0]
        assert torch.equal(con[i], sola), f"la pose {i} sale igual sola que en el lote"
        fuera, _ = _distancias(figura)
        anillo = (fuera > 0) & (fuera <= 9 * h / 1088)
        antes, despues = _luz_de_mas(sin[i].numpy(), pl)[anillo].sum(), _luz_de_mas(con[i].numpy(), pl)[anillo].sum()
        assert despues < 0.45 * antes, (i, float(despues / antes))


def test_sin_halo_no_se_come_al_personaje_cortado_por_el_cuadro():
    from scipy.ndimage import distance_transform_edt
    for h in (544, 1088):
        e = h / 1088
        img, pl, figura, _ = _escena(h, cy=1.0)  # media figura: el borde de abajo del cuadro la corta
        con, sin = _con_y_sin(img, pl)
        margen = int(8 * e) + 2
        dentro = distance_transform_edt(np.pad(figura, ((0, margen), (0, 0)), mode="edge"))[:h]  # el cuadro no es contorno
        interior = dentro > 8 * e
        cambio = np.abs(con - sin).max(-1)
        assert interior[-2:].sum() > 300 * e and cambio[-2:][interior[-2:]].max() == 0, (h, "las filas pegadas al borde son personaje")
        assert cambio[interior].max() == 0
        assert (cambio > 0).sum() > 500 * e * e, "el reborde del contorno de verdad si se corrige"


def test_nodo_componer_declara_sin_halo_opcional_y_al_final():
    entradas = B.PosesComponer.define_schema()["inputs"]
    assert [x.a[0] for x in entradas] == ["pintadas", "placa", "sin_halo"]
    assert entradas[2].k.get("optional") is True and entradas[2].k.get("default") is True
    assert not entradas[0].k.get("optional") and not entradas[1].k.get("optional")
    img, pl, _, _ = _escena(272)
    P, L = torch.from_numpy(img)[None], torch.from_numpy(pl)[None]
    con, sin = B.componer(P, L, sin_halo=True), B.componer(P, L, sin_halo=False)
    assert not torch.equal(con[0], sin[0])
    sin_la_entrada = B.PosesComponer.execute(P, L).args  # un grafo guardado antes de v0.13 no la trae
    assert len(sin_la_entrada) == 2 and torch.equal(sin_la_entrada[0], con[0]) and torch.equal(sin_la_entrada[1], con[1])
    assert torch.equal(B.PosesComponer.execute(P, L, sin_halo=None).args[0], con[0])
    assert torch.equal(B.PosesComponer.execute(P, L, sin_halo=True).args[0], con[0])
    assert torch.equal(B.PosesComponer.execute(P, L, sin_halo=False).args[0], sin[0])


if __name__ == "__main__":
    pruebas = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in pruebas:
        t()
        print("ok", t.__name__)
    print(len(pruebas), "pruebas")
