# -*- coding: utf-8 -*-
"""Prueba de integracion de la camara lenta, sin ComfyUI: plan (CP_LentaPlan) -> subgrafo (CP_BlockingLento) ->
un interprete pequeno que lo ejecuta con dobles de los nodos de LTX y los nodos de imagen y CP_LentaElegir de verdad.

Los dobles de LTX no generan nada: devuelven objetos etiquetados (modelo, conditioning con sus guias, latentes con su
tamano y numero de fotogramas) y el decodificado pinta en cada fotograma una marca con su ventana, su etapa y su indice
lento. Asi se ve de donde sale cada fotograma de la animacion final.
Se ejecuta con: python tests/test_lenta_integracion.py"""
import json
import os
import re
import sys
import types

import numpy as np
import torch
import torch.nn.functional as F

# ---- doble minimo de comfy_api.latest ----
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
        self.expand = k.get("expand")


class _IO:
    Schema = staticmethod(lambda **k: k)
    NodeOutput = _Salida
    ComfyNode = type("ComfyNode", (), {})


for _n in ("Image", "Int", "String", "Float", "Boolean", "Noise", "Model", "Vae", "LatentUpscaleModel", "Conditioning"):
    setattr(_IO, _n, _mk(_n))

latest.io = _IO
mod.latest = latest
sys.modules["comfy_api"] = mod
sys.modules["comfy_api.latest"] = latest


# ---- doble de comfy_execution.graph_utils (el reparto de prefijos de ComfyUI 0.42.1) ----
class GraphBuilder:
    _raiz, _llamada, _grafo = "", 0, 0

    @classmethod
    def set_default_prefix(cls, raiz, llamada, grafo=0):
        cls._raiz, cls._llamada, cls._grafo = raiz, llamada, grafo

    @classmethod
    def alloc_prefix(cls):
        cls._grafo += 1
        return f"{cls._raiz}.{cls._llamada}.{cls._grafo - 1}."


_ejec = types.ModuleType("comfy_execution")
_utiles = types.ModuleType("comfy_execution.graph_utils")
_utiles.GraphBuilder = GraphBuilder
_ejec.graph_utils = _utiles
sys.modules["comfy_execution"] = _ejec
sys.modules["comfy_execution.graph_utils"] = _utiles

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(AQUI, "..", "comfy_character_pipeline"))
import lenta as LE  # noqa: E402
import lenta_grafo as LG  # noqa: E402

with open(os.path.join(AQUI, "datos", "lenta_0003.json"), encoding="utf-8") as _f:
    DATOS = json.load(_f)

A, H = 128, 128  # pequeno: lo que se prueba es el cableado, no la imagen
FACTOR = 2.0
SIGMAS_8 = (1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0)
SIGMAS_3 = (0.909375, 0.725, 0.421875, 0.0)


# ---------------------------------------------------------------- objetos etiquetados

class Etiqueta:
    def __init__(self, papel):
        self.papel = papel

    def __repr__(self):
        return f"<{self.papel}>"


class Cond:
    """Conditioning: de que texto sale y que guias lleva encima, en orden."""

    def __init__(self, papel, guias=()):
        self.papel, self.guias = papel, tuple(guias)


class Latente:
    """Latente de video: tamano en pixeles, fotogramas, de que EmptyLTXVLatentVideo sale y guias anadidas."""

    def __init__(self, origen, ancho, alto, n, guias=(), muestreado=None):
        self.origen, self.ancho, self.alto, self.n = origen, ancho, alto, n
        self.guias, self.muestreado = tuple(guias), muestreado

    def con(self, **cambios):
        datos = dict(origen=self.origen, ancho=self.ancho, alto=self.alto, n=self.n, guias=self.guias, muestreado=self.muestreado)
        datos.update(cambios)
        return Latente(**datos)


def _ventana_etapa(origen):
    m = re.search(r"w(\d+)\.s(\d)\.", origen)
    assert m, origen
    return int(m.group(1)), int(m.group(2))


# Marca en el pixel (0, 0): canal 0 = etapa (0 guia, 1 render), canal 2 = ventana*0.2 + indice/1000.
def _marca(w, i):
    return w * 0.2 + i / 1000.0


def leer_marca(fotograma):
    v = int(round(float(fotograma[0, 0, 2]) * 1000))
    return int(round(float(fotograma[0, 0, 0]))), v // 200, v % 200


def _pose_de(imagen):
    """Las poses de prueba son de un color liso k/100 (lejos de la marca del arte)."""
    return int(round(float(imagen[..., 1].mean()) * 100))


# ---------------------------------------------------------------- dobles de los nodos

class Interprete:
    def __init__(self, grafo, valores_fuera):
        self.grafo, self.fuera = grafo, valores_fuera
        self.hecho, self.registro, self.orden = {}, [], []

    def valor(self, enlace):
        ident, salida = enlace
        if ident in self.fuera:
            return self.fuera[ident][salida]
        assert ident in self.grafo, f"enlace a un nodo que no existe: {enlace}"
        resultado = self.ejecutar(ident)
        assert 0 <= salida < len(resultado), f"{ident} no tiene salida {salida}"
        return resultado[salida]

    def ejecutar(self, ident):
        if ident not in self.hecho:
            nodo = self.grafo[ident]
            entradas = {k: self.valor(v) if LG._es_enlace(v) else v for k, v in nodo["inputs"].items()}
            fn = getattr(self, "n_" + nodo["class_type"], None)
            assert fn, f"clase sin doble: {nodo['class_type']}"
            self.hecho[ident] = tuple(fn(ident, **entradas))
            self.orden.append(ident)
        return self.hecho[ident]

    # ---- imagen (equivalentes a los de ComfyUI) ----
    def n_ImageFromBatch(self, ident, image, batch_index, length):
        # ComfyUI recorta el indice al ultimo sin avisar: aqui se exige que exista
        assert 0 <= batch_index < len(image) and length == 1, (ident, batch_index, len(image))
        return (image[batch_index:batch_index + length].clone(),)

    def n_ImageScale(self, ident, image, upscale_method, width, height, crop):
        assert upscale_method == "lanczos" and crop == "center", ident
        _, h, w, _ = image.shape
        if w / h > width / height:  # recorte centrado al aspecto de destino
            nuevo = int(round(h * width / height))
            image = image[:, :, (w - nuevo) // 2:(w - nuevo) // 2 + nuevo]
        elif w / h < width / height:
            nuevo = int(round(w * height / width))
            image = image[:, (h - nuevo) // 2:(h - nuevo) // 2 + nuevo]
        t = F.interpolate(image.movedim(-1, 1), size=(height, width), mode="bilinear", align_corners=False)
        return (t.movedim(1, -1),)

    # ---- muestreo ----
    def n_KSamplerSelect(self, ident, sampler_name):
        return (Etiqueta("sampler:" + sampler_name),)

    def n_ManualSigmas(self, ident, sigmas):
        return (tuple(float(x) for x in sigmas.split(",")),)

    def n_CFGGuider(self, ident, cfg, model, positive, negative):
        assert type(cfg) is float and cfg == 1.0, ident
        return ({"modelo": model, "positivo": positive, "negativo": negative},)

    def n_SamplerCustomAdvanced(self, ident, noise, guider, sampler, sigmas, latent_image):
        self.registro.append(("muestreo", ident, noise, guider, sampler, sigmas, latent_image))
        sale = latent_image.con(muestreado=ident)
        return (sale, sale)

    # ---- LTX ----
    def n_EmptyLTXVLatentVideo(self, ident, width, height, length, batch_size):
        assert all(type(x) is int for x in (width, height, length, batch_size)) and batch_size == 1, ident
        assert width % 32 == 0 and height % 32 == 0 and length % 8 == 1, (ident, width, height, length)
        return (Latente(ident, width, height, length),)

    def n_LTXVAddGuide(self, ident, positive, negative, vae, latent, image, frame_idx, strength):
        assert vae.papel == "vae" and type(strength) is float and strength == 1.0, ident
        assert image.shape[1:3] == (latent.alto, latent.ancho), (ident, image.shape, latent.ancho, latent.alto)
        assert frame_idx % 8 == 0 and 0 <= frame_idx <= latent.n - 1, (ident, frame_idx, latent.n)
        guia = ("pose", frame_idx, _pose_de(image))
        self.registro.append(("guia1", ident, guia))
        return (Cond(positive.papel, positive.guias + (guia,)), Cond(negative.papel, negative.guias + (guia,)),
                latent.con(guias=latent.guias + (guia,)))

    def n_LTXAddVideoICLoRAGuide(self, ident, positive, negative, vae, latent, image, frame_idx, strength,
                                 latent_downscale_factor, crop, use_tiled_encode, tile_size, tile_overlap,
                                 use_streaming, tile_frames):
        assert vae.papel == "vae" and strength == 1.0 and latent_downscale_factor == FACTOR, ident
        assert use_tiled_encode is False and use_streaming is False, ident
        guia = ("ic", frame_idx, crop, image)
        self.registro.append(("guia2", ident, guia))
        return (Cond(positive.papel, positive.guias + (guia,)), Cond(negative.papel, negative.guias + (guia,)),
                latent.con(guias=latent.guias + (guia,)))

    def n_LTXVCropGuides(self, ident, positive, negative, latent):
        # recibe el conditioning CON las guias: de ahi saca cuantas quitar
        assert positive.guias and positive.guias == negative.guias == latent.guias, ident
        assert latent.muestreado, ident
        return (Cond(positive.papel), Cond(negative.papel), latent.con(guias=()))

    def n_LTXVLatentUpsampler(self, ident, samples, upscale_model, vae):
        assert upscale_model.papel == "escalador" and vae.papel == "vae" and not samples.guias, ident
        return (samples.con(ancho=samples.ancho * 2, alto=samples.alto * 2),)

    def n_LTXVTiledVAEDecode(self, ident, vae, latents, horizontal_tiles, vertical_tiles, overlap, last_frame_fix,
                             working_device, working_dtype):
        assert vae.papel == "vae" and not latents.guias and latents.muestreado, ident
        w, etapa = _ventana_etapa(latents.origen)
        n, alto, ancho = latents.n, latents.alto, latents.ancho
        img = torch.full((n, alto, ancho, 3), 0.3)
        lado = ancho // 8
        for i in range(n):
            # un cuadrado que avanza: el recorrido de la guia tiene movimiento de verdad
            x = lado + int(round(i * (ancho - 3 * lado) / max(1, n - 1)))
            img[i, alto // 2 - lado // 2:alto // 2 + lado // 2, x:x + lado] = 0.9
            img[i, 0, 0] = torch.tensor([float(etapa - 1), 0.0, _marca(w, i)])
        return (img,)

    # ---- el de verdad ----
    def n_CP_LentaElegir(self, ident, **entradas):
        return LE.LentaElegir.execute(**entradas).args


ENLACES = {
    "ruido": ["semilla", 0], "poses": ["poses", 0], "arte": ["arte", 0],
    "modelo_suave": ["sage_suave", 0], "modelo_render": ["sage_render", 0], "factor_guia": ["ic_lora", 1],
    "vae": ["vae", 0], "escalador": ["escalador", 0],
    "positivo_suave": ["cond_suave", 0], "negativo_suave": ["cond_suave", 1],
    "positivo_render": ["cond_render", 0], "negativo_render": ["cond_render", 1],
}


def _valores_fuera(n_poses):
    # poses a 540 de alto en proporcion 16:9, pequenas: cada una de un color liso k/100
    poses = torch.stack([torch.full((27, 48, 3), k / 100.0) for k in range(n_poses)])
    arte = torch.full((1, H, A, 3), 0.95)
    return {
        "semilla": (Etiqueta("ruido"),), "poses": (poses,), "arte": (arte,),
        "sage_suave": (Etiqueta("modelo_suave"),), "sage_render": (Etiqueta("modelo_render"),),
        "ic_lora": (Etiqueta("modelo_ic"), FACTOR), "vae": (Etiqueta("vae"),), "escalador": (Etiqueta("escalador"),),
        "cond_suave": (Cond("positivo_suave"), Cond("negativo_suave")),
        "cond_render": (Cond("positivo_render"), Cond("negativo_render")),
    }


def _plan_real():
    """El plan como lo saca CP_LentaPlan del plano de control: mapa de poses y saltos medidos guardados."""
    inicios, total = LE.leer_mapa(json.dumps({"bloques": DATOS["bloques"]}), len(DATOS["inicios"]))
    return LE.planificar(inicios, total, DATOS["saltos_dis_480"])


def _plan_v8():
    """El plan de la version validada: ventanas y lentos puestos a mano."""
    inicios = DATOS["inicios"]
    return LE.leer_plan(LE.armar_plan(inicios, inicios[-1] + 1, DATOS["saltos_dis_480"], DATOS["lentos_v8"],
                                      [tuple(v) for v in DATOS["ventanas_v8"]]))


def _ejecutar(plan):
    GraphBuilder.set_default_prefix("40", 0, 0)
    salida = LG.BlockingLento.execute(json.dumps(plan), A, H, **ENLACES)
    grafo = salida.expand
    interprete = Interprete(grafo, _valores_fuera(len(plan["inicios"])))
    animacion = interprete.valor(salida.args[0])
    playblast = interprete.valor(salida.args[1])
    assert len(interprete.hecho) == len(grafo), "hay nodos del subgrafo que no alimentan ninguna salida"
    return grafo, interprete, animacion, playblast


def _comprobar(plan, grafo, interprete, animacion, playblast):
    ventanas, inicios, total = plan["ventanas"], plan["inicios"], plan["total"]
    pre = "40.0.0."

    # cada muestreador: el ruido del panel, el modelo y los textos de su etapa, sus sigmas y euler_ancestral
    muestreos = {r[1]: r for r in interprete.registro if r[0] == "muestreo"}
    assert len(muestreos) == 3 * len(ventanas)
    for ident, (_, _, ruido, guider, sampler, sigmas, latente) in muestreos.items():
        w, etapa = _ventana_etapa(ident)
        etiqueta = "suave" if etapa == 1 else "render"
        assert ruido.papel == "ruido" and sampler.papel == "sampler:euler_ancestral", ident
        assert guider["modelo"].papel == "modelo_" + etiqueta, ident
        assert guider["positivo"].papel == "positivo_" + etiqueta and guider["negativo"].papel == "negativo_" + etiqueta, ident
        assert latente.n == ventanas[w]["n"], ident
        if ident.endswith("muestreo1"):
            assert sigmas == SIGMAS_8 and (latente.ancho, latente.alto) == (A // 2, H // 2), ident
        else:
            assert sigmas == (SIGMAS_3 if etapa == 1 else SIGMAS_8) and (latente.ancho, latente.alto) == (A, H), ident

    for w, v in enumerate(ventanas):
        esperadas = [("pose", v["pos"][j], k) for j, k in enumerate(range(v["p0"], v["p1"] + 1))]
        # etapa 1: los dos pases fijan cada pose de la ventana en su fotograma lento, partiendo de los textos limpios
        for sufijo in ("1", "2"):
            guias = muestreos[f"{pre}w{w}.s1.muestreo{sufijo}"][3]["positivo"].guias
            assert list(guias) == esperadas, (w, sufijo, guias)
        # etapa 2: el playblast lento entero (n fotogramas), y el ancla como arte (-1) y como fotograma 0
        guias = muestreos[f"{pre}w{w}.s2.muestreo"][3]["positivo"].guias
        assert [(g[1], g[2]) for g in guias] == [(0, "disabled"), (-1, "center"), (0, "center")], (w, guias)
        guia, ancla_arte, ancla_pose = (g[3] for g in guias)
        assert guia.shape == (v["n"], H, A, 3) and leer_marca(guia[-1]) == (0, w, v["n"] - 1), w
        assert ancla_arte is ancla_pose and ancla_arte.shape == (1, H, A, 3)
        if w == 0:
            assert torch.equal(ancla_arte, interprete.fuera["arte"][0]), "el ancla de la primera ventana es el arte"
        else:
            render_anterior = interprete.hecho[f"{pre}w{w - 1}.s2.render"][0]
            assert torch.equal(ancla_arte[0], render_anterior[-1]), f"ancla de la ventana {w}"
            assert leer_marca(ancla_arte[0]) == (1, w - 1, ventanas[w - 1]["n"] - 1)

    # la animacion y el playblast: total fotogramas, al tamano del render y de los mismos indices lentos
    assert animacion.shape == (total, H, A, 3) and playblast.shape == (total, H, A, 3), (animacion.shape, playblast.shape)
    marcas = [leer_marca(f) for f in animacion]
    assert all(e == 1 for e, _, _ in marcas) and all(e == 0 for e, _, _ in (leer_marca(f) for f in playblast))
    assert [m[1:] for m in marcas] == [leer_marca(f)[1:] for f in playblast]
    assert sorted(marcas) == marcas, "el video va siempre hacia delante"
    # cada pose cae en su fotograma del blocking: la frontera entre ventanas es el lento 0 de la siguiente
    for k, inicio in enumerate(inicios):
        w = max(i for i, v in enumerate(ventanas) if v["p0"] <= k)
        v = ventanas[w]
        assert marcas[inicio][1:] == (w, v["pos"][k - v["p0"]]), (k, inicio, marcas[inicio])
    # y lo que sigue a la ultima pose es la pose final repetida
    assert all(m[1:] == (len(ventanas) - 1, ventanas[-1]["n"] - 1) for m in marcas[inicios[-1]:])
    return marcas


# ---------------------------------------------------------------- pruebas

def test_plan_real_de_principio_a_fin():
    plan = _plan_real()
    assert plan["total"] == 77 and len(plan["inicios"]) == 25
    grafo, interprete, animacion, playblast = _ejecutar(plan)
    _comprobar(plan, grafo, interprete, animacion, playblast)
    assert len(animacion) == 77 and len(playblast) == 77


def test_plan_v8_reproduce_los_fotogramas_validados_a_traves_del_grafo():
    """Con el recorrido guardado de v8 en lugar del medido, el grafo entero devuelve los 77 fotogramas de la version
    validada, en orden: el plan, el cableado del acumulado y la eleccion encajan de punta a punta."""
    plan = _plan_v8()
    guardado = DATOS["recorrido_por_ventana"]
    original = LE.recorrido

    def recorrido_guardado(guia, nodo="CP_LentaElegir"):
        _, w, _ = leer_marca(guia[0])
        assert len(guia) == len(guardado[w]) + 1
        return np.array(guardado[w], np.float64)

    LE.recorrido = recorrido_guardado
    try:
        grafo, interprete, animacion, playblast = _ejecutar(plan)
    finally:
        LE.recorrido = original
    marcas = _comprobar(plan, grafo, interprete, animacion, playblast)
    assert [[f"w{w}", i] for _, w, i in marcas] == DATOS["elegidos_v8"]


if __name__ == "__main__":
    pruebas = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in pruebas:
        t()
        print("ok", t.__name__)
    print(len(pruebas), "pruebas")
