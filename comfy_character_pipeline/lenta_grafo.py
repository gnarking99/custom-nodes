# -*- coding: utf-8 -*-
"""Camara lenta: el subgrafo que convierte un blocking a saltos en animacion suave y renderizada con LTX.

Un solo nodo (CP_BlockingLento) se EXPANDE en los nodos de serie con los que se valido la receta, una vez por ventana
del plan de CP_LentaPlan:
- etapa 1, playblast suave: todas las poses de la ventana fijadas muy separadas en el tiempo (LTXVAddGuide) y dos
  pases del modelo sin LoRAs: a mitad de tamano, subida del latente x2 y refinado a tamano completo;
- etapa 2, render: la IC-LoRA sigue ese playblast lento en UN pase a tamano completo, con un ancla como arte (-1) y
  como fotograma 0. El ancla de la primera ventana es el arte; la de las siguientes, el ultimo fotograma del render de
  la ventana anterior: el personaje es el mismo de principio a fin y no hay costuras;
- CP_LentaElegir vuelve a los tiempos del blocking y va acumulando lo elegido.

El subgrafo es un diccionario (prompt de API): se compara nodo a nodo, sin GPU, con los grafos validados de
tests/datos/ (tests/test_lenta_grafo.py). Lo que sale de leer el motor (ComfyUI 0.42.1):
- los nodos de una expansion NO pasan la validacion del motor: nadie convierte tipos ni rellena valores por defecto.
  Cada entrada va con su tipo exacto (int, float, bool, str) y todas las obligatorias van escritas;
- ningun nodo de guardado aqui dentro: llegaria al usuario como resultado;
- los objetos pesados entran con raw_link (llega el enlace [id, indice], no el valor) y se cablean tal cual;
- el motor ejecuta primero lo que antes da de alta, y da de alta recorriendo las entradas de cada nodo de la ultima a
  la primera. En CP_LentaElegir lo acumulado va DETRAS de guia y render: asi cada ventana termina antes de que empiece
  la siguiente. OJO: la cache del motor (RAM_PRESSURE) conserva hasta el final la guia y el render de TODAS las ventanas
  y lo acumulado por cada CP_LentaElegir: unos 40 GB de imagenes para un plano de 77 fotogramas en 4 ventanas.
"""

import json
import logging

from comfy_api.latest import io

try:
    from .lenta import leer_plan as _leer_plan
except ImportError:  # las pruebas cargan los modulos sueltos, sin el paquete
    from lenta import leer_plan as _leer_plan

NODO = "CP_BlockingLento"
SIGMAS_8 = "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"  # 8 pasos, la curva oficial del destilado
SIGMAS_3 = "0.909375, 0.725, 0.421875, 0.0"  # refinado del segundo pase de la etapa 1
MUESTREADOR = "euler_ancestral"
# El motor firma cada nodo para su cache recorriendo los ancestros con recursion (caching.py), y el encadenado por el
# ancla mete unos 6 niveles por ventana: con 64 son unos 400 y el limite de Python es 1000.
MAX_VENTANAS = 64
ENLACES = ("ruido", "poses", "arte", "modelo_suave", "modelo_render", "factor_guia", "vae", "escalador",
           "positivo_suave", "negativo_suave", "positivo_render", "negativo_render")


def _es_enlace(v):
    """Lo mismo que entiende el motor por enlace: [id de nodo, indice de salida]."""
    return (isinstance(v, (list, tuple)) and len(v) == 2 and isinstance(v[0], str)
            and isinstance(v[1], int) and not isinstance(v[1], bool))


def _es_entero(v):
    return isinstance(v, int) and not isinstance(v, bool)


def leer_plan(texto, nodo=NODO):
    """El plan de CP_LentaPlan (texto JSON) como diccionario, comprobado entero con el mismo lector que CP_LentaElegir
    (lenta.leer_plan): un plan que fallaria en la primera eleccion falla aqui, antes de gastar GPU."""
    return _leer_plan(texto, nodo)


def comprobar_plan(plan, nodo=NODO):
    """Lo minimo que necesita el subgrafo (grafo_lento la llama tambien con planes a medias, en las pruebas). El plan
    entero lo comprueba leer_plan en el nodo. Devuelve las ventanas. Un plan torcido tiene que fallar AQUI, antes de gastar
    GPU: los nodos de la expansion no se validan y fallarian a mitad del trabajo."""
    ventanas = plan.get("ventanas") if isinstance(plan, dict) else None
    if not isinstance(ventanas, list) or not ventanas:
        raise ValueError(f"[{nodo}] El plan no trae ventanas: conecta el plan de CP_LentaPlan.")
    if len(ventanas) > MAX_VENTANAS:
        raise ValueError(f"[{nodo}] El plan tiene {len(ventanas)} ventanas y el tope es {MAX_VENTANAS}: "
                         "parte el blocking en planos mas cortos.")
    lentos = plan.get("lentos")
    for w, v in enumerate(ventanas):
        if not isinstance(v, dict) or any(c not in v for c in ("p0", "p1", "pos", "n")):
            raise ValueError(f"[{nodo}] La ventana {w} del plan no se puede leer: le falta p0, p1, pos o n.")
        p0, p1, pos, n = v["p0"], v["p1"], v["pos"], v["n"]
        if not (_es_entero(p0) and _es_entero(p1) and _es_entero(n) and isinstance(pos, list) and all(_es_entero(x) for x in pos)):
            raise ValueError(f"[{nodo}] La ventana {w} del plan no se puede leer: p0, p1, n y pos tienen que ser enteros.")
        if p0 < 0 or p1 <= p0:
            raise ValueError(f"[{nodo}] La ventana {w} va de la pose {p0} a la {p1}: tiene que acabar en una pose posterior.")
        if len(pos) != p1 - p0 + 1:
            raise ValueError(f"[{nodo}] La ventana {w} va de la pose {p0} a la {p1} y trae {len(pos)} posiciones: "
                             f"tienen que ser {p1 - p0 + 1}.")
        if pos[0] != 0 or any(b <= a for a, b in zip(pos, pos[1:])):
            raise ValueError(f"[{nodo}] Las posiciones de la ventana {w} tienen que empezar en 0 y crecer: {pos}.")
        if any(x % 8 for x in pos):
            raise ValueError(f"[{nodo}] La ventana {w} fija poses en {pos}: LTX solo fija en fotogramas multiplos de 8.")
        if n != pos[-1] + 1 or n % 8 != 1:
            raise ValueError(f"[{nodo}] La ventana {w} tiene {n} fotogramas lentos: tienen que ser 8k+1 y acabar en su "
                             f"ultima pose (fotograma {pos[-1]}).")
        anterior = ventanas[w - 1]["p1"] if w else 0
        if p0 != anterior:
            raise ValueError(f"[{nodo}] La ventana {w} empieza en la pose {p0} y tenia que empezar en la {anterior}: "
                             "cada ventana comparte su primera pose con la anterior y la primera empieza en la pose 0.")
        if isinstance(lentos, list) and [b - a for a, b in zip(pos, pos[1:])] != lentos[p0:p1]:
            raise ValueError(f"[{nodo}] Las posiciones de la ventana {w} no salen de los lentos del plan: {pos}.")
    inicios = plan.get("inicios")
    if isinstance(inicios, list) and ventanas[-1]["p1"] != len(inicios) - 1:
        raise ValueError(f"[{nodo}] Las ventanas acaban en la pose {ventanas[-1]['p1']} y el blocking tiene "
                         f"{len(inicios)} poses: el plan no cubre el plano entero.")
    return ventanas


def comprobar_tamano(ancho, alto, nodo=NODO):
    """La etapa 1 hace su primer pase a la mitad y LTX trabaja en multiplos de 32: el tamano va en multiplos de 64."""
    if not (_es_entero(ancho) and _es_entero(alto)) or ancho < 128 or alto < 128 or ancho % 64 or alto % 64:
        raise ValueError(f"[{nodo}] El ancho ({ancho}) y el alto ({alto}) tienen que ser multiplos de 64 (minimo 128).")


def comprobar_enlaces(enlaces, nodo=NODO):
    """Copia limpia de los enlaces de entrada. Todos son enlaces [id, indice]; el factor de la guia puede ser un numero."""
    limpio = {}
    for nombre in ENLACES:
        v = enlaces.get(nombre) if isinstance(enlaces, dict) else None
        if _es_enlace(v):
            limpio[nombre] = [v[0], v[1]]
        elif nombre == "factor_guia" and isinstance(v, (int, float)) and not isinstance(v, bool):
            limpio[nombre] = float(v)
        else:
            raise ValueError(f"[{nodo}] La entrada '{nombre}' tiene que venir conectada a otro nodo.")
    return limpio


def _copia(v):
    """Cada entrada lleva su propia lista: el motor y las pruebas pueden tocar una sin tocar las demas."""
    return list(v) if isinstance(v, list) else v


def _nodo(grafo, ident, clase, **entradas):
    if ident in grafo:
        raise ValueError(f"[{NODO}] Id repetido en el subgrafo: {ident}.")
    grafo[ident] = {"class_type": clase, "inputs": {k: _copia(v) for k, v in entradas.items()}}
    return ident


def _fijar_poses(grafo, pre, escala, guia, ventana, enl, positivo, negativo, latente, ancho, alto):
    """Cadena de LTXVAddGuide: cada pose de la ventana, a ancho x alto, fijada en su fotograma lento.
    Devuelve (positivo, negativo, latente) de la ultima guia."""
    for j, k in enumerate(range(ventana["p0"], ventana["p1"] + 1)):
        e = _nodo(grafo, f"{pre}{escala}{k}", "ImageScale", image=[f"{pre}i{k}", 0], upscale_method="lanczos",
                  width=ancho, height=alto, crop="center")
        g = _nodo(grafo, f"{pre}{guia}{k}", "LTXVAddGuide", positive=positivo, negative=negativo, vae=enl["vae"],
                  latent=latente, image=[e, 0], frame_idx=ventana["pos"][j], strength=1.0)
        positivo, negativo, latente = [g, 0], [g, 1], [g, 2]
    return positivo, negativo, latente


def _muestrear(grafo, pre, sufijo, enl, modelo, sigmas, positivo, negativo, latente):
    """CFG 1 + muestreo + recorte de las guias. LTXVCropGuides recibe el conditioning CON las guias: de ahi saca cuantas
    quitar. Devuelve el enlace al latente ya recortado."""
    guider = _nodo(grafo, f"{pre}guider{sufijo}", "CFGGuider", cfg=1.0, model=modelo, positive=positivo, negative=negativo)
    muestreo = _nodo(grafo, f"{pre}muestreo{sufijo}", "SamplerCustomAdvanced", noise=enl["ruido"], guider=[guider, 0],
                     sampler=[f"{pre}sampler", 0], sigmas=[sigmas, 0], latent_image=latente)
    recorte = _nodo(grafo, f"{pre}recorte{sufijo}", "LTXVCropGuides", positive=positivo, negative=negativo, latent=[muestreo, 0])
    return [recorte, 2]


def _decodificar(grafo, ident, enl, latente):
    return [_nodo(grafo, ident, "LTXVTiledVAEDecode", horizontal_tiles=2, vertical_tiles=2, overlap=4, last_frame_fix=False,
                  working_device="auto", working_dtype="auto", vae=enl["vae"], latents=latente), 0]


def _etapa_suave(grafo, pre, ventana, enl, ancho, alto):
    """Etapa 1 de una ventana: el playblast lento, n fotogramas a ancho x alto. Devuelve su enlace."""
    for k in range(ventana["p0"], ventana["p1"] + 1):
        _nodo(grafo, f"{pre}i{k}", "ImageFromBatch", image=enl["poses"], batch_index=k, length=1)
    _nodo(grafo, pre + "latente", "EmptyLTXVLatentVideo", width=ancho // 2, height=alto // 2, length=ventana["n"], batch_size=1)
    _nodo(grafo, pre + "sampler", "KSamplerSelect", sampler_name=MUESTREADOR)
    _nodo(grafo, pre + "sigmas1", "ManualSigmas", sigmas=SIGMAS_8)
    _nodo(grafo, pre + "sigmas2", "ManualSigmas", sigmas=SIGMAS_3)
    # primer pase a mitad de tamano
    guiado = _fijar_poses(grafo, pre, "e", "k", ventana, enl, enl["positivo_suave"], enl["negativo_suave"],
                          [pre + "latente", 0], ancho // 2, alto // 2)
    medio = _muestrear(grafo, pre, "1", enl, enl["modelo_suave"], pre + "sigmas1", *guiado)
    _nodo(grafo, pre + "subir", "LTXVLatentUpsampler", samples=medio, upscale_model=enl["escalador"], vae=enl["vae"])
    # segundo pase: las mismas poses a tamano completo sobre el latente subido, partiendo del conditioning LIMPIO
    guiado = _fijar_poses(grafo, pre, "E", "K", ventana, enl, enl["positivo_suave"], enl["negativo_suave"],
                          [pre + "subir", 0], ancho, alto)
    entero = _muestrear(grafo, pre, "2", enl, enl["modelo_suave"], pre + "sigmas2", *guiado)
    return _decodificar(grafo, pre + "guia", enl, entero)


def _etapa_render(grafo, pre, ventana, enl, guia, ancla, ancho, alto):
    """Etapa 2 de una ventana: el render lento en un pase a tamano completo. Devuelve su enlace."""
    _nodo(grafo, pre + "latente", "EmptyLTXVLatentVideo", width=ancho, height=alto, length=ventana["n"], batch_size=1)
    _nodo(grafo, pre + "sampler", "KSamplerSelect", sampler_name=MUESTREADOR)
    _nodo(grafo, pre + "sigmas", "ManualSigmas", sigmas=SIGMAS_8)
    positivo, negativo, latente = enl["positivo_render"], enl["negativo_render"], [pre + "latente", 0]
    # el playblast lento entero, el ancla como arte (-1) y el ancla fijada en el fotograma 0
    for nombre, imagen, fotograma, recorte in (("g_guia", guia, 0, "disabled"), ("g_arte", ancla, -1, "center"),
                                               ("g_pose", ancla, 0, "center")):
        g = _nodo(grafo, pre + nombre, "LTXAddVideoICLoRAGuide", positive=positivo, negative=negativo, vae=enl["vae"],
                  latent=latente, image=imagen, frame_idx=fotograma, strength=1.0,
                  latent_downscale_factor=enl["factor_guia"], crop=recorte, use_tiled_encode=False, tile_size=256,
                  tile_overlap=64, use_streaming=False, tile_frames=97)
        positivo, negativo, latente = [g, 0], [g, 1], [g, 2]
    final = _muestrear(grafo, pre, "", enl, enl["modelo_render"], pre + "sigmas", positivo, negativo, latente)
    return _decodificar(grafo, pre + "render", enl, final)


def grafo_lento(plan, enlaces, ancho, alto, prefijo):
    """El subgrafo de todas las ventanas del plan, como prompt de API de ComfyUI: {id: {"class_type", "inputs"}}.

    plan: diccionario del plan de CP_LentaPlan. enlaces: [id, indice] de ruido, poses, arte, modelo_suave,
    modelo_render, factor_guia, vae, escalador, positivo_suave, negativo_suave, positivo_render y negativo_render.
    Los ids son prefijo + nombre legible: w2.s1.k13 es la guia de la pose 13 en el primer pase de la etapa 1 de la
    ventana 2. Devuelve (grafo, enlace a la animacion, enlace al playblast): las dos salidas del ultimo CP_LentaElegir."""
    ventanas = comprobar_plan(plan)
    comprobar_tamano(ancho, alto)
    enl = comprobar_enlaces(enlaces)
    texto = json.dumps(plan)
    grafo = {}
    ancla, render, elegido = enl["arte"], None, None
    for w, ventana in enumerate(ventanas):
        pre = f"{prefijo}w{w}."
        guia = _etapa_suave(grafo, pre + "s1.", ventana, enl, ancho, alto)
        if w:
            # ultimo fotograma del render anterior: ImageFromBatch lo copia, no retiene el lote entero
            ancla = [_nodo(grafo, pre + "ancla", "ImageFromBatch", image=render, batch_index=ventanas[w - 1]["n"] - 1, length=1), 0]
        render = _etapa_render(grafo, pre + "s2.", ventana, enl, guia, ancla, ancho, alto)
        entradas = {"guia": guia, "render": render, "plan": texto, "ventana": w}
        if w:
            # lo acumulado va lo ULTIMO: el orden de estas entradas decide el de ejecucion (ver la cabecera)
            entradas.update(anterior_render=[elegido, 0], anterior_guia=[elegido, 1])
        elegido = _nodo(grafo, pre + "elegir", "CP_LentaElegir", **entradas)
    return grafo, [elegido, 0], [elegido, 1]


class BlockingLento(io.ComfyNode):
    """Se expande en el subgrafo de la receta de camara lenta: una etapa suave y un render por ventana, encadenados."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=NODO,
            display_name="Cámara lenta · Blocking a animación",
            category="character-pipeline/poses",
            description="Convierte un blocking a saltos en animacion suave y renderizada: por cada ventana del plan genera "
                        "el playblast a camara lenta, lo renderiza encadenando con la ventana anterior y vuelve a los "
                        "tiempos del blocking.",
            inputs=[
                io.String.Input("plan", force_input=True, tooltip="El de CP_LentaPlan."),
                io.Int.Input("ancho", default=1920, min=128, max=4096, step=64,
                             tooltip="Multiplo de 64. La etapa suave hace su primer pase a la mitad."),
                io.Int.Input("alto", default=1088, min=128, max=4096, step=64, tooltip="Multiplo de 64."),
                io.Noise.Input("ruido", raw_link=True, tooltip="El mismo RandomNoise para todos los muestreos."),
                io.Image.Input("poses", raw_link=True, tooltip="Las de CP_PosesDetectar, las mismas que recibio el plan."),
                io.Image.Input("arte", raw_link=True, tooltip="La primera pose pintada y compuesta: ancla de la primera ventana."),
                io.Model.Input("modelo_suave", raw_link=True, tooltip="LTX sin LoRAs: genera el playblast lento."),
                io.Model.Input("modelo_render", raw_link=True, tooltip="LTX + IC-LoRA Layout-to-Render + LoRA del personaje."),
                io.Float.Input("factor_guia", force_input=True, raw_link=True,
                               tooltip="latent_downscale_factor: la segunda salida de LTXICLoRALoaderModelOnly."),
                io.Vae.Input("vae", raw_link=True),
                io.LatentUpscaleModel.Input("escalador", raw_link=True, tooltip="Escalador espacial x2 de latentes de LTX."),
                io.Conditioning.Input("positivo_suave", raw_link=True, tooltip="Ya pasado por LTXVConditioning a 24 fps."),
                io.Conditioning.Input("negativo_suave", raw_link=True),
                io.Conditioning.Input("positivo_render", raw_link=True, tooltip="Ya pasado por LTXVConditioning a 24 fps."),
                io.Conditioning.Input("negativo_render", raw_link=True),
            ],
            outputs=[
                io.Image.Output(display_name="animacion"),
                io.Image.Output(display_name="playblast"),
            ],
            enable_expand=True,
        )

    @classmethod
    def execute(cls, plan, ancho, alto, ruido, poses, arte, modelo_suave, modelo_render, factor_guia, vae, escalador,
                positivo_suave, negativo_suave, positivo_render, negativo_render) -> io.NodeOutput:
        from comfy_execution.graph_utils import GraphBuilder

        enlaces = dict(ruido=ruido, poses=poses, arte=arte, modelo_suave=modelo_suave, modelo_render=modelo_render,
                       factor_guia=factor_guia, vae=vae, escalador=escalador, positivo_suave=positivo_suave,
                       negativo_suave=negativo_suave, positivo_render=positivo_render, negativo_render=negativo_render)
        datos = leer_plan(plan)
        # el motor fija la raiz del prefijo (id de este nodo y numero de llamada) justo antes de llamar a execute
        grafo, animacion, playblast = grafo_lento(datos, enlaces, ancho, alto, GraphBuilder.alloc_prefix())
        ventanas = datos["ventanas"]
        logging.info("[%s] %d ventanas, %d fotogramas lentos, %d nodos a %dx%d.", NODO, len(ventanas),
                     sum(v["n"] for v in ventanas), len(grafo), ancho, alto)
        return io.NodeOutput(animacion, playblast, expand=grafo)
