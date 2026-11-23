# -*- coding: utf-8 -*-
"""Suplente de MiniMaxH3ImageConditioning para los motores locales con ComfyUI anterior a la 0.45.

La tarjeta MINIMAX H3 IMAGE usa el nodo del nucleo `MiniMaxH3ImageConditioning`, que existe desde ComfyUI 0.45.0
(comfy_extras/nodes_minimax_h3.py). La nube va en la 0.45.6, pero el motor local de cada puesto (el que carga la tarjeta
y construye el prompt que luego se manda a la nube) sigue en la 0.42.1 y no lo conoce: sin el, el frontend exporta el
nodo sin class_type y el panel no deja generar.

Este archivo registra un nodo con el MISMO node_id y el MISMO esquema que el de la v0.45.6 (entradas, tipos, rangos,
pasos, el Autogrow `images` con prefijo «image_» de 0 a 100 y las dos salidas), copiado de la etiqueta v0.45.6. Asi el
frontend local lo dibuja y serializa igual que lo haria un ComfyUI 0.45.6, y el prompt que sale es el que la nube espera.

- En el motor local NO se ejecuta: si alguien lo lanza en local, falla con un mensaje claro (la app nunca ejecuta las
  tarjetas en local; las manda a la nube).
- En la nube (0.45 o posterior) NO hace nada: nodes.py pasa los nombres del nucleo como `ignore` al cargar los custom
  nodes (init_external_custom_nodes -> load_custom_node(..., base_node_names); en la rama V3, `if schema.node_id not in
  ignore`), asi que gana siempre el nodo del nucleo. Lo mismo cuando el motor local suba a 0.45 o posterior.
- En la 0.42.1 ese nombre no esta entre los del nucleo, asi que el suplente si se registra.

Cuando todos los puestos tengan un motor 0.45 o posterior, este archivo se puede quitar.
"""

from comfy_api.latest import io

NODO = "MiniMaxH3ImageConditioning"
# Copiados de la v0.45.6: nodes_minimax_h3.CANVAS_MULTIPLE y nodes.MAX_RESOLUTION (este ultimo, igual en la 0.42.1).
CANVAS_MULTIPLE = 32
MAX_RESOLUTION = 16384

MENSAJE = (
    f"[{NODO}] Este nodo solo funciona en la nube, con ComfyUI 0.45 o posterior. Lo que hay en este equipo es un "
    "suplente del pack Character Pipeline para que el motor local (ComfyUI anterior a la 0.45) pueda abrir la tarjeta "
    "MINIMAX H3 IMAGE y preparar el envio. Lanza la tarjeta desde su interfaz en BITT Studio: se ejecuta en la nube."
)


class SuplenteMiniMaxH3ImageConditioning(io.ComfyNode):
    """Mismo esquema que MiniMaxH3ImageConditioning de ComfyUI v0.45.6; no se ejecuta en local."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=NODO,
            display_name="MiniMax H3 Image Conditioning",
            category="model/conditioning/minimax",
            description="Text-to-image, or with images an edit guided by them. The prompt refers to them as <Picture 1>, <Picture 2>, ...",
            inputs=[
                io.Clip.Input("clip"),
                io.Vae.Input("vae", optional=True, tooltip="The MiniMax-H3-Image VAE. With it the pictures are also anchored as reference latents, without it they only condition the text encoder."),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                io.Int.Input("width", default=0, min=0, max=MAX_RESOLUTION, step=CANVAS_MULTIPLE,
                             tooltip="0 uses the width of image_1 after its reference scaling, or 2048 without images. The model is tuned for 2048x2048 pixels of area"),
                io.Int.Input("height", default=0, min=0, max=MAX_RESOLUTION, step=CANVAS_MULTIPLE,
                             tooltip="0 uses the height of image_1 after its reference scaling, or 2048 without images."),
                io.Autogrow.Input("images", optional=True, template=io.Autogrow.TemplatePrefix(
                    input=io.Image.Input("image", tooltip="Reference picture, <Picture n> in the prompt; image_1 is the picture to edit."), prefix="image_", min=0, max=100)),
            ],
            outputs=[io.Conditioning.Output(display_name="positive"), io.Latent.Output()],
        )

    @classmethod
    def execute(cls, clip, prompt, width, height, vae=None, images=None) -> io.NodeOutput:
        raise RuntimeError(MENSAJE)
