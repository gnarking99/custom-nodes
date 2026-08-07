# -*- coding: utf-8 -*-
"""Selector de resolucion para LTX: eliges la entrega, el resto se calcula."""

from comfy_api.latest import io

# entregas habituales. None = tomar la del video de origen
DELIVERY = {
    "UHD 4K 3840x2160": (3840, 2160),
    "DCI 4K 4096x2160": (4096, 2160),
    "2K 2560x1440": (2560, 1440),
    "1080p 1920x1080": (1920, 1080),
    "720p 1280x720": (1280, 720),
    "vertical 4K 2160x3840": (2160, 3840),
    "vertical 1080p 1080x1920": (1080, 1920),
    "auto (igual que el video)": None,
}

STAGES = {"x2 (una etapa)": 2, "x4 (dos etapas)": 4}
MULT = {"32 (VAE de LTX)": 32, "64 (recomendado con LoRA)": 64}


def _snap(v, m):
    """Redondea SIEMPRE hacia arriba al multiplo.

    Hacia abajo la cadena se queda corta: 2160/4 = 540, que al redondear al 64
    mas cercano cae a 512 y x4 acaba en 2048, por debajo de la entrega. El
    ultimo salto lo tendria que dar un reescalado normal, sin detalle nuevo.
    Hacia arriba sobra un poco y el reescalado final es una reduccion de unos
    pocos pixeles, que no se ve.
    """
    import math
    return max(m, int(math.ceil(v / m)) * m)


class LTXResolution(io.ComfyNode):
    """Un solo selector: eliges lo que hay que ENTREGAR y el nodo calcula hacia
    atras todas las resoluciones intermedias, respetando la relacion de aspecto
    del video original.

    El problema que resuelve: poner la resolucion a mano en LTX obliga a escribir
    el mismo par de numeros en tres sitios (el latente vacio y las dos
    referencias redimensionadas), acordarse de si el multiplo es 32 o 64, y
    voltearlos cuando el video es vertical. Cualquier descuadre entre esos tres
    sitios da un fallo raro o una salida deformada.

    Aqui se calcula al reves: de la entrega hacia la base.

        base = entrega / etapas, ajustado al multiplo
        x2   = base * 2
        x4   = base * 4
        final = la entrega exacta

    Por que hacia atras: si eliges la base y multiplicas, acabas en medidas como
    4096x2304 cuando querias 3840x2160, y hay que recortar. Partiendo de la
    entrega, la unica correccion es un reescalado final de unos pocos pixeles,
    que es invisible.

    La relacion de aspecto sale del video de entrada, no de la entrega: si tu
    clip es 1280x720 y pides UHD, la salida mantiene 16:9. Si tu clip es 9:16 y
    pides UHD horizontal, el nodo respeta TU relacion y ajusta el alto.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CP_LTXResolution",
            display_name="LTX Resolution",
            category="character-pipeline/geometry",
            description="Elige la entrega y calcula base, x2, x4 y final respetando el aspecto del video.",
            inputs=[
                io.Image.Input("image", tooltip="Un frame del video de origen. Solo se lee su relacion de aspecto."),
                io.Combo.Input("entrega", options=list(DELIVERY.keys()), default="UHD 4K 3840x2160"),
                io.Combo.Input(
                    "etapas", options=list(STAGES.keys()), default="x4 (dos etapas)",
                    tooltip=(
                        "Cuantas pasadas del upscaler latente. x4 da mas detalle "
                        "pero la ultima etapa cuesta 16 veces mas atencion que la anterior."
                    ),
                ),
                io.Combo.Input(
                    "multiplo", options=list(MULT.keys()), default="64 (recomendado con LoRA)",
                    tooltip=(
                        "El VAE de LTX exige multiplo de 32. Con LoRA de personaje la "
                        "comunidad reporta mejor comportamiento con 64."
                    ),
                ),
                io.Boolean.Input(
                    "respetar_aspecto_original", default=True,
                    tooltip=(
                        "Activo: la relacion de aspecto sale de TU video y la entrega solo "
                        "marca cuantos pixeles. Apagado: se fuerza la relacion de la entrega "
                        "(deforma si no coinciden)."
                    ),
                ),
                io.Float.Input(
                    "limite_base_mp", default=1.2, min=0.1, max=4.0, step=0.05,
                    tooltip=(
                        "Techo de megapixeles para la etapa base. Protege de pedir 4K con "
                        "una sola etapa y acabar generando a 1920x1080 en el paso caro."
                    ),
                ),
            ],
            outputs=[
                io.Int.Output(display_name="base_w"),
                io.Int.Output(display_name="base_h"),
                io.Int.Output(display_name="x2_w"),
                io.Int.Output(display_name="x2_h"),
                io.Int.Output(display_name="x4_w"),
                io.Int.Output(display_name="x4_h"),
                io.Int.Output(display_name="final_w"),
                io.Int.Output(display_name="final_h"),
                io.String.Output(display_name="resumen"),
            ],
        )

    @classmethod
    def execute(cls, image, entrega, etapas, multiplo, respetar_aspecto_original,
                limite_base_mp) -> io.NodeOutput:
        if image is None or image.dim() != 4 or image.shape[0] == 0:
            raise ValueError("[LTXResolution] Necesito un frame del video de origen.")

        _, sh, sw, _ = image.shape
        src_ar = sw / float(sh)

        target = DELIVERY[entrega]
        if target is None:
            fw, fh = int(sw), int(sh)
        else:
            tw, th = target
            if respetar_aspecto_original:
                # mismos pixeles que la entrega, pero con TU relacion de aspecto
                px = tw * th
                fh = (px / src_ar) ** 0.5
                fw = fh * src_ar
                fw, fh = int(round(fw)), int(round(fh))
            else:
                fw, fh = tw, th

        n = STAGES[etapas]
        m = MULT[multiplo]

        base_w = _snap(fw / n, m)
        base_h = _snap(fh / n, m)

        # techo de megapixeles en la etapa base
        avisos = []
        mp = base_w * base_h / 1_000_000.0
        if mp > limite_base_mp:
            k = (limite_base_mp / mp) ** 0.5
            nw, nh = _snap(base_w * k, m), _snap(base_h * k, m)
            avisos.append(
                f"La base salia a {base_w}x{base_h} ({mp:.2f} MP), por encima del limite "
                f"{limite_base_mp} MP. Reducida a {nw}x{nh}. Sube `etapas` a x4 o el limite "
                f"si de verdad quieres esa base."
            )
            base_w, base_h = nw, nh

        x2_w, x2_h = base_w * 2, base_h * 2
        x4_w, x4_h = base_w * 4, base_h * 4

        alcanzado = (x2_w, x2_h) if n == 2 else (x4_w, x4_h)
        if alcanzado[0] < fw or alcanzado[1] < fh:
            avisos.append(
                f"Las etapas llegan a {alcanzado[0]}x{alcanzado[1]} y la entrega es "
                f"{fw}x{fh}: el ultimo salto lo hace un reescalado normal, sin detalle nuevo. "
                f"Sube `etapas` para que el modelo lo reconstruya."
            )

        resumen = "\n".join([
            f"origen      {sw}x{sh}   (aspecto {src_ar:.3f})",
            f"base        {base_w}x{base_h}   {base_w*base_h/1e6:.2f} MP  <- aqui se genera",
            f"etapa x2    {x2_w}x{x2_h}",
            (f"etapa x4    {x4_w}x{x4_h}" if n == 4 else "etapa x4    (desactivada)"),
            f"ENTREGA     {fw}x{fh}",
            f"multiplo    {m}",
            "",
            ("\n".join("AVISO: " + a for a in avisos) if avisos
             else "Todo cuadra: las etapas llegan a la entrega o la superan."),
        ])

        return io.NodeOutput(base_w, base_h, x2_w, x2_h, x4_w, x4_h, fw, fh, resumen)
