# -*- coding: utf-8 -*-
"""Planificador de GENERACION en LTX-2.3: un solo nodo decide modo, medidas y frames.

Distinto de `CP_LTXResolution`, que parte de un video existente (v2v) y calcula
hacia atras desde la entrega. Aqui no hay video de origen: hay un modo (t2v /
i2v / ref2v / flf2v), una duracion en segundos y una entrega. El nodo devuelve
las tres cosas que en LTX se escriben a mano en seis sitios distintos y que, si
no cuadran, no dan error: dan un video deformado o un fallo raro a mitad de cola.
"""

import math

from comfy_api.latest import io

# La entrega es lo que sale del ULTIMO upscaler latente, no del primer sampler.
DELIVERY = {
    "vertical 1080x1920 (9:16)": (1080, 1920),
    "vertical 1440x2560 (9:16)": (1440, 2560),
    "horizontal 1920x1080 (16:9)": (1920, 1080),
    "horizontal 2560x1440 (16:9)": (2560, 1440),
    "cuadrado 1440x1440 (1:1)": (1440, 1440),
    "cine 2048x858 (2.39:1)": (2048, 858),
    "auto (aspecto de la imagen)": None,
}

MODES = {
    # modo -> (usa_imagen_inicial, usa_hoja_referencia, usa_ultimo_frame)
    "t2v · solo texto": (False, False, False),
    "i2v · imagen inicial": (True, False, False),
    "ref2v · hoja de referencia (IC-LoRA)": (False, True, False),
    "i2v + ref2v · imagen inicial + hoja": (True, True, False),
    "flf2v · primer y ultimo frame": (True, False, True),
}

STAGES = {"x2 · dos pasadas": 2, "x4 · tres pasadas": 4}
MULT = {"32 (minimo del VAE)": 32, "64 (recomendado con LoRA)": 64}

# LTX-2.3 es nativo hasta 2560x1440. Por encima el upscaler latente sigue
# funcionando pero el sampler ya trabaja fuera de su distribucion de tokens.
NATIVE_PIXELS = 2560 * 1440


def _snap(v, m):
    """Redondea hacia arriba al multiplo. Ver la nota larga en `ltx_res.py`."""
    return max(m, int(math.ceil(v / m)) * m)


def _frames_8n1(seconds, fps):
    """LTX comprime 8x en el eje temporal: la longitud valida es 8n+1.

    ComfyUI no se queja si le pides 126 frames — redondea por dentro y el audio
    y el video acaban desalineados por unos milisegundos. Mejor redondear aqui
    y saber exactamente cuantos frames se van a generar.
    """
    raw = seconds * fps
    n = max(1, int(round((raw - 1) / 8.0)))
    return n * 8 + 1


class LTXGenPlan(io.ComfyNode):
    """Elige el modo y la entrega; el nodo calcula todo lo demas.

    Los tres problemas que resuelve, cada uno de una sesion perdida:

    1. **La resolucion base no es la entrega.** En LTX la etapa 1 genera a la
       MITAD (o a un cuarto) y el upscaler latente dobla. Poner la entrega en
       `EmptyLTXVLatentVideo` es el error mas caro del flujo: generas el paso
       lento a 4x los pixeles que necesitabas y el upscaler te saca del rango
       nativo del modelo.

    2. **`frames` tiene que ser 8n+1** y sale de duracion x fps, que son dos
       numeros que el artista si quiere tocar. 5 s x 24 fps = 120, que NO es
       8n+1: lo valido de al lado es 121.

    3. **El modo decide que ramas del grafo estan vivas.** Las salidas booleanas
       se cablean directamente a los `bypass` de los nodos de imagen y a los
       switches de grupo, asi que cambiar de t2v a ref2v es mover UN desplegable
       en vez de acordarse de cuatro bypass.

    Hay dos salidas de fps, `fps` (INT) y `fps_f` (FLOAT), porque LTX las pide de
    los dos tipos en el mismo flujo: `LTXVEmptyLatentAudio` quiere INT y
    `LTXVConditioning`, `CreateVideo` y `VHS_VideoCombine` quieren FLOAT. ComfyUI
    no convierte entre ambos al cablear, y tener dos primitivas separadas es
    justo la clase de duplicado que hace que el audio se desincronice del video
    cuando cambias uno y olvidas el otro.

    La salida `bypass_imagen` esta invertida a proposito: `LTXVImgToVideoInplace`
    y `LTXVImgToVideoConditionOnly` tienen una entrada llamada `bypass`, no
    `enable`. Cablear la booleana correcta evita el clasico "he puesto i2v y me
    sigue ignorando la imagen".
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CP_LTXGenPlan",
            display_name="LTX Gen Plan",
            category="character-pipeline/geometry",
            description="Modo (t2v/i2v/ref2v/flf2v) + entrega + duracion -> medidas por etapa, frames 8n+1 y booleanas de rama.",
            inputs=[
                io.Combo.Input("modo", options=list(MODES.keys()), default="i2v · imagen inicial"),
                io.Combo.Input("entrega", options=list(DELIVERY.keys()), default="vertical 1080x1920 (9:16)"),
                io.Combo.Input(
                    "etapas", options=list(STAGES.keys()), default="x2 · dos pasadas",
                    tooltip=(
                        "Cuantas pasadas del upscaler latente. x4 anade una tercera etapa "
                        "con 4 veces los pixeles: la atencion es O(n^2), asi que cuesta "
                        "~16 veces la anterior para darte 2-3 pasos de refinado."
                    ),
                ),
                io.Combo.Input(
                    "multiplo", options=list(MULT.keys()), default="64 (recomendado con LoRA)",
                    tooltip=(
                        "El VAE de LTX exige multiplo de 32. Con LoRA de personaje la "
                        "comunidad reporta perdida de movimiento en medidas que no son "
                        "multiplo de 64 (1280x704 si, 1280x720 no)."
                    ),
                ),
                io.Float.Input(
                    "duracion_s", default=5.0, min=0.5, max=60.0, step=0.5,
                    tooltip="Segundos de video. Se convierte a frames 8n+1 con el fps de abajo.",
                ),
                io.Int.Input("fps", default=24, min=8, max=60),
                io.Float.Input(
                    "limite_base_mp", default=0.60, min=0.10, max=4.00, step=0.05,
                    tooltip=(
                        "Techo de megapixeles de la ETAPA 1. Es el paso caro: se ejecuta "
                        "entero, con todos los frames y todos los pasos. 0.55-0.65 MP "
                        "(960x544, 1080x544...) es la zona donde LTX-2.3 fue entrenado."
                    ),
                ),
                io.Image.Input(
                    "imagen", optional=True,
                    tooltip=(
                        "Opcional. Solo se lee su relacion de aspecto, para 'auto' y para "
                        "avisarte si el encuadre de la imagen no coincide con la entrega."
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
                io.Int.Output(display_name="frames"),
                io.Int.Output(display_name="fps"),
                io.Float.Output(display_name="fps_f"),
                io.Boolean.Output(display_name="usar_imagen"),
                io.Boolean.Output(display_name="bypass_imagen"),
                io.Boolean.Output(display_name="usar_referencia"),
                io.Boolean.Output(display_name="usar_ultimo_frame"),
                io.String.Output(display_name="resumen"),
            ],
        )

    @classmethod
    def execute(cls, modo, entrega, etapas, multiplo, duracion_s, fps,
                limite_base_mp, imagen=None) -> io.NodeOutput:
        usa_img, usa_ref, usa_last = MODES[modo]
        n = STAGES[etapas]
        m = MULT[multiplo]
        avisos = []

        # --- aspecto y entrega ---------------------------------------------
        src_ar = None
        if imagen is not None and imagen.dim() == 4 and imagen.shape[0] > 0:
            _, sh, sw, _ = imagen.shape
            src_ar = sw / float(sh)

        target = DELIVERY[entrega]
        if target is None:
            if src_ar is None:
                raise ValueError(
                    "[LTXGenPlan] Entrega 'auto' necesita una imagen conectada para "
                    "sacar el aspecto. Conecta `imagen` o elige una entrega concreta."
                )
            # 2 MP con el aspecto de la imagen
            fh = (2_000_000 / src_ar) ** 0.5
            fw, fh = int(round(fh * src_ar)), int(round(fh))
        else:
            fw, fh = target

        if src_ar is not None and target is not None:
            dst_ar = fw / float(fh)
            if abs(src_ar - dst_ar) / dst_ar > 0.05:
                avisos.append(
                    f"La imagen es {src_ar:.3f}:1 y la entrega {dst_ar:.3f}:1. "
                    f"LTX no recorta por ti: la imagen entra estirada salvo que la "
                    f"redimensiones con crop='center' antes del preprocess."
                )

        # --- medidas por etapa ---------------------------------------------
        base_w = _snap(fw / n, m)
        base_h = _snap(fh / n, m)

        mp = base_w * base_h / 1_000_000.0
        if mp > limite_base_mp:
            k = (limite_base_mp / mp) ** 0.5
            nw, nh = _snap(base_w * k, m), _snap(base_h * k, m)
            avisos.append(
                f"La base salia a {base_w}x{base_h} ({mp:.2f} MP), por encima del limite "
                f"{limite_base_mp:.2f} MP. Reducida a {nw}x{nh}. Si de verdad quieres esa "
                f"base, sube `limite_base_mp`; si quieres mas entrega, sube `etapas`."
            )
            base_w, base_h = nw, nh

        x2_w, x2_h = base_w * 2, base_h * 2
        x4_w, x4_h = base_w * 4, base_h * 4
        alcanzado = (x2_w, x2_h) if n == 2 else (x4_w, x4_h)

        if alcanzado[0] < fw or alcanzado[1] < fh:
            avisos.append(
                f"Las etapas llegan a {alcanzado[0]}x{alcanzado[1]} y la entrega es "
                f"{fw}x{fh}: el ultimo salto lo da un reescalado normal, sin detalle "
                f"nuevo. Ese hueco lo tapa mejor SeedVR2 que un `ImageScale`."
            )
        if alcanzado[0] * alcanzado[1] > NATIVE_PIXELS * 1.15:
            avisos.append(
                f"{alcanzado[0]}x{alcanzado[1]} supera el rango nativo de LTX-2.3 "
                f"(2560x1440). El upscaler latente sigue funcionando, pero el sampler "
                f"de esa etapa trabaja fuera de su distribucion de tokens: espera "
                f"texturas repetidas. Para pasar de ahi, upscaler de video al final."
            )

        # --- frames ---------------------------------------------------------
        frames = _frames_8n1(duracion_s, fps)
        real_s = frames / float(fps)
        if abs(real_s - duracion_s) > 0.5 / fps:
            avisos.append(
                f"{duracion_s:.2f} s x {fps} fps = {duracion_s * fps:.0f} frames, que no es "
                f"8n+1. Ajustado a {frames} frames = {real_s:.2f} s."
            )
        if frames > 241:
            avisos.append(
                f"{frames} frames ({real_s:.1f} s) en una sola pasada: por encima de "
                f"~10 s la coherencia de LTX-2.3 se degrada y la VRAM sube con el "
                f"cuadrado de los tokens. Trocea y encadena con el ultimo frame."
            )

        # --- coherencia modo <-> entradas ------------------------------------
        if usa_img and imagen is None:
            avisos.append(
                f"Modo '{modo}' pero no hay imagen conectada a este nodo. No es un "
                f"error (la imagen puede entrar directa al flujo), pero sin ella no "
                f"puedo comprobarte el encuadre."
            )
        if usa_ref:
            avisos.append(
                "ref2v activo: la hoja de referencia entra por `LTXAddVideoICLoRAGuide` "
                "/ `LTXVAddGuide`, NO por el latente. Necesita su IC-LoRA cargado "
                "(Ingredients / MSR / Best-Face-ID) o el guide no hace nada."
            )

        resumen = "\n".join([
            f"modo        {modo}",
            f"base        {base_w}x{base_h}   {base_w * base_h / 1e6:.2f} MP  <- ETAPA 1, el paso caro",
            f"etapa x2    {x2_w}x{x2_h}",
            (f"etapa x4    {x4_w}x{x4_h}" if n == 4 else "etapa x4    (desactivada)"),
            f"ENTREGA     {fw}x{fh}   (multiplo {m})",
            f"frames      {frames}   =  {real_s:.2f} s a {fps} fps   (8n+1 con n={(frames - 1) // 8})",
            "",
            f"imagen inicial   {'SI' if usa_img else 'no'}",
            f"hoja referencia  {'SI' if usa_ref else 'no'}",
            f"ultimo frame     {'SI' if usa_last else 'no'}",
            "",
            ("\n".join("AVISO: " + a for a in avisos) if avisos
             else "Todo cuadra."),
        ])

        return io.NodeOutput(
            base_w, base_h, x2_w, x2_h, x4_w, x4_h, fw, fh,
            frames, int(fps), float(fps),
            usa_img, (not usa_img), usa_ref, usa_last,
            resumen,
        )
