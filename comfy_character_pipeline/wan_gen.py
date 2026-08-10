# -*- coding: utf-8 -*-
"""Planificador de GENERACION en Wan 2.2, incluido el troceado en tandas de 81.

Wan 2.2 se entreno con clips de 81 frames a 16 fps. Por encima de eso no da
error: da un video que se repite en bucle, o que pierde coherencia poco a poco.
La forma de hacer clips largos es generar tandas y encadenarlas pasando los
ultimos frames de una como frames iniciales de la siguiente.

Este nodo hace la aritmetica de ese troceado, que es donde se cuela el error:
cuantas tandas hacen falta, en que frame empieza cada una, cuantos frames
solapan y si la ultima se queda corta.
"""

import math

from comfy_api.latest import io

# Resoluciones que la comunidad reporta como bien portadas por Wan 2.2.
# Todas multiplo de 16, que es lo que exige el VAE (compresion 8x + patch 2).
DELIVERY = {
    "vertical 720x1280 (9:16)": (720, 1280),
    "vertical 704x1280 (9:16)": (704, 1280),
    "vertical 480x832 (9:16, rapido)": (480, 832),
    "horizontal 1280x704 (16:9)": (1280, 704),
    "horizontal 1024x576 (16:9)": (1024, 576),
    "horizontal 832x480 (16:9, rapido)": (832, 480),
    "horizontal 1536x864 (16:9, fuera de rango)": (1536, 864),
    "cuadrado 960x960": (960, 960),
    "auto (aspecto de la imagen)": None,
}

MODES = {
    # modo -> (usa_imagen_inicial, usa_referencia, es_continuacion)
    "t2v · solo texto": (False, False, False),
    "i2v · imagen inicial": (True, False, False),
    "ref2v · imagenes de referencia (Phantom)": (False, True, False),
    "continuacion · extiende el clip anterior": (True, False, True),
}

MULT = 16          # exigencia espacial del VAE de Wan
TEMPORAL = 4       # longitud valida = 4n+1
NATIVE_LEN = 81    # ventana con la que se entreno Wan 2.2
NATIVE_PIXELS = 1280 * 720


def _snap(v, m=MULT):
    return max(m, int(round(v / m)) * m)


def _len_4n1(n):
    """Hacia abajo. Para techos: nunca devuelve mas de lo que le pides."""
    n = max(1, int(n))
    return ((n - 1) // TEMPORAL) * TEMPORAL + 1


def _len_4n1_near(n):
    """Al mas cercano. Para longitudes: 5 s x 16 fps = 80 frames, y lo que se
    quiere ahi es 81 — el numero canonico de Wan — no 77."""
    n = max(1, int(n))
    return max(1, int(round((n - 1) / float(TEMPORAL)))) * TEMPORAL + 1


class WanGenPlan(io.ComfyNode):
    """Elige modo, entrega y duracion total; el nodo reparte el trabajo en tandas.

    Los cuatro errores que evita, todos silenciosos:

    1. **Pedir mas de 81 frames de una vez.** No falla: el video se repite en
       bucle o se degrada. El nodo tapa la longitud de cada tanda en 81 y te
       dice cuantas tandas hacen falta.

    2. **`length` tiene que ser 4n+1.** El VAE de Wan comprime 4x en el eje
       temporal. 80 no vale, 81 si. Y 81 frames a 16 fps son 5.06 s, no 5.

    3. **Resolucion multiplo de 16.** 1280x720 NO lo es (720 = 45x16 es correcto,
       pero 704 es lo que usa la comunidad porque 720 se sale del entreno). El
       nodo alinea y avisa.

    4. **El solape se cuenta dos veces.** Al encadenar tandas, los frames que
       compartes con la anterior no son frames nuevos. Si planificas 4 tandas de
       81 con 16 de solape esperando 324 frames, te salen 276. Aqui el conteo
       sale de una sola formula.

    Sobre `overlap_frames`: son los ultimos frames de la tanda anterior que se
    le pasan a `WanImageToVideo` como `start_image`. Wan 2.2 I2V los admite sin
    ningun LoRA extra (Wan 2.1 no podia). Mas solape = transicion mas suave y
    menos frames nuevos por tanda.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CP_WanGenPlan",
            display_name="Wan Gen Plan",
            category="character-pipeline/geometry",
            description="Modo + entrega + duracion total -> resolucion valida, longitud 4n+1 y el troceado en tandas de 81.",
            inputs=[
                io.Combo.Input("modo", options=list(MODES.keys()), default="i2v · imagen inicial"),
                io.Combo.Input("entrega", options=list(DELIVERY.keys()),
                               default="vertical 720x1280 (9:16)"),
                io.Float.Input(
                    "duracion_total_s", default=5.0, min=0.5, max=120.0, step=0.5,
                    tooltip="Segundos del clip COMPLETO, sumando todas las tandas.",
                ),
                io.Int.Input("fps", default=16, min=8, max=30,
                             tooltip="Wan 2.2 es nativo a 16 fps. Sube el frame rate al final, interpolando."),
                io.Int.Input(
                    "chunk_index", default=0, min=0, max=512,
                    tooltip="0 = primera tanda. Sube de uno en uno y vuelve a lanzar para recorrer el clip.",
                ),
                io.Int.Input(
                    "overlap_frames", default=16, min=0, max=64,
                    tooltip=(
                        "Frames de la tanda anterior que se reinyectan como imagen inicial. "
                        "0 en la primera tanda. 8-16 es el rango habitual; mas solape suaviza "
                        "la costura pero cada tanda aporta menos metraje nuevo."
                    ),
                ),
                io.Int.Input(
                    "max_frames_tanda", default=81, min=17, max=161,
                    tooltip=(
                        "Techo por tanda. 81 es la ventana de entreno de Wan 2.2 y el valor "
                        "seguro. Con control (VACE) se puede pasar de ahi perdiendo nitidez "
                        "pero no movimiento."
                    ),
                ),
                io.Image.Input(
                    "imagen", optional=True,
                    tooltip="Opcional. Solo se lee su relacion de aspecto, para 'auto' y para avisar del encuadre.",
                ),
            ],
            outputs=[
                io.Int.Output(display_name="width"),
                io.Int.Output(display_name="height"),
                io.Int.Output(display_name="length"),
                io.Int.Output(display_name="fps"),
                io.Float.Output(display_name="fps_f"),
                io.Int.Output(display_name="total_tandas"),
                io.Int.Output(display_name="start_frame"),
                io.Int.Output(display_name="overlap"),
                io.Boolean.Output(display_name="es_ultima"),
                io.Boolean.Output(display_name="usar_imagen"),
                io.Boolean.Output(display_name="usar_referencia"),
                io.String.Output(display_name="resumen"),
            ],
        )

    @classmethod
    def execute(cls, modo, entrega, duracion_total_s, fps, chunk_index, overlap_frames,
                max_frames_tanda, imagen=None) -> io.NodeOutput:
        usa_img, usa_ref, es_cont = MODES[modo]
        avisos = []

        # --- resolucion -----------------------------------------------------
        src_ar = None
        if imagen is not None and imagen.dim() == 4 and imagen.shape[0] > 0:
            _, sh, sw, _ = imagen.shape
            src_ar = sw / float(sh)

        target = DELIVERY[entrega]
        if target is None:
            if src_ar is None:
                raise ValueError(
                    "[WanGenPlan] Entrega 'auto' necesita una imagen conectada para sacar "
                    "el aspecto. Conecta `imagen` o elige una entrega concreta."
                )
            h = (0.75 * 1_000_000 / src_ar) ** 0.5
            w, h = _snap(h * src_ar), _snap(h)
        else:
            w, h = _snap(target[0]), _snap(target[1])
            if (w, h) != target:
                avisos.append(f"{target[0]}x{target[1]} no es multiplo de {MULT}; ajustado a {w}x{h}.")

        if src_ar is not None and target is not None:
            dst_ar = w / float(h)
            if abs(src_ar - dst_ar) / dst_ar > 0.05:
                avisos.append(
                    f"La imagen es {src_ar:.3f}:1 y la entrega {dst_ar:.3f}:1. Wan no recorta "
                    f"por ti: redimensiona con crop antes de `WanImageToVideo` o saldra estirada."
                )
        if w * h > NATIVE_PIXELS * 1.10:
            avisos.append(
                f"{w}x{h} esta por encima del rango nativo de Wan 2.2 (~720p). Funciona, pero "
                f"aparecen repeticiones y deriva. Genera a 720p y sube con SeedVR2 al final."
            )

        # --- troceado -------------------------------------------------------
        cap = _len_4n1(min(max_frames_tanda, 161))
        if cap > NATIVE_LEN:
            avisos.append(
                f"max_frames_tanda={cap} supera los {NATIVE_LEN} de entreno de Wan 2.2. "
                f"Sin senal de control (VACE / pose) la coherencia se degrada."
            )

        total_frames_deseados = max(1, int(round(duracion_total_s * fps)))
        ov = int(overlap_frames)
        if ov >= cap:
            raise ValueError(
                f"[WanGenPlan] overlap_frames ({ov}) debe ser menor que la longitud de tanda "
                f"({cap}); con solape >= longitud el encadenado no avanza nunca."
            )
        stride = cap - ov                      # frames NUEVOS por tanda

        if total_frames_deseados <= cap:
            total_tandas = 1
        else:
            total_tandas = 1 + math.ceil((total_frames_deseados - cap) / stride)

        if chunk_index >= total_tandas:
            raise ValueError(
                f"[WanGenPlan] chunk_index={chunk_index} pero el plan solo tiene "
                f"{total_tandas} tanda(s) para {total_frames_deseados} frames "
                f"({duracion_total_s:.1f} s a {fps} fps) con tandas de {cap} y solape {ov}.\n"
                "  Has terminado el clip. Baja chunk_index o sube la duracion."
            )

        es_primera = (chunk_index == 0)
        es_ultima = (chunk_index >= total_tandas - 1)
        start_frame = chunk_index * stride
        length = cap
        if es_ultima:
            restantes = total_frames_deseados - start_frame
            length = min(cap, max(TEMPORAL + 1, _len_4n1_near(restantes)))

        ov_efectivo = 0 if es_primera else ov
        frames_totales = stride * (total_tandas - 1) + cap
        seg_reales = frames_totales / float(fps)

        # --- coherencia del modo --------------------------------------------
        if es_cont and es_primera:
            avisos.append(
                "Modo continuacion con chunk_index=0: no hay tanda anterior de la que tirar. "
                "Genera la primera con i2v o t2v y vuelve aqui con chunk_index=1."
            )
        if usa_img and not es_cont and not es_primera:
            avisos.append(
                f"Estas en la tanda {chunk_index} pero el modo es '{modo}'. Para encadenar, "
                f"cambia a 'continuacion' y alimenta los ultimos {ov} frames de la tanda "
                f"anterior como imagen inicial."
            )
        if usa_ref:
            avisos.append(
                "ref2v usa `WanPhantomSubjectToVideo`, que necesita el modelo Phantom y solo "
                "actua en el paso LOW. El paso HIGH hay que saltarlo o dejar que solo "
                "establezca la composicion."
            )
        if fps != 16:
            avisos.append(
                f"{fps} fps: Wan 2.2 se entreno a 16. Generar a otro frame rate cambia la "
                f"velocidad percibida del movimiento. Lo normal es generar a 16 e interpolar."
            )

        resumen = "\n".join([
            f"modo         {modo}",
            f"resolucion   {w}x{h}   ({w * h / 1e6:.2f} MP, multiplo de {MULT})",
            f"tanda        {chunk_index + 1} de {total_tandas}",
            f"length       {length}   (4n+1, n={(length - 1) // TEMPORAL})",
            f"solape       {ov_efectivo}   ->  {length - ov_efectivo} frames nuevos en esta tanda",
            f"empieza en   frame {start_frame}",
            f"clip final   {frames_totales} frames  =  {seg_reales:.2f} s a {fps} fps",
            "",
            f"imagen inicial   {'SI' if usa_img else 'no'}",
            f"referencias      {'SI' if usa_ref else 'no'}",
            f"ultima tanda     {'SI' if es_ultima else 'no'}",
            "",
            ("\n".join("AVISO: " + a for a in avisos) if avisos else "Todo cuadra."),
        ])

        return io.NodeOutput(
            w, h, length, int(fps), float(fps),
            total_tandas, start_frame, ov_efectivo, es_ultima,
            usa_img, usa_ref, resumen,
        )
