# -*- coding: utf-8 -*-
"""Que un video con etiquetas mal codificadas no tumbe la carga.

Los .mov que exporta QuickTime en un sistema en espanol llevan en la pista de video la etiqueta
handler_name = "Gestor de contenido de video Apple" con la i acentuada en Latin-1 (byte 0xED). PyAV hasta la 18
(comprobado de la 14.2 a la 18.1) decodifica las etiquetas como UTF-8 estricto y av.open() falla entero con
"'utf-8' codec can't decode byte 0xed in position 24", asi que LoadVideo no llega a leer ni un fotograma.
PyAV 19 quito el argumento metadata_errors y ya tolera estas etiquetas; aqui se pide lo mismo a las versiones que
lo aceptan. Se usa "replace" y no "surrogateescape": con este ultimo falla despues quien vuelva a escribir la etiqueta.

Solo cambia como se decodifican las ETIQUETAS de texto del contenedor (los bytes que no son UTF-8 salen como el
caracter de sustitucion). No toca fotogramas, audio, tiempos ni la escritura de video.
"""

import logging

_MARCA = "_cp_metadatos_tolerantes"


def _envolver(original):
    """av.open que pide metadata_errors='replace' si quien llama no dice nada y la version lo admite."""
    estado = {"admite": None}  # None = sin probar; False = esta version de PyAV no tiene ese argumento

    def abrir(*args, **kwargs):
        # metadata_errors es el 8.o argumento posicional: con 8 o mas, quien llama ya lo ha indicado.
        if estado["admite"] is not False and "metadata_errors" not in kwargs and len(args) < 8:
            try:
                r = original(*args, metadata_errors="replace", **kwargs)
                estado["admite"] = True
                return r
            except TypeError as e:
                # PyAV 19+ ya no tiene el argumento (y ya tolera las etiquetas): se llama como siempre. Ese TypeError
                # salta antes de tocar el archivo, asi que repetir la apertura no tiene efectos.
                if "unexpected keyword argument 'metadata_errors'" not in str(e):
                    raise
                estado["admite"] = False
        return original(*args, **kwargs)

    abrir.__name__ = getattr(original, "__name__", "open")
    abrir.__doc__ = getattr(original, "__doc__", None)
    abrir.__wrapped__ = original
    setattr(abrir, _MARCA, True)
    return abrir


def tolerar_metadatos_de_video(av=None):
    """Sustituye av.open por la version tolerante. Devuelve 'puesto', 'ya estaba' o 'sin av'. No lanza nunca."""
    try:
        if av is None:
            import av  # noqa: PLC0415  (import pesado: solo al registrar el pack)
        original = av.open
        if getattr(original, _MARCA, False):
            return "ya estaba"
        tolerante = _envolver(original)
        av.open = tolerante
        contenedor = getattr(av, "container", None)
        for modulo in (contenedor, getattr(contenedor, "core", None)):
            if modulo is not None and getattr(modulo, "open", None) is original:
                modulo.open = tolerante
        return "puesto"
    except Exception as e:  # sin PyAV, o una version rara: el pack tiene que cargar igual
        logging.info("[CP] No se pudo preparar la lectura tolerante de etiquetas de video: %s", e)
        return "sin av"
