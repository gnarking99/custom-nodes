# -*- coding: utf-8 -*-
"""Camara lenta: la animacion a doses o a treses, como stop motion.

CP_BlockingLento saca la animacion a unos: un dibujo distinto en cada fotograma. Este nodo la pasa a una cadencia
de c fotogramas por dibujo REPITIENDO fotogramas de la propia animacion: no genera nada, no cambia la duracion ni los
fps y no cuesta GPU. Necesita el mismo plan de CP_LentaPlan que recibio CP_BlockingLento, porque la regla va salto a
salto:
- cada pose sigue en su fotograma (inicios del plan) y la espera quieta de un salto largo (los d - m primeros
  fotogramas, con m = min(d, transicion_max)) no se toca;
- los m fotogramas del movimiento se parten en n = max(1, m // c) bloques seguidos, de base + 1 los primeros y de
  base los demas (base, resto = divmod(m, n)), y cada bloque ensena su primer fotograma. El primer bloque empieza en
  la pose. Cuando m >= c cada bloque dura de c a 2c - 1 fotogramas; cuando m < c el salto entero es un bloque y se
  queda en la pose;
- lo que sigue a la ultima pose (la cola) tampoco se toca.
Con poses muy juntas (saltos de 2 a 5 fotogramas) a treses casi todos los saltos quedan en un bloque y la animacion
vuelve a ser el blocking: la linea info lo avisa.

Va DESPUES de todo lo que genera (la cadencia tiene que ser el ultimo paso) y solo sobre la animacion: el tensor que
recibe es de la cache del motor (la salida del ultimo CP_LentaElegir) y nunca se escribe encima. Con cadencia 1 se
devuelve el MISMO tensor; con otra, una sola copia indexada.
"""

import logging
import numbers

import torch
from comfy_api.latest import io

try:
    from .lenta import leer_plan as _leer_plan
except ImportError:  # las pruebas cargan los modulos sueltos, sin el paquete
    from lenta import leer_plan as _leer_plan

NODO = "CP_Cadencia"
MAX_CADENCIA = 4
NOMBRES = {1: "A unos", 2: "A doses", 3: "A treses", 4: "A cuatros"}


def comprobar_cadencia(cadencia, nodo=NODO):
    """La cadencia como entero de 1 a MAX_CADENCIA. Ni bool ni float (tampoco 2.0): un valor enlazado no lo valida
    el motor y un INT del panel siempre llega como entero."""
    if isinstance(cadencia, bool) or not isinstance(cadencia, numbers.Integral) or not 1 <= int(cadencia) <= MAX_CADENCIA:
        raise ValueError(
            f"[{nodo}] La cadencia tiene que ser un entero de 1 a {MAX_CADENCIA} (1 a unos, 2 a doses, 3 a treses, "
            f"4 a cuatros) y llego {cadencia!r}."
        )
    return int(cadencia)


def _es_entero(x):
    """El mismo criterio que comprobar_cadencia: un entero (tambien de numpy), ni bool ni float."""
    return isinstance(x, numbers.Integral) and not isinstance(x, bool)


def _comprobar_enteros(inicios, total, transicion_max, nodo):
    """Los tiempos tienen que ser enteros ANTES de convertirlos: int() truncaria 12.7, '12', True o 4.9 sin avisar."""
    for x in inicios:
        if not _es_entero(x):
            raise ValueError(f"[{nodo}] Los inicios de las poses tienen que ser enteros y llego {x!r}.")
    if not _es_entero(total):
        raise ValueError(f"[{nodo}] El total de fotogramas tiene que ser un entero y llego {total!r}.")
    if not _es_entero(transicion_max):
        raise ValueError(f"[{nodo}] transicion_max tiene que ser un entero y llego {transicion_max!r}.")


def _comprobar_tiempos(inicios, total, transicion_max, nodo):
    """Lo que la regla necesita del plan. leer_plan ya lo garantiza; aqui se repite para quien llame suelto."""
    if not inicios or inicios[0] != 0 or any(b <= a for a, b in zip(inicios, inicios[1:])):
        raise ValueError(f"[{nodo}] Los inicios de las poses tienen que empezar en 0 y ser crecientes: {list(inicios)[:6]}...")
    if total < inicios[-1] + 1:
        raise ValueError(f"[{nodo}] El plan dice tener {total} fotogramas y la ultima pose empieza en el {inicios[-1]}.")
    if transicion_max < 1:
        raise ValueError(f"[{nodo}] transicion_max tiene que ser al menos 1 y llego {transicion_max}.")


def _bloques(m, c):
    """(n, base, resto) del movimiento de un salto: n dibujos, los `resto` primeros de base + 1 fotogramas."""
    n = max(1, m // c)
    base, resto = divmod(m, n)
    return n, base, resto


def indices_cadencia(inicios, total, transicion_max, cadencia, nodo=NODO):
    """Por cada fotograma de salida, de que fotograma de la animacion de entrada sale (lista de total enteros).
    Con cadencia 1 es la identidad."""
    c = comprobar_cadencia(cadencia, nodo)
    _comprobar_enteros(inicios, total, transicion_max, nodo)
    inicios, total, transicion_max = [int(x) for x in inicios], int(total), int(transicion_max)
    _comprobar_tiempos(inicios, total, transicion_max, nodo)
    idx = list(range(total))
    if c == 1:
        return idx
    for a, b in zip(inicios, inicios[1:]):
        d = b - a
        m = min(d, transicion_max)
        f = a + d - m  # de a a f - 1 es la espera quieta: no se toca
        n, base, resto = _bloques(m, c)
        for i in range(n):
            t = base + (1 if i < resto else 0)
            idx[f:f + t] = [f] * t  # todo el bloque ensena su primer fotograma
            f += t
        if f != b:  # no puede pasar: n bloques que suman base * n + resto = m
            raise RuntimeError(f"[{nodo}] Los bloques del salto {a}-{b} acaban en el fotograma {f}.")
    return idx


def resumen_cadencia(inicios, total, transicion_max, cadencia):
    """Una linea para info: cuantos dibujos quedan y el aviso de los saltos que se quedan enteros en la pose."""
    c = comprobar_cadencia(cadencia)
    _comprobar_enteros(inicios, total, transicion_max, NODO)
    inicios, total, transicion_max = [int(x) for x in inicios], int(total), int(transicion_max)
    _comprobar_tiempos(inicios, total, transicion_max, NODO)
    movimientos = [min(b - a, transicion_max) for a, b in zip(inicios, inicios[1:])]
    dibujos = sum(_bloques(m, c)[0] for m in movimientos) + 1  # mas la ultima pose
    texto = f"{NOMBRES[c]}: {dibujos} dibujos en {total} fotogramas."
    if c == 1:
        return texto + " La animacion sale tal cual de CP_BlockingLento."
    # saltos con movimiento (m >= 2) que con esta cadencia son un solo bloque: se quedan enteros en la pose
    # m >= 2 y no d >= 2: con m = 1 (un salto que se mueve en un solo fotograma) la cadencia no cambia nada
    con_movimiento = [m for m in movimientos if m >= 2]
    quietos = [m for m in con_movimiento if _bloques(m, c)[0] == 1]
    uno = len(quietos) == 1
    if quietos and len(quietos) == len(con_movimiento):
        texto += (
            f" AVISO: {'el salto se mueve' if uno else f'los {len(quietos)} saltos se mueven'} en menos de {2 * c} "
            f"fotogramas y se queda{'' if uno else 'n'} entero{'' if uno else 's'} en la pose: "
            "la animacion queda como el blocking, a saltos. Luce con mas fotogramas entre pose y pose (mas ritmo o mas "
            "fotogramas por pose) o con una cadencia menor."
        )
    elif quietos:
        texto += (
            f" {len(quietos)} de {len(con_movimiento)} saltos se mueve{'' if uno else 'n'} en menos de {2 * c} fotogramas "
            f"y se queda{'' if uno else 'n'} entero{'' if uno else 's'} en la pose."
        )
    return texto


class Cadencia(io.ComfyNode):
    """La animacion de CP_BlockingLento a doses, treses o cuatros: cada dibujo dura c fotogramas, como stop motion."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=NODO,
            display_name="Cámara lenta · Cadencia",
            category="character-pipeline/poses",
            description="Pasa la animacion a doses, treses o cuatros, como stop motion: cada dibujo dura c fotogramas, "
                        "las poses siguen en su fotograma y las esperas quietas no cambian. No genera nada: repite "
                        "fotogramas de la propia animacion, con la misma duracion.",
            inputs=[
                io.Image.Input("imagenes", tooltip="La animacion ya elegida, entera: la salida animacion de CP_BlockingLento."),
                io.String.Input("plan", force_input=True, tooltip="El de CP_LentaPlan, el mismo que recibio CP_BlockingLento."),
                io.Int.Input("cadencia", default=1, min=1, max=MAX_CADENCIA,
                             tooltip="1: a unos (tal cual). 2: a doses. 3: a treses. 4: a cuatros. Fotogramas que dura "
                                     "cada dibujo."),
            ],
            outputs=[
                io.Image.Output(display_name="imagenes"),
                io.String.Output(display_name="info"),
            ],
        )

    @classmethod
    def execute(cls, imagenes, plan, cadencia) -> io.NodeOutput:
        datos = _leer_plan(plan, NODO)
        total = datos["total"]
        if imagenes is None or not hasattr(imagenes, "dim") or imagenes.dim() != 4 or len(imagenes) != total:
            if imagenes is None:
                trae = "nada"
            elif not hasattr(imagenes, "dim") or imagenes.dim() != 4:
                trae = f"algo que no es un lote de imagenes (forma {tuple(getattr(imagenes, 'shape', ()))})"
            else:
                trae = f"{len(imagenes)}"
            raise ValueError(
                f"[{NODO}] El plan es de {total} fotogramas y llegaron {trae}: conecta la animacion de CP_BlockingLento "
                "entera y el mismo plan de CP_LentaPlan que recibio."
            )
        idx = indices_cadencia(datos["inicios"], total, datos["transicion_max"], cadencia)
        info = resumen_cadencia(datos["inicios"], total, datos["transicion_max"], cadencia)
        logging.info("[%s] %s", NODO, info)
        if idx == list(range(total)):
            return io.NodeOutput(imagenes, info)  # el mismo tensor: ni copia ni memoria de mas
        # una sola copia indexada; el tensor de entrada es de la cache del motor y no se toca
        salida = imagenes.index_select(0, torch.tensor(idx, dtype=torch.long, device=imagenes.device))
        return io.NodeOutput(salida, info)
