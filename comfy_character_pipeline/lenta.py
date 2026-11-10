# -*- coding: utf-8 -*-
"""Camara lenta: del blocking a saltos a animacion suave, sin interpolador.

LTX genera el plano a camara lenta con TODAS las poses del blocking fijadas, muy separadas en el tiempo (16 o 32
fotogramas lentos por salto), y despues se eligen los fotogramas lentos que tocan para volver a los tiempos del
blocking. Aqui estan las dos piezas que no trae ComfyUI: el plan (cuantos lentos lleva cada salto y en que ventanas
se parte el plano) y la eleccion de fotogramas.

Medido el 10-11-2026 con un blocking de 25 poses en 77 fotogramas:
- A camara lenta el modelo no emborrona el movimiento. Con menos lentos en los saltos pequenos tiene menos tiempo
  para inventar gestos y se genera un 30 % menos de fotogramas.
- Los fotogramas se eligen por RECORRIDO (flujo optico acumulado) y no por indice: LTX carga el movimiento hacia un
  lado del salto y por indice salen parones y tirones.
- El render en un pase a 1920x1088 esta medido hasta 177 fotogramas por ventana: de ahi el tope.

Todo lo que mide movimiento lo hace a 480 px de ancho, asi los umbrales valen para cualquier resolucion.

El ritmo multiplica lo que dura cada pose (inicios, total y transicion_max salen ya escalados en el plan): mas
fotogramas de salida entre pose y pose sin que CP_LentaElegir ni CP_BlockingLento cambien. Lo que cuesta GPU son
los lentos generados, no los de salida.

Compatibilidad: sin ritmo ni mas_lentos (o con ritmo 1 y mas_lentos False) el plan es identico, byte a byte, al de
la v0.14 para cualquier mapa; tests/test_lenta_compat_v014.py lo comprueba contra planes generados con la v0.14.
"""

import json
import logging

import numpy as np
import torch
import torch.nn.functional as F
from comfy_api.latest import io

VERSION_PLAN = 1
ANCHO_MEDIDA = 480  # px: a este ancho se miden los saltos y el recorrido
PARTE_MOVIL = 20  # se promedia 1 de cada 20 pixeles: el 5 % que mas se mueve
LENTOS_CORTO = 16
LENTOS_LARGO = 32
UMBRAL_SALTO = 9.0  # px a 480 de ancho: desde aqui el salto es grande y lleva 32 lentos
MAX_VENTANA = 177  # el mayor tamano medido a 1920x1088 en un pase
TRANSICION_MAX = 12  # fotogramas del blocking que como mucho ocupa un salto; el resto es espera
MAX_TOTAL = 1601  # cada ventana deja unos 9 GB de imagenes en la cache del trabajo
# Fotogramas de salida (blocking x ritmo) como mucho cuando hay ritmo (> 1): cada uno a 1920x1088 en float32 son
# 25 MB y CP_LentaElegir acumula el render y la guia de todo el plano (1000 fotogramas = unos 50 GB). Con ritmo 1 no
# se comprueba: la v0.14 no tenia este tope y un blocking que aceptaba tiene que seguir aceptandolo.
MAX_SALIDA = 1000
MAX_RITMO = 8
# Regla de mas lentos (solo con mas_lentos=True o ritmo > 1; sin ellos, la regla de la v0.14). Fotogramas de salida
# de un salto hasta los que bastan 16 lentos: con 16 lentos y hasta 8 de salida hay al menos dos lentos por fotograma
# y la eleccion por recorrido tiene margen para saltarse los tirones del modelo. Con 9 a 16 de salida, 16 lentos ya no
# dejan margen (con 16 se cogen todos, tirones incluidos): pasan a 32, que esta validado en GPU, y vuelve a haber el
# doble. Hasta 8 no cambia nada.
TRAMO_CORTO = 8
SIN_MOVIMIENTO = 1e-6  # px: por debajo, el recorrido de un salto no dice nada


# ---------------------------------------------------------------- medir movimiento

def _cv2(nodo):
    try:
        import cv2
    except ImportError as e:
        raise RuntimeError(f"[{nodo}] Falta OpenCV (cv2): instala opencv-python-headless.") from e
    return cv2


def _gris(imagen, cv2):
    """Un fotograma [H,W,C] (tensor o numpy 0-1, o numpy uint8) -> gris uint8 a 480 px de ancho."""
    if isinstance(imagen, torch.Tensor):
        imagen = imagen[..., :3].detach().float().clamp(0.0, 1.0).mul(255.0).round().to(torch.uint8).cpu().numpy()
    else:
        imagen = np.asarray(imagen)[..., :3]
        if imagen.dtype != np.uint8:
            imagen = np.clip(np.rint(imagen.astype(np.float32) * 255.0), 0, 255).astype(np.uint8)
    alto, ancho = imagen.shape[:2]
    alto = max(16, int(round(alto * ANCHO_MEDIDA / ancho)))
    # Primero se reduce y despues se pasa a gris: es como se midieron los datos de control.
    chica = cv2.resize(np.ascontiguousarray(imagen), (ANCHO_MEDIDA, alto), interpolation=cv2.INTER_AREA)
    if chica.ndim == 3 and chica.shape[-1] == 3:
        chica = cv2.cvtColor(chica, cv2.COLOR_RGB2GRAY)
    return chica.reshape(alto, ANCHO_MEDIDA)


def _movimiento(flujo):
    """Media del 5 % de pixeles que mas se mueven: el personaje ocupa poco y el fondo no cuenta."""
    mag = np.hypot(flujo[..., 0], flujo[..., 1]).ravel()
    k = max(1, len(mag) // PARTE_MOVIL)
    return float(np.partition(mag, -k)[-k:].mean())


def medir_saltos(poses, nodo="CP_LentaPlan"):
    """Movimiento (px a 480 de ancho) entre cada pose y la siguiente, con flujo optico DIS: len(poses) - 1 valores.
    Es una medida gruesa, sin generar nada, y solo sirve para decidir entre 16 y 32 fotogramas lentos."""
    cv2 = _cv2(nodo)
    grises = [_gris(p, cv2) for p in poses]
    saltos = []
    for a, b in zip(grises, grises[1:]):
        flujo = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM).calc(a, b, None)
        saltos.append(_movimiento(flujo))
    return saltos


def recorrido(guia, nodo="CP_LentaElegir"):
    """Movimiento (px a 480 de ancho) entre fotogramas lentos consecutivos, con Farneback: len(guia) - 1 valores.
    Se convierte fotograma a fotograma: el lote entero a 1920x1088 no cabe dos veces en memoria."""
    cv2 = _cv2(nodo)
    medido, anterior = [], None
    for i in range(len(guia)):
        gris = _gris(guia[i], cv2)
        if anterior is not None:
            flujo = cv2.calcOpticalFlowFarneback(anterior, gris, None, 0.5, 4, 21, 3, 5, 1.1, 0)
            medido.append(_movimiento(flujo))
        anterior = gris
    return np.array(medido, np.float64)


# ---------------------------------------------------------------- plan

def _entero(x):
    if isinstance(x, bool) or int(x) != x:
        raise ValueError(f"{x!r} no es un entero")
    return int(x)


def _comprobar_tiempos(inicios, total, nodo):
    if not inicios or inicios[0] != 0 or any(b <= a for a, b in zip(inicios, inicios[1:])):
        raise ValueError(f"[{nodo}] Los inicios de las poses tienen que empezar en 0 y ser crecientes: {inicios}.")
    if total < inicios[-1] + 1:
        raise ValueError(f"[{nodo}] El blocking dice tener {total} fotogramas y la ultima pose empieza en el {inicios[-1]}.")


def leer_mapa(mapa, n_poses, nodo="CP_LentaPlan"):
    """Mapa de CP_PosesDetectar -> (inicios, total): el fotograma del blocking donde empieza cada pose y cuantos
    fotogramas tiene el blocking."""
    try:
        bloques = [(_entero(a), _entero(b)) for a, b in json.loads(mapa)["bloques"]]
    except (ValueError, KeyError, TypeError) as e:
        raise ValueError(f"[{nodo}] El mapa de poses no se puede leer: {e}") from e
    if not bloques:
        raise ValueError(f"[{nodo}] El mapa de poses esta vacio.")
    if len(bloques) != n_poses:
        raise ValueError(f"[{nodo}] Llegaron {n_poses} poses y el mapa tiene {len(bloques)}.")
    seguidos = all(b >= a for a, b in bloques) and all(c == b + 1 for (_, b), (c, _) in zip(bloques, bloques[1:]))
    if bloques[0][0] != 0 or not seguidos:
        raise ValueError(f"[{nodo}] El mapa de poses no es una cadencia seguida desde el fotograma 0: {bloques[:4]}...")
    return [a for a, _ in bloques], bloques[-1][1] + 1


def lentos_por_salto(saltos_px, duraciones, umbral=UMBRAL_SALTO, transicion_max=TRANSICION_MAX, mas_margen=False):
    """Fotogramas lentos de cada salto: 32 si el salto es grande (o, con mas_margen, si hay que sacar de el mas de
    TRAMO_CORTO fotogramas), 16 si no, y nunca menos que los fotogramas que hay que sacar de el (subido al multiplo
    de 8, que es cada cuanto fija LTX una pose). Sin mas_margen es la regla de la v0.14."""
    lentos = []
    for px, d in zip(saltos_px, duraciones):
        m = min(d, transicion_max)
        base = LENTOS_LARGO if px >= umbral or (mas_margen and m > TRAMO_CORTO) else LENTOS_CORTO
        lentos.append(max(base, 8 * -(-m // 8)))
    return lentos


def repartir_ventanas(lentos, max_ventana=MAX_VENTANA):
    """Cortes [(p0, p1)] del plano en ventanas que empiezan y acaban en pose y comparten la pose frontera.
    Primero el numero minimo de ventanas con n <= max_ventana (n = lentos de sus saltos + 1); despues, con ese
    numero, el reparto mas parejo: la menor suma de n^2. Si empatan, la de cortes mas tempranos. Un salto solo
    que no quepa forma su propia ventana aunque supere el tope."""
    saltos = len(lentos)
    if saltos == 0:
        raise ValueError("[CP_LentaPlan] No hay saltos que repartir en ventanas.")
    acum = [0]
    for x in lentos:
        acum.append(acum[-1] + int(x))

    def cabe(a, b):
        return b - a == 1 or acum[b] - acum[a] + 1 <= max_ventana

    # El minimo sale alargando cada ventana mientras quepa: un trozo de una ventana que cabe tambien cabe.
    minimo, a = 0, 0
    while a < saltos:
        b = a + 1
        while b < saltos and cabe(a, b + 1):
            b += 1
        minimo, a = minimo + 1, b

    # coste[j][a]: menor suma de n^2 al partir los saltos a..fin en j ventanas. Enteros: los empates son exactos.
    nada = float("inf")
    coste = [[nada] * (saltos + 1) for _ in range(minimo + 1)]
    coste[0][saltos] = 0
    for j in range(1, minimo + 1):
        for a in range(saltos - 1, -1, -1):
            for b in range(a + 1, saltos + 1):
                if not cabe(a, b):
                    break
                if coste[j - 1][b] != nada:
                    coste[j][a] = min(coste[j][a], (acum[b] - acum[a] + 1) ** 2 + coste[j - 1][b])
    cortes, a = [], 0
    for j in range(minimo, 0, -1):
        for b in range(a + 1, saltos + 1):
            if not cabe(a, b):
                break
            if coste[j - 1][b] != nada and (acum[b] - acum[a] + 1) ** 2 + coste[j - 1][b] == coste[j][a]:
                cortes.append((a, b))
                a = b
                break
    return cortes


def armar_plan(inicios, total, saltos_px, lentos, cortes, transicion_max=TRANSICION_MAX, ritmo=1, mas_lentos=False):
    """El plan con los lentos y los cortes ya decididos. pos[j] es el indice lento de la pose p0+j en su ventana.
    inicios, total y transicion_max van ya multiplicados por el ritmo. El ritmo solo se apunta si no es 1 y
    mas_lentos solo si es True (asi un plan sin ellos sale igual, byte a byte, que en la v0.14)."""
    ventanas = []
    for p0, p1 in cortes:
        pos = [0]
        for x in lentos[p0:p1]:
            pos.append(pos[-1] + int(x))
        ventanas.append({"p0": int(p0), "p1": int(p1), "pos": pos, "n": pos[-1] + 1})
    plan = {
        "version": VERSION_PLAN,
        "inicios": [int(x) for x in inicios],
        "total": int(total),
        "saltos_px": [float(x) for x in saltos_px],
        "lentos": [int(x) for x in lentos],
        "transicion_max": int(transicion_max),
        "ventanas": ventanas,
    }
    if int(ritmo) != 1:
        plan["ritmo"] = int(ritmo)
    if mas_lentos:
        plan["mas_lentos"] = True
    return plan


def _comprobar_ritmo(ritmo, nodo):
    try:
        entero = _entero(ritmo)
    except (ValueError, TypeError):
        entero = None
    if entero is None or not 1 <= entero <= MAX_RITMO:
        raise ValueError(f"[{nodo}] El ritmo tiene que ser un entero de 1 a {MAX_RITMO} y llego {ritmo!r}.")
    return entero


def planificar(inicios, total, saltos_px, umbral=UMBRAL_SALTO, max_ventana=MAX_VENTANA,
               transicion_max=TRANSICION_MAX, max_total=MAX_TOTAL, nodo="CP_LentaPlan", ritmo=1,
               max_salida=MAX_SALIDA, mas_lentos=False):
    """Plan de camara lenta: lentos por salto y ventanas. Devuelve el dict que viaja como JSON entre los nodos.
    ritmo multiplica lo que dura cada pose: el plan sale con inicios, total y transicion_max ya multiplicados.
    La regla de mas lentos (32 si de un salto salen mas de TRAMO_CORTO fotogramas) se aplica con mas_lentos=True o
    con ritmo > 1, y el tope de max_salida fotogramas de salida solo con ritmo > 1. Con ritmo 1 y mas_lentos False
    el plan (y cada error) es el de la v0.14."""
    inicios = [int(x) for x in inicios]
    if len(inicios) < 2:
        raise ValueError(
            f"[{nodo}] El blocking tiene una sola pose: no hay nada que animar. "
            "Revisa que exportaste el plano correcto: tiene que haber al menos dos poses distintas."
        )
    ritmo = _comprobar_ritmo(ritmo, nodo)
    _comprobar_tiempos(inicios, total, nodo)
    if ritmo > 1 and int(total) * ritmo > max_salida:
        raise ValueError(
            f"[{nodo}] Con ritmo {ritmo} salen {int(total) * ritmo} fotogramas de salida ({int(total)} del blocking x "
            f"{ritmo}) y con ritmo el tope es {max_salida}: baja el ritmo (con 1 no hay este tope) o parte el plano en dos. "
            "Es por memoria: cada fotograma a 1920x1088 ocupa 25 MB y se guardan el render y la guia de todo el plano."
        )
    mas_lentos = bool(mas_lentos)
    regla = mas_lentos or ritmo > 1
    inicios, total, transicion_max = [x * ritmo for x in inicios], int(total) * ritmo, int(transicion_max) * ritmo
    if len(saltos_px) != len(inicios) - 1:
        raise ValueError(f"[{nodo}] Con {len(inicios)} poses tocan {len(inicios) - 1} saltos medidos y llegaron {len(saltos_px)}.")
    # Se redondea ANTES de clasificar: quien lea el plan saca los mismos lentos de las medidas guardadas.
    saltos = [round(float(x), 3) for x in saltos_px]
    if not all(np.isfinite(saltos)):
        raise ValueError(f"[{nodo}] La medida de algun salto no es un numero: {saltos}.")
    duraciones = [b - a for a, b in zip(inicios, inicios[1:])]
    lentos = lentos_por_salto(saltos, duraciones, umbral, transicion_max, mas_margen=regla)
    cortes = repartir_ventanas(lentos, max_ventana)
    lentos_totales = sum(lentos) + len(cortes)
    if lentos_totales > max_total:
        raise ValueError(
            f"[{nodo}] Salen {lentos_totales} fotogramas lentos en {len(cortes)} ventanas y el tope es {max_total}. "
            "El blocking es demasiado largo para una sola pasada: partelo en planos mas cortos."
            + (f" Con ritmo {ritmo} cada pose dura {ritmo} veces mas: bajar el ritmo tambien lo arregla." if ritmo > 1 else "")
        )
    return leer_plan(armar_plan(inicios, total, saltos, lentos, cortes, transicion_max, ritmo, mas_lentos), nodo)


def leer_plan(texto, nodo="CP_LentaElegir"):
    """El plan (texto JSON o dict) comprobado de arriba abajo: quien lo usa ya no tiene que desconfiar."""
    plan = texto
    if not isinstance(plan, dict):
        try:
            plan = json.loads(texto)
        except (ValueError, TypeError) as e:
            raise ValueError(f"[{nodo}] El plan no se puede leer: {e}") from e
    if not isinstance(plan, dict):
        raise ValueError(f"[{nodo}] El plan no se puede leer: no es un objeto JSON.")
    if plan.get("version") != VERSION_PLAN:
        raise ValueError(
            f"[{nodo}] El plan es de la version {plan.get('version')!r} y este nodo entiende la {VERSION_PLAN}: "
            "vuelve a generarlo con CP_LentaPlan."
        )
    crudo = plan
    try:
        plan = {
            "version": VERSION_PLAN,
            "inicios": [_entero(x) for x in plan["inicios"]],
            "total": _entero(plan["total"]),
            "saltos_px": [float(x) for x in plan["saltos_px"]],
            "lentos": [_entero(x) for x in plan["lentos"]],
            "transicion_max": _entero(plan["transicion_max"]),
            "ventanas": [
                {"p0": _entero(v["p0"]), "p1": _entero(v["p1"]), "pos": [_entero(x) for x in v["pos"]], "n": _entero(v["n"])}
                for v in plan["ventanas"]
            ],
        }
        # El ritmo es informativo (todo lo demas ya viene multiplicado) y los planes sin el son de ritmo 1.
        if "ritmo" in crudo:
            plan["ritmo"] = _entero(crudo["ritmo"])
        # mas_lentos tambien es informativo (los lentos ya vienen decididos): solo se guarda si es True.
        if "mas_lentos" in crudo:
            if not isinstance(crudo["mas_lentos"], bool):
                raise ValueError(f"mas_lentos tiene que ser true o false y trae {crudo['mas_lentos']!r}")
            if crudo["mas_lentos"]:
                plan["mas_lentos"] = True
    except (KeyError, TypeError, ValueError) as e:
        raise ValueError(f"[{nodo}] Al plan le falta un dato o lo trae mal: {e!r}.") from e

    inicios, lentos, ventanas, transicion = plan["inicios"], plan["lentos"], plan["ventanas"], plan["transicion_max"]
    if len(inicios) < 2:
        raise ValueError(f"[{nodo}] El plan tiene una sola pose: no hay nada que animar.")
    _comprobar_tiempos(inicios, plan["total"], nodo)
    saltos = len(inicios) - 1
    if len(lentos) != saltos or len(plan["saltos_px"]) != saltos:
        raise ValueError(
            f"[{nodo}] El plan no cuadra: con {len(inicios)} poses tocan {saltos} saltos y trae "
            f"{len(lentos)} lentos y {len(plan['saltos_px'])} medidas."
        )
    if transicion < 1:
        raise ValueError(f"[{nodo}] transicion_max tiene que ser 1 o mas y el plan trae {transicion}.")
    if plan.get("ritmo", 1) < 1:
        raise ValueError(f"[{nodo}] El ritmo tiene que ser 1 o mas y el plan trae {plan['ritmo']}.")
    for k, x in enumerate(lentos):
        m = min(inicios[k + 1] - inicios[k], transicion)
        if x < 8 or x % 8:
            raise ValueError(f"[{nodo}] El salto {k} lleva {x} fotogramas lentos y tiene que ser un multiplo de 8: LTX fija las poses cada 8.")
        if x < m:
            raise ValueError(f"[{nodo}] El salto {k} lleva {x} fotogramas lentos y hay que sacar {m} de el.")
    seguidas = (
        bool(ventanas) and ventanas[0]["p0"] == 0 and ventanas[-1]["p1"] == saltos
        and all(v["p1"] > v["p0"] for v in ventanas)
        and all(a["p1"] == b["p0"] for a, b in zip(ventanas, ventanas[1:]))
    )
    if not seguidas:
        cortes = [(v["p0"], v["p1"]) for v in ventanas]
        raise ValueError(f"[{nodo}] Las ventanas del plan no cubren las {len(inicios)} poses una detras de otra: {cortes}.")
    for w, v in enumerate(ventanas):
        pos, n = v["pos"], v["n"]
        if len(pos) != v["p1"] - v["p0"] + 1:
            raise ValueError(f"[{nodo}] La ventana {w} va de la pose {v['p0']} a la {v['p1']} y trae {len(pos)} posiciones.")
        if n != pos[-1] + 1:
            raise ValueError(f"[{nodo}] La ventana {w} dice tener {n} fotogramas y su ultima pose esta en el {pos[-1]}: tocan {pos[-1] + 1}.")
        if n % 8 != 1:
            raise ValueError(f"[{nodo}] La ventana {w} tiene {n} fotogramas y LTX trabaja con 8k+1.")
        if any(x % 8 for x in pos):
            raise ValueError(f"[{nodo}] Las poses de la ventana {w} no caen en un multiplo de 8: {pos}.")
        esperado = [0]
        for x in lentos[v["p0"]:v["p1"]]:
            esperado.append(esperado[-1] + x)
        if pos != esperado:
            raise ValueError(f"[{nodo}] Las posiciones de la ventana {w} no salen de los lentos de sus saltos: {pos} y tocan {esperado}.")
    return plan


def resumir_plan(plan, max_ventana=MAX_VENTANA):
    """Una linea legible: poses, ventanas, lentos y avisos."""
    inicios, lentos, transicion = plan["inicios"], plan["lentos"], plan["transicion_max"]
    tamanos = [v["n"] for v in plan["ventanas"]]
    cuenta = {}
    for x in lentos:
        cuenta[x] = cuenta.get(x, 0) + 1
    reparto = ", ".join(f"{c} a {x}" for x, c in sorted(cuenta.items(), reverse=True))
    texto = (
        f"{len(inicios)} poses en {plan['total']} fotogramas: {len(tamanos)} ventana{'s' if len(tamanos) != 1 else ''} de "
        f"{', '.join(str(n) for n in tamanos)} = {sum(tamanos)} fotogramas lentos. Saltos: {reparto}."
    )
    for w, n in enumerate(tamanos):
        if n > max_ventana:
            texto += f" AVISO: la ventana {w} mide {n} fotogramas y el tope es {max_ventana} (un salto solo no se puede partir): puede no caber en memoria."
    esperas = sum(1 for a, b in zip(inicios, inicios[1:]) if b - a > transicion)
    if esperas:
        texto += f" Esperas de mas de {transicion} fotogramas: {esperas} (la pose se queda quieta y el salto ocupa los {transicion} ultimos)."
    return texto


def resumir_salida(plan):
    """Lo que sale y lo que se genera, y el aviso de saltos con mas lentos de los validados en GPU (no falla)."""
    ritmo = plan.get("ritmo", 1)
    generados = sum(v["n"] for v in plan["ventanas"])
    texto = f"Salida: {plan['total']} fotogramas"
    if ritmo != 1:
        texto += f" (ritmo {ritmo}: cada pose dura {ritmo} veces lo del blocking)"
    texto += f" a partir de {generados} fotogramas lentos generados."
    if ritmo != 1 or plan.get("mas_lentos"):
        motivo = "mas_lentos" if plan.get("mas_lentos") else f"ritmo {ritmo}"
        texto += (
            f" Regla de mas lentos activa ({motivo}): 32 lentos en los saltos de los que salen mas de {TRAMO_CORTO} fotogramas."
        )
    else:
        texto += " Regla de lentos de siempre (la de mas lentos solo con mas_lentos o ritmo > 1)."
    largos = [x for x in plan["lentos"] if x > LENTOS_LARGO]
    if largos:
        texto += (
            f" AVISO: {len(largos)} salto{'s' if len(largos) != 1 else ''} con mas de {LENTOS_LARGO} lentos (hasta {max(largos)}): "
            f"en GPU solo estan validados {LENTOS_CORTO} y {LENTOS_LARGO}. Baja el ritmo o transicion_max si se ve raro."
        )
    return texto


# ---------------------------------------------------------------- elegir

def elegir_indices(recorrido, lentos_ventana, d_ventana, transicion_max=TRANSICION_MAX):
    """Que fotograma lento de la ventana le toca a cada fotograma del blocking, sin la pose final de la ventana.

    recorrido: movimiento entre lentos consecutivos de la ventana (n - 1 valores). Por cada salto, con sus lentos
    y sus d fotogramas de salida: si d pasa de transicion_max los primeros son la pose quieta (una espera larga
    del blocking sigue siendo espera) y los m = min(d, transicion_max) ultimos recorren el salto a velocidad
    constante: el fotograma j toma el lento cuyo recorrido acumulado queda mas cerca de j/m del total."""
    medido = np.asarray(recorrido, np.float64).ravel()
    if len(lentos_ventana) != len(d_ventana):
        raise ValueError(f"[CP_LentaElegir] La ventana tiene {len(lentos_ventana)} saltos con lentos y {len(d_ventana)} con duracion.")
    if len(medido) != sum(lentos_ventana):
        raise ValueError(f"[CP_LentaElegir] El recorrido tiene {len(medido)} valores y la ventana {sum(lentos_ventana)} pasos entre fotogramas lentos.")
    if not np.isfinite(medido).all() or (medido < 0).any():
        raise ValueError("[CP_LentaElegir] El recorrido medido tiene valores negativos o que no son numeros.")
    elegidos, base = [], 0
    for lentos, d in zip(lentos_ventana, d_ventana):
        m = min(d, transicion_max)
        if m < 1 or lentos < m:
            raise ValueError(f"[CP_LentaElegir] De un salto de {lentos} fotogramas lentos no se pueden sacar {m} (salto de {d} fotogramas del blocking).")
        s = np.concatenate([[0.0], np.cumsum(medido[base:base + lentos])])
        total = s[-1]
        idx = [0] * (d - m + 1)  # la espera y la pose, que siempre es el lento 0 del salto
        for j in range(1, m):
            if total <= SIN_MOVIMIENTO:
                c = (2 * j * lentos + m) // (2 * m)  # poses casi iguales: reparto uniforme por indice
            else:
                objetivo = j / m * total
                c = int(np.searchsorted(s, objetivo))
                c = c if c >= lentos or abs(s[c] - objetivo) <= abs(s[c - 1] - objetivo) else c - 1
            # Siempre hacia delante y dejando sitio a los que faltan.
            idx.append(min(max(c, idx[-1] + 1), lentos - (m - j)))
        elegidos += [base + c for c in idx]
        base += lentos
    return elegidos


def indices_ventana(plan, ventana, medido):
    """Los lentos de la ventana que salen al video, en orden. La pose final (el ultimo lento) solo se anade en la
    ultima ventana: en las demas es el lento 0 de la siguiente. Si el blocking sigue despues de la ultima pose,
    se repite hasta completar sus fotogramas."""
    inicios = plan["inicios"]
    v = plan["ventanas"][ventana]
    p0, p1 = v["p0"], v["p1"]
    duraciones = [inicios[k + 1] - inicios[k] for k in range(p0, p1)]
    indices = elegir_indices(medido, plan["lentos"][p0:p1], duraciones, plan["transicion_max"])
    if ventana == len(plan["ventanas"]) - 1:
        indices += [v["n"] - 1] * (plan["total"] - inicios[-1])
    return indices


def montar_ventana(guia, render, plan, ventana, anterior_render=None, anterior_guia=None, medido=None, nodo="CP_LentaElegir"):
    """(render, guia) con lo acumulado de las ventanas anteriores mas lo elegido de esta. El recorrido se mide en
    la guia (medido lo sustituye en las pruebas). Lo elegido se COPIA: una vista retendria en la cache del trabajo
    los 4 GB del lote lento."""
    ventanas = plan["ventanas"]
    if not 0 <= ventana < len(ventanas):
        raise ValueError(f"[{nodo}] La ventana {ventana} no existe: el plan tiene {len(ventanas)} (de la 0 a la {len(ventanas) - 1}).")
    v = ventanas[ventana]
    for nombre, lote in (("la guia", guia), ("el render", render)):
        if lote is None or lote.dim() != 4 or len(lote) != v["n"]:
            trae = "nada" if lote is None else f"{len(lote)}"
            raise ValueError(
                f"[{nodo}] La ventana {ventana} tiene {v['n']} fotogramas lentos y {nombre} trae {trae}: "
                "la guia y el render tienen que ser los de esa ventana, enteros."
            )
    # Lo acumulado tiene que ser justo lo que va antes de esta ventana: asi el video acaba con los fotogramas del blocking.
    previos = plan["inicios"][v["p0"]]
    for nombre, lote, canales in (("anterior_render", anterior_render, render.shape[3]), ("anterior_guia", anterior_guia, guia.shape[3])):
        trae = 0 if lote is None else len(lote)
        if trae != previos:
            arreglo = "conecta las dos salidas del CP_LentaElegir de la ventana anterior" if previos else "es la primera, no lleva nada conectado ahi"
            raise ValueError(f"[{nodo}] Antes de la ventana {ventana} van {previos} fotogramas ya elegidos y {nombre} trae {trae}: {arreglo}.")
        if previos and tuple(lote.shape[1:]) != (render.shape[1], render.shape[2], canales):
            raise ValueError(
                f"[{nodo}] {nombre} mide {lote.shape[2]}x{lote.shape[1]} y el render de esta ventana {render.shape[2]}x{render.shape[1]}: "
                "todas las ventanas tienen que salir al mismo tamano."
            )
    if medido is None:
        medido = recorrido(guia, nodo)
    indices = torch.tensor(indices_ventana(plan, ventana, medido), dtype=torch.long)
    elegido = render.index_select(0, indices.to(render.device))
    elegida = guia.index_select(0, indices.to(guia.device))
    if elegida.shape[1:3] != elegido.shape[1:3]:
        elegida = F.interpolate(
            elegida.movedim(-1, 1).float(), size=tuple(elegido.shape[1:3]), mode="bilinear", align_corners=False, antialias=True,
        ).movedim(1, -1).clamp(0.0, 1.0).contiguous()
    if previos:
        elegido = torch.cat([anterior_render.to(elegido.device), elegido], dim=0)
        elegida = torch.cat([anterior_guia.to(elegida.device), elegida], dim=0)
    return elegido, elegida


# ---------------------------------------------------------------- nodos

class LentaPlan(io.ComfyNode):
    """Decide cuantos fotogramas lentos lleva cada salto del blocking y en que ventanas se parte el plano."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CP_LentaPlan",
            display_name="Cámara lenta · Plan",
            category="character-pipeline/poses",
            description="Mide cuanto se mueve cada salto del blocking y reparte el plano en ventanas a camara lenta.",
            inputs=[
                io.Image.Input("poses", tooltip="Una imagen por pose, de CP_PosesDetectar."),
                io.String.Input("mapa", force_input=True),
                io.Float.Input(
                    "umbral", default=UMBRAL_SALTO, min=0.0, max=100.0, step=0.5,
                    tooltip="Movimiento (px a 480 de ancho) desde el que un salto lleva 32 fotogramas lentos en vez de 16.",
                ),
                io.Int.Input(
                    "max_ventana", default=MAX_VENTANA, min=33, max=513,
                    tooltip="Fotogramas lentos por ventana como mucho. 177 es lo mayor medido a 1920x1088 en un pase.",
                ),
                io.Int.Input(
                    "transicion_max", default=TRANSICION_MAX, min=2, max=64,
                    tooltip="Fotogramas del blocking que como mucho ocupa un salto: lo que sobra de una espera larga sigue quieto.",
                ),
                io.Int.Input(
                    "max_total", default=MAX_TOTAL, min=65, max=8001,
                    tooltip="Tope de seguridad de fotogramas lentos en todo el plano: por encima, parte el blocking.",
                ),
                io.Int.Input(
                    "ritmo", default=1, min=1, max=MAX_RITMO, optional=True,
                    tooltip="Multiplica lo que dura cada pose (y transicion_max): mas fotogramas de salida entre pose y pose. "
                            f"Con 1, los tiempos del blocking. Con mas de 1: tope de {MAX_SALIDA} fotogramas de salida y "
                            "se activa la regla de mas lentos.",
                ),
                io.Boolean.Input(
                    "mas_lentos", default=False, optional=True,
                    tooltip=f"32 fotogramas lentos (no 16) en los saltos de los que salen mas de {TRAMO_CORTO} fotogramas: "
                            "dos lentos por fotograma de salida, a cambio de mas GPU. Con ritmo > 1 se aplica siempre. "
                            "Apagado y con ritmo 1, el plan de siempre.",
                ),
            ],
            outputs=[
                io.String.Output(display_name="plan"),
                io.Int.Output(display_name="n_ventanas"),
                io.Int.Output(display_name="fotogramas_lentos", tooltip="Los que se generan en total: la suma de las ventanas."),
                io.String.Output(display_name="info"),
            ],
        )

    @classmethod
    def execute(cls, poses, mapa, umbral=UMBRAL_SALTO, max_ventana=MAX_VENTANA, transicion_max=TRANSICION_MAX,
                max_total=MAX_TOTAL, ritmo=1, mas_lentos=False) -> io.NodeOutput:
        if poses is None or len(poses) == 0:
            raise ValueError("[CP_LentaPlan] No llegaron poses.")
        inicios, total = leer_mapa(mapa, len(poses), "CP_LentaPlan")
        saltos = medir_saltos(poses, "CP_LentaPlan") if len(inicios) > 1 else []
        if ritmo is None:
            ritmo = 1
        plan = planificar(inicios, total, saltos, umbral, max_ventana, transicion_max, max_total, ritmo=ritmo,
                          mas_lentos=bool(mas_lentos))
        info = resumir_plan(plan, max_ventana) + " " + resumir_salida(plan)
        logging.info("[CP_LentaPlan] %s", info)
        return io.NodeOutput(json.dumps(plan), len(plan["ventanas"]), sum(v["n"] for v in plan["ventanas"]), info)


class LentaElegir(io.ComfyNode):
    """Vuelve a los tiempos del blocking: de los fotogramas lentos de una ventana, los que tocan."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CP_LentaElegir",
            display_name="Cámara lenta · Elegir fotogramas",
            category="character-pipeline/poses",
            description="Elige los fotogramas lentos de una ventana repartiendo el movimiento por recorrido y los anade a lo anterior.",
            inputs=[
                io.Image.Input("guia", tooltip="El playblast lento de la ventana, entero: en el se mide el recorrido."),
                io.Image.Input("render", tooltip="El render lento de la misma ventana, entero."),
                io.String.Input("plan", force_input=True),
                io.Int.Input("ventana", default=0, min=0, max=4096, tooltip="Que ventana del plan es, desde 0."),
                io.Image.Input("anterior_render", optional=True, tooltip="La salida render del CP_LentaElegir de la ventana anterior."),
                io.Image.Input("anterior_guia", optional=True, tooltip="La salida guia del CP_LentaElegir de la ventana anterior."),
            ],
            outputs=[
                io.Image.Output(display_name="render"),
                io.Image.Output(display_name="guia", tooltip="Al tamano del render."),
            ],
        )

    @classmethod
    def execute(cls, guia, render, plan, ventana, anterior_render=None, anterior_guia=None) -> io.NodeOutput:
        datos = leer_plan(plan, "CP_LentaElegir")
        elegido, elegida = montar_ventana(guia, render, datos, ventana, anterior_render, anterior_guia)
        logging.info(
            "[CP_LentaElegir] Ventana %d de %d: %d fotogramas lentos, van %d de %d del blocking.",
            ventana + 1, len(datos["ventanas"]), len(render), len(elegido), datos["total"],
        )
        return io.NodeOutput(elegido, elegida)
