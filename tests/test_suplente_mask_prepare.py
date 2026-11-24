# -*- coding: utf-8 -*-
"""Prueba el suplente de InpaintMaskPrepareMEC (suplente_mask_prepare.py) sin ComfyUI:
- el esquema que declara frente al del nodo original (Code2Collapse, commit 9891293), guardado en
  tests/datos/inpaint_mask_prepare_mec_c2c_9891293.json;
- lo que hace con los valores de la tarjeta «enhance lora propio v9» (quitar islas de menos de 20 000 px y nada mas);
- cada opcion frente a una copia del algoritmo original (OpenCV, si esta instalado) y frente a una implementacion
  ingenua escrita aqui, pixel a pixel, que no usa ni scipy ni OpenCV.
La comprobacion dentro de un ComfyUI real es tools/comprobar_suplente_mask_prepare.py.
Se ejecuta con: python tests/test_suplente_mask_prepare.py   (necesita torch, numpy y scipy u opencv)"""
import json
import math
import os
import sys
import types

import numpy as np
import torch

# ---- stub minimo de comfy_api.latest: cada puerto guarda sus argumentos y su tipo ----
mod = types.ModuleType("comfy_api")
latest = types.ModuleType("comfy_api.latest")


class _Port:
    def __init__(self, *a, **k):
        self.a, self.k = a, k


def _mk(name):
    return type(name, (), {"Input": type(name + "Input", (_Port,), {}), "Output": type(name + "Output", (_Port,), {})})


class _NodeOutput:
    def __init__(self, *valores):
        self.valores = valores


class _IO:
    Schema = staticmethod(lambda **k: k)
    NodeOutput = _NodeOutput
    ComfyNode = type("ComfyNode", (), {})


for _n in ("Mask", "Image", "Int", "Float", "Boolean", "Combo", "String"):
    setattr(_IO, _n, _mk(_n))

latest.io = _IO
mod.latest = latest
sys.modules["comfy_api"] = mod
sys.modules["comfy_api.latest"] = latest

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(AQUI, "..", "comfy_character_pipeline"))
import suplente_mask_prepare as S  # noqa: E402

REF = json.load(open(os.path.join(AQUI, "datos", "inpaint_mask_prepare_mec_c2c_9891293.json"), encoding="utf-8"))
TIPOS = {"MaskInput": "MASK", "ImageInput": "IMAGE", "IntInput": "INT", "FloatInput": "FLOAT", "BooleanInput": "BOOLEAN"}
# Los valores del nodo 34 (y del 54) de la tarjeta publicada, en el orden de sus widgets.
TARJETA = dict(fill_holes=False, remove_small_regions=True, min_region_area=20000, grow_pixels=0,
               inpaint_edge_mode="hard_binary", stitch_edge_mode="gaussian", stitch_feather_radius=16,
               temporal_smooth=False, temporal_sigma=1.5)
comprobaciones = 0


def ok(condicion, que=""):
    global comprobaciones
    assert condicion, que
    comprobaciones += 1


def preparar(mask, **cambios):
    return S.preparar(mask, **{**TARJETA, **cambios})


# ============================================================ 1. esquema igual al del original
esquema = S.SuplenteInpaintMaskPrepareMEC.define_schema()
ok(esquema["node_id"] == "InpaintMaskPrepareMEC" == REF["name"])
ok(esquema["display_name"] == REF["display_name"])
ok(esquema["category"] == REF["category"])
ok(esquema["description"] == REF["description"])
entradas = {p.a[0]: p for p in esquema["inputs"]}
orden = [p.a[0] for p in esquema["inputs"]]
ok([n for n in orden if not entradas[n].k.get("optional")] == REF["input_order"]["required"], orden)
ok([n for n in orden if entradas[n].k.get("optional")] == REF["input_order"]["optional"], orden)
for nombre, p in entradas.items():
    tipo_ref, opts = REF["input"]["optional" if p.k.get("optional") else "required"][nombre]
    if type(p).__name__ == "ComboInput":
        # En /object_info un desplegable V1 es [[opciones], {...}].
        ok(isinstance(tipo_ref, list) and p.k["options"] == tipo_ref, (nombre, p.k["options"], tipo_ref))
    else:
        ok(TIPOS[type(p).__name__] == tipo_ref, (nombre, type(p).__name__, tipo_ref))
    for clave, valor in opts.items():
        ok(p.k.get(clave) == valor, (nombre, clave, p.k.get(clave), valor))
    ok(set(p.k) - {"optional", "options"} == set(opts), (nombre, sorted(p.k), sorted(opts)))
salidas = esquema["outputs"]
ok([type(s).__name__ for s in salidas] == ["MaskOutput", "MaskOutput", "ImageOutput", "StringOutput"])
ok(REF["output"] == ["MASK", "MASK", "IMAGE", "STRING"])
ok([s.k.get("display_name") for s in salidas] == REF["output_name"])
# La firma de execute lleva las entradas en el mismo orden (ComfyUI las pasa por nombre).
import inspect  # noqa: E402
ok(list(inspect.signature(S.SuplenteInpaintMaskPrepareMEC.execute).parameters) == orden)


# ============================================================ referencias independientes
def islas_ingenua(binaria, min_area):
    """Quita piezas (vecindad de 8) de menos de min_area con una pila, pixel a pixel. Sin scipy ni OpenCV."""
    h, w = binaria.shape
    vista = np.zeros((h, w), dtype=bool)
    salida = np.zeros((h, w), dtype=bool)
    for y0 in range(h):
        for x0 in range(w):
            if not binaria[y0, x0] or vista[y0, x0]:
                continue
            pila, pieza = [(y0, x0)], []
            vista[y0, x0] = True
            while pila:
                y, x = pila.pop()
                pieza.append((y, x))
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        yy, xx = y + dy, x + dx
                        if 0 <= yy < h and 0 <= xx < w and binaria[yy, xx] and not vista[yy, xx]:
                            vista[yy, xx] = True
                            pila.append((yy, xx))
            if len(pieza) >= min_area:
                for y, x in pieza:
                    salida[y, x] = True
    return salida


def huecos_ingenua(binaria):
    """Rellena lo que no se alcanza desde el borde andando por el fondo en cruz (vecindad de 4)."""
    h, w = binaria.shape
    fuera = np.zeros((h, w), dtype=bool)
    pila = [(y, x) for y in range(h) for x in (0, w - 1)] + [(y, x) for x in range(w) for y in (0, h - 1)]
    pila = [(y, x) for y, x in pila if not binaria[y, x]]
    for y, x in pila:
        fuera[y, x] = True
    while pila:
        y, x = pila.pop()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            yy, xx = y + dy, x + dx
            if 0 <= yy < h and 0 <= xx < w and not binaria[yy, xx] and not fuera[yy, xx]:
                fuera[yy, xx] = True
                pila.append((yy, xx))
    return ~fuera


try:
    import cv2
except ImportError:
    cv2 = None


def islas_original(binaria, min_area):
    """Copia del algoritmo del pack original (_remove_small_regions_torch con OpenCV)."""
    n, etiquetas, stats, _ = cv2.connectedComponentsWithStats(binaria.astype(np.uint8), 8)
    filtrada = np.zeros_like(binaria, dtype=np.uint8)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            filtrada[etiquetas == i] = 1
    return filtrada.astype(bool)


def huecos_original(binaria):
    """Copia del algoritmo del pack original (_fill_holes_torch con OpenCV)."""
    m = binaria.astype(np.uint8) * 255
    contornos, jerarquia = cv2.findContours(m, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if jerarquia is not None:
        for i, hj in enumerate(jerarquia[0]):
            if hj[3] >= 0:
                cv2.drawContours(m, contornos, i, 255, -1)
    return m > 127


def gauss_ingenua(binaria, sigma):
    """Desenfoque separable con borde repetido, en numpy y float64 (misma definicion que el original)."""
    radio = max(1, int(math.ceil(3.0 * sigma)))
    x = np.arange(-radio, radio + 1, dtype=np.float64)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    k /= k.sum()
    a = np.pad(binaria.astype(np.float64), ((0, 0), (radio, radio)), mode="edge")
    a = np.stack([np.convolve(f, k, mode="valid") for f in a])
    a = np.pad(a, ((radio, radio), (0, 0)), mode="edge")
    return np.stack([np.convolve(c, k, mode="valid") for c in a.T]).T


# ============================================================ 2. los valores de la tarjeta
H = W = 1024
m = torch.zeros(1, H, W)
m[0, 100:600, 100:500] = 1.0          # el personaje: 200 000 px
m[0, 300:350, 250:300] = 0.0          # con un hueco (entre antenas y cabeza): no se rellena
m[0, 700:800, 100:300] = 1.0          # 20 000 px justos: se queda
m[0, 700:800, 400:600] = 1.0
m[0, 799, 599] = 0.0                  # 19 999 px: fuera
m[0, 900:910, 900:910] = 0.7          # isla pequena y semitransparente: fuera
m[0, 50:60, 50:60] = 0.3              # por debajo del umbral: nunca fue mascara
copia = m.clone()
inpaint, mezcla, vista, info = preparar(m)
ok(torch.equal(m, copia), "no toca la entrada")
ok(inpaint.shape == (1, H, W) and inpaint.dtype == torch.float32)
ok(set(inpaint.unique().tolist()) == {0.0, 1.0}, "hard_binary sale en 0 y 1")
esperada = torch.zeros(1, H, W)
esperada[0, 100:600, 100:500] = 1.0
esperada[0, 300:350, 250:300] = 0.0
esperada[0, 700:800, 100:300] = 1.0
ok(torch.equal(inpaint, esperada), "queda el personaje con su hueco y la pieza de 20 000 px justos")
ok(int(inpaint.sum()) == 200000 - 2500 + 20000)
ok(mezcla.shape == (1, H, W) and float(mezcla.min()) >= 0.0 and float(mezcla.max()) <= 1.0 + 1e-6)
ok(vista.shape == (1, H, W, 3) and torch.equal(vista[..., 0], inpaint) and torch.equal(vista[..., 1], mezcla)
   and float(vista[..., 2].abs().max()) == 0.0)
ok("pixels removed: 20099" in info and "suplente" in info, info)
ok(np.allclose(mezcla[0].numpy(), gauss_ingenua(esperada[0].numpy() > 0.5, 16 * 0.4), atol=1e-5),
   "la mascara de mezcla es la binaria desenfocada con sigma 6,4")
# Una mascara 2D vale igual y sale con lote.
ok(torch.equal(preparar(m[0])[0], inpaint))
# Sin nada que quitar: la mascara binaria sale igual.
ok(torch.equal(preparar(esperada)[0], esperada))
# Mascara vacia y mascara llena.
ok(float(preparar(torch.zeros(1, 64, 64))[0].sum()) == 0.0)
ok(torch.equal(preparar(torch.ones(1, 200, 200))[0], torch.ones(1, 200, 200)))
ok(float(preparar(torch.ones(1, 100, 100))[0].sum()) == 0.0, "una imagen entera de 10 000 px no llega a 20 000")

# ============================================================ 3. vecindad de 8 y umbral
d = torch.zeros(1, 40, 40)
d[0, 5:10, 5:10] = 1.0
d[0, 10:15, 10:15] = 1.0              # se tocan solo por la esquina: una pieza de 50
ok(int(preparar(d, min_region_area=50)[0].sum()) == 50)
ok(int(preparar(d, min_region_area=51)[0].sum()) == 0)
d[0, 20:25, 30:35] = 1.0              # otra de 25, separada
ok(int(preparar(d, min_region_area=26)[0].sum()) == 50)
ok(int(preparar(d, min_region_area=0)[0].sum()) == 75, "con 0 no se quita nada")
ok(int(preparar(d, remove_small_regions=False)[0].sum()) == 75)

# ============================================================ 4. frente a las referencias, con mascaras al azar
rng = np.random.default_rng(20261124)
for caso in range(60):
    h, w = int(rng.integers(8, 70)), int(rng.integers(8, 70))
    lote = int(rng.integers(1, 4))
    suave = rng.random((lote, h, w)).astype(np.float32)
    # manchas: ruido suavizado a mano para que haya piezas, huecos y diagonales
    suave = (suave + np.roll(suave, 1, 1) + np.roll(suave, 1, 2) + np.roll(suave, -1, 1)) / 4
    umbral = float(rng.uniform(0.42, 0.58))
    mascara = torch.from_numpy(np.clip(suave - umbral + 0.5, 0, 1))
    area = int(rng.integers(1, 40))
    binaria = mascara.numpy() > 0.5
    salida = preparar(mascara, min_region_area=area)[0].numpy() > 0.5
    for b in range(lote):
        ok(np.array_equal(salida[b], islas_ingenua(binaria[b], area)), ("islas ingenua", caso, b))
        if cv2 is not None:
            ok(np.array_equal(salida[b], islas_original(binaria[b], area)), ("islas original", caso, b))
    rellena = preparar(mascara, fill_holes=True, remove_small_regions=False)[0].numpy() > 0.5
    for b in range(lote):
        ok(np.array_equal(rellena[b], huecos_ingenua(binaria[b])), ("huecos ingenua", caso, b))
        if cv2 is not None:
            ok(np.array_equal(rellena[b], huecos_original(binaria[b])), ("huecos original", caso, b))
    # las dos a la vez: primero huecos, luego islas
    ambas = preparar(mascara, fill_holes=True, min_region_area=area)[0].numpy() > 0.5
    for b in range(lote):
        ok(np.array_equal(ambas[b], islas_ingenua(huecos_ingenua(binaria[b]), area)), ("huecos + islas", caso, b))

# ============================================================ 5. sin scipy usa OpenCV y da lo mismo
if cv2 is not None:
    guardado = {k: sys.modules.get(k) for k in ("scipy", "scipy.ndimage")}
    con_scipy = preparar(m, fill_holes=True)[0]
    sys.modules["scipy"] = None
    sys.modules["scipy.ndimage"] = None
    try:
        ok(torch.equal(preparar(m, fill_holes=True)[0], con_scipy), "con OpenCV sale lo mismo que con scipy")
    finally:
        for k, v in guardado.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v

# ============================================================ 6. engordar, bordes y mezcla
p = torch.zeros(1, 41, 41)
p[0, 20, 20] = 1.0
g = preparar(p, remove_small_regions=False, grow_pixels=3)[0]
ok(int(g.sum()) == 49 and float(g[0, 17:24, 17:24].min()) == 1.0, "un pixel +3 px es un cuadrado de 7x7")
s = preparar(p, remove_small_regions=False, grow_pixels=3, inpaint_edge_mode="slight_feather")[0]
ok(0.0 < float(s[0, 20, 16]) < 1.0 and float(s.max()) <= 1.0 and float(s.min()) >= 0.0, "slight_feather deja borde suave")
ok(np.allclose(s[0].numpy(), gauss_ingenua(g[0].numpy() > 0.5, 1.5), atol=1e-5))
# Sin quitar islas, la mascara suave se pasa por el umbral de 0,5.
t = torch.tensor([[[0.2, 0.5, 0.51, 0.9]]])
ok(preparar(t, remove_small_regions=False)[0].tolist() == [[[0.0, 0.0, 1.0, 1.0]]])
for radio in (1, 4, 16, 40):
    z = preparar(esperada[:, :256, :256], stitch_feather_radius=radio)[1]
    ok(np.allclose(z[0].numpy(), gauss_ingenua(esperada[0, :256, :256].numpy() > 0.5, radio * 0.4), atol=1e-5), radio)

# ============================================================ 7. suavizado en el tiempo
v = torch.zeros(5, 32, 32)
v[2, 8:24, 8:24] = 1.0                # la mascara solo esta en el fotograma del medio
sin = preparar(v, remove_small_regions=False)[1]
con = preparar(v, remove_small_regions=False, temporal_smooth=True, temporal_sigma=1.5)[1]
ok(float(sin[1].max()) == 0.0 and float(con[1].max()) > 0.0 and float(con[2].max()) < float(sin[2].max()))
ok(torch.equal(preparar(v, remove_small_regions=False, temporal_smooth=True)[0], preparar(v, remove_small_regions=False)[0]),
   "la mascara de inpaint no se suaviza en el tiempo")
una = preparar(v[2:3], remove_small_regions=False, temporal_smooth=True)[1]
ok(torch.equal(una, sin[2:3]), "con un solo fotograma no hace nada")
# Referencia del suavizado: nucleo gaussiano de radio ceil(3*sigma), borde repetido, por pixel.
k = np.exp(-0.5 * (np.arange(-5, 6) / 1.5) ** 2)
k /= k.sum()
relleno = np.concatenate([np.repeat(sin.numpy()[:1], 5, 0), sin.numpy(), np.repeat(sin.numpy()[-1:], 5, 0)])
ref_t = np.stack([(relleno[i:i + 11] * k[:, None, None]).sum(0) for i in range(5)])
ok(np.allclose(con.numpy(), np.clip(ref_t, 0, 1), atol=1e-5))

# ============================================================ 8. errores y edge_aware
imagen = torch.zeros(1, 64, 64, 3)
try:
    preparar(torch.zeros(1, 64, 64), stitch_edge_mode="edge_aware", reference_image=imagen)
    raise AssertionError("edge_aware con imagen tiene que fallar")
except ValueError as e:
    ok("edge_aware" in str(e) and "suplente" in str(e), e)
ok(torch.equal(preparar(d, stitch_edge_mode="edge_aware")[1], preparar(d)[1]), "edge_aware sin imagen es gaussian, como el original")
ok(torch.equal(preparar(d, reference_image=imagen)[1], preparar(d)[1]), "gaussian no mira la imagen")
for malo, cambios in ((torch.zeros(1, 1, 8, 8), {}), ("no es un tensor", {}), (d, {"inpaint_edge_mode": "soft"}),
                      (d, {"stitch_edge_mode": "otro"})):
    try:
        preparar(malo, **cambios)
        raise AssertionError("tenia que fallar")
    except ValueError as e:
        ok("InpaintMaskPrepareMEC" in str(e), e)

# ============================================================ 9. el nodo devuelve las cuatro salidas
r = S.SuplenteInpaintMaskPrepareMEC.execute(m, **TARJETA)
ok(isinstance(r, _NodeOutput) and len(r.valores) == 4 and torch.equal(r.valores[0], inpaint) and isinstance(r.valores[3], str))
# Tal como llega del prompt de la tarjeta: los nueve widgets en orden y sin reference_image.
r2 = S.SuplenteInpaintMaskPrepareMEC.execute(m, False, True, 20000, 0, "hard_binary", "gaussian", 16, False, 1.5)
ok(torch.equal(r2.valores[0], inpaint))

print(f"test_suplente_mask_prepare: OK ({comprobaciones} comprobaciones; referencia de OpenCV: {'si' if cv2 is not None else 'NO, no esta instalado'})")
