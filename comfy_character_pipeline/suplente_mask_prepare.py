# -*- coding: utf-8 -*-
"""Suplente de InpaintMaskPrepareMEC: limpia una mascara (rellena huecos, quita islas sueltas, la engorda) y da la
mascara de inpaint y la de mezcla.

La tarjeta ENHANCE IMAGEN / enhance lora propio v9 usa `InpaintMaskPrepareMEC` para quitar las islas sueltas que deja
SAM (nodo 34, y el 54 de ELEGIR PINTANDO). Ese nodo era del pack Code2Collapse/ComfyUI-CustomNodePacks
(nodes/inpaint_suite.py, commit 9891293, licencia Apache-2.0), que salio de `nodes_config.json` el 28-10-2026 porque
ninguna tarjeta del repositorio lo usaba; esta tarjeta no esta en el repositorio y se quedo sin el nodo: desde ese
dia fallaba siempre en la nube con «nodo no instalado».

Este archivo registra un nodo con el MISMO node_id y el MISMO esquema (entradas, orden, tipos, rangos, opciones y
las cuatro salidas), asi que la tarjeta publicada y la pantalla Enhance de la app, que comprueba que el nodo 34 es de
este tipo, siguen valiendo sin tocarlas. A diferencia del suplente de MiniMax H3 Image, este SI se ejecuta: la
logica esta escrita aqui, sin el pack original.

Que hace igual que el original (comprobado en tests/test_suplente_mask_prepare.py contra una copia de su algoritmo):
- fill_holes: rellena los huecos interiores de la mascara binaria (umbral 0,5).
- remove_small_regions: quita las piezas sueltas (vecindad de 8) de menos de min_region_area pixeles; la que tiene
  justo ese tamano se queda. La mascara sale binaria.
- grow_pixels: engorda con un cuadrado de 2*grow+1.
- inpaint_edge_mode: hard_binary (umbral 0,5) o slight_feather (desenfoque gaussiano de sigma 1,5).
- stitch_edge_mode gaussian: la binaria desenfocada con sigma = stitch_feather_radius * 0,4.
- temporal_smooth: suaviza la mascara de mezcla entre fotogramas (solo con mas de uno).

Que NO hace: stitch_edge_mode «edge_aware» CON reference_image conectada (el original ajustaba el borde a los
contornos de la imagen). Falla con un mensaje claro en vez de dar otra cosa en silencio. Sin reference_image, el
original ya usaba gaussian, y este tambien.

Si el pack original estuviera instalado a la vez, los dos registrarian el mismo nombre y ganaria el ultimo en cargar.
En BITT no esta instalado.
"""

import math

import numpy as np
import torch
import torch.nn.functional as F
from comfy_api.latest import io

NODO = "InpaintMaskPrepareMEC"
INPAINT_EDGE_MODES = ["hard_binary", "slight_feather"]
STITCH_EDGE_MODES = ["gaussian", "edge_aware"]

MENSAJE_EDGE_AWARE = (
    f"[{NODO}] El modo «edge_aware» con reference_image no esta en el suplente del pack Character Pipeline. "
    "Usa stitch_edge_mode «gaussian» o desconecta reference_image."
)


def _etiquetar(binaria, vecindad):
    """Piezas conexas de una mascara binaria [H,W] de numpy, con vecindad de 4 u 8: (etiquetas, cuantas).

    scipy viene con ComfyUI; si faltara, se usa OpenCV. La importacion va aqui dentro para que un motor sin ninguna de
    las dos siga cargando el resto del pack."""
    try:
        from scipy import ndimage
        return ndimage.label(binaria, structure=np.ones((3, 3), dtype=bool) if vecindad == 8 else None)
    except ImportError:
        pass
    try:
        import cv2
        cuantas, etiquetas = cv2.connectedComponents(binaria.astype(np.uint8), connectivity=vecindad)
        return etiquetas, cuantas - 1
    except ImportError:
        raise RuntimeError(f"[{NODO}] Hace falta scipy u opencv-python para separar las piezas de la mascara.")


def quitar_islas(mask, min_area):
    """Quita de una mascara [B,H,W] las piezas (vecindad de 8) de menos de min_area pixeles. Binaria en float32."""
    if min_area <= 0:
        return mask
    salida = []
    for b in range(mask.shape[0]):
        binaria = mask[b].detach().cpu().numpy() > 0.5
        etiquetas, _ = _etiquetar(binaria, 8)
        conservar = np.bincount(etiquetas.ravel()) >= min_area
        conservar[0] = False  # la etiqueta 0 es el fondo
        salida.append(torch.from_numpy(conservar[etiquetas].astype(np.float32)))
    return torch.stack(salida, dim=0).to(mask.device)


def rellenar_huecos(mask):
    """Rellena los huecos interiores de una mascara [B,H,W]: el fondo que no llega al borde de la imagen. Binaria en float32."""
    salida = []
    for b in range(mask.shape[0]):
        binaria = mask[b].detach().cpu().numpy() > 0.5
        # El fondo se recorre con vecindad de 4, la contraria a la de 8 de las piezas.
        fondo, cuantas = _etiquetar(~binaria, 4)
        toca_borde = np.zeros(cuantas + 1, dtype=bool)
        toca_borde[np.concatenate([fondo[0], fondo[-1], fondo[:, 0], fondo[:, -1]])] = True
        toca_borde[0] = True  # la etiqueta 0 aqui es la propia mascara, no un hueco
        salida.append(torch.from_numpy((binaria | ~toca_borde[fondo]).astype(np.float32)))
    return torch.stack(salida, dim=0).to(mask.device)


def _nucleo_gauss(sigma, device, dtype):
    radio = max(1, int(math.ceil(3.0 * sigma)))
    x = torch.arange(2 * radio + 1, device=device, dtype=dtype) - radio
    nucleo = torch.exp(-0.5 * (x / sigma) ** 2)
    return nucleo / nucleo.sum()


def desenfocar(mask, sigma):
    """Desenfoque gaussiano separable de una mascara [B,H,W], repitiendo el borde."""
    if sigma <= 0:
        return mask
    nucleo = _nucleo_gauss(sigma, mask.device, mask.dtype)
    margen = len(nucleo) // 2
    m = mask.unsqueeze(1)
    m = F.conv2d(F.pad(m, (margen, margen, 0, 0), mode="replicate"), nucleo.view(1, 1, 1, -1))
    m = F.conv2d(F.pad(m, (0, 0, margen, margen), mode="replicate"), nucleo.view(1, 1, -1, 1))
    return m.squeeze(1)


def suavizar_en_el_tiempo(mask, sigma):
    """Desenfoque gaussiano a lo largo de los fotogramas de una mascara [B,H,W]."""
    b, h, w = mask.shape
    if b <= 1 or sigma <= 0:
        return mask
    nucleo = _nucleo_gauss(sigma, mask.device, mask.dtype)
    margen = len(nucleo) // 2
    plano = mask.reshape(b, h * w).permute(1, 0).unsqueeze(1)
    plano = F.conv1d(F.pad(plano, (margen, margen), mode="replicate"), nucleo.view(1, 1, -1))
    return plano.squeeze(1).permute(1, 0).reshape(b, h, w).clamp(0.0, 1.0)


def preparar(mask, fill_holes, remove_small_regions, min_region_area, grow_pixels, inpaint_edge_mode,
             stitch_edge_mode, stitch_feather_radius, temporal_smooth, temporal_sigma, reference_image=None):
    """La logica del nodo. Devuelve (inpaint_mask, stitch_blend_mask, debug_preview, info)."""
    if not isinstance(mask, torch.Tensor) or mask.ndim not in (2, 3):
        raise ValueError(f"[{NODO}] La mascara tiene que ser un tensor [H,W] o [B,H,W].")
    if inpaint_edge_mode not in INPAINT_EDGE_MODES:
        raise ValueError(f"[{NODO}] inpaint_edge_mode desconocido: {inpaint_edge_mode!r}. Vale: {INPAINT_EDGE_MODES}.")
    if stitch_edge_mode not in STITCH_EDGE_MODES:
        raise ValueError(f"[{NODO}] stitch_edge_mode desconocido: {stitch_edge_mode!r}. Vale: {STITCH_EDGE_MODES}.")
    if stitch_edge_mode == "edge_aware" and reference_image is not None:
        raise ValueError(MENSAJE_EDGE_AWARE)

    if mask.dim() == 2:
        mask = mask.unsqueeze(0)
    b, h, w = mask.shape
    trabajo = mask.clone()

    rellenados = quitados = 0
    if fill_holes:
        antes = (trabajo > 0.5).sum().item()
        trabajo = rellenar_huecos(trabajo)
        rellenados = int((trabajo > 0.5).sum().item() - antes)
    if remove_small_regions and min_region_area > 0:
        antes = (trabajo > 0.5).sum().item()
        trabajo = quitar_islas(trabajo, min_region_area)
        quitados = int(antes - (trabajo > 0.5).sum().item())
    if grow_pixels > 0:
        trabajo = F.max_pool2d(trabajo.unsqueeze(1), kernel_size=2 * grow_pixels + 1, stride=1,
                               padding=grow_pixels).squeeze(1).clamp(0.0, 1.0)

    binaria = (trabajo > 0.5).float()
    inpaint_mask = binaria if inpaint_edge_mode == "hard_binary" else desenfocar(binaria, 1.5).clamp(0.0, 1.0)
    stitch_blend_mask = desenfocar(binaria, stitch_feather_radius * 0.4)
    if temporal_smooth and b > 1:
        stitch_blend_mask = suavizar_en_el_tiempo(stitch_blend_mask, temporal_sigma)

    debug_preview = torch.stack([inpaint_mask, stitch_blend_mask, torch.zeros_like(inpaint_mask)], dim=-1)
    info = "\n".join([
        f"{NODO} (suplente de Character Pipeline):",
        f"  input: {b}x{h}x{w}",
        f"  fill_holes: {fill_holes} (pixels filled: {rellenados})",
        f"  remove_small_regions: {remove_small_regions} (pixels removed: {quitados})",
        f"  grow_pixels: {grow_pixels}",
        f"  inpaint_edge_mode: {inpaint_edge_mode}",
        f"  stitch_edge_mode: {stitch_edge_mode}, radius={stitch_feather_radius}",
        f"  temporal_smooth: {bool(temporal_smooth and b > 1)}",
    ])
    return inpaint_mask, stitch_blend_mask, debug_preview, info


class SuplenteInpaintMaskPrepareMEC(io.ComfyNode):
    """Mismo nombre y esquema que InpaintMaskPrepareMEC de Code2Collapse (commit 9891293); la logica es de este pack."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=NODO,
            display_name="Inpaint Mask Prepare",
            category="C2C/Inpaint",
            description="Clean, grow, and prepare dual masks: inpaint_mask for model + stitch_blend_mask for composite.",
            inputs=[
                io.Mask.Input("mask", tooltip="Raw input mask (B,H,W)"),
                io.Boolean.Input("fill_holes", default=True, tooltip="Fill interior holes in the mask"),
                io.Boolean.Input("remove_small_regions", default=True, tooltip="Remove small disconnected blobs"),
                io.Int.Input("min_region_area", default=100, min=0, max=100000, step=10,
                             tooltip="Minimum region area in pixels to keep"),
                io.Int.Input("grow_pixels", default=4, min=0, max=256, step=1, tooltip="Dilate mask by this many pixels"),
                io.Combo.Input("inpaint_edge_mode", options=INPAINT_EDGE_MODES, default="hard_binary",
                               tooltip="Edge style for inpaint mask: hard_binary or slight_feather"),
                io.Combo.Input("stitch_edge_mode", options=STITCH_EDGE_MODES, default="gaussian",
                               tooltip="Edge style for stitch blend mask: gaussian or edge_aware"),
                io.Int.Input("stitch_feather_radius", default=16, min=1, max=128, step=1,
                             tooltip="Feather radius for the stitch blend mask"),
                io.Boolean.Input("temporal_smooth", default=False,
                                 tooltip="Apply Gaussian temporal smoothing along batch dimension (for video)"),
                io.Float.Input("temporal_sigma", default=1.5, min=0.1, max=10.0, step=0.1,
                               tooltip="Temporal Gaussian sigma in frames"),
                io.Image.Input("reference_image", optional=True,
                               tooltip="Reference image for edge-aware stitch blend mask (required for edge_aware mode)"),
            ],
            outputs=[
                io.Mask.Output(display_name="inpaint_mask"),
                io.Mask.Output(display_name="stitch_blend_mask"),
                io.Image.Output(display_name="debug_preview"),
                io.String.Output(display_name="info"),
            ],
        )

    @classmethod
    def execute(cls, mask, fill_holes, remove_small_regions, min_region_area, grow_pixels, inpaint_edge_mode,
                stitch_edge_mode, stitch_feather_radius, temporal_smooth, temporal_sigma,
                reference_image=None) -> io.NodeOutput:
        return io.NodeOutput(*preparar(mask, fill_holes, remove_small_regions, min_region_area, grow_pixels,
                                       inpaint_edge_mode, stitch_edge_mode, stitch_feather_radius, temporal_smooth,
                                       temporal_sigma, reference_image))
