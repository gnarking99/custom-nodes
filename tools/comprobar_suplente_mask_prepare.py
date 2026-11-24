# -*- coding: utf-8 -*-
"""Comprueba el suplente de InpaintMaskPrepareMEC DENTRO de un ComfyUI real, sin servidor, sin GPU y sin modelos.

Carga el nucleo y este pack con el mismo camino que main.py (nodes.init_extra_nodes) y mira que:
- `InpaintMaskPrepareMEC` existe y lo da este pack;
- lo que serviria /object_info tiene las mismas entradas (nombres, orden, tipos, rangos, opciones) y salidas que el
  nodo original de Code2Collapse (tests/datos/inpaint_mask_prepare_mec_c2c_9891293.json);
- se ejecuta con los valores de la tarjeta «enhance lora propio v9» y quita las islas de menos de 20 000 px;
- el resto del pack carga igual (los 29 CP_ y el suplente de MiniMax H3 Image).

Solo carga este pack (los demas custom nodes de esa carpeta no se tocan). No escribe en la carpeta de ComfyUI.

    <python de ComfyUI> -I tools/comprobar_suplente_mask_prepare.py <carpeta de ComfyUI>   # con la 0.42.1 y con la de la nube
"""
import asyncio
import json
import os
import sys
import tempfile

PACK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAIZ = os.path.abspath(sys.argv[1])
NOMBRE = "InpaintMaskPrepareMEC"
REFERENCIA = os.path.join(PACK, "tests", "datos", "inpaint_mask_prepare_mec_c2c_9891293.json")

# Una carpeta de custom nodes desechable con solo este pack enlazado.
carpeta = tempfile.mkdtemp(prefix="suplente_mask_prepare_")
os.symlink(PACK, os.path.join(carpeta, "cp_pack"))

sys.argv = [sys.argv[0], "--cpu", "--disable-all-custom-nodes", "--whitelist-custom-nodes", "cp_pack"]
sys.path.insert(0, RAIZ)
os.chdir(RAIZ)
import comfy.options  # noqa: E402
comfy.options.enable_args_parsing()
import folder_paths  # noqa: E402
folder_paths.folder_names_and_paths["custom_nodes"] = ([carpeta], set())
import nodes  # noqa: E402
import torch  # noqa: E402

version = {}
exec(open(os.path.join(RAIZ, "comfyui_version.py"), encoding="utf-8").read(), version)
version = version["__version__"]

asyncio.run(nodes.init_extra_nodes(init_custom_nodes=True, init_api_nodes=False))
cls = nodes.NODE_CLASS_MAPPINGS.get(NOMBRE)
modulo = getattr(cls, "RELATIVE_PYTHON_MODULE", "") if cls else ""
cp = sorted(k for k in nodes.NODE_CLASS_MAPPINGS if k.startswith("CP_"))
ref = json.load(open(REFERENCIA, encoding="utf-8"))
fallos = []
diferencias = []

if cls is None:
    fallos.append(f"ComfyUI {version}: {NOMBRE} no esta registrado")
else:
    if modulo != "custom_nodes.cp_pack":
        fallos.append(f"ComfyUI {version}: {NOMBRE} lo da {modulo or 'nadie'}, no este pack")
    info = json.loads(json.dumps(cls.GET_NODE_INFO_V1()))  # como lo sirve /object_info (tuplas -> listas)

    def normal(entrada):
        """Un desplegable sale [[opciones], {...}] en un nodo V1 y ["COMBO", {"options": [...], ...}] en uno V3."""
        tipo, opts = entrada[0], dict(entrada[1] if len(entrada) > 1 else {})
        if isinstance(tipo, list):
            tipo, opts["options"] = "COMBO", tipo
        opts.pop("multiselect", None)
        return [tipo, opts]

    for clave in ("name", "display_name", "category", "description", "input_order", "output", "output_name"):
        if info.get(clave) != ref[clave]:
            diferencias.append(clave)
    for seccion in ("required", "optional"):
        suyas, mias = ref["input"].get(seccion, {}), info["input"].get(seccion, {})
        for nombre in sorted(set(suyas) | set(mias)):
            if nombre not in suyas or nombre not in mias or normal(suyas[nombre]) != normal(mias[nombre]):
                diferencias.append(f"input.{seccion}.{nombre}")
    if diferencias:
        fallos.append(f"el esquema no es el del nodo original en: {diferencias}")

    # Los widgets del nodo 34 de la tarjeta, en su orden, y una mascara con el personaje y dos islas.
    mascara = torch.zeros(1, 512, 512)
    mascara[0, 100:300, 100:300] = 1.0     # 40 000 px
    mascara[0, 400:500, 100:299] = 1.0     # 19 900 px
    mascara[0, 10:20, 10:20] = 1.0
    salida = cls.execute(mascara, False, True, 20000, 0, "hard_binary", "gaussian", 16, False, 1.5)
    valores = getattr(salida, "result", None) or getattr(salida, "args", None)
    if not valores or len(valores) != 4:
        fallos.append(f"execute no devolvio cuatro salidas: {salida!r}")
    else:
        limpia = valores[0]
        if tuple(limpia.shape) != (1, 512, 512) or int(limpia.sum()) != 40000 or float(limpia[0, 100:300, 100:300].min()) != 1.0:
            fallos.append(f"no quito las islas: quedan {int(limpia.sum())} px y tenian que quedar 40000")

if len(cp) != 29:
    fallos.append(f"se cargaron {len(cp)} nodos CP_ y el pack tiene 29")
if "MiniMaxH3ImageConditioning" not in nodes.NODE_CLASS_MAPPINGS:
    fallos.append("falta MiniMaxH3ImageConditioning (nucleo o suplente)")

print(json.dumps({"comfyui": version, "lo_da": modulo, "cp": len(cp), "diferencias_de_esquema": diferencias}, ensure_ascii=False))
print("FALLA\n- " + "\n- ".join(fallos) if fallos else "OK")
sys.exit(1 if fallos else 0)
