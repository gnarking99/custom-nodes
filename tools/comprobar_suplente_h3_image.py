# -*- coding: utf-8 -*-
"""Comprueba el suplente de MiniMaxH3ImageConditioning DENTRO de un ComfyUI real, sin servidor, sin GPU y sin modelos.

Carga el nucleo y este pack con el mismo camino que main.py (nodes.init_extra_nodes: primero comfy_extras, luego los
custom nodes con los nombres del nucleo como `ignore`) y mira quien se queda con el nombre:
- ComfyUI anterior a la 0.45 (p. ej. la 0.42.1 de los puestos): tiene que ganar el SUPLENTE de este pack, con el mismo
  esquema que el nodo del nucleo de la v0.45.6 (tests/datos/minimax_h3_image_conditioning_v0456.json, sacado del
  /object_info de la nube; solo cambia python_module), y ejecutarlo tiene que fallar con el mensaje del suplente.
- ComfyUI 0.45 o posterior: tiene que ganar el nodo del NUCLEO (el suplente se ignora) y el resto del pack cargar igual.
En los dos casos se cargan todos los CP_ del pack.

Solo carga este pack (los demas custom nodes de esa carpeta no se tocan). No escribe en la carpeta de ComfyUI.

    <python de ComfyUI> -I tools/comprobar_suplente_h3_image.py <carpeta de ComfyUI>
"""
import asyncio
import json
import os
import sys
import tempfile

PACK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAIZ = os.path.abspath(sys.argv[1])
NOMBRE = "MiniMaxH3ImageConditioning"
REFERENCIA = os.path.join(PACK, "tests", "datos", "minimax_h3_image_conditioning_v0456.json")

# Una carpeta de custom nodes desechable con solo este pack enlazado.
carpeta = tempfile.mkdtemp(prefix="suplente_h3_image_")
os.symlink(PACK, os.path.join(carpeta, "cp_pack"))

sys.argv = [sys.argv[0], "--cpu", "--disable-all-custom-nodes", "--whitelist-custom-nodes", "cp_pack"]
sys.path.insert(0, RAIZ)
os.chdir(RAIZ)
import comfy.options  # noqa: E402
comfy.options.enable_args_parsing()
import folder_paths  # noqa: E402
folder_paths.folder_names_and_paths["custom_nodes"] = ([carpeta], set())
import nodes  # noqa: E402

version = {}
exec(open(os.path.join(RAIZ, "comfyui_version.py"), encoding="utf-8").read(), version)
version = version["__version__"]
nueva = tuple(int(x) for x in version.split(".")[:2]) >= (0, 45)

asyncio.run(nodes.init_extra_nodes(init_custom_nodes=True, init_api_nodes=False))
cls = nodes.NODE_CLASS_MAPPINGS.get(NOMBRE)
modulo = getattr(cls, "RELATIVE_PYTHON_MODULE", "")
info = json.loads(json.dumps(cls.GET_NODE_INFO_V1()))  # como lo sirve /object_info (tuplas -> listas)
ref = json.load(open(REFERENCIA, encoding="utf-8"))
diferencias = sorted(k for k in set(ref) | set(info) if k != "python_module" and ref.get(k) != info.get(k))
cp = sorted(k for k in nodes.NODE_CLASS_MAPPINGS if k.startswith("CP_"))

fallos = []
if nueva and modulo != "comfy_extras.nodes_minimax_h3":
    fallos.append(f"ComfyUI {version}: {NOMBRE} lo da {modulo}, no el nucleo")
if not nueva and modulo != "custom_nodes.cp_pack":
    fallos.append(f"ComfyUI {version}: {NOMBRE} lo da {modulo or 'nadie'}, no el suplente")
if diferencias:
    fallos.append(f"el esquema no es el de la v0.45.6 en: {diferencias}")
if len(cp) != 29:
    fallos.append(f"se cargaron {len(cp)} nodos CP_ y el pack tiene 29")
if not nueva:
    try:
        cls.execute(clip=None, prompt="", width=0, height=0)
        fallos.append("el suplente se ejecuto sin fallar")
    except RuntimeError as e:
        if "solo funciona en la nube" not in str(e):
            fallos.append(f"el suplente fallo con otro mensaje: {e}")

print(json.dumps({"comfyui": version, "gana": modulo, "cp": len(cp), "diferencias_de_esquema": diferencias}, ensure_ascii=False))
print("FALLA\n- " + "\n- ".join(fallos) if fallos else "OK")
sys.exit(1 if fallos else 0)
