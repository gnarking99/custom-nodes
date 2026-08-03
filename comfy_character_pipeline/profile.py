# -*- coding: utf-8 -*-
"""Perfil de personaje: una sola fuente de verdad para los tres flujos."""

import json
import os
from comfy_api.latest import io

PROFILES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "profiles")

DEFAULTS = {
    "trigger": "",
    "lora_name": "",
    "strength_edit": 0.85,
    "strength_refine": 0.25,
    "canon_seed": 1234567890,
    "canon_prompt": "",
    "canon_negative": "",
    "identity_lock": (
        "Keep the character identity, pose, facial expression, camera angle and "
        "composition exactly as in the reference image. Change nothing else."
    ),
    "notes": "",
}


def _list_profiles():
    try:
        files = sorted(f for f in os.listdir(PROFILES_DIR) if f.lower().endswith(".json"))
    except FileNotFoundError:
        files = []
    return files or ["<no hay perfiles en profiles/>"]


class CharacterProfile(io.ComfyNode):
    """Carga la ficha de un personaje desde `profiles/<nombre>.json`.

    El problema que resuelve no es tecnico, es de organizacion. Ahora mismo el
    nombre del LoRA, su fuerza en cada pase, la palabra de disparo, la semilla del
    canon y el candado de identidad estan repetidos a mano en tres flujos
    distintos (Flux, LTX, Wan). Cambiar de personaje significa tocar una docena de
    widgets y equivocarse en alguno.

    Con este nodo cambias de personaje en UN desplegable y los tres flujos leen lo
    mismo. El perfil es un JSON versionable en el repo: queda historial de que
    fuerza funcionaba para cada personaje.

    NOTA: los perfiles se leen al arrancar ComfyUI. Si anades un JSON nuevo,
    reinicia (o recarga la pagina y vuelve a crear el nodo).
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CP_CharacterProfile",
            display_name="Character Profile",
            category="character-pipeline/organization",
            description="Ficha de personaje unica para los flujos de Flux, LTX y Wan.",
            inputs=[
                io.Combo.Input("profile", options=_list_profiles()),
            ],
            outputs=[
                io.String.Output(display_name="trigger"),
                io.String.Output(display_name="lora_name"),
                io.Float.Output(display_name="strength_edit"),
                io.Float.Output(display_name="strength_refine"),
                io.Int.Output(display_name="canon_seed"),
                io.String.Output(display_name="canon_prompt"),
                io.String.Output(display_name="canon_negative"),
                io.String.Output(display_name="identity_lock"),
                io.String.Output(display_name="profile_name"),
            ],
        )

    @classmethod
    def execute(cls, profile) -> io.NodeOutput:
        if not profile or profile.startswith("<"):
            raise ValueError(
                f"[CharacterProfile] No hay ningun perfil disponible.\n"
                f"  Carpeta esperada: {PROFILES_DIR}\n"
                f"  Crea ahi un JSON (por ejemplo `mipersonaje.json`) y reinicia ComfyUI.\n"
                f"  En instalaciones en la nube, comprueba que el repo se haya clonado "
                f"entero y no solo los archivos .py."
            )

        path = os.path.join(PROFILES_DIR, profile)
        if not os.path.isfile(path):
            raise ValueError(
                f"[CharacterProfile] No encuentro el perfil '{profile}'.\n"
                f"  Buscando en: {PROFILES_DIR}\n"
                f"  Si lo acabas de crear, reinicia ComfyUI: la lista se lee al arrancar."
            )

        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"[CharacterProfile] El perfil '{profile}' no es JSON valido.\n"
                f"  {exc}\n"
                f"  Suele ser una coma de mas al final o comillas tipograficas."
            ) from exc

        if not isinstance(data, dict):
            raise ValueError(
                f"[CharacterProfile] El perfil '{profile}' debe ser un objeto JSON "
                f"(llaves), no {type(data).__name__}."
            )

        unknown = [k for k in data if k not in DEFAULTS]
        if unknown:
            print(f"[CharacterProfile] Aviso: claves ignoradas en '{profile}': {', '.join(unknown)}")

        cfg = dict(DEFAULTS)
        cfg.update(data)

        missing = [k for k in ("trigger", "lora_name") if not str(cfg[k]).strip()]
        if missing:
            raise ValueError(
                f"[CharacterProfile] El perfil '{profile}' no define: {', '.join(missing)}."
            )

        def _num(key, cast):
            try:
                return cast(cfg[key])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"[CharacterProfile] '{key}' del perfil '{profile}' no es un numero "
                    f"valido (valor: {cfg[key]!r})."
                ) from exc

        return io.NodeOutput(
            str(cfg["trigger"]),
            str(cfg["lora_name"]),
            _num("strength_edit", float),
            _num("strength_refine", float),
            _num("canon_seed", int),
            str(cfg["canon_prompt"]),
            str(cfg["canon_negative"]),
            str(cfg["identity_lock"]),
            os.path.splitext(profile)[0],
        )
