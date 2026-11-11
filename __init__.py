# -*- coding: utf-8 -*-
"""ComfyUI Character Pipeline — nodos para flujos de recuperacion de detalle
de personajes con LoRA (Flux.2 Klein / LTX-2.3 / Wan 2.2).

API V3 (ComfyUI >= 0.23). Requiere `comfy_api.latest`.
"""

from comfy_api.latest import ComfyExtension, io

from .comfy_character_pipeline.guards import MaskGuard, BatchMatch
from .comfy_character_pipeline.abdiff import ABDiff
from .comfy_character_pipeline.geometry import SnapResolution, VideoChunk
from .comfy_character_pipeline.ltx_res import LTXResolution
from .comfy_character_pipeline.ltx_gen import LTXGenPlan
from .comfy_character_pipeline.wan_gen import WanGenPlan
from .comfy_character_pipeline.sampling import SigmasDenoise
from .comfy_character_pipeline.ltx_sigmas import LTXSigmas
from .comfy_character_pipeline.wan_split import WanSplit
from .comfy_character_pipeline.profile import CharacterProfile
from .comfy_character_pipeline.canon import CanonPanelBuilder, PromptTemplate
from .comfy_character_pipeline.lora_meta import LoraTriggerReader
from .comfy_character_pipeline.artist import QualityPreset, PipelineStatus
from .comfy_character_pipeline.panel import ControlPanel
from .comfy_character_pipeline.poses import PosesDetectar, PosesMontar, PosesSinEsperas
from .comfy_character_pipeline.poses_fotos import PosesDesdeFotos
from .comfy_character_pipeline.blocking import PosesClaves, PosesComponer, LTXGuiasPoses
from .comfy_character_pipeline.lenta import LentaPlan, LentaElegir
from .comfy_character_pipeline.lenta_grafo import BlockingLento
from .comfy_character_pipeline.cadencia import Cadencia

from .comfy_character_pipeline.compat_av import tolerar_metadatos_de_video

# Los .mov de QuickTime en espanol traen una etiqueta en Latin-1 que tumba av.open() en PyAV <= 18 (ver compat_av.py).
tolerar_metadatos_de_video()

# extension de frontend: js/control_panel.js
WEB_DIRECTORY = "./js"

NODES = [
    # artista
    ControlPanel,
    QualityPreset,
    PipelineStatus,
    # organizacion
    LoraTriggerReader,
    CharacterProfile,
    # guardas
    MaskGuard,
    BatchMatch,
    ABDiff,
    # geometria
    LTXResolution,
    LTXGenPlan,
    WanGenPlan,
    SnapResolution,
    VideoChunk,
    # muestreo
    SigmasDenoise,
    LTXSigmas,
    WanSplit,
    # canon
    CanonPanelBuilder,
    PromptTemplate,
    # render por poses
    PosesDetectar,
    PosesMontar,
    PosesSinEsperas,
    PosesDesdeFotos,
    # blocking a animacion
    PosesClaves,
    PosesComponer,
    LTXGuiasPoses,
    # camara lenta
    LentaPlan,
    LentaElegir,
    BlockingLento,
    Cadencia,
]


class CharacterPipelineExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return NODES


async def comfy_entrypoint() -> CharacterPipelineExtension:
    return CharacterPipelineExtension()


__all__ = ["comfy_entrypoint", "WEB_DIRECTORY"]
