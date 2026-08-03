# -*- coding: utf-8 -*-
"""ComfyUI Character Pipeline — nodos para flujos de recuperacion de detalle
de personajes con LoRA (Flux.2 Klein / LTX-2.3 / Wan 2.2).

API V3 (ComfyUI >= 0.23). Requiere `comfy_api.latest`.
"""

from comfy_api.latest import ComfyExtension, io

from .comfy_character_pipeline.guards import MaskGuard, BatchMatch
from .comfy_character_pipeline.geometry import SnapResolution, VideoChunk
from .comfy_character_pipeline.sampling import SigmasDenoise
from .comfy_character_pipeline.profile import CharacterProfile
from .comfy_character_pipeline.canon import CanonPanelBuilder, PromptTemplate

NODES = [
    CharacterProfile,
    MaskGuard,
    BatchMatch,
    SnapResolution,
    VideoChunk,
    SigmasDenoise,
    CanonPanelBuilder,
    PromptTemplate,
]


class CharacterPipelineExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return NODES


async def comfy_entrypoint() -> CharacterPipelineExtension:
    return CharacterPipelineExtension()
