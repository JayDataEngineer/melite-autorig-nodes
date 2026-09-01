"""Tech Noir Auto-Rig bridge nodes for ComfyUI.

Provides the glue node that connects TRELLIS output (STRING glb_path) to
SkinToken Rig input (TRIMESH). Without this bridge, the two node packs
cannot be chained in a single ComfyUI graph — TRELLIS emits a file path,
SkinToken consumes a Python trimesh object.

    Trellis2ImageTo3D → (STRING) → LoadGLBAsTrimesh → (TRIMESH) → SkinTokenRigTrimesh

This pack intentionally contains ONLY the bridge node. The actual rigging
logic lives in the upstream ComfyUI-SkinToken pack (vendored as a git
submodule at custom_nodes/ComfyUI-SkinToken/).
"""
from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
