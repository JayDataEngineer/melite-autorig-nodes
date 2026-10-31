"""Auto-rig bridge + motion-retarget nodes for ComfyUI.

Nodes:

  LoadGLBAsTrimesh — bridges STRING (glb_path) → TRIMESH (trimesh.Trimesh)
  SaveRiggedModel — OUTPUT_NODE that captures the rigged_path STRING and
    returns it with 3D-preview UI data. Without this, SkinTokenRigTrimesh's
    output is invisible to ComfyUI's response (it lacks OUTPUT_NODE=True).
  KimodoExportFBX — transfers the SOMA-77 template's hand-painted LBS
    weights onto a source mesh and writes the rigged GLB (the node writes
    itself; GLB is the only format — the FBX/retarget arms died with the
    blender lane).
  KimodoTransferWeights — copies the rig + skin weights from a SOMA-rigged
    CLEAN mesh onto a CLOTHED mesh (same body, AI-generated clothing/hair).
    Uses GPU KNN (torch.cdist). The clothing step of the character pipeline.

The actual rigging AI lives in custom_nodes/ComfyUI-SkinToken/ (upstream
git submodule). This pack is pure glue + retargeting logic.
"""
from __future__ import annotations

import logging
import os
import shutil
import time
import json as _json

log = logging.getLogger(__name__)


class LoadGLBAsTrimesh:
    """Load a GLB/GLTF/OBJ/STL file as a trimesh.Trimesh object.

    Bridges the type gap between melite-trellis-nodes (STRING output) and
    ComfyUI-SkinToken (TRIMESH input). Placed between them in the graph:

        Trellis2ImageTo3D.out(0) → LoadGLBAsTrimesh.glb_path → SkinTokenRigTrimesh.trimesh
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "glb_path": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": (
                            "Path to a GLB/GLTF/OBJ/STL mesh file. "
                            "Typically the output of Trellis2ImageTo3D."
                        ),
                        "forceInput": True,
                    },
                ),
            },
        }

    RETURN_TYPES = ("TRIMESH",)
    RETURN_NAMES = ("trimesh",)
    FUNCTION = "load"
    CATEGORY = "Melite/AutoRig"
    OUTPUT_NODE = False

    def load(self, glb_path: str):
        if not glb_path or not glb_path.strip():
            raise ValueError(
                "LoadGLBAsTrimesh: glb_path is empty. "
                "Connect a Trellis2ImageTo3D output or provide a valid path."
            )
        path = glb_path.strip()

        # Resolve relative paths against ComfyUI input/ and output/
        # directories — same logic as KimodoExportFBX. This lets the
        # frontend pass uploaded filenames (e.g. "kimodo_character.glb")
        # without knowing ComfyUI's internal directory layout.
        if not os.path.isabs(path) or not os.path.isfile(path):
            import folder_paths
            _input_dir = folder_paths.get_input_directory()
            _output_dir = folder_paths.get_output_directory()
            for d in (_input_dir, _output_dir):
                candidate = os.path.join(d, path)
                if os.path.isfile(candidate):
                    path = candidate
                    break
            else:
                for root, _dirs, files in os.walk(_output_dir):
                    if path in files:
                        path = os.path.join(root, path)
                        break

        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"LoadGLBAsTrimesh: file not found: {glb_path} "
                f"(checked abs, input/{glb_path}, output/{glb_path})"
            )

        import trimesh

        # force='mesh' ensures we get a Trimesh, not a Scene.
        # TRELLIS output is single-mesh, but this is defensive.
        mesh = trimesh.load(path, force="mesh", process=False)

        if mesh is None:
            raise RuntimeError(f"LoadGLBAsTrimesh: trimesh.load returned None for {path}")

        # Validate that the mesh has geometry SkinToken can consume.
        # SkinToken's _build_asset() reads .vertices and .faces as numpy arrays.
        verts = getattr(mesh, "vertices", None)
        faces = getattr(mesh, "faces", None)
        if verts is None or faces is None or len(verts) == 0 or len(faces) == 0:
            raise RuntimeError(
                f"LoadGLBAsTrimesh: loaded mesh has no geometry "
                f"(vertices={len(verts) if verts is not None else 'None'}, "
                f"faces={len(faces) if faces is not None else 'None'}). "
                f"File may be corrupt or empty: {path}"
            )

        log.info(
            "LoadGLBAsTrimesh: loaded %s — %d verts, %d faces",
            os.path.basename(path), len(verts), len(faces),
        )
        return (mesh,)


class SaveRiggedModel:
    """Output capture node for the auto-rig pipeline.

    SkinTokenRigTrimesh saves the rigged GLB/FBX to disk but does NOT
    declare OUTPUT_NODE=True, so ComfyUI never includes its result in the
    response. This node sits after SkinTokenRigTrimesh, takes the
    rigged_path STRING, and returns it with three_model UI data so the
    frontend can display the rigged mesh in the output area.

    Mirrors the ui pattern from Trellis2ImageTo3D's return dict.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "rigged_path": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Path to the rigged GLB/FBX (from SkinTokenRigTrimesh).",
                        "forceInput": True,
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("rigged_path",)
    FUNCTION = "save"
    CATEGORY = "Melite/AutoRig"
    OUTPUT_NODE = True

    def save(self, rigged_path: str):
        filename = os.path.basename(rigged_path) if rigged_path else ""
        # Extract subfolder relative to ComfyUI's output directory so /view
        # can find the file. SkinToken saves to output/3D/TrellisAutoRig_*.glb
        # — without the correct subfolder, ComfyUI's /view returns 404 and
        # extract_media silently skips the rigged GLB.
        subdir = ""
        if rigged_path:
            import folder_paths
            # THE OUTPUT-DIR TRUTH (weight-suite receipt 2026-10-31:
            # the copy/transfer lanes executed on silicon but nothing
            # harvested — COMFYUI_ROOT is unset under the service
            # launch, the /root/ComfyUI fallback escaped to "..",
            # subdir fell "", and /view 404'd the 3D/ file. The
            # engine's own folder_paths owns the base — no env).
            output_base = folder_paths.get_output_directory()
            try:
                rel = os.path.relpath(rigged_path, output_base)
                parent = os.path.dirname(rel)
                # Only set subdir if it's a real subfolder inside output/
                if parent and parent != "." and not parent.startswith(".."):
                    subdir = parent
            except Exception:
                pass
        return {
            "result": (rigged_path,),
            "ui": {
                "three_model": [
                    {
                        "filename": filename,
                        "subfolder": subdir,
                        "type": "output",
                    }
                ]
            },
        }


class KimodoExportFBX:
    """Transfer the SOMA-77 template's hand-painted LBS weights onto a
    source mesh and write the rigged GLB.

    The node IS the output capture: it writes the GLB itself and returns
    its path (the bench mounts the three_model preview). No format
    conversion, no motion retarget, no Blender — the FBX/retarget arms
    and the MIMO visual validation rode headless Blender subprocesses and
    were deleted with the blender lane (operator 2026-11-01: melite
    carries no blender — it rides the DSH profiles). GLB is the only
    output format; motion rides the kimodo family's RaySomaxBake path.

    Pipeline (NVIDIA SOMA-X SOTA):
      1. AABB-align SOMA template mesh into source mesh space
         (uniform scale via height ratio + axis permutation)
      2. BarycentricInterpolator: for each source vertex, find its
         containing triangle in the T_align'd SOMA template mesh.
         Interpolate the template's hand-painted weights via the
         triangle's barycentric coords. This preserves the template's
         anatomically-correct weight boundaries EXACTLY — structurally
         eliminating the torso/arm "wing" artifact that bone-heat
         smoothing caused on raised-arm motions.
      3. Cross-body decontamination (prune L↔R contamination on
         inner thighs, collar bones) + top-4 influence selection.
      4. Write GLB with Mixamo-named joints + correct IBMs.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_glb": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": (
                            "Original textured GLB to rig. The SOMA-77 "
                            "template's weights transfer onto THIS mesh "
                            "(AABB-aligned torch.cdist nearest-neighbor on "
                            "the RTX 4090, ~10s for 500K verts) and the "
                            "rigged GLB is written directly."
                        ),
                        "forceInput": True,
                    },
                ),
            },
            "optional": {
                "preserve_texture": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "When True (default), the output GLB keeps the "
                            "source mesh's textures. When False, all "
                            "textures are stripped and the mesh is given a "
                            "flat SOMA-blue material — useful when you want "
                            "to apply textures later in Pose Studio or an "
                            "external tool."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("rigged_path",)
    FUNCTION = "export"
    CATEGORY = "Melite/AutoRig"
    OUTPUT_NODE = True

    def export(self, source_glb: str, preserve_texture: bool = True):
        if not source_glb or not source_glb.strip():
            raise ValueError(
                "KimodoExportFBX: source_glb is empty. Provide the GLB to rig."
            )

        import folder_paths  # ComfyUI core
        output_dir = folder_paths.get_output_directory()

        source_glb = self._resolve_path("source_glb", source_glb.strip())

        ts = int(time.time())
        output_path = os.path.join(output_dir, f"3D/KimodoRigged_{ts}.glb")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        _nodes_dir = os.path.dirname(os.path.abspath(__file__))
        import importlib.util
        _mod_path = os.path.join(_nodes_dir, "soma_weight_transfer.py")
        _spec = importlib.util.spec_from_file_location(
            "soma_weight_transfer", _mod_path,
        )
        _gpu_mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_gpu_mod)

        log.info(
            "KimodoExportFBX: SOMA template weight transfer "
            "(source=%s — the pack's skin_standard.npz is the weight source)",
            os.path.basename(source_glb),
        )
        _gpu_mod.transfer_soma_weights_and_write_glb(
            source_path=source_glb,
            output_path=output_path,
            preserve_texture=preserve_texture,
        )
        log.info("KimodoExportFBX: SOMA transfer complete → %s", output_path)

        if not os.path.isfile(output_path):
            raise RuntimeError(
                f"KimodoExportFBX: output not created: {output_path}"
            )

        log.info("KimodoExportFBX: exported %s (%.1f KB)",
                 output_path, os.path.getsize(output_path) / 1024.0)

        filename = os.path.basename(output_path)
        subfolder = os.path.relpath(
            os.path.dirname(output_path), output_dir
        ) if os.path.dirname(output_path) != output_dir else ""

        return {
            "result": (output_path,),
            "ui": {
                "three_model": [
                    {
                        "filename": filename,
                        "subfolder": subfolder,
                        "type": "output",
                    }
                ]
            },
        }

    @staticmethod
    def _resolve_path(label: str, path: str) -> str:
        # Accept absolute paths, ComfyUI input/ filenames, output/
        # filenames (with optional subfolder), or relative paths — the
        # frontend passes a filename from upload_file() without knowing
        # ComfyUI's internal directory layout.
        import folder_paths  # ComfyUI core
        _input_dir = folder_paths.get_input_directory()
        _output_dir = folder_paths.get_output_directory()
        if os.path.isabs(path) and os.path.isfile(path):
            return path
        candidate = os.path.join(_input_dir, path)
        if os.path.isfile(candidate):
            return candidate
        candidate = os.path.join(_output_dir, path)
        if os.path.isfile(candidate):
            return candidate
        for root, _dirs, files in os.walk(_output_dir):
            if path in files:
                return os.path.join(root, path)
        raise FileNotFoundError(
            f"KimodoExportFBX: {label} not found: {path} "
            f"(checked abs, input/{path}, output/{path}, output/**/{path})"
        )


class KimodoTransferWeights:
    """Transfer skin weights from a rigged mesh onto a clothed mesh.

    The character pipeline generates TWO meshes from the same depth/seed:

      1. CLEAN mesh  — near-nude, hairless, gets SOMA-rigged (has weights)
      2. CLOTHED mesh — same body, but with AI-generated clothing + hair

    This node copies the rig from the clean mesh onto the clothed mesh using
    GPU KNN (torch.cdist). The clothed mesh inherits the clean mesh's
    skeleton, joint hierarchy, inverse-bind matrices, and skin weights —
    deforming correctly when animated. No physics simulation needed; the
    clothing is static geometry that rides along with the body.

    Pipeline placement:

      SOMA-rigged clean GLB ─┐
                            ├─→ KimodoTransferWeights → clothed+rigged GLB
      clothed GLB (no rig) ──┘                                 │
                                                                ▼
                                                    SaveRiggedModel / KimodoExportFBX

    Uses AABB-aligned nearest-neighbor on the GPU (~10s for 500K verts) and
    writes the GLB directly — no Blender involved.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "rigged_path": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": (
                            "Path to the CLEAN mesh that has already been "
                            "SOMA-rigged (has JOINTS_0/WEIGHTS_0 + armature). "
                            "The source of the rig + weights."
                        ),
                        "forceInput": True,
                    },
                ),
                "target_path": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": (
                            "Path to the CLOTHED mesh (geometry + textures, "
                            "no rig). Receives the rig from rigged_path."
                        ),
                        "forceInput": True,
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output_path",)
    FUNCTION = "transfer"
    CATEGORY = "Melite/AutoRig"
    OUTPUT_NODE = True

    def transfer(self, rigged_path: str, target_path: str):
        if not rigged_path or not rigged_path.strip():
            raise ValueError(
                "KimodoTransferWeights: rigged_path is empty. "
                "Provide the SOMA-rigged clean mesh path."
            )
        if not target_path or not target_path.strip():
            raise ValueError(
                "KimodoTransferWeights: target_path is empty. "
                "Provide the clothed mesh path."
            )

        rigged_path = rigged_path.strip()
        target_path = target_path.strip()

        # Resolve paths (same logic as KimodoExportFBX — accept absolute,
        # input/, output/, or bare filenames).
        import folder_paths
        _input_dir = folder_paths.get_input_directory()
        _output_dir = folder_paths.get_output_directory()

        def _resolve_path(label: str, path: str) -> str:
            if os.path.isabs(path) and os.path.isfile(path):
                return path
            for d in (_input_dir, _output_dir):
                candidate = os.path.join(d, path)
                if os.path.isfile(candidate):
                    return candidate
            for root, _dirs, files in os.walk(_output_dir):
                if path in files:
                    return os.path.join(root, path)
            raise FileNotFoundError(
                f"KimodoTransferWeights: {label} not found: {path} "
                f"(checked abs, input/{path}, output/{path}, output/**/{path})"
            )

        rigged_path = _resolve_path("rigged_path", rigged_path)
        target_path = _resolve_path("target_path", target_path)

        # ── VRAM yield (F39b: the 14GB Kimodo loader + 4.9GB transfer
        # weights OOM'd a 24GB card even solo — a previous brick's
        # ComfyUI-resident models starve the transfer's cdist tiles).
        # Same law as KimodoLoader and the Voice Studio: evict
        # ComfyUI residents BEFORE the heavy alloc. Loud when the
        # seat is missing — a transfer that cannot yield must not
        # run blind into an OOM traceback.
        try:
            import comfy.model_management as _mm
        except ImportError as exc:
            raise RuntimeError(
                "KimodoTransferWeights: comfy.model_management is "
                "unavailable — the transfer cannot yield VRAM before "
                "its GPU nearest-neighbor pass: %s" % exc
            ) from exc
        _mm.unload_all_models()
        _mm.soft_empty_cache()
        log.info("KimodoTransferWeights: evicted ComfyUI-resident "
                 "models before the transfer")

        # ── Build output path ───────────────────────────────────────────
        output_dir = folder_paths.get_output_directory()
        ts = int(time.time())
        output_filename = f"3D/KimodoClothed_{ts}.glb"
        output_path = os.path.join(output_dir, output_filename)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # ── Run GPU weight transfer ─────────────────────────────────────
        # Reuses gpu_weight_transfer.transfer_weights_and_write_glb —
        # AABB-align + torch.cdist KNN + copy weights by index.
        import importlib.util
        _nodes_dir = os.path.dirname(os.path.abspath(__file__))
        _mod_path = os.path.join(_nodes_dir, "gpu_weight_transfer.py")
        _spec = importlib.util.spec_from_file_location(
            "gpu_weight_transfer", _mod_path,
        )
        _gpu_mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_gpu_mod)

        log.info(
            "KimodoTransferWeights: clean(rigged)=%s → clothed(target)=%s",
            os.path.basename(rigged_path), os.path.basename(target_path),
        )
        t0 = time.time()
        _gpu_mod.transfer_weights_and_write_glb(
            rigged_path=rigged_path,
            source_path=target_path,
            output_path=output_path,
        )
        log.info(
            "KimodoTransferWeights: complete in %.1fs → %s (%.1f KB)",
            time.time() - t0, output_path,
            os.path.getsize(output_path) / 1024.0,
        )

        if not os.path.isfile(output_path):
            raise RuntimeError(
                f"KimodoTransferWeights: output not created: {output_path}"
            )

        filename = os.path.basename(output_path)
        subfolder = os.path.relpath(
            os.path.dirname(output_path), output_dir
        ) if os.path.dirname(output_path) != output_dir else ""

        return {
            "result": (output_path,),
            "ui": {
                "three_model": [
                    {
                        "filename": filename,
                        "subfolder": subfolder,
                        "type": "output",
                    }
                ],
            },
        }


def _resolve_glb(glb_path: str, label: str = "glb_path") -> str:
    """Resolve a GLB path the same way KimodoExportFBX does: accept absolute,
    ComfyUI input/, output/, or bare filename. Returns an absolute path."""
    glb_path = (glb_path or "").strip()
    if not glb_path:
        raise ValueError(f"{label}: path is empty.")
    if os.path.isabs(glb_path) and os.path.isfile(glb_path):
        return glb_path
    import folder_paths
    for d in (folder_paths.get_input_directory(), folder_paths.get_output_directory()):
        cand = os.path.join(d, glb_path)
        if os.path.isfile(cand):
            return cand
    raise FileNotFoundError(
        f"{label}: not found: {glb_path} (checked abs, input/, output/)")


def _import_surgery():
    """Import the weight-surgery functions (package-relative or absolute)."""
    try:
        from .weight_surgery import stage4a2_sync_sibling_weights, stage4a_clean_weights
    except ImportError:
        from weight_surgery import stage4a2_sync_sibling_weights, stage4a_clean_weights
    return stage4a2_sync_sibling_weights, stage4a_clean_weights


class RaySyncSiblingWeights:
    """Unify bone weights of coincident sibling vertices (xatlas UV-seam
    splits) so the mesh can't rip during animation. ComfyUI-node port of the
    host-side stage4a2_sync_sibling_weights — lets the one-flow character
    pipeline run with ZERO host-side deviations. Operates on GLB binary
    (numpy only — no Blender, no GPU)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "rigged_glb_path": (
                    "STRING",
                    {"default": "", "multiline": False,
                     "tooltip": "SOMA-rigged GLB. Coincident UV-seam sibling vertices get unified bone weights.",
                     "forceInput": True,
                     },
                ),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output_path",)
    FUNCTION = "sync"
    CATEGORY = "Melite/AutoRig"
    OUTPUT_NODE = True

    def sync(self, rigged_glb_path: str):
        import folder_paths
        sync_fn, _ = _import_surgery()
        src = _resolve_glb(rigged_glb_path, "rigged_glb_path")
        out_dir = folder_paths.get_output_directory()
        os.makedirs(out_dir, exist_ok=True)
        out_path = sync_fn(src, out_dir)
        return {
            "result": (out_path,),
            "ui": {"three_model": [{
                "filename": os.path.basename(out_path),
                "subfolder": "",
                "type": "output",
            }]},
        }


class RayCleanWeights:
    """Clean SOMAX weight binding errors before animation baking (arm-torso
    isolation, shoulder rebalancing, armpit decontamination, wrist blend zone,
    cross-body decontamination, renormalize). ComfyUI-node port of
    stage4a_clean_weights — eliminates the last host-side stage from the
    one-flow character pipeline. Operates on GLB binary (numpy only)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "rigged_glb_path": (
                    "STRING",
                    {"default": "", "multiline": False,
                     "tooltip": "SOMA-rigged GLB whose weights need cleanup.",
                     "forceInput": True,
                     },
                ),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output_path",)
    FUNCTION = "clean"
    CATEGORY = "Melite/AutoRig"
    OUTPUT_NODE = True

    def clean(self, rigged_glb_path: str):
        import folder_paths
        _, clean_fn = _import_surgery()
        src = _resolve_glb(rigged_glb_path, "rigged_glb_path")
        out_dir = folder_paths.get_output_directory()
        os.makedirs(out_dir, exist_ok=True)
        out_path = clean_fn(src, out_dir)
        return {
            "result": (out_path,),
            "ui": {"three_model": [{
                "filename": os.path.basename(out_path),
                "subfolder": "",
                "type": "output",
            }]},
        }


class SomaNPZToOpenPose:
    """Convert SOMA-77 motion NPZ → OpenPose COCO-18 image for ControlNet.

    Loads a specific frame from a SOMA motion NPZ (from KimodoTextToPose)
    and projects the 77 known 3D joint positions to 2D, drawing the
    OpenPose-format skeleton using the EXACT controlnet_aux color palette.

    This bypasses DWPose estimation entirely — the skeleton is
    mathematically exact (zero pixel drift). ControlNet OpenPose models
    interpret the output natively because the colors + limb connections
    match controlnet_aux exactly.

    Chain: KimodoTextToPose → (STRING npz_path) → SomaNPZToOpenPose
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "npz_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": (
                        "SOMA-77 motion NPZ path (from KimodoTextToPose). "
                        "Connect the STRING output."
                    ),
                    "forceInput": True,
                }),
                "frame_index": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 100000,
                    "tooltip": (
                        "Which frame to render. 0 = first frame. "
                        "Use -1 for the middle frame (peak motion)."
                    ),
                }),
                "img_size": ("INT", {
                    "default": 768,
                    "min": 256,
                    "max": 2048,
                    "step": 64,
                    "tooltip": (
                        "Output image size (square). Match your ControlNet's "
                        "expected resolution."
                    ),
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("openpose_image",)
    FUNCTION = "render"
    CATEGORY = "Melite/AutoRig"
    OUTPUT_NODE = True

    def render(self, npz_path: str, frame_index: int = 0, img_size: int = 768):
        import numpy as np
        import torch
        import folder_paths
        from PIL import Image

        npz_path = (npz_path or "").strip()
        if not npz_path:
            raise ValueError(
                "SomaNPZToOpenPose: npz_path is empty. "
                "Connect the KimodoTextToPose output."
            )

        # Resolve path (absolute, input/, output/, output/**)
        _input_dir = folder_paths.get_input_directory()
        _output_dir = folder_paths.get_output_directory()

        def _resolve_npz(path):
            if os.path.isabs(path) and os.path.isfile(path):
                return path
            for d in (_input_dir, _output_dir):
                candidate = os.path.join(d, path)
                if os.path.isfile(candidate):
                    return candidate
            for root, _dirs, files in os.walk(_output_dir):
                if path in files:
                    return os.path.join(root, path)
            raise FileNotFoundError(
                f"SomaNPZToOpenPose: NPZ not found: {path} "
                f"(checked abs, input/, output/)"
            )

        resolved = _resolve_npz(npz_path)

        # Load NPZ and extract joints for the requested frame
        data = np.load(resolved, allow_pickle=False)
        if "posed_joints" not in data:
            raise ValueError(
                f"SomaNPZToOpenPose: NPZ missing 'posed_joints' key. "
                f"Available: {sorted(data.keys())}"
            )
        posed_joints = np.asarray(data["posed_joints"], dtype=np.float32)  # (T, 77, 3)
        T = posed_joints.shape[0]
        # Negative index wraps from the end (-1 = last frame)
        actual_idx = frame_index if frame_index >= 0 else T + frame_index
        if actual_idx < 0 or actual_idx >= T:
            raise ValueError(
                f"SomaNPZToOpenPose: frame_index {frame_index} out of range "
                f"for {T}-frame NPZ."
            )
        joints_3d = posed_joints[actual_idx]  # (77, 3)

        # Render OpenPose PNG via the existing soma_to_openpose module.
        # Sibling-module import via importlib.util (NOT a flat ``from
        # soma_to_openpose import …``) — at node-EXECUTION time the custom
        # nodes pack dir is no longer on sys.path (ComfyUI removes it after
        # the loading phase), so the flat import raised
        # ``No module named 'soma_to_openpose'`` at run time. Same pattern
        # the file already uses for gpu_weight_transfer / weight_surgery.
        import importlib.util
        _nodes_dir = os.path.dirname(os.path.abspath(__file__))
        _mod_path = os.path.join(_nodes_dir, "soma_to_openpose.py")
        _spec = importlib.util.spec_from_file_location(
            "soma_to_openpose", _mod_path,
        )
        _s2o = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_s2o)
        soma_to_openpose_image = _s2o.soma_to_openpose_image

        out_subdir = os.path.join(_output_dir, "openpose")
        os.makedirs(out_subdir, exist_ok=True)
        out_filename = f"soma_openpose_frame_{actual_idx:04d}.png"
        out_path = os.path.join(out_subdir, out_filename)
        soma_to_openpose_image(joints_3d, out_path, img_size=img_size)

        # Load as ComfyUI IMAGE tensor: (1, H, W, 3) float32 0-1
        img = Image.open(out_path).convert("RGB")
        arr = np.array(img).astype(np.float32) / 255.0
        tensor = torch.from_numpy(arr).unsqueeze(0)

        log.info(
            "SomaNPZToOpenPose: frame %d/%d → %s (%dx%d)",
            actual_idx, T, out_filename, img_size, img_size,
        )

        return {
            "result": (tensor,),
            "ui": {"images": [{
                "filename": out_filename,
                "subfolder": "openpose",
                "type": "output",
            }]},
        }


NODE_CLASS_MAPPINGS = {
    "LoadGLBAsTrimesh": LoadGLBAsTrimesh,
    "SaveRiggedModel": SaveRiggedModel,
    "KimodoExportFBX": KimodoExportFBX,
    "KimodoTransferWeights": KimodoTransferWeights,
    "RaySyncSiblingWeights": RaySyncSiblingWeights,
    "RayCleanWeights": RayCleanWeights,
    "SomaNPZToOpenPose": SomaNPZToOpenPose,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadGLBAsTrimesh": "🔗 Load GLB → TRIMESH",
    "SaveRiggedModel": "💾 Save Rigged Model",
    "KimodoExportFBX": "🎬 Kimodo Export FBX (Retarget)",
    "KimodoTransferWeights": "👕 Kimodo Transfer Weights (Clean → Clothed)",
    "RaySyncSiblingWeights": "🧵 Sync Sibling Weights (UV-seam unification)",
    "RayCleanWeights": "🧹 Clean SOMAX Weights (arm-torso isolation)",
    "SomaNPZToOpenPose": "🦴 SOMA NPZ → OpenPose (ControlNet)",
}

