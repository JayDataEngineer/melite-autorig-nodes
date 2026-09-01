"""Auto-rig bridge + motion-retarget nodes for ComfyUI.

Nodes:

  LoadGLBAsTrimesh — bridges STRING (glb_path) → TRIMESH (trimesh.Trimesh)
  SaveRiggedModel — OUTPUT_NODE that captures the rigged_path STRING and
    returns it with 3D-preview UI data. Without this, SkinTokenRigTrimesh's
    output is invisible to ComfyUI's response (it lacks OUTPUT_NODE=True).
  KimodoExportFBX — takes a rigged character (Mixamo bones, from Step 1 or
    any source) + a SOMA-77 motion NPZ (from KimodoTextToPose) and produces
    an animated FBX/GLB via Blender headless retargeting.
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
import subprocess
import time
import json as _json

log = logging.getLogger(__name__)


# ── MIMO-2.5 visual validation ──────────────────────────────────────────────
# Renders frames from the output GLB and sends them to MIMO-2.5 vision API
# to check for body horror.  The user never discovers bad output manually.

_MIMO_RENDER_SCRIPT = r"""
import bpy, sys, os
glb_path = sys.argv[-3]
out_dir  = sys.argv[-2]
n        = int(sys.argv[-1])
os.makedirs(out_dir, exist_ok=True)
bpy.ops.wm.read_factory_settings(use_empty=True)
s = bpy.context.scene
s.render.engine = 'BLENDER_EEVEE_NEXT'
s.render.resolution_x = 384; s.render.resolution_y = 512
s.eevee.taa_render_samples = 16
bpy.ops.import_scene.gltf(filepath=glb_path)
arm = next((o for o in bpy.data.objects if o.type == 'ARMATURE'), None)
acts = bpy.data.actions
if arm and acts:
    ad = arm.animation_data or arm.animation_data_create()
    ad.action = acts[0]
    fs = max(1, int(acts[0].frame_range[0]))
    fe = max(fs+1, int(acts[0].frame_range[1]))
    step = max(1, (fe - fs) // n)
else:
    fs, fe, step = 0, 0, 1
cd = bpy.data.cameras.new("C"); cd.lens = 50
c = bpy.data.objects.new("C", cd); c.location = (0,-3.5,1.2); c.rotation_euler = (1.4,0,0)
bpy.context.collection.objects.link(c); s.camera = c
for nm,loc,e in [("K",(3,-3,5),800),("F",(-3,-2,3),300)]:
    L = bpy.data.lights.new(nm, type='AREA'); L.energy=e; L.size=5
    o = bpy.data.objects.new(nm, L); o.location = loc
    bpy.context.collection.objects.link(o)
w = bpy.data.worlds.new("W"); w.use_nodes = True
w.node_tree.nodes["Background"].inputs[0].default_value = (0.15,0.15,0.2,1)
w.node_tree.nodes["Background"].inputs[1].default_value = 1.0
s.world = w
for i in range(n):
    f = fs + i*step if arm and acts else 0
    s.frame_set(f); bpy.context.view_layer.update()
    s.render.filepath = os.path.join(out_dir, f"f{i:03d}.png")
    bpy.ops.render.render(write_still=True)
print(f"RENDERED:{n}")
"""

_MIMO_PROMPT = (
    "You are a 3D animation quality reviewer. These frames are from an "
    "animated 3D character. Check for BODY HORROR: limbs bending in "
    "impossible directions, joints dislocated, mesh inside-out, character "
    "melting/collapsing, severe texture stretching.\n\n"
    "Rate each frame 1-10 (10=perfect, 1=unwatchable). "
    "Overall verdict: PASS (>=6) or FAIL.\n"
    "Respond ONLY as JSON: "
    '{"overall_rating": N, "verdict": "PASS|FAIL", '
    '"frame_ratings": [N,...], "issues": "...", "summary": "..."}'
)

# MIMO API credentials — REQUIRED. Load from env (config/secrets.env).
# No hardcoded fallback by design: a missing key must fail loudly, not
# silently use a leaked secret committed to the repo.
_MIMO_API_KEY = os.environ.get("MIMO_API_KEY")
_MIMO_HEADERS = {
    "Authorization": f"Bearer {_MIMO_API_KEY}",
    "Content-Type": "application/json",
    "User-Agent": "Mozilla/5.0",
}


def _mimo_validate(glb_path: str, num_frames: int = 4) -> dict | None:
    """Render frames from GLB and ask MIMO-2.5 if the output is body horror.

    Returns dict with overall_rating, verdict, frame_ratings, issues, summary.
    Returns None on total failure (non-fatal — generation still succeeds).
    """
    import base64

    blender_bin = _resolve_blender_binary()
    script_path = os.path.join(os.path.dirname(glb_path), "_mimo_render.py")
    out_dir = os.path.join(os.path.dirname(glb_path), "_mimo_frames")

    with open(script_path, "w") as f:
        f.write(_MIMO_RENDER_SCRIPT)

    os.makedirs(out_dir, exist_ok=True)
    for old in os.listdir(out_dir):
        old_path = os.path.join(out_dir, old)
        if os.path.isfile(old_path):
            os.remove(old_path)

    cmd = [blender_bin, "--background", "--python", script_path,
           "--", glb_path, out_dir, str(num_frames)]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    if proc.returncode != 0:
        log.warning("MIMO render failed: %s", proc.stderr[-300:])
        return None

    frame_paths = sorted(
        os.path.join(out_dir, f) for f in os.listdir(out_dir) if f.endswith(".png")
    )
    if not frame_paths:
        log.warning("MIMO render produced no frames")
        return None

    content = [{"type": "text", "text": _MIMO_PROMPT}]
    for fp in frame_paths:
        with open(fp, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}"},
        })

    try:
        import requests
        if not _MIMO_API_KEY:
            log.error("MIMO_API_KEY env var not set — cannot validate. See config/secrets.env.example.")
            return None
        resp = requests.post(
            "https://opencode.ai/zen/go/v1/chat/completions",
            headers=_MIMO_HEADERS,
            json={
                "model": "mimo-v2.5",
                "messages": [{"role": "user", "content": content}],
                "max_tokens": 4096,
                "temperature": 0.1,
            },
            timeout=60,
        )
        if resp.status_code != 200:
            log.warning("MIMO API %d: %s", resp.status_code, resp.text[:200])
            return None
        text = resp.json()["choices"][0]["message"]["content"]
        if text is None:
            # MIMO-2.5 is a reasoning model — if max_tokens is too low, all
            # tokens are consumed by reasoning and content is None. Bump
            # the limit and retry so reasoning + content both fit.
            log.warning("MIMO returned None content (reasoning ate all tokens), retrying with higher limit")
            resp = requests.post(
                "https://opencode.ai/zen/go/v1/chat/completions",
                headers=_MIMO_HEADERS,
                json={
                    "model": "mimo-v2.5",
                    "messages": [{"role": "user", "content": content}],
                    "max_tokens": 4096,
                    "temperature": 0.1,
                },
                timeout=90,
            )
            if resp.status_code != 200:
                log.warning("MIMO retry failed %d: %s", resp.status_code, resp.text[:200])
                return None
            text = resp.json()["choices"][0]["message"]["content"]
            if text is None:
                log.warning("MIMO still None content after retry")
                return None
        text = text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()
        return _json.loads(text)
    except Exception as e:
        log.warning("MIMO API call failed: %s", e)
        return None


# ── Blender binary resolution (mirrors SkinToken's _resolve_blender_binary) ──
def _resolve_blender_binary() -> str:
    """Find the Blender headless binary.

    Priority:
      1. SKINTOKEN_BLENDER_BIN env var (shared with SkinToken)
      2. PATH lookup via shutil.which
    """
    env_path = os.environ.get("SKINTOKEN_BLENDER_BIN") or os.environ.get("BLENDER_BIN")
    if env_path and os.path.isfile(env_path):
        return env_path
    discovered = shutil.which("blender")
    if discovered:
        return discovered
    raise FileNotFoundError(
        "Blender binary not found. Set SKINTOKEN_BLENDER_BIN or ensure "
        "'blender' is on PATH."
    )


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
    CATEGORY = "TechNoir/AutoRig"
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
    CATEGORY = "TechNoir/AutoRig"
    OUTPUT_NODE = True

    def save(self, rigged_path: str):
        filename = os.path.basename(rigged_path) if rigged_path else ""
        # Extract subfolder relative to ComfyUI's output directory so /view
        # can find the file. SkinToken saves to output/3D/TrellisAutoRig_*.glb
        # — without the correct subfolder, ComfyUI's /view returns 404 and
        # extract_media silently skips the rigged GLB.
        subdir = ""
        if rigged_path:
            output_base = os.path.join(os.environ.get("COMFYUI_ROOT", "/root/ComfyUI"), "output")
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
    """Apply SOMA-77 motion data onto a rigged character → animated FBX/GLB.

    Takes a rigged character file (GLB or FBX with Mixamo bone names —
    typically the output of SkinTokenRigTrimesh from Step 1) and a SOMA-77
    motion NPZ (from KimodoTextToPose), then retargets the motion onto the
    character's skeleton via Blender headless.

    The retargeting uses position-based direction matching:
      For each bone, compute the rotation that aligns the bone's rest
      direction to the posed direction (derived from NPZ joint positions).
    Root translation is scaled to match the armature's proportions.

    Output: animated FBX (for Unity/Unreal/Kimodo) or GLB (for web/Three.js).
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
                            "Path to a rigged GLB/FBX with Mixamo bone names. "
                            "Connect from SaveRiggedModel or provide a path."
                        ),
                        "forceInput": True,
                    },
                ),
                "fps": (
                    "INT",
                    {
                        "default": 30,
                        "min": 1,
                        "max": 120,
                        "tooltip": "Output animation frame rate.",
                    },
                ),
                "file_format": (
                    ["fbx", "glb"],
                    {
                        "default": "fbx",
                        "tooltip": (
                            "FBX for Unity/Unreal/Kimodo. "
                            "GLB for web/Three.js."
                        ),
                    },
                ),
            },
            "optional": {
                "npz_path": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": (
                            "Optional: SOMA-77 motion NPZ for retargeting. "
                            "If provided, Blender retargets motion onto the "
                            "character. If empty, just converts format."
                        ),
                        "forceInput": True,
                    },
                ),
                "in_place": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "In-place motion: zero root horizontal drift so "
                            "character walks on a treadmill (stays centered). "
                            "Recommended for Pose Studio."
                        ),
                    },
                ),
                "decimate_ratio": (
                    "FLOAT",
                    {
                        "default": 0.2,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": (
                            "Mesh decimation ratio (0=off, 0.2=reduce to 20%% "
                            "of faces). Uses Collapse mode which preserves "
                            "skin weights. Reduces file size ~70%%."
                        ),
                    },
                ),
                "source_glb": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": (
                            "Optional: original textured GLB. When provided "
                            "(and no npz_path), GPU transfers the rig + skin "
                            "weights from rigged_path onto THIS mesh, "
                            "preserving textures. Uses AABB-aligned "
                            "torch.cdist nearest-neighbor on the RTX 4090 "
                            "(~10s for 500K verts) and writes the GLB "
                            "directly — no Blender involved."
                        ),
                        "forceInput": True,
                    },
                ),
                "preserve_texture": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "SOMAX standalone mode only. When True (default), "
                            "the output GLB keeps the source mesh's textures. "
                            "When False, all textures are stripped and the "
                            "mesh is given a flat SOMA-blue material — useful "
                            "when you want to apply textures later in Pose "
                            "Studio or an external tool. Ignored for retarget."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("animated_path",)
    FUNCTION = "export"
    CATEGORY = "TechNoir/AutoRig"
    OUTPUT_NODE = True

    def export(
        self,
        rigged_path: str,
        fps: int,
        file_format: str,
        npz_path: str = "",
        source_glb: str = "",
        in_place: bool = True,
        decimate_ratio: float = 0.2,
        preserve_texture: bool = True,
    ):
        if not rigged_path and not source_glb:
            raise ValueError(
                "KimodoExportFBX: both rigged_path and source_glb are empty. "
                "Provide a GLB path (rigged_path for retarget, or source_glb "
                "for SOMA standalone rigging)."
            )

        rigged_path = rigged_path.strip()
        npz_path = npz_path.strip() if npz_path else ""
        source_glb = source_glb.strip() if source_glb else ""

        # Resolve paths: accept absolute paths, ComfyUI input/ filenames,
        # output/ filenames (with optional subfolder), or relative paths.
        # This lets the frontend pass a filename from upload_file() without
        # knowing ComfyUI's internal directory layout.
        import folder_paths  # ComfyUI core
        _input_dir = folder_paths.get_input_directory()
        _output_dir = folder_paths.get_output_directory()

        def _resolve_path(label: str, path: str) -> str:
            # 1. Absolute path that exists.
            if os.path.isabs(path) and os.path.isfile(path):
                return path
            # 2. Relative to ComfyUI input/ (uploaded files).
            candidate = os.path.join(_input_dir, path)
            if os.path.isfile(candidate):
                return candidate
            # 3. Relative to ComfyUI output/ (generated files, e.g. "3D/foo.glb").
            candidate = os.path.join(_output_dir, path)
            if os.path.isfile(candidate):
                return candidate
            # 4. Bare filename in output/ (no subfolder).
            for root, _dirs, files in os.walk(_output_dir):
                if path in files:
                    return os.path.join(root, path)
            raise FileNotFoundError(
                f"KimodoExportFBX: {label} not found: {path} "
                f"(checked abs, input/{path}, output/{path}, output/**/{path})"
            )

        # rigged_path may be empty in SOMA standalone mode (source_glb only).
        if rigged_path:
            rigged_path = _resolve_path("rigged_path", rigged_path)
        if npz_path:
            npz_path = _resolve_path("npz_path", npz_path)
        if source_glb:
            source_glb = _resolve_path("source_glb", source_glb)

        # ── Resolve Blender binary ──────────────────────────────────────
        blender_bin = _resolve_blender_binary()
        log.info("KimodoExportFBX: blender=%s", blender_bin)

        # ── Choose action based on inputs ───────────────────────────────
        # Priority: rig_and_retarget (source+NPZ) > retarget (NPZ) >
        #           apply_weights (source_glb) > convert
        _nodes_dir = os.path.dirname(os.path.abspath(__file__))
        if source_glb and npz_path:
            action = "rig_and_retarget"
        elif npz_path:
            action = "retarget"
        elif source_glb:
            action = "apply_weights"
        else:
            action = "convert"

        # ── Build output path ───────────────────────────────────────────
        import folder_paths  # ComfyUI core
        output_dir = folder_paths.get_output_directory()
        ext = "fbx" if file_format.lower() == "fbx" else "glb"
        ts = int(time.time())
        animated_filename = f"3D/KimodoAnimated_{ts}.{ext}"
        output_path = os.path.join(output_dir, animated_filename)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # ── Combined mode: SOMA rig THEN retarget motion in one pass ───────
        # When the user provides BOTH a raw GLB and a motion NPZ, we rig
        # the mesh first (SOMA weight transfer, ~2s) then immediately
        # retarget the motion via Blender. This is the ONE-CLICK path:
        # TRELLIS GLB + motion NPZ → animated character.
        if action == "rig_and_retarget":
            import importlib.util
            _mod_path = os.path.join(_nodes_dir, "soma_weight_transfer.py")
            _spec = importlib.util.spec_from_file_location(
                "soma_weight_transfer", _mod_path,
            )
            _gpu_mod = importlib.util.module_from_spec(_spec)
            _spec.loader.exec_module(_gpu_mod)

            temp_rigged = os.path.join(
                os.path.dirname(output_path), f"_soma_temp_{ts}.glb",
            )
            log.info(
                "KimodoExportFBX: combined mode — SOMA rig → %s, then retarget",
                os.path.basename(temp_rigged),
            )
            _gpu_mod.transfer_soma_weights_and_write_glb(
                source_path=source_glb, output_path=temp_rigged,
                preserve_texture=preserve_texture,
            )
            log.info("KimodoExportFBX: SOMA rig complete, starting retarget")

            rigged_path = temp_rigged
            action = "retarget"
            # Fall through to the retarget logic below

        # ── SOMA template weight transfer (standalone rig mode) ───────────
        # When source_glb is provided WITHOUT a pre-rigged GLB, we use the
        # SOMA-77 template body (skin_standard.npz — a pre-rigged human with
        # hand-painted LBS weights) as the weight source. This REPLACES the
        # broken SkinToken approach.
        #
        # Pipeline (NVIDIA SOMA-X SOTA):
        #   1. AABB-align SOMA template mesh into source mesh space
        #      (uniform scale via height ratio + axis permutation)
        #   2. BarycentricInterpolator: for each source vertex, find its
        #      containing triangle in the T_align'd SOMA template mesh.
        #      Interpolate the template's hand-painted weights via the
        #      triangle's barycentric coords. This preserves the template's
        #      anatomically-correct weight boundaries EXACTLY — structurally
        #      eliminating the torso/arm "wing" artifact that bone-heat
        #      smoothing caused on raised-arm motions.
        #   3. Cross-body decontamination (prune L↔R contamination on
        #      inner thighs, collar bones) + top-4 influence selection.
        #   4. Write GLB with Mixamo-named joints + correct IBMs.
        if action == "apply_weights":
            import importlib.util
            _mod_path = os.path.join(_nodes_dir, "soma_weight_transfer.py")
            _spec = importlib.util.spec_from_file_location(
                "soma_weight_transfer", _mod_path,
            )
            _gpu_mod = importlib.util.module_from_spec(_spec)
            _spec.loader.exec_module(_gpu_mod)

            # If FBX requested, write GLB first then convert
            glb_output = output_path
            if file_format.lower() == "fbx":
                glb_output = output_path.replace(".fbx", ".glb")

            log.info(
                "KimodoExportFBX: SOMA template weight transfer "
                "(source=%s, no SkinToken — uses /opt/kimodo/.../skin_standard.npz)",
                os.path.basename(source_glb),
            )
            # NEW API: takes only source_path + output_path (no rigged_path
            # needed — the SOMA template at /opt/kimodo is the weight source).
            _gpu_mod.transfer_soma_weights_and_write_glb(
                source_path=source_glb,
                output_path=glb_output,
                preserve_texture=preserve_texture,
            )
            log.info("KimodoExportFBX: SOMA transfer complete → %s", glb_output)

            if file_format.lower() == "glb":
                # GLB already written — no Blender needed!
                output_path = glb_output
            else:
                # FBX: convert the GLB via Blender (format conversion only,
                # no weight computation)
                script_path = os.path.join(_nodes_dir, "blender_convert.py")
                cmd = [
                    blender_bin, "--background", "--python", script_path,
                    "--", glb_output, output_path, "fbx",
                ]
                log.info("KimodoExportFBX: FBX convert: %s", " ".join(cmd))
                result = subprocess.run(cmd, capture_output=True,
                                        text=True, timeout=600)
                if result.returncode != 0:
                    raise RuntimeError(
                        f"KimodoExportFBX: FBX conversion failed "
                        f"(exit {result.returncode}). "
                        f"stderr: {result.stderr[-500:]}"
                    )
                # Clean up temp GLB
                if os.path.isfile(glb_output) and glb_output != output_path:
                    os.remove(glb_output)

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

        # ── Retarget / Convert: run Blender headless ────────────────────
        if action == "retarget":
            script_path = os.path.join(_nodes_dir, "blender_retarget.py")
        else:
            script_path = os.path.join(_nodes_dir, "blender_convert.py")
        if not os.path.isfile(script_path):
            raise FileNotFoundError(
                f"KimodoExportFBX: {action} script not found: {script_path}"
            )

        if action == "retarget":
            cmd = [
                blender_bin,
                "--background",
                "--python",
                script_path,
                "--",
                rigged_path,
                npz_path,
                output_path,
                str(fps),
                file_format.lower(),
                str(in_place),
                str(decimate_ratio),
            ]
        else:
            cmd = [
                blender_bin,
                "--background",
                "--python",
                script_path,
                "--",
                rigged_path,
                output_path,
                file_format.lower(),
            ]
        log.info("KimodoExportFBX: running %s: %s", action, " ".join(cmd))

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 min max
        )

        # Blender prints to both stdout and stderr; log for debugging.
        if result.stdout:
            for line in result.stdout.strip().splitlines():
                if line.strip():
                    log.info("blender: %s", line.rstrip())
        if result.stderr:
            for line in result.stderr.strip().splitlines():
                stripped = line.rstrip()
                if any(kw in stripped.lower() for kw in
                       ("error", "traceback", "exception", "failed")):
                    log.error("blender: %s", stripped)
                elif stripped.startswith("Blender"):
                    log.info("blender: %s", stripped)
                else:
                    log.debug("blender stderr: %s", stripped)

        if result.returncode != 0:
            raise RuntimeError(
                f"KimodoExportFBX: Blender failed (exit {result.returncode}). "
                f"Last stderr: {result.stderr[-500:] if result.stderr else '(empty)'}"
            )

        if not os.path.isfile(output_path):
            raise RuntimeError(
                f"KimodoExportFBX: output file not created: {output_path}. "
                f"Blender stdout: {result.stdout[-500:]}"
            )

        log.info(
            "KimodoExportFBX: exported %s (%.1f KB)",
            output_path,
            os.path.getsize(output_path) / 1024.0,
        )

        filename = os.path.basename(output_path)
        subfolder = os.path.relpath(
            os.path.dirname(output_path), output_dir
        ) if os.path.dirname(output_path) != output_dir else ""

        ui_data = {
            "three_model": [
                {
                    "filename": filename,
                    "subfolder": subfolder,
                    "type": "output",
                }
            ],
        }

        # ── MIMO-2.5 visual validation ─────────────────────────────────
        # Render frames and send to MIMO-2.5 vision API to check for body
        # horror. The user NEVER has to discover bad output — MIMO catches
        # it first. Runs in <30s (4 frames EEVEE + API call).
        # NOTE: ComfyUI serializes UI dicts to list(keys), so we JSON-encode
        # the validation result as a string to survive the round-trip.
        try:
            validation = _mimo_validate(output_path)
            if validation:
                # ComfyUI converts dict→list(keys) and str→list(chars).
                # Wrap in a list to survive: ComfyUI preserves list-of-dict
                # (see three_model above). Server unwraps on the other side.
                ui_data["mimo_validation"] = [validation]
                verdict = validation.get("verdict", "UNKNOWN")
                rating = validation.get("overall_rating", -1)
                log.info(
                    "KimodoExportFBX: MIMO-2.5 verdict=%s rating=%s",
                    verdict, rating,
                )
        except Exception as ve:
            log.warning("MIMO validation failed (non-fatal): %s", ve)

        return {
            "result": (output_path,),
            "ui": ui_data,
        }


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
                                                    KimodoExportFBX (retarget)

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
    CATEGORY = "TechNoir/AutoRig"
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
    CATEGORY = "TechNoir/AutoRig"
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
    CATEGORY = "TechNoir/AutoRig"
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
    CATEGORY = "TechNoir/AutoRig"
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

