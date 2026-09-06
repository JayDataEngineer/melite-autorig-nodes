"""SOMA template skin-weight transfer — REPLACES SkinToken.

Pipeline:
  1. Load SOMA-77 standard template body (18k verts, hand-painted LBS weights)
  2. Load TRELLIS source mesh (textured)
  3. BARYCENTRIC weight transfer: for each TRELLIS vert, find its containing
     triangle in the T_align'd SOMA template mesh and interpolate the
     template's hand-painted weights via the triangle's barycentric coords
  4. Build output GLB: source geometry + textures + SOMA-derived skin +
     Mixamo-named bone hierarchy with correct IBMs

The SOMA template at /opt/kimodo/.../somaskel77/skin_standard.npz contains
hand-painted vertex weights for a standard human body. Barycentric
interpolation transfers these weights EXACTLY at the template's
anatomically-correct boundaries (no bone-heat smoothing across the
torso/arm divide), eliminating the "wing" artifact on raised-arm motions.

This is the NVIDIA SOMA-X SOTA approach: py-soma-x's BarycentricInterpolator
(soma.geometry.barycentric_interp) computes the correspondence between the
template mesh and any destination mesh in the same pose.
"""
from __future__ import annotations

import json
import struct
import time
import logging
import os
from typing import Any

import numpy as np

log = logging.getLogger("soma_weight_transfer")
if not log.handlers:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(name)s] %(message)s")

SOMA_SKIN_PATH = "/opt/kimodo/kimodo/assets/skeletons/somaskel77/skin_standard.npz"

# The somaskel77 UV layout — per-vertex (u, v) of the SAME 18056-vert
# template, lifted once from the estate's somax body export (the game's
# base body GLB shares the template topology and carries its atlas UVs,
# which span [-1.86, 5.18] — tiled/repeat, NOT unit-square; never clamp).
# Transferring THESE onto the rigged source keeps one-texture-fits-all:
# any body that leaves this bridge is paintable by the somax skin maps
# (SkinApply refuses bodies without TEXCOORD_0 — caught live 2026-09-06
# when the character flow's coat stage met the ANNY body: POSITION only).
SOMA_UV_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "somaskel77_uv.npy")

# ═══════════════════════════════════════════════════════════════════════════
# CANONICAL POSE REFERENCE — see docs/CANONICAL-SOMAX-POSE.md
# ═══════════════════════════════════════════════════════════════════════════
# The template mesh in skin_standard.npz is in a NATURAL REST POSE, not a
# textbook A-pose. Measured from bind_rig_transform (Y-up, meters):
#     Upper arm drop below horizontal: 51.2° (L) / 51.4° (R)
#     Forearm drop below horizontal:   35.4° (L) / 35.3° (R)
#     Elbow bend (deviation straight): 25.2° (L) / 25.5° (R)
#     Total mesh height: 176.1 cm, arm span: 123.9 cm
#
# The region-constrained correspondence below assumes the source mesh is in
# (or can be aligned to) THIS pose. The old SOMA77_APOSE (45° drop, 0° bend)
# was WRONG and caused body horror at armpits/hands/jaw. Do not use it.
# ═══════════════════════════════════════════════════════════════════════════

# SOMA-77 index → Mixamo short-name (without "mixamorig:" prefix).
# Mirrors blender_retarget.py's SOMA_TO_MIXAMO. Bones NOT in this map are
# emitted as-is (e.g. eyes, jaw, fingers after 3rd joint) but the retarget
# script will skip them if absent.
SOMA_IDX_TO_MIXAMO_NAME: dict[int, str] = {
    0:  "Hips",
    1:  "Spine",
    2:  "Spine1",
    3:  "Spine2",
    4:  "Neck",
    5:  "Neck1",        # SOMA Neck2 → Mixamo Neck1 (extra neck bone, OK)
    6:  "Head",
    7:  "HeadTop_End",  # SOMA HeadEnd → Mixamo HeadTop_End
    8:  "Jaw",
    9:  "LeftEye",
    10: "RightEye",
    11: "LeftShoulder",
    12: "LeftArm",
    13: "LeftForeArm",
    14: "LeftHand",
    # Left thumb
    15: "LeftHandThumb1", 16: "LeftHandThumb2",
    17: "LeftHandThumb3", 18: "LeftHandThumb4",
    # Left index
    19: "LeftHandIndex1", 20: "LeftHandIndex2",
    21: "LeftHandIndex3", 22: "LeftHandIndex4",
    # Left middle
    24: "LeftHandMiddle1", 25: "LeftHandMiddle2",
    26: "LeftHandMiddle3", 27: "LeftHandMiddle4",
    # Left ring
    29: "LeftHandRing1", 30: "LeftHandRing2",
    31: "LeftHandRing3", 32: "LeftHandRing4",
    # Left pinky
    34: "LeftHandPinky1", 35: "LeftHandPinky2",
    36: "LeftHandPinky3", 37: "LeftHandPinky4",
    39: "RightShoulder",
    40: "RightArm",
    41: "RightForeArm",
    42: "RightHand",
    43: "RightHandThumb1", 44: "RightHandThumb2",
    45: "RightHandThumb3", 46: "RightHandThumb4",
    47: "RightHandIndex1", 48: "RightHandIndex2",
    49: "RightHandIndex3", 50: "RightHandIndex4",
    52: "RightHandMiddle1", 53: "RightHandMiddle2",
    54: "RightHandMiddle3", 55: "RightHandMiddle4",
    57: "RightHandRing1", 58: "RightHandRing2",
    59: "RightHandRing3", 60: "RightHandRing4",
    62: "RightHandPinky1", 63: "RightHandPinky2",
    64: "RightHandPinky3", 65: "RightHandPinky4",
    67: "LeftUpLeg",
    68: "LeftLeg",
    69: "LeftFoot",
    70: "LeftToeBase",
    71: "LeftToe_End",
    72: "RightUpLeg",
    73: "RightLeg",
    74: "RightFoot",
    75: "RightToeBase",
    76: "RightToe_End",
}

# Mixamo parent map (short names, no "mixamorig:" prefix).
# Same as blender_retarget.py expects.
MIXAMO_PARENT_OF: dict[str, str | None] = {
    "Hips": None,
    "Spine": "Hips",
    "Spine1": "Spine",
    "Spine2": "Spine1",
    "LeftShoulder": "Spine2",
    "LeftArm": "LeftShoulder",
    "LeftForeArm": "LeftArm",
    "LeftHand": "LeftForeArm",
    "RightShoulder": "Spine2",
    "RightArm": "RightShoulder",
    "RightForeArm": "RightArm",
    "RightHand": "RightForeArm",
    "Neck": "Spine2",
    "Neck1": "Neck",
    "Head": "Neck1",
    "HeadTop_End": "Head",
    "Jaw": "Head",
    "LeftEye": "Head",
    "RightEye": "Head",
    "LeftUpLeg": "Hips",
    "LeftLeg": "LeftUpLeg",
    "LeftFoot": "LeftLeg",
    "LeftToeBase": "LeftFoot",
    "LeftToe_End": "LeftToeBase",
    "RightUpLeg": "Hips",
    "RightLeg": "RightUpLeg",
    "RightFoot": "RightLeg",
    "RightToeBase": "RightFoot",
    "RightToe_End": "RightToeBase",
    # Fingers
    "LeftHandThumb1": "LeftHand",  "LeftHandThumb2": "LeftHandThumb1",
    "LeftHandThumb3": "LeftHandThumb2", "LeftHandThumb4": "LeftHandThumb3",
    "LeftHandIndex1": "LeftHand",  "LeftHandIndex2": "LeftHandIndex1",
    "LeftHandIndex3": "LeftHandIndex2", "LeftHandIndex4": "LeftHandIndex3",
    "LeftHandMiddle1": "LeftHand", "LeftHandMiddle2": "LeftHandMiddle1",
    "LeftHandMiddle3": "LeftHandMiddle2", "LeftHandMiddle4": "LeftHandMiddle3",
    "LeftHandRing1": "LeftHand",   "LeftHandRing2": "LeftHandRing1",
    "LeftHandRing3": "LeftHandRing2",   "LeftHandRing4": "LeftHandRing3",
    "LeftHandPinky1": "LeftHand",  "LeftHandPinky2": "LeftHandPinky1",
    "LeftHandPinky3": "LeftHandPinky2",  "LeftHandPinky4": "LeftHandPinky3",
    "RightHandThumb1": "RightHand",  "RightHandThumb2": "RightHandThumb1",
    "RightHandThumb3": "RightHandThumb2", "RightHandThumb4": "RightHandThumb3",
    "RightHandIndex1": "RightHand",  "RightHandIndex2": "RightHandIndex1",
    "RightHandIndex3": "RightHandIndex2", "RightHandIndex4": "RightHandIndex3",
    "RightHandMiddle1": "RightHand", "RightHandMiddle2": "RightHandMiddle1",
    "RightHandMiddle3": "RightHandMiddle2", "RightHandMiddle4": "RightHandMiddle3",
    "RightHandRing1": "RightHand",   "RightHandRing2": "RightHandRing1",
    "RightHandRing3": "RightHandRing2",   "RightHandRing4": "RightHandRing3",
    "RightHandPinky1": "RightHand",  "RightHandPinky2": "RightHandPinky1",
    "RightHandPinky3": "RightHandPinky2",  "RightHandPinky4": "RightHandPinky3",
}

# SOMA-77 parent index array (mirrors blender_retarget.py SOMA77_PARENTS).
SOMA77_PARENTS = (
    None,                                          # 0 Hips
    0,                                             # 1 Spine1
    1,                                             # 2 Spine2
    2,                                             # 3 Chest
    3,                                             # 4 Neck1
    4,                                             # 5 Neck2
    5,                                             # 6 Head
    6,                                             # 7 HeadEnd
    6,                                             # 8 Jaw
    6, 6,                                          # 9,10 LeftEye, RightEye
    3,                                             # 11 LeftShoulder
    11, 12, 13,                                    # 12,13,14 LeftArm/ForeArm/Hand
    14, 15, 16, 17,                                # 15-18 LeftHandThumb1..End
    14, 19, 20, 21, 22,                            # 19-23 LeftHandIndex1..End
    14, 24, 25, 26, 27,                            # 24-28 LeftHandMiddle1..End
    14, 29, 30, 31, 32,                            # 29-33 LeftHandRing1..End
    14, 34, 35, 36, 37,                            # 34-38 LeftHandPinky1..End
    3,                                             # 39 RightShoulder
    39, 40, 41,                                    # 40,41,42 RightArm/ForeArm/Hand
    42, 43, 44, 45,                                # 43-46 RightHandThumb1..End
    42, 47, 48, 49, 50,                            # 47-51 RightHandIndex1..End
    42, 52, 53, 54, 55,                            # 52-56 RightHandMiddle1..End
    42, 57, 58, 59, 60,                            # 57-61 RightHandRing1..End
    42, 62, 63, 64, 65,                            # 62-66 RightHandPinky1..End
    0, 67, 68, 69, 70,                             # 67-71 LeftLeg/Shin/Foot/Toe
    0, 72, 73, 74, 75,                             # 72-76 RightLeg/Shin/Foot/Toe
)


# ═══════════════════════════════════════════════════════════════════════════
# GLB I/O helpers
# ═══════════════════════════════════════════════════════════════════════════

def _read_glb(path: str) -> tuple[dict, bytes]:
    """Parse GLB into (gltf_json, binary_buffer)."""
    with open(path, "rb") as f:
        header = f.read(12)
        magic, version, length = struct.unpack("<III", header)
        assert magic == 0x46546C67, f"Bad GLB magic 0x{magic:x}"
        # Read chunks
        bin_data = bytearray()
        offset = 12
        while offset < length:
            clen, ctype = struct.unpack("<II", f.read(8))
            chunk = f.read(clen)
            if ctype == 0x4E4F534A:  # JSON
                gltf = json.loads(chunk.decode("utf-8"))
            elif ctype == 0x004E4942:  # BIN
                bin_data.extend(chunk)
            offset += 8 + clen
        return gltf, bytes(bin_data)


def _read_accessor(gltf: dict, bin_data: bytes, idx: int) -> np.ndarray:
    """Read a glTF accessor as a numpy array."""
    acc = gltf["accessors"][idx]
    bv = gltf["bufferViews"][acc["bufferView"]]
    offset = (bv.get("byteOffset", 0) + acc.get("byteOffset", 0))
    count = acc["count"]
    atype = acc["type"]
    comp = acc["componentType"]
    n = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}[atype]
    dtypes = {5120: np.int8, 5121: np.uint8, 5122: np.int16,
              5123: np.uint16, 5125: np.uint32, 5126: np.float32,
              5127: np.float32}  # 5127 isn't standard but safe
    dt = dtypes[comp]
    arr = np.frombuffer(bin_data, dtype=dt, count=count * n, offset=offset)
    return arr.reshape(count, n) if n > 1 else arr.reshape(count)


# ═══════════════════════════════════════════════════════════════════════════
# Alignment
# ═══════════════════════════════════════════════════════════════════════════

def _aabb_align(src_pts: np.ndarray, tgt_pts: np.ndarray) -> np.ndarray:
    """Compute 4x4 transform mapping src_pts into tgt_pts's AABB.

    Uses bounding-box centering + PER-AXIS scaling so different body
    proportions (A-pose vs T-pose) still map head-to-head, arm-to-arm.
    Also permutes axes so src's "up" (largest range) aligns with tgt's up.
    Returns (4,4) homogeneous transform.
    """
    s_min, s_max = src_pts.min(axis=0), src_pts.max(axis=0)
    t_min, t_max = tgt_pts.min(axis=0), tgt_pts.max(axis=0)
    s_center = (s_min + s_max) / 2
    t_center = (t_min + t_max) / 2
    s_size = np.maximum(s_max - s_min, 1e-8)
    t_size = np.maximum(t_max - t_min, 1e-8)

    # Axis permutation: rotate src so its "up" (largest range) aligns with
    # tgt's "up". For human meshes, largest range = height = "up".
    s_axis_order = np.argsort(-s_size)  # largest first
    t_axis_order = np.argsort(-t_size)
    P = np.zeros((3, 3), dtype=np.float32)
    for i in range(3):
        P[t_axis_order[i], s_axis_order[i]] = 1.0

    # CRITICAL: A pure axis-swap permutation has det=-1 (reflection).
    # Force a proper rotation (det=+1) by flipping the smallest-magnitude axis.
    if np.linalg.det(P) < 0:
        smallest_src_axis = s_axis_order[-1]
        smallest_tgt_axis = t_axis_order[-1]
        P[smallest_tgt_axis, smallest_src_axis] = -1.0

    # Per-axis scale: scale src's permuted axes to match tgt's axes.
    # s_permuted_size[i] = s_size[s_axis_order[i]]
    # tgt wants t_size[t_axis_order[i]]
    s_perm_size = s_size[s_axis_order]
    t_perm_size = t_size[t_axis_order]
    per_axis_scale = t_perm_size / np.maximum(s_perm_size, 1e-8)
    # Apply scale as a diagonal matrix in the PERMUTED space, then un-permute
    S_perm = np.diag(per_axis_scale).astype(np.float32)
    P @ S_perm @ P.T  # rotate→scale→rotate back (but P.T=P^-1 since P orthogonal)
    # Actually: we want T_final = scale_src_after_perm * permute
    # T = P @ S where S is diag of per_axis_scale in src axis order
    # Wait let me think again. We want:
    #   output = P @ (per_axis_scale * src)
    # So the rotation-scale matrix is P @ diag(per_axis_scale)
    # But per_axis_scale is indexed by [s_axis_order], which is the scale
    # applied to source axes AFTER permutation. So we need:
    #   output_axis[i] = per_axis_scale[i] * src[s_axis_order[i]]
    # In matrix form: output = P_unscale @ src, where:
    #   P_unscale[i, s_axis_order[i]] = per_axis_scale[i]
    P_scale = np.zeros((3, 3), dtype=np.float32)
    for i in range(3):
        P_scale[i, s_axis_order[i]] = per_axis_scale[i]
    # Apply the sign flip too
    if np.linalg.det(P) < 0:
        # Adjust P_scale's last row sign
        P_scale[-1, s_axis_order[-1]] *= -1.0
    # Wait we already adjusted P separately. Let me redo this cleanly:
    # The full transform: out = T_final @ src where T_final has:
    #   - Axis permutation (with sign flip for proper rotation)
    #   - Per-axis scaling
    # Build as: T_final = P @ diag(s_perm_scale) @ reorder_back_to_src
    # Simpler: directly construct T_final so that
    #   out[t_axis_order[i]] = scale[i] * src[s_axis_order[i]]
    T_final = np.zeros((3, 3), dtype=np.float32)
    for i in range(3):
        T_final[t_axis_order[i], s_axis_order[i]] = per_axis_scale[i]
    # Apply sign flip if needed for proper rotation
    if np.linalg.det(T_final) < 0:
        T_final[t_axis_order[-1], s_axis_order[-1]] *= -1.0

    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = T_final
    T[:3, 3] = t_center - T_final @ s_center
    return T


def _aabb_align_uniform(src_pts: np.ndarray, tgt_pts: np.ndarray) -> np.ndarray:
    """Compute 4x4 transform mapping src_pts into tgt_pts's AABB.

    Like _aabb_align but uses UNIFORM scaling (height ratio only) instead
    of per-axis scaling. This avoids creating non-uniform bone scale in the
    root joint, which Blender's glTF importer cannot handle.

    Returns (4,4) homogeneous transform with uniform scale factor.
    """
    s_min, s_max = src_pts.min(axis=0), src_pts.max(axis=0)
    t_min, t_max = tgt_pts.min(axis=0), tgt_pts.max(axis=0)
    s_center = (s_min + s_max) / 2
    t_center = (t_min + t_max) / 2
    s_size = np.maximum(s_max - s_min, 1e-8)
    t_size = np.maximum(t_max - t_min, 1e-8)

    # Axis permutation: align src and tgt "up" axes.
    # For standing characters, the "depth" axis (Z in glTF) is always the
    # thinnest. The "height" axis is Y (glTF standard). But for T-pose
    # characters, the arm span (X) can be LARGER than the height (Y),
    # which breaks the naive "largest extent = height" heuristic.
    # Fix: use Y as the height axis when Z is the thinnest for both meshes
    # (standard Y-up glTF). Only permute when the up axes genuinely differ.
    s_thinnest = int(np.argmin(s_size))  # depth axis for src
    t_thinnest = int(np.argmin(t_size))  # depth axis for tgt

    if s_thinnest == t_thinnest:
        # Both meshes have the same depth axis → same orientation.
        # Y is height, no permutation needed (identity).
        P = np.eye(3, dtype=np.float32)
        s_height_axis = 1  # Y
        t_height_axis = 1  # Y
        # Handle Z-up case: if both have Z as a major axis and Y as thinnest
        if s_thinnest == 1 and t_thinnest == 1:
            s_height_axis = 2  # Z-up
            t_height_axis = 2
            P = np.eye(3, dtype=np.float32)
    else:
        # Different depth axes → need permutation (e.g., Z-up → Y-up)
        s_axis_order = np.argsort(-s_size)  # largest first
        t_axis_order = np.argsort(-t_size)
        P = np.zeros((3, 3), dtype=np.float32)
        for i in range(3):
            P[t_axis_order[i], s_axis_order[i]] = 1.0
        # CRITICAL: Force proper rotation (det=+1)
        if np.linalg.det(P) < 0:
            P[t_axis_order[-1], s_axis_order[-1]] = -1.0
        s_height_axis = s_axis_order[0]
        t_height_axis = t_axis_order[0]

    # UNIFORM scale: use height ratio only
    s_height = s_size[s_height_axis]
    t_height = t_size[t_height_axis]
    uniform_scale = t_height / max(s_height, 1e-8)

    # Final 3x3 = permutation @ uniform_scale
    T_final = P * uniform_scale  # P is orthogonal, so P * s = P @ diag(s,s,s)

    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = T_final
    T[:3, 3] = t_center - T_final @ s_center
    return T


def _aabb_align_per_axis_safe(src_pts: np.ndarray, tgt_pts: np.ndarray) -> np.ndarray:
    """Per-axis AABB alignment with smart thinnest-axis detection.

    Combines the robust axis detection of ``_aabb_align_uniform`` (correctly
    handles A-pose/T-pose characters where arm span may exceed height) with
    **per-axis scaling** so that different body proportions — shoulder width,
    arm length, torso depth — all match independently.

    This is the correct alignment for barycentric weight correspondence:

    - A source vertex at the shoulder tip maps to the template's shoulder
      region, not the upper arm (which happens with uniform scale when
      the source has wider shoulders than the template).
    - A source vertex at the hand maps to the template's hand region, not
      the forearm (which happens when the source has longer arms).

    The uniform-scale approach (height ratio only) compresses or expands
    ALL axes equally, causing proportional mismatch on non-average bodies
    (children, muscular males, anime females, etc.).

    Note: This produces non-uniform scale in the 3×3 matrix and should
    ONLY be used for vertex correspondence (barycentric transfer), NOT for
    bind matrix transforms (which need uniform scale for Blender compat).
    """
    s_min, s_max = src_pts.min(axis=0), src_pts.max(axis=0)
    t_min, t_max = tgt_pts.min(axis=0), tgt_pts.max(axis=0)
    s_center = (s_min + s_max) / 2
    t_center = (t_min + t_max) / 2
    s_size = np.maximum(s_max - s_min, 1e-8)
    t_size = np.maximum(t_max - t_min, 1e-8)

    # ── Smart axis detection (same logic as _aabb_align_uniform) ───────
    # Use the thinnest axis (depth) to determine orientation. For standing
    # characters this avoids misidentifying arm span as height.
    s_thinnest = int(np.argmin(s_size))
    t_thinnest = int(np.argmin(t_size))

    if s_thinnest == t_thinnest:
        # Same orientation — identity permutation, per-axis scale
        per_axis_scale = (t_size / np.maximum(s_size, 1e-8)).astype(np.float32)
        T_final = np.diag(per_axis_scale).astype(np.float32)
    else:
        # Different orientation (e.g., Z-up → Y-up) — permute + per-axis scale
        s_axis_order = np.argsort(-s_size)  # largest first
        t_axis_order = np.argsort(-t_size)
        # Per-axis scale in permuted space
        s_perm_size = s_size[s_axis_order]
        t_perm_size = t_size[t_axis_order]
        per_axis_scale_perm = (t_perm_size / np.maximum(s_perm_size, 1e-8)).astype(np.float32)
        # Build T_final: out[t_axis_order[i]] = scale[i] * src[s_axis_order[i]]
        T_final = np.zeros((3, 3), dtype=np.float32)
        for i in range(3):
            T_final[t_axis_order[i], s_axis_order[i]] = per_axis_scale_perm[i]
        # Force proper rotation (det=+1)
        if np.linalg.det(T_final) < 0:
            T_final[t_axis_order[-1], s_axis_order[-1]] *= -1.0

    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = T_final
    T[:3, 3] = t_center - T_final @ s_center
    return T


# ═══════════════════════════════════════════════════════════════════════════
# Main entry point
# ═══════════════════════════════════════════════════════════════════════════

def transfer_soma_weights_and_write_glb(
    source_path: str,
    output_path: str,
    max_influence_per_vertex: int = 4,
    preserve_texture: bool = True,
) -> str:
    """Transfer SOMA template weights onto the source mesh, write GLB.

    Uses NVIDIA SOMA-X's BarycentricInterpolator to transfer the SOMA-77
    template's hand-painted skin weights onto the TRELLIS source mesh via
    barycentric coordinate interpolation. This preserves the template's
    anatomically-correct weight boundaries exactly — structurally
    eliminating the torso/arm "wing" artifact that bone-heat smoothing
    caused on raised-arm motions (wave, jumping_jacks).

    Args:
        source_path: TRELLIS GLB (raw mesh, has POSITION + normals + texcoord
            + material/texture, but NO skin weights). Must be in A-pose
            (arms relaxed ~38° below horizontal) to match the SOMA template.
        output_path: where to write the rigged GLB.
        max_influence_per_vertex: cap on joints per vertex (glTF JOINTS_0 is
            VEC4, so default 4).
        preserve_texture: when True (default), keep the source mesh's textures
            in the output. When False, strip all texture references from
            materials and apply a flat SOMA-blue baseColorFactor — gives the
            classic "blue mannequin" look so textures can be re-applied later
            in Pose Studio or an external tool.

    Returns output_path.
    """
    t0 = time.time()
    import torch

    # ── Load SOMA template ─────────────────────────────────────────────
    log.info("[soma_xfer] loading SOMA template %s", SOMA_SKIN_PATH)
    soma = np.load(SOMA_SKIN_PATH, allow_pickle=True)
    soma_verts = soma["bind_vertices"].astype(np.float32)        # (18056, 3)
    soma_lbs_idx = soma["lbs_indices"].astype(np.int32)          # (18056, 8)
    soma_lbs_w = soma["lbs_weights"].astype(np.float32)          # (18056, 8)
    soma_joint_names = list(soma["rig_joint_names"])             # 77
    soma_bind = soma["bind_rig_transform"].astype(np.float32)    # (77, 4, 4)
    # UV layout ships WITH the pack (see SOMA_UV_PATH). Missing/mismatched
    # layout downgrades to no UV transfer (weights are unaffected) — the
    # run then fails only if a downstream consumer needs UVs, loudly.
    soma_uv: np.ndarray | None = None
    try:
        _uv = np.load(SOMA_UV_PATH).astype(np.float32)
        if _uv.shape == (len(soma_verts), 2):
            soma_uv = _uv
        else:
            log.warning("[soma_xfer] %s shape %s != (%d, 2) — UV transfer skipped",
                        SOMA_UV_PATH, _uv.shape, len(soma_verts))
    except (OSError, ValueError) as _e:
        log.warning("[soma_xfer] somaskel77 UV layout unavailable (%s) — UV transfer skipped", _e)
    log.info("[soma_xfer] template: %d verts, %d joints, uv=%s",
             len(soma_verts), len(soma_joint_names),
             soma_uv.shape if soma_uv is not None else None)

    # ── Load source GLB ────────────────────────────────────────────────
    gltf, bin_data = _read_glb(source_path)
    prim = gltf["meshes"][0]["primitives"][0]
    attrs = prim["attributes"]
    src_pos = _read_accessor(gltf, bin_data, attrs["POSITION"]).astype(np.float32)
    n_src = len(src_pos)
    log.info("[soma_xfer] source: %d verts", n_src)

    # ── AABB align SOMA → source ───────────────────────────────────────
    # TWO separate transforms:
    #
    # 1. T_align_verts (PER-AXIS): Scales each axis independently so that
    #    shoulder width, arm length, torso depth all match the source.
    #    Used for barycentric correspondence — ensures source verts match
    #    the correct template region. Without this, a character with wider
    #    shoulders than the template gets shoulder verts matched to upper-
    #    arm template verts → wrong weights → stretch artifacts.
    #
    # 2. T_align (UNIFORM): Height-ratio scale only. Used for bind matrices
    #    + motion data alignment. Preserves uniform root bone scale so the
    #    GLB writer's scale-stripping code works correctly.
    T_align_verts = _aabb_align_per_axis_safe(soma_verts, src_pos)
    soma_h = np.hstack([soma_verts, np.ones((len(soma_verts), 1), np.float32)])
    soma_aligned = (T_align_verts @ soma_h.T).T[:, :3].astype(np.float32)

    # Log scale ratios for diagnostics
    _v_diag = np.diag(T_align_verts[:3, :3])
    log.info("[soma_xfer] per-axis scale (verts): [%.3f, %.3f, %.3f]",
             _v_diag[0], _v_diag[1], _v_diag[2])

    T_align = _aabb_align_uniform(soma_verts, src_pos)

    # Transform bind matrices with UNIFORM scale (Blender-compatible)
    soma_bind_aligned = np.einsum("ij,njk->nik", T_align, soma_bind)
    # Re-normalize the bottom row to [0,0,0,1]
    soma_bind_aligned[:, 3, :3] = 0.0
    soma_bind_aligned[:, 3, 3] = 1.0

    # ── Barycentric weight transfer (SOMA-X SOTA pipeline) ─────────────
    # For each source vert, find its containing triangle in the T_align'd
    # SOMA template mesh and interpolate the template's hand-painted
    # weights via barycentric coordinates. This preserves the template's
    # anatomically-correct weight boundaries EXACTLY — including the
    # critical torso/arm divide that bone-heat smoothed across (causing
    # "wing" artifacts on raised-arm motions).
    #
    # Math proof (why wing is structurally impossible):
    #   A chest vert at |X|<4cm has its containing triangle in the SOMA
    #   template's spine region → barycentric coords interpolate spine
    #   weights ONLY → arm weight = 0.000000. Verified: torso core max
    #   arm weight = 0.000000 vs bone-heat's 0.05+ (which dragged 2189
    #   torso verts >5cm during a 90° arm raise → the "wing" membrane).
    #   With barycentric: 1 vert >5cm (99.95% reduction).
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_joints_soma = len(soma_joint_names)

    # 1. Densify template's sparse (18056, 8) → dense (18056, 77).
    # The template stores top-8 influences per vert as parallel (idx, w)
    # arrays. BarycentricInterpolator expects dense per-vertex attributes
    # to interpolate, so we scatter the sparse weights into a (V, J) grid.
    soma_faces = soma["faces"].astype(np.int64)
    soma_dense_w = np.zeros((len(soma_verts), n_joints_soma), dtype=np.float32)
    for slot in range(soma_lbs_idx.shape[1]):
        valid = soma_lbs_idx[:, slot] >= 0
        np.add.at(
            soma_dense_w,
            (np.arange(len(soma_verts))[valid], soma_lbs_idx[valid, slot]),
            soma_lbs_w[valid, slot],
        )
    log.info("[soma_xfer] densified template weights: %s (row-sum range %.4f..%.4f)",
             soma_dense_w.shape,
             float(soma_dense_w.sum(axis=1).min()),
             float(soma_dense_w.sum(axis=1).max()))

    # 2. Region-constrained correspondence (THE FIX for body horror) ──
    #
    # ROOT CAUSE of the armpit/hand/jaw distortion (user reports
    # 2026-07-29 "lower head, shoulders→arms, wrists→hand misalignment"
    # + 2026-07-30 "armpits static, hands mangled, face tilts up"):
    #   The ORIGINAL pipeline used trimesh.nearest.on_surface, which for
    #   each source vert finds the SINGLE geometrically closest triangle on
    #   the template surface. At CONCAVE JOINTS this picks the WRONG body
    #   part:
    #     • Armpit vert → closest tri is on the CHEST → 100% torso weight
    #       → armpit stays STATIC when the arm swings.
    #     • Finger vert → closest tri is a DIFFERENT finger or the palm
    #       → mangled hands.
    #     • Jaw vert → closest tri is upper FACE/neck → face tilts up;
    #       SOMAX literally cannot locate the jawline.
    #   The post-hoc containments further below (finger/blob/forearm) were
    #   whack-a-mole patches for individual symptoms — they never fixed the
    #   armpit or jaw, and they cleaned up AFTER the bad mapping happened.
    #
    # FIX: label every vertex's body part from the SOMA skeleton and NEVER
    # let a vert borrow weights from a different body part. An armpit vert
    # (arm region) can only map to arm/armpit template triangles — which
    # carry the correct arm-dominant blend, so the armpit moves WITH the
    # arm. A finger maps only to its own finger. The jaw maps only to
    # head/jaw triangles.
    #
    #  AMOUNT of DEVIATION will LEAD to UNACCEPTABLE BODY HORROR." This
    #  region constraint makes the transfer robust to the residual
    #  deviation between the TRELLIS mesh and the SOMA template, so the
    #  arm/shoulder/wrist/jaw connections come out clean even when the
    #  generated body isn't a perfect match for the template.)
    import trimesh as _trimesh
    from scipy.spatial import cKDTree as _cKDTree
    from soma.geometry.barycentric_interp import (
        fabricate_tet as _fabricate_tet,
        compute_barycentric_coords_3d as _bary3d,
    )

    # ── 2a. Joint → body-region map ───────────────────────────────────
    # Coarse enough that boundary blends (shoulder↔arm, hip↔leg) stay
    # within ONE region so the soft weight transition is preserved; fine
    # enough that hand / face / limb can't cross-contaminate each other.
    _R_HEAD, _R_TORSO = 0, 1
    _R_L_ARM, _R_L_HAND = 2, 3
    _R_R_ARM, _R_R_HAND = 4, 5
    _R_L_LEG, _R_R_LEG = 6, 7
    _JOINT_REGION: dict[int, int] = {}
    for _j in (4, 5, 6, 7, 8, 9, 10):   _JOINT_REGION[_j] = _R_HEAD    # neck+head+jaw+eyes
    for _j in (0, 1, 2, 3):              _JOINT_REGION[_j] = _R_TORSO   # hips+spine+chest
    for _j in range(11, 15):            _JOINT_REGION[_j] = _R_L_ARM   # L shoulder→wrist
    for _j in range(15, 38):            _JOINT_REGION[_j] = _R_L_HAND  # L fingers
    for _j in range(39, 43):            _JOINT_REGION[_j] = _R_R_ARM
    for _j in range(43, 67):            _JOINT_REGION[_j] = _R_R_HAND
    for _j in range(67, 72):            _JOINT_REGION[_j] = _R_L_LEG
    for _j in range(72, 77):            _JOINT_REGION[_j] = _R_R_LEG
    _REGION_NAMES = {
        _R_HEAD: "head", _R_TORSO: "torso",
        _R_L_ARM: "L.arm", _R_L_HAND: "L.hand",
        _R_R_ARM: "R.arm", _R_R_HAND: "R.hand",
        _R_L_LEG: "L.leg", _R_R_LEG: "R.leg",
    }

    # ── 2b. Template FACE → region ────────────────────────────────────
    # Each face's region = the body part of its strongest-weight joint
    # (total weight of each joint summed across the 3 verts). This labels
    # every template triangle with the body part it belongs to.
    _face_joint_w = soma_dense_w[soma_faces].sum(axis=1)   # (n_faces, 77)
    _face_dom_joint = np.argmax(_face_joint_w, axis=1)      # (n_faces,)
    _face_region = np.fromiter(
        (_JOINT_REGION.get(int(j), _R_TORSO) for j in _face_dom_joint),
        dtype=np.int64, count=len(_face_dom_joint),
    )
    del _face_joint_w  # 33 MB, free before the per-region queries

    # ── 2c. SOURCE vert → region (nearest aligned SOMA joint) ─────────
    # Place the skeleton inside the source mesh (AABB-aligned bind joints)
    # and label each source vert by its nearest joint's body part. The
    # armpit fold verts are closest to the shoulder joint → ARM region →
    # they move with the arm. The jaw verts are closest to the Jaw joint
    # → HEAD region → they don't pick up chest/neck weights.
    _aligned_joints = soma_bind_aligned[:, :3, 3]            # (77, 3) in src space
    _joint_tree = _cKDTree(_aligned_joints)
    _, _src_nearest_joint = _joint_tree.query(src_pos)       # (n_src,)
    _src_region = np.fromiter(
        (_JOINT_REGION.get(int(j), _R_TORSO) for j in _src_nearest_joint),
        dtype=np.int64, count=n_src,
    )

    # ── 2d. Region-constrained nearest-face query ─────────────────────
    # For each region, build a sub-mesh of ONLY that region's template
    # faces and query nearest surface point for source verts in the SAME
    # region. A vert in the LEFT HAND region can NEVER match a forearm or
    # torso triangle — this is what eliminates the cross-mapping.
    t_bary = time.time()
    face_ids = np.full(n_src, -1, dtype=np.int64)

    for _rid in sorted(set(_src_region.tolist())):
        _src_mask = _src_region == _rid
        _tgt_faces = np.where(_face_region == _rid)[0]
        if not _src_mask.any() or len(_tgt_faces) == 0:
            continue
        # Remap used verts to a dense [0, n) range so trimesh gets a valid
        # contiguous mesh for its internal BVH/KD-tree nearest query.
        _sub_faces_global = soma_faces[_tgt_faces]
        _used_verts = np.unique(_sub_faces_global)
        _remap = np.full(len(soma_aligned), -1, dtype=np.int64)
        _remap[_used_verts] = np.arange(len(_used_verts))
        _sub_mesh = _trimesh.Trimesh(
            vertices=soma_aligned[_used_verts],
            faces=_remap[_sub_faces_global],
            process=False,   # skip validation — we know the geometry is clean
        )
        _, _, _sub_face_ids = _sub_mesh.nearest.on_surface(src_pos[_src_mask])
        # Map sub-mesh local face ids back to GLOBAL template face ids.
        face_ids[_src_mask] = _tgt_faces[_sub_face_ids]

    # Fallback: any vert whose region had no template faces (face_ids==-1)
    # → global nearest face. Rare; only for degenerate/empty regions.
    _unassigned = face_ids == -1
    if _unassigned.any():
        _full_mesh = _trimesh.Trimesh(
            vertices=soma_aligned, faces=soma_faces, process=False,
        )
        _, _, _global_fids = _full_mesh.nearest.on_surface(src_pos[_unassigned])
        face_ids[_unassigned] = _global_fids
        log.warning(
            "[soma_xfer] %d verts had no template faces in their region — "
            "fell back to global nearest (unexpected for canonical parts)",
            int(_unassigned.sum()),
        )
    log.info(
        "[soma_xfer] region-constrained correspondence built in %.1fs "
        "(src regions: %s)",
        time.time() - t_bary,
        { _REGION_NAMES[r]: int((_src_region == r).sum())
          for r in sorted(set(_src_region.tolist())) },
    )

    # ── 2e. Barycentric weights for the region-matched faces ──────────
    # Same tetrahedral barycentric as BarycentricInterpolator: fabricate a
    # 4th vertex (normal offset) per triangle, then solve for 4 barycentric
    # coords. Handles source verts slightly OUTSIDE the template surface
    # (hair, loose clothing) without producing NaNs.
    #
    # We use normal_scale="edge" (not the default "area"): for small
    # template triangles the "area" scale produces a nearly-flat tet (P3
    # offset ≈ 2×area ≈ 1e-5), making the 3×3 solve matrix singular and
    # the barycentric coords explode to ±1000s. "edge" scales the offset
    # to the mean edge length → well-conditioned tetrahedra → stable
    # barycentric coords in [0, 1] after clamping below.
    _V_P3 = _fabricate_tet(
        soma_aligned[soma_faces[:, 0]],
        soma_aligned[soma_faces[:, 1]],
        soma_aligned[soma_faces[:, 2]],
        normal_scale="edge",
    )
    _V_tet = np.concatenate([soma_aligned, _V_P3], axis=0)
    _new_vert_idx = np.arange(len(soma_faces))[:, None] + len(soma_aligned)
    _F_tet = np.concatenate([soma_faces, _new_vert_idx], axis=1)  # (n_faces, 4)

    _tet = _F_tet[face_ids]          # (n_src, 4) global vert indices per src vert
    _v0 = _V_tet[_tet[:, 0]]
    _v1 = _V_tet[_tet[:, 1]]
    _v2 = _V_tet[_tet[:, 2]]
    _v3 = _V_tet[_tet[:, 3]]
    _bary = _bary3d(src_pos, _v0, _v1, _v2, _v3).astype(np.float32)  # (n_src, 4)

    # Clamp + renormalize barycentric coords. The raw tetrahedral solve
    # allows extrapolation (negative coords / coords > 1) for source verts
    # OUTSIDE the tet — which happens at region boundaries where the
    # nearest in-region face is geometrically far away. Unclamped, this
    # produces wild weight values (observed up to ±775 in tests). Clamping
    # to [0, ∞) and renormalizing converts extrapolation → interpolation:
    # the point projects onto the nearest tet face/edge/vertex. This is
    # the standard practice in mesh-deformation-transfer libraries and
    # keeps weight row-sums at exactly 1.0.
    _bary = np.maximum(_bary, 0.0)
    _bary /= np.maximum(_bary.sum(axis=1, keepdims=True), 1e-12)

    # ── 2f. Interpolate template weights via barycentric coords ───────
    # per_joint_w[i, j] = Σ_k bary[i,k] · w_padded[tet[i,k], j]
    #
    # The 4th tetrahedron vertex (P3) is a FABRICATED normal-offset point
    # with no real template weights. We give each face's P3 the AVERAGE of
    # its triangle's 3 vertex weights — the principled choice since P3
    # represents the triangle's normal direction, so its "attributes" are
    # the triangle's mean. For on-surface source verts b3≈0 (no effect);
    # for off-surface verts (hair, clothing, ears) the weight flows back
    # into the triangle's own joints instead of vanishing or going to a
    # nonsense cross-product (which the old BarycentricInterpolator did
    # implicitly when it ran fabricate_tet over weight channels).
    t_xfer = time.time()
    _face_avg_w = soma_dense_w[soma_faces].mean(axis=1)            # (n_faces, 77)
    _w_padded = np.concatenate([soma_dense_w, _face_avg_w], axis=0)  # (V+F, 77)
    per_joint_w = np.zeros((n_src, n_joints_soma), dtype=np.float32)
    for _k in range(4):
        per_joint_w += _bary[:, _k:_k + 1] * _w_padded[_tet[:, _k]]
    # Clamp negatives: barycentric extrapolation can produce them for
    # source verts slightly outside the template mesh (hair, clothing,
    # ears). These should be 0, not negative.
    per_joint_w = np.clip(per_joint_w, 0.0, None)
    log.info("[soma_xfer] weight interpolation done in %.1fs", time.time() - t_xfer)

    # ── Somax UV transfer (same barycentric pass, TEXCOORD not JOINTS) ──
    # P3 (fabricated tet apex) carries its triangle's mean UV, exactly as
    # it carries the triangle's mean weight. Values are NOT clamped: the
    # layout is a tiled atlas, and off-surface extrapolation may legitimately
    # land a hair's-breadth outside a tile — repeat wrap absorbs it.
    src_uv: np.ndarray | None = None
    if soma_uv is not None:
        _face_avg_uv = soma_uv[soma_faces].mean(axis=1)              # (n_faces, 2)
        _uv_padded = np.concatenate([soma_uv, _face_avg_uv], axis=0)  # (V+F, 2)
        src_uv = np.zeros((n_src, 2), dtype=np.float32)
        for _k in range(4):
            src_uv += _bary[:, _k:_k + 1] * _uv_padded[_tet[:, _k]]
        log.info("[soma_xfer] uv transfer done: range [%.3f, %.3f]",
                 float(src_uv.min()), float(src_uv.max()))

    # ── Finger weight containment ─────────────────────────────────────
    # Barycentric extrapolation assigns finger joint weights (Thumb, Index,
    # Middle, Ring, Pinky) to torso/limb verts that are OUTSIDE the SOMA
    # template's tight body surface (hair, clothing, ears, etc.). The
    # nearest template triangle for these "exterior" verts happens to be
    # in the finger region — so the vert gets LeftHandPinky4 weight even
    # though it's on the shoulder/chest/waist.
    #
    # During hand animation, these verts follow finger movements, causing
    # severe distortion: skin stretches behind the shoulder, chest ripples,
    # waist jiggles. This is the ROOT CAUSE of the "shoulder stretch"
    # artifact that stages 4b/4c/4d were trying (and failing) to fix.
    #
    # Fix: for each vert with finger weights, compute its distance to the
    # nearest hand joint. If it's outside a radius proportional to body
    # size, zero ALL finger weights. Verts that become zero-weight are
    # handled by the KNN-1 fallback below (assigned to nearest anatomical
    # joint by position).
    FINGER_KEYWORDS = ('Thumb', 'Index', 'Middle', 'Ring', 'Pinky')
    finger_joint_mask = np.zeros(n_joints_soma, dtype=bool)
    for ji in range(n_joints_soma):
        mix_name = SOMA_IDX_TO_MIXAMO_NAME.get(ji, '')
        if any(fk in mix_name for fk in FINGER_KEYWORDS):
            finger_joint_mask[ji] = True

    n_finger_joints = int(finger_joint_mask.sum())
    if n_finger_joints > 0:
        # Hand joint world positions (in source mesh space)
        left_hand_pos = soma_bind_aligned[14, :3, 3]   # SOMA joint 14 = LeftHand
        right_hand_pos = soma_bind_aligned[42, :3, 3]  # SOMA joint 42 = RightHand

        # Distance from each source vert to the NEAREST hand
        dist_left = np.linalg.norm(src_pos - left_hand_pos[None, :], axis=1)
        dist_right = np.linalg.norm(src_pos - right_hand_pos[None, :], axis=1)
        hand_dist = np.minimum(dist_left, dist_right)

        # Radius: verts within this distance of a hand may keep finger weights.
        # Fingers extend ~7% of body height from the wrist; the shoulder is
        # ~22% away. Using 12% gives a safe margin between legitimate finger
        # verts and torso verts that got finger weights from barycentric
        # extrapolation outside the template surface.
        body_height = float(src_pos[:, 1].max() - src_pos[:, 1].min())
        hand_radius = 0.12 * body_height

        # Find verts with finger weights that are far from BOTH hands
        has_finger_w = per_joint_w[:, finger_joint_mask].sum(axis=1) > 1e-6
        far_from_hands = hand_dist > hand_radius
        needs_fix = has_finger_w & far_from_hands

        n_finger_fixed = int(needs_fix.sum())
        if n_finger_fixed > 0:
            per_joint_w[np.ix_(needs_fix, finger_joint_mask)] = 0.0
            log.info(
                "[soma_xfer] finger containment: %d verts had finger weights "
                "outside hand radius (%.3fm) — zeroed (of %d total finger-weighted)",
                n_finger_fixed, hand_radius, int(has_finger_w.sum())
            )
        else:
            log.info(
                "[soma_xfer] finger containment: all finger-weighted verts "
                "are within hand radius (%.3fm) — no fixes needed",
                hand_radius
            )

        # ── Hand blob weight consolidation ─────────────────────────────
        # The TRELLIS mesh produces blob hands — ~280 verts per hand with
        # no distinguishable fingers. The barycentric transfer maps all
        # blob verts to whichever SOMAX template finger triangle is nearest,
        # clustering 80%+ of weight onto 1-2 finger bones (typically pinky
        # + ring). When ANY of those finger bones rotates during animation,
        # the entire blob distorts because the weight distribution doesn't
        # match the geometry at all.
        #
        # Analysis of c_v2_containment stage4_somax_rigged.glb showed:
        #   288 verts/hand, 56% rigidly locked to one bone (>0.94 weight)
        #   RightHandPinky2 alone had 72.8 total weight across all verts
        #   RightHandThumb had ZERO influence, Index had 0.1
        #   When pinky animates → entire hand blob distorts
        #
        # Fix: detect blob hands by counting verts that have finger weight
        # (NOT all verts within hand_radius, which includes forearm/wrist).
        # If the total is below the blob threshold, consolidate ALL finger
        # bone weights into the Hand bone for that side. The hand becomes a
        # rigid block that follows wrist rotation cleanly — no individual
        # finger animation, but no distortion either.
        #
        # For high-res meshes with real finger geometry (500+ finger-weighted
        # verts per hand), this pass is skipped so fingers keep their weights.
        n_finger_weighted = int(has_finger_w.sum())

        # A proper hand mesh has ~1000+ finger-weighted verts per hand
        # (5 fingers × 4 joints × ~50 verts per segment). Below 2000 total
        # → blob hands without real finger geometry.
        BLOB_FINGER_THRESHOLD = 2000
        is_blob_hand = n_finger_weighted < BLOB_FINGER_THRESHOLD

        if is_blob_hand:
            # Determine each finger-weighted vert's nearest hand (left/right)
            left_closer = dist_left <= dist_right

            for side, closer_mask, hand_soma_idx in [
                ('Left', left_closer, 14),    # SOMA joint 14 = LeftHand
                ('Right', ~left_closer, 42),  # SOMA joint 42 = RightHand
            ]:
                # Only process verts that HAVE finger weight AND are closest
                # to this hand
                this_hand = has_finger_w & closer_mask
                if not this_hand.any():
                    continue

                # Find finger bone indices for THIS side only
                side_finger_ji = []
                for ji in range(n_joints_soma):
                    mix_name = SOMA_IDX_TO_MIXAMO_NAME.get(ji, '')
                    if side in mix_name and any(
                        fk in mix_name for fk in FINGER_KEYWORDS
                    ):
                        side_finger_ji.append(ji)

                if not side_finger_ji:
                    continue

                # Snapshot total finger weight per vert before zeroing
                finger_w_sum = per_joint_w[
                    np.ix_(this_hand, side_finger_ji)
                ].sum(axis=1)

                # Zero all side-finger weights for these verts
                per_joint_w[np.ix_(this_hand, side_finger_ji)] = 0.0

                # Consolidate into the Hand bone for this side
                per_joint_w[this_hand, hand_soma_idx] += finger_w_sum

                # Re-normalize this hand's verts
                wsum_h = per_joint_w[this_hand].sum(axis=1, keepdims=True)
                wsum_h = np.maximum(wsum_h, 1e-10)
                per_joint_w[this_hand] /= wsum_h

                n_v = int(this_hand.sum())
                avg_fw = float(finger_w_sum.mean())
                log.info(
                    "[soma_xfer] hand blob consolidation (%s): %d verts — "
                    "finger weights (avg %.1f%%) consolidated to %sHand bone. "
                    "Hand animates as rigid block (blob has no finger geometry).",
                    side, n_v, 100 * avg_fw, side
                )
        else:
            log.info(
                "[soma_xfer] hand blob consolidation: skipped (%d finger-"
                "weighted verts ≥ %d threshold — likely has finger geometry)",
                n_finger_weighted, BLOB_FINGER_THRESHOLD
            )

    # ── Forearm upward containment ─────────────────────────────────────
    # Same principle as finger containment: barycentric transfer assigns
    # forearm weights to vertices ABOVE the elbow (shoulder, upper-arm,
    # chest-back area). When the forearm rotates during animation, these
    # verts get dragged, causing the persistent "shoulder stretch" /
    # "skin contorts behind the shoulder" artifact that stages 4b/4c/4d
    # were trying (and failing) to fix.
    #
    # Root cause: the SOMA template's forearm region is geometrically
    # close to the shoulder when the arms hang at the sides. Barycentric
    # extrapolation maps upper-arm and back-of-shoulder source verts to
    # template triangles that straddle the elbow, picking up forearm weight.
    #
    # Fix: for each side, define the arm axis (shoulder → elbow). For any
    # vertex with forearm weight that projects ABOVE the elbow along this
    # axis (i.e., toward the shoulder), zero the forearm weight. The
    # zeroed weight is redistributed by per-vertex re-normalization, which
    # boosts the remaining (correct) arm/shoulder/spine weights.
    FOREARM_NAMES_UP = ('LeftForeArm', 'RightForeArm')
    forearm_joint_indices = []
    for ji in range(n_joints_soma):
        mix_name = SOMA_IDX_TO_MIXAMO_NAME.get(ji, '')
        if mix_name in FOREARM_NAMES_UP:
            forearm_joint_indices.append((ji, mix_name))

    for fa_ji, fa_name in forearm_joint_indices:
        side = 'Left' if 'Left' in fa_name else 'Right'
        # SOMA skeleton indices: LeftForeArm=13, LeftArm=12
        #                    RightForeArm=41, RightArm=40
        elbow_soma_idx = 13 if side == 'Left' else 41
        arm_soma_idx = 12 if side == 'Left' else 40

        elbow_pos = soma_bind_aligned[elbow_soma_idx, :3, 3]
        shoulder_pos = soma_bind_aligned[arm_soma_idx, :3, 3]

        # Arm axis: from elbow UP toward shoulder
        arm_axis = shoulder_pos - elbow_pos
        axis_len = np.linalg.norm(arm_axis)
        if axis_len < 1e-6:
            continue
        arm_axis = arm_axis / axis_len

        # Find verts with THIS forearm's weight
        has_fa = per_joint_w[:, fa_ji] > 0.05
        if not has_fa.any():
            continue

        # Project relative position onto arm axis
        rel = src_pos[has_fa] - elbow_pos[None, :]
        proj = rel @ arm_axis  # >0 = above elbow (toward shoulder)

        # Verts above the elbow by >8% of arm length are contaminated.
        # The 8% threshold removes only the most contaminated verts (high
        # forearm weight far above the elbow). More aggressive thresholds
        # (2%, 5%) create sharper weight boundaries that increase mesh
        # tearing — the Laplacian smoothing pass below handles the gradual
        # transition zone better than hard zeroing.
        # A/B tested: 8% + Laplacian = 6.5% torn area vs 11.3% baseline.
        threshold = 0.08 * axis_len
        above = proj > threshold
        above_indices = np.where(has_fa)[0][above]

        if len(above_indices) > 0:
            avg_w = per_joint_w[above_indices, fa_ji].mean()
            per_joint_w[above_indices, fa_ji] = 0.0
            log.info(
                "[soma_xfer] forearm containment (%s): %d verts had %s "
                "weight above elbow (avg w=%.3f) — zeroed. "
                "These were the shoulder-stretch culprits.",
                side, len(above_indices), fa_name, avg_w
            )

    # ── Upper-arm upward containment ──────────────────────────────────
    # Same principle but for the LeftArm/RightArm bone: barycentric transfer
    # assigns arm weights to vertices ABOVE the shoulder joint (neck, upper
    # chest, upper back). When the arm raises, these verts get dragged,
    # stretching the neck/chest area.
    #
    # Fix: define the shoulder→spine axis. For any vertex with arm weight
    # that projects ABOVE the shoulder joint toward the spine, attenuate
    # the arm weight proportionally (gentler than forearm — the arm bone
    # legitimately influences some shoulder area vertices).
    ARM_NAMES_UP = ('LeftArm', 'RightArm')
    arm_joint_indices = []
    for ji in range(n_joints_soma):
        mix_name = SOMA_IDX_TO_MIXAMO_NAME.get(ji, '')
        if mix_name in ARM_NAMES_UP:
            arm_joint_indices.append((ji, mix_name))

    for arm_ji, arm_name in arm_joint_indices:
        side = 'Left' if 'Left' in arm_name else 'Right'
        # Spine2 is the joint above the shoulder in the SOMA skeleton
        # LeftShoulder=11, LeftArm=12, Spine2=3
        spine2_soma_idx = 3
        arm_soma_idx = 12 if side == 'Left' else 40

        shoulder_pos = soma_bind_aligned[arm_soma_idx, :3, 3]
        spine2_pos = soma_bind_aligned[spine2_soma_idx, :3, 3]

        # Axis: from shoulder UP toward spine2 (center of upper chest)
        up_axis = spine2_pos - shoulder_pos
        up_len = np.linalg.norm(up_axis)
        if up_len < 1e-6:
            continue
        up_axis = up_axis / up_len

        # Find verts with THIS arm's weight
        has_arm = per_joint_w[:, arm_ji] > 0.10
        if not has_arm.any():
            continue

        # Project relative position onto up axis
        rel = src_pos[has_arm] - shoulder_pos[None, :]
        proj = rel @ up_axis  # >0 = above shoulder (toward spine/neck)

        # Verts projecting above the shoulder by >15% of shoulder-spine
        # distance are in the neck/upper-chest zone. Attenuate (not zero)
        # their arm weight by 70% — reduces the drag without removing
        # legitimate shoulder influence entirely.
        up_threshold = 0.15 * up_len
        above = proj > up_threshold
        above_indices = np.where(has_arm)[0][above]

        if len(above_indices) > 0:
            avg_w = per_joint_w[above_indices, arm_ji].mean()
            # Gentle attenuation: reduce arm weight by 70%
            per_joint_w[above_indices, arm_ji] *= 0.3
            log.info(
                "[soma_xfer] upper-arm containment (%s): %d verts had %s "
                "weight above shoulder (avg w=%.3f) — attenuated 70%%.",
                side, len(above_indices), arm_name, avg_w
            )

    # Re-normalize after forearm + upper-arm containment
    wsum_fc = per_joint_w.sum(axis=1, keepdims=True)
    wsum_fc = np.maximum(wsum_fc, 1e-10)
    per_joint_w /= wsum_fc

    # ── Laplacian weight smoothing at shoulder-torso boundary ─────────
    # Sharp weight gradients between the arm/forearm bones and the spine
    # bones cause mesh tearing during animation: neighboring vertices with
    # very different weights move in different directions, stretching the
    # triangles between them.
    #
    # Fix: build a vertex adjacency graph from the source mesh faces, then
    # apply a localized Laplacian smoothing pass ONLY to vertices in the
    # shoulder band (y ∈ [shoulder_y - 0.1, shoulder_y + 0.1]). This blends
    # each shoulder vertex's weights with its neighbors' average, reducing
    # sharp gradients without affecting the (already good) limb weights.
    #
    # The smoothing factor is 0.35 (35% neighbor blend, 65% original) —
    # enough to soften transitions but not enough to destroy anatomical
    # weight boundaries.
    try:
        from scipy.sparse import csr_matrix

        # Read source mesh faces for adjacency graph
        if "indices" in prim:
            src_faces = _read_accessor(gltf, bin_data, prim["indices"]).reshape(-1, 3).astype(np.int64)
        else:
            # Non-indexed primitive: generate sequential faces
            n_src_verts = len(src_pos)
            src_faces = np.arange(n_src_verts, dtype=np.int64).reshape(-1, 3)

        # Build adjacency from faces
        n_src = len(src_pos)
        rows = np.concatenate([src_faces[:,0], src_faces[:,1], src_faces[:,2],
                               src_faces[:,1], src_faces[:,2], src_faces[:,0]])
        cols = np.concatenate([src_faces[:,1], src_faces[:,2], src_faces[:,0],
                               src_faces[:,0], src_faces[:,1], src_faces[:,2]])
        data = np.ones(len(rows), dtype=np.float64)
        adj = csr_matrix((data, (rows, cols)), shape=(n_src, n_src))
        # Normalize rows to get averaging operator
        row_sums = np.array(adj.sum(axis=1)).flatten()
        row_sums = np.maximum(row_sums, 1)
        adj_norm = csr_matrix((data / np.repeat(row_sums[rows], 1), (rows, cols)),
                              shape=(n_src, n_src))

        # Identify shoulder band + wrist band vertices for smoothing.
        y_coords = src_pos[:, 1]
        y_min, y_max = y_coords.min(), y_coords.max()
        y_range = y_max - y_min
        shoulder_y = y_min + 0.82 * y_range
        shoulder_mask = (y_coords > shoulder_y - 0.10 * y_range) & \
                        (y_coords < shoulder_y + 0.10 * y_range)
        shoulder_idx = np.where(shoulder_mask)[0]

        # Also identify wrist-band vertices. After hand blob consolidation,
        # the hand verts have 100% Hand bone weight while adjacent forearm
        # verts have ForeArm weight — a sharp gradient that tears during
        # wrist rotation. Smooth the wrist transition band.
        left_wrist_y = float(soma_bind_aligned[14, 1, 3])   # LeftHand Y
        right_wrist_y = float(soma_bind_aligned[42, 1, 3])  # RightHand Y
        wrist_mask = np.zeros(n_src, dtype=bool)
        for wrist_y in [left_wrist_y, right_wrist_y]:
            wrist_mask |= (y_coords > wrist_y - 0.04 * y_range) & \
                          (y_coords < wrist_y + 0.04 * y_range)
        wrist_idx = np.where(wrist_mask)[0]

        # Combine both bands for smoothing
        smooth_idx = np.unique(np.concatenate([shoulder_idx, wrist_idx])) \
            if len(shoulder_idx) > 0 else wrist_idx

        if len(smooth_idx) > 10:
            # Apply 2 iterations of Laplacian smoothing to the combined bands.
            # A/B tested: α=0.35/2iter gives 6.5% torn area (best for shoulder).
            # Same parameters applied to wrist band to smooth the Hand↔ForeArm
            # weight gradient created by blob consolidation.
            alpha = 0.35  # blend factor
            for _ in range(2):
                # Compute neighbor average for ALL verts (vectorized)
                neighbor_avg = adj_norm @ per_joint_w
                # Only blend the target verts
                per_joint_w[smooth_idx] = (
                    (1 - alpha) * per_joint_w[smooth_idx] +
                    alpha * neighbor_avg[smooth_idx]
                )
                # Re-normalize
                wsum = per_joint_w.sum(axis=1, keepdims=True)
                wsum = np.maximum(wsum, 1e-10)
                per_joint_w /= wsum

            log.info(
                "[soma_xfer] Laplacian weight smoothing: %d verts smoothed "
                "(shoulder=%d, wrist=%d, α=%.2f, 2 iter) — reduces weight "
                "gradients that cause mesh tearing at shoulder and wrist.",
                len(smooth_idx), len(shoulder_idx), len(wrist_idx), alpha
            )
        else:
            log.info("[soma_xfer] Laplacian smoothing: skipped (only %d verts)",
                     len(smooth_idx))
    except Exception as e:
        log.warning("[soma_xfer] Laplacian smoothing failed (non-fatal): %s", e)

    # ── Fallback for zero-weight verts (outside the SOMA template) ──────
    # Barycentric extrapolation can leave a small fraction of source verts
    # (typically hair, clothing, ears — geometry not present on the SOMA
    # template's tight body) with ALL-ZERO weights. These would fail the
    # golden contract's weight_sums check (sum != 1.0) and cause "floating
    # shard" artifacts in render (unskinned verts stay at bind pose).
    #
    # Fix: assign each zero-weight vert to its single nearest SOMA template
    # vert's TOP joint (KNN-1 on the aligned template positions). This is
    # a graceful fallback — the vert follows the closest rigid bone, which
    # is almost always correct for geometry just outside the body surface.
    zero_mask = per_joint_w.sum(axis=1) < 1e-8
    n_zero = int(zero_mask.sum())
    # Pre-compute aligned template tensor for KNN-1 fallback (used by both
    # the zero-weight fallback below and the post-normalization containment)
    soma_at = torch.from_numpy(soma_aligned).to(device)
    if n_zero > 0:
        # KNN-1: find nearest template vert for each zero-weight source vert
        src_zt = torch.from_numpy(src_pos[zero_mask]).to(device)
        d = torch.cdist(src_zt, soma_at)               # (n_zero, 18056)
        _, nn_idx = torch.topk(d, 1, dim=1, largest=False)  # (n_zero, 1)
        nn_idx = nn_idx.cpu().numpy().flatten()
        # For each zero vert, copy the nearest template vert's top-1 joint
        for i, tvi in enumerate(nn_idx):
            src_vi = int(np.where(zero_mask)[0][i])
            j = int(np.argmax(soma_dense_w[tvi]))
            per_joint_w[src_vi, j] = 1.0
        log.info("[soma_xfer] barycentric transfer done in %.1fs — shape %s, "
                 "zero-weight verts fallback (KNN-1 → nearest joint): %d (%.2f%%)",
                 time.time() - t_xfer, per_joint_w.shape,
                 n_zero, float(zero_mask.mean() * 100))
    else:
        log.info("[soma_xfer] barycentric transfer done in %.1fs — shape %s, "
                 "zero-weight verts: 0",
                 time.time() - t_xfer, per_joint_w.shape)

    # ── Cross-body decontamination ────────────────────────────────────
    # Barycentric transfer can assign weights to BOTH left and right joints
    # when source verts lie near the sagittal plane (inner thighs, collar
    # bones, spine). During animation, left and right bones move in OPPOSITE
    # directions, causing LBS collapse ("flattened pancake" / "candy
    # wrapper" effect) at those verts.
    #
    # Fix: for each vertex, determine its side by looking at which side has
    # the highest-weight bone. Then zero out all opposite-side joints and
    # re-normalize. Center joints (Spine, Head, Hips, Neck, etc.) are never
    # filtered.
    is_left_joint = np.zeros(n_joints_soma, dtype=bool)
    is_right_joint = np.zeros(n_joints_soma, dtype=bool)
    for ji, jname in enumerate(soma_joint_names):
        jl = jname.lower()
        if "left" in jl:
            is_left_joint[ji] = True
        elif "right" in jl:
            is_right_joint[ji] = True

    if is_left_joint.any() and is_right_joint.any():
        # Max weight from left-side joints vs right-side joints, per vertex
        left_max_w = per_joint_w[:, is_left_joint].max(axis=1)   # (N,)
        right_max_w = per_joint_w[:, is_right_joint].max(axis=1) # (N,)

        vertex_left = left_max_w > right_max_w    # (N,) bool
        vertex_right = right_max_w > left_max_w   # (N,) bool

        # Zero out cross-body contamination using broadcasting masks
        # Left-side vertices: zero out right joints
        cross_lr = vertex_left[:, None] & is_right_joint[None, :]
        # Right-side vertices: zero out left joints
        cross_rl = vertex_right[:, None] & is_left_joint[None, :]

        n_cleaned = (cross_lr | cross_rl).any(axis=1).sum()
        per_joint_w[cross_lr] = 0.0
        per_joint_w[cross_rl] = 0.0
        log.info(
            "[soma_xfer] cross-body decontam: %d/%d verts cleaned "
            "(L→R=%d, R→L=%d)",
            n_cleaned, n_src, cross_lr.any(axis=1).sum(),
            cross_rl.any(axis=1).sum(),
        )

    # Normalize per vertex (after decontamination)
    wsum = per_joint_w.sum(axis=1, keepdims=True)
    wsum = np.maximum(wsum, 1e-10)
    per_joint_w /= wsum

    # ── Post-normalization finger containment ──────────────────────────
    # After cross-body decontamination zeros opposite-side joints and
    # normalization rescales, tiny finger weights (<0.01) from verts
    # outside the template surface can be amplified into the top-4.
    # This second pass catches them on the FINAL weights, just before
    # top-M truncation.
    if n_finger_joints > 0:
        has_finger_w = per_joint_w[:, finger_joint_mask].sum(axis=1) > 1e-6
        far_from_hands = hand_dist > hand_radius
        needs_fix = has_finger_w & far_from_hands
        n_fix2 = int(needs_fix.sum())
        if n_fix2 > 0:
            per_joint_w[np.ix_(needs_fix, finger_joint_mask)] = 0.0
            # Re-normalize after zeroing finger weights
            wsum2 = np.maximum(per_joint_w.sum(axis=1, keepdims=True), 1e-10)
            per_joint_w /= wsum2
            # For verts that became zero-weight (ALL their weight was finger),
            # assign to nearest non-finger joint by KNN-1 on position
            all_zero = per_joint_w.sum(axis=1) < 1e-8
            n_all_zero = int(all_zero.sum())
            if n_all_zero > 0:
                src_zt2 = torch.from_numpy(src_pos[all_zero]).to(device)
                d2 = torch.cdist(src_zt2, soma_at)
                if d2 is not None:
                    _, nn2 = torch.topk(d2, 1, dim=1, largest=False)
                    nn2 = nn2.cpu().numpy().flatten()
                    for i2, tvi2 in enumerate(nn2):
                        src_vi2 = int(np.where(all_zero)[0][i2])
                        # Copy nearest template vert's top-1 NON-FINGER joint
                        w_copy = soma_dense_w[tvi2].copy()
                        w_copy[finger_joint_mask] = 0.0
                        j2 = int(np.argmax(w_copy))
                        per_joint_w[src_vi2, j2] = 1.0
            log.info(
                "[soma_xfer] post-norm finger containment: %d verts had "
                "amplified finger weights — zeroed (of %d finger-weighted, "
                "%d became zero→KNN-1)",
                n_fix2, int(has_finger_w.sum()), n_all_zero
            )
        else:
            log.info(
                "[soma_xfer] post-norm finger containment: no amplified "
                "finger weights found"
            )

    # ── Truncate to top-M influences per vertex (glTF VEC4 default) ────
    M = max_influence_per_vertex
    top_idx = np.argsort(-per_joint_w, axis=1)[:, :M]   # (N, M)
    top_w = np.take_along_axis(per_joint_w, top_idx, axis=1)  # (N, M)
    # Re-normalize after truncation
    top_w /= np.maximum(top_w.sum(axis=1, keepdims=True), 1e-10)
    # Threshold: drop near-zero weights, keep at least 1
    top_w[top_w < 0.01] = 0.0
    top_w /= np.maximum(top_w.sum(axis=1, keepdims=True), 1e-10)

    # Safety: replace any NaN with 0
    top_w = np.nan_to_num(top_w, nan=0.0)

    log.info("[soma_xfer] weight interpolation complete (M=%d influences)", M)

    # ── Build output GLB ───────────────────────────────────────────────
    _write_rigged_glb(
        source_gltf=gltf,
        source_bin=bin_data,
        source_pos=src_pos,
        joints=top_idx.astype(np.uint16),
        weights=top_w.astype(np.float32),
        src_uv=src_uv,
        soma_joint_names=soma_joint_names,
        soma_bind_aligned=soma_bind_aligned,
        output_path=output_path,
        # T_align maps SOMA template space → source mesh space. The server's
        # deform_custom endpoint applies this to Kimodo's motion data (which
        # is in SOMA template space) so the LBS World matrices match the IBM
        # (which was computed in source-aligned space = T_align(soma_bind)).
        # Without this, the motion is in the wrong coordinate frame: Kimodo's
        # ~1.7m-tall character with feet at Y=0 drives a ~1.0m-tall normalized
        # mesh whose IBM expects Y-centered coords → body horror.
        t_align=T_align.astype(np.float32),
        preserve_texture=preserve_texture,
    )

    elapsed = time.time() - t0
    log.info("[soma_xfer] complete in %.1fs → %s", elapsed, output_path)
    return output_path


# ═══════════════════════════════════════════════════════════════════════════
# GLB writer
# ═══════════════════════════════════════════════════════════════════════════

def _write_rigged_glb(
    source_gltf: dict,
    source_bin: bytes,
    source_pos: np.ndarray,
    joints: np.ndarray,           # (N, M) uint16 joint indices
    weights: np.ndarray,          # (N, M) float32
    soma_joint_names: list[str],
    soma_bind_aligned: np.ndarray,  # (77, 4, 4) world bind matrices (aligned)
    output_path: str,
    t_align: np.ndarray | None = None,  # (4, 4) SOMA-template → source-mesh transform
    preserve_texture: bool = True,
    src_uv: np.ndarray | None = None,    # (N, 2) somax-layout UVs for the source verts
) -> None:
    """Write a new GLB = source geometry/textures + skin weights + armature.

    When preserve_texture=False, strips every material's texture references
    and replaces them with a flat SOMA-blue baseColorFactor. Geometry, skin,
    and armature are unchanged — this is purely a material/texture swap so
    the character can be re-textured later in Pose Studio or another tool.
    """
    n_verts = len(source_pos)
    n_joints_soma = len(soma_joint_names)

    # Build the joint name list (Mixamo-renamed) and parent hierarchy
    joint_names_mix = [SOMA_IDX_TO_MIXAMO_NAME.get(i, f"SOMA_{i}")
                       for i in range(n_joints_soma)]
    # Parent indices: SOMA77_PARENTS gives SOMA parent, which is the same index
    # in our output (we preserve SOMA order)
    joint_parents = [SOMA77_PARENTS[i] if i < len(SOMA77_PARENTS) else None
                     for i in range(n_joints_soma)]

    # ── Build new buffer layout ────────────────────────────────────────
    # Strategy: keep source bufferView/accessor structure intact for geometry,
    # append JOINTS_0 + WEIGHTS_0 + IBM as new accessors at the end.
    gltf = json.loads(json.dumps(source_gltf))  # deep copy
    src_buffer = gltf.get("buffers", [{}])[0]
    src_byte_length = src_buffer.get("byteLength", len(source_bin))

    # ── Ensure Y-up output (glTF standard) ────────────────────────────
    # The source mesh (TRELLIS) may be in Z-up. The AABB alignment may
    # put bones in Z-up too. glTF REQUIRES all data in Y-up.
    # We detect the mesh height axis and convert EVERYTHING to Y-up if
    # the mesh is not already Y-up.
    prim_ref = gltf["meshes"][0]["primitives"][0]
    attrs_ref = prim_ref.get("attributes", {})
    pos_acc_idx_ref = attrs_ref.get("POSITION")

    _need_yup_fix = False
    if pos_acc_idx_ref is not None:
        pos_acc_ref = gltf["accessors"][pos_acc_idx_ref]
        pos_min = pos_acc_ref.get("min", [0, 0, 0])
        pos_max = pos_acc_ref.get("max", [0, 0, 0])
        mesh_spreads = [pos_max[i] - pos_min[i] for i in range(3)]
        mesh_height = max(range(3), key=lambda i: mesh_spreads[i])

        if mesh_height == 2:  # Z-up — need to convert to Y-up
            _need_yup_fix = True
            log.info("[soma_xfer] converting Z-up → Y-up for glTF output")

            # ── Convert mesh POSITION + NORMAL in the binary ──
            _src_bin_mutable = bytearray(source_bin[:src_byte_length])

            # POSITION: Z-up → Y-up = (x, y, z) → (x, z, -y)
            # This maps +Z (up in Z-up) to +Y (up in Y-up).
            pos_bv = gltf["bufferViews"][pos_acc_ref["bufferView"]]
            pos_off = pos_bv.get("byteOffset", 0) + pos_acc_ref.get("byteOffset", 0)
            pos_count = pos_acc_ref["count"]
            pos_bytes = pos_count * 3 * 4
            pos_data = np.frombuffer(bytes(_src_bin_mutable[pos_off:pos_off + pos_bytes]),
                                      dtype=np.float32).reshape(pos_count, 3).copy()
            new_pos = np.column_stack([pos_data[:, 0], pos_data[:, 2], -pos_data[:, 1]]).astype(np.float32)
            _src_bin_mutable[pos_off:pos_off + pos_bytes] = new_pos.tobytes()
            pos_acc_ref["min"] = [float(new_pos[:, i].min()) for i in range(3)]
            pos_acc_ref["max"] = [float(new_pos[:, i].max()) for i in range(3)]

            # NORMAL: same rotation
            nrm_acc_idx = attrs_ref.get("NORMAL")
            if nrm_acc_idx is not None:
                nrm_acc = gltf["accessors"][nrm_acc_idx]
                nrm_bv = gltf["bufferViews"][nrm_acc["bufferView"]]
                nrm_off = nrm_bv.get("byteOffset", 0) + nrm_acc.get("byteOffset", 0)
                nrm_count = nrm_acc["count"]
                nrm_bytes = nrm_count * 3 * 4
                nrm_data = np.frombuffer(bytes(_src_bin_mutable[nrm_off:nrm_off + nrm_bytes]),
                                          dtype=np.float32).reshape(nrm_count, 3).copy()
                new_nrm = np.column_stack([nrm_data[:, 0], nrm_data[:, 2], -nrm_data[:, 1]]).astype(np.float32)
                _src_bin_mutable[nrm_off:nrm_off + nrm_bytes] = new_nrm.tobytes()

            source_bin = bytes(_src_bin_mutable)

            # ── Convert soma_bind_aligned from Z-up to Y-up ──
            # LEFT-MULTIPLY by P4 (not conjugation). This transforms the
            # world-space bone matrices to Y-up, same as the mesh vertices.
            # The root bone's local transform gets the P4 rotation, which
            # correctly maps child local-Z translations to world +Y (up).
            P4 = np.array([
                [1, 0, 0, 0],
                [0, 0, 1, 0],
                [0, -1, 0, 0],
                [0, 0, 0, 1]
            ], dtype=np.float64)
            soma_bind_aligned = np.array([
                P4 @ m.astype(np.float64)
                for m in soma_bind_aligned
            ], dtype=np.float32)
        else:
            log.info("[soma_xfer] mesh already Y-up — no coordinate fix needed")

    # ── IBM computed AFTER scale stripping (see below) ──

    # Pad source bin to 4-byte alignment
    pad = (4 - (src_byte_length % 4)) % 4
    new_bin = bytearray(source_bin[:src_byte_length])
    if pad:
        new_bin.extend(b"\x00" * pad)

    def add_bufferView(data: bytes) -> int:
        nonlocal new_bin
        # 4-byte align
        cur_pad = (4 - (len(new_bin) % 4)) % 4
        if cur_pad:
            new_bin.extend(b"\x00" * cur_pad)
        offset = len(new_bin)
        new_bin.extend(data)
        bv_idx = len(gltf.setdefault("bufferViews", []))
        gltf["bufferViews"].append({
            "buffer": 0,
            "byteOffset": offset,
            "byteLength": len(data),
        })
        return bv_idx

    def add_accessor(bv_idx: int, count: int, atype: str,
                     comp: int, mins=None, maxs=None) -> int:
        acc = {
            "bufferView": bv_idx,
            "componentType": comp,
            "count": count,
            "type": atype,
        }
        if mins is not None:
            acc["min"] = mins
        if maxs is not None:
            acc["max"] = maxs
        gltf.setdefault("accessors", []).append(acc)
        return len(gltf["accessors"]) - 1

    # ── JOINTS_0 (uint16 vec4) ─────────────────────────────────────────
    joints_bytes = joints.astype(np.uint16).tobytes()
    bv_j = add_bufferView(joints_bytes)
    acc_j = add_accessor(bv_j, n_verts, "VEC4", 5123,
                          mins=[int(joints.min())],
                          maxs=[int(joints.max())])

    # ── WEIGHTS_0 (float32 vec4) ───────────────────────────────────────
    weights_bytes = weights.astype(np.float32).tobytes()
    bv_w = add_bufferView(weights_bytes)
    acc_w = add_accessor(bv_w, n_verts, "VEC4", 5126,
                          mins=[0.0, 0.0, 0.0, 0.0],
                          maxs=[1.0, 1.0, 1.0, 1.0])

    # Add JOINTS_0 + WEIGHTS_0 to mesh primitive
    prim = gltf["meshes"][0]["primitives"][0]
    prim["attributes"]["JOINTS_0"] = acc_j
    prim["attributes"]["WEIGHTS_0"] = acc_w

    # ── TEXCOORD_0 (float32 vec2, somax layout) ────────────────────────
    # Only when the source primitive has NO UVs of its own: a body that
    # arrives with UVs keeps them (its textures live in that space). The
    # somax layout spans beyond [0,1] (tiled atlas) — min/max stated from
    # the data, never clamped.
    if src_uv is not None and "TEXCOORD_0" not in prim["attributes"]:
        uv_bytes = src_uv.astype(np.float32).tobytes()
        bv_uv = add_bufferView(uv_bytes)
        acc_uv = add_accessor(bv_uv, n_verts, "VEC2", 5126,
                              mins=[float(src_uv[:, 0].min()), float(src_uv[:, 1].min())],
                              maxs=[float(src_uv[:, 0].max()), float(src_uv[:, 1].max())])
        prim["attributes"]["TEXCOORD_0"] = acc_uv

    # ── IBM buffer created later, after scale-stripping + FK recompute ──

    # ── Build nodes hierarchy ──────────────────────────────────────────
    # Existing nodes might include mesh node. We need to:
    #   1. Add n_joints_soma new nodes (the bones)
    #   2. Set parent-child relationships via the "children" field
    #   3. Attach the mesh node to the root bone (Hips) — actually glTF
    #      uses a SKIN attached to the mesh node, not a parent link.
    #   4. Set each bone's translation + rotation from local matrices

    existing_node_count = len(gltf.get("nodes", []))
    scenes = gltf.get("scenes", [{"nodes": []}])
    root_node = scenes[0].get("nodes", [None])[0]

    # Compute local transforms: local = inv(parent_world) @ world
    # For root (Hips): local = world
    bone_node_idxs = []
    local_transforms: list[tuple[list[float], list[float], list[float]]] = []  # (t, r, s)

    # Track the uniform scale from the root bone (AABB alignment)
    _root_uniform_scale: float | None = None

    for i in range(n_joints_soma):
        world = soma_bind_aligned[i]
        parent_idx = joint_parents[i]
        if parent_idx is None:
            local = world
        else:
            parent_world = soma_bind_aligned[parent_idx]
            local = np.linalg.inv(parent_world) @ world

        # Decompose into TRS (translation, rotation, scale).
        t = local[:3, 3]
        rmat = local[:3, :3].astype(np.float64)
        # Extract scale as column norms
        col_norms = np.linalg.norm(rmat, axis=0)
        col_norms = np.maximum(col_norms, 1e-10)
        # Normalize to get pure rotation
        rot_normalized = rmat / col_norms
        # Fix sign: if det < 0, flip the smallest column
        if np.linalg.det(rot_normalized) < 0:
            min_idx = np.argmin(col_norms)
            rot_normalized[:, min_idx] *= -1
            col_norms[min_idx] *= -1
        q = _mat3_to_quat_xyzw(rot_normalized)

        if parent_idx is None:
            # Root bone: capture the uniform scale for stripping.
            # AABB alignment uses uniform scale, so all 3 norms ≈ same.
            _root_uniform_scale = float(np.mean(col_norms))

        local_transforms.append((
            t.tolist(), q, [float(x) for x in col_norms]
        ))

    # ── STRIP SCALE from bone nodes ───────────────────────────────────
    # Blender's glTF importer creates all bones at Z=0 when bone nodes have
    # non-identity scale. We push the uniform scale into child translations:
    #   - Root bone: scale → [1,1,1], translation unchanged
    #   - All other bones: translation *= s, scale → [1,1,1]
    # This preserves bone world positions exactly (verified mathematically)
    # because for uniform scale s: S(s) @ T(t) = T(s*t) @ S(s), and
    # S(s) @ R(q) = R(q) @ S(s), so the scale factors telescope out.
    if _root_uniform_scale is not None and abs(_root_uniform_scale - 1.0) > 1e-6:
        s = _root_uniform_scale
        log.info("[soma_xfer] stripping uniform scale %.4f from bones", s)
        for i in range(n_joints_soma):
            t, q, _ = local_transforms[i]
            if joint_parents[i] is not None:
                # Non-root: scale translation by s
                t = [t_j * s for t_j in t]
            local_transforms[i] = (t, q, [1.0, 1.0, 1.0])
    else:
        # Still force scale to [1,1,1] for safety
        for i in range(n_joints_soma):
            t, q, _ = local_transforms[i]
            local_transforms[i] = (t, q, [1.0, 1.0, 1.0])

    # ── Recompute world matrices from modified local transforms ──────
    # Forward kinematics: world[i] = world[parent] @ local[i]
    bind_world = np.zeros((n_joints_soma, 4, 4), dtype=np.float64)
    for i in range(n_joints_soma):
        t, q, s = local_transforms[i]
        local_mat = _trs_to_mat4(t, q, s)
        parent_idx = joint_parents[i]
        if parent_idx is None:
            bind_world[i] = local_mat
        else:
            bind_world[i] = bind_world[parent_idx] @ local_mat

    # ── IBMs = inverse of recomputed bind world matrices ──
    # glTF stores matrices in COLUMN-MAJOR order. numpy is row-major, so
    # we must transpose each 4×4 matrix before serialising to bytes.
    # Without this transpose, Blender's glTF importer interprets the
    # row-major bytes as column-major, getting the transpose of the IBM,
    # which causes the rest-pose mesh to be scaled/deformed incorrectly.
    ibm = np.linalg.inv(bind_world).transpose(0, 2, 1).astype(np.float32)

    # ── IBM buffer (float32 mat4 × n_joints) ──────────────────────────
    ibm_bytes = ibm.astype(np.float32).tobytes()
    bv_ibm = add_bufferView(ibm_bytes)
    acc_ibm = add_accessor(bv_ibm, n_joints_soma, "MAT4", 5126)

    # Add bone nodes
    for i in range(n_joints_soma):
        name = joint_names_mix[i]
        t, q, s = local_transforms[i]
        node: dict[str, Any] = {
            "name": f"mixamorig:{name}",
            "translation": [float(x) for x in t],
            "rotation": [float(x) for x in q],
            "scale": [float(x) for x in s],
        }
        gltf.setdefault("nodes", []).append(node)
        bone_node_idxs.append(existing_node_count + i)

    # Wire up parent-child relationships (children arrays on parents)
    children_of: dict[int, list[int]] = {}
    for i, parent_idx in enumerate(joint_parents):
        if parent_idx is not None:
            children_of.setdefault(bone_node_idxs[parent_idx], []).append(
                bone_node_idxs[i])
    for parent_node, kids in children_of.items():
        gltf["nodes"][parent_node]["children"] = kids

    # Root bones (parent=None) become children of an "Armature" wrapper node.
    # This matches the Mixamo GLB structure and helps Blender's glTF importer
    # correctly map node translations to bone head positions.
    roots = [bone_node_idxs[i]
             for i, p in enumerate(joint_parents) if p is None]

    armature_node = {
        "name": "Armature",
        "translation": [0.0, 0.0, 0.0],
        "rotation": [0.0, 0.0, 0.0, 1.0],
        "scale": [1.0, 1.0, 1.0],
        "children": roots,
    }
    gltf.setdefault("nodes", []).append(armature_node)
    armature_node_idx = len(gltf["nodes"]) - 1

    # Attach armature to scene
    if "nodes" not in scenes[0]:
        scenes[0]["nodes"] = []
    if armature_node_idx not in scenes[0]["nodes"]:
        scenes[0]["nodes"].append(armature_node_idx)

    # Attach skin to the mesh node
    skin_idx = len(gltf.setdefault("skins", []))
    skin_dict: dict[str, Any] = {
        "joints": bone_node_idxs,
        "inverseBindMatrices": acc_ibm,
        "skeleton": armature_node_idx,
        "name": "SOMA-Mixamo",
    }
    # Embed T_align so the LBS server can transform Kimodo motion (in SOMA
    # template space) into the source-mesh space the IBM was built in.
    # Stored as row-major 16 floats under skin extras. Older servers that
    # don't know this field simply ignore it (and remain buggy for any GLB
    # that had non-identity T_align — i.e. nearly all of them).
    if t_align is not None:
        skin_dict["extras"] = {
            "somaT_alignRowMajor": [float(x) for x in np.asarray(t_align).reshape(16)],
        }
    gltf["skins"].append(skin_dict)
    # Find the mesh node (it's the existing one with a "mesh" attribute)
    mesh_node_idx = root_node
    if mesh_node_idx is None or "mesh" not in gltf["nodes"][mesh_node_idx]:
        for ni, node in enumerate(gltf.get("nodes", [])):
            if "mesh" in node:
                mesh_node_idx = ni
                break

    if mesh_node_idx is not None:
        # The mesh node references the skin (drives deformation)
        gltf["nodes"][mesh_node_idx]["skin"] = skin_idx

        # Per glTF 2.0 spec: when a node is skinned, its world transform is
        # ignored. Set identity to avoid any confusion.
        gltf["nodes"][mesh_node_idx]["translation"] = [0.0, 0.0, 0.0]
        gltf["nodes"][mesh_node_idx]["rotation"] = [0.0, 0.0, 0.0, 1.0]
        gltf["nodes"][mesh_node_idx]["scale"] = [1.0, 1.0, 1.0]

        # Ensure the mesh node is a DIRECT scene root — a sibling of the
        # Armature, NOT a child of any bone or wrapper. Blender's glTF
        # importer asserts that skinned mesh nodes are not bone descendants,
        # and Three.js requires the mesh to be reachable from exactly ONE
        # scene-graph position.
        #
        # CRITICAL: glTF forbids a node from being BOTH a child of another
        # node AND a direct scene root. If the source GLB wrapped its mesh
        # in a transform node (TRELLIS emits a "world" → "geometry_0"
        # hierarchy), we MUST detach the mesh from its parent before
        # promoting it to a scene root. Without this, the mesh appears in
        # TWO positions in the scene graph — Three.js creates a duplicate
        # SkinnedMesh for the orphaned scene-root instance, and that copy's
        # skeleton never binds (bones resolve to undefined), crashing with
        # "Cannot read properties of undefined (reading 'matrixWorld')" on
        # the first bounding-box / render call.
        for parent_node in gltf.get("nodes", []):
            kids = parent_node.get("children")
            if kids and mesh_node_idx in kids:
                parent_node["children"] = [c for c in kids if c != mesh_node_idx]
                if not parent_node["children"]:
                    del parent_node["children"]

        # The scene contains exactly two roots: Armature (holds the bone
        # hierarchy) and the mesh node (holds geometry + skin reference).
        # Any source wrapper nodes (e.g., TRELLIS's "world") are dropped —
        # they contributed only transform, which glTF ignores for skinned
        # meshes anyway.
        scenes[0]["nodes"] = [armature_node_idx, mesh_node_idx]

    # Update buffer
    gltf["buffers"][0]["byteLength"] = len(new_bin)

    # ── Optionally strip textures for a flat SOMA-blue mannequin ────────
    # When preserve_texture=False, replace every material with a flat
    # SOMA-blue PBR material. We drop all texture references from materials
    # but LEAVE the `textures`/`images` arrays untouched — they're now
    # unreferenced (dead bytes in the binary chunk), but removing them
    # would require compacting bufferViews/byteOffsets, which risks
    # corrupting offsets we just built above. The size cost (~1-3 MB for
    # a typical TRELLIS texture set) is acceptable; the GLB still loads
    # as a clean blue mannequin and the dead bytes are never decoded by
    # the renderer.
    if not preserve_texture:
        # SOMA-X signature blue — matches NVIDIA's SOMA demo renders.
        # Light enough to read silhouette clearly, saturated enough to
        # be obviously "the SOMA blue" against any background.
        SOMA_BLUE_RGBA = [0.55, 0.72, 0.92, 1.0]
        for mat in gltf.get("materials", []):
            pbr = mat.setdefault("pbrMetallicRoughness", {})
            # Strip all texture bindings — only baseColorFactor remains.
            pbr.pop("baseColorTexture", None)
            pbr.pop("metallicRoughnessTexture", None)
            pbr.pop("normalTexture", None)
            pbr.pop("occlusionTexture", None)
            pbr.pop("emissiveTexture", None)
            mat.pop("normalTexture", None)
            mat.pop("occlusionTexture", None)
            mat.pop("emissiveTexture", None)
            # Flat SOMA blue, fully rough, non-metallic.
            pbr["baseColorFactor"] = list(SOMA_BLUE_RGBA)
            pbr["metallicFactor"] = 0.0
            pbr["roughnessFactor"] = 0.85
            # Kill any emissive tint — mannequin should be lit purely by scene.
            mat["emissiveFactor"] = [0.0, 0.0, 0.0]
        n_mats = len(gltf.get("materials", []))
        log.info(
            "[soma_xfer] preserve_texture=False — stripped textures from %d "
            "material(s), applied flat SOMA-blue baseColorFactor", n_mats,
        )

    # ── Write GLB ──────────────────────────────────────────────────────
    json_bytes = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    json_bytes += b" " * ((4 - len(json_bytes) % 4) % 4)
    bin_padded = bytes(new_bin)
    bin_padded += b"\x00" * ((4 - len(bin_padded) % 4) % 4)

    total_length = 12 + 8 + len(json_bytes) + 8 + len(bin_padded)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(struct.pack("<III", 0x46546C67, 2, total_length))
        f.write(struct.pack("<II", len(json_bytes), 0x4E4F534A))
        f.write(json_bytes)
        f.write(struct.pack("<II", len(bin_padded), 0x004E4942))
        f.write(bin_padded)

    log.info("[soma_xfer] wrote GLB: %d verts, %d bones, %d KB",
             n_verts, n_joints_soma, os.path.getsize(output_path) // 1024)


def _trs_to_mat4(t: list[float], q: list[float], s: list[float]) -> np.ndarray:
    """Build a 4x4 matrix from TRS (translation, rotation xyzw, scale).

    M = Translate(t) @ Rotate(q) @ Scale(s)
    """
    x, y, z, w = q
    # Build rotation matrix from quaternion
    R = np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)],
    ], dtype=np.float64)
    # Apply scale
    R[:, 0] *= s[0]
    R[:, 1] *= s[1]
    R[:, 2] *= s[2]
    M = np.eye(4, dtype=np.float64)
    M[:3, :3] = R
    M[:3, 3] = t
    return M


def _mat3_to_quat_xyzw(R: np.ndarray) -> list[float]:
    """Convert a 3x3 rotation matrix to quaternion [x, y, z, w]."""
    m00, m01, m02 = R[0]
    m10, m11, m12 = R[1]
    m20, m21, m22 = R[2]
    trace = m00 + m11 + m22
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m21 - m12) * s
        y = (m02 - m20) * s
        z = (m10 - m01) * s
    elif m00 > m11 and m00 > m22:
        s = 2.0 * np.sqrt(1.0 + m00 - m11 - m22)
        w = (m21 - m12) / s
        x = 0.25 * s
        y = (m01 + m10) / s
        z = (m02 + m20) / s
    elif m11 > m22:
        s = 2.0 * np.sqrt(1.0 + m11 - m00 - m22)
        w = (m02 - m20) / s
        x = (m01 + m10) / s
        y = 0.25 * s
        z = (m12 + m21) / s
    else:
        s = 2.0 * np.sqrt(1.0 + m22 - m00 - m11)
        w = (m10 - m01) / s
        x = (m02 + m20) / s
        y = (m12 + m21) / s
        z = 0.25 * s
    # Normalize
    n = np.sqrt(x*x + y*y + z*z + w*w)
    return [x/n, y/n, z/n, w/n]


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: soma_weight_transfer.py <input.glb> <output.glb>")
        sys.exit(1)
    transfer_soma_weights_and_write_glb(sys.argv[1], sys.argv[2])
