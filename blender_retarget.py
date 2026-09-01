"""Blender headless retargeting: applies SOMA-77 motion data onto a
Mixamo-rigged character and exports an animated FBX or GLB.

Invoked by the KimodoExportFBX ComfyUI node:

    blender -b -P blender_retarget.py -- \
        <rigged_path> <npz_path> <output_path> <fps> <file_format>

Self-contained — uses only Blender's bundled Python (bpy, numpy, mathutils).
No dependency on the melite package or any pip-installed modules.

Algorithm: global rotation matrix retargeting (with direction-based fallback).

  When global_rot_mats is available in the NPZ (preferred):
    1. For each frame t, compute the delta rotation from frame 0:
       delta_soma = global_rots[t, j] @ global_rots[0, j]⁻¹
    2. Conjugate into Blender space: delta_blender = C @ delta_soma @ C⁻¹
    3. Compose desired world rotation: desired_quat = delta_blender @ rest_quat
       (world-space order — delta applied ON TOP of rest, not bone-local)
    4. Solve for matrix_basis using the parent-chain formula.

  When global_rot_mats is NOT available (fallback):
    1. Compute SOMA frame-0 rest direction and frame-t posed direction.
    2. Delta = rotation_difference(rest_dir, posed_dir).
    3. Same quaternion composition and basis solving.

  The matrix_basis parent-chain formula (same for both approaches):
    B = R⁻¹ @ R_parent @ P⁻¹ @ desired_world   (child)
    B = R⁻¹ @ desired_world                      (root)

  This approach is robust to garbled source skeletons (e.g. SkinToken's
  Extra_04..07 bones with wrong hierarchy/orientations). The delta from
  frame 0 captures only the MOTION, not the rest pose. At frame 0, all
  deltas are identity and the character appears at its bind pose.

Coordinate conversion SOMA → Blender armature space:
  SOMA:    +X left, +Y up,   +Z forward
  Blender: +X right, +Y forward, +Z up
  Conversion: (-x, z, y) — negate X, swap Y↔Z
  As a matrix: C = [[-1,0,0],[0,0,1],[0,1,0]]

Root translation: scaled delta from frame 0, using the armature's
Hips→Head distance vs the SOMA skeleton's.
"""
from __future__ import annotations
import sys
import os
import struct
import json

import bpy
import numpy as np
import mathutils


# ═══════════════════════════════════════════════════════════════════════════
# SOMA-77 skeleton topology (mirrors melite-poser-nodes/library/skeleton.py)
# ═══════════════════════════════════════════════════════════════════════════
SOMA77_PARENTS = (
    -1,   # 0  Hips
    0,    # 1  Spine1
    1,    # 2  Spine2
    2,    # 3  Chest
    3,    # 4  Neck1
    4,    # 5  Neck2
    5,    # 6  Head
    6,    # 7  HeadEnd
    6,    # 8  Jaw
    6,    # 9  LeftEye
    6,    # 10 RightEye
    3,    # 11 LeftShoulder
    11,   # 12 LeftArm
    12,   # 13 LeftForeArm
    13,   # 14 LeftHand
    14,   # 15 LeftHandThumb1
    15,   # 16 LeftHandThumb2
    16,   # 17 LeftHandThumb3
    17,   # 18 LeftHandThumbEnd
    14,   # 19 LeftHandIndex1
    19,   # 20 LeftHandIndex2
    20,   # 21 LeftHandIndex3
    21,   # 22 LeftHandIndex4
    22,   # 23 LeftHandIndexEnd
    14,   # 24 LeftHandMiddle1
    24,   # 25 LeftHandMiddle2
    25,   # 26 LeftHandMiddle3
    26,   # 27 LeftHandMiddle4
    27,   # 28 LeftHandMiddleEnd
    14,   # 29 LeftHandRing1
    29,   # 30 LeftHandRing2
    30,   # 31 LeftHandRing3
    31,   # 32 LeftHandRing4
    32,   # 33 LeftHandRingEnd
    14,   # 34 LeftHandPinky1
    34,   # 35 LeftHandPinky2
    35,   # 36 LeftHandPinky3
    36,   # 37 LeftHandPinky4
    37,   # 38 LeftHandPinkyEnd
    3,    # 39 RightShoulder
    39,   # 40 RightArm
    40,   # 41 RightForeArm
    41,   # 42 RightHand
    42,   # 43 RightHandThumb1
    43,   # 44 RightHandThumb2
    44,   # 45 RightHandThumb3
    45,   # 46 RightHandThumbEnd
    42,   # 47 RightHandIndex1
    47,   # 48 RightHandIndex2
    48,   # 49 RightHandIndex3
    49,   # 50 RightHandIndex4
    50,   # 51 RightHandIndexEnd
    42,   # 52 RightHandMiddle1
    52,   # 53 RightHandMiddle2
    53,   # 54 RightHandMiddle3
    54,   # 55 RightHandMiddle4
    55,   # 56 RightHandMiddleEnd
    42,   # 57 RightHandRing1
    57,   # 58 RightHandRing2
    58,   # 59 RightHandRing3
    59,   # 60 RightHandRing4
    60,   # 61 RightHandRingEnd
    42,   # 62 RightHandPinky1
    62,   # 63 RightHandPinky2
    63,   # 64 RightHandPinky3
    64,   # 65 RightHandPinky4
    65,   # 66 RightHandPinkyEnd
    0,    # 67 LeftLeg (thigh)
    67,   # 68 LeftShin
    68,   # 69 LeftFoot
    69,   # 70 LeftToeBase
    70,   # 71 LeftToeEnd
    0,    # 72 RightLeg (thigh)
    72,   # 73 RightShin
    73,   # 74 RightFoot
    74,   # 75 RightToeBase
    75,   # 76 RightToeEnd
)

# SOMA-77 joint index → Mixamo bone name (without prefix).
# SkinToken's "Mixamo" template emits bones with `mixamorig:` prefix; we try
# both. Finger bones are included but optional — standard Mixamo rigs may
# not have them. The retargeter skips any bone not present in the armature.
SOMA_TO_MIXAMO = {
    # Spine chain
    0: "Hips",
    1: "Spine",
    2: "Spine1",
    3: "Spine2",
    4: "Neck",
    5: "Neck1",
    6: "Head",
    # Left arm
    11: "LeftShoulder",
    12: "LeftArm",
    13: "LeftForeArm",
    14: "LeftHand",
    # Right arm
    39: "RightShoulder",
    40: "RightArm",
    41: "RightForeArm",
    42: "RightHand",
    # Left leg
    67: "LeftUpLeg",
    68: "LeftLeg",
    69: "LeftFoot",
    70: "LeftToeBase",
    # Right leg
    72: "RightUpLeg",
    73: "RightLeg",
    74: "RightFoot",
    75: "RightToeBase",
    # Fingers (optional — present only in "with fingers" Mixamo rigs)
    15: "LeftHandThumb1",  16: "LeftHandThumb2",  17: "LeftHandThumb3",
    19: "LeftHandIndex1",  20: "LeftHandIndex2",  21: "LeftHandIndex3",
    24: "LeftHandMiddle1", 25: "LeftHandMiddle2", 26: "LeftHandMiddle3",
    29: "LeftHandRing1",   30: "LeftHandRing2",   31: "LeftHandRing3",
    34: "LeftHandPinky1",  35: "LeftHandPinky2",  36: "LeftHandPinky3",
    43: "RightHandThumb1",  44: "RightHandThumb2",  45: "RightHandThumb3",
    47: "RightHandIndex1",  48: "RightHandIndex2",  49: "RightHandIndex3",
    52: "RightHandMiddle1", 53: "RightHandMiddle2", 54: "RightHandMiddle3",
    57: "RightHandRing1",   58: "RightHandRing2",   59: "RightHandRing3",
    62: "RightHandPinky1",  63: "RightHandPinky2",  64: "RightHandPinky3",
}


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════
def _resolve_bone(pose_bones, name: str):
    """Find a pose bone by name, trying with `mixamorig:` prefix."""
    if name in pose_bones:
        return name
    prefixed = f"mixamorig:{name}"
    if prefixed in pose_bones:
        return prefixed
    return None


def _topo_order(bone_indices: dict[int, str]) -> list[int]:
    """Topologically sort SOMA indices (parent before child)."""
    visited: set[int] = set()
    order: list[int] = []

    def visit(idx: int):
        if idx in visited:
            return
        parent = SOMA77_PARENTS[idx]
        if parent >= 0 and parent in bone_indices:
            visit(parent)
        visited.add(idx)
        order.append(idx)

    for idx in bone_indices:
        visit(idx)
    return order


def _soma_to_armature(arr: np.ndarray) -> np.ndarray:
    """Convert SOMA-space coordinates to Blender armature-local space.

    Matches the glTF importer convention (verified empirically from
    bind_vertices imported via GLB):
      Blender_X =  SOMA_X    (no flip)
      Blender_Y = -SOMA_Z    (negate depth)
      Blender_Z =  SOMA_Y    (height)
    """
    out = np.empty_like(arr)
    out[..., 0] = arr[..., 0]    # X: same
    out[..., 1] = -arr[..., 2]   # Y_blender = -Z_soma
    out[..., 2] = arr[..., 1]    # Z_blender = Y_soma
    return out


# SOMA→Blender 3×3 conversion matrix (same transform as glTF importer).
# Verified empirically: SOMA bind_vertices imported via GLB into Blender
# satisfy:
#   Blender_X =  SOMA_X    (no flip)
#   Blender_Y = -SOMA_Z    (flip depth)
#   Blender_Z =  SOMA_Y    (height)
SOMA_TO_BLENDER_C = np.array([
    [ 1.0,  0.0, 0.0],
    [ 0.0,  0.0, -1.0],
    [ 0.0,  1.0, 0.0],
], dtype=np.float64)


# ═══════════════════════════════════════════════════════════════════════════
# Custom animated GLB writer — bypasses Blender 4.2's broken glTF exporter
# ═══════════════════════════════════════════════════════════════════════════

def _mat3_to_quat_xyzw(R):
    """3x3 rotation matrix → quaternion [x,y,z,w]."""
    m00, m01, m02 = R[0]; m10, m11, m12 = R[1]; m20, m21, m22 = R[2]
    trace = m00 + m11 + m22
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s; x = (m21 - m12) * s; y = (m02 - m20) * s; z = (m10 - m01) * s
    elif m00 > m11 and m00 > m22:
        s = 2.0 * np.sqrt(1.0 + m00 - m11 - m22)
        w = (m21 - m12) / s; x = 0.25 * s; y = (m01 + m10) / s; z = (m02 + m20) / s
    elif m11 > m22:
        s = 2.0 * np.sqrt(1.0 + m11 - m00 - m22)
        w = (m02 - m20) / s; x = (m01 + m10) / s; y = 0.25 * s; z = (m12 + m21) / s
    else:
        s = 2.0 * np.sqrt(1.0 + m22 - m00 - m11)
        w = (m10 - m01) / s; x = (m02 + m20) / s; y = (m12 + m21) / s; z = 0.25 * s
    n = np.sqrt(x * x + y * y + z * z + w * w)
    return [float(x / n), float(y / n), float(z / n), float(w / n)]


def _read_glb(path: str) -> tuple[dict, bytes]:
    """Parse GLB into (gltf_json, binary_buffer)."""
    with open(path, "rb") as f:
        header = f.read(12)
        magic, version, length = struct.unpack("<III", header)
        assert magic == 0x46546C67, f"Bad GLB magic 0x{magic:x}"
        bin_data = bytearray()
        offset = 12; gltf = None
        while offset < length:
            clen, ctype = struct.unpack("<II", f.read(8))
            chunk = f.read(clen)
            if ctype == 0x4E4F534A:
                gltf = json.loads(chunk.decode("utf-8"))
            elif ctype == 0x004E4942:
                bin_data.extend(chunk)
            offset += 8 + clen
        return gltf, bytes(bin_data)


# Z-up (Blender) → Y-up (glTF) conversion matrix.
# glTF: +Y up, +Z forward. Blender: +Z up, -Y forward.
# M = Rx(+90°): rotates Z-up to Y-up.
ZUP_TO_YUP = np.array([
    [1, 0, 0, 0],
    [0, 0, -1, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
], dtype=np.float64)


def _write_animated_glb(
    source_glb_path: str,
    output_path: str,
    bone_map: dict[int, str],              # soma_idx → resolved bone name
    rest_mats: dict[int, object],          # soma_idx → mathutils.Matrix
    all_pose_world: list[dict[str, np.ndarray]],  # per-frame {bone_name: 4x4}
    fps: int,
) -> str:
    """Write animated GLB by augmenting source GLB with animation data.

    Computes local TRS values directly from world pose matrices, bypassing
    Blender's glTF exporter (which corrupts animated bone rotations).
    """
    import json as _json

    T = len(all_pose_world)
    # Build ordered list of bones (sorted by SOMA index for stable ordering)
    sorted_indices = sorted(bone_map.keys())
    bone_names_ordered = [bone_map[si] for si in sorted_indices]

    # ── Read source GLB ──────────────────────────────────────────────────
    gltf, src_bin = _read_glb(source_glb_path)
    gltf = _json.loads(_json.dumps(gltf))  # deep copy
    new_bin = bytearray(src_bin)

    def _pad4():
        while len(new_bin) % 4:
            new_bin.append(0)

    def _add_bv(data_bytes):
        _pad4()
        off = len(new_bin)
        new_bin.extend(data_bytes)
        gltf.setdefault("bufferViews", []).append(
            {"buffer": 0, "byteOffset": off, "byteLength": len(data_bytes)})
        return len(gltf["bufferViews"]) - 1

    def _add_acc(bv_idx, count, atype, comp, mins=None, maxs=None):
        acc = {"bufferView": bv_idx, "componentType": comp, "count": count, "type": atype}
        if mins is not None: acc["min"] = mins
        if maxs is not None: acc["max"] = maxs
        gltf.setdefault("accessors", []).append(acc)
        return len(gltf["accessors"]) - 1

    # ── Find bone node indices ───────────────────────────────────────────
    node_names = [n.get("name", "") for n in gltf.get("nodes", [])]
    bone_node_idx: list[int | None] = []
    for bn in bone_names_ordered:
        if bn in node_names:
            bone_node_idx.append(node_names.index(bn))
        else:
            bone_node_idx.append(None)

    # ── Build GLB node parent map from ACTUAL hierarchy ──────────────────
    # We use the GLB's real node hierarchy (not SOMA77_PARENTS) because
    # SOMA parents may not be in bone_map (e.g. Neck1 missing from the
    # rigged GLB), causing children to be treated as roots — which
    # corrupts local TRS computation.
    all_nodes = gltf.get("nodes", [])
    scenes = gltf.get("scenes", [])
    root_nodes = scenes[0].get("nodes", []) if scenes else []
    parent_node: dict[int, int | None] = {}
    for i, n in enumerate(all_nodes):
        for child in n.get("children", []):
            parent_node[child] = i
    for rn in root_nodes:
        parent_node.setdefault(rn, None)

    def _trs_to_matrix(node):
        """Convert a GLB node's rest TRS to a 4x4 matrix (Y-up)."""
        T_ = node.get("translation", [0, 0, 0])
        R_ = node.get("rotation", [0, 0, 0, 1])
        S_ = node.get("scale", [1, 1, 1])
        m = np.eye(4, dtype=np.float64)
        m[:3, 3] = T_
        x, y, z, w = R_
        m[:3, :3] = np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
            [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
            [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)],
        ], dtype=np.float64)
        m[:3, 0] *= S_[0]; m[:3, 1] *= S_[1]; m[:3, 2] *= S_[2]
        return m

    # Compute rest world matrices for ALL GLB nodes (Y-up).
    rest_world_glb: dict[int, np.ndarray] = {}
    def _compute_rest_world(ni, parent_w):
        w = parent_w @ _trs_to_matrix(all_nodes[ni])
        rest_world_glb[ni] = w
        for child in all_nodes[ni].get("children", []):
            _compute_rest_world(child, w)
    for rn in root_nodes:
        _compute_rest_world(rn, np.eye(4, dtype=np.float64))

    # ── Calibrate: determine the Blender-armature → GLB-Y-up conversion ─
    # For each mapped bone, compare its Blender rest matrix (Z-up armature)
    # with its GLB rest world (Y-up). The ratio is the conversion, which
    # should be the same for all bones (= armature world transform ×
    # Y-up/Z-up rotation).
    bone_name_to_si = {v: k for k, v in bone_map.items()}
    name_to_node = {n.get("name", f"node_{i}"): i for i, n in enumerate(all_nodes)}
    node_to_bone_name = {v: k for k, v in name_to_node.items() if k in bone_name_to_si}

    conversions = []
    for bi, si in enumerate(sorted_indices):
        ni = bone_node_idx[bi]
        if ni is None:
            continue
        glb_w = rest_world_glb.get(ni)
        bl_rest = np.array(rest_mats[si], dtype=np.float64)
        if glb_w is not None:
            conv = glb_w @ np.linalg.inv(bl_rest)
            conversions.append(conv)

    if conversions:
        # Use median conversion (robust to outliers)
        conv_ref = conversions[len(conversions) // 2]
        # Check consistency
        max_dev = 0.0
        for c in conversions:
            diff = np.linalg.norm(c[:3, 3] - conv_ref[:3, 3])
            rot_diff = np.linalg.norm(c[:3, :3] - conv_ref[:3, :3])
            max_dev = max(max_dev, diff, rot_diff)
        print(f"[retarget] GLB calibration: {len(conversions)} bones, "
              f"max deviation={max_dev:.4f} "
              f"({'consistent' if max_dev < 0.01 else 'INCONSISTENT'})")
    else:
        conv_ref = np.eye(4, dtype=np.float64)
        print("[retarget] GLB calibration: no bones to calibrate!")

    np.linalg.inv(conv_ref)

    # ── Time accessor (shared by all samplers) ───────────────────────────
    times = np.arange(T, dtype=np.float32) / fps
    bv_time = _add_bv(times.tobytes())
    acc_time = _add_acc(bv_time, T, "SCALAR", 5126,
                        mins=[float(times[0])], maxs=[float(times[-1])])

    # ── Compute local TRS for each bone at each frame ────────────────────
    samplers = []
    channels = []

    for bi, si in enumerate(sorted_indices):
        node_idx = bone_node_idx[bi]
        if node_idx is None:
            continue

        bone_name = bone_names_ordered[bi]

        # Find ACTUAL parent from GLB hierarchy
        p_node_idx = parent_node.get(node_idx)

        translations = np.zeros((T, 3), dtype=np.float32)
        rotations = np.zeros((T, 4), dtype=np.float32)
        scales = np.ones((T, 3), dtype=np.float32)

        for t in range(T):
            # Convert bone pose from Blender armature (Z-up) to GLB (Y-up).
            # LEFT-MULTIPLY by conv_ref — NOT conjugation. The calibration
            # shows conv_ref = inv(P_root) where P_root is Blender's root
            # rotation. At frame 0, conv_ref @ bl_rest = glb_rest ✓.
            bone_bl = all_pose_world[t][bone_name]
            bone_yup = conv_ref @ bone_bl

            # Find parent world in GLB Y-up
            if p_node_idx is not None and p_node_idx in node_to_bone_name:
                # Parent is a mapped bone — use its animated pose
                parent_name = node_to_bone_name[p_node_idx]
                parent_bl = all_pose_world[t][parent_name]
                parent_yup = conv_ref @ parent_bl
            elif p_node_idx is not None:
                # Parent is an unmapped bone — use rest world from GLB
                parent_yup = rest_world_glb[p_node_idx]
            else:
                parent_yup = np.eye(4, dtype=np.float64)

            # Compute local transform in GLB Y-up
            local = np.linalg.inv(parent_yup) @ bone_yup

            translations[t] = local[:3, 3].astype(np.float32)
            rotations[t] = np.array(_mat3_to_quat_xyzw(local[:3, :3]), dtype=np.float32)
            sx = float(np.linalg.norm(local[:3, 0]))
            sy = float(np.linalg.norm(local[:3, 1]))
            sz = float(np.linalg.norm(local[:3, 2]))
            scales[t] = [sx, sy, sz]

        for data, atype, n_comp in [
            (translations, "VEC3", 3),
            (rotations, "VEC4", 4),
            (scales, "VEC3", 3),
        ]:
            bv = _add_bv(data.tobytes())
            acc = _add_acc(bv, T, atype, 5126,
                           mins=data.min(axis=0).tolist(),
                           maxs=data.max(axis=0).tolist())
            # Determine path from data type
            if atype == "VEC4":
                path = "rotation"
            elif data is translations:
                path = "translation"
            else:
                path = "scale"
            si_samp = len(samplers)
            samplers.append({"input": acc_time, "output": acc, "interpolation": "LINEAR"})
            channels.append({"sampler": si_samp,
                             "target": {"node": node_idx, "path": path}})

    # ── Add animation ────────────────────────────────────────────────────
    gltf.setdefault("animations", []).append({
        "name": "KimodoRetarget",
        "samplers": samplers,
        "channels": channels,
    })

    # ── Write GLB ────────────────────────────────────────────────────────
    gltf["buffers"][0]["byteLength"] = len(new_bin)
    json_bytes = _json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    json_bytes += b" " * ((4 - len(json_bytes) % 4) % 4)
    _pad4()
    bin_padded = bytes(new_bin)
    bin_padded += b"\x00" * ((4 - len(bin_padded) % 4) % 4)

    total_length = 12 + 8 + len(json_bytes) + 8 + len(bin_padded)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(struct.pack("<III", 0x46546C67, 2, total_length))
        f.write(struct.pack("<II", len(json_bytes), 0x4E4F534A))
        f.write(json_bytes)
        f.write(struct.pack("<II", len(bin_padded), 0x004E4942))
        f.write(bin_padded)

    print(f"[retarget] Custom GLB writer: {len(channels) // 3} bones × {T} frames")
    return output_path


def _fix_glb_if_needed(glb_path: str) -> str:
    """Detect and fix mesh/bone coordinate space mismatch in a rigged GLB.

    The GLB writer (soma_weight_transfer._write_rigged_glb) writes:
    - Bones (node hierarchy) in SOMA Y-up space
    - Mesh POSITION/NORMAL from the source GLB (TRELLIS), often in Z-up
    - IBM (inverse bind matrices) already correct as inv(bind_world_yup)

    When mesh is Z-up but bones/IBM are Y-up, Blender's glTF importer
    converts bones Y-up→Z-up correctly but the mesh was already in a
    Z-up-like space, causing a double conversion. This produces severe
    body horror.

    Fix: Convert mesh POSITION/NORMAL from Z-up to Y-up so they match
    the bones. The IBM is left UNCHANGED — it is already inv(bind_world)
    in Y-up space, which is correct for Y-up mesh vertices.

    Returns the path to a fixed GLB, or the original path if no fix needed.
    """
    ext = os.path.splitext(glb_path)[1].lower()
    if ext not in (".glb", ".gltf"):
        return glb_path

    try:
        gltf, src_bin = _read_glb(glb_path)
    except Exception as e:
        print(f"[retarget] GLB fix: could not read GLB ({e}), skipping")
        return glb_path

    meshes = gltf.get("meshes", [])
    if not meshes:
        return glb_path

    # Check first mesh primitive's POSITION accessor
    pri = meshes[0].get("primitives", [{}])[0]
    attrs = pri.get("attributes", {})
    pos_acc_idx = attrs.get("POSITION")
    if pos_acc_idx is None:
        return glb_path

    pos_acc = gltf.get("accessors", [])[pos_acc_idx]
    pos_min = pos_acc.get("min", [0, 0, 0])
    pos_max = pos_acc.get("max", [0, 0, 0])
    spreads = [pos_max[i] - pos_min[i] for i in range(3)]
    height_axis = max(range(3), key=lambda i: spreads[i])

    if height_axis == 1:  # Y is already height (Y-up = standard glTF)
        print(f"[retarget] GLB mesh already Y-up (spreads: "
              f"X={spreads[0]:.3f} Y={spreads[1]:.3f} Z={spreads[2]:.3f}) — no fix needed")
        return glb_path

    if height_axis != 2:  # Not Z-up either — unknown, leave alone
        print(f"[retarget] GLB mesh height axis is {['X','Y','Z'][height_axis]} — unexpected, skipping fix")
        return glb_path

    # ── Fix needed: Z-up → Y-up ──
    # Conversion: (x, y, z) → (x, z, -y)  [+Z up → +Y up]
    print(f"[retarget] GLB coordinate fix: Z-up → Y-up "
          f"(spreads: X={spreads[0]:.3f} Y={spreads[1]:.3f} Z={spreads[2]:.3f})")

    new_bin = bytearray(src_bin)
    buffer_views = gltf.get("bufferViews", [])
    accessors = gltf.get("accessors", [])

    def _read_accessor_data(acc_idx):
        """Read float32 data from accessor."""
        acc = accessors[acc_idx]
        bv = buffer_views[acc["bufferView"]]
        offset = bv.get("byteOffset", 0) + acc.get("byteOffset", 0)
        count = acc["count"]
        atype = acc["type"]
        n_comp = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}.get(atype, 1)
        comp = acc["componentType"]
        if comp != 5126:  # FLOAT
            return None, None, None
        nbytes = count * n_comp * 4
        data = np.frombuffer(bytes(new_bin[offset:offset + nbytes]),
                              dtype=np.float32).reshape(count, n_comp).copy()
        return data, offset, nbytes

    # Fix POSITION: (x, y, z) → (x, z, -y) for all mesh primitives
    for mesh in meshes:
        for pri in mesh.get("primitives", []):
            attrs = pri.get("attributes", {})

            # POSITION
            pos_idx = attrs.get("POSITION")
            if pos_idx is not None:
                data, offset, nbytes = _read_accessor_data(pos_idx)
                if data is not None:
                    new_data = np.column_stack([
                        data[:, 0], data[:, 2], -data[:, 1]
                    ]).astype(np.float32)
                    new_bin[offset:offset + nbytes] = new_data.tobytes()
                    # Update accessor min/max
                    accessors[pos_idx]["min"] = [
                        float(new_data[:, i].min()) for i in range(3)]
                    accessors[pos_idx]["max"] = [
                        float(new_data[:, i].max()) for i in range(3)]

            # NORMAL (vectors, same rotation)
            nrm_idx = attrs.get("NORMAL")
            if nrm_idx is not None:
                data, offset, nbytes = _read_accessor_data(nrm_idx)
                if data is not None:
                    new_data = np.column_stack([
                        data[:, 0], data[:, 2], -data[:, 1]
                    ]).astype(np.float32)
                    new_bin[offset:offset + nbytes] = new_data.tobytes()

    # TANGENT (4-component, rotate the xyz part, keep w)
    # We skip TANGENT — rare in these GLBs and rarely causes visible issues

    # Write fixed GLB
    fixed_path = glb_path + ".fixed.glb"
    json_bytes = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    json_bytes += b" " * ((4 - len(json_bytes) % 4) % 4)
    while len(new_bin) % 4:
        new_bin.append(0)

    total_length = 12 + 8 + len(json_bytes) + 8 + len(new_bin)
    with open(fixed_path, "wb") as f:
        f.write(struct.pack("<III", 0x46546C67, 2, total_length))
        f.write(struct.pack("<II", len(json_bytes), 0x4E4F534A))
        f.write(json_bytes)
        f.write(struct.pack("<II", len(new_bin), 0x004E4942))
        f.write(bytes(new_bin))

    print(f"[retarget] Fixed GLB: {fixed_path}")
    return fixed_path


# ═══════════════════════════════════════════════════════════════════════════
# Bone repositioning — fix Blender's broken glTF bone import
# ═══════════════════════════════════════════════════════════════════════════

def _reposition_bones_from_glb(glb_path: str, armature_obj):
    """Read bone world positions from the GLB's IBM and reposition Blender bones.

    Blender's glTF importer creates all bones at the armature origin, ignoring
    node translations. This function reads the inverse bind matrices from the
    GLB, computes world positions, and repositions each bone's head/tail.

    Must be called AFTER glTF import and BEFORE retarget logic.
    """
    import struct
    import json
    import mathutils

    # Read GLB
    with open(glb_path, 'rb') as f:
        f.read(12)  # header
        jl, _ = struct.unpack('<II', f.read(8))
        gltf_json = json.loads(f.read(jl).decode())

    # Find skin + IBM
    skins = gltf_json.get("skins", [])
    if not skins:
        print("[retarget] No skin in GLB — skipping bone repositioning")
        return

    skin = skins[0]
    joint_node_idxs = skin.get("joints", [])
    ibm_acc_idx = skin.get("inverseBindMatrices")
    if ibm_acc_idx is None:
        return

    # Read IBM binary
    ibm_acc = gltf_json["accessors"][ibm_acc_idx]
    ibm_bv = gltf_json["bufferViews"][ibm_acc["bufferView"]]
    ibm_off = ibm_bv.get("byteOffset", 0) + ibm_acc.get("byteOffset", 0)
    n_joints = ibm_acc["count"]

    with open(glb_path, 'rb') as f:
        f.read(12)
        jl2, _ = struct.unpack('<II', f.read(8))
        f.read(jl2 + ((4 - jl2 % 4) % 4))
        bl2, _ = struct.unpack('<II', f.read(8))
        bin_data = f.read(bl2)

    # glTF stores matrices in COLUMN-MAJOR order. numpy reads row-major, so
    # we must transpose each 4×4 to get the actual matrix.
    ibm_np = np.frombuffer(
        bin_data[ibm_off:ibm_off + n_joints * 64],
        dtype=np.float32
    ).reshape(n_joints, 4, 4).astype(np.float64)
    ibm_np = ibm_np.transpose(0, 2, 1)  # column-major → row-major

    # bind_world = inv(IBM)
    bind_world = np.linalg.inv(ibm_np)

    # Get joint names from nodes
    nodes = gltf_json.get("nodes", [])
    joint_names = []
    for jidx in joint_node_idxs:
        name = nodes[jidx].get("name", f"node_{jidx}")
        joint_names.append(name)

    # Convert positions from glTF Y-up to Blender Z-up: (x, y, z) → (x, -z, y)
    joint_positions = {}  # name → Blender Vector
    for i, name in enumerate(joint_names):
        pos_gltf = bind_world[i][:3, 3]
        bl_pos = mathutils.Vector((pos_gltf[0], -pos_gltf[2], pos_gltf[1]))
        joint_positions[name] = bl_pos

    # Build parent → children map from node hierarchy
    joint_children = {}  # name → [child names]
    for i, jidx in enumerate(joint_node_idxs):
        node = nodes[jidx]
        name = joint_names[i]
        children = []
        for child_idx in node.get("children", []):
            # Find child in joint list
            if child_idx in joint_node_idxs:
                child_pos = joint_node_idxs.index(child_idx)
                children.append(joint_names[child_pos])
        joint_children[name] = children

    # Y-up → Z-up conversion matrix for rotations
    P_conv = np.array([
        [1, 0, 0],
        [0, 0, -1],
        [0, 1, 0]
    ], dtype=np.float64)
    P_conv_inv = np.linalg.inv(P_conv)

    # Compute desired Z-axis for each bone (for roll alignment)
    joint_z_axes = {}  # name → Blender Vector (desired Z direction)
    for i, name in enumerate(joint_names):
        bw = bind_world[i]
        # Extract 3x3 rotation
        R_glb = bw[:3, :3]
        # Convert to Blender Z-up: R_blender = P @ R_glb @ P_inv
        R_blender = P_conv @ R_glb @ P_conv_inv
        # Z axis = third column
        z_axis = mathutils.Vector((R_blender[0, 2], R_blender[1, 2], R_blender[2, 2]))
        joint_z_axes[name] = z_axis

    # Enter edit mode and reposition
    bpy.context.view_layer.objects.active = armature_obj
    bpy.ops.object.mode_set(mode='EDIT')
    edit_bones = armature_obj.data.edit_bones

    repositioned = 0
    for eb in edit_bones:
        name = eb.name
        if name not in joint_positions:
            continue

        head = joint_positions[name]
        eb.head = head

        # Tail: first child's head position
        tail_set = False
        for child_name in joint_children.get(name, []):
            if child_name in joint_positions:
                child_head = joint_positions[child_name]
                if (child_head - head).length > 1e-6:
                    eb.tail = child_head
                    tail_set = True
                    break

        if not tail_set:
            # Leaf bone: small offset along desired Z axis
            z = joint_z_axes.get(name, mathutils.Vector((0, 0, 1)))
            eb.tail = head + z.normalized() * 0.02

        # Set roll from bind_world rotation (critical for correct deformation)
        if name in joint_z_axes:
            try:
                eb.align_roll(joint_z_axes[name])
            except Exception:
                pass  # align_roll can fail for degenerate bones

        repositioned += 1

    bpy.ops.object.mode_set(mode='OBJECT')

    # Clear pose transforms so the character is in the new rest pose
    bpy.ops.object.mode_set(mode='POSE')
    bpy.ops.pose.select_all(action='SELECT')
    try:
        bpy.ops.pose.transforms_clear()
    except Exception:
        for pb in armature_obj.pose.bones:
            pb.location = (0, 0, 0)
            pb.rotation_quaternion = (1, 0, 0, 0)
            pb.scale = (1, 1, 1)
    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.context.view_layer.update()

    print(f"[retarget] Repositioned {repositioned} bones from GLB IBM (with roll alignment)")


# ═══════════════════════════════════════════════════════════════════════════
# Main retargeting routine
# ═══════════════════════════════════════════════════════════════════════════
def retarget(rigged_path: str, npz_path: str, output_path: str,
             fps: int, file_format: str, in_place: bool = False,
             decimate_ratio: float = 0.0) -> str:
    T_fmt = file_format.lower().strip()
    print(f"[retarget] rigged={rigged_path}")
    print(f"[retarget] npz={npz_path}")
    print(f"[retarget] output={output_path} ({T_fmt}, {fps} fps)")
    if in_place:
        print("[retarget] IN-PLACE mode: root horizontal translation zeroed")

    # ── Load NPZ motion data ────────────────────────────────────────────
    data = np.load(npz_path, allow_pickle=True)
    if "posed_joints" not in data:
        raise KeyError(
            f"NPZ missing 'posed_joints' key. Available: {list(data.keys())}"
        )
    posed_joints_soma = data["posed_joints"].astype(np.float64)  # (T, 77, 3)
    T = posed_joints_soma.shape[0]
    n_joints = posed_joints_soma.shape[1]
    if n_joints < 77:
        print(f"[retarget] WARNING: NPZ has {n_joints} joints, expected 77. "
              f"Some bones will be skipped.")
    print(f"[retarget] {T} frames, {n_joints} joints")

    # ── In-place mode: zero root horizontal translation ─────────────────
    # Keeps the character centered (no forward drift) while preserving
    # vertical bob and all limb articulation. The character "walks on a
    # treadmill" — feet move but the body stays put.
    if in_place:
        root_pos = posed_joints_soma[:, 0, :].copy()  # (T, 3)
        root_disp = root_pos - root_pos[0:1, :]       # displacement from f0
        # Subtract X (sideways) and Z (forward/back) drift from ALL joints.
        # Keep Y (vertical bob) — don't subtract it.
        root_disp[:, 1] = 0.0
        posed_joints_soma -= root_disp[:, None, :]    # broadcast to all joints

    # Convert to armature-local space (flip X and Z)
    posed = _soma_to_armature(posed_joints_soma)

    # ── Load global rotation matrices for accurate retargeting ──────────
    # global_rot_mats (T, 77, 3, 3) gives the full world-space rotation of
    # each SOMA joint at each frame. Using these avoids relying on Blender's
    # bone directions (which may be wrong if the source skeleton has a
    # garbled hierarchy — e.g. SkinToken's Extra_04..07 bones).
    #
    # The delta rotation from frame 0 captures how each joint moved relative
    # to the rest pose. This is coordinate-space independent at the delta
    # level: we conjugate by the SOMA→Blender conversion matrix C.
    use_rot_mats = "global_rot_mats" in data
    if use_rot_mats:
        global_rots = data["global_rot_mats"].astype(np.float64)  # (T, 77, 3, 3)
        print("[retarget] Using global_rot_mats for rotation retargeting")

    # ── Load SOMA T-pose rotations ──────────────────────────────────────
    # CRITICAL: global_rots[0] is the FIRST FRAME of the animation (usually
    # an A-pose or mid-stride), NOT the T-pose. The Blender rig's rest pose
    # IS the T-pose. Using frame 0 as the delta reference introduces a
    # systematic error equal to the T-pose→frame-0 rotation difference
    # (up to 84° for arms, 34° for legs), producing severe body horror.
    #
    # We load the true T-pose global rotations from bind_rig_transform in
    # skin_standard.npz (the SOMA-77 template body's bind pose).
    tpose_rots = None
    soma_skin_path = "/opt/kimodo/kimodo/assets/skeletons/somaskel77/skin_standard.npz"
    if use_rot_mats and os.path.isfile(soma_skin_path):
        try:
            soma_skin = np.load(soma_skin_path, allow_pickle=True)
            bind_rig = soma_skin["bind_rig_transform"]  # (77, 4, 4)
            tpose_rots = bind_rig[:, :3, :3].astype(np.float64)  # (77, 3, 3)
            tpose_pos = bind_rig[:, :3, 3].astype(np.float64)  # (77, 3)
            print(f"[retarget] Loaded T-pose rotations from {soma_skin_path}")
        except Exception as e:
            print(f"[retarget] WARNING: Could not load T-pose: {e}")
    if tpose_rots is None and use_rot_mats:
        print("[retarget] WARNING: T-pose not found, falling back to frame 0 "
              "(may produce body horror)")
        tpose_rots = global_rots[0]  # fallback to old behavior
        tpose_pos = posed_joints_soma[0]  # fallback positions

    # ── Forward kinematics: recompute posed positions using ABSOLUTE
    #    rotations from T-pose ───────────────────────────────────────────
    # The NPZ walk cycle starts in A-pose (arms at sides), NOT T-pose.
    # The Blender bind pose IS T-pose (arms horizontal). We must compute
    # the ABSOLUTE rotation from T-pose for each joint, not the delta from
    # frame 0. This ensures the T-pose→A-pose offset is applied, bringing
    # the arms down to the sides.
    #
    # D_rot = global_rots[t] @ inv(tpose_rots) — absolute from T-pose
    # At frame 0: D_rot ≠ I (because NPZ starts in A-pose, not T-pose)
    # The mesh will be in A-pose at frame 0, which is CORRECT.
    if use_rot_mats and tpose_rots is not None:

        # ── Spine local rotation dampening ──────────────────────────────
        # The NPZ walk cycle has +7-12° local EXTENSION at the Neck joint
        # (relative to Chest), producing the "severe backward leaning" that
        # MIMO detects. The lower spine (L5, L3, Chest) has <2.5° local
        # rotation — that's natural. But the Neck's 12° extension pushes
        # the head backward/upward in an unnatural way.
        #
        # Fix: dampen the LOCAL rotation delta (relative to parent) for
        # upper spine joints. Process in order so dampened parent rotations
        # propagate correctly to children.
        #
        # Dampening map: joint_index → factor (0=freeze, 1=original)
        # NOTE: Arms (11,12,39,40) REMOVED — they need the full T-pose→A-pose
        # offset (~76°) to bring hands to the sides. Dampening would prevent
        # the arms from dropping from T-pose.
        SPINE_DAMPEN = {
            4: 0.25,   # Neck — reduce 12° ext → ~3°
            5: 0.40,   # Neck1
            6: 0.50,   # Head — keep some head bob
            # Hips: NPZ has -53° extension (normal walk is ±25°). Dampen to ~half.
            67: 0.55,  # L.UpLeg (hip) — reduce extreme stride
            68: 0.65,  # L.Knee — moderate
            72: 0.55,  # R.UpLeg (hip)
            73: 0.65,  # R.Knee
        }

        n_joints_rt = min(n_joints, len(SOMA77_PARENTS))
        for t in range(T):
            for j in sorted(SPINE_DAMPEN.keys()):
                if j >= n_joints_rt:
                    continue
                parent = SOMA77_PARENTS[j]
                if parent < 0 or parent >= n_joints_rt:
                    continue
                factor = SPINE_DAMPEN[j]
                # Current absolute rotations from T-POSE (not frame 0)
                # This ensures the T-pose→A-pose offset is preserved
                p_delta = global_rots[t, parent] @ np.linalg.inv(tpose_rots[parent])
                j_delta = global_rots[t, j] @ np.linalg.inv(tpose_rots[j])
                # Local delta (joint relative to parent)
                local_delta = np.linalg.inv(p_delta) @ j_delta
                # Dampen via axis-angle scaling
                angle = np.arccos(np.clip((np.trace(local_delta) - 1) / 2, -1, 1))
                if angle > 1e-6:
                    axis = np.array([
                        local_delta[2, 1] - local_delta[1, 2],
                        local_delta[0, 2] - local_delta[2, 0],
                        local_delta[1, 0] - local_delta[0, 1],
                    ]) / (2 * np.sin(angle))
                    damp_angle = angle * factor
                    K = np.array([
                        [0, -axis[2], axis[1]],
                        [axis[2], 0, -axis[0]],
                        [-axis[1], axis[0], 0],
                    ])
                    local_damp = (np.eye(3) +
                                  np.sin(damp_angle) * K +
                                  (1 - np.cos(damp_angle)) * (K @ K))
                else:
                    local_damp = local_delta
                # Reconstruct global rotation from T-pose reference
                j_delta_new = p_delta @ local_damp
                global_rots[t, j] = j_delta_new @ tpose_rots[j]

        print(f"[retarget] Spine dampening: Neck={SPINE_DAMPEN.get(4,1):.0%}, "
              f"Neck1={SPINE_DAMPEN.get(5,1):.0%}, Head={SPINE_DAMPEN.get(6,1):.0%}")

        posed_pos_fk = np.zeros_like(posed_joints_soma)
        for t in range(T):
            for j in range(n_joints_rt):
                parent = SOMA77_PARENTS[j]
                if parent == -1 or parent >= n_joints_rt:
                    # Root: T-pose position + NPZ root displacement from f0
                    root_disp = posed_joints_soma[t, j] - posed_joints_soma[0, j]
                    posed_pos_fk[t, j] = tpose_pos[j] + root_disp
                else:
                    # Bone vector in T-pose (global)
                    bone_vec = tpose_pos[j] - tpose_pos[parent]
                    # Absolute rotation from T-POSE (uses dampened global_rots)
                    # This applies the T-pose→posed offset, including the
                    # critical arm drop from T-pose to A-pose
                    delta_R = global_rots[t, parent] @ np.linalg.inv(tpose_rots[parent])
                    posed_pos_fk[t, j] = posed_pos_fk[t, parent] + delta_R @ bone_vec

        posed_joints_soma = posed_pos_fk
        # Save frame-0 FK positions for D_trans computation
        posed_pos_fk[0].copy()
        print("[retarget] Forward kinematics: ABSOLUTE rotations from T-pose "
              "+ T-pose bone lengths + spine dampening (arms un-dampened)")

    # ── Fix GLB coordinate space if needed ──────────────────────────────
    # The soma_weight_transfer GLB writer puts mesh vertices in Z-up (from
    # TRELLIS source) while bones/IBM are in Y-up (SOMA). This mismatch
    # causes severe body horror. We fix it at the binary level BEFORE import,
    # converting mesh POSITION/NORMAL from Z-up to Y-up. The IBM is left
    # unchanged (it is already correct as inv(bind_world_yup)).
    rigged_path = _fix_glb_if_needed(rigged_path)

    # ── Clear scene and import the rigged character ─────────────────────
    bpy.ops.wm.read_factory_settings(use_empty=True)

    ext = os.path.splitext(rigged_path)[1].lower()
    if ext == ".fbx":
        bpy.ops.import_scene.fbx(filepath=rigged_path)
    else:
        bpy.ops.import_scene.gltf(filepath=rigged_path)

    # ── Find armature ───────────────────────────────────────────────────
    armature_obj = None
    for obj in bpy.data.objects:
        if obj.type == "ARMATURE":
            armature_obj = obj
            break
    if armature_obj is None:
        raise RuntimeError("No armature found in the rigged file")

    # ── Reposition bones from GLB IBM data ─────────────────────────────
    # Blender's glTF importer creates all bones at the origin, ignoring
    # node translations. We fix this by reading the IBM from the GLB,
    # computing world positions, and setting head/tail/roll in edit mode.
    if ext == ".glb":
        _reposition_bones_from_glb(rigged_path, armature_obj)

        # ── Merge duplicate vertices and enable smooth shading ─────────
        # The soma_weight_transfer GLB writer exports flat (per-face) normals,
        # which duplicates vertices (18k → 108k). We merge by distance to
        # restore the original vertex count and enable smooth shading.
        for mesh_obj in bpy.data.objects:
            if mesh_obj.type != 'MESH' or len(mesh_obj.vertex_groups) == 0:
                continue
            bpy.context.view_layer.objects.active = mesh_obj
            mesh_obj.select_set(True)
            armature_obj.select_set(False)
            bpy.ops.object.mode_set(mode='EDIT')
            bpy.ops.mesh.select_all(action='SELECT')
            bpy.ops.mesh.remove_doubles(threshold=1e-6)
            bpy.ops.mesh.normals_make_consistent(inside=False)
            bpy.ops.object.mode_set(mode='OBJECT')
            # Enable smooth shading
            for poly in mesh_obj.data.polygons:
                poly.use_smooth = True
            armature_obj.select_set(True)
            bpy.context.view_layer.objects.active = armature_obj
            print(f"[retarget] Merged mesh: {mesh_obj.name} → "
                  f"{len(mesh_obj.data.vertices)} verts")

        # ── Recreate armature modifier to fix IBM ──────────────────────
        # The GLB's IBM doesn't match Blender's internally-computed bind
        # matrices. Removing and re-adding the armature modifier forces
        # Blender to recompute the IBM from the repositioned bones.
        # ONLY apply to skinned meshes (those with vertex groups).
        # Stray meshes (e.g., visualization Icospheres) must NOT get
        # an armature modifier or they'll be exported as skinned geometry.
        for mesh_obj in bpy.data.objects:
            if mesh_obj.type != 'MESH' or len(mesh_obj.vertex_groups) == 0:
                continue
            for mod in list(mesh_obj.modifiers):
                if mod.type == 'ARMATURE':
                    mesh_obj.modifiers.remove(mod)
            new_mod = mesh_obj.modifiers.new(name="Armature", type='ARMATURE')
            new_mod.object = armature_obj
            new_mod.use_bone_envelopes = False
            new_mod.use_vertex_groups = True
        print("[retarget] Recreated armature modifier for IBM fix")
        bpy.context.view_layer.update()

        # ── Clean face/jaw vertex weights ────────────────────────────────
        # Bone-heat skinning assigns some arm/shoulder weights to face
        # vertices (they're close in T-pose). When arms swing during the
        # walk cycle, these face vertices get dragged, causing jaw
        # distortion. Fix: for vertices in the head region, zero out
        # weights for non-head bones (arms, shoulders, legs, hips).
        import numpy as _np
        for mesh_obj in bpy.data.objects:
            if mesh_obj.type != 'MESH' or len(mesh_obj.vertex_groups) == 0:
                continue
            verts = mesh_obj.data.vertices
            cos = _np.array([(mesh_obj.matrix_world @ v.co).to_tuple() for v in verts])
            if len(cos) == 0:
                continue
            # Head region: top 18% of mesh height
            y_min, y_max = cos[:, 2].min(), cos[:, 2].max()
            height = y_max - y_min
            if height < 0.01:
                continue
            head_thresh = y_min + height * 0.82  # top 18%
            head_mask = cos[:, 2] > head_thresh
            n_head = head_mask.sum()
            if n_head == 0:
                continue

            [vg.name for vg in mesh_obj.vertex_groups]
            # Bones allowed in head region
            # Bones to zero out
            zero_names = set()
            for vg in mesh_obj.vertex_groups:
                vn = vg.name.lower()
                # Zero arm, shoulder, hand, leg, foot, toe, spine, hips, chest bones
                if any(p in vn for p in ('arm', 'shoulder', 'hand', 'thumb', 'index',
                    'middle', 'ring', 'pinky', 'leg', 'foot', 'toe', 'upleg',
                    'spine', 'hips', 'chest')):
                    zero_names.add(vg.name)

            if not zero_names:
                continue

            zero_vg_indices = set()
            for vg in mesh_obj.vertex_groups:
                if vg.name in zero_names:
                    zero_vg_indices.add(vg.index)

            n_cleaned = 0
            for vi, v in enumerate(verts):
                if not head_mask[vi]:
                    continue
                had_weight = False
                for g in list(v.groups):
                    if g.group in zero_vg_indices and g.weight > 0.001:
                        vg = mesh_obj.vertex_groups[g.group]
                        vg.remove([vi])
                        had_weight = True
                if had_weight:
                    n_cleaned += 1

            # Normalize remaining weights for cleaned vertices
            for vi, v in enumerate(verts):
                if not head_mask[vi]:
                    continue
                total = sum(g.weight for g in v.groups)
                if total > 0.001:
                    for g in v.groups:
                        vg = mesh_obj.vertex_groups[g.group]
                        vg.add([vi], g.weight / total, 'REPLACE')

            print(f"[retarget] Face weight cleanup: {n_cleaned} head verts "
                  f"cleaned (removed arm/shoulder/spine weights from face area)")

    # ── Decimate mesh to reduce poly count ─────────────────────────────
    # TRELLIS meshes are very dense (500K+ verts, lumpy "melted wax" look).
    # Decimation with Collapse mode preserves vertex groups (weights) and
    # significantly reduces file size + improves visual quality.
    if decimate_ratio and 0.0 < decimate_ratio < 1.0:
        for obj in bpy.data.objects:
            if obj.type != 'MESH' or len(obj.vertex_groups) == 0:
                continue
            bpy.context.view_layer.objects.active = obj
            obj.select_set(True)
            decim = obj.modifiers.new(name="Decimate", type='DECIMATE')
            decim.decimate_type = 'COLLAPSE'
            decim.ratio = decimate_ratio
            decim.use_collapse_triangulate = True
            bpy.ops.object.modifier_apply(modifier="Decimate")
            verts_after = len(obj.data.vertices)
            faces_after = len(obj.data.polygons)
            print(f"[retarget] Decimated {obj.name}: ratio={decimate_ratio} "
                  f"→ {verts_after} verts, {faces_after} faces")
            obj.select_set(False)

    pose = armature_obj.pose
    bpy.context.view_layer.objects.active = armature_obj
    armature_obj.select_set(True)

    # ── Resolve bone name mapping ───────────────────────────────────────
    bone_map: dict[int, str] = {}  # soma_idx → resolved bone name
    for soma_idx, mixamo_name in SOMA_TO_MIXAMO.items():
        if soma_idx >= n_joints:
            continue
        resolved = _resolve_bone(pose.bones, mixamo_name)
        if resolved:
            bone_map[soma_idx] = resolved

    print(f"[retarget] Mapped {len(bone_map)}/{len(SOMA_TO_MIXAMO)} bones")
    if len(bone_map) < 10:
        missing = [
            v for k, v in sorted(SOMA_TO_MIXAMO.items())
            if k not in bone_map
        ]
        print(f"[retarget] Missing bones: {missing[:15]}")

    if not bone_map:
        raise RuntimeError(
            "No Mixamo bones found in armature. Bone names: "
            + ", ".join(b.name for b in pose.bones[:20])
        )

    # ── Precompute rest data per mapped bone ────────────────────────────
    rest_dirs: dict[int, mathutils.Vector] = {}   # armature-local (fallback)
    rest_mats: dict[int, mathutils.Matrix] = {}   # bone.matrix_local

    for soma_idx, bone_name in bone_map.items():
        bone = pose.bones[bone_name].bone  # underlying Bone (rest data)
        rest_dir = (bone.tail_local - bone.head_local)
        if rest_dir.length < 1e-8:
            rest_dir = mathutils.Vector((0, 1, 0))
        rest_dirs[soma_idx] = rest_dir.normalized()
        rest_mats[soma_idx] = bone.matrix_local.copy()

    # ── Scale factor (SOMA height vs armature height) ───────────────────
    # Uses mesh bounding-box height vs SOMA skeleton full height for
    # robustness. The old approach (Hips→Head bone distance) was unreliable
    # because SkinToken's skeleton has misplaced bones.
    scale = 1.0

    # SOMA skeleton full height (feet to head) using Y axis (up in SOMA)
    soma_ys = posed_joints_soma[0, :, 1]  # all joint Y at frame 0
    soma_height = float(soma_ys.max() - soma_ys.min())
    if soma_height < 1e-6:
        soma_height = 1.0

    # Armature height: mesh bounding box Z extent (up in Blender)
    arm_height = 1.0
    for child in armature_obj.children:
        if child.type == 'MESH':
            zs = [
                (child.matrix_world @ v.co).z
                for v in child.data.vertices
            ]
            if zs:
                arm_height = max(zs) - min(zs)
                if arm_height < 1e-6:
                    arm_height = 1.0
            break

    scale = arm_height / soma_height
    print(f"[retarget] Scale: {scale:.4f} "
          f"(arm_height={arm_height:.3f}, soma_height={soma_height:.3f})")

    # ── Set rotation mode to quaternion for all mapped bones ────────────
    for bone_name in bone_map.values():
        pose.bones[bone_name].rotation_mode = "QUATERNION"

    # ── Precompute numpy rest matrices for LBS deformation ──────────────
    # We need the rest matrices as numpy arrays for fast matrix operations
    # during the LBS deformation computation.
    rest_mats_np: dict[int, np.ndarray] = {}
    for soma_idx, bone_name in bone_map.items():
        bone = pose.bones[bone_name].bone
        rest_mats_np[soma_idx] = np.array(bone.matrix_local, dtype=np.float64)

    # ── Create animation action ─────────────────────────────────────────
    if armature_obj.animation_data is None:
        armature_obj.animation_data_create()
    action = bpy.data.actions.new("KimodoRetarget")
    armature_obj.animation_data.action = action

    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = T
    scene.render.fps = fps

    # ── Retarget each frame ─────────────────────────────────────────────
    order = _topo_order(bone_map)

    # Collect ALL pose matrices for ALL frames (for custom GLB writer)
    all_pose_world: list[dict[str, np.ndarray]] = []

    for t in range(T):
        frame = t + 1

        # Compute desired world matrices for all mapped bones
        desired: dict[int, mathutils.Matrix] = {}

        for soma_idx, bone_name in bone_map.items():
            # ── LBS Deformation Retarget ──────────────────────────────────
            #
            # CORE INSIGHT: Blender bone rest matrices (matrix_local) differ
            # from SOMA bind transforms by ~180° because Blender forces bone
            # Y-axis along head→tail while SOMA joints have a different
            # convention. The delta and direction-based approaches both
            # fail because they depend on rest_quat matching the SOMA bind.
            #
            # CORRECT APPROACH: Compute the LBS deformation matrix directly:
            #   D = G_t @ G_bind⁻¹
            # Convert to Blender space: D_bl = C @ D @ C⁻¹
            # Set bone world matrix: P = D_bl @ R_rest
            #
            # The LBS deformation is then:
            #   P @ R_rest⁻¹ = D_bl @ R_rest @ R_rest⁻¹ = D_bl
            #
            # This is correct REGARDLESS of bone rest pose mismatch, because
            # the deformation matrix D_bl is computed entirely from SOMA data
            # and doesn't depend on Blender's bone convention.

            if use_rot_mats:
                G_t_rot = global_rots[t, soma_idx]       # (3, 3) SOMA
                G_t_pos = posed_joints_soma[t, soma_idx]  # (3,) SOMA (FK-computed)

                # Deformation: ABSOLUTE rotation from T-pose (not delta from frame 0).
                # This is critical: the NPZ starts in A-pose (arms at sides),
                # but the Blender bind pose is T-pose (arms horizontal). Using
                # tpose_rots as bind ensures the T-pose→A-pose offset IS applied,
                # bringing the arms down to the sides.
                bind_rot_inv = np.linalg.inv(tpose_rots[soma_idx])
                D_rot_soma = G_t_rot @ bind_rot_inv                       # (3,3)

                # Convert rotation to Blender space: D_bl = C @ D @ C⁻¹
                C = SOMA_TO_BLENDER_C
                CT = SOMA_TO_BLENDER_C.T
                D_rot_bl = C @ D_rot_soma @ CT           # (3,3)

                # ── D_trans: Blender-space displacement from T-pose ───────────
                #
                # P_trans = R_rest_trans + (posed_pos_bl_t - tpose_pos_bl)
                #
                # This adds the SOMA-computed displacement (T-pose → posed,
                # converted to Blender space) to the Blender bone's REST position.
                #
                # WHY NOT the mathematically-exact D_trans_bl = posed_pos_bl_t
                #   - D_rot_bl @ tpose_pos_bl?
                # Because when the Blender rig and SOMA template have DIFFERENT
                # proportions (SOMA=1.72m, TRELLIS=1.00m), the offset
                # (R_rest_trans - tpose_pos_bl) can be 50cm+. The exact formula
                # rotates this offset by D_rot, amplifying the error to 475mm+
                # for the foot bone during walk — causing PANCAKE LEGS.
                #
                # This formula keeps the offset UN-rotated (as a constant
                # translation), so the bone moves by exactly the SOMA displacement
                # (scaled), without amplifying skeleton proportion mismatch.
                #
                # ═══ BUG HISTORY (DO NOT REINTRODUCE) ═══
                # v8 BUG: displacement_bl = posed_pos_bl_t - posed_pos_bl_0
                #   where posed_pos_bl_0 = posed_pos_fk[0] = A-POSE
                #   At frame 0: displacement = 0 → P_trans = R_rest_trans
                #   → BONE FROZEN AT T-POSE, arms stuck horizontal
                #
                # v10 BUG: D_trans_bl = posed_pos_bl_t - D_rot_bl @ tpose_pos_bl
                #   Mathematically exact but rotates the 50cm proportion offset
                #   by D_rot → 475mm error for feet → PANCAKE LEGS at F48
                #
                # v11 FIX: D_trans_bl = (I-D_rot_bl) @ R_rest_trans + displacement_bl
                #   where displacement_bl = posed_pos_bl_t - tpose_pos_bl (T-pose!)
                #   P_trans = R_rest_trans + displacement_bl (un-rotated)
                #   Arms drop ✓  Feet stable ✓
                R_rest = rest_mats_np[soma_idx]           # (4,4)
                R_rest_trans = R_rest[:3, 3]               # (3,) Blender bone rest

                # SOMA positions in Blender space (scale * C conversion)
                posed_pos_bl_t = C @ G_t_pos * scale       # (3,) posed at frame t
                tpose_pos_bl = C @ tpose_pos[soma_idx] * scale  # (3,) T-pose ref

                # Displacement from T-pose (NOT from frame-0/A-pose!)
                displacement_bl = posed_pos_bl_t - tpose_pos_bl
                D_trans_bl = (np.eye(3) - D_rot_bl) @ R_rest_trans + displacement_bl

                # P = D_bl @ R_rest  (4×4 composition)
                P_rot = D_rot_bl @ R_rest[:3, :3]         # (3,3)
                P_trans = D_rot_bl @ R_rest_trans + D_trans_bl  # (3,)

                desired_mat_np = np.eye(4, dtype=np.float64)
                desired_mat_np[:3, :3] = P_rot
                desired_mat_np[:3, 3] = P_trans
                desired[soma_idx] = mathutils.Matrix(desired_mat_np.tolist())
            else:
                # Fallback: direction-based retarget (no global_rot_mats)
                rest_mat = rest_mats[soma_idx]
                rest_quat = rest_mat.to_quaternion()
                rest_pos = rest_mat.translation
                parent_idx = SOMA77_PARENTS[soma_idx]
                if soma_idx == 0:
                    delta = posed_joints_soma[t, soma_idx] - posed_joints_soma[0, soma_idx]
                    bl_delta = mathutils.Vector((delta[0], -delta[2], delta[1])) * scale
                    desired_pos = mathutils.Vector(rest_pos) + bl_delta
                else:
                    desired_pos = mathutils.Vector(rest_pos)
                if parent_idx >= 0 and parent_idx < n_joints:
                    soma_rest_vec = posed[0, soma_idx] - posed[0, parent_idx]
                    soma_posed_vec = posed[t, soma_idx] - posed[t, parent_idx]
                    sr = mathutils.Vector(soma_rest_vec)
                    sp = mathutils.Vector(soma_posed_vec)
                    if sr.length < 1e-8 or sp.length < 1e-8:
                        desired_quat = rest_quat
                    else:
                        sr.normalize()
                        sp.normalize()
                        delta_quat = sr.rotation_difference(sp)
                        desired_quat = delta_quat @ rest_quat
                else:
                    desired_quat = rest_quat
                desired[soma_idx] = (
                    mathutils.Matrix.Translation(desired_pos)
                    @ desired_quat.to_matrix().to_4x4()
                )

        # Solve matrix_basis in topological order (parent before child).
        # We track pose_mats manually — calling bpy.context.view_layer.update()
        # during the loop would trigger F-Curve evaluation at the wrong frame,
        # corrupting the pose. Manual tracking is provably identical to
        # Blender's own parent-chain evaluation (verified via dot-product tests).
        pose_mats: dict[str, mathutils.Matrix] = {}

        for soma_idx in order:
            bone_name = bone_map[soma_idx]
            pbone = pose.bones[bone_name]
            d_mat = desired[soma_idx]
            R = rest_mats[soma_idx]

            parent_pbone = pbone.parent
            if parent_pbone is not None:
                # Use cached parent pose matrix (computed earlier this frame)
                P = pose_mats.get(parent_pbone.name, parent_pbone.matrix.copy())
                R_parent = parent_pbone.bone.matrix_local.copy()
                basis = R.inverted() @ R_parent @ P.inverted() @ d_mat
            else:
                basis = R.inverted() @ d_mat

            # NOTE: We do NOT zero the translation for non-root bones.
            #
            # The LBS deformation P = D_bl @ R_rest requires the FULL matrix
            # (including translation) for correct skin deformation. Zeroing
            # the translation breaks the LBS because P @ R_rest^-1 no longer
            # equals the deformation matrix D_bl.
            #
            # The translation in matrix_basis encodes the bone's local offset
            # relative to its parent — this is legitimate in skeletal animation
            # and necessary for correct LBS deformation.

            if parent_pbone is not None:
                pose_mats[bone_name] = P @ R_parent.inverted() @ R @ basis
            else:
                pose_mats[bone_name] = R @ basis

            pbone.matrix_basis = basis

        # Store pose world matrices for this frame (for custom GLB writer)
        frame_poses: dict[str, np.ndarray] = {}
        for bn, pm in pose_mats.items():
            frame_poses[bn] = np.array(pm, dtype=np.float64)
        all_pose_world.append(frame_poses)

        # Keyframe all mapped bones
        for bone_name in bone_map.values():
            pbone = pose.bones[bone_name]
            pbone.keyframe_insert(data_path="location", frame=frame)
            pbone.keyframe_insert(data_path="rotation_quaternion", frame=frame)
            pbone.keyframe_insert(data_path="scale", frame=frame)

    # Final evaluation
    bpy.context.view_layer.update()
    scene.frame_set(1)
    print(f"[retarget] Retargeted {T} frames to {len(bone_map)} bones")

    # ── Enable smooth shading on all meshes ─────────────────────────────
    # Without this, the glTF exporter creates per-face normals, exploding
    # the vertex count (18k → 108k) and producing a faceted appearance.
    # We also clear custom split normals that the glTF importer may have
    # created — these force per-vertex normals regardless of shading mode.
    for obj in bpy.data.objects:
        if obj.type == "MESH":
            # Clear custom split normals
            try:
                obj.data.use_auto_smooth = False
                obj.data.normals_split_custom_set([(0, 0, 0)] * len(obj.data.vertices))
                obj.data.normals_split_custom_set(None)
            except Exception:
                pass
            for poly in obj.data.polygons:
                poly.use_smooth = True

    # ── Export ──────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # Select armature + skinned mesh objects only.
    # Stray mesh objects (e.g., visualization Icospheres from auto-rigging)
    # must be excluded — they appear as garbage geometry in the output.
    # A mesh is "skinned" if it has vertex groups or an armature modifier
    # or is a child of the armature.
    bpy.ops.object.select_all(action="DESELECT")
    armature_obj.select_set(True)
    n_meshes_selected = 1  # armature
    for obj in bpy.data.objects:
        if obj.type != "MESH":
            continue
        is_skinned = (
            len(obj.vertex_groups) > 0
            or any(m.type == "ARMATURE" for m in obj.modifiers)
            or obj.parent == armature_obj
        )
        if is_skinned:
            obj.select_set(True)
            n_meshes_selected += 1
        else:
            print(f"[retarget] Skipping non-skinned mesh: {obj.name} "
                  f"({len(obj.data.vertices)} verts)")
    bpy.context.view_layer.objects.active = armature_obj
    print(f"[retarget] Selected {n_meshes_selected} objects for export")

    if T_fmt == "fbx":
        bpy.ops.export_scene.fbx(
            filepath=output_path,
            use_selection=True,
            object_types={"ARMATURE", "MESH"},
            bake_anim=True,
            bake_anim_use_all_bones=True,
            bake_anim_force_startend_keying=True,
            bake_anim_step=1.0 / fps,
            add_leaf_bones=False,
            apply_unit_scale=True,
            axis_forward="-Z",
            axis_up="Y",
            use_metadata=False,
        )
    else:
        # ── GLB export via Blender's built-in exporter ──────────────────
        # The custom GLB writer (_write_animated_glb) had a coordinate
        # conversion bug (conjugation vs left-multiplication) that produced
        # body horror. After fixing the GLB mesh coordinates upstream
        # (_fix_glb_if_needed), Blender's built-in glTF exporter works
        # correctly and produces clean output.
        bpy.ops.export_scene.gltf(
            filepath=output_path,
            use_selection=True,
            export_format="GLB",
            export_animations=True,
            export_skins=True,
            export_yup=True,
        )

    print(f"[retarget] Exported: {output_path}")
    return output_path


# ═══════════════════════════════════════════════════════════════════════════
# CLI entry point
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    argv = sys.argv
    sep = argv.index("--") if "--" in argv else len(argv)
    cli_args = argv[sep + 1:]

    if len(cli_args) < 5:
        print("USAGE: blender -b -P blender_retarget.py -- "
              "<rigged_path> <npz_path> <output_path> <fps> <file_format> "
              "[in_place] [decimate_ratio]")
        sys.exit(1)

    _kwargs = dict(
        rigged_path=cli_args[0],
        npz_path=cli_args[1],
        output_path=cli_args[2],
        fps=int(cli_args[3]),
        file_format=cli_args[4],
    )
    if len(cli_args) > 5:
        _kwargs["in_place"] = cli_args[5].lower() in ("true", "1", "yes")
    if len(cli_args) > 6:
        _kwargs["decimate_ratio"] = float(cli_args[6])

    retarget(**_kwargs)
