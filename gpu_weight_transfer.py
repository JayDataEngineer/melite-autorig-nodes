"""GPU-accelerated weight transfer — replaces SkinToken's brute-force transfer_rigging.

Does the ENTIRE transfer in the ComfyUI process (GPU available), writing the
output GLB directly. Blender is only used for FBX conversion if needed.

SkinToken's transfer_rigging does:
  1. estimate_similarity_transform (align source→target) — ESSENTIAL for quality
  2. Brute-force O(N×M) nearest-neighbor                    — SLOW (32B ops)
  3. Weight copy by index                                   — trivial

This replaces step 2 with torch.cdist on GPU (RTX 4090). Steps 1 and 3 are
identical. The output GLB is assembled from:
  - Source GLB geometry (POSITION, NORMAL, TEXCOORD_0, materials, textures)
  - Rigged GLB armature (nodes, skin, inverse bind matrices)
  - Transferred JOINTS_0 + WEIGHTS_0 (computed via GPU nearest-neighbor)

No Blender involved for weight transfer → no vertex reordering issues.
"""
from __future__ import annotations

import json
import struct
import logging
import time
import copy
import numpy as np

log = logging.getLogger(__name__)


# ── GLB parsing helpers ─────────────────────────────────────────────────

def _read_glb(path: str) -> tuple[dict, bytes]:
    """Read a GLB file → (gltf JSON dict, binary buffer)."""
    with open(path, "rb") as f:
        magic, version, length = struct.unpack("<III", f.read(12))
        assert magic == 0x46546C67, f"Not a GLB: {path}"
        json_len, json_type = struct.unpack("<II", f.read(8))
        assert json_type == 0x4E4F534A, "Bad JSON chunk type"
        # GLB JSON chunk may be padded with spaces (0x20) or nulls (0x00)
        gltf = json.loads(f.read(json_len).decode("utf-8").rstrip(" \x00"))
        bin_len, bin_type = struct.unpack("<II", f.read(8))
        assert bin_type == 0x004E4942, "Bad BIN chunk type"
        bin_data = f.read(bin_len)
    return gltf, bin_data


def _read_accessor(gltf: dict, bin_data: bytes, idx: int) -> np.ndarray:
    """Read accessor data from a glTF binary buffer."""
    acc = gltf["accessors"][idx]
    bv = gltf["bufferViews"][acc["bufferView"]]
    offset = bv.get("byteOffset", 0) + acc.get("byteOffset", 0)

    count = acc["count"]
    atype = acc["type"]
    comp = acc["componentType"]

    dtypes = {5120: np.int8, 5121: np.uint8, 5122: np.int16,
              5123: np.uint16, 5125: np.uint32, 5126: np.float32}
    np_dt = dtypes[comp]
    esizes = {5120: 1, 5121: 1, 5122: 2, 5123: 2, 5125: 4, 5126: 4}
    esize = esizes[comp]
    ncomps = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}[atype]

    stride = bv.get("byteStride", 0)  # 0 = tightly packed
    if stride == 0 or stride == esize * ncomps:
        arr = np.frombuffer(bin_data, dtype=np_dt, count=count * ncomps,
                            offset=offset).reshape(count, ncomps).copy()
    else:
        arr = np.empty((count, ncomps), dtype=np_dt)
        for i in range(count):
            arr[i] = np.frombuffer(bin_data, dtype=np_dt, count=ncomps,
                                   offset=offset + i * stride)
    return arr


# ── Similarity transform (AABB bounding-box alignment) ──────────────────

def _estimate_similarity_transform(
    src: np.ndarray, tgt: np.ndarray,
) -> np.ndarray:
    """Estimate a 4×4 similarity transform aligning src → tgt using AABB.

    Centers both meshes at the same point and scales uniformly to match
    sizes. Robust to very different coordinate spaces (e.g.,
    bottom_center_origin normalization). For character meshes of the same
    shape (simplified vs. full-detail), AABB alignment is sufficient for
    accurate nearest-vertex weight transfer — and unlike Kabsch/Umeyama,
    it never collapses when fed imperfect correspondences.
    """
    src_min = src.min(axis=0)
    src_max = src.max(axis=0)
    src_center = (src_min + src_max) / 2.0
    src_size = src_max - src_min

    tgt_min = tgt.min(axis=0)
    tgt_max = tgt.max(axis=0)
    tgt_center = (tgt_min + tgt_max) / 2.0
    tgt_size = tgt_max - tgt_min

    # Uniform scale: average of per-axis scale ratios
    scale_ratios = tgt_size / np.maximum(src_size, 1e-6)
    scale = float(np.mean(scale_ratios))

    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = scale * np.eye(3, dtype=np.float32)
    T[:3, 3] = (tgt_center - scale * src_center).astype(np.float32)
    return T


# ── Main entry point ────────────────────────────────────────────────────

def transfer_weights_and_write_glb(
    rigged_path: str,
    source_path: str,
    output_path: str,
) -> str:
    """Transfer skin weights from rigged GLB onto original textured GLB.

    1. Parse both GLBs
    2. GPU: AABB-align rigged→source, compute nearest-neighbor (torch.cdist)
    3. Transfer JOINTS_0 + WEIGHTS_0 using indices
    4. Write output GLB: source geometry + transferred weights + armature

    Returns output_path.
    """
    t0 = time.time()
    import torch

    # ── Load both GLBs ─────────────────────────────────────────────────
    rigged_gltf, rigged_bin = _read_glb(rigged_path)
    source_gltf, source_bin = _read_glb(source_path)

    # ── Extract rigged mesh data ───────────────────────────────────────
    rigged_prim = rigged_gltf["meshes"][0]["primitives"][0]
    ra = rigged_prim["attributes"]

    rigged_pos = _read_accessor(rigged_gltf, rigged_bin, ra["POSITION"]).astype(np.float32)
    rigged_joints = _read_accessor(rigged_gltf, rigged_bin, ra["JOINTS_0"]).astype(np.int32)
    rigged_weights = _read_accessor(rigged_gltf, rigged_bin, ra["WEIGHTS_0"]).astype(np.float32)

    # ── Extract source mesh data ───────────────────────────────────────
    source_prim = source_gltf["meshes"][0]["primitives"][0]
    sa = source_prim["attributes"]
    source_pos = _read_accessor(source_gltf, source_bin, sa["POSITION"]).astype(np.float32)

    n_src, n_tgt = len(rigged_pos), len(source_pos)
    log.info("[gpu_transfer] rigged=%d verts → source=%d verts", n_src, n_tgt)

    # ── GPU: AABB alignment + nearest-neighbor (torch.cdist) ───────────
    T = _estimate_similarity_transform(rigged_pos, source_pos)
    rigged_h = np.hstack([rigged_pos, np.ones((n_src, 1), dtype=np.float32)])
    rigged_aligned = (T @ rigged_h.T).T[:, :3]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    src_tensor = torch.from_numpy(rigged_aligned.astype(np.float32)).to(device)
    tgt_tensor = torch.from_numpy(source_pos.astype(np.float32)).to(device)

    indices = np.empty(n_tgt, dtype=np.int64)

    if device == "cuda":
        max_cells = 500_000_000  # ~2GB / 4 bytes
        chunk_size = max(256, min(8192, max_cells // max(n_src, 1)))
    else:
        chunk_size = 128

    t_nn = time.time()
    for i in range(0, n_tgt, chunk_size):
        end = min(i + chunk_size, n_tgt)
        dists = torch.cdist(tgt_tensor[i:end], src_tensor)
        indices[i:end] = dists.argmin(dim=1).cpu().numpy()
    log.info("[gpu_transfer] NN %d×%d on %s in %.1fs",
             n_tgt, n_src, device, time.time() - t_nn)

    # Safety: clamp indices to valid range
    indices = np.clip(indices, 0, n_src - 1)

    # ── Transfer weights ───────────────────────────────────────────────
    transferred_joints = rigged_joints[indices]    # (M, 4)
    transferred_weights = rigged_weights[indices]  # (M, 4)

    # Normalize weights per vertex (handle floating-point drift)
    w_sums = transferred_weights.sum(axis=1, keepdims=True)
    w_sums = np.maximum(w_sums, 1e-10)
    transferred_weights = transferred_weights / w_sums

    flat_tj = transferred_joints[:, 0]
    unique_tj = np.unique(flat_tj[transferred_weights[:, 0] > 0.1])
    log.info("[gpu_transfer] %d unique primary joints, weights normalized",
             len(unique_tj))

    # Free GPU tensors before GLB assembly (which may do a second GPU pass
    # for per-vertex weight redistribution in _sanitize_rig)
    del src_tensor, tgt_tensor, indices
    if device == "cuda":
        torch.cuda.empty_cache()

    # ── Write output GLB ───────────────────────────────────────────────
    # Pass T so bone translations and IBM can be transformed from rigged
    # space into source space — without this, the bones (height=2.0 from
    # bottom_center_origin) won't match the source vertices (height=1.0)
    # and any animation will produce the "abomination" deformation.
    _write_output_glb(
        source_gltf, source_bin,
        transferred_joints, transferred_weights,
        rigged_gltf, rigged_bin,
        T,
        output_path,
        source_pos,
    )

    elapsed = time.time() - t0
    log.info("[gpu_transfer] complete in %.1fs → %s", elapsed, output_path)
    return output_path


# ── Canonical Mixamo hierarchy ──────────────────────────────────────────
# Maps bone short-name (without "mixamorig:" prefix) to its parent.
# Used to REBUILD SkinToken's garbled hierarchy into the standard Mixamo
# structure. SkinToken nests bones incorrectly (e.g., LeftArm inside Head)
# and inserts Extra_XX bones that absorb weight but are never animated.

MIXAMO_PARENTS: dict[str, str | None] = {
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
    "Head": "Neck",
    "HeadTop_End": "Head",
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
    "LeftHandThumb1": "LeftHand", "LeftHandThumb2": "LeftHandThumb1",
    "LeftHandThumb3": "LeftHandThumb2", "LeftHandThumb4": "LeftHandThumb3",
    "LeftHandIndex1": "LeftHand", "LeftHandIndex2": "LeftHandIndex1",
    "LeftHandIndex3": "LeftHandIndex2", "LeftHandIndex4": "LeftHandIndex3",
    "LeftHandMiddle1": "LeftHand", "LeftHandMiddle2": "LeftHandMiddle1",
    "LeftHandMiddle3": "LeftHandMiddle2", "LeftHandMiddle4": "LeftHandMiddle3",
    "LeftHandRing1": "LeftHand", "LeftHandRing2": "LeftHandRing1",
    "LeftHandRing3": "LeftHandRing2", "LeftHandRing4": "LeftHandRing3",
    "LeftHandPinky1": "LeftHand", "LeftHandPinky2": "LeftHandPinky1",
    "LeftHandPinky3": "LeftHandPinky2", "LeftHandPinky4": "LeftHandPinky3",
    "RightHandThumb1": "RightHand", "RightHandThumb2": "RightHandThumb1",
    "RightHandThumb3": "RightHandThumb2", "RightHandThumb4": "RightHandThumb3",
    "RightHandIndex1": "RightHand", "RightHandIndex2": "RightHandIndex1",
    "RightHandIndex3": "RightHandIndex2", "RightHandIndex4": "RightHandIndex3",
    "RightHandMiddle1": "RightHand", "RightHandMiddle2": "RightHandMiddle1",
    "RightHandMiddle3": "RightHandMiddle2", "RightHandMiddle4": "RightHandMiddle3",
    "RightHandRing1": "RightHand", "RightHandRing2": "RightHandRing1",
    "RightHandRing3": "RightHandRing2", "RightHandRing4": "RightHandRing3",
    "RightHandPinky1": "RightHand", "RightHandPinky2": "RightHandPinky1",
    "RightHandPinky3": "RightHandPinky2", "RightHandPinky4": "RightHandPinky3",
}


def _short_name(name: str) -> str:
    """Strip 'mixamorig:' prefix from a bone name."""
    return name.replace("mixamorig:", "") if name else name


def _sanitize_rig(
    rigged_nodes: list,
    rigged_skin: dict,
    joints: np.ndarray,      # (M, 4) skin joint indices
    weights: np.ndarray,     # (M, 4)
    source_pos: np.ndarray,  # (M, 3)
) -> tuple[list, dict, np.ndarray, np.ndarray]:
    """Remove Extra bones, redistribute weights, fix hierarchy.

    SkinToken produces garbled skeleton hierarchies: Extra_XX bones that
    absorb 30%+ of mesh weights but are never animated, left arm nested
    inside Head, etc. This function:

    1. Computes dominant-weight centroids for all skin joints
    2. Redistributes Extra bone weights to nearest Mixamo bone
    3. Removes Extra bone nodes from the hierarchy
    4. Rebuilds parent-child relationships to canonical Mixamo structure

    After sanitization, every vertex is weighted to ANIMATED bones, and
    bone positions (computed downstream via centroids) are accurate.

    Returns:
        (new_nodes, new_skin, new_joints, new_weights) where:
        - new_nodes: node list with Extra bones removed + canonical hierarchy
        - new_skin: skin with cleaned joint list (no Extra bones)
        - new_joints: (M, 4) remapped skin joint indices
        - new_weights: (M, 4) redistributed + renormalized weights
    """
    skin_joints = rigged_skin["joints"]
    n_skin = len(skin_joints)
    joint_names = [_short_name(rigged_nodes[j].get("name", ""))
                   for j in skin_joints]

    # ── 1. Compute dominant-weight centroids ──────────────────────────
    # Only bones with ≥5 dominant vertices get a centroid.  SkinToken
    # often starves real Mixamo bones (e.g. LeftArm gets 0 dominant
    # verts because Extra bones absorb its area).  Starved bones have no
    # centroid and get their seed from bilateral mirroring instead.
    dominant_col = np.argmax(weights, axis=1)
    dominant_joint = joints[np.arange(len(joints)), dominant_col]

    dominant_centroids: dict[int, np.ndarray] = {}
    for j in range(n_skin):
        dom_mask = dominant_joint == j
        if int(dom_mask.sum()) >= 5:
            dominant_centroids[j] = (
                source_pos[dom_mask].astype(np.float64).mean(axis=0))

    # ── 2. Classify bones: Mixamo vs Extra ────────────────────────────
    extra_skin: set[int] = set()
    mixamo_skin: set[int] = set()
    for j, name in enumerate(joint_names):
        if "Extra" in name:
            extra_skin.add(j)
        else:
            mixamo_skin.add(j)

    if not extra_skin:
        log.info("[gpu_transfer] no Extra bones found, skipping sanitization")
        return rigged_nodes, rigged_skin, joints, weights

    # ── 3. Per-vertex spatial redistribution ───────────────────────────
    # For each VERTEX weighted to an Extra bone, find the NEAREST Mixamo
    # bone by vertex position.  Seed positions must be anatomically
    # correct — SkinToken's weights and IBMs are both garbled, so we
    # can't trust either directly.

    # SEED STRATEGY:
    #   - RIGHT-side limb bones have correct dominant-weight centroids
    #     (the right side of this character was rigged properly).
    #   - LEFT-side limb seeds = mirror of right-side centroid (X flip).
    #   - CENTRAL bones (Hips/Spine*) have correct centroids at X≈0.
    #   - Head is placed at the top of the mesh bounding box (its
    #     centroid is wrong — pulled into the arm area by garbage
    #     weights that gave Head 89% of its dominant verts in the arm).
    #   - Neck = midpoint between Spine2 and Head.
    _MIRROR = {
        "LeftShoulder": "RightShoulder", "RightShoulder": "LeftShoulder",
        "LeftArm": "RightArm", "RightArm": "LeftArm",
        "LeftForeArm": "RightForeArm", "RightForeArm": "LeftForeArm",
        "LeftHand": "RightHand", "RightHand": "LeftHand",
        "LeftUpLeg": "RightUpLeg", "RightUpLeg": "LeftUpLeg",
        "LeftLeg": "RightLeg", "RightLeg": "LeftLeg",
        "LeftFoot": "RightFoot", "RightFoot": "LeftFoot",
        "LeftToeBase": "RightToeBase", "RightToeBase": "LeftToeBase",
    }
    _CENTRAL = {"Hips", "Spine", "Spine1", "Spine2"}
    name_to_sj: dict[str, int] = {}
    for j, name in enumerate(joint_names):
        name_to_sj[_short_name(name)] = j

    # 3a. RIGHT-side limb centroids → seed; mirror for LEFT-side.
    mixamo_seed_pos: dict[int, np.ndarray] = {}
    for rname, lname in _MIRROR.items():
        if not rname.startswith("Right"):
            continue
        rsj = name_to_sj.get(rname)
        lsj = name_to_sj.get(lname)
        if rsj is None or rsj not in dominant_centroids:
            continue
        rpos = dominant_centroids[rsj].copy()
        mixamo_seed_pos[rsj] = rpos
        if lsj is not None:
            mpos = rpos.copy()
            mpos[0] = -mpos[0]  # mirror across X axis
            mixamo_seed_pos[lsj] = mpos

    # 3b. CENTRAL bones: dominant centroid with X forced to 0 (midline).
    for cname in _CENTRAL:
        csj = name_to_sj.get(cname)
        if csj is not None and csj in dominant_centroids:
            cpos = dominant_centroids[csj].copy()
            cpos[0] = 0.0
            mixamo_seed_pos[csj] = cpos

    # 3c. Head at top of mesh; Neck between Spine2 and Head.
    mesh_max_y = float(source_pos[:, 1].max())
    hsj = name_to_sj.get("Head")
    if hsj is not None:
        mixamo_seed_pos[hsj] = np.array([0.0, mesh_max_y * 0.9, 0.0])
    nsj = name_to_sj.get("Neck")
    s2sj = name_to_sj.get("Spine2")
    if nsj is not None and s2sj is not None and s2sj in mixamo_seed_pos:
        neck_y = (mixamo_seed_pos[s2sj][1] + mesh_max_y * 0.9) / 2.0
        mixamo_seed_pos[nsj] = np.array([0.0, neck_y, 0.0])

    # 3d. Fill any remaining gaps via mirror fallback.
    for short_name, mj in name_to_sj.items():
        if mj in mixamo_seed_pos or mj not in mixamo_skin:
            continue
        mirror_name = _MIRROR.get(short_name)
        if mirror_name and mirror_name in name_to_sj:
            mirror_sj = name_to_sj[mirror_name]
            if mirror_sj in mixamo_seed_pos:
                mpos = mixamo_seed_pos[mirror_sj].copy()
                mpos[0] = -mpos[0]
                mixamo_seed_pos[mj] = mpos
                log.info("[gpu_transfer] mirror seed: %s ← %s",
                         short_name, mirror_name)

    if not mixamo_seed_pos:
        log.warning("[gpu_transfer] no Mixamo seeds for redistribution")
        return rigged_nodes, rigged_skin, joints, weights

    # 3c. Build seed arrays for GPU nearest-neighbor
    seed_skin_joints = sorted(mixamo_seed_pos.keys())
    seed_positions = np.array(
        [mixamo_seed_pos[mj] for mj in seed_skin_joints],
        dtype=np.float32,
    )
    log.info("[gpu_transfer] per-vertex redistribution: %d Mixamo seeds "
             "for %d Extra bones", len(seed_skin_joints), len(extra_skin))

    # 3d. GPU: for each source vertex, find the 2 NEAREST Mixamo seeds.
    # The 2nd nearest is needed for Extra-dominant vertices whose original
    # weights are garbage (Head/Hips dominate in arm area) — we override
    # ALL their slots with a smooth blend of the 2 nearest bones.
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Clear any leftover GPU memory from the weight-transfer cdist
    if device == "cuda":
        torch.cuda.empty_cache()

    v_tensor = torch.from_numpy(source_pos.astype(np.float32)).to(device)
    s_tensor = torch.from_numpy(seed_positions).to(device)

    n_verts = len(source_pos)
    nearest_seed_idx = np.empty(n_verts, dtype=np.int64)
    second_seed_idx = np.empty(n_verts, dtype=np.int64)
    nearest_dist = np.empty(n_verts, dtype=np.float32)
    second_dist = np.empty(n_verts, dtype=np.float32)
    chunk_size = 4096 if device == "cuda" else 256
    for i in range(0, n_verts, chunk_size):
        end = min(i + chunk_size, n_verts)
        dists = torch.cdist(v_tensor[i:end], s_tensor)
        top2 = dists.topk(2, dim=1, largest=False)
        vals = top2.values.cpu().numpy()
        idxs = top2.indices.cpu().numpy()
        nearest_seed_idx[i:end] = idxs[:, 0]
        second_seed_idx[i:end] = idxs[:, 1]
        nearest_dist[i:end] = vals[:, 0]
        second_dist[i:end] = vals[:, 1]
        del dists, top2, vals, idxs

    # Free GPU tensors
    del v_tensor, s_tensor
    if device == "cuda":
        torch.cuda.empty_cache()

    # Map seed array index → skin joint index
    nearest_mixamo = np.array(
        [seed_skin_joints[si] for si in nearest_seed_idx],
        dtype=np.int64,
    )
    second_mixamo = np.array(
        [seed_skin_joints[si] for si in second_seed_idx],
        dtype=np.int64,
    )

    # 3e. Replace Extra bone influences with per-vertex nearest Mixamo bone.
    # For Extra-INFLUENCED vertices where a Mixamo bone is dominant, only
    # replace the Extra slots (preserving the good weights).
    # For Extra-DOMINANT vertices (whose weights are entirely garbage —
    # SkinToken gave Head/Hips more weight than arm bones in the arm area),
    # override ALL 4 slots with a smooth blend of the 2 nearest seeds.
    new_joints = joints.copy()
    new_weights = weights.astype(np.float64).copy()
    extra_arr = np.array(sorted(extra_skin), dtype=np.int64)

    # Identify Extra-dominant vertices (highest-weight bone is Extra)
    extra_dominant_mask = np.isin(dominant_joint, extra_arr)
    n_extra_dom = int(extra_dominant_mask.sum())

    # 3e-i. For non-Extra-dominant vertices: replace only Extra slots.
    # BUT only if the vertex's dominant bone is spatially consistent.
    # See 3e-iii for the misplaced-dominant override.
    non_dom = ~extra_dominant_mask
    for c in range(4):
        is_extra = non_dom & np.isin(joints[:, c], extra_arr)
        new_joints[is_extra, c] = nearest_mixamo[is_extra]

    # 3e-ii. For Extra-dominant vertices: full spatial override.
    # Blend the 2 nearest seeds using inverse-distance weighting.
    d1 = np.maximum(nearest_dist, 1e-6)
    d2 = np.maximum(second_dist, 1e-6)
    w1 = d2 / (d1 + d2)  # closer seed gets more weight
    w2 = d1 / (d1 + d2)

    ed = extra_dominant_mask
    new_joints[ed, 0] = nearest_mixamo[ed]
    new_joints[ed, 1] = second_mixamo[ed]
    new_joints[ed, 2] = nearest_mixamo[ed]
    new_joints[ed, 3] = second_mixamo[ed]
    new_weights[ed, 0] = w1[ed] * 0.6
    new_weights[ed, 1] = w2[ed] * 0.6
    new_weights[ed, 2] = w1[ed] * 0.4
    new_weights[ed, 3] = w2[ed] * 0.4

    # 3e-iii. Misplaced-dominant override.
    # SkinToken often makes a CENTRAL bone (Head, Hips, Spine) dominant
    # in a LIMB area (e.g., 19K Head-dominant verts in the left forearm).
    # These vertices are NOT Extra-dominant, so 3e-ii doesn't catch them.
    # Detect: if the vertex's dominant bone seed is much farther than the
    # nearest seed, the dominant bone is in the wrong territory → override.
    seed_pos_by_sj: dict[int, np.ndarray] = {
        sj_idx: seed_positions[seed_skin_joints.index(sj_idx)]
        for sj_idx in seed_skin_joints
    }

    # Build per-vertex dominant-bone seed distance.
    # For vertices whose dominant bone has a seed, compute distance.
    dom_seed_dist = np.full(n_verts, np.inf, dtype=np.float64)
    has_seed = np.zeros(n_verts, dtype=bool)
    for sj_idx, sp in seed_pos_by_sj.items():
        mask = dominant_joint == sj_idx
        if mask.any():
            diff = source_pos[mask].astype(np.float64) - sp
            dom_seed_dist[mask] = np.sqrt((diff ** 2).sum(axis=1))
            has_seed[mask] = True

    # Misplaced: dominant bone seed is > 3× farther than nearest seed
    MISPLACE_RATIO = 3.0
    misplace_mask = (
        has_seed
        & ~extra_dominant_mask
        & (dom_seed_dist > MISPLACE_RATIO * nearest_dist.astype(np.float64))
        & (dom_seed_dist > 0.03)  # absolute threshold: must be >3cm off
    )
    n_misplace = int(misplace_mask.sum())

    mp = misplace_mask
    new_joints[mp, 0] = nearest_mixamo[mp]
    new_joints[mp, 1] = second_mixamo[mp]
    new_joints[mp, 2] = nearest_mixamo[mp]
    new_joints[mp, 3] = second_mixamo[mp]
    new_weights[mp, 0] = w1[mp] * 0.6
    new_weights[mp, 1] = w2[mp] * 0.6
    new_weights[mp, 2] = w1[mp] * 0.4
    new_weights[mp, 3] = w2[mp] * 0.4

    log.info("[gpu_transfer] per-vertex redistribution: %d Extra-dominant, "
             "%d misplaced-dominant (full override), %d Extra-influenced "
             "(slot replacement)",
             n_extra_dom, n_misplace,
             int(np.any(np.isin(joints, extra_arr), axis=1).sum()) - n_extra_dom)

    # ── 5. Merge duplicate bones per vertex (vectorized) ──────────────
    # After replacement, a vertex may have the same bone in multiple slots.
    # Sort by joint index, then merge consecutive duplicates by summing weights.
    sort_idx = np.argsort(new_joints, axis=1)
    sj = np.take_along_axis(new_joints, sort_idx, axis=1)
    sw = np.take_along_axis(new_weights, sort_idx, axis=1)

    for c in range(3, 0, -1):
        dup = sj[:, c] == sj[:, c - 1]
        sw[dup, c - 1] += sw[dup, c]
        sw[dup, c] = 0.0

    # Reorder by weight descending (top 4 influences)
    w_sort = np.argsort(-sw, axis=1)
    final_joints = np.take_along_axis(sj, w_sort, axis=1)
    final_weights = np.take_along_axis(sw, w_sort, axis=1)

    # Renormalize
    w_sums = final_weights.sum(axis=1, keepdims=True)
    final_weights = final_weights / np.maximum(w_sums, 1e-10)

    # ── 6. Build cleaned skin joints (Extra removed) ──────────────────
    kept_skin = sorted(mixamo_skin)
    old_to_new_skin = {old: new for new, old in enumerate(kept_skin)}

    remapped_joints = np.zeros_like(final_joints)
    for old_j, new_j in old_to_new_skin.items():
        remapped_joints[final_joints == old_j] = new_j
    # Safety: any leftover Extra indices → 0
    for ej in extra_skin:
        remapped_joints[final_joints == ej] = 0

    new_skin_joints = [skin_joints[j] for j in kept_skin]

    # ── 7. Build new node list (Extra bones removed) ──────────────────
    node_remap: dict[int, int] = {}
    new_nodes: list[dict] = []
    for i, node in enumerate(rigged_nodes):
        name = _short_name(node.get("name", ""))
        if "Extra" in name:
            continue
        node_remap[i] = len(new_nodes)
        new_node = {}
        for k in ("name", "rotation", "scale", "extensions", "extras"):
            if k in node:
                new_node[k] = copy.deepcopy(node[k])
        if "translation" in node:
            new_node["translation"] = copy.deepcopy(node["translation"])
        new_nodes.append(new_node)

    # Remap skin joint node indices
    new_skin = dict(rigged_skin)
    new_skin["joints"] = [node_remap[j]
                          for j in new_skin_joints if j in node_remap]

    # ── 8. Rebuild hierarchy to canonical Mixamo ──────────────────────
    name_to_new: dict[str, int] = {}
    for i, node in enumerate(new_nodes):
        name = _short_name(node.get("name", ""))
        if name:
            name_to_new[name] = i

    children_map: dict[int, list[int]] = {}
    for name, new_idx in name_to_new.items():
        parent_name = MIXAMO_PARENTS.get(name)
        if parent_name and parent_name in name_to_new:
            parent_idx = name_to_new[parent_name]
            children_map.setdefault(parent_idx, []).append(new_idx)
        # Bones not in MIXAMO_PARENTS keep their original position (no
        # parent reassignment — they'll be roots or handled by caller)

    for parent_idx, child_list in children_map.items():
        new_nodes[parent_idx]["children"] = sorted(child_list)

    # Clear stale children references on bones whose hierarchy changed
    all_children = set()
    for cl in children_map.values():
        all_children.update(cl)
    for i, node in enumerate(new_nodes):
        if i not in children_map and i in all_children:
            node.pop("children", None)

    n_extra = len(extra_skin)
    log.info("[gpu_transfer] sanitized rig: removed %d Extra bones, "
             "redistributed %d → %d skin joints",
             n_extra, n_skin, len(new_skin_joints))

    return new_nodes, new_skin, remapped_joints, final_weights


# ── Bone position repair ────────────────────────────────────────────────

def _quat_to_mat3(q: list | tuple) -> np.ndarray:
    """Convert quaternion [x, y, z, w] to 3×3 rotation matrix."""
    x, y, z, w = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w),   2*(x*z+y*w)],
        [2*(x*y+z*w),   1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w),   2*(y*z+x*w),   1-2*(x*x+y*y)],
    ], dtype=np.float64)


def _repair_bone_positions(
    rigged_nodes: list,
    rigged_skin: dict,
    source_pos: np.ndarray,
    joints: np.ndarray,
    weights: np.ndarray,
) -> tuple[dict, dict, np.ndarray]:
    """Repair bone positions using mesh skin-weight centroids.

    SkinToken's skeleton often has garbled bone positions (wrong pivot
    points, bones pointing sideways instead of down). This computes the
    weighted centroid of each bone's influence area directly from the
    source mesh, giving anatomically correct pivot points.

    The deformation formula is:  v' = rest_pos + delta @ (v - rest_pos)
    — a pure rotation around rest_pos. If rest_pos is wrong, the limb
    swings in a distorted arc. Fixing rest_pos fixes walking, bending,
    and all articulated motion.

    Returns:
        (world_positions, new_translations, ibm_data) where:
        - world_positions: {node_idx: np.ndarray[3]} world space positions
        - new_translations: {node_idx: np.ndarray[3]} local translations
        - ibm_data: (n_joints, 16) inverse bind matrices (column-major)
    """
    skin_joint_nodes = rigged_skin["joints"]
    n_joints = len(skin_joint_nodes)

    # 1. Compute centroid for each skin joint.
    #
    # PRIMARY: dominant-weight centroid — only vertices where this bone has the
    # highest weight among its 4 influences. This gives the center of the bone's
    # exclusive territory, producing correct separation between adjacent bones
    # (e.g., LeftArm's centroid is in the upper-arm area, LeftForeArm's is in
    # the forearm area — 20+ cm apart, not overlapping).
    #
    # FALLBACK: weighted centroid of all influenced vertices (for bones with
    # too few dominant vertices, e.g., sparse finger/toe bones).
    dominant_col = np.argmax(weights, axis=1)  # (M,) which of 4 slots is highest
    dominant_joint = joints[np.arange(len(joints)), dominant_col]  # (M,) bone idx

    centroids: dict[int, np.ndarray] = {}
    n_dominant = 0
    n_fallback = 0
    for j in range(n_joints):
        node_idx = skin_joint_nodes[j]
        dom_mask = dominant_joint == j
        if dom_mask.sum() >= 5:
            # Dominant-weight centroid: simple mean of vertices where this
            # bone reigns supreme
            centroids[node_idx] = source_pos[dom_mask].astype(np.float64).mean(axis=0)
            n_dominant += 1
        else:
            # Fallback: weighted centroid of all influenced vertices
            mask = np.any(joints == j, axis=1)
            if not mask.any():
                continue
            bone_w = np.zeros(mask.sum(), dtype=np.float64)
            for c in range(4):
                jm = joints[mask, c] == j
                bone_w = np.maximum(bone_w, weights[mask, c].astype(np.float64) * jm)
            total = bone_w.sum()
            if total > 1e-6:
                centroids[node_idx] = (
                    bone_w[:, None] * source_pos[mask].astype(np.float64)
                ).sum(axis=0) / total
                n_fallback += 1

    log.info("[gpu_transfer] bone repair: %d/%d joints have centroids "
             "(%d dominant, %d fallback)",
             len(centroids), n_joints, n_dominant, n_fallback)

    # 2. Build children map and find roots
    children_map: dict[int, list] = {}
    all_children: set[int] = set()
    for i, node in enumerate(rigged_nodes):
        for child in node.get("children", []):
            children_map.setdefault(i, []).append(child)
            all_children.add(child)

    root_indices = [
        i for i in range(len(rigged_nodes)) if i not in all_children
    ]

    # 3. Walk hierarchy: compute world positions + rotations
    world_pos: dict[int, np.ndarray] = {}
    world_rot: dict[int, np.ndarray] = {}

    def walk(node_idx: int, parent_wp: np.ndarray, parent_wr: np.ndarray):
        node = rigged_nodes[node_idx]
        local_rot = _quat_to_mat3(node.get("rotation", [0, 0, 0, 1]))

        if node_idx in centroids:
            wp = centroids[node_idx]
        else:
            # Non-skin node: use original translation composed with parent
            local_t = np.array(
                node.get("translation", [0, 0, 0]), dtype=np.float64
            )
            wp = parent_wp + parent_wr @ local_t

        wr = parent_wr @ local_rot
        world_pos[node_idx] = wp
        world_rot[node_idx] = wr

        for child in children_map.get(node_idx, []):
            walk(child, wp, wr)

    for root in root_indices:
        if root in centroids:
            wp = centroids[root]
        else:
            wp = np.array(
                rigged_nodes[root].get("translation", [0, 0, 0]),
                dtype=np.float64,
            )
        wr = _quat_to_mat3(
            rigged_nodes[root].get("rotation", [0, 0, 0, 1])
        )
        world_pos[root] = wp
        world_rot[root] = wr
        for child in children_map.get(root, []):
            walk(child, wp, wr)

    # 4. Compute new local translations from world positions
    parent_map: dict[int, int] = {}
    for parent_idx, children in children_map.items():
        for child in children:
            parent_map[child] = parent_idx

    new_translations: dict[int, np.ndarray] = {}
    for node_idx in world_pos:
        parent_idx = parent_map.get(node_idx)
        if parent_idx is not None and parent_idx in world_pos:
            new_t = world_rot[parent_idx].T @ (
                world_pos[node_idx] - world_pos[parent_idx]
            )
        else:
            new_t = world_pos[node_idx]
        new_translations[node_idx] = new_t.astype(np.float32)

    # 5. Compute IBM for each skin joint
    ibm_data = np.zeros((n_joints, 16), dtype=np.float32)
    for j, node_idx in enumerate(skin_joint_nodes):
        wp = world_pos.get(node_idx, np.zeros(3))
        wr = world_rot.get(node_idx, np.eye(3))
        world_mat = np.eye(4, dtype=np.float64)
        world_mat[:3, :3] = wr
        world_mat[:3, 3] = wp
        ibm_mat = np.linalg.inv(world_mat)
        # Column-major storage (glTF convention)
        ibm_data[j] = ibm_mat.T.reshape(16).astype(np.float32)

    return world_pos, new_translations, ibm_data


# ── GLB assembly ────────────────────────────────────────────────────────

def _write_output_glb(
    source_gltf: dict,
    source_bin: bytes,
    joints: np.ndarray,
    weights: np.ndarray,
    rigged_gltf: dict,
    rigged_bin: bytes,
    T: np.ndarray,
    output_path: str,
    source_pos: np.ndarray,
) -> None:
    """Assemble output GLB: source geometry + transferred weights + armature.

    Takes the source GLB (has textures, materials, correct mesh) and injects:
    - JOINTS_0 + WEIGHTS_0 accessors on the mesh primitive
    - Skin definition (joints + IBM) from the rigged GLB
    - Bone node hierarchy from the rigged GLB

    CRITICAL: The bone translations and IBM are transformed from the rigged
    coordinate space into the source coordinate space using T (the AABB
    similarity transform). Without this, the bones are in a different scale
    than the mesh vertices → catastrophic skinning deformation.

    IMPORTANT: If the source GLB already has a skin (from a previous run),
    it is DISCARDED entirely. The output gets a SINGLE clean skin from the
    rigged GLB — no leftover bones, no duplicate skins, no index confusion.
    """
    gltf = copy.deepcopy(source_gltf)
    new_bin = bytearray(source_bin)

    gltf.setdefault("bufferViews", [])
    gltf.setdefault("accessors", [])

    def _pad4(b: bytearray) -> None:
        while len(b) % 4:
            b.append(0)

    # ── Sanitize rig: remove Extra bones, redistribute weights ───────────
    # SkinToken inserts Extra_XX bones that absorb 30%+ of mesh weights but
    # are never animated, and garbles the hierarchy (e.g., LeftArm nested
    # inside Head). This fixes both BEFORE we write any data.
    rigged_nodes = rigged_gltf.get("nodes", [])
    rigged_skins = rigged_gltf.get("skins", [])
    if not rigged_skins:
        raise RuntimeError("Rigged GLB has no skin definition")
    rigged_skin = rigged_skins[0]

    clean_nodes, clean_skin, joints, weights = _sanitize_rig(
        rigged_nodes, rigged_skin, joints, weights, source_pos,
    )

    n_verts = len(joints)

    # ── Add JOINTS_0 ───────────────────────────────────────────────────
    max_joint = int(joints.max())
    if max_joint > 255:
        joints_dt = np.uint16
        joints_comp = 5123  # UNSIGNED_SHORT
    else:
        joints_dt = np.uint8
        joints_comp = 5121  # UNSIGNED_BYTE

    joints_bytes = joints.astype(joints_dt).tobytes()

    _pad4(new_bin)
    joints_bv = len(gltf["bufferViews"])
    joints_off = len(new_bin)
    new_bin.extend(joints_bytes)
    gltf["bufferViews"].append({
        "buffer": 0, "byteOffset": joints_off,
        "byteLength": len(joints_bytes),
        "target": 34962,
    })
    joints_acc = len(gltf["accessors"])
    gltf["accessors"].append({
        "bufferView": joints_bv,
        "componentType": joints_comp,
        "count": n_verts,
        "type": "VEC4",
    })

    # ── Add WEIGHTS_0 ──────────────────────────────────────────────────
    weights_bytes = weights.astype(np.float32).tobytes()
    _pad4(new_bin)
    weights_bv = len(gltf["bufferViews"])
    weights_off = len(new_bin)
    new_bin.extend(weights_bytes)
    gltf["bufferViews"].append({
        "buffer": 0, "byteOffset": weights_off,
        "byteLength": len(weights_bytes),
        "target": 34962,
    })
    weights_acc = len(gltf["accessors"])
    gltf["accessors"].append({
        "bufferView": weights_bv,
        "componentType": 5126,
        "count": n_verts,
        "type": "VEC4",
    })

    # ── Update mesh primitive to use our JOINTS_0/WEIGHTS_0 ────────────
    source_prim = gltf["meshes"][0]["primitives"][0]
    source_prim["attributes"]["JOINTS_0"] = joints_acc
    source_prim["attributes"]["WEIGHTS_0"] = weights_acc

    # ── Repair bone positions using mesh skin-weight centroids ──────────
    # Now that Extra bones are gone and weights are redistributed, compute
    # accurate bone pivot points from dominant-weight centroids.
    _, repaired_translations, ibm_repaired = _repair_bone_positions(
        clean_nodes, clean_skin, source_pos, joints, weights,
    )

    # Write repaired IBM
    ibm_new_acc = None
    ibm_bytes = ibm_repaired.tobytes()
    _pad4(new_bin)
    ibm_bv = len(gltf["bufferViews"])
    ibm_off = len(new_bin)
    new_bin.extend(ibm_bytes)
    gltf["bufferViews"].append({
        "buffer": 0, "byteOffset": ibm_off,
        "byteLength": len(ibm_bytes),
    })
    ibm_new_acc = len(gltf["accessors"])
    gltf["accessors"].append({
        "bufferView": ibm_bv,
        "componentType": 5126,
        "count": len(ibm_repaired),
        "type": "MAT4",
    })

    # ── REBUILD node hierarchy from scratch ────────────────────────────
    # Collect mesh node info from the SOURCE GLB (keep mesh references,
    # discard everything else — old bones, old skin refs, armature, etc.)
    source_mesh_infos = []
    for node in source_gltf.get("nodes", []):
        if "mesh" in node:
            info = {"mesh": node["mesh"]}
            if "name" in node:
                info["name"] = node["name"]
            if "translation" in node:
                info["translation"] = node["translation"]
            if "rotation" in node:
                info["rotation"] = node["rotation"]
            if "scale" in node:
                info["scale"] = node["scale"]
            source_mesh_infos.append(info)

    # Build fresh node list from SANITIZED nodes + repaired translations.
    # The sanitized nodes already have:
    # - Extra bones removed
    # - Canonical Mixamo children references (from MIXAMO_PARENTS)
    # Bone ROTATIONS are preserved from the rigged GLB (they define the
    # bone orientation, which doesn't affect deformation — only the pivot
    # position matters for the formula v' = rest_pos + delta @ (v - rest_pos)).
    new_nodes = []
    for i, cn in enumerate(clean_nodes):
        new_node = {}
        for k in ("name", "rotation", "scale",
                   "extensions", "extras"):
            if k in cn:
                new_node[k] = copy.deepcopy(cn[k])
        if i in repaired_translations:
            new_node["translation"] = repaired_translations[i].tolist()
        elif "translation" in cn:
            new_node["translation"] = copy.deepcopy(cn["translation"])
        if "children" in cn:
            new_node["children"] = copy.deepcopy(cn["children"])
        new_nodes.append(new_node)

    # Mesh nodes (after bones)
    mesh_node_indices = []
    for info in source_mesh_infos:
        ni = len(new_nodes)
        mesh_node_indices.append(ni)
        new_nodes.append(info)

    gltf["nodes"] = new_nodes

    # ── SINGLE skin definition (replace any existing) ──────────────────
    # Preserve the source rig's identity fields (name, skeleton, extras)
    # by starting from clean_skin — a shallow copy of the source skin with
    # only `joints` remapped. The previous code built a fresh dict with
    # ONLY joints + IBM, which dropped `skin.name`. Downstream the
    # somax_bake node hard-requires `skin.name === "SOMA-Mixamo"` (77
    # joints) and rejects anything else — so a transfer_rig output could
    # not be baked. Preserving the lineage fields fixes that without
    # lying: this node's purpose IS to copy a SOMAX rig onto a new mesh,
    # so the output retains SOMAX identity. If the source wasn't SOMAX
    # (skin.name absent or different), the absence propagates and the
    # validator still correctly rejects it.
    skin = dict(clean_skin)
    skin["joints"] = list(clean_skin["joints"])
    if ibm_new_acc is not None:
        skin["inverseBindMatrices"] = ibm_new_acc
    gltf["skins"] = [skin]
    skin_idx = 0

    # Assign skin to mesh nodes
    for ni in mesh_node_indices:
        new_nodes[ni]["skin"] = skin_idx

    # ── Rebuild scene: root bones + mesh nodes ─────────────────────────
    all_children = set()
    for node in new_nodes:
        all_children.update(node.get("children", []))
    root_bones = [i for i in range(len(new_nodes))
                  if i not in all_children and i not in mesh_node_indices]
    scene_nodes = root_bones + mesh_node_indices
    gltf["scenes"] = [{"nodes": scene_nodes}]
    gltf["scene"] = 0

    # ── Update buffer length to include all appended data ─────────────
    # CRITICAL: The source GLB's buffer byteLength only covers the original
    # binary. Our new JOINTS_0/WEIGHTS_0/IBM data was appended beyond that.
    # Without updating, glTF loaders won't read the new data.
    gltf["buffers"][0]["byteLength"] = len(new_bin)

    # ── Write the GLB ──────────────────────────────────────────────────
    json_bytes = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    # GLB spec: JSON chunk padded with SPACE (0x20), NOT null bytes
    while len(json_bytes) % 4:
        json_bytes += b" "
    _pad4(new_bin)

    total_len = 12 + 8 + len(json_bytes) + 8 + len(new_bin)
    with open(output_path, "wb") as f:
        f.write(struct.pack("<III", 0x46546C67, 2, total_len))
        f.write(struct.pack("<II", len(json_bytes), 0x4E4F534A))
        f.write(json_bytes)
        f.write(struct.pack("<II", len(new_bin), 0x004E4942))
        f.write(new_bin)

    log.info("[gpu_transfer] wrote %s (%.1f MB)",
             output_path, len(new_bin) / 1024 / 1024)
