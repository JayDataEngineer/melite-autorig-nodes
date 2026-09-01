"""SOMAX weight-cleanup surgery — ComfyUI-node-callable port of the host-side
stage4a2_sync_sibling_weights + stage4a_clean_weights from
core/pipelines/character.py. Operates directly on GLB binary (numpy only —
no Blender, no GPU). Lifted verbatim so the one-flow ComfyUI workflow has ZERO
host-side deviations."""
from __future__ import annotations
import os
import logging
import struct
import json as _json
from collections import defaultdict
import numpy as np
log = logging.getLogger("melite.weight_surgery")

def stage4a2_sync_sibling_weights(rigged_glb_path: str, output_dir: str) -> str:
    """Unify bone weights of coincident (sibling) vertices so xatlas UV-seam
    splits can't rip during animation.

    xatlas re-unwrap cuts the watertight mesh along UV seams, duplicating
    vertices at each seam (siblings share the same 3D position, different UV).
    SOMAX/bone_dist can assign siblings different bone weights → when the
    skeleton animates, the two sides of a seam move at different rates and the
    mesh rips open (the 'see-through/hollow' artifact). Verified run 1bd272db:
    662/2799 sibling groups had different weights/joints.

    Fix: group vertices by exact 3D position; for each group, merge the
    (joint, weight) pairs across all siblings, take the top-4 by summed weight,
    renormalize, assign to every sibling. Siblings now have identical weights
    → they move in perfect sync → seams never pull apart.

    Operates directly on the GLB binary (mirrors stage4a_bone_distance_weights).
    """

    log.info("=" * 60)
    log.info("Stage 4a2: Sibling Weight Sync (UV-seam split unification)")
    log.info("=" * 60)

    with open(rigged_glb_path, "rb") as f:
        raw = f.read()
    magic, version, total_len = struct.unpack_from("<III", raw, 0)
    json_len, json_type = struct.unpack_from("<II", raw, 12)
    gltf = _json.loads(raw[20:20 + json_len].decode())
    bin_len, bin_type = struct.unpack_from("<II", raw, 20 + json_len)
    bindata = bytearray(raw[20 + json_len + 8:20 + json_len + 8 + bin_len])

    def _read_acc(acc_idx):
        acc = gltf['accessors'][acc_idx]
        bv = gltf['bufferViews'][acc['bufferView']]
        offset = bv.get('byteOffset', 0) + acc.get('byteOffset', 0)
        count = acc['count']
        dtype_map = {5126: 'f4', 5123: 'u2', 5121: 'u1', 5125: 'u4'}
        fmt = dtype_map[acc['componentType']]
        ncomp = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4}[acc['type']]
        arr = np.frombuffer(bytes(bindata), dtype=np.dtype(fmt), count=count * ncomp, offset=offset)
        return arr.reshape(count, ncomp) if ncomp > 1 else arr

    prim = gltf['meshes'][0]['primitives'][0]
    verts = _read_acc(prim['attributes']['POSITION']).astype(np.float64)
    j_dtype = gltf['accessors'][prim['attributes']['JOINTS_0']]['componentType']
    j_fmt = {5121: 'u1', 5123: 'u2', 5125: 'u4'}[j_dtype]
    joints = _read_acc(prim['attributes']['JOINTS_0']).astype(np.int64)
    weights = _read_acc(prim['attributes']['WEIGHTS_0']).astype(np.float64)

    # Group vertices by exact 3D position (siblings from the xatlas UV split).
    coord_map = defaultdict(list)
    for i, v in enumerate(verts):
        coord_map[tuple(np.round(v, 5))].append(i)

    new_joints = joints.copy()
    new_weights = weights.copy()
    n_synced = 0
    for coord, idxs in coord_map.items():
        if len(idxs) < 2:
            continue
        # Merge influences: sum weights per joint across all siblings.
        pair_w = {}
        for vi in idxs:
            for j, w in zip(joints[vi], weights[vi]):
                if w < 1e-6:
                    continue
                pair_w[int(j)] = pair_w.get(int(j), 0.0) + float(w)
        if not pair_w:
            continue
        top = sorted(pair_w.items(), key=lambda x: -x[1])[:4]
        tj = np.array([t[0] for t in top] + [0] * (4 - len(top)), dtype=np.int64)
        tw = np.array([t[1] for t in top] + [0.0] * (4 - len(top)), dtype=np.float64)
        tw_sum = tw.sum()
        if tw_sum > 1e-10:
            tw = tw / tw_sum
        for vi in idxs:
            new_joints[vi] = tj
            new_weights[vi] = tw
        n_synced += 1

    # Write JOINTS_0 + WEIGHTS_0 back (same dtype/shape → in-place overwrite).
    joints_bytes = new_joints.astype(np.dtype(j_fmt)).tobytes()
    weights_bytes = new_weights.astype(np.float32).tobytes()
    for acc_idx, new_bytes in [(prim['attributes']['JOINTS_0'], joints_bytes),
                                (prim['attributes']['WEIGHTS_0'], weights_bytes)]:
        acc = gltf['accessors'][acc_idx]
        bv = gltf['bufferViews'][acc['bufferView']]
        offset = bv.get('byteOffset', 0) + acc.get('byteOffset', 0)
        bindata[offset:offset + len(new_bytes)] = new_bytes

    json_str = _json.dumps(gltf, separators=(',', ':'))
    while len(json_str) % 4 != 0:
        json_str += ' '
    json_bytes = json_str.encode('utf-8')
    while len(bindata) % 4 != 0:
        bindata += b'\x00'
    total_len = 12 + 8 + len(json_bytes) + 8 + len(bindata)
    glb_out = bytearray()
    glb_out += struct.pack('<I', 0x46546C67)
    glb_out += struct.pack('<I', 2)
    glb_out += struct.pack('<I', total_len)
    glb_out += struct.pack('<I', len(json_bytes))
    glb_out += struct.pack('<I', 0x4E4F534A)
    glb_out += json_bytes
    glb_out += struct.pack('<I', len(bindata))
    glb_out += struct.pack('<I', 0x004E4942)
    glb_out += bytes(bindata)

    output_path = os.path.join(output_dir, "stage4a2_sync_weights.glb")
    with open(output_path, "wb") as f:
        f.write(glb_out)
    log.info("  Synced %d sibling groups → identical weights (%d bytes)",
             n_synced, len(glb_out))
    return output_path


def stage4a_clean_weights(rigged_glb_path: str, output_dir: str) -> str:
    """Clean SOMAX weight binding errors before animation baking.

    Six passes:
    1. Arm-torso isolation: zero arm/shoulder weights on torso vertices
       (|X| < 0.04 from centerline) to prevent wing/stretch artifacts.
    1b. Shoulder rebalancing: cap spine influence at 15% near the shoulder
        joint, redistribute excess to Shoulder+Arm. Fixes "thin shoulder".
    1c. Armpit decontamination: remove arm+torso cross-contamination in the
        armpit region. Fixes "sticky parts under the armpit" webbing.
    1d. Wrist blend zone: smooth hard 100% ForeArm->100% Hand transition to
        a gradient. Fixes "hands to wrist broken" candy-wrapper collapse.
    2. Cross-body decontamination: for each vertex, if both left and right
       arm bones have weight, keep only the dominant side.
    3. Renormalize all weights to sum to 1.0.

    Operates directly on the GLB binary — no external dependencies.
    """

    log.info("=" * 60)
    log.info("Stage 4a: Weight Cleanup (arm-torso isolation)")
    log.info("=" * 60)

    with open(rigged_glb_path, "rb") as f:
        raw = f.read()

    # Parse GLB
    magic, version, total_len = struct.unpack_from("<III", raw, 0)
    json_len, json_type = struct.unpack_from("<II", raw, 12)
    gltf = _json.loads(raw[20:20 + json_len].decode())
    bin_len, bin_type = struct.unpack_from("<II", raw, 20 + json_len)
    bindata = bytearray(raw[20 + json_len + 8:20 + json_len + 8 + bin_len])

    # Read accessors
    def _read_acc(acc_idx):
        acc = gltf['accessors'][acc_idx]
        bv = gltf['bufferViews'][acc['bufferView']]
        offset = bv.get('byteOffset', 0) + acc.get('byteOffset', 0)
        count = acc['count']
        dtype_map = {5126: 'f4', 5123: 'u2', 5121: 'u1', 5122: 'i2', 5125: 'u4'}
        fmt = dtype_map[acc['componentType']]
        ncomp = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4, 'MAT4': 16}[acc['type']]
        arr = np.frombuffer(bytes(bindata), dtype=np.dtype(fmt), count=count * ncomp, offset=offset)
        return arr.reshape(count, ncomp) if ncomp > 1 else arr

    prim = gltf['meshes'][0]['primitives'][0]
    verts = _read_acc(prim['attributes']['POSITION']).astype(np.float64)
    joints = _read_acc(prim['attributes']['JOINTS_0']).astype(int)
    weights = _read_acc(prim['attributes']['WEIGHTS_0']).astype(np.float64)
    V = len(verts)

    # Identify joint groups from skin
    skin = gltf['skins'][0]
    joint_names = []
    for node_idx in skin['joints']:
        name = gltf['nodes'][node_idx].get('name', f'joint_{node_idx}')
        joint_names.append(name)

    # Classify joints
    arm_joints = set()
    left_arm = set()
    right_arm = set()
    torso_joints = set()
    for idx, name in enumerate(joint_names):
        short = name.replace("mixamorig:", "")
        if any(k in short for k in ["Arm", "ForeArm", "Hand", "Shoulder",
                                     "Thumb", "Index", "Middle", "Ring", "Pinky"]):
            arm_joints.add(idx)
            if "Left" in short or "left" in short:
                left_arm.add(idx)
            elif "Right" in short or "right" in short:
                right_arm.add(idx)
        elif any(k in short for k in ["Spine", "Hips", "Neck", "Head", "Chest",
                                       "UpLeg", "Leg", "Foot", "Toe", "Jaw", "Eye"]):
            torso_joints.add(idx)

    log.info("  Arm joints: %d (L:%d R:%d), Torso/other: %d",
             len(arm_joints), len(left_arm), len(right_arm), len(torso_joints))

    # Build per-vertex full weight vectors
    n_joints = len(joint_names)
    wv = np.zeros((V, n_joints), dtype=np.float64)
    for v in range(V):
        for k in range(4):
            wv[v, joints[v, k]] += weights[v, k]

    fixed_arm = 0
    fixed_cross = 0

    # Pass 1: Arm-torso isolation
    # Torso vertices: |X| < 0.04m from centerline AND Y between hips and neck
    centerline_threshold = 0.04  # 4cm
    y_min, y_max = verts[:, 1].min(), verts[:, 1].max()
    torso_y_range = (verts[:, 1] > y_min + 0.15) & (verts[:, 1] < y_max - 0.15)
    near_centerline = np.abs(verts[:, 0]) < centerline_threshold
    torso_mask = torso_y_range & near_centerline

    for v in np.where(torso_mask)[0]:
        arm_w = sum(wv[v, j] for j in arm_joints if j < n_joints)
        if arm_w > 0.05:
            # Zero out all arm weights
            for j in arm_joints:
                if j < n_joints:
                    wv[v, j] = 0.0
            fixed_arm += 1

    log.info("  Arm-torso isolation: fixed %d vertices (>5%% arm weight on torso)", fixed_arm)

    # ── Read inverse bind matrices for joint positions ────────────────
    ibm_acc_idx = skin.get('inverseBindMatrices')
    joint_world_pos = {}
    if ibm_acc_idx is not None:
        ibm_raw = _read_acc(ibm_acc_idx)
        # glTF MAT4 is column-major — reshape and transpose
        ibms = ibm_raw.reshape(len(joint_names), 4, 4).swapaxes(1, 2)
        for ji in range(len(joint_names)):
            try:
                joint_world_pos[ji] = np.linalg.inv(ibms[ji])[:3, 3]
            except np.linalg.LinAlgError:
                pass

    def _joint_idx(short_name):
        """Get joint index by short name (without mixamorig: prefix)."""
        for ji, jn in enumerate(joint_names):
            if jn.replace("mixamorig:", "") == short_name:
                return ji
        return -1

    # Pass 1b: Shoulder rebalancing — reduce spine influence on shoulder verts
    # Root cause: Spine2 has ~40% weight on shoulder vertices, causing
    # pinching when the arm rotates (Spine2 fights with Shoulder/Arm).
    # Fix: cap torso weight at 15% near the shoulder joint, redistribute
    # the excess to the shoulder + arm joints.
    fixed_shoulder = 0
    for side in ["Left", "Right"]:
        sh_ji = _joint_idx(f"{side}Shoulder")
        arm_ji = _joint_idx(f"{side}Arm")
        if sh_ji < 0 or arm_ji < 0:
            continue
        sh_pos = joint_world_pos.get(sh_ji)
        arm_pos = joint_world_pos.get(arm_ji)
        if sh_pos is None or arm_pos is None:
            continue

        center = (sh_pos + arm_pos) / 2
        dists = np.linalg.norm(verts - center, axis=1)
        shoulder_verts = np.where(dists < 0.05)[0]

        # Torso joints that contaminate the shoulder
        torso_shoulder = set()
        for idx, name in enumerate(joint_names):
            short = name.replace("mixamorig:", "")
            if any(k in short for k in ["Spine", "Hips", "Neck"]):
                torso_shoulder.add(idx)

        for v in shoulder_verts:
            torso_w = sum(wv[v, j] for j in torso_shoulder if j < n_joints)
            if torso_w > 0.15:
                excess = torso_w - 0.15
                # Scale down torso joints proportionally
                scale = 0.15 / torso_w if torso_w > 0 else 0
                for j in torso_shoulder:
                    if j < n_joints:
                        wv[v, j] *= scale
                # Give excess to shoulder + arm
                sh_w = wv[v, sh_ji] if sh_ji < n_joints else 0
                arm_w = wv[v, arm_ji] if arm_ji < n_joints else 0
                cur = sh_w + arm_w
                if cur > 0.001:
                    wv[v, sh_ji] += excess * (sh_w / cur)
                    wv[v, arm_ji] += excess * (arm_w / cur)
                else:
                    wv[v, sh_ji] += excess
                fixed_shoulder += 1

    log.info("  Shoulder rebalancing: fixed %d vertices (spine >15%% → redistributed)",
             fixed_shoulder)

    # Pass 1c: Armpit decontamination — remove arm+torso cross-contamination
    # Root cause: ~30% of armpit vertices have BOTH arm and spine weights,
    # causing "webbing" or "sticky" mesh under the arms when they lift.
    # Fix: for armpit vertices, decide arm-vs-torso by distance to each
    # joint group, then zero out the loser.
    fixed_armpit = 0
    for side in ["Left", "Right"]:
        arm_ji = _joint_idx(f"{side}Arm")
        spine2_ji = _joint_idx("Spine2")
        if arm_ji < 0 or spine2_ji < 0:
            continue
        arm_pos = joint_world_pos.get(arm_ji)
        spine2_pos = joint_world_pos.get(spine2_ji)
        if arm_pos is None or spine2_pos is None:
            continue

        # Armpit: between arm and spine2, below the shoulder
        armpit_center = np.array([
            arm_pos[0] * 0.45,  # medial from arm
            (arm_pos[1] + spine2_pos[1]) / 2 - 0.02,  # slightly below
            arm_pos[2],
        ])
        dists = np.linalg.norm(verts - armpit_center, axis=1)
        armpit_verts = np.where(dists < 0.06)[0]

        side_arm_set = left_arm if side == "Left" else right_arm
        for v in armpit_verts:
            arm_w = sum(wv[v, j] for j in side_arm_set if j < n_joints)
            torso_w = sum(wv[v, j] for j in torso_joints if j < n_joints)
            if arm_w > 0.05 and torso_w > 0.05:
                # Decide by X-position: verts closer to centerline → torso,
                # verts further out → arm
                if abs(verts[v, 0]) > abs(arm_pos[0]) * 0.6:
                    # Closer to arm — zero torso
                    for j in torso_joints:
                        if j < n_joints:
                            wv[v, j] = 0.0
                else:
                    # Closer to torso — zero arm
                    for j in side_arm_set:
                        if j < n_joints:
                            wv[v, j] = 0.0
                fixed_armpit += 1

    log.info("  Armpit decontamination: fixed %d vertices (arm+torso webbing)", fixed_armpit)

    # Pass 1d: Wrist blend zone — smooth the hard ForeArm→Hand transition
    # Root cause: wrist vertices are 100% ForeArm, hand vertices are 100%
    # Hand — no gradient. This causes "candy-wrapper" collapse at the wrist.
    # Fix: for vertices near the wrist joint, blend ForeArm + Hand weights.
    fixed_wrist = 0
    for side in ["Left", "Right"]:
        hand_ji = _joint_idx(f"{side}Hand")
        fa_ji = _joint_idx(f"{side}ForeArm")
        if hand_ji < 0 or fa_ji < 0:
            continue
        hand_pos = joint_world_pos.get(hand_ji)
        fa_pos = joint_world_pos.get(fa_ji)
        if hand_pos is None or fa_pos is None:
            continue

        wrist_center = (hand_pos + fa_pos) / 2
        wrist_axis = hand_pos - fa_pos
        wrist_len = np.linalg.norm(wrist_axis)
        if wrist_len < 0.001:
            continue
        wrist_axis /= wrist_len

        # Find vertices in the wrist transition zone
        dists = np.linalg.norm(verts - wrist_center, axis=1)
        wrist_verts = np.where(dists < 0.03)[0]

        for v in wrist_verts:
            # Project onto wrist axis: -1 at forearm, +1 at hand
            rel = np.dot(verts[v] - wrist_center, wrist_axis) / (wrist_len / 2)
            # Clamp to [-1, 1]
            t = np.clip(rel, -1, 1)
            # Blend: 0% hand at forearm side → 100% hand at hand side
            hand_target = (t + 1) / 2  # 0 at forearm, 1 at hand

            cur_fa = wv[v, fa_ji] if fa_ji < n_joints else 0
            cur_hand = wv[v, hand_ji] if hand_ji < n_joints else 0
            combined = cur_fa + cur_hand
            if combined > 0.01:
                new_fa = combined * (1 - hand_target)
                new_hand = combined * hand_target
                if abs(new_fa - cur_fa) > 0.05 or abs(new_hand - cur_hand) > 0.05:
                    wv[v, fa_ji] = new_fa
                    wv[v, hand_ji] = new_hand
                    fixed_wrist += 1

    log.info("  Wrist blend zone: smoothed %d vertices (ForeArm↔Hand gradient)", fixed_wrist)

    # Pass 2: Cross-body decontamination
    for v in range(V):
        left_w = sum(wv[v, j] for j in left_arm if j < n_joints)
        right_w = sum(wv[v, j] for j in right_arm if j < n_joints)
        if left_w > 0.01 and right_w > 0.01:
            if left_w >= right_w:
                for j in right_arm:
                    if j < n_joints:
                        wv[v, j] = 0.0
            else:
                for j in left_arm:
                    if j < n_joints:
                        wv[v, j] = 0.0
            fixed_cross += 1

    log.info("  Cross-body decontamination: fixed %d vertices", fixed_cross)

    # Pass 3: Renormalize + truncate to top-4
    new_joints = np.zeros((V, 4), dtype=np.uint16)
    new_weights = np.zeros((V, 4), dtype=np.float32)
    for v in range(V):
        row = wv[v]
        top4 = np.argsort(row)[::-1][:4]
        w_sum = row[top4].sum()
        if w_sum > 0.001:
            for k in range(4):
                new_joints[v, k] = top4[k]
                new_weights[v, k] = row[top4[k]] / w_sum
        else:
            new_joints[v, 0] = 0  # bind to root
            new_weights[v, 0] = 1.0

    # Write cleaned weights back to GLB
    # Find the WEIGHTS_0 and JOINTS_0 bufferviews and overwrite in-place
    w_acc_idx = prim['attributes']['WEIGHTS_0']
    j_acc_idx = prim['attributes']['JOINTS_0']
    w_acc = gltf['accessors'][w_acc_idx]
    j_acc = gltf['accessors'][j_acc_idx]
    w_bv = gltf['bufferViews'][w_acc['bufferView']]
    j_bv = gltf['bufferViews'][j_acc['bufferView']]
    w_offset = w_bv.get('byteOffset', 0) + w_acc.get('byteOffset', 0)
    j_offset = j_bv.get('byteOffset', 0) + j_acc.get('byteOffset', 0)

    new_w_bytes = new_weights.astype(np.float32).tobytes()
    new_j_bytes = new_joints.astype(np.uint16).tobytes()

    # Verify sizes match (same vertex count)
    assert len(new_w_bytes) == V * 4 * 4, f"Weight bytes mismatch: {len(new_w_bytes)} vs {V * 16}"
    assert len(new_j_bytes) == V * 4 * 2, f"Joint bytes mismatch: {len(new_j_bytes)} vs {V * 8}"

    bindata[w_offset:w_offset + len(new_w_bytes)] = new_w_bytes
    bindata[j_offset:j_offset + len(new_j_bytes)] = new_j_bytes

    # Rebuild GLB
    json_bytes = _json.dumps(gltf).encode()
    while len(json_bytes) % 4:
        json_bytes += b' '
    bin_bytes = bytes(bindata)
    while len(bin_bytes) % 4:
        bin_bytes += b'\x00'
    total = 12 + 8 + len(json_bytes) + 8 + len(bin_bytes)
    glb_out = struct.pack('<III', 0x46546C67, 2, total)
    glb_out += struct.pack('<II', len(json_bytes), 0x4E4F534A) + json_bytes
    glb_out += struct.pack('<II', len(bin_bytes), 0x004E4942) + bin_bytes

    output_path = os.path.join(output_dir, "stage4a_cleaned.glb")
    with open(output_path, "wb") as f:
        f.write(glb_out)
    log.info("  Cleaned GLB: %s (%d bytes)", output_path, len(glb_out))
    log.info("  Total fixes: %d arm-torso + %d shoulder + %d armpit + %d wrist + %d cross-body = %d vertices cleaned",
             fixed_arm, fixed_shoulder, fixed_armpit, fixed_wrist, fixed_cross,
             fixed_arm + fixed_shoulder + fixed_armpit + fixed_wrist + fixed_cross)
    return output_path


# === Stage 4b: Weld duplicate vertices at joint boundaries =============

