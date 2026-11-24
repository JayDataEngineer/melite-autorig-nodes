#!/usr/bin/env python3
"""rig_preflight.py — scene judge and planner for the SOMA-77 weight transfer.

STANDARD LIBRARY ONLY. soma_weight_transfer.py imports it inside ComfyUI; the
estate's contract walls import it under a bare python3 (CI carries no numpy).

Contract (the seated-colleague burn, 2026-11-23): the transfer rigs a STANDING
humanoid in rest pose. Everything else is refused with a report naming the
measured reason. A scene is judged as a whole:
  * every node stays reachable from its scene (nothing is orphaned),
  * worn parts (body, head, hands, hair, clothes) are baked to world space and
    skinned; props (pedestals, seats, panels, weapons) stay static,
  * a Z-up source is converted wholesale (every node, every primitive) before
    axis roles are read; no per-mesh "largest spread" guess,
  * the template is aligned ONCE, from the worn cluster's own axes and facing.

Vocabulary: a PART is one node x primitive pair carrying POSITION. WORN parts
follow the skeleton. The CLUSTER is the seed part plus every part whose world
box overlaps it. Extents are metres in the scene's own frame.
"""
from __future__ import annotations

import array
import json
import math
import struct
import sys

FLOAT32 = 5126
MODE_TRIANGLES = 4
UP_AXIS = 1
# The template faces +Z of its own frame: toes extend 0.174 m past the ankle
# line against 0.083 m of heel, the nose 0.072 m past the neck column against
# none behind (somaskel77 skin_standard.npz, measured 2026-11-23).
TEMPLATE_FRONT_SIGN = 1
# bind_vertices extents (1.2393, 1.7605, 0.3561): the template stands in a
# T-ish pose, so its own depth/height 0.20 and lateral/height 0.70 sit mid-band.
DEPTH_OVER_HEIGHT = (0.07, 0.40)
LATERAL_OVER_HEIGHT = (0.20, 1.30)
CLUSTER_OVERLAP = 0.10
SEED_MIN_HEIGHT = 0.5
FOOT_SLAB = 0.035
FOOT_GAP = 0.03
SHIN_SLAB = (0.08, 0.25)
FACING_RATIO = 1.25

IDENTITY = ((1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0), (0.0, 0.0, 0.0, 1.0))


class RigRefusal(Exception):
    """The source cannot be rigged. ``violations`` is the report's CRITICAL list."""

    def __init__(self, violations, stats=None):
        self.violations = list(violations)
        self.stats = dict(stats or {})
        super().__init__("; ".join(self.violations))


# ── container ────────────────────────────────────────────────────────────────

def parse_glb(path):
    """Return (gltf dict, BIN chunk bytes). Refuses a non-GLB or a truncated one."""
    with open(path, "rb") as handle:
        blob = handle.read()
    if len(blob) < 20 or blob[:4] != b"glTF":
        raise RigRefusal(["not a GLB: the file lacks the glTF magic bytes"])
    version, total = struct.unpack_from("<II", blob, 4)
    if version != 2 or total > len(blob):
        raise RigRefusal(["not a GLB 2.0 or the file is truncated"])
    json_len, _json_type = struct.unpack_from("<II", blob, 12)
    gltf = json.loads(blob[20:20 + json_len].decode("utf-8"))
    bin_data = b""
    pos = 20 + json_len
    if len(blob) >= pos + 8:
        bin_len, bin_type = struct.unpack_from("<II", blob, pos)
        if bin_type == 0x004E4942:
            bin_data = blob[pos + 8:pos + 8 + bin_len]
    buffers = gltf.get("buffers") or []
    if buffers and len(bin_data) < int(buffers[0].get("byteLength", 0)):
        raise RigRefusal(["the GLB binary chunk is shorter than its buffer declares"])
    return gltf, bin_data


def pack_glb(gltf, bin_data):
    """Serialise (gltf, BIN) to GLB bytes; JSON is space-padded, BIN zero-padded to 4."""
    doc = json.dumps(gltf, separators=(",", ":"), sort_keys=True).encode("utf-8")
    doc += b" " * (-len(doc) % 4)
    blob = bytearray(bin_data)
    blob += b"\0" * (-len(blob) % 4)
    total = 12 + 8 + len(doc) + 8 + len(blob)
    out = bytearray(b"glTF" + struct.pack("<II", 2, total))
    out += struct.pack("<II", len(doc), 0x4E4F534A) + doc
    out += struct.pack("<II", len(blob), 0x004E4942) + blob
    return bytes(out)


# ── 4x4 affine matrices, row-major nested lists, p' = M @ [p, 1] ────────────

def mat_mul(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(4)) for j in range(4)] for i in range(4)]


def mat_apply(m, x, y, z):
    return (m[0][0] * x + m[0][1] * y + m[0][2] * z + m[0][3],
            m[1][0] * x + m[1][1] * y + m[1][2] * z + m[1][3],
            m[2][0] * x + m[2][1] * y + m[2][2] * z + m[2][3])


def _det3(m):
    return (m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]))


def normal_matrix(m, label):
    """Inverse-transpose of M's upper-left 3x3 (directions under a non-uniform scale)."""
    r = [row[:3] for row in m[:3]]
    det = _det3(r)
    if abs(det) < 1e-12:
        raise RigRefusal(["%s carries a degenerate transform (zero scale on an axis)" % label])
    c00, c01, c02 = r[1][1] * r[2][2] - r[1][2] * r[2][1], r[1][2] * r[2][0] - r[1][0] * r[2][2], r[1][0] * r[2][1] - r[1][1] * r[2][0]
    return [[c00 / det, c01 / det, c02 / det],
            [(r[0][2] * r[2][1] - r[0][1] * r[2][2]) / det, (r[0][0] * r[2][2] - r[0][2] * r[2][0]) / det, (r[0][1] * r[1][2] - r[0][2] * r[1][1]) / det],
            [(r[0][1] * r[1][2] - r[0][2] * r[1][1]) / det, (r[0][2] * r[1][0] - r[0][0] * r[1][2]) / det, (r[0][0] * r[1][1] - r[0][1] * r[1][0]) / det]]


def node_matrix(node):
    """A node's local 4x4 from either ``matrix`` (column-major) or TRS."""
    if "matrix" in node:
        m = [float(v) for v in node["matrix"]]
        return [[m[0], m[4], m[8], m[12]], [m[1], m[5], m[9], m[13]],
                [m[2], m[6], m[10], m[14]], [m[3], m[7], m[11], m[15]]]
    t = [float(v) for v in node.get("translation", (0.0, 0.0, 0.0))]
    x, y, z, w = [float(v) for v in node.get("rotation", (0.0, 0.0, 0.0, 1.0))]
    s = [float(v) for v in node.get("scale", (1.0, 1.0, 1.0))]
    norm = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return [[(1 - 2 * (y * y + z * z)) * s[0], (2 * (x * y - z * w)) * s[1], (2 * (x * z + y * w)) * s[2], t[0]],
            [(2 * (x * y + z * w)) * s[0], (1 - 2 * (x * x + z * z)) * s[1], (2 * (y * z - x * w)) * s[2], t[1]],
            [(2 * (x * z - y * w)) * s[0], (2 * (y * z + x * w)) * s[1], (1 - 2 * (x * x + y * y)) * s[2], t[2]],
            [0.0, 0.0, 0.0, 1.0]]


def is_identity(m):
    return all(abs(m[i][j] - IDENTITY[i][j]) <= 1e-9 for i in range(4) for j in range(4))


def scene_sets(gltf):
    """Per scene, the set of node indices reachable from its roots (cycle-safe)."""
    nodes = gltf.get("nodes") or []
    out = []
    for scene in gltf.get("scenes") or []:
        seen = set()
        stack = list(scene.get("nodes") or [])
        while stack:
            ni = stack.pop()
            if ni in seen:
                continue
            seen.add(ni)
            stack.extend(nodes[ni].get("children") or [])
        out.append(seen)
    return out


def world_matrices(gltf):
    """World 4x4 of every node reachable from any scene, keyed by node index."""
    nodes = gltf.get("nodes") or []
    out = {}
    stack = []
    for scene in gltf.get("scenes") or []:
        for root in scene.get("nodes") or []:
            stack.append((root, IDENTITY))
    seen = set()
    while stack:
        ni, parent = stack.pop()
        if ni in seen:
            continue
        seen.add(ni)
        world = mat_mul(parent, node_matrix(nodes[ni]))
        out[ni] = world
        for child in nodes[ni].get("children") or []:
            stack.append((child, world))
    return out


# ── buffer views ─────────────────────────────────────────────────────────────

def accessor_view(gltf, prim, attr, where):
    """(offset, stride, count, components) of a float32 attribute, or None when absent/dataless.
    xyz are always the leading components; TANGENT's w (handedness) follows them."""
    idx = (prim.get("attributes") or {}).get(attr)
    if idx is None:
        return None
    acc = gltf["accessors"][idx]
    if "bufferView" not in acc:
        return None
    if acc.get("sparse"):
        raise RigRefusal(["%s: %s uses a sparse accessor (not supported)" % (where, attr)])
    view = gltf["bufferViews"][acc["bufferView"]]
    if view.get("buffer", 0) != 0 or "uri" in gltf.get("buffers", [{}])[0]:
        raise RigRefusal(["%s: %s is not stored inside the GLB" % (where, attr)])
    components = {"VEC3": 3, "VEC4": 4}.get(acc.get("type"))
    if acc.get("componentType") != FLOAT32 or not components:
        raise RigRefusal(["%s: %s must be float32 VEC3 or VEC4" % (where, attr)])
    offset = int(view.get("byteOffset", 0)) + int(acc.get("byteOffset", 0))
    stride = int(view.get("byteStride") or 0) or 4 * components
    return offset, stride, int(acc["count"]), components


def read_vec3(bin_data, view):
    """xyz values of a view as a flat array('f') of length 3*count."""
    offset, stride, count, components = view
    if offset + (count - 1) * stride + 12 > len(bin_data):
        raise RigRefusal(["the buffer ends inside a float attribute"])
    if stride == 12:
        out = array.array("f")
        out.frombytes(bytes(bin_data[offset:offset + count * 12]))
    else:
        out = array.array("f", [0.0]) * (3 * count)
        for i in range(count):
            start = offset + i * stride
            out[3 * i:3 * i + 3] = array.array("f", bytes(bin_data[start:start + 12]))
    if sys.byteorder == "big":
        out.byteswap()
    return out


def write_vec3(bin_data, view, flat):
    offset, stride, count, components = view
    data = flat.tobytes() if sys.byteorder == "little" else array.array("f", flat).byteswap().tobytes()
    if stride == 12:
        bin_data[offset:offset + count * 12] = data
        return
    for i in range(count):
        start = offset + i * stride
        bin_data[start:start + 12] = data[i * 12:(i + 1) * 12]


def flip_tangent_w(bin_data, view):
    """Negate every tangent's handedness component (a mirrored node flips bitangents)."""
    offset, stride, count, _components = view
    for i in range(count):
        at = offset + i * stride + 12
        bin_data[at:at + 4] = struct.pack("<f", -struct.unpack_from("<f", bin_data, at)[0])


def transform_points(flat, m):
    for i in range(0, len(flat), 3):
        x, y, z = flat[i], flat[i + 1], flat[i + 2]
        flat[i] = m[0][0] * x + m[0][1] * y + m[0][2] * z + m[0][3]
        flat[i + 1] = m[1][0] * x + m[1][1] * y + m[1][2] * z + m[1][3]
        flat[i + 2] = m[2][0] * x + m[2][1] * y + m[2][2] * z + m[2][3]


def transform_directions(flat, n):
    """Unit directions through the inverse-transpose 3x3 ``n`` (xyz stride)."""
    for i in range(0, len(flat), 3):
        x, y, z = flat[i], flat[i + 1], flat[i + 2]
        nx, ny, nz = n[0][0] * x + n[0][1] * y + n[0][2] * z, n[1][0] * x + n[1][1] * y + n[1][2] * z, n[2][0] * x + n[2][1] * y + n[2][2] * z
        norm = math.sqrt(nx * nx + ny * ny + nz * nz)
        if norm > 1e-12:
            flat[i], flat[i + 1], flat[i + 2] = nx / norm, ny / norm, nz / norm


def extents(flat):
    lo = [math.inf] * 3
    hi = [-math.inf] * 3
    for i in range(0, len(flat), 3):
        for k in range(3):
            v = flat[i + k]
            if v < lo[k]:
                lo[k] = v
            if v > hi[k]:
                hi[k] = v
    return lo, hi


def _set_accessor_bounds(gltf, prim, attr, lo, hi):
    acc = gltf["accessors"][prim["attributes"][attr]]
    acc["min"], acc["max"] = [float(v) for v in lo], [float(v) for v in hi]


# ── parts and cluster ────────────────────────────────────────────────────────

def parts_of_scene(gltf, bin_data):
    """Every triangle primitive on a scene node, with world matrix and world box."""
    if gltf.get("skins"):
        raise RigRefusal(["the GLB already carries a rig (%d skin(s)); the auto lane rigs a static body — use mode copy to keep the donor rig" % len(gltf["skins"])])
    if any("KHR_draco_mesh_compression" in (prim.get("extensions") or {})
           for mesh in gltf.get("meshes") or [] for prim in mesh.get("primitives") or []):
        raise RigRefusal(["a primitive is Draco-compressed; decode it before rigging"])
    world = world_matrices(gltf)
    instances = {}
    for ni, node in enumerate(gltf.get("nodes") or []):
        if "mesh" in node:
            instances.setdefault(node["mesh"], []).append(ni)
    parts = []
    for ni in sorted(world):
        node = gltf["nodes"][ni]
        if "mesh" not in node:
            continue
        label = "node '%s'" % (node.get("name") or ni)
        for pi, prim in enumerate(gltf["meshes"][node["mesh"]].get("primitives") or []):
            if prim.get("mode", MODE_TRIANGLES) != MODE_TRIANGLES:
                continue
            view = accessor_view(gltf, prim, "POSITION", "%s primitive %d" % (label, pi))
            if view is None:
                continue
            acc = gltf["accessors"][prim["attributes"]["POSITION"]]
            lo, hi = acc.get("min"), acc.get("max")
            if lo is None or hi is None:
                lo, hi = extents(read_vec3(bin_data, view))
            wlo, whi = [math.inf] * 3, [-math.inf] * 3
            for cx in (lo[0], hi[0]):
                for cy in (lo[1], hi[1]):
                    for cz in (lo[2], hi[2]):
                        wx, wy, wz = mat_apply(world[ni], cx, cy, cz)
                        for k, v in enumerate((wx, wy, wz)):
                            wlo[k] = min(wlo[k], v)
                            whi[k] = max(whi[k], v)
            parts.append({"node": ni, "mesh": node["mesh"], "prim": pi, "view": view, "count": view[2],
                          "world": world[ni], "box": (wlo, whi), "label": "%s primitive %d" % (label, pi),
                          "instanced": len(instances[node["mesh"]]) > 1})
    return parts


def part_points(part, bin_data):
    """World-space positions of a part as a flat array('f')."""
    flat = read_vec3(bin_data, part["view"])
    transform_points(flat, part["world"])
    return flat


# Role hypotheses in preference order: the glTF convention first (+Y up, a humanoid modelled in a
# DCC faces +Z or -Z, so Z is depth and X lateral), then a figure turned sideways, then the two
# Z-up frames a mis-exported scene can arrive in.
ROLE_HYPOTHESES = (
    (UP_AXIS, 2), (UP_AXIS, 0),
    (2, 1), (2, 0),
    (0, 1), (0, 2),
)


def role_candidates(size):
    """[(roles, ratios)] for every hypothesis whose depth/height and lateral/height both sit in
    band, in preference order; empty when no framing reads as a standing humanoid."""
    out = []
    for up, depth in ROLE_HYPOTHESES:
        lateral = 3 - up - depth
        height = max(size[up], 1e-9)
        ratios = (size[depth] / height, size[lateral] / height)
        if (DEPTH_OVER_HEIGHT[0] <= ratios[0] <= DEPTH_OVER_HEIGHT[1]
                and LATERAL_OVER_HEIGHT[0] <= ratios[1] <= LATERAL_OVER_HEIGHT[1]):
            out.append(({"lateral": lateral, "up": up, "depth": depth}, ratios))
    return out


def roles_for_size(size):
    """(roles, ratios) to measure a box with: the first in-band hypothesis, else the glTF
    convention with its own readings (so a refusal names the honest numbers)."""
    candidates = role_candidates(size)
    if candidates:
        return candidates[0]
    up, depth = ROLE_HYPOTHESES[0]
    lateral = 3 - up - depth
    height = max(size[up], 1e-9)
    return {"lateral": lateral, "up": up, "depth": depth}, (size[depth] / height, size[lateral] / height)


def read_body(union, size):
    """Roles, facing and the standing verdict for one point cloud (world units), the single body
    reading every caller shares. Band-fit hypotheses are ranked by readable facing (feet pointing
    along the hypothesis' depth axis); ties and non-fit boxes fall back to the glTF convention."""
    roles, ratios = roles_for_size(size)
    sign, note = facing_sign(union, roles)
    if len(role_candidates(size)) > 1:
        for cand_roles, cand_ratios in role_candidates(size):
            cand_sign, cand_note = facing_sign(union, cand_roles)
            if cand_sign is not None:
                roles, ratios, sign, note = cand_roles, cand_ratios, cand_sign, cand_note
                break
    reach, reach_note = leg_reach(union, roles)
    stats = stance_readings(union, roles)
    stats["reach"] = reach
    violations = standing_violations(size, ratios, "the body")
    if not "solid column" in reach_note:
        violations.append(reach_note + ": not a standing humanoid with readable feet")
    elif reach < LEG_REACH_MIN:
        violations.append("%s (a standing body stays solid above each foot to %.0f%%): the figure is "
                          "seated, kneeling, or merged with furniture" % (reach_note, LEG_REACH_MIN * 100))
    return {"roles": roles, "ratios": ratios, "facing": sign, "facing_note": note,
            "reach": reach, "reach_note": reach_note, "stats": stats, "violations": violations}


def standing_violations(size, ratios, label):
    """CRITICAL lines when a box cannot be a standing humanoid; empty when it can."""
    out = []
    if not DEPTH_OVER_HEIGHT[0] <= ratios[0] <= DEPTH_OVER_HEIGHT[1]:
        out.append("%s: depth/height %.2f is outside %.2f-%.2f" % (label, ratios[0], DEPTH_OVER_HEIGHT[0], DEPTH_OVER_HEIGHT[1]))
    if not LATERAL_OVER_HEIGHT[0] <= ratios[1] <= LATERAL_OVER_HEIGHT[1]:
        out.append("%s: lateral/height %.2f is outside %.2f-%.2f" % (label, ratios[1], LATERAL_OVER_HEIGHT[0], LATERAL_OVER_HEIGHT[1]))
    if out:
        out.append("%s measures %.3f x %.3f x %.3f m (X x Y x Z): not a standing humanoid in rest pose, feet to crown"
                   % (label, size[0], size[1], size[2]))
    return out


def facing_sign(flat, roles):
    """(+1|-1|None, note): which way along the depth axis the feet point, from the ankle line."""
    up, depth = roles["up"], roles["depth"]
    ys = flat[up::3]
    lo, hi = min(ys), max(ys)
    height = hi - lo
    shins = [flat[i + depth] for i in range(0, len(flat), 3) if lo + SHIN_SLAB[0] * height < flat[i + up] < lo + SHIN_SLAB[1] * height]
    feet = [flat[i + depth] for i in range(0, len(flat), 3) if flat[i + up] < lo + FOOT_SLAB * height]
    if len(shins) < 8 or len(feet) < 8:
        return None, "no foot slab to read facing from; assumed the glTF front (+depth)"
    ankle = sum(shins) / len(shins)
    toward_min = ankle - min(feet)
    toward_max = max(feet) - ankle
    if min(toward_min, toward_max) <= 0.0 or max(toward_min, toward_max) < FACING_RATIO * min(toward_min, toward_max):
        return None, "facing ambiguous (foot extends %.3f m one way, %.3f m the other); assumed the glTF front (+depth)" % (toward_min, toward_max)
    return (-1 if toward_min > toward_max else 1), "feet point %s along the depth axis" % ("-" if toward_min > toward_max else "+")


def alignment_permutation(roles, front_sign):
    """3x3 rotation P with world = P @ template; the template front (+Z) lands on the source front."""
    perm = [[0.0] * 3 for _ in range(3)]
    perm[roles["lateral"]][0] = 1.0
    perm[roles["up"]][1] = 1.0
    perm[roles["depth"]][2] = float(front_sign)
    if _det3(perm) < 0.0:
        perm[roles["lateral"]][0] = -1.0
    return perm


SLICES = 20
COLUMN_RADIUS = 0.12   # horizontal reach around a foot's axis, in units of height
LEG_REACH_MIN = 0.60   # a standing body stays solid above each foot at least this high
FOOT_MAX_WIDTH = 0.40  # a bottom cluster wider than this is a base, not feet


def _pct(sorted_values, q):
    return sorted_values[min(len(sorted_values) - 1, int(q * len(sorted_values)))]


def stance_readings(flat, roles):
    """The 5-95% depth extent of the slice band between knees and hips, in units of height (a report
    statistic: a seated body carries its thighs there, a standing one only legs and hanging hands)."""
    up, depth = roles["up"], roles["depth"]
    ys = flat[up::3]
    lo, hi = min(ys), max(ys)
    height = hi - lo
    if height <= 0.0:
        return {"thigh": math.inf}
    band = []
    for i in range(0, len(flat), 3):
        s = int((flat[i + up] - lo) / height * SLICES)
        if 6 <= s < 10:
            band.append(flat[i + depth])
    band.sort()
    return {"thigh": (_pct(band, 0.95) - _pct(band, 0.05)) / height if band else math.inf}


def _grid_clusters(points, cell_size):
    """Clusters of (a, b) points: cells of side ``cell_size`` joined with their 8 neighbours.
    Returns per cluster {"count", "min", "max", "mean"} over (a, b)."""
    grid = {}
    for a, b in points:
        key = (int(a // cell_size), int(b // cell_size))
        cell = grid.get(key)
        if cell is None:
            grid[key] = [1, a, b, a, b, a, b]
        else:
            cell[0] += 1
            cell[1] += a
            cell[2] += b
            cell[3] = min(cell[3], a)
            cell[4] = min(cell[4], b)
            cell[5] = max(cell[5], a)
            cell[6] = max(cell[6], b)
    keys = list(grid)
    index = {key: i for i, key in enumerate(keys)}
    parent = list(range(len(keys)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for key, i in index.items():
        for da in (-1, 0, 1):
            for db in (-1, 0, 1):
                j = index.get((key[0] + da, key[1] + db))
                if j is not None and j != i:
                    parent[find(i)] = find(j)
    clusters = {}
    for key, i in index.items():
        cell = grid[key]
        out = clusters.setdefault(find(i), [0, math.inf, math.inf, -math.inf, -math.inf, 0.0, 0.0])
        out[0] += cell[0]
        out[1] = min(out[1], cell[3])
        out[2] = min(out[2], cell[4])
        out[3] = max(out[3], cell[5])
        out[4] = max(out[4], cell[6])
        out[5] += cell[1]
        out[6] += cell[2]
    return [{"count": c[0], "min": (c[1], c[2]), "max": (c[3], c[4]), "mean": (c[5] / c[0], c[6] / c[0])}
            for c in clusters.values()]


def leg_reach(flat, roles):
    """(reach, note): the lowest height, in units of the body's height, at which the solid column
    directly above a foot ends (the column's axis stays on the foot). A standing body keeps that
    column through thighs and pelvis to the shoulders; a seated one loses it above the knees, where
    the thigh turns back horizontally."""
    up, lateral, depth = roles["up"], roles["lateral"], roles["depth"]
    ys = flat[up::3]
    lo, hi = min(ys), max(ys)
    height = hi - lo
    if height <= 0.0:
        return 0.0, "the body has no height"
    slabs = [[] for _ in range(SLICES)]
    for i in range(0, len(flat), 3):
        slabs[min(SLICES - 1, int((flat[i + up] - lo) / height * SLICES))].append((flat[i + lateral], flat[i + depth]))
    feet = []
    for s in range(0, SLICES // 3):
        clusters = [c for c in _grid_clusters(slabs[s], FOOT_GAP * height) if c["count"] >= 4]
        if not clusters:
            continue
        clusters.sort(key=lambda c: (c["max"][0] - c["min"][0]) * (c["max"][1] - c["min"][1]), reverse=True)
        first = clusters[0]
        if first["max"][0] - first["min"][0] > FOOT_MAX_WIDTH * height:
            continue
        if len(clusters) > 1 and clusters[1]["count"] >= 0.25 * first["count"]:
            feet = [(c["mean"], s) for c in clusters[:2]]
        else:
            half = sorted(slabs[s])
            cut = len(half) // 2
            feet = [((sum(p[0] for p in half[side * cut:(side + 1) * cut]) / max(cut, 1),
                      sum(p[1] for p in half[side * cut:(side + 1) * cut]) / max(cut, 1)), s) for side in (0, 1)]
        break
    if not feet:
        return 0.0, "no feet found in the bottom third of the body"
    radius = COLUMN_RADIUS * height
    reach = math.inf
    for axis, start in feet:
        top = (start + 1) / SLICES
        gap = 0
        for s in range(start + 1, SLICES):
            near = [1 for a, b in slabs[s] if (a - axis[0]) ** 2 + (b - axis[1]) ** 2 < radius * radius]
            if len(near) >= 3:
                top = (s + 1) / SLICES
                gap = 0
            else:
                gap += 1
                if gap >= 2:
                    break
        reach = min(reach, top)
    return reach, "the solid column above the feet ends at %.0f%% of the body's height" % (reach * 100)


def _overlap_ratio(part, seed):
    (plo, phi), (slo, shi) = part["box"], seed["box"]
    for k in range(3):
        span = shi[k] - slo[k]
        cut = min(phi[k], shi[k]) - max(plo[k], slo[k])
        if span <= 0.0 or cut < CLUSTER_OVERLAP * span:
            return False
    return True


def cluster_around(parts, seed):
    return [part for part in parts if part is seed or _overlap_ratio(part, seed)]


def _box_size(part):
    lo, hi = part["box"]
    return [hi[k] - lo[k] for k in range(3)]


def seed_part(parts):
    """The part the body is read from: the largest humanoid-proportioned part, else the largest."""
    if not parts:
        raise RigRefusal(["the scene carries no triangle primitive with positions"])
    tallest = max(max(_box_size(part)) for part in parts)
    proportioned = [
        part for part in parts
        if _box_size(part)[roles_for_size(_box_size(part))[0]["up"]] >= SEED_MIN_HEIGHT * tallest
        and not standing_violations(_box_size(part), roles_for_size(_box_size(part))[1], "")
    ]
    return max(proportioned or parts, key=lambda part: part["count"])


def plan_for(parts, bin_data):
    """Roles, gate verdict and facing of the union of ``parts``, measured on their world points."""
    union = array.array("f")
    lo = [math.inf] * 3
    hi = [-math.inf] * 3
    for part in parts:
        if part["count"]:
            flat = part_points(part, bin_data)
            plo, phi = extents(flat)
            union.extend(flat)
            for k in range(3):
                lo[k] = min(lo[k], plo[k])
                hi[k] = max(hi[k], phi[k])
    if not union:
        raise RigRefusal(["the worn parts carry no vertices"])
    size = [hi[k] - lo[k] for k in range(3)]
    body = read_body(union, size)
    body.update({"parts": parts, "min": lo, "max": hi, "size": size})
    return body


def judge(path):
    """Read-only verdict on a GLB: what the transfer would see, do, and refuse."""
    gltf, bin_data = parse_glb(path)
    parts = parts_of_scene(gltf, bin_data)
    seed = seed_part(parts)
    cluster = cluster_around(parts, seed)
    plan = plan_for(cluster, bin_data)
    return {
        "parts": [{"node": p["node"], "label": p["label"], "vertices": p["count"], "instanced": p["instanced"],
                   "box": [p["box"][0], p["box"][1]]} for p in parts],
        "seed": seed["label"], "cluster": [p["label"] for p in cluster],
        "size": plan["size"], "roles": plan["roles"], "ratios": plan["ratios"],
        "facing": plan["facing"], "facing_note": plan["facing_note"], "violations": plan["violations"],
        "reach": plan["reach"], "stats": plan["stats"],
    }


# ── scene edits ──────────────────────────────────────────────────────────────

def _rotate_attribute(gltf, bin_data, prim, attr, perm, where):
    view = accessor_view(gltf, prim, attr, where)
    if view is None:
        return
    flat = read_vec3(bin_data, view)
    if attr == "POSITION":
        # Positions take the pure rotation: transform_directions renormalizes
        # every triple to unit length, which is right for NORMAL/TANGENT and
        # would crush the geometry into a unit sphere.
        rot4 = [list(perm[0]) + [0.0], list(perm[1]) + [0.0], list(perm[2]) + [0.0], [0.0, 0.0, 0.0, 1.0]]
        transform_points(flat, rot4)
    else:
        transform_directions(flat, perm)
    write_vec3(bin_data, view, flat)
    if attr == "POSITION":
        _set_accessor_bounds(gltf, prim, attr, *extents(flat))


def ensure_yup(gltf, bin_data, roles):
    """Convert a Z-up (or X-up) scene to Y-up wholesale: every node L -> R L R^T, every POSITION, NORMAL,
    TANGENT and morph POSITION rotated by R. Returns (gltf, bytearray, converted)."""
    if roles["up"] == UP_AXIS:
        return gltf, bin_data, False
    perm = [[0.0] * 3 for _ in range(3)]
    perm[roles["lateral"]][0] = 1.0
    perm[roles["up"]][1] = 1.0
    perm[roles["depth"]][2] = 1.0
    if _det3(perm) < 0.0:
        perm[roles["depth"]][2] = -1.0
    rot = [list(perm[0]) + [0.0], list(perm[1]) + [0.0], list(perm[2]) + [0.0], [0.0, 0.0, 0.0, 1.0]]
    rot_t = [[rot[j][i] for j in range(4)] for i in range(4)]
    bin_data = bytearray(bin_data)
    for node in gltf.get("nodes") or []:
        new = mat_mul(mat_mul(rot, node_matrix(node)), rot_t)
        for key in ("translation", "rotation", "scale"):
            node.pop(key, None)
        node["matrix"] = [new[0][0], new[1][0], new[2][0], 0.0, new[0][1], new[1][1], new[2][1], 0.0,
                          new[0][2], new[1][2], new[2][2], 0.0, new[0][3], new[1][3], new[2][3], 1.0]
    for mi, mesh in enumerate(gltf.get("meshes") or []):
        for pi, prim in enumerate(mesh.get("primitives") or []):
            where = "mesh %d primitive %d" % (mi, pi)
            for attr in ("POSITION", "NORMAL", "TANGENT"):
                _rotate_attribute(gltf, bin_data, prim, attr, perm, where)
            for target in prim.get("targets") or []:
                _rotate_attribute(gltf, bin_data, {"attributes": target}, "POSITION", perm, where + " morph target")
    return gltf, bin_data, True


def bake_to_world(gltf, bin_data, nodes):
    """Bake the world transform of ``nodes`` (mesh nodes) into their vertices and re-seat each as an identity
    scene root. A node with children keeps its transform for them and hands its mesh to a fresh root.
    Returns (gltf, bytearray, {old node index: skin node index})."""
    bin_data = bytearray(bin_data)
    world = world_matrices(gltf)
    reach = scene_sets(gltf)
    scenes = gltf.setdefault("scenes", [{"nodes": []}])
    for scene in scenes:
        scene.setdefault("nodes", [])
    seats = {}
    for ni in nodes:
        node = gltf["nodes"][ni]
        mesh_index = node["mesh"]
        label = "node '%s'" % (node.get("name") or ni)
        matrix = world[ni]
        if not is_identity(matrix):
            normal = normal_matrix(matrix, label)
            mirrored = _det3([row[:3] for row in matrix[:3]]) < 0.0
            for prim in gltf["meshes"][mesh_index].get("primitives") or []:
                view = accessor_view(gltf, prim, "POSITION", label)
                if view is None:
                    continue
                flat = read_vec3(bin_data, view)
                transform_points(flat, matrix)
                write_vec3(bin_data, view, flat)
                _set_accessor_bounds(gltf, prim, "POSITION", *extents(flat))
                for attr in ("NORMAL", "TANGENT"):
                    aview = accessor_view(gltf, prim, attr, label)
                    if aview is None:
                        continue
                    dirs = read_vec3(bin_data, aview)
                    transform_directions(dirs, normal)
                    write_vec3(bin_data, aview, dirs)
                    if attr == "TANGENT" and mirrored and aview[3] == 4:
                        flip_tangent_w(bin_data, aview)
        if node.get("children"):
            # The node stays as the group its children hang from; its mesh moves to a root-level seat.
            scenes_reached = [si for si, seen in enumerate(reach) if ni in seen]
            gltf["nodes"].append({"name": (node.get("name") or "mesh") + " (skinned)", "mesh": mesh_index})
            seat = len(gltf["nodes"]) - 1
            del node["mesh"]
        else:
            for other in gltf["nodes"]:
                children = other.get("children") or []
                if ni in children:
                    children.remove(ni)
            for key in ("matrix",):
                node.pop(key, None)
            node["translation"], node["rotation"], node["scale"] = [0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0]
            seat = ni
            scenes_reached = [si for si, seen in enumerate(reach) if ni in seen]
        for si in scenes_reached:
            if seat not in scenes[si]["nodes"]:
                scenes[si]["nodes"].append(seat)
        if not scenes_reached:
            raise RigRefusal(["%s is not reachable from any scene" % label])
        seats[ni] = seat
    return gltf, bin_data, seats


def _jsonable(value):
    """Non-finite floats (a degenerate box divides extents) become null —
    the CLI's stdout is JSON a consumer must be able to parse."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _main(argv):
    if len(argv) != 2:
        sys.stderr.write("usage: rig_preflight.py <model.glb>\n")
        return 2
    try:
        verdict = judge(argv[1])
    except RigRefusal as err:
        verdict = {"violations": err.violations}
    sys.stdout.write(json.dumps(_jsonable(verdict), indent=2, sort_keys=True) + "\n")
    return 1 if verdict.get("violations") else 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv))
