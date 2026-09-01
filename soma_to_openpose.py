"""
Convert SOMA-77 skeleton to standard OpenPose COCO-18 image for ControlNet.

Uses the EXACT color palette and limb connections from controlnet_aux
(the library ComfyUI/Automatic1111 use under the hood). This is critical:
ControlNet OpenPose models were trained on these specific colors.

Pipeline: SOMA skeleton → 2D projection → OpenPose PNG → ControlNet → Character

The resulting character will have proportions matching the skeleton,
making weight transfer and animation retargeting much cleaner.
"""
import math
import numpy as np

# ── Check for cv2 (preferred for exact controlnet_aux match) ──────
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

from PIL import Image, ImageDraw

# ═══════════════════════════════════════════════════════════════════
# SOMA-77 → COCO-18 keypoint mapping
# ═══════════════════════════════════════════════════════════════════
# NOTE (suggestion #5, 2026-07-31): this is a VENDORED COPY of the
# canonical mapping in melite-poser-nodes/library/skeleton.py
# (COCO_TO_SOMA_IDX, inverted). The two cannot share an import because
# ComfyUI nodepacks are self-contained + the dir names have hyphens. A
# drift-guard test (tests/unit/test_soma_coco_mapping_consolidated.py)
# asserts the two agree at PR time — if you update the canonical, update
# this copy too (or the test fails).
#
# The canonical source is what the main render pipeline
# (PoserRenderOpenPose) uses. This vendored copy is what melite-autorig-nodes
# uses for its standalone OpenPose rendering. They MUST agree so a
# character rendered through either path comes out in the same pose.
#
# COCO-18 indices:
#   0:Nose  1:Neck  2:RShoulder  3:RElbow  4:RWrist
#   5:LShoulder  6:LElbow  7:LWrist
#   8:RHip  9:RKnee  10:RAnkle
#   11:LHip  12:LKnee  13:LAnkle
#   14:REye  15:LEye  16:REar  17:LEar
SOMA_TO_COCO18 = {
    6: 0,    # Head → Nose
    4: 1,    # Neck → Neck
    40: 2,   # RightArm → RShoulder (arm joint, NOT clavicle J39)
    41: 3,   # RightForeArm → RElbow
    42: 4,   # RightHand → RWrist
    12: 5,   # LeftArm → LShoulder (arm joint, NOT clavicle J11)
    13: 6,   # LeftForeArm → LElbow
    14: 7,   # LeftHand → LWrist
    72: 8,   # RightLeg → RHip
    73: 9,   # RightShin → RKnee
    74: 10,  # RightFoot → RAnkle
    67: 11,  # LeftLeg → LHip
    68: 12,  # LeftShin → LKnee
    69: 13,  # LeftFoot → LAnkle
    10: 14,  # RightEye → REye
    9: 15,   # LeftEye → LEye
    # No ears (16, 17) in SOMA — ControlNet handles missing keypoints
}

# ═══════════════════════════════════════════════════════════════════
# Standard controlnet_aux color palette (EXACT match)
# ═══════════════════════════════════════════════════════════════════
# These colors form a rainbow gradient that ControlNet uses to
# distinguish left/right, front/back, and limb identification.
COLORS = [
    [255, 0, 0],       # 0: red
    [255, 85, 0],      # 1: red-orange
    [255, 170, 0],     # 2: orange
    [255, 255, 0],     # 3: yellow
    [170, 255, 0],     # 4: yellow-green
    [85, 255, 0],      # 5: green
    [0, 255, 0],       # 6: bright green
    [0, 255, 85],      # 7: teal
    [0, 255, 170],     # 8: cyan
    [0, 255, 255],     # 9: bright cyan
    [0, 170, 255],     # 10: light blue
    [0, 85, 255],      # 11: blue
    [0, 0, 255],       # 12: bright blue
    [85, 0, 255],      # 13: purple
    [170, 0, 255],     # 14: magenta
    [255, 0, 255],     # 15: bright magenta
    [255, 0, 170],     # 16: pink
    [255, 0, 85],      # 17: bright pink
]

# ═══════════════════════════════════════════════════════════════════
# Standard COCO-18 limb connections (0-based indices)
# ═══════════════════════════════════════════════════════════════════
# Same as controlnet_aux limbSeq, converted to 0-based.
# Each limb gets COLORS[limb_index] — this is the critical part.
LIMB_SEQ = [
    (1, 2),    # 0:  Neck → RShoulder      [255,0,0]     red
    (1, 5),    # 1:  Neck → LShoulder      [255,85,0]    red-orange
    (2, 3),    # 2:  RShoulder → RElbow    [255,170,0]   orange
    (3, 4),    # 3:  RElbow → RWrist       [255,255,0]   yellow
    (5, 6),    # 4:  LShoulder → LElbow    [170,255,0]   yellow-green
    (6, 7),    # 5:  LElbow → LWrist       [85,255,0]    green
    (1, 8),    # 6:  Neck → RHip           [0,255,0]     bright green
    (8, 9),    # 7:  RHip → RKnee          [0,255,85]    teal
    (9, 10),   # 8:  RKnee → RAnkle        [0,255,170]   cyan
    (1, 11),   # 9:  Neck → LHip           [0,255,255]   bright cyan
    (11, 12),  # 10: LHip → LKnee          [0,170,255]   light blue
    (12, 13),  # 11: LKnee → LAnkle        [0,85,255]    blue
    (1, 0),    # 12: Neck → Nose           [0,0,255]     bright blue
    (0, 14),   # 13: Nose → REye           [85,0,255]    purple
    # (14, 16), # 14: REye → REar          [170,0,255]   magenta (no ear)
    (0, 15),   # 15: Nose → LEye           [255,0,255]   bright magenta
    # (15, 17), # 16: LEye → LEar          [255,0,170]   pink (no ear)
]

# Active limbs (skip ear connections since SOMA has no ears)
ACTIVE_LIMBS = [
    (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7),
    (1, 8), (8, 9), (9, 10), (1, 11), (11, 12), (12, 13),
    (1, 0), (0, 14), (0, 15),
]


def soma_to_openpose_image(
    joint_positions_3d,
    output_path,
    img_size=768,
    bg_color=(0, 0, 0),
):
    """Convert SOMA-77 3D joint positions to a 2D OpenPose COCO-18 image.

    Uses the exact controlnet_aux color palette and drawing style so
    ControlNet models (IllustriousXL_openpose, etc.) interpret it correctly.

    Args:
        joint_positions_3d: (77, 3) array of joint positions
        output_path: where to save the PNG
        img_size: output image size (square)
    """
    joints_3d = np.array(joint_positions_3d)

    # Project to 2D: front view (X = horizontal, Y = vertical)
    # SOMA: X=left(+), Y=up(+), Z=forward(+)
    # Image: X→right (flip SOMA X), Y→down (flip SOMA Y)
    x_2d = -joints_3d[:, 0]  # SOMA +X (left) → image right
    y_2d = -joints_3d[:, 1]  # SOMA +Y (up) → image down

    # Normalize to image coordinates with padding
    x_min, x_max = x_2d.min(), x_2d.max()
    y_min, y_max = y_2d.min(), y_2d.max()
    max_range = max(x_max - x_min, y_max - y_min, 0.01)

    scale = (img_size * 0.92) / max_range  # fill 92% of canvas — minimal padding
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2

    x_px = (x_2d - x_center) * scale + img_size / 2
    y_px = (y_2d - y_center) * scale + img_size / 2

    # Build COCO-18 keypoints dict
    keypoints = {}  # coco18_idx → (x, y)
    for soma_idx, coco_idx in SOMA_TO_COCO18.items():
        if soma_idx < len(joints_3d):
            keypoints[coco_idx] = (float(x_px[soma_idx]), float(y_px[soma_idx]))

    if HAS_CV2:
        _draw_with_cv2(keypoints, output_path, img_size)
    else:
        _draw_with_pil(keypoints, output_path, img_size)

    print(f"OpenPose image saved: {output_path} ({img_size}x{img_size})")
    print("Format: COCO-18 (controlnet_aux compatible)")
    print(f"Keypoints: {len(keypoints)}/18")

    # Print keypoint positions for verification
    coco_names = {
        0: "Nose", 1: "Neck", 2: "RShoulder", 3: "RElbow", 4: "RWrist",
        5: "LShoulder", 6: "LElbow", 7: "LWrist",
        8: "RHip", 9: "RKnee", 10: "RAnkle",
        11: "LHip", 12: "LKnee", 13: "LAnkle",
        14: "REye", 15: "LEye",
    }
    for idx in sorted(keypoints.keys()):
        name = coco_names.get(idx, f"KP{idx}")
        color = COLORS[idx]
        print(f"  [{idx:2d}] {name:12s}: ({int(keypoints[idx][0])}, {int(keypoints[idx][1])}) "
              f"color=RGB({color[0]},{color[1]},{color[2]})")


def _draw_with_cv2(keypoints, output_path, img_size):
    """Draw using cv2 — exact match to controlnet_aux style."""
    canvas = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    stickwidth = 4

    # Draw limbs (as filled ellipses like controlnet_aux)
    for limb_idx, (k1, k2) in enumerate(ACTIVE_LIMBS):
        if k1 not in keypoints or k2 not in keypoints:
            continue

        color = COLORS[limb_idx]
        p1 = keypoints[k1]
        p2 = keypoints[k2]

        # controlnet_aux uses ellipse2Poly for smooth thick sticks
        mX = (p1[0] + p2[0]) / 2
        mY = (p1[1] + p2[1]) / 2
        length = math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
        angle = math.degrees(math.atan2(p1[1] - p2[1], p1[0] - p2[0]))

        polygon = cv2.ellipse2Poly(
            (int(mX), int(mY)),
            (int(length / 2), stickwidth),
            int(angle), 0, 360, 1
        )
        # Dim color to 60% like controlnet_aux
        dimmed = [int(float(c) * 0.6) for c in color]
        cv2.fillConvexPoly(canvas, polygon, dimmed)

    # Draw keypoints (circles with full color)
    for idx, (x, y) in keypoints.items():
        color = COLORS[idx] if idx < len(COLORS) else [255, 255, 255]
        cv2.circle(canvas, (int(x), int(y)), 4, color, thickness=-1)

    # cv2 uses BGR, convert to RGB before saving
    canvas_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    Image.fromarray(canvas_rgb).save(output_path)


def _draw_with_pil(keypoints, output_path, img_size):
    """Fallback: draw using PIL (slightly different style but same colors)."""
    img = Image.new("RGB", (img_size, img_size), (0, 0, 0))
    draw = ImageDraw.Draw(img)

    limb_width = max(4, int(img_size / 96))

    # Draw limbs
    for limb_idx, (k1, k2) in enumerate(ACTIVE_LIMBS):
        if k1 not in keypoints or k2 not in keypoints:
            continue
        color = tuple(COLORS[limb_idx])
        p1 = keypoints[k1]
        p2 = keypoints[k2]
        # Dim color slightly
        dimmed = tuple(int(c * 0.7) for c in color)
        draw.line([p1, p2], fill=dimmed, width=limb_width)

    # Draw keypoint circles
    kp_radius = max(4, int(img_size / 128))
    for idx, (x, y) in keypoints.items():
        color = tuple(COLORS[idx]) if idx < len(COLORS) else (255, 255, 255)
        draw.ellipse(
            [x - kp_radius, y - kp_radius, x + kp_radius, y + kp_radius],
            fill=color,
        )

    img.save(output_path)


def load_soma_tpose():
    """Load SOMA-77 T-pose joint positions."""
    skin = np.load(
        "/opt/kimodo/kimodo/assets/skeletons/somaskel77/skin_standard.npz",
        allow_pickle=True,
    )
    bind = skin["bind_rig_transform"]  # (77, 4, 4)
    joint_pos = bind[:, :3, 3]  # (77, 3)
    return joint_pos


if __name__ == "__main__":
    import sys
    output = sys.argv[1] if len(sys.argv) > 1 else "/tmp/soma_openpose.png"

    print("=== SOMA T-pose → OpenPose COCO-18 ===")
    tpose = load_soma_tpose()
    print(f"T-pose joints: {tpose.shape}")
    print(f"  X range: [{tpose[:,0].min():.3f}, {tpose[:,0].max():.3f}]")
    print(f"  Y range: [{tpose[:,1].min():.3f}, {tpose[:,1].max():.3f}]")
    print(f"  Z range: [{tpose[:,2].min():.3f}, {tpose[:,2].max():.3f}]")
    print(f"\nUsing {'cv2' if HAS_CV2 else 'PIL'} for drawing")
    print()
    soma_to_openpose_image(tpose, output)
