"""
ONE-COMMAND PIPELINE: SOMA Skeleton → OpenPose → ControlNet → TRELLIS → Rigged GLB

This script chains:
  1. Generate OpenPose image from SOMA-77 skeleton T-pose
  2. Submit ComfyUI ControlNet workflow (ILFlatMix + IllustriousXL_openpose)
  3. Submit TRELLIS image-to-3D workflow
  4. Run SOMA weight transfer
  5. Run Blender retarget with walk cycle

Usage (inside inference-comfyui container):
  python3 soma_openpose_pipeline.py [--npz /tmp/walking_forward.npz] [--seed 42]

Requires: ComfyUI running on localhost:18465, Blender, TRELLIS model.
"""
import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request

COMFY_URL = "http://127.0.0.1:18465"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def submit_workflow(workflow, client_id="pipeline", timeout=600):
    """Submit ComfyUI workflow and poll for completion."""
    prompt_data = {"prompt": workflow, "client_id": client_id}
    req = urllib.request.Request(
        f"{COMFY_URL}/prompt",
        data=json.dumps(prompt_data).encode(),
        headers={"Content-Type": "application/json"},
    )
    resp = json.loads(urllib.request.urlopen(req).read())
    if "node_errors" in resp and resp["node_errors"]:
        for nid, err in resp["node_errors"].items():
            print(f"  NODE ERROR [{nid}]: {err.get('exception_message','?')}")
        return None
    prompt_id = resp["prompt_id"]
    for attempt in range(timeout // 3):
        time.sleep(3)
        try:
            hist = json.loads(
                urllib.request.urlopen(f"{COMFY_URL}/history/{prompt_id}").read()
            )
            if prompt_id in hist:
                outputs = hist[prompt_id].get("outputs", {})
                status = hist[prompt_id].get("status", {})
                if status.get("completed"):
                    return outputs
                elif status.get("status_str") == "error":
                    return None
        except Exception:
            pass
        if attempt % 20 == 0:
            print(f"  [{attempt*3}s] waiting...")
    return None


def step_1_openpose(output_path):
    """Generate OpenPose image from SOMA skeleton."""
    print("\n" + "="*60)
    print("Step 1: SOMA Skeleton → OpenPose Image")
    print("="*60)
    script = os.path.join(SCRIPT_DIR, "soma_to_openpose.py")
    if not os.path.exists(script):
        # Generate inline
        sys.path.insert(0, "/tmp")
        from soma_to_openpose import load_soma_tpose, soma_to_openpose_image
        tpose = load_soma_tpose()
        soma_to_openpose_image(tpose, output_path)
    else:
        subprocess.run([sys.executable, script, output_path], check=True)


def step_2_controlnet(openpose_img, seed=42):
    """Generate character via ControlNet."""
    print("\n" + "="*60)
    print(f"Step 2: OpenPose → ControlNet → Character (seed={seed})")
    print("="*60)

    # Copy OpenPose image to ComfyUI input
    input_dir = "/root/ComfyUI/input"
    os.makedirs(input_dir, exist_ok=True)
    import shutil
    shutil.copy(openpose_img, f"{input_dir}/soma_openpose.png")

    workflow = {
        "load_pose": {"class_type": "LoadImage", "inputs": {"image": "soma_openpose.png"}},
        "checkpoint": {"class_type": "CheckpointLoaderSimple",
                       "inputs": {"ckpt_name": "Illustrious/ILFlatMix.safetensors"}},
        "positive": {"class_type": "CLIPTextEncode", "inputs": {
            "text": ("1girl, solo, full body, T-pose, arms spread horizontally, "
                     "standing straight, character design sheet, "
                     "wearing skin-tight bodysuit, bare arms, bare legs, "
                     "white background, simple flat colors, anime style, "
                     "neutral expression, looking at viewer, symmetrical pose, "
                     "high quality, masterpiece, clean lineart"),
            "clip": ["checkpoint", 1]}},
        "negative": {"class_type": "CLIPTextEncode", "inputs": {
            "text": ("deformed, bad anatomy, extra limbs, missing limbs, "
                     "multiple people, cropped, out of frame, "
                     "bad hands, bad face, distorted, "
                     "heavy clothing, baggy clothes, "
                     "text, watermark, signature"),
            "clip": ["checkpoint", 1]}},
        "controlnet": {"class_type": "ControlNetLoader",
                       "inputs": {"control_net_name": "IllustriousXL_openpose.safetensors"}},
        "apply_cn": {"class_type": "ControlNetApplyAdvanced", "inputs": {
            "positive": ["positive", 0], "negative": ["negative", 0],
            "control_net": ["controlnet", 0], "image": ["load_pose", 0],
            "strength": 0.85, "start_percent": 0.0, "end_percent": 0.9,
            "vae": ["checkpoint", 2]}},
        "latent": {"class_type": "EmptyLatentImage",
                   "inputs": {"width": 1024, "height": 1024, "batch_size": 1}},
        "sampler": {"class_type": "KSampler", "inputs": {
            "model": ["checkpoint", 0], "positive": ["apply_cn", 0],
            "negative": ["apply_cn", 1], "latent_image": ["latent", 0],
            "seed": seed, "steps": 30, "cfg": 5.0,
            "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 1.0}},
        "decode": {"class_type": "VAEDecode",
                   "inputs": {"samples": ["sampler", 0], "vae": ["checkpoint", 2]}},
        "save": {"class_type": "SaveImage",
                 "inputs": {"images": ["decode", 0], "filename_prefix": "soma_character_tpose"}},
    }

    outputs = submit_workflow(workflow, "controlnet_gen", timeout=120)
    if not outputs:
        raise RuntimeError("ControlNet generation failed")

    for node_output in outputs.values():
        if "images" in node_output:
            img = node_output["images"][0]
            char_path = f"/root/ComfyUI/output/{img['filename']}"
            # Copy to input for TRELLIS
            shutil.copy(char_path, f"{input_dir}/soma_character_tpose.png")
            print(f"  Character image: {char_path}")
            return char_path
    raise RuntimeError("No image in ControlNet output")


def step_3_trellis(char_img, seed=42):
    """Convert character image to 3D GLB via TRELLIS."""
    print("\n" + "="*60)
    print("Step 3: Character → TRELLIS → 3D GLB")
    print("="*60)

    workflow = {
        "load_image": {"class_type": "LoadImage",
                       "inputs": {"image": "soma_character_tpose.png"}},
        "load_trellis": {"class_type": "Trellis2Loader", "inputs": {
            "ckpt_path": "/models/3d/trellis/TRELLIS.2-4B/ckpts",
            "pipeline_type": "1024_cascade"}},
        "generate": {"class_type": "Trellis2ImageTo3D", "inputs": {
            "pipeline": ["load_trellis", 0], "image": ["load_image", 0],
            "seed": seed, "resolution": "1024",
            "decimation": 150000, "texture_size": 4096, "max_num_tokens": 8192}},
    }

    outputs = submit_workflow(workflow, "trellis_gen", timeout=600)
    if not outputs:
        raise RuntimeError("TRELLIS generation failed")

    for node_output in outputs.values():
        for key, val in node_output.items():
            if isinstance(val, list):
                for item in val:
                    if isinstance(item, dict) and "filename" in item:
                        glb_path = f"/root/ComfyUI/output/{item['filename']}"
                        print(f"  GLB: {glb_path}")
                        return glb_path
    raise RuntimeError("No GLB in TRELLIS output")


def step_4_weight_transfer(glb_path, output_path):
    """Run SOMA weight transfer on the GLB."""
    print("\n" + "="*60)
    print("Step 4: Weight Transfer (SOMA bone-heat skinning)")
    print("="*60)
    sys.path.insert(0, SCRIPT_DIR)
    from soma_weight_transfer import transfer_soma_weights_and_write_glb
    transfer_soma_weights_and_write_glb(glb_path, output_path)
    print(f"  Rigged: {output_path}")


def step_5_retarget(rigged_path, npz_path, output_path, decim=0.15):
    """Run Blender retarget."""
    print("\n" + "="*60)
    print("Step 5: Blender Retarget (SOMA walk cycle)")
    print("="*60)
    cmd = [
        "blender", "--background", "--python",
        os.path.join(SCRIPT_DIR, "blender_retarget.py"), "--",
        rigged_path, npz_path, output_path, "30", "glb", "true", str(decim),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    for line in result.stdout.splitlines():
        if line.startswith("[retarget]"):
            print(f"  {line}")
    if not os.path.exists(output_path):
        print("  STDERR:", result.stderr[-500:])
        raise RuntimeError("Retarget failed")
    print(f"  Output: {output_path}")


def step_6_validate(glb_path):
    """Run validation."""
    print("\n" + "="*60)
    print("Step 6: Validation")
    print("="*60)
    cmd = ["blender", "--background", "--python",
           os.path.join(SCRIPT_DIR, "validate_rig.py"), "--", glb_path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    for line in result.stdout.splitlines():
        if any(line.startswith(p) for p in ["✅", "❌", "⚠️", "  [", "Mesh", "Weight", "Frame", "Height"]):
            print(f"  {line}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SOMA OpenPose ControlNet Pipeline")
    parser.add_argument("--npz", default="/tmp/walking_forward.npz",
                        help="Path to SOMA NPZ motion file")
    parser.add_argument("--seed", type=int, default=42, help="Generation seed")
    parser.add_argument("--output", default="/tmp/onetool_soma_pipeline.glb",
                        help="Output GLB path")
    parser.add_argument("--skip-to", choices=["openpose", "controlnet", "trellis", "weights", "retarget"],
                        default=None, help="Skip to a specific step")
    args = parser.parse_args()

    t0 = time.time()

    # Step 1: OpenPose
    openpose_path = "/root/ComfyUI/input/soma_openpose.png"
    if args.skip_to in (None, "openpose"):
        step_1_openpose(openpose_path)

    # Step 2: ControlNet
    if args.skip_to in (None, "openpose", "controlnet"):
        char_img = step_2_controlnet(openpose_path, seed=args.seed)

    # Step 3: TRELLIS
    if args.skip_to in (None, "openpose", "controlnet", "trellis"):
        glb_path = step_3_trellis(char_img, seed=args.seed)

    # Step 4: Weight Transfer
    rigged_path = "/tmp/trellis_soma_rigged.glb"
    if args.skip_to in (None, "openpose", "controlnet", "trellis", "weights"):
        step_4_weight_transfer(glb_path, rigged_path)

    # Step 5: Retarget
    if args.skip_to in (None, "openpose", "controlnet", "trellis", "weights", "retarget"):
        step_5_retarget(rigged_path, args.npz, args.output)

    # Step 6: Validate
    step_6_validate(args.output)

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"PIPELINE COMPLETE in {elapsed:.0f}s")
    print(f"Output: {args.output}")
    print(f"{'='*60}")
