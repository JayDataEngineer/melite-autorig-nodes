"""Blender headless format conversion: GLB/FBX -> FBX/GLB.
    blender -b -P blender_convert.py -- <input_path> <output_path> <file_format>
"""
from __future__ import annotations
import sys
import os
import bpy

args = sys.argv[sys.argv.index("--") + 1:]
input_path = args[0]
output_path = args[1]
file_format = args[2].lower() if len(args) > 2 else "fbx"

bpy.ops.wm.read_factory_settings(use_empty=True)

ext = os.path.splitext(input_path)[1].lower()
if ext in (".glb", ".gltf"):
    bpy.ops.import_scene.gltf(filepath=input_path)
elif ext == ".fbx":
    bpy.ops.import_scene.fbx(filepath=input_path)
else:
    raise ValueError(f"Unsupported input format: {ext}")

out_dir = os.path.dirname(output_path)
if out_dir:
    os.makedirs(out_dir, exist_ok=True)

if file_format == "fbx":
    bpy.ops.export_scene.fbx(filepath=output_path)
else:
    bpy.ops.export_scene.gltf(filepath=output_path, export_format="GLB")

if not os.path.isfile(output_path):
    raise RuntimeError(f"Output file was not created: {output_path}")

print(f"CONVERT_OK: {output_path}")
