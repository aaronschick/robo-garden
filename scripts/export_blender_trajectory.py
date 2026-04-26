"""Extract per-joint and base trajectories from spheriball_rolling.blend.

Run headless with Blender's bundled Python (which has bpy + numpy):

    blender --background workspace/robots/urchin_v2/assets/spheriball_rolling.blend \
            --python scripts/export_blender_trajectory.py

Or open the blend in interactive Blender and run via Text Editor.
The same logic also runs in-process when the Blender MCP is connected
(see callers in the planning thread).

Outputs:
    workspace/datasets/urchin_v2_blender_trajectory.npz
        joint_pos    (T, 42)   float32   meters along URDF joint axis, >= 0
        joint_names  (42,)     str       URDF declaration order
        base_pos     (T, 3)    float32   world-space base translation
        base_quat    (T, 4)    float32   world-space base rotation, wxyz
        fps          ()        float32   24.0
        joint_pos_raw (T, 42)  float32   pre-calibration tile.location.z (debug)

URDF joint order (frozen): vc_00..vc_29 then sol_00..sol_11. Confirmed
against workspace/robots/urchin_v2/assets/urdf/urchin_v2.urdf and
urchin_v2/urchin_v2_cfg.py (NUM_VOICE_COILS=30, NUM_SOLENOIDS=12).

Calibration: each tile object's location.z at frame 1 is taken as the
joint=0 rest pose. The sign that makes (z(t) - z(rest)) non-negative
across the whole trajectory is selected per joint. Per-joint stroke
limits (0.030 vc, 0.040 sol) are clipped after sign resolution; values
within numeric tolerance are accepted, larger ones logged as warnings.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import bpy
import numpy as np


JOINT_ORDER = (
    [f"vc_{i:02d}" for i in range(30)]
    + [f"sol_{i:02d}" for i in range(12)]
)
ROOT_NAME = "Urchin_v2_ROOT"
DEFAULT_OUT = Path("workspace/datasets/urchin_v2_blender_trajectory.npz")


def euler_to_quat_wxyz(euler) -> np.ndarray:
    """Blender XYZ-extrinsic euler -> wxyz quaternion."""
    from mathutils import Euler
    q = Euler(euler, "XYZ").to_quaternion()
    return np.array([q.w, q.x, q.y, q.z], dtype=np.float32)


def extract(out_path: Path = DEFAULT_OUT) -> dict:
    scene = bpy.context.scene
    fps = float(scene.render.fps)
    f0, f1 = scene.frame_start, scene.frame_end
    T = f1 - f0 + 1

    tiles = []
    for name in JOINT_ORDER:
        tile_name = f"{name}_tile"
        if tile_name not in bpy.data.objects:
            raise KeyError(f"missing tile object: {tile_name}")
        tiles.append(bpy.data.objects[tile_name])

    if ROOT_NAME not in bpy.data.objects:
        raise KeyError(f"missing root empty: {ROOT_NAME}")
    root = bpy.data.objects[ROOT_NAME]

    raw_z = np.zeros((T, len(tiles)), dtype=np.float32)
    base_pos = np.zeros((T, 3), dtype=np.float32)
    base_quat = np.zeros((T, 4), dtype=np.float32)

    for ti, frame in enumerate(range(f0, f1 + 1)):
        scene.frame_set(frame)
        for ji, tile in enumerate(tiles):
            raw_z[ti, ji] = tile.location.z
        base_pos[ti] = list(root.location)
        base_quat[ti] = euler_to_quat_wxyz(root.rotation_euler)

    rest = raw_z[0]
    delta = raw_z - rest
    span_pos = delta.max(axis=0)
    span_neg = -delta.min(axis=0)
    sign = np.where(span_pos >= span_neg, 1.0, -1.0).astype(np.float32)
    joint_pos = (delta * sign).clip(min=0.0).astype(np.float32)

    stroke = np.array(
        [0.030] * 30 + [0.040] * 12, dtype=np.float32,
    )
    over = joint_pos.max(axis=0) - stroke
    if (over > 1e-3).any():
        bad = [(JOINT_ORDER[i], float(joint_pos[:, i].max()), float(stroke[i]))
               for i in np.where(over > 1e-3)[0]]
        print(f"[export] WARNING: joints exceed URDF stroke: {bad}", flush=True)
    joint_pos = np.minimum(joint_pos, stroke)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        joint_pos=joint_pos,
        joint_names=np.array(JOINT_ORDER),
        base_pos=base_pos,
        base_quat=base_quat,
        fps=np.float32(fps),
        joint_pos_raw=raw_z,
    )

    summary = {
        "frames": T,
        "fps": fps,
        "out": str(out_path.resolve()),
        "joint_min": float(joint_pos.min()),
        "joint_max": float(joint_pos.max()),
        "joint_mean_max": float(joint_pos.max(axis=0).mean()),
        "base_pos_range": {
            "x": [float(base_pos[:, 0].min()), float(base_pos[:, 0].max())],
            "y": [float(base_pos[:, 1].min()), float(base_pos[:, 1].max())],
            "z": [float(base_pos[:, 2].min()), float(base_pos[:, 2].max())],
        },
        "sign_flips": int((sign < 0).sum()),
    }
    print("[export]", json.dumps(summary, indent=2), flush=True)
    return summary


if __name__ == "__main__":
    out = DEFAULT_OUT
    if "--" in sys.argv:
        post = sys.argv[sys.argv.index("--") + 1:]
        if post:
            out = Path(post[0])
    extract(out)
