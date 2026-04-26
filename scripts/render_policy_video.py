"""Render a trained Brax PPO policy to an MP4.

Loads the most-recent ``workspace/checkpoints/brax_ppo_*/`` (or one chosen
via ``--checkpoint``), replays the policy against ``--robot`` (an MJCF in
``workspace/robots/``), captures frames with ``mujoco.Renderer``, and
writes an ``.mp4`` via ``imageio-ffmpeg``.

Designed to run inside WSL2 (where Brax + JAX GPU + the training venv live)
so we can reconstruct the inference function from saved params.  The
physics step is CPU-only MuJoCo so a GPU is not required for rendering.

Example:
    uv run python scripts/render_policy_video.py \\
        --robot urchin_v2 \\
        --out workspace/renders/urchin_v2_dryrun.mp4 \\
        --seconds 8
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Repo-root resolution so this runs from either side of the WSL boundary.
REPO_ROOT = Path(__file__).resolve().parent.parent


def find_latest_checkpoint(checkpoints_dir: Path) -> Path:
    """Return the newest ``brax_ppo_<timestamp>/`` directory, or raise."""
    candidates = sorted(
        checkpoints_dir.glob("brax_ppo_*"),
        key=lambda p: p.stat().st_mtime,
    )
    if not candidates:
        raise FileNotFoundError(
            f"No brax_ppo_* checkpoint directories found in {checkpoints_dir}"
        )
    return candidates[-1]


def load_brax_inference(ckpt_dir: Path):
    """Reconstruct the Brax PPO inference fn from a saved checkpoint.

    Returns a callable ``obs (np.ndarray) -> action (np.ndarray, in [-1, 1])``
    and the ``(obs_size, action_size)`` tuple for sanity-checking.
    """
    import jax
    import jax.numpy as jnp
    from brax.training.agents.ppo import networks as ppo_networks  # type: ignore
    from robo_garden.training.checkpoints import load_checkpoint

    meta_path = ckpt_dir.with_suffix(".json")
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing metadata sidecar {meta_path}")
    meta = json.loads(meta_path.read_text())
    obs_size = int(meta["obs_size"])
    action_size = int(meta["action_size"])

    ckpt = load_checkpoint(ckpt_dir)
    params = ckpt.get("params")
    if params is None:
        raise RuntimeError(f"Checkpoint {ckpt_dir} has no params payload")

    network = ppo_networks.make_ppo_networks(obs_size, action_size)
    make_inference_fn = ppo_networks.make_inference_fn(network)
    inference_fn = make_inference_fn(params, deterministic=True)
    rng = jax.random.PRNGKey(0)

    def policy(obs: np.ndarray) -> np.ndarray:
        jobs = jnp.asarray(obs, dtype=jnp.float32).reshape(1, -1)
        action, _ = inference_fn(jobs, rng)
        return np.asarray(action, dtype=np.float32).reshape(-1)

    return policy, (obs_size, action_size), meta


def render_policy(
    mjcf_path: Path,
    policy_fn,
    out_path: Path,
    *,
    seconds: float = 8.0,
    frame_rate: int = 30,
    width: int = 960,
    height: int = 540,
    camera: str | None = None,
) -> dict:
    """Step MuJoCo under ``policy_fn`` and write an mp4 of the trajectory.

    Returns a small info dict with per-episode metrics (speed, distance, etc.)
    so the caller can pick the "most successful" run.
    """
    import mujoco
    import imageio.v2 as imageio

    model = mujoco.MjModel.from_xml_path(str(mjcf_path))
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)

    nq, nv, nu = model.nq, model.nv, model.nu
    dt = model.opt.timestep
    substeps_per_frame = max(1, int(round(1.0 / (frame_rate * dt))))
    total_frames = int(round(seconds * frame_rate))

    ctrlrange = np.asarray(model.actuator_ctrlrange, dtype=np.float32)
    lo = ctrlrange[:, 0].copy()
    hi = ctrlrange[:, 1].copy()
    unlimited = hi <= lo
    lo[unlimited] = -1.0
    hi[unlimited] = 1.0
    center = (lo + hi) * 0.5
    halfspan = (hi - lo) * 0.5

    renderer = mujoco.Renderer(model, width=width, height=height)
    if camera is not None:
        try:
            renderer.camera = camera
        except Exception:
            pass

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        str(out_path),
        fps=frame_rate,
        codec="libx264",
        quality=8,
        macro_block_size=1,
    )

    peak_speed = 0.0
    start_xy = np.array([data.qpos[0], data.qpos[1]], dtype=np.float32)
    speeds: list[float] = []

    try:
        for frame_idx in range(total_frames):
            obs = np.concatenate([data.qpos, data.qvel]).astype(np.float32)
            raw = policy_fn(obs)
            raw = np.clip(raw[:nu], -1.0, 1.0)
            data.ctrl[:nu] = center + raw * halfspan

            for _ in range(substeps_per_frame):
                mujoco.mj_step(model, data)
                if np.any(np.isnan(data.qpos)):
                    raise RuntimeError(
                        f"Sim diverged (NaN qpos) at frame {frame_idx}"
                    )

            vx, vy = float(data.qvel[0]), float(data.qvel[1])
            speed_xy = float(np.sqrt(vx * vx + vy * vy))
            speeds.append(speed_xy)
            peak_speed = max(peak_speed, speed_xy)

            renderer.update_scene(data, camera=camera if camera else -1)
            pixels = renderer.render()
            writer.append_data(pixels)
    finally:
        writer.close()
        renderer.close()

    end_xy = np.array([data.qpos[0], data.qpos[1]], dtype=np.float32)
    distance = float(np.linalg.norm(end_xy - start_xy))
    mean_speed = float(np.mean(speeds)) if speeds else 0.0
    return {
        "seconds": seconds,
        "frames": total_frames,
        "peak_speed_mps": peak_speed,
        "mean_speed_mps": mean_speed,
        "distance_m": distance,
        "out_path": str(out_path),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--robot", required=True, help="Robot name (MJCF in workspace/robots/)")
    ap.add_argument("--checkpoint", help="Path to brax_ppo_<ts>/ dir. Default: newest.")
    ap.add_argument("--out", required=True, help="Output mp4 path")
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=540)
    ap.add_argument("--camera", default=None, help="Named camera in the MJCF, or None for free cam")
    ap.add_argument("--trials", type=int, default=3, help="Render this many episodes and keep the best")
    args = ap.parse_args()

    # Add src/ to path so ``import robo_garden.*`` works when this script is
    # invoked with plain ``python`` instead of ``uv run`` (which already
    # prepends the project root).
    sys.path.insert(0, str(REPO_ROOT / "src"))

    mjcf_path = REPO_ROOT / "workspace" / "robots" / f"{args.robot}.xml"
    if not mjcf_path.exists():
        raise FileNotFoundError(f"MJCF not found: {mjcf_path}")

    if args.checkpoint:
        ckpt_dir = Path(args.checkpoint)
    else:
        ckpt_dir = find_latest_checkpoint(REPO_ROOT / "workspace" / "checkpoints")
    print(f"Using checkpoint: {ckpt_dir}")

    policy, (obs_size, action_size), meta = load_brax_inference(ckpt_dir)
    print(f"Loaded Brax PPO inference: obs={obs_size}, action={action_size}, "
          f"best_reward={meta.get('best_reward', 'n/a')}, "
          f"total_timesteps={meta.get('total_timesteps', 'n/a')}")

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = REPO_ROOT / out_path

    # Render `args.trials` episodes, keep only the one with the largest
    # XY distance travelled (proxy for "most successful roll").
    best_info = None
    best_path = None
    for trial in range(max(1, args.trials)):
        # No explicit rng seed on the policy — Brax make_inference_fn with
        # deterministic=True ignores rng anyway.  Variation comes from
        # MuJoCo's initial qpos perturbation which we don't seed here; each
        # trial starts from identical qpos=0 so trials are currently
        # identical.  Keep the loop as a placeholder so we can add
        # randomised init later without restructuring callers.
        trial_out = out_path.with_name(f"{out_path.stem}_trial{trial}{out_path.suffix}")
        info = render_policy(
            mjcf_path=mjcf_path,
            policy_fn=policy,
            out_path=trial_out,
            seconds=args.seconds,
            frame_rate=args.fps,
            width=args.width,
            height=args.height,
            camera=args.camera,
        )
        print(f"  trial {trial}: distance={info['distance_m']:.3f} m, "
              f"peak_speed={info['peak_speed_mps']:.3f} m/s -> {trial_out.name}")
        if best_info is None or info["distance_m"] > best_info["distance_m"]:
            best_info = info
            best_path = trial_out

    # Copy the best trial to the canonical --out path.
    import shutil
    if best_path is not None and best_path != out_path:
        shutil.copy2(best_path, out_path)
    print(
        f"\nMost-successful trial:\n"
        f"  distance:   {best_info['distance_m']:.3f} m\n"
        f"  mean speed: {best_info['mean_speed_mps']:.3f} m/s\n"
        f"  peak speed: {best_info['peak_speed_mps']:.3f} m/s\n"
        f"  wrote:      {out_path}"
    )


if __name__ == "__main__":
    main()
