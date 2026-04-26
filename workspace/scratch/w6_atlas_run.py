"""W6 (2026-04-25) — Open-loop locomotion atlas Isaac harness.

Reads the LHS samples produced by `w6_atlas_lhs.py`, instantiates a
single urchin_v3 env (under `URCHIN_RESET_MODE=canonical`), and rolls
out 4 s of open-loop control per sample with the explicit three-field
`RollingEngine` driven by the sample's parameters.

Per-sample metrics (from `env.extras["diagnostics"]` populated by
`UrchinEnv._compute_diagnostics`, plus root pose deltas):

    mean_speed_mps       Σ(forward speed along goal direction) / steps
    net_displacement_m   ||spawn_xy - final_xy||_2
    mean_slip_ratio      mean of |slip_ratio| over the episode
    mean_cot             mean of cot_inst, ignoring near-zero-velocity steps
    mean_support_asym    mean of |support_asym|
    success              net_displacement_m > 0.5 AND mean_speed_mps > 0.10

Output:
    workspace/_tasks_out/w6_atlas/atlas.parquet (or atlas.csv if pyarrow
    is unavailable on this env).

Constraints:
    - This script imports Isaac Lab. It is meant to be invoked from a
      WSL shell. On Windows it must exit cleanly with a clear message.
    - Module-top imports are limited to stdlib + numpy. Isaac Lab is
      imported lazily inside `main()` after `_has_isaaclab()` confirms
      it's available.
    - Reuses the patterns from `record_primitive_dataset.py` (engine
      cfg construction, axis/SH-basis setup, env step loop). Does NOT
      modify rolling_engine.py or urchin_env_cfg.py.

Usage (from WSL):
    URCHIN_RESET_MODE=canonical \\
        uv run python workspace/scratch/w6_atlas_run.py \\
        --samples workspace/_tasks_out/w6_atlas/lhs_samples.csv \\
        --num-envs 4
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from pathlib import Path

import numpy as np


def _has_isaaclab() -> bool:
    """Import-test for Isaac Lab. Use a real try-import — find_spec
    raises ValueError on Isaac Lab installs (`isaaclab.app.__spec__ is None`)."""
    try:
        from isaaclab.app import AppLauncher  # noqa: F401
        return True
    except Exception:
        return False


def _read_lhs_csv(path: Path) -> tuple[list[str], np.ndarray]:
    """Return (header_names_minus_sample_id, (n, 5) float array)."""
    if not path.exists():
        raise SystemExit(f"--samples CSV not found: {path}")
    rows = []
    with path.open() as fh:
        reader = csv.reader(fh)
        header = next(reader)
        for row in reader:
            rows.append([float(v) for v in row[1:]])  # drop sample_id
    if not rows:
        raise SystemExit(f"--samples CSV {path} is empty")
    return header[1:], np.asarray(rows, dtype=np.float64)


def _write_atlas(path_no_ext: Path, rows: list[dict]) -> Path:
    """Write rows to atlas.parquet (preferred) or atlas.csv (fallback).

    Returns the actual path written.
    """
    path_no_ext.parent.mkdir(parents=True, exist_ok=True)
    try:
        import pandas as pd
        df = pd.DataFrame(rows)
        try:
            parquet_path = path_no_ext.with_suffix(".parquet")
            df.to_parquet(parquet_path, index=False)
            return parquet_path
        except Exception as exc:  # pyarrow / fastparquet not available
            print(f"[w6-run] parquet write failed ({exc!r}); falling "
                  "back to CSV.", file=sys.stderr)
    except Exception as exc:                                # pragma: no cover
        print(f"[w6-run] pandas unavailable ({exc!r}); writing CSV via "
              "stdlib.", file=sys.stderr)

    csv_path = path_no_ext.with_suffix(".csv")
    if rows:
        keys = list(rows[0].keys())
        with csv_path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
    return csv_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--samples", type=Path,
        default=Path("workspace/_tasks_out/w6_atlas/lhs_samples.csv"),
        help="Path to LHS samples CSV from w6_atlas_lhs.py.",
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("workspace/_tasks_out/w6_atlas/atlas"),
        help="Output path WITHOUT extension. .parquet preferred, "
             ".csv fallback if pyarrow is unavailable.",
    )
    parser.add_argument("--num-envs", type=int, default=4, dest="num_envs",
                        help="Parallel envs per Isaac instance (default 4).")
    parser.add_argument("--episode-s", type=float, default=4.0,
                        dest="episode_s")
    parser.add_argument(
        "--goal-distance", type=float, default=2.0, dest="goal_distance",
        help="Spawn at (0,0), goal at (goal_distance, 0). Long enough "
             "that the env doesn't auto-resample mid-episode.",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="torch seed before env.reset (init pose / contact RNG).",
    )
    parser.add_argument(
        "--max-samples", type=int, default=0,
        help="Cap the run at this many samples (0 = no cap). For smoke.",
    )
    parser.add_argument(
        "--reset-each-sample", action="store_true",
        default=True,
        help="env.reset() between samples. Always on; flag retained "
             "for clarity in the CLI surface.",
    )
    parser.add_argument(
        "--print-every", type=int, default=16,
        help="Stdout progress cadence (samples). Default 16.",
    )
    # Isaac AppLauncher only added inside main() after we confirm
    # we're running in WSL — Windows imports of AppLauncher fail.
    args, unknown = parser.parse_known_args(argv)

    if not _has_isaaclab():
        msg = (
            "[w6-run] Isaac Lab not importable on this interpreter. "
            "This script requires WSL Isaac Lab. Skipping cleanly.\n"
            "[w6-run] Run via:\n"
            "          URCHIN_RESET_MODE=canonical uv run python "
            "workspace/scratch/w6_atlas_run.py --samples <csv>\n"
        )
        print(msg, file=sys.stderr)
        return 0

    # ---- Isaac Lab path: hard-set canonical reset BEFORE module import ----
    os.environ.setdefault("URCHIN_RESET_MODE", "canonical")
    os.environ["URCHIN_EPISODE_S"] = str(max(args.episode_s + 1.0, 8.0))
    os.environ["URCHIN_GOAL_SAMPLING_MODE"] = "ring"
    os.environ["URCHIN_START_XY"] = "0.0,0.0"
    os.environ["URCHIN_GOAL_XY"] = f"{args.goal_distance},0.0"
    os.environ["URCHIN_DIST_SCALE_START"] = "1.0"
    os.environ["URCHIN_DIST_SCALE_END"] = "1.0"

    # Lazy imports — keep module-top clean for Windows.
    from isaaclab.app import AppLauncher

    # Parse just the AppLauncher args so headless flag and friends work.
    al_parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(al_parser)
    al_args, _ = al_parser.parse_known_args(unknown)
    al_args.headless = True

    app_launcher = AppLauncher(al_args)
    sim_app = app_launcher.app

    # Make urchin_v3 imports work the same way record_primitive_dataset does.
    import isaaclab as _il
    _ISAACLAB_SRC = Path(_il.__file__).parent / "source" / "isaaclab"
    if _ISAACLAB_SRC.exists() and str(_ISAACLAB_SRC) not in sys.path:
        sys.path.insert(0, str(_ISAACLAB_SRC))

    # __file__ = <repo>/workspace/scratch/w6_atlas_run.py
    # parents[1] = <repo>/workspace; need <repo>/workspace/robots/urchin_v3.
    urchin_root = (Path(__file__).resolve().parents[1]
                   / "robots" / "urchin_v3")
    sys.path.insert(0, str(urchin_root))

    import torch
    import json

    from urchin_v3.urchin_env_cfg import UrchinEnv, UrchinEnvCfg
    from urchin_v3.urchin_v3_cfg import NUM_PANELS
    from scripts.scripted_roll import build_sh_basis
    from scripts import rolling_engine
    from scripts.rolling_engine import RollingEngineCfg, new_state

    try:
        return _run(args, AppLauncher, sim_app, torch, json, UrchinEnv,
                    UrchinEnvCfg, NUM_PANELS, build_sh_basis,
                    rolling_engine, RollingEngineCfg, new_state, urchin_root)
    finally:
        try:
            sim_app.close()
        except Exception:
            pass


def _run(args, AppLauncher, sim_app, torch, json, UrchinEnv, UrchinEnvCfg,
         NUM_PANELS, build_sh_basis, rolling_engine, RollingEngineCfg,
         new_state, urchin_root):
    """Inner function — keeps the Isaac-loaded code path narrow."""
    headers, samples = _read_lhs_csv(args.samples)
    expected = ["rear_push_amp", "front_retract_amp", "lean_phase",
                "phase_velocity_hz", "push_duty"]
    if headers != expected:
        raise SystemExit(
            f"[w6-run] Unexpected CSV columns: {headers}. "
            f"Expected: {expected}"
        )

    if args.max_samples > 0:
        samples = samples[: args.max_samples]
    n_samples = samples.shape[0]
    print(f"[w6-run] loaded {n_samples} LHS samples from "
          f"{args.samples.resolve()}")

    torch.manual_seed(args.seed)

    # Load axes / SH basis the same way record_primitive_dataset.py does.
    meta_path = urchin_root / "assets" / "urdf" / "urdf_meta.json"
    meta = json.loads(meta_path.read_text())
    axes_cpu = torch.tensor(
        [p["axis"] for p in meta["panels"]], dtype=torch.float32,
    )
    assert axes_cpu.shape[0] == NUM_PANELS

    cfg = UrchinEnvCfg()
    cfg.scene.num_envs = args.num_envs
    env = UrchinEnv(cfg)
    device = env.device

    axes = axes_cpu.to(device)
    B = build_sh_basis(axes)
    B_pinv = torch.linalg.pinv(B)

    env_dt = cfg.sim.dt * cfg.decimation
    steps_per_ep = int(args.episode_s / env_dt)
    print(f"[w6-run] env_dt={env_dt:.4f}s steps/ep={steps_per_ep} "
          f"num_envs={args.num_envs}")

    # Fix yaw to goal_angle (=0 with goal=(D,0), spawn=(0,0)).
    env._yaw_span_override = 0.0  # type: ignore[attr-defined]

    rows: list[dict] = []
    t_run0 = time.time()

    for sid in range(n_samples):
        rear_push, front_retract, lean_phase_cycles, phase_hz, push_duty = (
            samples[sid].tolist()
        )

        # Convert lean_phase from cycles -> radians (controller expects rad).
        # The deep-research report parameterises lean_phase in cycles
        # (a fraction of one push cycle); rolling_engine.py's API takes
        # radians. 1 cycle = 2*pi rad.
        lean_phase_rad = float(lean_phase_cycles) * 2.0 * math.pi

        engine_cfg = RollingEngineCfg(
            rear_push_amp=float(rear_push),
            front_reach_amp=1.0,                    # outside W6 sweep
            front_retract_amp=float(front_retract),
            support_width=6.0,                      # default
            support_bias=0.0,
            duty_cycle=1.0,                         # legacy alias kept inert
            push_duty=float(push_duty),
            lean_duty=1.0,
            retract_duty=1.0,
            push_phase=0.0,
            lean_phase=lean_phase_rad,
            retract_phase=0.0,
            steering_bias=0.0,                      # straight-line atlas
            phase_velocity_hz=float(phase_hz),
            breathing_gain=0.0,
        )

        # Reset per sample. The env's _reset_idx (under canonical mode)
        # writes root link pose, COM velocity, and joint state in the
        # canonical four-step sequence and clears PhysX buffers.
        obs_dict, _ = env.reset()
        obs = (obs_dict["policy"] if isinstance(obs_dict, dict)
               else obs_dict)

        # Capture spawn position for net-displacement calc.
        robot = env.scene["robot"]
        spawn_xy = robot.data.root_pos_w[:, :2].clone()  # (N, 2)

        engine_state = new_state(
            num_envs=args.num_envs, device=device, dtype=obs.dtype,
        )

        speed_acc = torch.zeros(args.num_envs, device=device, dtype=torch.float32)
        slip_acc = torch.zeros_like(speed_acc)
        cot_acc = torch.zeros_like(speed_acc)
        cot_count = torch.zeros_like(speed_acc)
        sasym_acc = torch.zeros_like(speed_acc)

        with torch.no_grad():
            for step in range(steps_per_ep):
                if not sim_app.is_running():
                    raise RuntimeError("Isaac Sim app exited mid-atlas")

                # Advance phase, then forward.
                engine_state = rolling_engine.advance_phase(
                    engine_state, engine_cfg, env_dt,
                )

                panel_target = rolling_engine.forward(
                    axes_b=axes,
                    projected_gravity_b=obs[:, 6:9],
                    to_goal_b=obs[:, 93:95],
                    state=engine_state,
                    cfg=engine_cfg,
                )                                              # (N, 42)
                action_sh = panel_target @ B_pinv.T            # (N, 9)

                # Pull diagnostics for the OBS we just consumed (the
                # extras dict was written in the previous _get_observations
                # call — env.reset for step 0, env.step for step k>=1).
                diag = (env.extras.get("diagnostics", {})
                        if hasattr(env, "extras") and env.extras
                        else {})

                # Forward speed along goal direction (body frame goal x).
                # to_goal_b[:, :2] is body-frame goal; we use world-frame
                # speed component via root_lin_vel_w projected on travel_dir.
                vel_xy = robot.data.root_lin_vel_w[:, :2]
                pos_xy = robot.data.root_pos_w[:, :2]
                # Goal direction = unit vector from current pos to goal.
                # Goal is fixed at (goal_distance, 0).
                goal_xy = torch.tensor(
                    [args.goal_distance, 0.0], device=device,
                ).expand_as(pos_xy)
                delta = goal_xy - pos_xy
                travel = delta / (delta.norm(dim=-1, keepdim=True) + 1e-6)
                fwd_speed = (vel_xy * travel).sum(-1)
                speed_acc += fwd_speed.float()

                slip_t = diag.get("slip_ratio")
                if slip_t is not None:
                    slip_acc += slip_t.abs().float()

                cot_t = diag.get("cot_inst")
                if cot_t is not None:
                    # Only count steps where speed is meaningful (the
                    # diag function already zeroes near-stationary steps,
                    # so we count nonzero entries to avoid division-by-zero
                    # bias in the per-sample mean).
                    mask = (cot_t.abs() > 0).float()
                    cot_acc += cot_t.float() * mask
                    cot_count += mask

                sasym_t = diag.get("support_asym")
                if sasym_t is not None:
                    sasym_acc += sasym_t.abs().float()

                next_obs_dict, _r, _term, _trunc, _info = env.step(action_sh)
                obs = (next_obs_dict["policy"]
                       if isinstance(next_obs_dict, dict)
                       else next_obs_dict)

        final_xy = robot.data.root_pos_w[:, :2]
        net_disp = (final_xy - spawn_xy).norm(dim=-1)            # (N,)
        mean_speed = (speed_acc / steps_per_ep).cpu().numpy()
        mean_slip = (slip_acc / steps_per_ep).cpu().numpy()
        mean_sasym = (sasym_acc / steps_per_ep).cpu().numpy()
        mean_cot = torch.where(
            cot_count > 0,
            cot_acc / cot_count.clamp(min=1.0),
            torch.zeros_like(cot_acc),
        ).cpu().numpy()
        net_disp_np = net_disp.cpu().numpy()

        # Per-sample summary = mean across the parallel envs (they share
        # spawn/goal/cfg so this is a low-variance estimate).
        row = {
            "sample_id": int(sid),
            "rear_push_amp": float(rear_push),
            "front_retract_amp": float(front_retract),
            "lean_phase": float(lean_phase_cycles),
            "phase_velocity_hz": float(phase_hz),
            "push_duty": float(push_duty),
            "mean_speed_mps": float(np.mean(mean_speed)),
            "net_displacement_m": float(np.mean(net_disp_np)),
            "mean_slip_ratio": float(np.mean(mean_slip)),
            "mean_cot": float(np.mean(mean_cot)),
            "mean_support_asym": float(np.mean(mean_sasym)),
            "n_envs": int(args.num_envs),
        }
        row["success"] = bool(
            row["net_displacement_m"] > 0.5
            and row["mean_speed_mps"] > 0.10
        )
        rows.append(row)

        if (sid + 1) % args.print_every == 0 or sid == n_samples - 1:
            elapsed = time.time() - t_run0
            rate = (sid + 1) / max(1e-6, elapsed)
            print(f"[w6-run] sample {sid + 1}/{n_samples}  "
                  f"net_disp={row['net_displacement_m']:.2f}m  "
                  f"speed={row['mean_speed_mps']:.2f}m/s  "
                  f"slip={row['mean_slip_ratio']:.2f}  "
                  f"cot={row['mean_cot']:.2f}  "
                  f"success={row['success']}  "
                  f"rate={rate:.2f}/s",
                  flush=True)

    out_path = _write_atlas(args.output, rows)
    print(f"[w6-run] DONE wrote {len(rows)} rows -> {out_path.resolve()}",
          flush=True)
    # Hard-exit. Isaac Sim's teardown path can hang for 25+ minutes on
    # Windows after `sim_app.close()`; we have all the data we need
    # already, so bypass the slow shutdown.
    os._exit(0)


if __name__ == "__main__":
    try:
        rc = main()
    except SystemExit:
        raise
    except Exception:
        import traceback
        traceback.print_exc()
        rc = 1
    finally:
        # Ensure Isaac Sim's app handle gets released even on exception.
        # main() already does this in its finally; the bare except above
        # is for non-Isaac error paths.
        pass
    sys.exit(rc)
