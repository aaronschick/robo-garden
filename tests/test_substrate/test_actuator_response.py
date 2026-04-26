"""W4 — Actuator step-response identification (urchin_v3, sim-only).

Two tests:

  test_single_panel_step_response
      Single urchin instance, all panels at rest_pos. At t=0 step ONE selected
      panel from rest_pos to its full stroke (limit_upper). Sample joint pos
      every physics step for 0.5 s. Compute and write to CSV: rise time
      (10%->90%), settle time (within 5% of target), overshoot percentage,
      peak velocity. Repeat for 4 panels at different orientations on the
      shell to bound variance.

  test_paired_panel_step_response
      Same single instance / step protocol, but step TWO panels simultaneously,
      first as a diametrically-opposite pair, then as an adjacent pair. Used
      to bound coupling and contact-induced delay between panels.

Outputs:
  workspace/_tasks_out/w4_substrate_id/actuator_response/single_panel.csv
  workspace/_tasks_out/w4_substrate_id/actuator_response/paired_panel.csv

These tests instantiate the live UrchinEnv and bypass the SH residual decode
in `_pre_physics_step` — we don't want oracle/contact-push interfering with
a clean step response. We DO use `URCHIN_RESET_MODE=canonical` so the env
state is clean across panel selections.

Top-level imports are `pytest`-only (and stdlib). The Isaac Lab + urchin
imports are guarded behind `_has_isaaclab()` and only happen inside the
test bodies, so Windows-side `pytest --collect-only` doesn't crash.

Run (WSL):
    cd /mnt/c/Users/aaron/Documents/repositories/robo-garden && \
      URCHIN_RESET_MODE=canonical \
      C:/isaac-venv/Scripts/python.exe -m pytest \
        tests/test_substrate/test_actuator_response.py -x -s
"""
from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Capability detection (no hard Isaac Lab import at module top).
# ---------------------------------------------------------------------------


def _has_isaaclab() -> bool:
    """Return True iff `from isaaclab.app import AppLauncher` succeeds.

    `find_spec` is unreliable here: Isaac Lab installs with
    `isaaclab.app.__spec__ is None`, so `find_spec` raises ValueError.
    Do a real try-import instead.
    """
    try:
        from isaaclab.app import AppLauncher  # noqa: F401
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _has_isaaclab(),
    reason="requires Isaac Lab (run inside WSL2 Ubuntu-22.04)",
)


# ---------------------------------------------------------------------------
# Output paths.
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[2]
_OUT_DIR = _REPO_ROOT / "workspace" / "_tasks_out" / "w4_substrate_id" / "actuator_response"


def _ensure_out_dir() -> Path:
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    return _OUT_DIR


# ---------------------------------------------------------------------------
# Step-response analytics (pure numpy; reused by both tests).
# ---------------------------------------------------------------------------


def _step_metrics(
    times_s,
    pos,
    *,
    rest: float,
    target: float,
):
    """Compute rise / settle / overshoot / peak velocity from a 1-D step trace.

    Args:
        times_s: (T,) physics-step times in seconds.
        pos:     (T,) joint position over the step.
        rest:    starting position.
        target:  commanded final position.

    Returns:
        dict with rise_time_ms, settle_time_ms, overshoot_pct, peak_vel_mps.
        Any field that cannot be determined from the trace (e.g. response
        never crossed 90 % within the window) is set to None — the caller
        should treat None as "not converged in window".
    """
    import numpy as np

    times_s = np.asarray(times_s, dtype=np.float64)
    pos = np.asarray(pos, dtype=np.float64)
    span = target - rest
    if abs(span) < 1e-9:
        return {
            "rise_time_ms": None, "settle_time_ms": None,
            "overshoot_pct": None, "peak_vel_mps": None,
        }

    # Normalised response in [0, 1] (rest -> 0, target -> 1).
    norm = (pos - rest) / span

    # Rise: first time |norm| >= 0.1 to first time >= 0.9.
    rise_start = None
    rise_end = None
    for i in range(norm.shape[0]):
        if rise_start is None and norm[i] >= 0.1:
            rise_start = times_s[i]
        if norm[i] >= 0.9:
            rise_end = times_s[i]
            break
    rise_time_ms = (
        1000.0 * (rise_end - rise_start)
        if (rise_start is not None and rise_end is not None)
        else None
    )

    # Settle: last time |norm - 1| > 0.05; settle_time = t after that.
    out_of_band = np.abs(norm - 1.0) > 0.05
    if out_of_band.any():
        last_out = int(np.where(out_of_band)[0][-1])
        if last_out + 1 < times_s.shape[0]:
            settle_time_ms = 1000.0 * times_s[last_out + 1]
        else:
            settle_time_ms = None
    else:
        # In-band from t=0 — degenerate, but report 0.
        settle_time_ms = 0.0

    # Overshoot: max norm - 1, expressed as percent of span. Negative span
    # (retract) handled by sign-flipping: norm is always normalised so 1.0
    # is target and overshoot is norm > 1.
    peak = float(np.max(norm))
    overshoot_pct = max(0.0, (peak - 1.0)) * 100.0

    # Peak velocity: max |dx/dt| over the trace, in joint-units / sec
    # (panels are prismatic: m/s).
    if times_s.shape[0] >= 2:
        dt = np.diff(times_s)
        dpos = np.diff(pos)
        vel = np.abs(dpos / np.where(dt > 0, dt, 1e-9))
        peak_vel_mps = float(np.max(vel))
    else:
        peak_vel_mps = None

    return {
        "rise_time_ms": rise_time_ms,
        "settle_time_ms": settle_time_ms,
        "overshoot_pct": overshoot_pct,
        "peak_vel_mps": peak_vel_mps,
    }


def _coupling_delay_ms(
    times_s,
    pos_a,
    pos_b,
    *,
    rest: float,
    target: float,
):
    """Time difference between when panel A and panel B first cross 50 % of span.

    Useful for the paired-panel test: if both panels are commanded
    simultaneously and the env is mechanically perfect, the delay is 0.
    Contact / coupling pushes one panel later than the other.
    """
    import numpy as np

    times_s = np.asarray(times_s, dtype=np.float64)
    pos_a = np.asarray(pos_a, dtype=np.float64)
    pos_b = np.asarray(pos_b, dtype=np.float64)
    span = target - rest
    if abs(span) < 1e-9:
        return None

    half = rest + 0.5 * span

    def _first_cross(p):
        if span > 0:
            mask = p >= half
        else:
            mask = p <= half
        if not mask.any():
            return None
        return float(times_s[int(np.where(mask)[0][0])])

    ta = _first_cross(pos_a)
    tb = _first_cross(pos_b)
    if ta is None or tb is None:
        return None
    return 1000.0 * (tb - ta)


# ---------------------------------------------------------------------------
# Isaac Lab driver helpers (created INSIDE tests to avoid module-level imports).
# ---------------------------------------------------------------------------


def _launch_app_and_env():
    """Launch Isaac Lab AppLauncher and instantiate UrchinEnv.

    Caller MUST close `sim_app` when done (via `sim_app.close()`).
    Forces canonical reset + headless mode. Returns (env, sim_app, cfg).
    """
    os.environ["URCHIN_RESET_MODE"] = "canonical"

    # AppLauncher must be constructed before any other Isaac imports.
    from isaaclab.app import AppLauncher
    import argparse
    parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(parser)
    # Pass an empty arg list so pytest's argv doesn't pollute parsing.
    args = parser.parse_args([])
    args.headless = True
    args.enable_cameras = False
    app_launcher = AppLauncher(args)
    sim_app = app_launcher.app

    import isaaclab as _il
    _src = Path(_il.__file__).parent / "source" / "isaaclab"
    if _src.exists() and str(_src) not in sys.path:
        sys.path.insert(0, str(_src))

    # urchin_v3 lives at workspace/robots/urchin_v3/urchin_v3 (package).
    workspace_root = _REPO_ROOT / "workspace" / "robots" / "urchin_v3"
    if str(workspace_root) not in sys.path:
        sys.path.insert(0, str(workspace_root))
    from urchin_v3.urchin_env_cfg import UrchinEnv, UrchinEnvCfg  # noqa: E402

    cfg = UrchinEnvCfg()
    cfg.scene.num_envs = 1
    cfg.episode_length_s = 10.0   # generous; we time-slice the test loop
    cfg.reset_mode = "canonical"
    # Zero out RL shaping rewards — we only care about joint kinematics.
    cfg.progress_reward_weight = 0.0
    cfg.distance_penalty_weight = 0.0
    cfg.goal_bonus = 0.0
    cfg.wall_contact_penalty = 0.0

    env = UrchinEnv(cfg, render_mode=None)
    return env, sim_app, cfg


def _select_panels_by_orientation(axes_b):
    """Pick 4 panels at distinct orientations on the shell.

    We use the panel body-frame axes (`env._axes_b`, shape (42, 3)) and pick:
        - the panel closest to +Z (top)
        - the panel closest to -Z (bottom)
        - the panel closest to +X (front)
        - the panel closest to -X (back)
    This gives us a representative bound on per-panel response variance
    without sweeping all 42.
    """
    import numpy as np
    axes = axes_b.detach().cpu().numpy() if hasattr(axes_b, "detach") else np.asarray(axes_b)

    def _argclosest(direction):
        return int(np.argmax(axes @ np.asarray(direction, dtype=axes.dtype)))

    return {
        "top":    _argclosest((0.0, 0.0, +1.0)),
        "bottom": _argclosest((0.0, 0.0, -1.0)),
        "front":  _argclosest((+1.0, 0.0, 0.0)),
        "back":   _argclosest((-1.0, 0.0, 0.0)),
    }


def _diametrically_opposite_pair(axes_b):
    """Pick a panel + the panel whose axis is most anti-parallel to it."""
    import numpy as np
    axes = axes_b.detach().cpu().numpy() if hasattr(axes_b, "detach") else np.asarray(axes_b)
    a = int(np.argmax(axes @ np.asarray((0.0, 0.0, +1.0), dtype=axes.dtype)))
    b = int(np.argmin(axes @ axes[a]))  # most anti-parallel
    return a, b


def _adjacent_pair(axes_b):
    """Pick a panel + its closest non-self neighbor by axis-cosine."""
    import numpy as np
    axes = axes_b.detach().cpu().numpy() if hasattr(axes_b, "detach") else np.asarray(axes_b)
    a = int(np.argmax(axes @ np.asarray((0.0, 0.0, +1.0), dtype=axes.dtype)))
    cosines = axes @ axes[a]
    cosines[a] = -np.inf
    b = int(np.argmax(cosines))
    return a, b


def _drive_step_and_log(env, sim_app, panel_ids_to_step, target_value, n_steps):
    """Hold all panels at rest_pos; step `panel_ids_to_step` to `target_value`.

    Bypasses `_pre_physics_step` by directly calling
    `robot.set_joint_position_target(...)` so the oracle/contact-push doesn't
    interfere. Returns:
        times_s    (T,)
        positions  (T, P_logged) where P_logged = len(panel_ids_to_step)
        velocities (T, P_logged)
    """
    import numpy as np
    import torch

    # Reset to canonical state first.
    env.reset()
    robot = env.scene["robot"]
    # _pre_physics_step populates _panel_joint_ids_t. Drive one zero-action
    # step so it gets created without disturbing us.
    n_envs = env.num_envs
    device = env.device
    zero_action = torch.zeros((n_envs, 9), device=device)
    env.step(zero_action)
    env.reset()  # back to clean canonical state, joint ids now cached.

    rest = float(env.cfg.rest_pos)
    panel_joint_ids = env._panel_joint_ids       # python list
    panel_joint_ids_t = env._panel_joint_ids_t   # torch.long
    n_panels = panel_joint_ids_t.numel()

    # Build a target tensor: rest_pos for all panels except the stepped ones.
    targets = torch.full(
        (n_envs, n_panels), rest, device=device,
        dtype=robot.data.joint_pos.dtype,
    )
    for pj_idx in panel_ids_to_step:
        # pj_idx is the index INTO the panel-joint list, not the global
        # joint id. We look up the corresponding global joint id below.
        targets[:, pj_idx] = float(target_value)

    times = []
    positions = []
    velocities = []

    sim_dt = float(env.cfg.sim.dt)  # 1/240 s

    # We log AT EVERY PHYSICS STEP, not every env step (env step decimates).
    # The simplest portable way to do this is to call sim.step() directly
    # while overriding the joint target each call.
    for t in range(n_steps):
        # Re-issue the target every physics step so it never drifts.
        robot.set_joint_position_target(
            targets, joint_ids=panel_joint_ids,
        )
        # Step the simulator one physics tick.
        env.sim.step(render=False)
        # Update articulation buffers so joint_pos reflects the new state.
        env.scene.update(dt=sim_dt)

        times.append(t * sim_dt)
        jp = robot.data.joint_pos[0, panel_joint_ids_t]   # (n_panels,)
        jv = robot.data.joint_vel[0, panel_joint_ids_t]
        positions.append(jp[panel_ids_to_step].detach().cpu().numpy().copy())
        velocities.append(jv[panel_ids_to_step].detach().cpu().numpy().copy())

        if not sim_app.is_running():
            break

    return (
        np.asarray(times),
        np.asarray(positions),
        np.asarray(velocities),
    )


# ---------------------------------------------------------------------------
# Tests.
# ---------------------------------------------------------------------------


def test_single_panel_step_response():
    """Step one panel from rest_pos to limit_upper; characterize response."""
    import numpy as np

    out_dir = _ensure_out_dir()
    csv_path = out_dir / "single_panel.csv"

    env, sim_app, cfg = _launch_app_and_env()
    try:
        # First env step populates _axes_b and _panel_joint_ids.
        import torch
        zero = torch.zeros((env.num_envs, 9), device=env.device)
        env.reset()
        env.step(zero)

        rest = float(cfg.rest_pos)
        # Use limit_upper (per-panel stroke); v3 is homogeneous so this is
        # the same scalar for every panel, but read it from the env to
        # stay correct if that ever changes.
        target = float(env._strokes.max().item())

        panel_picks = _select_panels_by_orientation(env._axes_b)

        # 0.5 s @ 240 Hz = 120 physics steps.
        n_steps = 120

        rows = []
        for label, pj_idx in panel_picks.items():
            times, pos, vel = _drive_step_and_log(
                env, sim_app, [pj_idx], target, n_steps,
            )
            metrics = _step_metrics(
                times, pos[:, 0], rest=rest, target=target,
            )
            print(
                f"[w4-actuator] panel={label:6s} idx={pj_idx:2d} "
                f"rise={metrics['rise_time_ms']} settle={metrics['settle_time_ms']} "
                f"overshoot%={metrics['overshoot_pct']} "
                f"peak_v={metrics['peak_vel_mps']}",
                flush=True,
            )
            rows.append({
                "panel_label": label,
                "panel_index": pj_idx,
                "rest_pos_m": rest,
                "target_pos_m": target,
                "n_samples": int(len(times)),
                **metrics,
            })

        # Write CSV.
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=list(rows[0].keys()),
            )
            writer.writeheader()
            writer.writerows(rows)
        print(f"[w4-actuator] wrote {csv_path}", flush=True)

        # Print summary table.
        print("\n[w4-actuator] single-panel step summary:")
        print(f"{'panel':8s} {'rise_ms':>8s} {'settle_ms':>10s} "
              f"{'over%':>6s} {'peak_v_mps':>12s}")
        for r in rows:
            print(
                f"{r['panel_label']:8s} "
                f"{(r['rise_time_ms'] or float('nan')):8.2f} "
                f"{(r['settle_time_ms'] or float('nan')):10.2f} "
                f"{(r['overshoot_pct'] or float('nan')):6.2f} "
                f"{(r['peak_vel_mps'] or float('nan')):12.4f}"
            )
    finally:
        try:
            env.close()
        except Exception:
            pass
        try:
            sim_app.close()
        except Exception:
            pass


def test_paired_panel_step_response():
    """Step two panels (opposite, then adjacent); measure coupling delay."""
    import numpy as np

    out_dir = _ensure_out_dir()
    csv_path = out_dir / "paired_panel.csv"

    env, sim_app, cfg = _launch_app_and_env()
    try:
        import torch
        zero = torch.zeros((env.num_envs, 9), device=env.device)
        env.reset()
        env.step(zero)

        rest = float(cfg.rest_pos)
        target = float(env._strokes.max().item())

        opposite = _diametrically_opposite_pair(env._axes_b)
        adjacent = _adjacent_pair(env._axes_b)

        n_steps = 120
        rows = []
        for label, pair in (("opposite", opposite), ("adjacent", adjacent)):
            times, pos, vel = _drive_step_and_log(
                env, sim_app, list(pair), target, n_steps,
            )
            # Per-panel metrics + coupling delay.
            m_a = _step_metrics(times, pos[:, 0], rest=rest, target=target)
            m_b = _step_metrics(times, pos[:, 1], rest=rest, target=target)
            delay_ms = _coupling_delay_ms(
                times, pos[:, 0], pos[:, 1], rest=rest, target=target,
            )
            print(
                f"[w4-actuator] pair={label:9s} a={pair[0]:2d} b={pair[1]:2d} "
                f"rise_a={m_a['rise_time_ms']} rise_b={m_b['rise_time_ms']} "
                f"delay_ms={delay_ms}",
                flush=True,
            )
            rows.append({
                "pair_label": label,
                "panel_a": pair[0],
                "panel_b": pair[1],
                "rest_pos_m": rest,
                "target_pos_m": target,
                "rise_a_ms": m_a["rise_time_ms"],
                "rise_b_ms": m_b["rise_time_ms"],
                "settle_a_ms": m_a["settle_time_ms"],
                "settle_b_ms": m_b["settle_time_ms"],
                "overshoot_a_pct": m_a["overshoot_pct"],
                "overshoot_b_pct": m_b["overshoot_pct"],
                "peak_vel_a_mps": m_a["peak_vel_mps"],
                "peak_vel_b_mps": m_b["peak_vel_mps"],
                "coupling_delay_ms": delay_ms,
            })

        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"[w4-actuator] wrote {csv_path}", flush=True)

        print("\n[w4-actuator] paired-panel step summary:")
        print(f"{'pair':10s} {'delay_ms':>10s} {'rise_a':>8s} {'rise_b':>8s}")
        for r in rows:
            print(
                f"{r['pair_label']:10s} "
                f"{(r['coupling_delay_ms'] or float('nan')):10.2f} "
                f"{(r['rise_a_ms'] or float('nan')):8.2f} "
                f"{(r['rise_b_ms'] or float('nan')):8.2f}"
            )
    finally:
        try:
            env.close()
        except Exception:
            pass
        try:
            sim_app.close()
        except Exception:
            pass
