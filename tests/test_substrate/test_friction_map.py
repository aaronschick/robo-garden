"""W4 — Friction-map identification (urchin_v3, sim-only).

Two tests:

  test_coast_down_flat
      Initialize urchin at fixed angular velocity (5 rad/s) on flat ground via
      the canonical reset path. Command panels to hold rest_pos (zero-motion
      oracle). Log body angular velocity vs time for 5 s. Fit exponential
      decay tau and compute a rolling-resistance coefficient.

  test_incline_static_slip_onset
      Place urchin on flat ground (we tilt the world gravity vector instead
      of authoring an inclined-plane USD asset; this is dynamically equivalent
      to a tilt while keeping the scene asset graph untouched). Ramp the
      effective incline angle from 0 deg -> 30 deg over 10 s with panels held
      at rest_pos. Detect the onset of slipping (body translation exceeds a
      threshold) -> effective static mu_s.

Outputs:
  workspace/_tasks_out/w4_substrate_id/friction/coast_down.csv
  workspace/_tasks_out/w4_substrate_id/friction/incline_slip.csv

Both tests use `URCHIN_RESET_MODE=canonical` for clean state and reuse the
diagnostics helpers from `workspace/robots/urchin_v3/scripts/diagnostics.py`
(specifically `project_omega_roll`) so we never duplicate the metric math.

Top-level imports are pytest + stdlib only. Isaac Lab / urchin imports happen
inside the test bodies, behind the `_has_isaaclab` skip guard, so Windows
pytest collection does not fail.

Run (WSL):
    cd /mnt/c/Users/aaron/Documents/repositories/robo-garden && \
      URCHIN_RESET_MODE=canonical \
      C:/isaac-venv/Scripts/python.exe -m pytest \
        tests/test_substrate/test_friction_map.py -x -s
"""
from __future__ import annotations

import csv
import math
import os
import sys
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Capability detection.
# ---------------------------------------------------------------------------


def _has_isaaclab() -> bool:
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
_OUT_DIR = _REPO_ROOT / "workspace" / "_tasks_out" / "w4_substrate_id" / "friction"


def _ensure_out_dir() -> Path:
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    return _OUT_DIR


# ---------------------------------------------------------------------------
# Analytics (pure numpy).
# ---------------------------------------------------------------------------


def _fit_exp_decay_tau(times_s, values):
    """Fit |v(t)| = |v0| * exp(-t/tau) by least-squares on log|v|.

    Returns (tau_s, v0_fit) or (None, None) if too few finite samples.
    """
    import numpy as np

    times_s = np.asarray(times_s, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    abs_v = np.abs(values)
    mask = abs_v > 1e-4   # ignore near-zero tail (log noise blows up)
    if mask.sum() < 5:
        return None, None
    t = times_s[mask]
    log_v = np.log(abs_v[mask])
    # log|v| = log|v0| - t/tau   (linear in t; slope = -1/tau).
    A = np.vstack([t, np.ones_like(t)]).T
    sol, *_ = np.linalg.lstsq(A, log_v, rcond=None)
    slope, intercept = float(sol[0]), float(sol[1])
    if slope >= 0:
        return None, None
    tau = -1.0 / slope
    v0 = math.exp(intercept)
    return tau, v0


def _rolling_resistance_coef(tau_s, radius_m, g_m_s2: float = 9.81):
    """Approximate rolling-resistance coefficient from coast-down tau.

    Model: pure-rolling sphere on flat ground, only loss term is rolling
    resistance with linear-velocity-proportional torque
        I * dw/dt = -mu_r * (M*g) * R
    => exponential decay only if we approximate the resistance as -c*w with
       c = (mu_r * M*g*R) / I. tau = I / (mu_r * M*g*R).
    For a homogeneous sphere I = 2/5 M R^2 -> tau = (2/5 R) / (mu_r * g).
    => mu_r = (2/5 R) / (tau * g) = 0.4 R / (tau * g).

    This is a coarse first-order proxy — the real PhysX rolling resistance
    is sliding-friction-coupled and does NOT decay exactly exponentially.
    We report the fit and let downstream W6/W7 treat it as an order-of-
    magnitude anchor, not a calibrated parameter.
    """
    if tau_s is None or tau_s <= 0:
        return None
    return 0.4 * radius_m / (tau_s * g_m_s2)


# ---------------------------------------------------------------------------
# Driver helpers.
# ---------------------------------------------------------------------------


def _launch_app_and_env(*, episode_length_s: float = 12.0):
    """Launch AppLauncher + UrchinEnv in canonical-reset mode.

    Returns (env, sim_app, cfg).
    """
    os.environ["URCHIN_RESET_MODE"] = "canonical"

    from isaaclab.app import AppLauncher
    import argparse
    parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args([])
    args.headless = True
    args.enable_cameras = False
    app_launcher = AppLauncher(args)
    sim_app = app_launcher.app

    import isaaclab as _il
    _src = Path(_il.__file__).parent / "source" / "isaaclab"
    if _src.exists() and str(_src) not in sys.path:
        sys.path.insert(0, str(_src))

    workspace_root = _REPO_ROOT / "workspace" / "robots" / "urchin_v3"
    if str(workspace_root) not in sys.path:
        sys.path.insert(0, str(workspace_root))
    from urchin_v3.urchin_env_cfg import UrchinEnv, UrchinEnvCfg  # noqa: E402

    cfg = UrchinEnvCfg()
    cfg.scene.num_envs = 1
    cfg.episode_length_s = episode_length_s
    cfg.reset_mode = "canonical"
    cfg.progress_reward_weight = 0.0
    cfg.distance_penalty_weight = 0.0
    cfg.goal_bonus = 0.0
    cfg.wall_contact_penalty = 0.0
    # Also park the oracle so it can't fight our hold-rest commands.
    cfg.oracle_amplitude = 0.0

    env = UrchinEnv(cfg, render_mode=None)
    return env, sim_app, cfg


def _hold_rest_one_step(env):
    """Drive panels to rest_pos for one env step using the SH=0 action.

    With oracle_amplitude=0 the env's _pre_physics_step decodes residual=0
    and the smoothed target relaxes to rest_pos. This matches the "zero
    motion oracle" the W4 spec prescribes.
    """
    import torch
    zero = torch.zeros((env.num_envs, 9), device=env.device)
    return env.step(zero)


# ---------------------------------------------------------------------------
# Tests.
# ---------------------------------------------------------------------------


def test_coast_down_flat():
    """Spin the body at 5 rad/s, hold panels still, log angular-vel decay."""
    import numpy as np
    import torch

    out_dir = _ensure_out_dir()
    csv_path = out_dir / "coast_down.csv"

    env, sim_app, cfg = _launch_app_and_env(episode_length_s=8.0)
    try:
        # First reset + zero step so panel ids / SH basis are populated.
        env.reset()
        _ = _hold_rest_one_step(env)
        env.reset()

        robot = env.scene["robot"]

        # Reuse diagnostics.project_omega_roll for the rolling-axis projection.
        scripts_dir = (
            _REPO_ROOT / "workspace" / "robots" / "urchin_v3" / "scripts"
        )
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))
        from diagnostics import project_omega_roll  # noqa: E402

        # Initial conditions: spawn at canonical state, then overwrite the
        # body angular velocity directly via PhysX. Roll axis = body x in
        # world frame at yaw=0 cos/sin -> default identity quat means body
        # x is world x; angular_vel set on world y gives forward roll.
        n = env.num_envs
        device = env.device
        omega0 = 5.0   # rad/s

        # canonical reset already happened. Now write a non-zero ang vel:
        ang_vel = torch.zeros((n, 3), device=device)
        ang_vel[:, 1] = omega0   # world-y rotation = forward roll about y.
        lin_vel = torch.zeros((n, 3), device=device)
        # write_root_com_velocity_to_sim expects (lin (3), ang (3)) = (6,).
        vel6 = torch.cat([lin_vel, ang_vel], dim=-1)
        robot.write_root_com_velocity_to_sim(vel6)

        # Travel direction in world frame: +x (since we spun about +y).
        travel_dir_w = torch.tensor(
            [[1.0, 0.0, 0.0]] * n, device=device, dtype=torch.float32,
        )

        # Sample every env step (decimation=4 => 60 Hz). 5 s total = 300 samples.
        n_env_steps = 300
        sample_dt = float(cfg.sim.dt) * cfg.decimation   # 1/60 s
        times = []
        omega_world_y = []
        omega_roll_axis = []

        for k in range(n_env_steps):
            _ = _hold_rest_one_step(env)
            ang_w = robot.data.root_ang_vel_w[0]   # (3,)
            times.append(k * sample_dt)
            omega_world_y.append(float(ang_w[1].item()))

            # Project onto the rolling axis (z x travel_dir).
            wr = project_omega_roll(
                ang_w.unsqueeze(0),
                travel_dir_w[:1],
            )
            omega_roll_axis.append(float(wr[0].item()))

            if not sim_app.is_running():
                break

        times = np.asarray(times)
        omega_world_y = np.asarray(omega_world_y)
        omega_roll_axis = np.asarray(omega_roll_axis)

        tau_s, v0_fit = _fit_exp_decay_tau(times, omega_world_y)
        # Effective sphere radius from the urchin docstring: shell sits at
        # z ~ 0.149 m (panel rest_pos + base). 0.17 z-spawn - 0.010 rest =
        # 0.16 m approximate.
        sphere_radius_m = 0.16
        mu_r = _rolling_resistance_coef(tau_s, sphere_radius_m)

        print(
            f"[w4-friction] coast-down: omega0={omega0:.2f} rad/s "
            f"tau={tau_s} v0_fit={v0_fit} mu_r={mu_r}",
            flush=True,
        )

        # Write per-step trace + summary.
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(
                ["t_s", "omega_world_y_radps", "omega_roll_axis_radps"],
            )
            for t, w, wr in zip(times, omega_world_y, omega_roll_axis):
                writer.writerow([f"{t:.6f}", f"{w:.6f}", f"{wr:.6f}"])
        # Side-car summary CSV.
        summary_path = csv_path.with_suffix(".summary.csv")
        with summary_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=[
                    "omega0_radps", "tau_s", "v0_fit_radps",
                    "rolling_resistance_coef", "sphere_radius_m_assumed",
                    "n_samples",
                ],
            )
            writer.writeheader()
            writer.writerow({
                "omega0_radps": omega0,
                "tau_s": tau_s,
                "v0_fit_radps": v0_fit,
                "rolling_resistance_coef": mu_r,
                "sphere_radius_m_assumed": sphere_radius_m,
                "n_samples": int(times.shape[0]),
            })
        print(f"[w4-friction] wrote {csv_path} + {summary_path}", flush=True)
    finally:
        try:
            env.close()
        except Exception:
            pass
        try:
            sim_app.close()
        except Exception:
            pass


def test_incline_static_slip_onset():
    """Ramp gravity tilt from 0 deg -> 30 deg; find slip-onset angle.

    We tilt the gravity vector instead of authoring an inclined-plane USD
    asset. Mechanically equivalent for static-slip onset: the body sees a
    tangential gravity component F_t = M*g*sin(theta), restrained by
    F_n*mu_s = M*g*cos(theta)*mu_s. Slip onset -> tan(theta) = mu_s.
    """
    import numpy as np
    import torch

    out_dir = _ensure_out_dir()
    csv_path = out_dir / "incline_slip.csv"

    env, sim_app, cfg = _launch_app_and_env(episode_length_s=12.0)
    try:
        env.reset()
        _ = _hold_rest_one_step(env)
        env.reset()

        robot = env.scene["robot"]
        n = env.num_envs
        device = env.device

        # Sample every env step (60 Hz). 10 s ramp + 1 s settle = 11 s = 660 steps.
        ramp_duration_s = 10.0
        total_duration_s = 11.0
        sample_dt = float(cfg.sim.dt) * cfg.decimation
        n_env_steps = int(round(total_duration_s / sample_dt))

        # Initial spawn xy.
        spawn_xy0 = robot.data.root_pos_w[0, :2].detach().cpu().numpy().copy()

        # Slip detection threshold: 5 cm horizontal translation away from spawn.
        slip_threshold_m = 0.05

        times = []
        incline_deg = []
        translation_m = []
        slip_onset_deg = None

        for k in range(n_env_steps):
            t = k * sample_dt
            theta_deg = min(30.0, 30.0 * (t / ramp_duration_s))
            theta_rad = math.radians(theta_deg)

            # Tilt gravity. Default world gravity is (0,0,-g). We rotate
            # about world-y so gravity gains a -x component:
            #   g_world = (-g*sin, 0, -g*cos)
            # The robot is a free body so this is dynamically identical to
            # placing it on a +x-down slope.
            g_mag = 9.81
            g_vec = (-g_mag * math.sin(theta_rad), 0.0, -g_mag * math.cos(theta_rad))
            try:
                env.sim.set_gravity(g_vec)
            except Exception:
                # Fallback: some Isaac Lab versions expose set_gravity on
                # the underlying physics context instead.
                try:
                    env.sim.physics_sim_view.set_gravity(g_vec)
                except Exception:
                    pass

            _ = _hold_rest_one_step(env)
            pos_xy = robot.data.root_pos_w[0, :2].detach().cpu().numpy()
            d_xy = float(np.linalg.norm(pos_xy - spawn_xy0))

            times.append(t)
            incline_deg.append(theta_deg)
            translation_m.append(d_xy)

            if slip_onset_deg is None and d_xy > slip_threshold_m:
                slip_onset_deg = theta_deg
                print(
                    f"[w4-friction] slip onset: theta={theta_deg:.2f} deg "
                    f"d_xy={d_xy*1000:.1f} mm at t={t:.2f}s",
                    flush=True,
                )

            if not sim_app.is_running():
                break

        # Restore world gravity in case other tests follow.
        try:
            env.sim.set_gravity((0.0, 0.0, -9.81))
        except Exception:
            pass

        # mu_s = tan(theta_onset).  None if never slipped within 30 deg.
        mu_s = math.tan(math.radians(slip_onset_deg)) if slip_onset_deg else None

        print(
            f"[w4-friction] incline: onset_deg={slip_onset_deg} mu_s={mu_s}",
            flush=True,
        )

        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["t_s", "incline_deg", "translation_m"])
            for t, deg, d in zip(times, incline_deg, translation_m):
                writer.writerow([f"{t:.6f}", f"{deg:.6f}", f"{d:.6f}"])
        summary_path = csv_path.with_suffix(".summary.csv")
        with summary_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=[
                    "slip_onset_deg", "static_mu_s", "slip_threshold_m",
                    "ramp_duration_s", "n_samples",
                ],
            )
            writer.writeheader()
            writer.writerow({
                "slip_onset_deg": slip_onset_deg,
                "static_mu_s": mu_s,
                "slip_threshold_m": slip_threshold_m,
                "ramp_duration_s": ramp_duration_s,
                "n_samples": int(len(times)),
            })
        print(f"[w4-friction] wrote {csv_path} + {summary_path}", flush=True)
    finally:
        try:
            env.close()
        except Exception:
            pass
        try:
            sim_app.close()
        except Exception:
            pass
