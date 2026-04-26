"""W7 closed-loop controller smoke test (no Isaac, pure torch).

Walks the W6 atlas sample 17 parameter set through the engine for 4 s
at 60 Hz with synthetic diagnostics (slip cycling 0 -> 0.4 sinusoidally,
support_asym oscillating +/-5), once with the W7 closed-loop features
OFF and once with them ON. Reports the rear-push channel amplitude
difference as the observable signal that the slip-aware scheduler and
event-gated phase reset are doing their job.

Run:
    uv run python workspace/scratch/w7_smoke.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

# Engine lives under workspace/robots/urchin_v3/scripts/ (not an
# installed package); shim sys.path before importing.
_REPO = Path(__file__).resolve().parents[2]
_URCHIN_SCRIPTS = _REPO / "workspace" / "robots" / "urchin_v3" / "scripts"
if str(_URCHIN_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_URCHIN_SCRIPTS))

import rolling_engine as re  # noqa: E402


# --- W6 sample 17 (best of 256 LHS samples; see project memory) ----------
W6_SAMPLE_17 = dict(
    rear_push_amp=0.35,
    front_retract_amp=0.58,
    lean_phase=0.06 * 2.0 * math.pi,    # 0.06 cycles -> radians
    phase_velocity_hz=1.12,
    push_duty=0.39,
)


def _fibonacci_normals(n: int = 42, seed: int = 0) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    phi = math.pi * (3.0 - math.sqrt(5.0))
    idx = torch.arange(n, dtype=torch.float64)
    z = 1.0 - (idx / (n - 1)) * 2.0
    r = torch.sqrt(torch.clamp(1.0 - z * z, min=0.0))
    theta = phi * idx
    x = r * torch.cos(theta)
    y = r * torch.sin(theta)
    normals = torch.stack([x, y, z], dim=-1).to(torch.float32)
    jitter = 0.001 * (torch.rand((n, 3), generator=gen) - 0.5)
    normals = (normals + jitter)
    normals = normals / normals.norm(dim=-1, keepdim=True)
    return normals


def _build_rear_mask(axes_b: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
    """Boolean mask over the 42 panels marking the rear-bottom region.
    Used to summarise the rear-push channel amplitude.
    """
    grav = torch.tensor([0.0, 0.0, -1.0])
    down_b = grav / grav.norm()
    goal3 = torch.cat([goal[0], torch.zeros(1)])
    forward_b = goal3 - (goal3 * down_b).sum() * down_b
    forward_b = forward_b / forward_b.norm()
    d = (axes_b * down_b).sum(-1)
    f = (axes_b * forward_b).sum(-1)
    return (d > 0.4) & (f < -0.4)


def _run(*, w7_on: bool, num_steps: int, dt: float):
    """Roll the engine for `num_steps` and return per-step rear-push
    summaries plus the full per-step rear-channel amplitude trace.
    """
    cfg_kwargs = dict(W6_SAMPLE_17)
    if w7_on:
        cfg_kwargs.update(
            event_phase_reset=True,
            slip_aware_scheduler=True,
            # Tighten thresholds slightly so the synthetic diagnostics
            # actually trigger within the 4 s window.
            phase_reset_support_asym_threshold=4.0,
            slip_scale_low_threshold=0.15,
            slip_scale_high_threshold=0.30,
            slip_scale_min=0.5,
        )
    cfg = re.RollingEngineCfg(**cfg_kwargs)

    axes_b = _fibonacci_normals(42, seed=1)
    grav = torch.tensor([[0.0, 0.0, -1.0]])
    goal = torch.tensor([[1.0, 0.0]])
    rear_mask = _build_rear_mask(axes_b, goal)

    state = re.new_state(num_envs=1, phase_init=0.0)

    rear_amplitude = []
    reset_count = 0
    last_reset_seen = state.last_reset_phase.clone()

    for step in range(num_steps):
        # Synthetic telemetry: slip = 0.2 + 0.2*sin(2*pi*0.5*t) ranges
        # [0.0, 0.4] over the 4 s window. support_asym = 5*sin(2*pi*1*t)
        # oscillates +/- 5 (crosses the +/-4 threshold each cycle).
        t = step * dt
        slip = 0.2 + 0.2 * math.sin(2 * math.pi * 0.5 * t)
        support_asym = 5.0 * math.sin(2 * math.pi * 1.0 * t)
        diagnostics = {
            "slip_ratio": torch.tensor([slip], dtype=torch.float32),
            "support_asym": torch.tensor([support_asym], dtype=torch.float32),
        }

        out = re.forward(
            axes_b=axes_b, projected_gravity_b=grav, to_goal_b=goal,
            state=state, cfg=cfg, diagnostics=diagnostics,
        )                                                                  # (1, 42)
        rear_amplitude.append(out[0, rear_mask].abs().mean().item())

        # Detect that a reset fired by checking last_reset_phase change.
        if not torch.equal(state.last_reset_phase, last_reset_seen):
            reset_count += 1
            last_reset_seen = state.last_reset_phase.clone()

        state = re.advance_phase(state, cfg, dt)

    return torch.tensor(rear_amplitude), reset_count


def main() -> int:
    dt = 1.0 / 60.0
    num_steps = 240  # 4 s

    rear_off, resets_off = _run(w7_on=False, num_steps=num_steps, dt=dt)
    rear_on, resets_on = _run(w7_on=True, num_steps=num_steps, dt=dt)

    diff = (rear_on - rear_off).abs()
    mean_off = rear_off.mean().item()
    mean_on = rear_on.mean().item()
    mean_diff = diff.mean().item()
    max_diff = diff.max().item()

    print("W7 smoke test (W6 sample 17, 4 s @ 60 Hz, synthetic diagnostics)")
    print("-" * 64)
    print(f"  Steps                : {num_steps}")
    print(f"  Slip range           : [0.0, 0.4]")
    print(f"  Support_asym range   : [-5.0, +5.0]")
    print()
    print(f"  Rear-push amp (W7=OFF)  mean: {mean_off:.4f}")
    print(f"  Rear-push amp (W7=ON)   mean: {mean_on:.4f}")
    print(f"  |delta| mean / max         : {mean_diff:.4f} / {max_diff:.4f}")
    print(f"  Phase resets fired (W7=OFF): {resets_off}")
    print(f"  Phase resets fired (W7=ON) : {resets_on}")

    # Validation: closed-loop modulation should leave a visible footprint.
    if max_diff < 1e-4:
        print()
        print("FAIL: W7 OFF vs ON produced no observable difference in rear push.")
        return 1
    if resets_on == 0:
        print()
        print("FAIL: event_phase_reset never fired despite engineered support_asym.")
        return 1
    if resets_off != 0:
        print()
        print("FAIL: phase reset fired with event_phase_reset=False (regression).")
        return 1

    print()
    print("PASS: W7 closed-loop modulation visibly affects rear-push channel.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
