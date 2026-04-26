"""Tests for workspace/robots/urchin_v3/scripts/diagnostics.py (W2 telemetry).

Pure torch — no Isaac Lab. The diagnostics module lives under
workspace/robots/urchin_v3/scripts/ which is not an installed package, so
we add it to sys.path up-front (mirrors test_rolling_engine.py).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

_URCHIN_SCRIPTS = (
    Path(__file__).resolve().parents[2]
    / "workspace" / "robots" / "urchin_v3" / "scripts"
)
if str(_URCHIN_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_URCHIN_SCRIPTS))

import diagnostics as diag  # noqa: E402
from scripted_roll import build_sh_basis  # noqa: E402


# ---------------------------------------------------------------------------
# slip_ratio
# ---------------------------------------------------------------------------


def test_slip_ratio_pure_translation():
    """v=1, omega_roll=0, R=0.1 -> s = (0 - 1) / max(0, 1, eps) = -1.

    Pure forward slide with no spin: the wheel translates without rolling.
    """
    omega = torch.tensor([0.0])
    v = torch.tensor([1.0])
    s = diag.slip_ratio(omega, v, radius=0.1)
    assert torch.allclose(s, torch.tensor([-1.0]), atol=1e-6)


def test_slip_ratio_pure_roll():
    """omega_roll = v / R: pure rolling, s = 0."""
    R = 0.17
    v = torch.tensor([0.5])
    omega = v / R
    s = diag.slip_ratio(omega, v, radius=R)
    assert torch.allclose(s, torch.tensor([0.0]), atol=1e-6)


def test_slip_ratio_pure_spin():
    """v=0, omega_roll=10 -> s = (R*10 - 0) / max(R*10, 0, eps) = +1."""
    omega = torch.tensor([10.0])
    v = torch.tensor([0.0])
    s = diag.slip_ratio(omega, v, radius=0.1)
    assert torch.allclose(s, torch.tensor([1.0]), atol=1e-6)


def test_slip_ratio_zero_zero_safe():
    """v=0, omega=0: clamped denominator avoids div-by-zero. Result ~0."""
    omega = torch.tensor([0.0])
    v = torch.tensor([0.0])
    s = diag.slip_ratio(omega, v, radius=0.1, eps=1e-6)
    assert torch.isfinite(s).all()
    assert s.abs().item() < 1e-3


def test_slip_ratio_batched():
    """Batch dim flows through cleanly."""
    omega = torch.tensor([0.0, 5.0, 0.0])
    v = torch.tensor([1.0, 0.5, 0.0])
    s = diag.slip_ratio(omega, v, radius=0.1)
    # env 0: pure slide -> -1; env 1: pure roll (R*omega = 0.5 = v) -> 0;
    # env 2: rest -> ~0.
    assert torch.allclose(s[:2], torch.tensor([-1.0, 0.0]), atol=1e-6)
    assert s[2].abs().item() < 1e-3


# ---------------------------------------------------------------------------
# cost_of_transport
# ---------------------------------------------------------------------------


def test_cot_zero_velocity():
    """v=0 -> COT = 0 (convention: standstill has no transport to amortize)."""
    effort = torch.ones((1, 42)) * 5.0
    qdot = torch.ones((1, 42)) * 0.1
    v = torch.tensor([0.0])
    cot = diag.cost_of_transport(effort, qdot, v)
    assert torch.allclose(cot, torch.tensor([0.0]))
    assert torch.isfinite(cot).all()


def test_cot_nonzero_velocity():
    """Finite, positive COT under finite velocity."""
    effort = torch.ones((2, 42)) * 5.0
    qdot = torch.ones((2, 42)) * 0.1
    v = torch.tensor([1.0, 0.5])
    cot = diag.cost_of_transport(effort, qdot, v, mass=2.52, g=9.81)
    # power = 42 * 5 * 0.1 = 21 W; cot = 21 / (2.52 * 9.81 * v)
    expected = torch.tensor([21.0 / (2.52 * 9.81 * 1.0),
                             21.0 / (2.52 * 9.81 * 0.5)])
    assert torch.allclose(cot, expected, atol=1e-5)


# ---------------------------------------------------------------------------
# support_asymmetry
# ---------------------------------------------------------------------------


def test_support_asym_sign():
    """Front-loaded -> negative; rear-loaded -> positive."""
    # 4 panels: 2 front (+x), 2 rear (-x).
    panel_axes = torch.tensor([[
        [+1.0, 0.0, 0.0],
        [+1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
    ]])  # (1, 4, 3)
    travel = torch.tensor([[1.0, 0.0, 0.0]])  # (1, 3) along +x

    # Front-loaded: front forces big, rear small.
    front_heavy = torch.tensor([[5.0, 5.0, 1.0, 1.0]])
    sa_front = diag.support_asymmetry(front_heavy, panel_axes, travel)
    assert sa_front.item() < 0  # rear - front = (1+1) - (5+5) = -8

    # Rear-loaded: rear forces big, front small.
    rear_heavy = torch.tensor([[1.0, 1.0, 5.0, 5.0]])
    sa_rear = diag.support_asymmetry(rear_heavy, panel_axes, travel)
    assert sa_rear.item() > 0  # rear - front = 8


def test_support_asym_balanced_zero():
    panel_axes = torch.tensor([[
        [+1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
    ]])
    travel = torch.tensor([[1.0, 0.0, 0.0]])
    forces = torch.tensor([[3.0, 3.0]])
    sa = diag.support_asymmetry(forces, panel_axes, travel)
    assert torch.allclose(sa, torch.tensor([0.0]))


# ---------------------------------------------------------------------------
# sh_puck_mode
# ---------------------------------------------------------------------------


def _fibonacci_normals(n: int = 42) -> torch.Tensor:
    """Approximately-uniform unit vectors on a sphere (Fibonacci lattice)."""
    indices = torch.arange(0, n, dtype=torch.float32) + 0.5
    phi = torch.acos(1.0 - 2.0 * indices / n)
    theta = math.pi * (1.0 + math.sqrt(5.0)) * indices
    x = torch.sin(phi) * torch.cos(theta)
    y = torch.sin(phi) * torch.sin(theta)
    z = torch.cos(phi)
    return torch.stack([x, y, z], dim=-1)


def test_sh_a20_flat_disc():
    """Equatorial extension, polar retraction => high a_20 / E_2 (puck)."""
    axes = _fibonacci_normals(42)  # (42, 3)
    B = build_sh_basis(axes)       # (42, 9)

    # Synthetic panel state: extension proportional to (1 - 3*n_z^2).
    # That is by construction the *negative* of the a_20 basis column up
    # to RMS normalisation: the projection onto col 6 dominates while
    # cols 4, 5, 7, 8 stay near zero.
    nz = axes[:, 2]
    panel_state = (1.0 - 3.0 * nz * nz).unsqueeze(0)  # (1, 42)

    p = diag.sh_puck_mode(panel_state, B)
    assert p.item() > 0.99  # |a_20| dominates E_2


def test_sh_a20_uniform_extension():
    """All panels equally extended => no quadrupole content (low ratio).

    The DC term (col 0) absorbs uniform offsets; l=2 columns (4-8) stay
    near zero, so |a_20| / (E_2 + eps) ~ small.
    """
    axes = _fibonacci_normals(42)
    B = build_sh_basis(axes)
    panel_state = torch.full((1, 42), 0.05)
    p = diag.sh_puck_mode(panel_state, B)
    # E_2 should be near-zero under uniform input; a_20 also near-zero.
    # The ratio is bounded by eps-clamping; effectively undefined but
    # required to be small (< 0.5 — solid puck mode is > 0.99).
    assert p.item() < 0.5


# ---------------------------------------------------------------------------
# project_omega_roll (sanity)
# ---------------------------------------------------------------------------


def test_project_omega_roll_aligned():
    """ω along (z × travel) => projection equals |ω|."""
    travel = torch.tensor([[1.0, 0.0, 0.0]])  # +x
    # roll axis = z × x = +y. ω = (0, 5, 0) => projection = +5.
    omega = torch.tensor([[0.0, 5.0, 0.0]])
    proj = diag.project_omega_roll(omega, travel)
    assert torch.allclose(proj, torch.tensor([5.0]), atol=1e-6)


def test_project_omega_roll_orthogonal():
    """ω along travel direction => zero rolling-axis projection."""
    travel = torch.tensor([[1.0, 0.0, 0.0]])
    omega = torch.tensor([[3.0, 0.0, 0.0]])  # along +x, parallel to travel
    proj = diag.project_omega_roll(omega, travel)
    assert proj.abs().item() < 1e-6
