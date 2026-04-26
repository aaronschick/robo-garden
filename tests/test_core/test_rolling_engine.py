"""Tests for workspace/robots/urchin_v3/scripts/rolling_engine.py.

Purpose:
  1. Pin the regression guarantee -- at default cfg, the engine reproduces
     compute_contactpush_oracle(amplitude=1.0) within float eps. Any change
     to the oracle math or engine defaults must make a deliberate decision
     about this invariant.
  2. Determinism -- same inputs + state produce identical outputs across
     repeated calls.
  3. Steering bias -- non-zero bias rotates the effective forward axis in
     the expected direction (checked via mirror symmetry between +bias and
     -bias on a paired pair of panels).
  4. Phase envelope -- duty_cycle=1.0 is a literal identity, duty_cycle<1.0
     actually depends on phase.
  5. Guardrails -- populating style_params raises loudly instead of doing
     nothing silently.

Pure torch, no Isaac Lab. The engine and oracle both live under
workspace/robots/urchin_v3/scripts/, which is not an installed package; we
add it to sys.path up-front.
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

import rolling_engine as re  # noqa: E402
from scripted_roll import compute_contactpush_oracle  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _fibonacci_normals(n: int = 42, *, seed: int = 0) -> torch.Tensor:
    """Generate n approximately-uniform unit vectors on the sphere.

    Deterministic: same seed -> identical normals. Used to stand in for the
    real urchin panel axes without depending on the robot asset loader.
    """
    gen = torch.Generator().manual_seed(seed)
    phi = math.pi * (3.0 - math.sqrt(5.0))                       # golden angle
    idx = torch.arange(n, dtype=torch.float64)
    z = 1.0 - (idx / (n - 1)) * 2.0                              # [+1, -1]
    r = torch.sqrt(torch.clamp(1.0 - z * z, min=0.0))
    theta = phi * idx
    x = r * torch.cos(theta)
    y = r * torch.sin(theta)
    normals = torch.stack([x, y, z], dim=-1).to(torch.float32)
    # Tiny jitter to avoid degenerate (exactly-axial) panels that would
    # land on tanh/sigmoid inflection points and make floating-point
    # comparisons noisy.
    jitter = 0.001 * (torch.rand((n, 3), generator=gen) - 0.5)
    normals = normals + jitter
    normals = normals / normals.norm(dim=-1, keepdim=True)
    return normals


def _default_inputs(num_envs: int = 4):
    """Geometry + intent inputs for an easy straight-ahead scenario."""
    axes_b = _fibonacci_normals(42, seed=1)
    grav = torch.tensor([[0.0, 0.0, -1.0]]).expand(num_envs, 3).contiguous()
    goal = torch.tensor([[1.0, 0.0]]).expand(num_envs, 2).contiguous()
    return axes_b, grav, goal


# ---------------------------------------------------------------------------
# 1. Regression: engine defaults == compute_contactpush_oracle
# ---------------------------------------------------------------------------


def test_defaults_match_oracle_exactly():
    axes_b, grav, goal = _default_inputs(num_envs=4)
    cfg = re.RollingEngineCfg()
    state = re.new_state(num_envs=4)

    engine_out = re.forward(
        axes_b=axes_b, projected_gravity_b=grav, to_goal_b=goal,
        state=state, cfg=cfg,
    )
    oracle_out = compute_contactpush_oracle(
        axes_b=axes_b, projected_gravity_b=grav, to_goal_b=goal,
        amplitude=1.0,
    )
    # The only numerical difference is the `1.0 * envelope * contact_term`
    # extra multiply where envelope==1.0. IEEE 754 guarantees this is a
    # no-op, so allclose with tight tolerance should pass.
    torch.testing.assert_close(engine_out, oracle_out, rtol=1e-6, atol=1e-7)


def test_defaults_match_oracle_independent_of_phase():
    """duty_cycle=1.0 makes the envelope identically 1 regardless of phase."""
    axes_b, grav, goal = _default_inputs(num_envs=4)
    cfg = re.RollingEngineCfg()
    oracle_out = compute_contactpush_oracle(
        axes_b=axes_b, projected_gravity_b=grav, to_goal_b=goal,
        amplitude=1.0,
    )
    for phase_val in (0.0, math.pi / 2, math.pi, 1.337):
        state = re.new_state(num_envs=4, phase_init=phase_val)
        engine_out = re.forward(
            axes_b=axes_b, projected_gravity_b=grav, to_goal_b=goal,
            state=state, cfg=cfg,
        )
        torch.testing.assert_close(
            engine_out, oracle_out, rtol=1e-6, atol=1e-7,
            msg=f"phase={phase_val} should not affect output at duty_cycle=1.0",
        )


def test_defaults_match_oracle_at_nontrivial_grav_and_goal():
    """Regression parity should hold for arbitrary (non-axis-aligned) inputs."""
    axes_b = _fibonacci_normals(42, seed=2)
    # Gravity tilted from world-down, goal not axis-aligned.
    grav = torch.tensor([
        [0.1, -0.05, -0.98],
        [-0.2, 0.3, -0.9],
    ])
    goal = torch.tensor([
        [0.7, -0.3],
        [-0.5, 0.8],
    ])
    cfg = re.RollingEngineCfg()
    state = re.new_state(num_envs=2)

    engine_out = re.forward(
        axes_b=axes_b, projected_gravity_b=grav, to_goal_b=goal,
        state=state, cfg=cfg,
    )
    oracle_out = compute_contactpush_oracle(
        axes_b=axes_b, projected_gravity_b=grav, to_goal_b=goal,
        amplitude=1.0,
    )
    torch.testing.assert_close(engine_out, oracle_out, rtol=1e-6, atol=1e-7)


# ---------------------------------------------------------------------------
# 2. Determinism
# ---------------------------------------------------------------------------


def test_determinism_across_repeated_calls():
    axes_b, grav, goal = _default_inputs(num_envs=4)
    cfg = re.RollingEngineCfg(
        duty_cycle=0.5, phase_velocity_hz=1.0, steering_bias=0.3,
    )
    state = re.new_state(num_envs=4, phase_init=0.7)

    out_a = re.forward(
        axes_b=axes_b, projected_gravity_b=grav, to_goal_b=goal,
        state=state, cfg=cfg,
    )
    out_b = re.forward(
        axes_b=axes_b, projected_gravity_b=grav, to_goal_b=goal,
        state=state, cfg=cfg,
    )
    # Pure function + same inputs -> bit-identical outputs.
    assert torch.equal(out_a, out_b)


def test_advance_phase_returns_new_state_at_expected_offset():
    cfg = re.RollingEngineCfg(phase_velocity_hz=2.0)
    state = re.new_state(num_envs=4, phase_init=0.0)
    dt = 0.25
    # Expected advance: 2*pi * 2.0 * 0.25 = pi.
    new_state = re.advance_phase(state, cfg, dt)
    expected = torch.full((4,), math.pi)
    torch.testing.assert_close(new_state.phase, expected, rtol=0, atol=1e-5)
    # Original state is untouched (pure-function invariant).
    torch.testing.assert_close(
        state.phase, torch.zeros(4), rtol=0, atol=0,
    )


def test_advance_phase_wraps_to_2pi():
    cfg = re.RollingEngineCfg(phase_velocity_hz=10.0)
    state = re.new_state(num_envs=2, phase_init=0.0)
    # One second at 10 Hz => 20*pi radians; should wrap into [0, 2pi).
    state = re.advance_phase(state, cfg, dt=1.0)
    assert torch.all(state.phase >= 0.0)
    assert torch.all(state.phase < 2.0 * math.pi + 1e-5)


# ---------------------------------------------------------------------------
# 3. Steering bias: mirror symmetry and actual effect
# ---------------------------------------------------------------------------


def test_steering_bias_changes_output():
    axes_b, grav, goal = _default_inputs(num_envs=1)
    state = re.new_state(num_envs=1)
    out_straight = re.forward(
        axes_b=axes_b, projected_gravity_b=grav, to_goal_b=goal,
        state=state, cfg=re.RollingEngineCfg(),
    )
    out_steered = re.forward(
        axes_b=axes_b, projected_gravity_b=grav, to_goal_b=goal,
        state=state, cfg=re.RollingEngineCfg(steering_bias=0.5),
    )
    # The fields should meaningfully disagree -- not drift at float eps.
    diff_norm = (out_steered - out_straight).abs().mean().item()
    assert diff_norm > 1e-3, (
        f"steering_bias=0.5 produced no meaningful change "
        f"(mean |diff|={diff_norm:.2e}); engine rotation likely inert."
    )


def test_steering_bias_is_mirror_symmetric():
    """+bias on a +y panel should equal -bias on the mirror-paired -y panel.

    Setup: forward=(+x), down=(-z), so right=down x forward=(-y). Two
    panels at (0, +1, 0) and (0, -1, 0) are mirror images across the
    forward-down plane. At steering_bias=+s the two panels' f_align
    values swap (sign-flip) vs steering_bias=-s -- because rotating
    forward by +s about down is a mirror of rotating by -s.
    """
    # Just the two mirror-paired panels.
    axes_b = torch.tensor([
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
    ])
    grav = torch.tensor([[0.0, 0.0, -1.0]])
    goal = torch.tensor([[1.0, 0.0]])
    state = re.new_state(num_envs=1)

    out_pos = re.forward(
        axes_b=axes_b, projected_gravity_b=grav, to_goal_b=goal,
        state=state, cfg=re.RollingEngineCfg(steering_bias=+0.5),
    )[0]                                                                # (2,)
    out_neg = re.forward(
        axes_b=axes_b, projected_gravity_b=grav, to_goal_b=goal,
        state=state, cfg=re.RollingEngineCfg(steering_bias=-0.5),
    )[0]                                                                # (2,)

    # Swapping +bias<->-bias should swap the two panels' outputs.
    torch.testing.assert_close(
        out_pos, out_neg.flip(0), rtol=1e-5, atol=1e-6,
    )


# ---------------------------------------------------------------------------
# 4. Duty cycle / phase envelope
# ---------------------------------------------------------------------------


def test_duty_cycle_below_one_is_phase_dependent():
    axes_b, grav, goal = _default_inputs(num_envs=1)
    cfg = re.RollingEngineCfg(duty_cycle=0.2)

    out_on = re.forward(
        axes_b=axes_b, projected_gravity_b=grav, to_goal_b=goal,
        state=re.new_state(num_envs=1, phase_init=0.0),
        cfg=cfg,
    )
    out_off = re.forward(
        axes_b=axes_b, projected_gravity_b=grav, to_goal_b=goal,
        state=re.new_state(num_envs=1, phase_init=math.pi),
        cfg=cfg,
    )
    diff = (out_on - out_off).abs().mean().item()
    assert diff > 1e-3, (
        f"duty_cycle=0.2 should make phase=0 vs phase=pi meaningfully "
        f"differ; got mean |diff|={diff:.2e}."
    )


def test_phase_envelope_limits():
    """At duty=1.0 envelope is exactly 1; at duty=0 it's a raised cosine."""
    phase = torch.linspace(0.0, 2.0 * math.pi, 16)
    env_full = re._phase_envelope(phase, duty_cycle=1.0)
    env_zero = re._phase_envelope(phase, duty_cycle=0.0)
    torch.testing.assert_close(env_full, torch.ones_like(phase), rtol=0, atol=0)
    expected_zero = 0.5 * (1.0 + phase.cos())
    torch.testing.assert_close(env_zero, expected_zero, rtol=1e-6, atol=1e-7)


# ---------------------------------------------------------------------------
# 5. Guardrails
# ---------------------------------------------------------------------------


def test_style_params_raises():
    axes_b, grav, goal = _default_inputs(num_envs=1)
    cfg = re.RollingEngineCfg(style_params={"wobble": 0.3})
    state = re.new_state(num_envs=1)
    with pytest.raises(NotImplementedError, match="style_params"):
        re.forward(
            axes_b=axes_b, projected_gravity_b=grav, to_goal_b=goal,
            state=state, cfg=cfg,
        )


# ---------------------------------------------------------------------------
# 6. Breathing gain (isotropic phase pulse)
# ---------------------------------------------------------------------------


def test_breathing_default_zero_is_regression_safe():
    """breathing_gain=0 must leave engine output identical to the oracle.
    Covered indirectly by test_defaults_match_oracle_* -- this test pins
    the invariant explicitly so future changes to the breathing term
    must consciously break it."""
    axes_b, grav, goal = _default_inputs(num_envs=2)
    state = re.new_state(num_envs=2, phase_init=1.234)
    cfg = re.RollingEngineCfg(breathing_gain=0.0)
    engine_out = re.forward(
        axes_b=axes_b, projected_gravity_b=grav, to_goal_b=goal,
        state=state, cfg=cfg,
    )
    oracle_out = compute_contactpush_oracle(
        axes_b=axes_b, projected_gravity_b=grav, to_goal_b=goal, amplitude=1.0,
    )
    torch.testing.assert_close(engine_out, oracle_out, rtol=1e-6, atol=1e-7)


def test_breathing_gain_is_phase_dependent():
    """breathing_gain > 0 with all other amps zero -> output swings with phase."""
    axes_b, grav, goal = _default_inputs(num_envs=1)
    cfg = re.RollingEngineCfg(
        rear_push_amp=0.0, front_reach_amp=0.0, breathing_gain=0.5,
    )
    out_0 = re.forward(
        axes_b=axes_b, projected_gravity_b=grav, to_goal_b=goal,
        state=re.new_state(num_envs=1, phase_init=0.0), cfg=cfg,
    )
    out_half = re.forward(
        axes_b=axes_b, projected_gravity_b=grav, to_goal_b=goal,
        state=re.new_state(num_envs=1, phase_init=math.pi / 2), cfg=cfg,
    )
    out_pi = re.forward(
        axes_b=axes_b, projected_gravity_b=grav, to_goal_b=goal,
        state=re.new_state(num_envs=1, phase_init=math.pi), cfg=cfg,
    )
    # sin(0)=0 and sin(pi)=0 -> both should produce near-zero output.
    # sin(pi/2)=1 -> output should be near 0.5 everywhere.
    torch.testing.assert_close(out_0, torch.zeros_like(out_0), atol=1e-6, rtol=0)
    torch.testing.assert_close(out_pi, torch.zeros_like(out_pi), atol=1e-6, rtol=0)
    torch.testing.assert_close(
        out_half, torch.full_like(out_half, 0.5), atol=1e-6, rtol=0,
    )


def test_breathing_is_isotropic_across_panels():
    """Pure breathing output must be the same scalar across all 42 panels."""
    axes_b, grav, goal = _default_inputs(num_envs=1)
    cfg = re.RollingEngineCfg(
        rear_push_amp=0.0, front_reach_amp=0.0, breathing_gain=0.3,
    )
    state = re.new_state(num_envs=1, phase_init=0.7)  # sin(0.7) ~ 0.644
    out = re.forward(
        axes_b=axes_b, projected_gravity_b=grav, to_goal_b=goal,
        state=state, cfg=cfg,
    )[0]                                                               # (42,)
    # Every panel should see exactly the same breathing scalar.
    torch.testing.assert_close(
        out, torch.full((42,), out[0].item()), atol=1e-6, rtol=0,
    )


def test_breathing_plus_other_amps_superposes():
    """At breathing_gain + rear/front > 0, output = oracle-output + breathing."""
    axes_b, grav, goal = _default_inputs(num_envs=1)
    phase_init = math.pi / 2  # sin(pi/2) = 1 -> breathing contributes full gain
    state = re.new_state(num_envs=1, phase_init=phase_init)
    cfg_breath = re.RollingEngineCfg(breathing_gain=0.4)
    cfg_no_breath = re.RollingEngineCfg(breathing_gain=0.0)
    out_with = re.forward(
        axes_b=axes_b, projected_gravity_b=grav, to_goal_b=goal,
        state=state, cfg=cfg_breath,
    )
    out_without = re.forward(
        axes_b=axes_b, projected_gravity_b=grav, to_goal_b=goal,
        state=state, cfg=cfg_no_breath,
    )
    diff = out_with - out_without
    # Before the final clamp, diff should be ~0.4 everywhere. The clamp
    # may cap it where the base output is already near ±1, so test the
    # central (low-magnitude) region.
    low_mag_mask = out_without.abs() < 0.5
    assert low_mag_mask.any(), "expected some panels to be mid-range pre-clamp"
    torch.testing.assert_close(
        diff[low_mag_mask],
        torch.full_like(diff[low_mag_mask], 0.4),
        atol=1e-5, rtol=0,
    )


def test_forward_sh_coeffs_shape():
    axes_b, grav, goal = _default_inputs(num_envs=3)
    state = re.new_state(num_envs=3)
    cfg = re.RollingEngineCfg()
    # Build a minimal SH basis for the test normals.
    from scripted_roll import build_sh_basis
    B = build_sh_basis(axes_b)                  # (42, 9)
    B_pinv = torch.linalg.pinv(B)               # (9, 42)
    coeffs = re.forward_sh_coeffs(
        axes_b=axes_b, projected_gravity_b=grav, to_goal_b=goal,
        state=state, cfg=cfg, B_pinv=B_pinv,
    )
    assert coeffs.shape == (3, 9)


# ---------------------------------------------------------------------------
# 7. W3: explicit lean / push / front-retract decomposition
# ---------------------------------------------------------------------------
#
# Field-isolation tests. Setup uses the canonical "blob upright, goal +x"
# scene so {front, rear, top, bottom} regions on the panel sphere are
# unambiguous:
#   down_b    = (0, 0, -1)  -> d_align = -axes_z
#   forward_b = (+1, 0, 0)  -> f_align =  axes_x
# Bottom panels:    d_align > 0  (axes point downward, axes_z < 0)
# Top panels:       d_align < 0
# Front panels:     f_align > 0
# Rear panels:      f_align < 0
# The engine's contact_gate sigmoid centres at d_align=0.3, so panels
# with d_align well above 0.3 are "in contact" and well below are "off
# the floor".


def _w3_region_masks(axes_b, grav, goal):
    """Compute (front_bottom, rear_bottom, top) boolean masks.

    Returns three (42,) bool tensors with strict thresholds (well into
    each region's interior, away from the sigmoid inflection points)
    so amplitude-isolation tests can assert "these panels respond, the
    others stay near zero" without flakiness from boundary panels.
    """
    down_b = grav[0] / grav[0].norm()
    if goal.shape[-1] == 2:
        goal3 = torch.cat([goal[0], torch.zeros(1)])
    else:
        goal3 = goal[0]
    forward_b = goal3 - (goal3 * down_b).sum() * down_b
    forward_b = forward_b / forward_b.norm()
    d = (axes_b * down_b).sum(-1)                                       # (42,)
    f = (axes_b * forward_b).sum(-1)                                    # (42,)

    front_bottom = (d > 0.5) & (f > 0.5)
    rear_bottom = (d > 0.5) & (f < -0.5)
    top = d < -0.5
    return front_bottom, rear_bottom, top


def test_front_retract_amp_zero_matches_oracle():
    """Explicit pin: front_retract_amp=0.0 (with default phase / duty)
    reproduces compute_contactpush_oracle bit-exactly. This is the W3
    backward-compat invariant the rest of the field decomposition is
    designed around -- changing it requires conscious sign-off."""
    axes_b, grav, goal = _default_inputs(num_envs=4)
    cfg = re.RollingEngineCfg(front_retract_amp=0.0)
    # Confirm all the new W3 phase / duty defaults are at identity values.
    assert cfg.lean_phase == 0.0
    assert cfg.retract_phase == 0.0
    assert cfg.push_phase == 0.0
    assert cfg.lean_duty == 1.0
    assert cfg.retract_duty == 1.0
    assert cfg.push_duty == 1.0

    state = re.new_state(num_envs=4, phase_init=1.7)  # any phase, push_env=1
    engine_out = re.forward(
        axes_b=axes_b, projected_gravity_b=grav, to_goal_b=goal,
        state=state, cfg=cfg,
    )
    oracle_out = compute_contactpush_oracle(
        axes_b=axes_b, projected_gravity_b=grav, to_goal_b=goal,
        amplitude=1.0,
    )
    torch.testing.assert_close(engine_out, oracle_out, rtol=1e-6, atol=1e-7)


def test_front_retract_amp_isolated():
    """With all other amps zeroed and front_retract_amp=1.0, output is
    negative on front-bottom panels (retract command) and ~zero on
    rear and top panels (other regions don't satisfy contact * front)."""
    axes_b, grav, goal = _default_inputs(num_envs=1)
    cfg = re.RollingEngineCfg(
        rear_push_amp=0.0, front_reach_amp=0.0, front_retract_amp=1.0,
    )
    state = re.new_state(num_envs=1)
    out = re.forward(
        axes_b=axes_b, projected_gravity_b=grav, to_goal_b=goal,
        state=state, cfg=cfg,
    )[0]                                                                # (42,)
    front_bottom, rear_bottom, top = _w3_region_masks(axes_b, grav, goal)
    assert front_bottom.any() and rear_bottom.any() and top.any(), (
        "test fixture should produce panels in all three regions"
    )
    # Front-bottom: strongly negative.
    assert (out[front_bottom] < -0.3).all(), (
        f"front-bottom panels should retract (out<<0); got {out[front_bottom]}"
    )
    # Rear-bottom: front sigmoid kills the field (sigmoid(neg)~0).
    assert out[rear_bottom].abs().max() < 0.05, (
        f"rear-bottom panels should be ~0; got {out[rear_bottom]}"
    )
    # Top: contact_gate kills the field.
    assert out[top].abs().max() < 0.05, (
        f"top panels should be ~0; got {out[top]}"
    )


def test_lean_amp_isolated():
    """With all other amps zeroed and front_reach_amp=1.0, output is
    positive on front-top panels (extend / lean) and ~zero on rear and
    bottom panels."""
    axes_b, grav, goal = _default_inputs(num_envs=1)
    cfg = re.RollingEngineCfg(
        rear_push_amp=0.0, front_reach_amp=1.0, front_retract_amp=0.0,
    )
    state = re.new_state(num_envs=1)
    out = re.forward(
        axes_b=axes_b, projected_gravity_b=grav, to_goal_b=goal,
        state=state, cfg=cfg,
    )[0]                                                                # (42,)

    # Build a front-top mask analogously to _w3_region_masks.
    down_b = grav[0] / grav[0].norm()
    goal3 = torch.cat([goal[0], torch.zeros(1)])
    forward_b = goal3 - (goal3 * down_b).sum() * down_b
    forward_b = forward_b / forward_b.norm()
    d = (axes_b * down_b).sum(-1)
    f = (axes_b * forward_b).sum(-1)
    front_top = (d < -0.5) & (f > 0.5)
    rear_any = f < -0.5
    bottom_any = d > 0.5
    assert front_top.any(), "fixture should have front-top panels"
    assert rear_any.any() and bottom_any.any()

    # Front-top: strongly positive (extend).
    assert (out[front_top] > 0.3).all(), (
        f"front-top panels should extend (out>>0); got {out[front_top]}"
    )
    # Rear (any d): front_gate=sigmoid(6*(f-0.2))~0 for f<0 -> field~0.
    assert out[rear_any].abs().max() < 0.05, (
        f"rear panels should be ~0; got {out[rear_any]}"
    )
    # Bottom (any f): top_gate=sigmoid(-6*(d+0.2))~0 for d>0 -> field~0.
    assert out[bottom_any].abs().max() < 0.05, (
        f"bottom panels should be ~0; got {out[bottom_any]}"
    )


def test_push_amp_isolated():
    """With rear_push_amp=1.0 and other amps zero, output is positive on
    rear-bottom panels and ~zero on top panels.

    NOTE: rear_push_field is the original *bipolar* contact-zone dipole
    (the W2 oracle's `contact_gate * (-tanh(...))`), so when it is the
    only active field, front-bottom panels go negative -- they are the
    other half of the same dipole. The W3 split makes that front-side
    retract addressable independently via `front_retract_amp`, but does
    not strip it from the rear-push field (otherwise the
    front_retract_amp=0 backward-compat invariant fails). This test
    therefore asserts only the regions that are unambiguous: rear-bottom
    extends, top stays inactive.
    """
    axes_b, grav, goal = _default_inputs(num_envs=1)
    cfg = re.RollingEngineCfg(
        rear_push_amp=1.0, front_reach_amp=0.0, front_retract_amp=0.0,
    )
    state = re.new_state(num_envs=1)
    out = re.forward(
        axes_b=axes_b, projected_gravity_b=grav, to_goal_b=goal,
        state=state, cfg=cfg,
    )[0]                                                                # (42,)
    front_bottom, rear_bottom, top = _w3_region_masks(axes_b, grav, goal)
    assert rear_bottom.any() and top.any() and front_bottom.any()

    # Rear-bottom: strongly positive (extend / push the floor).
    assert (out[rear_bottom] > 0.3).all(), (
        f"rear-bottom panels should push (out>>0); got {out[rear_bottom]}"
    )
    # Top: contact_gate=sigmoid(6*(d-0.3))~0 for d<-0.5 -> field~0.
    assert out[top].abs().max() < 0.05, (
        f"top panels should be ~0; got {out[top]}"
    )
    # Documenting the bipolar nature: front-bottom is the negative lobe
    # of the same `-tanh(f_align)` factor. We assert it as a fact rather
    # than as a "should be zero" -- explicit independent retract is the
    # job of front_retract_amp, not of zeroing rear_push_amp.
    assert (out[front_bottom] < -0.3).all(), (
        "rear_push_field is the full bipolar dipole; front-bottom is "
        f"its negative lobe. Got {out[front_bottom]}"
    )


def test_phase_offsets_independent():
    """Each of lean_phase, retract_phase, push_phase shifts its field's
    envelope independently of the other two. Concretely: at phase=0 with
    duty<1, envelope=1 (peak); shifting offset by pi inverts to envelope
    =0 (trough), so the field's contribution flips between maximum and
    near-zero. The other two fields' envelopes (offset still 0) must be
    unchanged across the comparison."""
    axes_b, grav, goal = _default_inputs(num_envs=1)
    state = re.new_state(num_envs=1, phase_init=0.0)
    duty = 0.0  # raised cosine: envelope = 0.5*(1+cos(phase + offset))

    # --- push: offset=0 -> envelope(0)=1; offset=pi -> envelope(pi)=0.
    cfg_push_zero = re.RollingEngineCfg(
        rear_push_amp=1.0, front_reach_amp=0.0, front_retract_amp=0.0,
        push_duty=duty, push_phase=0.0,
    )
    cfg_push_pi = re.RollingEngineCfg(
        rear_push_amp=1.0, front_reach_amp=0.0, front_retract_amp=0.0,
        push_duty=duty, push_phase=math.pi,
    )
    out_push_zero = re.forward(
        axes_b=axes_b, projected_gravity_b=grav, to_goal_b=goal,
        state=state, cfg=cfg_push_zero,
    )
    out_push_pi = re.forward(
        axes_b=axes_b, projected_gravity_b=grav, to_goal_b=goal,
        state=state, cfg=cfg_push_pi,
    )
    # phase=0+pi -> cos(pi)=-1 -> envelope=0 -> output=0 everywhere.
    torch.testing.assert_close(
        out_push_pi, torch.zeros_like(out_push_pi), atol=1e-6, rtol=0,
    )
    # phase=0+0 -> envelope=1 -> output is the un-enveloped rear push.
    assert out_push_zero.abs().max() > 0.3, (
        "push_phase=0, duty=0 at phase=0 should be peak envelope, not zero"
    )

    # --- lean: same logic on front_reach_amp / lean_phase / lean_duty.
    cfg_lean_zero = re.RollingEngineCfg(
        rear_push_amp=0.0, front_reach_amp=1.0, front_retract_amp=0.0,
        lean_duty=duty, lean_phase=0.0,
    )
    cfg_lean_pi = re.RollingEngineCfg(
        rear_push_amp=0.0, front_reach_amp=1.0, front_retract_amp=0.0,
        lean_duty=duty, lean_phase=math.pi,
    )
    out_lean_zero = re.forward(
        axes_b=axes_b, projected_gravity_b=grav, to_goal_b=goal,
        state=state, cfg=cfg_lean_zero,
    )
    out_lean_pi = re.forward(
        axes_b=axes_b, projected_gravity_b=grav, to_goal_b=goal,
        state=state, cfg=cfg_lean_pi,
    )
    torch.testing.assert_close(
        out_lean_pi, torch.zeros_like(out_lean_pi), atol=1e-6, rtol=0,
    )
    assert out_lean_zero.abs().max() > 0.3

    # --- retract: same logic.
    cfg_ret_zero = re.RollingEngineCfg(
        rear_push_amp=0.0, front_reach_amp=0.0, front_retract_amp=1.0,
        retract_duty=duty, retract_phase=0.0,
    )
    cfg_ret_pi = re.RollingEngineCfg(
        rear_push_amp=0.0, front_reach_amp=0.0, front_retract_amp=1.0,
        retract_duty=duty, retract_phase=math.pi,
    )
    out_ret_zero = re.forward(
        axes_b=axes_b, projected_gravity_b=grav, to_goal_b=goal,
        state=state, cfg=cfg_ret_zero,
    )
    out_ret_pi = re.forward(
        axes_b=axes_b, projected_gravity_b=grav, to_goal_b=goal,
        state=state, cfg=cfg_ret_pi,
    )
    torch.testing.assert_close(
        out_ret_pi, torch.zeros_like(out_ret_pi), atol=1e-6, rtol=0,
    )
    assert out_ret_zero.abs().max() > 0.3

    # --- independence: setting lean_phase=pi must not change the push
    #     output. Run rear push only, with a non-default lean_phase, and
    #     compare to the push-only baseline (lean_phase=0).
    cfg_push_baseline = re.RollingEngineCfg(
        rear_push_amp=1.0, front_reach_amp=0.0, front_retract_amp=0.0,
        push_duty=0.5,  # arbitrary non-trivial duty
    )
    cfg_push_with_lean_offset = re.RollingEngineCfg(
        rear_push_amp=1.0, front_reach_amp=0.0, front_retract_amp=0.0,
        push_duty=0.5, lean_phase=math.pi, retract_phase=math.pi,
    )
    state2 = re.new_state(num_envs=1, phase_init=0.6)
    out_baseline = re.forward(
        axes_b=axes_b, projected_gravity_b=grav, to_goal_b=goal,
        state=state2, cfg=cfg_push_baseline,
    )
    out_with_offset = re.forward(
        axes_b=axes_b, projected_gravity_b=grav, to_goal_b=goal,
        state=state2, cfg=cfg_push_with_lean_offset,
    )
    torch.testing.assert_close(
        out_baseline, out_with_offset, rtol=0, atol=0,
        msg="lean_phase / retract_phase must not affect rear push output",
    )


# ---------------------------------------------------------------------------
# 8. W7: closed-loop event-gated reset + slip-aware scheduler
# ---------------------------------------------------------------------------


def test_w7_disabled_matches_w6_baseline():
    """With the W7 closed-loop features OFF (defaults), `forward()` must
    return bit-exactly the same output whether or not a `diagnostics`
    dict is passed. This protects W6 atlas reproducibility -- any
    silent leak of telemetry into the W6 regime would invalidate every
    sample in `workspace/_tasks_out/w6_atlas/atlas.parquet`.
    """
    axes_b, grav, goal = _default_inputs(num_envs=4)
    # Use a non-trivial cfg from W6 sample 17 to stress the comparison.
    cfg = re.RollingEngineCfg(
        rear_push_amp=0.35,
        front_retract_amp=0.58,
        lean_phase=0.06 * 2.0 * math.pi,
        phase_velocity_hz=1.12,
        push_duty=0.39,
    )
    # Sanity: W7 features are disabled at default.
    assert cfg.event_phase_reset is False
    assert cfg.slip_aware_scheduler is False

    state = re.new_state(num_envs=4, phase_init=1.234)
    out_no_diag = re.forward(
        axes_b=axes_b, projected_gravity_b=grav, to_goal_b=goal,
        state=state, cfg=cfg,
    )
    # Same call but with a populated diagnostics dict -- the slip values
    # and support_asym are deliberately picked to fire BOTH features if
    # they were enabled. They must be ignored.
    diagnostics = {
        "slip_ratio": torch.tensor([0.0, 0.4, 0.6, 1.0]),
        "support_asym": torch.tensor([0.0, 5.0, -10.0, 100.0]),
    }
    out_with_diag = re.forward(
        axes_b=axes_b, projected_gravity_b=grav, to_goal_b=goal,
        state=state, cfg=cfg, diagnostics=diagnostics,
    )
    assert torch.equal(out_no_diag, out_with_diag), (
        "W6 atlas regime: diagnostics must be ignored when "
        "event_phase_reset=slip_aware_scheduler=False"
    )


def test_slip_scheduler_monotonic():
    """Slip multiplier is monotone-decreasing in slip ratio, equals 1.0
    at and below the low threshold, equals slip_scale_min at and above
    the high threshold, and linearly interpolates between."""
    cfg = re.RollingEngineCfg(slip_aware_scheduler=True)
    # Defaults: low=0.15, high=0.30, min=0.5.
    assert cfg.slip_scale_low_threshold == 0.15
    assert cfg.slip_scale_high_threshold == 0.30
    assert cfg.slip_scale_min == 0.5

    slips = torch.tensor([0.0, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35])
    multipliers = re.slip_amplitude_scale({"slip_ratio": slips}, cfg)

    # Bound: shape preserved.
    assert multipliers.shape == slips.shape

    # Monotone-decreasing.
    diffs = multipliers[1:] - multipliers[:-1]
    assert (diffs <= 1e-7).all(), (
        f"multiplier should be monotone non-increasing in slip; got {multipliers}"
    )

    # Equality at the boundary points.
    assert multipliers[0].item() == pytest.approx(1.0, abs=1e-6)   # slip 0.00
    assert multipliers[1].item() == pytest.approx(1.0, abs=1e-6)   # slip 0.10
    assert multipliers[2].item() == pytest.approx(1.0, abs=1e-6)   # slip 0.15
    assert multipliers[5].item() == pytest.approx(0.5, abs=1e-6)   # slip 0.30
    assert multipliers[6].item() == pytest.approx(0.5, abs=1e-6)   # slip 0.35

    # Linear interpolation between 0.15 and 0.30:
    # slip 0.20 -> frac (0.20-0.15)/0.15 = 1/3  -> 1 - 1/3 * 0.5 = 0.8333
    # slip 0.25 -> frac 2/3                      -> 1 - 2/3 * 0.5 = 0.6667
    assert multipliers[3].item() == pytest.approx(1.0 - (1.0 / 3.0) * 0.5, abs=1e-5)
    assert multipliers[4].item() == pytest.approx(1.0 - (2.0 / 3.0) * 0.5, abs=1e-5)


def test_phase_reset_fires_on_support_asym():
    """Per-env support_asym above the configured threshold triggers the
    event-based phase reset. Below the threshold or at default
    `event_phase_reset=False`, no reset fires."""
    cfg = re.RollingEngineCfg(
        event_phase_reset=True,
        phase_reset_support_asym_threshold=2.0,
        # Disable clause (b) by setting the increment threshold high:
        # state.phase==last_reset_phase==0 at construction so increment is
        # always 0 here, but be explicit.
        phase_reset_roll_increment_rad=10.0,
    )
    state = re.new_state(num_envs=4, phase_init=0.0)
    diagnostics = {
        # Threshold is 2.0; envs 1 and 2 cross it (in absolute value).
        "support_asym": torch.tensor([1.0, 3.0, -3.0, 0.5]),
    }
    reset_mask = re.should_phase_reset(diagnostics, state, cfg)
    expected = torch.tensor([False, True, True, False])
    assert torch.equal(reset_mask, expected), (
        f"reset mask should fire for envs 1 and 2 only; got {reset_mask}"
    )

    # Cross-check: with event_phase_reset=False the same diagnostics
    # produce no resets at all.
    cfg_off = re.RollingEngineCfg(event_phase_reset=False)
    reset_mask_off = re.should_phase_reset(diagnostics, state, cfg_off)
    assert (~reset_mask_off).all(), (
        "event_phase_reset=False must suppress all resets regardless of diag"
    )


def test_phase_reset_fires_on_roll_increment():
    """Cumulative phase advance since the last reset, exceeding
    `phase_reset_roll_increment_rad`, triggers a reset and updates
    `state.last_reset_phase` to the new phase."""
    axes_b, grav, goal = _default_inputs(num_envs=1)

    # Threshold low enough that a single phase-pi advance crosses it.
    increment_threshold = math.pi / 4
    cfg = re.RollingEngineCfg(
        event_phase_reset=True,
        phase_reset_support_asym_threshold=1e9,  # disable clause (a)
        phase_reset_roll_increment_rad=increment_threshold,
        phase_reset_advance=math.pi / 2,
    )
    state = re.new_state(num_envs=1, phase_init=0.0)

    # Diagnostics with low support_asym so clause (a) cannot fire.
    diagnostics = {"support_asym": torch.tensor([0.0])}

    # Step 1: phase still at 0.0 -- increment is 0, no reset.
    reset_mask_0 = re.should_phase_reset(diagnostics, state, cfg)
    assert not reset_mask_0.any()

    # Drive the state's phase forward toward the threshold WITHOUT
    # crossing it. Incremental nudge to phase=pi/8 < pi/4.
    state.phase = torch.tensor([math.pi / 8])
    reset_mask_below = re.should_phase_reset(diagnostics, state, cfg)
    assert not reset_mask_below.any(), (
        f"increment {math.pi / 8} below threshold {increment_threshold} "
        f"should not trigger; got reset_mask={reset_mask_below}"
    )

    # Now drive state.phase past the threshold and call forward(); this
    # should fire the reset and update last_reset_phase.
    state.phase = torch.tensor([math.pi])  # increment = pi, > pi/4
    pre_reset_phase = state.phase.clone()
    reset_mask_above = re.should_phase_reset(diagnostics, state, cfg)
    assert reset_mask_above.all(), (
        "increment of pi must cross threshold pi/4"
    )

    # Run the controller -- it applies the reset and bookkeeping. We
    # confirm state.phase jumped by phase_reset_advance (mod 2*pi) and
    # last_reset_phase tracks the post-jump phase, re-arming clause (b).
    re.forward(
        axes_b=axes_b, projected_gravity_b=grav, to_goal_b=goal,
        state=state, cfg=cfg, diagnostics=diagnostics,
    )
    expected_post_phase = (pre_reset_phase + cfg.phase_reset_advance) % (
        2.0 * math.pi
    )
    torch.testing.assert_close(state.phase, expected_post_phase, atol=1e-6, rtol=0)
    torch.testing.assert_close(
        state.last_reset_phase, expected_post_phase, atol=1e-6, rtol=0,
        msg="last_reset_phase must update to the post-reset phase",
    )

    # Re-arm verified: immediately after the reset, increment (phase -
    # last_reset_phase) % 2pi == 0, so clause (b) does not fire again.
    reset_mask_after = re.should_phase_reset(diagnostics, state, cfg)
    assert not reset_mask_after.any(), (
        "after reset, last_reset_phase tracks phase -- clause (b) re-armed"
    )
