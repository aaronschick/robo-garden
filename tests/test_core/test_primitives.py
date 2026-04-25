"""Tests for workspace/robots/urchin_v3/scripts/primitives.py.

Verifies:
 1. Registry contains all expected primitives.
 2. Each primitive's reset() + step() returns a valid RollingEngineCfg.
 3. Static primitives (straight/arcs/stop/wobble) expose the intended cfg
    fields (steering sign, zero amps, phase animation).
 4. Dynamic primitives (accelerate/brake) ramp rear_push_amp over duration
    and clamp at the endpoints.
 5. Style layer: 'neutral' is a no-op, 'lazy' halves amps, 'snappy'
    scales amps up + pulses (duty_cycle<1 with nonzero phase_velocity),
    unknown styles raise loudly.
 6. Engine state advances across successive step() calls for primitives
    with phase_velocity_hz > 0.

Pure torch, no Isaac.
"""
from __future__ import annotations

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

import primitives as pr  # noqa: E402
import rolling_engine as re  # noqa: E402


EXPECTED_PRIMITIVES = {
    "straight_roll", "arc_left", "arc_right",
    "accelerate", "brake", "stop_settle", "wobble_idle",
    "jump_stationary", "jump_rolling", "slide",
}


def test_registry_has_all_primitives():
    assert set(pr.PRIMITIVES) == EXPECTED_PRIMITIVES


def test_all_primitives_reset_and_step():
    for name, prim in pr.PRIMITIVES.items():
        state = prim.reset(num_envs=2)
        cfg, new_state = prim.step(t=0.0, obs=None, state=state, dt=0.04)
        assert isinstance(cfg, re.RollingEngineCfg), name
        assert isinstance(new_state, pr.PrimitiveState), name
        assert new_state.engine_state.phase.shape == (2,), name


def test_straight_roll_matches_engine_defaults():
    prim = pr.PRIMITIVES["straight_roll"]
    state = prim.reset(num_envs=1)
    cfg, _ = prim.step(0.0, None, state, 0.04)
    default = re.RollingEngineCfg()
    for field_name in [
        "rear_push_amp", "front_reach_amp", "support_width",
        "support_bias", "duty_cycle", "steering_bias", "phase_velocity_hz",
    ]:
        assert getattr(cfg, field_name) == getattr(default, field_name), field_name


def test_arc_left_has_negative_steering():
    # obs=None: gate is fully open (introspection mode) -> target bias.
    prim = pr.PRIMITIVES["arc_left"]
    state = prim.reset(num_envs=1)
    cfg, _ = prim.step(0.0, None, state, 0.04)
    assert cfg.steering_bias < 0.0


def test_arc_right_has_positive_steering():
    prim = pr.PRIMITIVES["arc_right"]
    state = prim.reset(num_envs=1)
    cfg, _ = prim.step(0.0, None, state, 0.04)
    assert cfg.steering_bias > 0.0


def _obs_with_ang_vel(mag: float, num_envs: int = 2) -> torch.Tensor:
    # obs layout matches scripted_roll.OBS_*: ang_vel_b at [:, 3:6].
    # Minimum obs width needed by primitive step fns is 6; pad to 9 for
    # realism (ang_vel + gravity also present in real obs).
    obs = torch.zeros(num_envs, 9)
    obs[:, 3] = mag  # roll-axis ang_vel (x in body frame)
    return obs


def test_arc_left_gated_to_zero_at_rest():
    # At |ang_vel|=0 the gate is closed: arc_left emits straight_roll.
    prim = pr.PRIMITIVES["arc_left"]
    state = prim.reset(num_envs=2)
    obs = _obs_with_ang_vel(0.0, num_envs=2)
    cfg, _ = prim.step(0.0, obs, state, 0.04)
    assert cfg.steering_bias == pytest.approx(0.0)


def test_arc_right_gated_to_zero_at_rest():
    prim = pr.PRIMITIVES["arc_right"]
    state = prim.reset(num_envs=2)
    obs = _obs_with_ang_vel(0.0, num_envs=2)
    cfg, _ = prim.step(0.0, obs, state, 0.04)
    assert cfg.steering_bias == pytest.approx(0.0)


def test_arc_left_full_bias_at_speed():
    # |ang_vel| well above v_threshold (default 0.8 rad/s) -> full target.
    prim = pr.PRIMITIVES["arc_left"]
    state = prim.reset(num_envs=2)
    obs = _obs_with_ang_vel(5.0, num_envs=2)
    cfg, _ = prim.step(0.0, obs, state, 0.04)
    assert cfg.steering_bias == pytest.approx(-0.5)


def test_arc_right_full_bias_at_speed():
    prim = pr.PRIMITIVES["arc_right"]
    state = prim.reset(num_envs=2)
    obs = _obs_with_ang_vel(5.0, num_envs=2)
    cfg, _ = prim.step(0.0, obs, state, 0.04)
    assert cfg.steering_bias == pytest.approx(0.5)


def test_arc_partial_gate_scales_linearly():
    # At half threshold, effective bias = 0.5 * target.
    prim = pr.PRIMITIVES["arc_left"]
    state = prim.reset(num_envs=2)
    obs = _obs_with_ang_vel(0.4, num_envs=2)  # half of default 0.8
    cfg, _ = prim.step(0.0, obs, state, 0.04)
    assert cfg.steering_bias == pytest.approx(-0.5 * 0.5)


def test_arcs_match_straight_at_rest():
    # The key physical property: at v=0, arc_left / arc_right / straight
    # emit the same engine cfg. This is what makes the BC dataset
    # primitive_id signal honest — no ambiguous steering during warmup.
    straight = pr.PRIMITIVES["straight_roll"]
    arc_l = pr.PRIMITIVES["arc_left"]
    arc_r = pr.PRIMITIVES["arc_right"]
    obs = _obs_with_ang_vel(0.0, num_envs=2)
    s_state = straight.reset(num_envs=2)
    l_state = arc_l.reset(num_envs=2)
    r_state = arc_r.reset(num_envs=2)
    cfg_s, _ = straight.step(0.0, obs, s_state, 0.04)
    cfg_l, _ = arc_l.step(0.0, obs, l_state, 0.04)
    cfg_r, _ = arc_r.step(0.0, obs, r_state, 0.04)
    assert cfg_s.steering_bias == cfg_l.steering_bias == cfg_r.steering_bias


def test_arc_left_styled_respects_gate():
    # Style wrapping must not cancel the momentum gate: arc_left:snappy
    # at rest should still emit steering_bias=0, then ramp in as obs
    # reports rolling speed. Regression against a future styled_step
    # that forgets to chain through the base step's obs handling.
    base = pr.PRIMITIVES["arc_left"]
    styled = pr.with_style(base, "snappy")
    state = styled.reset(num_envs=2)
    obs_rest = _obs_with_ang_vel(0.0, num_envs=2)
    obs_fast = _obs_with_ang_vel(5.0, num_envs=2)
    cfg_rest, _ = styled.step(0.0, obs_rest, state, 0.04)
    cfg_fast, _ = styled.step(0.0, obs_fast, state, 0.04)
    assert cfg_rest.steering_bias == pytest.approx(0.0)
    assert cfg_fast.steering_bias == pytest.approx(-0.5)


def test_stop_settle_active_brake_then_hold():
    """stop_settle applies mild reverse contact-dipole for the first
    brake_s seconds (rear/front amps < 0), then holds amps at zero for
    the remainder of default_duration_s (settle phase)."""
    prim = pr.PRIMITIVES["stop_settle"]
    state = prim.reset(num_envs=1)
    cfg_start, _ = prim.step(0.0, None, state, 0.04)
    assert cfg_start.rear_push_amp < 0.0
    assert cfg_start.front_reach_amp < 0.0
    cfg_end, _ = prim.step(prim.default_duration_s, None, state, 0.04)
    assert cfg_end.rear_push_amp == 0.0
    assert cfg_end.front_reach_amp == 0.0


def test_wobble_idle_uses_breathing_no_translation():
    """wobble_idle should use breathing_gain + zero push/reach so there's
    no direction-biased force component."""
    prim = pr.PRIMITIVES["wobble_idle"]
    state = prim.reset(num_envs=1)
    cfg, _ = prim.step(0.0, None, state, 0.04)
    assert cfg.breathing_gain > 0.0
    assert cfg.phase_velocity_hz > 0.0
    assert cfg.rear_push_amp == 0.0
    assert cfg.front_reach_amp == 0.0


def test_accelerate_ramps_rear_push():
    prim = pr.PRIMITIVES["accelerate"]
    state = prim.reset(num_envs=1)
    cfg_start, state = prim.step(0.0, None, state, 0.04)
    cfg_mid, state = prim.step(prim.default_duration_s / 2, None, state, 0.04)
    cfg_end, state = prim.step(prim.default_duration_s * 2, None, state, 0.04)
    assert cfg_start.rear_push_amp < cfg_mid.rear_push_amp < cfg_end.rear_push_amp
    assert cfg_start.rear_push_amp == pytest.approx(0.15)
    assert cfg_end.rear_push_amp == pytest.approx(2.2)


def test_brake_flat_reverse_then_zero():
    """Active brake: flat reverse thrust (rear/front amps = -1.0) for
    duration_s seconds, then snaps to zero. Flat profile chosen over
    a ramp so the reverse signal survives the env's action LPF."""
    prim = pr.PRIMITIVES["brake"]
    state = prim.reset(num_envs=1)
    cfg_start, state = prim.step(0.0, None, state, 0.04)
    cfg_mid, state = prim.step(prim.default_duration_s / 2, None, state, 0.04)
    cfg_end, state = prim.step(prim.default_duration_s * 2, None, state, 0.04)
    assert cfg_start.rear_push_amp == pytest.approx(-1.0)
    assert cfg_start.front_reach_amp == pytest.approx(-1.0)
    assert cfg_mid.rear_push_amp == pytest.approx(-1.0)
    assert cfg_end.rear_push_amp == pytest.approx(0.0)
    assert cfg_end.front_reach_amp == pytest.approx(0.0)


def test_jump_stationary_pulse_then_zero():
    """jump_stationary emits breathing_gain>0 for launch_s seconds then
    drops to 0, with zero rolling drive throughout."""
    prim = pr.PRIMITIVES["jump_stationary"]
    state = prim.reset(num_envs=1)
    cfg_launch, state = prim.step(0.0, None, state, 0.04)
    cfg_post, _ = prim.step(prim.default_duration_s, None, state, 0.04)
    assert cfg_launch.breathing_gain > 0.0
    assert cfg_launch.rear_push_amp == 0.0
    assert cfg_launch.front_reach_amp == 0.0
    assert cfg_post.breathing_gain == 0.0


def test_jump_rolling_pulse_with_drive():
    """jump_rolling superimposes a breathing pulse on top of forward
    rolling drive (nonzero rear/front amps)."""
    prim = pr.PRIMITIVES["jump_rolling"]
    state = prim.reset(num_envs=1)
    cfg_launch, state = prim.step(0.0, None, state, 0.04)
    cfg_post, _ = prim.step(prim.default_duration_s, None, state, 0.04)
    assert cfg_launch.breathing_gain > 0.0
    assert cfg_launch.rear_push_amp > 0.0
    assert cfg_launch.front_reach_amp > 0.0
    assert cfg_post.breathing_gain == 0.0
    # rolling drive persists after the pulse
    assert cfg_post.rear_push_amp > 0.0


def test_slide_minimal_contact_drive():
    """slide runs a tight-ball coast: small positive rear push with zero
    front reach and zero breathing. Assumes caller warms up rolling
    first; low drive alone won't accelerate from rest."""
    prim = pr.PRIMITIVES["slide"]
    state = prim.reset(num_envs=1)
    cfg, _ = prim.step(0.0, None, state, 0.04)
    default = re.RollingEngineCfg()
    assert 0.0 < cfg.rear_push_amp < default.rear_push_amp
    assert cfg.front_reach_amp == 0.0
    assert cfg.breathing_gain == 0.0


def test_style_neutral_is_noop():
    base = pr.PRIMITIVES["straight_roll"]
    styled = pr.with_style(base, "neutral")
    assert styled is base  # short-circuit returns original


def test_style_lazy_scales_amps():
    base = pr.PRIMITIVES["straight_roll"]
    styled = pr.with_style(base, "lazy")
    state = styled.reset(num_envs=1)
    cfg, _ = styled.step(0.0, None, state, 0.04)
    default = re.RollingEngineCfg()
    assert cfg.rear_push_amp == pytest.approx(default.rear_push_amp * 0.5)
    assert cfg.front_reach_amp == pytest.approx(default.front_reach_amp * 0.5)
    # steering + duty should be unchanged
    assert cfg.duty_cycle == default.duty_cycle
    assert cfg.steering_bias == default.steering_bias


def test_style_snappy_pulses_and_scales_up():
    base = pr.PRIMITIVES["straight_roll"]
    styled = pr.with_style(base, "snappy")
    state = styled.reset(num_envs=1)
    cfg, _ = styled.step(0.0, None, state, 0.04)
    default = re.RollingEngineCfg()
    assert cfg.rear_push_amp == pytest.approx(default.rear_push_amp * 1.5)
    assert cfg.front_reach_amp == pytest.approx(default.front_reach_amp * 1.3)
    assert cfg.duty_cycle == pytest.approx(0.4)
    # phase must actually advance for duty_cycle gating to pulse visibly
    assert cfg.phase_velocity_hz > 0.0


def test_with_style_unknown_raises():
    base = pr.PRIMITIVES["straight_roll"]
    with pytest.raises(KeyError):
        pr.with_style(base, "does_not_exist")


def test_with_style_preserves_name_tag():
    base = pr.PRIMITIVES["straight_roll"]
    styled = pr.with_style(base, "lazy")
    assert styled.name == "straight_roll:lazy"


def test_engine_state_advances_with_steps():
    prim = pr.PRIMITIVES["wobble_idle"]
    state0 = prim.reset(num_envs=1)
    _, state1 = prim.step(0.0, None, state0, 0.04)
    _, state2 = prim.step(0.04, None, state1, 0.04)
    phase0 = float(state0.engine_state.phase.item())
    phase1 = float(state1.engine_state.phase.item())
    phase2 = float(state2.engine_state.phase.item())
    assert phase1 > phase0
    assert phase2 > phase1


def test_determinism_same_t_same_cfg():
    prim = pr.PRIMITIVES["accelerate"]
    s1 = prim.reset(num_envs=4)
    s2 = prim.reset(num_envs=4)
    c1, _ = prim.step(0.5, None, s1, 0.04)
    c2, _ = prim.step(0.5, None, s2, 0.04)
    assert c1.rear_push_amp == c2.rear_push_amp
    assert c1.steering_bias == c2.steering_bias


def test_styled_primitive_static_matches_direct_override():
    """with_style(straight_roll, 'lazy') should produce the same cfg as
    applying 'lazy' overrides directly to the default cfg."""
    base = pr.PRIMITIVES["straight_roll"]
    styled = pr.with_style(base, "lazy")
    state = styled.reset(num_envs=1)
    cfg, _ = styled.step(0.0, None, state, 0.04)
    expected = pr._apply_style(re.RollingEngineCfg(), pr.STYLES["lazy"])
    assert cfg.rear_push_amp == pytest.approx(expected.rear_push_amp)
    assert cfg.front_reach_amp == pytest.approx(expected.front_reach_amp)
