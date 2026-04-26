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
    "straight_roll",
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


def test_straight_roll_matches_w6_sample17_baseline():
    """W8: straight_roll's canonical baseline retuned away from engine
    defaults (oracle@1.0, COT>100, narrow basin) onto W6 sample 17:
        rear_push_amp=0.35, front_retract_amp=0.58,
        lean_phase=0.06 cycles, phase_velocity_hz=1.12, push_duty=0.39.
    See `project_urchin_v3_w6_full_atlas_results.md`.

    Fields the W6 atlas did NOT sweep (front_reach_amp, support_width,
    support_bias, duty_cycle alias, steering_bias) must still match
    engine defaults so the W3 retract field, lean envelope, and arc
    gates stay regression-safe.
    """
    import math as _math
    prim = pr.PRIMITIVES["straight_roll"]
    state = prim.reset(num_envs=1)
    cfg, _ = prim.step(0.0, None, state, 0.04)
    default = re.RollingEngineCfg()
    # W6 sample-17 fields:
    assert cfg.rear_push_amp == pytest.approx(0.35)
    assert cfg.front_retract_amp == pytest.approx(0.58)
    assert cfg.lean_phase == pytest.approx(0.06 * 2 * _math.pi)
    assert cfg.phase_velocity_hz == pytest.approx(1.12)
    assert cfg.push_duty == pytest.approx(0.39)
    # Untouched-by-W6 fields stay at engine defaults:
    for field_name in [
        "front_reach_amp", "support_width", "support_bias",
        "duty_cycle", "steering_bias",
    ]:
        assert getattr(cfg, field_name) == getattr(default, field_name), field_name


# W8: arc_left and arc_right primitives removed from the registry. With
# 42-panel symmetry the urchin has no inherent "front", so steering is
# implicit in the goal-direction observation rather than an explicit
# primitive label. The momentum-gated _make_arc_step helper was deleted
# alongside make_arc_left / make_arc_right. Goal-relative direction
# tracking is handled inside the engine via to_goal_b.


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


def test_style_neutral_matches_w6_sample17():
    """W8: neutral := the W6 sample-17 basin point, applied as
    absolute-value replacements. Since straight_roll's own baseline is
    also W6 sample 17, applying neutral on top is a value-identity
    no-op (cfg fields equal). Pre-W8 neutral was {} and the wrapper
    short-circuited to `styled is base`; the new contract is
    field-identity, not object-identity, so callers can rely on
    neutral always emitting the canonical baseline regardless of how
    the underlying primitive was constructed.
    """
    import math as _math
    base = pr.PRIMITIVES["straight_roll"]
    styled = pr.with_style(base, "neutral")
    state = styled.reset(num_envs=1)
    cfg, _ = styled.step(0.0, None, state, 0.04)
    assert cfg.rear_push_amp == pytest.approx(0.35)
    assert cfg.front_retract_amp == pytest.approx(0.58)
    assert cfg.lean_phase == pytest.approx(0.06 * 2 * _math.pi)
    assert cfg.phase_velocity_hz == pytest.approx(1.12)
    assert cfg.push_duty == pytest.approx(0.39)


def test_style_lazy_anchors_inside_w6_basin():
    """W8: lazy := W6 sample-86-like basin point (rear≈0.25, retract
    ≈0.40, slower clock, longer push_duty). Pre-W8 lazy was a 0.5x
    multiplier on engine defaults (oracle@1.0), which the W6 atlas
    showed was outside the success basin for this robot. Now an
    absolute-value replacement so 'lazy' is a real low-amplitude
    rolling regime, not a stiction-locked one."""
    import math as _math
    base = pr.PRIMITIVES["straight_roll"]
    styled = pr.with_style(base, "lazy")
    state = styled.reset(num_envs=1)
    cfg, _ = styled.step(0.0, None, state, 0.04)
    assert cfg.rear_push_amp == pytest.approx(0.25)
    assert cfg.front_retract_amp == pytest.approx(0.40)
    assert cfg.lean_phase == pytest.approx(0.04 * 2 * _math.pi)
    assert cfg.phase_velocity_hz == pytest.approx(0.80)
    assert cfg.push_duty == pytest.approx(0.50)
    # Untouched fields stay at engine defaults (steering, support_width).
    default = re.RollingEngineCfg()
    assert cfg.steering_bias == default.steering_bias


def test_style_snappy_anchors_inside_w6_basin():
    """W8: snappy := W6 sample-28-like basin point (rear≈0.45,
    retract≈0.66, faster clock, sharp pulse). Pre-W8 snappy was a 1.5x
    multiplier on engine defaults plus duty_cycle=0.4 + phase_hz=2.5
    -- which placed it well outside the W6 success basin. Now an
    absolute-value replacement inside that basin."""
    import math as _math
    base = pr.PRIMITIVES["straight_roll"]
    styled = pr.with_style(base, "snappy")
    state = styled.reset(num_envs=1)
    cfg, _ = styled.step(0.0, None, state, 0.04)
    assert cfg.rear_push_amp == pytest.approx(0.45)
    assert cfg.front_retract_amp == pytest.approx(0.66)
    assert cfg.lean_phase == pytest.approx(0.08 * 2 * _math.pi)
    assert cfg.phase_velocity_hz == pytest.approx(1.50)
    assert cfg.push_duty == pytest.approx(0.39)


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
