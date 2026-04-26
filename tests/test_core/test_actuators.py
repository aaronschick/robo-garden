"""Tests for the actuator catalog loader + linear/rotary validator dispatch.

Covers:
  * Linear actuator YAML loads end-to-end (no bridge patch required).
  * Rotary catalogs remain unaffected by the new dual-kind schema.
  * Validator dispatches on ``kind`` — same error class for
    missing linear ``force_n`` as for missing rotary ``torque_nm``.
  * Linear spec-range checks fire for implausible values.
  * Full-catalog parametrized sweep: every YAML entry in
    ``data/actuators/`` loads cleanly and passes spec validation.
"""

from __future__ import annotations

import yaml

import mujoco
import pytest

from robo_garden.building.actuators import load_catalog
from robo_garden.building.models import Actuator, ActuatorAssignment
from robo_garden.building.validator import (
    _validate_actuator_specs,
    validate_buildability,
)
from robo_garden.config import DATA_DIR


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


ACTUATOR_DIR = DATA_DIR / "actuators"


def _all_yaml_entries():
    """Yield (yaml_filename, entry_dict) for every actuator YAML entry."""
    for yaml_file in sorted(ACTUATOR_DIR.glob("*.yaml")):
        with open(yaml_file) as f:
            data = yaml.safe_load(f)
        for entry in data.get("actuators", []):
            yield yaml_file.name, entry


def _sample_model_with_joint(joint_name: str = "hinge") -> mujoco.MjModel:
    """Compile a minimal pendulum MJCF with a single named joint."""
    xml = f"""
    <mujoco>
      <worldbody>
        <body name="arm" pos="0 0 1">
          <joint name="{joint_name}" type="hinge" axis="0 1 0"/>
          <geom type="capsule" size="0.02" fromto="0 0 0 0 0 -0.1" mass="0.05"/>
        </body>
      </worldbody>
    </mujoco>
    """
    return mujoco.MjModel.from_xml_string(xml)


# ---------------------------------------------------------------------------
# Loader tests
# ---------------------------------------------------------------------------


def test_load_catalog_includes_linear_and_rotary():
    """The loader materializes every YAML entry — no linear skip, no field filter."""
    catalog = load_catalog()
    assert len(catalog) > 0
    kinds = {a.kind for a in catalog}
    assert kinds == {"linear", "rotary"}, (
        f"Expected both linear and rotary actuators in catalog, got {kinds}"
    )


def test_linear_yaml_loads_all_six_entries():
    """Every entry in linear.yaml materializes with kind == 'linear'."""
    catalog = load_catalog()
    linear = [a for a in catalog if a.kind == "linear"]
    # linear.yaml ships with exactly 6 entries per Agent A2's changelog.
    assert len(linear) == 6, f"Expected 6 linear entries, got {len(linear)}"
    for a in linear:
        assert isinstance(a, Actuator)
        assert a.type == "linear"
        assert a.force_n is not None and a.force_n > 0
        assert a.speed_mps is not None and a.speed_mps > 0
        assert a.stroke_mm is not None and a.stroke_mm > 0
        # Rotary fields stay None on linear entries.
        assert a.torque_nm is None
        assert a.speed_rpm is None


@pytest.mark.parametrize(
    "yaml_name",
    ["bldc_motors.yaml", "hobby_servos.yaml", "dynamixel.yaml"],
)
def test_rotary_yaml_unchanged(yaml_name: str):
    """Pre-existing rotary catalogs still load and still register as rotary."""
    with open(ACTUATOR_DIR / yaml_name) as f:
        data = yaml.safe_load(f)

    catalog_by_id = {a.id: a for a in load_catalog()}
    for entry in data["actuators"]:
        act = catalog_by_id[entry["id"]]
        assert act.kind == "rotary"
        assert act.torque_nm is not None and act.torque_nm > 0
        assert act.speed_rpm is not None and act.speed_rpm > 0


# ---------------------------------------------------------------------------
# Validator dispatch tests
# ---------------------------------------------------------------------------


def _make_rotary(**overrides) -> Actuator:
    base = dict(
        id="test_rotary", name="Test Rotary",
        type="servo", voltage=12.0, weight_g=50.0,
        torque_nm=1.0, speed_rpm=60.0,
    )
    base.update(overrides)
    return Actuator(**base)


def _make_linear(**overrides) -> Actuator:
    base = dict(
        id="test_linear", name="Test Linear",
        type="linear", voltage=12.0, weight_g=50.0,
        force_n=50.0, speed_mps=0.05, stroke_mm=50.0,
    )
    base.update(overrides)
    return Actuator(**base)


def test_spec_validation_passes_for_valid_rotary():
    assert _validate_actuator_specs(_make_rotary()) == []


def test_spec_validation_passes_for_valid_linear():
    assert _validate_actuator_specs(_make_linear()) == []


def test_linear_missing_force_fails_validation_same_way_as_rotary_missing_torque():
    """Missing linear ``force_n`` should surface the same error-type contract
    as a missing rotary ``torque_nm``: a non-empty errors list from
    ``_validate_actuator_specs`` that causes ``validate_buildability`` to
    flip ``passed = False``.
    """
    rotary = _make_rotary(torque_nm=None)
    linear = _make_linear(force_n=None)

    rotary_errs = _validate_actuator_specs(rotary)
    linear_errs = _validate_actuator_specs(linear)

    assert rotary_errs, "rotary missing torque_nm should error"
    assert linear_errs, "linear missing force_n should error"

    # Both should fail validate_buildability identically.
    model = _sample_model_with_joint()
    for act in (rotary, linear):
        report = validate_buildability(
            model,
            [ActuatorAssignment(joint_name="hinge", actuator=act)],
        )
        assert report.passed is False
        assert any("missing or non-positive" in e for e in report.errors)


def test_linear_out_of_range_force_fails_validation():
    bad = _make_linear(force_n=100_000.0)
    errs = _validate_actuator_specs(bad)
    assert errs
    assert any("force_n" in e and "outside plausible range" in e for e in errs)

    model = _sample_model_with_joint()
    report = validate_buildability(
        model, [ActuatorAssignment(joint_name="hinge", actuator=bad)]
    )
    assert report.passed is False


def test_linear_out_of_range_speed_fails_validation():
    bad = _make_linear(speed_mps=50.0)
    errs = _validate_actuator_specs(bad)
    assert any("speed_mps" in e and "outside plausible range" in e for e in errs)


def test_linear_out_of_range_stroke_fails_validation():
    bad = _make_linear(stroke_mm=10_000.0)
    errs = _validate_actuator_specs(bad)
    assert any("stroke_mm" in e and "outside plausible range" in e for e in errs)


def test_rotary_validation_behavior_unchanged_for_valid_entry():
    """Regression guard: the historical rotary torque-capacity +
    low-speed-warning path still runs and produces the legacy check
    string for a valid rotary assignment.
    """
    model = _sample_model_with_joint()
    act = _make_rotary(torque_nm=5.0, speed_rpm=60.0)
    report = validate_buildability(
        model, [ActuatorAssignment(joint_name="hinge", actuator=act)]
    )
    assert report.passed is True
    assert any("Test Rotary OK" in c for c in report.checks)


# ---------------------------------------------------------------------------
# Full-catalog sweep
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "yaml_name,entry",
    [(n, e) for n, e in _all_yaml_entries()],
    ids=[f"{n}::{e['id']}" for n, e in _all_yaml_entries()],
)
def test_every_catalog_entry_loads_and_validates(yaml_name: str, entry: dict):
    """Every real YAML entry must (a) materialize as Actuator and
    (b) pass spec validation with zero errors."""
    actuator = Actuator(**entry)
    errs = _validate_actuator_specs(actuator)
    assert errs == [], (
        f"{yaml_name}::{entry['id']} failed spec validation: {errs}"
    )
