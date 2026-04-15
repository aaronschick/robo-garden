"""Tests for MJCF/URDF format validation."""

from robo_garden.core.formats import validate_mjcf, model_info


def test_validate_valid_mjcf(minimal_mjcf):
    result = validate_mjcf(minimal_mjcf)
    assert result.valid
    assert len(result.errors) == 0
    assert result.model is not None


def test_validate_invalid_mjcf():
    result = validate_mjcf("<mujoco><invalid_tag/></mujoco>")
    assert not result.valid
    assert len(result.errors) > 0


def test_validate_empty_string():
    result = validate_mjcf("")
    assert not result.valid


def test_model_info(minimal_mjcf):
    result = validate_mjcf(minimal_mjcf)
    info = model_info(result.model)
    assert info["num_bodies"] >= 2
    assert info["num_joints"] >= 1
    assert info["num_actuators"] >= 1


def test_quadruped_model_info(quadruped_mjcf):
    result = validate_mjcf(quadruped_mjcf)
    assert result.valid
    info = model_info(result.model)
    assert info["num_actuators"] == 8  # 4 hips + 4 knees
