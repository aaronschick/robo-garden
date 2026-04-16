"""URDF/MJCF parsing, validation, and conversion utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco


@dataclass
class ValidationResult:
    valid: bool
    errors: list[str]
    warnings: list[str]
    model: mujoco.MjModel | None = None

    @property
    def summary(self) -> str:
        parts = []
        if self.valid:
            parts.append("VALID")
        else:
            parts.append("INVALID")
        if self.errors:
            parts.append(f"Errors: {'; '.join(self.errors)}")
        if self.warnings:
            parts.append(f"Warnings: {'; '.join(self.warnings)}")
        return " | ".join(parts)


def validate_mjcf(xml_string: str) -> ValidationResult:
    """Validate an MJCF XML string by attempting to compile it with MuJoCo."""
    errors = []
    warnings = []
    model = None

    try:
        model = mujoco.MjModel.from_xml_string(xml_string)
    except Exception as e:
        errors.append(f"MJCF compilation failed: {e}")
        return ValidationResult(valid=False, errors=errors, warnings=warnings)

    # Basic sanity checks
    if model.nbody < 2:
        warnings.append("Robot has fewer than 2 bodies (only worldbody)")
    if model.njnt == 0:
        warnings.append("Robot has no joints")
    if model.nu == 0:
        warnings.append("Robot has no actuators defined")

    return ValidationResult(valid=True, errors=errors, warnings=warnings, model=model)


def validate_urdf(xml_string: str) -> ValidationResult:
    """Validate a URDF XML string by converting through MuJoCo's URDF compiler."""
    errors = []
    warnings = []
    model = None

    try:
        model = mujoco.MjModel.from_xml_string(xml_string)
    except Exception as e:
        errors.append(f"URDF compilation failed: {e}")
        return ValidationResult(valid=False, errors=errors, warnings=warnings)

    if model.nbody < 2:
        warnings.append("Robot has fewer than 2 bodies (only worldbody)")
    if model.njnt == 0:
        warnings.append("Robot has no joints")
    # URDF robots commonly have nu==0 (no <transmission> tags); less alarming than MJCF
    if model.nu == 0:
        warnings.append("No actuators compiled from URDF — add <transmission> tags for control")

    return ValidationResult(valid=True, errors=errors, warnings=warnings, model=model)


def detect_format(xml_string: str) -> str:
    """Return 'urdf' if root element is <robot ...>, else 'mjcf'."""
    stripped = xml_string.lstrip("\ufeff \t\n\r")  # strip BOM + whitespace
    return "urdf" if stripped.startswith("<robot") else "mjcf"


def validate_robot_xml(xml_string: str) -> ValidationResult:
    """Auto-detect format (MJCF or URDF) and validate."""
    if detect_format(xml_string) == "urdf":
        return validate_urdf(xml_string)
    return validate_mjcf(xml_string)


def load_mjcf_file(path: Path) -> mujoco.MjModel:
    """Load an MJCF file from disk."""
    return mujoco.MjModel.from_xml_path(str(path))


def model_info(model: mujoco.MjModel) -> dict:
    """Extract summary information from a compiled MuJoCo model."""
    return {
        "num_bodies": model.nbody,
        "num_joints": model.njnt,
        "num_actuators": model.nu,
        "num_sensors": model.nsensor,
        "num_geoms": model.ngeom,
        "timestep": model.opt.timestep,
    }
