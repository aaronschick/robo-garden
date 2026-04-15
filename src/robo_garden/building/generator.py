"""Validate and post-process Claude-generated MJCF robot descriptions."""

from __future__ import annotations

from robo_garden.building.actuators import find_actuator
from robo_garden.building.materials import find_material
from robo_garden.building.models import ActuatorAssignment, MaterialAssignment
from robo_garden.core.formats import validate_mjcf, ValidationResult


def process_robot_generation(
    mjcf_xml: str,
    actuator_assignments: list[dict] | None = None,
    material_assignments: list[dict] | None = None,
) -> dict:
    """Validate MJCF and resolve actuator/material assignments.

    Returns a dict with validation results, resolved assignments, and any issues.
    """
    result = validate_mjcf(mjcf_xml)
    issues = list(result.errors)
    warnings = list(result.warnings)

    resolved_actuators = []
    if actuator_assignments:
        for assignment in actuator_assignments:
            actuator = find_actuator(assignment["actuator_id"])
            if actuator is None:
                issues.append(f"Unknown actuator: {assignment['actuator_id']}")
            else:
                resolved_actuators.append(
                    ActuatorAssignment(
                        joint_name=assignment["joint_name"],
                        actuator=actuator,
                        gear_ratio=assignment.get("gear_ratio", 1.0),
                    )
                )

    resolved_materials = []
    if material_assignments:
        for assignment in material_assignments:
            material = find_material(assignment["material_id"])
            if material is None:
                issues.append(f"Unknown material: {assignment['material_id']}")
            else:
                resolved_materials.append(
                    MaterialAssignment(
                        link_name=assignment["link_name"],
                        material=material,
                    )
                )

    return {
        "valid": result.valid and len(issues) == 0,
        "errors": issues,
        "warnings": warnings,
        "actuators": resolved_actuators,
        "materials": resolved_materials,
        "model": result.model,
    }
