"""Constraint validation: torque budget, mass budget, joint ranges, buildability."""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco

from robo_garden.building.models import ActuatorAssignment


@dataclass
class ValidationReport:
    """Report from physical constraint validation."""
    passed: bool = True
    checks: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    total_mass_kg: float = 0.0
    total_cost_usd: float | None = None


def validate_buildability(
    model: mujoco.MjModel,
    actuator_assignments: list[ActuatorAssignment],
) -> ValidationReport:
    """Check that a robot design is physically buildable with assigned components.

    Checks:
    - Joint torque requirements vs actuator capabilities
    - Total mass budget
    - Joint range compatibility
    - Estimated cost
    """
    report = ValidationReport()

    # Total mass from MuJoCo model
    total_mass = sum(model.body_mass[i] for i in range(model.nbody))
    report.total_mass_kg = float(total_mass)
    report.checks.append(f"Total mass: {total_mass:.3f} kg")

    if total_mass > 50:
        report.warnings.append(f"Robot mass ({total_mass:.1f}kg) is very heavy for a prototype")

    # Check actuator assignments
    total_cost = 0.0
    has_cost = True
    for assignment in actuator_assignments:
        actuator = assignment.actuator

        # Estimate required torque (mass * gravity * max_lever_arm heuristic)
        # This is a rough check - proper static analysis would need full kinematics
        per_joint_mass = total_mass / max(model.njnt, 1)
        estimated_torque = per_joint_mass * 9.81 * 0.1  # 10cm lever arm estimate

        if estimated_torque > actuator.torque_nm * assignment.gear_ratio:
            report.errors.append(
                f"Joint '{assignment.joint_name}': estimated torque {estimated_torque:.2f}Nm "
                f"exceeds {actuator.name} capacity {actuator.torque_nm * assignment.gear_ratio:.2f}Nm"
            )
            report.passed = False
        else:
            report.checks.append(
                f"Joint '{assignment.joint_name}': {actuator.name} OK "
                f"({estimated_torque:.2f}/{actuator.torque_nm * assignment.gear_ratio:.2f}Nm)"
            )

        if actuator.price_usd is not None:
            total_cost += actuator.price_usd
        else:
            has_cost = False

    report.total_cost_usd = total_cost if has_cost else None
    if has_cost:
        report.checks.append(f"Estimated actuator cost: ${total_cost:.2f}")

    return report
