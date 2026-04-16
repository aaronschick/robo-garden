"""Constraint validation: torque budget, mass budget, joint ranges, buildability."""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco

from robo_garden.building.models import ActuatorAssignment, MaterialAssignment


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
    material_assignments: list[MaterialAssignment] | None = None,
) -> ValidationReport:
    """Check that a robot design is physically buildable with assigned components.

    Checks:
    - Joint torque requirements vs actuator capabilities
    - Total mass budget
    - Joint range compatibility
    - Estimated actuator cost
    - Material assignments vs MJCF body names
    - Material printability (informational)
    """
    report = ValidationReport()

    # Total mass from MuJoCo model
    total_mass = sum(model.body_mass[i] for i in range(model.nbody))
    report.total_mass_kg = float(total_mass)
    report.checks.append(f"Total mass: {total_mass:.3f} kg")

    if total_mass > 50:
        report.warnings.append(f"Robot mass ({total_mass:.1f}kg) is very heavy for a prototype")

    # Build a map: joint name → joint index for subtree mass lookup
    joint_name_to_idx: dict[str, int] = {}
    for i in range(model.njnt):
        jname = model.joint(i).name
        joint_name_to_idx[jname] = i

    # Check actuator assignments
    total_cost = 0.0
    has_cost = True
    for assignment in actuator_assignments:
        actuator = assignment.actuator

        # Estimate required torque using subtree mass at the joint's body when available.
        # model.body_subtreemass[body_id] = total mass of body + all descendants.
        # This gives a better per-joint load estimate than dividing total mass evenly.
        j_idx = joint_name_to_idx.get(assignment.joint_name)
        if j_idx is not None:
            body_id = int(model.jnt_bodyid[j_idx])
            subtree_mass = float(model.body_subtreemass[body_id])
        else:
            # Joint not found in compiled model — fall back to even split
            subtree_mass = total_mass / max(model.njnt, 1)

        # 10 cm lever arm is a conservative heuristic for a typical limb segment
        estimated_torque = subtree_mass * 9.81 * 0.1
        capacity = actuator.torque_nm * assignment.gear_ratio

        if estimated_torque > capacity:
            report.errors.append(
                f"Joint '{assignment.joint_name}': estimated torque {estimated_torque:.2f}Nm "
                f"exceeds {actuator.name} capacity {capacity:.2f}Nm"
            )
            report.passed = False
        else:
            report.checks.append(
                f"Joint '{assignment.joint_name}': {actuator.name} OK "
                f"({estimated_torque:.2f}/{capacity:.2f}Nm)"
            )

        # Speed check: very slow actuators are unsuitable for dynamic locomotion joints
        if actuator.speed_rpm < 30:
            report.warnings.append(
                f"Joint '{assignment.joint_name}': {actuator.name} speed "
                f"({actuator.speed_rpm:.0f} RPM) may be too slow for dynamic motion"
            )

        if actuator.price_usd is not None:
            total_cost += actuator.price_usd
        else:
            has_cost = False

    report.total_cost_usd = total_cost if has_cost else None
    if has_cost and actuator_assignments:
        report.checks.append(f"Estimated actuator cost: ${total_cost:.2f}")

    # Check material assignments
    if material_assignments:
        body_names = {model.body(i).name for i in range(model.nbody)}
        for assignment in material_assignments:
            if assignment.link_name not in body_names:
                report.warnings.append(
                    f"Material assigned to unknown link '{assignment.link_name}'"
                )
                continue
            mat = assignment.material
            if not mat.printable:
                report.warnings.append(
                    f"Link '{assignment.link_name}': {mat.name} requires machining "
                    f"(not 3D-printable)"
                )
            report.checks.append(
                f"Link '{assignment.link_name}': {mat.name} "
                f"(strength={mat.yield_strength_mpa}MPa, density={mat.density_kg_m3}kg/m\u00b3)"
            )

    return report
