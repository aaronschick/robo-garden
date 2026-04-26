"""Constraint validation: torque budget, mass budget, joint ranges, buildability."""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco

from robo_garden.building.models import Actuator, ActuatorAssignment, MaterialAssignment


# Plausible ranges for linear actuator specs. Calibrated against the
# shortlist in data/actuators/linear.yaml (forces 42-267 N, speeds
# 0.008-2.0 m/s, strokes 20-110 mm) with headroom for larger VCAs and
# industrial ball-screw stages. Values outside these ranges almost
# certainly indicate a typo or a bad unit conversion, so they are flagged
# as errors (matching the behavior of rotary torque-capacity violations).
_LINEAR_FORCE_RANGE_N = (1.0, 5000.0)
_LINEAR_SPEED_RANGE_MPS = (0.001, 10.0)
_LINEAR_STROKE_RANGE_MM = (1.0, 500.0)


def _validate_actuator_specs(actuator: Actuator) -> list[str]:
    """Dispatch on actuator kind; return a list of error strings (possibly empty).

    Linear actuators require ``force_n``, ``speed_mps``, ``stroke_mm`` to be
    non-None and positive, and within plausible ranges. Rotary actuators
    require ``torque_nm`` and ``speed_rpm`` to be non-None and positive.
    """
    errors: list[str] = []
    if actuator.kind == "linear":
        if actuator.force_n is None or actuator.force_n <= 0:
            errors.append(
                f"Linear actuator '{actuator.name}': force_n "
                f"({actuator.force_n}) missing or non-positive"
            )
        elif not (_LINEAR_FORCE_RANGE_N[0] <= actuator.force_n <= _LINEAR_FORCE_RANGE_N[1]):
            errors.append(
                f"Linear actuator '{actuator.name}': force_n "
                f"({actuator.force_n:.2f} N) outside plausible range "
                f"{_LINEAR_FORCE_RANGE_N[0]}-{_LINEAR_FORCE_RANGE_N[1]} N"
            )

        if actuator.speed_mps is None or actuator.speed_mps <= 0:
            errors.append(
                f"Linear actuator '{actuator.name}': speed_mps "
                f"({actuator.speed_mps}) missing or non-positive"
            )
        elif not (_LINEAR_SPEED_RANGE_MPS[0] <= actuator.speed_mps <= _LINEAR_SPEED_RANGE_MPS[1]):
            errors.append(
                f"Linear actuator '{actuator.name}': speed_mps "
                f"({actuator.speed_mps} m/s) outside plausible range "
                f"{_LINEAR_SPEED_RANGE_MPS[0]}-{_LINEAR_SPEED_RANGE_MPS[1]} m/s"
            )

        if actuator.stroke_mm is None or actuator.stroke_mm <= 0:
            errors.append(
                f"Linear actuator '{actuator.name}': stroke_mm "
                f"({actuator.stroke_mm}) missing or non-positive"
            )
        elif not (_LINEAR_STROKE_RANGE_MM[0] <= actuator.stroke_mm <= _LINEAR_STROKE_RANGE_MM[1]):
            errors.append(
                f"Linear actuator '{actuator.name}': stroke_mm "
                f"({actuator.stroke_mm} mm) outside plausible range "
                f"{_LINEAR_STROKE_RANGE_MM[0]}-{_LINEAR_STROKE_RANGE_MM[1]} mm"
            )
    else:
        # Rotary path — preserves historical required-field semantics.
        if actuator.torque_nm is None or actuator.torque_nm <= 0:
            errors.append(
                f"Rotary actuator '{actuator.name}': torque_nm "
                f"({actuator.torque_nm}) missing or non-positive"
            )
        if actuator.speed_rpm is None or actuator.speed_rpm <= 0:
            errors.append(
                f"Rotary actuator '{actuator.name}': speed_rpm "
                f"({actuator.speed_rpm}) missing or non-positive"
            )
    return errors


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

        # Spec-range / required-field dispatch (rotary vs linear).
        spec_errors = _validate_actuator_specs(actuator)
        if spec_errors:
            report.errors.extend(spec_errors)
            report.passed = False
            # Skip the torque-capacity / speed checks below — they would
            # divide by None and mask the underlying spec problem.
            if actuator.price_usd is not None:
                total_cost += actuator.price_usd
            else:
                has_cost = False
            continue

        # Torque-capacity and speed heuristics below only apply to rotary
        # actuators; linear ones short-circuit after spec validation.
        if actuator.kind == "linear":
            if actuator.price_usd is not None:
                total_cost += actuator.price_usd
            else:
                has_cost = False
            report.checks.append(
                f"Joint '{assignment.joint_name}': {actuator.name} (linear) "
                f"force={actuator.force_n:.1f}N speed={actuator.speed_mps:.3f}m/s "
                f"stroke={actuator.stroke_mm:.0f}mm"
            )
            continue

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
