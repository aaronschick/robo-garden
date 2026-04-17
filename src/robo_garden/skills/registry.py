"""File-system registry for skills and policy specs.

Workspace layout::

    workspace/skills/<robot_name>/<skill_slug>/
        skill.json
        variants/<variant_id>/
            variant.json
            policy/        ← copy of the checkpoint

    workspace/policies/<robot_name>/<policy_name>/
        policy.json
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from robo_garden.skills import SkillEntry, SkillSpec, VariantSpec

log = logging.getLogger(__name__)


def _skills_root() -> Path:
    from robo_garden.config import SKILLS_DIR
    return SKILLS_DIR


def _policies_root() -> Path:
    from robo_garden.config import POLICIES_DIR
    return POLICIES_DIR


# ---------------------------------------------------------------------------
# Skills
# ---------------------------------------------------------------------------

def list_skills(robot_name: str | None = None) -> list[SkillEntry]:
    """Return a flat list of SkillEntry objects, optionally filtered by robot."""
    root = _skills_root()
    if not root.exists():
        return []

    entries: list[SkillEntry] = []
    robot_dirs = [root / robot_name] if robot_name else sorted(root.iterdir())

    for robot_dir in robot_dirs:
        if not robot_dir.is_dir():
            continue
        rname = robot_dir.name
        for skill_dir in sorted(robot_dir.iterdir()):
            if not skill_dir.is_dir():
                continue
            skill_json = skill_dir / "skill.json"
            if not skill_json.exists():
                continue
            try:
                spec = SkillSpec.from_dict(json.loads(skill_json.read_text(encoding="utf-8")))
            except Exception as exc:
                log.warning(f"Skipping malformed skill.json at {skill_json}: {exc}")
                continue

            variants = list_variants(rname, spec.skill_id)
            active_variant = spec.active_variant
            best_reward: float | None = None
            for v in variants:
                if v.variant_id == active_variant:
                    best_reward = v.best_reward
                    break

            entries.append(SkillEntry(
                robot_name=rname,
                skill_id=spec.skill_id,
                display_name=spec.display_name,
                task_description=spec.task_description,
                active_variant_id=active_variant,
                best_reward=best_reward,
                variant_count=len(variants),
                created_at_utc=spec.created_at_utc,
            ))

    return entries


def get_skill(robot_name: str, skill_id: str) -> SkillSpec | None:
    skill_json = _skills_root() / robot_name / skill_id / "skill.json"
    if not skill_json.exists():
        return None
    try:
        return SkillSpec.from_dict(json.loads(skill_json.read_text(encoding="utf-8")))
    except Exception as exc:
        log.warning(f"Could not read skill {robot_name}/{skill_id}: {exc}")
        return None


def save_skill(spec: SkillSpec) -> Path:
    skill_dir = _skills_root() / spec.robot_name / spec.skill_id
    skill_dir.mkdir(parents=True, exist_ok=True)
    skill_json = skill_dir / "skill.json"
    skill_json.write_text(
        json.dumps(spec.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return skill_json


def set_active_variant(robot_name: str, skill_id: str, variant_id: str) -> bool:
    spec = get_skill(robot_name, skill_id)
    if spec is None:
        return False
    spec.active_variant = variant_id
    save_skill(spec)
    return True


# ---------------------------------------------------------------------------
# Variants
# ---------------------------------------------------------------------------

def list_variants(robot_name: str, skill_id: str) -> list[VariantSpec]:
    variants_dir = _skills_root() / robot_name / skill_id / "variants"
    if not variants_dir.exists():
        return []
    specs: list[VariantSpec] = []
    for vdir in sorted(variants_dir.iterdir()):
        if not vdir.is_dir():
            continue
        vj = vdir / "variant.json"
        if not vj.exists():
            continue
        try:
            specs.append(VariantSpec.from_dict(json.loads(vj.read_text(encoding="utf-8"))))
        except Exception as exc:
            log.warning(f"Skipping malformed variant.json at {vj}: {exc}")
    return specs


def get_variant(robot_name: str, skill_id: str, variant_id: str) -> VariantSpec | None:
    vj = _skills_root() / robot_name / skill_id / "variants" / variant_id / "variant.json"
    if not vj.exists():
        return None
    try:
        return VariantSpec.from_dict(json.loads(vj.read_text(encoding="utf-8")))
    except Exception as exc:
        log.warning(f"Could not read variant {variant_id}: {exc}")
        return None


def save_variant(robot_name: str, skill_id: str, spec: VariantSpec) -> Path:
    vdir = _skills_root() / robot_name / skill_id / "variants" / spec.variant_id
    vdir.mkdir(parents=True, exist_ok=True)
    vj = vdir / "variant.json"
    vj.write_text(
        json.dumps(spec.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return vj


def variant_policy_dir(robot_name: str, skill_id: str, variant_id: str) -> Path:
    return _skills_root() / robot_name / skill_id / "variants" / variant_id / "policy"


# ---------------------------------------------------------------------------
# Policies
# ---------------------------------------------------------------------------

def list_policies(robot_name: str | None = None) -> list:
    """Return PolicySpec objects from workspace/policies/, optionally filtered by robot."""
    from robo_garden.skills import PolicySpec

    root = _policies_root()
    if not root.exists():
        return []

    specs: list = []
    robot_dirs = [root / robot_name] if robot_name else sorted(root.iterdir())

    for robot_dir in robot_dirs:
        if not robot_dir.is_dir():
            continue
        for policy_dir in sorted(robot_dir.iterdir()):
            if not policy_dir.is_dir():
                continue
            pj = policy_dir / "policy.json"
            if not pj.exists():
                continue
            try:
                specs.append(PolicySpec.from_dict(json.loads(pj.read_text(encoding="utf-8"))))
            except Exception as exc:
                log.warning(f"Skipping malformed policy.json at {pj}: {exc}")

    return specs


def get_policy(robot_name: str, policy_name: str) -> "PolicySpec | None":
    from robo_garden.skills import PolicySpec

    pj = _policies_root() / robot_name / policy_name / "policy.json"
    if not pj.exists():
        return None
    try:
        return PolicySpec.from_dict(json.loads(pj.read_text(encoding="utf-8")))
    except Exception as exc:
        log.warning(f"Could not read policy {robot_name}/{policy_name}: {exc}")
        return None


def save_policy(spec: "PolicySpec") -> Path:
    policy_dir = _policies_root() / spec.robot_name / spec.policy_name
    policy_dir.mkdir(parents=True, exist_ok=True)
    pj = policy_dir / "policy.json"
    pj.write_text(
        json.dumps(spec.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return pj


def delete_policy(robot_name: str, policy_name: str) -> bool:
    import shutil

    policy_dir = _policies_root() / robot_name / policy_name
    if not policy_dir.exists():
        return False
    try:
        shutil.rmtree(policy_dir)
        return True
    except Exception as exc:
        log.warning(f"Could not delete policy {robot_name}/{policy_name}: {exc}")
        return False
