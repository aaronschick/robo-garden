"""Promote a completed training run to a named skill in the Skills Library.

Usage::

    from robo_garden.skills.promote import promote_run_to_skill

    variant = promote_run_to_skill(
        run_id="run_20260417_013607_e7ee54",
        skill_id="walk_forward",
        display_name="Walk Forward",
        task_description="Move trunk +X at ~0.5 m/s while keeping torso upright",
    )
    # workspace/skills/go2_walker/walk_forward/variants/<variant_id>/policy/ created

"""

from __future__ import annotations

import logging
import re
import shutil
from pathlib import Path
from uuid import uuid4

from robo_garden.skills import SkillSpec, VariantSpec
from robo_garden.skills.registry import (
    get_skill,
    list_variants,
    save_skill,
    save_variant,
    set_active_variant,
    variant_policy_dir,
)
from robo_garden.training.history import find_run

log = logging.getLogger(__name__)

_SLUG_RE = re.compile(r"[^a-z0-9_]+")


def _slugify(text: str) -> str:
    return _SLUG_RE.sub("_", text.lower()).strip("_") or "skill"


def _resolve_checkpoint(checkpoint_path: str) -> Path | None:
    """Return an absolute Path to the checkpoint, or None if not found."""
    from robo_garden.config import PROJECT_ROOT, WORKSPACE_DIR

    p = Path(checkpoint_path)
    if p.is_absolute() and p.exists():
        return p

    # Try relative to project root and workspace
    for base in (PROJECT_ROOT, WORKSPACE_DIR, Path.cwd()):
        candidate = base / p
        if candidate.exists():
            return candidate

    log.warning(f"promote: checkpoint not found at {checkpoint_path!r}")
    return None


def promote_run_to_skill(
    run_id: str,
    skill_id: str,
    display_name: str,
    task_description: str = "",
    robot_name: str | None = None,
) -> VariantSpec:
    """Copy a training checkpoint into workspace/skills/ and write skill manifests.

    Parameters
    ----------
    run_id:
        ID of the training run to promote (must exist in workspace/runs/runs.jsonl).
    skill_id:
        Slug identifier for the skill (e.g. "walk_forward").  Will be slugified.
    display_name:
        Human-readable name shown in the Skills Library (e.g. "Walk Forward").
    task_description:
        One-sentence description of what the skill does.
    robot_name:
        Override robot name.  If omitted, taken from the run record.

    Returns
    -------
    VariantSpec
        The saved variant (including variant_id and checkpoint_path).

    Raises
    ------
    ValueError
        If the run_id is not found in history.
    """
    run = find_run(run_id)
    if run is None:
        raise ValueError(f"promote_run_to_skill: run_id {run_id!r} not found in history")

    rname = robot_name or run.get("robot_name", "unknown_robot")
    skill_id = _slugify(skill_id)
    variant_id = f"v_{uuid4().hex[:6]}"

    # ------------------------------------------------------------------
    # Copy checkpoint
    # ------------------------------------------------------------------
    raw_ckpt = run.get("checkpoint_path", "")
    policy_dest = variant_policy_dir(rname, skill_id, variant_id)

    if raw_ckpt:
        src = _resolve_checkpoint(raw_ckpt)
        if src is not None:
            policy_dest.parent.mkdir(parents=True, exist_ok=True)
            if src.is_dir():
                shutil.copytree(src, policy_dest)
            else:
                policy_dest.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, policy_dest / src.name)
            log.info(f"promote: copied checkpoint {src} → {policy_dest}")
        else:
            policy_dest.mkdir(parents=True, exist_ok=True)
            log.warning(f"promote: checkpoint {raw_ckpt!r} not found — variant saved without policy")
    else:
        policy_dest.mkdir(parents=True, exist_ok=True)
        log.warning("promote: no checkpoint_path in run record — variant saved without policy")

    # ------------------------------------------------------------------
    # Derive checkpoint_path to store in variant.json (relative to WORKSPACE_DIR)
    # ------------------------------------------------------------------
    from robo_garden.config import WORKSPACE_DIR
    try:
        rel_ckpt = str(policy_dest.relative_to(WORKSPACE_DIR))
    except ValueError:
        rel_ckpt = str(policy_dest)

    # ------------------------------------------------------------------
    # Write variant.json
    # ------------------------------------------------------------------
    variant = VariantSpec(
        variant_id=variant_id,
        run_id=run_id,
        algorithm=run.get("algorithm", ""),
        best_reward=float(run.get("best_reward") or 0.0),
        total_timesteps=int(run.get("total_timesteps") or 0),
        checkpoint_path=rel_ckpt,
        reward_function_id=run.get("reward_function_id", ""),
        environment_name=run.get("environment_name", ""),
    )
    save_variant(rname, skill_id, variant)

    # ------------------------------------------------------------------
    # Write / update skill.json
    # ------------------------------------------------------------------
    existing = get_skill(rname, skill_id)
    if existing is None:
        # Try to pull obs/action dims from the approved manifest
        obs_spec: dict = {}
        action_spec: dict = {}
        try:
            from robo_garden.config import APPROVED_DIR
            env_name = run.get("environment_name", "")
            manifest_path = APPROVED_DIR / f"{rname}__{env_name}.json"
            if manifest_path.exists():
                import json as _json
                m = _json.loads(manifest_path.read_text(encoding="utf-8"))
                dims = m.get("model_dims", {})
                obs_spec = {"nq": dims.get("nq"), "nv": dims.get("nv")}
                action_spec = {"nu": dims.get("nu")}
        except Exception:
            pass

        skill = SkillSpec(
            skill_id=skill_id,
            display_name=display_name,
            robot_name=rname,
            task_description=task_description,
            active_variant=variant_id,
            obs_spec=obs_spec,
            action_spec=action_spec,
        )
    else:
        # Skill already exists — add new variant and promote it as active
        existing.active_variant = variant_id
        if display_name and not existing.display_name:
            existing.display_name = display_name
        if task_description and not existing.task_description:
            existing.task_description = task_description
        skill = existing

    save_skill(skill)
    log.info(
        f"promote: skill {rname}/{skill_id} variant {variant_id} "
        f"(best_reward={variant.best_reward:.3f})"
    )
    return variant
