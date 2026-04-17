"""Skills and policy data model for Robo Garden.

A *skill* is a named behavior a specific robot body can perform, backed by one
or more trained *variants* (checkpoints).  A *policy* is a named composition of
skills (switcher, blend, sequence) that runs on the same robot body.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


@dataclass
class VariantSpec:
    """One trained checkpoint that implements a skill."""

    variant_id: str
    run_id: str
    algorithm: str
    best_reward: float
    total_timesteps: int
    # Relative to WORKSPACE_DIR (stored as str for JSON serialisability)
    checkpoint_path: str
    reward_function_id: str = ""
    environment_name: str = ""
    created_at_utc: str = ""
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.created_at_utc:
            self.created_at_utc = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict:
        return {
            "variant_id": self.variant_id,
            "run_id": self.run_id,
            "algorithm": self.algorithm,
            "best_reward": self.best_reward,
            "total_timesteps": self.total_timesteps,
            "checkpoint_path": self.checkpoint_path,
            "reward_function_id": self.reward_function_id,
            "environment_name": self.environment_name,
            "created_at_utc": self.created_at_utc,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "VariantSpec":
        return cls(
            variant_id=d["variant_id"],
            run_id=d.get("run_id", ""),
            algorithm=d.get("algorithm", ""),
            best_reward=float(d.get("best_reward", 0.0)),
            total_timesteps=int(d.get("total_timesteps", 0)),
            checkpoint_path=d.get("checkpoint_path", ""),
            reward_function_id=d.get("reward_function_id", ""),
            environment_name=d.get("environment_name", ""),
            created_at_utc=d.get("created_at_utc", ""),
            notes=d.get("notes", ""),
        )


@dataclass
class SkillSpec:
    """A named behavior a robot body can perform."""

    skill_id: str
    display_name: str
    robot_name: str
    task_description: str = ""
    tags: list[str] = field(default_factory=list)
    active_variant: str = ""
    obs_spec: dict = field(default_factory=dict)
    action_spec: dict = field(default_factory=dict)
    created_at_utc: str = ""

    def __post_init__(self) -> None:
        if not self.created_at_utc:
            self.created_at_utc = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict:
        return {
            "skill_id": self.skill_id,
            "display_name": self.display_name,
            "robot_name": self.robot_name,
            "task_description": self.task_description,
            "tags": self.tags,
            "active_variant": self.active_variant,
            "obs_spec": self.obs_spec,
            "action_spec": self.action_spec,
            "created_at_utc": self.created_at_utc,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SkillSpec":
        return cls(
            skill_id=d["skill_id"],
            display_name=d.get("display_name", d["skill_id"]),
            robot_name=d.get("robot_name", ""),
            task_description=d.get("task_description", ""),
            tags=list(d.get("tags", [])),
            active_variant=d.get("active_variant", ""),
            obs_spec=dict(d.get("obs_spec", {})),
            action_spec=dict(d.get("action_spec", {})),
            created_at_utc=d.get("created_at_utc", ""),
        )


@dataclass
class SkillEntry:
    """Flat summary of a skill + its active variant for display / SKILL_LIST."""

    robot_name: str
    skill_id: str
    display_name: str
    task_description: str
    active_variant_id: str
    best_reward: Optional[float]
    variant_count: int
    created_at_utc: str

    def to_dict(self) -> dict:
        return {
            "robot_name": self.robot_name,
            "skill_id": self.skill_id,
            "display_name": self.display_name,
            "task_description": self.task_description,
            "active_variant_id": self.active_variant_id,
            "best_reward": self.best_reward,
            "variant_count": self.variant_count,
            "created_at_utc": self.created_at_utc,
        }


@dataclass
class PolicySkillRef:
    skill_id: str
    variant_id: str
    trigger: str = ""  # gamepad button / axis name, or "" for default

    def to_dict(self) -> dict:
        return {"skill_id": self.skill_id, "variant_id": self.variant_id, "trigger": self.trigger}

    @classmethod
    def from_dict(cls, d: dict) -> "PolicySkillRef":
        return cls(
            skill_id=d["skill_id"],
            variant_id=d.get("variant_id", ""),
            trigger=d.get("trigger", ""),
        )


@dataclass
class PolicySpec:
    """A named composition of skills that runs on one robot body."""

    policy_name: str
    robot_name: str
    composition: str = "switcher"  # "switcher" | "blend" | "sequence"
    skills: list[PolicySkillRef] = field(default_factory=list)
    created_at_utc: str = ""

    def __post_init__(self) -> None:
        if not self.created_at_utc:
            self.created_at_utc = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict:
        return {
            "policy_name": self.policy_name,
            "robot_name": self.robot_name,
            "composition": self.composition,
            "skills": [s.to_dict() for s in self.skills],
            "created_at_utc": self.created_at_utc,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PolicySpec":
        return cls(
            policy_name=d["policy_name"],
            robot_name=d.get("robot_name", ""),
            composition=d.get("composition", "switcher"),
            skills=[PolicySkillRef.from_dict(s) for s in d.get("skills", [])],
            created_at_utc=d.get("created_at_utc", ""),
        )


__all__ = [
    "VariantSpec",
    "SkillSpec",
    "SkillEntry",
    "PolicySkillRef",
    "PolicySpec",
]
