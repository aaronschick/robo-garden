"""PolicyRunner: switcher composition for multiple skills.

A PolicyRunner holds one policy_fn per skill entry and switches between them
based on gamepad button triggers.  The currently active skill's policy_fn is
called on every step.

Usage::

    runner = PolicyRunner([
        PolicyEntry("walk_forward", "v_abc123", "button_0", policy_fn_walk),
        PolicyEntry("trot",         "v_def456", "button_1", policy_fn_trot),
    ])
    # Wrap as a plain obs→action callable (gamepad state provided via closure)
    gamepad_fn = lambda: gamepad_runner.latest()
    policy_fn  = runner.as_policy_fn(gamepad_fn)

    player = LivePolicyPlayer(policy_fn=policy_fn, ...)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class PolicyEntry:
    """One skill slot in a composed policy."""
    skill_id: str
    variant_id: str
    trigger: str          # "button_0", "button_1", etc.  "" = never triggered by button
    policy_fn: Callable   # obs → action


class PolicyRunner:
    """Switcher composition: routes obs to the currently active skill's policy_fn.

    On each ``step`` call the runner checks all trigger buttons in the
    supplied ``GamepadState``.  The first entry whose button is currently
    pressed becomes the new active entry.  If no button is pressed the active
    entry is held until another button fires.

    The first entry in the list is the default (active on startup with no
    button press required).
    """

    def __init__(self, entries: list[PolicyEntry]) -> None:
        if not entries:
            raise ValueError("PolicyRunner requires at least one entry")
        self._entries = list(entries)
        self._active_idx: int = 0

    # ------------------------------------------------------------------
    # Core step
    # ------------------------------------------------------------------

    def step(
        self,
        obs: np.ndarray,
        gamepad_state=None,  # GamepadState | None
    ) -> np.ndarray:
        """Check triggers, maybe switch active entry, return action for obs."""
        if gamepad_state is not None and gamepad_state.connected:
            for i, entry in enumerate(self._entries):
                if entry.trigger and gamepad_state.buttons.get(entry.trigger, False):
                    if i != self._active_idx:
                        log.debug(
                            f"PolicyRunner: switched to skill {entry.skill_id!r} "
                            f"via {entry.trigger}"
                        )
                    self._active_idx = i
                    break

        active = self._entries[self._active_idx]
        try:
            return np.asarray(active.policy_fn(obs), dtype=np.float32).reshape(-1)
        except Exception as exc:
            log.debug(f"PolicyRunner: policy_fn error for {active.skill_id!r} — {exc}")
            return np.zeros(1, dtype=np.float32)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def as_policy_fn(
        self,
        gamepad_state_fn: Callable | None = None,
    ) -> Callable[[np.ndarray], np.ndarray]:
        """Return an ``obs → action`` closure compatible with LivePolicyPlayer.

        ``gamepad_state_fn`` is called on every step to get the latest
        GamepadState (e.g. ``lambda: gamepad_runner.latest()``).
        Pass None to disable trigger switching (default entry is always active).
        """
        runner = self

        def _fn(obs: np.ndarray) -> np.ndarray:
            gs = gamepad_state_fn() if gamepad_state_fn is not None else None
            return runner.step(obs, gs)

        return _fn

    @property
    def active_skill_id(self) -> str:
        return self._entries[self._active_idx].skill_id

    @property
    def active_idx(self) -> int:
        return self._active_idx

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_policy_spec(
        cls,
        spec: "PolicySpec",  # type: ignore[name-defined]
        load_fn: Callable,  # load_policy_fn(VariantSpec) -> Callable
    ) -> "PolicyRunner":
        """Build a PolicyRunner from a PolicySpec, loading each variant's policy.

        ``load_fn`` is called once per entry (e.g. ``skills.inference.load_policy_fn``).
        """
        from robo_garden.skills.registry import get_variant

        entries: list[PolicyEntry] = []
        for ref in spec.skills:
            variant = get_variant(spec.robot_name, ref.skill_id, ref.variant_id)
            if variant is None:
                log.warning(
                    f"PolicyRunner.from_policy_spec: variant "
                    f"{spec.robot_name}/{ref.skill_id}/{ref.variant_id} not found, skipping"
                )
                continue
            pfn = load_fn(variant)
            entries.append(PolicyEntry(
                skill_id=ref.skill_id,
                variant_id=ref.variant_id,
                trigger=ref.trigger,
                policy_fn=pfn,
            ))

        if not entries:
            raise ValueError(
                f"PolicyRunner: no loadable variants in policy {spec.policy_name!r}"
            )
        return cls(entries)
