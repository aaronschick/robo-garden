"""Tool gating by session phase.

Design Studio and Training Gym expose different subsets of Claude tools.
The `Session.phase` attribute (``"design"`` or ``"training"``) controls which
subset is advertised to Claude on each turn. The ``approve_for_training``
tool is the only bridge that flips phase from design to training.

Exports:
    DESIGN_TOOL_NAMES   — set of tool names available in the design phase
    TRAINING_TOOL_NAMES — set of tool names available in the training phase
    tools_for_phase(phase, all_tools) — returns the filtered tool list

Both phases share some tools (query_catalog, evaluate) — anything that is
purely read-only / inspection.
"""

from __future__ import annotations

# Tools usable while iterating on the robot design.  `train` and
# `generate_reward` are intentionally absent: the user must click
# "Promote to Training" in the Studio UI (or Claude must call
# `approve_for_training`) before training tools unlock.
DESIGN_TOOL_NAMES: frozenset[str] = frozenset({
    "generate_robot",
    "simulate",
    "evaluate",
    "generate_environment",
    "query_catalog",
    "approve_for_training",
})

# Tools usable once a design has been approved.  The design tools are kept
# available because iterating on the reward function often surfaces problems
# that require small robot tweaks, but `approve_for_training` is retained so
# Claude can re-approve after edits.
TRAINING_TOOL_NAMES: frozenset[str] = frozenset({
    "generate_robot",
    "simulate",
    "evaluate",
    "generate_environment",
    "generate_reward",
    "train",
    "review_run",
    "promote_skill",
    "query_catalog",
    "approve_for_training",
})


def tools_for_phase(phase: str, all_tools: list[dict]) -> list[dict]:
    """Return the subset of *all_tools* permitted in *phase*.

    Unknown phases fall back to design (fail-closed).
    """
    if phase == "training":
        allowed = TRAINING_TOOL_NAMES
    else:
        allowed = DESIGN_TOOL_NAMES
    return [t for t in all_tools if t.get("name") in allowed]
