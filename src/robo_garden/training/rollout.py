"""Sample a short on-policy rollout for live viewport playback during training.

During a training run the Studio UI wants to animate the *current* policy in
Isaac Sim so the user can see the behaviour improving over time. Sampling a
full episode every progress tick is too expensive, so we cap rollouts at a
small, fixed number of frames and let ``handle_train`` decide how often to
invoke us.

The caller passes a ``policy_apply`` callable with the signature::

    policy_apply(obs: np.ndarray) -> np.ndarray

where ``obs`` is a 1-D observation (same layout as the training env: qpos +
qvel) and the returned action is a 1-D array in [-1, 1] of shape (nu,).

Returns a structured ``Rollout`` dataclass holding the qpos frames and the
timesteps (in simulated seconds) so ``bridge.stream_qpos_batch`` can forward
them straight to Isaac Sim.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

import mujoco
import numpy as np

log = logging.getLogger(__name__)


@dataclass
class Rollout:
    qpos: np.ndarray         # shape (num_frames, nq)
    timesteps: list[float]   # simulated seconds, len == num_frames
    rewards: np.ndarray      # shape (num_frames,), cumulative per-step reward
    success: bool            # False if the rollout diverged (NaN qpos)


def sample_rollout(
    mjcf_xml: str,
    policy_apply: Callable[[np.ndarray], np.ndarray] | None,
    *,
    num_frames: int = 150,
    seed: int = 0,
    reward_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], float] | None = None,
) -> Rollout:
    """Play one short episode with ``policy_apply`` and return qpos frames.

    If ``policy_apply`` is None, a zero-action rollout is returned (useful
    before the policy has learned anything so the UI still animates).
    """
    model = mujoco.MjModel.from_xml_string(mjcf_xml)
    data = mujoco.MjData(model)
    rng = np.random.default_rng(seed)
    mujoco.mj_resetData(model, data)

    nq = model.nq
    nv = model.nv
    nu = model.nu

    qpos_frames = np.zeros((num_frames, nq), dtype=np.float32)
    rewards = np.zeros(num_frames, dtype=np.float32)
    timesteps: list[float] = []
    ctrl_range = model.actuator_ctrlrange

    def _obs() -> np.ndarray:
        return np.concatenate([data.qpos, data.qvel]).astype(np.float32)

    diverged = False
    for t in range(num_frames):
        prev = _obs()
        if policy_apply is not None:
            try:
                raw = np.asarray(policy_apply(prev), dtype=np.float32).reshape(-1)[:nu]
            except Exception as exc:
                log.debug(f"sample_rollout: policy_apply failed: {exc}")
                raw = np.zeros(nu, dtype=np.float32)
        else:
            raw = rng.uniform(-0.1, 0.1, nu).astype(np.float32)

        if ctrl_range.shape[0] == nu:
            lo, hi = ctrl_range[:, 0], ctrl_range[:, 1]
            scaled = lo + (np.clip(raw, -1.0, 1.0) + 1.0) * 0.5 * (hi - lo)
        else:
            scaled = raw

        data.ctrl[:nu] = scaled
        mujoco.mj_step(model, data)

        qpos_frames[t] = data.qpos.astype(np.float32)
        timesteps.append(float(t * model.opt.timestep))

        if reward_fn is not None:
            try:
                rewards[t] = float(reward_fn(prev, raw, _obs()))
            except Exception:
                rewards[t] = 0.0

        if np.any(np.isnan(data.qpos)):
            diverged = True
            qpos_frames = qpos_frames[: t + 1]
            rewards = rewards[: t + 1]
            timesteps = timesteps[: t + 1]
            break

    return Rollout(
        qpos=qpos_frames,
        timesteps=timesteps,
        rewards=rewards,
        success=not diverged,
    )
