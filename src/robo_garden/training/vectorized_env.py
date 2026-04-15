"""MJX batched environment using jax.vmap for GPU-parallel simulation.

This is the most performance-critical module in the project. It creates N copies
of the simulation state and uses jax.vmap to step all environments in parallel
on the GPU.

Target: 64-256 parallel environments on RTX 3070 (8GB VRAM).

TODO (Phase 4): Full implementation with:
- jax.vmap over mjx.step for batched physics
- Vectorized observation extraction
- Vectorized reward computation
- Auto-reset on episode termination
- Gymnasium VectorEnv compatibility
"""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)


class MJXVectorizedEnv:
    """Vectorized MuJoCo environment using MJX + JAX for GPU parallelism.

    This wraps mjx.step with jax.vmap to run N environments simultaneously.
    All data stays on GPU (as JAX arrays) to avoid CPU-GPU transfer overhead.
    """

    def __init__(self, mjcf_xml: str, num_envs: int = 128, seed: int = 0):
        self.mjcf_xml = mjcf_xml
        self.num_envs = num_envs
        self.seed = seed
        self._initialized = False

    def initialize(self):
        """JIT-compile the vectorized step function.

        This is called lazily on first reset() to avoid slow import-time compilation.
        """
        import jax
        import jax.numpy as jnp
        import mujoco
        from mujoco import mjx

        self.model = mujoco.MjModel.from_xml_string(self.mjcf_xml)
        self.mjx_model = mjx.put_model(self.model)

        # Create initial data and batch it
        data = mujoco.MjData(self.model)
        self.mjx_data = mjx.put_data(self.model, data)

        # Vectorize the step function
        self._step_fn = jax.jit(jax.vmap(mjx.step, in_axes=(None, 0)))

        self._initialized = True
        log.info(f"MJX vectorized env initialized: {self.num_envs} envs on {jax.devices()}")

    def reset(self):
        """Reset all environments. Returns batched initial observations."""
        if not self._initialized:
            self.initialize()
        # TODO: Implement batched reset with random initial states
        raise NotImplementedError("Phase 4: Implement batched reset")

    def step(self, actions):
        """Step all environments with batched actions. Returns (obs, rewards, dones, infos)."""
        # TODO: Implement batched step
        raise NotImplementedError("Phase 4: Implement batched step")
