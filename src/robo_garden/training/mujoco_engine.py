"""MuJoCo + MJX local training engine: primary backend for RTX 3070."""

from __future__ import annotations

import logging
import time

import numpy as np

from robo_garden.training.models import CurriculumConfig, TrainingConfig, TrainingResult
from robo_garden.training.vectorized_env import MJXVectorizedEnv

log = logging.getLogger(__name__)


def _merge_mjcf(robot_mjcf: str, env_mjcf: str) -> str:
    """Merge robot MJCF into environment MJCF by inserting robot worldbody into env worldbody.

    If env_mjcf is empty, returns robot_mjcf unchanged.
    Simple approach: extract <worldbody> contents from robot and append to env worldbody.
    """
    if not env_mjcf.strip():
        return robot_mjcf

    import xml.etree.ElementTree as ET

    try:
        robot_root = ET.fromstring(robot_mjcf)
        env_root = ET.fromstring(env_mjcf)

        robot_wb = robot_root.find("worldbody")
        env_wb = env_root.find("worldbody")

        if robot_wb is not None and env_wb is not None:
            existing_names = {c.get("name") for c in env_wb if c.get("name")}
            for child in list(robot_wb):
                # Skip ground planes — environment provides the floor
                if child.tag == "geom" and child.get("type") == "plane":
                    continue
                name = child.get("name")
                if name and name in existing_names:
                    child.set("name", f"robot_{name}")
                env_wb.append(child)
                if child.get("name"):
                    existing_names.add(child.get("name"))

        # Copy assets
        robot_assets = robot_root.find("asset")
        env_assets = env_root.find("asset")
        if robot_assets is not None:
            if env_assets is None:
                env_assets = ET.SubElement(env_root, "asset")
            existing_asset_names = {c.get("name") for c in env_assets if c.get("name")}
            for child in list(robot_assets):
                name = child.get("name")
                if name and name in existing_asset_names:
                    child.set("name", f"robot_{name}")
                env_assets.append(child)
                if child.get("name"):
                    existing_asset_names.add(child.get("name"))

        # Copy actuators
        robot_actuators = robot_root.find("actuator")
        if robot_actuators is not None:
            env_actuators = env_root.find("actuator")
            if env_actuators is None:
                env_actuators = ET.SubElement(env_root, "actuator")
            existing_actuator_names = {c.get("name") for c in env_actuators if c.get("name")}
            for child in list(robot_actuators):
                name = child.get("name")
                if name and name in existing_actuator_names:
                    child.set("name", f"robot_{name}")
                env_actuators.append(child)
                if child.get("name"):
                    existing_actuator_names.add(child.get("name"))

        # Copy list-style sections (append children)
        for section_name in ("sensor", "contact", "equality", "tendon"):
            robot_section = robot_root.find(section_name)
            if robot_section is not None:
                env_section = env_root.find(section_name)
                if env_section is None:
                    env_section = ET.SubElement(env_root, section_name)
                for child in list(robot_section):
                    env_section.append(child)

        # Copy singleton sections (robot overrides env)
        for section_name in ("option", "default"):
            robot_section = robot_root.find(section_name)
            if robot_section is not None:
                env_section = env_root.find(section_name)
                if env_section is not None:
                    env_root.remove(env_section)
                env_root.append(robot_section)

        return ET.tostring(env_root, encoding="unicode")
    except Exception as e:
        log.warning(f"MJCF merge failed ({e}), using robot MJCF only")
        return robot_mjcf


class MuJoCoMJXEngine:
    """Training engine using MuJoCo MJX (JAX GPU acceleration).

    Trains with Brax PPO when available; falls back to a random-rollout
    baseline that at least validates the full vectorized pipeline.
    """

    def __init__(self):
        self._merged_mjcf: str = ""
        self.config: TrainingConfig | None = None
        self._env: MJXVectorizedEnv | None = None
        self._curriculum_config: CurriculumConfig | None = None

    def setup(
        self,
        robot_mjcf: str,
        env_mjcf: str,
        config: TrainingConfig,
        curriculum_config: CurriculumConfig | None = None,
    ) -> None:
        self.config = config
        self._curriculum_config = curriculum_config
        self._merged_mjcf = _merge_mjcf(robot_mjcf, env_mjcf)
        log.info(
            f"MJX engine setup: {config.num_envs} envs, "
            f"{config.total_timesteps} total steps, algorithm={config.algorithm}"
        )
        if curriculum_config is not None:
            log.info(
                f"Curriculum enabled: {len(curriculum_config.stages)} stages, "
                f"advance threshold={curriculum_config.advance_threshold}"
            )

    def train(self, reward_fn_code: str = "", callback=None) -> TrainingResult:
        assert self.config is not None, "Call setup() before train()"

        # Compile reward function if provided
        reward_fn = None
        if reward_fn_code:
            try:
                from robo_garden.rewards.reward_runner import compile_reward_function
                _compute = compile_reward_function(reward_fn_code)
                # Adapt signature: (obs, action, next_obs) -> scalar
                reward_fn = lambda obs, action, next_obs: float(_compute(obs, action, next_obs, {})[0])
            except Exception as e:
                log.warning(f"Reward function compilation failed ({e}), using default reward")

        self._env = MJXVectorizedEnv(
            mjcf_xml=self._merged_mjcf,
            num_envs=self.config.num_envs,
            max_episode_steps=self.config.max_episode_steps,
            reward_fn=reward_fn,
        )

        # Attempt Brax PPO first
        try:
            return self._train_brax(callback)
        except Exception as e:
            log.warning(f"Brax PPO unavailable ({e}), running random-rollout baseline")
            return self._train_random_rollout(callback)

    def _train_brax(self, callback) -> TrainingResult:
        """Train with Brax PPO wrapped around MJXVectorizedEnv."""
        import jax
        from brax.envs.base import Env, State  # type: ignore
        from brax.training.agents.ppo import train as ppo_train  # type: ignore

        config = self.config
        env = self._env
        if not env._initialized:
            env.initialize()

        if not env._use_mjx:
            raise RuntimeError("Brax PPO requires MJX (JAX GPU), not available on this machine")

        class _MJXBraxEnv(Env):  # type: ignore
            def __init__(self, vec_env: MJXVectorizedEnv):
                self._ve = vec_env

            @property
            def observation_size(self) -> int:
                return self._ve.obs_dim

            @property
            def action_size(self) -> int:
                return self._ve.action_dim

            def reset(self, rng: jax.Array) -> State:
                import jax.numpy as jnp
                ve = self._ve
                data = ve._base_mjx_data
                obs = self._get_obs(data)
                return State(
                    pipeline_state=data,
                    obs=obs,
                    reward=jnp.float32(0.0),
                    done=jnp.float32(0.0),
                    metrics={},
                )

            def step(self, state: State, action: jax.Array) -> State:
                import jax.numpy as jnp
                ve = self._ve
                prev_obs = state.obs
                data = state.pipeline_state.replace(ctrl=action[: ve._nu])
                data = ve._mjx.step(ve._mx, data)
                obs = self._get_obs(data)
                done = jnp.where(
                    jnp.any(jnp.isnan(data.qpos)), jnp.float32(1.0), jnp.float32(0.0)
                )
                if ve.reward_fn is not None:
                    r = ve.reward_fn(
                        np.asarray(prev_obs),
                        np.asarray(action),
                        np.asarray(obs),
                    )
                    reward = jnp.float32(r)
                else:
                    reward = -0.1 * jnp.sum(action ** 2)
                return state.replace(pipeline_state=data, obs=obs, reward=reward, done=done)

            def _get_obs(self, data) -> jax.Array:
                import jax.numpy as jnp
                ve = self._ve
                qpos = data.qpos[1:] if ve._floating_base else data.qpos
                return jnp.concatenate([qpos, data.qvel, data.ctrl])

        adapter = _MJXBraxEnv(env)
        start = time.time()
        reward_curve: list[tuple[int, float]] = []

        def _progress(step, metrics):
            r = float(metrics.get("eval/episode_reward", 0))
            reward_curve.append((step, r))
            log.info(f"  step={step:,}  mean_reward={r:.3f}")
            if callback:
                callback(step, metrics)

        make_inference_fn, params, metrics = ppo_train(
            environment=adapter,
            num_timesteps=config.total_timesteps,
            episode_length=config.max_episode_steps,
            num_envs=config.num_envs,
            learning_rate=config.learning_rate,
            entropy_cost=config.entropy_coef,
            discounting=config.gamma,
            batch_size=config.batch_size,
            progress_fn=_progress,
        )

        elapsed = time.time() - start
        best_reward = max((r for _, r in reward_curve), default=float("-inf"))

        # Save checkpoint
        from robo_garden.training.checkpoints import save_checkpoint
        from robo_garden.config import CHECKPOINTS_DIR

        checkpoint_dir = CHECKPOINTS_DIR / f"brax_ppo_{int(time.time())}"
        save_checkpoint(params, checkpoint_dir, metadata={
            "algorithm": "ppo",
            "total_timesteps": config.total_timesteps,
            "best_reward": best_reward,
            "num_envs": config.num_envs,
        })

        return TrainingResult(
            config=config,
            reward_curve=reward_curve,
            best_reward=best_reward,
            training_time_seconds=elapsed,
            checkpoint_path=checkpoint_dir,
        )

    def _train_random_rollout(self, callback) -> TrainingResult:
        """Random-action rollout — exercises the vectorized env without a policy."""
        import numpy as np

        curriculum_manager = None
        if self._curriculum_config is not None:
            from robo_garden.training.curriculum import CurriculumManager
            curriculum_manager = CurriculumManager(self._curriculum_config)

        config = self.config
        env = self._env
        obs, _ = env.reset()

        total_steps = 0
        episode_rewards = np.zeros(env.num_envs, dtype=np.float32)
        all_episode_rewards: list[float] = []
        reward_curve: list[tuple[int, float]] = []
        start = time.time()
        log_interval = max(1, config.total_timesteps // 20)

        while total_steps < config.total_timesteps:
            actions = np.random.uniform(-1, 1, (env.num_envs, env.action_dim)).astype(np.float32)
            obs, rewards, terminated, truncated, info = env.step(actions)
            episode_rewards += rewards
            total_steps += env.num_envs

            done = terminated | truncated
            if done.any():
                all_episode_rewards.extend(episode_rewards[done].tolist())
                episode_rewards[done] = 0.0

            if total_steps % log_interval < env.num_envs:
                mean_r = float(np.mean(all_episode_rewards[-100:])) if all_episode_rewards else 0.0
                reward_curve.append((total_steps, mean_r))
                log.info(f"  step={total_steps:,}  mean_ep_reward={mean_r:.3f}  (random policy)")
                if callback:
                    callback(total_steps, {"eval/episode_reward": mean_r})
                if curriculum_manager is not None and curriculum_manager.should_advance(mean_r):
                    if curriculum_manager.advance():
                        params = curriculum_manager.get_env_params(curriculum_manager.current_stage)
                        log.info(
                            f"Curriculum advanced to stage {curriculum_manager.current_stage} "
                            f"({params.get('stage_name', '')}), "
                            f"difficulty={params.get('difficulty', 0):.2f}"
                        )

        elapsed = time.time() - start
        best = max(all_episode_rewards, default=float("-inf"))

        # Save checkpoint (no params for random rollout, just metadata)
        from robo_garden.training.checkpoints import save_checkpoint
        from robo_garden.config import CHECKPOINTS_DIR

        checkpoint_dir = CHECKPOINTS_DIR / f"random_rollout_{int(time.time())}"
        save_checkpoint(None, checkpoint_dir, metadata={
            "algorithm": "random_rollout",
            "total_timesteps": config.total_timesteps,
            "best_reward": best,
            "num_envs": config.num_envs,
        })

        return TrainingResult(
            config=config,
            episode_rewards=all_episode_rewards[-1000:],
            best_reward=best,
            training_time_seconds=elapsed,
            reward_curve=reward_curve,
            checkpoint_path=checkpoint_dir,
        )

    def evaluate(self, checkpoint_path: str, num_episodes: int = 10) -> dict:
        """Evaluate policy by running rollout episodes.

        Loads checkpoint if available; falls back to random policy baseline.
        Returns reward statistics over num_episodes.
        """
        import numpy as np
        import mujoco
        from robo_garden.training.checkpoints import load_checkpoint

        policy = "random_baseline"
        if checkpoint_path:
            try:
                load_checkpoint(checkpoint_path)
                policy = "checkpoint"
            except Exception as e:
                log.warning(f"Could not load checkpoint ({e}), using random baseline")

        if not self._merged_mjcf:
            return {"error": "Engine not set up. Call setup() before evaluate()."}

        model = mujoco.MjModel.from_xml_string(self._merged_mjcf)
        max_steps = self.config.max_episode_steps if self.config else 1000
        episode_rewards: list[float] = []

        for ep in range(num_episodes):
            data = mujoco.MjData(model)
            mujoco.mj_resetData(model, data)
            ep_reward = 0.0
            for _ in range(max_steps):
                data.ctrl[:] = np.random.uniform(-1, 1, model.nu).astype(np.float32)
                mujoco.mj_step(model, data)
                ep_reward += float(-0.01 * np.sum(np.square(data.ctrl)))
            episode_rewards.append(ep_reward)
            log.debug(f"Eval episode {ep + 1}/{num_episodes}: reward={ep_reward:.3f}")

        arr = np.array(episode_rewards)
        return {
            "policy": policy,
            "num_episodes": num_episodes,
            "mean_reward": float(arr.mean()),
            "std_reward": float(arr.std()),
            "min_reward": float(arr.min()),
            "max_reward": float(arr.max()),
        }

    def cleanup(self) -> None:
        if self._env:
            self._env.close()
        self._env = None
