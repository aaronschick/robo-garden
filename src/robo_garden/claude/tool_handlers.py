"""Dispatch Claude tool_use blocks to local execution."""

from __future__ import annotations

import logging
from collections import OrderedDict
from typing import Any

from robo_garden.core.simulation import SimulationResult  # noqa: F401 – for type reference

log = logging.getLogger(__name__)

# Registry of tool name -> handler function
_HANDLERS: dict[str, Any] = {}

# Bounded caches — evict oldest when full
_MAX_SIM_RESULTS = 20
_MAX_REWARD_FNS = 50

_sim_results: OrderedDict[str, SimulationResult] = OrderedDict()
_reward_fns: OrderedDict = OrderedDict()


def get_reward_fn(reward_id: str):
    """Retrieve a stored reward function by ID (LRU: moves to end on access)."""
    rf = _reward_fns.get(reward_id)
    if rf is not None:
        _reward_fns.move_to_end(reward_id)
    return rf


def register_handler(tool_name: str):
    """Decorator to register a tool handler function."""
    def decorator(fn):
        _HANDLERS[tool_name] = fn
        return fn
    return decorator


def dispatch_tool(tool_name: str, tool_input: dict) -> dict:
    """Dispatch a tool call to its registered handler.

    Returns a dict that will be JSON-serialized as the tool_result content.
    """
    handler = _HANDLERS.get(tool_name)
    if handler is None:
        log.error(f"Unknown tool: {tool_name}")
        return {"error": f"Unknown tool: {tool_name}"}

    try:
        return handler(tool_input)
    except Exception as e:
        log.exception(f"Tool {tool_name} failed")
        return {"error": str(e)}


# --- Handler implementations ---


@register_handler("generate_robot")
def handle_generate_robot(input: dict) -> dict:
    """Validate Claude-generated robot XML (MJCF or URDF), save to workspace, notify Isaac Sim."""
    from robo_garden.core.formats import detect_format, validate_robot_xml, model_info
    from robo_garden.core.robot import Robot
    from robo_garden.config import ROBOTS_DIR
    from robo_garden.isaac import get_bridge

    # Accept robot_xml (new) or mjcf_xml (backward compat)
    robot_xml = input.get("robot_xml") or input.get("mjcf_xml", "")
    name = input.get("name", "unnamed")
    fmt = input.get("format") or detect_format(robot_xml)

    result = validate_robot_xml(robot_xml)
    if not result.valid:
        return {"success": False, "errors": result.errors}

    info = model_info(result.model)

    # Save to workspace with format-appropriate extension
    extension = ".urdf" if fmt == "urdf" else ".xml"
    robot_path = Robot(name=name, mjcf_xml=robot_xml).save(ROBOTS_DIR, extension=extension)

    # Resolve actuator assignments
    buildability = None
    actuator_assignments_input = input.get("actuator_assignments", [])
    act_assignments = []
    if actuator_assignments_input:
        from robo_garden.building.actuators import find_actuator
        from robo_garden.building.models import ActuatorAssignment

        unresolved = []
        for a in actuator_assignments_input:
            actuator = find_actuator(a["actuator_id"])
            if actuator:
                act_assignments.append(ActuatorAssignment(
                    joint_name=a["joint_name"],
                    actuator=actuator,
                    gear_ratio=float(a.get("gear_ratio", 1.0)),
                ))
            else:
                unresolved.append(a["actuator_id"])

        if unresolved:
            result.warnings.append(
                f"Actuator IDs not found in catalog: {', '.join(unresolved)}"
            )

    # Resolve material assignments
    material_assignments_input = input.get("material_assignments", [])
    mat_assignments = []
    if material_assignments_input:
        from robo_garden.building.materials import find_material
        from robo_garden.building.models import MaterialAssignment

        unresolved_mats = []
        for m in material_assignments_input:
            material = find_material(m["material_id"])
            if material:
                mat_assignments.append(MaterialAssignment(
                    link_name=m["link_name"],
                    material=material,
                ))
            else:
                unresolved_mats.append(m["material_id"])

        if unresolved_mats:
            result.warnings.append(
                f"Material IDs not found in catalog: {', '.join(unresolved_mats)}"
            )

    # Run buildability check if any assignments provided
    if act_assignments or mat_assignments:
        from robo_garden.building.validator import validate_buildability

        report = validate_buildability(
            result.model,
            act_assignments,
            material_assignments=mat_assignments if mat_assignments else None,
        )
        buildability = {
            "passed": report.passed,
            "errors": report.errors,
            "warnings": report.warnings,
            "total_mass_kg": report.total_mass_kg,
            "total_cost_usd": report.total_cost_usd,
        }

    # Notify Isaac Sim bridge (no-op if not connected)
    bridge = get_bridge()
    bridge.send_robot(name, robot_path, fmt=fmt)

    return {
        "success": True,
        "robot_name": name,
        "format": fmt,
        "model_info": info,
        "warnings": result.warnings,
        "robot_path": str(robot_path),
        "buildability": buildability,
    }


@register_handler("simulate")
def handle_simulate(input: dict) -> dict:
    """Run a physics simulation and optionally render to MP4."""
    import mujoco
    from robo_garden.core.simulation import simulate
    from robo_garden.config import ROBOTS_DIR, RENDERS_DIR
    from robo_garden.isaac import get_bridge

    robot_name = input.get("robot_name", "")
    duration = float(input.get("duration_seconds", 2.0))
    render_video = bool(input.get("render_video", False))

    # Try .xml (MJCF) then .urdf — whichever was saved by generate_robot
    xml_path = ROBOTS_DIR / f"{robot_name}.xml"
    urdf_path = ROBOTS_DIR / f"{robot_name}.urdf"
    if xml_path.exists():
        robot_path = xml_path
    elif urdf_path.exists():
        robot_path = urdf_path
    else:
        return {
            "success": False,
            "error": f"Robot '{robot_name}' not found in workspace. Call generate_robot first.",
        }

    model = mujoco.MjModel.from_xml_path(str(robot_path))
    result = simulate(model, duration=duration)

    # Store result so evaluate() can retrieve it (bounded LRU cache)
    _sim_results[robot_name] = result
    if len(_sim_results) > _MAX_SIM_RESULTS:
        _sim_results.popitem(last=False)

    # Stream simulation frames to Isaac Sim (no-op if not connected)
    bridge = get_bridge()
    bridge.stream_simulation(result, robot_name)

    response: dict = {
        "success": True,
        "robot_name": robot_name,
        "simulation_id": robot_name,
        "duration": result.duration,
        "num_steps": result.num_steps,
        "stable": result.stable,
        "diverged": result.diverged,
        "summary": result.summary,
    }

    if render_video:
        video_path = RENDERS_DIR / f"{robot_name}_sim_{int(duration)}s.mp4"
        try:
            import imageio
            renderer = mujoco.Renderer(model, height=480, width=640)
            data = mujoco.MjData(model)
            mujoco.mj_resetData(model, data)

            fps = 30
            timestep = model.opt.timestep
            steps_per_frame = max(1, int(1.0 / (fps * timestep)))
            total_steps = int(duration / timestep)

            frames = []
            for step in range(total_steps):
                mujoco.mj_step(model, data)
                if step % steps_per_frame == 0:
                    renderer.update_scene(data)
                    frames.append(renderer.render().copy())

            imageio.mimsave(str(video_path), frames, fps=fps)
            response["video_path"] = str(video_path)
            response["video_frames"] = len(frames)
        except ImportError:
            response["video_error"] = "imageio not installed. Run: pip install imageio[ffmpeg]"
        except Exception as exc:
            response["video_error"] = str(exc)

    return response


@register_handler("evaluate")
def handle_evaluate(input: dict) -> dict:
    """Evaluate simulation results against criteria."""
    robot_name = input.get("simulation_id", "")
    result = _sim_results.get(robot_name)
    if result is None:
        return {
            "success": False,
            "error": (
                f"No simulation found for '{robot_name}'. "
                "Call simulate first and use the returned simulation_id."
            ),
        }

    metrics_requested = input.get("metrics", [])
    computed: dict = {}

    if "stability" in metrics_requested:
        computed["stability"] = 1.0 if result.stable else 0.0
    if "diverged" in metrics_requested:
        computed["diverged"] = result.diverged
    if "com_height" in metrics_requested:
        computed["com_height"] = result.summary.get("final_com_z", 0.0)
    if "max_velocity" in metrics_requested:
        computed["max_velocity"] = result.summary.get("max_velocity", 0.0)
    if "forward_velocity" in metrics_requested:
        computed["forward_velocity"] = result.summary.get("forward_velocity", 0.0)
    if "energy" in metrics_requested:
        computed["energy"] = result.summary.get("energy", 0.0)
    if "energy_efficiency" in metrics_requested:
        computed["energy_efficiency"] = result.summary.get("energy_efficiency", 0.0)
    if "duration" in metrics_requested:
        computed["duration"] = result.duration
    if "num_steps" in metrics_requested:
        computed["num_steps"] = result.num_steps

    success_criteria = input.get("success_criteria", {})
    criteria_met = {
        k: float(computed.get(k, 0)) >= float(v)
        for k, v in success_criteria.items()
    }

    return {
        "success": True,
        "robot_name": robot_name,
        "metrics": computed,
        "criteria_met": criteria_met,
        "all_criteria_passed": all(criteria_met.values()) if criteria_met else True,
    }


@register_handler("generate_environment")
def handle_generate_environment(input: dict) -> dict:
    """Validate Claude-generated environment MJCF and save to workspace."""
    from robo_garden.core.formats import validate_robot_xml, model_info
    from robo_garden.config import ENVIRONMENTS_DIR

    name = input.get("name", "unnamed")
    mjcf_xml = input.get("mjcf_xml", "")

    result = validate_robot_xml(mjcf_xml)
    if not result.valid:
        return {"success": False, "errors": result.errors}

    info = model_info(result.model)
    path = ENVIRONMENTS_DIR / f"{name}.xml"
    path.write_text(mjcf_xml)

    return {
        "success": True,
        "env_name": name,
        "env_path": str(path),
        "model_info": info,
        "warnings": result.warnings,
    }


@register_handler("generate_reward")
def handle_generate_reward(input: dict) -> dict:
    """Validate and register a reward function."""
    from uuid import uuid4
    from robo_garden.rewards.reward_runner import compile_reward_function
    from robo_garden.rewards.models import RewardFunction

    task_description = input.get("task_description", "")
    reward_code = input.get("reward_code", "")

    try:
        compile_reward_function(reward_code)
    except Exception as e:
        return {"success": False, "error": str(e)}

    reward_id = f"reward_{uuid4().hex[:8]}"
    _reward_fns[reward_id] = RewardFunction(
        code=reward_code,
        task_description=task_description,
    )
    if len(_reward_fns) > _MAX_REWARD_FNS:
        _reward_fns.popitem(last=False)

    return {
        "success": True,
        "reward_function_id": reward_id,
        "task_description": task_description,
        "signature_valid": True,
    }


@register_handler("train")
def handle_train(input: dict) -> dict:
    """Launch an RL training run using MuJoCo MJX + Brax PPO."""
    from robo_garden.training.mujoco_engine import MuJoCoMJXEngine
    from robo_garden.training.models import TrainingConfig
    from robo_garden.config import ROBOTS_DIR, ENVIRONMENTS_DIR
    from robo_garden.isaac import get_bridge

    robot_name = input.get("robot_name", "")
    robot_path = ROBOTS_DIR / f"{robot_name}.xml"
    if not robot_path.exists():
        robot_path = ROBOTS_DIR / f"{robot_name}.urdf"
    if not robot_path.exists():
        return {"success": False, "error": f"Robot '{robot_name}' not found. Call generate_robot first."}

    env_name = input.get("environment_name", "")
    env_mjcf = ""
    if env_name:
        env_path = ENVIRONMENTS_DIR / f"{env_name}.xml"
        if env_path.exists():
            env_mjcf = env_path.read_text()
        else:
            log.warning(f"Environment '{env_name}' not found, training without environment MJCF")

    reward_fn_code = ""
    reward_id = input.get("reward_function_id", "")
    if reward_id:
        rf = get_reward_fn(reward_id)
        if rf:
            reward_fn_code = rf.code
        else:
            log.warning(f"Reward function '{reward_id}' not found, using default reward")

    config = TrainingConfig(
        algorithm=input.get("algorithm", "ppo"),
        num_envs=int(input.get("num_envs", 128)),
        total_timesteps=int(input["total_timesteps"]),
    )

    curriculum_config = None
    n_stages = input.get("curriculum_stages")
    if n_stages is not None:
        from robo_garden.training.models import CurriculumConfig, CurriculumStage
        stages = [CurriculumStage(name=f"stage_{i}") for i in range(int(n_stages))]
        curriculum_config = CurriculumConfig(stages=stages)

    bridge = get_bridge()
    updates: list[dict] = []

    def _progress(timestep: int, metrics: dict) -> None:
        mean_reward = float(metrics.get("eval/episode_reward", 0))
        updates.append({"timestep": timestep, "mean_reward": mean_reward})
        bridge.send_training_update(robot_name, timestep, mean_reward)

    engine = MuJoCoMJXEngine()
    engine.setup(robot_path.read_text(), env_mjcf, config, curriculum_config=curriculum_config)
    result = engine.train(reward_fn_code=reward_fn_code, callback=_progress)
    engine.cleanup()

    return {
        "success": True,
        "robot_name": robot_name,
        "best_reward": result.best_reward,
        "training_time_seconds": result.training_time_seconds,
        "reward_curve": result.reward_curve[-10:],
        "recent_updates": updates[-5:],
        "isaac_connected": bridge.connected,
    }


@register_handler("query_catalog")
def handle_query_catalog(input: dict) -> dict:
    """Search actuator/material/robot catalogs."""
    from robo_garden.building.actuators import query_actuators, find_actuator
    from robo_garden.building.materials import query_materials, find_material
    from dataclasses import asdict

    catalog = input.get("catalog", "")
    query = input.get("query", "").lower()
    filters = input.get("filters", {})
    limit = int(input.get("limit", 10))

    if catalog == "actuators":
        results = query_actuators(
            min_torque_nm=filters.get("min_torque_nm"),
            max_weight_g=filters.get("max_weight_g"),
            actuator_type=filters.get("type"),
            max_price_usd=filters.get("max_price_usd"),
        )
        # Filter by natural language query (name/id contains keywords)
        if query:
            keywords = query.split()
            results = [
                a for a in results
                if any(kw in a.name.lower() or kw in a.id.lower() for kw in keywords)
            ]
        items = [asdict(a) for a in results[:limit]]
        return {"catalog": "actuators", "count": len(items), "results": items}

    elif catalog == "materials":
        results = query_materials(
            printable=filters.get("printable"),
            min_strength_mpa=filters.get("min_strength_mpa"),
            material_type=filters.get("type"),
        )
        if query:
            keywords = query.split()
            results = [
                m for m in results
                if any(kw in m.name.lower() or kw in m.id.lower() for kw in keywords)
            ]
        items = [asdict(m) for m in results[:limit]]
        return {"catalog": "materials", "count": len(items), "results": items}

    elif catalog == "robots":
        import importlib
        import robot_descriptions

        all_names = list(robot_descriptions.DESCRIPTIONS.keys())

        if query:
            keywords = query.split()
            all_names = [
                n for n in all_names
                if any(kw in n.lower() for kw in keywords)
            ]

        items = []
        for name in all_names:
            if len(items) >= limit:
                break
            try:
                mod = importlib.import_module(f"robot_descriptions.{name}")
                mjcf_path = str(getattr(mod, "MJCF_PATH", None) or "")
                urdf_path = str(getattr(mod, "URDF_PATH", None) or "")
                items.append({
                    "name": name,
                    "has_mjcf": bool(mjcf_path),
                    "has_urdf": bool(urdf_path),
                    "mjcf_path": mjcf_path or None,
                    "urdf_path": urdf_path or None,
                })
            except Exception:
                pass

        return {"catalog": "robots", "count": len(items), "results": items}

    else:
        return {"error": f"Unknown catalog: '{catalog}'. Use 'actuators', 'materials', or 'robots'."}
