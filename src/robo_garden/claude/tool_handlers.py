"""Dispatch Claude tool_use blocks to local execution."""

from __future__ import annotations

import logging
import os
from collections import OrderedDict
from typing import Any, Callable  # noqa: F401  Callable used by set_approval_callback signature

from robo_garden.core.simulation import SimulationResult  # noqa: F401 – for type reference

log = logging.getLogger(__name__)

# Registry of tool name -> handler function
_HANDLERS: dict[str, Any] = {}

# Bounded caches — evict oldest when full
_MAX_SIM_RESULTS = 20
_MAX_REWARD_FNS = 50

_sim_results: OrderedDict[str, SimulationResult] = OrderedDict()
_reward_fns: OrderedDict = OrderedDict()

# Catalog robot path registry: robot_name -> Path to MJCF on disk
# Used when generate_robot loads from robot_descriptions catalog instead of XML string.
_catalog_paths: dict[str, "Path"] = {}

# Callback invoked when approve_for_training succeeds.  Set by Session to flip
# its phase state.  Signature: (robot_name: str, environment_name: str,
# manifest_path: Path) -> None.
_on_approve: Any | None = None

# When True, the Studio runtime (src/robo_garden/studio.py) owns the Isaac Sim
# bridge lifecycle and will call bridge.send_robot itself from
# Studio._on_tool_side_effects.  In that case handle_generate_robot must NOT
# also send LOAD_ROBOT — otherwise Isaac Sim deletes and re-imports the robot,
# racing with the kinematic-playback path and frequently leaving the viewport
# empty.  Chat/gym modes leave this False, so the handler keeps notifying
# Isaac directly.
_studio_mode: bool = False


def set_studio_mode(enabled: bool) -> None:
    """Toggle Studio-ownership of Isaac Sim LOAD_ROBOT dispatch."""
    global _studio_mode
    _studio_mode = bool(enabled)


def set_approval_callback(cb) -> None:
    """Register a callback fired on successful approve_for_training.

    Used by `Session` to flip its `phase` attribute from "design" to
    "training" so subsequent turns advertise the training tools.
    """
    global _on_approve
    _on_approve = cb


# Optional live-training progress sink for TUI mode.
# Signature: (timestep: int, metrics: dict) -> None
# Set by ChatScreen before starting a chat turn that may invoke training.
_tui_train_progress: Any | None = None


def set_tui_train_progress(cb) -> None:
    """Register a TUI callback for live training progress updates.

    Called by ChatScreen so TrainingScreen updates in real-time rather than
    only after the full training run completes.  Pass None to unregister.
    """
    global _tui_train_progress
    _tui_train_progress = cb


def get_sim_result(robot_name: str):
    """Retrieve the most recent SimulationResult for *robot_name* (or None).

    Exposed so callers outside the tool-dispatch path (e.g. Studio UI gate
    checklist) can inspect simulation state.
    """
    return _sim_results.get(robot_name)


def get_catalog_path(robot_name: str):
    """Return the catalog MJCF Path for *robot_name*, or None if not registered."""
    return _catalog_paths.get(robot_name)


def get_reward_fn(reward_id: str):
    """Retrieve a stored reward function by ID (LRU: moves to end on access)."""
    rf = _reward_fns.get(reward_id)
    if rf is not None:
        _reward_fns.move_to_end(reward_id)
    return rf


def _compute_model_dims_from_manifest(manifest: dict) -> dict:
    """Compute nq/nv/nu/obs_dim/action_dim from the files named in *manifest*.

    Used as a fallback when an older manifest was written before
    handle_approve_for_training started persisting model_dims.  Best-effort —
    returns {} on any failure.
    """
    from pathlib import Path as _P

    try:
        import mujoco
        from robo_garden.training.mujoco_engine import _merge_mjcf

        robot_path = _P(manifest.get("robot_path", ""))
        env_path = _P(manifest.get("environment_path", ""))
        if not robot_path.exists():
            return {}

        robot_xml = robot_path.read_text(encoding="utf-8")
        robot_name = manifest.get("robot_name", "")
        if robot_name in _catalog_paths:
            robot_xml = _absolutize_asset_paths(robot_xml, robot_path.parent)
        env_xml = env_path.read_text(encoding="utf-8") if env_path.exists() else ""
        merged = _merge_mjcf(robot_xml, env_xml)
        m = mujoco.MjModel.from_xml_string(merged)
        return {
            "nq": int(m.nq),
            "nv": int(m.nv),
            "nu": int(m.nu),
            "obs_dim": int(m.nq + m.nv),
            "action_dim": int(m.nu),
            "floating_base": bool(m.nq > m.nv),
        }
    except Exception as exc:
        log.debug(f"_compute_model_dims_from_manifest failed: {exc}")
        return {}


def _lookup_model_dims(robot_name: str) -> dict:
    """Best-effort lookup of (obs_dim, action_dim, nq, nv, nu) for *robot_name*.

    Reads the approval manifest written by handle_approve_for_training.  If
    the manifest predates the model_dims field (older runs), compute them on
    the fly from the files it references so the reward smoke-test always sees
    real shapes.  Returns {} only when no manifest exists.
    """
    import json
    from robo_garden.config import APPROVED_DIR

    if not robot_name:
        return {}
    for path in APPROVED_DIR.glob(f"{robot_name}__*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        dims = data.get("model_dims") or {}
        if dims and dims.get("obs_dim"):
            return dims
        # Manifest lacks model_dims — recover by rebuilding the model.
        return _compute_model_dims_from_manifest(data)
    return {}


def _latest_approved_robot() -> str:
    """Return the robot_name from the most-recently-written approval manifest.

    Used as a fallback when Claude calls generate_reward without robot_name.
    In practice there is at most one approval at a time (the Session gates it),
    so "most recent" is both unambiguous and correct.
    """
    import json
    from robo_garden.config import APPROVED_DIR

    manifests = sorted(
        APPROVED_DIR.glob("*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for path in manifests:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            name = data.get("robot_name", "")
            if name:
                return name
        except Exception:
            continue
    return ""


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


# --- Helpers ---

_SCENE_FLOOR_XML = """\
<mujoco>
  <include file="{robot_filename}"/>
  <worldbody>
    <light name="_scene_sun" pos="0 0 4" dir="0 0 -1" diffuse="1 1 1" castshadow="false"/>
    <geom name="_scene_floor" type="plane" size="50 50 0.05" rgba="0.75 0.8 0.85 1"
          contype="1" conaffinity="1" condim="3" friction="1 0.005 0.0001"/>
  </worldbody>
</mujoco>"""


def _absolutize_asset_paths(xml_text: str, base_dir: "Path") -> str:
    """Rewrite relative meshdir/texturedir paths to absolute so the XML can be
    compiled from a string (rather than from a file next to its assets).
    """
    import re
    from pathlib import Path

    def _fix(m: re.Match) -> str:
        attr, rel = m.group(1), m.group(2)
        if Path(rel).is_absolute():
            return m.group(0)
        abs_path = (Path(base_dir) / rel).resolve().as_posix()
        return f'{attr}="{abs_path}"'

    # Match meshdir="..." and texturedir="..."
    return re.sub(r'(meshdir|texturedir)="([^"]*)"', _fix, xml_text)


def _load_model_with_scene(robot_path: "Path") -> "mujoco.MjModel":
    """Load a MuJoCo model from *robot_path*, injecting a floor + light if the XML
    has no ground plane geom.

    For catalog robots (and any robot-only MJCF without a floor) this writes a
    thin wrapper XML next to the original file, loads it, then removes it.
    The wrapper uses a relative ``<include>`` so asset paths (meshdir, etc.)
    resolve correctly from the original file's directory.
    """
    import mujoco
    from pathlib import Path

    robot_path = Path(robot_path)

    # Quick check: does the XML mention a plane geom?
    xml_text = robot_path.read_text(encoding="utf-8", errors="replace")
    has_floor = 'type="plane"' in xml_text or "type='plane'" in xml_text

    if has_floor:
        return mujoco.MjModel.from_xml_path(str(robot_path))

    # Inject floor by writing a wrapper next to the original (so meshdir works)
    scene_xml = _SCENE_FLOOR_XML.format(robot_filename=robot_path.name)
    wrapper_path = robot_path.parent / f"_rg_scene_{robot_path.stem}.xml"
    try:
        wrapper_path.write_text(scene_xml, encoding="utf-8")
        return mujoco.MjModel.from_xml_path(str(wrapper_path))
    finally:
        wrapper_path.unlink(missing_ok=True)


# --- Handler implementations ---


@register_handler("generate_robot")
def handle_generate_robot(input: dict) -> dict:
    """Validate Claude-generated robot XML (MJCF or URDF), save to workspace, notify Isaac Sim."""
    import importlib
    from pathlib import Path
    from robo_garden.core.formats import detect_format, validate_robot_xml, model_info, load_mjcf_file
    from robo_garden.core.robot import Robot
    from robo_garden.config import ROBOTS_DIR
    from robo_garden.isaac import get_bridge

    name = input.get("name", "unnamed")
    catalog_name = input.get("catalog_name", "").strip()

    # --- Branch: load from robot_descriptions catalog ---
    if catalog_name:
        try:
            mod = importlib.import_module(f"robot_descriptions.{catalog_name}")
        except ModuleNotFoundError:
            return {
                "success": False,
                "errors": [
                    f"Catalog entry '{catalog_name}' not found in robot_descriptions. "
                    "Use query_catalog with catalog='robots' to list available names."
                ],
            }
        mjcf_path_str = getattr(mod, "MJCF_PATH", None)
        if not mjcf_path_str:
            return {
                "success": False,
                "errors": [f"Catalog entry '{catalog_name}' has no MJCF_PATH attribute."],
            }
        mjcf_path = Path(mjcf_path_str)
        if not mjcf_path.exists():
            return {
                "success": False,
                "errors": [f"MJCF file not found on disk: {mjcf_path}"],
            }

        try:
            model = load_mjcf_file(mjcf_path)
        except Exception as exc:
            return {"success": False, "errors": [f"MuJoCo failed to load catalog MJCF: {exc}"]}

        info = model_info(model)
        fmt = "mjcf"

        # Register in catalog path registry so simulate() can find it
        _catalog_paths[name] = mjcf_path

        # Notify Isaac Sim bridge (no-op if not connected).
        # In Studio mode the Studio runtime owns this dispatch; sending here
        # would cause a double-load race in the viewport.
        if not _studio_mode:
            bridge = get_bridge()
            bridge.send_robot(name, mjcf_path, fmt=fmt)

        return {
            "success": True,
            "robot_name": name,
            "format": fmt,
            "catalog_name": catalog_name,
            "mjcf_path": str(mjcf_path),
            "model_info": info,
            "warnings": [],
            "buildability": None,
            "note": (
                f"Loaded from robot_descriptions.{catalog_name}. "
                "The original MJCF file is used for simulation (mesh paths preserved)."
            ),
        }

    # --- Branch: Claude-generated XML ---
    # Accept robot_xml (new) or mjcf_xml (backward compat)
    robot_xml = input.get("robot_xml") or input.get("mjcf_xml", "")
    if not robot_xml:
        return {
            "success": False,
            "errors": ["Either robot_xml or catalog_name must be provided."],
        }

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

    # Notify Isaac Sim bridge (no-op if not connected).
    # In Studio mode the Studio runtime forwards this via _on_tool_side_effects.
    if not _studio_mode:
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

    # Check catalog path registry first (robots loaded from robot_descriptions)
    if robot_name in _catalog_paths:
        robot_path = _catalog_paths[robot_name]
    else:
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

    model = _load_model_with_scene(robot_path)
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
    # MJCF regularly contains non-ASCII characters (Claude emits curly
    # quotes, µ, °, ≤ in comments / names). Pathlib defaults to the
    # platform encoding — cp1252 on Windows — which crashes on those.
    # Force utf-8 to match how every other tool handler writes text.
    path.write_text(mjcf_xml, encoding="utf-8")

    return {
        "success": True,
        "env_name": name,
        "env_path": str(path),
        "model_info": info,
        "warnings": result.warnings,
    }


@register_handler("generate_reward")
def handle_generate_reward(input: dict) -> dict:
    """Validate and register a reward function.

    The smoke-test runs the reward against the real observation / action
    shape whenever we can resolve an approved robot — either from the
    explicit ``robot_name`` input or by falling back to the most recently
    approved one.  A failing smoke-test is returned as an error so Claude
    rewrites the reward instead of silently registering a broken function
    that returns 0 on every real step at training time.
    """
    from uuid import uuid4
    from robo_garden.rewards.reward_runner import compile_reward_function
    from robo_garden.rewards.models import RewardFunction

    task_description = input.get("task_description", "")
    reward_code = input.get("reward_code", "")
    robot_name = (input.get("robot_name") or "").strip()

    # Claude sometimes omits the optional robot_name — look up the latest
    # approved design so the smoke test still uses real dimensions.
    if not robot_name:
        robot_name = _latest_approved_robot()

    dims = _lookup_model_dims(robot_name) if robot_name else {}
    obs_dim = dims.get("obs_dim")
    action_dim = dims.get("action_dim")

    try:
        compile_reward_function(reward_code, obs_dim=obs_dim, action_dim=action_dim)
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "resolved_robot_name": robot_name,
            "obs_dim_used": obs_dim,
            "action_dim_used": action_dim,
            "hint": (
                "obs = concat(qpos, qvel). "
                f"For this robot: nq={dims.get('nq')}, nv={dims.get('nv')}, nu={dims.get('nu')}. "
                "Every hard-coded index in compute_reward must be < obs_dim / nu."
            ) if dims else (
                "No approved robot found — approve_for_training must run first "
                "so generate_reward can validate against real obs dimensions."
            ),
        }

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
        "resolved_robot_name": robot_name,
        "validated_obs_dim": obs_dim,
        "validated_action_dim": action_dim,
    }


@register_handler("train")
def handle_train(input: dict) -> dict:
    """Launch an RL training run using MuJoCo MJX + Brax PPO."""
    from robo_garden.training.mujoco_engine import MuJoCoMJXEngine
    from robo_garden.training.models import TrainingConfig
    from robo_garden.config import ROBOTS_DIR, ENVIRONMENTS_DIR
    from robo_garden.isaac import get_bridge

    robot_name = input.get("robot_name", "")
    # Check catalog path registry first
    if robot_name in _catalog_paths:
        robot_path = _catalog_paths[robot_name]
    else:
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

    reward_fn = None
    done_fn = None
    reward_fn_code = ""
    reward_id = input.get("reward_function_id", "")
    if reward_id:
        rf = get_reward_fn(reward_id)
        if rf:
            reward_fn_code = rf.code
            try:
                from robo_garden.rewards.reward_runner import (
                    compile_reward_function,
                    safe_reward,
                )
                _raw = compile_reward_function(reward_fn_code)
                # safe_reward returns (float, dict) and converts IndexError /
                # ValueError to (0.0, {"_error": ...}) so one bad index does
                # not kill a long training run.
                _safe = safe_reward(_raw, fallback=0.0)
                # Return (scalar, components) tuple so the gym env can surface component data
                reward_fn = (
                    lambda obs, action, next_obs, _r=_safe: _r(obs, action, next_obs, {})
                )
            except Exception as exc:
                log.warning(f"Could not compile reward function '{reward_id}': {exc}")

            # Compile compute_done if the reward code defines one.
            # This is critical for locomotion: without early termination after
            # collapse, the robot runs ~1000 steps at -100/step and the signal
            # is too noisy for PPO to learn from.
            try:
                _globs: dict = {}
                exec(compile(reward_fn_code, "<reward>", "exec"), _globs)
                if "compute_done" in _globs and callable(_globs["compute_done"]):
                    _cd = _globs["compute_done"]
                    done_fn = lambda next_obs, _f=_cd: bool(_f(next_obs))
                    log.info("handle_train: compiled compute_done from reward code")
            except Exception as exc:
                log.debug(f"handle_train: could not extract compute_done: {exc}")
        else:
            log.warning(f"Reward function '{reward_id}' not found, using default reward")

    # For floating-base robots (freejoint trunk, nq > nv) locomotion needs a
    # longer horizon — 2500 steps (5 s at dt=0.002) vs. the 1000-step default
    # which is sufficient for manipulation tasks. Without this the episode ends
    # before the robot can build up meaningful forward velocity.
    dims = _lookup_model_dims(robot_name)
    is_floating_base = bool(dims.get("floating_base", False))
    default_horizon = 2500 if is_floating_base else 1000

    config = TrainingConfig(
        algorithm=input.get("algorithm", "ppo"),
        num_envs=int(input.get("num_envs", 128)),
        total_timesteps=int(input["total_timesteps"]),
        max_episode_steps=int(input.get("max_episode_steps", default_horizon)),
    )

    curriculum_config = None
    n_stages = input.get("curriculum_stages")
    if n_stages is not None:
        from robo_garden.training.models import CurriculumConfig, CurriculumStage
        stages = [CurriculumStage(name=f"stage_{i}") for i in range(int(n_stages))]
        curriculum_config = CurriculumConfig(stages=stages)

    bridge = get_bridge()
    updates: list[dict] = []

    import time as _time
    from datetime import datetime, timezone
    from uuid import uuid4
    from robo_garden.training.history import append_run

    run_id = f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:6]}"
    started_at = _time.time()
    best_so_far: dict = {"reward": float("-inf")}

    bridge.send_train_run_start(
        run_id=run_id,
        robot_name=robot_name,
        environment_name=env_name,
        algorithm=config.algorithm,
        total_timesteps=config.total_timesteps,
        reward_function_id=reward_id,
    )

    def _progress(timestep: int, metrics: dict) -> None:
        mean_reward = float(metrics.get("eval/episode_reward", 0))
        if mean_reward > best_so_far["reward"]:
            best_so_far["reward"] = mean_reward
        elapsed = _time.time() - started_at
        tps = (timestep / elapsed) if elapsed > 0 else 0.0
        updates.append({"timestep": timestep, "mean_reward": mean_reward})
        bridge.send_training_update(
            robot_name,
            timestep,
            mean_reward,
            run_id=run_id,
            best_reward=best_so_far["reward"],
            elapsed_s=elapsed,
            total_timesteps=config.total_timesteps,
            algorithm=config.algorithm,
            timesteps_per_second=tps,
            backend=metrics.get("_backend", ""),
        )
        if _tui_train_progress is not None:
            try:
                _tui_train_progress(timestep, {
                    **metrics,
                    "best_reward": best_so_far["reward"],
                    "elapsed_s": elapsed,
                    "total_timesteps": config.total_timesteps,
                    "timesteps_per_second": tps,
                })
            except Exception:
                pass
        components = metrics.get("components")
        if components and isinstance(components, dict):
            from robo_garden.isaac.protocol import make_train_reward_breakdown
            bridge.send_raw(make_train_reward_breakdown(run_id, timestep, components))

    robot_xml = robot_path.read_text(encoding="utf-8")
    # Catalog robots have relative meshdir — rewrite to absolute so the engine
    # can compile the XML from a string without needing the original file context.
    if robot_name in _catalog_paths:
        robot_xml = _absolutize_asset_paths(robot_xml, robot_path.parent)

    # Pre-merge MJCF once for rollout sampling so we don't pay the cost on
    # every rollout tick.  Best-effort — failures just disable rollouts.
    try:
        from robo_garden.training.mujoco_engine import _merge_mjcf
        merged_for_rollout = _merge_mjcf(robot_xml, env_mjcf)
    except Exception as exc:
        log.warning(f"Could not merge MJCF for rollout streaming: {exc}")
        merged_for_rollout = ""

    def _rollout(timestep: int, policy_apply) -> None:
        if not merged_for_rollout:
            return
        try:
            from robo_garden.training.rollout import sample_rollout
            from robo_garden.isaac.protocol import make_train_rollout_preview

            rollout = sample_rollout(
                merged_for_rollout,
                policy_apply,
                num_frames=150,
                seed=int(timestep) & 0xFFFF,
            )
            if rollout.qpos.shape[0] > 0:
                bridge.stream_qpos_batch(
                    robot_name,
                    rollout.qpos,
                    list(rollout.timesteps),
                )
                bridge.send_raw(make_train_rollout_preview(
                    run_id, timestep, num_frames=int(rollout.qpos.shape[0])
                ))
        except Exception as exc:
            log.debug(f"rollout streaming failed at step {timestep}: {exc}")

    # GPU training path: when ROBO_GARDEN_TRAIN_IN_WSL=1 on Windows, dispatch
    # the whole run to a wsl.exe subprocess so Claude's reward executes against
    # JAX/MJX/Brax with CUDA. Progress is piped back through the same
    # ``_progress`` callback below (which drives the Isaac Sim training panel
    # and history), so behaviour from Claude's point of view is unchanged —
    # just faster. Rollout streaming is disabled in this path because the
    # trained policy params live in the WSL process and aren't easy to
    # marshal mid-training; we can revisit if the viewport rollouts become
    # important during gym-mode runs.
    from robo_garden.training import wsl_dispatch

    error_text = ""
    success = False
    result = None

    if wsl_dispatch.is_enabled():
        _wsl_flag = os.environ.get("ROBO_GARDEN_TRAIN_IN_WSL", "").strip()
        _wsl_reason = f"ROBO_GARDEN_TRAIN_IN_WSL={_wsl_flag}" if _wsl_flag else "WSL2 auto-detected"
        log.info(f"Dispatching training to WSL2 (run_id={run_id}) — {_wsl_reason}.")
        wsl_result = wsl_dispatch.run_in_wsl(
            run_id=run_id,
            robot_xml=robot_xml,
            env_mjcf=env_mjcf,
            reward_fn_code=reward_fn_code,
            robot_name=robot_name,
            environment_name=env_name,
            algorithm=config.algorithm,
            total_timesteps=config.total_timesteps,
            num_envs=config.num_envs,
            # Locomotion-friendly default; WSL worker honors it verbatim.
            max_episode_steps=config.max_episode_steps,
            progress_callback=_progress,
        )
        success = bool(wsl_result.get("success", False))
        error_text = str(wsl_result.get("error", ""))
        best_reward_wsl = float(wsl_result.get("best_reward", float("-inf")))
        reward_curve_wsl = wsl_result.get("reward_curve", []) or []
        checkpoint_path_wsl = str(wsl_result.get("checkpoint_path", ""))
        ended_at = _time.time()
        training_time = float(
            wsl_result.get("training_time_seconds", ended_at - started_at)
        )
        # Fall through to the existing run-record / bridge-end / return path
        # with the same variables that the in-process branch populates.
        best_reward = best_reward_wsl
        reward_curve = list(reward_curve_wsl)
        checkpoint_path = checkpoint_path_wsl
    else:
        engine = MuJoCoMJXEngine()
        engine.setup(robot_xml, env_mjcf, config, curriculum_config=curriculum_config)

        try:
            result = engine.train(
                reward_fn_code=reward_fn_code,
                reward_fn=reward_fn,
                done_fn=done_fn,
                callback=_progress,
                rollout_callback=_rollout,
            )
            success = True
        except Exception as exc:
            error_text = f"{type(exc).__name__}: {exc}"
            log.exception("Training failed")
        finally:
            engine.cleanup()

        ended_at = _time.time()
        training_time = ended_at - started_at
        best_reward = float(result.best_reward) if result is not None else float("-inf")
        checkpoint_path = str(result.checkpoint_path) if result is not None else ""
        reward_curve = list(result.reward_curve) if result is not None else []

    run_record = {
        "run_id": run_id,
        "robot_name": robot_name,
        "environment_name": env_name,
        "algorithm": config.algorithm,
        "total_timesteps": config.total_timesteps,
        "best_reward": best_reward,
        "training_time_seconds": training_time,
        "started_at": started_at,
        "ended_at": ended_at,
        "success": success,
        "checkpoint_path": checkpoint_path,
        "reward_function_id": reward_id,
        "error": error_text,
    }
    append_run(run_record)

    bridge.send_train_run_end(
        run_id=run_id,
        robot_name=robot_name,
        success=success,
        best_reward=best_reward if success else None,
        training_time_seconds=training_time,
        total_timesteps=config.total_timesteps,
        checkpoint_path=checkpoint_path,
        reward_curve=reward_curve[-50:],
        error=error_text,
    )

    if not success:
        return {
            "success": False,
            "error": error_text,
            "run_id": run_id,
            "training_time_seconds": training_time,
            "isaac_connected": bridge.connected,
        }

    # Describe reward curve trend to help Claude decide whether to refine.
    _curve_rewards = [r for _, r in reward_curve] if reward_curve else []
    if len(_curve_rewards) >= 2:
        _trend = _curve_rewards[-1] - _curve_rewards[0]
        _trend_desc = f"improved by {_trend:.3f}" if _trend > 0 else f"did not improve ({_trend:.3f})"
    else:
        _trend_desc = "insufficient data"

    return {
        "success": True,
        "run_id": run_id,
        "robot_name": robot_name,
        "best_reward": best_reward,
        "training_time_seconds": training_time,
        "reward_curve": reward_curve[-10:],
        "recent_updates": updates[-5:],
        "checkpoint_path": checkpoint_path,
        "isaac_connected": bridge.connected,
        "eureka_refinement": {
            "suggestion": (
                "To improve results iteratively (Eureka-style): call generate_reward "
                "with previous_stats below, then train again. Repeat 2-3 times."
            ),
            "previous_stats": {
                "mean_reward": best_reward,
                "reward_trend": _trend_desc,
                "reward_curve_tail": reward_curve[-5:],
                "reward_function_id": reward_id,
            },
        },
        "review_hint": (
            f"Call review_run(run_id='{run_id}') to replay this policy "
            "in the Isaac viewport and optionally save a video."
        ),
    }


@register_handler("approve_for_training")
def handle_approve_for_training(input: dict) -> dict:
    """Promote a robot + environment pair from Design to Training phase.

    Preconditions:
      1. Robot file exists (saved via generate_robot or loaded from catalog).
      2. Environment file exists (saved via generate_environment).
      3. A recent `simulate` call stored a non-diverged SimulationResult for
         the robot (indicating the passive physics is stable).

    On success, writes ``workspace/approved/<robot>__<env>.json`` and fires
    the registered approval callback so the Session flips to training phase.
    """
    import json
    from datetime import datetime, timezone
    from pathlib import Path
    from robo_garden.config import ROBOTS_DIR, ENVIRONMENTS_DIR, APPROVED_DIR

    robot_name = (input.get("robot_name") or "").strip()
    env_name = (input.get("environment_name") or "").strip()
    notes = input.get("notes", "")

    missing: list[str] = []

    if not robot_name:
        missing.append("robot_name is required")
    if not env_name:
        missing.append("environment_name is required")

    # Resolve robot path (catalog registry or saved file)
    robot_path: Path | None = None
    if robot_name:
        if robot_name in _catalog_paths:
            robot_path = _catalog_paths[robot_name]
        else:
            xml_path = ROBOTS_DIR / f"{robot_name}.xml"
            urdf_path = ROBOTS_DIR / f"{robot_name}.urdf"
            if xml_path.exists():
                robot_path = xml_path
            elif urdf_path.exists():
                robot_path = urdf_path
        if robot_path is None:
            missing.append(
                f"robot '{robot_name}' not found — call generate_robot first"
            )

    # Resolve environment path
    env_path = ENVIRONMENTS_DIR / f"{env_name}.xml" if env_name else None
    if env_path is not None and not env_path.exists():
        missing.append(
            f"environment '{env_name}' not found — call generate_environment first"
        )

    # Stability gate: most recent sim for this robot must be non-diverged
    sim = _sim_results.get(robot_name)
    if sim is None:
        missing.append(
            f"no recent simulate() result for '{robot_name}' — run a passive "
            "simulation (control_mode='passive', duration_seconds>=1.0) first"
        )
    elif sim.diverged:
        missing.append(
            f"latest simulation for '{robot_name}' diverged — fix physics "
            "instability before approving"
        )

    if missing:
        return {
            "success": False,
            "approved": False,
            "unmet_preconditions": missing,
            "message": (
                "Cannot approve — address the listed preconditions, then call "
                "approve_for_training again."
            ),
        }

    # Compute obs/action dims of the merged (robot + env) MJCF so later calls
    # to generate_reward can validate shapes without re-loading MuJoCo. We
    # tolerate failures here — worst case the reward smoke-test runs with a
    # stand-in shape as before.
    model_dims: dict = {}
    try:
        import mujoco
        from robo_garden.training.mujoco_engine import _merge_mjcf

        robot_xml_text = robot_path.read_text(encoding="utf-8")
        if robot_name in _catalog_paths:
            robot_xml_text = _absolutize_asset_paths(robot_xml_text, robot_path.parent)
        env_xml_text = env_path.read_text(encoding="utf-8") if env_path is not None else ""
        merged = _merge_mjcf(robot_xml_text, env_xml_text)
        m = mujoco.MjModel.from_xml_string(merged)
        model_dims = {
            "nq": int(m.nq),
            "nv": int(m.nv),
            "nu": int(m.nu),
            "obs_dim": int(m.nq + m.nv),
            "action_dim": int(m.nu),
            "floating_base": bool(m.nq > m.nv),
        }
    except Exception as exc:
        log.warning(f"approve_for_training: could not compute model dims: {exc}")

    manifest = {
        "robot_name": robot_name,
        "robot_path": str(robot_path),
        "environment_name": env_name,
        "environment_path": str(env_path),
        "notes": notes,
        "approved_at_utc": datetime.now(timezone.utc).isoformat(),
        "sim_summary": dict(sim.summary) if sim is not None else {},
        "sim_stable": bool(sim.stable) if sim is not None else False,
        "model_dims": model_dims,
    }
    manifest_path = APPROVED_DIR / f"{robot_name}__{env_name}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")

    # Flip the session phase (no-op if no callback registered — e.g. in
    # headless chat mode the user can still make progress, gating is still
    # enforced at the tool-list level on the next turn).
    if _on_approve is not None:
        try:
            _on_approve(robot_name, env_name, manifest_path)
        except Exception as exc:
            log.warning(f"approve_for_training callback failed: {exc}")

    return {
        "success": True,
        "approved": True,
        "robot_name": robot_name,
        "environment_name": env_name,
        "manifest_path": str(manifest_path),
        "phase": "training",
        "model_dims": model_dims,
        "message": (
            f"Design approved. Training tools (generate_reward, train) are now "
            f"available. Manifest saved to {manifest_path.name}."
            + (
                f" Pass robot_name='{robot_name}' to generate_reward so the reward "
                f"is smoke-tested against the real obs_dim={model_dims['obs_dim']} "
                f"and action_dim={model_dims['action_dim']}."
                if model_dims else ""
            )
        ),
    }


@register_handler("review_run")
def handle_review_run(input: dict) -> dict:
    """Load a training checkpoint, run a policy rollout, stream to viewport, optionally save video."""
    from pathlib import Path
    from robo_garden.training.history import load_recent
    from robo_garden.training.mujoco_engine import MuJoCoMJXEngine, _merge_mjcf
    from robo_garden.training.models import TrainingConfig
    from robo_garden.isaac import get_bridge
    from robo_garden.config import ROBOTS_DIR, ENVIRONMENTS_DIR, RENDERS_DIR

    run_id = input.get("run_id", "latest")
    num_frames = int(input.get("num_frames", 150))
    render_video = bool(input.get("render_video", False))

    runs = load_recent(limit=50)
    run = None
    if run_id == "latest":
        run = next((r for r in runs if r.get("success")), None)
    else:
        run = next((r for r in runs if r.get("run_id") == run_id), None)

    if run is None:
        return {
            "success": False,
            "error": (
                f"No run found with id='{run_id}'. "
                "Pass run_id='latest' or a run_id from the train tool result."
            ),
        }

    checkpoint_path = run.get("checkpoint_path", "")
    robot_name = run.get("robot_name", "")
    env_name = run.get("environment_name", "")
    resolved_run_id = run.get("run_id", run_id)

    if not checkpoint_path:
        return {"success": False, "error": "Run record has no checkpoint_path.", "run_id": resolved_run_id}

    robot_path: Path | None = None
    if robot_name in _catalog_paths:
        robot_path = _catalog_paths[robot_name]
    else:
        for ext in (".xml", ".urdf"):
            p = ROBOTS_DIR / f"{robot_name}{ext}"
            if p.exists():
                robot_path = p
                break

    if robot_path is None:
        return {
            "success": False,
            "error": f"Robot '{robot_name}' XML not found — it may have been generated in a different session.",
            "run_id": resolved_run_id,
        }

    robot_xml = robot_path.read_text(encoding="utf-8")
    if robot_name in _catalog_paths:
        robot_xml = _absolutize_asset_paths(robot_xml, robot_path.parent)

    env_xml = ""
    if env_name:
        env_path = ENVIRONMENTS_DIR / f"{env_name}.xml"
        if env_path.exists():
            env_xml = env_path.read_text(encoding="utf-8")

    merged_mjcf = _merge_mjcf(robot_xml, env_xml)

    engine = MuJoCoMJXEngine()
    engine._merged_mjcf = merged_mjcf
    engine.config = TrainingConfig(num_envs=1, total_timesteps=0)

    rollout = engine.rollout_from_checkpoint(checkpoint_path, num_frames=num_frames)
    if rollout is None:
        return {"success": False, "error": "Rollout failed — could not build merged MJCF.", "run_id": resolved_run_id}

    bridge = get_bridge()
    if bridge.connected and rollout.qpos.shape[0] > 0:
        bridge.stream_qpos_batch(robot_name, rollout.qpos, list(rollout.timesteps))

    policy_type = "sb3_checkpoint" if (Path(checkpoint_path) / "policy.zip").exists() else "zero_action"

    response: dict = {
        "success": True,
        "run_id": resolved_run_id,
        "robot_name": robot_name,
        "policy_type": policy_type,
        "num_frames": int(rollout.qpos.shape[0]),
        "rollout_success": rollout.success,
        "isaac_connected": bridge.connected,
        "best_reward": run.get("best_reward"),
        "algorithm": run.get("algorithm"),
    }

    if policy_type == "zero_action":
        response["note"] = (
            "Brax/JAX policies require the inference function from training to replay. "
            "Showing zero-action (passive dynamics) rollout instead. "
            "The robot's passive stability and joint layout are visible."
        )

    if render_video:
        video_path = RENDERS_DIR / f"{robot_name}_rollout_{resolved_run_id}.mp4"
        try:
            import mujoco
            import imageio
            import numpy as np

            model = mujoco.MjModel.from_xml_string(merged_mjcf)
            data = mujoco.MjData(model)
            renderer = mujoco.Renderer(model, height=480, width=640)
            mujoco.mj_resetData(model, data)

            frames = []
            for qpos_frame in rollout.qpos:
                data.qpos[:model.nq] = qpos_frame[:model.nq]
                mujoco.mj_kinematics(model, data)
                renderer.update_scene(data)
                frames.append(renderer.render().copy())

            imageio.mimsave(str(video_path), frames, fps=30)
            response["video_path"] = str(video_path)
        except Exception as exc:
            response["video_error"] = str(exc)

    return response


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


@register_handler("promote_skill")
def handle_promote_skill(input: dict) -> dict:
    """Copy a training checkpoint into the Skills Library and write manifests."""
    from robo_garden.training.history import load_recent, find_run
    from robo_garden.skills.promote import promote_run_to_skill

    run_id = (input.get("run_id") or "").strip()
    skill_id = (input.get("skill_id") or "").strip()
    display_name = (input.get("display_name") or "").strip()
    task_description = (input.get("task_description") or "").strip()

    if not skill_id:
        return {"success": False, "error": "skill_id is required"}
    if not display_name:
        return {"success": False, "error": "display_name is required"}

    if run_id == "latest":
        recent = load_recent(limit=50)
        run_record = next((r for r in recent if r.get("success")), None)
        if run_record is None:
            return {"success": False, "error": "No successful training run found in history"}
        run_id = run_record["run_id"]

    try:
        variant = promote_run_to_skill(
            run_id=run_id,
            skill_id=skill_id,
            display_name=display_name,
            task_description=task_description,
        )
    except ValueError as exc:
        return {"success": False, "error": str(exc)}
    except Exception as exc:
        log.exception("promote_skill failed")
        return {"success": False, "error": f"{type(exc).__name__}: {exc}"}

    return {
        "success": True,
        "run_id": run_id,
        "skill_id": skill_id,
        "display_name": display_name,
        "variant_id": variant.variant_id,
        "checkpoint_path": variant.checkpoint_path,
        "best_reward": variant.best_reward,
        "message": (
            f"Skill '{display_name}' (variant {variant.variant_id}) saved to the "
            f"Skills Library. It is now visible in the Skills tab."
        ),
    }
