"""Tool definitions registry for Claude API tool-use.

These define the contract between Claude and the local system.
Each tool has a name, description, and JSON Schema for its input.
"""

TOOLS = [
    {
        "name": "generate_robot",
        "description": (
            "Generate a robot description in MJCF or URDF format based on a design specification. "
            "Provide the full robot XML string, a list of joint names with their intended actuators "
            "from the actuator database, and material assignments for each link. The robot must be "
            "physically buildable with real-world components. Returns validation results and "
            "buildability analysis including any torque/material constraint violations."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Robot name identifier (alphanumeric + underscores)",
                },
                "robot_xml": {
                    "type": "string",
                    "description": "Complete robot XML string (MJCF or URDF depending on 'format')",
                },
                "format": {
                    "type": "string",
                    "enum": ["mjcf", "urdf"],
                    "description": (
                        "Robot description format. Use 'mjcf' (default) for complex actuator/contact "
                        "specs, tendons, and sensors. Use 'urdf' for ROS-compatible robots or simpler "
                        "kinematic chains."
                    ),
                    "default": "mjcf",
                },
                "actuator_assignments": {
                    "type": "array",
                    "description": "Map joints to real-world actuators from the catalog",
                    "items": {
                        "type": "object",
                        "properties": {
                            "joint_name": {"type": "string"},
                            "actuator_id": {"type": "string"},
                            "gear_ratio": {"type": "number", "default": 1.0},
                        },
                        "required": ["joint_name", "actuator_id"],
                    },
                },
                "material_assignments": {
                    "type": "array",
                    "description": "Map links to real-world materials from the catalog",
                    "items": {
                        "type": "object",
                        "properties": {
                            "link_name": {"type": "string"},
                            "material_id": {"type": "string"},
                        },
                        "required": ["link_name", "material_id"],
                    },
                },
                "catalog_name": {
                    "type": "string",
                    "description": (
                        "Load a reference robot from the robot_descriptions catalog instead of "
                        "providing robot_xml. Use the exact catalog key returned by query_catalog "
                        "(e.g. 'go2_mj_description', 'spot_mj_description'). "
                        "When set, robot_xml is not required."
                    ),
                },
                "description": {
                    "type": "string",
                    "description": "Text description of the robot design",
                },
            },
            "required": ["name"],
        },
    },
    {
        "name": "simulate",
        "description": (
            "Run a physics simulation of a robot in an environment for a specified duration. "
            "Returns state time-series (joint positions, velocities, COM trajectory), "
            "stability assessment, and performance metrics. "
            "Set render_video=true to save an MP4 to workspace/renders/ using the local "
            "MuJoCo renderer — no external tools required. The result will include video_path "
            "on success or video_error if rendering failed."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "robot_name": {
                    "type": "string",
                    "description": "Name of the robot to simulate",
                },
                "duration_seconds": {
                    "type": "number",
                    "description": "Simulation duration in seconds",
                    "default": 2.0,
                },
                "control_mode": {
                    "type": "string",
                    "enum": ["passive", "policy", "trajectory"],
                    "description": "passive=no control, policy=trained policy, trajectory=replay",
                    "default": "passive",
                },
                "policy_checkpoint": {
                    "type": "string",
                    "description": "Path to policy checkpoint (required if control_mode=policy)",
                },
                "render_video": {
                    "type": "boolean",
                    "description": "Save an MP4 video of the simulation using the local MuJoCo renderer. No Isaac Sim needed.",
                    "default": False,
                },
            },
            "required": ["robot_name", "duration_seconds"],
        },
    },
    {
        "name": "evaluate",
        "description": (
            "Evaluate simulation results against specified criteria. Returns structured "
            "metrics: stability, forward velocity, energy efficiency, task success, etc."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "simulation_id": {
                    "type": "string",
                    "description": "ID from a previous simulate call",
                },
                "metrics": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Metrics to compute: stability, forward_velocity, energy_efficiency, com_height, joint_range_usage",
                },
                "success_criteria": {
                    "type": "object",
                    "description": "Metric thresholds for success (e.g., {\"stability\": 0.9})",
                },
            },
            "required": ["simulation_id", "metrics"],
        },
    },
    {
        "name": "generate_reward",
        "description": (
            "Generate a Python reward function for RL training. Signature: "
            "def compute_reward(obs, action, next_obs, info) -> tuple[float, dict]. "
            "The dict holds per-component reward breakdowns for analysis. Only numpy "
            "(as np) is available.\n\n"
            "OBSERVATION LAYOUT (identical in every training path):\n"
            "  obs = np.concatenate([qpos, qvel])      # shape (nq + nv,)\n"
            "  - qpos[:nq] — generalized positions; for floating-base robots the\n"
            "    first 7 entries are [x, y, z, qw, qx, qy, qz]. Joint positions\n"
            "    follow in MJCF declaration order.\n"
            "  - qvel[nq:nq+nv] — generalized velocities; for floating-base robots\n"
            "    the first 6 entries are [vx, vy, vz, wx, wy, wz] (world-frame\n"
            "    linear + angular). Joint velocities follow.\n"
            "  - nq / nv / nu come from the merged MJCF. Always pass robot_name so\n"
            "    the smoke-test validates against the REAL dimensions and an index\n"
            "    error is caught at generate-time, not mid-training.\n"
            "\n"
            "ACTION LAYOUT: action is a np.ndarray of shape (nu,) in [-1, 1], one\n"
            "entry per MJCF actuator. The env scales this into each actuator's\n"
            "ctrlrange before stepping.\n"
            "\n"
            "If you reference any index, assert it fits nq/nv/nu first. A safe-\n"
            "guard wrapper returns 0.0 on IndexError during training, but the\n"
            "smoke test raises — so fix layout bugs here."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "task_description": {
                    "type": "string",
                    "description": "What the robot should learn to do",
                },
                "robot_name": {
                    "type": "string",
                    "description": (
                        "Name of the approved robot. When provided, the handler "
                        "smoke-tests the reward with the robot's real (nq + nv) "
                        "observation length and nu action length, so shape bugs "
                        "are caught now instead of crashing training. "
                        "Highly recommended."
                    ),
                },
                "observation_space_description": {
                    "type": "string",
                    "description": "Optional human-readable obs description (layout above is authoritative).",
                },
                "reward_code": {
                    "type": "string",
                    "description": "Python code defining compute_reward function",
                },
                "previous_stats": {
                    "type": "object",
                    "description": "Training stats from previous reward iteration (for Eureka refinement)",
                },
            },
            "required": ["task_description", "reward_code"],
        },
    },
    {
        "name": "generate_environment",
        "description": (
            "Generate a simulation environment in MJCF XML format. Includes terrain, "
            "obstacles, objects, lighting, and physics settings."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Environment name"},
                "description": {
                    "type": "string",
                    "description": "Natural language description of the environment",
                },
                "terrain_type": {
                    "type": "string",
                    "enum": ["flat", "heightfield", "stairs", "rough", "mixed"],
                    "default": "flat",
                },
                "size_meters": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "[width, length] in meters",
                    "default": [10, 10],
                },
                "objects": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "type": {"type": "string"},
                            "position": {
                                "type": "array",
                                "items": {"type": "number"},
                            },
                            "size": {
                                "type": "array",
                                "items": {"type": "number"},
                            },
                        },
                    },
                },
                "mjcf_xml": {
                    "type": "string",
                    "description": "Complete environment MJCF XML",
                },
            },
            "required": ["name", "mjcf_xml"],
        },
    },
    {
        "name": "train",
        "description": (
            "Start RL training for a robot in an environment with a specified reward function. "
            "Uses MuJoCo MJX with GPU acceleration locally (64-256 parallel environments). "
            "Returns training progress and the best policy checkpoint."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "robot_name": {"type": "string"},
                "environment_name": {"type": "string"},
                "reward_function_id": {"type": "string"},
                "algorithm": {
                    "type": "string",
                    "enum": ["ppo", "sac"],
                    "default": "ppo",
                },
                "num_envs": {
                    "type": "integer",
                    "description": "Parallel environments (64-256 for local RTX 3070)",
                    "default": 128,
                },
                "total_timesteps": {
                    "type": "integer",
                    "description": "Total training timesteps",
                    "default": 1000000,
                },
                "curriculum_stages": {
                    "type": "integer",
                    "description": "Number of curriculum stages (1-5). If provided, training uses progressive difficulty.",
                    "minimum": 1,
                    "maximum": 5,
                },
            },
            "required": ["robot_name", "environment_name", "reward_function_id", "total_timesteps"],
        },
    },
    {
        "name": "approve_for_training",
        "description": (
            "Promote a robot + environment pair from the Design phase into the Training phase. "
            "This tool is the ONLY bridge between design iteration and RL training. Call this when "
            "the user has explicitly approved the current design (or click the 'Promote to Training' "
            "button in the Studio UI). "
            "Requires: (1) the robot has been saved via generate_robot, (2) the environment has "
            "been saved via generate_environment, (3) a recent passive simulate call showed the "
            "robot is stable (did not diverge). "
            "On success: writes an approval manifest to workspace/approved/<robot>__<env>.json, "
            "flips the session into the Training phase, and unlocks generate_reward and train. "
            "If any precondition is missing, returns a structured error listing what still needs "
            "to happen."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "robot_name": {
                    "type": "string",
                    "description": "Name of the robot previously registered via generate_robot",
                },
                "environment_name": {
                    "type": "string",
                    "description": "Name of the environment previously registered via generate_environment",
                },
                "notes": {
                    "type": "string",
                    "description": (
                        "Short human-readable notes explaining why this design is ready for "
                        "training (e.g. 'passive stable at 0.41m CoM, no self-collision, actuator "
                        "torques within Dynamixel XM430 limits')."
                    ),
                },
            },
            "required": ["robot_name", "environment_name"],
        },
    },
    {
        "name": "query_catalog",
        "description": (
            "Search the actuator, material, or reference robot databases. Use this to find "
            "real-world components matching design requirements (torque, speed, weight, cost)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "catalog": {
                    "type": "string",
                    "enum": ["actuators", "materials", "robots"],
                    "description": "Which catalog to search",
                },
                "query": {
                    "type": "string",
                    "description": (
                        "Natural language search query. For robots, match against names like "
                        "'spot', 'humanoid', 'quadruped', 'ant', 'cassie', 'unitree'."
                    ),
                },
                "filters": {
                    "type": "object",
                    "description": "Structured filters (e.g., {\"min_torque_nm\": 2.0, \"type\": \"servo\"})",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max results to return",
                    "default": 10,
                },
            },
            "required": ["catalog", "query"],
        },
    },
]
