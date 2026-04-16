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
                "description": {
                    "type": "string",
                    "description": "Text description of the robot design",
                },
            },
            "required": ["name", "robot_xml"],
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
            "Generate a Python reward function for RL training. The function must have signature: "
            "def compute_reward(obs, action, next_obs, info) -> tuple[float, dict]. "
            "The dict should contain per-component reward breakdowns for analysis. "
            "Only numpy is available in the reward function scope."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "task_description": {
                    "type": "string",
                    "description": "What the robot should learn to do",
                },
                "observation_space_description": {
                    "type": "string",
                    "description": "Description of what the observation array contains",
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
