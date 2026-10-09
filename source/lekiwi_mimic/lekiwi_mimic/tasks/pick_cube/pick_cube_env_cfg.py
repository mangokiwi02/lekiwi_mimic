"""LeKiwi pick-cube tasks.

* ``Static``: the cube spawns within arm reach in front of the base; the base never moves. Source demos are recorded here.
* ``Far``: the robot starts at a random pose and the cube lies 1-3 m away in any direction; the robot must drive there.
"""

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
import numpy as np
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import FrameTransformerCfg, OffsetCfg
from isaaclab.utils import configclass
from isaaclab.utils.datasets.episode_data import EpisodeData
from leisaac.tasks.template.lekiwi_env_cfg import (
    LeKiwiEventCfg,
    LeKiwiObservationsCfg,
    LeKiwiTaskEnvCfg,
    LeKiwiTaskSceneCfg,
    LeKiwiTerminationsCfg,
)
from leisaac.utils.env_utils import delete_attribute
from leisaac.utils.robot_utils import convert_leisaac_action_to_lerobot

from ...actions import TCP_FRAME_NAME, LeKiwiMimicActionsCfg, init_lekiwi_mimic_action_cfg
from ...datagen.calibration import TCP_OFFSET
from . import mdp

CUBE_SIZE = 0.03

# Articulation joint order of LEKIWI_CFG (printed by scripts/inspect_robot.py): the virtual base joints come FIRST,
# ['base_x', 'base_y', 'base_theta', 'shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper']
ARM_GRIPPER_SLICE = slice(3, 9)

# Cube spawn region for source demos, in the base frame (+y forward). Top-down reach on the floor is ~0.11-0.31 m (scripts/calibrate_grasp.py).
STATIC_CUBE_DISTANCE = (0.18, 0.25)
STATIC_CUBE_BEARING = (-0.35, 0.35)

# Arm "ready" pose: gripper pointing roughly down, grasp point ~5 cm above the floor 0.19 m in front of the base
# (scripts/calibrate_grasp.py --tcp_height 0.07). Starting here keeps differential-IK motions small (a large
# reorientation from the default upright pose makes DLS IK diverge), and keeps the fingers clear of the cube when the
# first Mimic segment is interpolated in after driving up.
ARM_READY_POSE = {"shoulder_pan": 0.0, "shoulder_lift": -0.34, "elbow_flex": 0.9, "wrist_flex": 0.96, "wrist_roll": 0.0}


@configclass
class PickCubeSceneCfg(LeKiwiTaskSceneCfg):
    scene: AssetBaseCfg = AssetBaseCfg(prim_path="/World/GroundPlane", spawn=sim_utils.GroundPlaneCfg())

    cube: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        spawn=sim_utils.CuboidCfg(
            size=(CUBE_SIZE, CUBE_SIZE, CUBE_SIZE),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16, solver_velocity_iteration_count=1, max_depenetration_velocity=1.0
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.02),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.2, dynamic_friction=1.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.1, 0.1)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.23, CUBE_SIZE / 2)),
    )


@configclass
class PickCubeObservationsCfg(LeKiwiObservationsCfg):
    @configclass
    class PolicyCfg(LeKiwiObservationsCfg.PolicyCfg):
        base_pose = ObsTerm(func=mdp.base_pose_2d)
        cube_pose_base = ObsTerm(func=mdp.object_pose_in_base, params={"object_cfg": SceneEntityCfg("cube")})
        ee_pose_base = ObsTerm(func=mdp.ee_pose_in_base)

    @configclass
    class SubtaskCfg(ObsGroup):
        grasp_cube = ObsTerm(func=mdp.cube_grasped)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()
    subtask_terms: SubtaskCfg = SubtaskCfg()


@configclass
class PickCubeTerminationsCfg(LeKiwiTerminationsCfg):
    success = DoneTerm(func=mdp.cube_lifted, params={"height": 0.06})


@configclass
class StaticEventCfg(LeKiwiEventCfg):
    reset_cube = EventTerm(
        func=mdp.reset_object_relative_to_base,
        mode="reset",
        params={
            "distance_range": STATIC_CUBE_DISTANCE,
            "bearing_range": STATIC_CUBE_BEARING,
            "yaw_range": (-math.pi / 4, math.pi / 4),
        },
    )


@configclass
class FarEventCfg(LeKiwiEventCfg):
    reset_base = EventTerm(
        func=mdp.reset_base_pose_uniform,
        mode="reset",
        params={"x_range": (-1.0, 1.0), "y_range": (-1.0, 1.0), "yaw_range": (-math.pi, math.pi)},
    )
    reset_cube = EventTerm(
        func=mdp.reset_object_relative_to_base,
        mode="reset",
        params={"distance_range": (1.0, 3.0), "bearing_range": (-math.pi, math.pi)},
    )


@configclass
class LeKiwiPickCubeEnvCfg(LeKiwiTaskEnvCfg):
    scene: PickCubeSceneCfg = PickCubeSceneCfg(env_spacing=10.0)
    observations: PickCubeObservationsCfg = PickCubeObservationsCfg()
    actions: LeKiwiMimicActionsCfg = LeKiwiMimicActionsCfg()
    events: StaticEventCfg = StaticEventCfg()
    terminations: PickCubeTerminationsCfg = PickCubeTerminationsCfg()

    task_description: str = "Pick up the red cube."
    task_type: str = "lekiwi_mimic"
    dynamic_reset_gripper_effort_limit: bool = False

    base_link_offset_xy: tuple[float, float] = (0.0, 0.0)
    """Position of the base link in the root frame when all base joints are zero (see scripts/inspect_robot.py)."""

    cameras: bool = True
    """False drops the wrist/front cameras for fast, state-only generation."""

    def __post_init__(self) -> None:
        super().__post_init__()
        init_lekiwi_mimic_action_cfg(self.actions)
        # TCP = grasp point between the closed fingertips, rigidly attached to the gripper link
        frames = list(self.scene.ee_frame.target_frames)
        if TCP_FRAME_NAME not in [f.name for f in frames]:
            frames.append(
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/gripper", name=TCP_FRAME_NAME, offset=OffsetCfg(pos=TCP_OFFSET)
                )
            )
        self.scene.ee_frame.target_frames = frames
        self.scene.robot.init_state.pos = (0.0, 0.0, 0.035)
        self.scene.robot.init_state.joint_pos = {**self.scene.robot.init_state.joint_pos, **ARM_READY_POSE, "gripper": 1.0}
        self.viewer.eye = (0.8, -0.8, 0.8)
        self.viewer.lookat = (0.0, 0.2, 0.0)
        if not self.cameras:
            self.drop_cameras()

    def drop_cameras(self) -> None:
        self.cameras = False
        delete_attribute(self.scene, "wrist")
        delete_attribute(self.scene, "front")
        delete_attribute(self.observations.policy, "wrist")
        delete_attribute(self.observations.policy, "front")

    def use_teleop_device(self, teleop_device) -> None:
        # actions are fixed to the Mimic convention: [eef pose in base frame (7) | gripper (1) | base joint vel (3)]
        self.task_type = teleop_device

    def build_lerobot_frame(self, episode_data: EpisodeData, dataset_cfg) -> dict:
        """LeRobot frame with LeKiwi-native action: arm/gripper joint targets (deg) + body-frame base velocity."""
        obs = episode_data._data["obs"]
        joint_target = obs["joint_pos_target"][-1][ARM_GRIPPER_SLICE].unsqueeze(0)
        arm_action = convert_leisaac_action_to_lerobot(joint_target).squeeze(0)
        frame = {
            "action": np.concatenate([arm_action, obs["user_vel_action"][-1].cpu().numpy()]),
            "observation.state": np.concatenate(
                [
                    convert_leisaac_action_to_lerobot(obs["joint_pos"][-1][ARM_GRIPPER_SLICE].unsqueeze(0)).squeeze(0),
                    obs["user_vel_state"][-1].cpu().numpy(),
                ]
            ),
            "task": self.task_description,
        }
        for key in dataset_cfg.features:
            if key.startswith("observation.images"):
                frame[key] = obs[key.split(".")[-1]][-1].cpu().numpy()
        return frame


@configclass
class LeKiwiFarPickCubeEnvCfg(LeKiwiPickCubeEnvCfg):
    events: FarEventCfg = FarEventCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        self.episode_length_s = 60.0
