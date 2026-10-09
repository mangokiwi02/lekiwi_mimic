"""MDP terms for the LeKiwi pick-cube tasks."""

from __future__ import annotations

import math

import isaaclab.utils.math as math_utils
import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from leisaac.enhance.envs.mdp import *  # noqa: F401,F403  (re-export LeIsaac + Isaac Lab terms)

from ...actions import BASE_BODY_NAME, EEF_FRAME_NAME, TCP_FRAME_NAME, base_pose_in_root, frame_index
from ...frames import FORWARD_AXIS_YAW

# ----------------------------------------------------------------------------
# observations
# ----------------------------------------------------------------------------


def _base_idx(robot: Articulation) -> int:
    return robot.find_bodies(BASE_BODY_NAME)[0][0]


def base_pose_2d(env: ManagerBasedEnv, robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """(x, y, sin yaw, cos yaw) of the base link in the root frame."""
    robot: Articulation = env.scene[robot_cfg.name]
    pos, quat = base_pose_in_root(robot, _base_idx(robot))
    yaw = math_utils.euler_xyz_from_quat(quat)[2]
    return torch.stack([pos[:, 0], pos[:, 1], torch.sin(yaw), torch.cos(yaw)], dim=-1)


def object_pose_in_base(
    env: ManagerBasedEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Object position (3) and quaternion (4) in the moving base frame."""
    robot: Articulation = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    idx = _base_idx(robot)
    pos, quat = math_utils.subtract_frame_transforms(
        robot.data.body_pos_w[:, idx], robot.data.body_quat_w[:, idx], obj.data.root_pos_w, obj.data.root_quat_w
    )
    return torch.cat([pos, quat], dim=-1)


def ee_pose_in_base(env: ManagerBasedEnv, ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")) -> torch.Tensor:
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    i = frame_index(ee_frame, EEF_FRAME_NAME)
    return torch.cat([ee_frame.data.target_pos_source[:, i], ee_frame.data.target_quat_source[:, i]], dim=-1)


def cube_grasped(
    env: ManagerBasedEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    diff_threshold: float = 0.035,
    gripper_open_threshold: float = 0.5,
) -> torch.Tensor:
    """TCP near the cube and gripper closed. Looks the gripper joint up by name (LeKiwi lists base joints first)."""
    robot: Articulation = env.scene[robot_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    tcp_pos = ee_frame.data.target_pos_w[:, frame_index(ee_frame, TCP_FRAME_NAME), :]
    near = torch.linalg.vector_norm(obj.data.root_pos_w - tcp_pos, dim=1) < diff_threshold
    gripper = robot.data.joint_pos[:, robot.joint_names.index("gripper")]
    return near & (gripper < gripper_open_threshold)


# ----------------------------------------------------------------------------
# terminations
# ----------------------------------------------------------------------------


def cube_lifted(
    env: ManagerBasedEnv, object_cfg: SceneEntityCfg = SceneEntityCfg("cube"), height: float = 0.08
) -> torch.Tensor:
    obj: RigidObject = env.scene[object_cfg.name]
    return (obj.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]) > height


# ----------------------------------------------------------------------------
# events
# ----------------------------------------------------------------------------


def reset_base_pose_uniform(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    yaw_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Teleport the virtual base joints (base_x, base_y, base_theta) to a random planar pose."""
    robot: Articulation = env.scene[asset_cfg.name]
    joint_ids = [robot.joint_names.index(n) for n in ("base_x", "base_y", "base_theta")]
    n = len(env_ids)
    lo = torch.tensor([x_range[0], y_range[0], yaw_range[0]], device=env.device)
    hi = torch.tensor([x_range[1], y_range[1], yaw_range[1]], device=env.device)
    sample = lo + (hi - lo) * torch.rand(n, 3, device=env.device)
    joint_pos = robot.data.joint_pos[env_ids].clone()
    joint_vel = robot.data.joint_vel[env_ids].clone()
    joint_pos[:, joint_ids] = sample
    joint_vel[:, joint_ids] = 0.0
    robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
    robot.set_joint_position_target(joint_pos, env_ids=env_ids)
    robot.set_joint_velocity_target(joint_vel, env_ids=env_ids)


def reset_object_relative_to_base(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    distance_range: tuple[float, float],
    bearing_range: tuple[float, float],
    yaw_range: tuple[float, float] = (-math.pi, math.pi),
    lateral_range: tuple[float, float] = (0.0, 0.0),
    object_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Place an object at a random distance/bearing from the robot's current base pose.

    ``bearing`` is measured from the robot's forward axis (+y of the base link). Must run after the base reset.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    n = len(env_ids)
    dev = env.device

    def u(r):
        return r[0] + (r[1] - r[0]) * torch.rand(n, device=dev)

    # body poses are stale right after a joint teleport (no sim step yet), so derive the base pose from the joints
    jp = robot.data.joint_pos[env_ids]
    bx, by, byaw = (jp[:, robot.joint_names.index(j)] for j in ("base_x", "base_y", "base_theta"))
    offset = env.cfg.base_link_offset_xy
    heading = byaw + FORWARD_AXIS_YAW + u(bearing_range)
    dist = u(distance_range)
    lat = u(lateral_range)
    ox = bx + offset[0] + dist * torch.cos(heading) - lat * torch.sin(heading)
    oy = by + offset[1] + dist * torch.sin(heading) + lat * torch.cos(heading)
    yaw = u(yaw_range)

    root_state = obj.data.default_root_state[env_ids].clone()
    root_rot = robot.data.root_quat_w[env_ids]
    rel = torch.stack([ox, oy, root_state[:, 2]], dim=-1)
    pos_w = robot.data.root_pos_w[env_ids] + math_utils.quat_apply(root_rot, rel)
    quat_w = math_utils.quat_mul(root_rot, math_utils.quat_from_euler_xyz(torch.zeros_like(yaw), torch.zeros_like(yaw), yaw))
    root_state[:, :3] = pos_w
    root_state[:, 3:7] = quat_w
    root_state[:, 7:] = 0.0
    obj.write_root_state_to_sim(root_state, env_ids=env_ids)
