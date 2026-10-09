"""Planar (SE(2)) helpers for the LeKiwi holonomic base.

LeKiwi in LeIsaac keeps its articulation root fixed in the world and moves the ``base`` link through
three virtual joints ``base_x``, ``base_y`` (prismatic, root frame) and ``base_theta`` (revolute, z).
All planar quantities here are expressed in that root frame as ``(x, y, yaw)``.

The robot's forward direction is +y of the base link (see ``convert_lekiwi_wheel_action_env2robot``).

Pure torch, no Isaac dependency, so it can be unit-tested without the simulator.
"""

from __future__ import annotations

import math

import torch

BASE_JOINT_NAMES = ("base_x", "base_y", "base_theta")
FORWARD_AXIS_YAW = math.pi / 2
"""Yaw of the robot's forward axis (+y) inside its own base frame."""


def wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def rot2d(yaw: torch.Tensor) -> torch.Tensor:
    """(..., ) -> (..., 2, 2) rotation matrices."""
    c, s = torch.cos(yaw), torch.sin(yaw)
    return torch.stack([torch.stack([c, -s], -1), torch.stack([s, c], -1)], -2)


def world_to_body_xy(vec_xy: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    """Rotate planar vectors from the (root) world frame into the body frame given body yaw."""
    return (rot2d(-yaw) @ vec_xy.unsqueeze(-1)).squeeze(-1)


def body_to_world_xy(vec_xy: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    return (rot2d(yaw) @ vec_xy.unsqueeze(-1)).squeeze(-1)


def base_goal_from_object(
    obj_xy: torch.Tensor,
    rel_obj_xy: torch.Tensor,
    approach_from_xy: torch.Tensor,
) -> torch.Tensor:
    """Compute the base pose that puts an object at a desired position in the base frame.

    The base yaw is chosen so that the robot faces the object coming from ``approach_from_xy``
    (typically its current position): the bearing from the approach point to the object is rotated
    so the object lands at the bearing of ``rel_obj_xy`` inside the base frame.

    Args:
        obj_xy: (N, 2) object position in the root frame.
        rel_obj_xy: (N, 2) desired object position in the base frame (e.g. sampled from source demos).
        approach_from_xy: (N, 2) point the robot comes from.

    Returns:
        (N, 3) goal ``(x, y, yaw)`` of the base in the root frame.
    """
    delta = obj_xy - approach_from_xy
    bearing_world = torch.atan2(delta[:, 1], delta[:, 0])
    bearing_body = torch.atan2(rel_obj_xy[:, 1], rel_obj_xy[:, 0])
    yaw = wrap_to_pi(bearing_world - bearing_body)
    base_xy = obj_xy - body_to_world_xy(rel_obj_xy, yaw)
    return torch.cat([base_xy, yaw.unsqueeze(-1)], dim=-1)


def canonicalize_yaw(rel_yaw: torch.Tensor, symmetry: float) -> torch.Tensor:
    """Map a yaw to the equivalent one in [-symmetry/2, symmetry/2) for an object with rotational symmetry."""
    return torch.remainder(rel_yaw + symmetry / 2, symmetry) - symmetry / 2
