"""Scripted top-down pick for LeKiwi with a static base, used as a stand-in for teleoperated source demos.

Every target is expressed in the base frame, matching the Mimic action convention
``[gripper-link pose in base frame (7) | gripper +1 open / -1 close | base joint velocity (3)]``.
"""

from __future__ import annotations

import math

import isaaclab.utils.math as math_utils
import torch

from ..frames import FORWARD_AXIS_YAW

OPEN, CLOSE = 1.0, -1.0

# (phase name, number of steps at 60 Hz). No idle waits: a pause where the state does not change makes the next
# phase's start unobservable for a state-based policy trained on these demos (it stalls), so the arm keeps moving and
# "close" lasts only as long as the gripper needs to close (~15 steps).
PHASES = [
    ("approach", 90),
    ("descend", 50),
    ("close", 18),
    ("lift", 60),
    ("hold", 10),
]


def _lerp_pose(p0, q0, p1, q1, t):
    # constant velocity on purpose: an ease-in/ease-out profile stops at every phase boundary, where the next motion
    # direction then has (near) zero action signal for a state-based policy trained on these demos
    s = max(0.0, min(1.0, t))
    pos = p0 + (p1 - p0) * s
    quat = math_utils.quat_slerp(q0, q1, s) if hasattr(math_utils, "quat_slerp") else (q1 if s > 0.5 else q0)
    return pos, quat


class PickCubeStateMachine:
    """Single-env scripted pick; targets are planned for the TCP and converted to gripper-link poses.
    Constants come from scripts/calibrate_grasp.py and scripts/sweep_sm.py."""

    def __init__(
        self,
        down_quat: tuple[float, float, float, float],
        tcp_offset: tuple[float, float, float],
        approach_axis: tuple[float, float, float] = (0.0, 0.0, 1.0),
        hover_height: float = 0.05,
        grasp_height: float = 0.0,
        forward_offset: float = 0.0,
        align_roll: bool = True,
        lift_height: float = 0.09,
        noise_std: float = 0.0,
        device: str = "cuda:0",
    ):
        self.device = device
        self.down_quat = torch.tensor(down_quat, device=device)
        self.tcp_offset = torch.tensor(tcp_offset, device=device)
        self.approach_axis = torch.nn.functional.normalize(torch.tensor(approach_axis, device=device), dim=0)
        self.hover_height = hover_height
        self.grasp_height = grasp_height
        self.forward_offset = forward_offset
        self.align_roll = align_roll
        self.lift_height = lift_height
        self.noise_std = noise_std
        self.reset()

    def reset(self):
        self.step_count = 0
        self.start_pos = None
        self.start_quat = None
        self.grasp_quat = None
        self.cube_pos = None
        self._jitter = None

    @property
    def total_steps(self) -> int:
        return sum(n for _, n in PHASES)

    @property
    def done(self) -> bool:
        return self.step_count >= self.total_steps

    def _phase(self):
        k = self.step_count
        for name, n in PHASES:
            if k < n:
                return name, k / max(n - 1, 1)
            k -= n
        return PHASES[-1][0], 1.0

    def _grasp_quat(self, cube_quat_b: torch.Tensor, cube_pos_b: torch.Tensor) -> torch.Tensor:
        """Top-down orientation: the calibrated pan=0 pose swung to the cube's bearing (shoulder_pan), then rolled
        about the tool axis (wrist_roll) to the cube face closest to that natural yaw."""
        dev = self.device
        pan = math.atan2(cube_pos_b[1].item(), cube_pos_b[0].item()) - FORWARD_AXIS_YAW
        cube_yaw = math_utils.euler_xyz_from_quat(cube_quat_b.unsqueeze(0))[2][0].item()
        roll = math.remainder(cube_yaw - pan, math.pi / 2) if self.align_roll else 0.0
        q_pan = math_utils.quat_from_euler_xyz(*(torch.tensor([v], device=dev) for v in (0.0, 0.0, pan)))[0]
        q_roll = math_utils.quat_from_angle_axis(torch.tensor([roll], device=dev), self.approach_axis.unsqueeze(0))[0]
        return math_utils.quat_mul(math_utils.quat_mul(q_pan, self.down_quat), q_roll)

    def _gripper_target(self, tcp_pos: torch.Tensor, quat: torch.Tensor) -> torch.Tensor:
        """Gripper-link position that puts the TCP at ``tcp_pos`` with orientation ``quat``."""
        return tcp_pos - math_utils.quat_apply(quat.unsqueeze(0), self.tcp_offset.unsqueeze(0))[0]

    def get_action(self, ee_pose_b: torch.Tensor, cube_pose_b: torch.Tensor) -> torch.Tensor:
        """ee_pose_b / cube_pose_b: (7,) position + wxyz quaternion in the base frame. Returns (1, 11)."""
        name, t = self._phase()
        if self.start_pos is None:
            self.start_pos, self.start_quat = ee_pose_b[:3].clone(), ee_pose_b[3:7].clone()
            self._jitter = self.noise_std * torch.randn(3, device=self.device)
        if name == "approach":
            # keep tracking the cube until the descent: link poses are stale on the first step after a reset
            self.cube_pos = cube_pose_b[:3].clone()
            self.grasp_quat = self._grasp_quat(cube_pose_b[3:7], self.cube_pos)

        up = torch.tensor([0.0, 0.0, 1.0], device=self.device)
        cube = self.cube_pos + self._jitter * torch.tensor([1.0, 1.0, 0.3], device=self.device)
        if self.forward_offset:
            # shift along the horizontal direction from the shoulder to the cube (the jaw's closing direction)
            radial = torch.nn.functional.normalize(self.cube_pos * torch.tensor([1.0, 1.0, 0.0], device=self.device), dim=0)
            cube = cube + self.forward_offset * radial
        hover = self._gripper_target(cube + up * self.hover_height, self.grasp_quat)
        grasp = self._gripper_target(cube + up * self.grasp_height, self.grasp_quat)
        lifted = self._gripper_target(cube + up * self.lift_height, self.grasp_quat)

        gripper = OPEN
        if name == "approach":
            pos, quat = _lerp_pose(self.start_pos, self.start_quat, hover, self.grasp_quat, t)
        elif name == "descend":
            pos, quat = _lerp_pose(hover, self.grasp_quat, grasp, self.grasp_quat, t)
        elif name == "close":
            pos, quat, gripper = grasp, self.grasp_quat, CLOSE
        elif name == "lift":
            pos, quat = _lerp_pose(grasp, self.grasp_quat, lifted, self.grasp_quat, t)
            gripper = CLOSE
        else:
            pos, quat, gripper = lifted, self.grasp_quat, CLOSE

        self.step_count += 1
        base_vel = torch.zeros(3, device=self.device)
        return torch.cat([pos, quat, torch.tensor([gripper], device=self.device), base_vel]).unsqueeze(0)
