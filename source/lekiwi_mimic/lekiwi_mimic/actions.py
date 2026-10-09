"""Action terms and action configuration for LeKiwi Mimic.

Why a custom IK term: LeKiwi's articulation root is fixed in the world and the ``base`` link moves through
virtual joints. Isaac Lab's ``DifferentialInverseKinematicsAction`` interprets absolute pose commands in the
*root* frame, so a static-base demo recorded that way cannot be replayed after the base has moved.
``BaseFrameDifferentialIKAction`` takes the command in the moving ``base`` link frame instead and converts it to
the root frame every step, which makes one action convention valid for both static and mobile episodes.
"""

from __future__ import annotations

from dataclasses import MISSING

import isaaclab.envs.mdp as mdp
import isaaclab.utils.math as math_utils
import torch
from isaaclab.envs.mdp.actions.task_space_actions import DifferentialInverseKinematicsAction
from isaaclab.utils import configclass

ARM_JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
BASE_BODY_NAME = "base"
EEF_BODY_NAME = "gripper"
EEF_FRAME_NAME = "gripper"
"""``ee_frame`` target used as the Mimic / action end effector: the gripper link."""
TCP_FRAME_NAME = "tcp"
"""``ee_frame`` target added by the pick-cube task: grasp point between the closed fingertips."""


def frame_index(ee_frame, name: str) -> int:
    """Index of a FrameTransformer target by name.

    Never index targets by their position in the config: Isaac Lab groups targets by body and keeps the frames of one
    body in a ``set`` (frame_transformer.py), so two targets on the same body (here ``gripper`` and ``tcp``) come out
    in an order that depends on Python's per-process string hash seed.
    """
    return ee_frame.data.target_frame_names.index(name)
GRIPPER_OPEN_POS = 1.0
GRIPPER_CLOSE_POS = 0.0


def base_pose_in_root(asset, base_body_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Pose of the moving base link expressed in the (fixed) articulation root frame."""
    return math_utils.subtract_frame_transforms(
        asset.data.root_pos_w,
        asset.data.root_quat_w,
        asset.data.body_pos_w[:, base_body_idx],
        asset.data.body_quat_w[:, base_body_idx],
    )


class BaseFrameDifferentialIKAction(DifferentialInverseKinematicsAction):
    """Absolute-pose differential IK whose command is expressed in the ``base`` link frame.

    The SO-101 arm has 5 joints, so a 6-D pose target is generally over-constrained. Plain DLS then trades position
    error for orientation error (centimetres for a few degrees). This term solves a weighted DLS where the
    orientation residual and Jacobian rows are scaled by ``cfg.rot_weight``, so position is tracked first.
    """

    cfg: BaseFrameDifferentialIKActionCfg

    def __init__(self, cfg: BaseFrameDifferentialIKActionCfg, env):
        if cfg.controller.use_relative_mode:
            raise ValueError("BaseFrameDifferentialIKAction only supports absolute pose commands.")
        super().__init__(cfg, env)
        body_ids, _ = self._asset.find_bodies(cfg.base_body_name)
        self._base_body_idx = body_ids[0]

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        cmd = self.raw_actions * self._scale
        base_pos_r, base_quat_r = base_pose_in_root(self._asset, self._base_body_idx)
        pos_r, quat_r = math_utils.combine_frame_transforms(base_pos_r, base_quat_r, cmd[:, :3], cmd[:, 3:7])
        self._processed_actions[:] = torch.cat([pos_r, quat_r], dim=-1)
        ee_pos_curr, ee_quat_curr = self._compute_frame_pose()
        self._ik_controller.set_command(self._processed_actions, ee_pos_curr, ee_quat_curr)

    def apply_actions(self):
        ee_pos, ee_quat = self._compute_frame_pose()
        joint_pos = self._asset.data.joint_pos[:, self._joint_ids]
        pos_err, rot_err = math_utils.compute_pose_error(
            ee_pos, ee_quat, self._processed_actions[:, :3], self._processed_actions[:, 3:7], rot_error_type="axis_angle"
        )
        w = self.cfg.rot_weight
        err = torch.cat([pos_err, w * rot_err], dim=-1).unsqueeze(-1)
        jac = self._compute_frame_jacobian().clone()
        jac[:, 3:, :] *= w
        lam = self.cfg.damping
        jjt = jac @ jac.transpose(1, 2) + (lam**2) * torch.eye(6, device=self.device)
        delta = (jac.transpose(1, 2) @ torch.linalg.solve(jjt, err)).squeeze(-1)
        self._asset.set_joint_position_target(joint_pos + delta, self._joint_ids)


@configclass
class BaseFrameDifferentialIKActionCfg(mdp.DifferentialInverseKinematicsActionCfg):
    class_type: type = BaseFrameDifferentialIKAction
    base_body_name: str = BASE_BODY_NAME
    rot_weight: float = 0.1
    """Scale on orientation residual/Jacobian rows: 0.1 makes 0.1 rad of orientation error weigh like 1 cm."""
    damping: float = 0.04


class PerturbedBaseVelocityAction(mdp.JointVelocityAction):
    """Base joint velocity plus an execution-only offset ``env.base_exec_offset`` (root frame, (N, 3)).

    The recorder stores the action *before* this term processes it, so generated demos keep the clean controller
    command as the label while the robot actually executes a perturbed one (DART, Laskey et al. 2017).
    """

    def process_actions(self, actions: torch.Tensor):
        super().process_actions(actions)
        offset = getattr(self._env, "base_exec_offset", None)
        if offset is not None:
            self._processed_actions += offset


@configclass
class PerturbedBaseVelocityActionCfg(mdp.JointVelocityActionCfg):
    class_type: type = PerturbedBaseVelocityAction


@configclass
class LeKiwiMimicActionsCfg:
    """Action = [gripper-link pos (3), quat wxyz (4) in base frame | gripper (1) | base joint vel (3)] = 11 dims."""

    arm_action: mdp.ActionTermCfg = MISSING
    gripper_action: mdp.ActionTermCfg = MISSING
    wheel_action: mdp.ActionTermCfg = MISSING


def init_lekiwi_mimic_action_cfg(action_cfg):
    """Fill the LeKiwi actions used for scripted source demos, Mimic generation and policy rollout.

    The gripper is binary (+1 open / -1 close) like LeIsaac's state-machine actions, so Mimic replays it unchanged.
    """
    action_cfg.arm_action = BaseFrameDifferentialIKActionCfg(
        asset_name="robot",
        joint_names=ARM_JOINT_NAMES,
        body_name=EEF_BODY_NAME,
        controller=mdp.DifferentialIKControllerCfg(command_type="pose", ik_method="dls", ik_params={"lambda_val": 0.04}),
    )
    action_cfg.gripper_action = mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["gripper"],
        open_command_expr={"gripper": GRIPPER_OPEN_POS},
        close_command_expr={"gripper": GRIPPER_CLOSE_POS},
    )
    action_cfg.wheel_action = PerturbedBaseVelocityActionCfg(
        asset_name="robot", joint_names=["base_x", "base_y", "base_theta"], scale=1.0
    )
    return action_cfg
