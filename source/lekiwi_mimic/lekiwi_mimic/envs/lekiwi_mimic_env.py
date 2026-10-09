"""Mimic environment for LeKiwi with a moving base.

Differences from ``leisaac.enhance.envs.ManagerBasedRLLeIsaacMimicEnv``:

* Poses (eef, objects, targets) are expressed in the moving ``base`` link frame instead of the fixed articulation
  root, so object-centric subtask segments recorded with a static base stay valid after the base has moved.
* The action carries a base velocity command (3 dims) produced by a holonomic P controller owned by the env.
  ``get_navigation_state`` follows the Isaac Lab Mimic API added in isaac-sim/IsaacLab#3992, which is newer than
  the Isaac Lab version LeIsaac pins, so the generator in ``lekiwi_mimic.datagen`` drives navigation itself.
"""

from __future__ import annotations

from collections.abc import Sequence

import isaaclab.utils.math as PoseUtils
import torch
from isaaclab.envs import ManagerBasedRLMimicEnv

from ..actions import BASE_BODY_NAME, EEF_FRAME_NAME, base_pose_in_root, frame_index
from ..controllers.holonomic_p import HolonomicPController
from ..frames import FORWARD_AXIS_YAW, canonicalize_yaw


class LeKiwiMimicEnv(ManagerBasedRLMimicEnv):
    def __init__(self, cfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        robot = self.scene["robot"]
        self._base_body_idx = robot.find_bodies(BASE_BODY_NAME)[0][0]
        self._eef_name = list(self.cfg.subtask_configs.keys())[0]
        self.nav_controller = HolonomicPController(cfg.nav_controller, self.num_envs, self.device)
        # execution-only base velocity offset (root frame), see actions.PerturbedBaseVelocityAction
        self.base_exec_offset = torch.zeros(self.num_envs, 3, device=self.device)

    # ------------------------------------------------------------------
    # base helpers
    # ------------------------------------------------------------------

    def base_pose_root(self, env_ids: Sequence[int] | slice | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        env_ids = slice(None) if env_ids is None else env_ids
        pos, quat = base_pose_in_root(self.scene["robot"], self._base_body_idx)
        return pos[env_ids], quat[env_ids]

    def base_pose_2d(self, env_ids: Sequence[int] | slice | None = None) -> torch.Tensor:
        """Planar base pose ``(x, y, yaw)`` in the root frame; yaw is the base link's z rotation."""
        pos, quat = self.base_pose_root(env_ids)
        yaw = PoseUtils.euler_xyz_from_quat(quat)[2]
        return torch.cat([pos[:, :2], PoseUtils.wrap_to_pi(yaw).unsqueeze(-1)], dim=-1)

    def heading(self, env_ids=None) -> torch.Tensor:
        """Yaw of the robot's forward axis in the root frame."""
        return PoseUtils.wrap_to_pi(self.base_pose_2d(env_ids)[:, 2] + FORWARD_AXIS_YAW)

    def objects_in_root(self, env_ids=None) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        env_ids = slice(None) if env_ids is None else env_ids
        robot = self.scene["robot"]
        out = {}
        for name, obj in self.scene.rigid_objects.items():
            out[name] = PoseUtils.subtract_frame_transforms(
                robot.data.root_pos_w[env_ids],
                robot.data.root_quat_w[env_ids],
                obj.data.root_pos_w[env_ids],
                obj.data.root_quat_w[env_ids],
            )
        return out

    # ------------------------------------------------------------------
    # Mimic API
    # ------------------------------------------------------------------

    def get_robot_eef_pose(self, eef_name: str, env_ids: Sequence[int] | None = None) -> torch.Tensor:
        env_ids = slice(None) if env_ids is None else env_ids
        ee_frame = self.scene["ee_frame"]  # source prim is Robot/base
        i = frame_index(ee_frame, EEF_FRAME_NAME)
        pos = ee_frame.data.target_pos_source[env_ids, i]
        quat = ee_frame.data.target_quat_source[env_ids, i]
        return PoseUtils.make_pose(pos, PoseUtils.matrix_from_quat(quat))

    def get_object_poses(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        env_ids = slice(None) if env_ids is None else env_ids
        base_pos, base_quat = self.base_pose_root(env_ids)
        poses = {}
        for name, (pos_r, quat_r) in self.objects_in_root(env_ids).items():
            pos_b, quat_b = PoseUtils.subtract_frame_transforms(base_pos, base_quat, pos_r, quat_r)
            mode = self.cfg.object_frame_mode.get(name, "pose")
            if mode == "bearing":
                # yaw = direction of the object seen from the base; the object's own yaw is ignored. A 5-DOF arm's
                # top-down grasp is tied to that bearing (shoulder_pan), so Mimic should rotate source segments by the
                # change in bearing rather than by the object's yaw.
                yaw = torch.atan2(pos_b[:, 1], pos_b[:, 0])
                zero = torch.zeros_like(yaw)
                quat_b = PoseUtils.quat_from_euler_xyz(zero, zero, yaw)
            elif mode.startswith("symmetric"):
                # "symmetric:<period>" - canonicalize the yaw of a rotationally symmetric object
                roll, pitch, yaw = PoseUtils.euler_xyz_from_quat(quat_b)
                yaw = canonicalize_yaw(PoseUtils.wrap_to_pi(yaw), float(mode.split(":")[1]))
                quat_b = PoseUtils.quat_from_euler_xyz(roll, pitch, yaw)
            poses[name] = PoseUtils.make_pose(pos_b, PoseUtils.matrix_from_quat(quat_b))
        return poses

    def target_eef_pose_to_action(
        self,
        target_eef_pose_dict: dict,
        gripper_action_dict: dict,
        action_noise_dict: dict | None = None,
        env_id: int = 0,
    ) -> torch.Tensor:
        (target_eef_pose,) = target_eef_pose_dict.values()
        pos, rot = PoseUtils.unmake_pose(target_eef_pose)
        pose_action = torch.cat([pos, PoseUtils.quat_from_matrix(rot)], dim=0)
        if action_noise_dict is not None:
            pose_action = pose_action + action_noise_dict[self._eef_name] * torch.randn_like(pose_action)
        (gripper_action,) = gripper_action_dict.values()
        env_ids = torch.tensor([env_id], device=self.device)
        base_cmd = self.nav_controller.compute(self.base_pose_2d(env_ids), env_ids)[0]
        return torch.cat([pose_action, gripper_action, base_cmd], dim=0).unsqueeze(0)

    def action_to_target_eef_pose(self, action: torch.Tensor) -> dict[str, torch.Tensor]:
        pos, quat = action[:, :3], action[:, 3:7]
        return {self._eef_name: PoseUtils.make_pose(pos, PoseUtils.matrix_from_quat(quat)).clone()}

    def actions_to_gripper_actions(self, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        return {self._eef_name: actions[:, 7:8]}

    def get_subtask_term_signals(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        env_ids = slice(None) if env_ids is None else env_ids
        return {name: sig[env_ids] for name, sig in self.obs_buf["subtask_terms"].items()}

    def get_navigation_state(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        env_ids = slice(None) if env_ids is None else env_ids
        return {k: v[env_ids] for k, v in self.nav_controller.navigation_state().items()}

    def _reset_idx(self, env_ids: Sequence[int]):
        super()._reset_idx(env_ids)
        self.nav_controller.stop(torch.as_tensor(env_ids, device=self.device))
        self.base_exec_offset[env_ids] = 0.0
