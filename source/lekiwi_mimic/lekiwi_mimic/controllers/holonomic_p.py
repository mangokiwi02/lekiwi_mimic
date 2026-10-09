"""Proportional controller for a holonomic (omni-wheel) base.

Outputs the velocity of the virtual base joints ``(vx_root, vy_root, yaw_rate)`` that LeIsaac's
``JointVelocityActionCfg(joint_names=["base_x", "base_y", "base_theta"])`` consumes directly.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..frames import wrap_to_pi


@dataclass
class HolonomicPControllerCfg:
    kp_xy: float = 1.5
    kp_yaw: float = 2.0
    max_lin_vel: float = 0.3
    """m/s, close to the real LeKiwi's comfortable speed."""
    max_ang_vel: float = 1.0
    """rad/s"""
    pos_tol: float = 0.02
    yaw_tol: float = 0.05
    settle_steps: int = 10
    """Consecutive steps within tolerance before the goal counts as reached."""


class HolonomicPController:
    """Batched P controller with saturation and a debounced goal-reached flag."""

    def __init__(self, cfg: HolonomicPControllerCfg, num_envs: int, device: str | torch.device = "cpu"):
        self.cfg = cfg
        self.goal = torch.zeros(num_envs, 3, device=device)
        self._in_tol_count = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.is_navigating = torch.zeros(num_envs, dtype=torch.bool, device=device)

    def set_goal(self, goal: torch.Tensor, env_ids: torch.Tensor | slice | None = None) -> None:
        env_ids = slice(None) if env_ids is None else env_ids
        self.goal[env_ids] = goal
        self._in_tol_count[env_ids] = 0
        self.is_navigating[env_ids] = True

    def stop(self, env_ids: torch.Tensor | slice | None = None) -> None:
        env_ids = slice(None) if env_ids is None else env_ids
        self.is_navigating[env_ids] = False

    @property
    def goal_reached(self) -> torch.Tensor:
        return self._in_tol_count >= self.cfg.settle_steps

    def errors(self, pose: torch.Tensor, env_ids: torch.Tensor | slice | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        env_ids = slice(None) if env_ids is None else env_ids
        e_xy = self.goal[env_ids, :2] - pose[:, :2]
        e_yaw = wrap_to_pi(self.goal[env_ids, 2] - pose[:, 2])
        return e_xy, e_yaw

    def compute(self, pose: torch.Tensor, env_ids: torch.Tensor | slice | None = None) -> torch.Tensor:
        """Compute base joint velocities and update the goal-reached debounce for ``env_ids``.

        Args:
            pose: (K, 3) current base ``(x, y, yaw)`` in the root frame for the envs in ``env_ids``.
            env_ids: envs to compute for; all envs when None.

        Returns:
            (K, 3) ``(vx, vy, yaw_rate)`` in the root frame; zero for envs that are not navigating.
        """
        cfg = self.cfg
        env_ids = slice(None) if env_ids is None else env_ids
        e_xy, e_yaw = self.errors(pose, env_ids)

        v = cfg.kp_xy * e_xy
        speed = torch.linalg.vector_norm(v, dim=-1, keepdim=True)
        v = v * torch.clamp(cfg.max_lin_vel / speed.clamp_min(1e-9), max=1.0)
        w = torch.clamp(cfg.kp_yaw * e_yaw, -cfg.max_ang_vel, cfg.max_ang_vel)
        cmd = torch.cat([v, w.unsqueeze(-1)], dim=-1)

        in_tol = (torch.linalg.vector_norm(e_xy, dim=-1) < cfg.pos_tol) & (e_yaw.abs() < cfg.yaw_tol)
        count = self._in_tol_count[env_ids]
        self._in_tol_count[env_ids] = torch.where(in_tol, count + 1, torch.zeros_like(count))

        return torch.where(self.is_navigating[env_ids].unsqueeze(-1), cmd, torch.zeros_like(cmd))

    def navigation_state(self) -> dict[str, torch.Tensor]:
        """Same keys as Isaac Lab Mimic's ``get_navigation_state`` (isaac-sim/IsaacLab#3992)."""
        return {"is_navigating": self.is_navigating.clone(), "navigation_goal_reached": self.goal_reached.clone()}
