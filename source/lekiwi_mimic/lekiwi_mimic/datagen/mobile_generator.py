"""Mobile Mimic data generator: NAV (P controller) -> SETTLE -> MANIP (Mimic) per subtask.

Isaac Lab Mimic at the version LeIsaac pins (before isaac-sim/IsaacLab#3992) has no navigation hooks, so this
generator drives navigation itself and reuses Mimic's source selection, object-centric transform and interpolation
(``generate_eef_subtask_trajectory`` / ``merge_eef_subtask_trajectory``) for the manipulation segments.

Navigation goals are not taken from demos. For each navigation phase a source demo is sampled from the pool, the
object's position in the base frame at the start of that demo's subtask is read, and the base goal is the pose that
reproduces that relative position (plus noise) while facing the object along the approach direction. Because the
Mimic env expresses all poses in the moving base frame, the manipulation segment that follows is the same problem
as in the static source demos.

Only single-arm, sequential subtasks are supported (no subtask constraints / skillgen).
"""

from __future__ import annotations

import asyncio
import math
import random
from dataclasses import dataclass, field

import torch
from isaaclab.managers import TerminationTermCfg
from isaaclab_mimic.datagen.data_generator import DataGenerator
from isaaclab_mimic.datagen.waypoint import MultiWaypoint, Waypoint

from ..frames import base_goal_from_object


@dataclass
class NavPhaseStats:
    subtask_ind: int
    start: list[float]
    goal: list[float]
    steps: int
    reached: bool
    overshoot: bool = False
    """DART: this phase got the forward push past the goal."""
    stall: bool = False
    """DART: this phase got a brake push (slow while still far from the goal)."""


@dataclass
class EpisodeStats:
    nav: list[NavPhaseStats] = field(default_factory=list)
    src_demo_ind: int | None = None


class ExecPerturbation:
    """Execution-only base velocity offset for one navigation phase (DART), see ``MobileNavCfg.exec_noise_*``.

    An Ornstein-Uhlenbeck process (stronger near the goal, off inside the settle radius) plus, in a fraction of the
    episodes, a push toward the goal when the base first comes within ``overshoot_dist`` (too fast / past the goal),
    and in another fraction a push away from the goal at a random distance (slow while still far).
    """

    def __init__(self, cfg, dt: float):
        self.cfg, self.dt = cfg, dt
        self.x: torch.Tensor | None = None
        self.overshoot = random.random() < cfg.overshoot_prob
        self.push_dir: torch.Tensor | None = None
        self.push_left = 0.0
        self.stall = random.random() < cfg.stall_prob
        self.stall_at = random.uniform(*cfg.stall_dist_range)
        self.stall_dir: torch.Tensor | None = None
        self.stall_left = 0.0

    def step(self, base: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        out = torch.zeros(3, device=base.device)
        if cfg.exec_noise_std <= 0 and not self.overshoot and not self.stall:
            return out
        e = goal[:2] - base[:2]
        d = float(e.norm())
        if cfg.exec_noise_std > 0:
            a = self.dt / cfg.exec_noise_tau
            if self.x is None:
                self.x = torch.zeros(3, device=base.device)
            self.x = self.x * (1 - a) + math.sqrt(2 * a) * torch.randn_like(self.x)
            if d > cfg.exec_noise_settle_radius:
                gain = cfg.exec_noise_near_gain if d < cfg.exec_noise_near_dist else 1.0
                std = torch.tensor([cfg.exec_noise_std, cfg.exec_noise_std, cfg.exec_noise_yaw_std], device=base.device)
                out = gain * std * self.x
        if self.overshoot and self.push_dir is None and d < cfg.overshoot_dist:
            self.push_dir, self.push_left = e / max(d, 1e-6), cfg.overshoot_time
        if self.push_dir is not None and self.push_left > 0:
            out[:2] += cfg.overshoot_speed * self.push_dir
            self.push_left -= self.dt
        if self.stall and self.stall_dir is None and d < self.stall_at:
            self.stall_dir, self.stall_left = -e / max(d, 1e-6), cfg.stall_time
        if self.stall_dir is not None and self.stall_left > 0:
            out[:2] += cfg.stall_speed * self.stall_dir
            self.stall_left -= self.dt
        return out


class MobileDataGenerator(DataGenerator):
    def __init__(self, env, src_demo_datagen_info_pool=None, dataset_path=None, demo_keys=None):
        super().__init__(env, src_demo_datagen_info_pool, dataset_path, demo_keys)
        if len(self.env_cfg.subtask_configs) != 1:
            raise ValueError("MobileDataGenerator supports a single end effector.")
        if self.env_cfg.task_constraint_configs:
            raise ValueError("MobileDataGenerator does not support subtask constraints.")
        self.eef_name = next(iter(self.env_cfg.subtask_configs))
        self.subtask_configs = self.env_cfg.subtask_configs[self.eef_name]
        self.nav_cfg = self.env_cfg.nav
        self.last_episode_stats: dict[int, EpisodeStats] = {}

    # ------------------------------------------------------------------
    # navigation
    # ------------------------------------------------------------------

    def sample_nav_goal(self, env_id: int, subtask_ind: int, randomized_boundaries: dict) -> torch.Tensor:
        """Base goal (x, y, yaw) in the root frame for the navigation phase before ``subtask_ind``."""
        obj_name = self.subtask_configs[subtask_ind].object_ref
        if obj_name is None:
            raise ValueError(f"Subtask {subtask_ind} needs an object_ref to navigate to.")
        if self.nav_cfg.goal_rel_xy is not None:
            rel_xy = torch.tensor(self.nav_cfg.goal_rel_xy, device=self.env.device, dtype=torch.float32)
        else:
            pool = self.src_demo_datagen_info_pool
            k = random.randrange(len(pool.datagen_infos))
            start = randomized_boundaries[self.eef_name][k, subtask_ind, 0]
            rel_xy = pool.datagen_infos[k].object_poses[obj_name][start][:2, 3].to(self.env.device)
        rel_xy = rel_xy + self.nav_cfg.rel_xy_noise * torch.randn_like(rel_xy)

        obj_pos_root = self.env.objects_in_root([env_id])[obj_name][0]
        base = self.env.base_pose_2d([env_id])
        return base_goal_from_object(obj_pos_root[:, :2], rel_xy.unsqueeze(0), base[:, :2])[0]

    def _hold_waypoint(self, env_id: int, prev_traj: list[Waypoint] | None, lift: float = 0.0) -> Waypoint:
        """Arm target held during navigation: the last commanded target, else the current eef pose (gripper open),
        raised by ``lift`` so the gripper clears the object while the base drives up to it."""
        if prev_traj:
            last = prev_traj[-1]
            pose, gripper = last.pose.clone(), last.gripper_action
        else:
            pose = self.env.get_robot_eef_pose(eef_name=self.eef_name, env_ids=[env_id])[0].clone()
            gripper = self.src_demo_datagen_info_pool.datagen_infos[0].gripper_action[self.eef_name][0].to(self.env.device)
        pose[2, 3] += lift
        return Waypoint(pose=pose, gripper_action=gripper, noise=0.0)

    async def _execute(self, waypoint: Waypoint, env_id, success_term, env_action_queue, buf: dict) -> None:
        res = await MultiWaypoint({self.eef_name: waypoint}).execute(
            env=self.env, success_term=success_term, env_id=env_id, env_action_queue=env_action_queue
        )
        buf["actions"].extend(res["actions"])
        buf["success"] = buf["success"] or res["success"]

    async def navigate(
        self, env_id, subtask_ind, randomized_boundaries, prev_traj, success_term, env_action_queue, buf
    ) -> NavPhaseStats:
        env = self.env
        ids = torch.tensor([env_id], device=env.device)
        goal = self.sample_nav_goal(env_id, subtask_ind, randomized_boundaries)
        start = env.base_pose_2d([env_id])[0]
        hold = self._hold_waypoint(env_id, prev_traj, lift=self.nav_cfg.hold_lift)
        # raise the arm before driving
        for _ in range(self.nav_cfg.hold_lift_steps):
            await self._execute(hold, env_id, success_term, env_action_queue, buf)

        env.nav_controller.set_goal(goal.unsqueeze(0), ids)
        perturb = ExecPerturbation(self.nav_cfg, env.step_dt)
        steps, reached = 0, False
        while steps < self.nav_cfg.max_nav_steps:
            env.base_exec_offset[env_id] = perturb.step(env.base_pose_2d([env_id])[0], goal)
            await self._execute(hold, env_id, success_term, env_action_queue, buf)
            steps += 1
            if bool(env.get_navigation_state([env_id])["navigation_goal_reached"][0]):
                reached = True
                break
        env.base_exec_offset[env_id] = 0.0
        env.nav_controller.stop(ids)
        if reached:
            for _ in range(self.nav_cfg.settle_steps):
                await self._execute(hold, env_id, success_term, env_action_queue, buf)
        return NavPhaseStats(subtask_ind, start.tolist(), goal.tolist(), steps, reached, perturb.overshoot, perturb.stall)

    # ------------------------------------------------------------------
    # main loop
    # ------------------------------------------------------------------

    async def generate(  # noqa: C901
        self,
        env_id: int,
        success_term: TerminationTermCfg,
        env_reset_queue: asyncio.Queue | None = None,
        env_action_queue: asyncio.Queue | None = None,
        pause_subtask: bool = False,
        export_demo: bool = True,
        motion_planner=None,
    ) -> dict:
        env = self.env
        env_id_tensor = torch.tensor([env_id], dtype=torch.int64, device=env.device)
        env.recorder_manager.reset(env_ids=env_id_tensor)
        await env_reset_queue.put(env_id)
        await env_reset_queue.join()
        initial_state = env.scene.get_state(is_relative=True)

        async with self.src_demo_datagen_info_pool.asyncio_lock:
            boundaries = self.randomize_subtask_boundaries()

        stats = EpisodeStats()
        buf = {"actions": [], "success": False}
        # link poses (base, eef) are stale right after a joint-teleport reset; hold still for a few steps first
        hold = self._hold_waypoint(env_id, None)
        for _ in range(self.nav_cfg.warmup_steps):
            await self._execute(hold, env_id, success_term, env_action_queue, buf)
        selected = {self.eef_name: None}
        prev_traj: list[Waypoint] | None = None
        aborted = False

        for subtask_ind in range(len(self.subtask_configs)):
            if subtask_ind in self.nav_cfg.navigate_before_subtasks:
                nav = await self.navigate(
                    env_id, subtask_ind, boundaries, prev_traj, success_term, env_action_queue, buf
                )
                stats.nav.append(nav)
                if not nav.reached:
                    aborted = True
                    break
            subtask_traj = self.generate_eef_subtask_trajectory(env_id, self.eef_name, subtask_ind, boundaries, {}, selected)
            if stats.src_demo_ind is None:
                stats.src_demo_ind = selected[self.eef_name]
            seq = self.merge_eef_subtask_trajectory(env_id, self.eef_name, subtask_ind, prev_traj, subtask_traj)
            for waypoint in seq:
                await self._execute(waypoint, env_id, success_term, env_action_queue, buf)
            prev_traj = seq
            if pause_subtask:
                input(f"Paused after subtask {subtask_ind}. Press Enter to continue...")

        success = buf["success"] and not aborted
        env.recorder_manager.set_success_to_episodes(
            env_id_tensor, torch.tensor([[success]], dtype=torch.bool, device=env.device)
        )
        if export_demo:
            env.recorder_manager.export_episodes(env_id_tensor)
        self.last_episode_stats[env_id] = stats

        actions = torch.cat(buf["actions"], dim=0) if buf["actions"] else torch.empty(0)
        return dict(initial_state=initial_state, actions=actions, success=success, stats=stats)
