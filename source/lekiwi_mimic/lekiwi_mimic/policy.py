"""Small egocentric behavior-cloning policy used to compare datasets (static source demos vs generated far-pick).

Observation (all in the moving base frame, so the policy is invariant to where the robot is in the room):
    cube position (3) + cube yaw as sin/cos of 4*yaw (2, cube has 90 deg symmetry)
    eef position (3) + eef quaternion (4) + gripper joint position (1)
    optionally (``base_vel=True``) the measured base velocity in the body frame (3): without it the policy cannot
    tell how fast it is approaching and so when to start braking
Action:
    eef target position as a delta from the current eef position (3) + eef target quaternion (4) + gripper command (1)
    + base velocity in the *body* frame (3)

The position is predicted as a delta because the absolute target is almost equal to the current eef position, which
an MLP learns to copy ("copycat" problem) and the robot then stalls; normalized deltas keep the signal.

The env consumes base velocity in the root frame, so ``body_to_root_base_vel`` converts at rollout time.
With ``history`` > 1 the observation stacks past frames; with ``chunk`` > 1 the policy predicts the next ``chunk``
actions and ``RolloutHelper`` averages overlapping predictions (temporal ensembling), as in ACT.
"""

from __future__ import annotations

import h5py
import numpy as np
import torch
import torch.nn as nn

from .frames import body_to_world_xy, world_to_body_xy

OBS_DIM = 13
ACT_DIM = 11


def _yaw_from_quat(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q.unbind(-1)
    return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def make_obs(
    cube_pose_b: torch.Tensor,
    ee_pose_b: torch.Tensor,
    gripper_pos: torch.Tensor,
    base_vel_b: torch.Tensor | None = None,
) -> torch.Tensor:
    yaw4 = 4 * _yaw_from_quat(cube_pose_b[..., 3:7])
    parts = [cube_pose_b[..., :3], torch.sin(yaw4)[..., None], torch.cos(yaw4)[..., None], ee_pose_b, gripper_pos[..., None]]
    if base_vel_b is not None:
        parts.append(base_vel_b)
    return torch.cat(parts, dim=-1)


def base_vel_body(base_joint_vel: torch.Tensor, base_yaw: torch.Tensor) -> torch.Tensor:
    """(.., 3) base_x/base_y/base_theta joint velocities (root frame) -> (vx, vy, wz) in the body frame."""
    return torch.cat([world_to_body_xy(base_joint_vel[..., :2], base_yaw), base_joint_vel[..., 2:3]], dim=-1)


def root_to_body_base_vel(action: torch.Tensor, base_yaw: torch.Tensor) -> torch.Tensor:
    out = action.clone()
    out[..., 8:10] = world_to_body_xy(action[..., 8:10], base_yaw)
    return out


def body_to_root_base_vel(action: torch.Tensor, base_yaw: torch.Tensor) -> torch.Tensor:
    out = action.clone()
    out[..., 8:10] = body_to_world_xy(action[..., 8:10], base_yaw)
    return out


def _stack_history(obs: torch.Tensor, history: int) -> torch.Tensor:
    """(T, D) -> (T, history*D): each row holds obs[t-history+1 .. t], padded with obs[0]."""
    idx = torch.arange(obs.shape[0]).unsqueeze(1) + torch.arange(-history + 1, 1).unsqueeze(0)
    return obs[idx.clamp_min(0)].reshape(obs.shape[0], -1)


def _chunk_actions(act: torch.Tensor, ee_pos: torch.Tensor, chunk: int) -> torch.Tensor:
    """(T, A) absolute-target actions -> (T, chunk*A): a[t..t+chunk-1] (padded with the last action), positions
    expressed relative to the eef position at t."""
    T = act.shape[0]
    idx = (torch.arange(T).unsqueeze(1) + torch.arange(chunk).unsqueeze(0)).clamp_max(T - 1)
    out = act[idx].clone()  # (T, chunk, A)
    out[..., :3] -= ee_pos.unsqueeze(1)
    return out.reshape(T, -1)


def demo_arrays(demo, history: int, chunk: int, obs_cfg: dict, gripper_joint_index: int = 8):
    """One HDF5 demo group -> (stacked obs, chunked actions, info). Joint order puts base_x/y/theta at 0..2."""
    o = demo["obs"]
    bp = torch.from_numpy(o["base_pose"][:]).float()
    yaw = torch.atan2(bp[:, 2], bp[:, 3])
    cube = torch.from_numpy(o["cube_pose_base"][:]).float()
    base_vel = None
    if obs_cfg.get("base_vel"):
        base_vel = base_vel_body(torch.from_numpy(o["joint_vel"][:, :3]).float(), yaw)
    obs = make_obs(
        cube,
        torch.from_numpy(o["ee_pose_base"][:]).float(),
        torch.from_numpy(o["joint_pos"][:, gripper_joint_index]).float(),
        base_vel,
    )
    act = root_to_body_base_vel(torch.from_numpy(demo["actions"][:]).float(), yaw)
    ee_pos = torch.from_numpy(o["ee_pose_base"][:, :3]).float()
    info = {"cube_dist": cube[:, :2].norm(dim=-1).numpy()}
    return _stack_history(obs, history).numpy(), _chunk_actions(act, ee_pos, chunk).numpy(), info


def load_dataset(
    path: str,
    gripper_joint_index: int,
    max_demos: int | None = None,
    history: int = 1,
    chunk: int = 1,
    obs_cfg: dict | None = None,
):
    """Return (obs, act, n_demos) float32 arrays from an Isaac Lab HDF5 file, successful demos only."""
    obs_l, act_l = [], []
    with h5py.File(path, "r") as f:
        demos = [k for k in f["data"].keys() if f["data"][k].attrs.get("success", True)]
        demos = sorted(demos, key=lambda k: int(k.split("_")[-1]))[:max_demos]
        for k in demos:
            o, a, _ = demo_arrays(f["data"][k], history, chunk, obs_cfg or {}, gripper_joint_index)
            obs_l.append(o)
            act_l.append(a)
    return np.concatenate(obs_l), np.concatenate(act_l), len(demos)


class BCPolicy(nn.Module):
    """MLP over ``history`` stacked observations predicting ``chunk`` future actions (ACT-style chunking)."""

    def __init__(self, hidden: int = 512, depth: int = 4, history: int = 1, chunk: int = 1, base_vel: bool = False):
        super().__init__()
        self.history, self.chunk = history, chunk
        self.obs_cfg = {"base_vel": base_vel}
        self.kwargs = {"hidden": hidden, "depth": depth, "history": history, "chunk": chunk, "base_vel": base_vel}
        self.obs_dim = OBS_DIM + 3 * base_vel
        in_dim, out_dim = self.obs_dim * history, ACT_DIM * chunk
        layers, d = [], in_dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.LayerNorm(hidden), nn.GELU()]
            d = hidden
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)
        self.register_buffer("obs_mean", torch.zeros(in_dim))
        self.register_buffer("obs_std", torch.ones(in_dim))
        self.register_buffer("act_mean", torch.zeros(out_dim))
        self.register_buffer("act_std", torch.ones(out_dim))

    def set_normalization(self, obs: np.ndarray, act: np.ndarray) -> None:
        self.obs_mean.copy_(torch.from_numpy(obs.mean(0)))
        self.obs_std.copy_(torch.from_numpy(obs.std(0) + 1e-4))
        self.act_mean.copy_(torch.from_numpy(act.mean(0)))
        self.act_std.copy_(torch.from_numpy(act.std(0) + 1e-4))

    def forward_normalized(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net((obs - self.obs_mean) / self.obs_std)

    @torch.no_grad()
    def act_chunk(self, obs_hist: torch.Tensor) -> torch.Tensor:
        """(N, history*OBS_DIM) -> (N, chunk, ACT_DIM) with positions relative to the current eef position."""
        a = self.forward_normalized(obs_hist) * self.act_std + self.act_mean
        return a.reshape(a.shape[0], self.chunk, ACT_DIM)


class RolloutHelper:
    """Observation history + temporal ensembling of overlapping action chunks (as in ACT) for batched rollouts."""

    def __init__(self, policy: BCPolicy, num_envs: int, device, ensemble_decay: float = 0.1):
        self.p = policy
        self.H, self.K = policy.history, policy.chunk
        self.obs = torch.zeros(num_envs, self.H, policy.obs_dim, device=device)
        # preds[n, age, step]: absolute action predicted `age` steps ago for `step` steps after its prediction
        self.preds = torch.zeros(num_envs, self.K, self.K, ACT_DIM, device=device)
        self.count = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.w = torch.exp(-ensemble_decay * torch.arange(self.K, device=device, dtype=torch.float))

    def reset(self, env_ids=None):
        env_ids = slice(None) if env_ids is None else env_ids
        self.count[env_ids] = 0

    def __call__(self, obs: torch.Tensor, ee_pos: torch.Tensor) -> torch.Tensor:
        first = self.count == 0
        self.obs[first] = obs[first].unsqueeze(1).expand(-1, self.H, -1)
        self.obs = torch.cat([self.obs[:, 1:], obs.unsqueeze(1)], dim=1)
        chunk = self.p.act_chunk(self.obs.reshape(obs.shape[0], -1))
        chunk[..., :3] += ee_pos.unsqueeze(1)  # absolute targets
        self.preds = torch.cat([chunk.unsqueeze(1), self.preds[:, :-1]], dim=1)
        ages = torch.arange(self.K, device=obs.device)
        valid = (ages.unsqueeze(0) <= self.count.unsqueeze(1)).float()  # (N, K)
        cur = self.preds[:, ages, ages]  # (N, K, A): prediction for "now" made `age` steps ago
        w = (valid * self.w).unsqueeze(-1)
        a = (cur * w).sum(1) / w.sum(1)
        a[:, 3:7] = a[:, 3:7] / a[:, 3:7].norm(dim=-1, keepdim=True).clamp_min(1e-6)
        a[:, 7] = torch.where(a[:, 7] >= 0, 1.0, -1.0)
        self.count += 1
        return a
