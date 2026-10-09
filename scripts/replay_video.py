"""Replay episodes from an HDF5 dataset and render a chase-cam + overhead video.

    scripts/run.sh scripts/replay_video.py --headless --enable_cameras --dataset datasets/far_test.hdf5 --episodes 0 1
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", required=True)
parser.add_argument("--task", default="LeKiwiMimic-PickCube-Far-Mimic-v0")
parser.add_argument("--episodes", type=int, nargs="*", default=[0])
parser.add_argument("--out", default="outputs/videos")
parser.add_argument("--every", type=int, default=2, help="keep every n-th frame (60 Hz sim -> 30 fps video)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import math  # noqa: E402
import os  # noqa: E402

import gymnasium as gym  # noqa: E402
import imageio.v2 as imageio  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
import isaaclab.utils.math as mu  # noqa: E402
import lekiwi_mimic.tasks  # noqa: E402,F401
import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab.sensors import TiledCameraCfg  # noqa: E402
from isaaclab.utils.datasets import HDF5DatasetFileHandler  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402


def _quat(roll, pitch, yaw):
    q = mu.quat_from_euler_xyz(torch.tensor([roll]), torch.tensor([pitch]), torch.tensor([yaw]))[0]
    return tuple(q.tolist())


def main():
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
    cfg.terminations = None
    cfg.recorders = None
    # chase camera on the moving base (front-left of the robot, forward is +y), next to the robot's own wrist camera
    cfg.scene.chase_cam = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base/chase_cam",
        offset=TiledCameraCfg.OffsetCfg(pos=(0.75, 0.75, 0.55), rot=_quat(0.0, math.radians(28), math.radians(-135)), convention="world"),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=18.0, clipping_range=(0.01, 20.0)),
        width=480,
        height=360,
    )
    env = gym.make(args.task, cfg=cfg).unwrapped
    src = HDF5DatasetFileHandler()
    src.open(args.dataset)
    names = list(src.get_episode_names())
    os.makedirs(args.out, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.dataset))[0]
    with torch.inference_mode():
        for idx in args.episodes:
            ep = src.load_episode(names[idx], env.device)
            env.reset()
            env.reset_to(ep.data["initial_state"], None, is_relative=True)
            frames = []
            for t, action in enumerate(ep.data["actions"]):
                env.step(action.reshape(1, -1))
                if t % 60 == 0 or t == len(ep.data["actions"]) - 1:
                    cube = env.obs_buf["policy"]["cube_pose_base"][0]
                    print(f"  t={t} base={env.base_pose_2d()[0].tolist()} cube_in_base={cube[:3].tolist()}", flush=True)
                if t % args.every == 0:
                    a = env.scene["chase_cam"].data.output["rgb"][0].cpu().numpy()
                    b = env.scene["wrist"].data.output["rgb"][0].cpu().numpy()  # 640x480 robot wrist camera
                    b = b[::4, ::4].repeat(3, 0).repeat(3, 1)[:360, :480] if b.shape[0] != 360 else b
                    frames.append(np.concatenate([a, b], axis=1))
            lifted = env.scene["cube"].data.root_pos_w[0, 2].item() - env.scene.env_origins[0, 2].item()
            print(f"{names[idx]}: recorded success={ep.success} replay cube height={lifted:.3f}", flush=True)
            path = os.path.join(args.out, f"{base}_{names[idx]}.mp4")
            imageio.mimsave(path, frames, fps=30)
            imageio.imwrite(path.replace(".mp4", "_last.png"), frames[-1])
            print("saved", path, len(frames), "frames", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    app.close()
