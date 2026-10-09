"""Sweep scripted-grasp parameters and report lift success per setting (tuning aid for the source-demo script)."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="LeKiwiMimic-PickCube-Static-Mimic-v0")
parser.add_argument("--episodes", type=int, default=6)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import itertools  # noqa: E402

import gymnasium as gym  # noqa: E402
import lekiwi_mimic.tasks  # noqa: E402,F401
import torch  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from lekiwi_mimic.datagen.calibration import SM_KWARGS  # noqa: E402
from lekiwi_mimic.datagen.pick_cube_sm import PickCubeStateMachine  # noqa: E402

GRID = {
    "grasp_height": [-0.015, -0.01],
    "forward_offset": [-0.012, -0.006],
}


def main():
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
    cfg.drop_cameras()
    success_term = cfg.terminations.success
    cfg.terminations = None
    cfg.seed = 7
    env = gym.make(args.task, cfg=cfg).unwrapped
    results = {}
    with torch.inference_mode():
        for values in itertools.product(*GRID.values()):
            kw = dict(zip(GRID.keys(), values))
            sm = PickCubeStateMachine(device=env.device, **{**SM_KWARGS, **kw})
            ok = 0
            for _ in range(args.episodes):
                env.reset()
                sm.reset()
                succ = False
                while not sm.done:
                    o = env.obs_buf["policy"]
                    env.step(sm.get_action(o["ee_pose_base"][0], o["cube_pose_base"][0]))
                    succ = succ or bool(success_term.func(env, **success_term.params)[0])
                ok += succ
            results[str(kw)] = ok / args.episodes
            print(f"SWEEP {kw} success={ok}/{args.episodes}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    app.close()
