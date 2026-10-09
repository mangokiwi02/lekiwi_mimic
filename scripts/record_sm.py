"""Record scripted source demos (static base) for LeKiwi pick-cube into an Isaac Lab HDF5 dataset.

Stand-in for teleoperation: the same file layout as LeIsaac's teleop recordings (actions + initial_state + obs),
so annotate.py / generate_mobile.py treat it exactly like human demos.

    pixi run python scripts/record_sm.py --num_demos 20 --output_file datasets/static_src.hdf5
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="LeKiwiMimic-PickCube-Static-Mimic-v0")
parser.add_argument("--num_demos", type=int, default=20)
parser.add_argument("--max_attempts", type=int, default=100)
parser.add_argument("--output_file", default="datasets/static_src.hdf5")
parser.add_argument("--noise_std", type=float, default=0.004, help="target jitter so demos are not identical (m)")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--cameras", action="store_true", help="keep wrist/front cameras (needs --enable_cameras)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import os  # noqa: E402

import gymnasium as gym  # noqa: E402
import lekiwi_mimic.tasks  # noqa: E402,F401
import torch  # noqa: E402
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg  # noqa: E402
from isaaclab.managers import DatasetExportMode  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from lekiwi_mimic.datagen.calibration import SM_KWARGS  # noqa: E402
from lekiwi_mimic.datagen.pick_cube_sm import PickCubeStateMachine  # noqa: E402


def main():
    torch.manual_seed(args.seed)
    out_dir, out_name = os.path.split(os.path.abspath(args.output_file))
    os.makedirs(out_dir, exist_ok=True)
    if os.path.exists(args.output_file):
        raise FileExistsError(args.output_file)

    cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
    if not args.cameras:
        cfg.drop_cameras()
    cfg.seed = args.seed
    success_term = cfg.terminations.success
    cfg.terminations = None
    cfg.recorders = ActionStateRecorderManagerCfg()
    cfg.recorders.dataset_export_dir_path = out_dir
    cfg.recorders.dataset_filename = os.path.splitext(out_name)[0]
    cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY
    env = gym.make(args.task, cfg=cfg).unwrapped

    sm = PickCubeStateMachine(noise_std=args.noise_std, device=env.device, **SM_KWARGS)
    ids = [0]
    saved = attempts = 0
    with torch.inference_mode():
        while saved < args.num_demos and attempts < args.max_attempts:
            env.recorder_manager.reset()
            env.reset()
            sm.reset()
            attempts += 1
            success = False
            while not sm.done:
                obs = env.obs_buf["policy"]
                action = sm.get_action(obs["ee_pose_base"][0], obs["cube_pose_base"][0])
                env.step(action)
                success = success or bool(success_term.func(env, **success_term.params)[0])
            env.recorder_manager.record_pre_reset(ids, force_export_or_skip=False)
            env.recorder_manager.set_success_to_episodes(ids, torch.tensor([[success]], device=env.device))
            env.recorder_manager.export_episodes(ids)
            saved += int(success)
            print(f"attempt {attempts}: success={success} saved={saved}/{args.num_demos}", flush=True)

    print(f"DONE saved={saved} attempts={attempts} rate={saved / max(attempts, 1):.2f}")
    env.close()


if __name__ == "__main__":
    main()
    app.close()
