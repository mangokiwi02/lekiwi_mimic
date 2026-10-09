"""Auto-annotate source demos for Mimic: replay each episode and record datagen_info + subtask term signals.

Headless counterpart of LeIsaac's scripts/mimic/annotate_demos.py --auto, using the lekiwi_mimic env so that
poses are recorded in the moving base frame.

    pixi run python scripts/annotate.py --input_file datasets/static_src.hdf5 --output_file datasets/static_annotated.hdf5
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="LeKiwiMimic-PickCube-Static-Mimic-v0")
parser.add_argument("--input_file", required=True)
parser.add_argument("--output_file", required=True)
parser.add_argument("--cameras", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import os  # noqa: E402

import gymnasium as gym  # noqa: E402
import lekiwi_mimic.tasks  # noqa: E402,F401
import torch  # noqa: E402
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg  # noqa: E402
from isaaclab.managers import RecorderTerm, RecorderTermCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402
from isaaclab.utils.datasets import HDF5DatasetFileHandler  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402


class PreStepDatagenInfoRecorder(RecorderTerm):
    def record_pre_step(self):
        env = self._env
        return "obs/datagen_info", {
            "object_pose": env.get_object_poses(),
            "eef_pose": {name: env.get_robot_eef_pose(eef_name=name) for name in env.cfg.subtask_configs},
            "target_eef_pose": env.action_to_target_eef_pose(env.action_manager.action),
        }


class PreStepSubtaskTermsRecorder(RecorderTerm):
    def record_pre_step(self):
        return "obs/datagen_info/subtask_term_signals", self._env.get_subtask_term_signals()


@configclass
class MimicRecorderManagerCfg(ActionStateRecorderManagerCfg):
    record_pre_step_datagen_info = RecorderTermCfg(class_type=PreStepDatagenInfoRecorder)
    record_pre_step_subtask_term_signals = RecorderTermCfg(class_type=PreStepSubtaskTermsRecorder)


def main():
    src = HDF5DatasetFileHandler()
    src.open(args.input_file)
    out_dir, out_name = os.path.split(os.path.abspath(args.output_file))
    os.makedirs(out_dir, exist_ok=True)

    cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
    if not args.cameras:
        cfg.drop_cameras()
    cfg.env_name = args.task
    success_term = cfg.terminations.success
    cfg.terminations = None
    cfg.recorders = MimicRecorderManagerCfg()
    cfg.recorders.dataset_export_dir_path = out_dir
    cfg.recorders.dataset_filename = os.path.splitext(out_name)[0]
    env = gym.make(args.task, cfg=cfg).unwrapped
    env.reset()

    exported = 0
    names = list(src.get_episode_names())
    with torch.inference_mode():
        for name in names:
            ep = src.load_episode(name, env.device)
            env.seed(int(ep.seed)) if ep.seed is not None else None
            env.sim.reset()
            env.recorder_manager.reset()
            env.reset_to(ep.data["initial_state"], None, is_relative=True)
            for action in ep.data["actions"]:
                env.step(action.reshape(1, -1))
            ok = bool(success_term.func(env, **success_term.params)[0])
            signals = env.recorder_manager.get_episode(0).data["obs"]["datagen_info"]["subtask_term_signals"]
            missing = [k for k, v in signals.items() if not torch.any(torch.stack(v) if isinstance(v, list) else v)]
            if ok and not missing:
                env.recorder_manager.set_success_to_episodes(None, torch.tensor([[True]], device=env.device))
                env.recorder_manager.export_episodes()
                exported += 1
            print(f"{name}: replay_success={ok} missing_signals={missing}", flush=True)
    print(f"DONE annotated {exported}/{len(names)}")
    env.close()


if __name__ == "__main__":
    main()
    app.close()
