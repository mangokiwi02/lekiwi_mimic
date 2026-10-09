"""Run the scripted pick once and print tracking diagnostics every N steps (optionally save viewport frames)."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="LeKiwiMimic-PickCube-Static-Mimic-v0")
parser.add_argument("--every", type=int, default=15)
parser.add_argument("--episodes", type=int, default=1)
parser.add_argument("--frames", default=None, help="directory to save front-camera PNGs (needs --enable_cameras)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import os  # noqa: E402

import gymnasium as gym  # noqa: E402
import isaaclab.utils.math as mu  # noqa: E402
import lekiwi_mimic.tasks  # noqa: E402,F401
import torch  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from lekiwi_mimic.actions import frame_index  # noqa: E402
from lekiwi_mimic.datagen.calibration import SM_KWARGS  # noqa: E402
from lekiwi_mimic.datagen.pick_cube_sm import PickCubeStateMachine  # noqa: E402


def main():
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
    if not args.frames:
        cfg.drop_cameras()
    else:
        import isaaclab.sim as sim_utils
        from isaaclab.sensors import TiledCameraCfg

        # world-fixed side view (looking along -x) and front view (looking along -y) of the grasp area
        for name, pos, rot in [
            ("side_cam", (0.45, 0.22, 0.10), (0.0, 0.0, 0.0, 1.0)),
            ("front_view_cam", (0.0, 0.65, 0.12), (0.7071, 0.0, 0.0, -0.7071)),
        ]:
            setattr(
                cfg.scene,
                name,
                TiledCameraCfg(
                    prim_path="{ENV_REGEX_NS}/" + name,
                    offset=TiledCameraCfg.OffsetCfg(pos=pos, rot=rot, convention="world"),
                    data_types=["rgb"],
                    spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, clipping_range=(0.01, 10.0)),
                    width=480,
                    height=360,
                ),
            )
    success_term = cfg.terminations.success
    cfg.terminations = None
    env = gym.make(args.task, cfg=cfg).unwrapped
    robot = env.scene["robot"]
    g = robot.joint_names.index("gripper")
    sm = PickCubeStateMachine(device=env.device, **SM_KWARGS)
    torch.set_printoptions(precision=3, sci_mode=False)
    with torch.inference_mode():
        for ep in range(args.episodes):
            env.reset()
            sm.reset()
            while not sm.done:
                o = env.obs_buf["policy"]
                phase = sm._phase()[0]
                a = sm.get_action(o["ee_pose_base"][0], o["cube_pose_base"][0])
                env.step(a)
                k = sm.step_count
                if k % args.every == 0 or sm.done:
                    o = env.obs_buf["policy"]
                    ee = o["ee_pose_base"][0]
                    ef = env.scene["ee_frame"]
                    tcp = ef.data.target_pos_source[0, frame_index(ef, "tcp")]
                    pos_err = (a[0, :3] - ee[:3]).norm().item()
                    rot_err = mu.quat_error_magnitude(a[0, 3:7].unsqueeze(0), ee[3:7].unsqueeze(0)).item()
                    sig = env.obs_buf["subtask_terms"]["grasp_cube"][0].item()
                    print(
                        f"ep{ep} t={k:3d} {phase:8s} pos_err={pos_err:.3f} rot_err={rot_err:.3f} "
                        f"tcp={tcp.tolist()} cube={o['cube_pose_base'][0, :3].tolist()} "
                        f"grip={robot.data.joint_pos[0, g].item():.2f} grasp={sig} "
                        f"succ={bool(success_term.func(env, **success_term.params)[0])}",
                        flush=True,
                    )
                if args.frames:
                    os.makedirs(args.frames, exist_ok=True)
                    if k % 5 == 0:
                        import imageio.v2 as imageio

                        for cam in ("wrist",):
                            imageio.imwrite(os.path.join(args.frames, f"{cam}_ep{ep}_{k:04d}.png"), o[cam][0].cpu().numpy())
                        for cam in ("side_cam", "front_view_cam"):
                            img = env.scene[cam].data.output["rgb"][0].cpu().numpy()
                            imageio.imwrite(os.path.join(args.frames, f"{cam}_ep{ep}_{k:04d}.png"), img)
    env.close()


if __name__ == "__main__":
    main()
    app.close()
