"""Inspect the LeKiwi articulation in LeIsaac and calibrate constants used by lekiwi_mimic.

Prints joint/body layout, how the virtual base joints map to the base link pose, the gripper's down-pointing
orientation and the TCP (jaw) offset, and checks IK reach on the floor in front of the base.
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

import math  # noqa: E402

import gymnasium as gym  # noqa: E402
import isaaclab.utils.math as mu  # noqa: E402
import lekiwi_mimic.tasks  # noqa: E402,F401
import torch  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from lekiwi_mimic.actions import frame_index  # noqa: E402

TASK = "LeKiwiMimic-PickCube-Static-Mimic-v0"


def main():
    cfg = parse_env_cfg(TASK, device=args.device, num_envs=1)
    cfg.drop_cameras()
    cfg.terminations = None
    env = gym.make(TASK, cfg=cfg).unwrapped
    env.reset()
    robot = env.scene["robot"]
    torch.set_printoptions(precision=4, sci_mode=False)

    print("joint_names", robot.joint_names)
    print("body_names", robot.body_names)
    print("is_fixed_base", robot.is_fixed_base)
    print("joint limits", robot.data.joint_pos_limits[0])
    print("root pos/quat", robot.data.root_pos_w[0], robot.data.root_quat_w[0])
    print("action dim", env.action_manager.total_action_dim, env.action_manager.active_terms)

    def show(tag):
        bp, bq = env.base_pose_root([0])
        print(f"[{tag}] base link in root: pos {bp[0]}, yaw {math.degrees(mu.euler_xyz_from_quat(bq)[2][0]):.2f}")
        print(f"[{tag}] base_pose_2d", env.base_pose_2d([0])[0])
        print(f"[{tag}] eef in base", env.get_robot_eef_pose("lekiwi", [0])[0])
        print(f"[{tag}] objects in base", {k: v[0][:3, 3] for k, v in env.get_object_poses([0]).items()})

    hold = torch.zeros(1, env.action_manager.total_action_dim, device=env.device)
    eef = env.get_robot_eef_pose("lekiwi", [0])[0]
    hold[0, :3] = eef[:3, 3]
    hold[0, 3:7] = mu.quat_from_matrix(eef[:3, :3])
    hold[0, 7] = 1.0
    show("reset")

    # base joint mapping: teleport and compare
    jid = [robot.joint_names.index(n) for n in ("base_x", "base_y", "base_theta")]
    jp = robot.data.joint_pos.clone()
    jp[0, jid] = torch.tensor([0.5, 0.3, 0.7], device=env.device)
    robot.write_joint_state_to_sim(jp, torch.zeros_like(jp))
    env.sim.step(render=False)
    env.scene.update(env.physics_dt)
    show("teleport x=.5 y=.3 th=.7")

    # base velocity: drive +x in root frame for 1 s
    jp[0, jid] = 0.0
    robot.write_joint_state_to_sim(jp, torch.zeros_like(jp))
    a = hold.clone()
    a[0, 8:11] = torch.tensor([0.2, 0.0, 0.5], device=env.device)
    for _ in range(60):
        env.step(a)
    show("after 1s of vx=0.2, wz=0.5")

    # gripper body vs jaw (TCP) frame
    ee = env.scene["ee_frame"]
    gi, ji = frame_index(ee, "gripper"), frame_index(ee, "jaw")
    print("ee_frame targets:", ee.data.target_frame_names)
    gp, gq = ee.data.target_pos_w[0, gi], ee.data.target_quat_w[0, gi]
    jpw = ee.data.target_pos_w[0, ji]
    print("tcp offset in gripper frame", mu.quat_apply(mu.quat_inv(gq), jpw - gp))
    print("gripper quat in base", env.get_robot_eef_pose("lekiwi", [0])[0][:3, :3])
    for name in ("gripper", "jaw"):
        bid = robot.find_bodies(name)[0][0]
        print(name, "body z-axis in world", mu.matrix_from_quat(robot.data.body_quat_w[0, bid])[:, 2])
    env.close()


if __name__ == "__main__":
    main()
    app.close()
