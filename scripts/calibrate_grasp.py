"""Find top-down grasp configurations for LeKiwi by batched forward kinematics.

Sweeps shoulder_lift / elbow_flex / wrist_flex (pan = roll = 0) across many envs, one config per env, and keeps
configs whose approach vector (wrist -> jaw grasp point) is close to straight down. Prints the reachable floor range
in front of the base and the gripper-link orientation (base frame) to use as DOWN_QUAT, plus the TCP offset.
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, default=14, help="grid points per joint")
parser.add_argument("--tcp_height", type=float, default=0.0, help="TCP height above the cube centre (m)")
parser.add_argument("--max_err", type=float, default=0.15, help="max angle (rad) between approach axis and -z")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

import itertools  # noqa: E402

import gymnasium as gym  # noqa: E402
import isaaclab.utils.math as mu  # noqa: E402
import lekiwi_mimic.tasks  # noqa: E402,F401
import torch  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

TASK = "LeKiwiMimic-PickCube-Static-Mimic-v0"
CUBE_Z_BASE = 0.015 - 0.035  # cube centre height in the base-link frame


def main():
    n = args.n
    lift = torch.linspace(-1.7, 1.7, n)
    elbow = torch.linspace(-1.7, 1.55, n)
    flex = torch.linspace(-1.6, 1.6, n)
    grid = torch.tensor(list(itertools.product(lift.tolist(), elbow.tolist(), flex.tolist())))
    cfg = parse_env_cfg(TASK, device=args.device, num_envs=len(grid))
    cfg.drop_cameras()
    cfg.terminations = None
    cfg.scene.env_spacing = 2.0
    env = gym.make(TASK, cfg=cfg).unwrapped
    env.reset()
    robot = env.scene["robot"]
    ids = [robot.joint_names.index(j) for j in ("shoulder_lift", "elbow_flex", "wrist_flex")]
    g = robot.joint_names.index("gripper")
    jp = robot.data.default_joint_pos.clone()
    jp[:, ids] = grid.to(env.device)
    jp[:, g] = 0.0  # closed: the grasp point is defined between the closed fingertips
    # write state AND targets (otherwise the PD targets pull the joints back during the step), then read bodies
    # directly instead of the FrameTransformer, whose buffers can lag a teleport by one update
    robot.write_joint_state_to_sim(jp, torch.zeros_like(jp))
    robot.set_joint_position_target(jp)
    robot.write_data_to_sim()
    env.sim.step(render=False)
    env.scene.update(env.physics_dt)
    print("gripper joint after step:", float(robot.data.joint_pos[:, g].abs().max()))

    def body_in_base(name):
        b = robot.find_bodies("base")[0][0]
        i = robot.find_bodies(name)[0][0]
        return mu.subtract_frame_transforms(
            robot.data.body_pos_w[:, b], robot.data.body_quat_w[:, b], robot.data.body_pos_w[:, i], robot.data.body_quat_w[:, i]
        )

    gpos, gquat = body_in_base("gripper")
    jaw_pos, jaw_quat = body_in_base("jaw")
    jaw_cfg = env.cfg.scene.ee_frame.target_frames[1].offset
    tcp, _ = mu.combine_frame_transforms(
        jaw_pos,
        jaw_quat,
        torch.tensor(jaw_cfg.pos, device=env.device).expand_as(jaw_pos),
        torch.tensor(jaw_cfg.rot, device=env.device).expand_as(jaw_quat),
    )
    wrist_b, _ = body_in_base("wrist")
    approach = torch.nn.functional.normalize(tcp - wrist_b, dim=-1)
    down_err = torch.acos((-approach[:, 2]).clamp(-1, 1))
    # world z of each body vs floor: reject configs where the gripper link goes below the floor
    z0 = CUBE_Z_BASE + args.tcp_height
    ok = (down_err < args.max_err) & (gpos[:, 2] > CUBE_Z_BASE) & (tcp[:, 2] < z0 + 0.03) & (tcp[:, 2] > z0 - 0.01)
    print(f"{int(ok.sum())} / {len(grid)} near-vertical configs with TCP at cube height")
    sel = ok.nonzero().squeeze(-1)
    if len(sel) == 0:
        best = down_err.argsort()[:10]
        for i in best.tolist():
            print("closest", grid[i].tolist(), "tcp", tcp[i].tolist(), "err", float(down_err[i]))
    else:
        order = tcp[sel, 1].argsort()
        for i in sel[order].tolist():
            tcp_off = mu.quat_apply(mu.quat_inv(gquat[i : i + 1]), (tcp[i] - gpos[i]).unsqueeze(0))[0]
            a_g = mu.quat_apply(mu.quat_inv(gquat[i : i + 1]), approach[i : i + 1])[0]
            print(
                f"q={[round(v, 2) for v in grid[i].tolist()]} tcp={[round(v, 3) for v in tcp[i].tolist()]} "
                f"err={float(down_err[i]):.3f} gquat={[round(v, 4) for v in gquat[i].tolist()]} "
                f"tcp_off={[round(v, 4) for v in tcp_off.tolist()]} approach_g={[round(v, 4) for v in a_g.tolist()]}"
            )
    env.close()


if __name__ == "__main__":
    main()
    app.close()
