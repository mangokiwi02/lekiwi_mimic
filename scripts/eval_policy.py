"""Roll out a BC policy in the far-pick task and report success rate by start distance.

    pixi run python scripts/eval_policy.py --ckpt outputs/bc_far.pt --num_episodes 200 --num_envs 32 --headless
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", required=True)
parser.add_argument("--task", default="LeKiwiMimic-PickCube-Far-Mimic-v0")
parser.add_argument("--num_episodes", type=int, default=200)
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--max_steps", type=int, default=2400, help="40 s at 60 Hz")
parser.add_argument("--static", action="store_true", help="evaluate on the static (cube in reach) task instead")
parser.add_argument("--seed", type=int, default=1234)
parser.add_argument("--out", default=None)
parser.add_argument("--debug", action="store_true", help="print env 0 every 30 steps")
parser.add_argument("--trace", action="store_true", help="save per-step cube distance / speed command of the first batch")
parser.add_argument("--phase_dist", type=float, default=0.23, help="privileged phase flag turns on below this cube distance")
parser.add_argument("--ensemble_decay", type=float, default=0.1, help="temporal ensembling weight exp(-decay*age)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import json  # noqa: E402
import os  # noqa: E402

import gymnasium as gym  # noqa: E402
import lekiwi_mimic.tasks  # noqa: E402,F401
import torch  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from lekiwi_mimic.actions import frame_index  # noqa: E402
from lekiwi_mimic.policy import BCPolicy, RolloutHelper, base_vel_body, body_to_root_base_vel, make_obs  # noqa: E402


def main():
    if args.static:
        args.task = "LeKiwiMimic-PickCube-Static-Mimic-v0"
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    cfg.drop_cameras()
    cfg.seed = args.seed
    success_term = cfg.terminations.success
    cfg.terminations = None
    cfg.episode_length_s = 1e6
    # unique recorder file so evals can run next to generation/other evals (default path is shared and locked)
    cfg.recorders.dataset_filename = f"eval_{os.getpid()}"
    env = gym.make(args.task, cfg=cfg).unwrapped
    robot = env.scene["robot"]
    g_idx = robot.joint_names.index("gripper")
    base_idx = [robot.joint_names.index(n) for n in ("base_x", "base_y", "base_theta")]

    ck = torch.load(args.ckpt, map_location=env.device)
    kw = ck.get("policy_kwargs", {"history": ck.get("history", 1), "chunk": ck.get("chunk", 1)})
    policy = BCPolicy(**kw).to(env.device)
    policy.load_state_dict(ck["state_dict"])
    policy.eval()
    helper = RolloutHelper(policy, env.num_envs, env.device, ensemble_decay=args.ensemble_decay)

    results = []
    trace = {"dist": [], "cmd_speed": [], "speed": []}
    with torch.inference_mode():
        while len(results) < args.num_episodes:
            env.reset()
            helper.reset()
            # start distance between base and cube (planar)
            cube_b = env.obs_buf["policy"]["cube_pose_base"]
            dist0 = cube_b[:, :2].norm(dim=-1)
            done = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
            t_done = torch.full((env.num_envs,), -1, device=env.device)
            phase = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
            for t in range(args.max_steps):
                o = env.obs_buf["policy"]
                yaw = env.base_pose_2d()[:, 2]
                vel = base_vel_body(robot.data.joint_vel[:, base_idx], yaw) if policy.obs_cfg["base_vel"] else None
                ph = None
                if policy.obs_cfg.get("phase"):
                    phase |= o["cube_pose_base"][:, :2].norm(dim=-1) <= args.phase_dist
                    ph = phase.float()
                obs = make_obs(o["cube_pose_base"], o["ee_pose_base"], robot.data.joint_pos[:, g_idx], vel, ph)
                a = helper(obs, o["ee_pose_base"][:, :3])
                if args.trace and not results:
                    # done episodes are masked so the plot only shows the approach of unfinished ones
                    d = o["cube_pose_base"][:, :2].norm(dim=-1).masked_fill(done, float("nan"))
                    trace["dist"].append(d.cpu())
                    trace["cmd_speed"].append(a[:, 8:10].norm(dim=-1).cpu())
                    trace["speed"].append(robot.data.joint_vel[:, base_idx[:2]].norm(dim=-1).cpu())
                a = body_to_root_base_vel(a, yaw)
                if args.debug and t % 30 == 0:
                    ef = env.scene["ee_frame"]
                    tcp = ef.data.target_pos_source[0, frame_index(ef, "tcp")]
                    print(
                        f"t={t} ee={[round(v, 3) for v in o['ee_pose_base'][0, :3].tolist()]} "
                        f"tgt={[round(v, 3) for v in a[0, :3].tolist()]} grip_cmd={a[0, 7].item():+.0f} "
                        f"grip={robot.data.joint_pos[0, g_idx].item():.2f} tcp={[round(v, 3) for v in tcp.tolist()]} "
                        f"cube={[round(v, 3) for v in o['cube_pose_base'][0, :3].tolist()]} vel={[round(v, 2) for v in a[0, 8:].tolist()]}",
                        flush=True,
                    )
                env.step(a)
                s = success_term.func(env, **success_term.params)
                newly = s & ~done
                t_done[newly] = t
                done |= s
                if done.all():
                    break
            cube_end = env.obs_buf["policy"]["cube_pose_base"][:, :3]
            for i in range(env.num_envs):
                results.append(
                    {
                        "dist": float(dist0[i]),
                        "success": bool(done[i]),
                        "steps": int(t_done[i]),
                        "cube_end_base": [round(v, 3) for v in cube_end[i].tolist()],
                    }
                )
            print("final cube in base (first 6 envs):", [r["cube_end_base"] for r in results[-env.num_envs :][:6]])
            rate = sum(r["success"] for r in results) / len(results)
            print(f"{len(results)} episodes, success {rate:.3f}", flush=True)

    results = results[: args.num_episodes]
    bins = [(0.0, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 2.5), (2.5, 3.5)]
    summary = {"ckpt": args.ckpt, "task": args.task, "n": len(results)}
    summary["success_rate"] = sum(r["success"] for r in results) / len(results)
    summary["by_distance"] = {}
    for lo, hi in bins:
        sel = [r for r in results if lo <= r["dist"] < hi]
        if sel:
            summary["by_distance"][f"{lo}-{hi}"] = {"n": len(sel), "success": sum(r["success"] for r in sel) / len(sel)}
    print(json.dumps(summary, indent=1))
    out = args.out or os.path.splitext(args.ckpt)[0] + ".eval.json"
    if args.trace:
        import numpy as np

        np.savez(os.path.splitext(out)[0] + ".trace.npz", **{k: torch.stack(v).numpy() for k, v in trace.items()})
    with open(out, "w") as f:
        json.dump({"summary": summary, "episodes": results}, f)
    env.close()


if __name__ == "__main__":
    main()
    app.close()
