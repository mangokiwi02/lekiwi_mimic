"""Quality checks for a generated far-pick dataset, without training a policy (no simulator needed).

For each dataset prints:
- generation: success / attempts, nav goal reached, success split by overshoot push (selection bias check)
- stop pose: cube position in the base frame when the drive ends (spread = label ambiguity for "stop here")
- labels: max |executed - recorded| base velocity (> 0 only with execution perturbation)
- state coverage near the stop: share of drive steps that are slow but still far, fast and close, or past the goal,
  i.e. the states a closed-loop policy ends up in
- arm during the drive: eef motion while the base moves (the arm should hold)

    python scripts/check_dataset.py datasets/big6/far_gen.hdf5 datasets/dart2/far_gen.hdf5
"""

import argparse
import json
import os

import h5py
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("datasets", nargs="+")
p.add_argument("--n", type=int, default=300, help="demos per dataset (first n)")
p.add_argument("--out", default="outputs/dataset_checks.json")
args = p.parse_args()


def check(path):
    out = {}
    stats_path = os.path.splitext(path)[0] + ".stats.jsonl"
    if os.path.exists(stats_path):
        rows = [json.loads(line) for line in open(stats_path)]
        out["attempts"] = len(rows)
        out["success_rate"] = round(sum(r["success"] for r in rows) / len(rows), 3)
        out["nav_reached"] = round(sum(all(n["reached"] for n in r["nav"]) for r in rows) / len(rows), 3)
        for flag in (True, False):
            sel = [r for r in rows if r["nav"] and r["nav"][0].get("overshoot", False) == flag]
            if sel:
                key = "success_overshoot" if flag else "success_normal"
                out[key] = f"{sum(r['success'] for r in sel)}/{len(sel)}"

    stop, gap, slow_far, fast_close, past, arm_move, drive_steps = [], [], 0, 0, 0, [], 0
    with h5py.File(path, "r") as f:
        keys = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[-1]))[: args.n]
        for k in keys:
            d = f["data"][k]
            a = d["actions"][:]
            o = d["obs"]
            cube = o["cube_pose_base"][:, :2]
            dist = np.linalg.norm(cube, axis=1)
            speed = np.linalg.norm(a[:, 8:10], axis=1)
            moving = np.where(speed > 1e-4)[0]
            if len(moving) == 0:
                continue
            t0, t1 = moving[0], moving[-1] + 1
            stop.append(cube[t1])
            # executed base velocity (joint_vel at t+1) vs recorded command at t, during the drive
            jv = np.linalg.norm(o["joint_vel"][t0 + 1 : t1 + 1, :2] - a[t0:t1, 8:10], axis=1)
            gap.append(jv.max())
            d_stop = dist[t1]
            drv_d = dist[t0:t1] - d_stop  # distance still to go
            meas = np.linalg.norm(o["joint_vel"][t0:t1, :2], axis=1)
            drive_steps += t1 - t0
            slow_far += int(((meas < 0.08) & (drv_d > 0.15)).sum())
            fast_close += int(((meas > 0.15) & (drv_d < 0.08)).sum())
            past += int((drv_d < -0.02).sum())
            ee = o["ee_pose_base"][t0:t1, :3]
            arm_move.append(np.linalg.norm(ee - ee[0], axis=1).max())
    stop = np.array(stop)
    out["demos_checked"] = len(stop)
    out["stop_cube_xy_mean"] = [round(float(v), 3) for v in stop.mean(0)]
    out["stop_cube_xy_std_cm"] = [round(float(v), 2) for v in 100 * stop.std(0)]
    out["exec_minus_label_max"] = round(float(np.median(gap)), 3)
    out["drive_steps"] = int(drive_steps)
    out["share_slow_but_far"] = round(slow_far / drive_steps, 4)
    out["share_fast_and_close"] = round(fast_close / drive_steps, 4)
    out["share_past_goal"] = round(past / drive_steps, 4)
    out["arm_motion_during_drive_cm_p95"] = round(100 * float(np.percentile(arm_move, 95)), 2)
    return out


res = {}
for path in args.datasets:
    res[path] = check(path)
    print(path)
    for k, v in res[path].items():
        print(f"  {k:32s} {v}")
os.makedirs(os.path.dirname(args.out), exist_ok=True)
old = json.load(open(args.out)) if os.path.exists(args.out) else {}
old.update(res)
json.dump(old, open(args.out, "w"), indent=1)
