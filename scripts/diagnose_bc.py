"""Offline diagnosis of a BC checkpoint on held-out generated demos (no simulator needed).

1. Open-loop error: predicted vs recorded action on demos the policy did not see, split by phase
   (driving / last 0.5 m / stopped near the cube / manipulation).
2. Speed-distance curve: recorded base speed vs planar cube distance, the open-loop prediction on the same states,
   and (with --trace, written by eval_policy.py --trace) the closed-loop rollout.

    python scripts/diagnose_bc.py --ckpt outputs/policies/big6_far300.pt --data datasets/big6/far_gen.hdf5 \
        --skip 300 --n 100 --trace outputs/policies/big6_far300.trace.npz --out outputs/diag/far300
"""

import argparse
import json
import os
import sys

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "source", "lekiwi_mimic"))
from lekiwi_mimic.policy import BCPolicy, demo_arrays  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--ckpt", required=True)
p.add_argument("--data", required=True)
p.add_argument("--skip", type=int, default=300, help="held-out demos start after the ones used for training")
p.add_argument("--n", type=int, default=100)
p.add_argument("--trace", default=None)
p.add_argument("--out", required=True)
args = p.parse_args()
os.makedirs(args.out, exist_ok=True)

ck = torch.load(args.ckpt, map_location="cpu")
policy = BCPolicy(**ck.get("policy_kwargs", {"history": ck.get("history", 1), "chunk": ck.get("chunk", 1)}))
policy.load_state_dict(ck["state_dict"])
policy.eval()

rows = []  # dist, true speed, pred speed, phase, arm pos err, true vel(2), pred vel(2)
with h5py.File(args.data, "r") as f:
    keys = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[-1]))[args.skip : args.skip + args.n]
    for k in keys:
        obs, act, info = demo_arrays(f["data"][k], policy.history, policy.chunk, policy.obs_cfg)
        with torch.no_grad():
            pred = policy.act_chunk(torch.from_numpy(obs))[:, 0].numpy()
        true = act.reshape(len(act), policy.chunk, -1)[:, 0]
        v_true, v_pred = np.linalg.norm(true[:, 8:10], axis=1), np.linalg.norm(pred[:, 8:10], axis=1)
        moving = np.where(v_true > 1e-4)[0]
        t_stop = moving[-1] + 1 if len(moving) else 0
        dist = info["cube_dist"]
        phase = np.full(len(act), 3)  # manipulation
        phase[:t_stop] = 0  # driving
        phase[:t_stop][dist[:t_stop] < dist[t_stop] + 0.5] = 1  # last 0.5 m
        stop_win = slice(t_stop, min(t_stop + 30, len(act)))
        phase[stop_win] = 2  # first 0.5 s after the stop (should stay still, arm starts)
        arm_err = np.linalg.norm(pred[:, :3] - true[:, :3], axis=1)
        rows.append(np.column_stack([dist, v_true, v_pred, phase, arm_err, true[:, 8:10], pred[:, 8:10]]))
R = np.concatenate(rows)

names = ["driving", "last 0.5 m", "0.5 s after stop", "manipulation"]
report = {}
for i, n in enumerate(names):
    s = R[R[:, 3] == i]
    report[n] = {
        "steps": int(len(s)),
        "speed_true_mean": float(s[:, 1].mean()),
        "speed_pred_mean": float(s[:, 2].mean()),
        "vel_err_rmse": float(np.sqrt(((s[:, 5:7] - s[:, 7:9]) ** 2).sum(1).mean())),
        "arm_pos_err_mean_cm": float(100 * s[:, 4].mean()),
    }
print(json.dumps(report, indent=1))
json.dump(report, open(os.path.join(args.out, "open_loop.json"), "w"), indent=1)

fig, ax = plt.subplots(figsize=(8, 4.5))
drive = R[R[:, 3] < 3]
ax.scatter(drive[:, 0], drive[:, 1], s=1, alpha=0.3, label="data (P controller)")
ax.scatter(drive[:, 0], drive[:, 2], s=1, alpha=0.3, label="policy, open loop on data states")
if args.trace:
    tr = np.load(args.trace)
    d, v = tr["dist"], tr["cmd_speed"]  # (T, N)
    ok = ~np.isnan(d)
    ax.scatter(d[ok], v[ok], s=1, alpha=0.15, c="red", label="policy, closed loop")
ax.set_xlim(0, 1.5)
ax.set_xlabel("planar cube distance from base (m)")
ax.set_ylabel("base speed command (m/s)")
ax.legend(markerscale=8)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(args.out, "speed_vs_dist.png"), dpi=120)
print("saved", os.path.join(args.out, "speed_vs_dist.png"))
