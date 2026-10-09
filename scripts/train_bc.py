"""Train the egocentric BC policy on one or more HDF5 datasets (no simulator needed).

    pixi run python scripts/train_bc.py --data datasets/far_generated.hdf5 --out outputs/bc_far.pt
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "source", "lekiwi_mimic"))
from lekiwi_mimic.policy import BCPolicy, load_dataset  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--data", nargs="+", required=True)
p.add_argument("--max_demos", type=int, default=None, help="per dataset")
p.add_argument("--out", required=True)
p.add_argument("--gripper_joint_index", type=int, default=8, help="LeKiwi joint order puts base_x/y/theta first")
p.add_argument("--epochs", type=int, default=200)
p.add_argument("--batch", type=int, default=1024)
p.add_argument("--lr", type=float, default=3e-4)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--history", type=int, default=2)
p.add_argument("--chunk", type=int, default=10)
p.add_argument("--base_vel", action="store_true", help="add measured base velocity to the observation")
p.add_argument("--phase", action="store_true", help="privileged drive/grasp phase flag (upper bound only)")
p.add_argument("--vel_weight", type=float, default=1.0, help="loss weight of the base velocity dims")
args = p.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
obs_l, act_l, n_demos = [], [], 0
for path in args.data:
    o, a, n = load_dataset(path, args.gripper_joint_index, args.max_demos, args.history, args.chunk, {"base_vel": args.base_vel, "phase": args.phase})
    obs_l.append(o)
    act_l.append(a)
    n_demos += n
obs, act = np.concatenate(obs_l), np.concatenate(act_l)
print(f"{n_demos} demos, {len(obs)} samples")

dev = "cuda" if torch.cuda.is_available() else "cpu"
model = BCPolicy(history=args.history, chunk=args.chunk, base_vel=args.base_vel, phase=args.phase).to(dev)
model.set_normalization(obs, act)
O, A = torch.from_numpy(obs).to(dev), torch.from_numpy(act).to(dev)
A_n = (A - model.act_mean) / model.act_std
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
steps_per_epoch = max(1, len(O) // args.batch)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs * steps_per_epoch)
w = torch.ones(args.chunk, 11, device=dev)
w[:, 7] = 2.0  # gripper timing matters most
w[:, 8:] = args.vel_weight
w = w.reshape(-1)
for ep in range(args.epochs):
    perm = torch.randperm(len(O), device=dev)
    tot = 0.0
    for i in range(steps_per_epoch):
        idx = perm[i * args.batch : (i + 1) * args.batch]
        loss = (((model.forward_normalized(O[idx]) - A_n[idx]) ** 2) * w).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        tot += loss.item()
    if ep % 20 == 0 or ep == args.epochs - 1:
        print(f"epoch {ep} loss {tot / steps_per_epoch:.4f}", flush=True)

os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
torch.save(
    {
        "state_dict": model.state_dict(),
        "policy_kwargs": model.kwargs,
        "history": args.history,
        "chunk": args.chunk,
        "data": args.data,
        "n_demos": n_demos,
        "n_samples": len(obs),
    },
    args.out,
)
with open(os.path.splitext(args.out)[0] + ".json", "w") as f:
    json.dump({"data": args.data, "n_demos": n_demos, "n_samples": len(obs), "final_loss": tot / steps_per_epoch}, f)
print("saved", args.out)
