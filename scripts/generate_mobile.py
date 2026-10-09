"""Generate far-pick (navigate + manipulate) demos from static source demos.

    pixi run python scripts/generate_mobile.py --input_file datasets/static_annotated.hdf5 \
        --output_file datasets/far_generated.hdf5 --generation_num_trials 200 --num_envs 8 --headless

With --task ...-Static-Mimic-v0 this is plain Mimic (no navigation), useful as a baseline / PR 1 check.
Per-episode statistics (navigation start/goal/steps, success) are written next to the output as *.stats.jsonl.
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="LeKiwiMimic-PickCube-Far-Mimic-v0")
parser.add_argument("--input_file", required=True)
parser.add_argument("--output_file", required=True)
parser.add_argument("--generation_num_trials", type=int, default=None)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--cameras", action="store_true", help="record wrist/front images (needs --enable_cameras)")
parser.add_argument("--pause_subtask", action="store_true")
parser.add_argument("--exec_noise_std", type=float, default=0.0, help="DART: OU base velocity offset std (m/s), labels stay clean")
parser.add_argument("--goal_rel_xy", type=float, nargs=2, default=None, help="fixed cube (x, y) in the base frame at the nav goal")
parser.add_argument("--rel_xy_noise", type=float, default=None)
parser.add_argument("--exec_noise_near_dist", type=float, default=None)
parser.add_argument("--overshoot_prob", type=float, default=0.0, help="DART: fraction of episodes pushed past the nav goal")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import asyncio  # noqa: E402
import dataclasses  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import random  # noqa: E402
import traceback  # noqa: E402

import gymnasium as gym  # noqa: E402
import isaaclab_mimic.datagen.generation as gen  # noqa: E402
import lekiwi_mimic.tasks  # noqa: E402,F401
import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab_mimic.datagen.datagen_info_pool import DataGenInfoPool  # noqa: E402
from lekiwi_mimic.datagen.mobile_generator import MobileDataGenerator  # noqa: E402


def _jsonable(o):
    return o.tolist() if hasattr(o, "tolist") else str(o)


async def run_generator(env, env_id, reset_q, action_q, generator, success_term, stats_file):
    try:
        while True:
            res = await generator.generate(
                env_id=env_id,
                success_term=success_term,
                env_reset_queue=reset_q,
                env_action_queue=action_q,
                pause_subtask=args.pause_subtask,
            )
            gen.num_success += int(bool(res["success"]))
            gen.num_failures += int(not res["success"])
            gen.num_attempts += 1
            rec = {"env_id": env_id, "success": bool(res["success"]), "steps": int(len(res["actions"]))}
            rec.update(dataclasses.asdict(res["stats"]))
            stats_file.write(json.dumps(rec, default=_jsonable) + "\n")
            stats_file.flush()
    except Exception:
        # an exception inside a generator task would otherwise leave env_loop waiting for actions forever
        traceback.print_exc()
        os._exit(1)


def main():
    out_dir, out_name = os.path.split(os.path.abspath(args.output_file))
    os.makedirs(out_dir, exist_ok=True)
    env_cfg, success_term = gen.setup_env_config(
        env_name=args.task,
        output_dir=out_dir,
        output_file_name=os.path.splitext(out_name)[0],
        num_envs=args.num_envs,
        device=args.device,
        generation_num_trials=args.generation_num_trials,
    )
    if not args.cameras:
        env_cfg.drop_cameras()
    if args.seed is not None:
        env_cfg.datagen_config.seed = args.seed
    env_cfg.nav.exec_noise_std = args.exec_noise_std
    env_cfg.nav.overshoot_prob = args.overshoot_prob
    if args.goal_rel_xy is not None:
        env_cfg.nav.goal_rel_xy = tuple(args.goal_rel_xy)
    if args.rel_xy_noise is not None:
        env_cfg.nav.rel_xy_noise = args.rel_xy_noise
    if args.exec_noise_near_dist is not None:
        env_cfg.nav.exec_noise_near_dist = args.exec_noise_near_dist
    env = gym.make(args.task, cfg=env_cfg).unwrapped

    seed = env.cfg.datagen_config.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    env.reset()

    loop = asyncio.get_event_loop()
    reset_q, action_q = asyncio.Queue(), asyncio.Queue()
    pool = DataGenInfoPool(env, env.cfg, env.device, asyncio_lock=asyncio.Lock())
    pool.load_from_dataset_file(args.input_file)
    print(f"Loaded {pool.num_datagen_infos} source demos")
    generator = MobileDataGenerator(env=env, src_demo_datagen_info_pool=pool)

    stats_path = os.path.splitext(args.output_file)[0] + ".stats.jsonl"
    with open(stats_path, "w") as stats_file:
        tasks = [
            loop.create_task(run_generator(env, i, reset_q, action_q, generator, success_term, stats_file))
            for i in range(args.num_envs)
        ]
        try:
            asyncio.ensure_future(asyncio.gather(*tasks))
            gen.env_loop(env, reset_q, action_q, pool, loop)
        except asyncio.CancelledError:
            pass
    print(f"FINAL {gen.num_success}/{gen.num_attempts} successful; stats -> {stats_path}")


if __name__ == "__main__":
    main()
    app.close()
