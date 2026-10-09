# LeKiwi Mobile Mimic

Generates "drive to the object and pick it up" demos for LeKiwi from pick demos recorded with the base standing still.

Runs on [LeIsaac](https://github.com/LightwheelAI/leisaac) (Isaac Sim 5.1, Isaac Lab 2.3) and Isaac Lab Mimic. The
robot starts 1-3 m away from a cube, drives to it with a holonomic P controller, then executes a source grasp
transformed by Mimic.

The source demos are currently scripted (a state machine), not teleoperated. All numbers are from simulation.

## Motivation

On a real mobile manipulator, collecting far-pick data means re-recording the arm motion for every place the base can
stop. The drive part of a holonomic base is easy to generate. The arm part can be reused if the demos are expressed
relative to the base and the object, which is what Mimic does for fixed arms. This repo adds what is missing for a
moving base in LeIsaac.

## How a demo is generated

```
reset: robot at a random pose, cube 1-3 m away
nav:   pick a stopping pose relative to the cube, drive there with the P controller (arm held up)
manip: Isaac Lab Mimic subtasks (select a source segment, transform it by the object frame, interpolate, execute)
```

## Problems found and how they were handled

| Problem | Change | Files |
|---|---|---|
| LeKiwi's articulation root is fixed and the base moves through virtual joints `base_x/base_y/base_theta`. Poses and IK commands in the root frame stop matching a static demo once the base has moved. | Mimic poses (eef, objects, targets) and the IK command are in the `base` link frame (`BaseFrameDifferentialIKAction`). | `actions.py`, `envs/lekiwi_mimic_env.py` |
| Mimic's navigation support (isaac-sim/IsaacLab#3992, Dec 2025) is newer than the Isaac Lab version LeIsaac pins (Oct 2025). | Own generator loop that reuses Mimic's segment selection, transform and interpolation. The env has `get_navigation_state()` with the same keys as #3992. | `datagen/mobile_generator.py` |
| The arm has 5 DOF, so a 6-D pose target is over-constrained and plain DLS gives up several cm of position for a few degrees of orientation. | DLS with orientation rows weighted by 0.1, on the gripper link. IK at the TCP was tried and was worse (static generation 21% vs 75%), so the TCP is only used for the grasp check and the scripted demos. | `actions.py` |
| `FrameTransformer` keeps target bodies in a set, so the order of `target_frame_names` changes between runs. | Frames are looked up by name (`actions.frame_index`). | `actions.py`, `mdp.py` |
| The fingers hit the cube while driving up to it. | Higher ready pose, arm raised during the drive and lowered by the Mimic interpolation. | `pick_cube_env_cfg.py`, `mobile_generator.py` |
| Mimic rotates a segment with the cube's yaw, but for this arm the top-down grasp follows the direction of the cube seen from the shoulder. | Object frame mode `"bearing"`: yaw is the direction of the cube from the base. | `envs/lekiwi_mimic_env.py` |
| Link poses are stale for one step after a joint-teleport reset (base z off by 3.5 cm). | A few hold steps before planning; the scripted demo tracks the cube until it descends. | `mobile_generator.py`, `pick_cube_sm.py` |

## Layout

```
source/lekiwi_mimic/lekiwi_mimic/
  frames.py                   planar SE(2) helpers, nav goal from object (unit-tested)
  controllers/holonomic_p.py  P controller with saturation and debounced goal check (unit-tested)
  actions.py                  base-frame IK action term, base velocity term, action config
  envs/lekiwi_mimic_env.py    Mimic env for the moving base
  tasks/pick_cube/            Static (cube in reach) and Far (cube 1-3 m away) tasks
  datagen/                    scripted source demos, calibration constants, mobile generator
  policy.py                   small BC policy used to compare datasets
scripts/
  setup_env.sh run.sh                                            environment and runner
  inspect_robot.py calibrate_grasp.py sweep_sm.py debug_sm.py    calibration / debugging
  record_sm.py annotate.py generate_mobile.py                    data pipeline
  train_bc.py eval_policy.py diagnose_bc.py replay_video.py      policy check and videos
tests/                        unit tests (no simulator needed)
```

## Setup (Windows, RTX 50xx)

```bash
git clone --recursive https://github.com/mangokiwi02/lekiwi_mimic.git
pixi install
bash scripts/setup_env.sh   # Isaac Sim 5.1, Isaac Lab (LeIsaac submodule), LeIsaac, this package
```

Isaac Sim's extension cache has paths of about 200 characters, which go over Windows MAX_PATH under `.pixi/envs/...`
when long paths are disabled. The setup script therefore puts the Isaac stack in a venv at a short path (`C:/lkm`,
change with `LKM_VENV`) made from pixi's Python. The robot asset `lekiwi.usd` comes from
[LightwheelAI/leisaac_env](https://huggingface.co/LightwheelAI/leisaac_env) and goes into
`third_party/leisaac/assets/robots/`.

## Pipeline

```bash
scripts/run.sh scripts/record_sm.py   --headless --num_demos 20 --output_file datasets/static_src.hdf5
scripts/run.sh scripts/annotate.py    --headless --input_file datasets/static_src.hdf5 --output_file datasets/static_annotated.hdf5
scripts/run.sh scripts/generate_mobile.py --headless --num_envs 16 --generation_num_trials 600 \
    --input_file datasets/static_annotated.hdf5 --output_file datasets/far_gen.hdf5
scripts/run.sh scripts/train_bc.py    --data datasets/far_gen.hdf5 --out outputs/bc_far.pt
scripts/run.sh scripts/eval_policy.py --headless --ckpt outputs/bc_far.pt --num_episodes 200
```

## Results

Everything below starts from 20 scripted source demos (cube 18-25 cm in front of the standing robot); 17 passed Mimic
annotation. `scripts/summarize.py` collects the numbers into `outputs/results.md`.

### Data generation

| Task | Success / attempts | Rate | Nav goal reached |
|---|---|---|---|
| Far: random robot pose, cube 1-3 m away in any direction | 600 / 795 | 75% | 795 / 795 |
| Static: cube in reach, same as the source demos | 305 / 368 | 83% | - |

The failures are missed grasps after the drive (the cube slips or a finger clips it).

Generation rate over the versions: far 36, 41, 60, 67, 75%; static 42, 67, 77, 83%. The bearing object frame took
static from 42% to 67%. After that came the reset fix, the raised arm while driving, and the frame lookup by name.

### Policy trained on the generated data

To check the data, a small policy is trained on it: an MLP on the state in the base frame, 2 frames of history and
10-step action chunks with temporal ensembling. It is not meant to be a strong policy.

| Trained on | Eval task | Episodes | Success | Cube within 0.3 m at the end |
|---|---|---|---|---|
| 20 source demos | Static | 128 | 46.1% | - |
| 300 generated static demos | Static | 128 | 58.6% | - |
| 20 source demos | Far | 64 | 0% | 0% |
| 100 generated far demos | Far | 200 | 4.5% | 57% |
| 300 generated far demos | Far | 200 | 3.5% | 80% |
| 600 generated far demos | Far | 200 | 11.5% | 30% |

All policies are evaluated from the same initial states (same seed). A policy trained only on the static demos never
drives. The ones trained on generated data drive up to the cube but mostly fail to stop and grasp. The 95% interval
on 11.5% with 200 episodes is about ±4.5 points, so the 100/300/600 numbers are not a clean trend.

Why it fails to stop (`scripts/diagnose_bc.py`):

- Open loop on held-out demos the policy is accurate: base velocity error is about 1 cm/s, also in the last 0.5 m.
- In closed loop the speed-distance curve leaves the data from 0.4-1.0 m on. The robot then creeps at 1-5 cm/s,
  does not stop, and pushes the cube with the chassis at about 0.12 m.
- The clean P controller never visits these states, so the data has no corrections from them.
- The stopping point is copied from a random source demo, so it varies by about 2-3 cm and 9 degrees between demos.
  The same state is labelled "still driving" in one demo and "stopped" in another.

Changes being tested now:

- DART-style perturbation in generation. The robot executes command + offset, but the demo stores the clean command.
  The offset is an Ornstein-Uhlenbeck process, larger within 1 m of the goal. Some episodes get pushed past the goal.
  See `MobileNavCfg.exec_noise_*` and `actions.PerturbedBaseVelocityAction`.
- A fixed stopping pose relative to the cube (`MobileNavCfg.goal_rel_xy`).
- Base velocity in the policy observation.

## Limitations

- Scripted source demos. Human demos are noisier, so the generation rate will be different.
- LeIsaac's base is driven through virtual joints with no wheel contact, so driving is idealized.
- Open floor, no obstacles; the controller drives straight to the goal.
- One task: picking a 3 cm cube from the floor.
