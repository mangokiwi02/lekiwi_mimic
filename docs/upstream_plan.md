# Upstream plan (LeIsaac)

Notes for contributing parts of this repo to LeIsaac. Not submitted yet. Each PR is one change, following LeIsaac's
CONTRIBUTING.md; larger features would be discussed in an issue first.

Version facts (checked 2026-10-09):
- LeIsaac `main` pins Isaac Lab at `3c6e67b` (2025-10-28), Isaac Sim 5.1.
- Isaac Lab Mimic's loco-manipulation APIs (`get_navigation_state`, body end effector, nav P-controller hooks) landed in
  isaac-sim/IsaacLab#3992 on 2025-12-01, i.e. **not** in LeIsaac's pin. Everything below works on the pinned version.

## PR 0 — bug fix: LeKiwi LeRobot export reads the wrong joints (small, send first)

`source/leisaac/leisaac/tasks/template/lekiwi_env_cfg.py`, `build_lerobot_frame`:

```python
convert_leisaac_action_to_lerobot(obs_data["joint_pos"][-1][:-3] ...)   # assumes base joints are last
```

`LEKIWI_CFG`'s articulation order (printed by `scripts/inspect_robot.py`) is

```
['base_x', 'base_y', 'base_theta', 'shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper']
```

so `[:-3]` selects `base_x, base_y, base_theta, shoulder_pan, shoulder_lift, elbow_flex` and
`observation.state` in exported LeKiwi datasets would be wrong. Fix: select the arm joints by name
(`robot.joint_names.index(...)`) or slice `[3:]`. Also worth checking `action[:, :6]` for every LeKiwi device mapping.
To confirm with the maintainers before sending (they may use a different USD revision).

## PR 1 — LeKiwi static pick + Mimic env

Scope: everything needed to run the existing Mimic pipeline (record → annotate → generate) on LeKiwi with the base
standing still.

- `devices/action_process.py`: a Mimic/state-machine device for LeKiwi (IK arm + binary gripper + zero base velocity).
- Base-frame IK action term (`BaseFrameDifferentialIKAction`): LeKiwi's root is fixed and the base moves through
  virtual joints, so commands must be in the `base` link frame. Uses orientation-weighted DLS, which the 5-DOF SO-101
  arm needs (plain DLS loses centimetres of position for a few degrees of orientation). IK runs on the `gripper` link;
  an IK body offset to the TCP was tried and made generation worse (static 21% vs ~75%, far 0%), so the TCP frame is only used
  for the grasp signal and the scripted demos.
- `LeKiwi-PickCube` task (static) + Mimic env cfg; `cube_grasped` looks the gripper joint up by name.
- Docs page + manual test plan: record 20 scripted demos → annotate (17/20 replay) → generate.

Files here: `actions.py`, `envs/lekiwi_mimic_env.py` (without navigation), `tasks/pick_cube/*` (Static),
`datagen/pick_cube_sm.py`, `datagen/calibration.py`.

## PR 2 — navigation state + holonomic P controller

- `controllers/holonomic_p.py`, `frames.py` (+ unit tests, no simulator needed).
- `get_navigation_state()` on the LeKiwi Mimic env with the #3992 keys (`is_navigating`, `navigation_goal_reached`).
- Object frame mode `"bearing"` for top-down grasps with a 5-DOF arm.

## PR 3 — far-pick task, mobile generator, docs

- `LeKiwi-PickCube-Far` task (random base pose, cube 1–3 m away).
- `datagen/mobile_generator.py` + `scripts/generate_mobile.py`.
- Docs: pipeline, numbers, limitations. When LeIsaac moves to an Isaac Lab with #3992, the generator loop can be
  replaced by Mimic's own navigation support; the env API is already shaped for it.

## Isaac Lab issue to report (not LeIsaac)

`FrameTransformer` (pinned `3c6e67b`) collects target bodies in a `set`, so `data.target_frame_names` order differs
between processes (seen: `['gripper', 'jaw', 'tcp']` in one run, `['tcp', 'gripper', 'jaw']` in the next). Code that
indexes `target_pos_w[:, 0]` assuming config order silently reads the wrong frame. Here every lookup goes through
`actions.frame_index(ee_frame, name)`. LeIsaac's own `ee_frame` users should be checked for index-based access.
Before filing: check whether newer Isaac Lab already preserves config order.

## Not upstream

- Omni-wheel contact fidelity (LeIsaac's base is kinematic virtual joints) — separate repo if pursued.
- BC policy / evaluation scripts — portfolio material, not library code.
