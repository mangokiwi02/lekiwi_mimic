# Related work and how it maps to this repo

Checked against arXiv / project pages in October 2026. Items marked (unverified) could not be confirmed.

## Demo generation from few demos

| Work | Venue | Link | Main idea |
|---|---|---|---|
| MimicGen | CoRL 2023 | arXiv 2310.17596 | Object-centric transform of source segments, linear interpolation between them, keep successful episodes. Adds action noise during generation; the docs say generation without it gives worse policies (appendix numbers not checked). |
| SkillMimicGen | CoRL 2024 | arXiv 2410.18907 | Only contact skills are transformed; transit motion is planned. Policy also learns when a skill starts and ends. |
| DexMimicGen | ICRA 2025 | arXiv 2410.24185 | Bimanual / dexterous version with per-arm asynchronous segments. |
| MoMaGen | ICLR 2026 | arXiv 2510.18316 | Mobile manipulation: base pose sampled around the object under reachability and visibility constraints, navigation planned, success filtering. No recovery data. |
| Isaac Lab Mimic locomanipulation | IsaacLab PR #3992, Dec 2025 | github.com/isaac-sim/IsaacLab/pull/3992 | Base as an end effector, optional navigation P controller. Newer than the Isaac Lab version LeIsaac pins, which is why this repo has its own generator loop. |
| DemoGen | RSS 2025 | arXiv 2502.16932 | One demo, point-cloud editing; skill segments transformed, free-space segments re-planned. Also generates re-approaches when the object moves. |
| CP-Gen | arXiv 2025 (venue unverified) | arXiv 2508.03944 | Keypoint-trajectory constraints to generalize to new object shapes and poses. |
| DockAnywhere | arXiv 2026 | arXiv 2604.15023 | Separates docking-dependent base motion from docking-independent manipulation; samples docking poses and edits observations to stay consistent. |

## Choosing where the base stops

| Work | Venue | Link | Main idea |
|---|---|---|---|
| Mobi-π | CoRL 2025 | arXiv 2505.23692 | Choose base poses that are in-distribution for a given manipulation policy. |
| N2M | ICML 2026 | arXiv 2509.18671 | Learn which start poses the policy succeeds from, from rollouts. |

## Corrective / recovery data

| Work | Venue | Link | Main idea |
|---|---|---|---|
| DART | CoRL 2017 | arXiv 1703.09327 (id not re-checked) | Perturb the expert's execution, keep the expert's clean action as the label. |
| IntervenGen | IROS 2024 | arXiv 2405.01472 | MimicGen-style generation of corrective interventions from a few human interventions. |
| CCIL | ICLR 2024 | arXiv 2310.12972 | Corrective labels near demo states from a learned local dynamics model. |
| Diffusion Meets DAgger | RSS 2024 | arXiv 2402.17768 | Synthesized off-distribution wrist views with corrective labels. |
| PGDG | arXiv 2026 | arXiv 2605.21710 | Find under-covered states and relabel them with corrective actions from sampling-based control. |
| Belkhale et al., Data Quality in Imitation Learning | NeurIPS 2023 | arXiv 2306.02437 | Data quality = consistent actions for similar states + coverage of the states the policy visits. |

## Visual augmentation (for later, image-based policies)

R2R2R (CoRL 2025, arXiv 2505.09601) and RoboSplat (RSS 2025, arXiv 2504.13175) render many demos from Gaussian
splat scenes.

## What this repo takes from them

- Fixed stop pose relative to the cube (`MobileNavCfg.goal_rel_xy`). Copying the stop pose from a random source demo
  gave different labels for the same state, which Belkhale et al. call action divergence.
- Execution-only perturbation of the base with clean labels (DART): OU noise, pushes past the goal, brake pushes far
  from the goal (`MobileNavCfg.exec_noise_*`, `overshoot_*`, `stall_*`). None of the mobile generation papers above
  adds recovery data to the navigation part.
- Logging generation failures by perturbation type to check that filtering does not remove the recovery episodes
  (`scripts/check_dataset.py`).

Not done yet:

- Arm action noise at the MimicGen level (the current 0.002 is much smaller; scales are not directly comparable).
- A DAgger-like loop as in IntervenGen / PGDG: roll out the policy, collect the states where it fails, reset the
  simulator there and generate corrective demos.
- Constrained base pose sampling around the cube (MoMaGen) for more variety in the approach.
