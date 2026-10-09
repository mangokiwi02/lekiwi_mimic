from __future__ import annotations

from isaaclab.envs.mimic_env_cfg import MimicEnvCfg, SubTaskConfig
from isaaclab.utils import configclass

from ...controllers.holonomic_p import HolonomicPControllerCfg
from .pick_cube_env_cfg import LeKiwiFarPickCubeEnvCfg, LeKiwiPickCubeEnvCfg

EEF_NAME = "lekiwi"


@configclass
class MobileNavCfg:
    """Navigation phases the mobile generator inserts before subtasks."""

    navigate_before_subtasks: list[int] = [0]
    """Subtask indices that start with a drive to a pre-manipulation base pose."""
    goal_rel_xy: tuple[float, float] | None = None
    """Fixed object position (x, y) in the base frame at the nav goal. None: copy it from a random source demo, which
    spreads the stopping point by the source spread (~2-3 cm, 9 deg) that a policy cannot observe, so the same state
    is labelled "still driving" in one demo and "stopped" in another."""
    rel_xy_noise: float = 0.01
    """Std (m) of noise added to the object position in the base frame sampled from the source demo."""
    max_nav_steps: int = 1200
    """Abort the trial if the goal is not reached in time (20 s at 60 Hz)."""
    settle_steps: int = 0
    """Zero-velocity steps after arrival before the arm starts. Kept at 0: the goal-reached debounce already ends with
    the base nearly still, and idle steps with an unchanging state are ambiguous for state-based policies."""
    hold_lift: float = 0.06
    """Raise the held eef target by this much (m) while driving, so the gripper clears the object on arrival."""
    hold_lift_steps: int = 30
    """Steps spent raising the arm before the base starts moving."""
    warmup_steps: int = 3
    """Hold steps right after reset so link poses (stale after a joint teleport) are refreshed before planning."""

    # DART-style execution perturbation: the robot executes command + offset, the demo records the clean command, so
    # the data contains the controller's corrections from states the clean P controller never visits.
    exec_noise_std: float = 0.0
    """Std (m/s) of the Ornstein-Uhlenbeck velocity offset; 0 disables the perturbation."""
    exec_noise_yaw_std: float = 0.15
    """Std (rad/s) of the yaw-rate offset (used when ``exec_noise_std`` > 0)."""
    exec_noise_tau: float = 0.5
    """Correlation time (s) of the offset; white noise at 60 Hz would average out and barely move the state."""
    exec_noise_near_dist: float = 1.0
    """Within this distance (m) to the base goal the offset is scaled by ``exec_noise_near_gain``."""
    exec_noise_near_gain: float = 1.5
    exec_noise_settle_radius: float = 0.05
    """No offset within this distance to the goal, so the goal-reached debounce can still trigger."""
    overshoot_prob: float = 0.0
    """Fraction of episodes pushed past the goal: a push toward the goal when the base first gets within
    ``overshoot_dist``, so the data shows braking from too fast and backing up."""
    overshoot_dist: float = 0.5
    overshoot_speed: float = 0.15
    overshoot_time: float = 0.8
    stall_prob: float = 0.0
    """Fraction of episodes braked while still far: a push away from the goal (``stall_speed``) for ``stall_time`` when
    the goal distance drops below a random value in ``stall_dist_range``. Covers "slow but still far" states."""
    stall_dist_range: tuple[float, float] = (0.15, 0.8)
    stall_speed: float = 0.25
    stall_time: float = 0.6


def _pick_cube_subtasks() -> list[SubTaskConfig]:
    return [
        SubTaskConfig(
            object_ref="cube",
            subtask_term_signal="grasp_cube",
            subtask_term_offset_range=(5, 10),
            selection_strategy="nearest_neighbor_object",
            selection_strategy_kwargs={"nn_k": 3},
            action_noise=0.002,
            num_interpolation_steps=20,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
            description="Grasp cube",
            next_subtask_description="Lift cube",
        ),
        SubTaskConfig(
            object_ref="cube",
            subtask_term_signal=None,
            subtask_term_offset_range=(0, 0),
            selection_strategy="random",
            selection_strategy_kwargs={},
            action_noise=0.001,
            num_interpolation_steps=5,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
        ),
    ]


def _configure_datagen(cfg: MimicEnvCfg, name: str) -> None:
    cfg.datagen_config.name = name
    cfg.datagen_config.generation_guarantee = True
    cfg.datagen_config.generation_keep_failed = True
    cfg.datagen_config.generation_num_trials = 10
    cfg.datagen_config.generation_select_src_per_subtask = False
    cfg.datagen_config.generation_transform_first_robot_pose = False
    cfg.datagen_config.generation_interpolate_from_last_target_pose = True
    cfg.datagen_config.max_num_failures = 50
    cfg.datagen_config.seed = 42
    cfg.subtask_configs[EEF_NAME] = _pick_cube_subtasks()


@configclass
class LeKiwiPickCubeMimicEnvCfg(LeKiwiPickCubeEnvCfg, MimicEnvCfg):
    nav_controller: HolonomicPControllerCfg = HolonomicPControllerCfg()
    nav: MobileNavCfg = MobileNavCfg(navigate_before_subtasks=[])
    object_frame_mode: dict[str, str] = {"cube": "bearing"}

    def __post_init__(self):
        super().__post_init__()
        _configure_datagen(self, "lekiwi_pick_cube_static")


@configclass
class LeKiwiFarPickCubeMimicEnvCfg(LeKiwiFarPickCubeEnvCfg, MimicEnvCfg):
    nav_controller: HolonomicPControllerCfg = HolonomicPControllerCfg()
    nav: MobileNavCfg = MobileNavCfg()
    object_frame_mode: dict[str, str] = {"cube": "bearing"}

    def __post_init__(self):
        super().__post_init__()
        _configure_datagen(self, "lekiwi_pick_cube_far")
