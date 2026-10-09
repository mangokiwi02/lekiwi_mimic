import gymnasium as gym

_ENV = "lekiwi_mimic.envs.lekiwi_mimic_env:LeKiwiMimicEnv"

for name, entry, cfg in [
    ("LeKiwiMimic-PickCube-Static-v0", "isaaclab.envs:ManagerBasedRLEnv", "LeKiwiPickCubeEnvCfg"),
    ("LeKiwiMimic-PickCube-Far-v0", "isaaclab.envs:ManagerBasedRLEnv", "LeKiwiFarPickCubeEnvCfg"),
]:
    gym.register(
        id=name,
        entry_point=entry,
        disable_env_checker=True,
        kwargs={"env_cfg_entry_point": f"{__name__}.pick_cube_env_cfg:{cfg}"},
    )

for name, cfg in [
    ("LeKiwiMimic-PickCube-Static-Mimic-v0", "LeKiwiPickCubeMimicEnvCfg"),
    ("LeKiwiMimic-PickCube-Far-Mimic-v0", "LeKiwiFarPickCubeMimicEnvCfg"),
]:
    gym.register(
        id=name,
        entry_point=_ENV,
        disable_env_checker=True,
        kwargs={"env_cfg_entry_point": f"{__name__}.pick_cube_mimic_env_cfg:{cfg}"},
    )
