import math

import pytest
import torch

from lekiwi_mimic.controllers.holonomic_p import HolonomicPController, HolonomicPControllerCfg
from lekiwi_mimic.frames import (
    base_goal_from_object,
    body_to_world_xy,
    canonicalize_yaw,
    world_to_body_xy,
    wrap_to_pi,
)


def test_wrap_to_pi():
    a = torch.tensor([0.0, math.pi + 0.1, -math.pi - 0.1, 4 * math.pi])
    out = wrap_to_pi(a)
    assert torch.allclose(out, torch.tensor([0.0, -math.pi + 0.1, math.pi - 0.1, 0.0]), atol=1e-5)


def test_body_world_roundtrip():
    g = torch.Generator().manual_seed(0)
    v = torch.randn(64, 2, generator=g)
    yaw = torch.rand(64, generator=g) * 6.28 - 3.14
    assert torch.allclose(world_to_body_xy(body_to_world_xy(v, yaw), yaw), v, atol=1e-5)


def test_base_goal_puts_object_at_relative_position():
    g = torch.Generator().manual_seed(1)
    obj = torch.randn(32, 2, generator=g) * 2
    rel = torch.tensor([[0.0, 0.25]]).repeat(32, 1) + 0.03 * torch.randn(32, 2, generator=g)
    start = torch.randn(32, 2, generator=g) * 2
    goal = base_goal_from_object(obj, rel, start)
    obj_in_base = world_to_body_xy(obj - goal[:, :2], goal[:, 2])
    assert torch.allclose(obj_in_base, rel, atol=1e-5)
    # robot faces the object along the approach direction: goal lies between start and object
    to_obj = obj - start
    to_goal = goal[:, :2] - start
    cos = (to_obj * to_goal).sum(-1) / (to_obj.norm(dim=-1) * to_goal.norm(dim=-1))
    far = to_obj.norm(dim=-1) > 0.5
    assert (cos[far] > 0.95).all()


def test_canonicalize_yaw():
    sym = math.pi / 2
    y = torch.tensor([0.0, 0.7, math.pi / 2 + 0.1, -math.pi + 0.2])
    out = canonicalize_yaw(y, sym)
    assert (out >= -sym / 2).all() and (out < sym / 2).all()
    assert torch.allclose(torch.cos(4 * out), torch.cos(4 * y), atol=1e-5)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_p_controller_converges_with_kinematic_integration(seed):
    torch.manual_seed(seed)
    n, dt = 16, 1 / 30
    cfg = HolonomicPControllerCfg()
    ctrl = HolonomicPController(cfg, n)
    pose = torch.zeros(n, 3)
    goal = torch.cat([torch.rand(n, 2) * 6 - 3, torch.rand(n, 1) * 6.2 - 3.1], -1)
    ctrl.set_goal(goal)
    for _ in range(30 * 40):
        cmd = ctrl.compute(pose)
        assert (cmd[:, :2].norm(dim=-1) <= cfg.max_lin_vel + 1e-6).all()
        assert (cmd[:, 2].abs() <= cfg.max_ang_vel + 1e-6).all()
        pose = pose + cmd * dt
        if ctrl.goal_reached.all():
            break
    assert ctrl.goal_reached.all()
    state = ctrl.navigation_state()
    assert state["is_navigating"].all() and state["navigation_goal_reached"].all()


def test_p_controller_idle_envs_output_zero():
    ctrl = HolonomicPController(HolonomicPControllerCfg(), 4)
    ctrl.set_goal(torch.ones(2, 3), env_ids=torch.tensor([0, 2]))
    cmd = ctrl.compute(torch.zeros(4, 3))
    assert cmd[[1, 3]].abs().sum() == 0 and cmd[[0, 2]].abs().sum() > 0


def test_p_controller_subset_updates_only_selected_envs():
    cfg = HolonomicPControllerCfg(settle_steps=2)
    ctrl = HolonomicPController(cfg, 3)
    ctrl.set_goal(torch.zeros(3, 3))
    ids = torch.tensor([1])
    for _ in range(3):
        cmd = ctrl.compute(torch.zeros(1, 3), ids)
    assert cmd.shape == (1, 3)
    assert ctrl.goal_reached.tolist() == [False, True, False]
