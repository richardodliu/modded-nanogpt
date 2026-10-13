import copy
import unittest
import torch
from nanogpt_research.optimizers.schedulefree_plus import AdamCScheduleFreePlusPaper
from nanogpt_research.schedules.adaptive import (
    StabilityConfig,
    GradientStabilityController,
)


class ScheduleFreeTests(unittest.TestCase):
    def test_step_mode_and_resume(self):
        p = torch.nn.Parameter(torch.randn(4, 4))
        opt = AdamCScheduleFreePlusPaper([p])
        opt.train()
        for _ in range(3):
            opt.zero_grad()
            loss = p.square().mean()
            loss.backward()
            opt.step_func(loss.item())
        opt.eval()
        self.assertTrue(torch.isfinite(p).all())
        opt.train()
        state = copy.deepcopy(opt.state_dict())
        q = torch.nn.Parameter(p.detach().clone())
        other = AdamCScheduleFreePlusPaper([q])
        other.load_state_dict(state)
        for parameter, optimizer in ((p, opt), (q, other)):
            optimizer.zero_grad()
            loss = parameter.square().mean()
            loss.backward()
            optimizer.step_func(loss.item())
        torch.testing.assert_close(p, q, rtol=0, atol=0)

    def test_controller_resume(self):
        config = StabilityConfig(
            horizon=30, window=2, check_interval=1, min_step=4, persistence_checks=2
        )
        controller = GradientStabilityController(config)
        for step in range(6):
            controller.observe(step, 1.0)
        restored = GradientStabilityController(config)
        restored.load_state_dict(controller.state_dict())
        for step in range(6, 30):
            self.assertEqual(controller.lr_factor(step), restored.lr_factor(step))
