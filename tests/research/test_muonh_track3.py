import unittest
import torch
from nanogpt_research.optimizers.track3 import MuonH, NorMuonH
from nanogpt_research.schedules.track3 import power_decay_lr


class Track3Tests(unittest.TestCase):
    def test_hyperball_radius_and_optimizer_state(self):
        for cls in (MuonH, NorMuonH):
            p = torch.nn.Parameter(torch.randn(8, 12))
            radius = p.norm().detach().clone()
            opt = cls([p], lr=0.01)
            for _ in range(5):
                p.grad = torch.randn_like(p)
                opt.step()
            torch.testing.assert_close(p.norm(), radius)
            self.assertTrue(opt.state[p])

    def test_power_schedule(self):
        for step in range(3301):
            peak = 0.014
            floor = 1e-5
            if step < 100:
                expected = floor + (peak - floor) * step / 99
            elif step <= 199:
                expected = peak
            elif step <= 3200:
                expected = floor + (peak - floor) * (
                    1 - ((step - 199) / (3200 - 199)) ** 0.4
                )
            else:
                expected = floor
            self.assertEqual(power_decay_lr(step, peak), expected)
