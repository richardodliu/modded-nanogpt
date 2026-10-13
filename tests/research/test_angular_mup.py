import unittest
import torch
from nanogpt_research.geometry.hyperball_angle import fixed_angle_target, angle_between


class AngularMupTests(unittest.TestCase):
    def test_angle_and_radius(self):
        w = torch.randn(8, 8)
        u = torch.randn_like(w)
        target, _ = fixed_angle_target(w, u, w.norm().item(), 0.02)
        torch.testing.assert_close(target.norm(), w.norm())
        self.assertAlmostEqual(angle_between(w, target), 0.02, places=6)
