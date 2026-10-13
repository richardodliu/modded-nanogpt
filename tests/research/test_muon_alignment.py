import unittest
import torch
from nanogpt_research.optimizers.muon_alignment import (
    LayerwiseAlignedMuon,
    stable_weight_step_angle,
)


class MuonAlignmentTests(unittest.TestCase):
    def test_measured_angle(self):
        p = torch.nn.Parameter(torch.randn(8, 8))
        before = p.detach().clone()
        p.grad = torch.randn_like(p)
        payload = {"parameter_names": ["p"], "target_radians": [[0.01]]}
        opt = LayerwiseAlignedMuon(
            [("p", p)], payload, "muonh_matches_muon", quantization_aware=True
        )
        opt.set_target_step(0)
        opt.step()
        self.assertAlmostEqual(stable_weight_step_angle(before, p), 0.01, places=6)
