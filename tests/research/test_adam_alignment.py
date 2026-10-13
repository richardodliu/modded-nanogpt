import unittest
import torch
from nanogpt_research.optimizers.adam_alignment import MatrixAdam, angle


class AdamAlignmentTests(unittest.TestCase):
    def test_measured_angle(self):
        for family in ("adamw", "adamh"):
            p = torch.nn.Parameter(torch.randn(8, 8))
            before = p.detach().clone()
            p.grad = torch.randn_like(p)
            payload = {"parameter_names": ["p"], "target_radians": [[0.01]]}
            opt = MatrixAdam(
                [("p", p)], family + "_matches_adamh", lr=0.01, target_payload=payload
            )
            opt.set_target_step(0)
            opt.step()
            self.assertAlmostEqual(angle(before, p), 0.01, places=6)
