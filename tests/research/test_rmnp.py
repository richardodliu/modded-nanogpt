import unittest
import torch
from nanogpt_research.optimizers.rmnp import RMNP


class RMNPTests(unittest.TestCase):
    def test_update_matches_direct_formula(self):
        p = torch.nn.Parameter(torch.randn(8, 4))
        expected = p.detach().clone()
        m = torch.zeros_like(p)
        opt = RMNP([p], lr=0.02, momentum=0.9, weight_decay=0.1, eps=1e-12)
        for _ in range(3):
            g = torch.randn_like(p)
            p.grad = g
            m.mul_(0.9).add_(g)
            u = torch.nn.functional.normalize(g + 0.9 * m, dim=-1) * (2**0.5)
            expected.mul_(1 - 0.02 * 0.1).add_(u, alpha=-0.02)
            opt.step()
            torch.testing.assert_close(p, expected)
