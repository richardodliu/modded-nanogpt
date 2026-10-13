import unittest
import torch
from nanogpt_research.optimizers.sgd_alignment import NormHSGD
from nanogpt_research.geometry.sgd_angular import stored_angle


class SGDTests(unittest.TestCase):
    def test_momentum_matches_pytorch(self):
        for momentum, nesterov in ((0.0, False), (0.9, False), (0.9, True)):
            p = torch.nn.Parameter(torch.randn(4, 6))
            q = torch.nn.Parameter(p.detach().clone())
            actual = NormHSGD([("p", p)], lr=0.03, momentum=momentum, nesterov=nesterov)
            expected = torch.optim.SGD(
                [q], lr=0.03, momentum=momentum, nesterov=nesterov
            )
            for _ in range(4):
                p.grad = torch.randn_like(p)
                q.grad = p.grad.clone()
                actual.step()
                expected.step()
                torch.testing.assert_close(p, q, rtol=0, atol=0)

    def test_measured_angle(self):
        for family in ("sgd", "sgdh"):
            p = torch.nn.Parameter(torch.randn(8, 8))
            p.grad = torch.randn_like(p)
            before = p.detach().clone()
            opt = NormHSGD([("p", p)], family, aligned=True)
            opt.targets = {"p": 0.01}
            opt.step()
            self.assertAlmostEqual(stored_angle(before, p), 0.01, places=6)
            if family == "sgdh":
                torch.testing.assert_close(p.norm(), before.norm())
