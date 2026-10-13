import unittest
import torch
from nanogpt_research.optimizers.adamh import AdamH


class AdamHTests(unittest.TestCase):
    def test_radius_and_closure(self):
        p = torch.nn.Parameter(torch.randn(4, 6))
        r = p.norm().detach()
        opt = AdamH([p])

        def closure():
            opt.zero_grad()
            loss = p.square().sum()
            loss.backward()
            return loss

        loss = opt.step(closure)
        self.assertTrue(torch.isfinite(loss))
        torch.testing.assert_close(p.norm(), r)
