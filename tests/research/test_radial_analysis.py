import unittest
import torch
from nanogpt_research.geometry.radial import before_geometry, after_geometry, aggregate


class RadialTests(unittest.TestCase):
    def test_actual_stored_update(self):
        p = torch.nn.Parameter(torch.ones(2, 2))
        p.grad = torch.ones_like(p)
        named = [("blocks.0.weight", p)]
        before, geo = before_geometry(named)
        with torch.no_grad():
            p.mul_(0.5)
        rows = after_geometry(named, before, geo)
        total, blocks = aggregate(["blocks.0.weight"], rows)
        self.assertEqual(total, -2.0)
        self.assertEqual(blocks, {"0": -2.0})
