import unittest
import torch
from nanogpt_research.normalization import (
    Derf,
    TaperNorm,
    replace_normalization,
    prepare_step,
    finish_step,
    sites,
)
from nanogpt_research.model import GPT


class NormalizationTests(unittest.TestCase):
    def test_reference_gradients(self):
        for module in (Derf(8), TaperNorm(8)):
            x = torch.randn(2, 3, 8, requires_grad=True)
            module(x).square().mean().backward()
            self.assertTrue(
                all(
                    p.grad is not None and torch.isfinite(p.grad).all()
                    for p in module.parameters()
                )
            )

    def test_calibration_and_model_integration(self):
        model = replace_normalization(GPT(64, 1, 128), case="taper_g0p5")
        for step in (1, 2):
            prepare_step(model, step, warmup=2, case="taper_g0p5")
            loss, _ = model(torch.randint(64, (1, 4)), torch.randint(64, (1, 4)))
            loss.backward()
            finish_step(model)
        prepare_step(model, 3, warmup=2, case="taper_g0p5")
        self.assertTrue(
            all(
                bool(m.calibrated) and m.gate == 0.5
                for _, m in sites(model)
                if not m.final
            )
        )
