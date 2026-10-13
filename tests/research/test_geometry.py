import unittest
from nanogpt_research.geometry.angular import solve_muonh_lr, reconstructed_angle_muonh


class GeometryTests(unittest.TestCase):
    def test_inverse(self):
        for cosine in (-0.8, -0.2, 0.0, 0.4, 0.9):
            for angle in (0.0, 1e-6, 0.01, 0.1):
                result = solve_muonh_lr(angle, cosine)
                self.assertAlmostEqual(
                    reconstructed_angle_muonh(result.applied_lr, cosine),
                    angle,
                    places=12,
                )
        with self.assertRaises(ValueError):
            solve_muonh_lr(0.2, -1.0)
