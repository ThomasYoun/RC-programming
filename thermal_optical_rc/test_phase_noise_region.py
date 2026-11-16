"""Checks for the reduced, steeper effective phase-screen controls."""
import unittest
import numpy as np

from phase_only_optics import radial_profile
from phase_only_rc import simulate_phase_rc


class FlatOptics:
    optical_shape = (24, 24)

    def field_for_input(self, phase, input_vector):
        return np.exp(1j * phase)


class RegionTests(unittest.TestCase):
    def test_support_fraction_and_steeper_inside_contrast(self):
        shape = (128, 128)
        for fraction in (.4, .2):
            with self.subTest(fraction=fraction):
                original = radial_profile(shape, .12, fraction)
                steep = radial_profile(shape, .04, fraction)
                np.testing.assert_array_equal(original > 0, steep > 0)
                self.assertLessEqual(abs(np.mean(steep > 0) - fraction), 1 / steep.size)
                self.assertAlmostEqual(float(steep.max()), 1.)
                self.assertLess(steep[steep > 0].min(), original[original > 0].min())

    def test_memory_does_not_modify_unaffected_pixels(self):
        profile = radial_profile((24, 24), .04, .2)
        initial = np.random.default_rng(1).uniform(0, 6*np.pi, profile.shape)
        initial *= profile > 0
        u = np.array([.2, .7, .5])
        _, _, final = simulate_phase_rc(
            u, FlatOptics(), profile, initial, tau=5, beta=2*np.pi,
            camera_shape=(4, 4), slm_inputs=u[:, None], return_final_phase=True,
        )
        self.assertTrue(np.all(final[profile == 0] == 0))
        self.assertTrue(np.any(final[profile > 0] != initial[profile > 0]))


if __name__ == '__main__':
    unittest.main()
