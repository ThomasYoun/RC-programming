"""Regression tests for the phase-only optical verification model."""

import unittest

import numpy as np

from phase_only_optics import PhaseOnlyOptics, phase_trace, radial_profile, wrap_phase


class PhaseModelTests(unittest.TestCase):
    def test_regularized_radial_profile_is_finite_and_normalized(self):
        profile = radial_profile(shape=(33, 33), core_radius=0.12)
        self.assertEqual(profile.shape, (33, 33))
        self.assertTrue(np.isfinite(profile).all())
        self.assertAlmostEqual(float(profile.max()), 1.0)
        self.assertAlmostEqual(float(profile[16, 16]), 1.0)
        self.assertGreater(profile[16, 17], profile[16, 24])
        self.assertGreater(profile[16, 24], profile[0, 0])

    def test_support_fraction_limits_active_pixels(self):
        profile = radial_profile(shape=(32, 32), core_radius=0.12, support_fraction=0.25)
        self.assertEqual(np.count_nonzero(profile), round(32 * 32 * 0.25))
        self.assertAlmostEqual(float(profile.max()), 1.0)
        self.assertEqual(profile[0, 0], 0.0)

    def test_phase_decay_and_increment_timing(self):
        a = np.exp(-1.0 / 4.0)
        phase, next_phase = phase_trace(
            n_steps=4, tau=4.0, dt=1.0, initial_phase=2.0,
            increments=np.array([0.5, 0.0, 0.3, 0.0]),
        )
        np.testing.assert_allclose(phase[0], 2.0)
        np.testing.assert_allclose(next_phase, a * phase + [0.5, 0.0, 0.3, 0.0])
        np.testing.assert_allclose(phase[1:], next_phase[:-1])
        np.testing.assert_allclose(phase[1], 2.0 * a + 0.5)

    def test_phase_state_is_not_wrapped_during_dynamics(self):
        phase, next_phase = phase_trace(
            n_steps=2, tau=8.0, initial_phase=0.0,
            increments=np.array([4 * np.pi, 0.0]),
        )
        self.assertAlmostEqual(phase[1], 4 * np.pi)
        self.assertAlmostEqual(next_phase[1], np.exp(-1.0 / 8) * 4 * np.pi)
        self.assertGreater(next_phase[1], 2 * np.pi)

    def test_wrap_is_periodic_and_within_half_open_interval(self):
        unwrapped = np.array([-2 * np.pi, -0.5, 0.0, 2 * np.pi, 7 * np.pi])
        wrapped = wrap_phase(unwrapped)
        self.assertTrue(np.all((wrapped >= 0) & (wrapped < 2 * np.pi)))
        np.testing.assert_allclose(np.exp(1j * unwrapped), np.exp(1j * wrapped), atol=1e-14)


class PhaseOpticsTests(unittest.TestCase):
    def setUp(self):
        self.shape = (32, 32)
        self.optics = PhaseOnlyOptics(seed=42, optical_shape=self.shape, pupil_radius=(0.7, 0.7))

    def test_fixed_scatterer_reproducible(self):
        phase = radial_profile(shape=self.shape, core_radius=0.12)
        same = PhaseOnlyOptics(seed=42, optical_shape=self.shape, pupil_radius=(0.7, 0.7))
        np.testing.assert_allclose(self.optics.field(phase), self.optics.field(phase), atol=1e-12)
        np.testing.assert_allclose(self.optics.field(phase), same.field(phase), atol=1e-12)

    def test_uniform_phase_cannot_change_intensity(self):
        baseline = self.optics.field(np.zeros(self.shape))
        shifted = self.optics.field(np.full(self.shape, 1.37))
        # FFTs use complex64; compare a scale-independent aggregate error.
        before, after = np.abs(baseline) ** 2, np.abs(shifted) ** 2
        self.assertLess(np.linalg.norm(before - after) / np.linalg.norm(before), 3e-6)

    def test_pixelwise_two_pi_wrapping_preserves_field(self):
        rng = np.random.default_rng(10)
        phase = rng.normal(size=self.shape) * 5
        before, after = self.optics.field(phase), self.optics.field(wrap_phase(phase))
        self.assertLess(np.linalg.norm(before - after) / np.linalg.norm(before), 3e-6)

    def test_nonuniform_phase_changes_speckle(self):
        baseline = np.abs(self.optics.field(np.zeros(self.shape))) ** 2
        phase = 4.0 * radial_profile(shape=self.shape, core_radius=0.12)
        perturbed = np.abs(self.optics.field(phase)) ** 2
        relative_change = np.linalg.norm(perturbed - baseline) / np.linalg.norm(baseline)
        self.assertGreater(relative_change, 1e-3)


if __name__ == '__main__':
    unittest.main()
