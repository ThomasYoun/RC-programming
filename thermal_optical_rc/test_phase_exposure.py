"""Tests for intensity integration and exposure-dependent camera noise."""
import unittest
import numpy as np
import torch
from phase_exposure import integrated_speckle, camera_features


class InterferenceOptics:
    optical_shape = (4, 4)
    def field_for_input(self, phase, input_vector):
        return 1 + np.exp(1j*np.asarray(phase))


class ExposureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_time_average_is_intensity_not_average_phase(self):
        optics = InterferenceOptics(); tau = .2; exposure = 1.; n = 80
        phase = np.full((4, 4), np.pi)
        result = integrated_speckle(optics, .5, phase, tau, exposure, substeps=n)
        times = (np.arange(n)+.5)*exposure/n
        values = np.pi*np.exp(-times/tau)
        expected = np.mean(np.abs(1+np.exp(1j*values))**2)
        self.assertAlmostEqual(result[0, 0], expected, places=6)
        wrong = np.abs(1+np.exp(1j*values.mean()))**2
        self.assertGreater(abs(result[0, 0]-wrong), .1)

    def test_zero_phase_and_long_delay(self):
        optics = InterferenceOptics(); phase = np.full((4, 4), np.pi)
        ref = integrated_speckle(optics, .5, np.zeros((4, 4)), .2, 1., substeps=8)
        late = integrated_speckle(optics, .5, phase, .2, .1, delay=5., substeps=8)
        np.testing.assert_allclose(ref, 4.)
        np.testing.assert_allclose(late, ref, atol=1e-6)

    def test_seeded_noise_and_exposure_affects_shot_noise(self):
        image = np.ones((4, 4)); n = 500
        def draws(exposure):
            rng = np.random.default_rng(3)
            return np.array([camera_features(image, exposure, 200, 0, rng, (1, 1))[0] for _ in range(n)])
        short, long = draws(.1), draws(1.)
        np.testing.assert_array_equal(short, draws(.1))
        self.assertGreater(np.var(short), 5*np.var(long))
        self.assertAlmostEqual(short.mean(), 1., delta=.04)
        self.assertAlmostEqual(long.mean(), 1., delta=.02)

    def test_invalid_exposure_and_shape_rejected(self):
        optics=InterferenceOptics()
        for tau, exposure, delay in [(.2, 0, 0), (0, .1, 0), (.2, .1, -.1)]:
            with self.subTest(tau=tau, exposure=exposure, delay=delay):
                with self.assertRaises(ValueError):
                    integrated_speckle(optics, .5, np.zeros((4, 4)), tau, exposure, delay)
        with self.assertRaises(ValueError):
            camera_features(np.ones((4, 4)), 0, 100, 1, np.random.default_rng(1), (2, 2))


if __name__ == '__main__':
    unittest.main()
