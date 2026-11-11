"""Tests for the fixed scalar/vector SLM encoding and phase coupling."""
import unittest
import numpy as np
import torch

from spatial_input_optics import ThresholdSLMOptics, localized_component_weights
from phase_only_rc import simulate_phase_rc, free_phase_rollout


class ConstantModel:
    def predict(self, x):
        return np.full(len(x), .5)


class ThresholdEncodingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.o = ThresholdSLMOptics(4, optical_shape=(32, 32), pupil_radius=(.7, .7), seed=42, encoder_seed=4)

    def test_fixed_balanced_component_mask_and_reproducibility(self):
        same = ThresholdSLMOptics(4, optical_shape=(32, 32), pupil_radius=(.7, .7), seed=42, encoder_seed=4)
        np.testing.assert_array_equal(self.o.component_map, same.component_map)
        np.testing.assert_array_equal(self.o.thresholds, same.thresholds)
        np.testing.assert_array_equal(np.bincount(self.o.component_map.ravel()), [256]*4)
        self.assertTrue(np.all((self.o.thresholds >= 0) & (self.o.thresholds < 1)))

    def test_each_component_controls_its_own_threshold_pixels(self):
        u = np.full(4, .3)
        pattern1 = self.o.encode(u)
        u[2] = .8
        pattern2 = self.o.encode(u)
        changed = pattern1 != pattern2
        self.assertTrue(changed.any())
        self.assertTrue(np.all(self.o.component_map[changed] == 2))
        self.assertTrue(np.all(pattern2 >= pattern1))
        np.testing.assert_array_equal(self.o.encode(np.zeros(4)), np.zeros((32, 32)))
        np.testing.assert_array_equal(self.o.encode(np.ones(4)), np.ones((32, 32)))

    def test_pure_phase_slm_and_complex_field(self):
        pattern = self.o.encode(np.full(4, .5))
        np.testing.assert_allclose(np.abs(np.exp(1j*np.pi*pattern)), 1)
        phase = np.linspace(0, 2, 32**2).reshape(32, 32)
        f = self.o.field_for_input(phase, np.array([.2, .3, .4, .5]))
        g = self.o.field_for_input(phase, np.array([.2, .3, .4, .5]))
        np.testing.assert_array_equal(f, g)
        self.assertGreater(np.std(np.abs(f)**2), 0)

    def test_vector_input_drives_slm_and_scalar_drives_memory(self):
        phase0 = np.zeros((32, 32)); profile = np.ones((32, 32))
        vectors = np.array([[.2, .3, .4, .5], [.8, .3, .4, .5]])
        drive = np.array([.3, .7])
        features, peaks = simulate_phase_rc(drive, self.o, profile, phase0,
            tau=2, beta=2, mode='memory', camera_shape=(4, 4), slm_inputs=vectors)
        np.testing.assert_allclose(peaks, [0.6, np.exp(-.5)*.6+1.4])
        self.assertGreater(np.linalg.norm(features[0]-features[1]), 1e-8)
        with self.assertRaises(ValueError):
            simulate_phase_rc(drive, self.o, profile, phase0, tau=2, beta=2,
                camera_shape=(4, 4), slm_inputs=np.array([[.1, .2, .3, .4]]))

    def test_scalar_mg_and_free_prediction_use_same_slm_encoder(self):
        o = ThresholdSLMOptics(1, optical_shape=(32, 32), pupil_radius=(.7, .7))
        result = free_phase_rollout([.2, .4], ConstantModel(), o, np.ones((32, 32)),
            np.zeros((32, 32)), tau=2, beta=1, camera_shape=(4, 4),
            mode='memory', n_steps=3, encode_slm=True)
        np.testing.assert_allclose(result['predictions'], [.5, .5, .5])
        np.testing.assert_allclose(result['feedback_inputs'], [.5, .5])

    def test_local_weights_positive_normalized_and_sensitive_to_field(self):
        w = localized_component_weights(64)
        self.assertAlmostEqual(w.sum(), 1.)
        self.assertTrue(np.all(w > 0))
        self.assertGreater(w[0], w[32])
        x = np.zeros(64); x[0], x[32] = .5, -.5
        self.assertAlmostEqual(x.mean(), 0)
        self.assertNotAlmostEqual(float(w@x), 0)


if __name__ == '__main__':
    unittest.main()
