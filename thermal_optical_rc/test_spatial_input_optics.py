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
        expected_row = np.repeat(np.arange(4), 8)
        np.testing.assert_array_equal(self.o.component_map, np.tile(expected_row, (32, 1)))

    def test_column_layout_on_128_uses_two_columns_per_component(self):
        optics = ThresholdSLMOptics(64, optical_shape=(128, 128), encoder_seed=4)
        expected_row = np.repeat(np.arange(64), 2)
        np.testing.assert_array_equal(optics.component_map, np.tile(expected_row, (128, 1)))
        np.testing.assert_array_equal(np.bincount(optics.component_map.ravel()), np.full(64, 256))

    def test_ks_64_input_columns_expand_to_256_in_four_by_four_blocks(self):
        optics = ThresholdSLMOptics(64, optical_shape=(256, 256), encoding_shape=(64, 64),
                                    block_size=4, encoder_seed=4)
        expected_base = np.tile(np.arange(64), (64, 1))
        np.testing.assert_array_equal(optics.base_component_map, expected_base)
        np.testing.assert_array_equal(optics.component_map,
            np.repeat(np.repeat(expected_base, 4, axis=0), 4, axis=1))
        self.assertEqual(optics.base_thresholds.shape, (64, 64))
        np.testing.assert_array_equal(optics.thresholds,
            np.repeat(np.repeat(optics.base_thresholds, 4, axis=0), 4, axis=1))
        u = np.linspace(.01, .99, 64)
        base = optics.encode_base(u)
        slm = optics.encode(u)
        self.assertEqual(base.shape, (64, 64))
        self.assertEqual(slm.shape, (256, 256))
        np.testing.assert_array_equal(slm, np.repeat(np.repeat(base, 4, axis=0), 4, axis=1))
        np.testing.assert_array_equal(np.bincount(optics.component_map.ravel()), np.full(64, 1024))
        changed = u.copy(); changed[17] = 1.0
        changed_pixels = optics.encode(changed) != slm
        self.assertTrue(changed_pixels.any())
        self.assertTrue(np.all(optics.component_map[changed_pixels] == 17))

    def test_block_configuration_validation(self):
        with self.assertRaises(ValueError):
            ThresholdSLMOptics(64, optical_shape=(256, 256), encoding_shape=(64, 64), block_size=3)
        with self.assertRaises(ValueError):
            ThresholdSLMOptics(64, optical_shape=(255, 256), block_size=4)
        with self.assertRaises(ValueError):
            ThresholdSLMOptics(64, optical_shape=(256, 256), encoding_shape=(64, 64), block_size=0)

    def test_uneven_column_counts_and_optional_legacy_shuffle(self):
        column = ThresholdSLMOptics(5, optical_shape=(32, 32), pupil_radius=(.7, .7), encoder_seed=4)
        shuffled = ThresholdSLMOptics(5, optical_shape=(32, 32), pupil_radius=(.7, .7), encoder_seed=4, layout='shuffled')
        np.testing.assert_array_equal(np.unique(column.component_map[0], return_counts=True)[1], [7, 7, 6, 6, 6])
        np.testing.assert_array_equal(column.thresholds, shuffled.thresholds)
        self.assertFalse(np.array_equal(column.component_map, shuffled.component_map))
        rng = np.random.default_rng(4)
        labels = np.arange(32*32) % 5
        rng.shuffle(labels)
        np.testing.assert_array_equal(shuffled.component_map, labels.reshape(32, 32))
        np.testing.assert_array_equal(shuffled.thresholds, rng.random((32, 32)))
        with self.assertRaises(ValueError):
            ThresholdSLMOptics(33, optical_shape=(32, 32), layout='columns')
        with self.assertRaises(ValueError):
            ThresholdSLMOptics(5, optical_shape=(32, 32), layout='unknown')

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
        np.testing.assert_array_equal(np.unique(np.pi*pattern), np.array([0., np.pi], dtype=np.float32))
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
        legacy = ThresholdSLMOptics(1, optical_shape=(32, 32), pupil_radius=(.7, .7), layout='shuffled')
        np.testing.assert_array_equal(o.component_map, legacy.component_map)
        np.testing.assert_array_equal(o.thresholds, legacy.thresholds)
        np.testing.assert_array_equal(o.encode(np.array([.4])), legacy.encode(np.array([.4])))
        np.testing.assert_array_equal(o.field_for_input(np.zeros((32, 32)), np.array([.4])),
                                      legacy.field_for_input(np.zeros((32, 32)), np.array([.4])))
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
