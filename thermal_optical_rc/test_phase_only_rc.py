"""Tests for input-driven phase RC causality and chronological splitting."""
import unittest

import numpy as np

from phase_only_rc import chronological_indices, simulate_phase_rc
from optical_physical_memory import RidgeReadout


class DummyOptics:
    """Deterministic optics whose intensity changes with its phase input."""

    def __init__(self, optical_shape=(4, 4)):
        self.optical_shape = optical_shape
        self.received_phases = []

    def field(self, phase_map):
        phase = np.asarray(phase_map)
        self.received_phases.append(phase.copy())
        return (2.0 + phase).astype(complex)


class PhaseRCCausalityTests(unittest.TestCase):
    def setUp(self):
        self.profile = np.ones((4, 4))
        self.initial = np.full((4, 4), 0.25)
        self.optics = DummyOptics()

    def run_sequence(self, inputs, mode='memory', tau=2.0, beta=0.5):
        return simulate_phase_rc(np.asarray(inputs, dtype=float), self.optics,
            self.profile, self.initial, tau=tau, beta=beta,
            mode=mode, camera_shape=(2, 2))

    def test_postwrite_timing_and_formula(self):
        features, center = self.run_sequence([0.8, 0.0, 0.0])
        a = np.exp(-0.5)
        expected = [a*0.25+0.5*0.8, a*(a*0.25+0.5*0.8), a*a*(a*0.25+0.5*0.8)]
        self.assertEqual(features.shape, (3, 4))
        np.testing.assert_allclose(center, expected, atol=1e-6)
        np.testing.assert_allclose(features[:, 0], (2+np.array(expected))**2, rtol=1e-6)

    def test_only_memory_depends_on_previous_input(self):
        a = [1.0, 0.0, 0.0]
        b = [0.0, 0.0, 0.0]
        for mode in ('memory', 'instant', 'static'):
            ra, _ = self.run_sequence(a, mode)
            rb, _ = self.run_sequence(b, mode)
            if mode == 'memory':
                self.assertGreater(np.linalg.norm(ra[1]-rb[1]), 1e-5)
            else:
                np.testing.assert_allclose(ra[1:], rb[1:], atol=1e-7)

    def test_current_input_affects_memory_and_instant_not_static(self):
        a = [0.0, 1.0, 0.0]
        b = [0.0, 0.0, 0.0]
        for mode in ('memory', 'instant', 'static'):
            ra, _ = self.run_sequence(a, mode)
            rb, _ = self.run_sequence(b, mode)
            np.testing.assert_allclose(ra[0], rb[0], atol=1e-7)
            if mode == 'static':
                np.testing.assert_allclose(ra[1], rb[1], atol=1e-7)
            else:
                self.assertGreater(np.linalg.norm(ra[1]-rb[1]), 1e-5)

    def test_fixed_initial_state_and_optics_reproduce_features(self):
        first, _ = self.run_sequence([0.1, 0.8, 0.4])
        second, _ = self.run_sequence([0.1, 0.8, 0.4])
        np.testing.assert_allclose(first, second, atol=0)

    def test_spatial_profile_limits_increment(self):
        profile = np.zeros((4, 4))
        profile[1, 1] = 1.0
        simulate_phase_rc([1.0], self.optics, profile, self.initial, tau=2,
                          beta=0.5, mode='memory', camera_shape=(2, 2))
        expected = np.exp(-0.5)*self.initial + 0.5*profile
        np.testing.assert_allclose(self.optics.received_phases[0], expected, atol=1e-6)

    def test_camera_noise_is_seeded_and_does_not_change_phase_state(self):
        kwargs = dict(tau=2, beta=0.5, mode='memory', camera_shape=(2, 2),
                      photon_scale=10, read_noise_electrons=1)
        a, phase_a = simulate_phase_rc([0.1, 0.8, 0.4], self.optics, self.profile,
                                     self.initial, noise_seed=10, **kwargs)
        b, phase_b = simulate_phase_rc([0.1, 0.8, 0.4], self.optics, self.profile,
                                     self.initial, noise_seed=10, **kwargs)
        c, phase_c = simulate_phase_rc([0.1, 0.8, 0.4], self.optics, self.profile,
                                     self.initial, noise_seed=11, **kwargs)
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(phase_a, phase_b)
        np.testing.assert_array_equal(phase_a, phase_c)
        self.assertGreater(np.linalg.norm(a-c), 1e-5)


class PhaseRCSplitTests(unittest.TestCase):
    def test_split_lengths_boundaries_and_no_target_leakage(self):
        indices = chronological_indices(train_length=2000, washout=200,
            validation_length=400, test_length=1000, horizon=1)
        fit, valid, refit, test = (indices[k] for k in ('fit', 'valid', 'refit', 'test'))
        self.assertEqual((len(fit), len(valid), len(refit), len(test)), (1399, 399, 1799, 1000))
        self.assertEqual((fit[0], fit[-1]), (200, 1598))
        self.assertEqual((valid[0], valid[-1]), (1600, 1998))
        self.assertEqual((refit[0], refit[-1]), (200, 1998))
        self.assertEqual((test[0], test[-1]), (2000, 2999))
        self.assertLess(fit[-1]+1, valid[0])
        self.assertLess(refit[-1]+1, test[0])
        self.assertEqual(test[-1]+1, 3000)

    def test_readout_standardization_uses_only_fit_samples(self):
        indices = chronological_indices(train_length=2000, washout=200,
            validation_length=400, test_length=1000, horizon=1)
        rng = np.random.default_rng(2)
        features = rng.normal(size=(3000, 4))
        targets = rng.normal(size=3000)
        fit = indices['fit']
        readout = RidgeReadout(1.0).fit(features[fit], targets[fit])
        np.testing.assert_allclose(readout.mean_, features[fit].mean(axis=0))
        np.testing.assert_allclose(readout.scale_, features[fit].std(axis=0))
        # Arbitrary changes to held-out samples do not influence the fitted readout.
        altered = features.copy()
        altered[indices['valid']] += 1000
        altered[indices['test']] -= 1000
        second = RidgeReadout(1.0).fit(altered[fit], targets[fit])
        np.testing.assert_allclose(second.mean_, readout.mean_)
        np.testing.assert_allclose(second.weights_, readout.weights_)


if __name__ == '__main__':
    unittest.main()
