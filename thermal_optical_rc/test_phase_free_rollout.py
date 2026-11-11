"""Causal, continuation, feedback and camera-noise tests for free rollout."""
import unittest

import numpy as np
import torch

from phase_only_rc import free_phase_rollout, simulate_phase_rc


class DummyOptics:
    optical_shape = (2, 2)

    def field(self, phase):
        # Observable intensity = 1 + phase, for easy analytic checks.
        return np.sqrt(1 + np.asarray(phase)).astype(complex)


class PhaseReadout:
    def predict(self, features):
        return np.asarray(features)[:, 0] - 1


class ConstantReadout:
    def __init__(self, value):
        self.value = value

    def predict(self, features):
        return np.full(len(features), self.value)


class FreeRolloutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.optics = DummyOptics()
        self.profile = np.ones((2, 2))
        self.initial = np.zeros((2, 2))

    def simulate(self, inputs, **kwargs):
        return simulate_phase_rc(inputs, self.optics, self.profile,
                                 tau=2, beta=1, camera_shape=(2, 2), **kwargs)

    def rollout(self, history, model, **kwargs):
        return free_phase_rollout(history, model, self.optics, self.profile,
            self.initial, tau=2, beta=1, camera_shape=(2, 2), **kwargs)

    def test_split_continuation_equals_single_pass_with_noise(self):
        inputs = np.array([.2, .4, .1, .8, .7])
        kwargs = dict(mode='memory', photon_scale=50., read_noise_electrons=2.,
                      return_final_phase=True)
        rng_full = np.random.default_rng(89)
        whole, whole_peaks, whole_phase = self.simulate(
            inputs, initial_phase=self.initial, noise_rng=rng_full, **kwargs)
        rng_chunks = np.random.default_rng(89)
        first, first_peaks, state = self.simulate(
            inputs[:2], initial_phase=self.initial, noise_rng=rng_chunks, **kwargs)
        second, second_peaks, final_state = self.simulate(
            inputs[2:], initial_phase=state, noise_rng=rng_chunks, **kwargs)
        np.testing.assert_array_equal(np.vstack([first, second]), whole)
        np.testing.assert_allclose(np.r_[first_peaks, second_peaks], whole_peaks)
        np.testing.assert_allclose(final_state, whole_phase)

    def test_first_prediction_is_from_last_history_post_write(self):
        result = self.rollout([.2, .4], PhaseReadout(), mode='instant', n_steps=4)
        # The state after the last known input .4 predicts the first unseen
        # input; there is no extra unobserved or zero input injected first.
        np.testing.assert_allclose(result['predictions'], [.4]*4, atol=3e-7)
        np.testing.assert_allclose(result['feedback_inputs'], [.4]*3, atol=3e-7)
        np.testing.assert_allclose(result['phase_peak'], [.4]*4, atol=3e-7)

    def test_memory_phase_continues_from_last_training_state(self):
        result = self.rollout([.2, .4], ConstantReadout(.3), mode='memory', n_steps=3)
        a = np.exp(-.5)
        p0 = a*.2+.4
        self.assertAlmostEqual(result['phase_peak'][0], p0)
        self.assertAlmostEqual(result['phase_peak'][1], a*p0+.3)
        self.assertAlmostEqual(result['phase_peak'][2], a*(a*p0+.3)+.3)
        np.testing.assert_allclose(result['feedback_inputs'], [.3, .3])

    def test_raw_predictions_are_retained_and_feedback_clipped(self):
        result = self.rollout([.1, .2], ConstantReadout(1.5), mode='instant', n_steps=4)
        np.testing.assert_allclose(result['predictions'], [1.5]*4)
        np.testing.assert_allclose(result['feedback_inputs'], [1.]*3)
        np.testing.assert_allclose(result['phase_peak'], [.2, 1., 1., 1.])
        self.assertEqual(result['clipping_fraction'], 1.)

    def test_camera_noise_reproducible_and_changes_with_seed(self):
        kwargs = dict(mode='memory', n_steps=10, photon_scale=25., read_noise_electrons=2.)
        a = self.rollout([.2, .4], PhaseReadout(), noise_seed=31, **kwargs)
        b = self.rollout([.2, .4], PhaseReadout(), noise_seed=31, **kwargs)
        c = self.rollout([.2, .4], PhaseReadout(), noise_seed=32, **kwargs)
        np.testing.assert_array_equal(a['predictions'], b['predictions'])
        self.assertFalse(np.array_equal(a['predictions'], c['predictions']))

    def test_invalid_history_and_horizon_rejected(self):
        for history, steps in [([], 2), ([1.1], 2), ([.5], 0), ([.5], True)]:
            with self.subTest(history=history, steps=steps):
                with self.assertRaises(ValueError):
                    self.rollout(history, ConstantReadout(.5), n_steps=steps)
        with self.assertRaises(ValueError):
            self.simulate([.1], initial_phase=self.initial, noise_rng='not a generator')

    def test_one_step_requires_no_feedback(self):
        result = self.rollout([.2], ConstantReadout(.7), n_steps=1)
        self.assertEqual(result['predictions'].shape, (1,))
        self.assertEqual(result['feedback_inputs'].shape, (0,))
        self.assertEqual(result['clipping_fraction'], 0.)


if __name__ == '__main__':
    unittest.main()
