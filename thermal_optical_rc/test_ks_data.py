"""Numerical and interface checks for the KS benchmark data generator."""
import unittest

import numpy as np

from ks_data import generate_ks


class KSDataTests(unittest.TestCase):
    def test_shape_seed_reproducibility_and_nontrivial_finite_solution(self):
        a = generate_ks(n_samples=150, warmup_steps=50, seed=17)
        b = generate_ks(n_samples=150, warmup_steps=50, seed=17)
        c = generate_ks(n_samples=150, warmup_steps=50, seed=18)
        self.assertEqual(a.shape, (150, 64))
        self.assertTrue(np.isfinite(a).all())
        np.testing.assert_array_equal(a, b)
        self.assertGreater(np.linalg.norm(a-c), 1e-5)
        self.assertGreater(np.std(a), 0.1)
        self.assertGreater(np.linalg.norm(a[-1]-a[0]), 0.1)

    def test_constant_field_stationary_and_mean_conserved(self):
        constant = np.full(64, 0.7)
        fields = generate_ks(n_samples=40, warmup_steps=10, initial_state=constant)
        np.testing.assert_allclose(fields, 0.7, atol=1e-12)
        x = np.arange(64)*2*np.pi/64
        initial = 0.7+np.sin(x)+0.3*np.cos(2*x)
        fields = generate_ks(n_samples=100, warmup_steps=20, initial_state=initial)
        np.testing.assert_allclose(fields.mean(axis=1), initial.mean(), atol=1e-12)

    def test_periodic_grid_translation_equivariance(self):
        x = np.arange(64)*2*np.pi/64
        initial = np.sin(x)+0.3*np.cos(3*x)
        reference = generate_ks(n_samples=40, warmup_steps=10, initial_state=initial)
        shifted = generate_ks(n_samples=40, warmup_steps=10, initial_state=np.roll(initial, 7))
        np.testing.assert_allclose(shifted, np.roll(reference, 7, axis=1), atol=1e-11)

    def test_warmup_and_sampling_index(self):
        sample = generate_ks(n_samples=2, warmup_steps=5, sample_every=3)
        same_first = generate_ks(n_samples=1, warmup_steps=5)
        same_second = generate_ks(n_samples=1, warmup_steps=8)
        np.testing.assert_allclose(sample[0], same_first[0], atol=1e-14)
        np.testing.assert_allclose(sample[1], same_second[0], atol=1e-14)

    def test_short_horizon_refines_with_smaller_dt(self):
        # Same initial condition and sample times; chaotic long-term trajectories
        # should not be expected to converge pointwise indefinitely.
        a = generate_ks(n_samples=40, warmup_steps=0, dt=0.5)
        b = generate_ks(n_samples=40, warmup_steps=0, dt=0.25, sample_every=2)
        c = generate_ks(n_samples=40, warmup_steps=0, dt=0.125, sample_every=4)
        self.assertLess(np.linalg.norm(b-c), np.linalg.norm(a-c))
        self.assertLess(np.max(np.abs(b-c)), 0.002)

    def test_small_perturbations_can_grow(self):
        # Finite-time sensitivity is a useful sanity check, not a numerical
        # estimate of a Lyapunov exponent or a proof of asymptotic chaos.
        original = generate_ks(n_samples=1000, warmup_steps=0)
        perturbation = 1e-6*np.cos(2*np.pi*np.arange(64)/64)
        perturbed = generate_ks(n_samples=1000, warmup_steps=0,
                                initial_state=original[0]+perturbation)
        self.assertGreater(np.linalg.norm(original[-1]-perturbed[-1]),
                           100*np.linalg.norm(original[0]-perturbed[0]))

    def test_rejects_invalid_parameters(self):
        for kwargs in ({'n_samples': 0}, {'n_grid': 15}, {'dt': 0},
                       {'warmup_steps': -1}, {'sample_every': 0},
                       {'initial_state': [float('nan')]*64}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                generate_ks(**kwargs)


if __name__ == '__main__':
    unittest.main()
