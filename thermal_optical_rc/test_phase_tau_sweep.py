"""Regression tests for long time-constant phase-only RC comparisons."""
import unittest

import numpy as np

from phase_only_rc import simulate_phase_rc


TAUS = (10, 50, 100, 500, 1000)


class RecordingOptics:
    """Small deterministic detector used to inspect the input phase map."""

    optical_shape = (2, 2)

    def __init__(self):
        self.phases = []

    def field(self, phase_map):
        phase = np.asarray(phase_map)
        self.phases.append(phase.copy())
        return (phase + 1).astype(complex)


class PhaseTauSweepTests(unittest.TestCase):
    def setUp(self):
        self.profile = np.array([[1.0, 0.5], [0.2, 0.0]])
        self.initial = np.array([[6*np.pi, 3*np.pi], [np.pi, 0.0]])
        self.beta = 2*np.pi

    def run_phase(self, values, tau, initial=None):
        optics = RecordingOptics()
        _, center = simulate_phase_rc(values, optics, self.profile,
            self.initial if initial is None else initial, tau=tau,
            beta=self.beta, mode='memory', camera_shape=(1, 1))
        return np.asarray(optics.phases), center

    def test_zero_input_initial_phase_after_200_writes(self):
        # Index 199 is the image after 200 updates (post-write convention).
        residuals = []
        for tau in TAUS:
            with self.subTest(tau=tau):
                phases, _ = self.run_phase(np.zeros(200), tau)
                expected_fraction = np.exp(-200/tau)
                np.testing.assert_allclose(phases[199], self.initial*expected_fraction,
                                           rtol=2e-13, atol=1e-12)
                residuals.append(phases[199, 0, 0]/self.initial[0, 0])
        self.assertTrue(np.all(np.diff(residuals) > 0))
        self.assertGreater(residuals[-1], 0.8)  # 200 steps is short for tau=1000

    def test_identical_current_write_is_two_pi_independent_of_tau(self):
        zero_initial = np.zeros_like(self.initial)
        for tau in TAUS:
            with self.subTest(tau=tau):
                phases, center = self.run_phase([0, 1, 0], tau, zero_initial)
                np.testing.assert_allclose(phases[0], 0, atol=1e-12)
                np.testing.assert_allclose(phases[1], self.beta*self.profile, atol=1e-12)
                np.testing.assert_allclose(phases[2], self.beta*self.profile*np.exp(-1/tau), atol=1e-12)
                np.testing.assert_allclose(center[1], 2*np.pi, atol=1e-12)

    def test_constant_input_closed_form_does_not_wrap_stored_state(self):
        # For q0=0 and constant u, q_n=beta*u*(1-a^n)/(1-a).
        zero_initial = np.zeros_like(self.initial)
        u, n_steps = 0.5, 30
        for tau in TAUS:
            with self.subTest(tau=tau):
                phases, _ = self.run_phase(np.full(n_steps, u), tau, zero_initial)
                decay = np.exp(-1/tau)
                expected = self.beta*u*(1-decay**n_steps)/(1-decay)
                np.testing.assert_allclose(phases[-1], expected*self.profile,
                                           rtol=2e-13, atol=1e-12)
                self.assertGreater(phases[-1, 0, 0], 2*np.pi)


if __name__ == '__main__':
    unittest.main()
