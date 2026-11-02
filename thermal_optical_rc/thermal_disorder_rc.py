"""Optical reservoir with a local, exponentially relaxing thermal phase mask.

The model is deliberately phenomenological.  It does not try to infer a real
sample's refractive-index distribution.  Instead it implements the controlled
perturbation model

    T_k = B D(theta_k) A,
    D(theta_k) = diag(exp(i * phi_max * theta_k * G)),

where A and B are fixed random propagation operators and G is a spatially
Gaussian heater occupying exactly one quarter of the intermediate plane.
Only a linear ridge readout is trained.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class ThermalParameters:
    tau: float = 8.0
    dt: float = 1.0
    phase_max: float = 1.5
    area_fraction: float = 0.25
    sigma_fraction: float = 0.35
    drive_gain: float = 1.0

    @property
    def decay(self) -> float:
        if self.tau <= 0 or self.dt <= 0:
            raise ValueError("tau and dt must be positive")
        return float(np.exp(-self.dt / self.tau))


class ThermalDisorderRC:
    """Fixed random optical propagation plus one fading thermal state.

    The random propagation is represented implicitly by phase screens and
    orthonormal FFTs, so no dense N^2 x N^2 transmission matrix is allocated.
    The heater is inserted between two fixed mixing stages, exactly matching
    the B D(t) A perturbation model.
    """

    def __init__(
        self,
        optical_shape=(32, 32),
        feature_shape=(16, 16),
        thermal: ThermalParameters = ThermalParameters(),
        ridge_alpha: float = 1e-3,
        seed: int = 42,
        device: Optional[str] = None,
    ):
        self.optical_shape = tuple(map(int, optical_shape))
        self.feature_shape = tuple(map(int, feature_shape))
        self.thermal = thermal
        self.ridge_alpha = float(ridge_alpha)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.generator = torch.Generator(device=self.device).manual_seed(seed)

        self.input_mask = torch.rand(self.optical_shape, generator=self.generator, device=self.device)
        self.phase_a = self._random_unit_phase()
        self.phase_b = self._random_unit_phase()
        self.heater_profile, self.heater_support = self._make_heater()
        self.reset()
        self.readout = None

    def _random_unit_phase(self):
        phase = 2 * torch.pi * torch.rand(
            self.optical_shape, generator=self.generator, device=self.device
        )
        return torch.exp(1j * phase)

    def _make_heater(self):
        """Gaussian profile with an exactly area_fraction-sized support."""
        h, w = self.optical_shape
        n_active = int(round(self.thermal.area_fraction * h * w))
        if not 1 <= n_active <= h * w:
            raise ValueError("area_fraction must select at least one and at most all pixels")
        yy, xx = torch.meshgrid(
            torch.arange(h, device=self.device, dtype=torch.float32),
            torch.arange(w, device=self.device, dtype=torch.float32),
            indexing="ij",
        )
        cy, cx = (h - 1) / 2, (w - 1) / 2
        dist2 = (yy - cy).square() + (xx - cx).square()
        flat_order = torch.argsort(dist2.flatten())
        support_flat = torch.zeros(h * w, dtype=torch.bool, device=self.device)
        support_flat[flat_order[:n_active]] = True
        support = support_flat.reshape(h, w)

        equivalent_radius = np.sqrt(n_active / np.pi)
        sigma = max(self.thermal.sigma_fraction * equivalent_radius, 1e-6)
        gaussian = torch.exp(-dist2 / (2 * sigma**2)) * support
        gaussian = gaussian / gaussian.max().clamp_min(1e-12)
        return gaussian, support

    def reset(self, temperature: float = 0.0):
        self.temperature = torch.tensor(float(temperature), device=self.device)

    def _encode_input(self, u_t):
        u = torch.as_tensor(u_t, dtype=torch.float32, device=self.device).flatten()[0]
        u = u.clamp(0.0, 1.0)
        # Fixed-mask threshold encoding used in the supplied optical-RC code.
        binary = (self.input_mask <= u).float()
        return torch.exp(1j * torch.pi * binary), u

    def _update_temperature(self, drive):
        gamma = self.thermal.decay
        target = self.thermal.drive_gain * drive
        self.temperature = gamma * self.temperature + (1.0 - gamma) * target

    def _propagate(self, field):
        # A: fixed input-to-heater-plane random mixing.
        internal = torch.fft.fft2(field * self.phase_a, norm="ortho")
        # D(theta): pure phase. Outside the heated support it is exactly 1.
        phase = self.thermal.phase_max * self.temperature * self.heater_profile
        internal = internal * torch.exp(1j * phase)
        # B: fixed heater-plane-to-camera random mixing.
        camera_field = torch.fft.ifft2(internal * self.phase_b, norm="ortho")
        return camera_field, phase

    @torch.no_grad()
    def step(self, u_t, return_diagnostics=False):
        field, drive = self._encode_input(u_t)
        self._update_temperature(drive)
        camera_field, phase = self._propagate(field)
        intensity = camera_field.abs().square()
        intensity = intensity / intensity.mean().clamp_min(1e-12)
        features = F.adaptive_avg_pool2d(
            intensity[None, None], self.feature_shape
        )[0, 0].flatten()
        if not return_diagnostics:
            return features
        return features, {
            "temperature": float(self.temperature.cpu()),
            "phase": phase.cpu(),
            "intensity": intensity.cpu(),
            "heater_profile": self.heater_profile.cpu(),
            "active_fraction": float(self.heater_support.float().mean().cpu()),
        }

    @torch.no_grad()
    def transform(self, inputs, reset=True):
        if reset:
            self.reset()
        u = torch.as_tensor(inputs, dtype=torch.float32, device=self.device).flatten()
        states = torch.empty((len(u), np.prod(self.feature_shape)), device=self.device)
        temperatures = torch.empty(len(u), device=self.device)
        for k, value in enumerate(u):
            states[k] = self.step(value)
            temperatures[k] = self.temperature
        return states, temperatures

    def fit(self, inputs, targets, washout=50, reset=True):
        states, _ = self.transform(inputs, reset=reset)
        y = torch.as_tensor(targets, dtype=torch.float32, device=self.device)
        if y.ndim == 1:
            y = y[:, None]
        x = states[washout:]
        y = y[washout:]
        # Bias + current input + optical features, as in the supplied report.
        u = torch.as_tensor(inputs, dtype=torch.float32, device=self.device).flatten()[washout:, None]
        z = torch.cat([torch.ones((len(x), 1), device=self.device), u, x], dim=1)
        eye = torch.eye(z.shape[1], device=self.device)
        eye[0, 0] = 0.0  # do not penalise the bias
        self.readout = torch.linalg.solve(z.T @ z + self.ridge_alpha * eye, z.T @ y)
        return z @ self.readout

    def predict(self, inputs, reset=False):
        if self.readout is None:
            raise RuntimeError("Call fit before predict")
        states, _ = self.transform(inputs, reset=reset)
        u = torch.as_tensor(inputs, dtype=torch.float32, device=self.device).flatten()[:, None]
        z = torch.cat([torch.ones((len(states), 1), device=self.device), u, states], dim=1)
        return z @ self.readout


def pulse_decay_check(model: ThermalDisorderRC, n_steps=40):
    """Return simulated and analytic thermal impulse responses."""
    pulse = torch.zeros(n_steps)
    pulse[0] = 1.0
    _, simulated = model.transform(pulse, reset=True)
    gamma = model.thermal.decay
    analytic = torch.tensor(
        [(1 - gamma) * model.thermal.drive_gain * gamma**k for k in range(n_steps)],
        device=model.device,
    )
    return simulated, analytic


def linear_memory_curve(model: ThermalDisorderRC, n_steps=1800, max_delay=30, washout=100, seed=7):
    """Train independent ridge probes to reconstruct delayed random inputs.

    Returns R^2(delay).  Its sum is a common empirical linear memory-capacity
    diagnostic; a held-out temporal task should still be used for RC claims.
    """
    gen = torch.Generator().manual_seed(seed)
    u = torch.rand(n_steps, generator=gen)
    states, _ = model.transform(u, reset=True)
    z = torch.cat([torch.ones((n_steps, 1), device=model.device), states], dim=1)
    split = int(0.7 * n_steps)
    scores = []
    for delay in range(1, max_delay + 1):
        start = max(washout, delay)
        x_train, y_train = z[start:split], u[start - delay : split - delay, None].to(model.device)
        x_test, y_test = z[split:], u[split - delay : n_steps - delay, None].to(model.device)
        eye = torch.eye(x_train.shape[1], device=model.device)
        weights = torch.linalg.solve(
            x_train.T @ x_train + model.ridge_alpha * eye,
            x_train.T @ y_train,
        )
        pred = x_test @ weights
        ss_res = (y_test - pred).square().sum()
        ss_tot = (y_test - y_test.mean()).square().sum().clamp_min(1e-12)
        scores.append(float((1 - ss_res / ss_tot).cpu()))
    return np.arange(1, max_delay + 1), np.asarray(scores)
