"""Direct phase-screen dynamics with a fixed, coherent optical propagation.

This is an effective model: phase is specified directly, not inferred from
temperature or a calibrated refractive-index distribution.
"""
import numpy as np
import torch

from optical_physical_memory import PhysicalMemoryOptics, PhysicalNoise


def radial_profile(shape=(128, 128), core_radius=0.12, support_fraction=None):
    """Normalized regularized inverse-radius phase profile in a 2-D plane.

    Coordinates run from -1 to 1; core_radius is in those normalized units.
    If requested, the nearest support_fraction of pixels form a circular-ish
    support, truncated at the edges of the square optical plane.
    """
    if len(shape) != 2 or any(not isinstance(n, (int, np.integer)) or n < 2 for n in shape):
        raise ValueError('shape must contain two integer dimensions >= 2')
    if not np.isfinite(core_radius) or core_radius <= 0:
        raise ValueError('core_radius must be finite and positive')
    if support_fraction is not None and not 0 < support_fraction <= 1:
        raise ValueError('support_fraction must be in (0, 1] or None')
    y = np.linspace(-1, 1, shape[0])
    x = np.linspace(-1, 1, shape[1])
    radius2 = y[:, None] ** 2 + x[None, :] ** 2
    profile = core_radius / np.sqrt(radius2 + core_radius ** 2)
    profile /= profile.max()
    if support_fraction is not None:
        count = max(1, round(np.prod(shape) * support_fraction))
        support = np.zeros(profile.size, dtype=bool)
        support[np.argsort(radius2.ravel(), kind='stable')[:count]] = True
        profile *= support.reshape(shape)
    return profile


def phase_trace(n_steps, tau, dt=1.0, initial_phase=0.0, increments=None):
    """Return pre-exposure phase amplitudes and their next states, in rad.

    q[k+1] = exp(-dt/tau)*q[k] + increments[k]. Neither state is wrapped.
    increments[k] is added after observation k, and first appears at k+1.
    """
    if not isinstance(n_steps, (int, np.integer)) or n_steps < 1:
        raise ValueError('n_steps must be a positive integer')
    if not np.isfinite(tau) or not np.isfinite(dt) or tau <= 0 or dt <= 0:
        raise ValueError('tau and dt must be finite and positive')
    if not np.isfinite(initial_phase):
        raise ValueError('initial_phase must be finite')
    increments = np.zeros(n_steps) if increments is None else np.asarray(increments, dtype=float)
    if increments.shape != (n_steps,) or not np.all(np.isfinite(increments)):
        raise ValueError('increments must contain n_steps finite values')
    phase, next_phase = np.empty(n_steps), np.empty(n_steps)
    q, a = float(initial_phase), np.exp(-dt/tau)
    for k in range(n_steps):
        phase[k] = q
        q = a*q + increments[k]
        next_phase[k] = q
    return phase, next_phase


def wrap_phase(phase):
    """Wrap for presentation; never use the wrapped value in state decay."""
    return np.mod(phase, 2*np.pi)


class PhaseOnlyOptics:
    """Fixed input and scattering realization with a direct phase screen.

    The propagation reuses the previous model's objective pupils, pre-screen
    disorder and post-screen coherent mixing. No thermal state or noise is
    advanced here. Input and all random optical operators remain fixed.
    """
    def __init__(self, seed=42, optical_shape=(128, 128), pupil_radius=(0.25, 0.25)):
        ideal = PhysicalNoise(pupil_rms_rad=0, slm_phase_std_rad=0,
                              photon_scale=1, read_noise_electrons=0,
                              shot_noise=False)
        self.model = PhysicalMemoryOptics(optical_shape=optical_shape,
            slm_shape=optical_shape, pupil_radius=pupil_radius, seed=seed,
            noise=ideal)
        self.optical_shape = tuple(optical_shape)
        with torch.no_grad():
            _, _, self.slm_field = self.model._encode_input(0.5)

    def field(self, phase_map):
        phase = np.asarray(phase_map, dtype=np.float32)
        if phase.shape != self.optical_shape or not np.isfinite(phase).all():
            raise ValueError('phase_map must be finite and match optical_shape')
        m = self.model
        with torch.no_grad():
            field = m._objective(self.slm_field, m.pupil1)*m.scatter_pre
            field = field*torch.exp(1j*torch.as_tensor(phase, device=m.device))
            field = torch.fft.fft2(field, norm='ortho')
            field = m.optical_gain*torch.fft.ifft2(field*m.scatter_fourier, norm='ortho')
            field = m._objective(field, m.pupil2)
            return field.cpu().numpy().copy()


def intensity_correlation(reference, current):
    """Pearson correlation of raw camera intensities across pixels."""
    a, b = np.ravel(reference), np.ravel(current)
    if a.shape != b.shape or a.size < 2:
        raise ValueError('images must contain the same number of pixels >= 2')
    a, b = a-a.mean(), b-b.mean()
    denominator = np.linalg.norm(a)*np.linalg.norm(b)
    return float(np.dot(a, b)/denominator) if denominator else float('nan')
