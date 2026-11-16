"""Finite-exposure camera diagnostic for a decaying phase-screen pulse.

This helper is independent of the discrete RC update. It computes an
intensity average under an explicitly assumed impulsive phase write followed
by exponential decay; it is not a calibrated thermal or camera model.
"""
import numpy as np
import torch
import torch.nn.functional as F


def integrated_speckle(optics, probe_input, phase_at_pulse, tau,
                       exposure, delay=0., substeps=32):
    """Return average *intensity* during an exposure after one phase pulse.

    Times, including tau, exposure and delay, use the same arbitrary units.
    During exposure the SLM probe remains fixed and the phase map follows
    phase(t)=phase_at_pulse*exp(-t/tau). The exposure starts delay units after
    the pulse. No further writes or background phase are assumed.
    """
    phase = np.asarray(phase_at_pulse, dtype=float)
    if phase.shape != optics.optical_shape or not np.isfinite(phase).all():
        raise ValueError('phase_at_pulse must be finite and match optical_shape')
    if not np.isfinite(tau) or tau <= 0 or not np.isfinite(exposure) or exposure <= 0:
        raise ValueError('tau and exposure must be finite and positive')
    if not np.isfinite(delay) or delay < 0:
        raise ValueError('delay must be finite and non-negative')
    if (not isinstance(substeps, (int, np.integer)) or
            isinstance(substeps, (bool, np.bool_)) or substeps < 1):
        raise ValueError('substeps must be a positive integer')
    if not np.isfinite(probe_input) or not 0 <= probe_input <= 1:
        raise ValueError('probe_input must be in [0,1]')
    result = np.zeros(optics.optical_shape, dtype=np.float64)
    for j in range(substeps):
        t = delay+(j+.5)*exposure/substeps
        field = optics.field_for_input(phase*np.exp(-t/tau), np.array([probe_input]))
        result += np.abs(field)**2
    return result/substeps


def camera_features(intensity, exposure, photon_rate, read_noise_electrons,
                    rng, camera_shape=(16, 16)):
    """Poisson photons + per-frame read noise, then intensity downsampling.

    photon_rate is expected photoelectrons per unit normalized intensity per
    unit exposure. A shorter exposure collects proportionally fewer photons;
    read noise is not scaled down. Per-pixel negative values are clipped as in
    the existing RC simulation. No dark current or sensor saturation is modeled.
    """
    intensity = np.asarray(intensity, dtype=float)
    if intensity.ndim != 2 or not np.isfinite(intensity).all() or np.any(intensity < 0):
        raise ValueError('intensity must be a finite nonnegative image')
    if not np.isfinite(exposure) or exposure <= 0 or not np.isfinite(photon_rate) or photon_rate <= 0:
        raise ValueError('exposure and photon_rate must be finite and positive')
    if not np.isfinite(read_noise_electrons) or read_noise_electrons < 0:
        raise ValueError('read_noise_electrons must be finite and non-negative')
    if not isinstance(rng, np.random.Generator):
        raise ValueError('rng must be a numpy Generator')
    if len(camera_shape) != 2 or any(not isinstance(n, (int, np.integer)) or n < 1 for n in camera_shape):
        raise ValueError('camera_shape must contain two positive integers')
    if any(c > p for c, p in zip(camera_shape, intensity.shape)):
        raise ValueError('camera_shape cannot exceed intensity shape')
    scale = exposure*photon_rate
    electrons = rng.poisson(intensity*scale).astype(float)
    if read_noise_electrons:
        electrons += rng.normal(0, read_noise_electrons, intensity.shape)
    measured = np.maximum(electrons, 0)/scale
    image = torch.as_tensor(measured, dtype=torch.float32)[None, None]
    return F.adaptive_avg_pool2d(image, camera_shape).numpy().ravel()
