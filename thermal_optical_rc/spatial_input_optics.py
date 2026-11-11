"""Fixed-threshold scalar/vector SLM encoding and coherent scattering.

Each SLM pixel has one fixed input-component assignment and one fixed random
threshold. The current input only is encoded: no reservoir-state concatenation
and no camera-to-SLM feedback. Phase memory is handled separately.
"""
import numpy as np
import torch

from phase_only_optics import PhaseOnlyOptics


class ThresholdSLMOptics:
    """One immutable random threshold pattern shared across all time steps."""
    def __init__(self, n_inputs, optical_shape=(128, 128), pupil_radius=(.25, .25),
                 seed=42, encoder_seed=271):
        if not isinstance(n_inputs, (int, np.integer)) or isinstance(n_inputs, (bool, np.bool_)) or n_inputs < 1:
            raise ValueError('n_inputs must be a positive integer')
        self.base = PhaseOnlyOptics(seed=seed, optical_shape=optical_shape,
                                    pupil_radius=pupil_radius)
        self.model = self.base.model
        self.optical_shape = tuple(optical_shape)
        self.n_inputs = int(n_inputs)
        n_pixels = int(np.prod(self.optical_shape))
        if n_inputs > n_pixels:
            raise ValueError('each input needs at least one SLM pixel')
        rng = np.random.default_rng(encoder_seed)
        labels = np.arange(n_pixels) % n_inputs
        rng.shuffle(labels)
        self.component_map = labels.reshape(self.optical_shape)
        self.thresholds = rng.random(self.optical_shape)

    def encode(self, input_vector):
        """Binary B[p] = 1[u[component[p]] > threshold[p]]."""
        u = np.asarray(input_vector, dtype=float)
        if u.shape != (self.n_inputs,) or not np.isfinite(u).all() or np.any((u < 0) | (u > 1)):
            raise ValueError('input_vector must be finite, shape (n_inputs,), and within [0,1]')
        return (u[self.component_map] > self.thresholds).astype(np.float32)

    def field_for_input(self, phase_map, input_vector):
        """O2 B D(phi) A O1 exp(i*pi*B_input), at a fixed input."""
        phase = np.asarray(phase_map, dtype=np.float32)
        if phase.shape != self.optical_shape or not np.isfinite(phase).all():
            raise ValueError('phase_map must be finite and match optical_shape')
        pattern = torch.as_tensor(self.encode(input_vector), device=self.model.device)
        m = self.model
        with torch.no_grad():
            slm_field = torch.exp(1j*torch.pi*pattern)
            field = m._objective(slm_field, m.pupil1)*m.scatter_pre
            field *= torch.exp(1j*torch.as_tensor(phase, device=m.device))
            field = torch.fft.fft2(field, norm='ortho')
            field = m.optical_gain*torch.fft.ifft2(field*m.scatter_fourier, norm='ortho')
            field = m._objective(field, m.pupil2)
            return field.cpu().numpy().copy()


def localized_component_weights(n_inputs, width_fraction=.125):
    """Fixed positive weights representing an assumed local scalar write drive.

    The component index is treated as a periodic spatial coordinate. This is
    a phenomenological input-to-heater coupling, not optical absorption.
    """
    if not isinstance(n_inputs, (int, np.integer)) or n_inputs < 1:
        raise ValueError('n_inputs must be a positive integer')
    if not np.isfinite(width_fraction) or width_fraction <= 0:
        raise ValueError('width_fraction must be finite and positive')
    i = np.arange(n_inputs)
    distance = np.minimum(i, n_inputs-i).astype(float)
    sigma = max(width_fraction*n_inputs, 1e-6)
    weights = np.exp(-.5*(distance/sigma)**2)
    return weights/weights.sum()
