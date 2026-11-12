"""Fixed-threshold scalar/vector SLM encoding and coherent scattering.

Each input-plane pixel has one fixed input-component assignment and one fixed
random threshold. By default vector components occupy ordered columns. An
optional integer block expansion copies each encoded pixel onto the SLM. The
current input only is encoded: no reservoir-state concatenation or
camera-to-SLM feedback.
"""
import numpy as np
import torch

from phase_only_optics import PhaseOnlyOptics


class ThresholdSLMOptics:
    """Fixed thresholds, ordered input-plane columns, optional block expansion."""
    def __init__(self, n_inputs, optical_shape=(128, 128), pupil_radius=(.25, .25),
                 seed=42, encoder_seed=271, layout='columns',
                 encoding_shape=None, block_size=1):
        if not isinstance(n_inputs, (int, np.integer)) or isinstance(n_inputs, (bool, np.bool_)) or n_inputs < 1:
            raise ValueError('n_inputs must be a positive integer')
        if layout not in ('columns', 'shuffled'):
            raise ValueError("layout must be 'columns' or 'shuffled'")
        if (not isinstance(block_size, (int, np.integer)) or
                isinstance(block_size, (bool, np.bool_)) or block_size < 1):
            raise ValueError('block_size must be a positive integer')
        self.base = PhaseOnlyOptics(seed=seed, optical_shape=optical_shape,
                                    pupil_radius=pupil_radius)
        self.model = self.base.model
        self.optical_shape = tuple(optical_shape)
        self.n_inputs = int(n_inputs)
        self.block_size = int(block_size)
        if encoding_shape is None:
            if any(n % self.block_size for n in self.optical_shape):
                raise ValueError('optical_shape must be divisible by block_size')
            encoding_shape = tuple(n//self.block_size for n in self.optical_shape)
        if (len(encoding_shape) != 2 or
                any(not isinstance(n, (int, np.integer)) or n < 1 for n in encoding_shape)):
            raise ValueError('encoding_shape must contain two positive integers')
        self.encoding_shape = tuple(int(n) for n in encoding_shape)
        if tuple(n*self.block_size for n in self.encoding_shape) != self.optical_shape:
            raise ValueError('encoding_shape * block_size must equal optical_shape')
        n_pixels = int(np.prod(self.encoding_shape))
        if n_inputs > n_pixels:
            raise ValueError('each input needs at least one input-plane pixel')
        if layout == 'columns' and n_inputs > self.encoding_shape[1]:
            raise ValueError('column layout needs at least one input-plane column per input')
        rng = np.random.default_rng(encoder_seed)
        # Generate the legacy shuffled map first in either mode. This preserves
        # the original fixed thresholds for a matched layout-only comparison;
        # the default component assignment below is strictly column-ordered.
        labels = np.arange(n_pixels) % n_inputs
        rng.shuffle(labels)
        self.base_thresholds = rng.random(self.encoding_shape)
        if layout == 'columns':
            width = self.encoding_shape[1]
            counts = np.full(n_inputs, width//n_inputs, dtype=int)
            counts[:width % n_inputs] += 1
            columns = np.repeat(np.arange(n_inputs), counts)
            self.base_component_map = np.tile(columns, (self.encoding_shape[0], 1))
        else:
            self.base_component_map = labels.reshape(self.encoding_shape)
        self.component_map = self._expand(self.base_component_map)
        self.thresholds = self._expand(self.base_thresholds)
        self.layout = layout

    def _expand(self, array):
        return np.repeat(np.repeat(array, self.block_size, axis=0), self.block_size, axis=1)

    def encode_base(self, input_vector):
        """Threshold the input components on the coarse encoding plane."""
        u = np.asarray(input_vector, dtype=float)
        if u.shape != (self.n_inputs,) or not np.isfinite(u).all() or np.any((u < 0) | (u > 1)):
            raise ValueError('input_vector must be finite, shape (n_inputs,), and within [0,1]')
        return (u[self.base_component_map] > self.base_thresholds).astype(np.float32)

    def encode(self, input_vector):
        """Copy each thresholded input-plane pixel into a block_size² block."""
        return self._expand(self.encode_base(input_vector))

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
