"""SLM optical reservoir with electronic feedback and local thermal memory.

Fixed random optics are initialized once. A local heat-induced phase screen is
inserted between the two mixing stages so both SLM-region-to-CCD field maps
change together. Only the linear ridge readout is trained.
"""
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class PhysicalNoise:
    """Illustrative noise levels, not calibrated to any particular hardware.

    pupil_rms_rad: static wavefront phase RMS inside each objective pupil.
    slm_phase_std_rad: independent temporal SLM pixel phase jitter.
    photon_scale: expected photoelectrons per unit of optical intensity.
    read_noise_electrons: CCD Gaussian read-noise standard deviation.
    """
    pupil_rms_rad: float = 0.06
    slm_phase_std_rad: float = 0.015
    photon_scale: float = 50000.0
    read_noise_electrons: float = 1.0
    shot_noise: bool = True


class OpticalFeedbackThermalRC:
    """Optical system with concatenated previous state and input on an SLM.

    Temperature updates before each camera exposure, so state r[k+1] depends
    on r[k], u[k], and theta[k]. `feedback=False` fixes the upper SLM half to
    zero phase and isolates the thermal memory from electronic feedback.
    """
    def __init__(self, optical_shape=(128, 128), slm_shape=(128, 128),
                 input_shape=(32, 32), state_shape=(16, 16),
                 pupil_radius=(0.25, 0.25), heater_fraction=0.1,
                 heater_phase_max=6.0, thermal_tau=8.0, dt=1.0,
                 heater_mode='driven', feedback=True, leak=1.0,
                 camera_gain=40.0, scatter_scale=0.8, optical_gain=0.8,
                 noise=PhysicalNoise(), seed=42, noise_seed=123,
                 device='cpu'):
        self.optical_shape = tuple(optical_shape)
        self.slm_shape = tuple(slm_shape)
        self.input_shape = tuple(input_shape)
        self.state_shape = tuple(state_shape)
        self.device = torch.device(device)
        if not 0 < heater_fraction <= 1 or not 0 < leak <= 1:
            raise ValueError('heater_fraction and leak must be in (0,1]')
        if thermal_tau <= 0 or dt <= 0 or camera_gain <= 0:
            raise ValueError('thermal_tau, dt and camera_gain must be positive')
        if heater_mode not in ('off', 'static', 'instant', 'driven'):
            raise ValueError('heater_mode must be off, static, instant or driven')
        if any(v <= 0 for v in (*self.optical_shape, *self.slm_shape, *self.input_shape, *self.state_shape)):
            raise ValueError('dimensions must be positive')
        if len(pupil_radius) != 2 or any(not 0 < r <= 1 for r in pupil_radius):
            raise ValueError('pupil radii must be in (0,1]')
        if min(noise.pupil_rms_rad, noise.slm_phase_std_rad, noise.read_noise_electrons) < 0 or noise.photon_scale <= 0:
            raise ValueError('noise levels must be non-negative and photon_scale positive')
        self.pupil_radius = tuple(pupil_radius)
        self.heater_phase_max = float(heater_phase_max)
        self.thermal_tau, self.dt = float(thermal_tau), float(dt)
        self.heater_mode, self.feedback, self.leak = heater_mode, feedback, float(leak)
        self.camera_gain, self.optical_gain, self.noise = float(camera_gain), float(optical_gain), noise
        self.noise_seed = noise_seed

        fixed = torch.Generator(device=self.device).manual_seed(seed)
        self.input_mask = torch.rand(self.input_shape, generator=fixed, device=self.device)
        def complex_screen():
            real = 2*torch.rand(self.optical_shape, generator=fixed, device=self.device)-1
            imag = 2*torch.rand(self.optical_shape, generator=fixed, device=self.device)-1
            screen = torch.complex(real, imag)
            return scatter_scale*screen/screen.abs().square().mean().sqrt()
        self.scatter_pre, self.scatter_fourier = complex_screen(), complex_screen()
        self.pupil1, self.pupil1_support = self._pupil(pupil_radius[0], noise.pupil_rms_rad, sign=1)
        self.pupil2, self.pupil2_support = self._pupil(pupil_radius[1], noise.pupil_rms_rad, sign=-1)
        h, w = self.optical_shape
        yy, xx = torch.meshgrid(torch.arange(h, device=self.device), torch.arange(w, device=self.device), indexing='ij')
        radius2 = (yy-(h-1)/2)**2 + (xx-(w-1)/2)**2
        count = max(1, round(h*w*heater_fraction))
        support = torch.zeros(h*w, device=self.device, dtype=torch.bool)
        support[torch.argsort(radius2.flatten())[:count]] = True
        self.heater_support = support.reshape(h, w)
        sigma = max(0.5*np.sqrt(count/np.pi), 1e-6)
        self.heater_profile = torch.exp(-radius2/(2*sigma**2))*self.heater_support
        self.heater_profile /= self.heater_profile.max()
        self.reset()

    def _pupil(self, radius, rms, sign):
        h, w = self.optical_shape
        y, x = torch.meshgrid(torch.linspace(-1, 1, h, device=self.device),
                              torch.linspace(-1, 1, w, device=self.device), indexing='ij')
        support = (x.square()+y.square()) <= radius**2
        if not torch.any(support):
            raise ValueError('Pupil is empty; increase the radius or optical resolution')
        # A simple fixed defocus + astigmatism error; normalized inside pupil.
        phase_shape = (2*(x.square()+y.square()) + sign*(x.square()-y.square()))
        values = phase_shape[support]
        phase_shape = (phase_shape-values.mean())/(values.square().mean()-values.mean().square()).clamp_min(1e-12).sqrt()
        return support*torch.exp(1j*rms*phase_shape), support

    @staticmethod
    def _center_pad_crop(field, target):
        h, w = field.shape[-2:]
        th, tw = target
        ch, cw = min(h, th), min(w, tw)
        crop = field[..., (h-ch)//2:(h-ch)//2+ch, (w-cw)//2:(w-cw)//2+cw]
        top, left = (th-ch)//2, (tw-cw)//2
        return F.pad(crop, (left, tw-cw-left, top, th-ch-top))

    @staticmethod
    def _objective(field, pupil):
        spectrum = torch.fft.fftshift(torch.fft.fft2(field, norm='ortho'), dim=(-2, -1))
        return torch.fft.ifft2(torch.fft.ifftshift(spectrum*pupil, dim=(-2, -1)), norm='ortho')

    def reset(self, temperature=0.0):
        self.state = torch.zeros(self.state_shape, dtype=torch.float32, device=self.device)
        self.temperature = float(temperature)
        self.noise_generator = torch.Generator(device=self.device).manual_seed(self.noise_seed)

    def _slm(self, u, add_noise=True):
        input_img = (self.input_mask <= u).float()
        state_img = self.state if self.feedback else torch.zeros_like(self.state)
        top_h = self.slm_shape[0]//2
        def resize_and_center(img, target_h):
            # Expand to square pixels matching the target half-height, then
            # center on the half-screen, as in the original layout.
            img = F.interpolate(img[None, None], size=(target_h, target_h), mode='nearest')[0, 0]
            return self._center_pad_crop(img, (target_h, self.slm_shape[1]))
        phase_levels = torch.cat((resize_and_center(state_img, top_h),
                                  resize_and_center(input_img, self.slm_shape[0]-top_h)), dim=0)
        phase = torch.pi*phase_levels
        if add_noise and self.noise.slm_phase_std_rad:
            phase = phase + self.noise.slm_phase_std_rad*torch.randn(
                phase.shape, generator=self.noise_generator, device=self.device)
        return input_img, phase, torch.exp(1j*phase)

    def optical_field(self, slm_field, temperature):
        """Linear complex-field propagation at fixed temperature, no readout."""
        field = self._center_pad_crop(slm_field, self.optical_shape)
        field = self._objective(field, self.pupil1)
        field = field*self.scatter_pre
        phase = self.heater_phase_max*temperature*self.heater_profile
        field = field*torch.exp(1j*phase)
        field = torch.fft.fft2(field, norm='ortho')
        field = self.optical_gain*torch.fft.ifft2(field*self.scatter_fourier, norm='ortho')
        return self._objective(field, self.pupil2)

    def _update_temperature(self, u):
        if self.heater_mode == 'off':
            self.temperature = 0.0
        elif self.heater_mode == 'static':
            self.temperature = 0.5
        elif self.heater_mode == 'instant':
            self.temperature = float(u)
        else:
            decay = np.exp(-self.dt/self.thermal_tau)
            self.temperature = decay*self.temperature + (1-decay)*float(u)

    @torch.no_grad()
    def step(self, u, record=False):
        u = float(np.clip(float(u), 0, 1))
        self._update_temperature(u)
        state_before = self.state.clone()
        input_img, slm_phase, slm_field = self._slm(u)
        field_ccd = self.optical_field(slm_field, self.temperature)
        intensity = field_ccd.abs().square()
        expected_electrons = intensity*self.noise.photon_scale
        electrons = (torch.poisson(expected_electrons, generator=self.noise_generator)
                     if self.noise.shot_noise else expected_electrons)
        if self.noise.read_noise_electrons:
            electrons = electrons + self.noise.read_noise_electrons*torch.randn(
                electrons.shape, generator=self.noise_generator, device=self.device)
        measured = torch.clamp(electrons/self.noise.photon_scale, min=0)
        camera_state = F.adaptive_avg_pool2d(measured[None, None], self.state_shape)[0, 0]
        # Fixed camera gain avoids per-frame min-max normalization. Clipping
        # models the bounded digital signal sent back to the phase SLM.
        camera_state = torch.clamp(self.camera_gain*camera_state, 0, 1)
        self.state = (1-self.leak)*self.state + self.leak*camera_state
        if not record:
            return self.state.detach().cpu().numpy().copy(), self.temperature
        return self.state.detach().cpu().numpy().copy(), self.temperature, {
            'input_img': input_img.cpu().numpy(), 'state_before': state_before.cpu().numpy(),
            'slm_phase': slm_phase.cpu().numpy(), 'slm_field': slm_field.cpu().numpy(),
            'ccd_ideal': intensity.cpu().numpy(), 'ccd_measured': measured.cpu().numpy(),
            'camera_state': camera_state.cpu().numpy(), 'state_after': self.state.cpu().numpy(),
            'temperature': self.temperature,
        }

    def run(self, inputs, reset=True, snapshot_at=None):
        if reset:
            self.reset()
        u = np.asarray(inputs).reshape(-1)
        states = np.empty((len(u), np.prod(self.state_shape)), dtype=np.float32)
        temperatures = np.empty(len(u), dtype=np.float64)
        snapshot = None
        for k, val in enumerate(u):
            if k == snapshot_at:
                state, theta, snapshot = self.step(val, record=True)
            else:
                state, theta = self.step(val)
            states[k] = state.ravel()
            temperatures[k] = theta
        return states, temperatures, snapshot


class RidgeReadout:
    """ESN-style ridge readout fitted to training data only."""
    def __init__(self, alpha=1.0):
        if alpha <= 0:
            raise ValueError('ridge alpha must be positive')
        self.alpha = alpha

    def fit(self, x, y):
        x = np.asarray(x, dtype=np.float64)
        self.mean_ = x.mean(axis=0)
        self.scale_ = x.std(axis=0)
        self.scale_[self.scale_ < 1e-8] = 1.0
        z = np.column_stack((np.ones(len(x)), (x-self.mean_)/self.scale_))
        penalty = self.alpha*np.eye(z.shape[1]); penalty[0, 0] = 0
        self.weights_ = np.linalg.solve(z.T@z + penalty, z.T@np.asarray(y))
        return self

    def predict(self, x):
        z = np.column_stack((np.ones(len(x)), (x-self.mean_)/self.scale_))
        return z@self.weights_


def nrmse(y, prediction):
    return float(np.sqrt(np.mean((y-prediction)**2))/np.std(y))
