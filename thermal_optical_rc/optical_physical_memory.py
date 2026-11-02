"""Physical optical RC: input-driven heat is the only dynamical state.

At sample k the camera observes H(theta_k, u_k), after which heat evolves as
theta_{k+1} = a*theta_k + beta*u_k. The SLM carries only the current input;
no camera signal or previous speckle is ever sent back to the SLM.
"""
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class PhysicalNoise:
    pupil_rms_rad: float = 0.18      # fixed pupil wavefront error, RMS radians
    slm_phase_std_rad: float = 0.06  # per-sample phase jitter, radians
    photon_scale: float = 5000.0     # expected photoelectrons / unit intensity
    read_noise_electrons: float = 3.0
    shot_noise: bool = True


class PhysicalMemoryOptics:
    """Fixed SLM-to-CCD optical propagation with one scalar heat state."""
    def __init__(self, optical_shape=(128, 128), slm_shape=(128, 128),
                 input_shape=(32, 32), camera_shape=(16, 16),
                 pupil_radius=(0.25, 0.25), heater_fraction=0.8,
                 heater_phase_max=6.0, tau=8.0, dt=1.0, beta=None,
                 thermal_gain=1.3, heat_mode='driven', static_theta=None,
                 scatter_scale=0.8, optical_gain=0.8,
                 slm_encoding='continuous_global',
                 noise=PhysicalNoise(), seed=42, noise_seed=123, device='cpu'):
        self.optical_shape = tuple(optical_shape)
        self.slm_shape = tuple(slm_shape)
        self.input_shape = tuple(input_shape)
        self.camera_shape = tuple(camera_shape)
        self.device = torch.device(device)
        if any(len(s) != 2 or any(v <= 0 for v in s) for s in
               (self.optical_shape, self.slm_shape, self.input_shape, self.camera_shape)):
            raise ValueError('all shapes must contain two positive dimensions')
        if not (0 < heater_fraction <= 1 and tau > 0 and dt > 0 and thermal_gain >= 0):
            raise ValueError('heater_fraction must be in (0,1]; tau and dt positive; thermal_gain non-negative')
        if len(pupil_radius) != 2 or any(not 0 < r <= 1 for r in pupil_radius):
            raise ValueError('pupil radii must be in (0,1]')
        if heat_mode not in ('driven', 'off', 'static', 'instant'):
            raise ValueError('heat_mode must be driven, off, static or instant')
        if slm_encoding not in ('continuous_global', 'binary_threshold'):
            raise ValueError('slm_encoding must be continuous_global or binary_threshold')
        if min(noise.pupil_rms_rad, noise.slm_phase_std_rad, noise.read_noise_electrons) < 0 or noise.photon_scale <= 0:
            raise ValueError('noise amplitudes must be non-negative; photon_scale must be positive')
        self.tau, self.dt = float(tau), float(dt)
        self.a = float(np.exp(-self.dt/self.tau))
        # Default keeps the continuous constant-input steady-state gain fixed
        # across tau; passing beta explicitly uses exactly the requested form.
        self.beta = float((1-self.a)*thermal_gain if beta is None else beta)
        if self.beta < 0 or not np.isfinite(self.beta):
            raise ValueError('beta must be finite and non-negative')
        self.heat_mode, self.thermal_gain = heat_mode, float(thermal_gain)
        self.static_theta = 0.5*self.thermal_gain if static_theta is None else float(static_theta)
        self.heater_phase_max, self.optical_gain = float(heater_phase_max), float(optical_gain)
        self.noise, self.noise_seed = noise, noise_seed
        self.slm_encoding = slm_encoding

        fixed = torch.Generator(device=self.device).manual_seed(seed)
        self.input_mask = torch.rand(self.input_shape, generator=fixed, device=self.device)
        def complex_screen():
            real = 2*torch.rand(self.optical_shape, generator=fixed, device=self.device)-1
            imag = 2*torch.rand(self.optical_shape, generator=fixed, device=self.device)-1
            screen = torch.complex(real, imag)
            return scatter_scale*screen/screen.abs().square().mean().sqrt()
        self.scatter_pre, self.scatter_fourier = complex_screen(), complex_screen()
        self.pupil1, self.pupil1_support = self._pupil(pupil_radius[0], noise.pupil_rms_rad, 1)
        self.pupil2, self.pupil2_support = self._pupil(pupil_radius[1], noise.pupil_rms_rad, -1)
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
        support = x.square()+y.square() <= radius**2
        if not torch.any(support):
            raise ValueError('empty pupil; increase radius or resolution')
        aberration = 2*(x.square()+y.square())+sign*(x.square()-y.square())
        values = aberration[support]
        aberration = (aberration-values.mean())/values.std(unbiased=False).clamp_min(1e-6)
        return support*torch.exp(1j*rms*aberration), support

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
        ft = torch.fft.fftshift(torch.fft.fft2(field, norm='ortho'), dim=(-2, -1))
        return torch.fft.ifft2(torch.fft.ifftshift(ft*pupil, dim=(-2, -1)), norm='ortho')

    def reset(self, theta0=0.0):
        self.theta = float(theta0)
        self.noise_generator = torch.Generator(device=self.device).manual_seed(self.noise_seed)

    def _encode_input(self, u):
        if self.slm_encoding == 'continuous_global':
            # Every SLM pixel responds continuously to u; fixed spatial gain
            # varies in [0.5, 1] to avoid an invisible global phase shift.
            input_image = u*(0.5+0.5*self.input_mask)
        else:
            # Original notebook encoding; a small input change flips only the
            # pixels whose random thresholds are crossed.
            input_image = (self.input_mask <= u).float()
        slm_pattern = F.interpolate(input_image[None, None], size=self.slm_shape, mode='nearest')[0, 0]
        phase = torch.pi*slm_pattern
        if self.noise.slm_phase_std_rad:
            phase += self.noise.slm_phase_std_rad*torch.randn(phase.shape, generator=self.noise_generator, device=self.device)
        return input_image, phase, torch.exp(1j*phase)

    def optical_field(self, slm_field, theta):
        """O2 B D(theta) A O1 acting on a complex SLM field."""
        field = self._center_pad_crop(slm_field, self.optical_shape)
        field = self._objective(field, self.pupil1)*self.scatter_pre
        field = field*torch.exp(1j*self.heater_phase_max*theta*self.heater_profile)
        field = torch.fft.fft2(field, norm='ortho')
        field = self.optical_gain*torch.fft.ifft2(field*self.scatter_fourier, norm='ortho')
        return self._objective(field, self.pupil2)

    @torch.no_grad()
    def step(self, u, heat_input=None, record=False):
        """Observe r_k=H(theta_k,u_k), then write theta_{k+1}."""
        u = float(u)
        heat_input = u if heat_input is None else float(heat_input)
        if not (0 <= u <= 1 and 0 <= heat_input <= 1):
            raise ValueError('probe and heating inputs must be in [0,1]')
        if self.heat_mode == 'off':
            theta_read = 0.0
        elif self.heat_mode == 'static':
            theta_read = self.static_theta
        elif self.heat_mode == 'instant':
            theta_read = self.thermal_gain*heat_input
        else:
            theta_read = self.theta
        input_image, slm_phase, slm_field = self._encode_input(u)
        ideal = self.optical_field(slm_field, theta_read).abs().square()
        expected = ideal*self.noise.photon_scale
        electrons = torch.poisson(expected, generator=self.noise_generator) if self.noise.shot_noise else expected
        if self.noise.read_noise_electrons:
            electrons += self.noise.read_noise_electrons*torch.randn(electrons.shape, generator=self.noise_generator, device=self.device)
        measured = torch.clamp(electrons/self.noise.photon_scale, min=0)
        features = F.adaptive_avg_pool2d(measured[None, None], self.camera_shape)[0, 0].cpu().numpy().copy()
        theta_next = self.a*theta_read+self.beta*heat_input if self.heat_mode == 'driven' else theta_read
        self.theta = theta_next
        if not record:
            return features, theta_read, theta_next
        return features, theta_read, theta_next, {
            'input_image': input_image.cpu().numpy(), 'slm_phase': slm_phase.cpu().numpy(),
            'slm_field': slm_field.cpu().numpy(), 'ccd_ideal': ideal.cpu().numpy(),
            'ccd_measured': measured.cpu().numpy(), 'features': features,
            'theta_read': theta_read, 'theta_next': theta_next,
        }

    def run(self, inputs, heat_inputs=None, reset=True, theta0=0.0, snapshot_at=None):
        if reset:
            self.reset(theta0)
        u = np.asarray(inputs).reshape(-1)
        q = u if heat_inputs is None else np.asarray(heat_inputs).reshape(-1)
        if len(u) != len(q):
            raise ValueError('probe and heating input lengths differ')
        states = np.empty((len(u), np.prod(self.camera_shape)), dtype=np.float32)
        theta = np.empty(len(u), dtype=np.float64)
        next_theta = np.empty(len(u), dtype=np.float64)
        snapshot = None
        for k, (value, power) in enumerate(zip(u, q)):
            if k == snapshot_at:
                r, theta[k], next_theta[k], snapshot = self.step(value, power, record=True)
            else:
                r, theta[k], next_theta[k] = self.step(value, power)
            states[k] = r.ravel()
        return states, theta, next_theta, snapshot


class RidgeReadout:
    """ESN-style linear readout, training-set standardization and free bias."""
    def __init__(self, alpha=1.0):
        if alpha <= 0:
            raise ValueError('alpha must be positive')
        self.alpha = float(alpha)

    def fit(self, x, y):
        x = np.asarray(x, dtype=np.float64)
        self.mean_ = x.mean(axis=0)
        self.scale_ = x.std(axis=0)
        self.scale_[self.scale_ < 1e-8] = 1.0
        z = np.column_stack((np.ones(len(x)), (x-self.mean_)/self.scale_))
        penalty = self.alpha*np.eye(z.shape[1]); penalty[0, 0] = 0
        self.weights_ = np.linalg.solve(z.T@z+penalty, z.T@np.asarray(y))
        return self

    def predict(self, x):
        z = np.column_stack((np.ones(len(x)), (x-self.mean_)/self.scale_))
        return z@self.weights_


def nrmse(y, prediction):
    return float(np.sqrt(np.mean((y-prediction)**2))/np.std(y))
