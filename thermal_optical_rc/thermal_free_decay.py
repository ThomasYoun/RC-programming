"""Fixed disordered optics with a single, optionally input-driven thermal state.

All experiments share the same optical instance; no camera feedback is used.
"""

from dataclasses import dataclass
import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class Condition:
    name: str
    phase_max: float
    tau: float | None  # None means a static temperature
    theta0: float = 1.0
    mode: str = 'free'  # 'free', 'driven', 'instant', or 'static'

    def trajectory(self, inputs, dt=1.0):
        """Theta sampled after input k: exact zero-order-hold relaxation."""
        inputs = np.asarray(inputs, dtype=np.float64).reshape(-1)
        if self.mode == 'static' or (self.mode == 'free' and self.tau is None):
            return np.full(len(inputs), self.theta0)
        if self.mode == 'instant':
            if np.any((inputs < 0) | (inputs > 1)):
                raise ValueError('Driven heater power must be within [0, 1]')
            return inputs.copy()
        if self.mode not in ('free', 'driven'):
            raise ValueError('mode must be static, free, driven, or instant')
        if self.tau is None:
            raise ValueError('tau is required for a dynamic condition')
        if self.tau <= 0 or dt <= 0:
            raise ValueError('tau and dt must be positive')
        if self.mode == 'free':
            return self.theta0 * np.exp(-np.arange(len(inputs))*dt/self.tau)
        if np.any((inputs < 0) | (inputs > 1)):
            raise ValueError('Driven heater power must be within [0, 1]')
        decay = np.exp(-dt/self.tau)
        state = self.theta0  # theta immediately before first input
        values = np.empty(len(inputs))
        for k, drive in enumerate(inputs):
            state = decay*state + (1-decay)*drive
            values[k] = state
        return values


class FixedDisorderedOptics:
    """SLM -> objective 1 -> fixed scattering -> heater -> objective 2 -> CCD.

    The complete SLM encodes the current input image. No CCD image is fed back.
    A and B are fixed implicit random mixing operators, not measured media.
    """
    def __init__(self, shape=(128, 128), camera_shape=(16, 16),
                 input_shape=(64, 64), slm_shape=(128, 128),
                 objective1_radius=0.25, objective2_radius=0.25,
                 scatter_scale=0.8, optical_gain=0.8,
                 seed=42, device='cpu'):
        self.shape = tuple(shape)  # scattering / optical plane
        self.camera_shape = tuple(camera_shape)
        self.input_shape = tuple(input_shape)
        self.slm_shape = tuple(slm_shape)
        self.device = torch.device(device)
        if len(self.shape) != 2 or len(self.input_shape) != 2 or len(self.slm_shape) != 2:
            raise ValueError('All shapes must have two dimensions')
        if any(v <= 0 for v in (*self.shape, *self.camera_shape, *self.input_shape, *self.slm_shape)):
            raise ValueError('All dimensions must be positive')
        if not (0 < objective1_radius <= 1 and 0 < objective2_radius <= 1):
            raise ValueError('Objective pupil radii must be in (0, 1]')
        generator = torch.Generator(device=self.device).manual_seed(seed)
        self.input_mask = torch.rand(self.input_shape, generator=generator, device=self.device)
        def random_complex_screen():
            real = 2*torch.rand(self.shape, generator=generator, device=self.device)-1
            imag = 2*torch.rand(self.shape, generator=generator, device=self.device)-1
            screen = torch.complex(real, imag)
            return scatter_scale*screen/screen.abs().square().mean().sqrt()
        # Match the source notebook's fixed complex-amplitude screens.
        self.scatter_pre = random_complex_screen()
        self.scatter_fourier = random_complex_screen()
        self.optical_gain = float(optical_gain)
        self.pupil1 = self._make_pupil(objective1_radius)
        self.pupil2 = self._make_pupil(objective2_radius)
        h, w = self.shape
        yy, xx = torch.meshgrid(torch.arange(h, device=self.device), torch.arange(w, device=self.device), indexing='ij')
        dist2 = (yy - (h - 1)/2)**2 + (xx - (w - 1)/2)**2
        count = round(h*w/4)
        self.support = torch.zeros(h*w, dtype=torch.bool, device=self.device)
        self.support[torch.argsort(dist2.flatten())[:count]] = True
        self.support = self.support.reshape(self.shape)
        sigma = 0.5*np.sqrt(count/np.pi)
        self.profile = torch.exp(-dist2/(2*sigma**2))*self.support
        self.profile /= self.profile.max()

    def _make_pupil(self, radius):
        h, w = self.shape
        # Same normalized Fourier coordinates and centered circular pupil as
        # RC_opMGSLM.ipynb; radius is a fraction of the Fourier half-width.
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, h, device=self.device),
                                torch.linspace(-1, 1, w, device=self.device), indexing='ij')
        return ((xx.square() + yy.square()) <= radius**2).float()

    @staticmethod
    def _center_pad_crop(field, target_shape):
        """Center a batch of fields, padding the field amplitude with zeros."""
        th, tw = target_shape
        h, w = field.shape[-2:]
        ch, cw = min(h, th), min(w, tw)
        y0, x0 = (h-ch)//2, (w-cw)//2
        cropped = field[..., y0:y0+ch, x0:x0+cw]
        top, left = (th-ch)//2, (tw-cw)//2
        return F.pad(cropped, (left, tw-cw-left, top, th-ch-top))

    def _slm_field(self, u):
        """Threshold the fixed mask and expand the image over the entire SLM."""
        image = (self.input_mask[None] <= u[:, None, None]).float()
        phase_pattern = F.interpolate(image[:, None], size=self.slm_shape, mode='nearest')[:, 0]
        return image, phase_pattern, torch.exp(1j*torch.pi*phase_pattern)

    @staticmethod
    def _objective(field, pupil):
        spectrum = torch.fft.fftshift(torch.fft.fft2(field, norm='ortho'), dim=(-2, -1))
        return torch.fft.ifft2(torch.fft.ifftshift(spectrum*pupil, dim=(-2, -1)), norm='ortho')

    def _forward(self, u, theta, phase_max, keep_stages=False):
        image, phase_pattern, slm_field = self._slm_field(u)
        optical_field = self._center_pad_crop(slm_field, self.shape)
        focused = self._objective(optical_field, self.pupil1)
        # Fixed scattering operator, as in RC_opMGSLM. The thermal phase
        # screen is then applied on the effective plane before objective 2.
        internal = torch.fft.fft2(focused*self.scatter_pre, norm='ortho')
        scattered = self.optical_gain*torch.fft.ifft2(internal*self.scatter_fourier, norm='ortho')
        thermal_phase = phase_max*theta[:, None, None]*self.profile
        heated = scattered*torch.exp(1j*thermal_phase)
        camera_field = self._objective(heated, self.pupil2)
        intensity = camera_field.abs().square()
        if keep_stages:
            return {'input_image': image, 'slm_phase': torch.pi*phase_pattern,
                    'slm_field': slm_field, 'optical_field': optical_field,
                    'objective1_field': focused, 'thermal_phase': thermal_phase,
                    'scattered_field': scattered, 'heated_field': heated, 'camera_field': camera_field,
                    'ccd_intensity': intensity,
                    'reservoir_state': F.adaptive_avg_pool2d(intensity[:, None], self.camera_shape)[:, 0]}
        return intensity

    @torch.no_grad()
    def snapshot(self, input_value=0.5, theta=1.0, phase_max=1.0):
        u = torch.tensor([input_value], dtype=torch.float32, device=self.device).clamp(0, 1)
        t = torch.tensor([theta], dtype=torch.float32, device=self.device)
        return {name: value[0].detach().cpu().numpy() for name, value in
                self._forward(u, t, phase_max, keep_stages=True).items()}

    @torch.no_grad()
    def intensity(self, inputs, theta, phase_max, batch_size=32):
        """Full camera intensity for corresponding arrays of u and theta."""
        u = torch.as_tensor(np.asarray(inputs), dtype=torch.float32, device=self.device).flatten()
        th = torch.as_tensor(np.asarray(theta), dtype=torch.float32, device=self.device).flatten()
        if len(u) != len(th):
            raise ValueError('inputs and theta must have the same length')
        frames = []
        for start in range(0, len(u), batch_size):
            x = u[start:start+batch_size].clamp(0, 1)
            frames.append(self._forward(x, th[start:start+batch_size], phase_max).cpu())
        return torch.cat(frames).numpy()

    def features(self, inputs, theta, phase_max, batch_size=32):
        """Downsample camera intensity, keeping common physical scaling."""
        u = np.asarray(inputs)
        theta = np.asarray(theta)
        chunks = []
        for start in range(0, len(u), batch_size):
            frames = self.intensity(u[start:start+batch_size], theta[start:start+batch_size], phase_max, batch_size)
            values = F.adaptive_avg_pool2d(torch.from_numpy(frames)[:, None], self.camera_shape)
            chunks.append(values[:, 0].flatten(1).numpy())
        return np.concatenate(chunks)


def spatial_correlation(frames, reference):
    x = np.asarray(frames).reshape(len(frames), -1)
    y = np.asarray(reference).ravel()
    x = x-x.mean(axis=1, keepdims=True)
    y = y-y.mean()
    return (x@y)/(np.linalg.norm(x, axis=1)*np.linalg.norm(y)+1e-12)


def ridge_predict(train_x, train_y, test_x, alpha=10.0):
    """Fit feature scaling and linear ridge readout on training data only."""
    mu = train_x.mean(axis=0)
    sd = train_x.std(axis=0)
    sd[sd < 1e-8] = 1.0
    x = (train_x-mu)/sd
    xt = (test_x-mu)/sd
    x = np.column_stack((np.ones(len(x)), x)).astype(np.float64)
    xt = np.column_stack((np.ones(len(xt)), xt)).astype(np.float64)
    penalty = np.eye(x.shape[1])*alpha
    penalty[0, 0] = 0
    weights = np.linalg.solve(x.T@x + penalty, x.T@train_y)
    return xt@weights


def scores(target, prediction):
    rmse = np.sqrt(np.mean((target-prediction)**2))
    return {'RMSE': float(rmse), 'NRMSE': float(rmse/np.std(target))}
