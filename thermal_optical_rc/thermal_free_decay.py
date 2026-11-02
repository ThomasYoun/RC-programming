"""Fixed disordered optics with a single freely cooling phase screen.

All experiments share the same instance, hence exactly the same A and B.
The heat source is switched off before k=0; no input heats the sample.
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

    def temperature(self, k, dt=1.0):
        k = np.asarray(k, dtype=np.float64)
        if self.tau is None:
            return np.full_like(k, self.theta0)
        if self.tau <= 0 or dt <= 0:
            raise ValueError('tau and dt must be positive')
        return self.theta0 * np.exp(-k * dt / self.tau)


class FixedDisorderedOptics:
    """Implicit T(k) = B D(k) A; camera sees |T(k) E(u_k)|².

    A = FFT diag(random_phase_a), B = IFFT diag(random_phase_b).
    This is a structured random unitary mixing model, not a measured sample.
    """
    def __init__(self, shape=(32, 32), camera_shape=(12, 12), seed=42, device='cpu'):
        self.shape = tuple(shape)
        self.camera_shape = tuple(camera_shape)
        self.device = torch.device(device)
        generator = torch.Generator(device=self.device).manual_seed(seed)
        self.input_mask = torch.rand(self.shape, generator=generator, device=self.device)
        self.phase_a = torch.exp(2j * torch.pi * torch.rand(self.shape, generator=generator, device=self.device))
        self.phase_b = torch.exp(2j * torch.pi * torch.rand(self.shape, generator=generator, device=self.device))
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

    @torch.no_grad()
    def intensity(self, inputs, theta, phase_max, batch_size=128):
        """Full camera intensity for corresponding arrays of u and theta."""
        u = torch.as_tensor(np.asarray(inputs), dtype=torch.float32, device=self.device).flatten()
        th = torch.as_tensor(np.asarray(theta), dtype=torch.float32, device=self.device).flatten()
        if len(u) != len(th):
            raise ValueError('inputs and theta must have the same length')
        frames = []
        for start in range(0, len(u), batch_size):
            x = u[start:start+batch_size].clamp(0, 1)[:, None, None]
            binary = (self.input_mask[None] <= x).float()
            field = torch.exp(1j*torch.pi*binary)
            internal = torch.fft.fft2(field*self.phase_a, norm='ortho')
            delta = phase_max*th[start:start+batch_size, None, None]*self.profile
            field_out = torch.fft.ifft2(internal*torch.exp(1j*delta)*self.phase_b, norm='ortho')
            frames.append(field_out.abs().square().cpu())
        return torch.cat(frames).numpy()

    def features(self, inputs, theta, phase_max, batch_size=128):
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
