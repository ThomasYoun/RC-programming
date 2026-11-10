"""Input-driven direct-phase optical RC, with no electronic feedback."""
import numpy as np
import torch
import torch.nn.functional as F


def chronological_indices(train_length=2000, washout=200, validation_length=400,
                          test_length=1000, horizon=1):
    """Chronological feature indices; prediction target is input[k+horizon].

    Features are indexed by the input that was written immediately before
    observation. Fitting/validation targets must stay inside the training
    segment, so the final horizon samples are omitted at each boundary.
    """
    sizes = (train_length, washout, validation_length, test_length, horizon)
    if any(not isinstance(n, (int, np.integer)) for n in sizes):
        raise ValueError('all split lengths must be integers')
    if horizon < 0 or washout < 0 or validation_length <= horizon or test_length <= 0:
        raise ValueError('invalid split lengths')
    boundary = train_length-validation_length
    if boundary-horizon <= washout:
        raise ValueError('not enough training samples')
    return {'fit': np.arange(washout, boundary-horizon),
            'valid': np.arange(boundary, train_length-horizon),
            'refit': np.arange(washout, train_length-horizon),
            'test': np.arange(train_length, train_length+test_length)}


def simulate_phase_rc(inputs, optics, profile, initial_phase, tau, beta,
                      mode='memory', camera_shape=(16, 16), photon_scale=None,
                      read_noise_electrons=0.0, noise_seed=123):
    """Apply input k, then observe the resulting phase and CCD speckle.

    memory: phi_next = exp(-1/tau)*phi + beta*u[k]*profile;
    instant: phi_next = beta*u[k]*profile (no retained state);
    static: phi_next = initial_phase (independent of input).

    Returns features[k] from phi_next and phase_next at the profile peak.
    Only one full phase image is held at once. If photon_scale is provided,
    independent Poisson shot noise and Gaussian read noise are applied to
    CCD pixels before camera downsampling; otherwise observation is ideal.
    """
    u = np.asarray(inputs, dtype=float)
    g = np.asarray(profile, dtype=float)
    phase = np.asarray(initial_phase, dtype=float).copy()
    if u.ndim != 1 or not len(u) or not np.isfinite(u).all():
        raise ValueError('inputs must be a nonempty finite 1-D array')
    if phase.shape != optics.optical_shape or g.shape != phase.shape:
        raise ValueError('phase and profile must match optical_shape')
    if not np.isfinite(phase).all() or not np.isfinite(g).all():
        raise ValueError('phase and profile must be finite')
    if mode not in ('memory', 'instant', 'static'):
        raise ValueError('mode must be memory, instant or static')
    if not np.isfinite(tau) or tau <= 0 or not np.isfinite(beta):
        raise ValueError('tau must be positive and beta finite')
    if len(camera_shape) != 2 or any(not isinstance(n, (int, np.integer)) or n < 1 for n in camera_shape):
        raise ValueError('camera_shape must contain two positive integers')
    if any(c > p for c, p in zip(camera_shape, phase.shape)):
        raise ValueError('camera_shape cannot exceed optical_shape')
    if (photon_scale is not None and (not np.isfinite(photon_scale) or photon_scale <= 0)) or not np.isfinite(read_noise_electrons) or read_noise_electrons < 0:
        raise ValueError('invalid camera noise parameters')
    if photon_scale is None and read_noise_electrons:
        raise ValueError('read noise requires photon_scale')
    rng = np.random.default_rng(noise_seed)
    a = np.exp(-1/tau)
    features = np.empty((len(u), int(np.prod(camera_shape))), dtype=np.float32)
    center_phase = np.empty(len(u))
    peak = np.unravel_index(np.argmax(g), g.shape)
    for k, uk in enumerate(u):
        if mode == 'memory':
            phase = a*phase + beta*uk*g
        elif mode == 'instant':
            phase = beta*uk*g
        # Static means the initially random phase remains frozen.
        center_phase[k] = phase[peak]
        intensity = np.abs(optics.field(phase))**2
        if photon_scale is not None:
            electrons = rng.poisson(intensity*photon_scale).astype(float)
            if read_noise_electrons:
                electrons += rng.normal(0, read_noise_electrons, intensity.shape)
            intensity = np.maximum(electrons, 0)/photon_scale
        tensor = torch.as_tensor(np.ascontiguousarray(intensity), dtype=torch.float32)
        features[k] = F.adaptive_avg_pool2d(tensor[None, None], camera_shape).numpy().ravel()
    return features, center_phase


def r2_score(y, prediction):
    y = np.asarray(y, dtype=float)
    pred = np.asarray(prediction, dtype=float)
    if y.shape != pred.shape or y.size == 0:
        raise ValueError('target and prediction must have identical nonempty shape')
    denominator = np.sum((y-y.mean())**2)
    return float(1-np.sum((y-pred)**2)/denominator) if denominator else float('nan')
