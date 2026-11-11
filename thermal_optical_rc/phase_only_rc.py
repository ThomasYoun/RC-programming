"""Direct-phase optical RC and optional output-to-input free forecasting.

The reservoir itself has no CCD-speckle-to-SLM feedback; free forecasting
explicitly feeds the digital readout back into the external input.
"""
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
                      read_noise_electrons=0.0, noise_seed=123, noise_rng=None,
                      return_final_phase=False):
    """Apply input k, then observe the resulting phase and CCD speckle.

    memory: phi_next = exp(-1/tau)*phi + beta*u[k]*profile;
    instant: phi_next = beta*u[k]*profile (no retained state);
    static: phi_next = initial_phase (independent of input).

    Returns features[k] from phi_next and phase_next at the profile peak.
    Only one full phase image is held at once. If photon_scale is provided,
    independent Poisson shot noise and Gaussian read noise are applied to
    CCD pixels before camera downsampling; otherwise observation is ideal.
    Pass a persistent numpy Generator as noise_rng to continue the noise stream
    across chunks. Set return_final_phase to also return the final unwrapped
    phase map, allowing the next chunk to continue without resetting memory.
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
    if noise_rng is not None and not isinstance(noise_rng, np.random.Generator):
        raise ValueError('noise_rng must be a numpy.random.Generator or None')
    rng = np.random.default_rng(noise_seed) if noise_rng is None else noise_rng
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
    if return_final_phase:
        return features, center_phase, phase.copy()
    return features, center_phase


def free_phase_rollout(history_inputs, model, optics, profile, initial_phase,
                       tau, beta, mode='memory', camera_shape=(16, 16),
                       photon_scale=None, read_noise_electrons=0.,
                       noise_seed=123, n_steps=1000):
    """Autonomous one-step rollout seeded exclusively by observed history.

    Warm the phase with history_inputs, then use the final post-write feature to
    predict the first unseen input. Every later step clips the previous raw
    prediction to [0,1], feeds it to the phase screen, and predicts again.
    The physical phase and camera RNG are continuous across the boundary.

    The feedback path is an external readout-to-input forecasting loop; it is
    not CCD-to-SLM feedback within the phase-only physical reservoir.
    ``predictions`` are raw readout outputs, ``feedback_inputs`` are clipped
    values actually written, and ``phase_peak`` records the phase from which
    each prediction was made, starting with the final warm-up observation.
    """
    if (not isinstance(n_steps, (int, np.integer)) or
            isinstance(n_steps, (bool, np.bool_)) or n_steps < 1):
        raise ValueError('n_steps must be a positive integer')
    history = np.asarray(history_inputs, dtype=float)
    if history.ndim != 1 or history.size == 0 or not np.isfinite(history).all():
        raise ValueError('history_inputs must be nonempty finite 1-D')
    if np.any((history < 0) | (history > 1)):
        raise ValueError('history_inputs must be in [0,1]')
    rng = np.random.default_rng(noise_seed)
    kwargs = dict(optics=optics, profile=profile, tau=tau, beta=beta,
                  mode=mode, camera_shape=camera_shape, photon_scale=photon_scale,
                  read_noise_electrons=read_noise_electrons, noise_rng=rng,
                  return_final_phase=True)
    warm_features, warm_phase, phase = simulate_phase_rc(history, initial_phase=initial_phase, **kwargs)
    predictions = np.empty(n_steps, dtype=float)
    feedback_inputs = np.empty(n_steps-1, dtype=float)
    phase_peak = np.empty(n_steps, dtype=float)

    def predict_one(feature):
        prediction = np.asarray(model.predict(feature[None, :]))
        if prediction.size != 1 or not np.isfinite(prediction).all():
            raise ValueError('model must predict one finite scalar per feature')
        return float(prediction.reshape(-1)[0])

    predictions[0] = predict_one(warm_features[-1])
    phase_peak[0] = warm_phase[-1]
    for k in range(1, n_steps):
        feedback_inputs[k-1] = np.clip(predictions[k-1], 0.0, 1.0)
        features, peak, phase = simulate_phase_rc(
            feedback_inputs[k-1:k], initial_phase=phase, **kwargs)
        phase_peak[k] = peak[0]
        predictions[k] = predict_one(features[-1])
    clipping_fraction = float(np.mean(feedback_inputs != predictions[:-1])) if n_steps > 1 else 0.0
    return {'predictions': predictions, 'feedback_inputs': feedback_inputs,
            'clipping_fraction': clipping_fraction, 'phase_peak': phase_peak}


def r2_score(y, prediction):
    y = np.asarray(y, dtype=float)
    pred = np.asarray(prediction, dtype=float)
    if y.shape != pred.shape or y.size == 0:
        raise ValueError('target and prediction must have identical nonempty shape')
    denominator = np.sum((y-y.mean())**2)
    return float(1-np.sum((y-pred)**2)/denominator) if denominator else float('nan')
