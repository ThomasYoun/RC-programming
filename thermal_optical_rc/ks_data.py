"""Deterministic 1-D Kuramoto–Sivashinsky data on a periodic grid.

We solve u_t = -u*u_x - u_xx - u_xxxx with a Fourier pseudospectral
spatial discretization and contour-stabilized ETDRK4, following Kassam and
Trefethen, *Fourth-Order Time-Stepping for Stiff PDEs* (2005), §4:
https://doi.org/10.1137/S1064827502410633

This is a configurable small KS benchmark, not the dataset or experimental
configuration of Wang et al.'s optical NGRC paper. Its temporal and spatial
sampling should be reported when comparing forecast scores.
"""
import numpy as np


def generate_ks(n_samples=3001, n_grid=64, domain_length=22.0, dt=0.25,
                sample_every=1, warmup_steps=1000, seed=2026,
                initial_state=None):
    """Return real KS fields of shape (n_samples, n_grid).

    The periodic grid is x_j=j*domain_length/n_grid for j=0,...,n_grid-1;
    the endpoint is not duplicated. The first sample is recorded *after*
    warmup_steps integration steps; successive records are sample_every
    integration steps apart. Mean(u) is conserved. seed controls the small
    random perturbation to a smooth zero-mean initial state unless an explicit
    initial_state is provided. The 2/3 mask de-aliases quadratic nonlinearity.

    `dt` is the integration time step; the interval between returned data
    points is `dt * sample_every`. All quantities are dimensionless.
    """
    integers = (n_samples, n_grid, sample_every, warmup_steps)
    if any(not isinstance(n, (int, np.integer)) or isinstance(n, (bool, np.bool_)) for n in integers):
        raise ValueError('sample counts, grid size and warmup must be integers')
    if n_samples < 1 or n_grid < 16 or n_grid % 2 or sample_every < 1 or warmup_steps < 0:
        raise ValueError('n_samples and sample_every must be positive; n_grid must be even and >= 16; warmup nonnegative')
    if not np.isfinite(domain_length) or domain_length <= 0 or not np.isfinite(dt) or dt <= 0:
        raise ValueError('domain_length and dt must be finite and positive')
    if initial_state is None:
        x = np.arange(n_grid)*domain_length/n_grid
        z = 2*np.pi*x/domain_length
        rng = np.random.default_rng(seed)
        u0 = np.cos(z)*(1+np.sin(z)) + 0.1*rng.standard_normal(n_grid)
        u0 -= u0.mean()
    else:
        u0 = np.asarray(initial_state, dtype=float)
        if u0.shape != (n_grid,) or not np.isfinite(u0).all():
            raise ValueError('initial_state must be finite and have shape (n_grid,)')
    # Even-grid Nyquist has no well-defined odd derivative on the real grid.
    wave = 2*np.pi*np.fft.fftfreq(n_grid, d=domain_length/n_grid)
    linear = wave**2-wave**4
    derivative_wave = wave.copy()
    derivative_wave[n_grid//2] = 0
    modes = np.fft.fftfreq(n_grid)*n_grid
    dealias = np.abs(modes) <= n_grid//3
    v = np.fft.fft(u0)*dealias

    E, E2 = np.exp(dt*linear), np.exp(dt*linear/2)
    contour_count = 32
    roots = np.exp(1j*np.pi*(np.arange(contour_count)+0.5)/contour_count)
    z = dt*linear[:, None] + roots[None, :]
    Q = dt*np.real(np.mean((np.exp(z/2)-1)/z, axis=1))
    f1 = dt*np.real(np.mean((-4-z+np.exp(z)*(4-3*z+z*z))/(z**3), axis=1))
    f2 = dt*np.real(np.mean((2+z+np.exp(z)*(-2+z))/(z**3), axis=1))
    f3 = dt*np.real(np.mean((-4-3*z-z*z+np.exp(z)*(4-z))/(z**3), axis=1))
    multiplier = -0.5j*derivative_wave

    def nonlinear(vhat):
        u = np.fft.ifft(vhat).real
        return multiplier*np.fft.fft(u*u)*dealias

    def advance(vhat):
        nv = nonlinear(vhat)
        a = E2*vhat+Q*nv
        na = nonlinear(a)
        b = E2*vhat+Q*na
        nb = nonlinear(b)
        c = E2*a+Q*(2*nb-nv)
        nc = nonlinear(c)
        vnext = E*vhat+nv*f1+2*(na+nb)*f2+nc*f3
        # Project round-off back onto the real-valued Fourier subspace. This
        # avoids exponentially amplifying tiny non-Hermitian FFT round-off
        # in long trajectories without changing the real PDE state.
        vnext = np.fft.fft(np.fft.ifft(vnext).real)*dealias
        if not np.isfinite(vnext).all():
            raise FloatingPointError('KS integration diverged; reduce dt or check resolution')
        return vnext

    for _ in range(warmup_steps):
        v = advance(v)
    out = np.empty((n_samples, n_grid), dtype=float)
    out[0] = np.fft.ifft(v).real
    for j in range(1, n_samples):
        for _ in range(sample_every):
            v = advance(v)
        out[j] = np.fft.ifft(v).real
    return out
