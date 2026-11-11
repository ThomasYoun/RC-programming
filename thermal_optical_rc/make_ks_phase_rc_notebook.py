"""Generate a scalar-observer Kuramoto–Sivashinsky phase-only RC notebook."""
import json
from pathlib import Path
import sys
import textwrap

cells = []
def md(text):
    cells.append({'cell_type': 'markdown', 'metadata': {}, 'source': textwrap.dedent(text).strip().splitlines(keepends=True)})
def code(text):
    cells.append({'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': textwrap.dedent(text).strip().splitlines(keepends=True)})

md(r'''
# KS field reconstruction with a phase-memory optical reservoir

This is a **single-sensor KS observer**: one scalar KS measurement drives the same phase-only reservoir as the previous notebook, and a multi-output ridge readout reconstructs the 63 unmeasured spatial positions. It also forecasts those positions several steps ahead while receiving the true scalar observation at every step. This is teacher-forced, not autonomous forecasting.

It is not a reproduction of Wang et al., [Optical next generation reservoir computing](https://www.nature.com/articles/s41377-025-01927-6): their KS observer uses seven measured spatial positions with explicitly delayed inputs, and their forecasting experiment uses full multivariate fields. Here the SLM probe is fixed, there is no concatenation or electronic feedback, and the only input-driven memory is the phase state. The work is an illustrative uncalibrated simulation, not a physical measurement.

Run this notebook in `thermal_optical_rc` with `ks_data.py`, `phase_only_rc.py`, `phase_only_optics.py`, and `optical_physical_memory.py`. It generates its own KS data with NumPy, PyTorch and Matplotlib. Runtime may benefit from setting BLAS threads to one.
''')
md(r'''
## 1. The KS field and what one sensor sees

We use the periodic one-dimensional Kuramoto–Sivashinsky equation
\[
v_t=-v v_x-v_{xx}-v_{xxxx},\qquad x\in[0,L),
\]
with \(L=22\), 64 grid points and time step 0.25, matching the basic geometry and sampling in the cited paper, but not its input representation or evaluation protocol. A Fourier exponential time differencing method (ETDRK4) generates a deterministic trajectory from a seeded initial condition; see [Kassam and Trefethen](https://doi.org/10.1137/S1064827502410633). The solver discards an initial transient before creating the data. Chaotic trajectories are sensitive to numerical error, so equality of long trajectories is not a solver-validation criterion.

We observe **only** location \(x_0\). The scalar signal is normalized using training-fit data and clipped to \([0,1]\), then used as the phase-screen drive \(u_k\). No measured KS spatial vector or past inputs are concatenated with the optical state. The other 63 locations are used as training targets and later held-out ground truth, not as optical inputs. A single spatial observation is deliberately much less information than in the paper's observer experiment.
''')
code('''
import numpy as np
import matplotlib.pyplot as plt
import torch
from ks_data import generate_ks
from phase_only_optics import PhaseOnlyOptics, radial_profile
from phase_only_rc import simulate_phase_rc, chronological_indices
from optical_physical_memory import RidgeReadout

torch.set_num_threads(1)
plt.rcParams['figure.dpi'] = 100
train_length, washout, test_length, validation_length = 2000, 200, 1000, 400
horizons = [0, 1, 4, 16]  # present reconstruction; forecasts 0.25, 1, 4 model-time units
dt_ks, length, n_grid, sensor_index = .25, 22., 64, 0
ks = generate_ks(n_samples=train_length+test_length+max(horizons),
                 n_grid=n_grid, domain_length=length, dt=dt_ks,
                 sample_every=1, warmup_steps=1000, seed=2026)
assert ks.shape == (train_length+test_length+max(horizons), n_grid)
observed = ks[:train_length+test_length, sensor_index]
lo, hi = observed[:train_length-validation_length].min(), observed[:train_length-validation_length].max()
if hi <= lo: raise ValueError('training sensor is constant')
raw_u = (observed-lo)/(hi-lo)
u = np.clip(raw_u, 0, 1)
unseen = np.arange(n_grid) != sensor_index
print(f'KS trajectory: {ks.shape}; L={length:g}, grid={n_grid}, model dt={dt_ks:g}')
print(f'Optical input: 1 measured position x={sensor_index*length/n_grid:g}; {unseen.sum()} unobserved positions')
print(f'Test sensor clipping fraction: {np.mean(raw_u[2000:3000] != u[2000:3000]):.2%}')

fig, axes = plt.subplots(2, 1, figsize=(11, 6), constrained_layout=True)
limit = np.max(np.abs(ks))
im = axes[0].imshow(ks.T, origin='lower', aspect='auto', cmap='RdBu_r',
    extent=(0, len(ks)*dt_ks, 0, length), vmin=-limit, vmax=limit)
axes[0].axhline(sensor_index*length/n_grid, color='black', ls='--', lw=.8, label='input sensor')
axes[0].set(xlabel='KS time', ylabel='position x', title='Generated KS field: entire trajectory')
fig.colorbar(im, ax=axes[0], label='v(x,t)')
axes[1].plot(np.arange(len(u)), u, lw=.8)
for boundary in [200, 1600, 2000]: axes[1].axvline(boundary, color='gray', ls='--', lw=.8)
axes[1].set(xlabel='input step', ylabel='normalized sensor u[k]', title='Single scalar input (dashed lines: washout, validation, test)')
plt.show()
''')
md(r'''
## 2. Physical reservoir and matched controls

At each input step, a random initial **unwrapped phase map** decays while the measurement adds a direct phase increment:
\[
\Phi_{k+1}=e^{-1/\tau}\Phi_k+2\pi u_kG,\qquad
r_{k+1}=\operatorname{downsample}\!\left|O_2B\,\mathrm{diag}(e^{i\Phi_{k+1}})AO_1E_{\mathrm{SLM}}\right|^2.
\]
The phase-screen profile is the existing regularized inverse-radius profile with 80% support. \(\tau\) is measured in input samples; because one sample is 0.25 KS time units, its corresponding KS-time decay constant is \(0.25\tau\) if these clocks are identified. For fixed \(\beta=2\pi\), changing \(\tau\) also changes steady-state phase amplitude, not just memory duration. Same scattering, pupil, phase map and noise seed are used across conditions; the ideal comparator at \(\tau=10\) has no camera noise. Noise uses raw-intensity Poisson photoelectron counts and Gaussian read noise, before 16×16 camera downsampling.

At 200 washout steps, the initial phase contribution has fallen to \(e^{-200/\tau}\); for \(\tau=100\) that is still 13.5%. The requested split is kept for comparability, so early training may retain the arbitrary initial condition. For a contribution ten steps old, the corresponding factors are 0.67%, 13.5%, 36.8%, 81.9% and 90.5% for \(\tau=2,5,10,50,100\), respectively. These phase contributions are not the same as separately recoverable input information, and camera speckle is not equal to \(u_k\).
''')
code('''
shape, camera_shape = (128, 128), (16, 16)
G = radial_profile(shape, core_radius=.12, support_fraction=.8)
initial = np.random.default_rng(2026).uniform(0, 6*np.pi, shape)
optics = PhaseOnlyOptics(seed=42, optical_shape=shape, pupil_radius=(.25, .25))
cases = [('Static, noisy', 'static', 2, True), ('Instant, noisy', 'instant', 2, True),
         ('Fading τ=2, noisy', 'memory', 2, True), ('Fading τ=5, noisy', 'memory', 5, True),
         ('Fading τ=10, noisy', 'memory', 10, True), ('Fading τ=50, noisy', 'memory', 50, True),
         ('Fading τ=100, noisy', 'memory', 100, True), ('Fading τ=10, ideal', 'memory', 10, False)]
features = {}
for label, mode, tau, noisy in cases:
    X, phase = simulate_phase_rc(u, optics, G, initial, tau=tau, beta=2*np.pi, mode=mode,
        camera_shape=camera_shape, photon_scale=5000 if noisy else None,
        read_noise_electrons=3.0 if noisy else 0, noise_seed=123)
    features[label] = X
    print(f'{label:<22} features={X.shape}, initial-phase residual after washout={np.exp(-200/tau):.4f}' if mode == 'memory' else f'{label:<22} features={X.shape}')
''')
md(r'''
## 3. Reconstruction and forecast protocol

For array index \(k\), the reservoir sees \(u_k\) and its post-write feature is `features[k]`. It predicts the KS field at \(k+h\): \(h=0\) is current-state **reconstruction**, and \(h=1,4,16\) are forecasts by 0.25, 1 and 4 KS time units. The targets are the 63 unmeasured locations. A separate multi-output readout and ridge strength are fitted for each case and horizon.

The first 2,000 input steps form the training segment, including 200 washout; the last 400 training steps are validation. We exclude training examples whose targets cross a validation or test boundary. The last 1,000 input steps are test, with features driven by true sensor values throughout and phase carried across the boundary. Feature scaling and ridge hyperparameter selection use training data only.

**Same-information controls:** training-mean field, direct affine scalar-to-field readout, instant phase, and frozen static phase. A full-field persistence result is also printed as an **oracle reference only**: it is given the entire current ground-truth field, so it has far more information than the single-sensor optical model.

Field NRMSE is \(\sqrt{\sum_{k,x}(\hat v-v)^2 / \sum_{k,x}(v-\bar v_x)^2}\), where \(\bar v_x\) is the retrospective test mean at each unmeasured location. Thus one corresponds to this test-mean reference; below one is better. The associated \(R^2=1-\mathrm{NRMSE}^2\). The test mean is used **only** to score, never for fitting or normalization.
''')
code('''
def field_nrmse(true, pred):
    true, pred = np.asarray(true), np.asarray(pred)
    if true.shape != pred.shape: raise ValueError('field shapes must match')
    denominator = np.sum((true-true.mean(axis=0, keepdims=True))**2)
    return np.sqrt(np.sum((true-pred)**2)/denominator)

def select_and_fit(X, Y, split):
    alphas = [.1, 1., 10., 100., 1000.]
    valid_scores = [field_nrmse(Y[split['valid']], RidgeReadout(a).fit(X[split['fit']], Y[split['fit']]).predict(X[split['valid']])) for a in alphas]
    best = alphas[int(np.argmin(valid_scores))]
    return RidgeReadout(best).fit(X[split['refit']], Y[split['refit']]), best

scores, predictions, selected_alphas = {}, {}, {}
for h in horizons:
    split = chronological_indices(train_length=train_length, washout=washout,
        validation_length=validation_length, test_length=test_length, horizon=h)
    indices = np.arange(train_length+test_length)
    Y = ks[indices+h][:, unseen]
    te = split['test']
    scores[h], predictions[h], selected_alphas[h] = {}, {}, {}
    mean_pred = np.broadcast_to(Y[split['refit']].mean(axis=0), Y[te].shape)
    scores[h]['Training mean'] = field_nrmse(Y[te], mean_pred)
    scalar_model, alpha = select_and_fit(u[:, None], Y, split)
    scalar_pred = scalar_model.predict(u[te, None])
    scores[h]['Scalar affine'] = field_nrmse(Y[te], scalar_pred)
    predictions[h]['Scalar affine'] = scalar_pred
    if h:
        scores[h]['Full-field persistence (oracle)'] = field_nrmse(Y[te], ks[te][:, unseen])
    for label, _, _, _ in cases:
        model, alpha = select_and_fit(features[label], Y, split)
        pred = model.predict(features[label][te])
        predictions[h][label] = pred
        selected_alphas[h][label] = alpha
        scores[h][label] = field_nrmse(Y[te], pred)

print(f"{'Condition':<31}" + ''.join(f'{f"h={h}":>11}' for h in horizons))
labels = ['Training mean', 'Scalar affine', 'Full-field persistence (oracle)'] + [case[0] for case in cases]
for label in labels:
    print(f'{label:<31}' + ''.join(f'{scores[h].get(label, np.nan):>11.4f}' for h in horizons))
print('h=0 is field reconstruction; h>0 is teacher-forced forecasting. Lower is better.')
''')
md(r'''
## 4. Visualize the 63 unmeasured positions during the test segment

The same fixed color range is used for all panels. Time axis uses test steps 0–999; \(h=0\) is reconstruction, and \(h=1\) is one-step forecast. Figures are not individually normalized. More panels and different \(\tau\) can be selected by editing `show_labels` below.
''')
code('''
h = 0  # change to 1, 4, or 16 to inspect teacher-forced forecasts
te = np.arange(train_length, train_length+test_length)
truth = ks[te+h][:, unseen]
show_labels = ['Scalar affine', 'Instant, noisy', 'Fading τ=2, noisy', 'Fading τ=5, noisy',
               'Fading τ=10, noisy', 'Fading τ=50, noisy', 'Fading τ=100, noisy']
all_fields = [truth] + [predictions[h][label] for label in show_labels]
limit = max(np.max(np.abs(a)) for a in all_fields)
fig, axes = plt.subplots(len(all_fields), 1, figsize=(13, 2.2*len(all_fields)), sharex=True, constrained_layout=True)
for ax, array, label in zip(axes, all_fields, ['Ground truth']+show_labels):
    im = ax.imshow(array.T, origin='lower', aspect='auto', cmap='RdBu_r', vmin=-limit, vmax=limit,
                   extent=(0, len(te), 1, n_grid-1))
    score = '' if label == 'Ground truth' else f' (NRMSE={scores[h][label]:.3f})'
    ax.set(title=label+score, ylabel='unmeasured grid index')
axes[-1].set_xlabel('test time step (0–1000)')
fig.colorbar(im, ax=axes, label='KS field v', shrink=.7); plt.show()

fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
for label in ['Scalar affine', 'Instant, noisy', 'Fading τ=2, noisy', 'Fading τ=5, noisy',
              'Fading τ=10, noisy', 'Fading τ=50, noisy', 'Fading τ=100, noisy']:
    ax.plot(horizons, [scores[h][label] for h in horizons], 'o-', label=label)
ax.axhline(1, color='gray', lw=.8, ls='--', label='test-mean reference')
ax.set(xlabel='forecast horizon h (steps)', ylabel='unobserved-field NRMSE', xticks=horizons)
ax.grid(alpha=.2); ax.legend(fontsize=8, ncol=2); plt.show()
''')
md(r'''
## 5. Limitations and interpretation

Reconstruction from one scalar sensor is underdetermined in general; it may depend on correlations learned from this specific trajectory. The scalar-driven phase has only one driven memory coordinate after the initial transient, no matter how many pixels the camera has. Thus temporal fading, optical nonlinearity and 256 speckle features do not automatically reconstruct a 64-dimensional KS state. Teacher forcing uses the true scalar throughout testing and is not an autonomous PDE solver.

Use the direct scalar baseline and instant condition to see whether fading provides an improvement with matched input information. Do not compare the oracle full-field persistence as if it had the same information. A single realization and seed does not establish robustness; test independent KS and optical/noise seeds, time-step convergence, sensor positions and phase/noise parameters. No calibrated temperature evolution is included. To reproduce the paper's multi-sensor observer or full-field forecast, a separately designed multivariate optical encoder or parallel phase states, and the paper's explicit-delay protocol, would be required.
''')

notebook = {'cells': cells, 'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}, 'language_info': {'name': 'python', 'version': '3'}}, 'nbformat': 4, 'nbformat_minor': 4}
output = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('ks_phase_memory_rc.ipynb')
output.write_text(json.dumps(notebook, ensure_ascii=False, indent=1)+'\n', encoding='utf-8')
