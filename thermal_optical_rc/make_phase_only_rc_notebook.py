"""Generate the input-driven, phase-only optical RC benchmark notebook."""
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
# Reservoir computing with an input-driven phase screen

The phase-only model is tested on teacher-forced and free-running Mackey–Glass prediction and independent-input delay reconstruction. There is no temperature calculation or CCD-speckle-to-SLM state feedback. The SLM probe, objective pupils, and scattering operators remain fixed; input drives the phase screen directly. The free-running benchmark intentionally feeds the digital readout's prediction back as the next driving input; it is a closed-loop numerical experiment, not an autonomous feedback-free physical device. This is a simulation, not a calibrated thermo-optic experiment.

Run this notebook from the `thermal_optical_rc` directory, with `phase_only_rc.py`, `phase_only_optics.py`, `optical_physical_memory.py`, and `MackeyGlass_t17.txt` alongside it. NumPy, PyTorch and Matplotlib are required. For reproducible runtimes, limiting BLAS threads may help.
''')
md(r'''
## 1. Causal optical model and comparisons

At time \(k\), input \(u_k\in[0,1]\) writes to an **unwrapped** phase map and the resulting speckle is observed:
\[
\Phi_{k+1}=a\Phi_k+\beta u_k G,\quad a=e^{-1/\tau},\qquad
r_{k+1}=\operatorname{downsample}\left|O_2 B\operatorname{diag}(e^{i\Phi_{k+1}})A O_1E_{\rm SLM}\right|^2.
\]
For clarity, array `features[k]` stores \(r_{k+1}\), the image **after** writing \(u_k\). Its one-step prediction target is \(u_{k+1}\). This is the same causal update as the phase-verification notebook, shifted to index each camera frame by its most recent written input. We compare:

* **Static:** the same initial random phase is frozen and independent of input.
* **Instant:** \(\Phi_{k+1}=2\pi u_kG\); no retained input history.
* **Fading:** the displayed values of \(\tau\), with **fixed \(\beta=2\pi\)** so that an input in \([0,1]\) writes an increment in \([0,2\pi]\) radians at the peak of \(G\). At any pixel the increment is \(2\pi u_kG(x,y)\).

All conditions use the same initial unwrapped random map, scattering realization, SLM probe and radial \(G\) (80% support). Instant mode deliberately overwrites the initial phase on its first write. The random initial map is sampled from \(U[0,6\pi)\), treated as an unwrapped state, and decays in the fading cases. The \(6\pi\) upper limit is adjustable. Camera shot and read noise are applied before downsampling; the ideal condition omits both. Pupil aberrations and SLM jitter are not included. All values are model assumptions, not calibrated parameters.

With the same per-step \(\beta\), changing \(\tau\) also changes the steady-state response: the peak driven phase at constant input is \(\beta u/(1-a)\). Thus these comparisons change both memory duration and steady-state phase magnitude; they do not isolate \(\tau\) alone. Unwrapped phase is never capped at \(2\pi\), even though optical phase is periodic. We compare \(\tau=2,5,10,50,100\) with the same 200-step washout. This washout is intentionally held fixed for comparison, **not** assumed sufficient for every \(\tau\).

Only \(W_{\rm out}\) is trained by ridge regression. The camera features are standardized using the fit segment only; the bias is unregularized. Despite many CCD pixels, the long-term phase dynamics have only **one driven scalar memory** because the spatial write profile is fixed.
''')
code('''
import numpy as np
import matplotlib.pyplot as plt
import torch
from pathlib import Path
from phase_only_optics import PhaseOnlyOptics, radial_profile
from phase_only_rc import simulate_phase_rc, free_phase_rollout, chronological_indices, r2_score
from optical_physical_memory import RidgeReadout, nrmse

torch.set_num_threads(1)
plt.rcParams['figure.dpi'] = 100
shape, camera_shape = (128, 128), (16, 16)
G = radial_profile(shape, core_radius=0.12, support_fraction=0.8)
initial_phase_max = 6*np.pi  # adjustable upper limit of the unwrapped initial state
initial = np.random.default_rng(2026).uniform(0, initial_phase_max, shape)
optics = PhaseOnlyOptics(seed=42, optical_shape=shape, pupil_radius=(0.25, 0.25))
phase_increment_max = 2*np.pi  # u ∈ [0,1] maps to a peak write of 0–2π rad
cases = [('Static, noisy', 'static', 2, True),
         ('Instant, noisy', 'instant', 2, True),
         ('Fading τ=2, noisy', 'memory', 2, True),
         ('Fading τ=5, noisy', 'memory', 5, True),
         ('Fading τ=10, noisy', 'memory', 10, True),
         ('Fading τ=50, noisy', 'memory', 50, True),
         ('Fading τ=100, noisy', 'memory', 100, True),
         ('Fading τ=10, ideal', 'memory', 10, False)]
def features_for(inputs, case, return_phase=False):
    label, mode, tau, noisy = case
    beta = phase_increment_max
    features, phase = simulate_phase_rc(inputs, optics, G, initial, tau, beta, mode=mode,
        camera_shape=camera_shape, photon_scale=5000 if noisy else None,
        read_noise_electrons=3.0 if noisy else 0, noise_seed=123)
    return (features, phase) if return_phase else features

def fit_validated(features, target, fit_idx, valid_idx, refit_idx):
    alphas = [0.1, 1.0, 10.0, 100.0, 1000.0]
    validation = [r2_score(target[valid_idx], RidgeReadout(a).fit(features[fit_idx], target[fit_idx]).predict(features[valid_idx])) for a in alphas]
    best = alphas[int(np.argmax(validation))]
    return RidgeReadout(best).fit(features[refit_idx], target[refit_idx]), best

print(f'Phase support: {np.count_nonzero(G)/G.size:.2%}; max increment: {phase_increment_max/np.pi:.2f}π rad')
print(f'Initial unwrapped phase range: [0, {initial_phase_max/np.pi:g}π) rad')
print(f'CCD features: {np.prod(camera_shape)}; objective pupils and scattering fixed')
''')
md(r'''
### Initial-state persistence and phase build-up

After \(n\) writes, the initial phase component is multiplied by \(e^{-n/\tau}\). This is distinct from the phase accumulated from inputs. The table reports the residual after the 200-step washout and at the test boundary, plus the approximate number of steps required to reduce it below 1%. For \(\tau=100\), the threshold takes 461 steps, so the 200-step washout leaves a substantial initial-phase contribution in early training. We retain the requested split; later plots expose phase drift rather than silently changing the protocol.
''')
code('''
tau_values = [2, 5, 10, 50, 100]
print(f"{'τ':>6}{'after 200':>15}{'after 2000':>15}{'steps to <1%':>17}{'steady peak/π at u=1':>26}")
for tau in tau_values:
    # Strict inequality: n must exceed tau*log(100), even if it is an integer.
    steps_1pct = int(np.floor(tau*np.log(100)))+1
    print(f'{tau:>6}{np.exp(-200/tau):>15.3%}{np.exp(-2000/tau):>15.3%}{steps_1pct:>17}{(2/(1-np.exp(-1/tau))):>26.1f}')
''')
md(r'''
## 2. Mackey–Glass one-step prediction

The first 2,000 input steps are the training segment, including 200 washout steps. The final 400 training steps provide chronological validation for ridge strength; the chosen readout is then refit on all post-washout training data whose targets lie in the training segment. The last 1,000 input steps are the test segment. There is no state reset between training and test.

Normalization uses only the fitting portion of the training segment. Inputs are clipped to \([0,1]\); target values are normalized but not clipped. Training predictions with targets crossing the validation or test boundary are excluded. The test receives the true inputs sequentially: this is **teacher-forced one-step prediction**, not autonomous forecasting. The persistence baseline predicts \(u_{k+1}\) from the current true value \(u_k\); because Mackey–Glass inputs are correlated, a static current-input mapping may perform well without memory. Here the instant phase control provides that comparison.
''')
code('''
splits = chronological_indices()
series = np.loadtxt(Path('MackeyGlass_t17.txt')).reshape(-1)
assert len(series) >= 3001
lo, hi = series[:1600].min(), series[:1600].max()
if hi <= lo: raise ValueError('training input is constant')
raw = (series[:3001]-lo)/(hi-lo)
u, target = np.clip(raw[:3000], 0, 1), raw[1:3001]
te = splits['test']
print(f"Train segment=2000, washout=200, refit={len(splits['refit'])}, validation={len(splits['valid'])}, test={len(te)}")
print(f'Test relative steps: 0–{len(te)-1}; target indices: {te[0]+1}–{te[-1]+1}')
print(f'Test input clipping fraction: {np.mean(u[te] != raw[te]):.4f}')
print(f'Persistence baseline NRMSE: {nrmse(target[te], u[te]):.4f}')
mg_predictions, mg_scores, mg_phases, mg_models = {}, {}, {}, {}
for case in cases:
    label = case[0]
    X, mg_phases[label] = features_for(u, case, return_phase=True)
    model, alpha = fit_validated(X, target, splits['fit'], splits['valid'], splits['refit'])
    mg_models[label] = model
    pred = model.predict(X[te])
    mg_predictions[label] = pred
    mg_scores[label] = (nrmse(target[te], pred), r2_score(target[te], pred), alpha)
    print(f'{label:<22} NRMSE={mg_scores[label][0]:.4f}  R²={mg_scores[label][1]: .4f}  ridge={alpha:g}')
''')
code('''
nrows = (len(cases)+1)//2
fig, axes = plt.subplots(nrows, 2, figsize=(14, 2.8*nrows), sharex=True, constrained_layout=True, squeeze=False)
for ax, case in zip(axes.flat, cases):
    label = case[0]
    ax.plot(np.arange(len(te)), target[te], color='black', lw=.9, label='target')
    ax.plot(np.arange(len(te)), mg_predictions[label], lw=.8, label='prediction')
    ax.set_title(f'{label}: NRMSE={mg_scores[label][0]:.3f}', fontsize=9)
    ax.set_xlim(0, 1000); ax.grid(alpha=.15); ax.legend(fontsize=7)
for ax in axes[-1]: ax.set_xlabel('test time step (0–1000)')
for ax in list(axes.flat)[len(cases):]: ax.set_visible(False)
plt.show()

fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
for tau in tau_values:
    label = f'Fading τ={tau}, noisy'
    ax.plot(np.arange(len(u)), mg_phases[label]/np.pi, label=f'τ={tau}')
for boundary, label in [(200, 'washout ends'), (1600, 'validation starts'), (2000, 'test starts')]:
    ax.axvline(boundary, color='gray', ls='--', lw=.8)
    ax.text(boundary, .98, label, rotation=90, va='top', ha='right', transform=ax.get_xaxis_transform(), fontsize=8)
ax.set(xlabel='input step k', ylabel='unwrapped phase at profile peak / π', title='Phase trajectory: initial decay plus input-driven accumulation')
ax.grid(alpha=.2); ax.legend(ncol=3); plt.show()
''')
md(r'''
## 3. Free-running Mackey–Glass prediction

The **same** readout, physical parameters, fixed optical realization, camera-noise settings and training-only scaling as above are used; no weights are retrained or selected on the test set. First the true training inputs \(u_0,\ldots,u_{1999}\) drive the phase memory and camera, including the 200-step washout. The last training feature predicts the first test value \(\hat u_{2000}\). Its output is then used as the next external drive, producing the feature for \(\hat u_{2001}\), and so on for 1,000 test values.

This section intentionally introduces **digital output-to-input feedback** to test free prediction: previous \(W_{\mathrm{out}}\) predictions, not the preceding camera speckle, drive the phase screen. This feedback is distinct from the phase memory itself. It is not the feedback-free physical-memory architecture used for teacher forcing, and it does not demonstrate a physically autonomous device. After the training boundary no true test input is supplied, and the phase state and camera-noise random stream continue from the training drive.

The optical model accepts inputs in \([0,1]\), so each predicted value used as feedback is clipped to that range; the reported predictions and error use the **unclipped readout outputs**, while `clip fraction` counts how often feedback required clipping. Clipping affects the closed-loop dynamics. The true target is the normalized, unclipped series at indices 2000–2999. A free-running persistence baseline remains constant at the last observed training input, rather than using future true inputs. The original teacher-forced scores use target indices 2001–3000; the final two columns below compare both modes over their common target indices 2001–2999.

The model retains fixed \(\beta=2\pi\), and the initial phase continues across the boundary; for \(\tau=100\), a 200-step washout still leaves a substantial initial contribution early in training. Strong nonlinearity, phase wrapping, camera noise and clipping can accumulate error during the free rollout. No claim is made about long-term climate reconstruction from a single trajectory.
''')
code('''
free_target = raw[2000:3000]
assert len(free_target) == 1000
free_predictions, free_rollouts, free_scores = {}, {}, {}
free_persistence = np.full(len(free_target), u[1999])
print(f'Free target indices: 2000–2999; last true drive index: 1999; predictions: {len(free_target)}')
print(f'Free constant-persistence baseline NRMSE: {nrmse(free_target, free_persistence):.4f}')
print(f"{'Condition':<22}{'Free NRMSE':>13}{'Free R²':>11}{'Clip fraction':>16}{'Free common':>14}{'Teacher common':>17}")
for label, mode, tau, noisy in cases:
    result = free_phase_rollout(u[:2000], mg_models[label], optics, G, initial,
        tau=tau, beta=phase_increment_max, mode=mode, camera_shape=camera_shape,
        photon_scale=5000 if noisy else None,
        read_noise_electrons=3.0 if noisy else 0, noise_seed=123,
        n_steps=len(free_target))
    prediction = result['predictions']
    assert prediction.shape == free_target.shape
    free_predictions[label] = prediction
    free_rollouts[label] = result
    free_scores[label] = (nrmse(free_target, prediction), r2_score(free_target, prediction))
    common_target = raw[2001:3000]
    free_common = nrmse(common_target, prediction[1:])
    teacher_common = nrmse(common_target, mg_predictions[label][:-1])
    print(f"{label:<22}{free_scores[label][0]:>13.4f}{free_scores[label][1]:>11.4f}{result['clipping_fraction']:>16.2%}{free_common:>14.4f}{teacher_common:>17.4f}")
''')
code('''
test_steps = np.arange(len(free_target))
nrows = (len(cases)+1)//2
for view, stop in [('full test', len(free_target)), ('first 200 steps', 200)]:
    fig, axes = plt.subplots(nrows, 2, figsize=(14, 2.8*nrows), sharex=True, constrained_layout=True, squeeze=False)
    for ax, case in zip(axes.flat, cases):
        label = case[0]
        ax.plot(test_steps[:stop], free_target[:stop], color='black', lw=.9, label='target')
        ax.plot(test_steps[:stop], free_predictions[label][:stop], lw=.8, label='free prediction')
        ax.set_title(f'{label}: full-test NRMSE={free_scores[label][0]:.3f}; {view}', fontsize=9)
        ax.set_xlim(0, stop); ax.grid(alpha=.15); ax.legend(fontsize=7)
    for ax in axes[-1]: ax.set_xlabel('test time step')
    for ax in list(axes.flat)[len(cases):]: ax.set_visible(False)
    plt.show()
''')
md(r'''
## 4. Independent-input delay reconstruction

With independent uniform inputs, a memoryless current-input mapping cannot systematically reconstruct an earlier input. At frame `features[k]`, reconstruct \(u_{k-d}\) for delays up to 500 steps. Here \(d=0\) is the input just written; \(d\ge1\) probes past inputs retained by the physical phase dynamics. For \(d>200\), fitting begins at index \(d\) so every training target exists; all other splits remain fixed. Thus the longest delays have fewer training examples. A different ridge strength is selected per condition and delay, on training-segment validation only. The first 200 samples are washout, and evaluation uses the last 1,000 samples. Positive held-out \(R^2\) indicates performance beyond the retrospective test-mean baseline; negative values are possible. These sparse scores are not a total memory-capacity estimate.
''')
code('''
random_u = np.random.default_rng(213).uniform(0, 1, 3000)
delay_values = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 500]
fit = np.arange(200, 1600); valid = np.arange(1600, 2000)
refit = np.arange(200, 2000); test = np.arange(2000, 3000)
memory_scores, nonlinear_scores = {}, {}
random_x = 2*random_u-1
previous_x = np.r_[0.0, random_x[:-1]]  # index 0 is discarded by washout
nonlinear_targets = {'current P2': (3*random_x**2-1)/2,
                     'previous P2': (3*previous_x**2-1)/2,
                     'current × previous': random_x*previous_x}
print(f"{'Condition':<22}" + ''.join(f'{f"d={d}":>9}' for d in delay_values))
for case in cases:
    X = features_for(random_u, case)
    scores = []
    for d in delay_values:
        history_target = np.zeros_like(random_u)
        history_target[d:] = random_u[:len(random_u)-d] if d else random_u
        fit_d = np.arange(max(200, d), 1600)
        refit_d = np.arange(max(200, d), 2000)
        model, alpha = fit_validated(X, history_target, fit_d, valid, refit_d)
        scores.append(r2_score(history_target[test], model.predict(X[test])))
    memory_scores[case[0]] = scores
    nonlinear_scores[case[0]] = []
    for nonlinear_target in nonlinear_targets.values():
        model, alpha = fit_validated(X, nonlinear_target, fit, valid, refit)
        nonlinear_scores[case[0]].append(r2_score(nonlinear_target[test], model.predict(X[test])))
    print(f'{case[0]:<22}' + ''.join(f'{v:>9.3f}' for v in scores))
fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
for label, scores in memory_scores.items(): ax.plot(delay_values, scores, 'o-', label=label)
ax.axhline(0, color='gray', lw=.8); ax.set(xlabel='input delay d (symlog scale)', ylabel='held-out R²')
ax.set_xscale('symlog', linthresh=1)
ax.grid(alpha=.25); ax.legend(fontsize=8, ncol=2); plt.show()
''')
md(r'''
## 5. Nonlinear readout tasks

Using the same independent-input frames and train/validation/test split, test the second-order function \(P_2(x)=(3x^2-1)/2\) for the just-written input and for the preceding input, as well as their product \(x_kx_{k-1}\), where \(x_k=2u_k-1\). Each task uses its own validated ridge readout. The input-only task tests nonlinear optical encoding without requiring memory; the delayed and product tasks require information about past input. These scores are computed from the same simulation frames as the previous section.
''')
code('''
print(f"{'Condition':<22}" + ''.join(f'{name:>21}' for name in nonlinear_targets))
for label, scores in nonlinear_scores.items():
    print(f'{label:<22}' + ''.join(f'{value:>21.3f}' for value in scores))
''')
md(r'''
## 6. Interpretation and limitations

The random initial phase is a transient, not input-driven long-term memory. With \(\tau=100\), the original phase remains substantial after the fixed 200-step washout, though it decays almost entirely before the test segment. The input-driven part still has only one dynamical scalar coefficient of \(G\). A high Mackey–Glass score alone does not establish useful fading memory: compare with instant, persistence, and the independent-input task. Phase periodicity, large fixed-increment accumulation, nonstationary initial transients, camera noise and ridge regularization all affect recoverability. \(G\), \(\tau\), gain, camera noise, and random seeds here are illustrative. Test multiple independent optical realizations, initial maps, input/noise seeds and parameters before claiming robustness. No temperature dynamics or experimentally calibrated index changes are represented.
''')

notebook = {'cells': cells, 'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}, 'language_info': {'name': 'python', 'version': '3'}}, 'nbformat': 4, 'nbformat_minor': 4}
output = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('phase_only_rc_tests.ipynb')
output.write_text(json.dumps(notebook, ensure_ascii=False, indent=1)+'\n', encoding='utf-8')
