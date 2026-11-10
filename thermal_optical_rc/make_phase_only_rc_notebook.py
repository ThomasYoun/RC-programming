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

The phase-only model is now tested on one-step Mackey–Glass prediction and independent-input delay reconstruction. There is no temperature calculation, CCD-to-SLM feedback, or electronic reservoir recurrence. The SLM probe, objective pupils, and scattering operators remain fixed; input drives the phase screen directly. This is a simulation, not a calibrated thermo-optic experiment.

Run this notebook from the `thermal_optical_rc` directory, with `phase_only_rc.py`, `phase_only_optics.py`, `optical_physical_memory.py`, and `MackeyGlass_t17.txt` alongside it. NumPy, PyTorch and Matplotlib are required. For reproducible runtimes, limiting BLAS threads may help.
''')
md(r'''
## 1. Causal optical model and comparisons

At time \(k\), input \(u_k\) writes to an **unwrapped** phase map and the resulting speckle is observed:
\[
\Phi_{k+1}=a\Phi_k+\beta u_k G,\quad a=e^{-1/\tau},\qquad
r_{k+1}=\operatorname{downsample}\left|O_2 B\operatorname{diag}(e^{i\Phi_{k+1}})A O_1E_{\rm SLM}\right|^2.
\]
For clarity, array `features[k]` stores \(r_{k+1}\), the image **after** writing \(u_k\). Its one-step prediction target is \(u_{k+1}\). This is the same causal update as the phase-verification notebook, shifted to index each camera frame by its most recent written input. We compare:

* **Static:** the same initial random phase is frozen and independent of input.
* **Instant:** \(\Phi_{k+1}=g u_kG\); no retained input history.
* **Fading:** the displayed values of \(\tau\), with \(\beta=(1-a)g\). This holds the constant-input steady-state gain \(g\) fixed across \(\tau\); it is different from the fixed \(\beta=0.6\pi\) example in the phase verification notebook.

All conditions use the same initial unwrapped random map, scattering realization, SLM probe and radial \(G\) (80% support). Instant mode deliberately overwrites the initial phase on its first write. The random initial map is sampled from \(U[0,2\pi)\), treated as an unwrapped physical state, and decays in the fading cases. Camera shot and read noise are applied before downsampling; the ideal condition omits both. Pupil aberrations and SLM jitter are not included in this phase-only benchmark. All values are model assumptions, not calibrated parameters.

Only \(W_{\rm out}\) is trained by ridge regression. The camera features are standardized using the fit segment only; the bias is unregularized. Despite many CCD pixels, the long-term phase dynamics have only **one driven scalar memory** because the spatial write profile is fixed.
''')
code('''
import numpy as np
import matplotlib.pyplot as plt
import torch
from pathlib import Path
from phase_only_optics import PhaseOnlyOptics, radial_profile
from phase_only_rc import simulate_phase_rc, chronological_indices, r2_score
from optical_physical_memory import RidgeReadout, nrmse

torch.set_num_threads(1)
plt.rcParams['figure.dpi'] = 100
shape, camera_shape = (128, 128), (16, 16)
G = radial_profile(shape, core_radius=0.12, support_fraction=0.8)
initial = np.random.default_rng(2026).uniform(0, 2*np.pi, shape)
optics = PhaseOnlyOptics(seed=42, optical_shape=shape, pupil_radius=(0.25, 0.25))
phase_gain = 2*np.pi
cases = [('Static, noisy', 'static', 8, True),
         ('Instant, noisy', 'instant', 8, True),
         ('Fading τ=2, noisy', 'memory', 2, True),
         ('Fading τ=8, noisy', 'memory', 8, True),
         ('Fading τ=32, noisy', 'memory', 32, True),
         ('Fading τ=8, ideal', 'memory', 8, False)]
def features_for(inputs, case):
    label, mode, tau, noisy = case
    beta = (1-np.exp(-1/tau))*phase_gain if mode == 'memory' else phase_gain
    features, _ = simulate_phase_rc(inputs, optics, G, initial, tau, beta, mode=mode,
        camera_shape=camera_shape, photon_scale=5000 if noisy else None,
        read_noise_electrons=3.0 if noisy else 0, noise_seed=123)
    return features

def fit_validated(features, target, fit_idx, valid_idx, refit_idx):
    alphas = [0.1, 1.0, 10.0, 100.0, 1000.0]
    validation = [r2_score(target[valid_idx], RidgeReadout(a).fit(features[fit_idx], target[fit_idx]).predict(features[valid_idx])) for a in alphas]
    best = alphas[int(np.argmax(validation))]
    return RidgeReadout(best).fit(features[refit_idx], target[refit_idx]), best

print(f'Phase support: {np.count_nonzero(G)/G.size:.2%}; fixed-input gain: {phase_gain/np.pi:.2f}π rad')
print(f'CCD features: {np.prod(camera_shape)}; objective pupils and scattering fixed')
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
mg_predictions, mg_scores = {}, {}
for case in cases:
    label = case[0]
    X = features_for(u, case)
    model, alpha = fit_validated(X, target, splits['fit'], splits['valid'], splits['refit'])
    pred = model.predict(X[te])
    mg_predictions[label] = pred
    mg_scores[label] = (nrmse(target[te], pred), r2_score(target[te], pred), alpha)
    print(f'{label:<22} NRMSE={mg_scores[label][0]:.4f}  R²={mg_scores[label][1]: .4f}  ridge={alpha:g}')
''')
code('''
fig, axes = plt.subplots(3, 2, figsize=(14, 9), sharex=True, constrained_layout=True)
for ax, case in zip(axes.flat, cases):
    label = case[0]
    ax.plot(np.arange(len(te)), target[te], color='black', lw=.9, label='target')
    ax.plot(np.arange(len(te)), mg_predictions[label], lw=.8, label='prediction')
    ax.set_title(f'{label}: NRMSE={mg_scores[label][0]:.3f}', fontsize=9)
    ax.set_xlim(0, 1000); ax.grid(alpha=.15); ax.legend(fontsize=7)
for ax in axes[-1]: ax.set_xlabel('test time step (0–1000)')
plt.show()
''')
md(r'''
## 3. Independent-input delay reconstruction

With independent uniform inputs, a memoryless current-input mapping cannot systematically reconstruct an earlier input. At frame `features[k]`, reconstruct \(u_{k-d}\) for \(d=0,1,2,4,8,16\). Here \(d=0\) is the input just written; \(d\ge1\) probes past inputs retained by the physical phase dynamics. A different ridge strength is selected per condition and delay, on training-segment validation only. The first 200 samples are washout, and evaluation uses the last 1,000 samples. Positive held-out \(R^2\) indicates performance beyond the retrospective test-mean baseline; negative values are possible. These sparse scores are not a total memory-capacity estimate.
''')
code('''
random_u = np.random.default_rng(213).uniform(0, 1, 3000)
delay_values = [0, 1, 2, 4, 8, 16]
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
        model, alpha = fit_validated(X, history_target, fit, valid, refit)
        scores.append(r2_score(history_target[test], model.predict(X[test])))
    memory_scores[case[0]] = scores
    nonlinear_scores[case[0]] = []
    for nonlinear_target in nonlinear_targets.values():
        model, alpha = fit_validated(X, nonlinear_target, fit, valid, refit)
        nonlinear_scores[case[0]].append(r2_score(nonlinear_target[test], model.predict(X[test])))
    print(f'{case[0]:<22}' + ''.join(f'{v:>9.3f}' for v in scores))
fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
for label, scores in memory_scores.items(): ax.plot(delay_values, scores, 'o-', label=label)
ax.axhline(0, color='gray', lw=.8); ax.set(xlabel='input delay d', ylabel='held-out R²')
ax.grid(alpha=.25); ax.legend(fontsize=8, ncol=2); plt.show()
''')
md(r'''
## 4. Nonlinear readout tasks

Using the same independent-input frames and train/validation/test split, test the second-order function \(P_2(x)=(3x^2-1)/2\) for the just-written input and for the preceding input, as well as their product \(x_kx_{k-1}\), where \(x_k=2u_k-1\). Each task uses its own validated ridge readout. The input-only task tests nonlinear optical encoding without requiring memory; the delayed and product tasks require information about past input. These scores are computed from the same simulation frames as the previous section.
''')
code('''
print(f"{'Condition':<22}" + ''.join(f'{name:>21}' for name in nonlinear_targets))
for label, scores in nonlinear_scores.items():
    print(f'{label:<22}' + ''.join(f'{value:>21.3f}' for value in scores))
''')
md(r'''
## 5. Interpretation and limitations

The random initial phase is a transient, not input-driven long-term memory; after washout, the remaining driven state is essentially the scalar coefficient of \(G\). A high Mackey–Glass score alone does not establish useful fading memory: compare with instant, persistence, and the independent-input task. Phase periodicity can make different unwrapped states optically indistinguishable; camera noise and ridge regularization further limit recoverability. \(G\), \(\tau\), gain, camera noise, and random seeds here are illustrative. Test multiple independent optical realizations, initial maps, input/noise seeds and parameters before claiming robustness. No temperature dynamics or experimentally calibrated index changes are represented.
''')

notebook = {'cells': cells, 'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}, 'language_info': {'name': 'python', 'version': '3'}}, 'nbformat': 4, 'nbformat_minor': 4}
output = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('phase_only_rc_tests.ipynb')
output.write_text(json.dumps(notebook, ensure_ascii=False, indent=1)+'\n', encoding='utf-8')
