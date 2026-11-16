"""Generate a localized phase/noise and finite-exposure MG RC experiment."""
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
# Localized phase perturbations, stronger noise and camera timing

This notebook extends the MG **input-driven phase-screen** model without modifying the earlier notebook. It compares smaller phase-screen supports, a steeper center-to-edge profile, stronger camera noise, static and instantaneous phase controls, and fading \(\tau=2,5,10,50\). All calculations use the same fixed threshold SLM encoder, random scattering operators and pupil realization; there is no camera-speckle-to-SLM reservoir feedback.

Results are simulations with uncalibrated phase and noise parameters. A separate single-pulse finite-exposure calculation discusses whether a rapid transient could be detected. It does **not** claim that the current idealized instant control is a measured thermal response.

Run from `thermal_optical_rc`, with `phase_only_rc.py`, `phase_only_optics.py`, `phase_exposure.py`, `spatial_input_optics.py`, `optical_physical_memory.py`, and `MackeyGlass_t17.txt` in the same directory. NumPy, PyTorch, Matplotlib and Jupyter are required. To limit CPU overhead, launch with `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1` where applicable.
''')
md(r'''
## 1. Fixed optical path and support/contrast ablation

The scalar MG input is threshold-encoded over the full 128×128 phase-only SLM. The dynamic phase screen is placed inside a fixed optical path:
\[
\Phi_{k+1}=e^{-1/\tau}\Phi_k+2\pi u_kG,\qquad
\mathbf r_{k}=\operatorname{pool}\left|O_2B\,\mathrm{diag}(e^{i\Phi_{k+1}})AO_1E_{{\rm SLM},k}\right|^2.
\]
Here \(\mathbf r_k\) is indexed by the input \(u_k\) written immediately before that frame; the phase in that frame is the post-write state \(\Phi_{k+1}\).
The base support was approximately **80%** of the phase-screen plane. Half and one-quarter of that area are therefore **40% and 20% of the full plane**, not 50% and 25%. On each active support, a regularized inverse-radius profile is normalized to peak at one. We compare: (a) 80% reference, (b) 40% and 20% with the previous core radius for an area-only control, and (c) 40% and 20% with a smaller core radius for a steeper center-to-edge profile.

The absolute center-to-edge difference below means the unwrapped **single-write increment at \(u=1\)**, between the peak and the smallest nonzero value *inside* the support. The outside is zero in every case, so the peak-to-outside difference is always \(2\pi\). At a general \(u\), the single-write difference scales by \(u\); accumulated dynamic phase can exceed \(2\pi\). We keep the input-to-phase gain fixed at \(2\pi\) in all cases rather than attributing an increase in gain to reduced area. The hard cutoff and inverse-radius phase profile are assumed, not a calibrated heat-diffusion solution.

To ensure the **dynamic affected area** is really confined to the chosen support, the same initial random unwrapped map is masked by that support. The fixed scattering medium still extends across the entire optical plane. This differs from the previous notebook's initially full-plane transient; the initial-phase difference is not fully negligible after the fixed washout for \(\tau=50\).
''')
code('''
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from pathlib import Path
from phase_only_optics import radial_profile
from spatial_input_optics import ThresholdSLMOptics
from phase_only_rc import simulate_phase_rc, free_phase_rollout, chronological_indices, r2_score
from optical_physical_memory import RidgeReadout, nrmse
from phase_exposure import integrated_speckle, camera_features

torch.set_num_threads(1)
plt.rcParams['figure.dpi'] = 100
shape, camera_shape = (128, 128), (16, 16)
optics = ThresholdSLMOptics(n_inputs=1, optical_shape=shape, pupil_radius=(.25, .25), seed=42, encoder_seed=271)
base_initial = np.random.default_rng(2026).uniform(0, 6*np.pi, shape)
beta = 2*np.pi
specs = [('80% reference', .8, .12), ('40% area-only', .4, .12),
         ('20% area-only', .2, .12), ('40% steeper', .4, .04), ('20% steeper', .2, .04)]
profiles = {name: radial_profile(shape, core_radius=core, support_fraction=fraction)
            for name, fraction, core in specs}
initials = {name: base_initial*(profile > 0) for name, profile in profiles.items()}
print(f"{'Profile':<18}{'Support':>11}{'core':>9}{'G edge':>11}{'Δphase / π':>15}")
for name, fraction, core in specs:
    G = profiles[name]; edge = G[G > 0].min()
    print(f'{name:<18}{np.mean(G > 0):>11.2%}{core:>9.3f}{edge:>11.4f}{beta*(1-edge)/np.pi:>15.4f}')
fig, axes = plt.subplots(2, len(specs), figsize=(15, 5), constrained_layout=True)
for col, (name, _, _) in enumerate(specs):
    G = profiles[name]
    im = axes[0, col].imshow(G, cmap='inferno', vmin=0, vmax=1)
    axes[0, col].set_title(name, fontsize=9); axes[0, col].axis('off')
    axes[1, col].plot(np.linspace(-1, 1, shape[1]), beta*G[shape[0]//2]/np.pi)
    axes[1, col].set(xlabel='normalized position', ylabel='single-write phase / π', ylim=(-.05, 2.05))
    axes[1, col].grid(alpha=.2)
fig.colorbar(im, ax=axes[0], label='normalized phase profile', shrink=.7); plt.show()
''')
md(r'''
## 2. Camera noise and detector-plane effect

The previous camera settings were `photon_scale=5000` and read-noise standard deviation 3 electrons. The stronger illustrative condition uses **1200** and **8 electrons**, respectively: fewer photons increase relative shot noise and larger per-frame read noise increases electronic uncertainty. Poisson shot noise and Gaussian read noise are added before spatial camera downsampling; negative electron counts are clipped as in the previous code. At low signal levels this clipping produces an upward bias; no calibrated black level is included. These settings are not experimental calibration. Objective pupil geometry stays fixed; SLM jitter, dark current, saturation, drift and correlated camera noise are not included.

The following single-write comparison uses a fixed SLM input and ideal noiseless intensities to isolate how a local phase screen affects the camera. Spatial mixing can make a local perturbation visible across many output pixels, but no fixed influence-area claim is imposed. All camera images share one intensity scale.
''')
code('''
noise_previous = dict(photon_scale=5000., read_noise_electrons=3.)
noise_strong = dict(photon_scale=1200., read_noise_electrons=8.)
probe = np.array([.5])
I0 = np.abs(optics.field_for_input(np.zeros(shape), probe))**2
display_profiles = ['80% reference', '40% steeper', '20% steeper']
images = [I0]+[np.abs(optics.field_for_input(beta*profiles[name], probe))**2 for name in display_profiles]
scale = max(np.max(image) for image in images)
fig, axes = plt.subplots(1, len(images), figsize=(13, 3.5), constrained_layout=True)
for ax, image, name in zip(axes, images, ['No phase write']+display_profiles):
    im = ax.imshow(image, cmap='magma', vmin=0, vmax=scale)
    ax.set_title(name, fontsize=9); ax.axis('off')
fig.colorbar(im, ax=axes, label='ideal CCD intensity', shrink=.7); plt.show()
for name, image in zip(display_profiles, images[1:]):
    corr = np.corrcoef(I0.ravel(), image.ravel())[0, 1]
    diff = np.linalg.norm(image-I0)/np.linalg.norm(I0)
    print(f'{name:<18} CCD spatial correlation={corr:.4f}; relative intensity change={diff:.4f}')
''')
md(r'''
## 3. Static, instant and fading phase are distinct

* **Static:** the initial perturbation is frozen. The SLM still changes with \(u_k\), but the perturbation phase does not respond to input history.
* **Instant snapshot:** after each input, set \(\Phi=2\pi u_kG\), erase previous perturbations, and observe immediately. This is an idealized memoryless **post-write snapshot**, not a measured camera response or a separately calibrated fast thermal model.
* **Fading:** \(\Phi_{k+1}=e^{-1/\tau}\Phi_k+2\pi u_kG\), observed after the write. The same per-step write gain is used for all \(\tau\), so \(\tau\) also changes accumulated phase amplitude.

The main RC sweep retains the prior instantaneous-camera approximation for every condition. Section 6 separately models finite exposure for a pulse; do not interpret the main instant snapshot as if a real camera had infinite speed. The area-only controls are evaluated at instant and \(\tau=5\); the reference and the two steeper profiles cover instant and all four requested \(\tau\). Static is evaluated for each of the three support sizes. The prior camera noise is repeated for the 80%-reference \(\tau=5\) case.
''')
code('''
taus = [2, 5, 10, 50]
main_profiles = ['80% reference', '40% steeper', '20% steeper']
cases = []
for name in main_profiles:
    cases.append((f'{name} | static', name, 'static', 5, 'strong'))
    cases.append((f'{name} | instant', name, 'instant', 5, 'strong'))
    cases.extend((f'{name} | τ={tau}', name, 'memory', tau, 'strong') for tau in taus)
for name in ['40% area-only', '20% area-only']:
    cases.append((f'{name} | instant', name, 'instant', 5, 'strong'))
    cases.append((f'{name} | τ=5', name, 'memory', 5, 'strong'))
cases.append(('80% reference | τ=5 | previous noise', '80% reference', 'memory', 5, 'previous'))
def features_for(inputs, case):
    label, name, mode, tau, noise_name = case
    noise = noise_strong if noise_name == 'strong' else noise_previous
    result, _ = simulate_phase_rc(inputs, optics, profiles[name], initials[name],
        tau=tau, beta=beta, mode=mode, camera_shape=camera_shape,
        slm_inputs=np.asarray(inputs)[:, None], noise_seed=123, **noise)
    return result

def fit_validated(X, y, fit_idx, valid_idx, refit_idx):
    alphas = [.1, 1., 10., 100., 1000.]
    validation = [r2_score(y[valid_idx], RidgeReadout(a).fit(X[fit_idx], y[fit_idx]).predict(X[valid_idx])) for a in alphas]
    best = alphas[int(np.argmax(validation))]
    return RidgeReadout(best).fit(X[refit_idx], y[refit_idx]), best

print(f'{len(cases)} cases: {len(main_profiles)} support/contrast groups plus matched ablations.')
''')
md(r'''
## 4. Mackey–Glass: teacher forcing and free prediction

The first 2,000 input steps form the training segment, including 200 washout steps and an internal 400-step validation window. The last 1,000 input steps are test. The same training-only input normalization, boundary purges, 256 CCD features and ridge regression used in the earlier MG notebook are retained. For \(\tau=50\), 200 steps leave \(e^{-4}\approx1.83\%\) of the initial phase, so it is not completely erased. Teacher-forced testing supplies the true current input at every step. Compare it with the persistence baseline \(\hat u_{k+1}=u_k\).

For the main three profile groups, free prediction warms the phase with the 2,000 real training inputs; then it feeds each predicted scalar back to the input with clipping to \([0,1]\). This is an explicit **digital output-to-input loop**, not a feedback-free autonomous material. Scores use the raw un-clipped predictions; clip fraction describes the feedback inputs. Teacher-forced and free common-window NRMSE use the same target indices 2001–2999. The separate full free score uses targets 2000–2999, and free persistence holds the final training input constant.
''')
code('''
series = np.loadtxt(Path('MackeyGlass_t17.txt')).reshape(-1)
assert len(series) >= 3001
lo, hi = series[:1600].min(), series[:1600].max()
if hi <= lo: raise ValueError('training input is constant')
raw = (series[:3001]-lo)/(hi-lo)
u, target = np.clip(raw[:3000], 0, 1), raw[1:3001]
splits = chronological_indices(train_length=2000, washout=200, validation_length=400, test_length=1000, horizon=1)
te = splits['test']
models, mg_predictions, mg_scores, alpha_selected = {}, {}, {}, {}
print(f'Training segment 2000 (washout 200), test 1000; input test clipping: {np.mean(u[te] != raw[te]):.2%}')
print(f'One-step persistence NRMSE: {nrmse(target[te], u[te]):.4f}')
print(f"{'Case':<42}{'NRMSE':>10}{'R²':>10}{'ridge':>10}")
for case in cases:
    label = case[0]
    X = features_for(u, case)
    model, alpha = fit_validated(X, target, splits['fit'], splits['valid'], splits['refit'])
    pred = model.predict(X[te])
    models[label], mg_predictions[label], alpha_selected[label] = model, pred, alpha
    mg_scores[label] = (nrmse(target[te], pred), r2_score(target[te], pred))
    print(f'{label:<42}{mg_scores[label][0]:>10.4f}{mg_scores[label][1]:>10.4f}{alpha:>10g}')

fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
for name in main_profiles:
    ax.plot(taus, [mg_scores[f'{name} | τ={tau}'][0] for tau in taus], 'o-', label=name)
ax.axhline(nrmse(target[te], u[te]), color='black', ls='--', lw=1, label='persistence')
ax.set(xlabel='phase decay τ (steps)', ylabel='test NRMSE', xticks=taus, title='Teacher-forced MG: same optics, stronger camera noise')
ax.grid(alpha=.2); ax.legend(); plt.show()
''')
code('''
labels_to_plot = ['static', 'instant']+[f'τ={tau}' for tau in taus]
fig, axes = plt.subplots(len(labels_to_plot), len(main_profiles), figsize=(14, 11),
                         sharex=True, sharey=True, constrained_layout=True)
for row, mode in enumerate(labels_to_plot):
    for col, name in enumerate(main_profiles):
        ax = axes[row, col]
        ax.plot(np.arange(1000), target[te], color='black', lw=.65, label='target')
        ax.plot(np.arange(1000), mg_predictions[f'{name} | {mode}'],
                color='tab:blue', alpha=.8, lw=.65, label='one-step prediction')
        ax.set_title(f'{name} | {mode}', fontsize=9)
        ax.set_xlim(0, 1000); ax.grid(alpha=.15)
        if col == 0: ax.set_ylabel('normalized MG')
        if row == len(labels_to_plot)-1: ax.set_xlabel('test step (0–1000)')
axes[0, 0].legend(fontsize=7, loc='upper right')
fig.suptitle('Teacher-forced one-step MG predictions', fontsize=12); plt.show()
''')
code('''
free_cases = [case for case in cases if case[1] in main_profiles and case[4]=='strong']
free_target = raw[2000:3000]
free_persistence = np.full(len(free_target), u[1999])
free_results = {}
print(f'Free constant-persistence NRMSE: {nrmse(free_target, free_persistence):.4f}')
print(f"{'Case':<42}{'free full':>11}{'free common':>14}{'teacher common':>17}{'clip':>10}")
for case in free_cases:
    label, name, mode, tau, _ = case
    rollout = free_phase_rollout(u[:2000], models[label], optics, profiles[name], initials[name],
        tau=tau, beta=beta, mode=mode, camera_shape=camera_shape, noise_seed=123,
        n_steps=1000, encode_slm=True, **noise_strong)
    free_results[label] = rollout
    common = raw[2001:3000]
    print(f'{label:<42}{nrmse(free_target, rollout["predictions"]):>11.4f}{nrmse(common, rollout["predictions"][1:]):>14.4f}{nrmse(common, mg_predictions[label][:-1]):>17.4f}{rollout["clipping_fraction"]:>10.2%}')
fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True, constrained_layout=True)
for ax, name in zip(axes, main_profiles):
    label = f'{name} | τ=5'
    ax.plot(np.arange(1000), free_target, color='black', lw=.8, label='target')
    ax.plot(np.arange(1000), free_results[label]['predictions'], lw=.8, label='free prediction')
    ax.set_title(label); ax.set_xlim(0, 1000); ax.grid(alpha=.2); ax.legend(fontsize=8)
axes[-1].set_xlabel('free test step (0–1000)'); plt.show()
''')
md(r'''
## 5. Independent-input history reconstruction

MG is temporally correlated, so a static mapping can predict one step without physical memory. Here the input is i.i.d. uniform, and a separate readout reconstructs \(u_{k-d}\) for \(d=0,1,2,4,8,16\). The post-write camera frame at index \(k\) sees \(u_k\), so \(d=0\) is current input and \(d\ge1\) tests past inputs. The first 2,000 input steps include 200 washout and validation; the last 1,000 are held out. Ridge strength is selected on validation only. The result is a sparse set of held-out \(R^2\) scores, **not** a total memory-capacity estimate. Negative \(R^2\) is worse than the retrospective test-mean reference.
''')
code('''
random_u = np.random.default_rng(213).uniform(0, 1, 3000)
delays = [0, 1, 2, 4, 8, 16]
fit = np.arange(200, 1600); valid = np.arange(1600, 2000)
refit = np.arange(200, 2000); test = np.arange(2000, 3000)
memory_scores = {}
print(f"{'Case':<29}" + ''.join(f'{f"d={d}":>9}' for d in delays))
for case in free_cases:
    label = case[0]
    X = features_for(random_u, case)
    result = []
    for d in delays:
        target_history = np.zeros_like(random_u)
        target_history[d:] = random_u[:len(random_u)-d] if d else random_u
        model, _ = fit_validated(X, target_history, fit, valid, refit)
        result.append(r2_score(target_history[test], model.predict(X[test])))
    memory_scores[label] = np.array(result)
    print(f'{label:<29}' + ''.join(f'{value:>9.3f}' for value in result))
fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True, constrained_layout=True)
for ax, name in zip(axes, main_profiles):
    for mode in ['instant']+[f'τ={tau}' for tau in taus]:
        label = f'{name} | {mode}'
        ax.plot(delays, memory_scores[label], 'o-', label=mode)
    ax.axhline(0, color='gray', lw=.8); ax.set(title=name, xlabel='delay d', ylabel='held-out R²')
    ax.grid(alpha=.2); ax.legend(fontsize=8)
plt.show()
''')
md(r'''
## 6. Could a fast decay be recorded by a camera?

Yes, **in principle**, if the sample has a finite response time and the probe/exposure is synchronized appropriately. But the existing `instant` code only specifies a no-memory post-write snapshot; it does not specify a thermal decay time, the camera's exposure, or a physically realized actuator. It cannot by itself establish that a camera records a transient. Static means a time-independent phase screen; instant means an input-dependent response with no retained input history. Both still have an input-dependent SLM image.

To examine the timing separately, assume one short phase pulse, held probe input 0.5 and no further writes:
\[
\Phi(t)=\Phi(0^+)e^{-t/\tau_{\rm fast}},\qquad
I_{\rm camera}=\frac{1}{T_{\rm exp}}\int_{t_d}^{t_d+T_{\rm exp}}
\left|E_{\rm CCD}(\Phi(t))\right|^2dt.
\]
The camera integrates **intensity**, not the phase or complex field. The model below uses \(\tau_{\rm fast}=0.2\) input-step units: after one full step the original phase amplitude is \(e^{-1/0.2}\approx0.67\%\). For each exposure, expected photon count scales with exposure duration at a fixed photon flux; read noise is per frame. The reported finite-sample vector-norm diagnostic has a positive noise floor, so an independently sampled no-pulse/no-pulse comparison is also printed; this is an illustrative check, not a statistical detection test. In reality dark current, detector saturation, nonzero pulse width, device timing and finite readout rate also matter and are not modeled here.

Optical gating and pump–probe approaches can measure fast transient phase changes. For example, one [photothermal phase-imaging study](https://pmc.ncbi.nlm.nih.gov/articles/PMC6904725/) used ~70 ns probe bursts and integrated nine repeated pump–probe pairs in a 10 ms camera exposure. The gated temporal resolution is therefore not the rate of independent camera frames; repeating a pulse during one exposure is not the same as recording an independent frame for every arbitrary input. See also the [camera exposure documentation](https://docs.baslerweb.com/exposure-time) and [camera signal-to-noise discussion](https://camera.hamamatsu.com/jp/en/learn/technical_information/thechnical_guide/calculating_snr.html). All times and photon rates below are illustrative dimensionless assumptions, not measurements of the user's device.
''')
code('''
tau_fast = .2
pulse_profile = profiles['20% steeper']
pulse_phase = beta*pulse_profile
fixed_probe = .5
zero_intensity = np.abs(optics.field_for_input(np.zeros(shape), np.array([fixed_probe])))**2
exposures = [.05, .2, 1.]
trials, substeps = 64, 32
print(f'τ_fast={tau_fast:g} step; phase remaining one step later = {np.exp(-1/tau_fast):.2%}')
print(f"{'Exposure':<12}{'photon scale':>14}{'ideal ΔI / ||I0||':>20}{'paired signal/noise':>22}{'noise-only floor':>20}")
exposure_results = {}
for T in exposures:
    hot = integrated_speckle(optics, fixed_probe, pulse_phase, tau_fast, T, delay=0, substeps=substeps)
    rel = np.linalg.norm(hot-zero_intensity)/np.linalg.norm(zero_intensity)
    rng = np.random.default_rng(444)
    pairs = np.array([camera_features(hot, T, 1200., 8., rng, camera_shape)-camera_features(zero_intensity, T, 1200., 8., rng, camera_shape) for _ in range(trials)])
    mean_delta = pairs.mean(axis=0)
    noise_rms = np.sqrt(np.mean(np.sum((pairs-mean_delta)**2, axis=1)))
    snr = np.linalg.norm(mean_delta)/noise_rms if noise_rms else np.nan
    null_rng = np.random.default_rng(445)
    null_pairs = np.array([camera_features(zero_intensity, T, 1200., 8., null_rng, camera_shape)-camera_features(zero_intensity, T, 1200., 8., null_rng, camera_shape) for _ in range(trials)])
    null_mean = null_pairs.mean(axis=0)
    null_rms = np.sqrt(np.mean(np.sum((null_pairs-null_mean)**2, axis=1)))
    null_floor = np.linalg.norm(null_mean)/null_rms if null_rms else np.nan
    exposure_results[T] = (hot, rel, snr, null_floor)
    print(f'{T:<12g}{1200*T:>14.1f}{rel:>20.4f}{snr:>22.4f}{null_floor:>20.4f}')
print('Paired signal/noise is an empirical vector-norm diagnostic; finite-sample noise biases the signal-norm estimate upward.')
late = integrated_speckle(optics, fixed_probe, pulse_phase, tau_fast, .05, delay=1., substeps=substeps)
print(f'Relative ideal speckle change for a short exposure one step after pulse: {np.linalg.norm(late-zero_intensity)/np.linalg.norm(zero_intensity):.6f}')
fig, axes = plt.subplots(1, len(exposures)+1, figsize=(14, 3), constrained_layout=True)
all_images = [zero_intensity]+[exposure_results[T][0] for T in exposures]
vmax = max(image.max() for image in all_images)
for ax, image, label in zip(axes, all_images, ['No pulse']+[f'T_exp={T:g} steps' for T in exposures]):
    im = ax.imshow(image, cmap='magma', vmin=0, vmax=vmax); ax.set_title(label); ax.axis('off')
fig.colorbar(im, ax=axes, label='exposure-averaged ideal intensity', shrink=.7); plt.show()
''')
md(r'''
## 7. Interpretation and limits

The support fraction describes an effective phase-screen area, not a measured heated volume or the fraction of CCD pixels influenced by it. A smaller, steeper perturbation also changes spatial phase gradients and interference. Whether a performance difference is due to area, profile, noise or random realization requires the displayed ablations and multiple independent input/optics/noise seeds. The single scalar driven phase-memory channel and fixed \(2\pi\) write gain limit the model; no spatial heat diffusion or calibrated material parameters are represented.

The finite-exposure calculation assumes an impulsive phase write, exponential relaxation and a held probe; continuous heating or a camera with rolling shutter needs a different temporal model. It is an isolated optical detectability check, not proof that a sub-step transient can be recorded for every arbitrary input in real time.
''')

notebook = {'cells': cells, 'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}, 'language_info': {'name': 'python', 'version': '3'}}, 'nbformat': 4, 'nbformat_minor': 4}
output = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('phase_noise_region_rc.ipynb')
output.write_text(json.dumps(notebook, ensure_ascii=False, indent=1)+'\n', encoding='utf-8')
