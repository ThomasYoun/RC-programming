"""Generate the pure physical thermal-memory optical RC notebook."""
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
# Input-Driven Physical Thermal Memory: Optical Reservoir Computing Without Electronic Feedback

This framework treats temperature as the material's internal dynamical state. The external input drives thermal evolution and encodes a weak probe on the SLM. A thermal phase screen changes the fixed scattering propagation, and the CCD speckle provides observable features.

**There is no CCD-to-SLM electronic feedback and no state/input concatenation.** The entire active SLM area encodes only the current input; only the digital linear readout is trained. This notebook requires `optical_physical_memory.py` and `MackeyGlass_t17.txt` in the same directory, plus NumPy, PyTorch, Matplotlib and Jupyter.
''')
md(r'''
## 1. State equation and causal timing

\[
a=e^{-\Delta t/\tau},\qquad
\theta_{k+1}=a\theta_k+\beta u_k,
\qquad r_k=H(\theta_k,u_k),\qquad \hat y_k=W_{\rm out}[1;r_k].
\]

**At each time step:** first, the current input \(u_k\) and pre-exposure temperature \(\theta_k\) produce the speckle features \(r_k\); second, the current heating updates \(\theta_{k+1}\). Thus, \(r_k\) can contain the current input and thermal traces of \(u_0,\ldots,u_{k-1}\). The one-step prediction target in this notebook is \(y_k=u_{k+1}\).

`beta` can be specified directly. With the default `beta=None`, the model uses \(\beta=(1-a)g_{\rm th}\). The write term remains linear in \(u_k\), while the steady-state gain under a constant input remains \(g_{\rm th}\) as \(\tau\) changes. Set `beta` explicitly if the experiment requires a fixed injection per step.

\(u_k\) must control non-negative absorbed power. The code assumes that the same normalized input drives the heater and encodes the probe. How the two beams or input-dependent absorption are coupled is a **model assumption** that must be implemented and calibrated in the apparatus; a phase-only SLM does not automatically provide this coupling.
''')
md(r'''
## 2. Optical mapping and observation

\[
E_{\rm SLM,k}=e^{i\pi\Phi(u_k)},\qquad
E_{\rm CCD,k}=O_2 B D(\theta_k)A O_1E_{\rm SLM,k},
\]
\[
D_{pp}(\theta_k)=e^{i\phi_{\max}\theta_kG_p},\qquad
r_k=\operatorname{downsample}(|E_{\rm CCD,k}|^2).
\]

The input continuously modulates the entire SLM: \(\Phi_p(u_k)=u_k(0.5+0.5M_p)\), where the fixed random spatial mask \(M_p\in[0,1]\). Every SLM pixel therefore changes phase with \(u_k\), rather than only switching pixels whose thresholds are crossed. The spatial mask prevents the entire SLM from acquiring the same global phase; if the phase-only modulation were identical everywhere, ideal intensity detection would not reveal a change. Set `slm_encoding='binary_threshold'` to restore the earlier threshold encoding.

\(O_1,O_2\) are objective Fourier pupils, and \(A,B\) are fixed random complex-amplitude propagation operators generated once. The Gaussian heater support covers approximately **80%** of the effective intermediate plane. Its heating gain was increased from 1.0 to **1.3**, increasing the steady-state thermal state by 30% for the same heating input. \(D=1\) outside the support. The full-plane SLM input and the heater's spatial region are distinct. Increasing the support also broadens the Gaussian according to its area; temperature is not uniform inside the support.

If the physical temperature rise is \(\Delta T_p=T_s\theta_kG_p\), and the thermo-optic coefficient and effective heated thickness are approximately uniform locally, then
\[
\Delta n_p\approx\frac{dn}{dT}\Delta T_p,\qquad
\delta\phi_p\approx\frac{2\pi}{\lambda}L_{\rm eff}\frac{dn}{dT}T_s\theta_kG_p.
\]
The model combines these uncalibrated physical quantities into one adjustable parameter, \(\phi_{\max}\), providing an approximate link from temperature to refractive index to phase.

\(\theta_k\) is the internal reservoir state; \(r_k\) is a camera observation vector, not a feedback state. A scalar temperature supplies only one independent thermal dynamical variable; 256 CCD features do not automatically provide 256 independent thermal memories. The thermal phase screen is an effective approximation of the temperature-dependent transmission matrix, not a complete heat-diffusion or refractive-index solution.
''')
code('''
import numpy as np
import matplotlib.pyplot as plt
import torch
from pathlib import Path
from optical_physical_memory import PhysicalMemoryOptics, PhysicalNoise, RidgeReadout, nrmse

torch.set_num_threads(1)
plt.rcParams['figure.dpi'] = 100
noise = PhysicalNoise(pupil_rms_rad=0.18, slm_phase_std_rad=0.06,
                      photon_scale=5000, read_noise_electrons=3.0, shot_noise=True)
previous_noise = PhysicalNoise(pupil_rms_rad=0.06, slm_phase_std_rad=0.015,
                               photon_scale=50000, read_noise_electrons=1.0, shot_noise=True)
ideal = PhysicalNoise(pupil_rms_rad=0, slm_phase_std_rad=0,
                      photon_scale=50000, read_noise_electrons=0, shot_noise=False)
common = dict(optical_shape=(128, 128), slm_shape=(128, 128),
    input_shape=(32, 32), camera_shape=(16, 16), pupil_radius=(0.25, 0.25),
    heater_fraction=0.8, heater_phase_max=6.0, dt=1.0,
    thermal_gain=1.3, seed=42, noise_seed=123)
model = PhysicalMemoryOptics(**common, tau=8, beta=None, heat_mode='driven', noise=noise)
print(f'Heater support = {float(model.heater_support.float().mean()):.6f}')
print(f'a = {model.a:.6f}; beta = {model.beta:.6f}; thermal gain = {model.thermal_gain:.2f}')
print('Input, SLM, optical, CCD features:', model.input_shape, model.slm_shape, model.optical_shape, model.camera_shape)
''')
md(r'''
## 3. Inspect the SLM, pupils and CCD

The finite objective pupil is a deterministic filter. Fixed pupil wavefront error, SLM phase jitter, CCD Poisson shot noise and read noise are separately adjustable. Relative to the previous version, the example increases pupil aberration from 0.06 to 0.18 rad and SLM jitter from 0.015 to 0.06 rad, reduces `photon_scale` from 50,000 to 5,000 to increase relative shot noise, and increases read noise from 1 to 3 electrons. These values are uncalibrated examples. Pupil aberration is a static device difference; the other noise sources vary between exposures. Frames are not individually min–max normalized.

The white `|E_SLM|=1` image is the expected amplitude of a phase-only SLM; the grayscale phase image shows the continuous spatial phase and jitter across the whole SLM.
''')
code('''
sample_inputs = np.random.default_rng(7).random(50)
_, _, _, snap = model.run(sample_inputs, snapshot_at=49)
fig, axes = plt.subplots(2, 4, figsize=(14, 7), constrained_layout=True)
panels = [(snap['input_image'], 'continuous input phase pattern', 'gray', 0, 1),
          (snap['slm_phase'], 'full-SLM phase (rad)', 'gray', 0, np.pi),
          (np.abs(snap['slm_field']), '|exp(i phase)|', 'gray', 0, 1),
          (model.pupil1_support.cpu(), 'objective 1 pupil', 'gray', 0, 1),
          (model.heater_profile.cpu(), 'heater support ~80%', 'inferno', 0, 1),
          (snap['ccd_ideal'], 'ideal CCD', 'magma', None, None),
          (snap['ccd_measured'], 'noisy CCD', 'magma', None, None),
          (snap['features'], 'CCD feature r[k]', 'magma', None, None)]
for ax, (arr, title, cmap, vmin, vmax) in zip(axes.flat, panels):
    im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=9); ax.axis('off'); fig.colorbar(im, ax=ax, fraction=.045, pad=.03)
plt.show()
print('theta used for camera:', snap['theta_read'], 'next theta:', snap['theta_next'])
print('SLM amplitude min/max:', np.abs(snap['slm_field']).min(), np.abs(snap['slm_field']).max())
print('Mean expected CCD electrons per optical pixel:', snap['ccd_ideal'].mean()*noise.photon_scale)

# With jitter disabled, every SLM pixel has a nonzero input response.
check = PhysicalMemoryOptics(**common, tau=8, heat_mode='driven', noise=ideal)
_, _, _, low = check.run([0.25], snapshot_at=0)
_, _, _, high = check.run([0.75], snapshot_at=0)
coverage = np.mean(np.abs(high['slm_phase']-low['slm_phase']) > 1e-7)
print('Fraction of SLM pixels modulated by input change:', coverage)
assert coverage == 1.0

# Repeat a fixed input/temperature; this quantifies temporal noise remaining
# after 128×128 -> 16×16 pooling, not a calibrated experimental measurement.
for name, cfg in [('previous noise', previous_noise), ('strong noise', noise)]:
    repeat_model = PhysicalMemoryOptics(**common, tau=8, heat_mode='static', noise=cfg)
    repeats, _, _, _ = repeat_model.run(np.full(100, 0.5))
    relative_fluctuation = repeats.std(axis=0).mean()/repeats.mean()
    print(f'{name}: pooled-feature temporal std / mean = {relative_fluctuation:.2%}')
''')
md(r'''
## 4. Pulse input and exponential decay: verify the timing

The input pulse is \(u_0=1\), followed by zeros, with initial \(\theta_0=0\). The temperatures used by the camera should be \(\theta_0=0\), \(\theta_1=\beta\), \(\theta_2=a\beta\), followed by decay by a factor of \(a\) per step. The initial zero is intentional: this experiment updates the temperature after the exposure.
''')
code('''
thermal = PhysicalMemoryOptics(**common, tau=8, heat_mode='driven', noise=ideal)
pulse = np.r_[1.0, np.zeros(39)]
_, theta, theta_next, _ = thermal.run(pulse)
expected = np.r_[0.0, thermal.beta*thermal.a**np.arange(39)]
assert np.allclose(theta, expected)
assert np.allclose(theta_next, thermal.a*theta + thermal.beta*pulse)
fig, ax = plt.subplots(figsize=(8, 3.5))
ax.stem(np.arange(len(pulse)), theta, linefmt='C0-', markerfmt='C0o', basefmt=' ', label='theta[k]')
ax.plot(expected, 'C1--', label='analytic')
ax.set(xlabel='step k', ylabel='thermal state', title='Input pulse written into heat and fading')
ax.legend(); plt.show()
print('Maximum recursion error:', np.max(np.abs(theta-expected)))
''')
md(r'''
## 5. Same current input, different input histories

Keep the same random optical device and use the same probe at the final step; compare heating histories of all zeros and all ones over the preceding 40 steps. To isolate memory, noise is disabled and the result is compared with no-heat, static-heat and instantaneous-response controls. The final exposure uses \(\theta_k\), not the subsequently updated \(\theta_{k+1}\).
''')
code('''
history0 = np.r_[np.zeros(40), 0.5]
history1 = np.r_[np.ones(40), 0.5]
history_results = {}
for mode in ['off', 'static', 'instant', 'driven']:
    m = PhysicalMemoryOptics(**common, tau=8, heat_mode=mode, noise=ideal)
    r0, t0, _, s0 = m.run(history0, snapshot_at=40)
    r1, t1, _, s1 = m.run(history1, snapshot_at=40)
    delta = np.linalg.norm(r1[-1]-r0[-1])/max(np.linalg.norm(r0[-1]), 1e-12)
    history_results[mode] = delta
    print(f'{mode:<10} theta=({t0[-1]:.4f}, {t1[-1]:.4f})  relative speckle feature difference={delta:.5f}')
assert history_results['off'] < 1e-7 and history_results['static'] < 1e-7
assert history_results['instant'] < 1e-7 and history_results['driven'] > 1e-5
delta_image = np.abs(s1['ccd_ideal']-s0['ccd_ideal'])
print('CCD fraction with |ΔI| > 5% of mean reference intensity:',
      np.mean(delta_image > .05*np.mean(s0['ccd_ideal'])))
fig, axes = plt.subplots(1, 3, figsize=(11, 3.5), constrained_layout=True)
for ax, arr, title in zip(axes, [s0['ccd_ideal'], s1['ccd_ideal'], delta_image],
                          ['zero heating history', 'one heating history', 'absolute CCD difference']):
    im=ax.imshow(arr, cmap='magma'); ax.set_title(title); ax.axis('off'); fig.colorbar(im, ax=ax, fraction=.045)
plt.show()
''')
md(r'''
## 6. Mackey–Glass one-step prediction

Chronological split: **the first 2,000 steps form the training segment, including the first 200 washout steps, followed by 1,000 test steps**. The final 400 steps of the training segment are used internally to select the ridge parameter. Once selected, the readout is refit on all available post-washout training samples. Input normalization is determined only from the internal fitting segment; the thermal state **continues across the training/test boundary without resetting**.

The readout uses only \(r_k\) and a bias; its target is \(u_{k+1}\). The split follows **input time**: test inputs have zero-based indices 2000–2999, and their one-step targets have indices 2001–3000. To avoid using the first test value as a training target, the final training input still drives the system but is excluded from readout fitting. Thus, the 2,000-step training segment and 200-step washout leave **1,799 supervised training samples**. Real inputs continue to drive the system during testing, so this is teacher-forced prediction.

Controls include no heating, fixed heat, instantaneous heat, input-driven thermal states with different \(\tau\), noise settings, and persistence. A static optical mapping can also predict an autocorrelated time series, so interpret these results alongside delayed reconstruction of independent random inputs below.

`previous noise` refers to the earlier lower-noise setting, and `strong noise` to the higher-noise setting. Their static pupil aberrations also differ, so the overall difference cannot be attributed solely to time-varying random noise.

For every condition, \(W_{\rm out}\) is fitted by ridge regression, as in an ESN linear readout:
\[
W_{\rm out}=(Z^TZ+\lambda P)^{-1}Z^TY,
\quad Z_k=[1,\widetilde r_k^T],\quad P=\operatorname{diag}(0,1,\ldots,1).
\]
Feature means and variances are fitted only on the training data, and the bias is not regularized. The code solves a linear system rather than explicitly computing a matrix inverse.
''')
code('''
series = np.loadtxt(Path('MackeyGlass_t17.txt')).reshape(-1)
train_length, validation_length, test_length, washout = 2000, 400, 1000, 200
fit_end = train_length-validation_length
assert len(series) >= train_length+test_length+1
lo, hi = series[:fit_end].min(), series[:fit_end].max()
u_raw = (series[:train_length+test_length]-lo)/(hi-lo)
u = np.clip(u_raw, 0, 1)
y = (series[1:train_length+test_length+1]-lo)/(hi-lo)
train = np.arange(washout, fit_end-1)
valid = np.arange(fit_end, train_length-1)
fit_all = np.arange(washout, train_length-1)
test = np.arange(train_length, train_length+test_length)
print(f'Train steps: {train_length}, washout: {washout}, fit samples: {len(fit_all)}, test steps: {len(test)}')
print(f'Test input indices: {test[0]}–{test[-1]}; target indices: {test[0]+1}–{test[-1]+1}')
assert fit_all[-1]+1 < train_length and len(test)==test_length
print('Test input clipping fraction:', np.mean(u_raw[test] != u[test]))
print('Test correlation of u[k] and u[k+1]:', np.corrcoef(u[test], y[test])[0, 1])
cases = [
    ('No heat, ideal', 'off', 8, ideal),
    ('Thermal τ=8, ideal', 'driven', 8, ideal),
    ('Thermal τ=8, previous noise', 'driven', 8, previous_noise),
    ('No heat, strong noise', 'off', 8, noise),
    ('Static heat, strong noise', 'static', 8, noise),
    ('Instant heat, strong noise', 'instant', 8, noise),
    ('Thermal τ=2, strong noise', 'driven', 2, noise),
    ('Thermal τ=8, strong noise', 'driven', 8, noise),
    ('Thermal τ=32, strong noise', 'driven', 32, noise),
]
predictions, readouts = {}, {}
for label, mode, tau, noise_config in cases:
    m = PhysicalMemoryOptics(**common, tau=tau, heat_mode=mode, noise=noise_config)
    features, thermal_state, _, _ = m.run(u)
    lambdas = [0.1, 1.0, 10.0, 100.0]
    validation = [nrmse(y[valid], RidgeReadout(l).fit(features[train], y[train]).predict(features[valid])) for l in lambdas]
    alpha = lambdas[int(np.argmin(validation))]
    readout = RidgeReadout(alpha).fit(features[fit_all], y[fit_all])
    predictions[label], readouts[label] = readout.predict(features[test]), readout
    print(f'{label:<25} NRMSE={nrmse(y[test], predictions[label]):.4f}  ridge={alpha:g}')
print(f"{'Persistence':<25} NRMSE={nrmse(y[test], u[test]):.4f}")
print('W_out shape (bias + camera features):', readouts['Thermal τ=8, strong noise'].weights_.shape)
''')
code('''
fig, axes = plt.subplots(4, 1, figsize=(15, 9), sharex=True, constrained_layout=True)
test_steps = np.arange(len(test))
for ax, label in zip(axes, ['No heat, strong noise', 'Static heat, strong noise', 'Thermal τ=2, strong noise', 'Thermal τ=8, strong noise']):
    ax.plot(test_steps, y[test], color='black', lw=1.0, label='target')
    ax.plot(test_steps, predictions[label], lw=0.9, label=label)
    ax.set_title(f'{label}: NRMSE={nrmse(y[test], predictions[label]):.4f}', fontsize=9)
    ax.legend(fontsize=8)
axes[-1].set_xlim(0, len(test))
axes[-1].set_xlabel('test time step'); plt.show()

weights = readouts['Thermal τ=8, strong noise'].weights_[1:].reshape(common['camera_shape'])
fig, ax = plt.subplots(figsize=(5, 4)); im=ax.imshow(weights, cmap='coolwarm')
ax.set_title('W_out on standardized speckle features'); fig.colorbar(im, ax=ax); plt.show()
''')
md(r'''
## 7. Reconstructing independent random-input history

Generate 3,000 i.i.d. inputs and reconstruct \(u_{k-d}\) using only the current \(r_k\). The first 2,000 steps form the training segment, including 200 washout steps; the last 1,000 steps are for testing. This tests the readout of past inputs from speckle more directly than one-step Mackey–Glass prediction. A negative test \(R^2\) is worse than the test-mean baseline. The steady-state thermal gain is held fixed across \(\tau\). The experiment includes camera noise; test additional random seeds and parameters to assess robustness.
''')
code('''
random_u = np.random.default_rng(2026).random(3000)
memory_cases = cases[3:]
delays = np.array([1, 2, 4, 8, 16]); split_memory = 2000
memory_scores = {}
for label, mode, tau, noise_config in memory_cases:
    m = PhysicalMemoryOptics(**common, tau=tau, heat_mode=mode, noise=noise_config)
    features, _, _, _ = m.run(random_u)
    values = []
    for delay in delays:
        tr = np.arange(200, split_memory); te = np.arange(split_memory, len(random_u))
        pred = RidgeReadout(10.0).fit(features[tr], random_u[tr-delay]).predict(features[te])
        target = random_u[te-delay]
        values.append(1-np.sum((target-pred)**2)/np.sum((target-target.mean())**2))
    memory_scores[label] = np.array(values)
    print(f'{label:<25}', np.round(values, 3))
fig, ax = plt.subplots(figsize=(9, 4))
for label, result in memory_scores.items(): ax.plot(delays, result, 'o-', label=label)
ax.axhline(0, color='0.5', lw=.8); ax.set(xlabel='delay d', ylabel='held-out R²')
ax.legend(fontsize=7); ax.grid(alpha=.2); plt.show()
''')
md(r'''
## 8. Is the nonlinearity observable by the readout?

Use independent random inputs \(x_k=2u_k-1\) to test the second-order function \(P_2(x)=(3x^2-1)/2\) and the product of current and past inputs. The first 2,000 steps form the training segment, including 200 washout steps; the last 1,000 are for testing, with a validation split inside training. Compare no heating, thermal memory at different \(\tau\), and a fixed probe with only the heating sequence changing. The fixed probe helps separate the current-input SLM encoding from the nonlinearity of thermal phase, interference and CCD detection. For each task, select the ridge parameter on validation data only and report test \(R^2\).
''')
code('''
nl_u = np.random.default_rng(213).random(3000)
x = 2*nl_u-1
targets = {'current P2': .5*(3*x*x-1),
           'delay-1 P2': .5*(3*np.roll(x, 1)**2-1),
           'current × delay-1': x*np.roll(x, 1)}
nl_train = np.arange(200, 1600); nl_valid = np.arange(1600, 2000); nl_test = np.arange(2000, len(nl_u))
nl_cases = [('No heat', 'off', 2, False), ('Thermal τ=2', 'driven', 2, False),
            ('Thermal τ=8', 'driven', 8, False), ('τ=2 fixed probe', 'driven', 2, True)]
def score_r2(target, prediction, indices):
    y0 = target[indices]
    return 1-np.sum((y0-prediction)**2)/np.sum((y0-y0.mean())**2)
print(f"{'Condition':<22}" + ''.join(f'{name:>21}' for name in targets))
for label, mode, tau, fixed_probe in nl_cases:
    m = PhysicalMemoryOptics(**common, tau=tau, heat_mode=mode, noise=noise)
    probe = np.full(len(nl_u), 0.5) if fixed_probe else nl_u
    features, _, _, _ = m.run(probe, heat_inputs=nl_u)
    result = []
    for target in targets.values():
        candidates = [1.0, 10.0, 100.0, 1000.0]
        validation = [score_r2(target, RidgeReadout(a).fit(features[nl_train], target[nl_train]).predict(features[nl_valid]), nl_valid) for a in candidates]
        best = candidates[int(np.argmax(validation))]
        fit = np.r_[nl_train, nl_valid]
        prediction = RidgeReadout(best).fit(features[fit], target[fit]).predict(features[nl_test])
        result.append(score_r2(target, prediction, nl_test))
    print(f'{label:<22}' + ''.join(f'{v:>21.3f}' for v in result))
''')
md(r'''
## 9. Limitations

1. The only recurrent variable in this version is \(\theta\); there is no previous-speckle feedback, electronic leak or output-to-SLM feedback.
2. A thermal phase screen inserted into fixed random propagation is a simplified effective model of \(\Delta T\rightarrow\Delta n\rightarrow\Delta\phi\rightarrow T\). Neither \(\phi_{\max}\) nor the input-to-heating coupling has been calibrated to a material, wavelength or thickness.
3. The model assumes one spatial Gaussian profile and one relaxation time. Multiple spatial regions, time constants or a heat-diffusion equation require additional modeling.
4. The figures and metrics show one device realization and random seed; a single simulation cannot establish experimental performance.
''')

notebook = {'cells': cells, 'metadata': {'kernelspec': {'display_name':'Python 3','language':'python','name':'python3'}, 'language_info': {'name':'python','version':'3'}}, 'nbformat':4, 'nbformat_minor':4}
output_path = Path(sys.argv[1]) if len(sys.argv)>1 else Path('optical_physical_memory_rc.ipynb')
output_path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1)+'\n', encoding='utf-8')
