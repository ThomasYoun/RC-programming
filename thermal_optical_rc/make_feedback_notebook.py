"""Generate the feedback + thermal-memory optical RC notebook."""
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
# Optical feedback and a thermally varying transmission matrix: RC with physical noise

Following the optical path in `RC_opMGSLM.ipynb`, this version places **the previous CCD state and the current input image in the upper and lower halves of the SLM**. It therefore includes both electronic feedback and thermal memory. All random scattering screens are initialized once; the same thermal phase screen changes both \(W_{\rm in}\) and \(W_{\rm res}\).

Place `optical_feedback_thermal.py` and `MackeyGlass_t17.txt` alongside this notebook. NumPy, PyTorch, Matplotlib and Jupyter are required. Temperature, phase, noise and time parameters are adjustable examples and have not been calibrated to a real device.
''')
md(r'''
## 1. Optical path and state update

\[
u_k\;\xrightarrow{\text{fixed mask}}\;\Phi_k,\qquad
E_{\mathrm{SLM},k}=\exp\left(i\pi\begin{bmatrix}r_k\\\Phi_k\end{bmatrix}\right).
\]

The two images are separately enlarged and centered in the upper and lower SLM halves; zero-phase regions still transmit light. The complex field is center-padded or cropped to the optical plane, then passes through

\[
E_{\rm CCD,k}=O_2\,B\,D(\theta_k)\,A\,O_1 E_{\rm SLM,k},\qquad
r_{k+1}=(1-\alpha)r_k+\alpha\,\operatorname{camera}(|E_{\rm CCD,k}|^2).
\]

`camera` includes noise, fixed gain, downsampling and clipping. The standard leaky update is a convex combination; at \(\alpha=1\), there is no additional digital leak, but feedback through the SLM remains. \(A,B\) are two fixed random scattering stages and \(O_1,O_2\) are the objective Fourier pupils. The thermal screen lies between the scattering stages:

\[
a=e^{-\Delta t/\tau},\quad \theta_k=a\theta_{k-1}+(1-a)u_k,\quad
D_{pp}(\theta_k)=e^{i\phi_{\max}\theta_k G_p}.
\]

The same input is assumed to **separately control** normalized heating power; phase modulation alone does not guarantee this coupling. The Gaussian \(G_p\) is nonzero over about 10% of the internal plane, with \(D=1\) elsewhere. Measurement noise is part of the CCD state and enters the feedback.

Let \(P_{\rm res},P_{\rm in}\) project onto the upper and lower SLM regions. The two effective blocks of the linear complex-field propagation are
\[
W_{\rm res}(k)=O_2BD(\theta_k)AO_1P_{\rm res},\quad
W_{\rm in}(k)=O_2BD(\theta_k)AO_1P_{\rm in}.
\]
These are neither resampled random matrices at each step nor stored as dense matrices. The current input and state first undergo exponential phase encoding at the SLM, followed by square-law CCD detection. The dynamic \(W\) blocks describe effective maps of the complex field.
''')
code('''
import numpy as np
import matplotlib.pyplot as plt
import torch
from pathlib import Path
from optical_feedback_thermal import OpticalFeedbackThermalRC, PhysicalNoise, RidgeReadout, nrmse

torch.set_num_threads(1)
plt.rcParams['figure.dpi'] = 100
noise = PhysicalNoise(pupil_rms_rad=0.06, slm_phase_std_rad=0.015,
                      photon_scale=50000, read_noise_electrons=1.0, shot_noise=True)
ideal = PhysicalNoise(pupil_rms_rad=0.0, slm_phase_std_rad=0.0,
                     photon_scale=50000, read_noise_electrons=0.0, shot_noise=False)
common = dict(optical_shape=(128, 128), slm_shape=(128, 128),
    input_shape=(32, 32), state_shape=(16, 16), pupil_radius=(0.25, 0.25),
    heater_fraction=0.1, heater_phase_max=6.0, thermal_tau=8.0, dt=1.0,
    camera_gain=40.0, leak=1.0, seed=42, noise_seed=123)
model = OpticalFeedbackThermalRC(**common, heater_mode='driven', feedback=True, noise=noise)
print('Optical / SLM / state shapes:', model.optical_shape, model.slm_shape, model.state_shape)
print('Heated support fraction:', float(model.heater_support.float().mean()))
print('Noise parameters:', noise)
''')
md(r'''
## 2. Objective pupils and physical noise

A finite pupil performs deterministic spatial-frequency filtering; it is **not noise**. The available parameters are:

- `pupil_rms_rad`: fixed defocus/astigmatism wavefront error in the pupils (radians RMS);
- `slm_phase_std_rad`: per-frame SLM phase jitter standard deviation;
- `shot_noise` / `photon_scale`: Poisson shot noise sampled from the expected CCD photon count;
- `read_noise_electrons`: CCD electronic readout noise standard deviation.

Noise is added before fixed camera gain and downsampling; frames are not individually min–max normalized. Values are illustrative, and `photon_scale` in particular is not a measured optical power. Change pupil radius, resolution, gain and leak rate in `common` to scan them.
''')
code('''
example_input = np.random.default_rng(7).random(50)
_, _, snap = model.run(example_input, snapshot_at=49)
fig, axes = plt.subplots(3, 3, figsize=(12, 10), constrained_layout=True)
panels = [
    (snap['state_before'], 'previous state r[k]', 'viridis', None, None),
    (snap['input_img'], 'input image', 'gray', 0, 1),
    (snap['slm_phase'], 'SLM concatenated phase (rad)', 'gray', 0, np.pi),
    (np.abs(snap['slm_field']), '|exp(i phase)|', 'gray', 0, 1),
    (model.pupil1_support.cpu(), 'objective 1 pupil', 'gray', 0, 1),
    (torch.angle(model.pupil1).cpu(), 'pupil 1 phase error (rad)', 'coolwarm', -0.2, 0.2),
    (snap['ccd_ideal'], 'ideal CCD intensity', 'magma', None, None),
    (snap['ccd_measured'], 'noisy CCD intensity', 'magma', None, None),
    (snap['state_after'], 'state r[k+1]', 'viridis', 0, 1),
]
for ax, (arr, title, cmap, vmin, vmax) in zip(axes.flat, panels):
    im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=9); ax.axis('off'); fig.colorbar(im, ax=ax, fraction=.045, pad=.03)
plt.show()
print('SLM field amplitude range:', np.abs(snap['slm_field']).min(), np.abs(snap['slm_field']).max())
print('Unit amplitude gives a white |E| image; phase information is in the phase panel.')
''')
md(r'''
## 3. Do both effective propagation blocks change with temperature?

With the same random device and per-frame noise disabled, retain the upper and lower halves of the same test complex field in turn and compare the CCD **complex fields** at \(\theta=0\) and \(\theta=1\). This illustrates changes in the two effective operators; it is not a measurement of the full transmission matrix.
''')
code('''
probe_model = OpticalFeedbackThermalRC(**common, heater_mode='driven', feedback=True, noise=ideal)
generator = torch.Generator().manual_seed(90)
probe = torch.exp(2j*torch.pi*torch.rand(common['slm_shape'], generator=generator))
differences = {}
for name, part in [('W_res', 'upper'), ('W_in', 'lower')]:
    mask = torch.zeros(common['slm_shape'])
    if part == 'upper': mask[:common['slm_shape'][0]//2] = 1
    else: mask[common['slm_shape'][0]//2:] = 1
    field0 = probe_model.optical_field(probe*mask, 0.0)
    field1 = probe_model.optical_field(probe*mask, 1.0)
    differences[name] = float(torch.linalg.vector_norm(field1-field0)/torch.linalg.vector_norm(field0))
    print(f'{name} complex field relative change: {differences[name]:.5f}')
assert min(differences.values()) > 1e-6
# The pure-phase heater preserves local amplitude and is 1 outside support.
heat_mask = torch.exp(1j*common['heater_phase_max']*probe_model.heater_profile)
assert torch.allclose(heat_mask.abs(), torch.ones_like(heat_mask.abs()), atol=1e-6)
assert torch.allclose(heat_mask[~probe_model.heater_support], torch.ones_like(heat_mask[~probe_model.heater_support]))

# Illustrate spatial influence with a common probe and ideal camera. The
# threshold is illustrative, not an experimentally calibrated noise floor.
intensity0 = probe_model.optical_field(probe, 0.0).abs().square().cpu().numpy()
intensity1 = probe_model.optical_field(probe, 1.0).abs().square().cpu().numpy()
delta_intensity = np.abs(intensity1-intensity0)
affected = np.mean(delta_intensity > .05*intensity0.mean())
print(f'CCD pixels changing by more than 5% of mean reference intensity: {affected:.2%}')
fig, axes = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
im = axes[0].imshow(probe_model.heater_profile.cpu(), cmap='inferno')
axes[0].set_title('Thermal support ~10%'); fig.colorbar(im, ax=axes[0])
im = axes[1].imshow(delta_intensity, cmap='magma')
axes[1].set_title('CCD |I(θ=1) − I(θ=0)|'); fig.colorbar(im, ax=axes[1])
for ax in axes: ax.axis('off')
plt.show()
''')
md(r'''
## 4. Ridge readout, as used to train ESN outputs

Flatten \(r_{k+1}\), standardize using training data and add a bias to obtain the row vector \(z_k=[1,\widetilde r_{k+1}^T]\). The training target is \(y_k=u_{k+1}\):

\[
W_{\rm out}=(Z^TZ+\lambda P)^{-1}Z^TY,\quad P=\mathrm{diag}(0,1,\ldots,1),\quad \hat Y=ZW_{\rm out}.
\]

Only \(W_{\rm out}\) is trained; the bias is not regularized. The code solves a linear system, fits feature standardization on the training set, selects \(\lambda\) from fixed candidates on the validation set, then fits on training plus validation. The test set is excluded from fitting. Prediction is teacher-forced: the true \(u_k\) is supplied throughout testing.
''')
code('''
series = np.loadtxt(Path('MackeyGlass_t17.txt')).reshape(-1)
n_train, n_valid, n_test, washout = 1600, 400, 1000, 100
split = n_train+n_valid
assert len(series) >= split+n_test+1
lo, hi = series[:n_train].min(), series[:n_train].max()
u_raw = (series[:split+n_test]-lo)/(hi-lo)
u = np.clip(u_raw, 0, 1)
y = (series[1:split+n_test+1]-lo)/(hi-lo)
train = np.arange(washout, n_train-1)
valid = np.arange(n_train, split-1)
fit_all = np.arange(washout, split-1)
test = np.arange(split, split+n_test)
print('Test clipping fraction:', np.mean(u[test] != u_raw[test]))
print('Mackey–Glass test lag-one correlation:', np.corrcoef(u[test], y[test])[0, 1])

cases = [
    ('No heat + feedback, ideal', 'off', True, ideal, 8.0),
    ('Thermal τ=8 + feedback, ideal', 'driven', True, ideal, 8.0),
    ('No heat + feedback, noisy', 'off', True, noise, 8.0),
    ('Static heat + feedback, noisy', 'static', True, noise, 8.0),
    ('Instant heat + feedback, noisy', 'instant', True, noise, 8.0),
    ('Thermal τ=8 + feedback, noisy', 'driven', True, noise, 8.0),
    ('Thermal τ=8, no feedback, noisy', 'driven', False, noise, 8.0),
    ('Thermal τ=2 + feedback, noisy', 'driven', True, noise, 2.0),
    ('Thermal τ=32 + feedback, noisy', 'driven', True, noise, 32.0),
]
outputs, readouts, state_bank, temperatures = {}, {}, {}, {}
for label, heat, feedback, noise_config, tau in cases:
    params = {**common, 'thermal_tau': tau}
    instance = OpticalFeedbackThermalRC(**params, heater_mode=heat, feedback=feedback, noise=noise_config)
    states, temps, _ = instance.run(u)
    # All cases use identical splits, candidate regularizers and optical seed.
    candidates = [0.1, 1.0, 10.0, 100.0]
    validation = [nrmse(y[valid], RidgeReadout(a).fit(states[train], y[train]).predict(states[valid])) for a in candidates]
    alpha = candidates[int(np.argmin(validation))]
    readout = RidgeReadout(alpha).fit(states[fit_all], y[fit_all])
    outputs[label] = readout.predict(states[test])
    readouts[label], state_bank[label], temperatures[label] = readout, states, temps
    print(f'{label:<32} NRMSE={nrmse(y[test], outputs[label]):.4f}  ridge={alpha:g}')
print(f"{'Persistence':<32} NRMSE={nrmse(y[test], u[test]):.4f}")
print('W_out shape (bias + CCD state):', readouts['Thermal τ=8 + feedback, noisy'].weights_.shape)
''')
code('''
fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True, constrained_layout=True)
display_cases = ['No heat + feedback, ideal', 'Thermal τ=8 + feedback, ideal',
                 'No heat + feedback, noisy', 'Thermal τ=8 + feedback, noisy']
for ax, label in zip(axes, display_cases):
    ax.plot(y[test][:200], color='black', lw=1.2, label='target')
    ax.plot(outputs[label][:200], lw=1, label=label)
    ax.set_title(f'{label}: NRMSE={nrmse(y[test], outputs[label]):.4f}', fontsize=9)
    ax.legend(fontsize=8)
axes[-1].set_xlabel('test time step'); plt.show()

weights = readouts['Thermal τ=8 + feedback, noisy'].weights_[1:].reshape(common['state_shape'])
fig, ax = plt.subplots(figsize=(5, 4)); im=ax.imshow(weights, cmap='coolwarm')
ax.set_title('Ridge W_out on standardized CCD state'); fig.colorbar(im, ax=ax); plt.show()
''')
md(r'''
## 5. Memory control: delayed reconstruction of independent random inputs

Adjacent Mackey–Glass samples are highly correlated, so a static mapping or persistence can also predict one step ahead; low error alone does not demonstrate fading memory. Independent random inputs train separate readouts to reconstruct past \(u_{k-d}\) from the current CCD state. Train and test are consecutive time segments; \(R^2<0\) means worse than the test-mean baseline.

No heater plus feedback can have **electronic feedback memory**; thermal plus no feedback (\(\alpha=1\)) isolates thermal memory, while thermal plus feedback includes both. A scalar \(\theta\) still represents only one independent thermal state. Assess statistical stability over multiple random seeds, noise levels and time constants.
''')
code('''
random_u = np.random.default_rng(2026).random(2000)
memory_cases = [cases[2], cases[5], cases[6], cases[3], cases[4], cases[7], cases[8]]
delays = np.array([1, 2, 4, 8, 16])
split_memory = 1400
memory_scores = {}
for label, heat, feedback, noise_config, tau in memory_cases:
    params = {**common, 'thermal_tau': tau}
    instance = OpticalFeedbackThermalRC(**params, heater_mode=heat, feedback=feedback, noise=noise_config)
    states, _, _ = instance.run(random_u)
    scores = []
    for delay in delays:
        train_indices = np.arange(100, split_memory)
        test_indices = np.arange(split_memory, len(random_u))
        readout = RidgeReadout(10.0).fit(states[train_indices], random_u[train_indices-delay])
        pred = readout.predict(states[test_indices]); target = random_u[test_indices-delay]
        scores.append(1-np.sum((target-pred)**2)/np.sum((target-target.mean())**2))
    memory_scores[label] = np.array(scores)
    print(f'{label:<32}', np.round(scores, 3))
fig, ax = plt.subplots(figsize=(9, 4))
for label, result in memory_scores.items(): ax.plot(delays, result, 'o-', label=label)
ax.axhline(0, color='0.5', lw=.8); ax.set(xlabel='delay d', ylabel='held-out R²')
ax.legend(fontsize=7); ax.grid(alpha=.2); plt.show()
''')

notebook = {'cells': cells, 'metadata': {'kernelspec': {'display_name':'Python 3','language':'python','name':'python3'}, 'language_info': {'name':'python','version':'3'}}, 'nbformat':4, 'nbformat_minor':4}
output_path = Path(sys.argv[1]) if len(sys.argv)>1 else Path('optical_feedback_thermal_rc.ipynb')
output_path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1)+'\n', encoding='utf-8')
