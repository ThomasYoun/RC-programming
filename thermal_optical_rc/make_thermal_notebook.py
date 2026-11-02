"""Build the tutorial notebook from the cell sources below."""
import json
import sys
from pathlib import Path
import textwrap

cells = []
def md(text):
    cells.append({'cell_type': 'markdown', 'metadata': {}, 'source': textwrap.dedent(text).strip().splitlines(keepends=True)})
def code(text):
    cells.append({'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': textwrap.dedent(text).strip().splitlines(keepends=True)})

md(r'''
# A feedback-free optical reservoir with thermal fading memory

Workflow: build the SLM–objective–scattering medium–thermal phase screen–objective–CCD optical path → compare speckle under different thermal conditions → predict Mackey–Glass → test whether input history is stored.

**Model assumption:** the continuous input both generates the SLM pattern and linearly controls normalized heating power; thermal relaxation preserves past inputs. There is no electronic feedback or concatenation of the previous reservoir state. Heating power proportional to input is a simplified initial assumption requiring experimental calibration.

Place `thermal_free_decay.py` and `MackeyGlass_t17.txt` alongside this notebook. `numpy`, `torch`, `matplotlib` and Jupyter are required.
''')
md(r'''
## 1. Mathematical model and controls

$$a=e^{-\Delta t/\tau},\quad\theta_k=a\theta_{k-1}+(1-a)u_k,\quad
G_p=\mathbf1_{p\in S}\exp[-r_p^2/(2\sigma^2)],\quad |S|/N=1/4.$$

$$E_\mathrm{SLM}=e^{i\pi\Phi(u_k)},\quad
E_\mathrm{CCD}=O_2D_k H O_1E_\mathrm{SLM},\quad
D_{pp,k}=e^{i\phi_{\max}\theta_kG_p},\quad r_k=\operatorname{downsample}(|E_\mathrm{CCD}|^2).$$

Here $O_1,O_2$ are the Fourier pupils of the two objectives, and $H$ is the fixed implicit random scattering operator. Defining $A=HO_1$ and $B=O_2$ gives $T_k=BD_kA$. The thermal screen lies in an **effective plane** after fixed scattering and before the second objective; it is not a complete microscopic model of the sample.

The current input is thresholded using a fixed random mask to produce a 64×64 binary image, enlarged by nearest-neighbor interpolation to the full 128×128 SLM; there is no upper/lower concatenation. The CCD produces the current observed state $r_k$, which is **not fed back** to the SLM. Zero-phase locations still transmit light (complex amplitude 1). When the SLM field is center-padded or cropped to the optical plane, only the area outside the SLM is zero-padded.

Every condition uses the **same** $A,B$, input mask, initial thermal state and phase coefficient. $\tau$ and $\Delta t$ use the same time unit. `phase_max` is the maximum phase per unit temperature rise; it is illustrative rather than material-calibrated. The 25% refers to the spatial pixel fraction of the **internal heated plane**, not 25% of transmission-matrix elements.

| Condition | Phase | Thermal state |
|---|---|---|
| No heating | $\phi_{\max}=0$ | No effect on propagation |
| Static thermal perturbation | Finite | $\theta=0.5$ |
| Instant thermal response | Finite | $\theta_k=u_k$, no history memory |
| Free cooling | Finite | Single heating event, no input writing |
| Input-driven thermal memory | Finite | Same input, different $\tau$ |

The fixed scattering section follows the original notebook: random complex-amplitude screen → orthonormal FFT → random complex-amplitude screen → orthonormal IFFT. Each objective is FFT → circular pupil → IFFT; scattering screens and the input mask remain fixed across conditions.
''')
code('''
import numpy as np
import matplotlib.pyplot as plt
import torch
from pathlib import Path
from thermal_free_decay import Condition, FixedDisorderedOptics, spatial_correlation, ridge_predict, scores

torch.set_num_threads(1)  # avoids excessive CPU threading for small FFTs
plt.rcParams['figure.dpi'] = 110
DT = 1.0
PHASE_MAX = 8.0
optics = FixedDisorderedOptics(shape=(128, 128), input_shape=(64, 64),
    slm_shape=(128, 128), camera_shape=(16, 16),
    objective1_radius=0.25, objective2_radius=0.25, seed=42)
conditions = [
    Condition('No heater', 0.0, None, mode='static'),
    Condition('Static heater', PHASE_MAX, None, theta0=0.5, mode='static'),
    Condition('Instant response', PHASE_MAX, None, theta0=0.0, mode='instant'),
    Condition('Free decay τ=8', PHASE_MAX, 8, theta0=1.0, mode='free'),
    Condition('Driven τ=2', PHASE_MAX, 2, theta0=0.0, mode='driven'),
    Condition('Driven τ=8', PHASE_MAX, 8, theta0=0.0, mode='driven'),
    Condition('Driven τ=32', PHASE_MAX, 32, theta0=0.0, mode='driven'),
]
print('Heater support fraction:', float(optics.support.float().mean()))
print('SLM / optical / camera state shapes:', optics.slm_shape, optics.shape, optics.camera_shape)
''')
md(r'''
## 2. Planes in the optical path

The figure shows each plane for one input. For ordinary computers, the 128×128 CCD image is average-downsampled to 16×16. Adjust `shape`, `slm_shape`, `input_shape` and `camera_shape` in the previous cell; a larger camera state increases linear regression memory and computation.
''')
code('''
snap = optics.snapshot(input_value=0.5, theta=1.0, phase_max=PHASE_MAX)
fig, axes = plt.subplots(2, 4, figsize=(14, 7), constrained_layout=True)
panels = [(snap['input_image'], 'input image', 'gray'),
          (snap['slm_phase'], 'SLM phase: input across full SLM', 'twilight'),
          (optics.pupil1.cpu(), 'objective 1 pupil', 'gray'),
          (np.abs(snap['objective1_field'])**2, 'after objective 1', 'magma'),
          (np.abs(snap['scattered_field'])**2, 'after fixed scattering', 'magma'),
          (snap['thermal_phase'], 'heater phase', 'inferno'),
          (snap['ccd_intensity'], 'CCD intensity after objective 2', 'magma'),
          (snap['reservoir_state'], 'downsampled reservoir state', 'magma')]
for ax, (value, title, cmap) in zip(axes.flat, panels):
    im = ax.imshow(value, cmap=cmap); ax.set_title(title, fontsize=9); ax.axis('off')
    fig.colorbar(im, ax=ax, fraction=.046, pad=.04)
plt.show()

# Zero temperature and zero phase strength must produce the same optical field.
no_heat_a = optics.snapshot(.5, 1.0, 0.0)
no_heat_b = optics.snapshot(.5, 0.0, PHASE_MAX)
assert np.allclose(no_heat_a['ccd_intensity'], no_heat_b['ccd_intensity'])
# A pure-phase heater conserves power at its own plane, before objective 2.
assert np.isclose(np.sum(np.abs(snap['scattered_field'])**2),
                  np.sum(np.abs(snap['heated_field'])**2), rtol=1e-5)
print('Optical checks: zero heating agrees; phase screen preserves power.')
''')
md(r'''
## 3. Heating profile and time course

Plot the normalized spatial profile and $\theta_k$. The input is 1 at times 0–39 and 0 thereafter; input-driven conditions cool exponentially by $e^{-\Delta t/\tau}$ after heating stops. Free cooling begins at temperature rise 1 and is independent of input. The Gaussian is truncated outside its finite support; 1/4 refers to spatial support, not Gaussian energy.
''')
code('''
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
im = ax[0].imshow(optics.profile.cpu(), cmap='inferno', vmin=0, vmax=1)
ax[0].set_title('Gaussian phase profile, 25% support'); fig.colorbar(im, ax=ax[0])
times = np.arange(121)
heating_pulse = (times < 40).astype(float)
ax[1].plot(times, heating_pulse, 'k:', label='heating input')
for condition in conditions[1:]:
    ax[1].plot(times, condition.trajectory(heating_pulse, DT), label=condition.name)
ax[1].set(xlabel='time step k', ylabel='normalized temperature θ', title='Heating followed by relaxation')
ax[1].legend(fontsize=8); fig.tight_layout(); plt.show()
''')
md(r'''
## 4. Fixed probe: observe speckle for each condition

Write the heating sequence from the previous section into the thermal state while fixing the weak probe pattern at $u=0.5$, so image changes arise from the thermal phase screen. Heater and probe controls are temporarily separated for this diagnostic; in the Mackey–Glass section both are controlled by the same input. All images share a color scale without per-frame min–max normalization. Correlation describes speckle changes and does not directly measure memory capacity.
''')
code('''
snap_times = np.array([0, 20, 40, 100])
probe = np.full(len(snap_times), 0.5)
reference = optics.intensity([0.5], [0.0], PHASE_MAX)[0]
all_frames = {c.name: optics.intensity(probe, c.trajectory(heating_pulse, DT)[snap_times], c.phase_max) for c in conditions}
vmax = max(np.percentile(frames, 99.5) for frames in all_frames.values())
fig, axes = plt.subplots(len(conditions), len(snap_times), figsize=(12, 13), constrained_layout=True)
for row, condition in enumerate(conditions):
    for col, k in enumerate(snap_times):
        im = axes[row, col].imshow(all_frames[condition.name][col], cmap='magma', vmin=0, vmax=vmax)
        axes[row, col].set_xticks([]); axes[row, col].set_yticks([])
        axes[row, col].set_title(f'{condition.name} | k={k}', fontsize=9)
fig.colorbar(im, ax=axes, shrink=0.65, label='intensity'); plt.show()

fig, ax = plt.subplots(figsize=(9, 4))
for condition in conditions:
    theta = condition.trajectory(heating_pulse, DT)
    frames = optics.intensity(np.full(len(theta), 0.5), theta, condition.phase_max)
    ax.plot(times, spatial_correlation(frames, reference), label=condition.name)
ax.set(xlabel='time step k', ylabel='spatial correlation with no heater', ylim=(-1.02, 1.02))
ax.legend(fontsize=8); ax.grid(alpha=.25); plt.show()
''')
md(r'''
## 5. Same current input, different histories

The two sequences have the same current input at the same time but histories of all zeros and all ones. No heating and instantaneous response have no history memory; the input-driven thermal state should produce different speckle for the same probe. This test does not rely on electronic feedback.
''')
code('''
history_a = np.r_[np.zeros(40), 0.5]
history_b = np.r_[np.ones(40), 0.5]
history_differences = {}
for case in conditions:
    theta_a = case.trajectory(history_a, DT)[-1]
    theta_b = case.trajectory(history_b, DT)[-1]
    frames = optics.intensity([0.5, 0.5], [theta_a, theta_b], case.phase_max)
    difference = np.linalg.norm(frames[0]-frames[1]) / max(np.linalg.norm(frames[0]), 1e-12)
    history_differences[case.name] = difference
    print(f'{case.name:<24} theta=({theta_a:.4f}, {theta_b:.4f})  relative CCD difference={difference:.6f}')
assert history_differences['No heater'] < 1e-7
assert history_differences['Instant response'] < 1e-7
assert history_differences['Driven τ=8'] > 1e-5
case = conditions[-2]  # tau=8
decay = np.exp(-DT/case.tau)
sequence = case.trajectory(np.r_[np.ones(20), np.zeros(10)], DT)
assert np.allclose(sequence[21:]/sequence[20:-1], decay)
print('Driven heater decays exponentially after heating is switched off.')
''')
md(r'''
## 6. Mackey–Glass: one-step prediction by condition

The local `MackeyGlass_t17.txt` is the user's existing sequence. The chronological split uses the first 2,000 points for training and the next 2,000 for testing, with training scaling applied to test data. The target is $y_k=u_{k+1}$; each test step receives true $u_k$ (teacher forcing), with no prediction or CCD electronic feedback. The thermal state evolves continuously from training into testing and is **not reset at the boundary**.

Each readout receives the current $u_k$ and current speckle features and uses the same ridge parameter. Persistence $\hat y_{k+1}=u_k$ and a linear readout using only current input are also shown. NRMSE uses the test target standard deviation. To avoid leakage at the boundary, the final training target remains within the training period.
''')
code('''
data_file = Path('MackeyGlass_t17.txt')
if not data_file.exists():
    raise FileNotFoundError('Put MackeyGlass_t17.txt next to this notebook, or set data_file to its path.')
series = np.loadtxt(data_file).reshape(-1)
n_train, n_test = 2000, 2000
assert len(series) >= n_train+n_test+1
train_min, train_max = series[:n_train].min(), series[:n_train].max()
u_raw = (series[:n_train+n_test]-train_min)/(train_max-train_min)
u = np.clip(u_raw, 0, 1)
y = (series[1:n_train+n_test+1]-train_min)/(train_max-train_min)
washout = 100
train_idx = np.arange(washout, n_train-1)  # remove initial transient; target stays in train
test_idx = np.arange(n_train, n_train+n_test)
k = np.arange(len(u))
print('Clipped test input fraction:', np.mean(u_raw[test_idx] != u[test_idx]))
predictions = {'Persistence': u[test_idx],
               'Current input linear': ridge_predict(u[train_idx, None], y[train_idx], u[test_idx, None])}
for c in conditions:
    features = optics.features(u, c.trajectory(u, DT), c.phase_max)
    z = np.column_stack((u, features))
    predictions[c.name] = ridge_predict(z[train_idx], y[train_idx], z[test_idx])
print(f"{'Condition':<26} {'RMSE':>10} {'NRMSE':>10}")
for name, pred in predictions.items():
    s = scores(y[test_idx], pred)
    print(f"{name:<26} {s['RMSE']:10.5f} {s['NRMSE']:10.5f}")
''')
code('''
fig, axes = plt.subplots(len(conditions)+1, 1, figsize=(12, 13), sharex=True, constrained_layout=True)
view = slice(0, 200)
for ax, name in zip(axes, ['Persistence'] + [c.name for c in conditions]):
    ax.plot(y[test_idx][view], 'k', lw=1.2, label='target')
    ax.plot(predictions[name][view], lw=1, alpha=.85, label=name)
    ax.set_ylabel('scaled value'); ax.set_title(f"{name}: NRMSE={scores(y[test_idx], predictions[name])['NRMSE']:.3f}", fontsize=9)
    ax.legend(loc='upper right', fontsize=7)
axes[-1].set_xlabel('test time step'); plt.show()
''')
md(r'''
## 7. Random-input delay reconstruction: can input history be read out?

Use independent random inputs to train separate linear readouts to predict $u_{k-d}$ using only the current speckle state, without supplying past values. Use consecutive train/test splits and report test $R^2$; a negative value is worse than the test-set constant baseline. This is a finite-sample test of history readout; compare multiple random seeds as well.
''')
code('''
rng = np.random.default_rng(123)
random_u = rng.random(2000)
memory_conditions = [conditions[0], conditions[2], *conditions[4:]]
delays = np.array([1, 2, 4, 8, 16])
split = 1400
memory_scores = {}
for c in memory_conditions:
    x = optics.features(random_u, c.trajectory(random_u, DT), c.phase_max)
    result = []
    for delay in delays:
        train = np.arange(100, split)
        test = np.arange(split, len(random_u))
        prediction = ridge_predict(x[train], random_u[train-delay], x[test])
        target = random_u[test-delay]
        result.append(1 - np.sum((target-prediction)**2) / np.sum((target-target.mean())**2))
    memory_scores[c.name] = np.array(result)
    print(f'{c.name:<24}', np.round(result, 3))
fig, ax = plt.subplots(figsize=(8, 4))
for name, result in memory_scores.items():
    ax.plot(delays, result, 'o-', label=name)
ax.axhline(0, color='gray', lw=.8); ax.set(xlabel='delay d', ylabel='held-out R²', title='Delayed random-input reconstruction')
ax.legend(fontsize=8); ax.grid(alpha=.2); plt.show()
''')
md(r'''
## 8. Interpreting the results

1. The input-driven model stores an exponentially weighted input history in one scalar thermal state; a fixed spatial Gaussian profile produces many CCD observations, not many independent thermal states.
2. The same-input/different-history test and random-delay reconstruction test whether memory is observable in the optical field and readable by a linear readout, respectively.
3. Adjacent Mackey–Glass samples are strongly correlated; a static map can also predict the next step. Interpret results alongside persistence, instantaneous response, no-heating and free-cooling controls.
4. The same $u_k$ controlling the SLM and heating power is an explicit modeling assumption, not an automatic consequence of a phase-only SLM. Actual absorption, heat diffusion, temperature scale and thermo-optic response need experimental calibration. This example compares one random seed and a fixed ridge parameter.
''')

notebook = {'cells': cells, 'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}, 'language_info': {'name': 'python', 'version': '3'}}, 'nbformat': 4, 'nbformat_minor': 4}
output_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('thermal_free_decay_comparison.ipynb')
output_path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1)+'\n', encoding='utf-8')
