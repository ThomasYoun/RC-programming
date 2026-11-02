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
# 光学反馈 + 热致动态传输矩阵：带物理噪声的 RC

此版本按原 `RC_opMGSLM.ipynb` 的光路，将 **上一时刻 CCD state 和当前 input image 上下拼接到 SLM**。因此它同时包含电子反馈和热记忆。所有随机散射屏只初始化一次；(W_{\rm in}) 和 (W_{\rm res}) 的变化由同一个热相位屏产生。

需要同一目录下的 `optical_feedback_thermal.py` 与 `MackeyGlass_t17.txt`，依赖 NumPy、PyTorch、Matplotlib、Jupyter。温升、相位、噪声与时间参数均为可调示例，未对真实装置标定。
''')
md(r'''
## 1. 光路和状态更新

\[
u_k\;\xrightarrow{\text{fixed mask}}\;\Phi_k,\qquad
E_{\mathrm{SLM},k}=\exp\left(i\pi\begin{bmatrix}r_k\\\Phi_k\end{bmatrix}\right).
\]

两幅图分别扩展后居中放入 SLM 的上、下半区；相位为零的 SLM 区域仍透光。复光场中心 pad/crop 至光学平面，再经过

\[
E_{\rm CCD,k}=O_2\,B\,D(\theta_k)\,A\,O_1 E_{\rm SLM,k},\qquad
r_{k+1}=(1-\alpha)r_k+\alpha\,\operatorname{camera}(|E_{\rm CCD,k}|^2).
\]

`camera` 包含噪声、固定增益、下采样与截断；这里使用标准 leaky-update 的凸组合，\(\alpha=1\) 时不额外保留数字 leak，仍有通过 SLM 的光学反馈。\(A,B\) 是固定的两段随机散射，\(O_1,O_2\) 是 objective 的 Fourier pupil。热屏插在两段散射之间：

\[
a=e^{-\Delta t/\tau},\quad \theta_k=a\theta_{k-1}+(1-a)u_k,\quad
D_{pp}(\theta_k)=e^{i\phi_{\max}\theta_k G_p}.
\]

假设同一输入**另行控制**归一化加热功率；这不是仅有相位调制便必然发生的结果。高斯 (G_p) 在约 10% 的内部平面上非零，其余位置 (D=1)。测量噪声是 CCD state 的一部分，会进入反馈。

记上、下 SLM 区域投影为 \(P_{\rm res},P_{\rm in}\)，则线性复光场传播的两个有效块为
\[
W_{\rm res}(k)=O_2BD(\theta_k)AO_1P_{\rm res},\quad
W_{\rm in}(k)=O_2BD(\theta_k)AO_1P_{\rm in}.
\]
它们不是每步重新抽样的随机矩阵，也没有存储为稠密矩阵；当前输入、状态先经过 SLM 的指数相位编码，CCD 再平方律探测。动态 \(W\) 对应的是复光场层面的有效映射。
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
## 2. 物镜 pupil 与物理噪声

有限 pupil 是确定性的空间频率滤波，**不是噪声**。这里分别提供：

- `pupil_rms_rad`：两个 pupil 内固定的 defocus/astigmatism 波前误差（弧度 RMS）；
- `slm_phase_std_rad`：逐帧 SLM 相位抖动标准差；
- `shot_noise` / `photon_scale`：按 CCD 期望光子数采样 Poisson 散粒噪声；
- `read_noise_electrons`：CCD 电子读出噪声标准差。

噪声在相机固定增益和下采样之前加入；不对每帧做 min–max 归一化。所有值是示例，尤其 `photon_scale` 不是已知光功率。改变 `common` 中的 pupil radius、分辨率、增益和泄漏率即可扫描。
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
## 3. 两个有效传播块是否同时随温度变化？

使用同一组随机器件、关闭逐帧噪声，对同一个测试复场分别只保留上下半屏；比较 \(\theta=0\) 和 \(\theta=1\) 时 CCD 的**复光场**。这是检验两个有效算子变化的示例，不是完整 transmission matrix 的测量。
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
## 4. Ridge readout，与 ESN 的输出训练方式相同

用 (r_{k+1}) 展开成向量，训练数据标准化后加 bias，得到行向量 \(z_k=[1,\widetilde r_{k+1}^T]\)。训练目标为 \(y_k=u_{k+1}\)：

\[
W_{\rm out}=(Z^TZ+\lambda P)^{-1}Z^TY,\quad P=\mathrm{diag}(0,1,\ldots,1),\quad \hat Y=ZW_{\rm out}.
\]

仅训练 \(W_{\rm out}\)，bias 不正则化。代码通过线性方程求解，训练集拟合特征标准化，验证集从固定候选中选 \(\lambda\)，再用训练+验证段拟合；测试段不参与任何拟合。预测为 teacher-forced：测试时持续输入真实 \(u_k\)。
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
## 5. 记忆的对照：独立随机输入延迟重建

Mackey–Glass 相邻点高度相关，静态映射或 persistence 也可能预测一步；低误差本身不能证明 fading memory。下面用独立随机序列训练不同 readout，以当前 CCD state 重建过去的 \(u_{k-d}\)。训练/测试按时间连续划分；\(R^2<0\) 表示比测试均值基线更差。

无热 + feedback 可以有**电子反馈记忆**；thermal + no feedback（\(\alpha=1\)）用于观察热记忆；thermal + feedback 包含两者。一个标量 \(\theta\) 仍只表示一个独立热状态。要评估统计稳定性，应再更换多个随机种子、噪声水平与时间常数。
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
