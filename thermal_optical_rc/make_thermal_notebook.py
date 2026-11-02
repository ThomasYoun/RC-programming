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
# 具有热致 fading memory 的无反馈光学储备池

研究顺序：构建 SLM–objective–散射介质–热相位屏–objective–CCD 光路 → 比较不同热条件下的 speckle → Mackey–Glass 预测 → 检验输入历史是否真正被储存。

**重要模型约定：** 连续输入既生成 SLM 图案，也线性控制归一化加热功率；介质的热弛豫保存过去输入。没有电子反馈，也不拼接上一时刻的 reservoir state。加热功率与输入成正比是第一版简化假设，需要实验标定。

文件：`thermal_free_decay.py` 和 `MackeyGlass_t17.txt` 放在这个 notebook 的同一目录。需要 `numpy`, `torch`, `matplotlib`, Jupyter。
''')
md(r'''
## 1. 数学模型和对照

$$a=e^{-\Delta t/\tau},\quad\theta_k=a\theta_{k-1}+(1-a)u_k,\quad
G_p=\mathbf1_{p\in S}\exp[-r_p^2/(2\sigma^2)],\quad |S|/N=1/4.$$

$$E_\mathrm{SLM}=e^{i\pi\Phi(u_k)},\quad
E_\mathrm{CCD}=O_2D_k H O_1E_\mathrm{SLM},\quad
D_{pp,k}=e^{i\phi_{\max}\theta_kG_p},\quad r_k=\operatorname{downsample}(|E_\mathrm{CCD}|^2).$$

这里 $O_1,O_2$ 分别为两个 objective 的 Fourier pupil，$H$ 是固定的隐式随机散射算子。将 $A=HO_1$、$B=O_2$ 包括在内，仍为 $T_k=BD_kA$。热屏是在固定散射后、第二片 objective 前的**有效平面**，并非样品内部完整微观模型。

当前输入通过固定随机 mask 阈值化，得到 64×64 二值图，并用最近邻扩展到整个 128×128 SLM；没有上下拼接。CCD 得到的 $r_k$ 是当前观测状态，**不反馈**到 SLM。相位为零的位置仍然透光（复振幅为 1）。SLM 场中心 pad/crop 到光学平面时，超出 SLM 的部分才补零。

每个条件使用**同一个** $A,B$、输入 mask、初始热状态和相位系数。$\tau$ 与 $\Delta t$ 使用相同时间单位。`phase_max` 表示单位温升下的最大相位，当前是示例参数而非材料标定。25% 是**内部加热平面**的空间像素比例，并不表示 transmission matrix 的 25% 元素。

| 条件 | 相位 | 热状态 |
|---|---|---|
| 无热 | $\phi_{\max}=0$ | 不影响传播 |
| 静态热扰动 | 有限 | $\theta=0.5$ |
| 即时热响应 | 有限 | $\theta_k=u_k$，无历史记忆 |
| 自由冷却 | 有限 | 单次加热，无输入写入 |
| 输入驱动热记忆 | 有限 | 同一输入，不同 $\tau$ |

固定散射部分参考原 notebook：随机复振幅屏 → 正交 FFT → 随机复振幅屏 → 正交 IFFT。两个 objective 均为 FFT → 圆形 pupil → IFFT；散射屏和输入 mask 在所有条件下保持固定。
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
## 2. 光路各平面

下图展示单个输入在各平面的状态。为了在一般电脑上运行，CCD 图像从 128×128 平均下采样为 16×16。可以在上一单元中调整 `shape`、`slm_shape`、`input_shape` 和 `camera_shape`；较大的 camera state 会增加线性回归的内存和计算量。
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
## 3. 加热形状与时间过程

绘制归一化空间分布和 $\theta_k$。输入在时间 0–39 为 1，之后为 0；输入驱动条件关闭加热后按 $e^{-\Delta t/\tau}$ 指数冷却。自由冷却条件从初始温升 1 开始，完全不受输入影响。高斯在有限 support 外截断；空间上的 1/4 指 support 而不是高斯能量的 1/4。
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
## 4. 固定 probe：逐组观察 speckle

用上一节的加热序列分别写入热状态，同时固定弱 probe 的图案为 $u=0.5$；因此观察期间的图像变化来自热相位屏。这里为诊断目的把 heater 控制与 probe 暂时分开；Mackey–Glass 部分两者由同一个输入控制。所有图共用色标，不对每一帧分别做 min–max 归一化。相关系数描述 speckle 变化，不直接代表记忆容量。
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
## 5. 相同当前输入、不同历史

两条序列在同一时刻使用相同当前输入，但历史分别为全 0 与全 1。无热和即时响应没有历史记忆；输入驱动的热状态应使同一 probe 对应不同 speckle。这个检验不依赖电子反馈。
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
## 6. Mackey–Glass：逐组 one-step prediction

本地 `MackeyGlass_t17.txt` 是用户已有序列。时间顺序划分：前 2000 点训练，后 2000 点测试；训练期间的缩放用于测试。训练目标为 $y_k=u_{k+1}$，测试时每步输入真实 $u_k$（teacher forcing），没有预测输出或 CCD 电子反馈。热状态从训练连续演化到测试，**不在 train/test 之间重置**。

每个读出接收当前 $u_k$ 和当前 speckle 特征，使用同样的 ridge 参数。另列 $\hat y_{k+1}=u_k$ 的 persistence 及只用当前输入的线性 readout。NRMSE 使用测试目标的标准差。为避免分界处目标泄漏，训练的最后一个目标仍在训练时间段内。
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
## 7. 随机输入延迟重建：检查输入历史能否读出

使用独立随机输入，训练单独的线性 readout 预测 $u_{k-d}$；只使用当前 speckle state，不输入历史值。采用连续训练/测试划分并报告测试 $R^2$，负值表示比测试集常数基线更差。它提供对读出历史能力的有限样本检验；应进一步比较多个随机种子。

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
## 8. 如何解释结果

1. 输入驱动模型用单个标量热状态保存输入的指数加权历史；固定的空间高斯 profile 使大量 CCD 像素成为不同观测，而非很多独立热状态。
2. 同一当前输入、不同历史的检验，以及随机延迟重建，分别检查记忆是否在光场中可观测、是否可由线性 readout 读出。
3. Mackey–Glass 相邻样本强相关；静态映射也可能预测下一步。需要与 persistence、即时响应、无热和自由冷却对照一起解释。
4. $u_k$ 同时控制 SLM 与加热功率是显式建模假设，不可从相位 SLM 自动推出；真实吸收、热扩散、温度尺度和热光响应仍需实验标定。此示例只比较一个随机种子和固定 ridge。
''')

notebook = {'cells': cells, 'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}, 'language_info': {'name': 'python', 'version': '3'}}, 'nbformat': 4, 'nbformat_minor': 4}
output_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('thermal_free_decay_comparison.ipynb')
output_path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1)+'\n', encoding='utf-8')
