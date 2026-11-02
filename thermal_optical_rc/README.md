# 热扰动散射介质与光学 reservoir computing

本目录包含可运行的 Jupyter notebook、模型代码和 Mackey–Glass 数据。当前主实验为 `optical_physical_memory_rc.ipynb`：整个 SLM 仅编码当前输入，不拼接、不做电子反馈；输入驱动材料热状态，局部热相位改变固定光学传播，CCD speckle 形成读出特征。

## 运行

推荐 Python 3.10 或更高版本。下面的命令在克隆仓库后执行（Windows 可将激活虚拟环境的命令换成 `.venv\Scripts\activate`）：

```bash
cd thermal_optical_rc
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m notebook
```

在浏览器中打开 `optical_physical_memory_rc.ipynb`，选择 **Kernel → Restart Kernel and Run All Cells**。请从本目录启动 Jupyter，因为 notebook 通过相对路径读取同目录下的 `optical_physical_memory.py` 和 `MackeyGlass_t17.txt`。不需要 GPU 或外部下载的数据。

安装 PyTorch 时可能下载较大的软件包。原始示例已在 CPU 上执行；本机已验证的版本为 Python 3.14.4、NumPy 2.4.4、PyTorch 2.11.0、Matplotlib 3.10.8。

## 文件说明

- `optical_physical_memory_rc.ipynb`：当前主 notebook，纯物理热记忆、全 SLM 输入、物镜 pupil、噪声和多组 RC 对照；GitHub 版本未保存执行输出。
- `optical_physical_memory.py`：无电子反馈的光学传播、显式热状态、可配置噪声和 ridge readout。
- `make_physical_memory_notebook.py`：生成上述 notebook 的可选脚本，会覆盖同名 notebook。
- `optical_feedback_thermal_rc.ipynb`：先前的电子反馈对照版本，与当前主实验不同。
- `optical_feedback_thermal.py`：反馈版光学传播、可配置噪声和 ridge readout。
- `make_feedback_notebook.py`：生成上述 notebook 的可选脚本，会覆盖同名 notebook。
- `thermal_free_decay_comparison.ipynb`：较早的无电子反馈 notebook，模型与当前主实验不同。
- `thermal_free_decay.py`：共享固定传播算子、热相位屏、相机特征和 ridge readout。
- `MackeyGlass_t17.txt`：notebook 使用的本地输入数据。
- `thermal_disorder_rc.py`、`run_thermal_rc_demo.py` 和 `THERMAL_RC_MODEL.md`：上一版输入驱动热模型及演示，模型假设与主 notebook 不同。
- `make_thermal_notebook.py` 和 `execute_thermal_notebook.py`：可选的 notebook 生成和运行脚本。生成脚本会覆盖当前目录中的同名 notebook。
- `output/thermal_rc_diagnostics.png`：上一版输入驱动模型的诊断结果。
- `references/`：建模时参考的部分原始 notebook 和项目 PDF。主实验不依赖这些参考文件。

## 模型范围

主实验光路是 `O2 B D[k] A O1`：`A`、`B` 是固定随机散射的两个阶段，`O1` 和 `O2` 为圆形 pupil，`D[k]` 是散射内部有效平面上的高斯热相位屏，support 占该平面约 80%；默认 `thermal_gain=1.3` 表示提高热输入增益。高斯分布并非 support 内均匀加热。整个 SLM 仅编码当前输入，默认 `phase = π u (0.5 + 0.5 M)`，固定随机空间 mask `M` 使所有 SLM 像素连续响应输入。CCD 的降采样强度是观测特征，不反馈至 SLM。这是简化模型，尚未对特定材料标定。

输入驱动条件按 `a = exp(-dt/tau)`、`theta[k+1] = a*theta[k] + beta*u[k]` 演化，读出采用 `r[k] = H(theta[k], u[k])`，没有电子反馈。默认 `beta=(1-a)*thermal_gain`，也可以显式设置 `beta`；同一输入控制加热功率是需要实验支持的假设。pupil 是确定性滤波，模型另提供 pupil 静态波前误差、SLM 相位抖动、CCD 光子散粒噪声和电子读出噪声。本版将示例噪声增大并与前一版比较，这些值未经实验标定。`W_out` 用 ESN 常见的 ridge 回归拟合；Mackey–Glass teacher-forced 单步预测应结合 persistence、随机输入延迟重建和二阶任务一起理解。

时序实验统一使用前 2,000 步作为训练段，其中前 200 步仅用于 washout；随后 1,000 步为测试段。Mackey–Glass 的 ridge 超参数在训练段内部验证，最终拟合不使用测试目标；热状态跨分界连续演化。
