# 热扰动散射介质与光学 reservoir computing

本目录包含可运行的 Jupyter notebook、模型代码和 Mackey–Glass 数据。当前主实验为 `optical_feedback_thermal_rc.ipynb`：按 `RC_opMGSLM.ipynb` 将上一状态与当前输入上下拼接到 SLM，经两个 objective、带热相位的固定散射介质和 CCD 形成电子反馈；同时比较热记忆、反馈和可开关的物理噪声。

## 运行

推荐 Python 3.10 或更高版本。下面的命令在克隆仓库后执行（Windows 可将激活虚拟环境的命令换成 `.venv\Scripts\activate`）：

```bash
cd thermal_optical_rc
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m notebook
```

在浏览器中打开 `optical_feedback_thermal_rc.ipynb`，选择 **Kernel → Restart Kernel and Run All Cells**。请从本目录启动 Jupyter，因为 notebook 通过相对路径读取同目录下的 `optical_feedback_thermal.py` 和 `MackeyGlass_t17.txt`。不需要 GPU 或外部下载的数据。

安装 PyTorch 时可能下载较大的软件包。原始示例已在 CPU 上执行；本机已验证的版本为 Python 3.14.4、NumPy 2.4.4、PyTorch 2.11.0、Matplotlib 3.10.8。

## 文件说明

- `optical_feedback_thermal_rc.ipynb`：当前主 notebook，SLM 拼接、动态 `W_in`/`W_res`、物镜 pupil、噪声和多组 RC 对照；GitHub 版本未保存执行输出。
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

反馈版光路是 `O2 B D[k] A O1`：`A`、`B` 是固定随机散射的两个阶段，`O1` 和 `O2` 为圆形 pupil，`D[k]` 是散射内部有效平面上的高斯热相位屏，占该平面约十分之一。后续混合传播使 CCD 更大范围发生变化。这是简化模型，尚未对特定材料标定。SLM 上下拼接 state 和 input；CCD 的降采样强度经可调 leak 后反馈至下一步。

输入驱动条件采用 `a = exp(-dt/tau)`、`theta[k] = a*theta[k-1] + (1-a)*u[k]`，假设同一输入另行控制归一化加热功率。系统同时含电子反馈和热记忆；无热反馈、有热反馈和有热无反馈等条件用于区分贡献。pupil 是确定性滤波，模型另提供 pupil 静态波前误差、SLM 相位抖动、CCD 光子散粒噪声和电子读出噪声。`W_out` 用 ESN 常见的 ridge 回归拟合。Mackey–Glass 测试是 teacher-forced 单步预测；应结合 persistence baseline 和随机输入延迟重建一起理解。
