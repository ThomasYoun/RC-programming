# 热扰动散射介质与光学 reservoir computing

本目录包含可运行的 Jupyter notebook、模型代码和 Mackey–Glass 数据。主 notebook 按 `RC_opMGSLM.ipynb` 构建 SLM → objective 1 → 固定随机散射 → 热相位屏 → objective 2 → CCD → 降采样 → 线性读出的光路，对比无热源、静态及即时热扰动、自由冷却和不同热弛豫时间的输入驱动模型。

## 运行

推荐 Python 3.10 或更高版本。下面的命令在克隆仓库后执行（Windows 可将激活虚拟环境的命令换成 `.venv\Scripts\activate`）：

```bash
cd thermal_optical_rc
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m notebook
```

在浏览器中打开 `thermal_free_decay_comparison.ipynb`，选择 **Kernel → Restart Kernel and Run All Cells**。请从本目录启动 Jupyter，因为 notebook 通过相对路径读取同目录下的 `thermal_free_decay.py` 和 `MackeyGlass_t17.txt`。不需要 GPU 或外部下载的数据。

安装 PyTorch 时可能下载较大的软件包。原始示例已在 CPU 上执行；本机已验证的版本为 Python 3.14.4、NumPy 2.4.4、PyTorch 2.11.0、Matplotlib 3.10.8。

## 文件说明

- `thermal_free_decay_comparison.ipynb`：主 notebook，运行后生成光路、speckle 图和 Mackey–Glass 对比；GitHub 版本未保存执行输出。
- `thermal_free_decay.py`：共享固定传播算子、热相位屏、相机特征和 ridge readout。
- `MackeyGlass_t17.txt`：notebook 使用的本地输入数据。
- `thermal_disorder_rc.py`、`run_thermal_rc_demo.py` 和 `THERMAL_RC_MODEL.md`：上一版输入驱动热模型及演示，模型假设与主 notebook 不同。
- `make_thermal_notebook.py` 和 `execute_thermal_notebook.py`：可选的 notebook 生成和运行脚本。生成脚本会覆盖当前目录中的同名 notebook。
- `output/thermal_rc_diagnostics.png`：上一版输入驱动模型的诊断结果。
- `references/`：建模时参考的部分原始 notebook 和项目 PDF。主实验不依赖这些参考文件。

## 模型范围

光路是 `O2 D[k] H O1`：`H` 是固定随机散射，`O1` 和 `O2` 为圆形 pupil，`D[k]` 是散射后有效平面上的高斯热相位屏，占该平面四分之一。整个 SLM 用于输入图案，没有输入与状态拼接；CCD 的降采样强度是当前 reservoir state，不反馈到 SLM。

输入驱动条件采用 `a = exp(-dt/tau)`、`theta[k] = a*theta[k-1] + (1-a)*u[k]`：假设同一输入控制归一化加热功率，热弛豫记录输入历史，不需要电子反馈。自由冷却、静态、即时和无热源条件作为对照。该线性加热关系、热光响应和材料参数尚未经过实验标定；固定空间分布对应单个独立热状态。Mackey–Glass 测试是 teacher-forced 单步预测；应结合 persistence baseline、历史可区分性及随机输入延迟重建一起理解。
