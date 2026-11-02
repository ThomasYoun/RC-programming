# 热扰动散射介质与光学 reservoir computing

本目录包含可运行的 Jupyter notebook、模型代码和 Mackey–Glass 数据。主 notebook 比较固定随机传播算子下的无热源、静态热扰动及不同热弛豫时间，并在同一组数据上测试单步预测。

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

- `thermal_free_decay_comparison.ipynb`：主 notebook，包含已运行结果、speckle 图和 Mackey–Glass 对比。
- `thermal_free_decay.py`：共享固定传播算子、热相位屏、相机特征和 ridge readout。
- `MackeyGlass_t17.txt`：notebook 使用的本地输入数据。
- `thermal_disorder_rc.py`、`run_thermal_rc_demo.py` 和 `THERMAL_RC_MODEL.md`：上一版输入驱动热模型及演示，模型假设与主 notebook 不同。
- `make_thermal_notebook.py` 和 `execute_thermal_notebook.py`：可选的 notebook 生成和运行脚本。生成脚本会覆盖当前目录中的同名 notebook。
- `output/thermal_rc_diagnostics.png`：上一版输入驱动模型的诊断结果。

## 模型范围

主 notebook 假设样品在初始时刻之前加热一次，之后只自由冷却：`theta[k] = theta[0] exp(-k dt/tau)`。在固定的 `A` 和 `B` 之间插入高斯相位屏，即 `T[k] = B D[k] A`；空间 support 占中间平面的四分之一。它是简化模型，尚未对特定材料标定。

主实验没有电子反馈，也没有通过输入控制加热。因此热状态随时间衰减，但不能储存输入序列的历史。Mackey–Glass 测试是 teacher-forced 单步预测；应与 notebook 中的 persistence baseline 一起理解，不能单凭该结果证明热记忆 RC。
