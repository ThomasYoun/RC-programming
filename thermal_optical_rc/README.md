# Thermally Perturbed Scattering Media for Optical Reservoir Computing

This directory contains runnable Jupyter notebooks, model code, and Mackey–Glass data. The main experiment is `optical_physical_memory_rc.ipynb`: the entire SLM encodes only the current input, with no concatenation or electronic feedback. The input drives the material's thermal state, a local thermal phase perturbation modifies the fixed optical propagation, and the CCD speckle provides readout features.

## Running the experiment

Python 3.10 or later is recommended. Run the following commands after cloning the repository (on Windows, replace the virtual environment activation command with `.venv\Scripts\activate`):

```bash
cd thermal_optical_rc
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m notebook
```

Open `optical_physical_memory_rc.ipynb` in the browser and select **Kernel → Restart Kernel and Run All Cells**. Start Jupyter from this directory because the notebook uses relative paths to read `optical_physical_memory.py` and `MackeyGlass_t17.txt` from the same directory. No GPU or externally downloaded data is required.

Installing PyTorch may involve downloading large packages. The original example was run on a CPU; the locally verified versions are Python 3.14.4, NumPy 2.4.4, PyTorch 2.11.0, and Matplotlib 3.10.8.

## File descriptions

- `ks_phase_memory_rc.ipynb`: KS field reconstruction and teacher-forced multi-horizon forecast from one measured spatial point, using the phase-only model with τ = 2, 5, 10, 50, 100. The other 63 grid points are targets, not inputs. This is a one-sensor observer feasibility test, not a reproduction of the cited paper's multi-input experiments. The data are generated at runtime, so there is no external KS data download.
- `ks_data.py`, `test_ks_data.py`, and `make_ks_phase_rc_notebook.py`: de-aliased Fourier ETDRK4 KS data generator, numerical tests, and notebook generator. Run `python -m unittest -v test_ks_data`.
- `phase_only_verification.ipynb`: an initially random unwrapped phase in [0, 6π) fades while input in [0, 1] writes peak phase increments of 2πu with an assumed regularized inverse-radius profile; unwrapped and 2π-wrapped maps use π-labeled colorbars. This is an effective phase-only experiment, not a calibrated temperature or refractive-index model.
- `phase_only_optics.py`, `make_phase_only_notebook.py`, and `test_phase_only.py`: phase-only model, notebook generator, and unit tests. Run the tests from this directory with `python -m unittest -v test_phase_only`.
- `phase_only_rc_tests.ipynb`: Mackey–Glass teacher-forced and free-running prediction, independent-input delay reconstruction and nonlinear tasks for the input-driven phase-only model, including τ = 2, 5, 10, 50, 100 and instant, static, persistence, noisy, and ideal controls. Free prediction explicitly feeds the digital readout back as input, clips feedback to [0, 1], and does not use true test inputs. With a fixed 2π per-unit input increment, changing τ also changes the steady-state phase magnitude. The notebook reports initial-state retention and phase drift: the 200-step washout is insufficient for τ=100. The fixed SLM probe does not encode the input directly. The first 2,000 input steps include 200 washout steps; the remaining 1,000 are the test segment.
- `phase_only_rc.py`, `make_phase_only_rc_notebook.py`, `test_phase_only_rc.py`, `test_phase_free_rollout.py`, and `test_phase_tau_sweep.py`: streaming RC simulation, free rollout, notebook generator, and tests. Run `python -m unittest -v test_phase_only_rc test_phase_free_rollout test_phase_tau_sweep`.
- `optical_physical_memory_rc.ipynb`: the main notebook, covering purely physical thermal memory, full-SLM input, objective pupils, noise, and multiple RC comparisons. The GitHub version does not contain saved execution outputs.
- `optical_physical_memory.py`: optical propagation without electronic feedback, an explicit thermal state, configurable noise, and ridge readout.
- `make_physical_memory_notebook.py`: optional script that generates the notebook above and overwrites the notebook of the same name.
- `optical_feedback_thermal_rc.ipynb`: an earlier electronic-feedback comparison, distinct from the current main experiment.
- `optical_feedback_thermal.py`: optical propagation, configurable noise, and ridge readout for the feedback version.
- `make_feedback_notebook.py`: optional script that generates the notebook above and overwrites the notebook of the same name.
- `thermal_free_decay_comparison.ipynb`: an earlier notebook without electronic feedback, using a different model from the current main experiment.
- `thermal_free_decay.py`: shared fixed propagation operators, thermal phase screen, camera features, and ridge readout.
- `MackeyGlass_t17.txt`: local input data used by the notebooks.
- `thermal_disorder_rc.py`, `run_thermal_rc_demo.py`, and `THERMAL_RC_MODEL.md`: an earlier input-driven thermal model and demonstration, with assumptions different from the main notebook.
- `make_thermal_notebook.py` and `execute_thermal_notebook.py`: optional notebook generation and execution scripts. The generation script overwrites the notebook of the same name in the current directory.
- `output/thermal_rc_diagnostics.png`: diagnostic results for the earlier input-driven model.
- `references/`: some original notebooks and the project PDF used as modeling references. The main experiment does not depend on these files.

## Model scope

The optical path in the main experiment is `O2 B D[k] A O1`: `A` and `B` are two stages of fixed random scattering, `O1` and `O2` are circular pupils, and `D[k]` is a Gaussian thermal phase screen on an effective plane inside the scattering medium, with a support covering approximately 80% of that plane. The default `thermal_gain=1.3` increases the thermal input gain. The Gaussian profile does not represent uniform heating within the support. The entire SLM encodes only the current input, with the default `phase = π u (0.5 + 0.5 M)`. The fixed random spatial mask `M` makes every SLM pixel respond continuously to the input. Downsampled CCD intensity provides the observed features and is not fed back to the SLM. This simplified model has not been calibrated to a specific material.

The input-driven condition evolves according to `a = exp(-dt/tau)` and `theta[k+1] = a*theta[k] + beta*u[k]`, with readout features `r[k] = H(theta[k], u[k])` and no electronic feedback. By default, `beta=(1-a)*thermal_gain`; `beta` can also be set explicitly. The assumption that the same input controls heating power requires experimental support. Pupils provide deterministic filtering; the model also includes static pupil wavefront error, SLM phase jitter, CCD photon shot noise, and electronic readout noise. This version increases the example noise and compares it with the previous version; these values have not been experimentally calibrated. `W_out` is fitted using ridge regression, as commonly done in ESNs. Teacher-forced one-step Mackey–Glass predictions should be interpreted alongside persistence, random-input delay reconstruction, and second-order tasks.

The time-series experiments consistently use the first 2,000 steps as the training segment, with the first 200 steps used only for washout, followed by 1,000 test steps. Mackey–Glass ridge hyperparameters are validated within the training segment, and test targets are excluded from the final fit. The thermal state evolves continuously across the boundary.
