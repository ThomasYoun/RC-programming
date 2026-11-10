"""Generate the phase-only optical perturbation verification notebook."""
import json
from pathlib import Path
import sys
import textwrap

cells = []

def md(text):
    cells.append({'cell_type': 'markdown', 'metadata': {},
                  'source': textwrap.dedent(text).strip().splitlines(keepends=True)})

def code(text):
    cells.append({'cell_type': 'code', 'metadata': {}, 'execution_count': None,
                  'outputs': [], 'source': textwrap.dedent(text).strip().splitlines(keepends=True)})

md(r'''
# Phase-only verification of a dynamic scattering medium

This notebook starts directly from an initially random, fading optical phase map and an externally driven phase increment, without a temperature model, a calibrated refractive-index change, or electronic feedback. It checks how their sum affects a fixed coherent scattering system. It is **not yet a reservoir-computing benchmark** and does not by itself validate a physical thermo-optic material.

Run with `phase_only_optics.py` and `optical_physical_memory.py` in the same directory. Requirements: NumPy, PyTorch, Matplotlib, and Jupyter.
''')
md(r'''
## 1. Phase state and increments

At observation step \(k\), the entire unwrapped phase map \(\Phi_k(x,y)\) is observed. The external signal \(u_k\in[0,1]\) writes a direct increment \(\delta q_k=2\pi u_k\) **after** the observation:
\[
\Phi_{k+1}(x,y)=a\Phi_k(x,y)+2\pi u_kG(x,y),\qquad a=e^{-\Delta t/\tau}.
\]

Here \(\Phi_0(x,y)\) is sampled once, independently per pixel, from \(U[0,6\pi)\), interpreted as an **unwrapped initial perturbation relative to zero**. The maximum \(6\pi\) is an adjustable illustration, not a calibrated value. The corresponding wrapped phase alone would not determine its later decay: initial values differing by \(2\pi\) have the same optical phase initially but different later unwrapped states. This chosen distribution and exponential decay are assumptions, not a unique physical reconstruction.

We separate \(\Phi_k=a^k\Phi_0+q_kG\), with \(q_0=0\) and \(q_{k+1}=aq_k+2\pi u_k\). Both initial and driven components use the same \(\tau\) in this simple example. Time, phase gain, and \(\tau\) are uncalibrated. The increment spans \([0,2\pi]\) at the maximum of \(G\); the local increment is \(2\pi u_kG(x,y)\). State evolution remains unwrapped; wrapped maps display the phase modulo \(2\pi\). The optical screen is \(D_k=e^{i\Phi_k}\).
''')
code('''
import numpy as np
import matplotlib.pyplot as plt
import torch
from phase_only_optics import (PhaseOnlyOptics, radial_profile, phase_trace,
                               phase_map_trace, wrap_phase, intensity_correlation)

torch.set_num_threads(1)
plt.rcParams['figure.dpi'] = 110
n_steps, dt, tau = 80, 1.0, 12.0
shape = (128, 128)
core_radius, support_fraction = 0.12, 0.8
G = radial_profile(shape, core_radius=core_radius, support_fraction=support_fraction)
initial_phase_max = 6*np.pi  # adjustable upper limit of the unwrapped initial phase
initial_phase = np.random.default_rng(2026).uniform(0, initial_phase_max, size=shape)
u = np.zeros(n_steps)
rng = np.random.default_rng(7)
u[4:25] = rng.uniform(0, 1, 21)
u[35:55] = rng.uniform(0, 1, 20)
beta = 2*np.pi  # u=0 -> 0 rad; u=1 -> 2π rad at the peak of G
increments = beta*u
q, q_next = phase_trace(n_steps, tau, dt, initial_phase=0, increments=increments)
total_phase, initial_decay, driven_phase = phase_map_trace(initial_phase, increments, G, tau, dt)
t = np.arange(n_steps)*dt
pixel = (shape[0]//2, shape[1]//2)
fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True, constrained_layout=True)
axes[0].step(t, u, where='post', label='external input u[k]')
axes[0].set(ylabel='u[k]', title='External input writes phase after the current observation')
axes[1].stem(t, increments/np.pi, basefmt=' ')
axes[1].set(ylabel='increment / π')
axes[2].plot(t, initial_decay[:, pixel[0], pixel[1]]/np.pi, label='decaying initial phase')
axes[2].plot(t, driven_phase[:, pixel[0], pixel[1]]/np.pi, label='input-driven phase')
axes[2].plot(t, total_phase[:, pixel[0], pixel[1]]/np.pi, label='total unwrapped phase')
axes[2].plot(t, wrap_phase(total_phase[:, pixel[0], pixel[1]])/np.pi, '--', label='total modulo 2π')
axes[2].set(xlabel='time step k', ylabel='phase / π', title='Example pixel near screen center')
for ax in axes: ax.grid(alpha=.25)
axes[0].legend(); axes[2].legend(ncol=2); plt.show()
print(f'Decay factor per step: {np.exp(-dt/tau):.5f}')
print(f'Initial phase range: {initial_phase.min():.4f} to {initial_phase.max():.4f} rad')
print(f'First nonzero input at k=4: q[4]={q[4]:.3f}, q[5]={q[5]:.3f} rad')
''')
md(r'''
## 2. An inverse-radius phase profile

Let \(r\) denote distance from the center of the phase-screen plane, **not** distance between a source and detector. We assume the normalized effective perturbation
\[
G(r)\propto\frac{1}{\sqrt{r^2+r_c^2}},\qquad \max G=1.
\]
The finite core \(r_c\) avoids a singularity. We retain the previous approximate **80% screen support**; outside it \(G=0\). The hard support edge is a modeling choice, not a prediction of heat diffusion. Coordinates and core size are dimensionless. Here \(1/r\) describes an assumed **projected phase perturbation**, not the absolute refractive index or an optical scattering Green function. Deriving a projected phase from a three-dimensional \(\Delta n\) requires integrating along the actual light paths.
''')
code('''
coords = np.linspace(-1, 1, shape[1])
fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
im = axes[0].imshow(G, origin='lower', extent=[-1, 1, -1, 1], cmap='inferno', vmin=0, vmax=1)
axes[0].set(title='Normalized inverse-radius profile G', xlabel='x', ylabel='y')
fig.colorbar(im, ax=axes[0], label='G')
axes[1].plot(coords, G[shape[0]//2])
axes[1].set(title='Central profile (hard support edge)', xlabel='x', ylabel='G')
axes[1].grid(alpha=.25); plt.show()
print(f'Actual discrete support: {np.count_nonzero(G)/G.size:.2%}')
''')
md(r'''
## 3. Spatial phase snapshots and the \(2\pi\) range

Each column is the same time step; rows separately show the fading random initial phase, the input-driven phase, and their total. First, unwrapped maps show the actual phase magnitudes decreasing or accumulating, on a shared scale with \(\pi\) ticks. The second figure shows the same maps modulo \(2\pi\), with a fixed \(0\) to \(2\pi\) color scale. Wrapping is only for display and never resets the underlying state. The initial random map extends across the plane, while the input-driven phase is limited to the chosen 80% support.
''')
code('''
snapshot_steps = [0, 10, 25, 45, 79]
components = [('initial decay', initial_decay), ('input-driven', driven_phase), ('total', total_phase)]
upper_pi = int(np.ceil(max(np.max(maps[snapshot_steps]) for _, maps in components)/np.pi))
tick_step = max(1, int(np.ceil(upper_pi/6)))
upper_pi = int(np.ceil(upper_pi/tick_step)*tick_step)
fig, axes = plt.subplots(3, len(snapshot_steps), figsize=(15, 8), constrained_layout=True)
for row, (label, maps) in enumerate(components):
    for col, k in enumerate(snapshot_steps):
        ax = axes[row, col]
        im = ax.imshow(maps[k], cmap='viridis', vmin=0, vmax=upper_pi*np.pi)
        ax.set_title(f'{label}, k={k}', fontsize=9); ax.axis('off')
pi_ticks = np.arange(0, upper_pi+1, tick_step)
cbar = fig.colorbar(im, ax=axes, ticks=pi_ticks*np.pi, label='unwrapped phase (rad)', shrink=.7)
cbar.ax.set_yticklabels([f'{v:g}π' for v in pi_ticks]); plt.show()

fig, axes = plt.subplots(3, len(snapshot_steps), figsize=(15, 8), constrained_layout=True)
for row, (label, maps) in enumerate(components):
    for col, k in enumerate(snapshot_steps):
        ax = axes[row, col]
        im = ax.imshow(wrap_phase(maps[k]), cmap='twilight', vmin=0, vmax=2*np.pi)
        ax.set_title(f'{label}, k={k}', fontsize=9); ax.axis('off')
cbar = fig.colorbar(im, ax=axes, ticks=np.arange(5)*np.pi/2, label='wrapped local phase (rad)', shrink=.7)
cbar.ax.set_yticklabels(['0', 'π/2', 'π', '3π/2', '2π']); plt.show()
print('Wrapped total phase range:', wrap_phase(total_phase).min(), 'to', wrap_phase(total_phase).max(), 'rad')
''')
md(r'''
## 4. Fixed scattering: phase to speckle

The same fixed SLM probe, objective pupils, and random scattering realization are used for every frame. The random initial phase is a separate, **decaying perturbation**; the underlying scattering operators remain fixed. Here the external input drives the phase increment directly, not the SLM probe. The optical path is the existing coherent effective model:
\[
E_{\rm CCD}=O_2\,B\,\mathrm{diag}(e^{i\Phi_k})\,A\,O_1E_{\rm SLM},\qquad I_k=|E_{\rm CCD}|^2.
\]
This is a fixed disordered propagation with one inserted phase screen, **not** an exact single-scattering or Maxwell solver. There is no electronic feedback. Noise is disabled so the effect of phase alone is visible. The CCD images share one raw intensity scale. A no-input control uses the same initial phase and optical operators but sets every increment to zero. Both controls share the exact same initial frame.
''')
code('''
optics = PhaseOnlyOptics(seed=42, optical_shape=shape, pupil_radius=(0.25, 0.25))
intensities = np.array([np.abs(optics.field(phase))**2 for phase in total_phase])
no_drive_intensities = np.array([np.abs(optics.field(phase))**2 for phase in initial_decay])
reference = intensities[0]
correlations = np.array([intensity_correlation(reference, image) for image in intensities])
control_correlations = np.array([intensity_correlation(reference, image) for image in no_drive_intensities])
drive_change = np.array([np.linalg.norm(image-control)/np.linalg.norm(control) for image, control in zip(intensities, no_drive_intensities)])
vmax = max(intensities[k].max() for k in snapshot_steps)
fig, axes = plt.subplots(1, len(snapshot_steps), figsize=(15, 3.3), constrained_layout=True)
for ax, k in zip(axes, snapshot_steps):
    im = ax.imshow(intensities[k], cmap='magma', vmin=0, vmax=vmax)
    ax.set_title(f'k={k}, corr={correlations[k]:.3f}'); ax.axis('off')
fig.colorbar(im, ax=axes, label='raw CCD intensity', shrink=.8); plt.show()
fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True, constrained_layout=True)
axes[0].plot(t, correlations, label='with input-driven increment')
axes[0].plot(t, control_correlations, '--', label='initial phase decay only')
axes[0].set(ylabel='Pearson correlation', title='Speckle response relative to the common initial frame')
axes[0].legend()
axes[1].plot(t, drive_change); axes[1].set(xlabel='time step k', ylabel='relative change vs no-input')
for ax in axes: ax.grid(alpha=.25)
plt.show()
''')
md(r'''
## 5. Sanity checks: periodic local phase versus peak amplitude

Adding \(2\pi\) to **every local phase** must preserve the complex field and intensity. A spatially uniform global phase also leaves intensity unchanged. By contrast, \(q=2\pi\) with a nonuniform \(G\) does **not** add \(2\pi\) to every pixel: \(2\pi G(x,y)\) is generally nonuniform, so its speckle need not match \(q=0\).
''')
code('''
phase = total_phase[10]
field = optics.field(phase)
field_2pi = optics.field(phase+2*np.pi)
zero = optics.field(np.zeros(shape))
global_phase = optics.field(np.ones(shape)*1.3)
peak_2pi = optics.field(2*np.pi*G)
print('Max |exp(iφ)| deviation from 1:', np.max(np.abs(np.abs(np.exp(1j*phase))-1)))
print('Relative field error after local +2π:', np.linalg.norm(field_2pi-field)/np.linalg.norm(field))
print('Relative intensity change for uniform phase:', np.linalg.norm(np.abs(global_phase)**2-np.abs(zero)**2)/np.linalg.norm(np.abs(zero)**2))
print('Relative intensity change for peak q=2π:', np.linalg.norm(np.abs(peak_2pi)**2-np.abs(zero)**2)/np.linalg.norm(np.abs(zero)**2))
assert np.allclose(np.exp(1j*(phase+2*np.pi)), np.exp(1j*phase), atol=1e-6)
assert np.allclose(np.abs(global_phase)**2, np.abs(zero)**2, rtol=1e-4, atol=1e-7)
''')
md(r'''
## Interpretation and limitations

This verifies numerical phase wrapping, a chosen fading random phase, an input-driven phase increment, and the resulting speckle response of a fixed optical model. It does **not** establish that a real material has a \(1/r\) refractive-index perturbation or exponential phase dynamics. Those require physical assumptions, geometry, calibration, and experimental comparison. The phase-only screen neglects possible absorption changes, diffusion-dependent profile evolution, and full multiple-scattering changes. There is no trained readout or RC benchmark in this phase verification; that can be added after studying this response.
''')

notebook = {'cells': cells, 'metadata': {'kernelspec': {'display_name': 'Python 3',
    'language': 'python', 'name': 'python3'}, 'language_info': {'name': 'python', 'version': '3'}},
    'nbformat': 4, 'nbformat_minor': 4}
output = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('phase_only_verification.ipynb')
output.write_text(json.dumps(notebook, ensure_ascii=False, indent=1)+'\n', encoding='utf-8')
