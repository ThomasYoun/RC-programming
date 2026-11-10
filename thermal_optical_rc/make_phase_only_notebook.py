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

This notebook starts directly from an imposed optical phase, without a temperature model, a calibrated refractive-index change, or electronic feedback. It checks how a prescribed phase memory and an assumed inverse-radius spatial profile affect a fixed coherent scattering system. It is **not yet a reservoir-computing benchmark** and does not by itself validate a physical thermo-optic material.

Run with `phase_only_optics.py` and `optical_physical_memory.py` in the same directory. Requirements: NumPy, PyTorch, Matplotlib, and Jupyter.
''')
md(r'''
## 1. Phase state and increments

At observation step \(k\), the unwrapped amplitude is \(q_k\) in radians. The direct phase increment \(\delta q_k\) is applied **after** that observation:
\[
q_{k+1}=e^{-\Delta t/\tau}q_k+\delta q_k,\qquad
\phi_k(x,y)=q_kG(x,y).
\]

The first plot shows both the memory \(q_k\) and the injected increments. Time is in arbitrary steps; neither \(q\), the increments, nor \(\tau\) is calibrated to a material. \(q_k\) is kept unwrapped for the update. For visualization only, use \(\phi_{\rm wrapped}=\phi\bmod 2\pi\in[0,2\pi)\). The optical screen is \(D=e^{i\phi}\).
''')
code('''
import numpy as np
import matplotlib.pyplot as plt
import torch
from phase_only_optics import (PhaseOnlyOptics, radial_profile, phase_trace,
                               wrap_phase, intensity_correlation)

torch.set_num_threads(1)
plt.rcParams['figure.dpi'] = 110
n_steps, dt, tau = 80, 1.0, 12.0
increments = np.zeros(n_steps)
increments[0] = 1.4*np.pi
increments[28] = 2.0*np.pi  # crosses 2π so local phase wrapping can be inspected
q, q_next = phase_trace(n_steps, tau, dt, initial_phase=0, increments=increments)
t = np.arange(n_steps)*dt
fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True, constrained_layout=True)
axes[0].plot(t, q/np.pi, label='unwrapped phase amplitude q[k]')
axes[0].plot(t, wrap_phase(q)/np.pi, '--', label='q[k] modulo 2π (display)')
axes[0].set(ylabel='phase / π', title='Direct phase memory: observe first, add increment afterwards')
axes[0].legend(); axes[0].grid(alpha=.25)
axes[1].stem(t, increments/np.pi, basefmt=' ')
axes[1].set(xlabel='time step k', ylabel='increment / π')
axes[1].grid(alpha=.25); plt.show()
print(f'Decay factor per step: {np.exp(-dt/tau):.5f}')
print(f'First increment at k=0 is visible in q[1]={q[1]/np.pi:.2f}π, not q[0]={q[0]:.1f}.')
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
shape = (128, 128)
core_radius, support_fraction = 0.12, 0.8
G = radial_profile(shape, core_radius=core_radius, support_fraction=support_fraction)
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

The color scale is fixed at \(0\) to \(2\pi\). When an unwrapped local phase exceeds \(2\pi\), its displayed phase wraps back to zero; this is not a reset of the underlying phase memory. Different locations wrap at different times because \(G\) is nonuniform.
''')
code('''
snapshot_steps = [0, 1, 10, 29, 50]
fig, axes = plt.subplots(1, len(snapshot_steps), figsize=(15, 3.3), constrained_layout=True)
for ax, k in zip(axes, snapshot_steps):
    im = ax.imshow(wrap_phase(q[k]*G), cmap='twilight', vmin=0, vmax=2*np.pi)
    ax.set_title(f'k={k}, peak={q[k]/np.pi:.2f}π'); ax.axis('off')
fig.colorbar(im, ax=axes, label='wrapped local phase (rad)', shrink=.8); plt.show()
print('Local phase range:', wrap_phase(q[:, None, None]*G).min(), 'to', wrap_phase(q[:, None, None]*G).max(), 'rad')
''')
md(r'''
## 4. Fixed scattering: phase to speckle

The same fixed SLM input, objective pupils, and random scattering realization are used for every frame. The optical path is the existing coherent effective model:
\[
E_{\rm CCD}=O_2\,B\,\mathrm{diag}(e^{i\phi_k})\,A\,O_1E_{\rm SLM},\qquad I_k=|E_{\rm CCD}|^2.
\]
This is a fixed disordered propagation with one inserted phase screen, **not** an exact single-scattering or Maxwell solver. There is no electronic feedback and no changing input in this verification. Noise is disabled so the effect of phase alone is visible. The CCD images share one raw intensity scale; the correlation uses raw images and is computed relative to the zero-phase image.
''')
code('''
optics = PhaseOnlyOptics(seed=42, optical_shape=shape, pupil_radius=(0.25, 0.25))
intensities = np.array([np.abs(optics.field(qk*G))**2 for qk in q])
reference = intensities[0]
correlations = np.array([intensity_correlation(reference, image) for image in intensities])
relative_change = np.array([np.linalg.norm(image-reference)/np.linalg.norm(reference) for image in intensities])
vmax = max(intensities[k].max() for k in snapshot_steps)
fig, axes = plt.subplots(1, len(snapshot_steps), figsize=(15, 3.3), constrained_layout=True)
for ax, k in zip(axes, snapshot_steps):
    im = ax.imshow(intensities[k], cmap='magma', vmin=0, vmax=vmax)
    ax.set_title(f'k={k}, corr={correlations[k]:.3f}'); ax.axis('off')
fig.colorbar(im, ax=axes, label='raw CCD intensity', shrink=.8); plt.show()
fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True, constrained_layout=True)
axes[0].plot(t, correlations); axes[0].set(ylabel='Pearson correlation', title='Speckle response relative to k=0')
axes[1].plot(t, relative_change); axes[1].set(xlabel='time step k', ylabel='relative intensity change')
for ax in axes: ax.grid(alpha=.25)
plt.show()
''')
md(r'''
## 5. Sanity checks: periodic local phase versus peak amplitude

Adding \(2\pi\) to **every local phase** must preserve the complex field and intensity. A spatially uniform global phase also leaves intensity unchanged. By contrast, \(q=2\pi\) with a nonuniform \(G\) does **not** add \(2\pi\) to every pixel: \(2\pi G(x,y)\) is generally nonuniform, so its speckle need not match \(q=0\).
''')
code('''
phase = q[10]*G
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

This verifies numerical phase wrapping, a chosen fading-phase law, and the resulting speckle response of a fixed optical model. It does **not** establish that a real material has a \(1/r\) refractive-index perturbation or exponential phase dynamics. Those require physical assumptions, geometry, calibration, and experimental comparison. The phase-only screen neglects possible absorption changes, diffusion-dependent profile evolution, and full multiple-scattering changes. Once this basic response is understood, an input-dependent increment and a trained linear readout can be added for reservoir-computing tests.
''')

notebook = {'cells': cells, 'metadata': {'kernelspec': {'display_name': 'Python 3',
    'language': 'python', 'name': 'python3'}, 'language_info': {'name': 'python', 'version': '3'}},
    'nbformat': 4, 'nbformat_minor': 4}
output = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('phase_only_verification.ipynb')
output.write_text(json.dumps(notebook, ensure_ascii=False, indent=1)+'\n', encoding='utf-8')
