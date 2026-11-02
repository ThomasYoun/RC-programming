# Local thermal fading memory in a disordered optical reservoir

This implementation combines the useful pieces of the supplied materials:

- fixed random optical mixing and camera speckle;
- fixed-mask threshold encoding of the scalar input;
- the report's first-order thermal relaxation;
- the shared chat's physically interpretable perturbation `T(k) = B D(k) A`;
- a camera-intensity feature vector followed by a ridge-regression readout.

## Model

For each discrete input sample `u[k]`, the lumped temperature state is

```text
gamma      = exp(-dt / tau)
theta[k+1] = gamma theta[k] + (1-gamma) drive_gain u[k].
```

Thus a unit pulse leaves a trace proportional to `gamma**k`.  The spatial
temperature is `theta[k] G(x,y)`, where `G` is Gaussian inside a circular set
containing exactly 25% of the pixels and zero elsewhere.  The induced phase is

```text
delta_phi(x,y,k) = phase_max theta[k] G(x,y),
D(k)              = exp(i delta_phi(x,y,k)).
```

`D=1` outside the support.  This detail matters: multiplying by
`phase_max*G*exp(i phase_max*G)` would incorrectly absorb all light outside the
heated region.  The effective transmission operator is

```text
T(k) = B D(k) A,
```

with `A` and `B` held fixed throughout every comparison.  They are implemented
with fixed random phase screens and orthonormal FFTs rather than a dense matrix.
The camera supplies the nonlinearity through square-law detection.

## Run

```bash
python3 run_thermal_rc_demo.py
```

The script verifies the exact exponential pulse response, verifies the 25%
support, and saves a heater/decay/memory diagnostic figure under `output/`.

## Scientific controls

Compare at least these cases with the same input sequence and the same random
seed: `phase_max=0` (no thermal observability), finite `tau` (fading memory),
and a static/non-decaying perturbation.  Report a held-out temporal prediction
metric and the delay-wise memory curve.  Speckle decorrelation by itself is not
evidence of useful memory.

The model is a controlled first approximation, not a prediction for a specific
material.  `phase_max` combines wavelength, thermo-optic coefficient, effective
heated thickness, and temperature scale.  Those quantities can be separated
later when experimental calibration is available.
