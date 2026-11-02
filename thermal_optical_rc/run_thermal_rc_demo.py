"""Reproducible smoke test and figures for thermal_disorder_rc.py."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from thermal_disorder_rc import (
    ThermalDisorderRC,
    ThermalParameters,
    linear_memory_curve,
    pulse_decay_check,
)


def main():
    output = Path("output")
    output.mkdir(exist_ok=True)
    model = ThermalDisorderRC(
        optical_shape=(32, 32),
        feature_shape=(12, 12),
        thermal=ThermalParameters(tau=8, phase_max=8.0, area_fraction=0.25),
        ridge_alpha=1e-1,
    )

    simulated, analytic = pulse_decay_check(model)
    error = float((simulated - analytic).abs().max().cpu())
    assert error < 1e-6, f"thermal recursion mismatch: {error}"
    _, diagnostic = model.step(0.5, return_diagnostics=True)
    assert diagnostic["active_fraction"] == 0.25

    delays, r2 = linear_memory_curve(model, n_steps=2000, max_delay=25)
    memory_capacity = float(np.maximum(r2, 0).sum())

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    axes[0].imshow(diagnostic["heater_profile"], cmap="inferno")
    axes[0].set_title("Gaussian heater (25% support)")
    axes[0].axis("off")
    axes[1].plot(simulated.cpu(), label="recursion")
    axes[1].plot(analytic.cpu(), "--", label="analytic")
    axes[1].set(xlabel="time step", ylabel="thermal state", title="Fading thermal trace")
    axes[1].legend(frameon=False)
    axes[2].plot(delays, r2, "o-")
    axes[2].axhline(0, color="0.6", lw=0.8)
    axes[2].set(xlabel="delay", ylabel="$R^2$", title=f"Memory curve (MC={memory_capacity:.2f})")
    fig.tight_layout()
    fig.savefig(output / "thermal_rc_diagnostics.png", dpi=180)
    print(f"active fraction: {diagnostic['active_fraction']:.3f}")
    print(f"decay factor exp(-dt/tau): {model.thermal.decay:.6f}")
    print(f"pulse check max error: {error:.3e}")
    print(f"positive linear memory capacity: {memory_capacity:.3f}")


if __name__ == "__main__":
    main()
