#!/usr/bin/env python3
"""Plot compact scalar diagnostics from a PyFluid AMReX scalars.tsv file."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from validation_common import read_float_tsv


def positive(values: list[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return np.maximum(arr, 1.0e-18)


def relative_change(rows: list[dict[str, float]], key: str) -> list[float]:
    initial = rows[0][key]
    scale = max(abs(initial), 1.0e-300)
    return [abs(row[key] - initial) / scale for row in rows]


def read_rows(path: Path) -> list[dict[str, float]]:
    rows = read_float_tsv(path)
    if not rows:
        raise ValueError(f"no complete scalar rows found in {path}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--title", default="AMReX scalar diagnostics")
    args = parser.parse_args()

    rows = read_rows(args.input)
    t = [row["time"] for row in rows]
    markers = {"marker": "o"} if len(rows) <= 16 else {}
    has_mhd = (
        "magnetic_energy" in rows[0]
        and "divb_l1" in rows[0]
        and (
            any(abs(row["magnetic_energy"]) > 0.0 for row in rows)
            or any(abs(row["divb_l1"]) > 0.0 for row in rows)
        )
    )
    has_diffusion = (
        "diffusion_residual" in rows[0]
        and (
            any(abs(row["diffusion_residual"]) > 0.0 for row in rows)
            or any(abs(row["diffusion_iterations"]) > 0.0 for row in rows)
        )
    )
    has_angular_momentum = (
        "angular_mom1" in rows[0]
        and any(
            abs(row["angular_mom1"]) > 0.0
            or abs(row["angular_mom2"]) > 0.0
            or abs(row["angular_mom3"]) > 0.0
            for row in rows
        )
    )
    has_wall_work = (
        "wall_work" in rows[0]
        and (
            any(abs(row["wall_work"]) > 0.0 for row in rows)
            or ("wall_power" in rows[0] and any(abs(row["wall_power"]) > 0.0 for row in rows))
        )
    )
    has_gravity = (
        "gravity_potential_energy" in rows[0]
        and "gravity_total_energy_rel_drift" in rows[0]
        and (
            any(abs(row["gravity_potential_energy"]) > 0.0 for row in rows)
            or any(abs(row["gravity_total_energy_rel_drift"]) > 0.0 for row in rows)
        )
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    ncols = 2 + int(has_angular_momentum) + int(has_wall_work) + int(has_gravity) + int(has_mhd) + int(has_diffusion)
    fig, axs = plt.subplots(2, ncols, figsize=(4.7 * ncols, 7), constrained_layout=True)
    fig.suptitle(args.title)

    energy_drift_label = "total energy" if has_gravity else "energy"
    energy_value_key = "gravity_total_energy" if has_gravity else "energy"
    axs[0, 0].semilogy(t, positive(relative_change(rows, "mass")), label="mass", linewidth=2.0, **markers)
    axs[0, 0].semilogy(t, positive(relative_change(rows, energy_value_key)), label=energy_drift_label, linewidth=2.0, **markers)
    axs[0, 0].set_title("Mass / Total Energy Change" if has_gravity else "Mass / Energy Change")
    axs[0, 0].set_xlabel("time")
    axs[0, 0].set_ylabel("relative change")
    axs[0, 0].legend()

    axs[0, 1].plot(t, [row["mom1"] for row in rows], label="mom1", linewidth=2.0, **markers)
    axs[0, 1].plot(t, [row["mom2"] for row in rows], label="mom2", linewidth=2.0, **markers)
    axs[0, 1].plot(t, [row["mom3"] for row in rows], label="mom3", linewidth=2.0, **markers)
    axs[0, 1].set_title("Total Momentum")
    axs[0, 1].set_xlabel("time")
    axs[0, 1].legend()

    axs[1, 0].plot(t, [row["min_rho"] for row in rows], label="min rho", linewidth=2.0, **markers)
    axs[1, 0].plot(t, [row["min_pressure"] for row in rows], label="min P", linewidth=2.0, **markers)
    axs[1, 0].set_title("Floors")
    axs[1, 0].set_xlabel("time")
    axs[1, 0].legend()

    axs[1, 1].plot(t, [row["dt"] for row in rows], label="dt", linewidth=2.0, **markers)
    if "sim_time_per_s" in rows[0]:
        axs[1, 1].plot(t, [row["sim_time_per_s"] for row in rows], color="tab:green", label="sim time/s", linewidth=2.0, **markers)
    ax2 = axs[1, 1].twinx()
    ax2.plot(t, [row["mcells_per_s"] for row in rows], color="tab:orange", label="active Mcells/s", linewidth=2.0, **markers)
    if "layout_mcells_per_s" in rows[0]:
        ax2.plot(
            t,
            [row["layout_mcells_per_s"] for row in rows],
            color="tab:red",
            linestyle="--",
            label="layout Mcells/s",
            linewidth=1.8,
            **markers,
        )
    axs[1, 1].set_title("Timestep / Performance")
    axs[1, 1].set_xlabel("time")
    axs[1, 1].set_ylabel("dt, sim time/s")
    ax2.set_ylabel("Mcells/s")
    lines, labels = axs[1, 1].get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    axs[1, 1].legend(lines + lines2, labels + labels2)

    next_col = 2
    if has_angular_momentum:
        col = next_col
        next_col += 1
        axs[0, col].plot(t, [row["angular_mom1"] for row in rows], label="l1", linewidth=2.0, **markers)
        axs[0, col].plot(t, [row["angular_mom2"] for row in rows], label="l2", linewidth=2.0, **markers)
        axs[0, col].plot(t, [row["angular_mom3"] for row in rows], label="l3", linewidth=2.0, **markers)
        axs[0, col].set_title("Angular Momentum")
        axs[0, col].set_xlabel("time")
        axs[0, col].legend()

        angular_mag = [
            np.sqrt(row["angular_mom1"] ** 2 + row["angular_mom2"] ** 2 + row["angular_mom3"] ** 2)
            for row in rows
        ]
        axs[1, col].plot(t, angular_mag, label="|L|", linewidth=2.0, **markers)
        axs[1, col].set_title("Angular Momentum Magnitude")
        axs[1, col].set_xlabel("time")
        axs[1, col].legend()

    if has_wall_work:
        col = next_col
        next_col += 1
        energy0 = rows[0]["energy"]
        energy_change = [row["energy"] - energy0 for row in rows]
        axs[0, col].plot(t, energy_change, label="energy change", linewidth=2.3, zorder=2, **markers)
        axs[0, col].plot(
            t,
            [row["wall_work"] for row in rows],
            label="wall work",
            linewidth=2.4,
            linestyle=":",
            zorder=5,
            **markers,
        )
        axs[0, col].set_title("Moving Wall Work")
        axs[0, col].set_xlabel("time")
        axs[0, col].legend()

        axs[1, col].semilogy(
            t,
            positive([abs(row["energy_minus_wall_work"]) for row in rows]),
            label="E - E0 - work",
            linewidth=2.0,
            **markers,
        )
        if "wall_work_rel_error" in rows[0]:
            axs[1, col].semilogy(
                t,
                positive([row["wall_work_rel_error"] for row in rows]),
                label="relative error",
                linewidth=2.0,
                **markers,
            )
        axs[1, col].set_title("Work Residual")
        axs[1, col].set_xlabel("time")
        axs[1, col].legend()

    if has_gravity:
        col = next_col
        next_col += 1
        axs[0, col].plot(t, [row["energy"] for row in rows], label="fluid", linewidth=2.0, **markers)
        axs[0, col].plot(t, [row["gravity_potential_energy"] for row in rows], label="potential", linewidth=2.0, **markers)
        axs[0, col].plot(t, [row["gravity_total_energy"] for row in rows], label="total", linewidth=2.3, **markers)
        axs[0, col].set_title("Gravity Energy")
        axs[0, col].set_xlabel("time")
        axs[0, col].legend()

        axs[1, col].semilogy(
            t,
            positive([row["gravity_total_energy_rel_drift"] for row in rows]),
            label="total energy",
            linewidth=2.0,
            **markers,
        )
        axs[1, col].set_title("Gravity Total Drift")
        axs[1, col].set_xlabel("time")
        axs[1, col].set_ylabel("relative change")
        axs[1, col].legend()

    if has_mhd:
        col = next_col
        next_col += 1
        magnetic = [row["magnetic_energy"] for row in rows]
        magnetic0 = magnetic[0]
        magnetic_drift = [abs(value - magnetic0) / max(abs(magnetic0), 1.0e-300) for value in magnetic]
        axs[0, col].semilogy(t, positive(magnetic_drift), label="magnetic", linewidth=2.0, **markers)
        axs[0, col].set_title("Magnetic Energy")
        axs[0, col].set_xlabel("time")
        axs[0, col].set_ylabel("relative change")
        axs[0, col].legend()

        axs[1, col].semilogy(t, positive([row["divb_l1"] for row in rows]), label="divb L1", linewidth=2.0, **markers)
        axs[1, col].set_title("Magnetic Constraint")
        axs[1, col].set_xlabel("time")
        axs[1, col].legend()

    if has_diffusion:
        col = next_col
        next_col += 1
        axs[0, col].semilogy(
            t,
            positive([row["diffusion_residual"] for row in rows]),
            label="residual",
            linewidth=2.0,
            **markers,
        )
        axs[0, col].set_title("Diffusion Residual")
        axs[0, col].set_xlabel("time")
        axs[0, col].legend()

        axs[1, col].plot(t, [row["diffusion_iterations"] for row in rows], label="SOR sweeps", linewidth=2.0, **markers)
        axs[1, col].set_title("Implicit Diffusion")
        axs[1, col].set_xlabel("time")
        axs[1, col].legend()

    for ax in axs.flat:
        ax.grid(True, which="major", alpha=0.25)
        try:
            ax.ticklabel_format(useOffset=False)
        except AttributeError:
            pass
    fig.savefig(args.output, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
