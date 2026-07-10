#!/usr/bin/env python3
"""Plot AMR hierarchy health from a PyFluid AMReX amr_levels.tsv file."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_rows(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if any(value is None or value == "" for value in row.values()):
                continue
            try:
                rows.append({key: float(value) for key, value in row.items()})
            except ValueError:
                continue
    return rows


def level_columns(row: dict[str, float], prefix: str) -> list[str]:
    cols = [name for name in row if name.startswith(prefix)]
    return sorted(cols, key=lambda name: int(name.removeprefix(prefix)))


def directional_dx_columns(row: dict[str, float]) -> list[str]:
    cols = [
        name
        for name in row
        if name.startswith("finest_dx")
        and name != "finest_dx"
        and name.removeprefix("finest_dx").isdigit()
    ]
    return sorted(cols, key=lambda name: int(name.removeprefix("finest_dx")))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--title", default="AMReX AMR hierarchy")
    args = parser.parse_args()

    rows = read_rows(args.input)
    if not rows:
        raise SystemExit(f"no AMR rows found in {args.input}")

    t = np.asarray([row["time"] for row in rows], dtype=float)
    markers = {"marker": "o"} if len(rows) <= 16 else {}
    cell_cols = level_columns(rows[0], "cells_L")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig, axs = plt.subplots(2, 3, figsize=(15.5, 8.0), constrained_layout=True)
    fig.suptitle(args.title)

    for col in cell_cols:
        values = np.asarray([row[col] for row in rows], dtype=float)
        if np.any(values > 0.0):
            axs[0, 0].plot(t, values, label=col.replace("cells_", ""), linewidth=2.0, **markers)
    axs[0, 0].set_title("Cells Per Level")
    axs[0, 0].set_xlabel("time")
    axs[0, 0].set_ylabel("cells")
    axs[0, 0].legend(ncol=2)

    axs[0, 1].plot(t, [row["active_finest_level"] for row in rows], label="active finest", linewidth=2.0, **markers)
    axs[0, 1].plot(t, [row["finest_headroom"] for row in rows], label="headroom", linewidth=2.0, **markers)
    axs[0, 1].plot(t, [row["max_level"] for row in rows], label="max level", linewidth=1.6, linestyle=":")
    axs[0, 1].set_title("Level Headroom")
    axs[0, 1].set_xlabel("time")
    axs[0, 1].set_ylabel("levels")
    axs[0, 1].legend()

    total_budget = np.asarray([row["total_budget_fraction"] for row in rows], dtype=float)
    peak_budget = np.asarray([row["peak_level_budget_fraction"] for row in rows], dtype=float)
    if np.any(total_budget >= 0.0):
        axs[1, 0].plot(t, total_budget, label="total", linewidth=2.0, **markers)
    if np.any(peak_budget >= 0.0):
        axs[1, 0].plot(t, peak_budget, label="peak level", linewidth=2.0, **markers)
    axs[1, 0].axhline(1.0, color="0.2", linestyle="--", linewidth=1.4, label="budget")
    axs[1, 0].set_title("Budget Fraction")
    axs[1, 0].set_xlabel("time")
    axs[1, 0].set_ylabel("fraction")
    axs[1, 0].set_ylim(bottom=0.0)
    axs[1, 0].legend()

    total_cells = np.asarray([max(row.get("total_cells", 0.0), 1.0) for row in rows], dtype=float)
    for col in cell_cols:
        values = np.asarray([row[col] / total for row, total in zip(rows, total_cells)], dtype=float)
        if np.any(values > 0.0):
            axs[1, 1].plot(t, values, label=col.replace("cells_", ""), linewidth=2.0, **markers)
    axs[1, 1].set_title("Cell Fraction Per Level")
    axs[1, 1].set_xlabel("time")
    axs[1, 1].set_ylabel("fraction")
    axs[1, 1].set_ylim(bottom=0.0)
    axs[1, 1].legend(ncol=2)

    if "finest_dx" in rows[0]:
        finest_dx = np.asarray([row["finest_dx"] for row in rows], dtype=float)
        min_feature_length = np.asarray([row["min_feature_length"] for row in rows], dtype=float)
        dx_cols = directional_dx_columns(rows[0])
        if dx_cols:
            for col in dx_cols:
                values = np.asarray([row[col] for row in rows], dtype=float)
                axs[0, 2].semilogy(t, values, label=col, linewidth=2.0, **markers)
            axs[0, 2].semilogy(t, finest_dx, label="max dx", linewidth=1.6, linestyle=":")
        else:
            axs[0, 2].semilogy(t, finest_dx, label="finest dx", linewidth=2.0, **markers)
        if np.any(min_feature_length > 0.0):
            positive_limit = np.where(min_feature_length > 0.0, min_feature_length, np.nan)
            axs[0, 2].semilogy(t, positive_limit, label="min feature length", linewidth=1.6, linestyle="--")
        if "solid_curvature_error" in rows[0] and "solid_curvature_tolerance" in rows[0]:
            curvature_error = np.asarray([row["solid_curvature_error"] for row in rows], dtype=float)
            curvature_tolerance = np.asarray([row["solid_curvature_tolerance"] for row in rows], dtype=float)
            if np.any(curvature_error > 0.0):
                axs[0, 2].semilogy(
                    t,
                    np.where(curvature_error > 0.0, curvature_error, np.nan),
                    label="curve error",
                    linewidth=2.0,
                    **markers,
                )
            if np.any(curvature_tolerance > 0.0):
                axs[0, 2].semilogy(
                    t,
                    np.where(curvature_tolerance > 0.0, curvature_tolerance, np.nan),
                    label="curve tolerance",
                    linewidth=1.6,
                    linestyle="--",
                )
        axs[0, 2].set_title("Resolution Cutoff")
        axs[0, 2].set_xlabel("time")
        axs[0, 2].set_ylabel("length")
        axs[0, 2].legend()

    if "resolution_limited_by_max_level" in rows[0]:
        limited = np.asarray([row["resolution_limited_by_max_level"] for row in rows], dtype=float)
        touches = np.asarray([row["touches_max_level"] for row in rows], dtype=float)
        if np.any(limited > 0.5):
            axs[1, 2].set_facecolor("#fff0f0")
            axs[1, 2].text(
                0.02,
                0.95,
                "under-resolved",
                color="crimson",
                fontweight="bold",
                transform=axs[1, 2].transAxes,
                va="top",
            )
        axs[1, 2].plot(t, touches, label="touches max", linewidth=2.0, **markers)
        axs[1, 2].plot(t, limited, label="max limited", linewidth=2.0, **markers)
        axs[1, 2].set_title("Ceiling Status")
        axs[1, 2].set_xlabel("time")
        axs[1, 2].set_ylabel("flag")
        axs[1, 2].set_ylim(-0.05, 1.05)
        axs[1, 2].legend()
    else:
        axs[0, 2].axis("off")
        axs[1, 2].axis("off")

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
