#!/usr/bin/env python3
"""Measure whether per-frame AMReX solid masks move on a fixed display grid."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_amrex_boxed_movie import (  # noqa: E402
    composite_slice,
    default_solid_geometry_file,
    parse_limits,
    parse_plot_bounds,
    parse_solid_boxes,
)
from plotfile_reader import sorted_plotfiles


ROOT = Path(__file__).resolve().parents[2]


def display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(path)


def analytic_solid_mask(
    shape: tuple[int, int],
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    boxes: list[tuple[float, float, float, float]],
    circles: list[tuple[float, float, float]],
) -> np.ndarray:
    ny, nx = shape
    x = xlim[0] + (np.arange(nx, dtype=float) + 0.5) * (xlim[1] - xlim[0]) / nx
    y = ylim[0] + (np.arange(ny, dtype=float) + 0.5) * (ylim[1] - ylim[0]) / ny
    xx, yy = np.meshgrid(x, y)
    solid = np.zeros(shape, dtype=bool)
    for xlo, xhi, ylo, yhi in boxes:
        solid |= (xx >= xlo) & (xx <= xhi) & (yy >= ylo) & (yy <= yhi)
    for cx, cy, radius in circles:
        solid |= (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2
    return solid


def load_solid_geometry_from_metadata(
    path: Path | None,
) -> tuple[list[tuple[float, float, float, float]], list[tuple[float, float, float]]]:
    if path is None or not path.exists():
        return [], []
    boxes: list[tuple[float, float, float, float]] = []
    circles: list[tuple[float, float, float]] = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            dim = int(row["dim"])
            if dim < 2:
                continue
            if row.get("kind") == "box":
                xlo = float(row["xlo"])
                xhi = float(row["xhi"])
                ylo = float(row["ylo"])
                yhi = float(row["yhi"])
                if xlo < xhi and ylo < yhi:
                    boxes.append((xlo, xhi, ylo, yhi))
            elif row.get("kind") == "circle":
                radius = float(row["radius"])
                if radius > 0.0:
                    circles.append((float(row["center_x"]), float(row["center_y"]), radius))
    return boxes, circles


def write_tsv(path: Path, rows: list[dict[str, float]]) -> None:
    fields = [
        "step",
        "time",
        "solid_fraction",
        "changed_fraction_vs_initial",
        "changed_pixels_vs_initial",
        "analytic_mismatch_fraction",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_rows(path: Path, rows: list[dict[str, float]], title: str) -> None:
    t = np.asarray([row["time"] for row in rows], dtype=float)
    solid_fraction = np.asarray([row["solid_fraction"] for row in rows], dtype=float)
    changed = np.asarray([row["changed_fraction_vs_initial"] for row in rows], dtype=float)
    mismatch = np.asarray([row["analytic_mismatch_fraction"] for row in rows], dtype=float)
    markers = {"marker": "o"} if len(rows) <= 32 else {}

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axs = plt.subplots(1, 3, figsize=(14.0, 4.2), constrained_layout=True)
    fig.suptitle(title)
    axs[0].plot(t, solid_fraction, linewidth=2.0, **markers)
    axs[0].set_title("Solid Area")
    axs[0].set_xlabel("time")
    axs[0].set_ylabel("fraction")
    axs[1].semilogy(t, np.maximum(changed, 1.0e-16), linewidth=2.0, **markers)
    axs[1].set_title("Mask Change Vs Initial")
    axs[1].set_xlabel("time")
    axs[1].set_ylabel("changed fraction")
    axs[2].semilogy(t, np.maximum(mismatch, 1.0e-16), linewidth=2.0, **markers)
    axs[2].set_title("Mismatch Vs Analytic Geometry")
    axs[2].set_xlabel("time")
    axs[2].set_ylabel("fraction")
    for ax in axs:
        ax.grid(True, which="major", alpha=0.25)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mask-plot-dir", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "runs/current/solid_mask_stability.tsv")
    parser.add_argument("--plot", type=Path, default=ROOT / "plots/current/scans/solid_mask_stability_018.png")
    parser.add_argument("--title", default="Solid mask stability")
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=5.0 / 3.0)
    parser.add_argument("--ref-ratio", type=int, default=2)
    parser.add_argument("--z-fraction", type=float, default=0.5)
    parser.add_argument("--max-pixels", type=int, default=900)
    parser.add_argument("--xlim", nargs=2, type=float)
    parser.add_argument("--ylim", nargs=2, type=float)
    parser.add_argument("--solid-box-lo", nargs="*", type=float)
    parser.add_argument("--solid-box-hi", nargs="*", type=float)
    parser.add_argument("--solid-geometry-file", type=Path)
    args = parser.parse_args()

    plotfiles = sorted_plotfiles(args.mask_plot_dir, prefix="auxGF", require=True)
    problo, probhi = parse_plot_bounds(plotfiles[0])
    xlim = parse_limits(args.xlim, (problo[0], probhi[0]))
    ylim = parse_limits(args.ylim, (problo[1], probhi[1]))
    boxes = parse_solid_boxes(args.solid_box_lo, args.solid_box_hi)
    circles: list[tuple[float, float, float]] = []
    if not boxes:
        geometry_file = args.solid_geometry_file or default_solid_geometry_file(args.mask_plot_dir)
        boxes, circles = load_solid_geometry_from_metadata(geometry_file)

    initial = None
    analytic = None
    rows: list[dict[str, float]] = []
    for path in plotfiles:
        image, _boxes, time_value, _ndim, _names, _xlim, _ylim = composite_slice(
            path,
            "solid_mask",
            args.gamma,
            args.ref_ratio,
            args.z_fraction,
            False,
            None,
            args.max_pixels,
            xlim,
            ylim,
        )
        mask = image > args.mask_threshold
        if initial is None:
            initial = mask.copy()
        if analytic is None and (boxes or circles):
            analytic = analytic_solid_mask(mask.shape, xlim, ylim, boxes, circles)
        changed = mask != initial
        mismatch_fraction = float(np.mean(mask != analytic)) if analytic is not None else float("nan")
        rows.append(
            {
                "step": float(plotfile_step(path)),
                "time": float(time_value),
                "solid_fraction": float(np.mean(mask)),
                "changed_fraction_vs_initial": float(np.mean(changed)),
                "changed_pixels_vs_initial": float(np.count_nonzero(changed)),
                "analytic_mismatch_fraction": mismatch_fraction,
            }
        )

    write_tsv(args.output, rows)
    plot_rows(args.plot, rows, args.title)
    max_changed = max(row["changed_fraction_vs_initial"] for row in rows)
    max_mismatch = max(
        row["analytic_mismatch_fraction"]
        for row in rows
        if np.isfinite(row["analytic_mismatch_fraction"])
    ) if boxes or circles else float("nan")
    print(f"frames={len(rows)} max_changed_fraction={max_changed:.8e} max_analytic_mismatch={max_mismatch:.8e}")
    print(f"wrote {display_path(args.output)}")
    print(f"wrote {display_path(args.plot)}")


if __name__ == "__main__":
    main()
