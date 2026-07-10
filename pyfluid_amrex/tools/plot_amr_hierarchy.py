#!/usr/bin/env python3
"""Plot AMReX hierarchy coverage from a plotfile Header."""

from __future__ import annotations

import argparse
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plotfile_reader import (  # noqa: E402
    level_count,
    level_index_boxes,
    parse_level_cell_sizes,
    parse_plot_bounds,
    parse_plot_header,
)


@dataclass(frozen=True)
class LevelBox:
    level: int
    bounds: tuple[tuple[float, float], ...]


def plotfile_level_boxes(plotfile: Path, ndim: int) -> list[LevelBox]:
    prob_lo, _prob_hi = parse_plot_bounds(plotfile)
    cell_sizes = parse_level_cell_sizes(plotfile)
    boxes: list[LevelBox] = []
    for level in range(level_count(plotfile)):
        dx = cell_sizes[level]
        for lo, hi in level_index_boxes(plotfile, level, ndim):
            bounds = tuple(
                (prob_lo[axis] + lo[axis] * dx[axis], prob_lo[axis] + (hi[axis] + 1) * dx[axis])
                for axis in range(ndim)
            )
            boxes.append(LevelBox(level=level, bounds=bounds))
    if not boxes:
        raise ValueError(f"no AMR level boxes found in {plotfile}")
    return boxes


def parse_limits(values: list[float] | None, default: tuple[float, float]) -> tuple[float, float]:
    if values is None:
        return default
    lo, hi = values
    if lo >= hi:
        raise ValueError("limits must satisfy lo < hi")
    return lo, hi


def coverage_image(
    boxes: list[LevelBox],
    ndim: int,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    z_value: float,
    pixels: int,
) -> np.ndarray:
    x_width = xlim[1] - xlim[0]
    y_width = ylim[1] - ylim[0]
    nx = pixels
    ny = max(1, int(round(pixels * y_width / x_width)))
    image = np.full((ny, nx), -1, dtype=int)
    for box in sorted(boxes, key=lambda item: item.level):
        if ndim == 3 and not (box.bounds[2][0] <= z_value <= box.bounds[2][1]):
            continue
        x0, x1 = box.bounds[0]
        y0, y1 = box.bounds[1]
        if x1 <= xlim[0] or x0 >= xlim[1] or y1 <= ylim[0] or y0 >= ylim[1]:
            continue
        ix0 = int(np.floor((max(x0, xlim[0]) - xlim[0]) * nx / x_width))
        ix1 = int(np.ceil((min(x1, xlim[1]) - xlim[0]) * nx / x_width))
        iy0 = int(np.floor((max(y0, ylim[0]) - ylim[0]) * ny / y_width))
        iy1 = int(np.ceil((min(y1, ylim[1]) - ylim[0]) * ny / y_width))
        ix0 = min(max(ix0, 0), nx - 1)
        ix1 = min(max(ix1, ix0 + 1), nx)
        iy0 = min(max(iy0, 0), ny - 1)
        iy1 = min(max(iy1, iy0 + 1), ny)
        image[iy0:iy1, ix0:ix1] = np.maximum(image[iy0:iy1, ix0:ix1], box.level)
    return image


def visible_boxes(
    boxes: list[LevelBox],
    ndim: int,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    z_value: float,
) -> list[LevelBox]:
    result: list[LevelBox] = []
    for box in boxes:
        if ndim == 3 and not (box.bounds[2][0] <= z_value <= box.bounds[2][1]):
            continue
        x0, x1 = box.bounds[0]
        y0, y1 = box.bounds[1]
        if x1 <= xlim[0] or x0 >= xlim[1] or y1 <= ylim[0] or y0 >= ylim[1]:
            continue
        result.append(box)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plotfile", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--title", default="AMR hierarchy")
    parser.add_argument("--z-fraction", type=float, default=0.5)
    parser.add_argument("--pixels", type=int, default=1000)
    parser.add_argument("--draw-boxes", action="store_true")
    parser.add_argument("--box-linewidth", type=float, default=0.9)
    parser.add_argument("--xlim", nargs=2, type=float)
    parser.add_argument("--ylim", nargs=2, type=float)
    args = parser.parse_args()

    _names, ndim, time_value = parse_plot_header(args.plotfile)
    if ndim < 2:
        raise ValueError("hierarchy plots require a 2D or 3D plotfile")
    prob_lo, prob_hi = parse_plot_bounds(args.plotfile)
    xlim = parse_limits(args.xlim, (prob_lo[0], prob_hi[0]))
    ylim = parse_limits(args.ylim, (prob_lo[1], prob_hi[1]))
    z_value = 0.0
    if ndim == 3:
        z_value = prob_lo[2] + args.z_fraction * (prob_hi[2] - prob_lo[2])

    boxes = plotfile_level_boxes(args.plotfile, ndim)
    visible = visible_boxes(boxes, ndim, xlim, ylim, z_value)
    level_counts = {level: 0 for level in sorted({box.level for box in boxes})}
    for box in visible:
        level_counts[box.level] = level_counts.get(box.level, 0) + 1
    max_level = max(level_counts)
    image = coverage_image(boxes, ndim, xlim, ylim, z_value, args.pixels)

    colors = ["#f2f2f2", "#8ecae6", "#219ebc", "#ffb703", "#fb8500", "#d62828", "#6a4c93", "#111111"]
    cmap = ListedColormap(colors[: max_level + 2])
    norm = BoundaryNorm(np.arange(-1.5, max_level + 1.1, 1.0), cmap.N)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.2, 6.8), constrained_layout=True)
    im = ax.imshow(
        image,
        origin="lower",
        extent=(xlim[0], xlim[1], ylim[0], ylim[1]),
        interpolation="nearest",
        cmap=cmap,
        norm=norm,
    )
    if args.draw_boxes:
        for box in visible:
            x0, x1 = box.bounds[0]
            y0, y1 = box.bounds[1]
            ax.add_patch(
                Rectangle(
                    (max(x0, xlim[0]), max(y0, ylim[0])),
                    min(x1, xlim[1]) - max(x0, xlim[0]),
                    min(y1, ylim[1]) - max(y0, ylim[0]),
                    fill=False,
                    edgecolor="black" if box.level <= 1 else "white",
                    linewidth=args.box_linewidth,
                    alpha=0.75,
                )
            )
    counts = ", ".join(f"L{level}:{count}" for level, count in sorted(level_counts.items()))
    title = "\n".join(textwrap.wrap(args.title, width=78))
    z_text = f", z={z_value:.3f}" if ndim == 3 else ""
    ax.set_title(f"{title}\nt={time_value:.5f}{z_text}; boxes {counts}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    cbar = fig.colorbar(im, ax=ax, ticks=np.arange(-1, max_level + 1))
    cbar.ax.set_yticklabels(["none"] + [f"L{level}" for level in range(max_level + 1)])
    cbar.set_label("finest AMR level covering pixel")
    fig.savefig(args.output, dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()
