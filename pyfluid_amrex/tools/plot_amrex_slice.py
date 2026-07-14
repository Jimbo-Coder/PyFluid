#!/usr/bin/env python3
"""Plot a 2D slice from a PyFluid AMReX cell-centered plotfile."""

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from amr_plot_data import HierarchySlice, component_values, plot_dimensions
from plot_labels import field_label
from plotfile_reader import (
   full_shape,
   level_chunks,
   level_count,
   parse_fab_index,
   parse_level_cell_sizes,
   parse_level_shapes,
   parse_plot_bounds,
   parse_plot_header,
   read_fab,
   read_level_state,
)


def axis_labels(ndim: int, slice_axis: str) -> tuple[str, str]:
    labels = ("x", "y", "z")
    dim0, dim1 = plot_dimensions(ndim, slice_axis)
    return labels[dim0], labels[dim1]


def composite_slice(
    plotfile: Path,
    component: str,
    gamma: float,
    ref_ratio: int,
    slice_axis: str,
    slice_fraction: float,
    max_pixels: int,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    method: str = "nearest",
) -> tuple[np.ndarray, list[tuple[int, float, float, float, float]], float, int, list[str], tuple[float, float], tuple[float, float]]:
    _ = ref_ratio
    result = HierarchySlice(
        plotfile,
        component,
        gamma,
        slice_axis=slice_axis,
        slice_fraction=slice_fraction,
        xlim=xlim,
        ylim=ylim,
    ).uniform(max_pixels, method=method)
    return (
        result.values,
        list(result.boxes),
        result.time,
        result.ndim,
        list(result.names),
        result.xlim,
        result.ylim,
    )


def parse_limits(values: list[float] | None, default: tuple[float, float]) -> tuple[float, float]:
    if values is None:
        return default
    if len(values) != 2:
        raise ValueError("limits must contain exactly two values")
    lo, hi = values
    if not lo < hi:
        raise ValueError("limits must satisfy lo < hi")
    return lo, hi


def select_component(state: np.ndarray, names: list[str], component: str, gamma: float) -> tuple[np.ndarray, str]:
    return component_values(state, names, component, gamma), field_label(component)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plotfile", required=True, type=Path)
    parser.add_argument("--component", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--title", default="AMReX slice")
    parser.add_argument("--level", type=int, help="debug mode: render only one AMR level instead of the default composite")
    parser.add_argument("--ref-ratio", type=int, default=2)
    parser.add_argument("--max-pixels", type=int, default=1400)
    parser.add_argument(
        "--resample-method",
        choices=("nearest", "linear", "average"),
        default="nearest",
        help="Uniform-view sampling; native AMR movies do not resample.",
    )
    parser.add_argument("--slice-fraction", type=float, default=0.5)
    parser.add_argument("--slice-axis", choices=["x", "y", "z"], default="z")
    parser.add_argument("--slice-index", type=int)
    parser.add_argument("--gamma", type=float, default=5.0 / 3.0)
    parser.add_argument("--xlim", nargs=2, type=float)
    parser.add_argument("--ylim", nargs=2, type=float)
    parser.add_argument("--show-boxes", action="store_true")
    parser.add_argument("--mask-plotfile", type=Path)
    parser.add_argument("--mask-component", default="solid_mask")
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    args = parser.parse_args()

    names, ndim, time_value = parse_plot_header(args.plotfile)
    if ndim < 2:
        raise ValueError("slice plots require a 2D or 3D plotfile")
    problo, probhi = parse_plot_bounds(args.plotfile)
    dim0, dim1 = plot_dimensions(ndim, args.slice_axis)
    xlim = parse_limits(args.xlim, (problo[dim0], probhi[dim0]))
    ylim = parse_limits(args.ylim, (problo[dim1], probhi[dim1]))
    xlabel, ylabel = axis_labels(ndim, args.slice_axis)
    component_label = field_label(args.component)

    composite_slice_fraction = args.slice_fraction
    if args.level is None:
        if not 0.0 <= composite_slice_fraction <= 1.0:
            raise ValueError("--slice-fraction must be in [0, 1]")
        if args.slice_index is not None and ndim == 3:
            finest_shape = parse_level_shapes(args.plotfile)[-1]
            axis = {"x": 0, "y": 1, "z": 2}[args.slice_axis]
            if args.slice_index < 0 or args.slice_index >= finest_shape[axis]:
                raise ValueError(f"--slice-index must be in [0, {finest_shape[axis] - 1}] for the finest level")
            composite_slice_fraction = (args.slice_index + 0.5) / finest_shape[axis]
        image, boxes, time_value, _ndim, _names, xlim, ylim = composite_slice(
            args.plotfile,
            args.component,
            args.gamma,
            args.ref_ratio,
            args.slice_axis,
            composite_slice_fraction,
            args.max_pixels,
            xlim,
            ylim,
            args.resample_method,
        )
        level_label = "composite"
    else:
        state = read_level_state(args.plotfile, names, ndim, args.level)
        values, component_label = select_component(state, names, args.component, args.gamma)
        if ndim == 2:
            image = values.T
        else:
            axis = {"x": 0, "y": 1, "z": 2}[args.slice_axis]
            index = args.slice_index if args.slice_index is not None else values.shape[axis] // 2
            image = np.take(values, index, axis=axis).T
        boxes = []
        level_label = f"L{args.level}"

    mask_image = None
    if args.mask_plotfile is not None:
        mask_names, mask_ndim, _ = parse_plot_header(args.mask_plotfile)
        if mask_ndim != ndim:
            raise ValueError("mask plotfile dimensionality does not match the state plotfile")
        if args.level is None:
            mask_image, _mask_boxes, _mask_time, _mask_ndim, _mask_names, _mask_xlim, _mask_ylim = composite_slice(
                args.mask_plotfile,
                args.mask_component,
                args.gamma,
                args.ref_ratio,
                args.slice_axis,
                composite_slice_fraction,
                args.max_pixels,
                xlim,
                ylim,
                args.resample_method,
            )
        else:
            mask_state = read_level_state(args.mask_plotfile, mask_names, ndim, args.level)
            mask_values, _ = select_component(mask_state, mask_names, args.mask_component, args.gamma)
            if ndim == 2:
                mask_image = mask_values.T
            else:
                mask_slice_axis = {"x": 0, "y": 1, "z": 2}[args.slice_axis]
                mask_index = args.slice_index if args.slice_index is not None else mask_values.shape[mask_slice_axis] // 2
                mask_image = np.take(mask_values, mask_index, axis=mask_slice_axis).T

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.5, 6), constrained_layout=True)
    im = ax.imshow(
        image,
        origin="lower",
        aspect="equal",
        cmap="viridis",
        extent=(xlim[0], xlim[1], ylim[0], ylim[1]),
        interpolation="nearest",
    )
    if mask_image is not None:
        solid = np.ma.masked_less_equal(mask_image, args.mask_threshold)
        ax.imshow(
            solid,
            origin="lower",
            aspect="equal",
            cmap="gray_r",
            alpha=0.85,
            vmin=0.0,
            vmax=1.0,
            extent=(xlim[0], xlim[1], ylim[0], ylim[1]),
            interpolation="nearest",
        )
    if args.show_boxes:
        colors = ["#36c9ff", "#ffd84d", "#ff4d6d", "#ffffff"]
        for lev, xlo, xhi, ylo, yhi in boxes:
            ax.add_patch(
                Rectangle(
                    (xlo, ylo),
                    xhi - xlo,
                    yhi - ylo,
                    fill=False,
                    edgecolor=colors[(lev - 1) % len(colors)],
                    linewidth=1.0,
                    alpha=0.9,
                )
            )
    title = "\n".join(textwrap.wrap(args.title, width=78))
    slice_text = f", {args.slice_axis}={composite_slice_fraction:.2f}" if ndim == 3 and args.level is None else ""
    ax.set_title(f"{title}\n{level_label} {component_label}{slice_text}, t={time_value:.5f}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    fig.colorbar(im, ax=ax, label=component_label)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
