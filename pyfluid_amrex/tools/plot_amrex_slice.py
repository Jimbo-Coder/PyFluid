#!/usr/bin/env python3
"""Plot a 2D slice from a PyFluid AMReX cell-centered plotfile."""

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

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


def canvas_shape(xlim: tuple[float, float], ylim: tuple[float, float], max_pixels: int) -> tuple[int, int]:
    width = max(xlim[1] - xlim[0], 1.0e-30)
    height = max(ylim[1] - ylim[0], 1.0e-30)
    if width >= height:
        nx = max_pixels
        ny = max(1, int(round(max_pixels * height / width)))
    else:
        ny = max_pixels
        nx = max(1, int(round(max_pixels * width / height)))
    return ny, nx


def crop_pixel_extent(
    lo: int,
    hi: int,
    n_level: int,
    domain: tuple[float, float],
    crop: tuple[float, float],
    n_canvas: int,
) -> tuple[int, int]:
    width = crop[1] - crop[0]
    dwidth = domain[1] - domain[0]
    x0 = domain[0] + lo * dwidth / n_level
    x1 = domain[0] + (hi + 1) * dwidth / n_level
    p0 = int(np.floor((x0 - crop[0]) * n_canvas / width))
    p1 = int(np.ceil((x1 - crop[0]) * n_canvas / width))
    p0 = min(max(p0, 0), n_canvas - 1)
    p1 = min(max(p1, p0 + 1), n_canvas)
    return p0, p1


def resize_nearest(tile: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    if out_h <= 0 or out_w <= 0:
        return np.empty((0, 0), dtype=tile.dtype)
    yi = np.minimum((np.arange(out_h) * tile.shape[0]) // out_h, tile.shape[0] - 1)
    xi = np.minimum((np.arange(out_w) * tile.shape[1]) // out_w, tile.shape[1] - 1)
    return tile[np.ix_(yi, xi)]


def plot_dimensions(ndim: int, slice_axis: str) -> tuple[int, int]:
    if ndim == 2:
        return 0, 1
    axis = {"x": 0, "y": 1, "z": 2}[slice_axis]
    dims = [dim for dim in range(3) if dim != axis]
    return dims[0], dims[1]


def axis_labels(ndim: int, slice_axis: str) -> tuple[str, str]:
    labels = ("x", "y", "z")
    dim0, dim1 = plot_dimensions(ndim, slice_axis)
    return labels[dim0], labels[dim1]


def slice_level_index(level_shape: tuple[int, ...], ndim: int, slice_axis: str, slice_fraction: float) -> int:
    if ndim == 2:
        return 0
    axis = {"x": 0, "y": 1, "z": 2}[slice_axis]
    return min(max(int(slice_fraction * level_shape[axis]), 0), level_shape[axis] - 1)


def extract_plane(
    values: np.ndarray,
    lo: tuple[int, ...],
    hi: tuple[int, ...],
    ndim: int,
    slice_axis: str,
    slice_index: int,
) -> np.ndarray | None:
    if ndim == 2:
        return values
    axis = {"x": 0, "y": 1, "z": 2}[slice_axis]
    if slice_index < lo[axis] or slice_index > hi[axis]:
        return None
    local = slice_index - lo[axis]
    if axis == 0:
        return values[local, :, :]
    if axis == 1:
        return values[:, local, :]
    return values[:, :, local]


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
) -> tuple[np.ndarray, list[tuple[int, float, float, float, float]], float, int, list[str], tuple[float, float], tuple[float, float]]:
    names, ndim, time_value = parse_plot_header(plotfile)
    if ndim < 2:
        raise ValueError("slice plots require a 2D or 3D plotfile")
    problo, probhi = parse_plot_bounds(plotfile)
    nlevels = level_count(plotfile)
    level_shapes = parse_level_shapes(plotfile)
    dim0, dim1 = plot_dimensions(ndim, slice_axis)
    xlim = xlim if xlim is not None else (problo[dim0], probhi[dim0])
    ylim = ylim if ylim is not None else (problo[dim1], probhi[dim1])
    canvas_ny, canvas_nx = canvas_shape(xlim, ylim, max_pixels)
    image = np.full((canvas_ny, canvas_nx), np.nan)

    for lev in range(nlevels):
        chunks = level_chunks(plotfile, lev, ndim)
        level_shape = level_shapes[lev]
        slice_index = slice_level_index(level_shape, ndim, slice_axis, slice_fraction)
        for lo, hi, data in chunks:
            values, _ = select_component(data, names, component, gamma)
            plane = extract_plane(values, lo, hi, ndim, slice_axis, slice_index)
            if plane is None:
                continue
            frac0 = (xlim[0] - problo[dim0]) / (probhi[dim0] - problo[dim0])
            frac1 = (xlim[1] - problo[dim0]) / (probhi[dim0] - problo[dim0])
            g0 = max(lo[dim0], int(np.floor(frac0 * level_shape[dim0])))
            g1 = min(hi[dim0] + 1, int(np.ceil(frac1 * level_shape[dim0])))
            frac0 = (ylim[0] - problo[dim1]) / (probhi[dim1] - problo[dim1])
            frac1 = (ylim[1] - problo[dim1]) / (probhi[dim1] - problo[dim1])
            h0 = max(lo[dim1], int(np.floor(frac0 * level_shape[dim1])))
            h1 = min(hi[dim1] + 1, int(np.ceil(frac1 * level_shape[dim1])))
            if g0 >= g1 or h0 >= h1:
                continue
            local0 = slice(g0 - lo[dim0], g1 - lo[dim0])
            local1 = slice(h0 - lo[dim1], h1 - lo[dim1])
            subplane = plane[local0, local1].T
            x0, x1 = crop_pixel_extent(g0, g1 - 1, level_shape[dim0], (problo[dim0], probhi[dim0]), xlim, canvas_nx)
            y0, y1 = crop_pixel_extent(h0, h1 - 1, level_shape[dim1], (problo[dim1], probhi[dim1]), ylim, canvas_ny)
            image[y0:y1, x0:x1] = resize_nearest(subplane, y1 - y0, x1 - x0)

    boxes: list[tuple[int, float, float, float, float]] = []
    for lev in range(1, nlevels):
        level_shape = level_shapes[lev]
        slice_index = slice_level_index(level_shape, ndim, slice_axis, slice_fraction)
        for lo, hi, _data in level_chunks(plotfile, lev, ndim):
            if ndim == 3:
                axis = {"x": 0, "y": 1, "z": 2}[slice_axis]
                if slice_index < lo[axis] or slice_index > hi[axis]:
                    continue
            xlo = problo[dim0] + lo[dim0] * (probhi[dim0] - problo[dim0]) / level_shape[dim0]
            xhi = problo[dim0] + (hi[dim0] + 1) * (probhi[dim0] - problo[dim0]) / level_shape[dim0]
            ylo = problo[dim1] + lo[dim1] * (probhi[dim1] - problo[dim1]) / level_shape[dim1]
            yhi = problo[dim1] + (hi[dim1] + 1) * (probhi[dim1] - problo[dim1]) / level_shape[dim1]
            if xhi <= xlim[0] or xlo >= xlim[1] or yhi <= ylim[0] or ylo >= ylim[1]:
                continue
            boxes.append((lev, max(xlo, xlim[0]), min(xhi, xlim[1]), max(ylo, ylim[0]), min(yhi, ylim[1])))

    return image, boxes, time_value, ndim, names, xlim, ylim


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
    if component in names:
        return state[names.index(component)], component
    velocity_names = {
        "v1": "mom1",
        "v2": "mom2",
        "v3": "mom3",
    }
    if component in velocity_names and "rho" in names and velocity_names[component] in names:
        return state[names.index(velocity_names[component])] / state[names.index("rho")], component
    if component == "velocity_mag" and all(name in names for name in ["rho", "mom1", "mom2", "mom3"]):
        rho = state[names.index("rho")]
        return np.sqrt(
            state[names.index("mom1")] ** 2
            + state[names.index("mom2")] ** 2
            + state[names.index("mom3")] ** 2
        ) / rho, component
    if component == "pressure":
        required = ["rho", "mom1", "mom2", "mom3", "energy"]
        if all(name in names for name in required):
            rho = state[names.index("rho")]
            kinetic = 0.5 * (
                state[names.index("mom1")] ** 2
                + state[names.index("mom2")] ** 2
                + state[names.index("mom3")] ** 2
            ) / rho
            magnetic = np.zeros_like(rho)
            if all(name in names for name in ["b1", "b2", "b3"]):
                magnetic = 0.5 * (
                    state[names.index("b1")] ** 2
                    + state[names.index("b2")] ** 2
                    + state[names.index("b3")] ** 2
                )
            return (state[names.index("energy")] - kinetic - magnetic) * (gamma - 1.0), component
    if component.endswith("_mag"):
        prefix = component.removesuffix("_mag")
        vector_names = [f"{prefix}1", f"{prefix}2", f"{prefix}3"]
        if all(name in names for name in vector_names):
            values = np.zeros_like(state[names.index(vector_names[0])])
            for name in vector_names:
                values += state[names.index(name)] ** 2
            return np.sqrt(values), component
    raise ValueError(f"{component} not found; available components: {', '.join(names)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plotfile", required=True, type=Path)
    parser.add_argument("--component", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--title", default="AMReX slice")
    parser.add_argument("--level", type=int, help="debug mode: render only one AMR level instead of the default composite")
    parser.add_argument("--ref-ratio", type=int, default=2)
    parser.add_argument("--max-pixels", type=int, default=1400)
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
    component_label = args.component

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
