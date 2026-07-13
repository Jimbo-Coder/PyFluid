#!/usr/bin/env python3
"""Render AMReX slice frames with AMR boxes and optionally assemble a GIF.

Angular plots with a radial axis are drawn in physical geometry: r-phi as an
annulus and r-theta as a meridional plane. Angular-only or z-phi slices stay as
unwrapped coordinate rectangles. Mixed multipatch plotfiles can be rendered in
physical patch geometry instead of the internal packed AMReX layout.
"""

from __future__ import annotations

import argparse
import csv
import textwrap
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle
from PIL import Image

from amr_plot_data import HierarchySlice, component_values as _component_values
from plotfile_reader import (
   component_plotfile_timeline,
   parse_fab_index,
   parse_level_shapes,
   parse_plot_bounds,
   parse_plot_header,
   plotfile_step as _plotfile_step,
   read_fab,
)
from plotfile_reader import level_chunks as _reader_level_chunks
from plotfile_reader import level_count as _reader_level_count


def plotfile_step(path: Path) -> int:
    return _plotfile_step(path)


def parse_box_levels(text: str | None) -> set[int] | None:
    if text is None:
        return None
    levels = set()
    for raw in text.split(","):
        raw = raw.strip()
        if raw:
            levels.add(int(raw))
    return levels


def parse_key_value_file(path: Path | None) -> dict[str, str]:
    if path is None or not path.exists():
        return {}
    params: dict[str, str] = {}
    with path.open() as f:
        for raw in f:
            line = raw.split("#", 1)[0].strip()
            if not line or "=" not in line:
                continue
            key, value = line.split("=", 1)
            params[key.strip()] = value.strip()
    return params


def param_ints(params: dict[str, str], key: str, default: tuple[int, ...]) -> tuple[int, ...]:
    raw = params.get(key)
    if raw is None:
        return default
    values = tuple(int(value) for value in raw.replace(",", " ").split())
    return values if values else default


def param_int(params: dict[str, str], key: str, default: int) -> int:
    raw = params.get(key)
    return int(raw) if raw is not None else default


def param_float(params: dict[str, str], key: str, default: float) -> float:
    raw = params.get(key)
    return float(raw) if raw is not None else default


def mixed_patch_view_available(params: dict[str, str]) -> bool:
    return params.get("mesh.patch_system", "single") in {
        "mixed_thornburg04",
        "multi_mixed_thornburg04",
        "mixed_thornburg13",
        "multi_mixed_thornburg13",
        "mixed_cylindrical5",
        "multi_mixed_cylindrical5",
        "mixed_cylindrical9",
        "multi_mixed_cylindrical9",
    }


def default_curvilinear_view(params: dict[str, str]) -> str:
    if mixed_patch_view_available(params):
        return "multipatch"
    axes = params.get("mesh.axes", "").replace(",", " ").split()
    if len(axes) < 2:
        return "native"
    if axes[0] == "r" and axes[1] == "phi":
        return "r-phi"
    if axes[0] == "r" and axes[1] == "theta":
        return "r-theta"
    return "native"


def default_axis_labels(params: dict[str, str], view: str) -> tuple[str, str]:
    if view == "multipatch":
        axes = params.get("mesh.axes", "").replace(",", " ").split()
        if len(axes) >= 2:
            return axes[0], axes[1]
        return "x", "y"
    if view == "r-phi":
        return "x", "y"
    if view == "r-theta":
        return "x", "z"
    axes = params.get("mesh.axes", "").replace(",", " ").split()
    if len(axes) >= 2:
        return axes[0], axes[1]
    return "x", "y"


def parse_solid_boxes(lo: list[float] | None, hi: list[float] | None) -> list[tuple[float, float, float, float]]:
    if lo is None and hi is None:
        return []
    if lo is None or hi is None:
        raise ValueError("--solid-box-lo and --solid-box-hi must be provided together")
    if len(lo) != len(hi) or len(lo) % 2 != 0:
        raise ValueError("solid box coordinates must contain nboxes * 2 values for 2D movie overlays")
    boxes = []
    for i in range(0, len(lo), 2):
        xlo, ylo = lo[i], lo[i + 1]
        xhi, yhi = hi[i], hi[i + 1]
        if not (xlo < xhi and ylo < yhi):
            raise ValueError("solid-box lower bounds must be smaller than upper bounds")
        boxes.append((xlo, xhi, ylo, yhi))
    return boxes


def load_solid_boxes_from_metadata(path: Path | None) -> list[tuple[float, float, float, float]]:
    if path is None or not path.exists():
        return []
    boxes = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row.get("kind") != "box":
                continue
            dim = int(row["dim"])
            if dim < 2:
                continue
            xlo = float(row["xlo"])
            xhi = float(row["xhi"])
            ylo = float(row["ylo"])
            yhi = float(row["yhi"])
            if xlo < xhi and ylo < yhi:
                boxes.append((xlo, xhi, ylo, yhi))
    return boxes


def load_solid_geometry_from_metadata(path: Path | None) -> list[dict[str, float | str]]:
    if path is None or not path.exists():
        return []
    geometries: list[dict[str, float | str]] = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            parsed: dict[str, float | str] = {}
            for key, value in row.items():
                if key in {"kind", "motion", "wall_condition"}:
                    parsed[key] = value
                else:
                    try:
                        parsed[key] = float(value)
                    except ValueError:
                        parsed[key] = np.nan
            geometries.append(parsed)
    return geometries


def plotfile_has_component(path: Path, component: str) -> bool:
    names, _ndim, _time_value = parse_plot_header(path)
    return component in names


def load_mask_plotfiles(mask_plot_dir: Path | None, component: str, *, required: bool) -> dict[int, Path]:
    if mask_plot_dir is None:
        return {}
    if not mask_plot_dir.exists():
        if required:
            raise ValueError(f"mask plot directory does not exist: {mask_plot_dir}")
        return {}
    mask_plotfiles = {}
    for path in mask_plot_dir.iterdir():
        if path.is_dir() and _is_aux_plotfile_dir(path):
            if plotfile_has_component(path, component):
                mask_plotfiles[plotfile_step(path)] = path
    if not mask_plotfiles:
        if required:
            raise ValueError(f"no aux plotfiles with {component} found in {mask_plot_dir}")
        return {}
    return mask_plotfiles


def is_amrex_plotfile_dir(path: Path) -> bool:
   if not path.is_dir():
      return False
   return (
        _is_state_plotfile_dir(path)
        or _is_aux_plotfile_dir(path)
   )


def _is_plotfile_name_with_step(name: str, prefix: str) -> bool:
   return name.startswith(prefix) and name[len(prefix):].isdigit()


def _is_state_plotfile_dir(path: Path) -> bool:
   return _is_plotfile_name_with_step(path.name, "stateGF")


def _is_aux_plotfile_dir(path: Path) -> bool:
   return _is_plotfile_name_with_step(path.name, "auxGF")


def mask_solid_boxes(
    image: np.ndarray,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    solid_boxes: list[tuple[float, float, float, float]],
) -> np.ndarray:
    if not solid_boxes:
        return image
    masked = image.copy()
    ny, nx = masked.shape
    x = xlim[0] + (np.arange(nx, dtype=float) + 0.5) * (xlim[1] - xlim[0]) / nx
    y = ylim[0] + (np.arange(ny, dtype=float) + 0.5) * (ylim[1] - ylim[0]) / ny
    xx, yy = np.meshgrid(x, y)
    solid = np.zeros_like(masked, dtype=bool)
    for xlo, xhi, ylo, yhi in solid_boxes:
        solid |= (xx >= xlo) & (xx <= xhi) & (yy >= ylo) & (yy <= yhi)
    masked[solid] = np.nan
    return masked


def transform_curvilinear_coordinates(x: np.ndarray, y: np.ndarray, view: str) -> tuple[np.ndarray, np.ndarray]:
    if view == "r-theta":
        return x * np.sin(y), x * np.cos(y)
    if view == "r-phi":
        return x * np.cos(y), x * np.sin(y)
    return x, y


def transformed_cell_edges(
    image: np.ndarray,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    view: str,
) -> tuple[np.ndarray, np.ndarray]:
    ny, nx = image.shape
    x_edges = np.linspace(xlim[0], xlim[1], nx + 1)
    y_edges = np.linspace(ylim[0], ylim[1], ny + 1)
    xx, yy = np.meshgrid(x_edges, y_edges)
    return transform_curvilinear_coordinates(xx, yy, view)


def transformed_limits(
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    view: str,
) -> tuple[tuple[float, float], tuple[float, float]]:
    samples = 256
    r = np.linspace(xlim[0], xlim[1], samples)
    a = np.linspace(ylim[0], ylim[1], samples)
    edges_x = np.concatenate([r, r, np.full_like(a, xlim[0]), np.full_like(a, xlim[1])])
    edges_y = np.concatenate([np.full_like(r, ylim[0]), np.full_like(r, ylim[1]), a, a])
    xx, yy = transform_curvilinear_coordinates(edges_x, edges_y, view)
    pad_x = 0.025 * max(np.ptp(xx), 1.0e-12)
    pad_y = 0.025 * max(np.ptp(yy), 1.0e-12)
    return (float(np.min(xx) - pad_x), float(np.max(xx) + pad_x)), (float(np.min(yy) - pad_y), float(np.max(yy) + pad_y))


def draw_coordinate_box_outline(
    ax: plt.Axes,
    xlo: float,
    xhi: float,
    ylo: float,
    yhi: float,
    view: str,
    **kwargs,
) -> None:
    if view == "native":
        ax.add_patch(Rectangle((xlo, ylo), xhi - xlo, yhi - ylo, fill=False, **kwargs))
        return
    samples = 96
    x_edge = np.linspace(xlo, xhi, samples)
    y_edge = np.linspace(ylo, yhi, samples)
    segments = (
        (x_edge, np.full_like(x_edge, ylo)),
        (x_edge, np.full_like(x_edge, yhi)),
        (np.full_like(y_edge, xlo), y_edge),
        (np.full_like(y_edge, xhi), y_edge),
    )
    for xs, ys in segments:
        xp, yp = transform_curvilinear_coordinates(xs, ys, view)
        ax.plot(xp, yp, **kwargs)


def finite_or(value: float | str, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if np.isfinite(number) else default


@dataclass(frozen=True)
class VisualPatch:
    patch_id: int
    coordinate_map: str
    local_lo: tuple[float, float, float]
    local_hi: tuple[float, float, float]
    local_axes: tuple[int, int]
    cells: tuple[int, int]
    owned_lo: tuple[int, int]
    owned_hi: tuple[int, int]
    origin: tuple[float, float, float]
    display_components: tuple[int, int]


@dataclass(frozen=True)
class PlotTile:
    level: int
    edge_x: np.ndarray
    edge_y: np.ndarray
    values: np.ndarray


@dataclass(frozen=True)
class BoxOutline:
    level: int
    segments: list[tuple[np.ndarray, np.ndarray]]


def draw_tile_pcolormesh(
    ax: plt.Axes,
    tile: PlotTile,
    *,
    cmap: matplotlib.colors.Colormap,
    vmin: float,
    vmax: float,
):
    return ax.pcolormesh(
        tile.edge_x,
        tile.edge_y,
        tile.values,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        shading="flat",
        edgecolors="none",
        linewidth=0.0,
        antialiased=False,
        rasterized=True,
    )


def patch_face_map_name(system: str, axis: int, sign: int) -> str:
    prefix = "cylindrical_shell" if "cylindrical" in system else "thornburg_shell"
    axis_name = ("x", "y", "z")[axis]
    sign_name = "plus" if sign > 0 else "minus"
    return f"{prefix}_{sign_name}_{axis_name}"


def cartesian_axes_from_params(params: dict[str, str], ndim: int) -> tuple[int, ...]:
    names = {"x": 0, "y": 1, "z": 2}
    raw_axes = params.get("mesh.axes", "")
    tokens = raw_axes.replace(",", " ").lower().split()
    if not tokens:
        tokens = ["x", "y", "z"][:ndim]
    if len(tokens) != ndim:
        raise ValueError("mesh.axes must list exactly the plotfile dimension")
    try:
        return tuple(names[token] for token in tokens)
    except KeyError as exc:
        raise ValueError("mixed patch visualization expects Cartesian mesh.axes tokens x/y/z") from exc


def patch_system_maps(system: str, active_axes: tuple[int, ...]) -> list[str]:
    maps = ["thornburg_central_cartesian"]
    if "cylindrical" in system:
        if active_axes[:2] != (0, 1):
            raise ValueError("mixed cylindrical 2D visualization supports the r-phi/x-y plane")
        shell_axes = (0, 1)
    else:
        shell_axes = active_axes
    for axis in shell_axes:
        maps.append(patch_face_map_name(system, axis, 1))
        maps.append(patch_face_map_name(system, axis, -1))
    if system in {"mixed_thornburg13", "multi_mixed_thornburg13", "mixed_cylindrical9", "multi_mixed_cylindrical9"}:
        for axis in shell_axes:
            maps.append(patch_face_map_name(system, axis, 1))
            maps.append(patch_face_map_name(system, axis, -1))
    return maps


def thornburg_tangent_component(primary_axis: int, other_axis: int) -> int:
    if primary_axis == 0:
        return 0 if other_axis == 2 else 1
    if primary_axis == 1:
        return 0 if other_axis == 2 else 1
    return 0 if other_axis == 1 else 1


def shell_axis_from_map(coordinate_map: str) -> int:
    axis, _sign = shell_axis_and_sign(coordinate_map)
    return axis


def shell_local_axes(system: str, coordinate_map: str, active_axes: tuple[int, ...]) -> tuple[int, int]:
    if "cylindrical" in system:
        return 0, 2
    primary_axis = shell_axis_from_map(coordinate_map)
    if len(active_axes) != 2:
        raise ValueError("physical multipatch movie view currently supports 2D plotfiles")
    other_axis = active_axes[1] if primary_axis == active_axes[0] else active_axes[0]
    return thornburg_tangent_component(primary_axis, other_axis), 2


def patch_cells_for_shell(system: str, params: dict[str, str], outer_shell: bool, ndim: int) -> tuple[int, int]:
    angular_cells = param_int(params, "mesh.patch_angular_cells", 16)
    radial_key = "mesh.patch_outer_radial_cells" if outer_shell else "mesh.patch_radial_cells"
    radial_cells = param_int(params, radial_key, 16)
    axial_cells = param_int(params, "mesh.patch_axial_cells", 16)
    if ndim != 2:
        raise ValueError("physical multipatch movie view currently supports 2D plotfiles")
    if "cylindrical" in system:
        return angular_cells, radial_cells
    return angular_cells, radial_cells


def build_visual_patches(params: dict[str, str], ndim: int, refinement: int) -> list[VisualPatch]:
    if ndim != 2:
        raise ValueError("physical multipatch movie view currently supports 2D plotfiles")
    system = params.get("mesh.patch_system", "single")
    if not mixed_patch_view_available(params):
        raise ValueError("physical multipatch view requires a mixed mesh.patch_system")
    if param_int(params, "mesh.patch_gap_cells", 0) != 0:
        raise ValueError("physical multipatch view currently requires mesh.patch_gap_cells = 0")
    pack_direction = param_int(params, "mesh.patch_pack_direction", 0)
    if pack_direction not in (0, 1):
        raise ValueError("mesh.patch_pack_direction is outside the 2D plot dimensions")

    inner_radius = param_float(params, "mesh.patch_inner_radius", 1.0)
    medium_radius = param_float(params, "mesh.patch_medium_radius", 2.5)
    outer_radius = param_float(params, "mesh.patch_outer_radius", 4.0)
    origin = (
        param_float(params, "mesh.patch_origin1", 0.0),
        param_float(params, "mesh.patch_origin2", 0.0),
        param_float(params, "mesh.patch_origin3", 0.0),
    )
    central_default = (16, 16)
    central_cells_raw = param_ints(params, "mesh.patch_central_cells", central_default)
    if len(central_cells_raw) < 2:
        raise ValueError("mesh.patch_central_cells must list at least two values for 2D visualization")
    central_cells = tuple(central_cells_raw[:2])
    active_axes = cartesian_axes_from_params(params, ndim)
    maps = patch_system_maps(system, active_axes)
    patches: list[VisualPatch] = []
    next_small = 0

    def append_patch(
        patch_id: int,
        coordinate_map: str,
        local_lo: tuple[float, float, float],
        local_hi: tuple[float, float, float],
        axes: tuple[int, int],
        cells: tuple[int, int],
    ) -> None:
        nonlocal next_small
        refined_cells = (cells[0] * refinement, cells[1] * refinement)
        owned_lo = [0, 0]
        owned_hi = [refined_cells[0] - 1, refined_cells[1] - 1]
        owned_lo[pack_direction] += next_small
        owned_hi[pack_direction] += next_small
        patches.append(
            VisualPatch(
                patch_id=patch_id,
                coordinate_map=coordinate_map,
                local_lo=local_lo,
                local_hi=local_hi,
                local_axes=axes,
                cells=refined_cells,
                owned_lo=(owned_lo[0], owned_lo[1]),
                owned_hi=(owned_hi[0], owned_hi[1]),
                origin=origin,
                display_components=(active_axes[0], active_axes[1]),
            )
        )
        next_small += refined_cells[pack_direction]

    central_lo = [0.0, 0.0, 0.0]
    central_hi = [0.0, 0.0, 0.0]
    for component in active_axes:
        central_lo[component] = -inner_radius
        central_hi[component] = inner_radius
    append_patch(
        0,
        maps[0],
        tuple(central_lo),
        tuple(central_hi),
        (active_axes[0], active_axes[1]),
        central_cells,
    )
    for patch_id, coordinate_map in enumerate(maps[1:], start=1):
        outer_shell = (
            system in {"mixed_thornburg13", "multi_mixed_thornburg13"}
            and patch_id >= 1 + 2 * ndim
        ) or (
            system in {"mixed_cylindrical9", "multi_mixed_cylindrical9"}
            and patch_id >= 5
        )
        radial_lo = medium_radius if outer_shell else inner_radius
        radial_hi = outer_radius if outer_shell else (medium_radius if "13" in system or "9" in system else outer_radius)
        if "cylindrical" in system:
            local_lo = (-0.25 * np.pi, param_float(params, "mesh.patch_axial_lo", -1.0), radial_lo)
            local_hi = (0.25 * np.pi, param_float(params, "mesh.patch_axial_hi", 1.0), radial_hi)
        else:
            local_lo = (-0.25 * np.pi, -0.25 * np.pi, radial_lo)
            local_hi = (0.25 * np.pi, 0.25 * np.pi, radial_hi)
        append_patch(
            patch_id,
            coordinate_map,
            local_lo,
            local_hi,
            shell_local_axes(system, coordinate_map, active_axes),
            patch_cells_for_shell(system, params, outer_shell, ndim),
        )
    return patches


def box_intersection(
    lo: tuple[int, ...],
    hi: tuple[int, ...],
    patch: VisualPatch,
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    inter_lo = (max(lo[0], patch.owned_lo[0]), max(lo[1], patch.owned_lo[1]))
    inter_hi = (min(hi[0], patch.owned_hi[0]), min(hi[1], patch.owned_hi[1]))
    if inter_lo[0] > inter_hi[0] or inter_lo[1] > inter_hi[1]:
        return None
    return inter_lo, inter_hi


def local_coordinate_grid(
    patch: VisualPatch,
    lo: tuple[int, int],
    hi: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    local_lo = np.array(patch.local_lo, dtype=float)
    local_hi = np.array(patch.local_hi, dtype=float)
    coords = []
    for direction in range(2):
        component = patch.local_axes[direction]
        edge0 = lo[direction] - patch.owned_lo[direction]
        edge1 = hi[direction] - patch.owned_lo[direction] + 1
        edges = np.arange(edge0, edge1 + 1, dtype=float)
        spacing = (local_hi[component] - local_lo[component]) / patch.cells[direction]
        coords.append(local_lo[component] + edges * spacing)
    q0, q1 = np.meshgrid(coords[0], coords[1])
    q = [
        np.full_like(q0, 0.5 * (patch.local_lo[component] + patch.local_hi[component]))
        for component in range(3)
    ]
    q[patch.local_axes[0]] = q0
    q[patch.local_axes[1]] = q1
    return q[0], q[1], q[2]


def shell_axis_and_sign(coordinate_map: str) -> tuple[int, float]:
    if coordinate_map.endswith("_plus_x"):
        return 0, 1.0
    if coordinate_map.endswith("_minus_x"):
        return 0, -1.0
    if coordinate_map.endswith("_plus_y"):
        return 1, 1.0
    if coordinate_map.endswith("_minus_y"):
        return 1, -1.0
    if coordinate_map.endswith("_plus_z"):
        return 2, 1.0
    if coordinate_map.endswith("_minus_z"):
        return 2, -1.0
    raise ValueError(f"unsupported shell coordinate map: {coordinate_map}")


def patch_local_to_physical_xy(
    patch: VisualPatch,
    q1: np.ndarray,
    q2: np.ndarray,
    q3: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    o1, o2, o3 = patch.origin
    if patch.coordinate_map in {"cartesian_identity", "thornburg_central_cartesian"}:
        global_components = (o1 + q1, o2 + q2, o3 + q3)
        return global_components[patch.display_components[0]], global_components[patch.display_components[1]]
    axis, sign = shell_axis_and_sign(patch.coordinate_map)
    if patch.coordinate_map.startswith("thornburg_shell"):
        angular_norm = np.sqrt(1.0 + np.tan(q1) ** 2 + np.tan(q2) ** 2)
        radial_inner = patch.local_lo[2]
        radial_outer = patch.local_hi[2]
        radial_fraction = (q3 - radial_inner) / (radial_outer - radial_inner)
        blend = 1.0 - radial_fraction + radial_fraction / angular_norm
        primary = sign * q3 * blend
        if axis == 0:
            global_components = (
                o1 + primary,
                o2 + primary * np.tan(q2),
                o3 + primary * np.tan(q1),
            )
            return global_components[patch.display_components[0]], global_components[patch.display_components[1]]
        if axis == 1:
            global_components = (
                o1 + primary * np.tan(q2),
                o2 + primary,
                o3 + primary * np.tan(q1),
            )
            return global_components[patch.display_components[0]], global_components[patch.display_components[1]]
        global_components = (
            o1 + primary * np.tan(q2),
            o2 + primary * np.tan(q1),
            o3 + primary,
        )
        return global_components[patch.display_components[0]], global_components[patch.display_components[1]]
    if patch.coordinate_map.startswith("cylindrical_shell"):
        angular_norm = np.sqrt(1.0 + np.tan(q1) ** 2)
        radial_inner = patch.local_lo[2]
        radial_outer = patch.local_hi[2]
        radial_fraction = (q3 - radial_inner) / (radial_outer - radial_inner)
        blend = 1.0 - radial_fraction + radial_fraction / angular_norm
        primary = sign * q3 * blend
        if axis == 0:
            global_components = (o1 + primary, o2 + primary * np.tan(q1), o3 + q2)
            return global_components[patch.display_components[0]], global_components[patch.display_components[1]]
        if axis == 1:
            global_components = (o1 + primary * np.tan(q1), o2 + primary, o3 + q2)
            return global_components[patch.display_components[0]], global_components[patch.display_components[1]]
    raise ValueError(f"physical 2D plot does not support coordinate map {patch.coordinate_map}")


def physical_patch_edges(patch: VisualPatch, lo: tuple[int, int], hi: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    q1, q2, q3 = local_coordinate_grid(patch, lo, hi)
    return patch_local_to_physical_xy(patch, q1, q2, q3)


def physical_outline_segments(
    patch: VisualPatch,
    lo: tuple[int, int],
    hi: tuple[int, int],
) -> list[tuple[np.ndarray, np.ndarray]]:
    x, y = physical_patch_edges(patch, lo, hi)
    return [
        (x[0, :], y[0, :]),
        (x[-1, :], y[-1, :]),
        (x[:, 0], y[:, 0]),
        (x[:, -1], y[:, -1]),
    ]


def physical_multipatch_limits(params: dict[str, str], ndim: int) -> tuple[tuple[float, float], tuple[float, float]]:
    patches = build_visual_patches(params, ndim, 1)
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for patch in patches:
        x, y = physical_patch_edges(patch, patch.owned_lo, patch.owned_hi)
        xs.append(x.ravel())
        ys.append(y.ravel())
    all_x = np.concatenate(xs)
    all_y = np.concatenate(ys)
    pad_x = 0.025 * max(float(np.ptp(all_x)), 1.0e-12)
    pad_y = 0.025 * max(float(np.ptp(all_y)), 1.0e-12)
    return (
        (float(np.min(all_x) - pad_x), float(np.max(all_x) + pad_x)),
        (float(np.min(all_y) - pad_y), float(np.max(all_y) + pad_y)),
    )


def physical_multipatch_tiles(
    plotfile: Path,
    component: str,
    gamma: float,
    ref_ratio: int,
    z_fraction: float,
    log10: bool,
    subtract_constant: float | None,
    show_base_boxes: bool,
    params: dict[str, str],
) -> tuple[
    list[PlotTile],
    list[BoxOutline],
    float,
    int,
]:
    names, ndim, time_value = parse_plot_header(plotfile)
    if ndim != 2:
        raise ValueError("physical multipatch movie view currently supports 2D plotfiles")
    nlevels = level_count(plotfile)
    tiles: list[PlotTile] = []
    boxes: list[BoxOutline] = []
    for lev in range(nlevels):
        refinement = ref_ratio**lev
        patches = build_visual_patches(params, ndim, refinement)
        for lo, hi, data in level_chunks(plotfile, lev, ndim):
            values = component_values(data, names, component, gamma)
            if component != "schlieren" and subtract_constant is not None:
                values = values - subtract_constant
            if log10:
                values = np.log10(np.maximum(values, 1.0e-14))
            for patch in patches:
                overlap = box_intersection(lo, hi, patch)
                if overlap is None:
                    continue
                inter_lo, inter_hi = overlap
                local_x = slice(inter_lo[0] - lo[0], inter_hi[0] - lo[0] + 1)
                local_y = slice(inter_lo[1] - lo[1], inter_hi[1] - lo[1] + 1)
                tile_values = values[local_x, local_y].T
                edge_x, edge_y = physical_patch_edges(patch, inter_lo, inter_hi)
                tiles.append(PlotTile(lev, edge_x, edge_y, tile_values))
        if lev > 0 or show_base_boxes:
            for lo, hi, _data in level_chunks(plotfile, lev, ndim):
                for patch in patches:
                    overlap = box_intersection(lo, hi, patch)
                    if overlap is None:
                        continue
                    inter_lo, inter_hi = overlap
                    boxes.append(BoxOutline(lev, physical_outline_segments(patch, inter_lo, inter_hi)))
    return tiles, boxes, time_value, ndim


def regular_cell_edges(
    lo: tuple[int, ...],
    hi: tuple[int, ...],
    level_shape: tuple[int, ...],
    problo: tuple[float, ...],
    probhi: tuple[float, ...],
    view: str,
) -> tuple[np.ndarray, np.ndarray]:
    x_edges = problo[0] + np.arange(lo[0], hi[0] + 2, dtype=float) * (probhi[0] - problo[0]) / level_shape[0]
    y_edges = problo[1] + np.arange(lo[1], hi[1] + 2, dtype=float) * (probhi[1] - problo[1]) / level_shape[1]
    xx, yy = np.meshgrid(x_edges, y_edges)
    return transform_curvilinear_coordinates(xx, yy, view)


def regular_box_outline(
    lo: tuple[int, ...],
    hi: tuple[int, ...],
    level_shape: tuple[int, ...],
    problo: tuple[float, ...],
    probhi: tuple[float, ...],
    view: str,
) -> list[tuple[np.ndarray, np.ndarray]]:
    xlo = problo[0] + lo[0] * (probhi[0] - problo[0]) / level_shape[0]
    xhi = problo[0] + (hi[0] + 1) * (probhi[0] - problo[0]) / level_shape[0]
    ylo = problo[1] + lo[1] * (probhi[1] - problo[1]) / level_shape[1]
    yhi = problo[1] + (hi[1] + 1) * (probhi[1] - problo[1]) / level_shape[1]
    if view == "native":
        return [
            (np.array([xlo, xhi]), np.array([ylo, ylo])),
            (np.array([xlo, xhi]), np.array([yhi, yhi])),
            (np.array([xlo, xlo]), np.array([ylo, yhi])),
            (np.array([xhi, xhi]), np.array([ylo, yhi])),
        ]
    samples = 96
    x_edge = np.linspace(xlo, xhi, samples)
    y_edge = np.linspace(ylo, yhi, samples)
    coordinate_segments = (
        (x_edge, np.full_like(x_edge, ylo)),
        (x_edge, np.full_like(x_edge, yhi)),
        (np.full_like(y_edge, xlo), y_edge),
        (np.full_like(y_edge, xhi), y_edge),
    )
    return [transform_curvilinear_coordinates(xs, ys, view) for xs, ys in coordinate_segments]


def angular_render_block_cells(level_shape: tuple[int, ...], problo: tuple[float, ...], probhi: tuple[float, ...]) -> int:
    if len(level_shape) < 2:
        return 1
    dangle = (probhi[1] - problo[1]) / max(level_shape[1], 1)
    if dangle <= 0.0:
        return 1
    # Keep each QuadMesh locally well behaved in physical x-y. Large angular
    # spans fold around the polar origin and Matplotlib leaves seam artifacts.
    return max(1, min(64, int(np.floor((np.pi / 6.0) / dangle))))


def append_regular_tiles(
    tiles: list[PlotTile],
    lo: tuple[int, ...],
    hi: tuple[int, ...],
    values_yx: np.ndarray,
    level: int,
    level_shape: tuple[int, ...],
    problo: tuple[float, ...],
    probhi: tuple[float, ...],
    view: str,
) -> None:
    if view not in {"r-theta", "r-phi"}:
        edge_x, edge_y = regular_cell_edges(lo, hi, level_shape, problo, probhi, view)
        tiles.append(PlotTile(level, edge_x, edge_y, values_yx))
        return

    y_cells = hi[1] - lo[1] + 1
    block = angular_render_block_cells(level_shape, problo, probhi)
    for y0 in range(0, y_cells, block):
        y1 = min(y_cells, y0 + block)
        sub_lo = list(lo)
        sub_hi = list(hi)
        sub_lo[1] = lo[1] + y0
        sub_hi[1] = lo[1] + y1 - 1
        edge_x, edge_y = regular_cell_edges(tuple(sub_lo), tuple(sub_hi), level_shape, problo, probhi, view)
        tiles.append(PlotTile(level, edge_x, edge_y, values_yx[y0:y1, :]))


def regular_mapped_tiles(
    plotfile: Path,
    component: str,
    gamma: float,
    ref_ratio: int,
    z_fraction: float,
    log10: bool,
    subtract_constant: float | None,
    show_base_boxes: bool,
    view: str,
) -> tuple[list[PlotTile], list[BoxOutline], float, int]:
    _ = ref_ratio
    hierarchy = HierarchySlice(
        plotfile,
        component,
        gamma,
        slice_axis="z",
        slice_fraction=z_fraction,
    )
    problo, probhi = hierarchy.problo, hierarchy.probhi
    tiles: list[PlotTile] = []
    boxes: list[BoxOutline] = []
    for tile in hierarchy.tiles():
        values = tile.values
        if subtract_constant is not None:
            values = values - subtract_constant
        if component != "schlieren" and log10:
            values = np.log10(np.maximum(values, 1.0e-14))
        append_regular_tiles(
            tiles,
            tile.lo,
            tile.hi,
            values,
            tile.level,
            tile.level_shape,
            problo,
            probhi,
            view,
        )
        if tile.level > 0 or show_base_boxes:
            boxes.append(
                BoxOutline(
                    tile.level,
                    regular_box_outline(tile.lo, tile.hi, tile.level_shape, problo, probhi, view),
                )
            )
    return tiles, boxes, hierarchy.time, hierarchy.ndim


def mapped_view_tiles(
    plotfile: Path,
    component: str,
    gamma: float,
    ref_ratio: int,
    z_fraction: float,
    log10: bool,
    subtract_constant: float | None,
    show_base_boxes: bool,
    view: str,
    params: dict[str, str],
) -> tuple[list[PlotTile], list[BoxOutline], float, int]:
    if view == "multipatch":
        return physical_multipatch_tiles(
            plotfile,
            component,
            gamma,
            ref_ratio,
            z_fraction,
            log10,
            subtract_constant,
            show_base_boxes,
            params,
        )
    return regular_mapped_tiles(
        plotfile,
        component,
        gamma,
        ref_ratio,
        z_fraction,
        log10,
        subtract_constant,
        show_base_boxes,
        view,
    )


def tile_limits(tiles: list[PlotTile]) -> tuple[tuple[float, float], tuple[float, float]]:
    if not tiles:
        raise ValueError("cannot compute limits for an empty tile set")
    xs = np.concatenate([tile.edge_x.ravel() for tile in tiles])
    ys = np.concatenate([tile.edge_y.ravel() for tile in tiles])
    pad_x = 0.025 * max(float(np.ptp(xs)), 1.0e-12)
    pad_y = 0.025 * max(float(np.ptp(ys)), 1.0e-12)
    return (
        (float(np.nanmin(xs) - pad_x), float(np.nanmax(xs) + pad_x)),
        (float(np.nanmin(ys) - pad_y), float(np.nanmax(ys) + pad_y)),
    )


def translated_value(geometry: dict[str, float | str], key: str, velocity_key: str, time_value: float) -> float:
    return finite_or(geometry.get(key, np.nan), 0.0) + finite_or(geometry.get(velocity_key, 0.0), 0.0) * time_value


def circle_radius_at_time(geometry: dict[str, float | str], time_value: float) -> float:
    radius = finite_or(geometry.get("radius", np.nan), 0.0)
    if str(geometry.get("motion", "")) != "pulse":
        return radius
    amplitude = finite_or(geometry.get("radius_amplitude", 0.0), 0.0)
    frequency = finite_or(geometry.get("radius_frequency", 0.0), 0.0)
    phase = finite_or(geometry.get("radius_phase", 0.0), 0.0)
    ramp_time = finite_or(geometry.get("radius_ramp_time", 0.0), 0.0)
    envelope = 1.0
    if ramp_time > 0.0:
        if time_value <= 0.0:
            envelope = 0.0
        elif time_value >= ramp_time:
            envelope = 1.0
        else:
            u = time_value / ramp_time
            envelope = u * u * u * (10.0 + u * (-15.0 + 6.0 * u))
    return max(radius + amplitude * envelope * np.sin(phase + 2.0 * np.pi * frequency * time_value), 0.0)


def solid_geometry_mask(
    shape: tuple[int, int],
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    geometries: list[dict[str, float | str]],
    time_value: float,
) -> np.ndarray:
    ny, nx = shape
    x = xlim[0] + (np.arange(nx, dtype=float) + 0.5) * (xlim[1] - xlim[0]) / nx
    y = ylim[0] + (np.arange(ny, dtype=float) + 0.5) * (ylim[1] - ylim[0]) / ny
    xx, yy = np.meshgrid(x, y)
    solid = np.zeros(shape, dtype=bool)
    for geometry in geometries:
        kind = str(geometry.get("kind", ""))
        if kind == "box":
            xlo = translated_value(geometry, "xlo", "v1", time_value)
            xhi = translated_value(geometry, "xhi", "v1", time_value)
            ylo = translated_value(geometry, "ylo", "v2", time_value)
            yhi = translated_value(geometry, "yhi", "v2", time_value)
            solid |= (xx >= xlo) & (xx <= xhi) & (yy >= ylo) & (yy <= yhi)
        elif kind == "circle":
            cx = translated_value(geometry, "center_x", "v1", time_value)
            cy = translated_value(geometry, "center_y", "v2", time_value)
            radius = circle_radius_at_time(geometry, time_value)
            solid |= (xx - cx) ** 2 + (yy - cy) ** 2 <= radius * radius
        elif kind == "wedge":
            x0 = translated_value(geometry, "wedge_x0", "v1", time_value)
            y0 = translated_value(geometry, "wedge_y0", "v2", time_value)
            angle = finite_or(geometry.get("wedge_angle_deg", 0.0), 0.0)
            slope = np.tan(np.deg2rad(angle))
            solid |= (xx >= x0) & (yy <= y0 + slope * (xx - x0))
    return solid


def mask_solid_geometry(
    image: np.ndarray,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    geometries: list[dict[str, float | str]],
    time_value: float,
) -> np.ndarray:
    if not geometries:
        return image
    masked = image.copy()
    masked[solid_geometry_mask(masked.shape, xlim, ylim, geometries, time_value)] = np.nan
    return masked


def box_extents_at_time(geometry: dict[str, float | str], time_value: float) -> tuple[float, float, float, float]:
    return (
        translated_value(geometry, "xlo", "v1", time_value),
        translated_value(geometry, "xhi", "v1", time_value),
        translated_value(geometry, "ylo", "v2", time_value),
        translated_value(geometry, "yhi", "v2", time_value),
    )


def circle_at_time(geometry: dict[str, float | str], time_value: float) -> tuple[float, float, float]:
    return (
        translated_value(geometry, "center_x", "v1", time_value),
        translated_value(geometry, "center_y", "v2", time_value),
        circle_radius_at_time(geometry, time_value),
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


def level_count(plotfile: Path) -> int:
    return _reader_level_count(plotfile)


def level_chunks(plotfile: Path, level: int, ndim: int):
    return _reader_level_chunks(plotfile, level, ndim)


def full_shape(chunks, ndim: int, fallback: tuple[int, ...] | None = None) -> tuple[int, ...]:
    if fallback is not None:
        return fallback
    return tuple(max(chunk[1][d] for chunk in chunks) + 1 for d in range(ndim))


def component_values(
    data: np.ndarray,
    names: list[str],
    component: str,
    gamma: float,
    cell_size: tuple[float, ...] | None = None,
) -> np.ndarray:
    return _component_values(data, names, component, gamma, cell_size)


def slice_to_xy(values: np.ndarray, lo: tuple[int, ...], hi: tuple[int, ...], slice_index: int) -> np.ndarray | None:
    if len(lo) == 2:
        return values
    if slice_index < lo[2] or slice_index > hi[2]:
        return None
    return values[:, :, slice_index - lo[2]]


def canvas_shape(xlim: tuple[float, float], ylim: tuple[float, float], max_pixels: int) -> tuple[int, int]:
    nx0 = max(xlim[1] - xlim[0], 1.0e-30)
    ny0 = max(ylim[1] - ylim[0], 1.0e-30)
    if nx0 >= ny0:
        nx = max_pixels
        ny = max(1, int(round(max_pixels * ny0 / nx0)))
    else:
        ny = max_pixels
        nx = max(1, int(round(max_pixels * nx0 / ny0)))
    return ny, nx


def pixel_extent(lo: int, hi: int, n_level: int, n_canvas: int) -> tuple[int, int]:
    p0 = int(np.floor(lo * n_canvas / n_level))
    p1 = int(np.ceil((hi + 1) * n_canvas / n_level))
    p0 = min(max(p0, 0), n_canvas - 1)
    p1 = min(max(p1, p0 + 1), n_canvas)
    return p0, p1


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


def sample_plotfile_values(
    plotfile: Path,
    component: str,
    gamma: float,
    ref_ratio: int,
    z_fraction: float,
    log10: bool,
    subtract_constant: float | None,
    max_samples_per_chunk: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    names, ndim, _time_value = parse_plot_header(plotfile)
    nlevels = level_count(plotfile)
    level_shapes = parse_level_shapes(plotfile)
    problo, probhi = parse_plot_bounds(plotfile)
    samples: list[tuple[np.ndarray, np.ndarray]] = []
    for lev in range(nlevels):
        level_shape = level_shapes[lev]
        cell_size = tuple((probhi[d] - problo[d]) / level_shape[d] for d in range(ndim))
        z_level = 0 if ndim == 2 else min(max(int(z_fraction * level_shape[2]), 0), level_shape[2] - 1)
        for lo, hi, data in level_chunks(plotfile, lev, ndim):
            values = component_values(data, names, component, gamma, cell_size)
            if subtract_constant is not None:
                values = values - subtract_constant
            plane = slice_to_xy(values, lo, hi, z_level)
            if plane is None:
                continue
            if log10:
                plane = np.log10(np.maximum(plane, 1.0e-14))
            finite = plane[np.isfinite(plane)].ravel()
            if finite.size == 0:
                continue
            stride = max(1, int(np.ceil(finite.size / max_samples_per_chunk)))
            sampled = finite[::stride]
            represented_measure = stride * cell_size[0] * cell_size[1]
            samples.append((sampled, np.full(sampled.shape, represented_measure)))
    return samples


def weighted_percentile(values: np.ndarray, weights: np.ndarray, percentile: float) -> float:
    if values.size == 0 or values.shape != weights.shape:
        raise ValueError("weighted percentile requires equally shaped nonempty values and weights")
    order = np.argsort(values)
    sorted_values = values[order]
    cumulative_weights = np.cumsum(weights[order])
    target = np.clip(percentile, 0.0, 100.0) * cumulative_weights[-1] / 100.0
    return float(sorted_values[min(np.searchsorted(cumulative_weights, target, side="left"), values.size - 1)])


def percentile_limits(
    plotfiles: list[Path],
    component: str,
    gamma: float,
    ref_ratio: int,
    z_fraction: float,
    log10: bool,
    subtract_constant: float | None,
    symmetric_limits: bool,
    max_samples_per_chunk: int,
) -> tuple[float, float]:
    samples: list[tuple[np.ndarray, np.ndarray]] = []
    for path in plotfiles:
        samples.extend(
            sample_plotfile_values(
                path,
                component,
                gamma,
                ref_ratio,
                z_fraction,
                log10,
                subtract_constant,
                max_samples_per_chunk,
            )
        )
    if not samples:
        raise ValueError(f"no finite values found for {component}")
    finite = np.concatenate([values for values, _weights in samples])
    weights = np.concatenate([sample_weights for _values, sample_weights in samples])
    if finite.size < 3:
        raise ValueError(f"not enough finite values to determine limits for {component}")

    vmin = weighted_percentile(finite, weights, 2.0)
    vmax = weighted_percentile(finite, weights, 98.0)
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        raise ValueError(f"non-finite sample values for {component}")
    if vmin == vmax:
        vmin = float(finite.min())
        vmax = float(finite.max())
        if vmin == vmax:
            return vmin, vmax

    linear_span = vmax - vmin
    centered = weighted_percentile(finite, weights, 50.0)
    centered_half_span = weighted_percentile(np.abs(finite - centered), weights, 90.0)
    if centered_half_span > 0.0 and centered_half_span * 2.0 < linear_span:
        vmin = centered - centered_half_span
        vmax = centered + centered_half_span
    elif centered_half_span > 0.0 and centered_half_span <= max(abs(centered), 1.0) * 0.4:
        vmin = centered - centered_half_span
        vmax = centered + centered_half_span
    if symmetric_limits or component == "vorticity3":
        vmax_abs = max(abs(vmin), abs(vmax), 1.0e-30)
        vmin = -vmax_abs
        vmax = vmax_abs
    return vmin, vmax


def composite_slice(
    plotfile: Path,
    component: str,
    gamma: float,
    ref_ratio: int,
    z_fraction: float,
    log10: bool,
    subtract_constant: float | None,
    max_pixels: int,
    xlim: tuple[float, float] = (0.0, 1.0),
    ylim: tuple[float, float] = (0.0, 1.0),
    show_base_boxes: bool = False,
    method: str = "nearest",
):
    _ = ref_ratio
    stored_component = "rho" if component == "schlieren" else component
    result = HierarchySlice(
        plotfile,
        stored_component,
        gamma,
        slice_axis="z",
        slice_fraction=z_fraction,
        xlim=xlim,
        ylim=ylim,
    ).uniform(max_pixels, method=method)
    image = result.values
    if subtract_constant is not None:
        image = image - subtract_constant
    if component != "schlieren" and log10:
        image = np.log10(np.maximum(image, 1.0e-14))

    if component == "schlieren":
        pixel_dx = (xlim[1] - xlim[0]) / image.shape[1]
        pixel_dy = (ylim[1] - ylim[0]) / image.shape[0]
        grad_y, grad_x = np.gradient(image, pixel_dy, pixel_dx, edge_order=1)
        image = np.sqrt(grad_x * grad_x + grad_y * grad_y)
        if subtract_constant is not None:
            image -= subtract_constant
        if log10:
            image = np.log10(np.maximum(image, 1.0e-14))

    boxes = list(result.boxes)
    if show_base_boxes:
        base = HierarchySlice(
            plotfile,
            stored_component,
            gamma,
            slice_axis="z",
            slice_fraction=z_fraction,
            xlim=xlim,
            ylim=ylim,
        )
        domain_x = (base.problo[0], base.probhi[0])
        domain_y = (base.problo[1], base.probhi[1])
        for tile in base.tiles():
            if tile.level != 0:
                continue
            level_x, level_y = tile.level_shape
            boxes.append(
                (
                    0,
                    domain_x[0] + tile.lo[0] * (domain_x[1] - domain_x[0]) / level_x,
                    domain_x[0] + (tile.hi[0] + 1) * (domain_x[1] - domain_x[0]) / level_x,
                    domain_y[0] + tile.lo[1] * (domain_y[1] - domain_y[0]) / level_y,
                    domain_y[0] + (tile.hi[1] + 1) * (domain_y[1] - domain_y[0]) / level_y,
                )
            )
    return image, boxes, result.time, result.ndim, list(result.names), xlim, ylim


def render_frame(
    plotfile: Path,
    component: str,
    output: Path,
    title: str,
    gamma: float,
    ref_ratio: int,
    z_fraction: float,
    log10: bool,
    subtract_constant: float | None,
    vmin: float,
    vmax: float,
    max_pixels: int,
    box_levels: set[int] | None,
    box_alpha: float,
    box_linewidth: float,
    hide_boxes: bool,
    show_base_boxes: bool,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    mask_plotfile: Path | None,
    mask_component: str,
    mask_threshold: float,
    solid_geometries: list[dict[str, float | str]],
    solid_alpha: float,
    solid_facecolor: str,
    solid_edgecolor: str,
    xlabel: str,
    ylabel: str,
    curvilinear_view: str,
    run_params: dict[str, str],
) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 6.2), constrained_layout=True)
    label = f"log10 {component}" if log10 else component
    if subtract_constant is not None:
        label = f"{component} - {subtract_constant:g}"
    if (component == "vorticity3" or subtract_constant is not None) and not log10:
        cmap = plt.get_cmap("coolwarm").copy()
    else:
        cmap = plt.get_cmap("magma" if log10 else "viridis").copy()
    cmap.set_bad(solid_facecolor if solid_geometries or mask_plotfile is not None else "#151515")
    tiles, physical_boxes, time_value, ndim = mapped_view_tiles(
        plotfile,
        component,
        gamma,
        ref_ratio,
        z_fraction,
        log10,
        subtract_constant,
        show_base_boxes,
        curvilinear_view,
        run_params,
    )
    if not tiles:
        raise ValueError(f"no mapped tiles produced for {plotfile}")
    im = None
    for tile in tiles:
        im = draw_tile_pcolormesh(ax, tile, cmap=cmap, vmin=vmin, vmax=vmax)
    if im is None:
        raise ValueError(f"no finite image mappable produced for {plotfile}")

    if mask_plotfile is not None:
        mask_tiles, _mask_boxes, _mask_time, _mask_ndim = mapped_view_tiles(
            mask_plotfile,
            mask_component,
            gamma,
            ref_ratio,
            z_fraction,
            False,
            None,
            False,
            curvilinear_view,
            run_params,
        )
        mask_cmap = plt.get_cmap("gray_r").copy()
        for tile in mask_tiles:
            masked_values = np.ma.masked_less_equal(tile.values, mask_threshold)
            mask_tile = PlotTile(tile.level, tile.edge_x, tile.edge_y, masked_values)
            draw_tile_pcolormesh(ax, mask_tile, cmap=mask_cmap, vmin=0.0, vmax=1.0).set_alpha(0.85)

    plot_xlim, plot_ylim = xlim, ylim
    colors = ["#36c9ff", "#ffd84d", "#ff4d6d", "#ffffff"]
    if not hide_boxes:
        for outline in physical_boxes:
            if box_levels is not None and outline.level not in box_levels:
                continue
            for xs, ys in outline.segments:
                ax.plot(
                    xs,
                    ys,
                    color=colors[(outline.level - 1) % len(colors)],
                    linewidth=box_linewidth,
                    alpha=box_alpha,
                )
    for geometry in solid_geometries:
        if str(geometry.get("kind", "")) == "box":
            xlo, xhi, ylo, yhi = box_extents_at_time(geometry, time_value)
            if xhi <= xlim[0] or xlo >= xlim[1] or yhi <= ylim[0] or ylo >= ylim[1]:
                continue
            draw_coordinate_box_outline(
                ax,
                max(xlo, xlim[0]),
                min(xhi, xlim[1]),
                max(ylo, ylim[0]),
                min(yhi, ylim[1]),
                curvilinear_view,
                color=solid_edgecolor,
                linewidth=1.4,
                alpha=solid_alpha,
                zorder=10,
            )
        elif str(geometry.get("kind", "")) == "circle":
            cx, cy, radius = circle_at_time(geometry, time_value)
            if cx + radius <= xlim[0] or cx - radius >= xlim[1] or cy + radius <= ylim[0] or cy - radius >= ylim[1]:
                continue
            ax.add_patch(
                Circle(
                    (cx, cy),
                    radius,
                    fill=False,
                    edgecolor=solid_edgecolor,
                    linewidth=1.4,
                    alpha=solid_alpha,
                    zorder=10,
                )
            )
    if run_params.get("pyfluid.boundary_profile", "").strip().lower() == "double_mach_reflection":
        wall_start = float(run_params.get("pyfluid.shock_x0", 1.0 / 6.0))
        wall_y = float(run_params.get("geometry.prob_lo", "0.0 0.0").split()[1])
        ax.plot(
            [wall_start, plot_xlim[1]],
            [wall_y, wall_y],
            color="#f5f5f5",
            linewidth=3.0,
            solid_capstyle="butt",
            zorder=12,
        )
        ax.plot(
            wall_start,
            wall_y,
            marker="^",
            markersize=7,
            markerfacecolor="#f5f5f5",
            markeredgecolor="#151515",
            markeredgewidth=0.7,
            zorder=13,
        )
    wrapped = "\n".join(textwrap.wrap(title, width=56))
    slice_text = f"z={z_fraction:.2f}, " if ndim == 3 else ""
    ax.set_title(f"{wrapped}\n{label}, {slice_text}t={time_value:.4f}, step={plotfile_step(plotfile)}", fontsize=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_aspect("equal")
    ax.set_xlim(plot_xlim)
    ax.set_ylim(plot_ylim)
    fig.colorbar(im, ax=ax, label=label)
    fig.savefig(output, dpi=170)
    plt.close(fig)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--gif", type=Path)
    parser.add_argument("--component", default="j_mag")
    parser.add_argument("--title", default="AMReX boxed slice")
    parser.add_argument("--gamma", type=float, default=5.0 / 3.0)
    parser.add_argument("--ref-ratio", type=int, default=2)
    parser.add_argument("--z-fraction", type=float, default=0.5)
    parser.add_argument("--log10", action="store_true")
    parser.add_argument("--subtract-constant", type=float)
    parser.add_argument("--symmetric-limits", action="store_true")
    parser.add_argument("--vmin", type=float)
    parser.add_argument("--vmax", type=float)
    parser.add_argument("--duration-ms", type=int, default=220)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--target-frames", type=int, help="Select this many frames uniformly across the available plotfiles.")
    parser.add_argument("--keep-frames", action="store_true", help="Keep per-frame PNGs after writing a GIF.")
    parser.add_argument("--max-pixels", type=int, default=1400)
    parser.add_argument("--max-samples-per-chunk", type=int, default=4096)
    parser.add_argument("--box-levels", help="Comma-separated AMR levels to outline; default outlines all refined levels.")
    parser.add_argument("--box-alpha", type=float, default=0.95)
    parser.add_argument("--box-linewidth", type=float, default=1.1)
    parser.add_argument("--hide-boxes", action="store_true")
    parser.add_argument("--show-base-boxes", action="store_true", help="Also outline level-0 AMReX grid boxes; useful for single-level geometry previews.")
    parser.add_argument("--xlim", nargs=2, type=float)
    parser.add_argument("--ylim", nargs=2, type=float)
    parser.add_argument("--mask-plotfile", type=Path)
    parser.add_argument("--mask-plot-dir", type=Path)
    parser.add_argument("--mask-component", default="solid_mask")
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--solid-box-lo", nargs="*", type=float)
    parser.add_argument("--solid-box-hi", nargs="*", type=float)
    parser.add_argument("--solid-geometry-file", type=Path, help="Explicit analytic solid overlay metadata. Default uses only discrete aux masks.")
    parser.add_argument("--solid-alpha", type=float, default=0.92)
    parser.add_argument("--solid-facecolor", default="#101010")
    parser.add_argument("--solid-edgecolor", default="#f4f4f4")
    parser.add_argument("--input", type=Path, help="Resolved run input file. Default: <plot-dir>/../run.in if present.")
    parser.add_argument("--xlabel")
    parser.add_argument("--ylabel")
    parser.add_argument(
        "--curvilinear-view",
        choices=("auto", "native", "r-theta", "r-phi", "multipatch"),
        default="auto",
        help=(
            "Render mapped data in physical geometry instead of rectangular coordinate space. "
            "auto selects multipatch for mixed patch systems and r-angle views for single curvilinear runs."
        ),
    )
    args = parser.parse_args()
    if args.mask_plotfile is not None and args.mask_plot_dir is not None:
        raise ValueError("--mask-plotfile and --mask-plot-dir are mutually exclusive")
    if args.frame_stride < 1:
        raise ValueError("--frame-stride must be >= 1")
    if args.target_frames is not None and args.target_frames < 2:
        raise ValueError("--target-frames must be >= 2")
    if args.target_frames is not None and args.frame_stride != 1:
        raise ValueError("--target-frames and --frame-stride are mutually exclusive")

    timeline = component_plotfile_timeline(
        args.plot_dir,
        args.component,
        require=True,
        frame_stride=args.frame_stride,
        target_frames=args.target_frames or 0,
    )
    plotfiles = [sample.path for sample in timeline]
    if not plotfiles:
        raise ValueError(f"no plotfiles found in {args.plot_dir}")

    if args.vmin is not None or args.vmax is not None:
        if args.vmin is None or args.vmax is None:
            raise ValueError("--vmin and --vmax must be supplied together")
        if args.vmin >= args.vmax:
            raise ValueError("--vmin must be smaller than --vmax")
        vmin, vmax = args.vmin, args.vmax
    else:
        vmin, vmax = percentile_limits(
            plotfiles,
            args.component,
            args.gamma,
            args.ref_ratio,
            args.z_fraction,
            args.log10,
            args.subtract_constant,
            args.symmetric_limits,
            args.max_samples_per_chunk,
        )

    box_levels = parse_box_levels(args.box_levels)
    input_path = args.input if args.input is not None else args.plot_dir.parent / "run.in"
    run_params = parse_key_value_file(input_path)
    curvilinear_view = default_curvilinear_view(run_params) if args.curvilinear_view == "auto" else args.curvilinear_view
    default_xlabel, default_ylabel = default_axis_labels(run_params, curvilinear_view)
    xlabel = args.xlabel if args.xlabel is not None else default_xlabel
    ylabel = args.ylabel if args.ylabel is not None else default_ylabel
    problo, probhi = parse_plot_bounds(plotfiles[0])
    if curvilinear_view == "multipatch":
        _names, ndim, _time_value = parse_plot_header(plotfiles[0])
        default_xlim, default_ylim = physical_multipatch_limits(run_params, ndim)
        xlim = parse_limits(args.xlim, default_xlim)
        ylim = parse_limits(args.ylim, default_ylim)
    else:
        coordinate_xlim = parse_limits(args.xlim, (problo[0], probhi[0]))
        coordinate_ylim = parse_limits(args.ylim, (problo[1], probhi[1]))
        if curvilinear_view in {"r-theta", "r-phi"}:
            xlim, ylim = transformed_limits(coordinate_xlim, coordinate_ylim, curvilinear_view)
        else:
            xlim, ylim = coordinate_xlim, coordinate_ylim
    manual_boxes = parse_solid_boxes(args.solid_box_lo, args.solid_box_hi)
    solid_geometries: list[dict[str, float | str]] = [
        {
            "kind": "box",
            "motion": "static",
            "xlo": xlo,
            "xhi": xhi,
            "ylo": ylo,
            "yhi": yhi,
            "v1": 0.0,
            "v2": 0.0,
        }
        for xlo, xhi, ylo, yhi in manual_boxes
    ]
    if not solid_geometries and args.solid_geometry_file is not None:
        solid_geometries = load_solid_geometry_from_metadata(args.solid_geometry_file)
    mask_plot_dir = args.mask_plot_dir
    mask_plotfiles = load_mask_plotfiles(mask_plot_dir, args.mask_component, required=mask_plot_dir is not None)
    if args.mask_plotfile is None and mask_plot_dir is None:
        aux_plot_dir = args.plot_dir.parent / "auxGF"
        mask_plotfiles = load_mask_plotfiles(aux_plot_dir, args.mask_component, required=False)
    frame_tag = "nobox" if args.hide_boxes else "boxes"
    if args.show_base_boxes and not args.hide_boxes:
        frame_tag = "baseboxes"
    if curvilinear_view != "native":
        frame_tag = f"{frame_tag}_{curvilinear_view.replace('-', '')}"
    frame_paths = []
    for path in plotfiles:
        step = plotfile_step(path)
        mask_plotfile = args.mask_plotfile
        if mask_plotfiles:
            if step not in mask_plotfiles:
                raise ValueError(f"no matching aux mask plotfile for step {step}")
            mask_plotfile = mask_plotfiles[step]
        frame_paths.append(
            render_frame(
                path,
                args.component,
                args.output_dir / f"{args.component}_{frame_tag}_step{step:05d}.png",
                args.title,
                args.gamma,
                args.ref_ratio,
                args.z_fraction,
                args.log10,
                args.subtract_constant,
                vmin,
                vmax,
                args.max_pixels,
                box_levels,
                args.box_alpha,
                args.box_linewidth,
                args.hide_boxes,
                args.show_base_boxes,
                xlim,
                ylim,
                mask_plotfile,
                args.mask_component,
                args.mask_threshold,
                solid_geometries,
                args.solid_alpha,
                args.solid_facecolor,
                args.solid_edgecolor,
                xlabel,
                ylabel,
                curvilinear_view,
                run_params,
            )
        )

    if args.gif is not None:
        gif_path = args.gif
        gif_path.parent.mkdir(parents=True, exist_ok=True)
        frames = [Image.open(path).convert("P", palette=Image.Palette.ADAPTIVE) for path in frame_paths]
        try:
            frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=args.duration_ms, loop=0)
        finally:
            for frame in frames:
                frame.close()
        if not args.keep_frames:
            for path in frame_paths:
                path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
