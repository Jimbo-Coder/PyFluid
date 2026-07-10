#!/usr/bin/env python3
"""Draw mixed multipatch grid geometry without running a fluid evolution."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plot_amrex_boxed_movie import build_visual_patches, physical_patch_edges


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTDIR = ROOT / "plots/current/multipatch_geometry"


def base_params(
   patch_system: str,
   axes: str,
   inner_radius: float,
   outer_radius: float,
   medium_radius: float,
   central_cells: str,
   angular_cells: int,
   radial_cells: int,
   outer_radial_cells: int,
) -> dict[str, str]:
   return {
      "mesh.patch_system": patch_system,
      "mesh.axes": axes,
      "mesh.patch_gap_cells": "0",
      "mesh.patch_inner_radius": f"{inner_radius:.16g}",
      "mesh.patch_medium_radius": f"{medium_radius:.16g}",
      "mesh.patch_outer_radius": f"{outer_radius:.16g}",
      "mesh.patch_central_cells": central_cells,
      "mesh.patch_angular_cells": str(angular_cells),
      "mesh.patch_radial_cells": str(radial_cells),
      "mesh.patch_outer_radial_cells": str(outer_radial_cells),
      "mesh.patch_axial_cells": str(radial_cells),
      "mesh.patch_axial_lo": "-1.0",
      "mesh.patch_axial_hi": "1.0",
   }


def geometry_cases() -> list[tuple[str, dict[str, str]]]:
   cases: list[tuple[str, dict[str, str]]] = []
   cases.append(
      (
         "Thornburg04 xy, inner=1 outer=4",
         base_params("mixed_thornburg04", "x y", 1.0, 4.0, 2.5, "16 16", 16, 12, 12),
      )
   )
   cases.append(
      (
         "Thornburg04 xz, inner=1 outer=4",
         base_params("mixed_thornburg04", "x z", 1.0, 4.0, 2.5, "16 16", 16, 12, 12),
      )
   )
   cases.append(
      (
         "Thornburg13 xy, medium shell",
         base_params("mixed_thornburg13", "x y", 1.0, 5.0, 2.4, "16 16", 16, 10, 14),
      )
   )
   cases.append(
      (
         "Thornburg13 xz, medium shell",
         base_params("mixed_thornburg13", "x z", 1.0, 5.0, 2.4, "16 16", 16, 10, 14),
      )
   )
   cases.append(
      (
         "Cylindrical5 xy, inner=1 outer=4",
         base_params("mixed_cylindrical5", "x y", 1.0, 4.0, 2.5, "16 16", 16, 12, 12),
      )
   )
   cases.append(
      (
         "Cylindrical9 xy, medium shell",
         base_params("mixed_cylindrical9", "x y", 1.0, 5.0, 2.4, "16 16", 16, 10, 14),
      )
   )
   cases.append(
      (
         "Thornburg04 xy, larger Cartesian core",
         base_params("mixed_thornburg04", "x y", 1.6, 5.0, 3.0, "20 20", 18, 12, 12),
      )
   )
   cases.append(
      (
         "Thornburg04 xy, smaller Cartesian core",
         base_params("mixed_thornburg04", "x y", 0.65, 4.0, 2.0, "12 12", 18, 14, 14),
      )
   )
   return cases


def patch_color(patch_id: int) -> str:
   palette = plt.get_cmap("tab20").colors
   return palette[patch_id % len(palette)]


def draw_sparse_grid(
   ax: plt.Axes,
   params: dict[str, str],
   *,
   max_lines_per_direction: int,
   line_width: float,
   alpha: float,
) -> None:
   patches = build_visual_patches(params, ndim=2, refinement=1)
   all_x: list[np.ndarray] = []
   all_y: list[np.ndarray] = []
   for patch in patches:
      color = patch_color(patch.patch_id)
      nx, ny = patch.cells
      x_edges, y_edges = physical_patch_edges(patch, patch.owned_lo, patch.owned_hi)
      all_x.append(x_edges.ravel())
      all_y.append(y_edges.ravel())

      stride_x = max(1, nx // max_lines_per_direction)
      stride_y = max(1, ny // max_lines_per_direction)
      for i in range(0, nx + 1, stride_x):
         ax.plot(x_edges[:, i], y_edges[:, i], color=color, lw=line_width, alpha=alpha)
      if nx % stride_x != 0:
         ax.plot(x_edges[:, -1], y_edges[:, -1], color=color, lw=line_width, alpha=alpha)
      for j in range(0, ny + 1, stride_y):
         ax.plot(x_edges[j, :], y_edges[j, :], color=color, lw=line_width, alpha=alpha)
      if ny % stride_y != 0:
         ax.plot(x_edges[-1, :], y_edges[-1, :], color=color, lw=line_width, alpha=alpha)

      ax.plot(x_edges[0, :], y_edges[0, :], color="black", lw=1.1)
      ax.plot(x_edges[-1, :], y_edges[-1, :], color="black", lw=1.1)
      ax.plot(x_edges[:, 0], y_edges[:, 0], color="black", lw=1.1)
      ax.plot(x_edges[:, -1], y_edges[:, -1], color="black", lw=1.1)

      center_x = float(np.mean(x_edges))
      center_y = float(np.mean(y_edges))
      ax.text(center_x, center_y, str(patch.patch_id), ha="center", va="center", fontsize=8, color=color)

   x = np.concatenate(all_x)
   y = np.concatenate(all_y)
   pad = 0.05 * max(float(np.ptp(x)), float(np.ptp(y)), 1.0e-12)
   ax.set_xlim(float(np.min(x) - pad), float(np.max(x) + pad))
   ax.set_ylim(float(np.min(y) - pad), float(np.max(y) + pad))
   ax.set_aspect("equal", adjustable="box")
   ax.grid(False)


def save_case_panel(outdir: Path) -> Path:
   cases = geometry_cases()
   fig, axes = plt.subplots(2, 4, figsize=(15.5, 7.6), constrained_layout=True)
   for ax, (title, params) in zip(axes.ravel(), cases, strict=True):
      draw_sparse_grid(ax, params, max_lines_per_direction=8, line_width=0.55, alpha=0.72)
      ax.set_title(title, fontsize=10)
      ax.set_xlabel("physical axis 1")
      ax.set_ylabel("physical axis 2")
   path = outdir / "mixed_patch_geometry_overview.png"
   fig.suptitle("Mixed patch geometry: Cartesian core plus true curvilinear shell patches", fontsize=13)
   fig.savefig(path, dpi=180)
   plt.close(fig)
   return path


def save_transition_zoom(outdir: Path) -> Path:
   cases = [
      (
         "Thornburg04 xy seam, inner=1",
         base_params("mixed_thornburg04", "x y", 1.0, 4.0, 2.5, "24 24", 24, 18, 18),
      ),
      (
         "Thornburg04 xz seam, inner=1",
         base_params("mixed_thornburg04", "x z", 1.0, 4.0, 2.5, "24 24", 24, 18, 18),
      ),
      (
         "Thornburg04 xy seam, inner=1.6",
         base_params("mixed_thornburg04", "x y", 1.6, 5.0, 3.0, "28 28", 28, 18, 18),
      ),
   ]
   fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4), constrained_layout=True)
   for ax, (title, params) in zip(axes, cases, strict=True):
      draw_sparse_grid(ax, params, max_lines_per_direction=14, line_width=0.45, alpha=0.68)
      inner_radius = float(params["mesh.patch_inner_radius"])
      ax.set_xlim(-1.4 * inner_radius, 2.4 * inner_radius)
      ax.set_ylim(-0.25 * inner_radius, 1.7 * inner_radius)
      ax.set_title(title, fontsize=10)
      ax.set_xlabel("physical axis 1")
      ax.set_ylabel("physical axis 2")
   path = outdir / "mixed_patch_geometry_transition_zoom.png"
   fig.suptitle("Cartesian-to-shell transition zoom", fontsize=13)
   fig.savefig(path, dpi=200)
   plt.close(fig)
   return path


def physical_cell_centers(edge_x: np.ndarray, edge_y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
   center_x = 0.25 * (edge_x[:-1, :-1] + edge_x[1:, :-1] + edge_x[:-1, 1:] + edge_x[1:, 1:])
   center_y = 0.25 * (edge_y[:-1, :-1] + edge_y[1:, :-1] + edge_y[:-1, 1:] + edge_y[1:, 1:])
   return center_x, center_y


def cell_areas(edge_x: np.ndarray, edge_y: np.ndarray) -> np.ndarray:
   x0, y0 = edge_x[:-1, :-1], edge_y[:-1, :-1]
   x1, y1 = edge_x[:-1, 1:], edge_y[:-1, 1:]
   x2, y2 = edge_x[1:, 1:], edge_y[1:, 1:]
   x3, y3 = edge_x[1:, :-1], edge_y[1:, :-1]
   twice_area = (
      x0 * y1 - y0 * x1
      + x1 * y2 - y1 * x2
      + x2 * y3 - y2 * x3
      + x3 * y0 - y3 * x0
   )
   return 0.5 * np.abs(twice_area)


def cell_aspect_ratios(edge_x: np.ndarray, edge_y: np.ndarray) -> np.ndarray:
   x0, y0 = edge_x[:-1, :-1], edge_y[:-1, :-1]
   x1, y1 = edge_x[:-1, 1:], edge_y[:-1, 1:]
   x2, y2 = edge_x[1:, 1:], edge_y[1:, 1:]
   x3, y3 = edge_x[1:, :-1], edge_y[1:, :-1]
   lengths = np.stack(
      [
         np.hypot(x1 - x0, y1 - y0),
         np.hypot(x2 - x1, y2 - y1),
         np.hypot(x3 - x2, y3 - y2),
         np.hypot(x0 - x3, y0 - y3),
      ],
      axis=0,
   )
   return np.max(lengths, axis=0) / np.maximum(np.min(lengths, axis=0), 1.0e-300)


def synthetic_field(x: np.ndarray, y: np.ndarray) -> np.ndarray:
   radius = np.sqrt(x * x + y * y)
   angle = np.arctan2(y, x)
   return np.exp(-0.08 * radius * radius) * (1.0 + 0.25 * np.cos(3.0 * angle)) + 0.18 * np.sin(1.7 * x - 0.9 * y)


def draw_synthetic_render(ax: plt.Axes, params: dict[str, str]) -> None:
   patches = build_visual_patches(params, ndim=2, refinement=1)
   all_x: list[np.ndarray] = []
   all_y: list[np.ndarray] = []
   for patch in patches:
      edge_x, edge_y = physical_patch_edges(patch, patch.owned_lo, patch.owned_hi)
      center_x, center_y = physical_cell_centers(edge_x, edge_y)
      values = synthetic_field(center_x, center_y)
      ax.pcolormesh(edge_x, edge_y, values, shading="flat", cmap="viridis", vmin=0.0, vmax=1.35)
      all_x.append(edge_x.ravel())
      all_y.append(edge_y.ravel())
      ax.plot(edge_x[0, :], edge_y[0, :], color="white", lw=0.6, alpha=0.95)
      ax.plot(edge_x[-1, :], edge_y[-1, :], color="white", lw=0.6, alpha=0.95)
      ax.plot(edge_x[:, 0], edge_y[:, 0], color="white", lw=0.6, alpha=0.95)
      ax.plot(edge_x[:, -1], edge_y[:, -1], color="white", lw=0.6, alpha=0.95)
   x = np.concatenate(all_x)
   y = np.concatenate(all_y)
   pad = 0.05 * max(float(np.ptp(x)), float(np.ptp(y)), 1.0e-12)
   ax.set_xlim(float(np.min(x) - pad), float(np.max(x) + pad))
   ax.set_ylim(float(np.min(y) - pad), float(np.max(y) + pad))
   ax.set_aspect("equal", adjustable="box")
   ax.grid(False)


def save_synthetic_render_panel(outdir: Path) -> Path:
   cases = [
      (
         "Thornburg04 xy",
         base_params("mixed_thornburg04", "x y", 1.0, 4.0, 2.5, "20 20", 24, 16, 16),
      ),
      (
         "Thornburg04 xz",
         base_params("mixed_thornburg04", "x z", 1.0, 4.0, 2.5, "20 20", 24, 16, 16),
      ),
      (
         "Thornburg13 xy",
         base_params("mixed_thornburg13", "x y", 1.0, 5.0, 2.4, "20 20", 24, 12, 18),
      ),
      (
         "Cylindrical9 xy",
         base_params("mixed_cylindrical9", "x y", 1.0, 5.0, 2.4, "20 20", 24, 12, 18),
      ),
   ]
   fig, axes = plt.subplots(1, 4, figsize=(15.5, 4.2), constrained_layout=True)
   for ax, (title, params) in zip(axes, cases, strict=True):
      draw_synthetic_render(ax, params)
      ax.set_title(title, fontsize=10)
      ax.set_xlabel("physical axis 1")
      ax.set_ylabel("physical axis 2")
   path = outdir / "mixed_patch_geometry_synthetic_render.png"
   fig.suptitle("Honest physical-space render of a smooth synthetic field", fontsize=13)
   fig.savefig(path, dpi=180)
   plt.close(fig)
   return path


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


def shell_global_to_local(
   coordinate_map: str,
   global_components: tuple[np.ndarray, np.ndarray, np.ndarray],
   origin: tuple[float, float, float],
   radial_inner: float,
   radial_outer: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
   centered = tuple(component - offset for component, offset in zip(global_components, origin, strict=True))
   axis, _sign = shell_axis_and_sign(coordinate_map)
   eps = 1.0e-300
   if coordinate_map.startswith("thornburg_shell"):
      if axis == 0:
         primary = np.where(np.abs(centered[0]) > eps, centered[0], np.nan)
         tangent_a = centered[2] / primary
         tangent_b = centered[1] / primary
      if axis == 1:
         primary = np.where(np.abs(centered[1]) > eps, centered[1], np.nan)
         tangent_a = centered[2] / primary
         tangent_b = centered[0] / primary
      if axis == 2:
         primary = np.where(np.abs(centered[2]) > eps, centered[2], np.nan)
         tangent_a = centered[1] / primary
         tangent_b = centered[0] / primary
      physical_radius = np.sqrt(centered[0] ** 2 + centered[1] ** 2 + centered[2] ** 2)
      angular_norm = np.sqrt(1.0 + tangent_a**2 + tangent_b**2)
      radial_denominator = radial_outer - radial_inner
      quadratic = (1.0 - angular_norm) / radial_denominator
      linear = angular_norm - quadratic * radial_inner
      radial_coordinate = 2.0 * physical_radius / (linear + np.sqrt(linear**2 + 4.0 * quadratic * physical_radius))
      return np.arctan(tangent_a), np.arctan(tangent_b), radial_coordinate
   if coordinate_map.startswith("cylindrical_shell"):
      if axis == 0:
         primary = np.where(np.abs(centered[0]) > eps, centered[0], np.nan)
         tangent_a = centered[1] / primary
      else:
         primary = np.where(np.abs(centered[1]) > eps, centered[1], np.nan)
         tangent_a = centered[0] / primary
      physical_radius = np.sqrt(centered[0] ** 2 + centered[1] ** 2)
      angular_norm = np.sqrt(1.0 + tangent_a**2)
      radial_denominator = radial_outer - radial_inner
      quadratic = (1.0 - angular_norm) / radial_denominator
      linear = angular_norm - quadratic * radial_inner
      radial_coordinate = 2.0 * physical_radius / (linear + np.sqrt(linear**2 + 4.0 * quadratic * physical_radius))
      return np.arctan(tangent_a), centered[2], radial_coordinate
   raise ValueError(f"unsupported coordinate map: {coordinate_map}")


def patch_contains_points(
   patch,
   x: np.ndarray,
   y: np.ndarray,
) -> np.ndarray:
   components = [np.full_like(x, patch.origin[index]) for index in range(3)]
   components[patch.display_components[0]] = x
   components[patch.display_components[1]] = y
   global_components = (components[0], components[1], components[2])
   if patch.coordinate_map in {"cartesian_identity", "thornburg_central_cartesian"}:
      local = tuple(component - offset for component, offset in zip(global_components, patch.origin, strict=True))
      sign_ok = np.ones_like(x, dtype=bool)
   else:
      axis, sign = shell_axis_and_sign(patch.coordinate_map)
      centered_axis = global_components[axis] - patch.origin[axis]
      sign_ok = sign * centered_axis >= -1.0e-12
      local = shell_global_to_local(
         patch.coordinate_map,
         global_components,
         patch.origin,
         patch.local_lo[2],
         patch.local_hi[2],
      )

   inside = sign_ok
   for component in patch.local_axes:
      lo = patch.local_lo[component]
      hi = patch.local_hi[component]
      value = local[component]
      inside = inside & np.isfinite(value) & (value >= lo - 1.0e-12) & (value <= hi + 1.0e-12)
   return inside


def draw_coverage_count(ax: plt.Axes, params: dict[str, str], samples: int = 420) -> None:
   patches = build_visual_patches(params, ndim=2, refinement=1)
   x_limits, y_limits = [], []
   for patch in patches:
      edge_x, edge_y = physical_patch_edges(patch, patch.owned_lo, patch.owned_hi)
      x_limits.extend([float(np.min(edge_x)), float(np.max(edge_x))])
      y_limits.extend([float(np.min(edge_y)), float(np.max(edge_y))])
   pad = 0.03 * max(max(x_limits) - min(x_limits), max(y_limits) - min(y_limits), 1.0e-12)
   xs = np.linspace(min(x_limits) - pad, max(x_limits) + pad, samples)
   ys = np.linspace(min(y_limits) - pad, max(y_limits) + pad, samples)
   xx, yy = np.meshgrid(xs, ys)
   count = np.zeros_like(xx, dtype=float)
   for patch in patches:
      count += patch_contains_points(patch, xx, yy)
      edge_x, edge_y = physical_patch_edges(patch, patch.owned_lo, patch.owned_hi)
      ax.plot(edge_x[0, :], edge_y[0, :], color="black", lw=0.45, alpha=0.8)
      ax.plot(edge_x[-1, :], edge_y[-1, :], color="black", lw=0.45, alpha=0.8)
      ax.plot(edge_x[:, 0], edge_y[:, 0], color="black", lw=0.45, alpha=0.8)
      ax.plot(edge_x[:, -1], edge_y[:, -1], color="black", lw=0.45, alpha=0.8)
   displayed = np.ma.masked_equal(count, 0.0)
   cmap = plt.get_cmap("viridis", 4).copy()
   cmap.set_bad("#151515")
   ax.pcolormesh(xs, ys, displayed, shading="auto", cmap=cmap, vmin=0.0, vmax=4.0)
   ax.set_aspect("equal", adjustable="box")
   ax.set_xlim(xs[0], xs[-1])
   ax.set_ylim(ys[0], ys[-1])
   ax.grid(False)


def save_coverage_panel(outdir: Path) -> Path:
   cases = [
      (
         "Thornburg04 xy",
         base_params("mixed_thornburg04", "x y", 1.0, 4.0, 2.5, "20 20", 20, 14, 14),
      ),
      (
         "Thornburg04 xz",
         base_params("mixed_thornburg04", "x z", 1.0, 4.0, 2.5, "20 20", 20, 14, 14),
      ),
      (
         "Thornburg13 xy",
         base_params("mixed_thornburg13", "x y", 1.0, 5.0, 2.4, "20 20", 20, 10, 16),
      ),
      (
         "Cylindrical9 xy",
         base_params("mixed_cylindrical9", "x y", 1.0, 5.0, 2.4, "20 20", 20, 10, 16),
      ),
   ]
   fig, axes = plt.subplots(1, 4, figsize=(15.5, 4.2), constrained_layout=True)
   for ax, (title, params) in zip(axes, cases, strict=True):
      draw_coverage_count(ax, params)
      ax.set_title(title, fontsize=10)
      ax.set_xlabel("physical axis 1")
      ax.set_ylabel("physical axis 2")
   path = outdir / "mixed_patch_geometry_coverage_count.png"
   fig.suptitle("Physical coverage count: black=hole, one color unit per covering patch", fontsize=13)
   fig.savefig(path, dpi=180)
   plt.close(fig)
   return path


def draw_cell_quality(ax: plt.Axes, params: dict[str, str], quantity: str) -> None:
   patches = build_visual_patches(params, ndim=2, refinement=1)
   all_x: list[np.ndarray] = []
   all_y: list[np.ndarray] = []
   all_area: list[np.ndarray] = []
   for patch in patches:
      edge_x, edge_y = physical_patch_edges(patch, patch.owned_lo, patch.owned_hi)
      all_x.append(edge_x.ravel())
      all_y.append(edge_y.ravel())
      all_area.append(cell_areas(edge_x, edge_y).ravel())
   reference_area = float(np.median(np.concatenate(all_area)))

   image = None
   for patch in patches:
      edge_x, edge_y = physical_patch_edges(patch, patch.owned_lo, patch.owned_hi)
      if quantity == "area":
         values = np.log10(cell_areas(edge_x, edge_y) / reference_area)
         image = ax.pcolormesh(edge_x, edge_y, values, shading="flat", cmap="coolwarm", vmin=-0.7, vmax=0.7)
      elif quantity == "aspect":
         values = cell_aspect_ratios(edge_x, edge_y)
         image = ax.pcolormesh(edge_x, edge_y, values, shading="flat", cmap="magma", vmin=1.0, vmax=2.4)
      else:
         raise ValueError(f"unknown cell-quality quantity {quantity}")
      ax.plot(edge_x[0, :], edge_y[0, :], color="black", lw=0.45, alpha=0.75)
      ax.plot(edge_x[-1, :], edge_y[-1, :], color="black", lw=0.45, alpha=0.75)
      ax.plot(edge_x[:, 0], edge_y[:, 0], color="black", lw=0.45, alpha=0.75)
      ax.plot(edge_x[:, -1], edge_y[:, -1], color="black", lw=0.45, alpha=0.75)

   x = np.concatenate(all_x)
   y = np.concatenate(all_y)
   pad = 0.05 * max(float(np.ptp(x)), float(np.ptp(y)), 1.0e-12)
   ax.set_xlim(float(np.min(x) - pad), float(np.max(x) + pad))
   ax.set_ylim(float(np.min(y) - pad), float(np.max(y) + pad))
   ax.set_aspect("equal", adjustable="box")
   ax.grid(False)
   if image is not None:
      label = "log10(cell area / median)" if quantity == "area" else "cell aspect ratio"
      plt.colorbar(image, ax=ax, label=label, fraction=0.046, pad=0.02)


def write_quality_summary(outdir: Path, cases: list[tuple[str, dict[str, str]]]) -> Path:
   path = outdir / "mixed_patch_geometry_quality.tsv"
   with path.open("w", encoding="utf-8") as stream:
      stream.write("case\tpatch_id\tarea_min_ratio\tarea_max_ratio\taspect_max\n")
      for title, params in cases:
         patches = build_visual_patches(params, ndim=2, refinement=1)
         all_area: list[np.ndarray] = []
         patch_edges: list[tuple[int, np.ndarray, np.ndarray]] = []
         for patch in patches:
            edge_x, edge_y = physical_patch_edges(patch, patch.owned_lo, patch.owned_hi)
            patch_edges.append((patch.patch_id, edge_x, edge_y))
            all_area.append(cell_areas(edge_x, edge_y).ravel())
         reference_area = float(np.median(np.concatenate(all_area)))
         for patch_id, edge_x, edge_y in patch_edges:
            area_ratio = cell_areas(edge_x, edge_y) / reference_area
            aspect = cell_aspect_ratios(edge_x, edge_y)
            stream.write(
               f"{title}\t{patch_id}\t{np.min(area_ratio):.8e}\t"
               f"{np.max(area_ratio):.8e}\t{np.max(aspect):.8e}\n"
            )
   return path


def save_cell_quality_panel(outdir: Path) -> Path:
   cases = [
      (
         "Thornburg04 xy",
         base_params("mixed_thornburg04", "x y", 1.0, 4.0, 2.5, "20 20", 24, 16, 16),
      ),
      (
         "Thornburg13 xy",
         base_params("mixed_thornburg13", "x y", 1.0, 5.0, 2.4, "20 20", 24, 12, 18),
      ),
      (
         "Cylindrical9 xy",
         base_params("mixed_cylindrical9", "x y", 1.0, 5.0, 2.4, "20 20", 24, 12, 18),
      ),
   ]
   fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.0), constrained_layout=True)
   for column, (title, params) in enumerate(cases):
      draw_cell_quality(axes[0, column], params, "area")
      draw_cell_quality(axes[1, column], params, "aspect")
      axes[0, column].set_title(f"{title}: area", fontsize=10)
      axes[1, column].set_title(f"{title}: aspect", fontsize=10)
      axes[1, column].set_xlabel("physical axis 1")
      axes[0, column].set_ylabel("physical axis 2")
      axes[1, column].set_ylabel("physical axis 2")
   path = outdir / "mixed_patch_geometry_cell_quality.png"
   fig.suptitle("Physical cell-size diagnostics for mixed patch construction", fontsize=13)
   fig.savefig(path, dpi=180)
   plt.close(fig)
   write_quality_summary(outdir, cases)
   return path


def main() -> None:
   parser = argparse.ArgumentParser(description=__doc__)
   parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
   args = parser.parse_args()

   args.outdir.mkdir(parents=True, exist_ok=True)
   paths = [
      save_case_panel(args.outdir),
      save_transition_zoom(args.outdir),
      save_synthetic_render_panel(args.outdir),
      save_coverage_panel(args.outdir),
      save_cell_quality_panel(args.outdir),
   ]
   for path in paths:
      print(path)


if __name__ == "__main__":
   main()
