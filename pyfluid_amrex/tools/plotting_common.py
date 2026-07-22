#!/usr/bin/env python3
"""Shared postprocessing plots for PyFluid AMReX runs."""

from __future__ import annotations

import math
import textwrap
from pathlib import Path
from typing import Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from plot_amrex_slice import (
   axis_labels,
   composite_slice,
   parse_plot_header,
   select_component,
)
from plotfile_reader import latest_plotfile, plotfile_step
from plot_labels import component_math, diagnostic_family_components
from validation_common import next_numbered_path, parse_float, read_tsv


def positive(values: Sequence[float]) -> np.ndarray:
   arr = np.asarray(values, dtype=float)
   return np.maximum(np.abs(arr), 1.0e-18)


def float_scalar_rows(path: Path) -> list[dict[str, float]]:
   rows: list[dict[str, float]] = []
   for row in read_tsv(path):
      parsed: dict[str, float] = {}
      complete = True
      for key, value in row.items():
         number = parse_float(value)
         if not math.isfinite(number):
            complete = False
            break
         parsed[key] = number
      if complete:
         rows.append(parsed)
   if not rows:
      raise ValueError(f"no complete scalar rows found in {path}")
   return rows


def relative_change(rows: list[dict[str, float]], key: str) -> np.ndarray:
   initial = rows[0][key]
   scale = max(abs(initial), 1.0e-300)
   return np.asarray([abs(row[key] - initial) / scale for row in rows], dtype=float)


def has_activity(rows: list[dict[str, float]], key: str) -> bool:
   return key in rows[0] and any(abs(row[key]) > 0.0 for row in rows)


def plot_scalar_report(
   scalars_path: Path,
   output: Path,
   *,
   title: str,
) -> None:
   rows = float_scalar_rows(scalars_path)
   t = np.asarray([row["time"] for row in rows], dtype=float)
   markers = {"marker": "o", "markersize": 4.0} if len(rows) <= 24 else {}
   has_gravity = has_activity(rows, "gravity_potential_energy")
   energy_key = (
      "gravity_total_energy"
      if has_gravity and "gravity_total_energy" in rows[0]
      else "energy"
   )
   energy_label = "total energy" if energy_key == "gravity_total_energy" else "energy"
   has_mhd = (
      has_activity(rows, "newtonian_magnetic_energy")
      or has_activity(rows, "rmhd_electromagnetic_energy")
      or has_activity(rows, "divb_l1")
   )
   has_diffusion = has_activity(rows, "diffusion_residual") or has_activity(rows, "diffusion_iterations")
   has_wall = has_activity(rows, "wall_work") or has_activity(rows, "wall_power")
   has_amr = "amr_active_finest_level" in rows[0]

   output.parent.mkdir(parents=True, exist_ok=True)
   fig, axs = plt.subplots(2, 3, figsize=(14, 7.4), constrained_layout=True)
   fig.suptitle("\n".join(textwrap.wrap(title, 95)), fontsize=12)

   ax = axs[0, 0]
   if "mass" in rows[0]:
      ax.semilogy(t, positive(relative_change(rows, "mass")), label="mass", linewidth=2.0, **markers)
   if energy_key in rows[0]:
      ax.semilogy(t, positive(relative_change(rows, energy_key)), label=energy_label, linewidth=2.0, **markers)
   if has_activity(rows, "passive_mass"):
      ax.semilogy(t, positive(relative_change(rows, "passive_mass")), label="passive", linewidth=2.0, **markers)
   if not ax.lines:
      ax.plot(t, np.zeros_like(t), label="not selected", linewidth=2.0, **markers)
   ax.set_title("Conserved Drift")
   ax.set_xlabel("time")
   ax.set_ylabel("relative change")
   ax.legend()

   ax = axs[0, 1]
   for key in diagnostic_family_components("conserved_momentum"):
      if key in rows[0]:
         ax.plot(
            t,
            [row[key] for row in rows],
            label=f"${component_math(key)}$",
            linewidth=2.0,
            **markers,
         )
   if not ax.lines:
      ax.plot(t, np.zeros_like(t), label="not selected", linewidth=2.0, **markers)
   ax.set_title("Total Momentum")
   ax.set_xlabel("time")
   ax.legend()

   ax = axs[0, 2]
   if has_mhd and "divb_l1" in rows[0]:
      ax.semilogy(t, positive([row["divb_l1"] for row in rows]), label="divb L1", linewidth=2.0, **markers)
   if has_diffusion and "diffusion_residual" in rows[0]:
      ax.semilogy(t, positive([row["diffusion_residual"] for row in rows]), label="diffusion residual", linewidth=2.0, **markers)
   if has_wall and "energy_minus_wall_work" in rows[0]:
      ax.semilogy(t, positive([row["energy_minus_wall_work"] for row in rows]), label="E - E0 - work", linewidth=2.0, **markers)
   has_external_budget = all(
      key in rows[0]
      for key in (
         "external_budget_valid",
         "external_mass_rel_error",
         "external_energy_rel_error",
      )
   )
   if has_external_budget and any(row["external_budget_valid"] > 0.5 for row in rows):
      ax.semilogy(
         t,
         positive([row["external_mass_rel_error"] for row in rows]),
         label="mass ledger",
         linewidth=2.0,
         **markers,
      )
      ax.semilogy(
         t,
         positive([row["external_energy_rel_error"] for row in rows]),
         label="energy ledger",
         linewidth=2.0,
         linestyle=":",
         zorder=5,
         **markers,
      )
   if not ax.lines:
      ax.plot(t, np.zeros_like(t), label="no active constraint residual", linewidth=2.0, **markers)
   ax.set_title("Constraint / Residual")
   ax.set_xlabel("time")
   ax.legend()

   ax = axs[1, 0]
   if "min_rho" in rows[0]:
      ax.plot(t, [row["min_rho"] for row in rows], label="min rho", linewidth=2.0, **markers)
   if "max_rho" in rows[0]:
      ax.plot(t, [row["max_rho"] for row in rows], label="max rho", linewidth=2.0, **markers)
   if "min_pressure" in rows[0]:
      ax.plot(t, [row["min_pressure"] for row in rows], label="min P", linewidth=2.0, **markers)
   if not ax.lines:
      ax.plot(t, np.zeros_like(t), label="not selected", linewidth=2.0, **markers)
   ax.set_title("Floors")
   ax.set_xlabel("time")
   ax.legend()

   ax = axs[1, 1]
   if "dt" in rows[0]:
      ax.plot(t, [row["dt"] for row in rows], label="dt", linewidth=2.0, **markers)
   if "sim_time_per_s" in rows[0]:
      ax.plot(t, [row["sim_time_per_s"] for row in rows], label="sim time/s", linewidth=2.0, **markers)
   ax2 = ax.twinx()
   if "mcells_per_s" in rows[0]:
      ax2.plot(t, [row["mcells_per_s"] for row in rows], color="tab:orange", label="active Mcells/s", linewidth=2.0, **markers)
   if "layout_mcells_per_s" in rows[0]:
      ax2.plot(t, [row["layout_mcells_per_s"] for row in rows], color="tab:red", linestyle=":", label="layout Mcells/s", linewidth=2.0, **markers)
   if not ax.lines and not ax2.lines:
      ax.plot(t, np.zeros_like(t), label="not selected", linewidth=2.0, **markers)
   ax.set_title("Timestep / Throughput")
   ax.set_xlabel("time")
   ax.set_ylabel("dt, sim time/s")
   ax2.set_ylabel("Mcells/s")
   lines, labels = ax.get_legend_handles_labels()
   lines2, labels2 = ax2.get_legend_handles_labels()
   ax.legend(lines + lines2, labels + labels2)

   ax = axs[1, 2]
   if has_gravity and "gravity_total_energy_rel_drift" in rows[0]:
      ax.semilogy(t, positive([row["gravity_total_energy_rel_drift"] for row in rows]), label="gravity total drift", linewidth=2.0, **markers)
   if has_mhd and "newtonian_magnetic_energy" in rows[0]:
      ax.plot(
         t,
         [row["newtonian_magnetic_energy"] for row in rows],
         label="Newtonian magnetic energy",
         linewidth=2.0,
         **markers,
      )
   if has_mhd and "rmhd_electromagnetic_energy" in rows[0]:
      ax.plot(
         t,
         [row["rmhd_electromagnetic_energy"] for row in rows],
         label="Eulerian electromagnetic energy",
         linewidth=2.0,
         **markers,
      )
   if has_diffusion and "diffusion_iterations" in rows[0]:
      ax.plot(t, [row["diffusion_iterations"] for row in rows], label="diffusion sweeps", linewidth=2.0, **markers)
   if has_wall and "wall_work" in rows[0]:
      ax.plot(t, [row["wall_work"] for row in rows], label="wall work", linewidth=2.0, **markers)
   if has_amr:
      ax.plot(t, [row["amr_active_finest_level"] for row in rows], label="AMR finest L", linewidth=2.0, **markers)
      if "amr_total_budget_fraction" in rows[0]:
         budget = [row["amr_total_budget_fraction"] for row in rows]
         if any(value >= 0.0 for value in budget):
            ax.plot(t, budget, label="AMR budget frac", linewidth=2.0, **markers)
      if "amr_peak_level_budget_fraction" in rows[0]:
         peak_budget = [row["amr_peak_level_budget_fraction"] for row in rows]
         if any(value >= 0.0 for value in peak_budget):
            ax.plot(t, peak_budget, label="AMR peak L frac", linewidth=2.0, linestyle=":", **markers)
   if not ax.lines and energy_key in rows[0]:
      ax.plot(t, [row[energy_key] for row in rows], label=energy_label, linewidth=2.0, **markers)
   if not ax.lines:
      ax.plot(t, np.zeros_like(t), label="no active physics diagnostic", linewidth=2.0, **markers)
   ax.set_title("Active Physics / AMR")
   ax.set_xlabel("time")
   ax.legend()

   for ax in axs.flat:
      ax.grid(True, which="major", alpha=0.25)
      try:
         ax.ticklabel_format(useOffset=False)
      except AttributeError:
         pass
   fig.savefig(output, dpi=180)
   plt.close(fig)


def default_snapshot_components(names: list[str]) -> list[str]:
   components: list[str] = []
   if "rho" in names:
      components.append("rho")
   if all(name in names for name in ("rho", "mom1", "mom2", "mom3", "energy")):
      components.append("pressure")
      components.append("velocity_mag")
   if all(name in names for name in ("b1", "b2", "b3")):
      components.append("b_mag")
   return components[:4]


def default_aux_components(names: list[str]) -> list[str]:
   preferred = ["solid_mask", "divb", "mach", "plasma_beta", "j_mag", "e_mag", "vorticity_mag", "amr_tag_metric"]
   out: list[str] = []
   for component in preferred:
      if component in names:
         out.append(component)
      elif component.endswith("_mag"):
         prefix = component.removesuffix("_mag")
         if all(f"{prefix}_{axis}" in names for axis in ("x", "y", "z")):
            out.append(component)
   return out[:4]


def plot_multivar_snapshot(
   plotfile: Path,
   output: Path,
   *,
   components: Sequence[str],
   title: str,
   gamma: float = 5.0 / 3.0,
   ref_ratio: int = 2,
   max_pixels: int = 1000,
   slice_axis: str = "z",
   slice_fraction: float = 0.5,
   show_boxes: bool = False,
) -> list[str]:
   names, ndim, _time_value = parse_plot_header(plotfile)
   if ndim < 2:
      return []
   selected = list(components)
   if not selected:
      selected = default_snapshot_components(names)
   if not selected:
      return []

   ncols = min(len(selected), 4)
   nrows = int(math.ceil(len(selected) / ncols))
   fig, axs = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 4.1 * nrows), squeeze=False, constrained_layout=True)
   fig.suptitle("\n".join(textwrap.wrap(title, 100)), fontsize=12)
   xlabel, ylabel = axis_labels(ndim, slice_axis)
   rendered: list[str] = []
   boxes_for_overlay: list[tuple[int, float, float, float, float]] = []

   for ax, component in zip(axs.flat, selected, strict=False):
      try:
         image, boxes, time_value, _ndim, _names, xlim, ylim = composite_slice(
            plotfile,
            component,
            gamma,
            ref_ratio,
            slice_axis,
            slice_fraction,
            max_pixels,
         )
      except ValueError:
         ax.axis("off")
         continue
      boxes_for_overlay = boxes
      im = ax.imshow(
         image,
         origin="lower",
         aspect="equal",
         cmap="viridis",
         extent=(xlim[0], xlim[1], ylim[0], ylim[1]),
         interpolation="nearest",
      )
      if show_boxes:
         colors = ["#36c9ff", "#ffd84d", "#ff4d6d", "#ffffff"]
         for lev, xlo, xhi, ylo, yhi in boxes_for_overlay:
            ax.add_patch(
               Rectangle(
                  (xlo, ylo),
                  xhi - xlo,
                  yhi - ylo,
                  fill=False,
                  edgecolor=colors[(lev - 1) % len(colors)],
                  linewidth=0.8,
                  alpha=0.8,
               )
            )
      slice_text = f", {slice_axis}={slice_fraction:.2f}" if ndim == 3 else ""
      ax.set_title(f"{component}{slice_text}, t={time_value:.5f}")
      ax.set_xlabel(xlabel)
      ax.set_ylabel(ylabel)
      fig.colorbar(im, ax=ax, label=component)
      rendered.append(component)

   for ax in axs.flat[len(selected):]:
      ax.axis("off")
   output.parent.mkdir(parents=True, exist_ok=True)
   fig.savefig(output, dpi=180)
   plt.close(fig)
   return rendered


def numbered_plot_path(plot_dir: Path, stem: str) -> Path:
   return next_numbered_path(plot_dir, stem, ".png")


def stable_plot_path(plot_dir: Path, stem: str) -> Path:
   plot_dir.mkdir(parents=True, exist_ok=True)
   return plot_dir / f"{stem}.png"
