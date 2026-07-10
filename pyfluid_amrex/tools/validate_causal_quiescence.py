#!/usr/bin/env python3
"""Check that a static background stays static outside a source causal cone."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_amrex_boxed_movie import (
   component_values,
   level_chunks,
   level_count,
)
from plotfile_reader import (
   parse_level_cell_sizes,
   parse_plot_bounds,
   parse_plot_header,
)
from plotfile_reader import plotfile_step, sorted_plotfiles
from validation_common import read_key_values, repo_relative, write_tsv


FIELDS = [
   "step",
   "time",
   "quiet_cells",
   "inner_quiet_radius",
   "outer_quiet_radius",
   "max_rho_quiet",
   "max_pressure_quiet",
   "max_velocity_quiet",
   "status",
]


def param_float(params: dict[str, str], key: str, default: float) -> float:
   try:
      value = float(params.get(key, default))
   except (TypeError, ValueError):
      return default
   return value if math.isfinite(value) else default


def source_radial_bounds(params: dict[str, str], t: float, safety: float, width: float) -> tuple[float, float]:
   gamma = param_float(params, "pyfluid.gamma", 5.0 / 3.0)
   rho0 = param_float(params, "pyfluid.rho0", 1.0)
   pressure0 = param_float(params, "pyfluid.pressure0", 1.0)
   source_radius = param_float(params, "pyfluid.source_radius", 0.0)
   source_sigma = param_float(params, "pyfluid.source_radial_sigma", 0.0)
   sound_speed = math.sqrt(max(gamma * pressure0 / max(rho0, 1.0e-300), 0.0))
   causal_radius = safety * sound_speed * t + width * source_sigma
   return source_radius - causal_radius, source_radius + causal_radius


def plotfile_quiet_row(
   plotfile: Path,
   params: dict[str, str],
   *,
   safety: float,
   source_widths: float,
   tolerance: float,
) -> dict[str, object]:
   names, ndim, time_value = parse_plot_header(plotfile)
   if ndim != 2:
      raise SystemExit("causal quiescence currently supports 2D plotfiles")
   if "rho" not in names or "energy" not in names:
      raise SystemExit(f"{plotfile} is missing rho/energy components")

   gamma = param_float(params, "pyfluid.gamma", 5.0 / 3.0)
   rho0 = param_float(params, "pyfluid.rho0", 1.0)
   pressure0 = param_float(params, "pyfluid.pressure0", 1.0)
   velocity0 = np.asarray([
      param_float(params, "pyfluid.v1_0", 0.0),
      param_float(params, "pyfluid.v2_0", 0.0),
      param_float(params, "pyfluid.v3_0", 0.0),
   ])
   problo, _probhi = parse_plot_bounds(plotfile)
   level_dx = parse_level_cell_sizes(plotfile)
   nlevels = level_count(plotfile)
   source_kind = params.get("pyfluid.source", "none").strip().lower()
   if source_kind in ("", "none", "off"):
      inner, outer = -math.inf, math.inf
   else:
      inner, outer = source_radial_bounds(params, time_value, safety, source_widths)

   quiet_cells = 0
   max_rho = 0.0
   max_pressure = 0.0
   max_velocity = 0.0
   for lev in range(nlevels):
      dx = level_dx[lev]
      for lo, hi, data in level_chunks(plotfile, lev, ndim):
         rho = component_values(data, names, "rho", gamma)
         pressure = component_values(data, names, "pressure", gamma)
         v1 = component_values(data, names, "v1", gamma)
         v2 = component_values(data, names, "v2", gamma)
         v3 = component_values(data, names, "v3", gamma)
         shape = rho.shape
         ii = np.arange(lo[0], hi[0] + 1, dtype=float)
         radius = problo[0] + (ii + 0.5) * dx[0]
         quiet = np.ones_like(radius, dtype=bool) if math.isinf(inner) else (radius < inner) | (radius > outer)
         if not np.any(quiet):
            continue
         mask = quiet[:, np.newaxis]
         quiet_cells += int(np.count_nonzero(np.broadcast_to(mask, shape)))
         max_rho = max(max_rho, float(np.nanmax(np.abs(rho[mask.repeat(shape[1], axis=1)] - rho0))))
         max_pressure = max(
            max_pressure,
            float(np.nanmax(np.abs(pressure[mask.repeat(shape[1], axis=1)] - pressure0))),
         )
         velocity_delta = np.sqrt(
            (v1 - velocity0[0]) ** 2
            + (v2 - velocity0[1]) ** 2
            + (v3 - velocity0[2]) ** 2
         )
         max_velocity = max(max_velocity, float(np.nanmax(velocity_delta[mask.repeat(shape[1], axis=1)])))

   metric = max(max_rho / max(abs(rho0), 1.0), max_pressure / max(abs(pressure0), 1.0), max_velocity)
   return {
      "step": plotfile_step(plotfile),
      "time": time_value,
      "quiet_cells": quiet_cells,
      "inner_quiet_radius": inner,
      "outer_quiet_radius": outer,
      "max_rho_quiet": max_rho,
      "max_pressure_quiet": max_pressure,
      "max_velocity_quiet": max_velocity,
      "status": "PASS" if quiet_cells > 0 and metric <= tolerance else "FAIL",
   }


def write_plot(path: Path, rows: list[dict[str, object]], title: str) -> None:
   t = np.asarray([float(row["time"]) for row in rows])
   fig, ax = plt.subplots(figsize=(7.5, 4.8), constrained_layout=True)
   for key, label in [
      ("max_rho_quiet", "rho"),
      ("max_pressure_quiet", "pressure"),
      ("max_velocity_quiet", "velocity"),
   ]:
      y = np.asarray([float(row[key]) for row in rows])
      ax.semilogy(t, np.maximum(y, 1.0e-300), marker="o", linewidth=2.0, label=label)
   ax.set_title(title)
   ax.set_xlabel("time")
   ax.set_ylabel("max quiet-region change")
   ax.grid(True, which="major", alpha=0.25)
   ax.legend()
   path.parent.mkdir(parents=True, exist_ok=True)
   fig.savefig(path, dpi=180)
   plt.close(fig)


def main() -> None:
   parser = argparse.ArgumentParser()
   parser.add_argument("run_dir", type=Path)
   parser.add_argument("--input", type=Path)
   parser.add_argument("--output", type=Path)
   parser.add_argument("--plot", type=Path)
   parser.add_argument("--safety", type=float, default=1.20)
   parser.add_argument("--source-widths", type=float, default=4.0)
   parser.add_argument("--tolerance", type=float, default=1.0e-8)
   args = parser.parse_args()

   run_dir = args.run_dir.resolve()
   input_path = args.input if args.input is not None else run_dir / "run.in"
   params = read_key_values(input_path)
   plotfiles = sorted_plotfiles(run_dir, require=True)
   rows = [
      plotfile_quiet_row(
         plotfile,
         params,
         safety=args.safety,
         source_widths=args.source_widths,
         tolerance=args.tolerance,
      )
      for plotfile in plotfiles
   ]

   output = args.output if args.output is not None else run_dir / "causal_quiescence.tsv"
   write_tsv(output, rows, FIELDS)
   if args.plot is not None:
      write_plot(args.plot, rows, f"{run_dir.name}: causal quiescence")
   failures = [row for row in rows if row["status"] != "PASS"]
   status = "PASS" if not failures else "FAIL"
   worst = max(
      max(float(row["max_rho_quiet"]), float(row["max_pressure_quiet"]), float(row["max_velocity_quiet"]))
      for row in rows
   )
   print(f"{status} causal_quiescence run={repo_relative(run_dir)} worst={worst:.3e} rows={len(rows)}")
   if failures:
      raise SystemExit(1)


if __name__ == "__main__":
   main()
