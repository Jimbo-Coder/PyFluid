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

from causal_validation import infer_localized_causal_model
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
   "max_magnetic_quiet",
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
   source_sigma = param_float(params, "pyfluid.source_sigma", 0.0)
   sound_speed = math.sqrt(max(gamma * pressure0 / max(rho0, 1.0e-300), 0.0))
   causal_radius = safety * sound_speed * t + width * source_sigma
   return source_radius - causal_radius, source_radius + causal_radius


def mesh_axes(params: dict[str, str], ndim: int) -> tuple[str, ...]:
   axes = tuple(params.get("mesh.axes", "").replace(",", " ").split())
   if axes:
      return axes
   coordinate = params.get("mesh.coordinate", "cartesian").strip().lower()
   return {
      "cartesian": ("x", "y", "z"),
      "cylindrical": ("r", "phi", "z"),
      "spherical": ("r", "theta", "phi"),
   }.get(coordinate, ("x", "y", "z"))[:ndim]


def physical_coordinates(
   params: dict[str, str],
   axes: tuple[str, ...],
   mesh: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
   shape = mesh[0].shape
   coordinate = params.get("mesh.coordinate", "cartesian").strip().lower()
   values = {
      "x": np.zeros(shape),
      "y": np.zeros(shape),
      "z": np.zeros(shape),
      "r": np.ones(shape),
      "theta": np.full(shape, 0.5 * math.pi),
      "phi": np.zeros(shape),
   }
   values.update({axis: mesh[direction] for direction, axis in enumerate(axes)})
   if coordinate == "cylindrical":
      return (
         values["r"] * np.cos(values["phi"]),
         values["r"] * np.sin(values["phi"]),
         values["z"],
      )
   if coordinate == "spherical":
      return (
         values["r"] * np.sin(values["theta"]) * np.cos(values["phi"]),
         values["r"] * np.sin(values["theta"]) * np.sin(values["phi"]),
         values["r"] * np.cos(values["theta"]),
      )
   return values["x"], values["y"], values["z"]


def mesh_velocity_to_cartesian(
   params: dict[str, str],
   axes: tuple[str, ...],
   mesh: list[np.ndarray],
   velocity: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
   """Convert runtime-direction velocity components to physical Cartesian."""
   coordinate = params.get("mesh.coordinate", "cartesian").strip().lower()
   shape = mesh[0].shape
   values = {
      "x": np.zeros(shape),
      "y": np.zeros(shape),
      "z": np.zeros(shape),
      "r": np.zeros(shape),
      "theta": np.zeros(shape),
      "phi": np.zeros(shape),
   }
   values.update({axis: velocity[direction] for direction, axis in enumerate(axes)})
   coordinates = {
      "r": np.ones(shape),
      "theta": np.full(shape, 0.5 * math.pi),
      "phi": np.zeros(shape),
   }
   coordinates.update({axis: mesh[direction] for direction, axis in enumerate(axes)})
   if coordinate == "cylindrical":
      cosine = np.cos(coordinates["phi"])
      sine = np.sin(coordinates["phi"])
      return (
         values["r"] * cosine - values["phi"] * sine,
         values["r"] * sine + values["phi"] * cosine,
         values["z"],
      )
   if coordinate == "spherical":
      sin_theta = np.sin(coordinates["theta"])
      cos_theta = np.cos(coordinates["theta"])
      sin_phi = np.sin(coordinates["phi"])
      cos_phi = np.cos(coordinates["phi"])
      return (
         values["r"] * sin_theta * cos_phi
         + values["theta"] * cos_theta * cos_phi
         - values["phi"] * sin_phi,
         values["r"] * sin_theta * sin_phi
         + values["theta"] * cos_theta * sin_phi
         + values["phi"] * cos_phi,
         values["r"] * cos_theta - values["theta"] * sin_theta,
      )
   return values["x"], values["y"], values["z"]


def pressure_bump_quiet_mask(
   params: dict[str, str],
   model,
   mesh: list[np.ndarray],
   time_value: float,
   safety: float,
) -> np.ndarray:
   axes = mesh_axes(params, len(mesh))
   if params.get("mesh.coordinate", "cartesian").strip().lower() == "cartesian":
      problo = tuple(float(value) for value in params.get("geometry.prob_lo", "").split())
      probhi = tuple(float(value) for value in params.get("geometry.prob_hi", "").split())
      periodic = tuple(int(value) for value in params.get("geometry.is_periodic", "").split())
      radius2 = np.zeros_like(mesh[0])
      for direction in range(len(mesh)):
         delta = np.abs(mesh[direction] - model.center[direction])
         if direction < len(periodic) and periodic[direction] != 0:
            extent = probhi[direction] - problo[direction]
            delta = np.minimum(delta, extent - delta)
         radius2 += delta**2
      return radius2 > model.radius(time_value, safety=safety) ** 2
   x, y, z = physical_coordinates(params, axes, mesh)
   center_mesh = [np.full_like(mesh[0], model.center[direction]) for direction in range(len(mesh))]
   cx, cy, cz = physical_coordinates(params, axes, center_mesh)
   radius2 = (x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2
   return radius2 > model.radius(time_value, safety=safety) ** 2


def plotfile_quiet_row(
   plotfile: Path,
   params: dict[str, str],
   *,
   safety: float,
   source_widths: float,
   tolerance: float,
) -> dict[str, object]:
   names, ndim, time_value = parse_plot_header(plotfile)
   if "rho" not in names or "energy" not in names:
      raise SystemExit(f"{plotfile} is missing rho/energy components")

   gamma = param_float(params, "pyfluid.gamma", 5.0 / 3.0)
   rho0 = param_float(params, "pyfluid.rho0", 1.0)
   pressure0 = param_float(params, "pyfluid.pressure0", 1.0)
   configured_velocity = np.asarray([
      param_float(params, "pyfluid.v1_0", 0.0),
      param_float(params, "pyfluid.v2_0", 0.0),
      param_float(params, "pyfluid.v3_0", 0.0),
   ])
   problo, _probhi = parse_plot_bounds(plotfile)
   level_dx = parse_level_cell_sizes(plotfile)
   nlevels = level_count(plotfile)
   source_kind = params.get("pyfluid.source", "none").strip().lower()
   causal_model = infer_localized_causal_model(params)
   if source_kind in ("", "none", "off"):
      inner, outer = -math.inf, math.inf
   else:
      inner, outer = source_radial_bounds(params, time_value, safety, source_widths)

   quiet_cells = 0
   max_rho = 0.0
   max_pressure = 0.0
   max_velocity = 0.0
   max_magnetic = 0.0
   for lev in range(nlevels):
      dx = level_dx[lev]
      for lo, hi, data in level_chunks(plotfile, lev, ndim):
         rho = component_values(data, names, "rho", gamma)
         pressure = component_values(data, names, "pressure", gamma)
         v1 = component_values(data, names, "v1", gamma)
         v2 = component_values(data, names, "v2", gamma)
         v3 = component_values(data, names, "v3", gamma)
         shape = rho.shape
         index_mesh = np.meshgrid(
            *(np.arange(lo[direction], hi[direction] + 1, dtype=float) for direction in range(ndim)),
            indexing="ij",
         )
         mesh = [problo[direction] + (index_mesh[direction] + 0.5) * dx[direction] for direction in range(ndim)]
         axes = mesh_axes(params, ndim)
         if causal_model is not None:
            quiet = pressure_bump_quiet_mask(params, causal_model, mesh, time_value, safety)
         elif math.isinf(inner):
            quiet = np.ones(shape, dtype=bool)
         else:
            radius = mesh[0]
            quiet = (radius < inner) | (radius > outer)
         if not np.any(quiet):
            continue
         quiet_cells += int(np.count_nonzero(quiet))
         max_rho = max(max_rho, float(np.nanmax(np.abs(rho[quiet] - rho0))))
         max_pressure = max(
            max_pressure,
            float(np.nanmax(np.abs(pressure[quiet] - pressure0))),
         )
         vx, vy, vz = mesh_velocity_to_cartesian(params, axes, mesh, (v1, v2, v3))
         ic = params.get("pyfluid.ic", "").strip().lower()
         if ic in ("density_cloud", "acoustic_packet_x"):
            vx0, vy0, vz0 = (
               np.full(shape, configured_velocity[0]),
               np.full(shape, configured_velocity[1]),
               np.full(shape, configured_velocity[2]),
            )
         else:
            reference = tuple(np.full(shape, value) for value in configured_velocity)
            vx0, vy0, vz0 = mesh_velocity_to_cartesian(params, axes, mesh, reference)
         velocity_delta = np.sqrt((vx - vx0) ** 2 + (vy - vy0) ** 2 + (vz - vz0) ** 2)
         max_velocity = max(max_velocity, float(np.nanmax(velocity_delta[quiet])))
         if all(component in names for component in ("b1", "b2", "b3")):
            magnetic_delta2 = np.zeros(shape)
            for component in (1, 2, 3):
               magnetic = component_values(data, names, f"b{component}", gamma)
               magnetic0 = param_float(params, f"pyfluid.b{component}_0", 0.0)
               magnetic_delta2 += (magnetic - magnetic0) ** 2
            max_magnetic = max(max_magnetic, float(np.nanmax(np.sqrt(magnetic_delta2)[quiet])))

   metric = max(
      max_rho / max(abs(rho0), 1.0),
      max_pressure / max(abs(pressure0), 1.0),
      max_velocity,
      max_magnetic,
   )
   return {
      "step": plotfile_step(plotfile),
      "time": time_value,
      "quiet_cells": quiet_cells,
      "inner_quiet_radius": inner,
      "outer_quiet_radius": outer,
      "max_rho_quiet": max_rho,
      "max_pressure_quiet": max_pressure,
      "max_velocity_quiet": max_velocity,
      "max_magnetic_quiet": max_magnetic,
      "status": "N/A" if quiet_cells == 0 else ("PASS" if metric <= tolerance else "FAIL"),
   }


def physics_validation_row(
   run_dir: Path,
   params: dict[str, str],
   *,
   tolerance: float = 1.0e-7,
   safety: float = 1.20,
) -> dict[str, object] | None:
   if infer_localized_causal_model(params) is None:
      return None
   plotfiles = sorted_plotfiles(run_dir)
   if not plotfiles:
      return None
   rows = [
      plotfile_quiet_row(
         plotfile,
         params,
         safety=safety,
         source_widths=4.0,
         tolerance=tolerance,
      )
      for plotfile in plotfiles
   ]
   write_tsv(run_dir / "causal_quiescence.tsv", rows, FIELDS)
   applicable = [row for row in rows if row["status"] != "N/A"]
   if not applicable:
      return None
   value = max(
      max(
         float(row["max_rho_quiet"]) / max(abs(param_float(params, "pyfluid.rho0", 1.0)), 1.0),
         float(row["max_pressure_quiet"]) / max(abs(param_float(params, "pyfluid.pressure0", 1.0)), 1.0),
         float(row["max_velocity_quiet"]),
         float(row["max_magnetic_quiet"]),
      )
      for row in applicable
   )
   return {
      "hook": "causal_quiescence",
      "status": "PASS" if value <= tolerance else "FAIL",
      "reason": "" if value <= tolerance else f"causal exterior change={value:.3e} > {tolerance:.3e}",
      "metric": "max_causal_exterior_change",
      "value": value,
      "tolerance": tolerance,
      "run_dir": str(run_dir),
   }


def write_plot(path: Path, rows: list[dict[str, object]], title: str) -> None:
   t = np.asarray([float(row["time"]) for row in rows])
   fig, ax = plt.subplots(figsize=(7.5, 4.8), constrained_layout=True)
   for key, label in [
      ("max_rho_quiet", "rho"),
      ("max_pressure_quiet", "pressure"),
      ("max_velocity_quiet", "velocity"),
      ("max_magnetic_quiet", "magnetic field"),
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
   parser.add_argument(
      "--tolerance",
      type=float,
      default=1.0e-7,
      help="Allowed exterior change; accounts for finite-stencil tails outside the continuum signal cone.",
   )
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
   failures = [row for row in rows if row["status"] == "FAIL"]
   status = "PASS" if not failures else "FAIL"
   worst = max(
      max(
         float(row["max_rho_quiet"]),
         float(row["max_pressure_quiet"]),
         float(row["max_velocity_quiet"]),
         float(row["max_magnetic_quiet"]),
      )
      for row in rows
   )
   print(f"{status} causal_quiescence run={repo_relative(run_dir)} worst={worst:.3e} rows={len(rows)}")
   if failures:
      raise SystemExit(1)


if __name__ == "__main__":
   main()
