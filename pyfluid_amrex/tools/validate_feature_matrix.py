#!/usr/bin/env python3
"""Run a compact AMReX feature matrix.

This is a smoke/coverage check for the supported runtime switches. It is not a
convergence suite. Each case writes its concrete input file and scalar output
under runs/current so failures are reproducible without guessing CLI overrides.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from amr_plot_data import compare_hierarchy_slices
from plotfile_reader import latest_plotfile, level_count, parse_plot_header, read_level_state

from validation_common import (
   BAROTROPIC_DIFFUSIVE_POLICY,
   BAROTROPIC_POLICY,
   BAROTROPIC_WALL_POLICY,
   CLOSED_POLICY,
   DIFFUSIVE_POLICY,
   DIFFUSIVE_WALL_POLICY,
   DRIVEN_POLICY,
   OPEN_POLICY,
   OPEN_MHD_POLICY,
   SMOKE_POLICY,
   ScalarPolicy,
   WALL_POLICY,
   amrex_executable,
   append_validation_history as append_common_validation_history,
   materialize_run_input,
   scalar_checks,
   scalar_validation_row,
   run_amrex,
   select_named_cases,
   write_tsv,
)


ROOT = Path(__file__).resolve().parents[2]
INPUTS = ROOT / "pyfluid_amrex" / "inputs"
RUN_ROOT = ROOT / "runs" / "current" / "amrex_feature_matrix"
CASE_INPUTS = RUN_ROOT / "inputs"
SUMMARY = RUN_ROOT / "feature_matrix.tsv"
HISTORY = ROOT / "runs" / "current" / "validation_history.tsv"
RECONSTRUCTION_METHODS = (
   ("pwc", "1"),
   ("pwl", "2"),
   ("wenoz", "3"),
   ("wenoz-char", "3"),
   ("mp5", "3"),
   ("mp5-char", "3"),
   ("teno5", "3"),
   ("teno5-char", "3"),
)


@dataclass(frozen=True)
class Case:
   name: str
   dim: int
   base_input: str
   overrides: dict[str, str] = field(default_factory=dict)
   ranks: int = 3
   checks: ScalarPolicy = CLOSED_POLICY
   require_amr: bool = False
   min_amr_level: int = 1
   min_amr_headroom: int = 2
   min_regrid_noops: int = 0
   expected_uniform_state: dict[str, float] = field(default_factory=dict)
   uniform_tolerance: float = 0.0
   comparison_reference: str | None = None
   comparison_components: tuple[str, ...] = ()
   comparison_rel_l2_tolerance: float = 0.0


COMMON_FAST = {
   "pyfluid.max_step": "2",
   "pyfluid.final_time": "-1.0",
   "pyfluid.scalar_int": "1",
   "pyfluid.plot_int": "-1",
   "pyfluid.checkpoint_int": "-1",
   "pyfluid.aux_gfs": "none",
   "pyfluid.aux_gf_int": "-1",
   "pyfluid.aux_gf_compute_int": "-1",
}

AUTO_AMR_COVERAGE = {
   "amr.n_cell": "24 24",
   "amr.max_level": "8",
   "amr.ref_ratio": "2 2",
   "amr.max_grid_size": "24",
   "amr.blocking_factor": "8",
   "amr.regrid_int": "1",
   "amr.tag_metric": "lohner",
   "amr.min_feature_length": "0.001",
   "amr.total_cell_budget": "1000000",
   "amr.initial_refine": "none",
   "amr.initial_refine_levels": "0",
   "amr.initial_refine_fixed_levels": "0",
}

DEEP_AMR_COVERAGE = {
   "amr.n_cell": "16 16",
   "amr.max_level": "8",
   "amr.ref_ratio": "2 2 2 2 2 2 2 2",
   "amr.max_grid_size": "32",
   "amr.blocking_factor": "8",
   "amr.regrid_mode": "synchronized",
   "amr.regrid_int": "1",
   "amr.state_interp": "conservative",
   "amr.tag_metric": "lohner",
   "amr.min_feature_length": "0.0005",
   "amr.total_cell_budget": "1000000",
   "amr.initial_refine": "box",
   "amr.initial_refine_levels": "3",
   "amr.initial_refine_fixed_levels": "0",
   "amr.initial_refine_box_lo": "0.30 0.30",
   "amr.initial_refine_box_hi": "0.70 0.70",
}

SELF_GRAVITY_POLICY = scalar_checks("self_gravity", energy_tol=1.0e-10)
SELF_GRAVITY_AMR_POLICY = scalar_checks("self_gravity_amr", energy_tol=2.0e-6)


def case_output_dir(name: str) -> Path:
   return RUN_ROOT / name


def write_case_input(case: Case, force: bool) -> Path:
   output_dir = case_output_dir(case.name)
   if force and output_dir.exists():
      shutil.rmtree(output_dir)
   output_dir.mkdir(parents=True, exist_ok=True)
   CASE_INPUTS.mkdir(parents=True, exist_ok=True)

   values = dict(COMMON_FAST)
   values["pyfluid.run_name"] = case.name
   values["pyfluid.output_dir"] = str(output_dir.relative_to(ROOT))
   values.update(case.overrides)

   path = CASE_INPUTS / f"{case.name}.in"
   return materialize_run_input(INPUTS / case.base_input, path, values)


def case_checks(case: Case) -> ScalarPolicy:
   name = case.name
   is_barotropic = case.overrides.get("pyfluid.eos") == "polytrope"
   if any(marker in name for marker in ("_bc_outflow", "_bc_diode", "inflow_mixed")):
      if case_is_mhd(case):
         return OPEN_MHD_POLICY
      return OPEN_POLICY
   if any(marker in name for marker in ("obstacle_forward", "obstacle_wedge", "obstacle_two_boxes")):
      return OPEN_POLICY
   if "reflecting" in name:
      if any(marker in name for marker in ("diffusion", "resistivity", "resistive")):
         if is_barotropic:
            return BAROTROPIC_WALL_POLICY
         return DIFFUSIVE_WALL_POLICY
      if is_barotropic:
         return BAROTROPIC_WALL_POLICY
      return WALL_POLICY
   if any(marker in name for marker in ("diffusion", "resistivity", "resistive")):
      if is_barotropic:
         return BAROTROPIC_DIFFUSIVE_POLICY
      return DIFFUSIVE_POLICY
   if any(marker in name for marker in ("moving_box", "rotating_cylinder", "pulsating_circle")):
      return DRIVEN_POLICY
   if name.endswith("_smoke"):
      return SMOKE_POLICY
   if is_barotropic:
      return BAROTROPIC_POLICY
   return case.checks


def case_is_mhd(case: Case) -> bool:
   return (
      case.name.startswith("mhd")
      or case.overrides.get("pyfluid.physics") == "mhd"
      or "mhd" in case.base_input
      or "resistive" in case.base_input
   )


def run_case(case: Case, force: bool, timeout: int, ranks_override: int | None = None) -> dict[str, str]:
   exe = amrex_executable(case.dim)
   checks = case_checks(case)
   ranks = case.ranks if ranks_override is None else ranks_override
   if not exe.exists():
      return {
         "case": case.name,
         "status": "FAIL",
         "reason": f"missing executable {exe.relative_to(ROOT)}",
         "dim": str(case.dim),
         "input": case.base_input,
         "policy": checks.name,
         "ranks": str(ranks),
      }

   input_path = write_case_input(case, force)
   proc = run_amrex(dim=case.dim, input_path=input_path, ranks=ranks, timeout=timeout)
   output_dir = case_output_dir(case.name)
   output_dir.mkdir(parents=True, exist_ok=True)
   (output_dir / "run.log").write_text(proc.stdout)
   if proc.returncode != 0:
      tail = "\n".join(proc.stdout.splitlines()[-8:])
      return {
         "case": case.name,
         "status": "FAIL",
         "reason": tail.replace("\t", " "),
         "dim": str(case.dim),
         "input": case.base_input,
         "policy": checks.name,
         "ranks": str(ranks),
      }

   row = scalar_validation_row(
      output_dir,
      case=case.name,
      input_file=input_path,
      checks=checks,
      mhd=case_is_mhd(case),
      dim=case.dim,
      ranks=ranks,
      base_input=case.base_input,
      kind="feature_matrix",
   )
   if case.require_amr:
      active_level = int(float(row.get("amr_active_finest_level", 0)))
      headroom = int(float(row.get("amr_finest_headroom", 0)))
      failures = []
      if active_level < case.min_amr_level:
         failures.append(
            f"automatic AMR reached L{active_level}, expected at least L{case.min_amr_level}"
         )
      if headroom < case.min_amr_headroom:
         failures.append(
            f"automatic AMR retained {headroom} level(s) of headroom, expected at least {case.min_amr_headroom}"
         )
      if failures:
         row["status"] = "FAIL"
         row["reason"] = "; ".join(filter(None, (str(row.get("reason", "")), *failures)))
   if case.min_regrid_noops > 0:
      match = re.search(
         r"AMR regrid attempts=(\d+) noops=(\d+) modified_levels=(\d+)",
         proc.stdout,
      )
      noops = int(match.group(2)) if match else 0
      row["regrid_attempts"] = match.group(1) if match else "0"
      row["regrid_noops"] = str(noops)
      row["regrid_modified_levels"] = match.group(3) if match else "0"
      if noops < case.min_regrid_noops:
         row["status"] = "FAIL"
         failure = f"AMR recorded {noops} no-op regrids, expected at least {case.min_regrid_noops}"
         row["reason"] = "; ".join(filter(None, (str(row.get("reason", "")), failure)))
   if case.expected_uniform_state:
      plotfile = latest_plotfile(output_dir / "stateGF", require=True)
      names, ndim, _time = parse_plot_header(plotfile)
      name_to_component = {name: component for component, name in enumerate(names)}
      max_error = 0.0
      for lev in range(level_count(plotfile)):
         state = read_level_state(plotfile, names, ndim, lev)
         for name, expected in case.expected_uniform_state.items():
            component = name_to_component[name]
            max_error = max(
               max_error,
               float(np.nanmax(np.abs(state[component] - expected))),
            )
      row["uniform_state_max_error"] = f"{max_error:.17g}"
      if max_error > case.uniform_tolerance:
         row["status"] = "FAIL"
         failure = (
            f"uniform-state error {max_error:.3e} exceeds "
            f"{case.uniform_tolerance:.3e} across the AMR hierarchy"
         )
         row["reason"] = "; ".join(filter(None, (str(row.get("reason", "")), failure)))
   return row


def hydro_method_cases() -> list[Case]:
   cases: list[Case] = []
   for timestep in ["euler", "corrector", "midpoint", "ssprk3", "rk4"]:
      cases.append(Case(
         f"hydro_timestep_{timestep}",
         2,
         "hydro_sine_2d.in",
         {"pyfluid.timestep": timestep, "pyfluid.recon": "pwl", "pyfluid.riemann": "hllc", "pyfluid.ng": "2"},
      ))
   for recon, ng in RECONSTRUCTION_METHODS:
      cases.append(Case(
         f"hydro_recon_{recon.replace('-', '_')}",
         2,
         "hydro_sine_2d.in",
         {"pyfluid.timestep": "ssprk3", "pyfluid.recon": recon, "pyfluid.riemann": "hllc", "pyfluid.ng": ng},
      ))
   for riemann in ["llf", "hll", "hlle", "hllc"]:
      cases.append(Case(
         f"hydro_riemann_{riemann}",
         2,
         "hydro_sine_2d.in",
         {"pyfluid.timestep": "ssprk3", "pyfluid.recon": "pwl", "pyfluid.riemann": riemann, "pyfluid.ng": "2"},
      ))
   cases.append(Case(
      "hydro_passive_dye_advection",
      2,
      "hydro_sine_2d.in",
      {
         "pyfluid.timestep": "ssprk3",
         "pyfluid.recon": "pwl",
         "pyfluid.riemann": "hllc",
         "pyfluid.ng": "2",
         "pyfluid.passive_scalars": "1",
         "pyfluid.passive_profile": "gaussian",
         "pyfluid.passive_background": "0.0",
         "pyfluid.passive_amplitude": "1.0",
         "pyfluid.passive_sigma": "0.08",
         "pyfluid.passive_x0": "0.35",
         "pyfluid.passive_y0": "0.5",
      },
   ))
   return cases


def mhd_method_cases() -> list[Case]:
   cases: list[Case] = []
   for timestep in ["euler", "corrector", "midpoint", "ssprk3", "rk4"]:
      cases.append(Case(
         f"mhd_timestep_{timestep}",
         2,
         "resistive_wave_x_2d_ct.in",
         {
            "pyfluid.timestep": timestep,
            "pyfluid.recon": "pwl",
            "pyfluid.riemann": "hlld",
            "pyfluid.ng": "2",
            "pyfluid.diffusion": "none",
            "pyfluid.resistivity": "0.0",
         },
      ))
   for recon, ng in RECONSTRUCTION_METHODS:
      cases.append(Case(
         f"mhd_recon_{recon.replace('-', '_')}",
         2,
         "resistive_wave_x_2d_ct.in",
         {
            "pyfluid.timestep": "ssprk3",
            "pyfluid.recon": recon,
            "pyfluid.riemann": "hlld",
            "pyfluid.ng": ng,
            "pyfluid.diffusion": "none",
            "pyfluid.resistivity": "0.0",
         },
      ))
   for riemann in ["llf", "hll", "hlle", "hlld"]:
      cases.append(Case(
         f"mhd_riemann_{riemann}",
         2,
         "resistive_wave_x_2d_ct.in",
         {
            "pyfluid.timestep": "ssprk3",
            "pyfluid.recon": "pwl",
            "pyfluid.riemann": riemann,
            "pyfluid.ng": "2",
            "pyfluid.diffusion": "none",
            "pyfluid.resistivity": "0.0",
         },
      ))
   cases.append(Case(
      "mhd_passive_dye_ct",
      2,
      "circular_alfven_x_2d_ct.in",
      {
         "pyfluid.max_step": "2",
         "pyfluid.final_time": "-1.0",
         "pyfluid.timestep": "ssprk3",
         "pyfluid.recon": "pwl",
         "pyfluid.riemann": "hlld",
         "pyfluid.ng": "2",
         "pyfluid.diffusion": "none",
         "pyfluid.viscosity": "0.0",
         "pyfluid.resistivity": "0.0",
         "pyfluid.passive_scalars": "1",
         "pyfluid.passive_profile": "sin_wave",
         "pyfluid.passive_background": "0.5",
         "pyfluid.passive_amplitude": "0.25",
      },
   ))
   cases.append(Case(
      "mhd_orszag_tang_wenoz_char_hlld_ct",
      2,
      "orszag_tang_2d_capstone_ct.in",
      {
         "amr.n_cell": "32 32",
         "pyfluid.diffusion": "none",
         "pyfluid.viscosity": "0.0",
         "pyfluid.resistivity": "0.0",
      },
   ))
   return cases


def eos_cases() -> list[Case]:
   return [
      Case("hydro_eos_polytrope", 2, "hydro_sine_2d.in", {
         "pyfluid.eos": "polytrope",
         "pyfluid.polytropic_k": "1.0",
         "pyfluid.polytropic_gamma": "1.6666666666666667",
         "pyfluid.polytropic_density_breaks": "1.0",
         "pyfluid.polytropic_segment_gammas": "1.4",
         "pyfluid.amplitude": "0.2",
         "pyfluid.timestep": "ssprk3",
         "pyfluid.recon": "pwl",
         "pyfluid.riemann": "hllc",
         "pyfluid.ng": "2",
      }),
      Case("hydro_eos_hybrid_polytrope", 2, "hydro_sine_2d.in", {
         "pyfluid.eos": "hybrid_polytrope",
         "pyfluid.polytropic_k": "0.2",
         "pyfluid.polytropic_gamma": "1.6666666666666667",
         "pyfluid.polytropic_density_breaks": "1.0",
         "pyfluid.polytropic_segment_gammas": "1.4",
         "pyfluid.thermal_gamma": "1.6666666666666667",
         "pyfluid.amplitude": "0.2",
         "pyfluid.timestep": "ssprk3",
         "pyfluid.recon": "pwl",
         "pyfluid.riemann": "hllc",
         "pyfluid.ng": "2",
      }),
      Case("mhd_eos_polytrope_wenoz_char_hlld_ct", 2, "hydro_sine_2d.in", {
         "pyfluid.physics": "mhd",
         "pyfluid.ct": "on",
         "pyfluid.eos": "polytrope",
         "pyfluid.polytropic_k": "1.0",
         "pyfluid.polytropic_gamma": "1.6666666666666667",
         "pyfluid.polytropic_density_breaks": "1.0",
         "pyfluid.polytropic_segment_gammas": "1.4",
         "pyfluid.amplitude": "0.2",
         "pyfluid.b1_0": "0.2",
         "pyfluid.b2_0": "0.1",
         "pyfluid.diffusion": "none",
         "pyfluid.resistivity": "0.0",
         "pyfluid.timestep": "rk4",
         "pyfluid.recon": "wenoz-char",
         "pyfluid.riemann": "hlld",
         "pyfluid.ng": "3",
      }),
      Case("mhd_eos_hybrid_polytrope_resistive_ct", 2, "resistive_wave_x_2d_ct.in", {
         "pyfluid.eos": "hybrid_polytrope",
         "pyfluid.polytropic_k": "0.2",
         "pyfluid.polytropic_gamma": "1.6666666666666667",
         "pyfluid.polytropic_density_breaks": "0.9",
         "pyfluid.polytropic_segment_gammas": "1.4",
         "pyfluid.thermal_gamma": "1.6666666666666667",
         "pyfluid.diffusion": "explicit",
         "pyfluid.resistivity": "1.0e-3",
         "pyfluid.viscosity": "0.0",
         "pyfluid.timestep": "ssprk3",
         "pyfluid.recon": "pwl",
         "pyfluid.riemann": "hlld",
         "pyfluid.ng": "2",
      }),
      Case("mhd_eos_polytrope_implicit_resistive_ct", 2, "resistive_wave_x_2d_ct.in", {
         "pyfluid.eos": "polytrope",
         "pyfluid.polytropic_k": "1.0",
         "pyfluid.polytropic_gamma": "1.6666666666666667",
         "pyfluid.polytropic_density_breaks": "0.9",
         "pyfluid.polytropic_segment_gammas": "1.4",
         "pyfluid.diffusion": "implicit",
         "pyfluid.resistivity": "1.0e-3",
         "pyfluid.viscosity": "0.0",
         "pyfluid.timestep": "ssprk3",
         "pyfluid.recon": "wenoz-char",
         "pyfluid.riemann": "hlld",
         "pyfluid.ng": "3",
         "pyfluid.diffusion_tolerance": "1.0e-10",
         "pyfluid.diffusion_max_iterations": "300",
      }),
      Case("mhd_eos_hybrid_polytrope_implicit_resistive_ct", 2, "resistive_wave_x_2d_ct.in", {
         "pyfluid.eos": "hybrid_polytrope",
         "pyfluid.polytropic_k": "0.2",
         "pyfluid.polytropic_gamma": "1.6666666666666667",
         "pyfluid.polytropic_density_breaks": "0.9",
         "pyfluid.polytropic_segment_gammas": "1.4",
         "pyfluid.thermal_gamma": "1.6666666666666667",
         "pyfluid.diffusion": "implicit",
         "pyfluid.resistivity": "1.0e-3",
         "pyfluid.viscosity": "0.0",
         "pyfluid.timestep": "ssprk3",
         "pyfluid.recon": "wenoz-char",
         "pyfluid.riemann": "hlld",
         "pyfluid.ng": "3",
         "pyfluid.diffusion_tolerance": "1.0e-10",
         "pyfluid.diffusion_max_iterations": "300",
      }),
   ]


def bc_cases() -> list[Case]:
   base = {
      "geometry.is_periodic": "0 0",
      "pyfluid.ic": "pressure_bump",
      "pyfluid.bc": "outflow",
      "pyfluid.recon": "pwl",
      "pyfluid.riemann": "hllc",
      "pyfluid.timestep": "ssprk3",
      "pyfluid.ng": "2",
      "amr.max_level": "0",
   }
   cases = [
      Case("hydro_bc_periodic", 2, "hydro_sine_2d.in", {
         "geometry.is_periodic": "1 1",
         "pyfluid.bc": "periodic",
         "pyfluid.recon": "pwl",
         "pyfluid.riemann": "hllc",
         "pyfluid.timestep": "ssprk3",
         "pyfluid.ng": "2",
         "amr.max_level": "0",
      }),
   ]
   for bc in ["outflow", "diode", "reflecting", "symmetry"]:
      values = dict(base)
      values["pyfluid.bc"] = bc
      cases.append(Case(f"hydro_bc_{bc}", 2, "hydro_sine_2d.in", values))
   inflow = dict(base)
   inflow.update({
      "pyfluid.bc_xlo": "inflow",
      "pyfluid.bc_xhi": "outflow",
      "pyfluid.bc_ylo": "reflecting",
      "pyfluid.bc_yhi": "reflecting",
      "pyfluid.inflow_v1": "0.1",
   })
   cases.append(Case("hydro_bc_inflow_mixed", 2, "hydro_sine_2d.in", inflow))
   mhd_base = {
      "geometry.is_periodic": "0 0",
      "amr.max_level": "0",
      "pyfluid.diffusion": "none",
      "pyfluid.resistivity": "0.0",
      "pyfluid.recon": "pwl",
      "pyfluid.riemann": "hlld",
      "pyfluid.timestep": "ssprk3",
      "pyfluid.ng": "2",
   }
   cases.append(Case("mhd_ct_bc_periodic", 2, "resistive_wave_x_2d_ct.in", {
      "geometry.is_periodic": "1 1",
      "amr.max_level": "0",
      "pyfluid.bc": "periodic",
      "pyfluid.diffusion": "none",
      "pyfluid.resistivity": "0.0",
      "pyfluid.recon": "pwl",
      "pyfluid.riemann": "hlld",
      "pyfluid.timestep": "ssprk3",
      "pyfluid.ng": "2",
   }))
   for bc in ["outflow", "diode", "reflecting", "symmetry"]:
      values = dict(mhd_base)
      values["pyfluid.bc"] = bc
      cases.append(Case(f"mhd_ct_bc_{bc}", 2, "resistive_wave_x_2d_ct.in", values))
   mhd_inflow = dict(mhd_base)
   mhd_inflow.update({
      "pyfluid.bc": "outflow",
      "pyfluid.bc_xlo": "inflow",
      "pyfluid.bc_xhi": "outflow",
      "pyfluid.bc_ylo": "reflecting",
      "pyfluid.bc_yhi": "reflecting",
      "pyfluid.inflow_v1": "0.1",
      "pyfluid.inflow_b3": "1.0e-3",
   })
   cases.append(Case("mhd_ct_bc_inflow_mixed", 2, "resistive_wave_x_2d_ct.in", mhd_inflow))
   return cases


def diffusion_cases() -> list[Case]:
   return [
      Case("hydro_diffusion_explicit", 2, "hydro_sine_2d.in", {
         "pyfluid.diffusion": "explicit",
         "pyfluid.viscosity": "1.0e-4",
         "pyfluid.recon": "pwl",
         "pyfluid.riemann": "hllc",
         "pyfluid.timestep": "ssprk3",
         "pyfluid.ng": "2",
      }),
      Case("hydro_diffusion_implicit", 2, "hydro_sine_2d.in", {
         "pyfluid.diffusion": "implicit",
         "pyfluid.viscosity": "1.0e-4",
         "pyfluid.recon": "pwl",
         "pyfluid.riemann": "hllc",
         "pyfluid.timestep": "ssprk3",
         "pyfluid.ng": "2",
         "pyfluid.diffusion_max_iterations": "500",
      }),
      Case("mhd_resistivity_explicit_ct", 2, "resistive_wave_x_2d_ct.in", {
         "pyfluid.diffusion": "explicit",
         "pyfluid.resistivity": "1.0e-3",
         "pyfluid.viscosity": "0.0",
         "pyfluid.recon": "pwl",
         "pyfluid.riemann": "hlld",
         "pyfluid.timestep": "ssprk3",
         "pyfluid.ng": "2",
      }),
      Case("mhd_resistivity_implicit_ct", 2, "resistive_wave_x_2d_ct.in", {
         "pyfluid.diffusion": "implicit",
         "pyfluid.resistivity": "1.0e-3",
         "pyfluid.viscosity": "0.0",
         "pyfluid.recon": "pwl",
         "pyfluid.riemann": "hlld",
         "pyfluid.timestep": "ssprk3",
         "pyfluid.ng": "2",
         "pyfluid.diffusion_tolerance": "1.0e-10",
         "pyfluid.diffusion_max_iterations": "300",
      }),
      Case("mhd_curvilinear_ct_explicit_cylindrical_rz", 2, "resistive_wave_x_2d_ct.in", {
         "mesh.coordinate": "cylindrical",
         "mesh.axes": "r z",
         "geometry.coord_sys": "1",
         "geometry.prob_lo": "1.0 0.0",
         "geometry.prob_hi": "2.0 1.0",
         "geometry.is_periodic": "0 0",
         "amr.n_cell": "24 24",
         "amr.max_grid_size": "24",
         "pyfluid.bc": "outflow",
         "pyfluid.diffusion": "explicit",
         "pyfluid.resistivity": "1.0e-3",
         "pyfluid.viscosity": "0.0",
         "pyfluid.recon": "pwl",
         "pyfluid.riemann": "hlld",
         "pyfluid.timestep": "ssprk3",
         "pyfluid.ng": "2",
      }, checks=OPEN_POLICY),
      Case("mhd_curvilinear_ct_explicit_cylindrical_rphi", 2, "cylindrical_uniform_static_2d.in", {
         "mesh.axes": "r phi",
         "geometry.coord_sys": "0",
         "geometry.prob_lo": "1.0 0.0",
         "geometry.prob_hi": "2.0 6.283185307179586",
         "geometry.is_periodic": "0 1",
         "amr.n_cell": "24 24",
         "amr.max_grid_size": "24",
         "pyfluid.physics": "mhd",
         "pyfluid.ct": "on",
         "pyfluid.diffusion": "explicit",
         "pyfluid.resistivity": "1.0e-3",
         "pyfluid.viscosity": "0.0",
         "pyfluid.b2_0": "0.1",
         "pyfluid.bc": "outflow",
         "pyfluid.bc_ylo": "periodic",
         "pyfluid.bc_yhi": "periodic",
         "pyfluid.recon": "pwl",
         "pyfluid.riemann": "hlld",
         "pyfluid.timestep": "ssprk3",
         "pyfluid.ng": "2",
      }, checks=OPEN_POLICY),
      Case("mhd_curvilinear_ct_explicit_cylindrical_phiz", 2, "cylindrical_uniform_static_2d.in", {
         "mesh.axes": "phi z",
         "geometry.coord_sys": "0",
         "geometry.prob_lo": "0.0 0.0",
         "geometry.prob_hi": "6.283185307179586 1.0",
         "geometry.is_periodic": "1 0",
         "amr.n_cell": "24 24",
         "amr.max_grid_size": "24",
         "pyfluid.physics": "mhd",
         "pyfluid.ct": "on",
         "pyfluid.diffusion": "explicit",
         "pyfluid.resistivity": "1.0e-3",
         "pyfluid.viscosity": "0.0",
         "pyfluid.b2_0": "0.1",
         "pyfluid.bc": "outflow",
         "pyfluid.bc_xlo": "periodic",
         "pyfluid.bc_xhi": "periodic",
         "pyfluid.bc_ylo": "outflow",
         "pyfluid.bc_yhi": "outflow",
         "pyfluid.recon": "pwl",
         "pyfluid.riemann": "hlld",
         "pyfluid.timestep": "ssprk3",
         "pyfluid.ng": "2",
      }, checks=OPEN_POLICY),
      Case("mhd_curvilinear_ct_explicit_spherical_rtheta", 2, "spherical_uniform_static_2d.in", {
         "amr.n_cell": "24 24",
         "amr.max_grid_size": "24",
         "pyfluid.physics": "mhd",
         "pyfluid.ct": "on",
         "pyfluid.diffusion": "explicit",
         "pyfluid.resistivity": "1.0e-3",
         "pyfluid.viscosity": "0.0",
         "pyfluid.b3_0": "0.1",
         "pyfluid.recon": "pwl",
         "pyfluid.riemann": "hlld",
         "pyfluid.timestep": "ssprk3",
         "pyfluid.ng": "2",
      }, checks=OPEN_POLICY),
      Case("mhd_curvilinear_ct_explicit_spherical_thetaphi", 2, "spherical_uniform_static_2d.in", {
         "mesh.axes": "theta phi",
         "geometry.coord_sys": "0",
         "geometry.prob_lo": "0.2 0.0",
         "geometry.prob_hi": "2.941592653589793 6.283185307179586",
         "geometry.is_periodic": "0 1",
         "amr.n_cell": "24 24",
         "amr.max_grid_size": "24",
         "pyfluid.physics": "mhd",
         "pyfluid.ct": "on",
         "pyfluid.diffusion": "explicit",
         "pyfluid.resistivity": "1.0e-3",
         "pyfluid.viscosity": "0.0",
         "pyfluid.b2_0": "0.1",
         "pyfluid.bc": "outflow",
         "pyfluid.bc_ylo": "periodic",
         "pyfluid.bc_yhi": "periodic",
         "pyfluid.recon": "pwl",
         "pyfluid.riemann": "hlld",
         "pyfluid.timestep": "ssprk3",
         "pyfluid.ng": "2",
      }, checks=OPEN_POLICY),
      Case("mhd_curvilinear_ct_explicit_spherical_rphi", 2, "spherical_uniform_static_rphi_2d.in", {
         "amr.n_cell": "24 24",
         "amr.max_grid_size": "24",
         "pyfluid.physics": "mhd",
         "pyfluid.ct": "on",
         "pyfluid.diffusion": "explicit",
         "pyfluid.resistivity": "1.0e-3",
         "pyfluid.viscosity": "0.0",
         "pyfluid.b2_0": "0.1",
         "pyfluid.recon": "pwl",
         "pyfluid.riemann": "hlld",
         "pyfluid.timestep": "ssprk3",
         "pyfluid.ng": "2",
      }, checks=OPEN_POLICY),
   ]


def amr_cases() -> list[Case]:
   cases = []
   relativistic_kh_common = {
      "pyfluid.max_step": "64",
      "pyfluid.final_time": "0.02",
      "pyfluid.plot_int": "1",
      "pyfluid.scalar_int": "1",
      "pyfluid.aux_gfs": "none",
      "pyfluid.aux_gf_compute_int": "-1",
      "pyfluid.aux_gf_int": "-1",
   }
   cases.append(Case(
      "rhd_kh_amr_no_subcycling",
      2,
      "rhd_kelvin_helmholtz_2d_amr.in",
      {
         **relativistic_kh_common,
         "pyfluid.subcycling": "off",
      },
      require_amr=True,
      min_amr_level=2,
   ))
   cases.append(Case(
      "rhd_kh_amr_subcycling",
      2,
      "rhd_kelvin_helmholtz_2d_amr.in",
      {
         **relativistic_kh_common,
         "pyfluid.subcycling": "on",
         "pyfluid.subcycling_steps": "local_cfl",
      },
      require_amr=True,
      min_amr_level=2,
      comparison_reference="rhd_kh_amr_no_subcycling",
      comparison_components=("dens", "mom1", "mom2", "mom3", "tau"),
      comparison_rel_l2_tolerance=2.0e-4,
   ))
   cases.append(Case(
      "rmhd_invariant_magnetic_amr",
      1,
      "rmhd_entropy_wave_1d.in",
      {
         "amr.n_cell": "16",
         "amr.max_level": "2",
         "amr.ref_ratio": "2 2",
         "amr.max_grid_size": "32",
         "amr.blocking_factor": "8",
         "amr.regrid_int": "1",
         "amr.tag_metric": "lohner",
         "amr.min_feature_length": "1.0e-5",
         "amr.total_cell_budget": "100000",
         "pyfluid.ic": "sod_periodic_plane",
         "pyfluid.shock_x0": "0.5",
         "pyfluid.shock_nx": "1.0",
         "pyfluid.rho_left": "1.0",
         "pyfluid.rho_right": "1.0",
         "pyfluid.pressure_left": "1.0",
         "pyfluid.pressure_right": "1.0",
         "pyfluid.v1_left": "0.0",
         "pyfluid.v1_right": "0.0",
         "pyfluid.v2_left": "0.0",
         "pyfluid.v2_right": "0.0",
         "pyfluid.v3_left": "0.0",
         "pyfluid.v3_right": "0.0",
         "pyfluid.b1_left": "0.2",
         "pyfluid.b1_right": "0.2",
         "pyfluid.b2_left": "0.05",
         "pyfluid.b2_right": "1.0",
         "pyfluid.b3_left": "0.0",
         "pyfluid.b3_right": "0.0",
         "pyfluid.ct": "off",
         "pyfluid.max_step": "0",
         "pyfluid.final_time": "-1",
         "pyfluid.plot_int": "1",
         "pyfluid.scalar_int": "1",
         "pyfluid.aux_gfs": "none",
         "pyfluid.aux_gf_compute_int": "-1",
         "pyfluid.aux_gf_int": "-1",
      },
      checks=SMOKE_POLICY,
      require_amr=True,
      min_amr_level=2,
      # This isolated sensor check uses a true magnetic discontinuity, so the
      # configured level is deliberately the stopping condition.
      min_amr_headroom=0,
   ))
   for timestep in ["euler", "corrector", "midpoint", "ssprk3", "rk4"]:
      cases.append(Case(
         f"hydro_amr_subcycling_{timestep}",
         2,
         "pressure_bump_2d_amr.in",
         {
            **AUTO_AMR_COVERAGE,
            "pyfluid.max_step": "2",
            "pyfluid.timestep": timestep,
            "pyfluid.recon": "pwl",
            "pyfluid.riemann": "hllc",
            "pyfluid.ng": "2",
            "pyfluid.subcycling": "on",
            "pyfluid.plot_int": "-1",
            "amr.state_interp": "quartic-conservative",
            "amr.regrid_int": "1",
         },
         require_amr=True,
      ))
   cases.append(Case(
      "hydro_amr_subcycling_aligned_regrid",
      2,
      "pressure_bump_2d_amr.in",
      {
         **AUTO_AMR_COVERAGE,
         "amr.regrid_mode": "aligned",
         "amr.regrid_int": "8",
         "pyfluid.max_step": "2",
         "pyfluid.timestep": "ssprk3",
         "pyfluid.recon": "pwl",
         "pyfluid.riemann": "hllc",
         "pyfluid.ng": "2",
         "pyfluid.subcycling": "on",
         "amr.state_interp": "quartic-conservative",
      },
      require_amr=True,
   ))
   cases.append(Case(
      "hydro_amr_aligned_noop_contract",
      2,
      "pressure_bump_2d_amr.in",
      {
         "geometry.is_periodic": "1 1",
         "amr.n_cell": "32 32",
         "amr.max_level": "2",
         "amr.ref_ratio": "2 2",
         "amr.max_grid_size": "32",
         "amr.blocking_factor": "8",
         "amr.regrid_mode": "aligned",
         "amr.regrid_int": "1",
         "amr.tag_metric": "none",
         "amr.initial_refine": "box",
         "amr.initial_refine_levels": "2",
         "amr.initial_refine_fixed_levels": "1",
         "amr.initial_refine_box_lo": "0.25 0.25",
         "amr.initial_refine_box_hi": "0.75 0.75",
         "pyfluid.ic": "uniform",
         "pyfluid.rho0": "1.0",
         "pyfluid.pressure0": "1.0",
         "pyfluid.v1_0": "0.0",
         "pyfluid.v2_0": "0.0",
         "pyfluid.v3_0": "0.0",
         "pyfluid.max_step": "4",
         "pyfluid.timestep": "euler",
         "pyfluid.recon": "pwc",
         "pyfluid.riemann": "llf",
         "pyfluid.ng": "1",
         "pyfluid.subcycling": "on",
      },
      require_amr=True,
      min_amr_headroom=1,
      min_regrid_noops=1,
   ))
   cases.append(Case(
      "hydro_amr_subcycling_aligned_deep_l7",
      2,
      "hydro_sine_2d.in",
      {
         "geometry.is_periodic": "1 1",
         "amr.n_cell": "16 16",
         "amr.max_level": "9",
         "amr.ref_ratio": "2 2 2 2 2 2 2 2 2",
         "amr.max_grid_size": "16",
         "amr.blocking_factor": "8",
         "amr.regrid_mode": "aligned",
         "amr.regrid_int": "17",
         "amr.tag_metric": "none",
         "amr.initial_refine": "box",
         "amr.initial_refine_levels": "7",
         "amr.initial_refine_fixed_levels": "7",
         "amr.initial_refine_box_lo": "0.45 0.45",
         "amr.initial_refine_box_hi": "0.55 0.55",
         "pyfluid.ic": "uniform",
         "pyfluid.rho0": "1.0",
         "pyfluid.pressure0": "1.0",
         "pyfluid.v1_0": "0.0",
         "pyfluid.v2_0": "0.0",
         "pyfluid.v3_0": "0.0",
         "pyfluid.max_step": "2",
         "pyfluid.final_time": "-1",
         "pyfluid.timestep": "euler",
         "pyfluid.recon": "pwc",
         "pyfluid.riemann": "llf",
         "pyfluid.ng": "1",
         "pyfluid.subcycling": "on",
         "pyfluid.subcycling_steps": "local_cfl",
         "pyfluid.plot_int": "2",
         "pyfluid.checkpoint_int": "-1",
      },
      require_amr=True,
      min_amr_level=7,
      min_amr_headroom=2,
      min_regrid_noops=1,
      expected_uniform_state={
         "rho": 1.0,
         "mom1": 0.0,
         "mom2": 0.0,
         "mom3": 0.0,
         "energy": 1.5,
      },
      uniform_tolerance=5.0e-15,
   ))
   for timestep in ["euler", "corrector", "midpoint", "ssprk3", "rk4"]:
      cases.append(Case(
         f"mhd_amr_ct_subcycling_{timestep}",
         2,
         "kelvin_helmholtz_mhd_2d_single_sheet_amr_ct.in",
         {
            **AUTO_AMR_COVERAGE,
            "pyfluid.max_step": "2",
            "pyfluid.timestep": timestep,
            "pyfluid.recon": "pwl",
            "pyfluid.riemann": "hlld",
            "pyfluid.ng": "2",
            "pyfluid.subcycling": "on",
            "pyfluid.diffusion": "none",
            "pyfluid.resistivity": "0.0",
            "pyfluid.viscosity": "0.0",
            "pyfluid.plot_int": "-1",
            "amr.state_interp": "quartic-conservative",
            "amr.regrid_int": "1",
         },
         require_amr=True,
      ))
   cases.append(Case(
      "mhd_amr_ct_subcycling_aligned_regrid",
      2,
      "kelvin_helmholtz_mhd_2d_single_sheet_amr_ct.in",
      {
         **AUTO_AMR_COVERAGE,
         "amr.regrid_mode": "aligned",
         "amr.regrid_int": "16",
         "pyfluid.max_step": "2",
         "pyfluid.timestep": "ssprk3",
         "pyfluid.recon": "pwl",
         "pyfluid.riemann": "hlld",
         "pyfluid.ng": "2",
         "pyfluid.subcycling": "on",
         "pyfluid.diffusion": "none",
         "pyfluid.resistivity": "0.0",
         "pyfluid.viscosity": "0.0",
         "amr.state_interp": "quartic-conservative",
      },
      require_amr=True,
   ))
   cases.append(Case("mhd_amr_ct_subcycling_explicit_resistive", 2, "kelvin_helmholtz_mhd_2d_single_sheet_amr_ct.in", {
      **DEEP_AMR_COVERAGE,
      "pyfluid.max_step": "3",
      "pyfluid.timestep": "ssprk3",
      "pyfluid.recon": "pwl",
      "pyfluid.riemann": "hlld",
      "pyfluid.ng": "2",
      "pyfluid.subcycling": "on",
      "pyfluid.diffusion": "explicit",
      "pyfluid.resistivity": "1.0e-4",
      "pyfluid.viscosity": "1.0e-4",
      "pyfluid.plot_int": "-1",
   }, require_amr=True, min_amr_level=3, min_amr_headroom=2))
   for diffusion_method in ("explicit", "implicit"):
      cases.append(Case(
         f"hydro_amr_subcycling_{diffusion_method}_diffusion",
         2,
         "pressure_bump_2d_amr.in",
         {
            **DEEP_AMR_COVERAGE,
            "pyfluid.max_step": "3",
            "pyfluid.timestep": "ssprk3",
            "pyfluid.recon": "pwl",
            "pyfluid.riemann": "hllc",
            "pyfluid.ng": "2",
            "pyfluid.subcycling": "on",
            "pyfluid.diffusion": diffusion_method,
            "pyfluid.viscosity": "1.0e-4",
            **({
               "pyfluid.diffusion_tolerance": "1.0e-12",
               "pyfluid.diffusion_max_iterations": "500",
            } if diffusion_method == "implicit" else {}),
            "pyfluid.plot_int": "-1",
         },
         checks=DIFFUSIVE_POLICY,
         require_amr=True,
         min_amr_level=3,
         min_amr_headroom=2,
      ))
   cases.append(Case(
      "hydro_amr_subcycling_aligned_implicit_diffusion",
      2,
      "pressure_bump_2d_amr.in",
      {
         **AUTO_AMR_COVERAGE,
         "amr.regrid_mode": "aligned",
         "amr.regrid_int": "8",
         "pyfluid.max_step": "2",
         "pyfluid.timestep": "ssprk3",
         "pyfluid.recon": "pwl",
         "pyfluid.riemann": "hllc",
         "pyfluid.ng": "2",
         "pyfluid.subcycling": "on",
         "pyfluid.diffusion": "implicit",
         "pyfluid.viscosity": "1.0e-4",
         "pyfluid.diffusion_tolerance": "1.0e-12",
         "pyfluid.diffusion_max_iterations": "500",
         "pyfluid.plot_int": "-1",
      },
      checks=DIFFUSIVE_POLICY,
      require_amr=True,
   ))
   cases.append(Case(
      "hydro_source_uniform_acceleration_amr",
      2,
      "hydro_sine_2d.in",
      {
         **AUTO_AMR_COVERAGE,
         "pyfluid.max_step": "2",
         "pyfluid.amplitude": "0.1",
         "pyfluid.timestep": "ssprk3",
         "pyfluid.recon": "pwl",
         "pyfluid.riemann": "hllc",
         "pyfluid.ng": "2",
         "pyfluid.subcycling": "on",
         "pyfluid.source": "uniform_acceleration",
         "pyfluid.source_a1": "0.2",
         "pyfluid.source_a2": "-0.15",
         "pyfluid.source_a3": "0.0",
      },
      checks=DRIVEN_POLICY,
      require_amr=True,
   ))
   cases.append(Case("hydro_amr_subcycling_local_cfl_ssprk3", 2, "pressure_bump_2d_amr.in", {
      **AUTO_AMR_COVERAGE,
      "pyfluid.max_step": "2",
      "pyfluid.timestep": "ssprk3",
      "pyfluid.recon": "pwl",
      "pyfluid.riemann": "hllc",
      "pyfluid.ng": "2",
      "pyfluid.subcycling": "on",
      "pyfluid.subcycling_steps": "local_cfl",
      "pyfluid.plot_int": "-1",
      "amr.state_interp": "quartic-conservative",
      "amr.regrid_int": "1",
   }, require_amr=True))
   cases.append(Case("hydro_amr_subcycling_local_cfl_rk4", 2, "pressure_bump_2d_amr.in", {
      **AUTO_AMR_COVERAGE,
      "pyfluid.max_step": "2",
      "pyfluid.timestep": "rk4",
      "pyfluid.recon": "pwl",
      "pyfluid.riemann": "hllc",
      "pyfluid.ng": "2",
      "pyfluid.subcycling": "on",
      "pyfluid.subcycling_steps": "local_cfl",
      "pyfluid.plot_int": "-1",
      "amr.state_interp": "quartic-conservative",
      "amr.regrid_int": "1",
   }, require_amr=True))
   cases.append(Case("hydro_sod_deep_amr_subcycling_local_cfl", 1, "sod_x_1d.in", {
      "pyfluid.max_step": "100000",
      "pyfluid.final_time": "0.016",
      "pyfluid.cfl": "0.9",
      "pyfluid.physical_check_int": "1",
      "pyfluid.timestep": "ssprk3",
      "pyfluid.recon": "pwl",
      "pyfluid.riemann": "hllc",
      "pyfluid.ng": "2",
      "pyfluid.subcycling": "on",
      "pyfluid.subcycling_steps": "local_cfl",
      "pyfluid.plot_int": "-1",
      "amr.n_cell": "128",
      "amr.max_level": "4",
      "amr.ref_ratio": "2 2 2 2",
      "amr.max_grid_size": "128",
      "amr.blocking_factor": "8",
      "amr.regrid_int": "100000",
      "amr.tag_metric": "none",
      "amr.initial_refine": "box",
      "amr.initial_refine_levels": "4",
      "amr.initial_refine_fixed_levels": "4",
      "amr.initial_refine_box_lo": "0.0",
      "amr.initial_refine_box_hi": "1.0",
      "amr.state_interp": "quartic-conservative",
   }, checks=scalar_checks("sod_deep_local_cfl", momentum=False)))
   cases.append(Case("mhd_amr_ct_subcycling_local_cfl_ssprk3", 2, "kelvin_helmholtz_mhd_2d_single_sheet_amr_ct.in", {
      **AUTO_AMR_COVERAGE,
      "pyfluid.max_step": "2",
      "pyfluid.timestep": "ssprk3",
      "pyfluid.recon": "pwl",
      "pyfluid.riemann": "hlld",
      "pyfluid.ng": "2",
      "pyfluid.subcycling": "on",
      "pyfluid.subcycling_steps": "local_cfl",
      "pyfluid.diffusion": "none",
      "pyfluid.resistivity": "0.0",
      "pyfluid.viscosity": "0.0",
      "pyfluid.plot_int": "-1",
      "amr.state_interp": "quartic-conservative",
      "amr.regrid_int": "1",
   }, require_amr=True))
   cases.append(Case("hydro_amr_implicit_diffusion_no_subcycling", 2, "pressure_bump_2d_amr.in", {
      **AUTO_AMR_COVERAGE,
      "pyfluid.max_step": "2",
      "pyfluid.timestep": "ssprk3",
      "pyfluid.recon": "pwl",
      "pyfluid.riemann": "hllc",
      "pyfluid.ng": "2",
      "pyfluid.subcycling": "off",
      "pyfluid.diffusion": "implicit",
      "pyfluid.viscosity": "1.0e-4",
      "pyfluid.diffusion_tolerance": "1.0e-12",
      "pyfluid.diffusion_max_iterations": "500",
      "pyfluid.plot_int": "-1",
   }, checks=DIFFUSIVE_POLICY, require_amr=True))
   self_gravity_common = {
      "pyfluid.max_step": "100",
      "pyfluid.amplitude": "0.1",
      "pyfluid.timestep": "ssprk3",
      "pyfluid.recon": "pwc",
      "pyfluid.riemann": "llf",
      "pyfluid.ng": "1",
      "pyfluid.gravity": "poisson",
      "pyfluid.gravity_coupling": "athena",
      "pyfluid.gravity_constant": "1.0",
      "pyfluid.gravity_tolerance": "1.0e-10",
      "pyfluid.gravity_max_iterations": "200",
      "pyfluid.gravity_max_fmg_iterations": "20",
      "pyfluid.gravity_max_order": "2",
   }
   cases.append(Case("hydro_self_gravity_uniform_ssprk3", 2, "hydro_sine_2d.in", {
      **self_gravity_common,
      "amr.max_level": "0",
      "pyfluid.subcycling": "off",
   }, checks=SELF_GRAVITY_POLICY))
   self_gravity_collapse = {
      **self_gravity_common,
      "geometry.is_periodic": "1 1",
      "pyfluid.ic": "density_cloud",
      "pyfluid.rho0": "1.0",
      "pyfluid.pressure0": "0.08",
      "pyfluid.amplitude": "4.0",
      "pyfluid.bump_sigma": "0.105",
      "pyfluid.bump_x0": "0.5",
      "pyfluid.bump_y0": "0.5",
      "pyfluid.v1_0": "0.0",
      "pyfluid.v2_0": "0.0",
      "pyfluid.v3_0": "0.0",
      "pyfluid.final_time": "-1.0",
      "pyfluid.max_step": "100",
      "pyfluid.scalar_int": "1",
      "pyfluid.physical_check_int": "1",
   }
   deep_self_gravity_amr = {
      **self_gravity_collapse,
      **DEEP_AMR_COVERAGE,
      "amr.regrid_int": "2",
      "pyfluid.final_time": "0.02",
      "pyfluid.plot_int": "1",
   }
   cases.append(Case("hydro_self_gravity_collapse_uniform", 2, "hydro_sine_2d.in", {
      **self_gravity_collapse,
      "amr.n_cell": "256 256",
      "amr.max_level": "0",
      "pyfluid.subcycling": "off",
   }, checks=SELF_GRAVITY_POLICY))
   cases.append(Case("hydro_self_gravity_amr_no_subcycling", 2, "hydro_sine_2d.in", {
      **deep_self_gravity_amr,
      "pyfluid.subcycling": "off",
   }, checks=SELF_GRAVITY_AMR_POLICY, require_amr=True, min_amr_level=3, min_amr_headroom=2))
   cases.append(Case("hydro_self_gravity_amr_subcycling", 2, "hydro_sine_2d.in", {
      **deep_self_gravity_amr,
      "pyfluid.subcycling": "on",
      "pyfluid.subcycling_steps": "local_cfl",
   },
      checks=SELF_GRAVITY_AMR_POLICY,
      require_amr=True,
      min_amr_level=3,
      min_amr_headroom=2,
      comparison_reference="hydro_self_gravity_amr_no_subcycling",
      comparison_components=("rho", "pressure"),
      comparison_rel_l2_tolerance=1.0e-3,
   ))
   cases.append(Case("hydro_obstacle_forward_step", 2, "forward_step_hydro_2d_obstacle.in", {
      "pyfluid.max_step": "2",
      "pyfluid.scalar_int": "1",
      "pyfluid.plot_int": "-1",
      "pyfluid.aux_gfs": "solid_mask",
      "pyfluid.aux_gf_int": "1",
      "pyfluid.aux_gf_compute_int": "-1",
   }))
   cases.append(Case("hydro_obstacle_two_boxes", 2, "forward_step_hydro_2d_obstacle.in", {
      "pyfluid.max_step": "2",
      "pyfluid.scalar_int": "1",
      "pyfluid.plot_int": "-1",
      "pyfluid.aux_gfs": "solid_mask",
      "pyfluid.aux_gf_int": "1",
      "pyfluid.aux_gf_compute_int": "-1",
      "pyfluid.solid_box_lo": "0.0 0.0  1.1 0.45",
      "pyfluid.solid_box_hi": "0.6 0.2  1.35 0.55",
   }))
   cases.append(Case("hydro_obstacle_wedge", 2, "oblique_shock_wedge_2d_obstacle.in", {
      "pyfluid.max_step": "2",
      "pyfluid.scalar_int": "1",
      "pyfluid.plot_int": "-1",
      "pyfluid.aux_gfs": "solid_mask",
      "pyfluid.aux_gf_int": "1",
      "pyfluid.aux_gf_compute_int": "-1",
   }))
   obstacle_half_base = {
      "geometry.prob_lo": "0.0 0.0",
      "geometry.prob_hi": "1.0 1.0",
      "geometry.is_periodic": "0 0",
      "amr.n_cell": "128 64",
      "amr.max_level": "0",
      "amr.max_grid_size": "32",
      "amr.blocking_factor": "8",
      "pyfluid.ic": "uniform",
      "pyfluid.rho0": "1.0",
      "pyfluid.pressure0": "1.0",
      "pyfluid.v1_0": "0.0",
      "pyfluid.v2_0": "0.0",
      "pyfluid.v3_0": "0.0",
      "pyfluid.bc": "reflecting",
      "pyfluid.recon": "pwc",
      "pyfluid.riemann": "hllc",
      "pyfluid.timestep": "euler",
      "pyfluid.ng": "1",
      "pyfluid.solid_geometry": "boxes",
      "pyfluid.aux_gfs": "solid_mask",
      "pyfluid.aux_gf_int": "1",
      "pyfluid.aux_gf_compute_int": "-1",
   }
   bottom_half = dict(obstacle_half_base)
   bottom_half.update({
      "pyfluid.solid_box_lo": "0.0 0.0",
      "pyfluid.solid_box_hi": "1.0 0.5",
   })
   cases.append(Case("hydro_obstacle_half_bottom", 2, "hydro_sine_2d.in", bottom_half))
   left_half = dict(obstacle_half_base)
   left_half.update({
      "pyfluid.solid_box_lo": "0.0 0.0",
      "pyfluid.solid_box_hi": "0.5 1.0",
   })
   cases.append(Case("hydro_obstacle_half_left", 2, "hydro_sine_2d.in", left_half))
   cases.append(Case("hydro_moving_box_piston", 2, "hydro_moving_wall_smush_2d.in", {
      "amr.n_cell": "128 64",
      "amr.max_level": "0",
      "pyfluid.max_step": "4",
      "pyfluid.final_time": "-1",
      "pyfluid.scalar_int": "1",
      "pyfluid.plot_int": "-1",
      "pyfluid.aux_gfs": "none",
      "pyfluid.aux_gf_int": "-1",
      "pyfluid.aux_gf_compute_int": "-1",
      "pyfluid.checkpoint_int": "-1",
   }))
   cases.append(Case("hydro_rotating_cylinder_no_slip", 2, "hydro_rotating_cylinder_2d_amr.in", {
      "pyfluid.max_step": "2",
      "pyfluid.final_time": "-1",
      "pyfluid.scalar_int": "1",
      "pyfluid.plot_int": "-1",
      "pyfluid.aux_gfs": "none",
      "pyfluid.aux_gf_int": "-1",
      "pyfluid.aux_gf_compute_int": "-1",
      "pyfluid.checkpoint_int": "-1",
   }))
   cases.append(Case("mhd_rotor_ct", 2, "mhd_rotor_2d_ct.in", {
      "pyfluid.max_step": "2",
      "pyfluid.scalar_int": "1",
      "pyfluid.plot_int": "-1",
      "pyfluid.aux_gfs": "none",
      "pyfluid.aux_gf_int": "-1",
      "pyfluid.aux_gf_compute_int": "-1",
   }))
   cases.append(Case(
      "mhd_current_sheet_amr_ct_explicit_resistive",
      2,
      "resistive_current_sheet_2d_amr_ct.in",
      {
         **AUTO_AMR_COVERAGE,
         "amr.level_cell_budget": "2000000",
         "amr.level_cell_budget_side": "0",
         "amr.total_cell_budget": "2000000",
         "amr.regrid_int": "16",
         "amr.state_interp": "quartic-conservative",
         "pyfluid.max_step": "3",
         "pyfluid.final_time": "-1",
         "pyfluid.bump_sigma": "0.05",
         "pyfluid.alfven_amplitude": "0.01",
         "pyfluid.recon": "pwl",
         "pyfluid.ng": "2",
         "pyfluid.plot_int": "-1",
         "pyfluid.checkpoint_int": "-1",
         "pyfluid.aux_gfs": "j,e",
         "pyfluid.aux_gf_compute_int": "1",
         "pyfluid.aux_gf_int": "-1",
      },
      checks=DIFFUSIVE_POLICY,
      require_amr=True,
      min_amr_level=3,
      min_amr_headroom=2,
   ))
   return cases


def combo_cases() -> list[Case]:
   return [
      Case("mhd_combo_amr_ct_subcycling_resistive_aux", 2, "kelvin_helmholtz_mhd_2d_single_sheet_amr_ct.in", {
         **AUTO_AMR_COVERAGE,
         "amr.state_interp": "quartic-conservative",
         "pyfluid.max_step": "2",
         "pyfluid.timestep": "ssprk3",
         "pyfluid.recon": "wenoz-char",
         "pyfluid.riemann": "hlld",
         "pyfluid.ng": "3",
         "pyfluid.subcycling": "on",
         "pyfluid.subcycling_steps": "local_cfl",
         "pyfluid.diffusion": "explicit",
         "pyfluid.viscosity": "1.0e-3",
         "pyfluid.resistivity": "1.0e-3",
         "pyfluid.aux_gfs": "j,e",
         "pyfluid.aux_gf_compute_int": "1",
         "pyfluid.aux_gf_int": "1",
      }, require_amr=True),
      Case("mhd_combo_3d_amr_ct_subcycling_resistive_aux", 3, "magnetic_blast_noise_3d_amr_ct.in", {
         "amr.n_cell": "16 16 16",
         "amr.max_level": "1",
         "amr.regrid_int": "1",
         "pyfluid.max_step": "2",
         "pyfluid.timestep": "ssprk3",
         "pyfluid.recon": "pwl",
         "pyfluid.riemann": "hll",
         "pyfluid.ng": "2",
         "pyfluid.subcycling": "on",
         "pyfluid.subcycling_steps": "spatial",
         "pyfluid.diffusion": "explicit",
         "pyfluid.viscosity": "0.0",
         "pyfluid.resistivity": "1.0e-3",
         "pyfluid.aux_gfs": "j,e",
         "pyfluid.aux_gf_compute_int": "1",
         "pyfluid.aux_gf_int": "-1",
      }),
      Case("hydro_combo_bc_outflow_obstacle_amr_localcfl_aux", 2, "hydro_obstacle_wake_2d_amr.in", {
         "amr.n_cell": "96 32",
         "amr.max_level": "1",
         "amr.regrid_int": "1",
         "pyfluid.max_step": "3",
         "pyfluid.timestep": "ssprk3",
         "pyfluid.recon": "pwl",
         "pyfluid.riemann": "hllc",
         "pyfluid.ng": "2",
         "pyfluid.subcycling": "on",
         "pyfluid.subcycling_steps": "local_cfl",
         "pyfluid.aux_gfs": "solid_mask,solid_phi",
         "pyfluid.aux_gf_compute_int": "-1",
         "pyfluid.aux_gf_int": "1",
      }),
   ]


def dimensional_cases() -> list[Case]:
   return [
      Case("hydro_1d_smoke", 1, "hydro_sine_1d.in", {
         "pyfluid.timestep": "ssprk3",
         "pyfluid.recon": "pwl",
         "pyfluid.riemann": "hllc",
         "pyfluid.ng": "2",
      }),
      Case("hydro_3d_smoke", 3, "hydro_sine_3d.in", {
         "amr.n_cell": "16 16 16",
         "pyfluid.timestep": "ssprk3",
         "pyfluid.recon": "pwl",
         "pyfluid.riemann": "hllc",
         "pyfluid.ng": "2",
      }),
      Case("mhd_3d_ct_explicit_resistive_smoke", 3, "resistive_wave_x_3d_ct.in", {
         "amr.n_cell": "16 16 16",
         "pyfluid.diffusion": "explicit",
         "pyfluid.resistivity": "1.0e-3",
         "pyfluid.timestep": "ssprk3",
         "pyfluid.recon": "pwl",
         "pyfluid.riemann": "hlld",
         "pyfluid.ng": "2",
      }),
   ]


def all_cases() -> list[Case]:
   cases: list[Case] = []
   for suite in suite_cases().values():
      cases.extend(suite)
   return cases


def suite_cases() -> dict[str, list[Case]]:
   return {
      "hydro": hydro_method_cases(),
      "mhd": mhd_method_cases(),
      "eos": eos_cases(),
      "bc": bc_cases(),
      "diffusion": diffusion_cases(),
      "amr": amr_cases(),
      "combo": combo_cases(),
      "dimensional": dimensional_cases(),
   }


def select_cases(suites: list[str] | None, only: list[str] | None) -> list[Case]:
   by_suite = suite_cases()
   if not suites or "all" in suites:
      cases = all_cases()
   else:
      cases = []
      for suite in suites:
         cases.extend(by_suite[suite])

   selected = select_named_cases(cases, only, known_cases=all_cases())
   known = {case.name: case for case in all_cases()}
   selected_names = {case.name for case in selected}
   for case in tuple(selected):
      if case.comparison_reference is not None:
         selected_names.add(case.comparison_reference)
   return [case for case in all_cases() if case.name in selected_names and case.name in known]


def apply_hierarchy_comparisons(rows: list[dict[str, str]], cases: list[Case]) -> None:
   rows_by_name = {row["case"]: row for row in rows}
   for case in cases:
      if case.comparison_reference is None:
         continue
      row = rows_by_name[case.name]
      reference_row = rows_by_name.get(case.comparison_reference)
      if reference_row is None:
         row["status"] = "FAIL"
         row["reason"] = f"missing comparison reference {case.comparison_reference}"
         continue
      if reference_row["status"] != "PASS" or row["status"] != "PASS":
         continue
      try:
         reference_plotfile = latest_plotfile(case_output_dir(case.comparison_reference) / "stateGF", require=True)
         candidate_plotfile = latest_plotfile(case_output_dir(case.name) / "stateGF", require=True)
         comparisons = compare_hierarchy_slices(
            reference_plotfile,
            candidate_plotfile,
            case.comparison_components,
         )
      except (FileNotFoundError, RuntimeError, ValueError) as exc:
         row["status"] = "FAIL"
         row["reason"] = f"hierarchy comparison failed: {exc}"
         continue
      max_relative_l2 = max((item.relative_l2 for item in comparisons), default=0.0)
      row["comparison_reference"] = case.comparison_reference
      row["comparison_max_rel_l2"] = f"{max_relative_l2:.17g}"
      row["comparison_metrics"] = "; ".join(
         f"{item.component}:L1={item.relative_l1:.3e},L2={item.relative_l2:.3e},Linf={item.relative_linf:.3e}"
         for item in comparisons
      )
      if max_relative_l2 > case.comparison_rel_l2_tolerance:
         row["status"] = "FAIL"
         row["reason"] = (
            f"hierarchy relative L2 {max_relative_l2:.3e} exceeds "
            f"{case.comparison_rel_l2_tolerance:.3e}"
         )


def failure_row(case: Case, reason: str, ranks_override: int | None = None) -> dict[str, str]:
   ranks = case.ranks if ranks_override is None else ranks_override
   return {
      "case": case.name,
      "status": "FAIL",
      "reason": reason,
      "dim": str(case.dim),
      "input": case.base_input,
      "policy": case_checks(case).name,
      "ranks": str(ranks),
   }


def write_summary(rows: list[dict[str, str]]) -> None:
   fields = [
      "case",
      "status",
      "reason",
      "dim",
      "input",
      "policy",
      "ranks",
      "steps",
      "time",
      "min_rho",
      "min_pressure",
      "mass_rel_drift",
      "energy_rel_drift",
      "gravity_total_energy_rel_drift",
      "passive_mass_rel_drift",
      "momentum_abs_drift",
      "divb_l1",
      "avg_mcells_per_s",
      "avg_sim_time_per_s",
      "diffusion_residual",
      "diffusion_iterations",
      "amr_active_finest_level",
      "amr_finest_headroom",
      "amr_total_budget_fraction",
      "uniform_state_max_error",
      "comparison_reference",
      "comparison_max_rel_l2",
      "comparison_metrics",
      "run_dir",
   ]
   write_tsv(SUMMARY, rows, fields)


def append_feature_history(rows: list[dict[str, str]]) -> None:
   append_common_validation_history(HISTORY, [{"kind": "feature_matrix", **row} for row in rows])


def main() -> int:
   parser = argparse.ArgumentParser(description=__doc__)
   parser.add_argument("--force", action="store_true", help="remove old per-case output directories before running")
   parser.add_argument("--timeout", type=int, default=120, help="seconds allowed per case")
   parser.add_argument(
      "--suite",
      action="append",
      choices=["all", *suite_cases().keys()],
      help="case suite to run; may repeat; default is all suites",
   )
   parser.add_argument("--only", action="append", help="run one named case; may repeat")
   parser.add_argument("--ranks", type=int, help="override per-case MPI rank counts")
   parser.add_argument("--list-cases", action="store_true", help="print case names and exit")
   args = parser.parse_args()

   try:
      cases = select_cases(args.suite, args.only)
   except ValueError as exc:
      raise SystemExit(str(exc)) from exc
   if not cases:
      raise SystemExit("no cases selected")

   if args.list_cases:
      for case in cases:
         print(case.name)
      return 0

   rows: list[dict[str, str]] = []
   for idx, case in enumerate(cases, start=1):
      print(f"[{idx:02d}/{len(cases):02d}] {case.name}", flush=True)
      try:
         row = run_case(case, args.force, args.timeout, args.ranks)
      except subprocess.TimeoutExpired:
         row = failure_row(case, f"timeout after {args.timeout}s", args.ranks)
      rows.append(row)
      print(f"   {row['status']}: {row.get('reason', '')[:160]}", flush=True)
      write_summary(rows)

   apply_hierarchy_comparisons(rows, cases)
   write_summary(rows)
   append_feature_history(rows)
   failed = [row for row in rows if row["status"] != "PASS"]
   print(f"\nsummary: {len(rows) - len(failed)} passed, {len(failed)} failed")
   print(f"wrote {SUMMARY.relative_to(ROOT)}")
   print(f"appended {HISTORY.relative_to(ROOT)}")
   return 1 if failed else 0


if __name__ == "__main__":
   raise SystemExit(main())
