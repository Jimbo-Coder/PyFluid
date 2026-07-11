#!/usr/bin/env python3
"""Smoke method switches that should be free for curvilinear finite-volume runs.

These are not convergence tests. They guard that curvilinear hydro and
cell-centered MHD compose with the normal timesteppers, reconstructors,
Riemann solvers, AMR refluxing/subcycling, viscosity, and resistivity paths.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

from validation_common import (
   DIFFUSIVE_POLICY,
   DIFFUSIVE_WALL_POLICY,
   DRIVEN_POLICY,
   ROOT,
   ScalarPolicy,
   WALL_POLICY,
   append_validation_history,
   filter_cases_by_attribute,
   repo_relative,
   run_amrex,
   scalar_checks,
   scalar_validation_row,
   select_named_cases,
   write_tsv,
)


RUN_ROOT = ROOT / "runs/current/curvilinear_methods"
CASE_INPUTS = RUN_ROOT / "inputs"
SUMMARY = RUN_ROOT / "summary.tsv"
HISTORY = ROOT / "runs/current/validation_history.tsv"
ANGULAR_DIFFUSIVE_POLICY = scalar_checks("angular_diffusive", energy=False, momentum=False)


def configure_run_root(path: Path) -> None:
   global RUN_ROOT, CASE_INPUTS, SUMMARY
   RUN_ROOT = path if path.is_absolute() else ROOT / path
   CASE_INPUTS = RUN_ROOT / "inputs"
   SUMMARY = RUN_ROOT / "summary.tsv"


@dataclass(frozen=True)
class CurvilinearCase:
   name: str
   base_input: str
   coordinate: str
   branch: str
   dim: int = 3
   mhd: bool = False
   policy: ScalarPolicy = WALL_POLICY
   overrides: dict[str, str] = field(default_factory=dict)
   expect_failure: str = ""


COMMON_DYNAMIC = {
   "pyfluid.ic": "pressure_bump",
   "pyfluid.amplitude": "0.2",
   "pyfluid.bump_sigma": "0.12",
   "pyfluid.max_step": "4",
   "pyfluid.final_time": "-1.0",
   "pyfluid.scalar_int": "1",
   "pyfluid.plot_int": "-1",
   "pyfluid.checkpoint_int": "-1",
   "pyfluid.aux_gfs": "none",
   "pyfluid.aux_gf_int": "-1",
   "pyfluid.aux_gf_compute_int": "-1",
   "amr.max_level": "0",
}

SPHERICAL_BUMP = {
   **COMMON_DYNAMIC,
   "pyfluid.bump_x0": "1.5",
   "pyfluid.bump_y0": "1.5707963267948966",
   "pyfluid.bump_z0": "3.141592653589793",
}

CYLINDRICAL_BUMP = {
   **COMMON_DYNAMIC,
   "pyfluid.bump_x0": "1.5",
   "pyfluid.bump_y0": "0.0",
   "pyfluid.bump_z0": "3.141592653589793",
}

COMMON_STATIC_DIFFUSION = {
   "pyfluid.max_step": "4",
   "pyfluid.final_time": "-1.0",
   "pyfluid.scalar_int": "1",
   "pyfluid.plot_int": "-1",
   "pyfluid.checkpoint_int": "-1",
   "pyfluid.aux_gfs": "none",
   "pyfluid.aux_gf_int": "-1",
   "pyfluid.aux_gf_compute_int": "-1",
   "pyfluid.diffusion": "explicit",
   "pyfluid.viscosity": "1.0e-3",
   "pyfluid.resistivity": "0.0",
   "pyfluid.diffusion_cfl": "0.9",
   "amr.max_level": "0",
}

COMMON_IMPLICIT_SHEAR_DIFFUSION = {
   "mesh.axes": "phi",
   "geometry.coord_sys": "0",
   "geometry.prob_lo": "0.0",
   "geometry.prob_hi": "6.283185307179586",
   "geometry.is_periodic": "1",
   "amr.n_cell": "48",
   "amr.max_grid_size": "24",
   "amr.max_level": "0",
   "pyfluid.physics": "hydro",
   "pyfluid.ic": "shear_wave_x",
   "pyfluid.rho0": "1.0",
   "pyfluid.pressure0": "1.0",
   "pyfluid.amplitude": "1.0e-3",
   "pyfluid.wave_number": "1.0",
   "pyfluid.max_step": "4",
   "pyfluid.final_time": "-1.0",
   "pyfluid.ng": "2",
   "pyfluid.bc": "periodic",
   "pyfluid.recon": "pwl",
   "pyfluid.riemann": "hllc",
   "pyfluid.timestep": "ssprk3",
   "pyfluid.scalar_int": "1",
   "pyfluid.plot_int": "-1",
   "pyfluid.checkpoint_int": "-1",
   "pyfluid.aux_gfs": "none",
   "pyfluid.aux_gf_int": "-1",
   "pyfluid.aux_gf_compute_int": "-1",
   "pyfluid.diffusion": "implicit",
   "pyfluid.viscosity": "1.0e-2",
   "pyfluid.resistivity": "0.0",
   "pyfluid.diffusion_cfl": "0.9",
   "pyfluid.diffusion_tolerance": "1.0e-10",
   "pyfluid.diffusion_max_iterations": "500",
   "pyfluid.diffusion_sor_omega": "1.5",
}

COMMON_MHD_STATIC = {
   "pyfluid.max_step": "4",
   "pyfluid.final_time": "-1.0",
   "pyfluid.scalar_int": "1",
   "pyfluid.plot_int": "-1",
   "pyfluid.checkpoint_int": "-1",
   "pyfluid.aux_gfs": "none",
   "pyfluid.aux_gf_int": "-1",
   "pyfluid.aux_gf_compute_int": "-1",
   "pyfluid.physics": "mhd",
   "pyfluid.ct": "off",
   "pyfluid.ic": "uniform",
   "pyfluid.b1_0": "0.0",
   "pyfluid.b2_0": "0.05",
   "pyfluid.b3_0": "0.0",
   "pyfluid.recon": "pwl",
   "pyfluid.riemann": "hlle",
   "pyfluid.timestep": "ssprk3",
   "pyfluid.ng": "2",
   "amr.max_level": "0",
}

COMMON_MHD_CT_STATIC = {
   **COMMON_MHD_STATIC,
   "pyfluid.ct": "on",
   "pyfluid.diffusion": "none",
   "pyfluid.viscosity": "0.0",
   "pyfluid.resistivity": "0.0",
}


def recon_ng(recon: str) -> str:
   if recon in ("wenoz", "wenoz-char"):
      return "3"
   if recon == "pwl":
      return "2"
   return "1"


def method_cases(level: str) -> list[CurvilinearCase]:
   cases: list[CurvilinearCase] = []
   for timestep in ("euler", "corrector", "midpoint", "ssprk3", "rk4"):
      overrides = {
         **SPHERICAL_BUMP,
         "pyfluid.timestep": timestep,
         "pyfluid.recon": "pwl",
         "pyfluid.riemann": "hllc",
         "pyfluid.ng": "2",
      }
      cases.append(CurvilinearCase(
         name=f"spherical3d_timestep_{timestep}",
         base_input="spherical_uniform_static_3d.in",
         coordinate="spherical",
         branch="timestep",
         overrides=overrides,
      ))

   for recon in ("pwc", "pwl", "wenoz", "wenoz-char"):
      overrides = {
         **SPHERICAL_BUMP,
         "pyfluid.timestep": "ssprk3",
         "pyfluid.recon": recon,
         "pyfluid.riemann": "hllc",
         "pyfluid.ng": recon_ng(recon),
      }
      cases.append(CurvilinearCase(
         name=f"spherical3d_recon_{recon.replace('-', '_')}",
         base_input="spherical_uniform_static_3d.in",
         coordinate="spherical",
         branch="recon",
         overrides=overrides,
      ))

   for riemann in ("llf", "hll", "hlle", "hllc"):
      overrides = {
         **SPHERICAL_BUMP,
         "pyfluid.timestep": "ssprk3",
         "pyfluid.recon": "pwl",
         "pyfluid.riemann": riemann,
         "pyfluid.ng": "2",
      }
      cases.append(CurvilinearCase(
         name=f"spherical3d_riemann_{riemann}",
         base_input="spherical_uniform_static_3d.in",
         coordinate="spherical",
         branch="riemann",
         overrides=overrides,
      ))

   cases.append(CurvilinearCase(
      name="cylindrical3d_wenoz_char_hllc_rk4",
      base_input="cylindrical_uniform_static_3d.in",
      coordinate="cylindrical",
      branch="combined",
      overrides={
         **CYLINDRICAL_BUMP,
         "pyfluid.timestep": "rk4",
         "pyfluid.recon": "wenoz-char",
         "pyfluid.riemann": "hllc",
         "pyfluid.ng": "3",
      },
   ))

   cases.append(CurvilinearCase(
      name="cylindrical_phi_1d_source_uniform_acceleration",
      base_input="cylindrical_uniform_static_1d.in",
      coordinate="cylindrical",
      branch="source",
      dim=1,
      policy=DRIVEN_POLICY,
      overrides={
         "mesh.axes": "phi",
         "geometry.prob_lo": "0.0",
         "geometry.prob_hi": "6.283185307179586",
         "geometry.is_periodic": "1",
         "geometry.coord_sys": "0",
         "pyfluid.bc": "periodic",
         "amr.n_cell": "48",
         "amr.max_grid_size": "24",
         "pyfluid.ic": "uniform",
         "pyfluid.source": "uniform_acceleration",
         "pyfluid.source_a1": "0.02",
         "pyfluid.timestep": "ssprk3",
         "pyfluid.recon": "pwl",
         "pyfluid.riemann": "hllc",
         "pyfluid.ng": "2",
         "pyfluid.max_step": "4",
      },
   ))

   cases.append(CurvilinearCase(
      name="spherical_phi_1d_polytrope_passive",
      base_input="spherical_uniform_static_1d.in",
      coordinate="spherical",
      branch="eos_passive",
      dim=1,
      overrides={
         "mesh.axes": "phi",
         "geometry.prob_lo": "0.0",
         "geometry.prob_hi": "6.283185307179586",
         "geometry.is_periodic": "1",
         "geometry.coord_sys": "0",
         "pyfluid.bc": "periodic",
         "amr.n_cell": "48",
         "amr.max_grid_size": "24",
         "pyfluid.eos": "polytrope",
         "pyfluid.polytropic_k": "1.0",
         "pyfluid.polytropic_gamma": "1.6666666666666667",
         "pyfluid.passive_scalars": "1",
         "pyfluid.passive_profile": "sin_wave",
         "pyfluid.passive_background": "0.5",
         "pyfluid.passive_amplitude": "0.25",
         "pyfluid.timestep": "ssprk3",
         "pyfluid.recon": "pwl",
         "pyfluid.riemann": "hllc",
         "pyfluid.ng": "2",
         "pyfluid.max_step": "4",
      },
   ))

   cases.append(CurvilinearCase(
      name="cylindrical_rz_2d_wenoz_hllc_rk4",
      base_input="cylindrical_axis_uniform_static_2d.in",
      coordinate="cylindrical",
      branch="slice_method",
      dim=2,
      overrides={
         **COMMON_DYNAMIC,
         "mesh.axes": "r z",
         "geometry.coord_sys": "1",
         "pyfluid.bump_x0": "0.35",
         "pyfluid.bump_y0": "0.0",
         "pyfluid.timestep": "rk4",
         "pyfluid.recon": "wenoz",
         "pyfluid.riemann": "hllc",
         "pyfluid.ng": "3",
      },
   ))

   cases.append(CurvilinearCase(
      name="cylindrical_phiz_2d_wenoz_hllc_rk4",
      base_input="cylindrical_uniform_static_2d.in",
      coordinate="cylindrical",
      branch="slice_method",
      dim=2,
      overrides={
         **COMMON_DYNAMIC,
         "mesh.axes": "phi z",
         "geometry.coord_sys": "0",
         "geometry.prob_lo": "0.0 -0.5",
         "geometry.prob_hi": "6.283185307179586 0.5",
         "geometry.is_periodic": "1 0",
         "pyfluid.bc": "reflecting",
         "pyfluid.bc_xlo": "periodic",
         "pyfluid.bc_xhi": "periodic",
         "pyfluid.bc_ylo": "reflecting",
         "pyfluid.bc_yhi": "reflecting",
         "amr.n_cell": "32 32",
         "amr.max_grid_size": "16",
         "pyfluid.bump_x0": "3.141592653589793",
         "pyfluid.bump_y0": "0.0",
         "pyfluid.timestep": "rk4",
         "pyfluid.recon": "wenoz",
         "pyfluid.riemann": "hllc",
         "pyfluid.ng": "3",
      },
   ))

   cases.append(CurvilinearCase(
      name="spherical_thetaphi_2d_wenoz_hllc_rk4",
      base_input="spherical_uniform_static_2d.in",
      coordinate="spherical",
      branch="slice_method",
      dim=2,
      overrides={
         **COMMON_DYNAMIC,
         "mesh.axes": "theta phi",
         "geometry.coord_sys": "0",
         "geometry.prob_lo": "0.3 0.0",
         "geometry.prob_hi": "2.8415926535897933 6.283185307179586",
         "geometry.is_periodic": "0 1",
         "pyfluid.bc": "reflecting",
         "pyfluid.bc_ylo": "periodic",
         "pyfluid.bc_yhi": "periodic",
         "amr.n_cell": "32 32",
         "amr.max_grid_size": "16",
         "pyfluid.bump_x0": "1.5707963267948966",
         "pyfluid.bump_y0": "3.141592653589793",
         "pyfluid.timestep": "rk4",
         "pyfluid.recon": "wenoz",
         "pyfluid.riemann": "hllc",
         "pyfluid.ng": "3",
      },
   ))

   cases.append(CurvilinearCase(
      name="spherical3d_amr_metric_reflux",
      base_input="spherical_uniform_static_3d.in",
      coordinate="spherical",
      branch="amr",
      overrides={
         **SPHERICAL_BUMP,
         "pyfluid.timestep": "ssprk3",
         "pyfluid.recon": "pwl",
         "pyfluid.riemann": "hllc",
         "pyfluid.ng": "2",
         "amr.max_level": "1",
         "amr.ref_ratio": "2",
         "amr.regrid_int": "2",
         "amr.tag_metric": "auto",
         "amr.tag_threshold": "0.08",
         "amr.initial_refine": "box",
         "amr.initial_refine_levels": "1",
         "amr.initial_refine_box_lo": "1.25 1.0 2.4",
         "amr.initial_refine_box_hi": "1.75 2.1 3.9",
         "pyfluid.subcycling": "off",
      },
   ))

   cases.append(CurvilinearCase(
      name="spherical3d_amr_subcycling",
      base_input="spherical_uniform_static_3d.in",
      coordinate="spherical",
      branch="subcycling",
      overrides={
         **SPHERICAL_BUMP,
         "pyfluid.timestep": "ssprk3",
         "pyfluid.recon": "pwl",
         "pyfluid.riemann": "hllc",
         "pyfluid.ng": "2",
         "amr.max_level": "1",
         "amr.ref_ratio": "2",
         "amr.regrid_int": "2",
         "amr.tag_metric": "auto",
         "amr.tag_threshold": "0.08",
         "amr.initial_refine": "box",
         "amr.initial_refine_levels": "1",
         "amr.initial_refine_box_lo": "1.25 1.0 2.4",
         "amr.initial_refine_box_hi": "1.75 2.1 3.9",
         "pyfluid.subcycling": "on",
      },
   ))

   cases.append(CurvilinearCase(
      name="cylindrical3d_amr_metric_reflux",
      base_input="cylindrical_uniform_static_3d.in",
      coordinate="cylindrical",
      branch="amr",
      overrides={
         **CYLINDRICAL_BUMP,
         "pyfluid.timestep": "ssprk3",
         "pyfluid.recon": "pwl",
         "pyfluid.riemann": "hllc",
         "pyfluid.ng": "2",
         "amr.max_level": "1",
         "amr.ref_ratio": "2",
         "amr.regrid_int": "2",
         "amr.tag_metric": "auto",
         "amr.tag_threshold": "0.08",
         "amr.initial_refine": "box",
         "amr.initial_refine_levels": "1",
         "amr.initial_refine_box_lo": "1.25 -0.25 2.4",
         "amr.initial_refine_box_hi": "1.75 0.25 3.9",
         "pyfluid.subcycling": "off",
      },
   ))

   cases.append(CurvilinearCase(
      name="cylindrical3d_amr_subcycling",
      base_input="cylindrical_uniform_static_3d.in",
      coordinate="cylindrical",
      branch="subcycling",
      overrides={
         **CYLINDRICAL_BUMP,
         "pyfluid.timestep": "ssprk3",
         "pyfluid.recon": "pwl",
         "pyfluid.riemann": "hllc",
         "pyfluid.ng": "2",
         "amr.max_level": "1",
         "amr.ref_ratio": "2",
         "amr.regrid_int": "2",
         "amr.tag_metric": "auto",
         "amr.tag_threshold": "0.08",
         "amr.initial_refine": "box",
         "amr.initial_refine_levels": "1",
         "amr.initial_refine_box_lo": "1.25 -0.25 2.4",
         "amr.initial_refine_box_hi": "1.75 0.25 3.9",
         "pyfluid.subcycling": "on",
      },
   ))

   cases.append(CurvilinearCase(
      name="spherical3d_explicit_viscosity_static",
      base_input="spherical_uniform_static_3d.in",
      coordinate="spherical",
      branch="diffusion",
      overrides=COMMON_STATIC_DIFFUSION,
   ))

   cases.append(CurvilinearCase(
      name="cylindrical3d_explicit_viscosity_static",
      base_input="cylindrical_uniform_static_3d.in",
      coordinate="cylindrical",
      branch="diffusion",
      overrides=COMMON_STATIC_DIFFUSION,
   ))

   cases.append(CurvilinearCase(
      name="spherical3d_implicit_viscosity_static",
      base_input="spherical_uniform_static_3d.in",
      coordinate="spherical",
      branch="diffusion",
      overrides={**COMMON_STATIC_DIFFUSION, "pyfluid.diffusion": "implicit"},
   ))

   cases.append(CurvilinearCase(
      name="cylindrical_phi_1d_implicit_viscosity_shear",
      base_input="cylindrical_uniform_static_1d.in",
      coordinate="cylindrical",
      branch="diffusion",
      dim=1,
      policy=ANGULAR_DIFFUSIVE_POLICY,
      overrides=COMMON_IMPLICIT_SHEAR_DIFFUSION,
   ))

   cases.append(CurvilinearCase(
      name="spherical_phi_1d_implicit_viscosity_shear",
      base_input="spherical_uniform_static_1d.in",
      coordinate="spherical",
      branch="diffusion",
      dim=1,
      policy=ANGULAR_DIFFUSIVE_POLICY,
      overrides=COMMON_IMPLICIT_SHEAR_DIFFUSION,
   ))

   cases.append(CurvilinearCase(
      name="cylindrical3d_mhd_cell_centered_static",
      base_input="cylindrical_uniform_static_3d.in",
      coordinate="cylindrical",
      branch="mhd",
      mhd=True,
      overrides=COMMON_MHD_STATIC,
   ))

   cases.append(CurvilinearCase(
      name="cylindrical_phiz_2d_mhd_ct_resistive_wave_explicit",
      base_input="cylindrical_uniform_static_2d.in",
      coordinate="cylindrical",
      branch="resistive_mhd_ct",
      dim=2,
      mhd=True,
      policy=DIFFUSIVE_POLICY,
      overrides={
         **COMMON_MHD_CT_STATIC,
         "mesh.axes": "phi z",
         "geometry.coord_sys": "0",
         "geometry.prob_lo": "0.0 -0.5",
         "geometry.prob_hi": "6.283185307179586 0.5",
         "geometry.is_periodic": "1 1",
         "pyfluid.bc": "periodic",
         "amr.n_cell": "48 24",
         "amr.max_grid_size": "24",
         "pyfluid.ic": "resistive_wave_x",
         "pyfluid.alfven_amplitude": "1.0e-3",
         "pyfluid.wave_number": "1.0",
         "pyfluid.diffusion": "explicit",
         "pyfluid.resistivity": "1.0e-2",
         "pyfluid.max_step": "4",
      },
   ))

   cases.append(CurvilinearCase(
      name="spherical3d_mhd_ct_toroidal_resistivity_explicit",
      base_input="spherical_uniform_static_3d.in",
      coordinate="spherical",
      branch="resistive_mhd_ct",
      mhd=True,
      policy=DIFFUSIVE_WALL_POLICY,
      overrides={
         **COMMON_MHD_CT_STATIC,
         "pyfluid.b1_0": "0.0",
         "pyfluid.b2_0": "0.0",
         "pyfluid.b3_0": "0.03",
         "pyfluid.diffusion": "explicit",
         "pyfluid.resistivity": "1.0e-2",
         "pyfluid.max_step": "4",
      },
   ))

   cases.append(CurvilinearCase(
      name="cylindrical3d_mhd_amr_subcycling_static",
      base_input="cylindrical_uniform_static_3d.in",
      coordinate="cylindrical",
      branch="mhd_amr_subcycling",
      mhd=True,
      overrides={
         **COMMON_MHD_STATIC,
         "pyfluid.b1_0": "0.0",
         "pyfluid.b2_0": "0.0",
         "pyfluid.b3_0": "0.0",
         "amr.max_level": "1",
         "amr.ref_ratio": "2",
         "amr.regrid_int": "2",
         "amr.tag_metric": "auto",
         "amr.tag_threshold": "0.08",
         "amr.initial_refine": "box",
         "amr.initial_refine_levels": "1",
         "amr.initial_refine_box_lo": "1.25 -0.25 1.0",
         "amr.initial_refine_box_hi": "1.75 0.25 5.0",
         "pyfluid.subcycling": "on",
      },
   ))

   cases.append(CurvilinearCase(
      name="spherical3d_mhd_amr_subcycling_static",
      base_input="spherical_uniform_static_3d.in",
      coordinate="spherical",
      branch="mhd_amr_subcycling",
      mhd=True,
      overrides={
         **COMMON_MHD_STATIC,
         "pyfluid.b1_0": "0.0",
         "pyfluid.b2_0": "0.0",
         "pyfluid.b3_0": "0.0",
         "amr.max_level": "1",
         "amr.ref_ratio": "2",
         "amr.regrid_int": "2",
         "amr.tag_metric": "auto",
         "amr.tag_threshold": "0.08",
         "amr.initial_refine": "box",
         "amr.initial_refine_levels": "1",
         "amr.initial_refine_box_lo": "1.25 1.0 2.4",
         "amr.initial_refine_box_hi": "1.75 2.1 3.9",
         "pyfluid.subcycling": "on",
      },
   ))

   cases.append(CurvilinearCase(
      name="spherical_rphi_2d_mhd_cell_centered_static",
      base_input="spherical_uniform_static_rphi_2d.in",
      coordinate="spherical",
      branch="mhd",
      dim=2,
      mhd=True,
      overrides={
         **COMMON_MHD_STATIC,
         "pyfluid.b1_0": "0.0",
         "pyfluid.b2_0": "0.03",
         "pyfluid.b3_0": "0.0",
      },
   ))

   cases.append(CurvilinearCase(
      name="cylindrical2d_mhd_cell_centered_static",
      base_input="cylindrical_uniform_static_2d.in",
      coordinate="cylindrical",
      branch="mhd",
      dim=2,
      mhd=True,
      overrides={
         **COMMON_MHD_STATIC,
         "pyfluid.b1_0": "0.0",
         "pyfluid.b2_0": "0.04",
         "pyfluid.b3_0": "0.0",
      },
   ))

   cases.append(CurvilinearCase(
      name="spherical_rtheta_2d_mhd_cell_centered_static",
      base_input="spherical_uniform_static_2d.in",
      coordinate="spherical",
      branch="mhd",
      dim=2,
      mhd=True,
      overrides={
         **COMMON_MHD_STATIC,
         "pyfluid.b1_0": "0.0",
         "pyfluid.b2_0": "0.0",
         "pyfluid.b3_0": "0.0",
      },
   ))

   cases.append(CurvilinearCase(
      name="cylindrical2d_mhd_ct_static",
      base_input="cylindrical_uniform_static_2d.in",
      coordinate="cylindrical",
      branch="mhd_ct",
      dim=2,
      mhd=True,
      overrides={
         **COMMON_MHD_CT_STATIC,
         "pyfluid.b1_0": "0.0",
         "pyfluid.b2_0": "0.04",
         "pyfluid.b3_0": "0.0",
      },
   ))

   cases.append(CurvilinearCase(
      name="cylindrical_rphi_2d_mhd_ct_static",
      base_input="cylindrical_uniform_static_2d.in",
      coordinate="cylindrical",
      branch="mhd_ct",
      dim=2,
      mhd=True,
      overrides={
         **COMMON_MHD_CT_STATIC,
         "mesh.axes": "r phi",
         "geometry.coord_sys": "0",
         "geometry.is_periodic": "0 1",
         "pyfluid.bc": "reflecting",
         "pyfluid.bc_xlo": "reflecting",
         "pyfluid.bc_xhi": "reflecting",
         "pyfluid.bc_ylo": "periodic",
         "pyfluid.bc_yhi": "periodic",
         "pyfluid.b1_0": "0.0",
         "pyfluid.b2_0": "0.04",
         "pyfluid.b3_0": "0.0",
      },
   ))

   cases.append(CurvilinearCase(
      name="spherical_rtheta_2d_mhd_ct_static",
      base_input="spherical_uniform_static_2d.in",
      coordinate="spherical",
      branch="mhd_ct",
      dim=2,
      mhd=True,
      overrides={
         **COMMON_MHD_CT_STATIC,
         "pyfluid.b1_0": "0.0",
         "pyfluid.b2_0": "0.0",
         "pyfluid.b3_0": "0.04",
      },
   ))

   cases.append(CurvilinearCase(
      name="spherical_rphi_2d_mhd_ct_static",
      base_input="spherical_uniform_static_rphi_2d.in",
      coordinate="spherical",
      branch="mhd_ct",
      dim=2,
      mhd=True,
      overrides={
         **COMMON_MHD_CT_STATIC,
         "pyfluid.b1_0": "0.0",
         "pyfluid.b2_0": "0.04",
         "pyfluid.b3_0": "0.0",
      },
   ))

   cases.append(CurvilinearCase(
      name="cylindrical3d_mhd_ct_static",
      base_input="cylindrical_uniform_static_3d.in",
      coordinate="cylindrical",
      branch="mhd_ct",
      mhd=True,
      overrides={
         **COMMON_MHD_CT_STATIC,
         "pyfluid.b1_0": "0.0",
         "pyfluid.b2_0": "0.04",
         "pyfluid.b3_0": "0.0",
      },
   ))

   cases.append(CurvilinearCase(
      name="spherical3d_mhd_ct_zero_static",
      base_input="spherical_uniform_static_3d.in",
      coordinate="spherical",
      branch="mhd_ct",
      mhd=True,
      overrides={
         **COMMON_MHD_CT_STATIC,
         "pyfluid.b1_0": "0.0",
         "pyfluid.b2_0": "0.0",
         "pyfluid.b3_0": "0.0",
      },
   ))

   cases.append(CurvilinearCase(
      name="spherical3d_mhd_ct_pressure_bump",
      base_input="spherical_uniform_static_3d.in",
      coordinate="spherical",
      branch="mhd_ct_dynamic",
      mhd=True,
      overrides={
         **COMMON_MHD_CT_STATIC,
         "pyfluid.ic": "pressure_bump",
         "pyfluid.amplitude": "0.03",
         "pyfluid.bump_x0": "1.5",
         "pyfluid.bump_y0": "1.5707963267948966",
         "pyfluid.bump_z0": "3.141592653589793",
         "pyfluid.max_step": "4",
         "pyfluid.b1_0": "0.0",
         "pyfluid.b2_0": "0.0",
         "pyfluid.b3_0": "0.02",
      },
   ))

   cases.append(CurvilinearCase(
      name="cylindrical3d_mhd_ct_wenoz_char_hlld_rk4_pressure_bump",
      base_input="cylindrical_uniform_static_3d.in",
      coordinate="cylindrical",
      branch="mhd_ct_dynamic",
      mhd=True,
      overrides={
         **COMMON_MHD_CT_STATIC,
         "pyfluid.ic": "pressure_bump",
         "pyfluid.amplitude": "0.04",
         "pyfluid.bump_x0": "1.5",
         "pyfluid.bump_y0": "3.141592653589793",
         "pyfluid.bump_z0": "0.0",
         "pyfluid.max_step": "4",
         "pyfluid.b1_0": "0.0",
         "pyfluid.b2_0": "0.03",
         "pyfluid.b3_0": "0.01",
         "pyfluid.recon": "wenoz-char",
         "pyfluid.riemann": "hlld",
         "pyfluid.timestep": "rk4",
         "pyfluid.ng": "3",
      },
   ))

   cases.append(CurvilinearCase(
      name="cylindrical2d_implicit_ct_resistivity_expected_reject",
      base_input="cylindrical_uniform_static_2d.in",
      coordinate="cylindrical",
      branch="expected_reject",
      dim=2,
      mhd=True,
      expect_failure="implicit resistive MHD is disabled",
      overrides={
         **COMMON_MHD_CT_STATIC,
         "pyfluid.diffusion": "implicit",
         "pyfluid.resistivity": "1.0e-3",
         "pyfluid.b1_0": "0.0",
         "pyfluid.b2_0": "0.04",
         "pyfluid.b3_0": "0.0",
      },
   ))

   cases.append(CurvilinearCase(
      name="spherical2d_fixed_gravity_expected_reject",
      base_input="spherical_uniform_static_2d.in",
      coordinate="spherical",
      branch="expected_reject",
      dim=2,
      expect_failure="curvilinear cylindrical/spherical gravity is not implemented yet",
      overrides={
         **COMMON_DYNAMIC,
         "pyfluid.gravity": "fixed",
         "pyfluid.gravity_fixed_potential": "uniform_acceleration",
         "pyfluid.gravity_a1": "0.01",
      },
   ))

   cases.append(CurvilinearCase(
      name="cylindrical2d_mhd_ct_pressure_bump",
      base_input="cylindrical_uniform_static_2d.in",
      coordinate="cylindrical",
      branch="mhd_ct_dynamic",
      dim=2,
      mhd=True,
      overrides={
         **COMMON_MHD_CT_STATIC,
         "pyfluid.ic": "pressure_bump",
         "pyfluid.amplitude": "0.1",
         "pyfluid.bump_x0": "1.5",
         "pyfluid.bump_y0": "0.5",
         "pyfluid.max_step": "8",
         "pyfluid.b1_0": "0.0",
         "pyfluid.b2_0": "0.04",
         "pyfluid.b3_0": "0.0",
      },
   ))

   cases.append(CurvilinearCase(
      name="spherical_rtheta_2d_mhd_ct_pressure_bump",
      base_input="spherical_uniform_static_2d.in",
      coordinate="spherical",
      branch="mhd_ct_dynamic",
      dim=2,
      mhd=True,
      overrides={
         **COMMON_MHD_CT_STATIC,
         "pyfluid.ic": "pressure_bump",
         "pyfluid.amplitude": "0.05",
         "pyfluid.bump_x0": "1.5",
         "pyfluid.bump_y0": "1.5707963267948966",
         "pyfluid.max_step": "8",
         "pyfluid.b1_0": "0.0",
         "pyfluid.b2_0": "0.0",
         "pyfluid.b3_0": "0.03",
      },
   ))

   cases.append(CurvilinearCase(
      name="cylindrical3d_mhd_ct_pressure_bump",
      base_input="cylindrical_uniform_static_3d.in",
      coordinate="cylindrical",
      branch="mhd_ct_dynamic",
      mhd=True,
      overrides={
         **COMMON_MHD_CT_STATIC,
         "pyfluid.ic": "pressure_bump",
         "pyfluid.amplitude": "0.05",
         "pyfluid.bump_x0": "1.5",
         "pyfluid.bump_y0": "0.0",
         "pyfluid.bump_z0": "3.141592653589793",
         "pyfluid.max_step": "6",
         "pyfluid.b1_0": "0.0",
         "pyfluid.b2_0": "0.03",
         "pyfluid.b3_0": "0.0",
      },
   ))

   cases.append(CurvilinearCase(
      name="cylindrical2d_mhd_amr_ct_pressure_bump",
      base_input="cylindrical_uniform_static_2d.in",
      coordinate="cylindrical",
      branch="mhd_amr_ct",
      dim=2,
      mhd=True,
      overrides={
         **COMMON_MHD_CT_STATIC,
         "pyfluid.ic": "pressure_bump",
         "pyfluid.amplitude": "0.1",
         "pyfluid.bump_x0": "1.5",
         "pyfluid.bump_y0": "0.5",
         "pyfluid.max_step": "8",
         "pyfluid.b1_0": "0.0",
         "pyfluid.b2_0": "0.04",
         "pyfluid.b3_0": "0.0",
         "amr.max_level": "1",
         "amr.ref_ratio": "2",
         "amr.regrid_int": "2",
         "amr.tag_metric": "auto",
         "amr.tag_threshold": "0.08",
         "amr.initial_refine": "box",
         "amr.initial_refine_levels": "1",
         "amr.initial_refine_box_lo": "1.25 0.2",
         "amr.initial_refine_box_hi": "1.75 0.8",
         "pyfluid.subcycling": "off",
      },
   ))

   cases.append(CurvilinearCase(
      name="spherical_rtheta_2d_mhd_amr_ct_pressure_bump",
      base_input="spherical_uniform_static_2d.in",
      coordinate="spherical",
      branch="mhd_amr_ct",
      dim=2,
      mhd=True,
      overrides={
         **COMMON_MHD_CT_STATIC,
         "pyfluid.ic": "pressure_bump",
         "pyfluid.amplitude": "0.05",
         "pyfluid.bump_x0": "1.5",
         "pyfluid.bump_y0": "1.5707963267948966",
         "pyfluid.max_step": "8",
         "pyfluid.b1_0": "0.0",
         "pyfluid.b2_0": "0.0",
         "pyfluid.b3_0": "0.03",
         "amr.max_level": "1",
         "amr.ref_ratio": "2",
         "amr.regrid_int": "2",
         "amr.tag_metric": "auto",
         "amr.tag_threshold": "0.08",
         "amr.initial_refine": "box",
         "amr.initial_refine_levels": "1",
         "amr.initial_refine_box_lo": "1.25 1.1",
         "amr.initial_refine_box_hi": "1.75 2.0",
         "pyfluid.subcycling": "off",
      },
   ))

   cases.append(CurvilinearCase(
      name="cylindrical2d_mhd_amr_ct_subcycling_pressure_bump",
      base_input="cylindrical_uniform_static_2d.in",
      coordinate="cylindrical",
      branch="mhd_amr_ct_subcycling",
      dim=2,
      mhd=True,
      overrides={
         **COMMON_MHD_CT_STATIC,
         "pyfluid.ic": "pressure_bump",
         "pyfluid.amplitude": "0.1",
         "pyfluid.bump_x0": "1.5",
         "pyfluid.bump_y0": "0.5",
         "pyfluid.max_step": "8",
         "pyfluid.b1_0": "0.0",
         "pyfluid.b2_0": "0.04",
         "pyfluid.b3_0": "0.0",
         "amr.max_level": "1",
         "amr.ref_ratio": "2",
         "amr.regrid_int": "2",
         "amr.tag_metric": "auto",
         "amr.tag_threshold": "0.08",
         "amr.initial_refine": "box",
         "amr.initial_refine_levels": "1",
         "amr.initial_refine_box_lo": "1.25 0.2",
         "amr.initial_refine_box_hi": "1.75 0.8",
         "pyfluid.subcycling": "on",
      },
   ))

   if level == "quick":
      return cases
   if level == "full":
      cases.append(CurvilinearCase(
         name="cylindrical3d_mhd_amr_ct_pressure_bump",
         base_input="cylindrical_uniform_static_3d.in",
         coordinate="cylindrical",
         branch="mhd_amr_ct",
         mhd=True,
         overrides={
            **COMMON_MHD_CT_STATIC,
            "pyfluid.ic": "pressure_bump",
            "pyfluid.amplitude": "0.05",
            "pyfluid.bump_x0": "1.5",
            "pyfluid.bump_y0": "0.0",
            "pyfluid.bump_z0": "3.141592653589793",
            "pyfluid.max_step": "6",
            "pyfluid.b1_0": "0.0",
            "pyfluid.b2_0": "0.03",
            "pyfluid.b3_0": "0.0",
            "amr.max_level": "1",
            "amr.ref_ratio": "2",
            "amr.regrid_int": "2",
            "amr.tag_metric": "auto",
            "amr.tag_threshold": "0.08",
            "amr.initial_refine": "box",
            "amr.initial_refine_levels": "1",
            "amr.initial_refine_box_lo": "1.25 -0.25 2.4",
            "amr.initial_refine_box_hi": "1.75 0.25 3.9",
            "pyfluid.subcycling": "off",
         },
      ))
      for timestep in ("euler", "corrector", "midpoint", "ssprk3", "rk4"):
         cases.append(CurvilinearCase(
            name=f"cylindrical3d_timestep_{timestep}",
            base_input="cylindrical_uniform_static_3d.in",
            coordinate="cylindrical",
            branch="timestep",
            overrides={
               **CYLINDRICAL_BUMP,
               "pyfluid.timestep": timestep,
               "pyfluid.recon": "pwl",
               "pyfluid.riemann": "hllc",
               "pyfluid.ng": "2",
            },
         ))
   return cases


def read_input(path: Path) -> list[str]:
   return path.read_text().splitlines()


def write_case_input(case: CurvilinearCase, force: bool) -> Path:
   run_dir = RUN_ROOT / case.name
   if force and run_dir.exists():
      shutil.rmtree(run_dir)
   run_dir.mkdir(parents=True, exist_ok=True)
   CASE_INPUTS.mkdir(parents=True, exist_ok=True)

   lines = read_input(ROOT / "pyfluid_amrex/inputs" / case.base_input)
   values = {
      "pyfluid.run_name": case.name,
      "pyfluid.output_dir": repo_relative(run_dir),
      **case.overrides,
   }
   lines.append("")
   lines.append("# validate_curvilinear_methods.py overrides")
   for key, value in values.items():
      lines.append(f"{key} = {value}")

   path = CASE_INPUTS / f"{case.name}.in"
   path.write_text("\n".join(lines) + "\n")
   return path


def failure_row(case: CurvilinearCase, reason: str, ranks: int) -> dict[str, object]:
   return {
      "kind": "curvilinear_methods",
      "case": case.name,
      "label": case.name,
      "status": "FAIL",
      "reason": reason,
      "dim": str(case.dim),
      "input": case.base_input,
      "policy": case.policy.name,
      "ranks": str(ranks),
      "coordinate": case.coordinate,
      "branch": case.branch,
      "run_dir": repo_relative(RUN_ROOT / case.name),
   }


def collect_expected_reject_backtraces(run_dir: Path) -> None:
   backtrace_dir = run_dir / "backtraces"
   for index, backtrace_path in enumerate(sorted(ROOT.glob("Backtrace.*"))):
      backtrace_dir.mkdir(parents=True, exist_ok=True)
      destination = backtrace_dir / backtrace_path.name
      if destination.exists():
         destination = backtrace_dir / f"{backtrace_path.name}.{index}"
      backtrace_path.replace(destination)


def run_case(case: CurvilinearCase, ranks: int, force: bool, timeout: int) -> dict[str, object]:
   input_path = write_case_input(case, force)
   run_ranks = 1 if case.expect_failure else ranks
   run_timeout = min(timeout, 15) if case.expect_failure else timeout
   result = run_amrex(dim=case.dim, input_path=input_path, ranks=run_ranks, timeout=run_timeout)
   run_dir = RUN_ROOT / case.name
   (run_dir / "run.log").write_text(result.stdout)

   if case.expect_failure:
      if result.returncode != 0 and case.expect_failure in result.stdout:
         collect_expected_reject_backtraces(run_dir)
         return {
            "kind": "curvilinear_methods",
            "case": case.name,
            "label": case.name,
            "status": "PASS",
            "reason": "expected rejection",
            "dim": str(case.dim),
            "input": case.base_input,
            "policy": "expected_reject",
            "ranks": str(run_ranks),
            "coordinate": case.coordinate,
            "branch": case.branch,
            "run_dir": repo_relative(run_dir),
            "avg_mcells_per_s": "",
            "avg_sim_time_per_s": "",
            "log": repo_relative(run_dir / "run.log"),
         }
      tail = " | ".join(result.stdout.splitlines()[-8:])
      return failure_row(case, f"expected rejection missing: {tail}", run_ranks)

   if result.returncode != 0:
      tail = " | ".join(result.stdout.splitlines()[-8:])
      return failure_row(case, f"returncode={result.returncode}: {tail}", ranks)
   if not (run_dir / "run.in").exists():
      raise RuntimeError(f"solver did not write canonical run configuration: {run_dir / 'run.in'}")

   row = scalar_validation_row(
      run_dir,
      case=case.name,
      input_file=run_dir / "run.in",
      policy=case.policy,
      mhd=case.mhd,
      dim=case.dim,
      ranks=ranks,
      base_input=case.base_input,
      kind="curvilinear_methods",
   )
   row["coordinate"] = case.coordinate
   row["branch"] = case.branch
   row["log"] = repo_relative(run_dir / "run.log")
   return row


def write_summary(rows: list[dict[str, object]]) -> None:
   write_tsv(
      SUMMARY,
      rows,
      [
         "case",
         "status",
         "reason",
         "coordinate",
         "branch",
         "dim",
         "input",
         "policy",
         "ranks",
         "steps",
         "time",
         "mass_rel_drift",
         "energy_rel_drift",
         "momentum_abs_drift",
         "avg_mcells_per_s",
         "avg_sim_time_per_s",
         "run_dir",
         "log",
      ],
   )


def main() -> int:
   parser = argparse.ArgumentParser(description=__doc__)
   parser.add_argument("--level", choices=("quick", "full"), default="quick")
   parser.add_argument("--force", action="store_true")
   parser.add_argument("--ranks", type=int, default=3)
   parser.add_argument("--timeout", type=int, default=120)
   parser.add_argument("--list-cases", action="store_true")
   parser.add_argument("--list-coordinates", action="store_true")
   parser.add_argument("--list-branches", action="store_true")
   parser.add_argument("--coordinate", action="append", default=[], help="Run one coordinate group; may be repeated.")
   parser.add_argument("--branch", action="append", default=[], help="Run one branch group; may be repeated.")
   parser.add_argument("--only", action="append", default=[], help="Run only the named case; may be repeated.")
   parser.add_argument(
      "--run-root",
      type=Path,
      default=RUN_ROOT,
      help="Run/output directory. Relative paths are interpreted from the repository root.",
   )
   args = parser.parse_args()

   os.environ.setdefault("UV_CACHE_DIR", str(ROOT / ".uv-cache"))
   os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib"))
   configure_run_root(args.run_root)
   RUN_ROOT.mkdir(parents=True, exist_ok=True)
   cases = method_cases(args.level)
   if args.list_coordinates:
      for coordinate in sorted({case.coordinate for case in cases}):
         print(coordinate)
      return 0
   if args.list_branches:
      for branch in sorted({case.branch for case in cases}):
         print(branch)
      return 0
   try:
      cases = filter_cases_by_attribute(cases, args.coordinate, attribute="coordinate", label="coordinate")
      cases = filter_cases_by_attribute(cases, args.branch, attribute="branch", label="branch")
   except ValueError as exc:
      print(str(exc), file=sys.stderr)
      return 2
   if args.list_cases:
      for case in cases:
         print(case.name)
      return 0
   if args.only:
      try:
         cases = select_named_cases(cases, args.only)
      except ValueError as exc:
         print(str(exc), file=sys.stderr)
         return 2

   rows: list[dict[str, object]] = []
   for index, case in enumerate(cases, start=1):
      print(f"[{index:02d}/{len(cases):02d}] {case.name}", flush=True)
      row = run_case(case, args.ranks, args.force, args.timeout)
      rows.append(row)
      print(f"   {row['status']}: {row.get('reason', '')}", flush=True)
      write_summary(rows)

   append_validation_history(HISTORY, rows)
   failed = [row for row in rows if row["status"] != "PASS"]
   print(f"summary: {len(rows) - len(failed)} passed, {len(failed)} failed")
   print(f"wrote {repo_relative(SUMMARY)}")
   print(f"appended {repo_relative(HISTORY)}")
   return 1 if failed else 0


if __name__ == "__main__":
   raise SystemExit(main())
