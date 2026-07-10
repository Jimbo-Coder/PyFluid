#!/usr/bin/env python3
"""Compare degenerate AMReX executables against lower-dimensional runs.

These checks catch drift where a 2D/3D specialization starts doing different
math from the lower-dimensional path when the extra direction has one cell and
the IC is independent of that direction.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "pyfluid_amrex" / "tools"))

from plot_amrex_slice import parse_plot_header, read_level_state  # noqa: E402
from plotfile_reader import latest_plotfile
from validation_common import mpi_prefix
from validation_common import append_validation_history, repo_relative  # noqa: E402


BUILD = ROOT / "build" / "pyfluid_amrex"
INPUTS = ROOT / "pyfluid_amrex" / "inputs"
RUN_ROOT = ROOT / "runs" / "current" / "dimensional_consistency"
INPUT_ROOT = RUN_ROOT / "inputs"
RESULTS = RUN_ROOT / "results.tsv"
HISTORY = ROOT / "runs" / "current" / "validation_history.tsv"


@dataclass(frozen=True)
class RunSpec:
   label: str
   dim: int
   base_input: str
   overrides: tuple[str, ...]


@dataclass(frozen=True)
class StateTransform:
   spatial_permutation: tuple[int, ...] | None = None
   spatial_reversals: tuple[int, ...] = ()
   component_map: tuple[tuple[str, str, float], ...] = ()


@dataclass(frozen=True)
class Pair:
   name: str
   reference: RunSpec
   degenerate: RunSpec
   transform: StateTransform = StateTransform()
   tolerance: float = 1.0e-10


COMMON = (
   "pyfluid.max_step=100",
   "pyfluid.final_time=0.01",
   "pyfluid.cfl=0.9",
   "pyfluid.scalar_int=1",
   "pyfluid.plot_int=1",
   "pyfluid.checkpoint_int=-1",
   "pyfluid.aux_gfs=none",
   "pyfluid.aux_gf_int=-1",
   "pyfluid.aux_gf_compute_int=-1",
   "amr.max_level=0",
   "amr.max_grid_size=32",
   "amr.blocking_factor=1",
   "pyfluid.subcycling=off",
)


HYDRO_METHOD = (
   "pyfluid.physics=hydro",
   "pyfluid.timestep=ssprk3",
   "pyfluid.recon=pwl",
   "pyfluid.riemann=hllc",
   "pyfluid.ng=2",
   "pyfluid.bc=periodic",
)


MHD_CT_METHOD = (
   "pyfluid.physics=mhd",
   "pyfluid.ct=on",
   "pyfluid.timestep=ssprk3",
   "pyfluid.recon=pwl",
   "pyfluid.riemann=hlld",
   "pyfluid.ng=2",
   "pyfluid.bc=periodic",
   "pyfluid.diffusion=none",
   "pyfluid.viscosity=0.0",
   "pyfluid.resistivity=0.0",
)


SWAP_XY = StateTransform(
   spatial_permutation=(1, 0),
   component_map=(
      ("mom1", "mom2", 1.0),
      ("mom2", "mom1", 1.0),
      ("b1", "b2", 1.0),
      ("b2", "b1", 1.0),
   ),
)

SWAP_XY_3D = StateTransform(
   spatial_permutation=(1, 0, 2),
   component_map=(
      ("mom1", "mom2", 1.0),
      ("mom2", "mom1", 1.0),
      ("b1", "b2", 1.0),
      ("b2", "b1", 1.0),
   ),
)

ACTIVE_Y_TO_X = StateTransform(
   component_map=(
      ("mom1", "mom2", 1.0),
      ("mom2", "mom1", 1.0),
      ("b1", "b2", 1.0),
      ("b2", "b1", 1.0),
   ),
)

ACTIVE_XZ_TO_XY = StateTransform(
   component_map=(
      ("mom2", "mom3", 1.0),
      ("mom3", "mom2", 1.0),
      ("b2", "b3", 1.0),
      ("b3", "b2", 1.0),
   ),
)

ACTIVE_YZ_TO_XY = StateTransform(
   component_map=(
      ("mom1", "mom2", 1.0),
      ("mom2", "mom3", 1.0),
      ("mom3", "mom1", 1.0),
      ("b1", "b2", 1.0),
      ("b2", "b3", 1.0),
      ("b3", "b1", 1.0),
   ),
)

ACTIVE_XZ_TO_XY_3D = StateTransform(
   spatial_permutation=(0, 2, 1),
   component_map=(
      ("mom2", "mom3", 1.0),
      ("mom3", "mom2", 1.0),
      ("b2", "b3", 1.0),
      ("b3", "b2", 1.0),
   ),
)

ACTIVE_YZ_TO_XY_3D = StateTransform(
   spatial_permutation=(1, 2, 0),
   component_map=(
      ("mom1", "mom2", 1.0),
      ("mom2", "mom3", 1.0),
      ("mom3", "mom1", 1.0),
      ("b1", "b2", 1.0),
      ("b2", "b3", 1.0),
      ("b3", "b1", 1.0),
   ),
)

CYCLIC_YZX_TO_XYZ_3D = StateTransform(
   spatial_permutation=(1, 2, 0),
   component_map=(
      ("mom1", "mom2", 1.0),
      ("mom2", "mom3", 1.0),
      ("mom3", "mom1", 1.0),
      ("b1", "b2", 1.0),
      ("b2", "b3", 1.0),
      ("b3", "b1", 1.0),
   ),
)

CYCLIC_ZXY_TO_XYZ_3D = StateTransform(
   spatial_permutation=(2, 0, 1),
   component_map=(
      ("mom1", "mom3", 1.0),
      ("mom2", "mom1", 1.0),
      ("mom3", "mom2", 1.0),
      ("b1", "b3", 1.0),
      ("b2", "b1", 1.0),
      ("b3", "b2", 1.0),
   ),
)

REFLECT_X = StateTransform(
   spatial_reversals=(0,),
   component_map=(
      ("mom1", "mom1", -1.0),
      ("b1", "b1", -1.0),
   ),
)

REFLECT_Z_3D = StateTransform(
   spatial_reversals=(2,),
   component_map=(
      ("mom3", "mom3", -1.0),
      ("b3", "b3", -1.0),
   ),
)

REFLECT_Y = StateTransform(
   spatial_reversals=(1,),
   component_map=(
      ("mom2", "mom2", -1.0),
      ("b2", "b2", -1.0),
   ),
)


HYDRO_PRESSURE_1D = RunSpec(
   "1d",
   1,
   "acoustic_wave_1d.in",
   COMMON
   + HYDRO_METHOD
   + (
      "geometry.prob_lo=0.0",
      "geometry.prob_hi=1.0",
      "geometry.is_periodic=1",
      "amr.n_cell=32",
      "pyfluid.ic=pressure_bump",
      "pyfluid.rho0=1.0",
      "pyfluid.pressure0=1.0",
      "pyfluid.amplitude=1.0e-2",
      "pyfluid.bump_sigma=0.12",
      "pyfluid.bump_x0=0.5",
      "pyfluid.bump_y0=0.5",
      "pyfluid.bump_z0=0.5",
      "pyfluid.v1_0=0.0",
      "pyfluid.v2_0=0.0",
      "pyfluid.v3_0=0.0",
   ),
)

HYDRO_PRESSURE_2D = RunSpec(
   "2d",
   2,
   "pressure_bump_2d_amr.in",
   COMMON
   + HYDRO_METHOD
   + (
      "geometry.prob_lo=0.0 0.0",
      "geometry.prob_hi=1.0 1.0",
      "geometry.is_periodic=1 1",
      "amr.n_cell=16 16",
      "pyfluid.ic=pressure_bump",
      "pyfluid.rho0=1.0",
      "pyfluid.pressure0=1.0",
      "pyfluid.amplitude=1.0e-2",
      "pyfluid.bump_sigma=0.12",
      "pyfluid.bump_x0=0.5",
      "pyfluid.bump_y0=0.5",
      "pyfluid.bump_z0=0.5",
      "pyfluid.v1_0=0.0",
      "pyfluid.v2_0=0.0",
      "pyfluid.v3_0=0.0",
   ),
)

MHD_PRESSURE_2D = RunSpec(
   "2d",
   2,
   "pressure_bump_2d_amr.in",
   COMMON
   + MHD_CT_METHOD
   + (
      "geometry.prob_lo=0.0 0.0",
      "geometry.prob_hi=1.0 1.0",
      "geometry.is_periodic=1 1",
      "amr.n_cell=16 16",
      "pyfluid.ic=pressure_bump",
      "pyfluid.rho0=1.0",
      "pyfluid.pressure0=1.0",
      "pyfluid.amplitude=1.0e-2",
      "pyfluid.bump_sigma=0.12",
      "pyfluid.bump_x0=0.5",
      "pyfluid.bump_y0=0.5",
      "pyfluid.bump_z0=0.5",
      "pyfluid.v1_0=0.0",
      "pyfluid.v2_0=0.0",
      "pyfluid.v3_0=0.0",
      "pyfluid.b1_0=0.1",
      "pyfluid.b2_0=0.05",
      "pyfluid.b3_0=0.2",
   ),
)

HYDRO_DIAGONAL_SOD_METHOD = (
   "pyfluid.physics=hydro",
   "pyfluid.timestep=ssprk3",
   "pyfluid.recon=pwl",
   "pyfluid.riemann=hllc",
   "pyfluid.ng=2",
   "pyfluid.bc=periodic",
)


def diagonal_sod_spec(
   label: str,
   shock_nx: float,
   shock_ny: float,
   shock_x0: float = 0.5,
   shock_y0: float = 0.5,
) -> RunSpec:
   return RunSpec(
      label,
      2,
      "pressure_bump_2d_amr.in",
      COMMON
      + HYDRO_DIAGONAL_SOD_METHOD
      + (
         "geometry.prob_lo=0.0 0.0",
         "geometry.prob_hi=1.0 1.0",
         "geometry.is_periodic=1 1",
         "amr.n_cell=32 32",
         "pyfluid.final_time=0.006",
         "pyfluid.max_step=8",
         "pyfluid.ic=sod_periodic_plane",
         f"pyfluid.shock_x0={shock_x0:.17g}",
         f"pyfluid.shock_y0={shock_y0:.17g}",
         f"pyfluid.shock_nx={shock_nx:.17g}",
         f"pyfluid.shock_ny={shock_ny:.17g}",
         "pyfluid.rho_left=1.0",
         "pyfluid.pressure_left=1.0",
         "pyfluid.v1_left=0.0",
         "pyfluid.v2_left=0.0",
         "pyfluid.v3_left=0.0",
         "pyfluid.rho_right=0.125",
         "pyfluid.pressure_right=0.1",
         "pyfluid.v1_right=0.0",
         "pyfluid.v2_right=0.0",
         "pyfluid.v3_right=0.0",
      ),
   )


def diagonal_sod3d_spec(
   label: str,
   shock_nx: float,
   shock_ny: float,
   shock_nz: float,
   shock_x0: float,
   shock_y0: float,
   shock_z0: float,
) -> RunSpec:
   return RunSpec(
      label,
      3,
      "pressure_bump_3d_amr.in",
      COMMON
      + HYDRO_DIAGONAL_SOD_METHOD
      + (
         "geometry.prob_lo=0.0 0.0 0.0",
         "geometry.prob_hi=1.0 1.0 1.0",
         "geometry.is_periodic=1 1 1",
         "amr.n_cell=12 12 12",
         "pyfluid.final_time=0.004",
         "pyfluid.max_step=4",
         "pyfluid.ic=sod_periodic_plane",
         f"pyfluid.shock_x0={shock_x0:.17g}",
         f"pyfluid.shock_y0={shock_y0:.17g}",
         f"pyfluid.shock_z0={shock_z0:.17g}",
         f"pyfluid.shock_nx={shock_nx:.17g}",
         f"pyfluid.shock_ny={shock_ny:.17g}",
         f"pyfluid.shock_nz={shock_nz:.17g}",
         "pyfluid.rho_left=1.0",
         "pyfluid.pressure_left=1.0",
         "pyfluid.v1_left=0.0",
         "pyfluid.v2_left=0.0",
         "pyfluid.v3_left=0.0",
         "pyfluid.rho_right=0.125",
         "pyfluid.pressure_right=0.1",
         "pyfluid.v1_right=0.0",
         "pyfluid.v2_right=0.0",
         "pyfluid.v3_right=0.0",
      ),
   )


DIAG_X0 = 0.17
DIAG_Y0 = 0.23
DIAG_Z0 = 0.31

HYDRO_DIAGONAL_SOD_PP = diagonal_sod_spec("diag_pp", 1.0, 1.0, DIAG_X0, DIAG_Y0)
HYDRO_DIAGONAL_SOD_SWAP = diagonal_sod_spec("diag_swap", 1.0, 1.0, DIAG_Y0, DIAG_X0)
HYDRO_DIAGONAL_SOD_MP = diagonal_sod_spec("diag_mp", -1.0, 1.0, 1.0 - DIAG_X0, DIAG_Y0)
HYDRO_DIAGONAL_SOD_PM = diagonal_sod_spec("diag_pm", 1.0, -1.0, DIAG_X0, 1.0 - DIAG_Y0)
HYDRO_DIAGONAL_SOD_3D_XY_PP = diagonal_sod3d_spec("diag_xy_pp", 1.0, 1.0, 0.0, DIAG_X0, DIAG_Y0, DIAG_Z0)
HYDRO_DIAGONAL_SOD_3D_XY_SWAP = diagonal_sod3d_spec("diag_xy_swap", 1.0, 1.0, 0.0, DIAG_Y0, DIAG_X0, DIAG_Z0)
HYDRO_DIAGONAL_SOD_3D_XY_MP = diagonal_sod3d_spec("diag_xy_mp", -1.0, 1.0, 0.0, 1.0 - DIAG_X0, DIAG_Y0, DIAG_Z0)
HYDRO_DIAGONAL_SOD_3D_XY_PM = diagonal_sod3d_spec("diag_xy_pm", 1.0, -1.0, 0.0, DIAG_X0, 1.0 - DIAG_Y0, DIAG_Z0)
HYDRO_DIAGONAL_SOD_3D_XZ_PP = diagonal_sod3d_spec("diag_xz_pp", 1.0, 0.0, 1.0, DIAG_X0, DIAG_Z0, DIAG_Y0)
HYDRO_DIAGONAL_SOD_3D_XZ_MP = diagonal_sod3d_spec("diag_xz_mp", -1.0, 0.0, 1.0, 1.0 - DIAG_X0, DIAG_Z0, DIAG_Y0)
HYDRO_DIAGONAL_SOD_3D_XZ_PM = diagonal_sod3d_spec("diag_xz_pm", 1.0, 0.0, -1.0, DIAG_X0, DIAG_Z0, 1.0 - DIAG_Y0)
HYDRO_DIAGONAL_SOD_3D_YZ_PP = diagonal_sod3d_spec("diag_yz_pp", 0.0, 1.0, 1.0, DIAG_Z0, DIAG_X0, DIAG_Y0)
HYDRO_DIAGONAL_SOD_3D_YZ_MP = diagonal_sod3d_spec("diag_yz_mp", 0.0, -1.0, 1.0, DIAG_Z0, 1.0 - DIAG_X0, DIAG_Y0)
HYDRO_DIAGONAL_SOD_3D_YZ_PM = diagonal_sod3d_spec("diag_yz_pm", 0.0, 1.0, -1.0, DIAG_Z0, DIAG_X0, 1.0 - DIAG_Y0)
HYDRO_DIAGONAL_SOD_3D_XYZ_PPP = diagonal_sod3d_spec("diag_xyz_ppp", 1.0, 1.0, 1.0, DIAG_X0, DIAG_Y0, DIAG_Z0)
HYDRO_DIAGONAL_SOD_3D_XYZ_YZX = diagonal_sod3d_spec("diag_xyz_yzx", 1.0, 1.0, 1.0, DIAG_Y0, DIAG_Z0, DIAG_X0)
HYDRO_DIAGONAL_SOD_3D_XYZ_ZXY = diagonal_sod3d_spec("diag_xyz_zxy", 1.0, 1.0, 1.0, DIAG_Z0, DIAG_X0, DIAG_Y0)
HYDRO_DIAGONAL_SOD_3D_XYZ_MPP = diagonal_sod3d_spec("diag_xyz_mpp", -1.0, 1.0, 1.0, 1.0 - DIAG_X0, DIAG_Y0, DIAG_Z0)
HYDRO_DIAGONAL_SOD_3D_XYZ_PMP = diagonal_sod3d_spec("diag_xyz_pmp", 1.0, -1.0, 1.0, DIAG_X0, 1.0 - DIAG_Y0, DIAG_Z0)
HYDRO_DIAGONAL_SOD_3D_XYZ_PPM = diagonal_sod3d_spec("diag_xyz_ppm", 1.0, 1.0, -1.0, DIAG_X0, DIAG_Y0, 1.0 - DIAG_Z0)


PAIRS = (
   Pair(
      "hydro_acoustic_1d_vs_2d_flat",
      RunSpec(
         "1d",
         1,
         "acoustic_wave_1d.in",
         COMMON
         + HYDRO_METHOD
         + (
            "geometry.prob_lo=0.0",
            "geometry.prob_hi=1.0",
            "geometry.is_periodic=1",
            "amr.n_cell=32",
            "pyfluid.ic=acoustic_wave_x",
            "pyfluid.wave_number=1.0",
            "pyfluid.rho0=1.0",
            "pyfluid.pressure0=1.0",
            "pyfluid.amplitude=1.0e-4",
            "pyfluid.v1_0=0.0",
            "pyfluid.v2_0=0.0",
            "pyfluid.v3_0=0.0",
         ),
      ),
      RunSpec(
         "2d_nx1",
         2,
         "hydro_sine_2d.in",
         COMMON
         + HYDRO_METHOD
         + (
            "geometry.prob_lo=0.0 0.0",
            "geometry.prob_hi=1.0 1.0",
            "geometry.is_periodic=1 1",
            "amr.n_cell=32 1",
            "pyfluid.ic=acoustic_wave_x",
            "pyfluid.wave_number=1.0",
            "pyfluid.rho0=1.0",
            "pyfluid.pressure0=1.0",
            "pyfluid.amplitude=1.0e-4",
            "pyfluid.v1_0=0.0",
            "pyfluid.v2_0=0.0",
            "pyfluid.v3_0=0.0",
         ),
      ),
   ),
   Pair(
      "hydro_pressure_bump_1d_vs_2d_nx1",
      HYDRO_PRESSURE_1D,
      RunSpec(
         "2d_nx1",
         2,
         "hydro_sine_2d.in",
         COMMON
         + HYDRO_METHOD
         + (
            "geometry.prob_lo=0.0 0.0",
            "geometry.prob_hi=1.0 1.0",
            "geometry.is_periodic=1 1",
            "amr.n_cell=32 1",
            "pyfluid.ic=pressure_bump",
            "pyfluid.rho0=1.0",
            "pyfluid.pressure0=1.0",
            "pyfluid.amplitude=1.0e-2",
            "pyfluid.bump_sigma=0.12",
            "pyfluid.bump_x0=0.5",
            "pyfluid.bump_y0=0.5",
            "pyfluid.bump_z0=0.5",
            "pyfluid.v1_0=0.0",
            "pyfluid.v2_0=0.0",
            "pyfluid.v3_0=0.0",
         ),
      ),
   ),
   Pair(
      "hydro_pressure_bump_1d_vs_2d_1xn",
      HYDRO_PRESSURE_1D,
      RunSpec(
         "2d_1xn",
         2,
         "hydro_sine_2d.in",
         COMMON
         + HYDRO_METHOD
         + (
            "geometry.prob_lo=0.0 0.0",
            "geometry.prob_hi=1.0 1.0",
            "geometry.is_periodic=1 1",
            "amr.n_cell=1 32",
            "pyfluid.ic=pressure_bump",
            "pyfluid.rho0=1.0",
            "pyfluid.pressure0=1.0",
            "pyfluid.amplitude=1.0e-2",
            "pyfluid.bump_sigma=0.12",
            "pyfluid.bump_x0=0.5",
            "pyfluid.bump_y0=0.5",
            "pyfluid.bump_z0=0.5",
            "pyfluid.v1_0=0.0",
            "pyfluid.v2_0=0.0",
            "pyfluid.v3_0=0.0",
         ),
      ),
      ACTIVE_Y_TO_X,
   ),
   Pair(
      "hydro_pressure_bump_2d_vs_3d_flat",
      HYDRO_PRESSURE_2D,
      RunSpec(
         "3d_nxnx1",
         3,
         "pressure_bump_3d_amr.in",
         COMMON
         + HYDRO_METHOD
         + (
            "geometry.prob_lo=0.0 0.0 0.0",
            "geometry.prob_hi=1.0 1.0 1.0",
            "geometry.is_periodic=1 1 1",
            "amr.n_cell=16 16 1",
            "pyfluid.ic=pressure_bump",
            "pyfluid.rho0=1.0",
            "pyfluid.pressure0=1.0",
            "pyfluid.amplitude=1.0e-2",
            "pyfluid.bump_sigma=0.12",
            "pyfluid.bump_x0=0.5",
            "pyfluid.bump_y0=0.5",
            "pyfluid.bump_z0=0.5",
            "pyfluid.v1_0=0.0",
            "pyfluid.v2_0=0.0",
            "pyfluid.v3_0=0.0",
         ),
      ),
   ),
   Pair(
      "hydro_pressure_bump_2d_vs_3d_nx1xn",
      HYDRO_PRESSURE_2D,
      RunSpec(
         "3d_nx1xn",
         3,
         "pressure_bump_3d_amr.in",
         COMMON
         + HYDRO_METHOD
         + (
            "geometry.prob_lo=0.0 0.0 0.0",
            "geometry.prob_hi=1.0 1.0 1.0",
            "geometry.is_periodic=1 1 1",
            "amr.n_cell=16 1 16",
            "pyfluid.ic=pressure_bump",
            "pyfluid.rho0=1.0",
            "pyfluid.pressure0=1.0",
            "pyfluid.amplitude=1.0e-2",
            "pyfluid.bump_sigma=0.12",
            "pyfluid.bump_x0=0.5",
            "pyfluid.bump_y0=0.5",
            "pyfluid.bump_z0=0.5",
            "pyfluid.v1_0=0.0",
            "pyfluid.v2_0=0.0",
            "pyfluid.v3_0=0.0",
         ),
      ),
      ACTIVE_XZ_TO_XY,
   ),
   Pair(
      "hydro_pressure_bump_2d_vs_3d_1xnxn",
      HYDRO_PRESSURE_2D,
      RunSpec(
         "3d_1xnxn",
         3,
         "pressure_bump_3d_amr.in",
         COMMON
         + HYDRO_METHOD
         + (
            "geometry.prob_lo=0.0 0.0 0.0",
            "geometry.prob_hi=1.0 1.0 1.0",
            "geometry.is_periodic=1 1 1",
            "amr.n_cell=1 16 16",
            "pyfluid.ic=pressure_bump",
            "pyfluid.rho0=1.0",
            "pyfluid.pressure0=1.0",
            "pyfluid.amplitude=1.0e-2",
            "pyfluid.bump_sigma=0.12",
            "pyfluid.bump_x0=0.5",
            "pyfluid.bump_y0=0.5",
            "pyfluid.bump_z0=0.5",
            "pyfluid.v1_0=0.0",
            "pyfluid.v2_0=0.0",
            "pyfluid.v3_0=0.0",
         ),
      ),
      ACTIVE_YZ_TO_XY,
   ),
   Pair(
      "hydro_pressure_bump_2d_xy_swap",
      HYDRO_PRESSURE_2D,
      HYDRO_PRESSURE_2D,
      SWAP_XY,
   ),
   Pair(
      "hydro_diagonal_sod_xy_swap",
      HYDRO_DIAGONAL_SOD_PP,
      HYDRO_DIAGONAL_SOD_SWAP,
      SWAP_XY,
   ),
   Pair(
      "hydro_diagonal_sod_reflect_x",
      HYDRO_DIAGONAL_SOD_PP,
      HYDRO_DIAGONAL_SOD_MP,
      REFLECT_X,
   ),
   Pair(
      "hydro_diagonal_sod_reflect_y",
      HYDRO_DIAGONAL_SOD_PP,
      HYDRO_DIAGONAL_SOD_PM,
      REFLECT_Y,
   ),
   Pair(
      "hydro_diagonal_sod_3d_xy_swap",
      HYDRO_DIAGONAL_SOD_3D_XY_PP,
      HYDRO_DIAGONAL_SOD_3D_XY_SWAP,
      SWAP_XY_3D,
   ),
   Pair(
      "hydro_diagonal_sod_3d_xy_reflect_x",
      HYDRO_DIAGONAL_SOD_3D_XY_PP,
      HYDRO_DIAGONAL_SOD_3D_XY_MP,
      REFLECT_X,
   ),
   Pair(
      "hydro_diagonal_sod_3d_xy_reflect_y",
      HYDRO_DIAGONAL_SOD_3D_XY_PP,
      HYDRO_DIAGONAL_SOD_3D_XY_PM,
      REFLECT_Y,
   ),
   Pair(
      "hydro_diagonal_sod_3d_xz_vs_xy",
      HYDRO_DIAGONAL_SOD_3D_XY_PP,
      HYDRO_DIAGONAL_SOD_3D_XZ_PP,
      ACTIVE_XZ_TO_XY_3D,
   ),
   Pair(
      "hydro_diagonal_sod_3d_xz_reflect_x",
      HYDRO_DIAGONAL_SOD_3D_XZ_PP,
      HYDRO_DIAGONAL_SOD_3D_XZ_MP,
      REFLECT_X,
   ),
   Pair(
      "hydro_diagonal_sod_3d_xz_reflect_z",
      HYDRO_DIAGONAL_SOD_3D_XZ_PP,
      HYDRO_DIAGONAL_SOD_3D_XZ_PM,
      REFLECT_Z_3D,
   ),
   Pair(
      "hydro_diagonal_sod_3d_yz_vs_xy",
      HYDRO_DIAGONAL_SOD_3D_XY_PP,
      HYDRO_DIAGONAL_SOD_3D_YZ_PP,
      ACTIVE_YZ_TO_XY_3D,
   ),
   Pair(
      "hydro_diagonal_sod_3d_yz_reflect_y",
      HYDRO_DIAGONAL_SOD_3D_YZ_PP,
      HYDRO_DIAGONAL_SOD_3D_YZ_MP,
      REFLECT_Y,
   ),
   Pair(
      "hydro_diagonal_sod_3d_yz_reflect_z",
      HYDRO_DIAGONAL_SOD_3D_YZ_PP,
      HYDRO_DIAGONAL_SOD_3D_YZ_PM,
      REFLECT_Z_3D,
   ),
   Pair(
      "hydro_diagonal_sod_3d_xyz_cycle_yzx",
      HYDRO_DIAGONAL_SOD_3D_XYZ_PPP,
      HYDRO_DIAGONAL_SOD_3D_XYZ_YZX,
      CYCLIC_YZX_TO_XYZ_3D,
   ),
   Pair(
      "hydro_diagonal_sod_3d_xyz_cycle_zxy",
      HYDRO_DIAGONAL_SOD_3D_XYZ_PPP,
      HYDRO_DIAGONAL_SOD_3D_XYZ_ZXY,
      CYCLIC_ZXY_TO_XYZ_3D,
   ),
   Pair(
      "hydro_diagonal_sod_3d_xyz_reflect_x",
      HYDRO_DIAGONAL_SOD_3D_XYZ_PPP,
      HYDRO_DIAGONAL_SOD_3D_XYZ_MPP,
      REFLECT_X,
   ),
   Pair(
      "hydro_diagonal_sod_3d_xyz_reflect_y",
      HYDRO_DIAGONAL_SOD_3D_XYZ_PPP,
      HYDRO_DIAGONAL_SOD_3D_XYZ_PMP,
      REFLECT_Y,
   ),
   Pair(
      "hydro_diagonal_sod_3d_xyz_reflect_z",
      HYDRO_DIAGONAL_SOD_3D_XYZ_PPP,
      HYDRO_DIAGONAL_SOD_3D_XYZ_PPM,
      REFLECT_Z_3D,
   ),
   Pair(
      "mhd_ct_pressure_bump_2d_vs_3d_flat",
      MHD_PRESSURE_2D,
      RunSpec(
         "3d_nxnx1",
         3,
         "pressure_bump_3d_amr.in",
         COMMON
         + MHD_CT_METHOD
         + (
            "geometry.prob_lo=0.0 0.0 0.0",
            "geometry.prob_hi=1.0 1.0 1.0",
            "geometry.is_periodic=1 1 1",
            "amr.n_cell=16 16 1",
            "pyfluid.ic=pressure_bump",
            "pyfluid.rho0=1.0",
            "pyfluid.pressure0=1.0",
            "pyfluid.amplitude=1.0e-2",
            "pyfluid.bump_sigma=0.12",
            "pyfluid.bump_x0=0.5",
            "pyfluid.bump_y0=0.5",
            "pyfluid.bump_z0=0.5",
            "pyfluid.v1_0=0.0",
            "pyfluid.v2_0=0.0",
            "pyfluid.v3_0=0.0",
            "pyfluid.b1_0=0.1",
            "pyfluid.b2_0=0.05",
            "pyfluid.b3_0=0.2",
         ),
      ),
   ),
   Pair(
      "mhd_ct_pressure_bump_2d_vs_3d_nx1xn",
      MHD_PRESSURE_2D,
      RunSpec(
         "3d_nx1xn",
         3,
         "pressure_bump_3d_amr.in",
         COMMON
         + MHD_CT_METHOD
         + (
            "geometry.prob_lo=0.0 0.0 0.0",
            "geometry.prob_hi=1.0 1.0 1.0",
            "geometry.is_periodic=1 1 1",
            "amr.n_cell=16 1 16",
            "pyfluid.ic=pressure_bump",
            "pyfluid.rho0=1.0",
            "pyfluid.pressure0=1.0",
            "pyfluid.amplitude=1.0e-2",
            "pyfluid.bump_sigma=0.12",
            "pyfluid.bump_x0=0.5",
            "pyfluid.bump_y0=0.5",
            "pyfluid.bump_z0=0.5",
            "pyfluid.v1_0=0.0",
            "pyfluid.v2_0=0.0",
            "pyfluid.v3_0=0.0",
            "pyfluid.b1_0=0.1",
            "pyfluid.b2_0=0.2",
            "pyfluid.b3_0=0.05",
         ),
      ),
      ACTIVE_XZ_TO_XY,
   ),
   Pair(
      "mhd_ct_pressure_bump_2d_vs_3d_1xnxn",
      MHD_PRESSURE_2D,
      RunSpec(
         "3d_1xnxn",
         3,
         "pressure_bump_3d_amr.in",
         COMMON
         + MHD_CT_METHOD
         + (
            "geometry.prob_lo=0.0 0.0 0.0",
            "geometry.prob_hi=1.0 1.0 1.0",
            "geometry.is_periodic=1 1 1",
            "amr.n_cell=1 16 16",
            "pyfluid.ic=pressure_bump",
            "pyfluid.rho0=1.0",
            "pyfluid.pressure0=1.0",
            "pyfluid.amplitude=1.0e-2",
            "pyfluid.bump_sigma=0.12",
            "pyfluid.bump_x0=0.5",
            "pyfluid.bump_y0=0.5",
            "pyfluid.bump_z0=0.5",
            "pyfluid.v1_0=0.0",
            "pyfluid.v2_0=0.0",
            "pyfluid.v3_0=0.0",
            "pyfluid.b1_0=0.2",
            "pyfluid.b2_0=0.1",
            "pyfluid.b3_0=0.05",
         ),
      ),
      ACTIVE_YZ_TO_XY,
   ),
)


def executable(dim: int) -> Path:
   return BUILD / f"pyfluid_amrex_{dim}d"


def write_input(pair: Pair, spec: RunSpec) -> tuple[Path, Path]:
   run_dir = RUN_ROOT / pair.name / spec.label
   input_dir = INPUT_ROOT / pair.name
   input_dir.mkdir(parents=True, exist_ok=True)
   run_dir.mkdir(parents=True, exist_ok=True)
   lines = (INPUTS / spec.base_input).read_text().splitlines()
   lines.append("")
   lines.append("# validate_dimensional_consistency.py overrides")
   lines.append(f"pyfluid.run_name={pair.name}_{spec.label}")
   lines.append(f"pyfluid.output_dir={repo_relative(run_dir)}")
   lines.extend(spec.overrides)
   path = input_dir / f"{spec.label}.in"
   path.write_text("\n".join(lines) + "\n")
   return path, run_dir


def run_spec(pair: Pair, spec: RunSpec, ranks: int) -> tuple[int, Path, str]:
   input_path, run_dir = write_input(pair, spec)
   cmd = [*mpi_prefix(ranks), str(executable(spec.dim)), str(input_path)]
   proc = subprocess.run(
      cmd,
      cwd=ROOT,
      text=True,
      stdout=subprocess.PIPE,
      stderr=subprocess.STDOUT,
      env={**os.environ, "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "1")},
      check=False,
   )
   (run_dir / "run.log").write_text(proc.stdout)
   return proc.returncode, run_dir, "\n".join(proc.stdout.splitlines()[-8:])


def read_squeezed_state(plotfile: Path) -> tuple[list[str], float, np.ndarray]:
   names, ndim, time_value = parse_plot_header(plotfile)
   state = read_level_state(plotfile, names, ndim, 0)
   singleton_axes = tuple(axis for axis in range(1, state.ndim) if state.shape[axis] == 1)
   if singleton_axes:
      state = np.squeeze(state, axis=singleton_axes)
   return names, time_value, state


def transform_state(
   source_names: list[str],
   source_state: np.ndarray,
   target_names: list[str],
   transform: StateTransform,
) -> np.ndarray:
   state = source_state
   spatial_ndim = state.ndim - 1
   if transform.spatial_permutation is not None:
      if len(transform.spatial_permutation) != spatial_ndim:
         raise ValueError(
            f"spatial permutation {transform.spatial_permutation} does not match state rank {spatial_ndim}"
         )
      state = np.transpose(state, (0,) + tuple(axis + 1 for axis in transform.spatial_permutation))
   for axis in transform.spatial_reversals:
      if axis < 0 or axis >= spatial_ndim:
         raise ValueError(f"cannot reverse spatial axis {axis} for state rank {spatial_ndim}")
      state = np.flip(state, axis=axis + 1)

   source_by_target = {
      target: (source, sign)
      for target, source, sign in transform.component_map
      if target in target_names or source in source_names
   }
   source_index = {name: idx for idx, name in enumerate(source_names)}
   transformed_components = []
   for target in target_names:
      source, sign = source_by_target.get(target, (target, 1.0))
      if source not in source_index:
         raise ValueError(f"component {source!r} needed for target {target!r} is missing from {source_names}")
      transformed_components.append(sign * state[source_index[source]])
   return np.stack(transformed_components, axis=0)


def compare_pair(pair: Pair, ranks: int) -> dict[str, object]:
   ref_code, ref_dir, ref_tail = run_spec(pair, pair.reference, ranks)
   deg_code, deg_dir, deg_tail = run_spec(pair, pair.degenerate, ranks)
   row: dict[str, object] = {
      "case": pair.name,
      "status": "PASS",
      "reason": "",
      "ranks": ranks,
      "reference": pair.reference.label,
      "degenerate": pair.degenerate.label,
      "ref_run_dir": repo_relative(ref_dir),
      "deg_run_dir": repo_relative(deg_dir),
      "max_abs_error": math.nan,
      "max_rel_error": math.nan,
      "worst_component": "",
      "time_ref": math.nan,
      "time_degenerate": math.nan,
   }
   failures: list[str] = []
   if ref_code != 0:
      failures.append(f"{pair.reference.label} returncode={ref_code}: {ref_tail}")
   if deg_code != 0:
      failures.append(f"{pair.degenerate.label} returncode={deg_code}: {deg_tail}")
   if failures:
      row["status"] = "FAIL"
      row["reason"] = " | ".join(failures)
      return row

   ref_names, ref_time, ref_state = read_squeezed_state(latest_plotfile(ref_dir, require=True))
   deg_names, deg_time, deg_state = read_squeezed_state(latest_plotfile(deg_dir, require=True))
   row["time_ref"] = ref_time
   row["time_degenerate"] = deg_time
   try:
      deg_state = transform_state(deg_names, deg_state, ref_names, pair.transform)
   except ValueError as exc:
      failures.append(str(exc))
   if ref_state.shape != deg_state.shape:
      failures.append(f"state shapes differ after squeeze: {ref_state.shape} != {deg_state.shape}")
   if abs(ref_time - deg_time) > 1.0e-14:
      failures.append(f"times differ: {ref_time:.17g} != {deg_time:.17g}")
   if not failures:
      delta = np.abs(ref_state - deg_state)
      scale = np.maximum(np.abs(ref_state), np.abs(deg_state))
      rel = delta / np.maximum(scale, 1.0e-300)
      component_errors = np.max(delta.reshape(delta.shape[0], -1), axis=1)
      worst_index = int(np.argmax(component_errors))
      max_abs = float(component_errors[worst_index])
      max_rel = float(np.max(rel))
      row["max_abs_error"] = max_abs
      row["max_rel_error"] = max_rel
      row["worst_component"] = ref_names[worst_index]
      if not math.isfinite(max_abs) or max_abs > pair.tolerance:
         failures.append(f"max_abs_error={max_abs:.3e} > {pair.tolerance:.3e}")

   row["status"] = "PASS" if not failures else "FAIL"
   row["reason"] = "; ".join(failures)
   return row


def write_results(rows: list[dict[str, object]], path: Path) -> None:
   fields = [
      "case",
      "status",
      "reason",
      "ranks",
      "reference",
      "degenerate",
      "time_ref",
      "time_degenerate",
      "max_abs_error",
      "max_rel_error",
      "worst_component",
      "ref_run_dir",
      "deg_run_dir",
   ]
   path.parent.mkdir(parents=True, exist_ok=True)
   with path.open("w", newline="") as f:
      writer = csv.DictWriter(f, delimiter="\t", fieldnames=fields, extrasaction="ignore")
      writer.writeheader()
      writer.writerows(rows)


def append_history(rows: list[dict[str, object]]) -> None:
   append_validation_history(
      HISTORY,
      [
         {
            "kind": "dimensional_consistency",
            "case": row["case"],
            "label": f"{row['reference']}_vs_{row['degenerate']}",
            "status": row["status"],
            "reason": row["reason"],
            "ranks": row["ranks"],
            "time": row["time_ref"],
            "description": (
               f"max_abs={float(row['max_abs_error']):.3e}; "
               f"max_rel={float(row['max_rel_error']):.3e}; "
               f"worst={row['worst_component']}"
            ),
         }
         for row in rows
      ],
   )


def main() -> int:
   parser = argparse.ArgumentParser(description=__doc__)
   parser.add_argument("--force", action="store_true")
   parser.add_argument("--ranks", type=int, default=3)
   parser.add_argument("--only", action="append", choices=[pair.name for pair in PAIRS])
   parser.add_argument("--list-cases", action="store_true")
   args = parser.parse_args()

   selected = [pair for pair in PAIRS if not args.only or pair.name in set(args.only)]
   if args.list_cases:
      for pair in selected:
         print(pair.name)
      return 0

   if args.force and RUN_ROOT.exists():
      shutil.rmtree(RUN_ROOT)
   RUN_ROOT.mkdir(parents=True, exist_ok=True)

   rows = []
   for pair in selected:
      row = compare_pair(pair, args.ranks)
      rows.append(row)
      print(
         f"{row['status']:>4} {row['case']}: "
         f"max_abs={float(row['max_abs_error']):.3e} "
         f"max_rel={float(row['max_rel_error']):.3e} "
         f"worst={row['worst_component']}"
      )
      if row["status"] != "PASS":
         print(f"     {row['reason']}")
   write_results(rows, RESULTS)
   append_history(rows)
   print(f"wrote {repo_relative(RESULTS)}")
   failed = [row for row in rows if row["status"] != "PASS"]
   return 1 if failed else 0


if __name__ == "__main__":
   raise SystemExit(main())
