#!/usr/bin/env python3
"""Run compact AMR validation cases and regenerate their diagnostic plots."""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plotfile_reader import checkpoint_path, latest_plotfile as discover_latest_plotfile  # noqa: E402
from validation_common import (  # noqa: E402
    build_cached_target,
    check_scalars,
    materialize_run_input,
    read_scalar_rows,
    run_mpi,
)


ROOT = Path(__file__).resolve().parents[2]
SOLVER_2D = ROOT / "build/pyfluid_amrex/pyfluid_amrex_2d"
CHECKPOINT_COMPARE_2D = ROOT / "build/pyfluid_amrex/pyfluid_amrex_compare_checkpoint_2d"
SUMMARY = ROOT / "runs/current/amr_validation_summary.tsv"


@dataclass(frozen=True)
class Case:
    name: str
    suite: str
    input_file: Path
    output_dir: Path
    plot_root: Path
    title: str
    overrides: dict[str, str]
    hierarchy: bool = True
    scalars: bool = True
    require_metric_stop: bool = False


@dataclass(frozen=True)
class Comparison:
    name: str
    suite: str
    reference: str
    candidate: str
    components: tuple[str, ...]
    title: str
    rel_l2_tolerance: float = 1.0e-3
    rel_linf_tolerance: float = 1.0e-2
    enforce_tolerance: bool = True


@dataclass(frozen=True)
class ImprovementCheck:
    name: str
    suite: str
    baseline_comparison: str
    improved_comparison: str
    component: str
    metric: str
    min_factor: float
    title: str


@dataclass(frozen=True)
class RestartCheck:
    name: str
    suite: str
    input_file: Path
    full_output_dir: Path
    restart_output_dir: Path
    checkpoint_step: int
    final_step: int
    tolerance: str
    overrides: dict[str, str]


PRESSURE_INPUT = ROOT / "pyfluid_amrex/inputs/pressure_bump_2d_amr.in"
KH_INPUT = ROOT / "pyfluid_amrex/inputs/kelvin_helmholtz_mhd_2d_movie_ct.in"
HYDRO_SINE_INPUT = ROOT / "pyfluid_amrex/inputs/hydro_sine_2d.in"
RESISTIVE_WAVE_INPUT = ROOT / "pyfluid_amrex/inputs/resistive_wave_x_2d_ct.in"
PRESSURE_PLOTS = ROOT / "plots/current/IC/pressure_bump"
KH_PLOTS = ROOT / "plots/current/IC/kelvin_helmholtz_mhd"

PRESSURE_AUTO_BASE = {
    "amr.max_level": "12",
    "amr.n_cell": "32 32",
    "amr.ref_ratio": "2 2",
    "amr.regrid_int": "2",
    "amr.n_error_buf": "8 8",
    "amr.min_feature_length": "0.000001",
    "amr.total_cell_budget": "2000000",
    "amr.initial_refine": "none",
    "amr.initial_refine_levels": "0",
    "amr.initial_refine_fixed_levels": "0",
    "amr.state_interp": "quartic-conservative",
    "pyfluid.final_time": "0.02",
    "pyfluid.max_step": "100000",
    "pyfluid.ng": "3",
    "pyfluid.recon": "wenoz",
    "pyfluid.riemann": "hllc",
    "pyfluid.timestep": "ssprk3",
    "pyfluid.plot_int": "100000",
    "pyfluid.scalar_int": "5",
    "pyfluid.checkpoint_int": "-1",
}

CASES = [
    Case(
        name="pressure_uniform64",
        suite="pressure",
        input_file=PRESSURE_INPUT,
        output_dir=ROOT / "runs/current/pressure_bump2d_uniform64_t002_001",
        plot_root=PRESSURE_PLOTS,
        title="Pressure bump 2D uniform64, WENO-Z/HLLC/SSPRK3, CFL=0.9",
        hierarchy=False,
        overrides={
            "amr.max_level": "0",
            "amr.n_cell": "64 64",
            "pyfluid.final_time": "0.02",
            "pyfluid.max_step": "100000",
            "pyfluid.ng": "3",
            "pyfluid.recon": "wenoz",
            "pyfluid.riemann": "hllc",
            "pyfluid.timestep": "ssprk3",
            "pyfluid.plot_int": "100000",
            "pyfluid.scalar_int": "5",
            "pyfluid.checkpoint_int": "-1",
        },
    ),
    Case(
        name="pressure_uniform256",
        suite="pressure",
        input_file=PRESSURE_INPUT,
        output_dir=ROOT / "runs/current/pressure_bump2d_uniform256_t002_001",
        plot_root=PRESSURE_PLOTS,
        title="Pressure bump 2D uniform256, WENO-Z/HLLC/SSPRK3, CFL=0.9",
        hierarchy=False,
        overrides={
            "amr.max_level": "0",
            "amr.n_cell": "256 256",
            "pyfluid.final_time": "0.02",
            "pyfluid.max_step": "100000",
            "pyfluid.ng": "3",
            "pyfluid.recon": "wenoz",
            "pyfluid.riemann": "hllc",
            "pyfluid.timestep": "ssprk3",
            "pyfluid.plot_int": "100000",
            "pyfluid.scalar_int": "5",
            "pyfluid.checkpoint_int": "-1",
        },
    ),
    Case(
        name="pressure_uniform512",
        suite="pressure",
        input_file=PRESSURE_INPUT,
        output_dir=ROOT / "runs/current/pressure_bump2d_uniform512_t002_001",
        plot_root=PRESSURE_PLOTS,
        title="Pressure bump 2D uniform512 reference, WENO-Z/HLLC/SSPRK3, CFL=0.9",
        hierarchy=False,
        overrides={
            "amr.max_level": "0",
            "amr.n_cell": "512 512",
            "pyfluid.final_time": "0.02",
            "pyfluid.max_step": "100000",
            "pyfluid.ng": "3",
            "pyfluid.recon": "wenoz",
            "pyfluid.riemann": "hllc",
            "pyfluid.timestep": "ssprk3",
            "pyfluid.plot_int": "100000",
            "pyfluid.scalar_int": "8",
            "pyfluid.checkpoint_int": "-1",
        },
    ),
    Case(
        name="pressure_amr_auto_loose",
        suite="pressure",
        input_file=PRESSURE_INPUT,
        output_dir=ROOT / "runs/current/pressure_bump2d_amr_auto_loose_001",
        plot_root=PRESSURE_PLOTS,
        title="Pressure bump 2D automatic AMR, threshold=0.16, WENO-Z/HLLC/SSPRK3, CFL=0.9",
        overrides={
            **PRESSURE_AUTO_BASE,
            "amr.tag_threshold": "0.16",
            "pyfluid.subcycling": "on",
        },
        require_metric_stop=True,
    ),
    Case(
        name="pressure_amr_auto_medium",
        suite="pressure",
        input_file=PRESSURE_INPUT,
        output_dir=ROOT / "runs/current/pressure_bump2d_amr_auto_medium_001",
        plot_root=PRESSURE_PLOTS,
        title="Pressure bump 2D automatic AMR, threshold=0.08, WENO-Z/HLLC/SSPRK3, CFL=0.9",
        overrides={
            **PRESSURE_AUTO_BASE,
            "amr.tag_threshold": "0.08",
            "pyfluid.subcycling": "on",
        },
        require_metric_stop=True,
    ),
    Case(
        name="pressure_amr_auto_tight",
        suite="pressure",
        input_file=PRESSURE_INPUT,
        output_dir=ROOT / "runs/current/pressure_bump2d_amr_auto_tight_001",
        plot_root=PRESSURE_PLOTS,
        title="Pressure bump 2D automatic AMR, threshold=0.02, WENO-Z/HLLC/SSPRK3, CFL=0.9",
        overrides={
            **PRESSURE_AUTO_BASE,
            "amr.tag_threshold": "0.02",
            "pyfluid.subcycling": "on",
        },
        require_metric_stop=True,
    ),
    Case(
        name="pressure_amr_nosub",
        suite="pressure",
        input_file=PRESSURE_INPUT,
        output_dir=ROOT / "runs/current/pressure_bump2d_amr_auto_medium_nosub_001",
        plot_root=PRESSURE_PLOTS,
        title="Pressure bump 2D automatic AMR/no-subcycling, threshold=0.08, WENO-Z/HLLC/SSPRK3, CFL=0.9",
        overrides={
            **PRESSURE_AUTO_BASE,
            "amr.tag_threshold": "0.08",
            "pyfluid.subcycling": "off",
        },
        require_metric_stop=True,
    ),
    Case(
        name="kh_uniform512",
        suite="kh",
        input_file=KH_INPUT,
        output_dir=ROOT / "runs/current/kh2d_uniform512_t016_001",
        plot_root=KH_PLOTS,
        title="KH MHD 2D uniform512 reference, WENO-Z/HLLD/SSPRK3/CT, explicit eta=nu=1.5e-4, CFL=0.9",
        hierarchy=False,
        overrides={
            "amr.max_level": "0",
            "amr.n_cell": "512 512",
            "pyfluid.final_time": "0.16",
            "pyfluid.max_step": "100000",
            "pyfluid.plot_int": "100000",
            "pyfluid.scalar_int": "32",
            "pyfluid.aux_gfs": "none",
            "pyfluid.aux_gf_int": "-1",
            "pyfluid.aux_gf_compute_int": "-1",
            "pyfluid.checkpoint_int": "-1",
        },
    ),
    Case(
        name="kh_fixedsheet",
        suite="kh",
        input_file=KH_INPUT,
        output_dir=ROOT / "runs/current/kh2d_movie_input_fixedsheet_t016_001",
        plot_root=KH_PLOTS,
        title="KH MHD 2D fixed-sheet AMR, WENO-Z/HLLD/SSPRK3/CT, explicit eta=nu=1.5e-4, CFL=0.9",
        overrides={
            "pyfluid.final_time": "0.16",
            "pyfluid.max_step": "100000",
            "pyfluid.plot_int": "100000",
            "pyfluid.scalar_int": "16",
            "pyfluid.aux_gfs": "none",
            "pyfluid.aux_gf_int": "-1",
            "pyfluid.aux_gf_compute_int": "-1",
            "pyfluid.checkpoint_int": "-1",
            "amr.state_interp": "quartic",
            "amr.tag_threshold": "0.10",
            "amr.n_error_buf": "8 8",
            "amr.initial_refine_fixed_levels": "3",
        },
    ),
    Case(
        name="kh_fullauto_buf16",
        suite="kh",
        input_file=KH_INPUT,
        output_dir=ROOT / "runs/current/kh2d_fullamr_quartic_thr003_buf16_t016_001",
        plot_root=KH_PLOTS,
        title="KH MHD 2D full-auto AMR, threshold=0.03, buffer=16, WENO-Z/HLLD/SSPRK3/CT, explicit eta=nu=1.5e-4, CFL=0.9",
        overrides={
            "pyfluid.final_time": "0.16",
            "pyfluid.max_step": "100000",
            "pyfluid.plot_int": "100000",
            "pyfluid.scalar_int": "16",
            "pyfluid.aux_gfs": "none",
            "pyfluid.aux_gf_int": "-1",
            "pyfluid.aux_gf_compute_int": "-1",
            "pyfluid.checkpoint_int": "-1",
            "amr.state_interp": "quartic",
            "amr.tag_threshold": "0.03",
            "amr.n_error_buf": "16 16",
            "amr.initial_refine_fixed_levels": "0",
        },
    ),
]

COMPARISONS = [
    Comparison(
        name="pressure_uniform64_vs_uniform256",
        suite="pressure",
        reference="pressure_uniform256",
        candidate="pressure_uniform64",
        components=("pressure", "rho"),
        title="Pressure bump 2D uniform64 vs uniform256, WENO-Z/HLLC/SSPRK3, CFL=0.9",
        rel_l2_tolerance=5.0e-2,
        rel_linf_tolerance=5.0e-1,
    ),
    Comparison(
        name="pressure_uniform64_vs_uniform512",
        suite="pressure",
        reference="pressure_uniform512",
        candidate="pressure_uniform64",
        components=("pressure", "rho"),
        title="Pressure bump 2D uniform64 vs uniform512 reference, WENO-Z/HLLC/SSPRK3, CFL=0.9",
        rel_l2_tolerance=5.0e-2,
        rel_linf_tolerance=5.0e-1,
    ),
    Comparison(
        name="pressure_amr_auto_loose_vs_uniform512",
        suite="pressure",
        reference="pressure_uniform512",
        candidate="pressure_amr_auto_loose",
        components=("pressure", "rho"),
        title="Pressure bump 2D automatic AMR threshold=0.16 vs uniform512 reference",
        rel_l2_tolerance=5.0e-3,
        rel_linf_tolerance=5.0e-2,
        enforce_tolerance=False,
    ),
    Comparison(
        name="pressure_amr_auto_medium_vs_uniform512",
        suite="pressure",
        reference="pressure_uniform512",
        candidate="pressure_amr_auto_medium",
        components=("pressure", "rho"),
        title="Pressure bump 2D automatic AMR threshold=0.08 vs uniform512 reference",
        rel_l2_tolerance=2.0e-3,
        rel_linf_tolerance=2.0e-2,
        enforce_tolerance=False,
    ),
    Comparison(
        name="pressure_amr_auto_medium_vs_uniform256",
        suite="pressure",
        reference="pressure_uniform256",
        candidate="pressure_amr_auto_medium",
        components=("pressure", "rho"),
        title="Pressure bump 2D automatic AMR threshold=0.08 vs uniform256",
        rel_l2_tolerance=5.0e-4,
        rel_linf_tolerance=5.0e-3,
        enforce_tolerance=False,
    ),
    Comparison(
        name="pressure_amr_auto_tight_vs_uniform512",
        suite="pressure",
        reference="pressure_uniform512",
        candidate="pressure_amr_auto_tight",
        components=("pressure", "rho"),
        title="Pressure bump 2D automatic AMR threshold=0.02 vs uniform512 reference",
        rel_l2_tolerance=1.0e-3,
        rel_linf_tolerance=1.0e-2,
        enforce_tolerance=False,
    ),
    Comparison(
        name="pressure_amr_nosub_vs_uniform",
        suite="pressure",
        reference="pressure_uniform256",
        candidate="pressure_amr_nosub",
        components=("pressure",),
        title="Pressure bump 2D automatic AMR/no-subcycling threshold=0.08 vs uniform256",
        rel_l2_tolerance=5.0e-4,
        rel_linf_tolerance=5.0e-3,
        enforce_tolerance=False,
    ),
    Comparison(
        name="pressure_subcycling_vs_nosub",
        suite="pressure",
        reference="pressure_amr_nosub",
        candidate="pressure_amr_auto_medium",
        components=("pressure", "rho"),
        title="Pressure bump 2D automatic AMR subcycling vs no-subcycling, WENO-Z/HLLC/SSPRK3, CFL=0.9",
        rel_l2_tolerance=2.0e-3,
        rel_linf_tolerance=2.0e-2,
    ),
    Comparison(
        name="kh_fixedsheet_vs_uniform512",
        suite="kh",
        reference="kh_uniform512",
        candidate="kh_fixedsheet",
        components=("pressure", "rho"),
        title="KH MHD 2D fixed-sheet AMR vs uniform512, WENO-Z/HLLD/SSPRK3/CT, explicit eta=nu=1.5e-4, CFL=0.9",
        rel_l2_tolerance=1.0e-3,
        rel_linf_tolerance=1.0e-2,
    ),
    Comparison(
        name="kh_fullauto_buf16_vs_uniform512",
        suite="kh",
        reference="kh_uniform512",
        candidate="kh_fullauto_buf16",
        components=("pressure", "rho"),
        title="KH MHD 2D full-auto AMR vs uniform512, WENO-Z/HLLD/SSPRK3/CT, explicit eta=nu=1.5e-4, CFL=0.9",
        rel_l2_tolerance=2.0e-3,
        rel_linf_tolerance=2.0e-2,
    ),
]

IMPROVEMENT_CHECKS = [
    ImprovementCheck(
        name="pressure_amr_auto_medium_improves_over_uniform64",
        suite="pressure",
        baseline_comparison="pressure_uniform64_vs_uniform256",
        improved_comparison="pressure_amr_auto_medium_vs_uniform256",
        component="pressure",
        metric="rel_l2",
        min_factor=2.0,
        title="Automatic AMR should reduce error versus uniform64",
    ),
    ImprovementCheck(
        name="pressure_amr_auto_loose_improves_over_uniform64",
        suite="pressure",
        baseline_comparison="pressure_uniform64_vs_uniform512",
        improved_comparison="pressure_amr_auto_loose_vs_uniform512",
        component="pressure",
        metric="rel_l2",
        min_factor=2.0,
        title="Loose automatic AMR should reduce error versus uniform64",
    ),
    ImprovementCheck(
        name="pressure_amr_auto_medium_improves_over_loose",
        suite="pressure",
        baseline_comparison="pressure_amr_auto_loose_vs_uniform512",
        improved_comparison="pressure_amr_auto_medium_vs_uniform512",
        component="pressure",
        metric="rel_l2",
        min_factor=1.05,
        title="Tighter automatic AMR threshold should reduce pressure error",
    ),
    ImprovementCheck(
        name="pressure_amr_auto_tight_improves_over_medium",
        suite="pressure",
        baseline_comparison="pressure_amr_auto_medium_vs_uniform512",
        improved_comparison="pressure_amr_auto_tight_vs_uniform512",
        component="pressure",
        metric="rel_l2",
        min_factor=1.05,
        title="Tight automatic AMR threshold should further reduce pressure error",
    ),
    ImprovementCheck(
        name="pressure_amr_nosub_improves_over_uniform64",
        suite="pressure",
        baseline_comparison="pressure_uniform64_vs_uniform256",
        improved_comparison="pressure_amr_nosub_vs_uniform",
        component="pressure",
        metric="rel_l2",
        min_factor=2.0,
        title="Pressure bump AMR/no-subcycling should reduce error versus uniform64",
    ),
]

RESTART_CHECKS = [
    RestartCheck(
        name="pressure_aligned_subcycling",
        suite="pressure",
        input_file=PRESSURE_INPUT,
        full_output_dir=ROOT / "runs/current/pressure_bump2d_amr_restart_full_001",
        restart_output_dir=ROOT / "runs/current/pressure_bump2d_amr_restart_from3_001",
        checkpoint_step=3,
        final_step=6,
        tolerance="1e-12",
        overrides={
            "amr.max_level": "2",
            "amr.n_cell": "64 64",
            "amr.ref_ratio": "2 2",
            "amr.regrid_mode": "aligned",
            "amr.regrid_int": "7",
            "amr.n_error_buf": "8 8",
            "amr.tag_threshold": "0.03",
            "amr.state_interp": "quartic",
            "pyfluid.max_step": "6",
            "pyfluid.ng": "3",
            "pyfluid.recon": "wenoz",
            "pyfluid.riemann": "hllc",
            "pyfluid.timestep": "rk4",
            "pyfluid.subcycling": "on",
            "pyfluid.plot_int": "3",
            "pyfluid.scalar_int": "1",
            "pyfluid.checkpoint_int": "3",
        },
    ),
    RestartCheck(
        name="kh_ct_subcycling",
        suite="kh",
        input_file=KH_INPUT,
        full_output_dir=ROOT / "runs/current/kh2d_ct_restart_full_001",
        restart_output_dir=ROOT / "runs/current/kh2d_ct_restart_from2_001",
        checkpoint_step=2,
        final_step=4,
        tolerance="1e-10",
        overrides={
            "pyfluid.max_step": "4",
            "pyfluid.final_time": "-1",
            "pyfluid.scalar_int": "1",
            "pyfluid.plot_int": "2",
            "pyfluid.checkpoint_int": "2",
            "pyfluid.aux_gfs": "none",
            "pyfluid.aux_gf_compute_int": "-1",
            "pyfluid.aux_gf_int": "-1",
            "pyfluid.diffusion": "explicit",
            "pyfluid.viscosity": "0.0",
            "pyfluid.resistivity": "1e-4",
            "amr.max_level": "3",
            "amr.initial_refine_levels": "3",
            "amr.initial_refine_fixed_levels": "1",
            "amr.level_cell_budget_side": "512",
            "amr.total_cell_budget": "600000",
            "amr.regrid_int": "8",
        },
    ),
    RestartCheck(
        name="kh_ct_nosubcycling",
        suite="kh",
        input_file=KH_INPUT,
        full_output_dir=ROOT / "runs/current/kh2d_ct_nosub_restart_full_001",
        restart_output_dir=ROOT / "runs/current/kh2d_ct_nosub_restart_from2_001",
        checkpoint_step=2,
        final_step=4,
        tolerance="1e-10",
        overrides={
            "pyfluid.max_step": "4",
            "pyfluid.final_time": "-1",
            "pyfluid.scalar_int": "1",
            "pyfluid.plot_int": "2",
            "pyfluid.checkpoint_int": "2",
            "pyfluid.aux_gfs": "none",
            "pyfluid.aux_gf_compute_int": "-1",
            "pyfluid.aux_gf_int": "-1",
            "pyfluid.subcycling": "off",
            "amr.max_level": "2",
            "amr.initial_refine_levels": "2",
            "amr.initial_refine_fixed_levels": "1",
            "amr.level_cell_budget_side": "512",
            "amr.total_cell_budget": "300000",
            "amr.regrid_int": "2",
        },
    ),
    RestartCheck(
        name="implicit_resistive_ct",
        suite="elliptic",
        input_file=RESISTIVE_WAVE_INPUT,
        full_output_dir=ROOT / "runs/current/resistive_ct_implicit_restart_full_001",
        restart_output_dir=ROOT / "runs/current/resistive_ct_implicit_restart_from2_001",
        checkpoint_step=2,
        final_step=4,
        tolerance="1e-12",
        overrides={
            "pyfluid.max_step": "4",
            "pyfluid.final_time": "-1",
            "pyfluid.diffusion": "implicit",
            "pyfluid.diffusion_tolerance": "1e-12",
            "pyfluid.diffusion_max_iterations": "500",
            "pyfluid.scalar_int": "1",
            "pyfluid.plot_int": "2",
            "pyfluid.checkpoint_int": "2",
            "pyfluid.aux_gfs": "none",
            "pyfluid.aux_gf_compute_int": "-1",
            "pyfluid.aux_gf_int": "-1",
            "amr.max_level": "0",
        },
    ),
    RestartCheck(
        name="self_gravity_deep_amr_subcycling",
        suite="elliptic",
        input_file=HYDRO_SINE_INPUT,
        full_output_dir=ROOT / "runs/current/self_gravity_deep_restart_full_001",
        restart_output_dir=ROOT / "runs/current/self_gravity_deep_restart_from2_001",
        checkpoint_step=2,
        final_step=4,
        tolerance="1e-12",
        overrides={
            "geometry.is_periodic": "1 1",
            "amr.n_cell": "16 16",
            "amr.max_level": "10",
            "amr.ref_ratio": "2 2 2 2 2 2 2 2 2 2",
            "amr.max_grid_size": "16",
            "amr.blocking_factor": "8",
            "amr.regrid_mode": "aligned",
            "amr.regrid_int": "2",
            "amr.n_error_buf": "4 4",
            "amr.tag_metric": "lohner",
            "amr.tag_threshold": "0.12",
            "amr.state_interp": "quartic-conservative",
            "amr.initial_refine": "box",
            "amr.initial_refine_levels": "3",
            "amr.initial_refine_fixed_levels": "0",
            "amr.initial_refine_box_lo": "0.25 0.25",
            "amr.initial_refine_box_hi": "0.75 0.75",
            "amr.total_cell_budget": "1000000",
            "amr.min_feature_length": "0.0005",
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
            "pyfluid.max_step": "4",
            "pyfluid.final_time": "-1",
            "pyfluid.timestep": "ssprk3",
            "pyfluid.recon": "pwc",
            "pyfluid.riemann": "llf",
            "pyfluid.ng": "1",
            "pyfluid.subcycling": "on",
            "pyfluid.subcycling_steps": "local_cfl",
            "pyfluid.gravity": "poisson",
            "pyfluid.gravity_coupling": "athena",
            "pyfluid.gravity_constant": "1.0",
            "pyfluid.gravity_tolerance": "1e-10",
            "pyfluid.gravity_max_iterations": "200",
            "pyfluid.gravity_max_fmg_iterations": "20",
            "pyfluid.gravity_max_order": "2",
            "pyfluid.physical_check_int": "1",
            "pyfluid.scalar_int": "1",
            "pyfluid.plot_int": "2",
            "pyfluid.checkpoint_int": "2",
        },
    ),
]


def rel(path: Path) -> str:
   return str(path.relative_to(ROOT))


def final_scalars(output_dir: Path) -> dict[str, str]:
   return read_scalar_rows(output_dir)[-1]


def final_amr_state(output_dir: Path) -> dict[str, str]:
    with (output_dir / "amr_levels.tsv").open(newline="") as handle:
        rows = list(csv.DictReader(
            (line for line in handle if not line.startswith("#")),
            delimiter="\t",
        ))
    if not rows:
        raise RuntimeError(f"{output_dir} has no AMR hierarchy rows")
    return rows[-1]


def require_metric_limited_hierarchy(case: Case) -> dict[str, str]:
    row = final_amr_state(case.output_dir)
    active_level = int(row["active_finest_level"])
    headroom = int(row["finest_headroom"])
    budget_fraction = float(row["total_budget_fraction"])
    terminal_decision = row.get(f"refinement_decision_L{active_level}", "")
    failures = []
    if active_level < 1:
        failures.append("automatic AMR did not refine")
    if headroom < 2:
        failures.append(f"only {headroom} level(s) of headroom")
    if int(row["touches_max_level"]) != 0:
        failures.append("hierarchy touched amr.max_level")
    if budget_fraction >= 0.8:
        failures.append(f"cell-budget fraction {budget_fraction:.3f} >= 0.8")
    if terminal_decision != "smooth":
        failures.append(f"terminal refinement decision is {terminal_decision!r}, not 'smooth'")
    if failures:
        raise RuntimeError(f"{case.name}: " + "; ".join(failures))
    return row


def run_tool(args: list[str]) -> str:
    result = subprocess.run(args, cwd=ROOT, check=True, text=True, capture_output=True)
    return result.stdout.strip()


def run_solver(input_file: Path, output_dir: Path, overrides: dict[str, str]) -> None:
    parameters = {"pyfluid.output_dir": rel(output_dir), **overrides}
    inactive_amr_parameters = ("amr.ref_ratio",) if overrides.get("amr.max_level") == "0" else ()
    run_input = materialize_run_input(
        input_file,
        output_dir / "run.in",
        parameters,
        remove_parameters=inactive_amr_parameters,
    )
    completed = run_mpi(
        SOLVER_2D,
        [run_input],
        ranks=3,
    )
    (output_dir / "run.log").write_text(completed.stdout)
    if completed.returncode != 0:
        raise RuntimeError(completed.stdout)


def run_case(case: Case, force: bool) -> Path:
   plot_dir = case.output_dir
   if force and plot_dir.exists():
      shutil.rmtree(plot_dir)
   existing = discover_latest_plotfile(plot_dir, "stateGF")
   if existing is None or force:
      run_solver(case.input_file, case.output_dir, case.overrides)
   plotfile = discover_latest_plotfile(plot_dir, "stateGF")
   if plotfile is None:
      raise RuntimeError(f"{case.name} did not write a plotfile")
   scalar_failures = check_scalars(read_scalar_rows(case.output_dir), mhd=case.suite == "kh")
   if scalar_failures:
      raise RuntimeError(f"{case.name}: " + "; ".join(scalar_failures))
   if case.require_metric_stop:
      require_metric_limited_hierarchy(case)
   return plotfile


def plot_case(case: Case, plotfile: Path) -> None:
    case.plot_root.mkdir(parents=True, exist_ok=True)
    if case.hierarchy:
        run_tool(
            [
                "uv",
                "run",
                "python",
                "pyfluid_amrex/tools/plot_amr_hierarchy.py",
                "--plotfile",
                rel(plotfile),
                "--output",
                rel(case.plot_root / f"{case.name}_hierarchy_001.png"),
                "--title",
                f"{case.title} hierarchy",
                "--draw-boxes",
            ]
        )
    if case.scalars:
        run_tool(
            [
                "uv",
                "run",
                "python",
                "pyfluid_amrex/tools/plot_amrex_scalars.py",
                "--input",
                rel(case.output_dir / "scalars.tsv"),
                "--output",
                rel(case.plot_root / f"{case.name}_scalars_001.png"),
                "--title",
                case.title,
            ]
        )


def plotfile_for(name: str, case_plotfiles: dict[str, Path]) -> Path:
    if name in case_plotfiles:
        return case_plotfiles[name]
    raise KeyError(name)


def parse_comparison_stdout(text: str, component: str) -> dict[str, str]:
    row = {}
    for token in text.split():
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        row[f"{component}_{key}"] = value
    return row


def parse_number(row: dict[str, str], key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, ValueError):
        return float("nan")


def run_comparison(comparison: Comparison, case_map: dict[str, Case], case_plotfiles: dict[str, Path]) -> dict[str, str]:
    candidate = case_map[comparison.candidate]
    row = {
        "kind": "comparison",
        "name": comparison.name,
        "suite": comparison.suite,
        "reference": comparison.reference,
        "candidate": comparison.candidate,
        "criterion": "absolute_tolerance" if comparison.enforce_tolerance else "convergence_sequence",
        "rel_l2_tolerance": f"{comparison.rel_l2_tolerance:.8e}" if comparison.enforce_tolerance else "",
        "rel_linf_tolerance": f"{comparison.rel_linf_tolerance:.8e}" if comparison.enforce_tolerance else "",
    }
    for component in comparison.components:
        output = candidate.plot_root / f"{comparison.name}_{component}_001.png"
        text = run_tool(
            [
                "uv",
                "run",
                "python",
                "pyfluid_amrex/tools/compare_amrex_plotfiles.py",
                "--reference",
                rel(plotfile_for(comparison.reference, case_plotfiles)),
                "--candidate",
                rel(plotfile_for(comparison.candidate, case_plotfiles)),
                "--component",
                component,
                "--output",
                rel(output),
                "--title",
                comparison.title,
            ]
        )
        row.update(parse_comparison_stdout(text, component))
    failures = []
    if comparison.enforce_tolerance:
        for component in comparison.components:
            rel_l2 = parse_number(row, f"{component}_rel_l2")
            rel_linf = parse_number(row, f"{component}_rel_linf")
            if not rel_l2 <= comparison.rel_l2_tolerance:
                failures.append(f"{component}_rel_l2={rel_l2:.3e}>{comparison.rel_l2_tolerance:.3e}")
            if not rel_linf <= comparison.rel_linf_tolerance:
                failures.append(f"{component}_rel_linf={rel_linf:.3e}>{comparison.rel_linf_tolerance:.3e}")
    row["status"] = "PASS" if not failures else "FAIL"
    row["reason"] = "; ".join(failures)
    return row


def run_improvement_check(check: ImprovementCheck, rows: list[dict[str, str]]) -> dict[str, str]:
    by_name = {row.get("name", ""): row for row in rows if row.get("kind") == "comparison"}
    baseline = by_name.get(check.baseline_comparison)
    improved = by_name.get(check.improved_comparison)
    key = f"{check.component}_{check.metric}"
    baseline_value = parse_number(baseline or {}, key)
    improved_value = parse_number(improved or {}, key)
    factor = baseline_value / improved_value if improved_value > 0.0 else float("inf")
    failures = []
    if baseline is None:
        failures.append(f"missing baseline comparison {check.baseline_comparison}")
    if improved is None:
        failures.append(f"missing improved comparison {check.improved_comparison}")
    if not factor >= check.min_factor:
        failures.append(f"improvement_factor={factor:.3e}<{check.min_factor:.3e}")
    return {
        "kind": "improvement",
        "status": "PASS" if not failures else "FAIL",
        "reason": "; ".join(failures),
        "name": check.name,
        "suite": check.suite,
        "baseline": check.baseline_comparison,
        "improved": check.improved_comparison,
        "component": check.component,
        "metric": check.metric,
        "baseline_error": f"{baseline_value:.8e}",
        "improved_error": f"{improved_value:.8e}",
        "improvement_factor": f"{factor:.8e}",
        "min_factor": f"{check.min_factor:.8e}",
    }


def run_restart_check(check: RestartCheck, force: bool) -> dict[str, str]:
    if force:
        for output_dir in (check.full_output_dir, check.restart_output_dir):
            if output_dir.exists():
                shutil.rmtree(output_dir)
    full_checkpoint = checkpoint_path(check.full_output_dir, check.final_step)
    restart_checkpoint = checkpoint_path(check.restart_output_dir, check.final_step)
    if force or not full_checkpoint.exists():
        run_solver(check.input_file, check.full_output_dir, check.overrides)
    if force or not restart_checkpoint.exists():
        restart_file = checkpoint_path(check.full_output_dir, check.checkpoint_step)
        if not restart_file.exists():
            raise RuntimeError(f"missing restart checkpoint: {rel(restart_file)}")
        restart_overrides = {
            **check.overrides,
            "pyfluid.restart_file": rel(restart_file),
        }
        run_solver(check.input_file, check.restart_output_dir, restart_overrides)
    completed = run_mpi(
        CHECKPOINT_COMPARE_2D,
        [full_checkpoint, restart_checkpoint, check.tolerance],
        ranks=3,
    )
    text = completed.stdout.strip()
    (check.restart_output_dir / "checkpoint_compare.log").write_text(completed.stdout)
    row = {
        "kind": "restart",
        "status": "PASS" if completed.returncode == 0 else "FAIL",
        "reason": "" if completed.returncode == 0 else "checkpoint comparison failed; see checkpoint_compare.log",
        "name": check.name,
        "suite": check.suite,
        "candidate": check.restart_output_dir.name,
        "reference": check.full_output_dir.name,
        "tolerance": check.tolerance,
    }
    for token in text.split():
        if "=" in token:
            key, value = token.split("=", 1)
            row[key] = value

    return row


def selected_suites(value: str) -> set[str]:
    if value == "all":
           return {"pressure", "kh", "elliptic"}
    return {value}


def write_summary(rows: list[dict[str, str]]) -> None:
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row})
    with SUMMARY.open("w", newline="") as f:
           writer = csv.DictWriter(f, fieldnames=keys, delimiter="\t")
           writer.writeheader()
           writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=["pressure", "kh", "elliptic", "all"], default="all")
    parser.add_argument("--case", choices=[case.name for case in CASES], action="append")
    parser.add_argument("--force", action="store_true", help="rerun cases even if plotfiles already exist")
    parser.add_argument("--restart-checks", action="store_true", help="also run checkpoint/restart comparisons")
    parser.add_argument(
        "--restart-case",
        choices=[check.name for check in RESTART_CHECKS],
        action="append",
        help="run only the selected checkpoint/restart comparison (implies --restart-checks)",
    )
    args = parser.parse_args()
    write_summary([{"status": "INCOMPLETE", "reason": "validation in progress"}])

    if args.restart_checks or args.restart_case:
        build_cached_target(CHECKPOINT_COMPARE_2D.name)

    suites = selected_suites(args.suite)
    selected_cases = [
           case for case in CASES
           if case.suite in suites and (args.case is None or case.name in args.case)
    ]
    case_map = {case.name: case for case in CASES}
    case_plotfiles = {}
    rows: list[dict[str, str]] = []

    for case in selected_cases:
           plotfile = run_case(case, args.force)
           case_plotfiles[case.name] = plotfile
           plot_case(case, plotfile)
           scalars = final_scalars(case.output_dir)
           amr_state = final_amr_state(case.output_dir) if case.hierarchy else {}
           rows.append(
               {
                   "kind": "case",
                   "status": "PASS",
                   "reason": "",
                   "name": case.name,
                   "suite": case.suite,
                   "plotfile": rel(plotfile),
                   "time": scalars.get("time", ""),
                   "mass_rel_drift": scalars.get("mass_rel_drift", ""),
                   "energy_rel_drift": scalars.get("energy_rel_drift", ""),
                   "divb_l1": scalars.get("divb_l1", ""),
                   "mcells_per_s": scalars.get("mcells_per_s", ""),
                   "sim_time_per_s": scalars.get("sim_time_per_s", ""),
                   "amr_active_finest_level": amr_state.get("active_finest_level", "0"),
                   "amr_finest_headroom": amr_state.get("finest_headroom", "0"),
                   "amr_terminal_decision": amr_state.get(
                       f"refinement_decision_L{amr_state.get('active_finest_level', '0')}",
                       "",
                   ),
               }
           )

    for comparison in COMPARISONS:
           if comparison.suite not in suites:
               continue
           needed = {comparison.reference, comparison.candidate}
           if args.case is not None and not needed.issubset(case_plotfiles.keys()):
               continue
           rows.append(run_comparison(comparison, case_map, case_plotfiles))

    available_comparisons = {
           row.get("name", "") for row in rows if row.get("kind") == "comparison"
    }
    for check in IMPROVEMENT_CHECKS:
           if check.suite not in suites:
               continue
           needed = {check.baseline_comparison, check.improved_comparison}
           if args.case is not None and not needed.issubset(available_comparisons):
               continue
           rows.append(run_improvement_check(check, rows))

    if args.restart_checks or args.restart_case:
           for check in RESTART_CHECKS:
               if check.suite in suites and (args.restart_case is None or check.name in args.restart_case):
                   rows.append(run_restart_check(check, args.force))

    write_summary(rows)
    for row in rows:
           if row["kind"] == "case":
               print(
                   row["name"],
                   "energy_drift=" + row.get("energy_rel_drift", "nan"),
                   "divb=" + row.get("divb_l1", "nan"),
                   "Mcells/s=" + row.get("mcells_per_s", "nan"),
                   "sim_time/s=" + row.get("sim_time_per_s", "nan"),
                   "AMR=L" + row.get("amr_active_finest_level", "0"),
                   "headroom=" + row.get("amr_finest_headroom", "0"),
                   "stop=" + row.get("amr_terminal_decision", ""),
               )
           elif row["kind"] == "comparison":
               print(
                   row["name"],
                   row.get("status", ""),
                   "pressure_l2=" + row.get("pressure_rel_l2", "nan"),
                   "rho_l2=" + row.get("rho_rel_l2", "nan"),
                   "tol_l2=" + row.get("rel_l2_tolerance", "nan"),
               )
           else:
               if row["kind"] == "improvement":
                   print(
                       row["name"],
                       row.get("status", ""),
                       "factor=" + row.get("improvement_factor", "nan"),
                       "min=" + row.get("min_factor", "nan"),
                   )
               else:
                   print(
                       row["name"],
                       row.get("status", ""),
                       "checkpoint_max=" + row.get("max_field_abs", "nan"),
                       "total_l1=" + row.get("total_l1", "nan"),
                       "tol=" + row.get("tolerance", "nan"),
                   )
    failures = [row for row in rows if row.get("status") == "FAIL"]
    print(f"summary={rel(SUMMARY)}")
    if failures:
           raise SystemExit(1)


if __name__ == "__main__":
    main()
