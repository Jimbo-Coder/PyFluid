#!/usr/bin/env python3
"""Run compact AMR validation cases and regenerate their diagnostic plots."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plotfile_reader import checkpoint_path, latest_plotfile as discover_latest_plotfile  # noqa: E402
from validation_common import mpi_prefix
from validation_common import mpi_prefix  # noqa: E402


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
    title: str
    overrides: dict[str, str]


PRESSURE_INPUT = ROOT / "pyfluid_amrex/inputs/pressure_bump_2d_amr.in"
KH_INPUT = ROOT / "pyfluid_amrex/inputs/kelvin_helmholtz_mhd_2d_movie_ct.in"
PRESSURE_PLOTS = ROOT / "plots/current/IC/pressure_bump"
KH_PLOTS = ROOT / "plots/current/IC/kelvin_helmholtz_mhd"

CASES = [
    Case(
        name="pressure_uniform64",
        suite="pressure",
        input_file=PRESSURE_INPUT,
        output_dir=ROOT / "runs/current/pressure_bump2d_uniform64_t002_001",
        plot_root=PRESSURE_PLOTS,
        title="Pressure bump 2D uniform64, WENO-Z/HLLC/RK4, CFL=0.9",
        hierarchy=False,
        overrides={
            "amr.max_level": "0",
            "amr.n_cell": "64 64",
            "pyfluid.final_time": "0.02",
            "pyfluid.max_step": "100000",
            "pyfluid.ng": "3",
            "pyfluid.recon": "wenoz",
            "pyfluid.riemann": "hllc",
            "pyfluid.timestep": "rk4",
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
        title="Pressure bump 2D uniform256, WENO-Z/HLLC/RK4, CFL=0.9",
        hierarchy=False,
        overrides={
            "amr.max_level": "0",
            "amr.n_cell": "256 256",
            "pyfluid.final_time": "0.02",
            "pyfluid.max_step": "100000",
            "pyfluid.ng": "3",
            "pyfluid.recon": "wenoz",
            "pyfluid.riemann": "hllc",
            "pyfluid.timestep": "rk4",
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
        title="Pressure bump 2D uniform512 reference, WENO-Z/HLLC/RK4, CFL=0.9",
        hierarchy=False,
        overrides={
            "amr.max_level": "0",
            "amr.n_cell": "512 512",
            "pyfluid.final_time": "0.02",
            "pyfluid.max_step": "100000",
            "pyfluid.ng": "3",
            "pyfluid.recon": "wenoz",
            "pyfluid.riemann": "hllc",
            "pyfluid.timestep": "rk4",
            "pyfluid.plot_int": "100000",
            "pyfluid.scalar_int": "8",
            "pyfluid.checkpoint_int": "-1",
        },
    ),
    Case(
        name="pressure_amr32_l2_subcycling",
        suite="pressure",
        input_file=PRESSURE_INPUT,
        output_dir=ROOT / "runs/current/pressure_bump2d_amr32_l2_t002_001",
        plot_root=PRESSURE_PLOTS,
        title="Pressure bump 2D AMR32-L2/subcycling, WENO-Z/HLLC/RK4, CFL=0.9",
        overrides={
            "amr.max_level": "2",
            "amr.n_cell": "32 32",
            "amr.ref_ratio": "2 2",
            "amr.regrid_int": "2",
            "amr.n_error_buf": "8 8",
            "amr.tag_threshold": "0.03",
            "amr.state_interp": "quartic",
            "pyfluid.final_time": "0.02",
            "pyfluid.max_step": "100000",
            "pyfluid.ng": "3",
            "pyfluid.recon": "wenoz",
            "pyfluid.riemann": "hllc",
            "pyfluid.timestep": "rk4",
            "pyfluid.subcycling": "on",
            "pyfluid.plot_int": "100000",
            "pyfluid.scalar_int": "5",
            "pyfluid.checkpoint_int": "-1",
        },
    ),
    Case(
        name="pressure_amr_subcycling",
        suite="pressure",
        input_file=PRESSURE_INPUT,
        output_dir=ROOT / "runs/current/pressure_bump2d_amr64_l2_t002_001",
        plot_root=PRESSURE_PLOTS,
        title="Pressure bump 2D AMR64-L2/subcycling, WENO-Z/HLLC/RK4, CFL=0.9",
        overrides={
            "amr.max_level": "2",
            "amr.n_cell": "64 64",
            "amr.ref_ratio": "2 2",
            "amr.regrid_int": "2",
            "amr.n_error_buf": "8 8",
            "amr.tag_threshold": "0.03",
            "amr.state_interp": "quartic",
            "pyfluid.final_time": "0.02",
            "pyfluid.max_step": "100000",
            "pyfluid.ng": "3",
            "pyfluid.recon": "wenoz",
            "pyfluid.riemann": "hllc",
            "pyfluid.timestep": "rk4",
            "pyfluid.subcycling": "on",
            "pyfluid.plot_int": "100000",
            "pyfluid.scalar_int": "5",
            "pyfluid.checkpoint_int": "-1",
        },
    ),
    Case(
        name="pressure_amr_nosub",
        suite="pressure",
        input_file=PRESSURE_INPUT,
        output_dir=ROOT / "runs/current/pressure_bump2d_amr64_l2_nosub_t002_001",
        plot_root=PRESSURE_PLOTS,
        title="Pressure bump 2D AMR64-L2/no-subcycling, WENO-Z/HLLC/RK4, CFL=0.9",
        overrides={
            "amr.max_level": "2",
            "amr.n_cell": "64 64",
            "amr.ref_ratio": "2 2",
            "amr.regrid_int": "2",
            "amr.n_error_buf": "8 8",
            "amr.tag_threshold": "0.03",
            "amr.state_interp": "quartic",
            "pyfluid.final_time": "0.02",
            "pyfluid.max_step": "100000",
            "pyfluid.ng": "3",
            "pyfluid.recon": "wenoz",
            "pyfluid.riemann": "hllc",
            "pyfluid.timestep": "rk4",
            "pyfluid.subcycling": "off",
            "pyfluid.plot_int": "100000",
            "pyfluid.scalar_int": "5",
            "pyfluid.checkpoint_int": "-1",
        },
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
        title="Pressure bump 2D uniform64 vs uniform256, WENO-Z/HLLC/RK4, CFL=0.9",
        rel_l2_tolerance=5.0e-2,
        rel_linf_tolerance=5.0e-1,
    ),
    Comparison(
        name="pressure_uniform64_vs_uniform512",
        suite="pressure",
        reference="pressure_uniform512",
        candidate="pressure_uniform64",
        components=("pressure", "rho"),
        title="Pressure bump 2D uniform64 vs uniform512 reference, WENO-Z/HLLC/RK4, CFL=0.9",
        rel_l2_tolerance=5.0e-2,
        rel_linf_tolerance=5.0e-1,
    ),
    Comparison(
        name="pressure_amr32_l2_vs_uniform512",
        suite="pressure",
        reference="pressure_uniform512",
        candidate="pressure_amr32_l2_subcycling",
        components=("pressure", "rho"),
        title="Pressure bump 2D AMR32-L2/subcycling vs uniform512 reference, WENO-Z/HLLC/RK4, CFL=0.9",
        rel_l2_tolerance=5.0e-3,
        rel_linf_tolerance=5.0e-2,
    ),
    Comparison(
        name="pressure_amr64_l2_vs_uniform512",
        suite="pressure",
        reference="pressure_uniform512",
        candidate="pressure_amr_subcycling",
        components=("pressure", "rho"),
        title="Pressure bump 2D AMR64-L2/subcycling vs uniform512 reference, WENO-Z/HLLC/RK4, CFL=0.9",
        rel_l2_tolerance=2.0e-3,
        rel_linf_tolerance=2.0e-2,
    ),
    Comparison(
        name="pressure_amr_subcycling_vs_uniform",
        suite="pressure",
        reference="pressure_uniform256",
        candidate="pressure_amr_subcycling",
        components=("pressure", "rho"),
        title="Pressure bump 2D AMR64-L2/subcycling vs uniform256, WENO-Z/HLLC/RK4, CFL=0.9",
        rel_l2_tolerance=5.0e-4,
        rel_linf_tolerance=5.0e-3,
    ),
    Comparison(
        name="pressure_amr_nosub_vs_uniform",
        suite="pressure",
        reference="pressure_uniform256",
        candidate="pressure_amr_nosub",
        components=("pressure",),
        title="Pressure bump 2D AMR64-L2/no-subcycling vs uniform256, WENO-Z/HLLC/RK4, CFL=0.9",
        rel_l2_tolerance=5.0e-4,
        rel_linf_tolerance=5.0e-3,
    ),
    Comparison(
        name="pressure_subcycling_vs_nosub",
        suite="pressure",
        reference="pressure_amr_nosub",
        candidate="pressure_amr_subcycling",
        components=("pressure", "rho"),
        title="Pressure bump 2D AMR64-L2 subcycling vs no-subcycling, WENO-Z/HLLC/RK4, CFL=0.9",
        rel_l2_tolerance=1.0e-5,
        rel_linf_tolerance=1.0e-4,
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
        name="pressure_amr_subcycling_improves_over_uniform64",
        suite="pressure",
        baseline_comparison="pressure_uniform64_vs_uniform256",
        improved_comparison="pressure_amr_subcycling_vs_uniform",
        component="pressure",
        metric="rel_l2",
        min_factor=2.0,
        title="Pressure bump AMR/subcycling should reduce error versus uniform64",
    ),
    ImprovementCheck(
        name="pressure_amr32_l2_improves_over_uniform64",
        suite="pressure",
        baseline_comparison="pressure_uniform64_vs_uniform512",
        improved_comparison="pressure_amr32_l2_vs_uniform512",
        component="pressure",
        metric="rel_l2",
        min_factor=2.0,
        title="Pressure bump AMR32-L2 should reduce error versus uniform64",
    ),
    ImprovementCheck(
        name="pressure_amr64_l2_improves_over_amr32_l2",
        suite="pressure",
        baseline_comparison="pressure_amr32_l2_vs_uniform512",
        improved_comparison="pressure_amr64_l2_vs_uniform512",
        component="pressure",
        metric="rel_l2",
        min_factor=1.5,
        title="Pressure bump AMR64-L2 should reduce error versus AMR32-L2",
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
        name="pressure_subcycling",
        suite="pressure",
        input_file=PRESSURE_INPUT,
        full_output_dir=ROOT / "runs/current/pressure_bump2d_amr_restart_full_001",
        restart_output_dir=ROOT / "runs/current/pressure_bump2d_amr_restart_from3_001",
        checkpoint_step=3,
        final_step=6,
        tolerance="1e-12",
        title="Pressure bump 2D AMR restart check, WENO-Z/HLLC/RK4/subcycling, CFL=0.9",
        overrides={
            "amr.max_level": "2",
            "amr.n_cell": "64 64",
            "amr.ref_ratio": "2 2",
            "amr.regrid_int": "2",
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
        title="KH MHD 2D AMR restart check, WENO-Z/HLLD/RK4/CT, explicit eta=nu=1.5e-4, CFL=0.9",
        overrides={
            "pyfluid.max_step": "4",
            "pyfluid.final_time": "-1",
            "pyfluid.scalar_int": "1",
            "pyfluid.plot_int": "2",
            "pyfluid.checkpoint_int": "2",
            "pyfluid.aux_gfs": "none",
            "pyfluid.aux_gf_compute_int": "-1",
            "pyfluid.aux_gf_int": "-1",
            "amr.max_level": "2",
            "amr.initial_refine_levels": "2",
            "amr.initial_refine_fixed_levels": "1",
            "amr.level_cell_budget_side": "512",
            "amr.total_cell_budget": "300000",
            "amr.regrid_int": "2",
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
        title="KH MHD 2D AMR restart check, WENO-Z/HLLD/RK4/CT/no-subcycling, explicit eta=nu=1.5e-4, CFL=0.9",
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
]


def rel(path: Path) -> str:
   return str(path.relative_to(ROOT))


def final_scalars(output_dir: Path) -> dict[str, str]:
   with (output_dir / "scalars.tsv").open(newline="") as f:
      rows = list(csv.DictReader(f, delimiter="\t"))
   return rows[-1]


def run_tool(args: list[str]) -> str:
    result = subprocess.run(args, cwd=ROOT, check=True, text=True, capture_output=True)
    return result.stdout.strip()


def run_solver(input_file: Path, output_dir: Path, overrides: dict[str, str]) -> None:
    command = [
        *mpi_prefix(3),
        rel(SOLVER_2D),
        rel(input_file),
        f"pyfluid.output_dir={rel(output_dir)}",
        *[f"{key}={value}" for key, value in overrides.items()],
    ]
    subprocess.run(command, cwd=ROOT, check=True)


def run_case(case: Case, force: bool) -> Path:
   plot_dir = case.output_dir
   existing = discover_latest_plotfile(plot_dir, "stateGF")
   if existing is None or force:
      run_solver(case.input_file, case.output_dir, case.overrides)
   plotfile = discover_latest_plotfile(plot_dir, "stateGF")
   if plotfile is None:
      raise RuntimeError(f"{case.name} did not write a plotfile")
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
        "rel_l2_tolerance": f"{comparison.rel_l2_tolerance:.8e}",
        "rel_linf_tolerance": f"{comparison.rel_linf_tolerance:.8e}",
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
    text = run_tool(
        [
            *mpi_prefix(3),
            rel(CHECKPOINT_COMPARE_2D),
            rel(full_checkpoint),
            rel(restart_checkpoint),
            check.tolerance,
        ]
    )
    row = {
        "kind": "restart",
        "status": "PASS",
        "reason": "",
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

    plot_root = KH_PLOTS if check.suite == "kh" else PRESSURE_PLOTS
    run_tool(
        [
            "uv",
            "run",
            "python",
            "pyfluid_amrex/tools/compare_amrex_plotfiles.py",
            "--reference",
            rel(check.full_output_dir / "stateGF" / f"stateGF{check.final_step:05d}"),
            "--candidate",
            rel(check.restart_output_dir / "stateGF" / f"stateGF{check.final_step:05d}"),
            "--component",
            "pressure",
            "--output",
            rel(plot_root / f"{check.name}_restart_pressure_001.png"),
            "--title",
            check.title,
        ]
    )
    return row


def selected_suites(value: str) -> set[str]:
    if value == "all":
        return {"pressure", "kh"}
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
    parser.add_argument("--suite", choices=["pressure", "kh", "all"], default="all")
    parser.add_argument("--case", choices=[case.name for case in CASES], action="append")
    parser.add_argument("--force", action="store_true", help="rerun cases even if plotfiles already exist")
    parser.add_argument("--restart-checks", action="store_true", help="also run checkpoint/restart comparisons")
    args = parser.parse_args()

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

    if args.restart_checks:
        for check in RESTART_CHECKS:
            if check.suite in suites:
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
