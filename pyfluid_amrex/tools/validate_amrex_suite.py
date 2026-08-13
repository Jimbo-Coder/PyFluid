#!/usr/bin/env python3
"""Run the local AMReX validation suite.

The step list below is the single place to define the normal validation
workflow. Child tools keep concrete cases and outputs; this driver selects
quick/standard/full slices, captures logs, and writes one compact summary.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from validation_common import append_validation_history, recent_history_path, write_tsv


ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = ROOT / "runs/current/amrex_standard_gate"
SUMMARY = RUN_ROOT / "summary.tsv"
LOG_DIR = RUN_ROOT / "logs"
HISTORY = ROOT / "runs/current/validation_history.tsv"


@dataclass(frozen=True)
class GateStep:
   name: str
   command: tuple[str, ...]
   kind: str
   levels: tuple[str, ...] = ("standard", "full")


def feature_matrix_command(*cases: str) -> tuple[str, ...]:
   command = (
      "uv", "run", "python", "pyfluid_amrex/tools/validate_feature_matrix.py", "--force",
   )
   return command + tuple(
      argument for case in cases for argument in ("--only", case)
   )


STEPS = [
   GateStep(
      name="architecture",
      kind="feature",
      levels=("quick", "standard", "full"),
      command=("uv", "run", "python", "pyfluid_amrex/tools/check_architecture.py"),
   ),
   GateStep(
      name="exact_reference_sampling",
      kind="analytic",
      levels=("quick", "standard", "full"),
      command=(
         "uv",
         "run",
         "python",
         "pyfluid_amrex/tools/validate_exact_reference.py",
      ),
   ),
   GateStep(
      name="quick_feature_matrix",
      kind="feature",
      levels=("quick",),
      command=feature_matrix_command(
         "hydro_timestep_ssprk3",
         "mhd_amr_ct_subcycling_euler",
      ),
   ),
   GateStep(
      name="standard_feature_matrix",
      kind="feature",
      levels=("standard",),
      command=feature_matrix_command(
         "hydro_timestep_ssprk3",
         "hydro_amr_subcycling_ssprk3",
         "mhd_timestep_ssprk3",
         "mhd_amr_ct_subcycling_ssprk3",
         "hydro_passive_dye_advection",
         "mhd_passive_dye_ct",
         "hydro_diffusion_explicit",
         "hydro_amr_subcycling_explicit_diffusion",
         "hydro_amr_subcycling_implicit_diffusion",
         "mhd_resistivity_explicit_ct",
         "mhd_amr_ct_subcycling_explicit_resistive",
         "hydro_source_uniform_acceleration_amr",
         "hydro_self_gravity_amr_subcycling",
         "mhd_combo_amr_ct_subcycling_resistive_aux",
         "hydro_obstacle_half_bottom",
         "hydro_moving_box_piston",
      ),
   ),
   GateStep(
      name="feature_matrix",
      kind="feature",
      levels=("full",),
      command=("uv", "run", "python", "pyfluid_amrex/tools/validate_feature_matrix.py", "--force"),
   ),
   GateStep(
      name="deep_aligned_subcycling_l7",
      kind="amr",
      levels=("standard",),
      command=(
         "uv",
         "run",
         "python",
         "pyfluid_amrex/tools/validate_feature_matrix.py",
         "--force",
         "--only",
         "hydro_amr_subcycling_aligned_deep_l7",
      ),
   ),
   GateStep(
      name="rhd_standard",
      kind="analytic",
      levels=("standard", "full"),
      command=(
         "uv",
         "run",
         "python",
         "pyfluid_amrex/tools/validate_rhd.py",
         "--level",
         "standard",
         "--ranks",
         "3",
         "--force",
      ),
   ),
   GateStep(
      name="rmhd_standard",
      kind="reference",
      levels=("standard", "full"),
      command=(
         "uv",
         "run",
         "python",
         "pyfluid_amrex/tools/validate_rmhd.py",
         "--level",
         "standard",
         "--ranks",
         "3",
         "--force",
      ),
   ),
   GateStep(
      name="rmhd_balsara3_amr_prolongation",
      kind="amr",
      levels=("standard", "full"),
      command=(
         "uv",
         "run",
         "python",
         "pyfluid_amrex/tools/validate_rmhd_balsara3_amr_prolongation.py",
         "--ranks",
         "3",
         "--force",
      ),
   ),
   GateStep(
      name="rmhd_advanced_riemann",
      kind="reference",
      levels=("standard", "full"),
      command=(
         "uv",
         "run",
         "python",
         "pyfluid_amrex/tools/validate_rmhd_advanced_riemann.py",
         "--ranks",
         "3",
         "--cells",
         "512",
         "--force",
      ),
   ),
   GateStep(
      name="fixed_spacetime_standard",
      kind="analytic",
      levels=("standard", "full"),
      command=(
         "uv",
         "run",
         "python",
         "pyfluid_amrex/tools/validate_fixed_spacetime.py",
         "--level",
         "standard",
         "--ranks",
         "3",
         "--force",
      ),
   ),
   GateStep(
      name="initial_data_symmetry",
      kind="symmetry",
      levels=("full",),
      command=(
         "uv",
         "run",
         "python",
         "pyfluid_amrex/tools/validate_initial_data_symmetry.py",
         "--ranks",
         "3",
         "--force",
      ),
   ),
   GateStep(
      name="reduced_initial_data_handoff",
      kind="initial_data",
      levels=("standard", "full"),
      command=(
         "uv",
         "run",
         "python",
         "pyfluid_amrex/tools/validate_reduced_initial_data.py",
         "--level",
         "standard",
         "--ranks",
         "3",
         "--force",
      ),
   ),
   GateStep(
      name="obstacle_mask_performance",
      kind="feature",
      levels=("full",),
      command=(
         "uv",
         "run",
         "python",
         "pyfluid_amrex/tools/validate_obstacle_masks.py",
         "--force",
         "--ranks",
         "3",
         "--n-cell",
         "512",
         "256",
         "--skip-plot",
      ),
   ),
   GateStep(
      name="ct_divb_compare",
      kind="feature",
      levels=("standard", "full"),
      command=("uv", "run", "python", "pyfluid_amrex/tools/validate_ct_divb.py", "--force", "--skip-plot"),
   ),
   GateStep(
      name="amr_global_tagging",
      kind="amr",
      levels=("standard", "full"),
      command=("uv", "run", "python", "pyfluid_amrex/tools/validate_amr_global_tagging.py", "--force"),
   ),
   GateStep(
      name="amr_dynamic_regrid",
      kind="amr",
      levels=("standard", "full"),
      command=(
         "uv",
         "run",
         "python",
         "pyfluid_amrex/tools/validate_amr_dynamic_regrid.py",
         "--force",
         "--ranks",
         "3",
      ),
   ),
   GateStep(
      name="double_mach_reflection",
      kind="reference",
      levels=("standard", "full"),
      command=(
         "uv",
         "run",
         "python",
         "pyfluid_amrex/tools/validate_double_mach_reflection.py",
         "--level",
         "quick",
         "--force",
         "--ranks",
         "3",
      ),
   ),
   GateStep(
      name="causal_open_boundary",
      kind="analytic",
      levels=("standard", "full"),
      command=("uv", "run", "python", "pyfluid_amrex/tools/validate_causal_policy.py", "--force", "--ranks", "3"),
   ),
   GateStep(
      name="boundary_tiling",
      kind="feature",
      levels=("standard", "full"),
      command=(
         "uv",
         "run",
         "python",
         "pyfluid_amrex/tools/validate_boundary_tiling.py",
      ),
   ),
   GateStep(
      name="multipatch_cartesian",
      kind="feature",
      levels=("full",),
      command=("uv", "run", "python", "pyfluid_amrex/tools/validate_multipatch.py", "--ranks", "3"),
   ),
   GateStep(
      name="production_multipatch_hydro_amr",
      kind="feature",
      levels=("standard", "full"),
      command=("uv", "run", "python", "pyfluid_amrex/tools/validate_production_multipatch.py", "--ranks", "3", "--force"),
   ),
   GateStep(
      name="mixed_patch_crawl",
      kind="feature",
      levels=("standard", "full"),
      command=("uv", "run", "python", "pyfluid_amrex/tools/validate_mixed_patch_crawl.py", "--ranks", "1", "--force", "--skip-plot"),
   ),
   GateStep(
      name="obstacle_wall_amr",
      kind="feature",
      command=("uv", "run", "python", "pyfluid_amrex/tools/validate_obstacle_wall.py", "--force", "--ranks", "3"),
   ),
   GateStep(
      name="solid_geometry_refinement",
      kind="feature",
      command=("uv", "run", "python", "pyfluid_amrex/tools/validate_solid_geometry_refinement.py", "--force"),
   ),
   GateStep(
      name="solid_volume_fraction",
      kind="feature",
      command=("uv", "run", "python", "pyfluid_amrex/tools/validate_solid_volume_fraction.py", "--force"),
   ),
   GateStep(
      name="pulsating_solid_geometry",
      kind="feature",
      command=(
         "uv",
         "run",
         "python",
         "pyfluid_amrex/tools/validate_pulsating_solid_geometry.py",
         "--force",
         "--skip-amr-evolution",
      ),
   ),
   GateStep(
      name="solid_sphere_geometry",
      kind="feature",
      command=("uv", "run", "python", "pyfluid_amrex/tools/validate_solid_sphere_geometry.py", "--force"),
   ),
   GateStep(
      name="dimensional_consistency",
      kind="feature",
      levels=("standard", "full"),
      command=("uv", "run", "python", "pyfluid_amrex/tools/validate_dimensional_consistency.py", "--force", "--ranks", "3"),
   ),
   GateStep(
      name="rank_reproducibility",
      kind="feature",
      levels=("standard", "full"),
      command=("uv", "run", "python", "pyfluid_amrex/tools/validate_rank_reproducibility.py", "--force"),
   ),
   GateStep(
      name="performance",
      kind="performance",
      levels=("full",),
      command=("uv", "run", "python", "pyfluid_amrex/tools/benchmark_pyfluid.py", "--force", "--allow-warn"),
   ),
   GateStep(
      name="codegen_contract",
      kind="feature",
      levels=("standard",),
      command=(
         "uv",
         "run",
         "python",
         "pyfluid_amrex/tools/validate_codegen_flux.py",
         "--skip-generated-comparison",
      ),
   ),
   GateStep(
      name="codegen_flux",
      kind="feature",
      levels=("full",),
      command=("uv", "run", "python", "pyfluid_amrex/tools/validate_codegen_flux.py"),
   ),
   GateStep(
      name="radiation",
      kind="feature",
      levels=("standard", "full"),
      command=(
         "uv", "run", "python", "pyfluid_amrex/tools/validate_radiation.py",
         "--force", "--ranks", "3",
      ),
   ),
   GateStep(
      name="curvilinear_geometry",
      kind="feature",
      levels=("standard", "full"),
      command=("uv", "run", "python", "pyfluid_amrex/tools/validate_curvilinear_geometry.py", "--force", "--ranks", "3"),
   ),
   GateStep(
      name="curvilinear_methods_cylindrical",
      kind="feature",
      levels=("standard", "full"),
      command=(
         "uv",
         "run",
         "python",
         "pyfluid_amrex/tools/validate_curvilinear_methods.py",
         "--level",
         "quick",
         "--coordinate",
         "cylindrical",
         "--run-root",
         "runs/current/curvilinear_methods/cylindrical",
         "--force",
         "--ranks",
         "3",
      ),
   ),
   GateStep(
      name="curvilinear_methods_spherical",
      kind="feature",
      levels=("standard", "full"),
      command=(
         "uv",
         "run",
         "python",
         "pyfluid_amrex/tools/validate_curvilinear_methods.py",
         "--level",
         "quick",
         "--coordinate",
         "spherical",
         "--run-root",
         "runs/current/curvilinear_methods/spherical",
         "--force",
         "--ranks",
         "3",
      ),
   ),
   GateStep(
      name="curvilinear_origin",
      kind="feature",
      levels=("standard", "full"),
      command=(
         "uv",
         "run",
         "python",
         "pyfluid_amrex/tools/validate_curvilinear_origin.py",
         "--level",
         "quick",
         "--force",
         "--ranks",
         "1",
      ),
   ),
   GateStep(
      name="curvilinear_subcycling_equivalence",
      kind="amr",
      levels=("standard", "full"),
      command=(
         "uv",
         "run",
         "python",
         "pyfluid_amrex/tools/validate_subcycling_benefit.py",
         "--curvilinear-equivalence",
         "--force",
         "--skip-plot",
         "--ranks",
         "3",
      ),
   ),
   GateStep(
      name="curvilinear_seeded_amr_equivalence",
      kind="amr",
      levels=("standard", "full"),
      command=(
         "uv",
         "run",
         "python",
         "pyfluid_amrex/tools/validate_curvilinear_cartesian_equivalence.py",
         "--nr",
         "32",
         "--angular-factor",
         "8",
         "--amr-max-level",
         "2",
         "--center-refine-radius",
         "0.55",
         "--final-time",
         "0.006",
         "--tag-threshold",
         "0.005",
         "--n-error-buf",
         "8",
         "--regrid-int",
         "100000",
         "--min-headroom",
         "0",
         "--skip-plot",
         "--force",
         "--ranks",
         "3",
      ),
   ),
   GateStep(
      name="curvilinear_regrid_freestream",
      kind="amr",
      levels=("standard", "full"),
      command=(
         "uv",
         "run",
         "python",
         "pyfluid_amrex/tools/validate_curvilinear_regrid_freestream.py",
         "--force",
         "--ranks",
         "3",
      ),
   ),
   GateStep(
      name="sod_exact",
      kind="analytic",
      levels=("standard", "full"),
      command=(
         "uv",
         "run",
         "python",
         "pyfluid_amrex/tools/validate_sod_exact.py",
         "--force",
         "--method-ladder",
         "--nx",
         "256",
         "--skip-plots",
      ),
   ),
   GateStep(
      name="source_uniform_acceleration",
      kind="analytic",
      levels=("standard", "full"),
      command=("uv", "run", "python", "pyfluid_amrex/tools/validate_sources.py", "--force"),
   ),
   GateStep(
      name="gravity",
      kind="analytic",
      levels=("standard", "full"),
      command=("uv", "run", "python", "pyfluid_amrex/tools/validate_gravity.py", "--force"),
   ),
   GateStep(
      name="brio_wu_reference",
      kind="reference",
      levels=("standard", "full"),
      command=(
         "uv",
         "run",
         "python",
         "pyfluid_amrex/tools/validate_brio_wu_reference.py",
         "--force",
         "--skip-plot",
         "--include-amr-ct",
      ),
   ),
   GateStep(
      name="eos_acoustic",
      kind="analytic",
      levels=("standard", "full"),
      command=(
         "uv",
         "run",
         "python",
         "pyfluid_amrex/tools/validate_eos_acoustic.py",
         "--force",
         "--resolutions",
         "32",
         "64",
         "128",
         "--skip-plots",
      ),
   ),
   GateStep(
      name="advection_medley",
      kind="analytic",
      levels=("standard", "full"),
      command=(
         "uv",
         "run",
         "python",
         "pyfluid_amrex/tools/validate_advection_medley.py",
         "--force",
         "--skip-plot",
      ),
   ),
   GateStep(
      name="characteristic_medley",
      kind="reference",
      levels=("standard", "full"),
      command=(
         "uv",
         "run",
         "python",
         "pyfluid_amrex/tools/validate_characteristic_medley.py",
         "--force",
         "--skip-plot",
      ),
   ),
   GateStep(
      name="reconstruction_convergence",
      kind="analytic",
      levels=("full",),
      command=(
         "uv",
         "run",
         "python",
         "pyfluid_amrex/tools/validate_reconstruction_convergence.py",
         "--force",
         "--skip-plot",
      ),
   ),
   GateStep(
      name="mhd_characteristic_medley",
      kind="reference",
      levels=("full",),
      command=(
         "uv",
         "run",
         "python",
         "pyfluid_amrex/tools/validate_mhd_characteristic_medley.py",
         "--force",
         "--include-amr",
         "--skip-plot",
      ),
   ),
   GateStep(
      name="dispersion_relations",
      kind="analytic",
      levels=("standard", "full"),
      command=("uv", "run", "python", "pyfluid_amrex/tools/validate_dispersion_relations.py", "--force", "--skip-plots"),
   ),
   GateStep(
      name="diffusion_amplitude",
      kind="analytic",
      levels=("standard", "full"),
      command=("uv", "run", "python", "pyfluid_amrex/tools/validate_diffusion_amplitude.py", "--force", "--skip-plot"),
   ),
   GateStep(
      name="rank_scaling",
      kind="performance",
      levels=("full",),
      command=("uv", "run", "python", "pyfluid_amrex/tools/benchmark_rank_scaling.py", "--force"),
   ),
   GateStep(
      name="production_rank_scaling",
      kind="performance",
      levels=("full",),
      command=(
         "uv",
         "run",
         "python",
         "pyfluid_amrex/tools/benchmark_rank_scaling.py",
         "--suite",
         "production",
         "--force",
      ),
   ),
   GateStep(
      name="subcycling_benefit",
      kind="performance",
      levels=("full",),
      command=("uv", "run", "python", "pyfluid_amrex/tools/validate_subcycling_benefit.py", "--force", "--skip-plot"),
   ),
   GateStep(
      name="hydro_cfl_edge",
      kind="stability",
      levels=("stability",),
      command=(
         "uv",
         "run",
         "python",
         "pyfluid_amrex/tools/scan_hydro_cfl_edge.py",
         "--force",
         "--cfl-values",
         "0.98,0.999,1.0,1.0001,1.001,1.01,1.05,1.1,1.2,1.3,1.4",
         "--steps",
         "1000",
      ),
   ),
   GateStep(
      name="hydro_subcycling_cfl_edge",
      kind="stability",
      levels=("stability",),
      command=(
         "uv",
         "run",
         "python",
         "pyfluid_amrex/tools/scan_hydro_cfl_edge.py",
         "--force",
         "--timesteps",
         "ssprk3",
         "--amr-modes",
         "amr_nosub,amr_spatial,amr_local_cfl",
         "--cfl-values",
         "0.8,0.9,0.98,1.0,1.02,1.05,1.1,1.2",
         "--steps",
         "500",
      ),
   ),
   GateStep(
      name="nonlinear_cfl_shock",
      kind="stability",
      levels=("stability",),
      command=(
         "uv",
         "run",
         "python",
         "pyfluid_amrex/tools/scan_cfl_stability.py",
         "--force",
         "--suite",
         "nonlinear",
         "--cfl-values",
         "0.8,0.9,0.98,1.0,1.02,1.05,1.1,1.2",
         "--require-pass-cfl",
         "0.8",
      ),
   ),
   GateStep(
      name="stability_matrix",
      kind="stability",
      levels=("stability",),
      command=(
         "uv",
         "run",
         "python",
         "pyfluid_amrex/tools/scan_cfl_stability.py",
         "--force",
         "--suite",
         "diffusion",
         "--suite",
         "subcycling",
         "--suite",
         "amr_metrics",
         "--suite",
         "ct",
         "--cfl-values",
         "0.8,0.9,0.98,1.0,1.02",
         "--require-pass-cfl",
         "0.9",
      ),
   ),
   GateStep(
      name="amr_validation",
      kind="amr",
      levels=("full",),
      command=("uv", "run", "python", "pyfluid_amrex/tools/validate_amr.py", "--suite", "all", "--force", "--restart-checks"),
   ),
]


def run_step(step: GateStep) -> dict[str, str]:
   LOG_DIR.mkdir(parents=True, exist_ok=True)
   log_path = LOG_DIR / f"{step.name}.log"
   env = dict(os.environ)
   env.setdefault("UV_CACHE_DIR", str(ROOT / ".uv-cache"))
   env.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib"))
   start = time.perf_counter()
   completed = subprocess.run(
      step.command,
      cwd=ROOT,
      env=env,
      text=True,
      stdout=subprocess.PIPE,
      stderr=subprocess.STDOUT,
      check=False,
   )
   elapsed = time.perf_counter() - start
   log_path.write_text(completed.stdout)
   return {
      "step": step.name,
      "kind": step.kind,
      "status": "PASS" if completed.returncode == 0 else "FAIL",
      "returncode": str(completed.returncode),
      "seconds": f"{elapsed:.6g}",
      "log": str(log_path.relative_to(ROOT)),
      "tail": " | ".join(completed.stdout.splitlines()[-6:]),
   }


def write_summary(rows: list[dict[str, str]]) -> None:
   write_tsv(SUMMARY, rows, ["step", "kind", "status", "returncode", "seconds", "log", "tail"])


def append_gate_history(rows: list[dict[str, str]]) -> None:
   append_validation_history(HISTORY, [{"suite": "amrex_gate", **row} for row in rows])


def selected_steps(level: str, kinds: list[str] | None, only: list[str] | None) -> list[GateStep]:
   known = {step.name for step in STEPS}
   if only:
      unknown = sorted(set(only) - known)
      if unknown:
         raise ValueError(f"unknown gate step(s): {', '.join(unknown)}")
      return [step for step in STEPS if step.name in set(only)]

   steps = [step for step in STEPS if level in step.levels]
   if kinds:
      wanted = set(kinds)
      steps = [step for step in steps if step.kind in wanted]
   return steps


def main() -> int:
   parser = argparse.ArgumentParser(description=__doc__)
   parser.add_argument("--level", choices=["quick", "standard", "full", "stability"], default="standard")
   parser.add_argument("--full", action="store_true", help="alias for --level full")
   parser.add_argument("--kind", action="append", choices=sorted({step.kind for step in STEPS}))
   parser.add_argument("--only", action="append", choices=[step.name for step in STEPS])
   parser.add_argument("--list-steps", action="store_true")
   args = parser.parse_args()
   level = "full" if args.full else args.level

   try:
      steps = selected_steps(level, args.kind, args.only)
   except ValueError as exc:
      raise SystemExit(str(exc)) from exc
   if not steps:
      raise SystemExit("no suite steps selected")

   if args.list_steps:
      for step in steps:
         print(f"{step.name}\t{step.kind}\t{','.join(step.levels)}")
      return 0

   if RUN_ROOT.exists():
      shutil.rmtree(RUN_ROOT)

   rows: list[dict[str, str]] = []
   for step in steps:
      print(f"[gate] {step.name}", flush=True)
      row = run_step(step)
      rows.append(row)
      write_summary(rows)
      print(f"   {row['status']} {row['seconds']}s log={row['log']}", flush=True)
      if row["status"] != "PASS":
         print(row["tail"], flush=True)
         break

   append_gate_history(rows)
   failed = [row for row in rows if row["status"] != "PASS"]
   print(f"summary={SUMMARY.relative_to(ROOT)}")
   print(f"history={HISTORY.relative_to(ROOT)}")
   print(f"recent_history={recent_history_path(HISTORY).relative_to(ROOT)}")
   return 1 if failed else 0


if __name__ == "__main__":
   raise SystemExit(main())
