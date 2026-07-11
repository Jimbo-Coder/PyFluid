#!/usr/bin/env python3
"""Validate curvilinear AMR regridding preserves a uniform freestream."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from validation_common import (
   CLOSED_POLICY,
   ROOT,
   append_validation_history,
   materialize_run_input,
   repo_relative,
   run_amrex,
   scalar_validation_row,
   write_tsv,
)


INPUT = ROOT / "pyfluid_amrex/inputs/cylindrical_spiral_radiation_rphi_2d_amr.in"
RUN_ROOT = ROOT / "runs/current/curvilinear_regrid_freestream"
SUMMARY = RUN_ROOT / "summary.tsv"
HISTORY = ROOT / "runs/current/validation_history.tsv"
FIELDS = [
   "kind",
   "suite",
   "case",
   "status",
   "reason",
   "returncode",
   "seconds",
   "dim",
   "ranks",
   "mass_rel_drift",
   "energy_rel_drift",
   "momentum_abs_drift",
   "quiet_change",
   "avg_mcells_per_s",
   "avg_sim_time_per_s",
   "run_dir",
]


OVERRIDES = {
   "pyfluid.run_name": "cylindrical_origin_static_regrid_freestream",
   "pyfluid.output_dir": "runs/current/curvilinear_regrid_freestream/cylindrical_origin_static_regrid_freestream",
   "mesh.coordinate": "cylindrical",
   "mesh.axes": "r phi",
   "mesh.patch_system": "single",
   "geometry.prob_lo": "0.0 0.0",
   "geometry.prob_hi": "1.10 6.283185307179586",
   "geometry.is_periodic": "0 1",
   "pyfluid.bc": "outflow",
   "pyfluid.bc_xlo": "origin",
   "pyfluid.bc_xhi": "outflow",
   "pyfluid.bc_ylo": "periodic",
   "pyfluid.bc_yhi": "periodic",
   "amr.n_cell": "32 64",
   "amr.max_level": "4",
   "amr.refinement_tensor": "auto",
   "amr.tag_metric": "auto",
   "amr.tag_threshold": "0.08",
   "amr.initial_refine": "box",
   "amr.initial_refine_levels": "3",
   "amr.initial_refine_fixed_levels": "0",
   "amr.initial_refine_box_lo": "0.0 0.0",
   "amr.initial_refine_box_hi": "0.35 6.283185307179586",
   "amr.min_feature_length_1": "0.0015",
   "amr.min_feature_length_2": "0.012",
   "amr.total_cell_budget": "1500000",
   "pyfluid.physics": "hydro",
   "pyfluid.ct": "off",
   "pyfluid.ic": "uniform",
   "pyfluid.eos": "ideal_gas",
   "pyfluid.rho0": "1.0",
   "pyfluid.pressure0": "0.5",
   "pyfluid.v1_0": "0.0",
   "pyfluid.v2_0": "0.0",
   "pyfluid.v3_0": "0.0",
   "pyfluid.source": "none",
   "pyfluid.recon": "wenoz",
   "pyfluid.riemann": "hllc",
   "pyfluid.timestep": "ssprk3",
   "pyfluid.ng": "3",
   "pyfluid.cfl": "0.9",
   "pyfluid.final_time": "-1.0",
   "pyfluid.max_step": "8",
   "pyfluid.scalar_int": "1",
   "pyfluid.plot_int": "1",
   "pyfluid.checkpoint_int": "-1",
   "pyfluid.aux_gfs": "none",
   "pyfluid.aux_gf_compute_int": "-1",
   "pyfluid.aux_gf_int": "-1",
   "pyfluid.diffusion": "none",
   "pyfluid.gravity": "none",
}


def run_quiet_check(run_dir: Path, tolerance: float) -> tuple[int, str, float]:
   command = [
      sys.executable,
      str(ROOT / "pyfluid_amrex/tools/validate_causal_quiescence.py"),
      str(run_dir),
      "--input",
      str(run_dir / "run.in"),
      "--output",
      str(run_dir / "freestream_quiescence.tsv"),
      "--plot",
      str(run_dir / "freestream_quiescence.png"),
      "--tolerance",
      f"{tolerance:.17g}",
   ]
   completed = subprocess.run(
      command,
      cwd=ROOT,
      text=True,
      stdout=subprocess.PIPE,
      stderr=subprocess.STDOUT,
      check=False,
   )
   quiet_change = float("nan")
   table = run_dir / "freestream_quiescence.tsv"
   if table.exists():
      import csv

      with table.open(newline="") as f:
         rows = list(csv.DictReader(f, delimiter="\t"))
      if rows:
         quiet_change = max(
            max(
               abs(float(row["max_rho_quiet"])),
               abs(float(row["max_pressure_quiet"])),
               abs(float(row["max_velocity_quiet"])),
            )
            for row in rows
         )
   return completed.returncode, completed.stdout, quiet_change


def main() -> int:
   parser = argparse.ArgumentParser(description=__doc__)
   parser.add_argument("--force", action="store_true")
   parser.add_argument("--ranks", type=int, default=3)
   parser.add_argument("--timeout", type=int, default=120)
   parser.add_argument("--tolerance", type=float, default=1.0e-10)
   args = parser.parse_args()

   run_dir = ROOT / OVERRIDES["pyfluid.output_dir"]
   if args.force and run_dir.exists():
      shutil.rmtree(run_dir)
   input_path = materialize_run_input(INPUT, run_dir / "run.in", OVERRIDES)

   result = run_amrex(
      dim=2,
      input_path=input_path,
      ranks=args.ranks,
      timeout=args.timeout,
   )
   run_dir.mkdir(parents=True, exist_ok=True)
   (run_dir / "run.log").write_text(result.stdout)
   if result.returncode == 0 and not (run_dir / "run.in").exists():
      raise RuntimeError(f"solver did not write canonical run configuration: {run_dir / 'run.in'}")

   row = scalar_validation_row(
      run_dir,
      case="cylindrical_origin_static_regrid_freestream",
      input_file=run_dir / "run.in",
      policy=CLOSED_POLICY,
      mhd=False,
      dim=2,
      ranks=args.ranks,
      base_input=repo_relative(INPUT),
      kind="amr",
   )
   quiet_code, quiet_out, quiet_change = run_quiet_check(run_dir, args.tolerance)
   (run_dir / "freestream_quiescence.log").write_text(quiet_out)

   failures: list[str] = []
   if result.returncode != 0:
      failures.append(f"run returncode={result.returncode}")
   if row["status"] != "PASS":
      failures.append(str(row.get("reason", "")))
   if quiet_code != 0:
      failures.append(f"quiet_change={quiet_change:.3e} > {args.tolerance:.3e}")

   out = {
      **row,
      "suite": "curvilinear_regrid_freestream",
      "status": "PASS" if not failures else "FAIL",
      "reason": "; ".join(reason for reason in failures if reason),
      "returncode": result.returncode if result.returncode != 0 else quiet_code,
      "seconds": f"{result.seconds:.6g}",
      "quiet_change": quiet_change,
      "run_dir": repo_relative(run_dir),
   }
   write_tsv(SUMMARY, [out], FIELDS)
   append_validation_history(HISTORY, [out])
   print(
      f"{out['status']} curvilinear_regrid_freestream: "
      f"quiet_change={quiet_change:.3e} tol={args.tolerance:.3e} "
      f"run={repo_relative(run_dir)}"
   )
   return 0 if out["status"] == "PASS" else 1


if __name__ == "__main__":
   raise SystemExit(main())
