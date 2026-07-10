#!/usr/bin/env python3
"""Check that compact AMReX runs reproduce across MPI decompositions."""

from __future__ import annotations

import argparse
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

from plot_amrex_boxed_movie import composite_slice  # noqa: E402
from plotfile_reader import latest_plotfile  # noqa: E402
from validation_common import mpi_prefix
from validation_common import (  # noqa: E402
   CLOSED_POLICY,
   DIFFUSIVE_POLICY,
   ScalarPolicy,
   append_validation_history,
   read_scalar_rows,
   repo_relative,
   scalar_validation_row,
   write_tsv,
)


BUILD = ROOT / "build" / "pyfluid_amrex"
INPUTS = ROOT / "pyfluid_amrex" / "inputs"
RUN_ROOT = ROOT / "runs" / "current" / "rank_reproducibility"
INPUT_ROOT = RUN_ROOT / "inputs"
RESULTS = RUN_ROOT / "results.tsv"
HISTORY = ROOT / "runs" / "current" / "validation_history.tsv"


@dataclass(frozen=True)
class Case:
   name: str
   dim: int
   base_input: str
   components: tuple[str, ...]
   overrides: tuple[str, ...]
   policy: ScalarPolicy
   mhd: bool
   tolerance_l2: float = 1.0e-11
   tolerance_linf: float = 1.0e-10


COMMON = (
   "pyfluid.cfl=0.9",
   "pyfluid.scalar_int=1",
   "pyfluid.plot_int=4",
   "pyfluid.checkpoint_int=-1",
   "pyfluid.aux_gfs=none",
   "pyfluid.aux_gf_int=-1",
   "pyfluid.aux_gf_compute_int=-1",
   "amr.max_grid_size=32",
   "amr.blocking_factor=8",
)


CASES = (
   Case(
      name="hydro_pressure_bump_amr",
      dim=2,
      base_input="pressure_bump_2d_amr.in",
      components=("rho", "pressure"),
      policy=CLOSED_POLICY,
      mhd=False,
      overrides=COMMON
      + (
         "pyfluid.max_step=4",
         "pyfluid.final_time=-1.0",
         "pyfluid.ng=2",
         "pyfluid.recon=pwl",
         "pyfluid.riemann=hllc",
         "pyfluid.timestep=ssprk3",
         "pyfluid.subcycling=on",
         "pyfluid.diffusion=none",
         "pyfluid.viscosity=0.0",
         "pyfluid.resistivity=0.0",
         "amr.n_cell=32 32",
         "amr.max_level=1",
         "amr.ref_ratio=2",
         "amr.regrid_int=2",
         "amr.tag_metric=auto",
         "amr.tag_threshold=0.08",
         "amr.state_interp=quartic",
      ),
   ),
   Case(
      name="mhd_kh_ct_amr_subcycling",
      dim=2,
      base_input="kelvin_helmholtz_mhd_2d_movie_ct.in",
      components=("rho", "pressure", "b1", "b2", "b3"),
      policy=DIFFUSIVE_POLICY,
      mhd=True,
      tolerance_l2=1.0e-10,
      tolerance_linf=1.0e-9,
      overrides=COMMON
      + (
         "pyfluid.max_step=4",
         "pyfluid.final_time=-1.0",
         "pyfluid.ng=3",
         "pyfluid.recon=pwl",
         "pyfluid.riemann=hlld",
         "pyfluid.timestep=ssprk3",
         "pyfluid.subcycling=on",
         "pyfluid.diffusion=explicit",
         "pyfluid.viscosity=1.5e-4",
         "pyfluid.resistivity=1.5e-4",
         "amr.n_cell=32 32",
         "amr.max_level=2",
         "amr.ref_ratio=2 2",
         "amr.regrid_int=2",
         "amr.tag_metric=auto",
         "amr.tag_threshold=0.12",
         "amr.state_interp=quartic-conservative",
         "amr.initial_refine_levels=1",
         "amr.initial_refine_fixed_levels=1",
         "amr.level_cell_budget_side=256",
         "amr.total_cell_budget=100000",
      ),
   ),
)


def executable(dim: int) -> Path:
   return BUILD / f"pyfluid_amrex_{dim}d"


def run_dir(case: Case, ranks: int) -> Path:
   return RUN_ROOT / case.name / f"ranks{ranks}"


def write_input(case: Case, ranks: int) -> tuple[Path, Path]:
   output_dir = run_dir(case, ranks)
   input_dir = INPUT_ROOT / case.name
   input_dir.mkdir(parents=True, exist_ok=True)
   output_dir.mkdir(parents=True, exist_ok=True)
   lines = (INPUTS / case.base_input).read_text().splitlines()
   lines.append("")
   lines.append("# validate_rank_reproducibility.py overrides")
   lines.append(f"pyfluid.run_name={case.name}_ranks{ranks}")
   lines.append(f"pyfluid.output_dir={repo_relative(output_dir)}")
   lines.extend(case.overrides)
   path = input_dir / f"ranks{ranks}.in"
   path.write_text("\n".join(lines) + "\n")
   return path, output_dir


def run_case(case: Case, ranks: int, force: bool, timeout: int) -> tuple[int, Path, str, Path]:
   output_dir = run_dir(case, ranks)
   if force and output_dir.exists():
      shutil.rmtree(output_dir)
   input_path, output_dir = write_input(case, ranks)
   if not force:
      try:
         if latest_plotfile(output_dir, "stateGF") is None:
            raise FileNotFoundError(f"no plotfiles in {output_dir}")
         return 0, output_dir, "", input_path
      except FileNotFoundError:
         pass
   command = [*mpi_prefix(ranks), str(executable(case.dim)), str(input_path)]
   completed = subprocess.run(
      command,
      cwd=ROOT,
      text=True,
      stdout=subprocess.PIPE,
      stderr=subprocess.STDOUT,
      timeout=timeout,
      env={**os.environ, "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "1")},
      check=False,
   )
   output_dir.mkdir(parents=True, exist_ok=True)
   (output_dir / "run.log").write_text(completed.stdout)
   return completed.returncode, output_dir, "\n".join(completed.stdout.splitlines()[-8:]), input_path


def compare_component(reference: Path, candidate: Path, component: str, max_pixels: int) -> tuple[float, float, float]:
   ref, _boxes, _tref, _ndim, _names, xlim, ylim = composite_slice(
      reference, component, 5.0 / 3.0, 2, 0.5, False, None, max_pixels, (0.0, 1.0), (0.0, 1.0)
   )
   cand, _boxes, _tcand, _ndim, _names, _xlim, _ylim = composite_slice(
      candidate, component, 5.0 / 3.0, 2, 0.5, False, None, max_pixels, xlim, ylim
   )
   mask = np.isfinite(ref) & np.isfinite(cand)
   if not np.any(mask):
      raise ValueError(f"{component}: no finite overlap")
   diff = cand - ref
   scale = max(float(np.sqrt(np.mean(ref[mask] ** 2))), 1.0e-300)
   rel_l1 = float(np.mean(np.abs(diff[mask]))) / scale
   rel_l2 = float(np.sqrt(np.mean(diff[mask] ** 2))) / scale
   rel_linf = float(np.max(np.abs(diff[mask]))) / scale
   return rel_l1, rel_l2, rel_linf


def average_positive(rows: list[dict[str, str]], key: str) -> float:
   values = []
   for row in rows:
      try:
         value = float(row.get(key, "nan"))
      except ValueError:
         value = math.nan
      if math.isfinite(value) and value > 0.0:
         values.append(value)
   return sum(values) / len(values) if values else math.nan


def scalar_row(case: Case, ranks: int, output_dir: Path, input_path: Path) -> dict[str, object]:
   return scalar_validation_row(
      output_dir,
      case=f"{case.name}_ranks{ranks}",
      input_file=input_path,
      checks=case.policy,
      mhd=case.mhd,
      dim=case.dim,
      ranks=ranks,
      base_input=case.base_input,
      kind="rank_reproducibility_scalar",
   )


def validate_case(case: Case, args: argparse.Namespace) -> list[dict[str, object]]:
   rows: list[dict[str, object]] = []
   run_info: dict[int, tuple[Path, Path]] = {}
   for ranks in (args.reference_ranks, args.candidate_ranks):
      code, output_dir, tail, input_path = run_case(case, ranks, args.force, args.timeout)
      if code != 0:
         rows.append({
            "case": case.name,
            "label": f"ranks{ranks}",
            "kind": "run",
            "status": "FAIL",
            "reason": f"rank {ranks} run failed: {tail}",
            "ranks": ranks,
            "run_dir": repo_relative(output_dir),
         })
         return rows
      rows.append(scalar_row(case, ranks, output_dir, input_path))
      run_info[ranks] = (output_dir, input_path)

   reference_dir = run_info[args.reference_ranks][0]
   reference = latest_plotfile(reference_dir, "stateGF")
   if reference is None:
      raise FileNotFoundError(f"no plotfiles in {reference_dir}")
   candidate_dir = run_info[args.candidate_ranks][0]
   candidate = latest_plotfile(candidate_dir, "stateGF")
   if candidate is None:
      raise FileNotFoundError(f"no plotfiles in {candidate_dir}")
   row: dict[str, object] = {
      "case": case.name,
      "label": f"{args.reference_ranks}_vs_{args.candidate_ranks}",
      "kind": "comparison",
      "status": "PASS",
      "reason": "",
      "reference_ranks": args.reference_ranks,
      "candidate_ranks": args.candidate_ranks,
      "reference_plotfile": repo_relative(reference),
      "candidate_plotfile": repo_relative(candidate),
      "max_rel_l1": 0.0,
      "max_rel_l2": 0.0,
      "max_rel_linf": 0.0,
      "worst_component": "",
      "tolerance_l2": case.tolerance_l2,
      "tolerance_linf": case.tolerance_linf,
   }
   failures = []
   for component in case.components:
      try:
         rel_l1, rel_l2, rel_linf = compare_component(reference, candidate, component, args.max_pixels)
      except ValueError as exc:
         failures.append(str(exc))
         continue
      row[f"{component}_rel_l1"] = rel_l1
      row[f"{component}_rel_l2"] = rel_l2
      row[f"{component}_rel_linf"] = rel_linf
      if rel_linf >= float(row["max_rel_linf"]):
         row["worst_component"] = component
      row["max_rel_l1"] = max(float(row["max_rel_l1"]), rel_l1)
      row["max_rel_l2"] = max(float(row["max_rel_l2"]), rel_l2)
      row["max_rel_linf"] = max(float(row["max_rel_linf"]), rel_linf)
   if float(row["max_rel_l2"]) > case.tolerance_l2:
      failures.append(f"max_rel_l2={float(row['max_rel_l2']):.3e} > {case.tolerance_l2:.3e}")
   if float(row["max_rel_linf"]) > case.tolerance_linf:
      failures.append(f"max_rel_linf={float(row['max_rel_linf']):.3e} > {case.tolerance_linf:.3e}")
   scalar_failures = [str(srow["reason"]) for srow in rows if srow.get("status") != "PASS"]
   failures.extend(reason for reason in scalar_failures if reason)
   row["status"] = "PASS" if not failures else "FAIL"
   row["reason"] = "; ".join(failures)

   reference_scalars = read_scalar_rows(run_info[args.reference_ranks][0])
   candidate_scalars = read_scalar_rows(run_info[args.candidate_ranks][0])
   row["reference_mcells_per_s"] = average_positive(reference_scalars, "mcells_per_s")
   row["candidate_mcells_per_s"] = average_positive(candidate_scalars, "mcells_per_s")
   row["reference_sim_time_per_s"] = average_positive(reference_scalars, "sim_time_per_s")
   row["candidate_sim_time_per_s"] = average_positive(candidate_scalars, "sim_time_per_s")
   rows.append(row)
   return rows


def write_results(rows: list[dict[str, object]]) -> None:
   fields = [
      "case",
      "label",
      "kind",
      "status",
      "reason",
      "ranks",
      "reference_ranks",
      "candidate_ranks",
      "max_rel_l1",
      "max_rel_l2",
      "max_rel_linf",
      "worst_component",
      "tolerance_l2",
      "tolerance_linf",
      "reference_mcells_per_s",
      "candidate_mcells_per_s",
      "reference_sim_time_per_s",
      "candidate_sim_time_per_s",
      "run_dir",
      "reference_plotfile",
      "candidate_plotfile",
   ]
   write_tsv(RESULTS, rows, fields)


def append_history(rows: list[dict[str, object]]) -> None:
   history_rows = []
   for row in rows:
      kind = str(row.get("kind", ""))
      if kind.endswith("scalar"):
         history_rows.append(row)
      elif kind == "comparison":
         history_rows.append(
            {
               "kind": "rank_reproducibility",
               "case": row["case"],
               "label": row["label"],
               "status": row["status"],
               "reason": row["reason"],
               "ranks": row["candidate_ranks"],
               "avg_mcells_per_s": row.get("candidate_mcells_per_s", ""),
               "avg_sim_time_per_s": row.get("candidate_sim_time_per_s", ""),
               "description": (
                  f"max_rel_l2={float(row['max_rel_l2']):.3e}; "
                  f"max_rel_linf={float(row['max_rel_linf']):.3e}; "
                  f"worst={row['worst_component']}"
               ),
            }
         )
   append_validation_history(HISTORY, history_rows)


def main() -> int:
   parser = argparse.ArgumentParser(description=__doc__)
   parser.add_argument("--force", action="store_true")
   parser.add_argument("--reference-ranks", type=int, default=1)
   parser.add_argument("--candidate-ranks", type=int, default=3)
   parser.add_argument("--max-pixels", type=int, default=256)
   parser.add_argument("--timeout", type=int, default=120)
   parser.add_argument("--only", action="append", choices=[case.name for case in CASES])
   parser.add_argument("--list-cases", action="store_true")
   args = parser.parse_args()

   selected = [case for case in CASES if not args.only or case.name in set(args.only)]
   if args.list_cases:
      for case in selected:
         print(case.name)
      return 0

   if args.force and RUN_ROOT.exists():
      shutil.rmtree(RUN_ROOT)
   RUN_ROOT.mkdir(parents=True, exist_ok=True)

   rows: list[dict[str, object]] = []
   for case in selected:
      case_rows = validate_case(case, args)
      rows.extend(case_rows)
      comparison = case_rows[-1]
      print(
         f"{comparison['status']:>4} {case.name}: "
         f"max_rel_l2={float(comparison.get('max_rel_l2', math.nan)):.3e} "
         f"max_rel_linf={float(comparison.get('max_rel_linf', math.nan)):.3e} "
         f"worst={comparison.get('worst_component', '')}"
      )
      if comparison["status"] != "PASS":
         print(f"     {comparison['reason']}")

   write_results(rows)
   append_history(rows)
   print(f"wrote {repo_relative(RESULTS)}")
   failed = [row for row in rows if row.get("status") != "PASS"]
   return 1 if failed else 0


if __name__ == "__main__":
   raise SystemExit(main())
