#!/usr/bin/env python3
"""Measure MPI rank scaling for one identical AMReX initial-data solve."""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

from validation_common import (
   ROOT,
   materialize_run_input,
   repo_relative,
   run_amrex,
   write_tsv,
)


BASE_INPUT = ROOT / "pyfluid_amrex/inputs/rotating_star_xcts_3d.in"
RUN_ROOT = ROOT / "runs/current/initial_data_scaling"
RESULTS = RUN_ROOT / "results.tsv"
SOLVE_PATTERN = re.compile(
   r"xcts initial data: matter_iterations=(?P<matter>[0-9]+) "
   r"relative_matter_update=(?P<update>[0-9.eE+-]+) "
   r"coupled_iterations=(?P<coupled>[0-9]+)"
)


def main() -> int:
   parser = argparse.ArgumentParser(description=__doc__)
   parser.add_argument("--ranks", type=int, nargs="+", default=(1, 2, 3))
   parser.add_argument("--cells", type=int, default=24)
   parser.add_argument("--max-grid-size", type=int, default=8)
   parser.add_argument("--omega", type=float, default=0.001)
   parser.add_argument("--timeout", type=float, default=1800.0)
   parser.add_argument("--force", action="store_true")
   args = parser.parse_args()

   if any(rank < 1 for rank in args.ranks):
      raise SystemExit("--ranks entries must be positive")
   if args.cells < 8 or args.max_grid_size < 4:
      raise SystemExit("benchmark grids are too small")

   RUN_ROOT.mkdir(parents=True, exist_ok=True)
   rows: list[dict[str, object]] = []
   for ranks in args.ranks:
      name = f"n{args.cells}_omega{args.omega:g}_r{ranks}"
      run_dir = RUN_ROOT / name
      if args.force and run_dir.exists():
         shutil.rmtree(run_dir)
      run_dir.mkdir(parents=True, exist_ok=True)
      input_path = materialize_run_input(
         BASE_INPUT,
         run_dir / "input.in",
         {
            "amr.n_cell": f"{args.cells} {args.cells} {args.cells}",
            "amr.max_level": "0",
            "amr.max_grid_size": str(args.max_grid_size),
            "initial_data.angular_velocity": str(args.omega),
            "initial_data.rotation_continuation_steps": "1",
            "pyfluid.run_name": name,
            "pyfluid.output_dir": repo_relative(run_dir),
            "pyfluid.max_step": "0",
            "pyfluid.plot_int": "-1",
            "pyfluid.checkpoint_int": "-1",
            "pyfluid.scalar_diagnostics": "standard",
            "pyfluid.aux_gfs": "none",
            "pyfluid.aux_gf_compute_int": "-1",
            "pyfluid.aux_gf_int": "-1",
         },
      )
      completed = run_amrex(
         dim=3,
         input_path=input_path,
         ranks=ranks,
         timeout=args.timeout,
      )
      (run_dir / "run.log").write_text(completed.stdout)
      match = SOLVE_PATTERN.search(completed.stdout)
      if completed.returncode != 0 or match is None:
         raise RuntimeError(
            f"initial-data scaling case {name} failed; see "
            f"{repo_relative(run_dir / 'run.log')}"
         )
      matter_iterations = int(match.group("matter"))
      rows.append({
         "ranks": ranks,
         "cells_per_axis": args.cells,
         "max_grid_size": args.max_grid_size,
         "omega": args.omega,
         "matter_iterations": matter_iterations,
         "final_coupled_iterations": int(match.group("coupled")),
         "relative_matter_update": float(match.group("update")),
         "seconds": completed.seconds,
         "seconds_per_matter_iteration": completed.seconds / matter_iterations,
         "run_dir": repo_relative(run_dir),
      })

   reference_seconds = float(rows[0]["seconds"])
   reference_ranks = int(rows[0]["ranks"])
   for row in rows:
      ranks = int(row["ranks"])
      speedup = reference_seconds / float(row["seconds"])
      row["speedup_vs_first"] = speedup
      row["parallel_efficiency_vs_first"] = (
         speedup / (ranks / reference_ranks)
      )

   fields = [
      "ranks",
      "cells_per_axis",
      "max_grid_size",
      "omega",
      "matter_iterations",
      "final_coupled_iterations",
      "relative_matter_update",
      "seconds",
      "seconds_per_matter_iteration",
      "speedup_vs_first",
      "parallel_efficiency_vs_first",
      "run_dir",
   ]
   write_tsv(RESULTS, rows, fields)
   print(f"wrote {repo_relative(RESULTS)}")
   return 0


if __name__ == "__main__":
   raise SystemExit(main())
