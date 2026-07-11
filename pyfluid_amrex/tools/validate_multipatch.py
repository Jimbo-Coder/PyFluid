#!/usr/bin/env python3
"""Validate multipatch maps, IC sampling, ghost transfer, overlap, and AMR."""

from __future__ import annotations

import argparse
import os
import subprocess
import time
from pathlib import Path

from validation_common import ROOT, append_validation_history, run_mpi, write_tsv


RUN_ROOT = ROOT / "runs/current/multipatch"
LOG_DIR = RUN_ROOT / "logs"
SUMMARY = RUN_ROOT / "summary.tsv"
HISTORY = ROOT / "runs/current/validation_history.tsv"


def run_command(command: list[str], log_path: Path) -> tuple[int, float, str]:
   env = dict(os.environ)
   env.setdefault("OMP_NUM_THREADS", "1")
   start = time.perf_counter()
   completed = subprocess.run(
      command,
      cwd=ROOT,
      env=env,
      text=True,
      stdout=subprocess.PIPE,
      stderr=subprocess.STDOUT,
      check=False,
   )
   elapsed = time.perf_counter() - start
   log_path.parent.mkdir(parents=True, exist_ok=True)
   log_path.write_text(completed.stdout)
   tail = "\n".join(completed.stdout.strip().splitlines()[-8:])
   return completed.returncode, elapsed, tail


def build_targets(dims: list[int]) -> None:
   targets = [f"pyfluid_amrex_multipatch_check_{dim}d" for dim in dims]
   command = ["cmake", "--build", "build/pyfluid_amrex", "--target", *targets, "--parallel", "4"]
   returncode, _, tail = run_command(command, LOG_DIR / "build.log")
   if returncode != 0:
      raise SystemExit(f"multipatch check build failed:\n{tail}")


def run_checks(dims: list[int], ranks: int) -> list[dict[str, object]]:
   rows: list[dict[str, object]] = []
   for dim in dims:
      executable = ROOT / "build" / "pyfluid_amrex" / f"pyfluid_amrex_multipatch_check_{dim}d"
      log_path = LOG_DIR / f"multipatch_check_{dim}d.log"
      completed = run_mpi(executable, ranks=ranks)
      log_path.parent.mkdir(parents=True, exist_ok=True)
      log_path.write_text(completed.stdout)
      returncode = completed.returncode
      seconds = completed.seconds
      tail = "\n".join(completed.stdout.strip().splitlines()[-8:])
      status = "pass" if returncode == 0 else "fail"
      rows.append(
         {
            "kind": "multipatch",
            "suite": "multipatch",
            "step": f"multipatch_{dim}d",
            "case": f"multipatch_topology_transfer_{dim}d",
            "label": f"{dim}D multipatch map/grid/layout/sampling/ghost/evolution/AMR transfer check",
            "status": status,
            "reason": "" if status == "pass" else tail,
            "returncode": returncode,
            "seconds": f"{seconds:.6g}",
            "dim": dim,
            "input": "",
            "policy": "multipatch_exchange",
            "ranks": ranks,
            "log": str(log_path.relative_to(ROOT)),
            "tail": tail,
            "description": (
               "Thornburg/cylindrical mixed-patch map, grid, logical-layout, and IC sampling checks, plus non-overlapping and overlapping "
               "Cartesian splits, four-patch tiled layouts, and T-junction "
               "layouts fill neighbor ghosts, advect a localized packet across patch interfaces with exact "
               "single-domain agreement, evolve localized packets across aligned and overlapping mixed-resolution "
               "interfaces in both coarse-to-fine and fine-to-coarse directions with exact single-hierarchy "
               "agreement, and conserve through AMReX coarse/fine transfer for one refined patch, all refined "
               "patches, an overlapping refined patch, and a refined T-junction branch"
            ),
         }
      )
   return rows


def main() -> int:
   parser = argparse.ArgumentParser(description=__doc__)
   parser.add_argument("--dims", nargs="+", type=int, default=[2, 3], choices=[2, 3])
   parser.add_argument("--ranks", type=int, default=3)
   parser.add_argument("--no-build", action="store_true")
   args = parser.parse_args()

   dims = sorted(set(args.dims))
   RUN_ROOT.mkdir(parents=True, exist_ok=True)
   if not args.no_build:
      build_targets(dims)

   rows = run_checks(dims, args.ranks)
   fields = [
      "kind",
      "suite",
      "step",
      "case",
      "label",
      "status",
      "returncode",
      "seconds",
      "dim",
      "ranks",
      "log",
      "tail",
      "description",
      "reason",
   ]
   write_tsv(SUMMARY, rows, fields)
   append_validation_history(HISTORY, rows)

   failed = [row for row in rows if row["status"] != "pass"]
   for row in rows:
      print(
         f"{row['status']}\t{row['case']}\t"
         f"dim={row['dim']}\tranks={row['ranks']}\tseconds={row['seconds']}"
      )
   if failed:
      print(f"summary: {SUMMARY.relative_to(ROOT)}")
      return 1
   print(f"summary: {SUMMARY.relative_to(ROOT)}")
   return 0


if __name__ == "__main__":
   raise SystemExit(main())
