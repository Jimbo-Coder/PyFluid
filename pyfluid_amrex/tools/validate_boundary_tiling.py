#!/usr/bin/env python3
"""Validate ownership-safe tiled boundary ghost fills in 2D and 3D."""

from __future__ import annotations

import argparse
import os
import subprocess
import time

from validation_common import (
   ROOT,
   append_validation_history,
   build_cached_target,
   write_tsv,
)


RUN_ROOT = ROOT / "runs/current/boundary_tiling"
LOG_DIR = RUN_ROOT / "logs"
SUMMARY = RUN_ROOT / "summary.tsv"
HISTORY = ROOT / "runs/current/validation_history.tsv"
PASS_MARKER = "PASS boundary grown-tile ownership check"


def run_check(dim: int) -> dict[str, object]:
   target = f"pyfluid_amrex_boundary_tiling_check_{dim}d"
   log_path = LOG_DIR / f"{target}.log"
   started = time.perf_counter()
   try:
      executable = build_cached_target(target)
      env = dict(os.environ)
      env.setdefault("OMP_NUM_THREADS", "2")
      completed = subprocess.run(
         (str(executable),),
         cwd=ROOT,
         env=env,
         check=False,
         stdout=subprocess.PIPE,
         stderr=subprocess.STDOUT,
         text=True,
      )
      output = completed.stdout
      failures = []
      if completed.returncode != 0:
         failures.append(f"returncode={completed.returncode}")
      if PASS_MARKER not in output:
         failures.append("missing PASS marker")
      returncode = completed.returncode
   except RuntimeError as error:
      output = f"{error}\n"
      failures = ["cached target build failed"]
      returncode = 1

   seconds = time.perf_counter() - started
   LOG_DIR.mkdir(parents=True, exist_ok=True)
   log_path.write_text(output)
   tail = " | ".join(output.strip().splitlines()[-8:])
   status = "PASS" if not failures else "FAIL"
   return {
      "kind": "feature",
      "suite": "boundary_tiling",
      "step": f"boundary_tiling_{dim}d",
      "case": f"boundary_tiling_{dim}d",
      "label": f"{dim}D tiled boundary ghost ownership",
      "status": status,
      "reason": "; ".join(failures),
      "returncode": returncode,
      "seconds": f"{seconds:.6g}",
      "dim": dim,
      "policy": "kernel_ownership",
      "ranks": 1,
      "log": str(log_path.relative_to(ROOT)),
      "tail": tail,
      "description": (
         "Forced small CPU tiles exercise ordinary, patch-union reflecting, "
         "relativistic, and constrained-transport boundary ghost writes"
      ),
   }


def main() -> int:
   parser = argparse.ArgumentParser(description=__doc__)
   parser.add_argument(
      "--dims", nargs="+", type=int, choices=(2, 3), default=(2, 3)
   )
   args = parser.parse_args()

   rows = [run_check(dim) for dim in sorted(set(args.dims))]
   fields = [
      "kind",
      "suite",
      "step",
      "case",
      "label",
      "status",
      "reason",
      "returncode",
      "seconds",
      "dim",
      "policy",
      "ranks",
      "log",
      "tail",
      "description",
   ]
   write_tsv(SUMMARY, rows, fields)
   append_validation_history(HISTORY, rows)

   for row in rows:
      print(
         f"{row['status']}\t{row['case']}\t"
         f"seconds={row['seconds']}\tlog={row['log']}"
      )
   print(f"summary: {SUMMARY.relative_to(ROOT)}")
   return 1 if any(row["status"] != "PASS" for row in rows) else 0


if __name__ == "__main__":
   raise SystemExit(main())
