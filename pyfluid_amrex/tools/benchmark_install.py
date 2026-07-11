#!/usr/bin/env python3
"""Smoke benchmark for the local AMReX install.

This times AMReX's installed 2D advection example. It is not a PyFluid MHD
benchmark; it only checks MPI/OpenMP runtime behavior for the local AMReX setup.
"""

from __future__ import annotations

import argparse
import csv
import os
import tempfile
from pathlib import Path

from validation_common import materialize_run_input, run_mpi


DEFAULT_EXE = Path("/private/tmp/amrex-omp-test/install_test_2d")
DEFAULT_INPUTS = Path("/Users/mrizzo/src/amrex/Tests/Amr/Advection_AmrCore/Exec/inputs")
DEFAULT_OUTPUT = Path("runs/current/amrex_install_scaling.tsv")


def parse_layouts(text: str) -> list[tuple[int, int]]:
   layouts: list[tuple[int, int]] = []
   for item in text.split(","):
      ranks_text, threads_text = item.lower().split("x", 1)
      layouts.append((int(ranks_text), int(threads_text)))
   return layouts


def run_case(exe: Path, inputs: Path, ranks: int, threads: int, nx: int, ny: int, steps: int) -> dict[str, object]:
   env = os.environ.copy()
   env["OMP_NUM_THREADS"] = str(threads)
   env.setdefault("OMP_PROC_BIND", "spread")
   env.setdefault("OMP_PLACES", "cores")
   env.setdefault("AMREX_CUDA_ERROR_CAPTURE_THIS", "0")
   with tempfile.TemporaryDirectory(prefix=f"amrex_np{ranks}_t{threads}_", dir="/private/tmp") as tmpdir:
      input_path = materialize_run_input(
         inputs,
         Path(tmpdir) / "run.in",
         {
            "max_step": str(steps),
            "amr.n_cell": f"{nx} {ny} 8",
            "amr.max_level": "0",
            "amr.max_grid_size": "32",
            "amr.blocking_factor_x": "8",
            "amr.blocking_factor_y": "8",
            "amr.plot_int": "-1",
            "amr.chk_int": "-1",
            "amr.v": "0",
            "adv.do_reflux": "0",
         },
      )
      completed = run_mpi(
         exe,
         [input_path],
         ranks=ranks,
         cwd=tmpdir,
         extra_env=env,
      )
      elapsed = completed.seconds

   cell_updates = float(nx * ny * steps)
   return {
      "ranks": ranks,
      "omp_threads": threads,
      "total_workers": ranks * threads,
      "nx": nx,
      "ny": ny,
      "steps": steps,
      "elapsed_s": elapsed,
      "mcells_s": cell_updates / elapsed / 1.0e6,
      "returncode": completed.returncode,
      "tail": "\n".join(completed.stdout.splitlines()[-8:]),
   }


def main() -> None:
   parser = argparse.ArgumentParser(description=__doc__)
   parser.add_argument("--exe", type=Path, default=DEFAULT_EXE)
   parser.add_argument("--inputs", type=Path, default=DEFAULT_INPUTS)
   parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
   parser.add_argument("--layouts", default="1x1,1x4,2x2,4x1,5x1,8x1,10x1")
   parser.add_argument("--nx", type=int, default=1024)
   parser.add_argument("--ny", type=int, default=1024)
   parser.add_argument("--steps", type=int, default=100)
   parser.add_argument("--repeats", type=int, default=1)
   args = parser.parse_args()

   rows = []
   for ranks, threads in parse_layouts(args.layouts):
      layout_rows = []
      for repeat in range(args.repeats):
         row = run_case(args.exe, args.inputs, ranks, threads, args.nx, args.ny, args.steps)
         row["repeat"] = repeat
         rows.append(row)
         layout_rows.append(row)
         if row["returncode"] != 0:
            print(f"{ranks} ranks x {threads} threads repeat {repeat}: FAILED")
            print(row["tail"])
      good = [row for row in layout_rows if row["returncode"] == 0]
      if good:
         avg_mcells_s = sum(float(row["mcells_s"]) for row in good) / len(good)
         avg_elapsed = sum(float(row["elapsed_s"]) for row in good) / len(good)
         print(
            f"{ranks} ranks x {threads} threads: "
            f"{avg_mcells_s:.2f} Mcells/s avg, elapsed={avg_elapsed:.3f}s avg over {len(good)}"
         )

   args.output.parent.mkdir(parents=True, exist_ok=True)
   fieldnames = [
      "ranks",
      "omp_threads",
      "total_workers",
      "repeat",
      "nx",
      "ny",
      "steps",
      "elapsed_s",
      "mcells_s",
      "returncode",
   ]
   with args.output.open("w", newline="") as f:
      writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
      writer.writeheader()
      for row in rows:
         writer.writerow({name: row[name] for name in fieldnames})
   print(f"wrote {args.output}")


if __name__ == "__main__":
   main()
