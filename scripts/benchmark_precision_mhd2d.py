#!/usr/bin/env python3
import csv
import json
import pathlib
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "plots" / "precision_benchmarks"

SIZE = 128
CASE_NAMES = ("hydro_fast", "hydro_wenoz_rk4", "mhd_wenoz_rk4_ct")
PRECISIONS = (("x64", True), ("float32", False))


def run_precision_case(case_name, precision_name, x64):
   code = f"""
import json
from dataclasses import replace

from scripts.benchmark_mhd2d import CASES, run_case

case0 = next(case for case in CASES if case.name == {case_name!r})
case = replace(case0, changes={{**case0.changes, "x64": {x64!r}}})
row = run_case(case, {SIZE})
row["precision"] = {precision_name!r}
print(json.dumps(row, sort_keys=True))
"""
   result = subprocess.run(
      [sys.executable, "-c", code],
      cwd=ROOT,
      check=True,
      text=True,
      capture_output=True,
   )
   return json.loads(result.stdout.strip().splitlines()[-1])


def write_csv(rows, outpath):
   keys = tuple(rows[0])
   with outpath.open("w", newline="") as f:
      writer = csv.DictWriter(f, fieldnames=keys)
      writer.writeheader()
      writer.writerows(rows)


def plot_results(rows, outpath):
   fig, ax = plt.subplots(figsize=(8.5, 5), constrained_layout=True)
   width = 0.35
   xlocs = list(range(len(CASE_NAMES)))
   for offset, (precision_name, _) in zip((-width / 2, width / 2), PRECISIONS):
      ys = [
         next(row["million_cell_updates_per_second"] for row in rows if row["case"] == case_name and row["precision"] == precision_name)
         for case_name in CASE_NAMES
      ]
      ax.bar([x + offset for x in xlocs], ys, width=width, label=precision_name)
   ax.set_xticks(xlocs, CASE_NAMES, rotation=20, ha="right")
   ax.set_ylabel("M cells/s")
   ax.set_title(f"Precision benchmark, {SIZE}x{SIZE}")
   ax.grid(True, axis="y", alpha=0.25)
   ax.legend()
   fig.savefig(outpath, dpi=160)
   plt.close(fig)


def main():
   OUTDIR.mkdir(parents=True, exist_ok=True)
   rows = []
   for case_name in CASE_NAMES:
      for precision_name, x64 in PRECISIONS:
         row = run_precision_case(case_name, precision_name, x64)
         rows.append(row)
         print(
            f"{case_name:18s} {precision_name:7s} "
            f"{row['million_cell_updates_per_second']:8.3f} Mcells/s "
            f"dM={row['mass_rel_drift_max']:.2e} dE={row['energy_rel_drift_max']:.2e}"
         )
   write_csv(rows, OUTDIR / "precision_benchmark.csv")
   plot_results(rows, OUTDIR / "precision_benchmark.png")
   print(f"Wrote precision benchmark results to {OUTDIR}")


if __name__ == "__main__":
   main()
