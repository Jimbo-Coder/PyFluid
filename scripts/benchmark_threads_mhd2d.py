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
OUTDIR = ROOT / "plots" / "thread_benchmarks"

CASE_NAME = "mhd_wenoz_rk4_ct"
SIZES = (128, 256, 512)
STEPS = 32
THREAD_COUNTS = (1, 2, 4)
MAX_SCALING_THREADS = 4


def run_thread_case(thread_count, size):
   xla_flags = f"--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads={thread_count}"
   code = f"""
import json
import os
from dataclasses import replace

os.environ["XLA_FLAGS"] = {xla_flags!r}

import jax

from scripts.benchmark_mhd2d import CASES, benchmark_config
import main_mhd2d_jax as mhd2d
from pyfluid.diagnostics import scalar_health_summary

case = next(case for case in CASES if case.name == {CASE_NAME!r})
config = replace(benchmark_config(case, {size}), nsteps={STEPS}, store_every={STEPS}, checkname=f"thread_benchmark_{{case.name}}_{size}_{thread_count}")
state = mhd2d.run_simulation(config)
summary = scalar_health_summary(state)
row = {{
   "case": case.name,
   "nx": config.nx,
   "ny": config.ny,
   "steps": config.nsteps,
   "store_every": config.store_every,
   "threads": {thread_count},
   "cpu_devices": len(jax.devices("cpu")),
   "x64": config.x64,
   "mhd": state.boolmag,
   "nvar": state.nvar,
   "run_time": summary["run_time"],
   "evolution_run_time": summary["evolution_run_time"],
   "cell_updates": summary["cell_updates"],
   "cell_updates_per_second": summary["cell_updates_per_second"],
   "million_cell_updates_per_second": summary["million_cell_updates_per_second"],
   "mass_rel_drift_max": summary["mass_rel_drift_max"],
   "energy_rel_drift_max": summary["energy_rel_drift_max"],
   "momentum_drift_max": summary["momentum_drift_max"],
   "min_rho": summary["min_rho"],
   "min_pressure": summary["min_pressure"],
   "divb_max": summary["divb_max"],
}}
row["threads"] = {thread_count}
row["xla_flags"] = {xla_flags!r}
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


def benchmark_label(rows, thread_counts):
   first = rows[0]
   sizes = ", ".join(f"{size}x{size}" for size in SIZES)
   threads = ", ".join(str(thread) for thread in thread_counts)
   return (
      f"JAX/XLA CPU thread scaling test: case={CASE_NAME}, sizes={sizes}, threads={threads}\n"
      f"MHD={first['mhd']}, CT case, WENO-Z/RK4/LLF, steps={STEPS}, x64={first['x64']}, metric=evolution Mcells/s"
   )


def plot_results(rows, outpath):
   thread_counts = tuple(thread for thread in THREAD_COUNTS if thread <= MAX_SCALING_THREADS)
   fig, axs = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
   for size in SIZES:
      size_rows = [row for row in rows if row["nx"] == size and int(row["threads"]) in thread_counts]
      size_rows.sort(key=lambda row: row["threads"])
      threads = [row["threads"] for row in size_rows]
      rates = [row["million_cell_updates_per_second"] for row in size_rows]
      baseline = rates[0]
      speedups = [rate / baseline for rate in rates]
      ideal = [thread / threads[0] for thread in threads]
      axs[0].plot(threads, rates, marker="o", label=f"{size}x{size}")
      axs[1].plot(threads, speedups, marker="o", label=f"{size}x{size}")
      if size == SIZES[-1]:
         axs[1].plot(threads, ideal, color="black", linestyle="--", linewidth=1.0, label="ideal")
   axs[0].set_xlabel("XLA intra-op CPU threads")
   axs[0].set_ylabel("throughput (Mcells/s)")
   axs[0].set_title(f"{CASE_NAME} throughput")
   axs[1].set_xlabel("XLA intra-op CPU threads")
   axs[1].set_ylabel("speedup vs 1 thread")
   axs[1].set_title(f"{CASE_NAME} scaling")
   for ax in axs:
      ax.grid(True, alpha=0.25)
      ax.legend()
      ax.set_xticks(thread_counts)
      ax.set_xlim(min(thread_counts) - 0.15, max(thread_counts) + 0.15)
   fig.suptitle(benchmark_label(rows, thread_counts), fontsize=11)
   fig.savefig(outpath, dpi=160)
   plt.close(fig)


def main():
   OUTDIR.mkdir(parents=True, exist_ok=True)
   rows = []
   for size in SIZES:
      for thread_count in THREAD_COUNTS:
         row = run_thread_case(thread_count, size)
         rows.append(row)
         print(
            f"{size:4d}x{size:<4d} threads={thread_count:2d} "
            f"{row['million_cell_updates_per_second']:8.3f} Mcells/s "
            f"runtime={row['evolution_run_time']:.3f}s",
            flush=True,
         )
   write_csv(rows, OUTDIR / "thread_benchmark.csv")
   plot_results(rows, OUTDIR / "thread_benchmark.png")
   print(f"Wrote thread benchmark results to {OUTDIR}")


if __name__ == "__main__":
   main()
