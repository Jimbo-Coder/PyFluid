#!/usr/bin/env python3
import argparse
import csv
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numba

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pyfluid_numba import (
   DEFAULT_THREADS,
   HydroRunConfig,
   bc_tuple_name,
   hydro_initial_condition_name,
   limiter_name,
   recon_name,
   recon_vars_name,
   riemann_name,
   run_hydro_case,
   timestep_name,
)


OUTDIR = ROOT / "plots" / "numba_benchmarks"

MAX_SCALING_THREADS = 4
DEFAULT_THREAD_COUNTS = (1, 2, min(DEFAULT_THREADS, MAX_SCALING_THREADS))
FULL_THREAD_COUNTS = tuple(range(1, min(DEFAULT_THREADS, MAX_SCALING_THREADS) + 1))
SIZES = (256, 512, 1024)
STEPS = 64


def run_case(size, threads):
   return run_hydro_case(HydroRunConfig(nx=size, ny=size, steps=STEPS, threads=threads))


def benchmark_label(thread_counts):
   config = HydroRunConfig(nx=SIZES[-1], ny=SIZES[-1], steps=STEPS)
   sizes = ", ".join(f"{size}x{size}" for size in SIZES)
   threads = ", ".join(str(thread) for thread in thread_counts)
   return (
      f"Numba thread scaling test: sizes={sizes}, threads={threads}\n"
      f"IC={hydro_initial_condition_name(config.ic)}, non-magnetized, diffusion off, no CT; "
      f"{recon_name(config.recon).upper()} {recon_vars_name(config.recon_vars)}/"
      f"{limiter_name(config.limiter).upper()}/{riemann_name(config.riemann).upper()}/"
      f"{timestep_name(config.timestep).upper()}, BC={bc_tuple_name(config.bc)}, CFL={config.cfl:g}, {STEPS} steps, float64"
   )


def with_scaling_metrics(rows):
   rows = [dict(row) for row in rows]
   for size in sorted({row["nx"] for row in rows}):
      size_rows = [row for row in rows if row["nx"] == size]
      base_rate = next(row["million_cell_updates_per_second"] for row in size_rows if row["threads"] == 1)
      for row in size_rows:
         row["speedup_vs_1_thread"] = row["million_cell_updates_per_second"] / base_rate
         row["parallel_efficiency_vs_1_thread"] = row["speedup_vs_1_thread"] / row["threads"]
   return rows


def write_csv(rows, outpath):
   with outpath.open("w", newline="") as f:
      writer = csv.DictWriter(f, fieldnames=tuple(rows[0]))
      writer.writeheader()
      writer.writerows(rows)


def plot_results(rows, thread_counts, outpath):
   thread_counts = tuple(thread for thread in thread_counts if thread <= MAX_SCALING_THREADS)
   fig, axs = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
   for size in SIZES:
      size_rows = [row for row in rows if row["nx"] == size and int(row["threads"]) in thread_counts]
      size_rows.sort(key=lambda row: row["threads"])
      threads = [row["threads"] for row in size_rows]
      rates = [row["million_cell_updates_per_second"] for row in size_rows]
      speedups = [row["speedup_vs_1_thread"] for row in size_rows]
      efficiencies = [row["parallel_efficiency_vs_1_thread"] for row in size_rows]
      axs[0].plot(threads, rates, marker="o", label=f"{size}x{size}")
      axs[1].plot(threads, speedups, marker="o", label=f"{size}x{size}")
      axs[2].plot(threads, efficiencies, marker="o", label=f"{size}x{size}")
   axs[1].plot(thread_counts, thread_counts, color="black", linestyle="--", linewidth=1.0, label="ideal")
   axs[2].axhline(1.0, color="black", linestyle="--", linewidth=1.0, label="ideal")
   axs[0].set_title("Numba hydro throughput")
   axs[0].set_xlabel("Numba threads")
   axs[0].set_ylabel("full-step throughput (M cells/s)")
   axs[1].set_title("Numba speedup")
   axs[1].set_xlabel("Numba threads")
   axs[1].set_ylabel("speedup vs 1 thread")
   axs[2].set_title("Numba efficiency")
   axs[2].set_xlabel("Numba threads")
   axs[2].set_ylabel("speedup / Numba threads")
   for ax in axs:
      ax.grid(True, alpha=0.25)
      ax.legend()
      ax.set_xticks(thread_counts)
      ax.set_xlim(min(thread_counts) - 0.15, max(thread_counts) + 0.15)
   fig.suptitle(benchmark_label(thread_counts), fontsize=12)
   fig.savefig(outpath, dpi=160, bbox_inches="tight")
   plt.close(fig)


def parse_args():
   parser = argparse.ArgumentParser(description="Benchmark Numba hydro throughput.")
   parser.add_argument("--full", action="store_true", help=f"Run the full 1..{MAX_SCALING_THREADS} Numba thread sweep instead of the default 1/{DEFAULT_THREAD_COUNTS[-1]} check.")
   return parser.parse_args()


def main():
   args = parse_args()
   thread_counts = FULL_THREAD_COUNTS if args.full else DEFAULT_THREAD_COUNTS
   OUTDIR.mkdir(parents=True, exist_ok=True)
   print(f"Numba max thread pool: {numba.config.NUMBA_NUM_THREADS}")
   rows = []
   for size in SIZES:
      size_rows = []
      for threads in thread_counts:
         size_rows.append(run_case(size, threads))
      size_rows = with_scaling_metrics(size_rows)
      rows.extend(size_rows)
      for row in size_rows:
         print(
            f"{size:4d}x{size:<4d} threads={row['threads']:2d} "
            f"full-step={row['million_cell_updates_per_second']:8.2f} M cells/s "
            f"rhs-stage={row['rhs_stage_million_cell_updates_per_second']:8.2f} M cells/s "
            f"speedup={row['speedup_vs_1_thread']:.2f}x "
            f"eff={row['parallel_efficiency_vs_1_thread']:.2f} "
            f"dM={row['mass_rel_drift']:.2e} dE={row['energy_rel_drift']:.2e}",
            flush=True,
         )
   write_csv(rows, OUTDIR / "numba_hydro2d_benchmark.csv")
   plot_results(rows, thread_counts, OUTDIR / "numba_hydro2d_benchmark.png")
   print(f"Wrote Numba benchmark results to {OUTDIR}")


if __name__ == "__main__":
   main()
