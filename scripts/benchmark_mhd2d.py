#!/usr/bin/env python3
import csv
import pathlib
import sys
from dataclasses import dataclass, replace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import main_mhd2d_jax as mhd2d
from pyfluid.config import SimulationConfig
from pyfluid.diagnostics import scalar_health_summary


SIZES = (64, 128)
STEPS = 128
OUTDIR = mhd2d.plotdir / "benchmarks"


@dataclass(frozen=True)
class BenchmarkCase:
   name: str
   changes: dict


CASES = (
   BenchmarkCase("hydro_fast", {"ic": 7, "recon": 0, "timestep": 0, "riemann": 0, "fluxct": False}),
   BenchmarkCase("hydro_wenoz_rk4", {"ic": 7, "recon": 2, "timestep": 3, "riemann": 0, "fluxct": False}),
   BenchmarkCase("mhd_wenoz_rk4_ct", {"ic": 8, "recon": 2, "timestep": 3, "riemann": 0, "fluxct": True}),
)


def benchmark_config(case, size):
   return replace(
      SimulationConfig(),
      nx=size,
      ny=size,
      nsteps=STEPS,
      store_every=STEPS,
      verbose=False,
      print_devices=False,
      print_summary=False,
      checkname=f"benchmark_{case.name}_{size}",
      **case.changes,
   )


def run_case(case, size):
   config = benchmark_config(case, size)
   state = mhd2d.run_simulation(config)
   summary = scalar_health_summary(state)
   return {
      "case": case.name,
      "nx": size,
      "ny": size,
      "steps": config.nsteps,
      "store_every": config.store_every,
      "ic": config.ic,
      "recon": config.recon,
      "timestep": config.timestep,
      "riemann": config.riemann,
      "fluxct": config.fluxct,
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
   }


def write_csv(rows, outpath):
   keys = tuple(rows[0])
   with outpath.open("w", newline="") as f:
      writer = csv.DictWriter(f, fieldnames=keys)
      writer.writeheader()
      writer.writerows(rows)


def plot_results(rows, outpath):
   fig, ax = plt.subplots(figsize=(8.5, 5), constrained_layout=True)
   for case in CASES:
      case_rows = [row for row in rows if row["case"] == case.name]
      xs = [row["nx"] for row in case_rows]
      ys = [row["million_cell_updates_per_second"] for row in case_rows]
      ax.plot(xs, ys, marker="o", label=case.name)
   ax.set_xlabel("grid size per dimension")
   ax.set_ylabel("throughput (Mcells/s)")
   ax.set_title(f"2D JAX solver benchmark: sizes={SIZES}, steps={STEPS}, metric=evolution Mcells/s")
   ax.grid(True, alpha=0.25)
   ax.legend()
   fig.savefig(outpath, dpi=160)
   plt.close(fig)


def main():
   OUTDIR.mkdir(parents=True, exist_ok=True)
   rows = []
   for size in SIZES:
      for case in CASES:
         row = run_case(case, size)
         rows.append(row)
         print(
            f"{case.name:18s} {size:4d}x{size:<4d} "
            f"{row['million_cell_updates_per_second']:8.3f} Mcells/s "
            f"dM={row['mass_rel_drift_max']:.2e} dE={row['energy_rel_drift_max']:.2e}"
         )
   write_csv(rows, OUTDIR / "benchmark.csv")
   plot_results(rows, OUTDIR / "benchmark.png")
   print(f"Wrote benchmark results to {OUTDIR}")


if __name__ == "__main__":
   main()
