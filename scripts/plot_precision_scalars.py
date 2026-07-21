#!/usr/bin/env python3
import argparse
import csv
import json
import pathlib
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "plots" / "precision_scalars"

CASE = {
   "ic": 8,
   "recon": 2,
   "timestep": 3,
   "riemann": 0,
   "fluxct": True,
   "nx": 128,
   "ny": 128,
   "nsteps": 128,
   "store_every": 8,
   "verbose": False,
   "print_devices": False,
   "print_summary": False,
}
PRECISIONS = (("x64", True), ("float32", False))


def dtype_name(value):
   dtype = getattr(value, "dtype", None)
   return None if dtype is None else str(dtype)


def is_float_dtype(dtype_text):
   if dtype_text is None:
      return False
   try:
      return np.issubdtype(np.dtype(dtype_text), np.floating)
   except TypeError:
      return False


def relative_drift(series):
   series = np.asarray(series)
   denom = max(abs(float(series[0])), 1.0)
   return np.abs(series - series[0]) / denom


def worker_run(precision_name, outdir):
   sys.path.insert(0, str(ROOT))

   import jax
   import main_mhd2d_jax as mhd2d
   from pyfluid.config import SimulationConfig
   from pyfluid.diagnostics import momentum_drift_norm, scalar_health_summary, stored_min
   from pyfluid.state import D, P

   x64 = dict(PRECISIONS)[precision_name]
   config = SimulationConfig(
      **CASE,
      x64=x64,
      checkname=f"precision_scalars_{precision_name}",
   )
   state = mhd2d.run_simulation(config)

   diagnostics = np.asarray(state.diagnostics)
   ts = np.asarray(state.ts)
   dts = np.concatenate(([np.nan], np.asarray(state.dts)))
   mass_rel = relative_drift(diagnostics[:, D.MASS])
   energy_rel = relative_drift(diagnostics[:, D.ENERGY])
   momentum = momentum_drift_norm(diagnostics)
   min_rho = stored_min(state, P.RHO)
   min_pressure = stored_min(state, P.PRESS)

   rows = []
   for i, time_value in enumerate(ts):
      rows.append(
         {
            "precision": precision_name,
            "time": float(time_value),
            "dt": float(dts[i]),
            "mass": float(diagnostics[i, D.MASS]),
            "mass_rel_drift": float(mass_rel[i]),
            "energy": float(diagnostics[i, D.ENERGY]),
            "energy_rel_drift": float(energy_rel[i]),
            "momentum_drift": float(momentum[i]),
            "internal_energy": float(diagnostics[i, D.INTERNAL_ENERGY]),
            "kinetic_energy": float(diagnostics[i, D.KINETIC_ENERGY]),
            "magnetic_energy": float(diagnostics[i, D.MAGNETIC_ENERGY]),
            "divb": float(diagnostics[i, D.DIVB_L2]) if state.boolmag else 0.0,
            "min_rho": float(min_rho[i]),
            "min_pressure": float(min_pressure[i]),
         }
      )

   scalar_path = outdir / f"{precision_name}_scalars.csv"
   with scalar_path.open("w", newline="") as f:
      writer = csv.DictWriter(f, fieldnames=tuple(rows[0]))
      writer.writeheader()
      writer.writerows(rows)

   tracked = {}
   for name, value in vars(state).items():
      dtype_text = dtype_name(value)
      if dtype_text is not None:
         tracked[name] = dtype_text

   expected_float_dtype = "float64" if x64 else "float32"
   unexpected_float_arrays = {
      name: dtype_text
      for name, dtype_text in tracked.items()
      if is_float_dtype(dtype_text) and dtype_text != expected_float_dtype
   }
   key_arrays = {
      name: tracked.get(name)
      for name in (
         "Uvec",
         "Pvec",
         "Uvecs",
         "Pvecs",
         "diagnostics",
         "ts",
         "dts",
         "dUvec",
         "Fvec",
         "alphaintx",
         "cs2vec",
         "stress",
         "dPveclimy",
         "Uvecexacts",
         "Pvecexacts",
         "diagnostics_exact",
      )
   }

   summary = scalar_health_summary(state)
   report = {
      "precision": precision_name,
      "config_x64": config.x64,
      "jax_enable_x64": bool(jax.config.jax_enable_x64),
      "expected_float_dtype": expected_float_dtype,
      "key_arrays": key_arrays,
      "unexpected_float_arrays": unexpected_float_arrays,
      "precision_ok": not unexpected_float_arrays,
      "scalar_csv": str(scalar_path.relative_to(ROOT)),
      "summary": summary,
   }
   (outdir / f"{precision_name}_dtypes.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
   print(json.dumps(report, sort_keys=True))


def run_worker(precision_name, outdir):
   command = [
      sys.executable,
      str(pathlib.Path(__file__).resolve()),
      "--worker",
      precision_name,
      "--outdir",
      str(outdir),
   ]
   result = subprocess.run(command, cwd=ROOT, check=True, text=True, capture_output=True)
   if result.stderr:
      print(result.stderr, end="", file=sys.stderr)
   report = json.loads(result.stdout.strip().splitlines()[-1])
   summary = report["summary"]
   print(
      f"{precision_name:7s} "
      f"{summary['million_cell_updates_per_second']:8.3f} Mcells/s "
      f"dM={summary['mass_rel_drift_max']:.2e} "
      f"dE={summary['energy_rel_drift_max']:.2e} "
      f"divB={summary['divb_max']:.2e} "
      f"dtypes_ok={report['precision_ok']}"
   )
   return report


def read_scalar_csv(path):
   with path.open(newline="") as f:
      rows = list(csv.DictReader(f))
   return {
      key: np.asarray([float(row[key]) for row in rows])
      for key in rows[0]
      if key != "precision"
   }


def positive(series):
   return np.maximum(np.asarray(series), 1.0e-30)


def plot_scalars(reports, outdir):
   data = {
      precision_name: read_scalar_csv(outdir / f"{precision_name}_scalars.csv")
      for precision_name, _ in PRECISIONS
   }
   fig, axs = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
   axs = axs.ravel()

   for precision_name, _ in PRECISIONS:
      scalars = data[precision_name]
      summary = reports[precision_name]["summary"]
      label = f"{precision_name} ({summary['million_cell_updates_per_second']:.2f} Mcells/s)"
      t = scalars["time"]
      axs[0].semilogy(t, positive(scalars["mass_rel_drift"]), label=label)
      axs[1].semilogy(t, positive(scalars["energy_rel_drift"]), label=label)
      axs[2].semilogy(t, positive(scalars["momentum_drift"]), label=label)
      axs[3].semilogy(t, positive(scalars["divb"]), label=label)
      axs[4].plot(t, scalars["min_rho"], label=f"{precision_name} min rho")
      axs[4].plot(t, scalars["min_pressure"], linestyle="--", label=f"{precision_name} min P")
      axs[5].plot(t[1:], scalars["dt"][1:], label=label)

   titles = (
      "Mass Relative Drift",
      "Energy Relative Drift",
      "Total Momentum Drift",
      "normalized divB",
      "Primitive Floors",
      "Timestep",
   )
   ylabels = (
      "|M - M0| / max(|M0|, 1)",
      "|E - E0| / max(|E0|, 1)",
      "|Ptot - Ptot0|",
      "normalized divB",
      "minimum value",
      "dt",
   )
   for ax, title, ylabel in zip(axs, titles, ylabels):
      ax.set_title(title)
      ax.set_xlabel("time")
      ax.set_ylabel(ylabel)
      ax.grid(True, alpha=0.25)
      ax.legend(fontsize=8)

   fig.suptitle(
      "Precision scalar reductions: 128x128 MHD rotor, WENO-Z/RK4/LLF/flux-CT",
      fontsize=13,
   )
   outpath = outdir / "precision_scalar_timeseries.png"
   fig.savefig(outpath, dpi=170)
   plt.close(fig)
   return outpath


def write_summary_csv(reports, outdir):
   rows = []
   for precision_name, _ in PRECISIONS:
      report = reports[precision_name]
      summary = report["summary"]
      rows.append(
         {
            "precision": precision_name,
            "precision_ok": report["precision_ok"],
            "jax_enable_x64": report["jax_enable_x64"],
            "expected_float_dtype": report["expected_float_dtype"],
            "million_cell_updates_per_second": summary["million_cell_updates_per_second"],
            "mass_rel_drift_max": summary["mass_rel_drift_max"],
            "energy_rel_drift_max": summary["energy_rel_drift_max"],
            "momentum_drift_max": summary["momentum_drift_max"],
            "divb_max": summary["divb_max"],
            "min_rho": summary["min_rho"],
            "min_pressure": summary["min_pressure"],
         }
      )
   path = outdir / "precision_scalar_summary.csv"
   with path.open("w", newline="") as f:
      writer = csv.DictWriter(f, fieldnames=tuple(rows[0]))
      writer.writeheader()
      writer.writerows(rows)


def parse_args():
   parser = argparse.ArgumentParser(description="Compare x64 and float32 scalar reductions for the 2D JAX MHD path.")
   parser.add_argument("--outdir", type=pathlib.Path, default=OUTDIR, help=f"Output directory. Default: {OUTDIR}.")
   parser.add_argument("--worker", choices=tuple(name for name, _ in PRECISIONS), help=argparse.SUPPRESS)
   return parser.parse_args()


def main():
   args = parse_args()
   args.outdir.mkdir(parents=True, exist_ok=True)
   if args.worker:
      worker_run(args.worker, args.outdir)
      return

   reports = {
      precision_name: run_worker(precision_name, args.outdir)
      for precision_name, _ in PRECISIONS
   }
   dtype_report_path = args.outdir / "dtype_report.json"
   dtype_report_path.write_text(json.dumps(reports, indent=2, sort_keys=True) + "\n")
   write_summary_csv(reports, args.outdir)
   plot_path = plot_scalars(reports, args.outdir)
   print(f"Wrote scalar precision diagnostics to {args.outdir}")
   print(f"Plot: {plot_path}")
   print(f"Dtype report: {dtype_report_path}")


if __name__ == "__main__":
   main()
