#!/usr/bin/env python3
import argparse
import pathlib
import sys
from dataclasses import replace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import main_mhd2d_jax as mhd2d
from pyfluid.config import add_config_arguments, config_from_namespace
from pyfluid.diagnostics import (
   abs_relative_drift,
   add_health_arguments,
   health_criteria_from_args,
   momentum_drift_norm,
   scalar_health_summary,
   stored_min,
)
from pyfluid.state import D, P


def parse_values(text):
   return [float(item) for item in text.split(",") if item.strip()]


def make_config(args, cflr):
   config = config_from_namespace(args)
   return replace(
      config,
      cfl=cflr,
      checkname=f"{config.checkname}_plot_cfl_ic{config.ic}_r{config.recon}_t{config.timestep}_f{config.riemann}_{cflr:g}",
   )


def run_cases(args, criteria):
   cases = []
   for cflr in parse_values(args.values):
      try:
         state = mhd2d.run_simulation(make_config(args, cflr))
      except Exception as err:
         cases.append({"cfl": cflr, "state": None, "error": str(err), "summary": None, "reasons": [str(err)]})
      else:
         summary = scalar_health_summary(state)
         reasons = criteria.failure_reasons(summary)
         cases.append({"cfl": cflr, "state": state, "error": None, "summary": summary, "reasons": reasons})
   return cases


def plot_timeseries(cases, outpath):
   fig, axs = plt.subplots(3, 3, figsize=(15, 11), constrained_layout=True)
   axs = axs.ravel()
   for case in cases:
      cflr = case["cfl"]
      state = case["state"]
      label = f"CFL {cflr:g}"
      if state is None:
         continue
      ts = np.asarray(state.ts)
      diagnostics = np.asarray(state.diagnostics)
      axs[0].semilogy(ts, np.maximum(diagnostics[:, D.DIVB_L2], 1e-30), label=label)
      axs[1].plot(ts, stored_min(state, P.RHO), label=label)
      axs[2].plot(ts, stored_min(state, P.PRESS), label=label)
      axs[3].semilogy(ts, np.maximum(abs_relative_drift(diagnostics[:, D.MASS]), 1e-30), label=label)
      axs[4].semilogy(ts, np.maximum(abs_relative_drift(diagnostics[:, D.ENERGY]), 1e-30), label=label)
      axs[5].semilogy(ts, np.maximum(momentum_drift_norm(diagnostics), 1e-30), label=label)
      axs[6].plot(ts, diagnostics[:, D.MAGNETIC_ENERGY], label=label)
      axs[7].plot(ts, diagnostics[:, D.KINETIC_ENERGY], label=label)
      axs[8].plot(ts[1:], np.asarray(state.dts), label=label)

   titles = (
      "normalized divB",
      "min density",
      "min pressure",
      "|relative mass drift|",
      "|relative energy drift|",
      "|total momentum drift|",
      "magnetic energy",
      "kinetic energy",
      "stored dt",
   )
   ylabels = ("normalized divB", "rho", "P", "|dM/M0|", "|dE/E0|", "|dMom|", "magE", "KE", "dt")
   for ax, title, ylabel in zip(axs, titles, ylabels):
      ax.set_title(title)
      ax.set_xlabel("time")
      ax.set_ylabel(ylabel)
      ax.grid(True, alpha=0.25)
      ax.legend(fontsize=8)
   fig.suptitle("CFL diagnostics")
   fig.savefig(outpath, dpi=160)
   plt.close(fig)


def plot_final_fields(cases, outdir):
   for case in cases:
      cflr = case["cfl"]
      state = case["state"]
      if state is None:
         continue
      pvecs = np.asarray(state.Pvecs)
      finite_steps = np.where(np.all(np.isfinite(pvecs), axis=(1, 2, 3)))[0]
      if len(finite_steps) == 0:
         continue
      plot_it = int(finite_steps[-1])
      fields = ((P.RHO, "rho"), (P.PRESS, "pressure"))
      if state.boolmag:
         fields += ((P.BX, "Bx"), (P.BY, "By"))
      else:
         fields += ((P.VX, "vx"), (P.VY, "vy"))
      fig, axs = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
      for ax, (component, title) in zip(axs.ravel(), fields):
         data = pvecs[plot_it, component, ...]
         pcm = ax.imshow(data.T, origin="lower", extent=(state.x0, state.x1, state.y0, state.y1), aspect="equal")
         fig.colorbar(pcm, ax=ax, shrink=0.85)
         ax.set_title(title)
         ax.set_xlabel("x")
         ax.set_ylabel("y")
      label = "final" if plot_it == len(pvecs) - 1 else "last finite"
      fig.suptitle(f"{label.capitalize()} fields, CFL {cflr:g}, t={state.ts[plot_it]:.6g}")
      fig.savefig(outdir / f"final_fields_cfl_{cflr:g}.png", dpi=160)
      plt.close(fig)


def write_summary(cases, outpath):
   lines = [
      "cfl,status,time,dt_mean,million_cell_updates_per_second,min_rho,min_pressure,divb_max,mass_rel_drift_max,"
      "energy_rel_drift_max,momentum_drift_max,primitive_amplification,reasons"
   ]
   for case in cases:
      cflr = case["cfl"]
      summary = case["summary"]
      reasons = case["reasons"]
      if summary is None:
         lines.append(f"{cflr},FAIL,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,{'; '.join(reasons)!r}")
         continue
      status = "PASS" if not reasons else "FAIL"
      lines.append(
         f"{cflr},{status},{summary['time']:.16e},{summary['dt_mean']:.16e},"
         f"{summary['million_cell_updates_per_second']:.16e},"
         f"{summary['min_rho']:.16e},{summary['min_pressure']:.16e},{summary['divb_max']:.16e},"
         f"{summary['mass_rel_drift_max']:.16e},{summary['energy_rel_drift_max']:.16e},"
         f"{summary['momentum_drift_max']:.16e},{summary['primitive_amplification']:.16e},{'; '.join(reasons)!r}"
      )
   outpath.write_text("\n".join(lines) + "\n")


def parse_args():
   parser = argparse.ArgumentParser(description="Save diagnostic plots for a CFL sweep.")
   add_config_arguments(parser)
   parser.add_argument(
      "--values",
      default="1.0,2.0,2.5,2.53125,2.5390625",
      help="Comma-separated CFL values to plot. Default: '1.0,2.0,2.5,2.53125,2.5390625'.",
   )
   parser.add_argument(
      "--outdir",
      type=pathlib.Path,
      default=mhd2d.plotdir / "cfl_diagnostics",
      help=f"Directory for generated plots and summary CSV. Default: {mhd2d.plotdir / 'cfl_diagnostics'}.",
   )
   add_health_arguments(parser)
   return parser.parse_args()


def main():
   args = parse_args()
   criteria = health_criteria_from_args(args)
   args.outdir.mkdir(parents=True, exist_ok=True)
   cases = run_cases(args, criteria)
   plot_timeseries(cases, args.outdir / "timeseries.png")
   plot_final_fields(cases, args.outdir)
   write_summary(cases, args.outdir / "summary.csv")
   print(f"Wrote CFL diagnostics to {args.outdir}")


if __name__ == "__main__":
   main()
