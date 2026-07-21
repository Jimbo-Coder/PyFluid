#!/usr/bin/env python3
import argparse
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import main_mhd2d_jax as mhd2d
from pyfluid.config import add_config_arguments, config_from_namespace
from pyfluid.diagnostics import plot_field_snapshot, plot_scalar_timeseries, write_scalar_summary_csv


def parse_args():
   parser = argparse.ArgumentParser(description="Run one case and save compact scalar/field diagnostics.")
   add_config_arguments(parser)
   parser.add_argument(
      "--outdir",
      type=pathlib.Path,
      default=mhd2d.plotdir / "clean_diagnostics",
      help=f"Directory for scalar and field diagnostic plots. Default: {mhd2d.plotdir / 'clean_diagnostics'}.",
   )
   return parser.parse_args()


def main():
   args = parse_args()
   args.outdir.mkdir(parents=True, exist_ok=True)
   state = mhd2d.run_simulation(config_from_namespace(args))
   plot_scalar_timeseries(state, args.outdir / "scalar_timeseries.png")
   plot_field_snapshot(state, args.outdir / "final_fields.png")
   write_scalar_summary_csv(state, args.outdir / "summary.csv")
   print(f"Wrote diagnostics to {args.outdir}")


if __name__ == "__main__":
   main()
