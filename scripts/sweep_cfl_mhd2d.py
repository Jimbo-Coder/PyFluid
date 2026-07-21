#!/usr/bin/env python3
import argparse
import pathlib
import sys
import traceback
from dataclasses import replace

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import main_mhd2d_jax as mhd2d
from pyfluid.config import add_config_arguments, config_from_namespace
from pyfluid.diagnostics import add_health_arguments, health_criteria_from_args, scalar_health_summary


def parse_values(text):
   return [float(item) for item in text.split(",") if item.strip()]


def make_config(args, cflr):
   config = config_from_namespace(args)
   return replace(
      config,
      cfl=cflr,
      checkname=f"{config.checkname}_cfl_ic{config.ic}_r{config.recon}_t{config.timestep}_f{config.riemann}_{cflr:g}",
   )


def run_case(args, criteria, cflr):
   state = mhd2d.run_simulation(make_config(args, cflr))
   summary = scalar_health_summary(state)
   reasons = criteria.failure_reasons(summary)
   return {"passed": len(reasons) == 0, "reasons": reasons, **summary}


def print_result(cflr, result):
   status = "PASS" if result["passed"] else "FAIL"
   print(
      f"{status} cfl={cflr:.6g} "
      f"t={result['time']:.6e} dt_mean={result['dt_mean']:.6e} "
      f"Mcells/s={result['million_cell_updates_per_second']:.3f} "
      f"min_rho={result['min_rho']:.6e} min_P={result['min_pressure']:.6e} "
      f"dMrel={result['mass_rel_drift_max']:.3e} dErel={result['energy_rel_drift_max']:.3e} "
      f"dP={result['momentum_drift_max']:.3e} divB={result['divb_max']:.3e} "
      f"amp={result['primitive_amplification']:.3e}"
   )
   for reason in result.get("reasons", ()):
      print(f"  {reason}")


def try_case(args, criteria, cflr):
   try:
      result = run_case(args, criteria, cflr)
   except Exception as err:
      result = {
         "passed": False,
         "time": np.nan,
         "dt_mean": np.nan,
         "million_cell_updates_per_second": np.nan,
         "min_rho": np.nan,
         "min_pressure": np.nan,
         "mass_rel_drift_max": np.nan,
         "energy_rel_drift_max": np.nan,
         "momentum_drift_max": np.nan,
         "divb_max": np.nan,
         "primitive_amplification": np.nan,
         "reasons": [str(err)],
         "error": str(err),
      }
      if args.tracebacks:
         traceback.print_exc()
   print_result(cflr, result)
   if "error" in result:
      print(f"  error: {result['error']}")
   return result


def bisect_failure(args, criteria):
   low = args.low
   high = args.high
   low_result = try_case(args, criteria, low)
   high_result = try_case(args, criteria, high)
   if not low_result["passed"]:
      raise SystemExit(f"low CFL {low} failed; choose a lower bracket")
   if high_result["passed"]:
      raise SystemExit(f"high CFL {high} passed; choose a higher bracket")
   for _ in range(args.bisect_steps):
      mid = 0.5*(low + high)
      result = try_case(args, criteria, mid)
      if result["passed"]:
         low = mid
      else:
         high = mid
   print(f"BRACKET stable<={low:.8g} unstable>={high:.8g}")


def parse_args():
   parser = argparse.ArgumentParser(description="Sweep CFL stability for the 2D JAX MHD solver.")
   add_config_arguments(parser)
   parser.add_argument("--values", default="0.45,0.7,0.9,1.0,1.1",
                       help="Comma-separated CFL values to test. Default: '0.45,0.7,0.9,1.0,1.1'.")
   parser.add_argument("--bisect", action="store_true", help="Bisect between --low and --high after testing that low passes and high fails.")
   parser.add_argument("--low", type=float, default=0.5, help="Stable CFL bracket endpoint for --bisect. Default: 0.5.")
   parser.add_argument("--high", type=float, default=1.5, help="Unstable CFL bracket endpoint for --bisect. Default: 1.5.")
   parser.add_argument("--bisect-steps", type=int, default=8, help="Bisection iterations after checking --low and --high. Default: 8.")
   add_health_arguments(parser)
   parser.add_argument("--tracebacks", action="store_true", help="Print Python tracebacks for failed CFL cases. Default: False.")
   return parser.parse_args()


def main():
   args = parse_args()
   criteria = health_criteria_from_args(args)
   if args.bisect:
      bisect_failure(args, criteria)
   else:
      for cflr in parse_values(args.values):
         try_case(args, criteria, cflr)


if __name__ == "__main__":
   main()
