#!/usr/bin/env python3
"""Regress Balsara-3 dynamic-regrid prolongation with subcycled AMR."""

from __future__ import annotations

import argparse
import math
import subprocess
import time

from validate_rmhd import (
   EXACT_BALSARA_CASES,
   HISTORY,
   INPUTS,
   RUN_ROOT,
   exact_balsara_comparison,
   relativistic_recovery_counts,
   rmhd_riemann_hll_fallback_count,
   run_case,
)
from validation_common import (
   ROOT,
   append_validation_history,
   build_cached_target,
   check_scalars,
   read_scalar_rows,
   repo_relative,
   write_tsv,
)


RUN_NAME = "balsara3_amr_prolongation"
RESULTS = RUN_ROOT / f"{RUN_NAME}.tsv"
MAXIMUM_NORMALIZED_COMPONENT_L1 = 4.5e-3
SCALAR_CONSERVATION_TOLERANCE = 1.0e-9
FORMER_FAILURE_STEP = 16


def run_captured_interpolation_check() -> None:
   target = "pyfluid_amrex_state_interpolation_check"
   check_binary = build_cached_target(target)
   check = subprocess.run(
      (str(check_binary),),
      cwd=ROOT,
      check=False,
      capture_output=True,
      text=True,
   )
   if check.returncode != 0:
      raise RuntimeError(
         f"{target} failed:\n{check.stdout}\n{check.stderr}"
      )
   pass_marker = "PASS relativistic conservative state interpolation check"
   if pass_marker not in check.stdout:
      raise RuntimeError(f"{target} did not report its PASS marker")


def case_spec() -> dict[str, object]:
   return next(spec for spec in EXACT_BALSARA_CASES if int(spec["case"]) == 3)


def overrides(spec: dict[str, object]) -> dict[str, str]:
   values = {
      "geometry.prob_lo": "-1.0",
      "geometry.prob_hi": "1.0",
      "geometry.is_periodic": "0",
      "amr.n_cell": "128",
      "amr.max_grid_size": "128",
      "amr.blocking_factor": "8",
      "amr.max_level": "9",
      "amr.ref_ratio": "2 2 2 2 2 2 2 2 2",
      "amr.regrid_mode": "synchronized",
      "amr.regrid_int": "2",
      "amr.n_error_buf": "4 4 4 4 4 4 4 4 4",
      "amr.grid_eff": "0.80",
      "amr.tag_metric": "lohner",
      "amr.tag_threshold": "0.08",
      "amr.min_feature_length": "0.001",
      "amr.level_cell_budget_side": "8192",
      "amr.total_cell_budget": "30000",
      "amr.initial_refine": "box",
      "amr.initial_refine_levels": "4",
      "amr.initial_refine_fixed_levels": "0",
      "amr.initial_refine_box_lo": "-0.08",
      "amr.initial_refine_box_hi": "0.08",
      "amr.state_interp": "coupled-conservative",
      "pyfluid.gamma": repr(spec["gamma"]),
      "pyfluid.final_time": repr(spec["time"]),
      "pyfluid.max_step": "10000",
      "pyfluid.ng": "3",
      "pyfluid.cfl": "0.9",
      "pyfluid.recon": "mp5",
      "pyfluid.riemann": "hlld",
      "pyfluid.timestep": "ssprk3",
      "pyfluid.rmhd_candidate_guard": "pwc",
      "pyfluid.rmhd_candidate_riemann": "same",
      "pyfluid.physical_check_int": "1",
      "pyfluid.scalar_int": "16",
      "pyfluid.plot_int": "1000000",
      "pyfluid.checkpoint_int": "-1",
      "pyfluid.subcycling": "on",
      "pyfluid.subcycling_steps": "local_cfl",
   }
   for side, state in (("left", spec["left"]), ("right", spec["right"])):
      for name, value in zip(
         ("rho", "pressure", "v1", "v2", "v3", "b1", "b2", "b3"),
         state,
         strict=True,
      ):
         values[f"pyfluid.{name}_{side}"] = repr(value)
   return values


def parse_args() -> argparse.Namespace:
   parser = argparse.ArgumentParser(description=__doc__)
   parser.add_argument("--ranks", type=int, default=3)
   parser.add_argument("--timeout", type=int, default=60)
   parser.add_argument("--force", action="store_true")
   return parser.parse_args()


def main() -> int:
   args = parse_args()
   spec = case_spec()
   started = time.perf_counter()
   run_captured_interpolation_check()
   run_dir, scalar = run_case(
      RUN_NAME,
      INPUTS / "rmhd_balsara1_1d.in",
      overrides(spec),
      args,
   )
   seconds = time.perf_counter() - started
   rows = read_scalar_rows(run_dir)
   failures = check_scalars(
      rows,
      mhd=True,
      momentum=False,
      mass_tol=SCALAR_CONSERVATION_TOLERANCE,
      energy_tol=SCALAR_CONSERVATION_TOLERANCE,
      divb_tol=SCALAR_CONSERVATION_TOLERANCE,
   )

   final_step = int(float(rows[-1]["step"]))
   final_time = float(rows[-1]["time"])
   if final_step <= FORMER_FAILURE_STEP:
      failures.append(
         f"final_step={final_step} did not pass former failure step "
         f"{FORMER_FAILURE_STEP}"
      )
   if not math.isclose(final_time, float(spec["time"]), rel_tol=0.0, abs_tol=1.0e-12):
      failures.append(f"final_time={final_time:.17g} != {spec['time']}")

   active_levels = [int(float(row["amr_active_finest_level"])) for row in rows]
   headrooms = [int(float(row["amr_finest_headroom"])) for row in rows]
   total_cells = [int(float(row["amr_total_cells"])) for row in rows]
   maximum_active_level = max(active_levels)
   minimum_headroom = min(headrooms)
   if maximum_active_level < 4:
      failures.append(f"active finest level={maximum_active_level} < 4")
   if minimum_headroom < 1:
      failures.append(f"minimum AMR headroom={minimum_headroom} < 1")
   if len(set(total_cells)) < 2:
      failures.append("AMR total cell count did not change")

   _, numerical, _, maximum_l1, maximum_overshoot, maximum_tv_ratio = (
      exact_balsara_comparison(3, spec, run_dir)
   )
   if not math.isfinite(maximum_l1):
      failures.append("maximum normalized conserved L1 is non-finite")
   elif maximum_l1 > MAXIMUM_NORMALIZED_COMPONENT_L1:
      failures.append(
         f"maximum normalized conserved L1={maximum_l1:.6e} > "
         f"{MAXIMUM_NORMALIZED_COMPONENT_L1:.6e}"
      )

   c2p_fallbacks, atmosphere_resets = relativistic_recovery_counts(run_dir)
   hll_fallbacks = rmhd_riemann_hll_fallback_count(run_dir)
   if c2p_fallbacks != 0:
      failures.append(f"c2p_fallbacks={c2p_fallbacks} != 0")
   if atmosphere_resets != 0:
      failures.append(f"atmosphere_resets={atmosphere_resets} != 0")
   if hll_fallbacks != 0:
      failures.append(f"rmhd_riemann_hll_fallbacks={hll_fallbacks} != 0")

   result = {
      "case": "Balsara 3 dynamic AMR prolongation",
      "status": "PASS" if not failures else "FAIL",
      "reason": "; ".join(failures),
      "seconds": seconds,
      "ranks": args.ranks,
      "base_cells": 128,
      "composite_cells": numerical.shape[1],
      "final_step": final_step,
      "final_time": final_time,
      "maximum_active_level": maximum_active_level,
      "minimum_level_headroom": minimum_headroom,
      "initial_total_cells": total_cells[0],
      "peak_total_cells": max(total_cells),
      "unique_total_cell_counts": len(set(total_cells)),
      "maximum_normalized_component_l1": maximum_l1,
      "maximum_normalized_global_overshoot": maximum_overshoot,
      "maximum_component_tv_ratio": maximum_tv_ratio,
      "mass_rel_drift": scalar["mass_rel_drift"],
      "energy_rel_drift": scalar["energy_rel_drift"],
      "divb_l1": scalar["divb_l1"],
      "c2p_fallbacks": c2p_fallbacks,
      "atmosphere_resets": atmosphere_resets,
      "rmhd_riemann_hll_fallbacks": hll_fallbacks,
      "run_dir": repo_relative(run_dir),
   }
   write_tsv(RESULTS, [result], list(result))
   scalar.update({
      "suite": RUN_NAME,
      "step": RUN_NAME,
      "status": result["status"],
      "reason": result["reason"],
      "seconds": seconds,
   })
   append_validation_history(HISTORY, [scalar])

   print(
      f"{result['status']} {RUN_NAME}: L1={maximum_l1:.6e} "
      f"level={maximum_active_level} headroom={minimum_headroom} "
      f"cells={total_cells[0]}->{total_cells[-1]} peak={max(total_cells)} "
      f"C2P/reset/HLL={c2p_fallbacks}/{atmosphere_resets}/{hll_fallbacks} "
      f"mass={float(scalar['mass_rel_drift']):.3e} "
      f"energy={float(scalar['energy_rel_drift']):.3e} "
      f"seconds={seconds:.3f}"
   )
   print(f"wrote {repo_relative(RESULTS)}")
   if failures:
      print("failures: " + "; ".join(failures))
      return 1
   return 0


if __name__ == "__main__":
   raise SystemExit(main())
