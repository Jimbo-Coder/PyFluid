#!/usr/bin/env python3
"""Test whether state-following AMR resolves the narrow Balsara-3 shell.

This is a controlled reference experiment, not a new general AMR gate.  It
compares uniform N=128, a base-N=128 hierarchy whose physical feature-length
guard stops refinement at N=1024-equivalent spacing, and uniform N=1024.
Every run uses production MP5, HLL or HLLD, SSPRK3, and the normal PWC
candidate guard.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from diagnose_rmhd_operator import recover_conserved  # noqa: E402
from plotfile_reader import (  # noqa: E402
   latest_plotfile,
   parse_plot_bounds,
   parse_plot_header_1d,
   read_composite_state_1d,
)
from scan_rmhd_balsara_reconstruction import (  # noqa: E402
   COMPONENT_NAMES,
   exact_comparison,
)
from validate_rmhd import (  # noqa: E402
   EXACT_BALSARA_CASES,
   INPUTS,
   RUN_ROOT,
   relativistic_recovery_counts,
   run_case,
)
from validation_common import (  # noqa: E402
   ROOT,
   read_scalar_rows,
   repo_relative,
   write_tsv,
)


RESULTS = RUN_ROOT / "balsara3_amr_resolution_{riemann}.tsv"
PLOT = ROOT / "plots/current/scans/rmhd_balsara3_amr_resolution_{riemann}.png"


def case_spec() -> dict[str, object]:
   return next(spec for spec in EXACT_BALSARA_CASES if int(spec["case"]) == 3)


def common_overrides(
   spec: dict[str, object], cells: int, riemann: str
) -> dict[str, str]:
   overrides = {
      "geometry.prob_lo": "-1.0",
      "geometry.prob_hi": "1.0",
      "amr.n_cell": str(cells),
      "amr.max_grid_size": "128",
      "amr.blocking_factor": "8",
      "pyfluid.gamma": repr(spec["gamma"]),
      "pyfluid.final_time": repr(spec["time"]),
      "pyfluid.max_step": "10000",
      "pyfluid.ng": "3",
      "pyfluid.cfl": "0.9",
      "pyfluid.recon": "mp5",
      "pyfluid.riemann": riemann,
      "pyfluid.timestep": "ssprk3",
      "pyfluid.rmhd_candidate_guard": "pwc",
      "pyfluid.rmhd_candidate_riemann": "same",
      "pyfluid.physical_check_int": "1",
      "pyfluid.scalar_int": "16",
      "pyfluid.plot_int": "1000000",
   }
   for side, state in (("left", spec["left"]), ("right", spec["right"])):
      for name, value in zip(
         ("rho", "pressure", "v1", "v2", "v3", "b1", "b2", "b3"),
         state,
         strict=True,
      ):
         overrides[f"pyfluid.{name}_{side}"] = repr(value)
   return overrides


def run_configuration(
   *,
   name: str,
   label: str,
   cells: int,
   amr: bool,
   riemann: str,
   args: argparse.Namespace,
) -> tuple[dict[str, object], tuple[np.ndarray, np.ndarray, np.ndarray]]:
   spec = case_spec()
   overrides = common_overrides(spec, cells, riemann)
   if amr:
      overrides.update({
         "amr.max_level": "6",
         "amr.ref_ratio": "2 2 2 2 2 2",
         "amr.regrid_mode": "synchronized",
         "amr.regrid_int": "2",
         "amr.n_error_buf": "4 4 4 4 4 4",
         "amr.grid_eff": "0.80",
         "amr.tag_metric": "lohner",
         "amr.tag_threshold": "0.08",
         "amr.min_feature_length": "0.002",
         "amr.level_cell_budget_side": "4096",
         "amr.total_cell_budget": "12000",
         "amr.initial_refine": "box",
         "amr.initial_refine_levels": "3",
         "amr.initial_refine_fixed_levels": "0",
         "amr.initial_refine_box_lo": "-0.08",
         "amr.initial_refine_box_hi": "0.08",
         "amr.state_interp": "coupled-conservative",
      })
   else:
      overrides["amr.max_level"] = "0"
   run_dir, scalar = run_case(
      name,
      INPUTS / "rmhd_balsara1_1d.in",
      overrides,
      args,
   )
   (
      centers,
      numerical,
      exact,
      maximum_l1,
      maximum_overshoot,
      maximum_tv_ratio,
   ) = exact_comparison(3, spec, run_dir)
   fallback_count, reset_count = relativistic_recovery_counts(run_dir)
   scalars = read_scalar_rows(run_dir)
   peak_cells = max(int(float(row["amr_total_cells"])) for row in scalars)
   active_level = max(
      int(float(row["amr_active_finest_level"])) for row in scalars
   )
   headroom = min(int(float(row["amr_finest_headroom"])) for row in scalars)
   row = {
      "case": 3,
      "label": label,
      "riemann": riemann,
      "base_cells": cells,
      "composite_cells": numerical.shape[1],
      "maximum_normalized_component_l1": maximum_l1,
      "maximum_normalized_global_overshoot": maximum_overshoot,
      "maximum_component_tv_ratio": maximum_tv_ratio,
      "c2p_fallbacks": fallback_count,
      "atmosphere_resets": reset_count,
      "mass_rel_drift": scalar["mass_rel_drift"],
      "energy_rel_drift": scalar["energy_rel_drift"],
      "peak_amr_cells": peak_cells,
      "active_finest_level": active_level,
      "minimum_level_headroom": headroom,
      "final_mcells_per_s": scalars[-1]["mcells_per_s"],
      "final_sim_time_per_s": scalars[-1]["sim_time_per_s"],
      "run_dir": repo_relative(run_dir),
   }
   return row, (centers, numerical, exact)


def recovered_profile(state: np.ndarray, gamma: float) -> np.ndarray:
   recovery = recover_conserved(state, gamma)
   if not np.all(recovery.success):
      raise RuntimeError("saved accepted Balsara-3 state failed C2P")
   return recovery.primitive


def make_plot(
   profiles: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
   riemann: str,
   plot_path: Path,
) -> None:
   gamma = float(case_spec()["gamma"])
   figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
   colors = {
      "uniform N=128": "#9d9d9d",
      "AMR 128 to 1024": "#4c78a8",
      "uniform N=1024": "#f58518",
   }
   finest = profiles["uniform N=1024"]
   axes[0].plot(finest[0], finest[2][0], color="black", label="exact FV")
   exact_primitive = recovered_profile(finest[2], gamma)
   axes[1].plot(finest[0], exact_primitive[0], color="black", label="exact FV")
   for label, (x, numerical, _) in profiles.items():
      axes[0].plot(x, numerical[0], color=colors[label], label=label)
      axes[1].plot(
         x,
         recovered_profile(numerical, gamma)[0],
         color=colors[label],
         label=label,
      )
   axes[0].set(xlim=(0.34, 0.43), xlabel="x", ylabel=r"$D$")
   axes[1].set(xlim=(0.34, 0.43), xlabel="x", ylabel=r"$\rho$")
   axes[0].set_title("Conserved shell")
   axes[1].set_title("Recovered primitive shell")
   figure.suptitle(f"Balsara 3 with {riemann.upper()}")
   for axis in axes:
      axis.grid(alpha=0.25)
   axes[0].legend(frameon=False, fontsize=8)
   plot_path.parent.mkdir(parents=True, exist_ok=True)
   figure.savefig(plot_path, dpi=180)
   plt.close(figure)


def parse_args() -> argparse.Namespace:
   parser = argparse.ArgumentParser(description=__doc__)
   parser.add_argument("--ranks", type=int, default=3)
   parser.add_argument("--timeout", type=int, default=180)
   parser.add_argument("--force", action="store_true")
   parser.add_argument("--skip-plot", action="store_true")
   parser.add_argument("--riemann", choices=("hll", "hlld"), default="hll")
   return parser.parse_args()


def main() -> int:
   args = parse_args()
   suffix = "" if args.riemann == "hll" else f"_{args.riemann}"
   results_path = Path(str(RESULTS).format(riemann=args.riemann))
   plot_path = Path(str(PLOT).format(riemann=args.riemann))
   configurations = (
      (f"research_balsara3_uniform_n128{suffix}", "uniform N=128", 128, False),
      (f"research_balsara3_amr_base128{suffix}", "AMR 128 to 1024", 128, True),
      (f"research_balsara3_uniform_n1024{suffix}", "uniform N=1024", 1024, False),
   )
   rows: list[dict[str, object]] = []
   profiles: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
   for name, label, cells, amr in configurations:
      row, profile = run_configuration(
         name=name,
         label=label,
         cells=cells,
         amr=amr,
         riemann=args.riemann,
         args=args,
      )
      rows.append(row)
      profiles[label] = profile
      print(
         f"{label}: L1={row['maximum_normalized_component_l1']:.6e} "
         f"TV={row['maximum_component_tv_ratio']:.4f} "
         f"peak_cells={row['peak_amr_cells']} "
         f"level={row['active_finest_level']} headroom={row['minimum_level_headroom']}"
      )
   results_path.parent.mkdir(parents=True, exist_ok=True)
   write_tsv(results_path, rows, list(rows[0]))
   if not args.skip_plot:
      make_plot(profiles, args.riemann, plot_path)
   print(f"wrote {repo_relative(results_path)}")
   if not args.skip_plot:
      print(f"wrote {repo_relative(plot_path)}")
   return 0


if __name__ == "__main__":
   raise SystemExit(main())
