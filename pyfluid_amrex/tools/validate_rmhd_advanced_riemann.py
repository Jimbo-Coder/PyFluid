#!/usr/bin/env python3
"""Validate RMHD HLLC/HLLD against exact and integration controls."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import numpy as np

from validate_rmhd import (
   EXACT_BALSARA_CASES,
   HISTORY,
   INPUTS,
   RUN_ROOT,
   exact_rmhd_conserved,
   relativistic_recovery_counts,
   rmhd_candidate_correction_counts,
   rmhd_riemann_hll_fallback_count,
   run_case,
   validate_exact_balsara,
)
from plotfile_reader import (
   latest_plotfile,
   level_count,
   parse_plot_header,
   parse_plot_header_1d,
   read_composite_state_1d,
   read_level_state,
)
from validation_common import (
   ROOT,
   amrex_executable,
   append_validation_history,
   build_cached_target,
   repo_relative,
   write_tsv,
)


RESULTS = RUN_ROOT / "rmhd_advanced_riemann.tsv"
POINTWISE_CHECK = ROOT / "build/pyfluid_amrex/pyfluid_amrex_rmhd_riemann_check"
CT_DIVB_TOLERANCE = 1.0e-12
HLLD_ROTATED_CT_TOLERANCE = 2.0e-10
# HLLC restores contact/shear antidiffusion but is not the production MHD
# choice: a grid-aligned strong shock can amplify transverse roundoff.  Keep a
# tight bounded-control gate here while HLLD retains the symmetry-level gate.
HLLC_ROTATED_CT_TOLERANCE = 1.0e-7


def results_path(args: argparse.Namespace) -> Path:
   unfiltered = (
      args.riemann_methods is None
      and args.case_numbers is None
      and args.cells == 512
      and args.candidate_riemann == "hll"
      and not args.contact_only
      and not args.skip_integration
      and not args.integration_only
   )
   if unfiltered:
      return RESULTS
   methods = "-".join(args.riemann_methods or ("hllc", "hlld"))
   cases = "-".join(str(case) for case in (args.case_numbers or ("all",)))
   if args.contact_only:
      scope = "contact"
   elif args.integration_only:
      scope = "integration"
   elif args.skip_integration:
      scope = "balsara"
   else:
      scope = "selected"
   suffix = (
      f"{methods}_{scope}_cases-{cases}_n{args.cells}_"
      f"candidate-{args.candidate_riemann}"
   )
   return RESULTS.with_name(f"{RESULTS.stem}_{suffix}{RESULTS.suffix}")


def require_no_recovery(
   run_dir,
   *,
   allow_riemann_hll: bool = False,
) -> tuple[int, int, int]:
   c2p_fallbacks, atmosphere_resets = relativistic_recovery_counts(run_dir)
   riemann_hll_fallbacks = rmhd_riemann_hll_fallback_count(run_dir)
   if c2p_fallbacks != 0 or atmosphere_resets != 0:
      raise RuntimeError(
         f"{run_dir.name} used a post-update recovery/reset: "
         f"{c2p_fallbacks}/{atmosphere_resets}"
      )
   if riemann_hll_fallbacks != 0 and not allow_riemann_hll:
      raise RuntimeError(
         f"{run_dir.name} unexpectedly used HLL "
         f"{riemann_hll_fallbacks} times"
      )
   return c2p_fallbacks, atmosphere_resets, riemann_hll_fallbacks


def integration_row(
   *,
   kind: str,
   case: str,
   metric: str,
   value: object,
   run_dir,
   recovery_counts: tuple[int, int, int],
   correction_counts: tuple[int, int] = (0, 0),
) -> dict[str, object]:
   c2p_fallbacks, atmosphere_resets, riemann_hll_fallbacks = recovery_counts
   pwl_corrections, pwc_corrections = correction_counts
   return {
      "kind": kind,
      "case": case,
      "status": "PASS",
      "metric": metric,
      "value": value,
      "c2p_fallbacks": c2p_fallbacks,
      "atmosphere_resets": atmosphere_resets,
      "rmhd_pwl_corrections": pwl_corrections,
      "rmhd_pwc_corrections": pwc_corrections,
      "rmhd_riemann_hll_fallbacks": riemann_hll_fallbacks,
      "run_dir": repo_relative(run_dir),
   }


def validate_uniform_ct_dimension(
   args: argparse.Namespace,
   riemann_method: str,
   dim: int,
) -> tuple[dict[str, object], dict[str, object]]:
   run_name = f"uniform_{dim}d_ct_{riemann_method}"
   overrides = {
      "pyfluid.riemann": riemann_method,
      "pyfluid.cfl": "0.9",
      "pyfluid.plot_int": "1000000",
   }
   if dim == 3:
      overrides.update({
         "geometry.prob_lo": "0.0 0.0 0.0",
         "geometry.prob_hi": "1.0 1.0 1.0",
         "geometry.is_periodic": "1 1 1",
         "amr.n_cell": "8 8 8",
         "amr.max_grid_size": "8",
      })
   run_dir, scalar = run_case(
      run_name,
      INPUTS / "rmhd_uniform_2d_ct.in",
      overrides,
      args,
      dim=dim,
   )
   plotfile = latest_plotfile(run_dir, require=True)
   names, output_dim, _ = parse_plot_header(plotfile)
   state = read_level_state(plotfile, names, output_dim, 0)
   spread = float(np.max(np.ptp(state.reshape(state.shape[0], -1), axis=1)))
   if spread > 64.0 * np.finfo(float).eps:
      raise RuntimeError(f"{run_name} changed a free stream: spread={spread:.3e}")
   if float(scalar["divb_l1"]) > CT_DIVB_TOLERANCE:
      raise RuntimeError(f"{run_name} changed divB: {scalar['divb_l1']}")
   counts = require_no_recovery(run_dir)
   return integration_row(
      kind="ct",
      case=run_name,
      metric="max_component_spread",
      value=spread,
      run_dir=run_dir,
      recovery_counts=counts,
   ), scalar


def validate_uniform_ct(
   args: argparse.Namespace,
   riemann_method: str,
) -> tuple[dict[str, object], dict[str, object]]:
   return validate_uniform_ct_dimension(args, riemann_method, 2)


def validate_uniform_3d_ct(
   args: argparse.Namespace,
   riemann_method: str,
) -> tuple[dict[str, object], dict[str, object]]:
   return validate_uniform_ct_dimension(args, riemann_method, 3)


def validate_fixed_curved_metric(
   args: argparse.Namespace,
   riemann_method: str,
) -> tuple[dict[str, object], dict[str, object]]:
   run_name = f"rindler_1d_{riemann_method}"
   run_dir, scalar = run_case(
      run_name,
      INPUTS / "rmhd_uniform_1d.in",
      {
         "geometry.is_periodic": "0",
         "spacetime.fixed_metric": "rindler",
         "spacetime.acceleration1": "0.1",
         "spacetime.origin1": "0.0",
         "pyfluid.validation_policy": "smoke",
         "pyfluid.riemann": riemann_method,
         "pyfluid.timestep": "ssprk3",
         "pyfluid.bc": "outflow",
         "pyfluid.cfl": "0.9",
         "pyfluid.max_step": "2",
         "pyfluid.final_time": "-1",
         "pyfluid.plot_int": "2",
      },
      args,
   )
   counts = require_no_recovery(run_dir)
   return integration_row(
      kind="curved_fixed",
      case=run_name,
      metric="accepted_steps",
      value=2,
      run_dir=run_dir,
      recovery_counts=counts,
   ), scalar


def validate_evolved_metric(
   args: argparse.Namespace,
   riemann_method: str,
) -> tuple[dict[str, object], dict[str, object]]:
   run_name = f"z4c_coupled_rmhd_{riemann_method}"
   run_dir, scalar = run_case(
      run_name,
      INPUTS / "z4c_coupled_uniform_1d.in",
      {
         "geometry.is_periodic": "1",
         "pyfluid.physics": "rmhd",
         "pyfluid.rho0": "1.0e-4",
         "pyfluid.pressure0": "1.0e-6",
         "pyfluid.b1_0": "7.5e-5",
         "pyfluid.b2_0": "3.0e-5",
         "pyfluid.b3_0": "-2.0e-5",
         "pyfluid.riemann": riemann_method,
         "pyfluid.cfl": "0.9",
         "pyfluid.bc": "periodic",
         "pyfluid.ct": "off",
      },
      args,
   )
   counts = require_no_recovery(run_dir)
   return integration_row(
      kind="curved_evolved",
      case=run_name,
      metric="accepted_steps",
      value=2,
      run_dir=run_dir,
      recovery_counts=counts,
   ), scalar


def validate_amr_ct(
   args: argparse.Namespace,
   riemann_method: str,
) -> tuple[dict[str, object], dict[str, object]]:
   run_name = f"entropy_2d_amr_ct_{riemann_method}"
   run_dir, scalar = run_case(
      run_name,
      INPUTS / "rmhd_entropy_wave_2d_amr_ct.in",
      {
         "pyfluid.ng": "3",
         "pyfluid.recon": "mp5",
         "pyfluid.riemann": riemann_method,
         "pyfluid.rmhd_candidate_guard": "pwc",
         "pyfluid.rmhd_candidate_riemann": "hll",
         "pyfluid.cfl": "0.9",
         "pyfluid.plot_int": "1000000",
      },
      args,
      dim=2,
   )
   plotfile = latest_plotfile(run_dir, require=True)
   levels = level_count(plotfile)
   if levels < 3 or float(scalar["divb_l1"]) > CT_DIVB_TOLERANCE:
      raise RuntimeError(
         f"{run_name} failed CT/AMR: levels={levels} divb={scalar['divb_l1']}"
      )
   counts = require_no_recovery(run_dir)
   return integration_row(
      kind="amr_ct",
      case=run_name,
      metric="divb_l1",
      value=scalar["divb_l1"],
      run_dir=run_dir,
      recovery_counts=counts,
   ), scalar


def validate_rotated_ct_shock(
   args: argparse.Namespace,
   riemann_method: str,
) -> tuple[dict[str, object], dict[str, object]]:
   common = {
      "pyfluid.ic": "sod_plane",
      "pyfluid.shock_x0": "0.0",
      "pyfluid.shock_y0": "0.0",
      "amr.max_level": "0",
      "amr.max_grid_size": "128",
      "amr.blocking_factor": "8",
      "pyfluid.final_time": "0.1",
      "pyfluid.max_step": "10000",
      "pyfluid.ng": "3",
      "pyfluid.cfl": "0.9",
      "pyfluid.recon": "mp5",
      "pyfluid.riemann": riemann_method,
      "pyfluid.rmhd_candidate_guard": "off",
      "pyfluid.ct": "on",
      "pyfluid.plot_int": "1000000",
   }
   x_name = f"balsara1_x_ct_{riemann_method}"
   x_dir, x_scalar = run_case(
      x_name,
      INPUTS / "rmhd_balsara1_1d.in",
      {
         **common,
         "geometry.prob_lo": "-0.5 0.0",
         "geometry.prob_hi": "0.5 1.0",
         "geometry.is_periodic": "0 1",
         "amr.n_cell": "128 8",
         "pyfluid.shock_nx": "1.0",
         "pyfluid.shock_ny": "0.0",
         "pyfluid.bc_xlo": "outflow",
         "pyfluid.bc_xhi": "outflow",
         "pyfluid.bc_ylo": "periodic",
         "pyfluid.bc_yhi": "periodic",
      },
      args,
      dim=2,
   )
   y_name = f"balsara1_y_ct_{riemann_method}"
   y_dir, y_scalar = run_case(
      y_name,
      INPUTS / "rmhd_balsara1_1d.in",
      {
         **common,
         "geometry.prob_lo": "0.0 -0.5",
         "geometry.prob_hi": "1.0 0.5",
         "geometry.is_periodic": "1 0",
         "amr.n_cell": "8 128",
         "pyfluid.shock_nx": "0.0",
         "pyfluid.shock_ny": "1.0",
         "pyfluid.b1_left": "-1.0",
         "pyfluid.b2_left": "0.5",
         "pyfluid.b1_right": "1.0",
         "pyfluid.b2_right": "0.5",
         "pyfluid.bc_xlo": "periodic",
         "pyfluid.bc_xhi": "periodic",
         "pyfluid.bc_ylo": "outflow",
         "pyfluid.bc_yhi": "outflow",
      },
      args,
      dim=2,
   )
   x_plotfile = latest_plotfile(x_dir, require=True)
   y_plotfile = latest_plotfile(y_dir, require=True)
   x_names, x_dim, _ = parse_plot_header(x_plotfile)
   y_names, y_dim, _ = parse_plot_header(y_plotfile)
   if x_names != y_names:
      raise RuntimeError("rotated RMHD CT runs use different state schemas")
   x_state = read_level_state(x_plotfile, x_names, x_dim, 0)
   y_state = read_level_state(y_plotfile, y_names, y_dim, 0)
   x_profile = np.mean(x_state, axis=2)
   y_profile = np.mean(y_state, axis=1)
   mapped_y = y_profile.copy()
   mapped_y[x_names.index("mom1")] = y_profile[x_names.index("mom2")]
   mapped_y[x_names.index("mom2")] = -y_profile[x_names.index("mom1")]
   mapped_y[x_names.index("b1")] = y_profile[x_names.index("b2")]
   mapped_y[x_names.index("b2")] = -y_profile[x_names.index("b1")]
   scales = np.maximum(np.max(np.abs(x_profile), axis=1), 1.0e-14)
   rotation_error = float(np.max(
      np.mean(np.abs(x_profile - mapped_y), axis=1) / scales
   ))
   transverse_spread = max(
      float(np.max(np.ptp(x_state, axis=2))),
      float(np.max(np.ptp(y_state, axis=1))),
   )
   tolerance = (
      HLLC_ROTATED_CT_TOLERANCE
      if riemann_method == "hllc"
      else HLLD_ROTATED_CT_TOLERANCE
   )
   maximum_defect = max(rotation_error, transverse_spread)
   if maximum_defect > tolerance:
      raise RuntimeError(
         f"{riemann_method} rotated CT mismatch: error={rotation_error:.3e} "
         f"transverse_spread={transverse_spread:.3e} tolerance={tolerance:.3e}"
      )
   if max(float(x_scalar["divb_l1"]), float(y_scalar["divb_l1"])) \
         > CT_DIVB_TOLERANCE:
      raise RuntimeError(f"{riemann_method} rotated CT shock changed divB")
   x_counts = require_no_recovery(x_dir, allow_riemann_hll=True)
   y_counts = require_no_recovery(y_dir, allow_riemann_hll=True)
   counts = tuple(x + y for x, y in zip(x_counts, y_counts, strict=True))
   x_corrections = rmhd_candidate_correction_counts(x_dir)
   y_corrections = rmhd_candidate_correction_counts(y_dir)
   corrections = tuple(
      x + y for x, y in zip(x_corrections, y_corrections, strict=True)
   )
   row = integration_row(
      kind="rotated_ct",
      case=f"balsara1_xy_ct_{riemann_method}",
      metric="maximum_rotated_ct_defect",
      value=maximum_defect,
      run_dir=y_dir,
      recovery_counts=counts,
      correction_counts=corrections,
   )
   row["rotation_l1"] = rotation_error
   row["transverse_spread"] = transverse_spread
   row["tolerance"] = tolerance
   return row, y_scalar


def validate_isolated_contact(
   args: argparse.Namespace,
   riemann_method: str,
) -> tuple[dict[str, object], dict[str, object]]:
   run_name = f"isolated_contact_{riemann_method}"
   run_dir, scalar = run_case(
      run_name,
      INPUTS / "rmhd_contact_1d.in",
      {
         "pyfluid.riemann": riemann_method,
         "pyfluid.plot_int": "1000000",
      },
      args,
   )
   plotfile = latest_plotfile(run_dir, require=True)
   names, _ = parse_plot_header_1d(plotfile)
   component_names = ("dens", "mom1", "mom2", "mom3", "tau", "b1", "b2", "b3")
   state = read_composite_state_1d(plotfile, names)
   numerical = np.vstack([
      state[names.index(name)] for name in component_names
   ])
   velocity = np.asarray((0.0, 0.7, 0.2))
   magnetic = np.asarray((5.0, 1.0, 0.5))
   comoving_magnetic_squared = (
      np.dot(magnetic, magnetic) * (1.0 - np.dot(velocity, velocity))
      + np.dot(velocity, magnetic) ** 2
   )
   total_pressure = 1.0 + 0.5 * comoving_magnetic_squared
   primitive_profile = np.asarray([
      (0.0, 10.0, total_pressure, 0.0, 0.7, 0.2, 1.0, 0.5),
      (0.0, 1.0, total_pressure, 0.0, 0.7, 0.2, 1.0, 0.5),
   ])
   sides = exact_rmhd_conserved(
      primitive_profile,
      normal_magnetic=5.0,
      gamma=5.0 / 3.0,
   )
   expected = np.empty_like(numerical)
   midpoint = numerical.shape[1] // 2
   expected[:, :midpoint] = sides[:, 0, np.newaxis]
   expected[:, midpoint:] = sides[:, 1, np.newaxis]
   scales = np.maximum(1.0, np.max(np.abs(expected), axis=1))
   maximum_error = float(np.max(np.abs(numerical - expected) / scales[:, np.newaxis]))
   c2p_fallbacks, atmosphere_resets = relativistic_recovery_counts(run_dir)
   riemann_hll_fallbacks = rmhd_riemann_hll_fallback_count(run_dir)
   if maximum_error > 2.0e-12:
      raise RuntimeError(
         f"{run_name} did not preserve the isolated contact: error={maximum_error:.3e}"
      )
   if c2p_fallbacks != 0 or atmosphere_resets != 0 or riemann_hll_fallbacks != 0:
      raise RuntimeError(
         f"{run_name} used a recovery/reset/Riemann fallback: "
         f"{c2p_fallbacks}/{atmosphere_resets}/{riemann_hll_fallbacks}"
      )
   return {
      "kind": "exact",
      "case": run_name,
      "status": "PASS",
      "metric": "maximum_scaled_state_error",
      "value": maximum_error,
      "c2p_fallbacks": c2p_fallbacks,
      "atmosphere_resets": atmosphere_resets,
      "rmhd_pwl_corrections": 0,
      "rmhd_pwc_corrections": 0,
      "rmhd_riemann_hll_fallbacks": riemann_hll_fallbacks,
      "run_dir": repo_relative(run_dir),
   }, scalar


def main() -> int:
   parser = argparse.ArgumentParser(description=__doc__)
   parser.add_argument(
      "--riemann",
      action="append",
      choices=("hllc", "hlld"),
      dest="riemann_methods",
      help="solver to validate; repeat to select both (default: both)",
   )
   parser.add_argument(
      "--case",
      action="append",
      type=int,
      choices=tuple(range(1, len(EXACT_BALSARA_CASES) + 1)),
      dest="case_numbers",
      help="Balsara case number; repeat to select several (default: all)",
   )
   parser.add_argument("--cells", type=int, default=512)
   parser.add_argument(
      "--candidate-riemann",
      choices=("same", "llf", "hll"),
      default="hll",
      help="Riemann solver used by the optional PWC candidate correction",
   )
   parser.add_argument("--ranks", type=int, default=3)
   parser.add_argument("--timeout", type=int, default=240)
   parser.add_argument("--force", action="store_true")
   parser.add_argument(
      "--contact-only",
      action="store_true",
      help="run the pointwise and isolated-contact checks without Balsara tubes",
   )
   parser.add_argument(
      "--skip-integration",
      action="store_true",
      help="skip fixed/evolved-metric and CT/AMR integration checks",
   )
   parser.add_argument(
      "--integration-only",
      action="store_true",
      help="run pointwise, contact, and integration checks without Balsara tubes",
   )
   args = parser.parse_args()
   output_path = results_path(args)
   if not amrex_executable(1).exists():
      raise SystemExit(
         "build pyfluid_amrex_1d first"
      )
   if args.cells < 32:
      raise SystemExit("--cells must be at least 32")

   pointwise_check = build_cached_target(POINTWISE_CHECK.name)
   subprocess.run([str(pointwise_check)], check=True)
   methods = tuple(dict.fromkeys(args.riemann_methods or ("hllc", "hlld")))
   selected_numbers = set(args.case_numbers or range(1, len(EXACT_BALSARA_CASES) + 1))
   cases = tuple(
      spec for spec in EXACT_BALSARA_CASES if int(spec["case"]) in selected_numbers
   )
   rows: list[dict[str, object]] = []
   scalar_rows: list[dict[str, object]] = []
   for method in methods:
      contact_row, contact_scalar = validate_isolated_contact(args, method)
      rows.append(contact_row)
      scalar_rows.append(contact_scalar)
      if args.contact_only:
         continue
      if not args.skip_integration:
         for validator in (
            validate_uniform_ct,
            validate_uniform_3d_ct,
            validate_fixed_curved_metric,
            validate_evolved_metric,
            validate_amr_ct,
            validate_rotated_ct_shock,
         ):
            integration_result, integration_scalar = validator(args, method)
            rows.append(integration_result)
            scalar_rows.append(integration_scalar)
      if args.integration_only:
         continue
      method_rows, method_scalars = validate_exact_balsara(
         args,
         riemann_method=method,
         candidate_riemann_method=args.candidate_riemann,
         cases=cases,
         cells=args.cells,
      )
      rows.extend(method_rows)
      scalar_rows.extend(method_scalars)

   write_tsv(
      output_path,
      rows,
      [
         "kind",
         "case",
         "status",
         "metric",
         "value",
         "rotation_l1",
         "transverse_spread",
         "tolerance",
         "c2p_fallbacks",
         "atmosphere_resets",
         "rmhd_pwl_corrections",
         "rmhd_pwc_corrections",
         "rmhd_riemann_hll_fallbacks",
         "run_dir",
      ],
   )
   append_validation_history(HISTORY, scalar_rows)
   for row in rows:
      print(
         f"PASS {row['case']}: {row['metric']}={row['value']} "
         f"riemann_hll_fallbacks={row['rmhd_riemann_hll_fallbacks']}"
      )
   print(f"wrote {repo_relative(output_path)}")
   return 0


if __name__ == "__main__":
   raise SystemExit(main())
