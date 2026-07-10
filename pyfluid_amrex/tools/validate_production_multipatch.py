#!/usr/bin/env python3
"""Validate production Cartesian multipatch on real AMR finite-volume runs."""

from __future__ import annotations

import argparse
import math
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_amrex_boxed_movie import composite_slice  # noqa: E402
from plotfile_reader import latest_plotfile  # noqa: E402
from run_case import merged_params, write_resolved_input, write_run_log  # noqa: E402
from validation_common import (  # noqa: E402
   CLOSED_POLICY,
   DIFFUSIVE_POLICY,
   ROOT,
   VALIDATION_HISTORY_FIELDS,
   WALL_POLICY,
   append_validation_history,
   parse_float,
   read_tsv,
   repo_relative,
   run_amrex,
   scalar_validation_row,
   write_tsv,
)


INPUT_2D = ROOT / "pyfluid_amrex/inputs/pressure_bump_2d_amr.in"
INPUT_3D = ROOT / "pyfluid_amrex/inputs/pressure_bump_3d_amr.in"
PATCH_UNION_2D = ROOT / "pyfluid_amrex/inputs/hydro_t_corridor_patch_union_2d_amr.in"
RUN_ROOT = ROOT / "runs/current/multipatch/production_hydro_amr"
SUMMARY = RUN_ROOT / "summary.tsv"
HISTORY = ROOT / "runs/current/validation_history.tsv"
HEADROOM_MAX_LEVEL = 10


def safe_clean(path: Path) -> None:
   resolved = path.resolve()
   runs_root = (ROOT / "runs").resolve()
   if not resolved.is_relative_to(runs_root) or resolved == runs_root:
      raise ValueError(f"refusing to clean non-run output directory: {resolved}")
   if resolved.exists():
      shutil.rmtree(resolved)


def run_case(
   name: str,
   *,
   dim: int,
   input_path: Path,
   overrides: list[str],
   ranks: int,
   timeout: int | None,
   checks=CLOSED_POLICY,
   mhd: bool = False,
   description: str = "production hydro AMR Cartesian multipatch run",
) -> tuple[Path, dict[str, object]]:
   run_dir = RUN_ROOT / name
   safe_clean(run_dir)
   full_overrides = [
      *overrides,
      f"pyfluid.run_name={name}",
      f"pyfluid.output_dir={repo_relative(run_dir)}",
   ]
   result = run_amrex(
      dim=dim,
      input_path=input_path,
      ranks=ranks,
      overrides=full_overrides,
      timeout=timeout,
   )
   write_run_log(run_dir, result.command, result.stdout, result.seconds)
   params = merged_params(input_path, full_overrides)
   input_copy = run_dir / "run.in"
   write_resolved_input(input_copy, params)
   if result.returncode != 0:
      tail = "\n".join(result.stdout.strip().splitlines()[-10:])
      return run_dir, {
         "kind": "feature",
         "suite": "production_multipatch",
         "step": name,
         "case": name,
         "label": name,
         "status": "FAIL",
         "reason": tail,
         "returncode": result.returncode,
         "seconds": result.seconds,
         "dim": dim,
         "input": repo_relative(input_path),
         "policy": "closed",
         "ranks": ranks,
         "log": repo_relative(run_dir / "run.log"),
         "description": description,
      }
   row = scalar_validation_row(
      run_dir,
      case=name,
      input_file=input_copy,
      checks=checks,
      mhd=mhd,
      dim=dim,
      ranks=ranks,
      base_input=repo_relative(input_path),
      kind="feature",
   )
   row.update(
      {
         "suite": "production_multipatch",
         "step": name,
         "returncode": result.returncode,
         "seconds": result.seconds,
         "log": repo_relative(run_dir / "run.log"),
         "description": description,
      }
   )
   return run_dir, row


def compare_component(reference: Path, candidate: Path, component: str, tolerance: float) -> dict[str, object]:
   ref, _boxes, tref, _ndim, _names, xlim, ylim = composite_slice(
      reference, component, 5.0 / 3.0, 2, 0.5, False, None, 512, (0.0, 1.0), (0.0, 1.0)
   )
   cand, _boxes, tcand, _ndim, _names, _xlim, _ylim = composite_slice(
      candidate, component, 5.0 / 3.0, 2, 0.5, False, None, 512, xlim, ylim
   )
   mask = np.isfinite(ref) & np.isfinite(cand)
   if not np.any(mask):
      return {
         "component": component,
         "status": "FAIL",
         "reason": "no overlapping finite composite pixels",
         "rel_l1": math.nan,
         "rel_l2": math.nan,
         "rel_linf": math.nan,
      }
   diff = cand - ref
   scale = max(float(np.mean(np.abs(ref[mask]))), 1.0e-30)
   rel_l1 = float(np.mean(np.abs(diff[mask]))) / scale
   rel_l2 = float(np.sqrt(np.mean(diff[mask] ** 2))) / scale
   rel_linf = float(np.max(np.abs(diff[mask]))) / scale
   time_error = abs(tref - tcand)
   status = "PASS" if rel_l2 <= tolerance and rel_linf <= 10.0 * tolerance and time_error <= 1.0e-12 else "FAIL"
   reason = "" if status == "PASS" else (
      f"{component} rel_l2={rel_l2:.3e}, rel_linf={rel_linf:.3e}, dt={time_error:.3e}"
   )
   return {
      "component": component,
      "status": status,
      "reason": reason,
      "rel_l1": rel_l1,
      "rel_l2": rel_l2,
      "rel_linf": rel_linf,
   }


def amr_headroom_row(run_dir: Path, *, case: str, label: str, dim: int, input_path: Path, ranks: int) -> dict[str, object]:
   rows = read_tsv(run_dir / "amr_levels.tsv")
   failures: list[str] = []
   if not rows:
      failures.append("missing amr_levels.tsv rows")
      final: dict[str, str] = {}
      max_active_level = math.nan
      max_level = math.nan
   else:
      final = rows[-1]
      max_active_level = max(parse_float(row.get("active_finest_level")) for row in rows)
      max_level = parse_float(final.get("max_level"))
      if max_level < HEADROOM_MAX_LEVEL:
         failures.append(f"max_level={max_level:g} < required safety headroom {HEADROOM_MAX_LEVEL}")
      capped_rows = [
         row for row in rows
         if parse_float(row.get("resolution_limited_by_max_level"), 0.0) > 0.5
      ]
      if capped_rows:
         failures.append("resolution was capped by max_level before meeting the configured cutoff")
      if max_active_level < 1.0:
         failures.append("AMR never activated a refined level")
   return {
      "kind": "feature",
      "suite": "production_multipatch",
      "step": case,
      "case": case,
      "label": label,
      "status": "PASS" if not failures else "FAIL",
      "reason": "; ".join(failures),
      "returncode": 0 if not failures else 1,
      "seconds": "",
      "dim": dim,
      "input": repo_relative(input_path),
      "policy": "amr_headroom",
      "ranks": ranks,
      "log": repo_relative(run_dir / "amr_levels.tsv") if (run_dir / "amr_levels.tsv").exists() else "",
      "amr_active_finest_level": final.get("active_finest_level", ""),
      "amr_finest_headroom": final.get("finest_headroom", ""),
      "amr_total_budget_fraction": final.get("total_budget_fraction", ""),
      "amr_peak_level_budget_fraction": final.get("peak_level_budget_fraction", ""),
      "description": (
         f"max active level {max_active_level:g} with max_level {max_level:g}; "
         "resolution must stop on the configured AMR cutoff, not max_level"
      ),
   }


def comparison_row(
   reference_run: Path,
   candidate_run: Path,
   *,
   dim: int,
   case: str,
   label: str,
   input_path: Path,
   components: list[str],
   tolerance: float,
   ranks: int,
) -> dict[str, object]:
   reference = latest_plotfile(reference_run, require=True)
   candidate = latest_plotfile(candidate_run, require=True)
   component_rows = [compare_component(reference, candidate, component, tolerance) for component in components]
   failures = [row for row in component_rows if row["status"] != "PASS"]
   max_l2 = max(float(row["rel_l2"]) for row in component_rows)
   max_linf = max(float(row["rel_linf"]) for row in component_rows)
   return {
      "kind": "feature",
      "suite": "production_multipatch",
      "step": case,
      "case": case,
      "label": label,
      "status": "PASS" if not failures else "FAIL",
      "reason": "; ".join(str(row["reason"]) for row in failures),
      "returncode": 0 if not failures else 1,
      "seconds": "",
      "dim": dim,
      "input": repo_relative(input_path),
      "policy": "plotfile_composite",
      "ranks": ranks,
      "log": "",
      "description": (
         f"final AMR plotfile composite comparison over {', '.join(components)}; "
         f"max rel_l2={max_l2:.3e}, max rel_linf={max_linf:.3e}"
      ),
   }


def patch_union_corridor_rows(args: argparse.Namespace) -> list[dict[str, object]]:
   rows: list[dict[str, object]] = []
   common = [
      "pyfluid.final_time=0.75",
      "pyfluid.max_step=100000",
      "pyfluid.scalar_int=4",
      "pyfluid.plot_int=-1",
      "pyfluid.checkpoint_int=-1",
      "pyfluid.physical_check_int=1",
      "pyfluid.recon=pwl",
      "pyfluid.riemann=hllc",
      "pyfluid.timestep=ssprk3",
   ]
   _noamr_run, noamr_row = run_case(
      "hydro_t_corridor_patch_union_2d_noamr",
      dim=2,
      input_path=PATCH_UNION_2D,
      overrides=[
         *common,
         "amr.max_level=0",
         "amr.initial_refine_levels=0",
         "amr.initial_refine_fixed_levels=0",
         "pyfluid.subcycling=off",
      ],
      ranks=args.ranks,
      timeout=args.timeout,
      checks=WALL_POLICY,
      description=(
         "2D Cartesian patch-union corridor without AMR; catches internal "
         "patch-union wall flux conservation regressions"
      ),
   )
   rows.append(noamr_row)

   amr_run, amr_row = run_case(
      "hydro_t_corridor_patch_union_2d_amr_subcycling",
      dim=2,
      input_path=PATCH_UNION_2D,
      overrides=[
         *common,
         f"amr.max_level={HEADROOM_MAX_LEVEL}",
         "amr.min_feature_length=0.004",
         "amr.initial_refine_levels=2",
         "amr.initial_refine_fixed_levels=0",
         "amr.regrid_int=4",
         "pyfluid.subcycling=on",
         "pyfluid.subcycling_steps=spatial",
      ],
      ranks=args.ranks,
      timeout=args.timeout,
      checks=WALL_POLICY,
      description=(
         "2D Cartesian patch-union corridor with AMR+subcycling; validates "
         "production FillPatch/reflux/average-down behavior on a non-rectangular domain"
      ),
   )
   rows.append(amr_row)
   if amr_row["status"] == "PASS":
      rows.append(
         amr_headroom_row(
            amr_run,
            case="hydro_t_corridor_patch_union_2d_amr_headroom",
            label="2D Cartesian patch-union corridor AMR headroom",
            dim=2,
            input_path=PATCH_UNION_2D,
            ranks=args.ranks,
         )
      )
   return rows


def write_summary(rows: list[dict[str, object]]) -> None:
   write_tsv(SUMMARY, rows, VALIDATION_HISTORY_FIELDS)
   append_validation_history(HISTORY, rows)


def print_summary(rows: list[dict[str, object]]) -> None:
   for row in rows:
      print(
         f"{row['status']}\t{row['case']}\tAMR=L{row.get('amr_active_finest_level', '')}\t"
         f"Mcells/s={row.get('avg_mcells_per_s', '')}\t{row.get('description', '')}"
      )
      if row.get("reason"):
         print(f"  reason: {row['reason']}")
   print(f"summary: {repo_relative(SUMMARY)}")


def main() -> int:
   parser = argparse.ArgumentParser(description=__doc__)
   parser.add_argument("--ranks", type=int, default=3)
   parser.add_argument("--timeout", type=int, default=120)
   parser.add_argument("--tolerance", type=float, default=1.0e-10)
   parser.add_argument("--hydro-only", action="store_true", help="Run only pure HD Cartesian multipatch cases.")
   parser.add_argument(
      "--patch-union-only",
      action="store_true",
      help="Run only the real non-rectangular patch-union corridor conservation gates.",
   )
   parser.add_argument("--force", action="store_true", help="Accepted for suite-driver consistency; outputs are overwritten either way.")
   args = parser.parse_args()

   if args.patch_union_only:
      rows = patch_union_corridor_rows(args)
      write_summary(rows)
      print_summary(rows)
      return 0 if all(row["status"] == "PASS" for row in rows) else 1

   common_2d = [
      "amr.n_cell=32 32",
      "amr.max_level=1",
      "amr.ref_ratio=2",
      "amr.max_grid_size=32",
      "amr.blocking_factor=8",
      "amr.initial_refine=box",
      "amr.initial_refine_levels=1",
      "amr.initial_refine_fixed_levels=1",
      "amr.initial_refine_box_lo=0.35 0.35",
      "amr.initial_refine_box_hi=0.65 0.65",
      "amr.regrid_int=-1",
      "amr.state_interp=quartic-conservative",
      "pyfluid.max_step=6",
      "pyfluid.scalar_int=1",
      "pyfluid.plot_int=6",
      "pyfluid.ng=2",
      "pyfluid.recon=pwl",
      "pyfluid.riemann=hllc",
      "pyfluid.timestep=ssprk3",
      "pyfluid.subcycling=on",
      "pyfluid.physical_check_int=1",
   ]
   single_run, single_row = run_case(
      "pressure_bump_2d_single_amr",
      dim=2,
      input_path=INPUT_2D,
      overrides=[*common_2d, "mesh.patch_system=single"],
      ranks=args.ranks,
      timeout=args.timeout,
   )
   tjunction_run, tjunction_row = run_case(
      "pressure_bump_2d_cartesian_tjunction_amr",
      dim=2,
      input_path=INPUT_2D,
      overrides=[
         *common_2d,
         "mesh.patch_system=multi_homogeneous_cartesian",
         "mesh.patch_box_small=0 0  16 0  16 16",
         "mesh.patch_box_big=15 31  31 15  31 31",
      ],
      ranks=args.ranks,
      timeout=args.timeout,
   )
   rows = [single_row, tjunction_row]
   if single_row["status"] == "PASS" and tjunction_row["status"] == "PASS":
      rows.append(
         comparison_row(
            single_run,
            tjunction_run,
            dim=2,
            case="2d_single_vs_tjunction_amr_compare",
            label="2D production hydro AMR single patch vs Cartesian T-junction",
            input_path=INPUT_2D,
            components=["rho", "mom1", "mom2", "energy"],
            tolerance=args.tolerance,
            ranks=args.ranks,
         )
      )
   counts_overlap_run, counts_overlap_row = run_case(
      "pressure_bump_2d_cartesian_counts_overlap_amr",
      dim=2,
      input_path=INPUT_2D,
      overrides=[
         *common_2d,
         "mesh.patch_system=multi_homogeneous_cartesian",
         "mesh.patch_counts=2 1",
         "mesh.patch_overlap_cells=2",
      ],
      ranks=args.ranks,
      timeout=args.timeout,
   )
   rows.append(counts_overlap_row)
   if single_row["status"] == "PASS" and counts_overlap_row["status"] == "PASS":
      rows.append(
         comparison_row(
            single_run,
            counts_overlap_run,
            dim=2,
            case="2d_single_vs_counts_overlap_amr_compare",
            label="2D production hydro AMR single patch vs tiled Cartesian patch-count layout with overlap context",
            input_path=INPUT_2D,
            components=["rho", "mom1", "mom2", "energy"],
            tolerance=args.tolerance,
            ranks=args.ranks,
         )
      )
   large_counts_overlap_run, large_counts_overlap_row = run_case(
      "pressure_bump_2d_cartesian_counts_large_overlap_amr",
      dim=2,
      input_path=INPUT_2D,
      overrides=[
         *common_2d,
         "mesh.patch_system=multi_homogeneous_cartesian",
         "mesh.patch_counts=2 1",
         "mesh.patch_overlap_cells=8",
      ],
      ranks=args.ranks,
      timeout=args.timeout,
   )
   rows.append(large_counts_overlap_row)
   if single_row["status"] == "PASS" and large_counts_overlap_row["status"] == "PASS":
      rows.append(
         comparison_row(
            single_run,
            large_counts_overlap_run,
            dim=2,
            case="2d_single_vs_counts_large_overlap_amr_compare",
            label="2D production hydro AMR single patch vs tiled Cartesian layout with large overlap context",
            input_path=INPUT_2D,
            components=["rho", "mom1", "mom2", "energy"],
            tolerance=args.tolerance,
            ranks=args.ranks,
         )
      )
   overlap_run, overlap_row = run_case(
      "pressure_bump_2d_cartesian_overlap_owner_first_amr",
      dim=2,
      input_path=INPUT_2D,
      overrides=[
         *common_2d,
         "mesh.patch_system=multi_homogeneous_cartesian",
         "mesh.patch_box_ownership=owner_first",
         "mesh.patch_box_small=0 0  14 0  14 14",
         "mesh.patch_box_big=17 31  31 17  31 31",
      ],
      ranks=args.ranks,
      timeout=args.timeout,
   )
   rows.append(overlap_row)
   if single_row["status"] == "PASS" and overlap_row["status"] == "PASS":
      rows.append(
         comparison_row(
            single_run,
            overlap_run,
            dim=2,
            case="2d_single_vs_overlap_owner_first_amr_compare",
            label="2D production hydro AMR single patch vs small-overlap Cartesian T-junction owner-first layout",
            input_path=INPUT_2D,
            components=["rho", "mom1", "mom2", "energy"],
            tolerance=args.tolerance,
            ranks=args.ranks,
         )
      )
   overlap_last_run, overlap_last_row = run_case(
      "pressure_bump_2d_cartesian_overlap_owner_last_amr",
      dim=2,
      input_path=INPUT_2D,
      overrides=[
         *common_2d,
         "mesh.patch_system=multi_homogeneous_cartesian",
         "mesh.patch_box_ownership=owner_last",
         "mesh.patch_box_small=0 0  14 0  14 14",
         "mesh.patch_box_big=17 31  31 17  31 31",
      ],
      ranks=args.ranks,
      timeout=args.timeout,
   )
   rows.append(overlap_last_row)
   if single_row["status"] == "PASS" and overlap_last_row["status"] == "PASS":
      rows.append(
         comparison_row(
            single_run,
            overlap_last_run,
            dim=2,
            case="2d_single_vs_overlap_owner_last_amr_compare",
            label="2D production hydro AMR single patch vs small-overlap Cartesian T-junction owner-last layout",
            input_path=INPUT_2D,
            components=["rho", "mom1", "mom2", "energy"],
            tolerance=args.tolerance,
            ranks=args.ranks,
         )
      )

   cross_overlap_run, cross_overlap_row = run_case(
      "pressure_bump_2d_cartesian_overlap_cross_owner_last_amr",
      dim=2,
      input_path=INPUT_2D,
      overrides=[
         *common_2d,
         "mesh.patch_system=multi_homogeneous_cartesian",
         "mesh.patch_box_ownership=owner_last",
         "mesh.patch_box_small=0 0  12 0  0 12",
         "mesh.patch_box_big=31 31  19 31  31 19",
      ],
      ranks=args.ranks,
      timeout=args.timeout,
   )
   rows.append(cross_overlap_row)
   if single_row["status"] == "PASS" and cross_overlap_row["status"] == "PASS":
      rows.append(
         comparison_row(
            single_run,
            cross_overlap_run,
            dim=2,
            case="2d_single_vs_overlap_cross_owner_last_amr_compare",
            label="2D production hydro AMR single patch vs cross-overlap Cartesian owner-last layout",
            input_path=INPUT_2D,
            components=["rho", "mom1", "mom2", "energy"],
            tolerance=args.tolerance,
            ranks=args.ranks,
         )
      )

   common_2d_resolution_jump = [
      item
      for item in common_2d
      if not item.startswith("amr.max_level=")
      and not item.startswith("amr.ref_ratio=")
      and not item.startswith("amr.n_error_buf=")
      and not item.startswith("amr.initial_refine_levels=")
      and not item.startswith("amr.initial_refine_fixed_levels=")
      and not item.startswith("amr.initial_refine_box_lo=")
      and not item.startswith("amr.initial_refine_box_hi=")
   ]
   common_2d_resolution_jump.extend(
      [
         f"amr.max_level={HEADROOM_MAX_LEVEL}",
         "amr.ref_ratio=2",
         "amr.n_error_buf=3",
         "amr.min_feature_length=0.01",
         "amr.initial_refine_levels=2",
         "amr.initial_refine_fixed_levels=2",
         "amr.initial_refine_box_lo=0.50 0.00",
         "amr.initial_refine_box_hi=1.00 0.50",
      ]
   )
   single_jump_run, single_jump_row = run_case(
      "pressure_bump_2d_single_resolution_jump_amr",
      dim=2,
      input_path=INPUT_2D,
      overrides=[*common_2d_resolution_jump, "mesh.patch_system=single"],
      ranks=args.ranks,
      timeout=args.timeout,
   )
   tjunction_jump_run, tjunction_jump_row = run_case(
      "pressure_bump_2d_tjunction_resolution_jump_amr",
      dim=2,
      input_path=INPUT_2D,
      overrides=[
         *common_2d_resolution_jump,
         "mesh.patch_system=multi_homogeneous_cartesian",
         "mesh.patch_box_small=0 0  16 0  16 16",
         "mesh.patch_box_big=15 31  31 15  31 31",
      ],
      ranks=args.ranks,
      timeout=args.timeout,
   )
   rows.extend([single_jump_row, tjunction_jump_row])
   rows.append(
      amr_headroom_row(
         single_jump_run,
         case="2d_single_resolution_jump_amr_headroom",
         label="2D production AMR headroom with mixed-resolution refined patch",
         dim=2,
         input_path=INPUT_2D,
         ranks=args.ranks,
      )
   )
   rows.append(
      amr_headroom_row(
         tjunction_jump_run,
         case="2d_tjunction_resolution_jump_amr_headroom",
         label="2D Cartesian multipatch AMR headroom with mixed-resolution refined branch",
         dim=2,
         input_path=INPUT_2D,
         ranks=args.ranks,
      )
   )
   if single_jump_row["status"] == "PASS" and tjunction_jump_row["status"] == "PASS":
      rows.append(
         comparison_row(
            single_jump_run,
            tjunction_jump_run,
            dim=2,
            case="2d_single_vs_tjunction_resolution_jump_amr_compare",
            label="2D production hydro AMR single patch vs Cartesian T-junction with fixed refined patch",
            input_path=INPUT_2D,
            components=["rho", "mom1", "mom2", "energy"],
            tolerance=args.tolerance,
            ranks=args.ranks,
         )
      )

   if args.hydro_only:
      common_2d_diffusion = []
   else:
      common_2d_diffusion = [
      item
      for item in common_2d
      if not item.startswith("pyfluid.max_step=")
      and not item.startswith("pyfluid.plot_int=")
      and not item.startswith("amr.ref_ratio=")
      ]
      common_2d_diffusion.extend(
         [
            "amr.ref_ratio=2",
            "pyfluid.max_step=4",
            "pyfluid.plot_int=4",
            "pyfluid.diffusion=explicit",
            "pyfluid.viscosity=0.01",
            "pyfluid.diffusion_cfl=0.9",
         ]
      )
   if not args.hydro_only:
      single_diffusion_run, single_diffusion_row = run_case(
      "pressure_bump_2d_single_explicit_diffusion_amr",
      dim=2,
      input_path=INPUT_2D,
      overrides=[*common_2d_diffusion, "mesh.patch_system=single"],
      ranks=args.ranks,
      timeout=args.timeout,
      checks=DIFFUSIVE_POLICY,
   )
      tjunction_diffusion_run, tjunction_diffusion_row = run_case(
      "pressure_bump_2d_tjunction_explicit_diffusion_amr",
      dim=2,
      input_path=INPUT_2D,
      overrides=[
         *common_2d_diffusion,
         "mesh.patch_system=multi_homogeneous_cartesian",
         "mesh.patch_box_small=0 0  16 0  16 16",
         "mesh.patch_box_big=15 31  31 15  31 31",
      ],
      ranks=args.ranks,
      timeout=args.timeout,
      checks=DIFFUSIVE_POLICY,
   )
      rows.extend([single_diffusion_row, tjunction_diffusion_row])
      if single_diffusion_row["status"] == "PASS" and tjunction_diffusion_row["status"] == "PASS":
         rows.append(
            comparison_row(
            single_diffusion_run,
            tjunction_diffusion_run,
            dim=2,
            case="2d_single_vs_tjunction_explicit_diffusion_amr_compare",
            label="2D explicit viscous hydro AMR single patch vs Cartesian T-junction",
            input_path=INPUT_2D,
            components=["rho", "mom1", "mom2", "energy"],
            tolerance=args.tolerance,
            ranks=args.ranks,
            )
         )

      overlap_diffusion_run, overlap_diffusion_row = run_case(
      "pressure_bump_2d_overlap_owner_first_explicit_diffusion_amr",
      dim=2,
      input_path=INPUT_2D,
      overrides=[
         *common_2d_diffusion,
         "mesh.patch_system=multi_homogeneous_cartesian",
         "mesh.patch_box_ownership=owner_first",
         "mesh.patch_box_small=0 0  14 0  14 14",
         "mesh.patch_box_big=17 31  31 17  31 31",
      ],
      ranks=args.ranks,
      timeout=args.timeout,
      checks=DIFFUSIVE_POLICY,
   )
      rows.append(overlap_diffusion_row)
      if single_diffusion_row["status"] == "PASS" and overlap_diffusion_row["status"] == "PASS":
         rows.append(
            comparison_row(
            single_diffusion_run,
            overlap_diffusion_run,
            dim=2,
            case="2d_single_vs_overlap_explicit_diffusion_amr_compare",
            label="2D explicit viscous hydro AMR single patch vs small-overlap owner-first layout",
            input_path=INPUT_2D,
            components=["rho", "mom1", "mom2", "energy"],
            tolerance=args.tolerance,
            ranks=args.ranks,
            )
         )

   if args.hydro_only:
      common_2d_mhd_ct = []
   else:
      common_2d_mhd_ct = [
      item
      for item in common_2d
      if not item.startswith("pyfluid.physics=")
      and not item.startswith("pyfluid.ct=")
      and not item.startswith("pyfluid.max_step=")
      and not item.startswith("pyfluid.plot_int=")
      and not item.startswith("pyfluid.recon=")
      and not item.startswith("pyfluid.riemann=")
      and not item.startswith("pyfluid.timestep=")
      ]
      common_2d_mhd_ct.extend(
         [
            "pyfluid.physics=mhd",
            "pyfluid.ct=on",
            "pyfluid.max_step=3",
            "pyfluid.plot_int=3",
            "pyfluid.recon=pwl",
            "pyfluid.riemann=hlld",
            "pyfluid.timestep=ssprk3",
            "pyfluid.b1_0=0.5",
            "pyfluid.b2_0=0.2",
            "pyfluid.b3_0=0.1",
            "pyfluid.diffusion=none",
            "pyfluid.resistivity=0.0",
            "pyfluid.viscosity=0.0",
         ]
      )
   if not args.hydro_only:
      mhd_single_run, mhd_single_row = run_case(
      "pressure_bump_2d_mhd_ct_single_amr",
      dim=2,
      input_path=INPUT_2D,
      overrides=[*common_2d_mhd_ct, "mesh.patch_system=single"],
      ranks=args.ranks,
      timeout=args.timeout,
      mhd=True,
      description="production MHD+CT AMR Cartesian multipatch run",
   )
      mhd_tjunction_run, mhd_tjunction_row = run_case(
      "pressure_bump_2d_mhd_ct_tjunction_amr",
      dim=2,
      input_path=INPUT_2D,
      overrides=[
         *common_2d_mhd_ct,
         "mesh.patch_system=multi_homogeneous_cartesian",
         "mesh.patch_box_small=0 0  16 0  16 16",
         "mesh.patch_box_big=15 31  31 15  31 31",
      ],
      ranks=args.ranks,
      timeout=args.timeout,
      mhd=True,
      description="production MHD+CT AMR Cartesian multipatch run",
   )
      rows.extend([mhd_single_row, mhd_tjunction_row])
      if mhd_single_row["status"] == "PASS" and mhd_tjunction_row["status"] == "PASS":
         rows.append(
            comparison_row(
            mhd_single_run,
            mhd_tjunction_run,
            dim=2,
            case="2d_mhd_ct_single_vs_tjunction_amr_compare",
            label="2D MHD+CT AMR single patch vs Cartesian T-junction",
            input_path=INPUT_2D,
            components=["rho", "mom1", "mom2", "energy", "b1", "b2", "b3"],
            tolerance=args.tolerance,
            ranks=args.ranks,
            )
         )
      mhd_counts_overlap_run, mhd_counts_overlap_row = run_case(
      "pressure_bump_2d_mhd_ct_counts_overlap_amr",
      dim=2,
      input_path=INPUT_2D,
      overrides=[
         *common_2d_mhd_ct,
         "mesh.patch_system=multi_homogeneous_cartesian",
         "mesh.patch_counts=2 1",
         "mesh.patch_overlap_cells=2",
      ],
      ranks=args.ranks,
      timeout=args.timeout,
      mhd=True,
      description="production MHD+CT AMR Cartesian multipatch run",
   )
      rows.append(mhd_counts_overlap_row)
      if mhd_single_row["status"] == "PASS" and mhd_counts_overlap_row["status"] == "PASS":
         rows.append(
            comparison_row(
            mhd_single_run,
            mhd_counts_overlap_run,
            dim=2,
            case="2d_mhd_ct_single_vs_counts_overlap_amr_compare",
            label="2D MHD+CT AMR single patch vs tiled Cartesian patch-count layout with overlap context",
            input_path=INPUT_2D,
            components=["rho", "mom1", "mom2", "energy", "b1", "b2", "b3"],
            tolerance=args.tolerance,
            ranks=args.ranks,
            )
         )

   common_3d = [
      "amr.n_cell=16 16 16",
      "amr.max_level=1",
      "amr.ref_ratio=2",
      "amr.max_grid_size=16",
      "amr.blocking_factor=8",
      "amr.initial_refine=box",
      "amr.initial_refine_levels=1",
      "amr.initial_refine_fixed_levels=1",
      "amr.initial_refine_box_lo=0.35 0.35 0.35",
      "amr.initial_refine_box_hi=0.65 0.65 0.65",
      "amr.regrid_int=-1",
      "amr.state_interp=quartic-conservative",
      "pyfluid.max_step=3",
      "pyfluid.scalar_int=1",
      "pyfluid.plot_int=3",
      "pyfluid.ng=2",
      "pyfluid.recon=pwl",
      "pyfluid.riemann=hllc",
      "pyfluid.timestep=ssprk3",
      "pyfluid.subcycling=on",
      "pyfluid.physical_check_int=1",
   ]
   single_3d_run, single_3d_row = run_case(
      "pressure_bump_3d_single_amr",
      dim=3,
      input_path=INPUT_3D,
      overrides=[*common_3d, "mesh.patch_system=single"],
      ranks=args.ranks,
      timeout=args.timeout,
   )
   tjunction_3d_run, tjunction_3d_row = run_case(
      "pressure_bump_3d_cartesian_tjunction_amr",
      dim=3,
      input_path=INPUT_3D,
      overrides=[
         *common_3d,
         "mesh.patch_system=multi_homogeneous_cartesian",
         "mesh.patch_box_small=0 0 0  8 0 0  8 8 0",
         "mesh.patch_box_big=7 15 15  15 7 15  15 15 15",
      ],
      ranks=args.ranks,
      timeout=args.timeout,
   )
   rows.extend([single_3d_row, tjunction_3d_row])
   if single_3d_row["status"] == "PASS" and tjunction_3d_row["status"] == "PASS":
      rows.append(
         comparison_row(
            single_3d_run,
            tjunction_3d_run,
            dim=3,
            case="3d_single_vs_tjunction_amr_compare",
            label="3D production hydro AMR single patch vs Cartesian T-junction",
            input_path=INPUT_3D,
            components=["rho", "mom1", "mom2", "mom3", "energy"],
            tolerance=args.tolerance,
            ranks=args.ranks,
         )
      )

   counts_overlap_3d_run, counts_overlap_3d_row = run_case(
      "pressure_bump_3d_cartesian_counts_large_overlap_amr",
      dim=3,
      input_path=INPUT_3D,
      overrides=[
         *common_3d,
         "mesh.patch_system=multi_homogeneous_cartesian",
         "mesh.patch_counts=2 1 1",
         "mesh.patch_overlap_cells=4",
      ],
      ranks=args.ranks,
      timeout=args.timeout,
   )
   rows.append(counts_overlap_3d_row)
   if single_3d_row["status"] == "PASS" and counts_overlap_3d_row["status"] == "PASS":
      rows.append(
         comparison_row(
            single_3d_run,
            counts_overlap_3d_run,
            dim=3,
            case="3d_single_vs_counts_large_overlap_amr_compare",
            label="3D production hydro AMR single patch vs tiled Cartesian layout with large overlap context",
            input_path=INPUT_3D,
            components=["rho", "mom1", "mom2", "mom3", "energy"],
            tolerance=args.tolerance,
            ranks=args.ranks,
         )
      )

   overlap_3d_run, overlap_3d_row = run_case(
      "pressure_bump_3d_cartesian_overlap_cross_owner_last_amr",
      dim=3,
      input_path=INPUT_3D,
      overrides=[
         *common_3d,
         "mesh.patch_system=multi_homogeneous_cartesian",
         "mesh.patch_box_ownership=owner_last",
         "mesh.patch_box_small=0 0 0  6 0 0  0 6 0  0 0 6",
         "mesh.patch_box_big=15 15 15  9 15 15  15 9 15  15 15 9",
      ],
      ranks=args.ranks,
      timeout=args.timeout,
   )
   rows.append(overlap_3d_row)
   if single_3d_row["status"] == "PASS" and overlap_3d_row["status"] == "PASS":
      rows.append(
         comparison_row(
            single_3d_run,
            overlap_3d_run,
            dim=3,
            case="3d_single_vs_overlap_cross_owner_last_amr_compare",
            label="3D production hydro AMR single patch vs cross-overlap Cartesian owner-last layout",
            input_path=INPUT_3D,
            components=["rho", "mom1", "mom2", "mom3", "energy"],
            tolerance=args.tolerance,
            ranks=args.ranks,
         )
      )

   if args.hydro_only:
      common_3d_mhd_ct = []
   else:
      common_3d_mhd_ct = [
      item
      for item in common_3d
      if not item.startswith("pyfluid.physics=")
      and not item.startswith("pyfluid.ct=")
      and not item.startswith("pyfluid.max_step=")
      and not item.startswith("pyfluid.plot_int=")
      and not item.startswith("pyfluid.recon=")
      and not item.startswith("pyfluid.riemann=")
      and not item.startswith("pyfluid.timestep=")
      ]
      common_3d_mhd_ct.extend(
         [
            "pyfluid.physics=mhd",
            "pyfluid.ct=on",
            "pyfluid.max_step=2",
            "pyfluid.plot_int=2",
            "pyfluid.recon=pwl",
            "pyfluid.riemann=hlld",
            "pyfluid.timestep=ssprk3",
            "pyfluid.b1_0=0.4",
            "pyfluid.b2_0=0.2",
            "pyfluid.b3_0=0.1",
            "pyfluid.diffusion=none",
            "pyfluid.resistivity=0.0",
            "pyfluid.viscosity=0.0",
         ]
      )
   if not args.hydro_only:
      mhd_3d_single_run, mhd_3d_single_row = run_case(
      "pressure_bump_3d_mhd_ct_single_amr",
      dim=3,
      input_path=INPUT_3D,
      overrides=[*common_3d_mhd_ct, "mesh.patch_system=single"],
      ranks=args.ranks,
      timeout=args.timeout,
      mhd=True,
      description="production 3D MHD+CT AMR Cartesian multipatch run",
   )
      mhd_3d_tjunction_run, mhd_3d_tjunction_row = run_case(
      "pressure_bump_3d_mhd_ct_tjunction_amr",
      dim=3,
      input_path=INPUT_3D,
      overrides=[
         *common_3d_mhd_ct,
         "mesh.patch_system=multi_homogeneous_cartesian",
         "mesh.patch_box_small=0 0 0  8 0 0  8 8 0",
         "mesh.patch_box_big=7 15 15  15 7 15  15 15 15",
      ],
      ranks=args.ranks,
      timeout=args.timeout,
      mhd=True,
      description="production 3D MHD+CT AMR Cartesian multipatch run",
   )
      rows.extend([mhd_3d_single_row, mhd_3d_tjunction_row])
      if mhd_3d_single_row["status"] == "PASS" and mhd_3d_tjunction_row["status"] == "PASS":
         rows.append(
            comparison_row(
            mhd_3d_single_run,
            mhd_3d_tjunction_run,
            dim=3,
            case="3d_mhd_ct_single_vs_tjunction_amr_compare",
            label="3D MHD+CT AMR single patch vs Cartesian T-junction",
            input_path=INPUT_3D,
            components=["rho", "mom1", "mom2", "mom3", "energy", "b1", "b2", "b3"],
            tolerance=args.tolerance,
            ranks=args.ranks,
            )
         )

   if args.hydro_only:
      common_3d_diffusion = []
   else:
      common_3d_diffusion = [
      item
      for item in common_3d
      if not item.startswith("pyfluid.max_step=")
      and not item.startswith("pyfluid.plot_int=")
      ]
      common_3d_diffusion.extend(
         [
            "pyfluid.max_step=2",
            "pyfluid.plot_int=2",
            "pyfluid.diffusion=explicit",
            "pyfluid.viscosity=0.005",
            "pyfluid.diffusion_cfl=0.9",
         ]
      )
   if not args.hydro_only:
      single_3d_diffusion_run, single_3d_diffusion_row = run_case(
      "pressure_bump_3d_single_explicit_diffusion_amr",
      dim=3,
      input_path=INPUT_3D,
      overrides=[*common_3d_diffusion, "mesh.patch_system=single"],
      ranks=args.ranks,
      timeout=args.timeout,
      checks=DIFFUSIVE_POLICY,
   )
      tjunction_3d_diffusion_run, tjunction_3d_diffusion_row = run_case(
      "pressure_bump_3d_tjunction_explicit_diffusion_amr",
      dim=3,
      input_path=INPUT_3D,
      overrides=[
         *common_3d_diffusion,
         "mesh.patch_system=multi_homogeneous_cartesian",
         "mesh.patch_box_small=0 0 0  8 0 0  8 8 0",
         "mesh.patch_box_big=7 15 15  15 7 15  15 15 15",
      ],
      ranks=args.ranks,
      timeout=args.timeout,
      checks=DIFFUSIVE_POLICY,
   )
      rows.extend([single_3d_diffusion_row, tjunction_3d_diffusion_row])
      if single_3d_diffusion_row["status"] == "PASS" and tjunction_3d_diffusion_row["status"] == "PASS":
         rows.append(
            comparison_row(
            single_3d_diffusion_run,
            tjunction_3d_diffusion_run,
            dim=3,
            case="3d_single_vs_tjunction_explicit_diffusion_amr_compare",
            label="3D explicit viscous hydro AMR single patch vs Cartesian T-junction",
            input_path=INPUT_3D,
            components=["rho", "mom1", "mom2", "mom3", "energy"],
            tolerance=args.tolerance,
            ranks=args.ranks,
            )
         )

   rows.extend(patch_union_corridor_rows(args))
   write_summary(rows)
   print_summary(rows)
   return 0 if all(row["status"] == "PASS" for row in rows) else 1


if __name__ == "__main__":
   raise SystemExit(main())
