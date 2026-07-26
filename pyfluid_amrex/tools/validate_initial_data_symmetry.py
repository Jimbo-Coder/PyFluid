#!/usr/bin/env python3
"""Validate elliptic stellar-ID reflection sectors and saved-ID expansion."""

from __future__ import annotations

import argparse
import math
import re
import shutil
from pathlib import Path

from validation_common import (
   ROOT,
   SMOKE_POLICY,
   append_validation_history,
   materialize_run_input,
   read_scalar_rows,
   repo_relative,
   run_amrex,
   scalar_checks,
   scalar_validation_row,
   write_tsv,
)


TOV_FULL_INPUT = ROOT / "pyfluid_amrex/inputs/tov_xcts_3d.in"
TOV_OCTANT_INPUT = ROOT / "pyfluid_amrex/inputs/tov_xcts_octant_3d.in"
RNS_FULL_INPUT = ROOT / "pyfluid_amrex/inputs/rotating_star_xcts_3d.in"
RNS_HALF_INPUT = (
   ROOT / "pyfluid_amrex/inputs/rotating_star_xcts_equatorial_symmetry_3d.in"
)
RUN_ROOT = ROOT / "runs/current/initial_data_symmetry_validation"
RESULTS = RUN_ROOT / "initial_data_symmetry.tsv"
HISTORY = ROOT / "runs/current/validation_history.tsv"

ROTATING_PATTERN = re.compile(
   r"rotating star: omega=(?P<omega>[0-9.eE+-]+) "
   r"maximum_density=(?P<maximum_density>[0-9.eE+-]+) "
   r"maximum_velocity=(?P<maximum_velocity>[0-9.eE+-]+) "
   r"injection_energy=(?P<injection_energy>[0-9.eE+-]+) "
   r"boundary_density_excess=(?P<boundary_density_excess>[0-9.eE+-]+) "
   r"equatorial_radius=(?P<equatorial_radius>[0-9.eE+-]+) "
   r"polar_radius=(?P<polar_radius>[0-9.eE+-]+) "
   r"coordinate_axis_ratio=(?P<coordinate_axis_ratio>[0-9.eE+-]+) "
   r"mass_shedding_indicator=(?P<mass_shedding_indicator>[0-9.eE+-]+)"
)
RECOVERY_PATTERN = re.compile(
   r"Relativistic recovery c2p_fallbacks=(\d+) atmosphere_resets=(\d+)"
)


def run_case(
   name: str,
   base_input: Path,
   overrides: dict[str, str],
   args: argparse.Namespace,
   *,
   checks: object,
) -> tuple[Path, dict[str, object]]:
   run_dir = RUN_ROOT / name
   if args.force and run_dir.exists():
      shutil.rmtree(run_dir)
   run_dir.mkdir(parents=True, exist_ok=True)
   input_path = materialize_run_input(
      base_input,
      run_dir / "input.in",
      {
         "pyfluid.run_name": name,
         "pyfluid.output_dir": repo_relative(run_dir),
         "pyfluid.aux_gfs": "none",
         "pyfluid.aux_gf_compute_int": "-1",
         "pyfluid.aux_gf_int": "-1",
         **overrides,
      },
   )
   if args.force or not (run_dir / "scalars.tsv").exists():
      completed = run_amrex(
         dim=3,
         input_path=input_path,
         ranks=args.ranks,
         timeout=args.timeout,
      )
      (run_dir / "run.log").write_text(completed.stdout)
      if completed.returncode != 0:
         raise RuntimeError(
            f"{name} failed; see {repo_relative(run_dir / 'run.log')}"
         )
   scalar = scalar_validation_row(
      run_dir,
      case=name,
      input_file=run_dir / "run.in",
      checks=checks,
      mhd=False,
      dim=3,
      ranks=args.ranks,
      kind="symmetry",
   )
   if scalar["status"] != "PASS":
      raise RuntimeError(f"{name} scalar validation failed: {scalar['reason']}")
   return run_dir, scalar


def initial_row(run_dir: Path) -> dict[str, str]:
   rows = read_scalar_rows(run_dir)
   if not rows or int(rows[0]["step"]) != 0:
      raise RuntimeError(f"{repo_relative(run_dir)} has no step-zero scalars")
   return rows[0]


def maximum_relative_difference(
   reference: dict[str, str],
   candidate: dict[str, str],
   columns: tuple[str, ...],
) -> float:
   return max(
      abs(float(candidate[name]) - float(reference[name]))
      / max(abs(float(reference[name])), 1.0e-14)
      for name in columns
   )


def parse_rotating(run_dir: Path) -> dict[str, float]:
   text = (run_dir / "run.log").read_text()
   match = ROTATING_PATTERN.search(text)
   recovery = RECOVERY_PATTERN.search(text)
   if match is None or recovery is None:
      raise RuntimeError(
         f"{repo_relative(run_dir)} lacks rotating-ID diagnostics"
      )
   values = {name: float(value) for name, value in match.groupdict().items()}
   values["c2p_fallbacks"] = float(recovery.group(1))
   values["atmosphere_resets"] = float(recovery.group(2))
   return values


def validate_tov(
   args: argparse.Namespace,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
   full_dir, full_scalar = run_case(
      "tov_full",
      TOV_FULL_INPUT,
      {"pyfluid.plot_int": "-1"},
      args,
      checks=SMOKE_POLICY,
   )
   octant_artifact = RUN_ROOT / "tov_octant" / "physical_id"
   octant_dir, octant_scalar = run_case(
      "tov_octant",
      TOV_OCTANT_INPUT,
      {
         "initial_data.output_file": repo_relative(octant_artifact),
         "pyfluid.plot_int": "-1",
      },
      args,
      checks=SMOKE_POLICY,
   )
   expanded_dir, expanded_scalar = run_case(
      "tov_octant_expanded",
      TOV_FULL_INPUT,
      {
         "initial_data.input_file": repo_relative(octant_artifact),
         "pyfluid.plot_int": "-1",
      },
      args,
      checks=SMOKE_POLICY,
   )

   columns = ("mass", "energy", "max_rho", "min_lapse")
   full_octant_error = maximum_relative_difference(
      initial_row(full_dir), initial_row(octant_dir), columns
   )
   expansion_error = maximum_relative_difference(
      initial_row(octant_dir), initial_row(expanded_dir), columns
   )
   expanded = initial_row(expanded_dir)
   expanded_constraint = max(
      abs(float(expanded[name]))
      for name in (
         "z4c_hamiltonian_l2",
         "z4c_hamiltonian_linf",
         "z4c_momentum_linf",
      )
   )
   if (
      full_octant_error > 2.0e-2
      or expansion_error > 1.0e-4
      or not math.isfinite(expanded_constraint)
      or expanded_constraint > 1.0e-2
   ):
      raise RuntimeError(
         "TOV symmetry validation failed: "
         f"full/octant={full_octant_error:.3e}, "
         f"expansion={expansion_error:.3e}, "
         f"expanded constraint={expanded_constraint:.3e}"
      )
   return ([
      {
         "kind": "symmetry",
         "case": "tov_xcts_full_octant",
         "status": "PASS",
         "metric": "maximum_relative_scalar_difference",
         "value": full_octant_error,
         "constraint_linf": expanded_constraint,
         "run_dir": repo_relative(octant_dir),
      },
      {
         "kind": "symmetry",
         "case": "tov_physical_id_octant_expansion",
         "status": "PASS",
         "metric": "maximum_relative_scalar_difference",
         "value": expansion_error,
         "constraint_linf": expanded_constraint,
         "run_dir": repo_relative(expanded_dir),
      },
   ], [full_scalar, octant_scalar, expanded_scalar])


def validate_rns(
   args: argparse.Namespace,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
   evolution_checks = scalar_checks(
      "rotating_id_smoke",
      mass_tol=2.0e-7,
      energy_tol=1.0e-4,
      momentum_tol=2.0e-9,
   )
   full_dir, full_scalar = run_case(
      "rns_full",
      RNS_FULL_INPUT,
      {"pyfluid.plot_int": "-1"},
      args,
      checks=evolution_checks,
   )
   half_dir, half_scalar = run_case(
      "rns_equatorial_half",
      RNS_HALF_INPUT,
      {"pyfluid.plot_int": "-1"},
      args,
      checks=evolution_checks,
   )
   scalar_columns = (
      "mass",
      "energy",
      "angular_mom3",
      "max_rho",
      "min_lapse",
   )
   scalar_error = maximum_relative_difference(
      initial_row(full_dir), initial_row(half_dir), scalar_columns
   )
   full_rotating = parse_rotating(full_dir)
   half_rotating = parse_rotating(half_dir)
   surface_columns = (
      "maximum_density",
      "maximum_velocity",
      "injection_energy",
      "equatorial_radius",
      "polar_radius",
      "coordinate_axis_ratio",
      "mass_shedding_indicator",
   )
   surface_error = maximum_relative_difference(
      {name: str(full_rotating[name]) for name in surface_columns},
      {name: str(half_rotating[name]) for name in surface_columns},
      surface_columns,
   )
   recovery_events = max(
      full_rotating["c2p_fallbacks"]
      + full_rotating["atmosphere_resets"],
      half_rotating["c2p_fallbacks"]
      + half_rotating["atmosphere_resets"],
   )
   if scalar_error > 6.0e-2 or surface_error > 3.0e-2 or recovery_events != 0:
      raise RuntimeError(
         "RNS equatorial-symmetry validation failed: "
         f"scalars={scalar_error:.3e}, surfaces={surface_error:.3e}, "
         f"recovery events={recovery_events:.0f}"
      )
   return ([
      {
         "kind": "symmetry",
         "case": "rns_xcts_equatorial_half",
         "status": "PASS",
         "metric": "maximum_relative_scalar_difference",
         "value": scalar_error,
         "surface_relative_difference": surface_error,
         "recovery_events": recovery_events,
         "run_dir": repo_relative(half_dir),
      },
   ], [full_scalar, half_scalar])


def main() -> int:
   parser = argparse.ArgumentParser(description=__doc__)
   parser.add_argument("--only", choices=("all", "tov", "rns"), default="all")
   parser.add_argument("--ranks", type=int, default=3)
   parser.add_argument("--timeout", type=int, default=1800)
   parser.add_argument("--force", action="store_true")
   args = parser.parse_args()

   RUN_ROOT.mkdir(parents=True, exist_ok=True)
   results: list[dict[str, object]] = []
   scalars: list[dict[str, object]] = []
   if args.only in ("all", "tov"):
      rows, scalar_rows = validate_tov(args)
      results.extend(rows)
      scalars.extend(scalar_rows)
   if args.only in ("all", "rns"):
      rows, scalar_rows = validate_rns(args)
      results.extend(rows)
      scalars.extend(scalar_rows)
   write_tsv(
      RESULTS,
      results,
      [
         "kind",
         "case",
         "status",
         "metric",
         "value",
         "surface_relative_difference",
         "constraint_linf",
         "recovery_events",
         "run_dir",
      ],
   )
   append_validation_history(HISTORY, scalars)
   for row in results:
      print(
         f"PASS {row['case']}: {row['metric']}={float(row['value']):.6e}"
      )
   return 0


if __name__ == "__main__":
   raise SystemExit(main())
