#!/usr/bin/env python3
"""Validate cylindrical/spherical finite-volume geometry."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from validation_common import (
   ROOT,
   append_validation_history,
   materialize_run_input,
   parameters_from_assignments,
   repo_relative,
   scalar_validation_row,
   write_tsv,
)


RUN_ROOT = ROOT / "runs/current/curvilinear_geometry"
SUMMARY = RUN_ROOT / "summary.tsv"
HISTORY = ROOT / "runs/current/validation_history.tsv"


@dataclass(frozen=True)
class CurvilinearGeometryCase:
   name: str
   input_path: Path
   run_dir: Path
   dim: int
   overrides: tuple[str, ...] = ()
   mhd: bool = False
   policy: str = "wall"


CASES = (
   CurvilinearGeometryCase(
      name="cylindrical_uniform_static_1d",
      input_path=ROOT / "pyfluid_amrex/inputs/cylindrical_uniform_static_1d.in",
      run_dir=ROOT / "runs/current/uniform/cylindrical_uniform_static_1d",
      dim=1,
   ),
   CurvilinearGeometryCase(
      name="cylindrical_z_uniform_static_1d",
      input_path=ROOT / "pyfluid_amrex/inputs/cylindrical_uniform_static_1d.in",
      run_dir=ROOT / "runs/current/uniform/cylindrical_z_uniform_static_1d",
      dim=1,
      overrides=(
         "pyfluid.run_name=cylindrical_z_uniform_static_1d",
         "mesh.axes=z",
         "geometry.prob_lo=0.0",
         "geometry.prob_hi=1.0",
         "geometry.coord_sys=0",
         "amr.n_cell=96",
      ),
   ),
   CurvilinearGeometryCase(
      name="cylindrical_phi_uniform_static_1d",
      input_path=ROOT / "pyfluid_amrex/inputs/cylindrical_uniform_static_1d.in",
      run_dir=ROOT / "runs/current/uniform/cylindrical_phi_uniform_static_1d",
      dim=1,
      overrides=(
         "pyfluid.run_name=cylindrical_phi_uniform_static_1d",
         "mesh.axes=phi",
         "geometry.prob_lo=0.0",
         "geometry.prob_hi=6.283185307179586",
         "geometry.is_periodic=1",
         "geometry.coord_sys=0",
         "pyfluid.bc=periodic",
         "amr.n_cell=96",
      ),
   ),
   CurvilinearGeometryCase(
      name="spherical_uniform_static_1d",
      input_path=ROOT / "pyfluid_amrex/inputs/spherical_uniform_static_1d.in",
      run_dir=ROOT / "runs/current/uniform/spherical_uniform_static_1d",
      dim=1,
   ),
   CurvilinearGeometryCase(
      name="spherical_theta_uniform_static_1d",
      input_path=ROOT / "pyfluid_amrex/inputs/spherical_uniform_static_1d.in",
      run_dir=ROOT / "runs/current/uniform/spherical_theta_uniform_static_1d",
      dim=1,
      overrides=(
         "pyfluid.run_name=spherical_theta_uniform_static_1d",
         "mesh.axes=theta",
         "geometry.prob_lo=0.2",
         "geometry.prob_hi=2.941592653589793",
         "geometry.coord_sys=0",
         "amr.n_cell=96",
      ),
   ),
   CurvilinearGeometryCase(
      name="spherical_phi_uniform_static_1d",
      input_path=ROOT / "pyfluid_amrex/inputs/spherical_uniform_static_1d.in",
      run_dir=ROOT / "runs/current/uniform/spherical_phi_uniform_static_1d",
      dim=1,
      overrides=(
         "pyfluid.run_name=spherical_phi_uniform_static_1d",
         "mesh.axes=phi",
         "geometry.prob_lo=0.0",
         "geometry.prob_hi=6.283185307179586",
         "geometry.is_periodic=1",
         "geometry.coord_sys=0",
         "pyfluid.bc=periodic",
         "amr.n_cell=96",
      ),
   ),
   CurvilinearGeometryCase(
      name="cylindrical_axis_uniform_static_1d",
      input_path=ROOT / "pyfluid_amrex/inputs/cylindrical_axis_uniform_static_1d.in",
      run_dir=ROOT / "runs/current/uniform/cylindrical_axis_uniform_static_1d",
      dim=1,
   ),
   CurvilinearGeometryCase(
      name="spherical_origin_uniform_static_1d",
      input_path=ROOT / "pyfluid_amrex/inputs/spherical_origin_uniform_static_1d.in",
      run_dir=ROOT / "runs/current/uniform/spherical_origin_uniform_static_1d",
      dim=1,
   ),
   CurvilinearGeometryCase(
      name="cylindrical_uniform_static_2d",
      input_path=ROOT / "pyfluid_amrex/inputs/cylindrical_uniform_static_2d.in",
      run_dir=ROOT / "runs/current/uniform/cylindrical_uniform_static_2d",
      dim=2,
   ),
   CurvilinearGeometryCase(
      name="cylindrical_phiz_uniform_static_2d",
      input_path=ROOT / "pyfluid_amrex/inputs/cylindrical_uniform_static_2d.in",
      run_dir=ROOT / "runs/current/uniform/cylindrical_phiz_uniform_static_2d",
      dim=2,
      overrides=(
         "pyfluid.run_name=cylindrical_phiz_uniform_static_2d",
         "mesh.axes=phi z",
         "geometry.prob_lo=0.0 0.0",
         "geometry.prob_hi=6.283185307179586 1.0",
         "geometry.is_periodic=1 0",
         "geometry.coord_sys=0",
         "pyfluid.bc=reflecting",
         "pyfluid.bc_xlo=periodic",
         "pyfluid.bc_xhi=periodic",
         "pyfluid.bc_ylo=reflecting",
         "pyfluid.bc_yhi=reflecting",
         "amr.n_cell=96 64",
      ),
   ),
   CurvilinearGeometryCase(
      name="cylindrical_axis_uniform_static_2d",
      input_path=ROOT / "pyfluid_amrex/inputs/cylindrical_axis_uniform_static_2d.in",
      run_dir=ROOT / "runs/current/uniform/cylindrical_axis_uniform_static_2d",
      dim=2,
   ),
   CurvilinearGeometryCase(
      name="cylindrical_axis_uniform_static_mhd_ct_2d",
      input_path=ROOT / "pyfluid_amrex/inputs/cylindrical_axis_uniform_static_2d.in",
      run_dir=ROOT / "runs/current/uniform/cylindrical_axis_uniform_static_mhd_ct_2d",
      dim=2,
      overrides=(
         "pyfluid.run_name=cylindrical_axis_uniform_static_mhd_ct_2d",
         "pyfluid.physics=mhd",
         "pyfluid.ct=on",
         "pyfluid.riemann=hlld",
         "pyfluid.b2_0=0.1",
      ),
      mhd=True,
   ),
   CurvilinearGeometryCase(
      name="cylindrical_axis_pressure_bump_mhd_ct_2d",
      input_path=ROOT / "pyfluid_amrex/inputs/cylindrical_axis_uniform_static_2d.in",
      run_dir=ROOT / "runs/current/pressure_bump/cylindrical_axis_pressure_bump_mhd_ct_2d",
      dim=2,
      overrides=(
         "pyfluid.run_name=cylindrical_axis_pressure_bump_mhd_ct_2d",
         "pyfluid.physics=mhd",
         "pyfluid.ct=on",
         "pyfluid.ic=pressure_bump",
         "pyfluid.amplitude=0.05",
         "pyfluid.bump_sigma=0.12",
         "pyfluid.bump_x0=0.35",
         "pyfluid.bump_y0=0.0",
         "pyfluid.b1_0=0.0",
         "pyfluid.b2_0=0.04",
         "pyfluid.b3_0=0.0",
         "pyfluid.riemann=hlld",
         "pyfluid.max_step=12",
         "pyfluid.scalar_int=1",
         "pyfluid.plot_int=-1",
      ),
      mhd=True,
   ),
   CurvilinearGeometryCase(
      name="cylindrical_axis_magnetic_tube_explicit_resistive_ct_2d",
      input_path=ROOT / "pyfluid_amrex/inputs/cylindrical_axis_uniform_static_2d.in",
      run_dir=ROOT / "runs/current/magnetic_tube_z/cylindrical_axis_magnetic_tube_explicit_resistive_ct_2d",
      dim=2,
      overrides=(
         "pyfluid.run_name=cylindrical_axis_magnetic_tube_explicit_resistive_ct_2d",
         "pyfluid.physics=mhd",
         "pyfluid.ct=on",
         "pyfluid.ic=magnetic_tube_z",
         "pyfluid.amplitude=0.02",
         "pyfluid.bump_sigma=0.12",
         "pyfluid.bump_x0=0.35",
         "pyfluid.bump_y0=0.0",
         "pyfluid.b1_0=0.0",
         "pyfluid.b2_0=0.0",
         "pyfluid.b3_0=0.0",
         "pyfluid.diffusion=explicit",
         "pyfluid.viscosity=0.0",
         "pyfluid.resistivity=1.0e-3",
         "pyfluid.diffusion_cfl=0.9",
         "pyfluid.riemann=hlld",
         "pyfluid.max_step=12",
         "pyfluid.scalar_int=1",
         "pyfluid.plot_int=-1",
      ),
      mhd=True,
      policy="diffusive_wall",
   ),
   CurvilinearGeometryCase(
      name="spherical_uniform_static_2d",
      input_path=ROOT / "pyfluid_amrex/inputs/spherical_uniform_static_2d.in",
      run_dir=ROOT / "runs/current/uniform/spherical_uniform_static_2d",
      dim=2,
   ),
   CurvilinearGeometryCase(
      name="spherical_uniform_static_rphi_2d",
      input_path=ROOT / "pyfluid_amrex/inputs/spherical_uniform_static_rphi_2d.in",
      run_dir=ROOT / "runs/current/uniform/spherical_uniform_static_rphi_2d",
      dim=2,
   ),
   CurvilinearGeometryCase(
      name="spherical_origin_rphi_uniform_static_mhd_ct_2d",
      input_path=ROOT / "pyfluid_amrex/inputs/spherical_uniform_static_rphi_2d.in",
      run_dir=ROOT / "runs/current/uniform/spherical_origin_rphi_uniform_static_mhd_ct_2d",
      dim=2,
      overrides=(
         "pyfluid.run_name=spherical_origin_rphi_uniform_static_mhd_ct_2d",
         "geometry.prob_lo=0.0 0.0",
         "geometry.prob_hi=1.0 6.283185307179586",
         "geometry.is_periodic=0 1",
         "pyfluid.bc=reflecting",
         "pyfluid.bc_xlo=origin",
         "pyfluid.bc_xhi=reflecting",
         "pyfluid.bc_ylo=periodic",
         "pyfluid.bc_yhi=periodic",
         "pyfluid.physics=mhd",
         "pyfluid.ct=on",
         "pyfluid.riemann=hlld",
         "pyfluid.b1_0=0.0",
         "pyfluid.b2_0=0.04",
         "pyfluid.b3_0=0.0",
         "pyfluid.max_step=12",
         "pyfluid.scalar_int=1",
         "pyfluid.plot_int=-1",
      ),
      mhd=True,
   ),
   CurvilinearGeometryCase(
      name="spherical_thetaphi_uniform_static_2d",
      input_path=ROOT / "pyfluid_amrex/inputs/spherical_uniform_static_2d.in",
      run_dir=ROOT / "runs/current/uniform/spherical_thetaphi_uniform_static_2d",
      dim=2,
      overrides=(
         "pyfluid.run_name=spherical_thetaphi_uniform_static_2d",
         "mesh.axes=theta phi",
         "geometry.prob_lo=0.2 0.0",
         "geometry.prob_hi=2.941592653589793 6.283185307179586",
         "geometry.is_periodic=0 1",
         "geometry.coord_sys=0",
         "pyfluid.bc=reflecting",
         "pyfluid.bc_ylo=periodic",
         "pyfluid.bc_yhi=periodic",
         "amr.n_cell=64 96",
      ),
   ),
   CurvilinearGeometryCase(
      name="cylindrical_uniform_static_3d",
      input_path=ROOT / "pyfluid_amrex/inputs/cylindrical_uniform_static_3d.in",
      run_dir=ROOT / "runs/current/uniform/cylindrical_uniform_static_3d",
      dim=3,
   ),
   CurvilinearGeometryCase(
      name="spherical_uniform_static_3d",
      input_path=ROOT / "pyfluid_amrex/inputs/spherical_uniform_static_3d.in",
      run_dir=ROOT / "runs/current/uniform/spherical_uniform_static_3d",
      dim=3,
   ),
   CurvilinearGeometryCase(
      name="cylindrical_pressure_bump_2d_smoke",
      input_path=ROOT / "pyfluid_amrex/inputs/cylindrical_pressure_bump_2d.in",
      run_dir=ROOT / "runs/current/pressure_bump/cylindrical_pressure_bump_2d_smoke",
      dim=2,
      overrides=(
         "pyfluid.run_name=cylindrical_pressure_bump_2d_smoke",
         "amr.n_cell=48 48",
         "amr.max_grid_size=24",
         "pyfluid.max_step=24",
         "pyfluid.scalar_int=1",
         "pyfluid.plot_int=-1",
      ),
   ),
   CurvilinearGeometryCase(
      name="spherical_pressure_bump_2d_smoke",
      input_path=ROOT / "pyfluid_amrex/inputs/spherical_pressure_bump_2d.in",
      run_dir=ROOT / "runs/current/pressure_bump/spherical_pressure_bump_2d_smoke",
      dim=2,
      overrides=(
         "pyfluid.run_name=spherical_pressure_bump_2d_smoke",
         "amr.n_cell=48 48",
         "amr.max_grid_size=24",
         "pyfluid.max_step=24",
         "pyfluid.scalar_int=1",
         "pyfluid.plot_int=-1",
      ),
   ),
   CurvilinearGeometryCase(
      name="spherical_pressure_bump_rphi_2d_smoke",
      input_path=ROOT / "pyfluid_amrex/inputs/spherical_pressure_bump_rphi_2d.in",
      run_dir=ROOT / "runs/current/pressure_bump/spherical_pressure_bump_rphi_2d_smoke",
      dim=2,
      overrides=(
         "pyfluid.run_name=spherical_pressure_bump_rphi_2d_smoke",
         "amr.n_cell=48 48",
         "amr.max_grid_size=24",
         "pyfluid.max_step=24",
         "pyfluid.scalar_int=1",
         "pyfluid.plot_int=-1",
      ),
   ),
)


def run_case(case: CurvilinearGeometryCase, ranks: int) -> subprocess.CompletedProcess[str]:
   env = dict(os.environ)
   env.setdefault("UV_CACHE_DIR", str(ROOT / ".uv-cache"))
   env.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib"))
   input_path = materialize_run_input(
      case.input_path,
      case.run_dir / "run.in",
      parameters_from_assignments(
         [
            *case.overrides,
            f"pyfluid.output_dir={repo_relative(case.run_dir)}",
         ]
      ),
   )
   command = [
      sys.executable,
      str(ROOT / "pyfluid_amrex/tools/run_case.py"),
      str(input_path),
      "--ranks",
      str(ranks),
      "--keep-existing",
      "--skip-plots",
   ]
   return subprocess.run(
      command,
      cwd=ROOT,
      env=env,
      text=True,
      stdout=subprocess.PIPE,
      stderr=subprocess.STDOUT,
      check=False,
   )


def main() -> int:
   parser = argparse.ArgumentParser(description=__doc__)
   parser.add_argument("--force", action="store_true", help="Accepted for suite consistency; run_case overwrites by default.")
   parser.add_argument("--ranks", type=int, default=3)
   args = parser.parse_args()

   RUN_ROOT.mkdir(parents=True, exist_ok=True)
   rows: list[dict[str, object]] = []
   ok = True
   for case in CASES:
      completed = run_case(case, args.ranks)
      log_path = RUN_ROOT / f"{case.name}.log"
      log_path.write_text(completed.stdout)
      if completed.returncode != 0:
         row = {
            "kind": "curvilinear_geometry",
            "case": case.name,
            "label": case.name,
            "status": "FAIL",
            "reason": f"run_case returncode={completed.returncode}",
            "dim": str(case.dim),
            "input": repo_relative(case.input_path),
            "policy": case.policy,
            "run_dir": repo_relative(case.run_dir),
            "ranks": str(args.ranks),
            "mass_rel_drift": float("nan"),
            "energy_rel_drift": float("nan"),
            "momentum_abs_drift": float("nan"),
            "avg_mcells_per_s": float("nan"),
            "avg_sim_time_per_s": float("nan"),
            "log": repo_relative(log_path),
         }
         ok = False
      else:
         row = scalar_validation_row(
            case.run_dir,
            case=case.name,
            input_file=case.run_dir / "run.in",
            policy=case.policy,
            mhd=case.mhd,
            dim=case.dim,
            ranks=args.ranks,
            base_input=repo_relative(case.input_path),
            kind="curvilinear_geometry",
         )
         row["log"] = repo_relative(log_path)
         if row["status"] != "PASS":
            ok = False
      rows.append(row)
      reason = f" reason={row['reason']}" if row.get("reason") else ""
      print(
         f"{row['status']} {case.name}: "
         f"mass={float(row['mass_rel_drift']):.3e} "
         f"energy={float(row['energy_rel_drift']):.3e} "
         f"mom={float(row['momentum_abs_drift']):.3e} "
         f"divB={float(row.get('divb_l1', 0.0)):.3e} "
         f"Mcells/s={float(row['avg_mcells_per_s']):.3e}{reason}"
      )
   write_tsv(SUMMARY, rows, ["case", "status", "reason", "mass_rel_drift", "energy_rel_drift", "momentum_abs_drift", "divb_l1", "avg_mcells_per_s", "avg_sim_time_per_s", "log"])
   append_validation_history(HISTORY, rows)
   print(f"summary={repo_relative(SUMMARY)}")
   print(f"history={repo_relative(HISTORY)}")
   return 0 if ok else 1


if __name__ == "__main__":
   raise SystemExit(main())
