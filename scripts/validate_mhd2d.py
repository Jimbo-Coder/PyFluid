#!/usr/bin/env python3
import argparse
import pathlib
import sys
import time
import traceback
from dataclasses import replace

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import main_mhd2d_jax as mhd2d
from pyfluid.config import SimulationConfig
from pyfluid.initial_data import SAFE_INITIAL_CONDITION_OVERRIDES, SUPPORTED_INITIAL_CONDITIONS, ic_has_magnetic_fields, ic_has_perturbation
from pyfluid.state import D, MAG_P, MAG_U, NVAR_HYDRO, NVAR_MHD, P, U


def scalar_summary(state):
   diagnostics = state.diagnostics
   mass_drift = float(diagnostics[-1, D.MASS] - diagnostics[0, D.MASS])
   energy_drift = float(diagnostics[-1, D.ENERGY] - diagnostics[0, D.ENERGY])
   divb = float(diagnostics[-1, D.DIVB_L2]) if state.boolmag else 0.0
   return mass_drift, energy_drift, divb


def require_finite_state(state):
   arrays = {
      "Uvec": np.asarray(state.Uvec),
      "Pvec": np.asarray(state.Pvec),
      "diagnostics": np.asarray(state.diagnostics),
   }
   for name, array in arrays.items():
      if not np.all(np.isfinite(array)):
         raise AssertionError(f"{name} contains non-finite values")
   if np.min(np.asarray(state.Pvec[P.RHO])) <= 0.0:
      raise AssertionError("primitive density is non-positive")
   if np.min(np.asarray(state.Pvec[P.PRESS])) <= 0.0:
      raise AssertionError("primitive pressure is non-positive")


def require_conserved(state, rel_tol=1e-12, abs_tol=1e-11):
   mass_drift, energy_drift, _ = scalar_summary(state)
   mass0 = float(state.diagnostics[0, D.MASS])
   energy0 = float(state.diagnostics[0, D.ENERGY])
   mass_rel = abs(mass_drift)/max(abs(mass0), 1.0)
   energy_rel = abs(energy_drift)/max(abs(energy0), 1.0)
   if abs(mass_drift) > abs_tol and mass_rel > rel_tol:
      raise AssertionError(f"mass drift {mass_drift:.3e} rel={mass_rel:.3e} exceeds rel={rel_tol:.3e}, abs={abs_tol:.3e}")
   if abs(energy_drift) > abs_tol and energy_rel > rel_tol:
      raise AssertionError(f"energy drift {energy_drift:.3e} rel={energy_rel:.3e} exceeds rel={rel_tol:.3e}, abs={abs_tol:.3e}")


def require_nonmagnetic_state(state):
   if state.boolmag:
      raise AssertionError("state unexpectedly enabled magnetic physics")
   if state.nvar != NVAR_HYDRO or state.pvar != NVAR_HYDRO:
      raise AssertionError(f"non-magnetic run allocated nvar={state.nvar}, pvar={state.pvar}; expected {NVAR_HYDRO},{NVAR_HYDRO}")
   if state.voi != NVAR_HYDRO:
      raise AssertionError(f"non-magnetic variable-of-interest count is {state.voi}, expected {NVAR_HYDRO}")
   if not np.allclose(np.asarray(state.Uvec[MAG_U]), 0.0):
      raise AssertionError("non-magnetic conserved magnetic slots are non-zero")
   if not np.allclose(np.asarray(state.Pvec[MAG_P]), 0.0):
      raise AssertionError("non-magnetic primitive magnetic slots are non-zero")


def require_magnetic_state(state):
   if not state.boolmag:
      raise AssertionError("state unexpectedly disabled magnetic physics")
   if state.nvar != NVAR_MHD or state.pvar != NVAR_MHD:
      raise AssertionError(f"magnetic run allocated nvar={state.nvar}, pvar={state.pvar}; expected {NVAR_MHD},{NVAR_MHD}")
   if state.voi != NVAR_MHD:
      raise AssertionError(f"magnetic variable-of-interest count is {state.voi}, expected {NVAR_MHD}")


def make_case_config(**changes):
   config = SimulationConfig(
      nx=16,
      ny=16,
      nsteps=4,
      store_every=4,
      save_hdf5=False,
      save_final_state=False,
      recover_state=False,
      verbose=False,
      print_devices=False,
      print_summary=False,
   )
   return replace(config, **changes)


def run_supported_case(name, config, check_magnetic=None, check_conservation=True):
   state = mhd2d.run_simulation(config)
   require_finite_state(state)
   if check_magnetic is True:
      require_magnetic_state(state)
   elif check_magnetic is False:
      require_nonmagnetic_state(state)
   if check_conservation and not state.booldiff:
      require_conserved(state)
   return state


def case_initial_conditions():
   for ic in SUPPORTED_INITIAL_CONDITIONS:
      changes = {
         "ic": ic,
         "nx": 8,
         "ny": 8,
         "nsteps": 0,
         "store_every": 1,
         "recon": 0,
         "timestep": 0,
         "riemann": 0,
         "diffusion": False,
         "checkname": f"validate_ic_{ic}",
      }
      changes.update(SAFE_INITIAL_CONDITION_OVERRIDES.get(ic, {}))
      state = run_supported_case(
         f"initial_condition_{ic}",
         make_case_config(**changes),
         check_magnetic=ic_has_magnetic_fields(ic),
         check_conservation=False,
      )
      if ic_has_perturbation(ic):
         pvec = np.asarray(state.Pvec)
         component_ranges = np.max(pvec, axis=(1, 2)) - np.min(pvec, axis=(1, 2))
         if float(np.max(component_ranges)) == 0.0:
            raise AssertionError(f"initial condition {ic} did not perturb primitive fields")


def case_hydro_reconstructors():
   for recon in (0, 1, 2):
      run_supported_case(
         f"hydro_recon_{recon}",
         make_case_config(ic=7, recon=recon, timestep=3, diffusion=False, checkname=f"validate_hydro_recon_{recon}"),
         check_magnetic=False,
      )


def case_mhd_reconstructors():
   for recon in (0, 1, 2):
      run_supported_case(
         f"mhd_recon_{recon}",
         make_case_config(ic=8, recon=recon, timestep=3, diffusion=False, checkname=f"validate_mhd_recon_{recon}"),
         check_magnetic=True,
      )


def case_timesteppers():
   for timestep in (0, 1, 2, 3, 4):
      run_supported_case(
         f"hydro_timestep_{timestep}",
         make_case_config(ic=7, recon=0, timestep=timestep, diffusion=False, checkname=f"validate_timestep_{timestep}"),
         check_magnetic=False,
      )


def case_riemann_solvers():
   for riemann in (0, 1, 2):
      run_supported_case(
         f"mhd_flux_{riemann}",
         make_case_config(ic=8, recon=0, riemann=riemann, timestep=0, diffusion=False, checkname=f"validate_mhd_flux_{riemann}"),
         check_magnetic=True,
      )
   run_supported_case(
      "hydro_hllc",
      make_case_config(ic=7, recon=0, riemann=3, timestep=0, diffusion=False, checkname="validate_hydro_hllc"),
      check_magnetic=False,
   )
   for ic, axis_name in ((0, "x"), (2, "y")):
      run_supported_case(
         f"hydro_hllc_sod_{axis_name}",
         make_case_config(ic=ic, recon=0, riemann=3, timestep=0, diffusion=False,
                          nsteps=2, store_every=2, checkname=f"validate_hydro_hllc_sod_{axis_name}"),
         check_magnetic=False,
      )


def case_reject_broken_options():
   try:
      run_supported_case("hllD_unimplemented", make_case_config(riemann=4), check_magnetic=True)
   except NotImplementedError:
      pass
   else:
      raise AssertionError("HLLD unexpectedly ran despite being unimplemented")

   try:
      run_supported_case("magnetic_hllc", make_case_config(ic=8, riemann=3), check_magnetic=True)
   except ValueError as err:
      if "HLLC is hydro-only" not in str(err):
         raise
   else:
      raise AssertionError("magnetic HLLC unexpectedly ran")

   try:
      run_supported_case("implicit_imex_unwired", make_case_config(timestep=5), check_magnetic=True)
   except NotImplementedError as err:
      if "Implicit Euler/IMEX" not in str(err):
         raise
   else:
      raise AssertionError("implicit/IMEX timestep unexpectedly ran")


def case_reflective_boundaries():
   run_supported_case(
      "reflective_boundaries",
      make_case_config(ic=0, bc=(2, 2, 2, 2), recon=0, timestep=0, diffusion=False, checkname="validate_reflective"),
      check_magnetic=False,
      check_conservation=False,
   )


def case_diffusion():
   state = run_supported_case(
      "diffusion",
      make_case_config(ic=8, recon=2, timestep=3, diffusion=True, checkname="validate_diffusion"),
      check_magnetic=True,
      check_conservation=False,
   )
   if not np.all(np.isfinite(state.differrs)):
      raise AssertionError("diffusion errors contain non-finite values")
   if np.max(state.differrs) > 1e-8:
      raise AssertionError(f"diffusion fixed-point error too large: {np.max(state.differrs):.3e}")


def case_output_roundtrip():
   state = run_supported_case(
      "output_write",
      make_case_config(ic=8, recon=0, timestep=0, save_hdf5=True, save_final_state=True, checkname="validate_output"),
      check_magnetic=True,
   )
   output_dir = mhd2d.checkdir / state.checkname
   required = ("UPvec_final.h5", "grid.h5", "grid.npz", "params.npz", "run.visit")
   missing = [name for name in required if not (output_dir / name).exists()]
   if missing:
      raise AssertionError(f"missing output files: {missing}")


def case_short_ct_comparison():
   common = {"ic": 8, "recon": 2, "timestep": 3, "diffusion": False, "nsteps": 32, "store_every": 8}
   no_ct = run_supported_case("ct_compare_noct", make_case_config(**common, fluxct=False, checkname="validate_ct_noct"), check_magnetic=True)
   ct = run_supported_case("ct_compare_ct", make_case_config(**common, fluxct=True, checkname="validate_ct_ct"), check_magnetic=True)
   no_ct_divb = float(no_ct.diagnostics[-1, D.DIVB_L2])
   ct_divb = float(ct.diagnostics[-1, D.DIVB_L2])
   if not ct_divb < no_ct_divb:
      raise AssertionError(f"short CT comparison did not reduce normalized divB: CT={ct_divb:.3e}, noCT={no_ct_divb:.3e}")


def linear_wave_density_pressure_errors(state):
   x = np.asarray(state.xg[state.imask])
   pvec = np.asarray(state.Pvec)
   kx = float(state.kx)
   phase = float(state.cs0 * state.kx * state.t)
   amp = float(state.delrho)
   rho_exact = float(state.rhob) + amp * np.cos(phase) * np.sin(kx * x)
   pressure_exact = float(state.presb) + float(state.cs20) * amp * np.cos(phase) * np.sin(kx * x)
   rho_error = np.sqrt(np.mean((pvec[P.RHO][state.imask] - rho_exact) ** 2)) / amp
   pressure_error = np.sqrt(np.mean((pvec[P.PRESS][state.imask] - pressure_exact) ** 2)) / (float(state.cs20) * amp)
   return float(rho_error), float(pressure_error)


def case_linear_wave_accuracy():
   nsteps = 142
   state = run_supported_case(
      "linear_wave_accuracy",
      make_case_config(
         ic=6,
         nx=32,
         ny=32,
         nsteps=nsteps,
         store_every=nsteps,
         recon=2,
         timestep=3,
         riemann=0,
         fluxct=False,
         delrho_factor=0.01,
         checkname="validate_linear_wave",
      ),
      check_magnetic=False,
   )
   rho_error, pressure_error = linear_wave_density_pressure_errors(state)
   max_error = max(rho_error, pressure_error)
   if max_error > 5e-3:
      raise AssertionError(f"linear wave normalized L2 error too large: rho={rho_error:.3e}, pressure={pressure_error:.3e}")
   return {"rho_l2": rho_error, "pressure_l2": pressure_error}


def case_float32_smoke():
   cases = (
      ("hydro_float32", make_case_config(ic=7, recon=0, timestep=0, riemann=0, fluxct=False, x64=False, nsteps=16, store_every=16, checkname="validate_hydro_float32"), False),
      ("mhd_float32", make_case_config(ic=8, recon=2, timestep=3, riemann=0, fluxct=True, x64=False, nsteps=16, store_every=16, checkname="validate_mhd_float32"), True),
   )
   results = []
   for name, config, check_magnetic in cases:
      state = run_supported_case(name, config, check_magnetic=check_magnetic, check_conservation=False)
      require_conserved(state, rel_tol=1e-4, abs_tol=1e-4)
      mass_drift, energy_drift, divb = scalar_summary(state)
      results.append((name, mass_drift, energy_drift, divb))
   return results


def case_rectangular_grid():
   state = run_supported_case(
      "rectangular_mhd_wenoz",
      make_case_config(
         ic=8,
         nx=32,
         ny=24,
         recon=2,
         timestep=3,
         riemann=0,
         fluxct=True,
         nsteps=4,
         store_every=4,
         checkname="validate_rectangular_mhd_wenoz",
      ),
      check_magnetic=True,
   )
   return {"shape": tuple(state.Uvec.shape), "time": float(state.t)}


def case_cfl_sweep():
   results = []
   for cflr in (0.45, 0.7, 0.9):
      state = run_supported_case(
         f"cfl_{cflr}",
         make_case_config(ic=8, recon=0, timestep=0, diffusion=False, cfl=cflr, nsteps=16, store_every=16, checkname=f"validate_cfl_{cflr}"),
         check_magnetic=True,
      )
      results.append((cflr, float(state.t), float(np.min(state.Pvec[P.RHO])), float(np.min(state.Pvec[P.PRESS]))))
   return results


CASES = (
   ("initial_conditions", case_initial_conditions),
   ("hydro_reconstructors", case_hydro_reconstructors),
   ("mhd_reconstructors", case_mhd_reconstructors),
   ("timesteppers", case_timesteppers),
   ("riemann_solvers", case_riemann_solvers),
   ("reject_broken_options", case_reject_broken_options),
   ("reflective_boundaries", case_reflective_boundaries),
   ("diffusion", case_diffusion),
   ("output_roundtrip", case_output_roundtrip),
   ("short_ct_comparison", case_short_ct_comparison),
   ("linear_wave_accuracy", case_linear_wave_accuracy),
   ("float32_smoke", case_float32_smoke),
   ("rectangular_grid", case_rectangular_grid),
   ("cfl_sweep", case_cfl_sweep),
)


def parse_args():
   parser = argparse.ArgumentParser(description="Run 2D JAX MHD validation cases.")
   parser.add_argument("--case", action="append", choices=[name for name, _ in CASES],
                       help="Run only the selected case. Can be passed more than once.")
   return parser.parse_args()


def main():
   args = parse_args()
   selected = set(args.case or [name for name, _ in CASES])
   failed = []
   for name, func in CASES:
      if name not in selected:
         continue
      start = time.time()
      try:
         result = func()
      except Exception as err:
         failed.append(name)
         print(f"FAIL {name}: {err}")
         traceback.print_exc()
      else:
         elapsed = time.time() - start
         suffix = f" {result}" if result is not None else ""
         print(f"PASS {name} ({elapsed:.2f}s){suffix}")
   if failed:
      print(f"FAILED {len(failed)} case(s): {', '.join(failed)}")
      return 1
   print("All selected validation cases passed.")
   return 0


if __name__ == "__main__":
   raise SystemExit(main())
