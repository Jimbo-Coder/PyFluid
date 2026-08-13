#!/usr/bin/env python3
"""Test whether changing RMHD HLLD outer speeds repairs its fan closure.

This is an offline falsification experiment, not a production Riemann solver.
For each Balsara initial jump it holds the published HLLD jump construction
fixed and minimizes two residuals:

* equality of the left/right contact normal velocity (the HLLD closure), and
* disagreement between the assumed total pressure and the total pressure
  recovered from each of the four intermediate conserved states.

The first fit may move the outer speeds anywhere inside the causal cone.  It
therefore diagnoses degrees of freedom but is not automatically a valid flux.
The second fit may only widen the production signal-speed envelope.  This
separates a useful general modification from an exact-shock-tube speed fit.
Exact Balsara data are used only for a final, independent interface-flux audit.
"""

from __future__ import annotations

import argparse
import dataclasses
import subprocess
from pathlib import Path

import matplotlib
import numpy as np
from scipy.optimize import least_squares

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from diagnose_rmhd_balsara_onset import case_spec  # noqa: E402
from diagnose_rmhd_operator import (  # noqa: E402
   ORACLE,
   _run_oracle,
   exact_primitives_at,
   primitive_state_and_flux,
)
from validate_rmhd import (  # noqa: E402
   EXACT_BALSARA_ROOT,
   RUN_ROOT,
   read_last_exact_balsara_profile,
)
from validation_common import ROOT, repo_relative, write_tsv  # noqa: E402


RESULTS = RUN_ROOT / "balsara_hlld_thermodynamic_closure.tsv"
PLOT = ROOT / "plots/current/scans/rmhd_hlld_thermodynamic_closure.png"
STATE_NAMES = ("fast_l", "contact_l", "contact_r", "fast_r")


@dataclasses.dataclass(frozen=True)
class FanEvaluation:
   valid: bool
   physical: bool
   contacts_valid: bool
   speeds: np.ndarray
   total_pressure: float
   contact_residual: float
   states: np.ndarray
   recovered_pressure: np.ndarray
   recovery_success: np.ndarray
   maximum_conserved_residual: float

   @property
   def pressure_log_residual(self) -> np.ndarray:
      if (
         not self.valid
         or not np.all(self.recovery_success)
         or not np.all(self.recovered_pressure > 0.0)
         or not self.total_pressure > 0.0
      ):
         return np.full(4, 50.0)
      return np.log(self.recovered_pressure / self.total_pressure)


class LineOracle:
   """Line-buffered access to one diagnostic mode of the C++ oracle."""

   def __init__(self, mode: str) -> None:
      self.process = subprocess.Popen(
         [str(ORACLE), mode],
         stdin=subprocess.PIPE,
         stdout=subprocess.PIPE,
         stderr=subprocess.PIPE,
         text=True,
         bufsize=1,
      )
      self.row_id = 0

   def call(self, payload: str) -> np.ndarray:
      if self.process.stdin is None or self.process.stdout is None:
         raise RuntimeError("oracle pipe is unavailable")
      row_id = self.row_id
      self.row_id += 1
      self.process.stdin.write(f"{row_id} {payload}\n")
      self.process.stdin.flush()
      line = self.process.stdout.readline()
      if not line:
         stderr = (
            self.process.stderr.read()
            if self.process.stderr is not None
            else ""
         )
         raise RuntimeError(f"oracle ended before row {row_id}: {stderr}")
      values = np.fromstring(line, sep=" ")
      if values.size == 0 or int(values[0]) != row_id:
         raise RuntimeError(f"oracle row mismatch: expected {row_id}: {line}")
      return values

   def close(self) -> None:
      if self.process.stdin is not None and not self.process.stdin.closed:
         self.process.stdin.close()
      return_code = self.process.wait(timeout=5.0)
      stderr = (
         self.process.stderr.read()
         if self.process.stderr is not None
         else ""
      )
      if return_code != 0:
         raise RuntimeError(f"oracle exited with {return_code}: {stderr}")

   def __enter__(self) -> LineOracle:
      return self

   def __exit__(self, *_: object) -> None:
      self.close()


class ClosureEvaluator:
   def __init__(self) -> None:
      self.fan_oracle = LineOracle("--hlld-evaluation-table")
      self.c2p_oracle = LineOracle("--c2p-table")
      self.cache: dict[tuple[object, ...], FanEvaluation] = {}

   @staticmethod
   def _primitive_text(state: tuple[float, ...]) -> str:
      return " ".join(f"{value:.17g}" for value in state)

   def _recover_state(
      self,
      conserved: np.ndarray,
      gamma: float,
   ) -> tuple[bool, float, float]:
      row = self.c2p_oracle.call(
         f"{gamma:.17g} "
         + " ".join(f"{value:.17g}" for value in conserved)
      )
      succeeded = bool(int(row[1]))
      if not succeeded:
         return False, np.nan, float(row[6])
      # rho, eps, gas pressure, velocity, magnetic field.
      pressure = float(row[9])
      velocity = row[10:13]
      magnetic = row[13:16]
      velocity_squared = float(np.dot(velocity, velocity))
      lorentz_squared_inverse = 1.0 - velocity_squared
      comoving_magnetic_squared = (
         float(np.dot(magnetic, magnetic)) * lorentz_squared_inverse
         + float(np.dot(magnetic, velocity)) ** 2
      )
      return True, pressure + 0.5 * comoving_magnetic_squared, float(row[6])

   def evaluate(
      self,
      spec: dict[str, object],
      speed_l: float,
      speed_r: float,
      total_pressure: float,
   ) -> FanEvaluation:
      cache_key = (
         int(spec["case"]),
         float(speed_l),
         float(speed_r),
         float(total_pressure),
      )
      if cache_key in self.cache:
         return self.cache[cache_key]
      gamma = float(spec["gamma"])
      row = self.fan_oracle.call(
         f"{gamma:.17g} {self._primitive_text(spec['left'])} "
         f"{self._primitive_text(spec['right'])} {speed_l:.17g} "
         f"{speed_r:.17g} {total_pressure:.17g}"
      )
      valid = bool(int(row[1]))
      physical = bool(int(row[2]))
      contacts_valid = bool(int(row[3]))
      if not valid or not contacts_valid:
         evaluation = FanEvaluation(
            valid=valid,
            physical=False,
            contacts_valid=contacts_valid,
            speeds=np.zeros(5),
            total_pressure=total_pressure,
            contact_residual=np.nan,
            states=np.zeros((4, 8)),
            recovered_pressure=np.full(4, np.nan),
            recovery_success=np.zeros(4, dtype=bool),
            maximum_conserved_residual=np.inf,
         )
         self.cache[cache_key] = evaluation
         return evaluation
      states = row[11:43].reshape(4, 8)
      recovered = [self._recover_state(state, gamma) for state in states]
      recovery_success = np.asarray([entry[0] for entry in recovered])
      recovered_pressure = np.asarray([entry[1] for entry in recovered])
      conserved_residuals = np.asarray([entry[2] for entry in recovered])
      evaluation = FanEvaluation(
         valid=True,
         physical=physical,
         contacts_valid=True,
         speeds=row[4:9].copy(),
         total_pressure=float(row[9]),
         contact_residual=float(row[10]),
         states=states,
         recovered_pressure=recovered_pressure,
         recovery_success=recovery_success,
         maximum_conserved_residual=float(np.max(conserved_residuals)),
      )
      self.cache[cache_key] = evaluation
      return evaluation

   def close(self) -> None:
      self.fan_oracle.close()
      self.c2p_oracle.close()

   def __enter__(self) -> ClosureEvaluator:
      return self

   def __exit__(self, *_: object) -> None:
      self.close()


def standard_fans() -> dict[int, tuple[float, float, float]]:
   rows: list[str] = []
   for index in range(1, 6):
      spec = case_spec(index)
      values = (*spec["left"], *spec["right"])
      rows.append(
         f"{index - 1} {float(spec['gamma']):.17g} "
         + " ".join(f"{float(value):.17g}" for value in values)
         + "\n"
      )
   table = _run_oracle("--hlld-fan-table", rows)
   if not np.all(table[:, 1].astype(bool)):
      failed = np.flatnonzero(~table[:, 1].astype(bool)) + 1
      raise RuntimeError(f"standard HLLD fan failed for cases {failed.tolist()}")
   return {
      case: (float(table[case - 1, 2]), float(table[case - 1, 6]),
             float(table[case - 1, 7]))
      for case in range(1, 6)
   }


def fit_fan(
   evaluator: ClosureEvaluator,
   spec: dict[str, object],
   standard: tuple[float, float, float],
   variant: str,
) -> tuple[FanEvaluation, object]:
   standard_l, standard_r, standard_pressure = standard
   causal_epsilon = 1.0e-9
   if variant == "unconstrained":
      lower = np.asarray((-1.0 + causal_epsilon, causal_epsilon,
                          np.log(standard_pressure) - np.log(1.0e4)))
      upper = np.asarray((-causal_epsilon, 1.0 - causal_epsilon,
                          np.log(standard_pressure) + np.log(1.0e4)))
   elif variant == "signal_bounded":
      lower = np.asarray((-1.0 + causal_epsilon, standard_r,
                          np.log(standard_pressure) - np.log(1.0e4)))
      upper = np.asarray((standard_l, 1.0 - causal_epsilon,
                          np.log(standard_pressure) + np.log(1.0e4)))
   else:
      raise ValueError(f"unknown fit variant {variant}")
   start = np.asarray((standard_l, standard_r, np.log(standard_pressure)))
   start = np.maximum(lower + 1.0e-12, np.minimum(upper - 1.0e-12, start))

   def objective(parameters: np.ndarray) -> np.ndarray:
      evaluation = evaluator.evaluate(
         spec, float(parameters[0]), float(parameters[1]),
         float(np.exp(parameters[2]))
      )
      if not evaluation.valid or not np.all(evaluation.recovery_success):
         return np.full(5, 50.0)
      return np.concatenate((
         np.asarray((1.0e4 * evaluation.contact_residual,)),
         evaluation.pressure_log_residual,
      ))

   result = least_squares(
      objective,
      start,
      bounds=(lower, upper),
      x_scale="jac",
      max_nfev=240,
      ftol=1.0e-11,
      xtol=1.0e-11,
      gtol=1.0e-11,
   )
   evaluation = evaluator.evaluate(
      spec, float(result.x[0]), float(result.x[1]),
      float(np.exp(result.x[2]))
   )
   return evaluation, result


def sampled_fan_flux(
   evaluation: FanEvaluation,
   state_l: np.ndarray,
   state_r: np.ndarray,
   flux_l: np.ndarray,
   flux_r: np.ndarray,
) -> np.ndarray:
   speed_l, alfven_l, contact, alfven_r, speed_r = evaluation.speeds
   fast_l, contact_l, contact_r, fast_r = evaluation.states
   flux_fast_l = flux_l + speed_l * (fast_l - state_l)
   flux_contact_l = flux_fast_l + alfven_l * (contact_l - fast_l)
   flux_fast_r = flux_r + speed_r * (fast_r - state_r)
   flux_contact_r = flux_fast_r + alfven_r * (contact_r - fast_r)
   if speed_l >= 0.0:
      return flux_l
   if alfven_l >= 0.0:
      return flux_fast_l
   if contact >= 0.0:
      return flux_contact_l
   if alfven_r >= 0.0:
      return flux_contact_r
   if speed_r >= 0.0:
      return flux_fast_r
   return flux_r


def exact_interface_flux(spec: dict[str, object]) -> np.ndarray:
   profile = read_last_exact_balsara_profile(
      EXACT_BALSARA_ROOT / f"solution_b{int(spec['case'])}.dat"
   )
   primitive = exact_primitives_at(
      profile,
      spec,
      float(spec["time"]),
      np.asarray((0.0,)),
   )
   _, flux = primitive_state_and_flux(primitive, float(spec["gamma"]))
   return flux[:, 0]


def flux_metrics(
   candidate: np.ndarray,
   exact: np.ndarray,
   flux_l: np.ndarray,
   flux_r: np.ndarray,
) -> tuple[float, float, int]:
   scale = np.maximum.reduce((np.abs(exact), np.abs(flux_l), np.abs(flux_r)))
   scale = np.maximum(scale, 1.0e-12)
   normalized = np.abs(candidate - exact) / scale
   return float(np.mean(normalized)), float(np.max(normalized)), int(np.argmax(normalized))


def result_row(
   *,
   case: int,
   variant: str,
   standard: tuple[float, float, float],
   evaluation: FanEvaluation,
   optimizer: object | None,
   exact_flux: np.ndarray,
   input_state: np.ndarray,
   input_flux: np.ndarray,
) -> dict[str, object]:
   candidate_flux = sampled_fan_flux(
      evaluation,
      input_state[:, 0], input_state[:, 1],
      input_flux[:, 0], input_flux[:, 1],
   )
   flux_l1, flux_linf, flux_component = flux_metrics(
      candidate_flux, exact_flux, input_flux[:, 0], input_flux[:, 1]
   )
   pressure_residual = evaluation.pressure_log_residual
   standard_l, standard_r, _ = standard
   return {
      "case": case,
      "variant": variant,
      "valid": int(evaluation.valid),
      "physical": int(evaluation.physical),
      "signal_bounded": int(
         evaluation.speeds[0] <= standard_l + 1.0e-12
         and evaluation.speeds[4] >= standard_r - 1.0e-12
      ),
      "speed_l": evaluation.speeds[0],
      "alfven_l": evaluation.speeds[1],
      "contact": evaluation.speeds[2],
      "alfven_r": evaluation.speeds[3],
      "speed_r": evaluation.speeds[4],
      "total_pressure": evaluation.total_pressure,
      "contact_residual": evaluation.contact_residual,
      "pressure_log_rms": float(np.sqrt(np.mean(pressure_residual**2))),
      "pressure_log_max": float(np.max(np.abs(pressure_residual))),
      **{
         f"recovered_pressure_{name}": evaluation.recovered_pressure[index]
         for index, name in enumerate(STATE_NAMES)
      },
      "maximum_conserved_residual": evaluation.maximum_conserved_residual,
      "exact_flux_normalized_l1": flux_l1,
      "exact_flux_normalized_linf": flux_linf,
      "exact_flux_worst_component": flux_component,
      "optimizer_success": int(getattr(optimizer, "success", True)),
      "optimizer_evaluations": int(getattr(optimizer, "nfev", 0)),
      "optimizer_cost": float(getattr(optimizer, "cost", np.nan)),
   }


def make_plot(rows: list[dict[str, object]]) -> None:
   variants = ("standard", "unconstrained", "signal_bounded")
   labels = {
      "standard": "production HLLD",
      "unconstrained": "free outer speeds",
      "signal_bounded": "signal-bounded fit",
   }
   colors = {
      "standard": "#4c78a8",
      "unconstrained": "#f58518",
      "signal_bounded": "#54a24b",
   }
   cases = np.arange(1, 6)
   figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), constrained_layout=True)
   for variant in variants:
      selected = [row for row in rows if row["variant"] == variant]
      axes[0].semilogy(
         cases,
         [float(row["pressure_log_rms"]) for row in selected],
         marker="o",
         color=colors[variant],
         label=labels[variant],
      )
      axes[1].semilogy(
         cases,
         [float(row["exact_flux_normalized_linf"]) for row in selected],
         marker="o",
         color=colors[variant],
         label=labels[variant],
      )
   axes[0].set(title="Intermediate-state pressure closure", xlabel="Balsara case",
               ylabel="RMS log pressure mismatch")
   axes[1].set(title="Independent exact interface-flux audit", xlabel="Balsara case",
               ylabel="maximum normalized flux error")
   for axis in axes:
      axis.set_xticks(cases)
      axis.grid(alpha=0.25)
   axes[0].legend(frameon=False, fontsize=8)
   PLOT.parent.mkdir(parents=True, exist_ok=True)
   figure.savefig(PLOT, dpi=180)
   plt.close(figure)


def parse_args() -> argparse.Namespace:
   parser = argparse.ArgumentParser(description=__doc__)
   parser.add_argument("--skip-plot", action="store_true")
   return parser.parse_args()


def main() -> int:
   args = parse_args()
   if not ORACLE.is_file():
      raise FileNotFoundError(
         f"missing {ORACLE}; build pyfluid_amrex_rmhd_riemann_check"
      )
   standard_by_case = standard_fans()
   rows: list[dict[str, object]] = []
   with ClosureEvaluator() as evaluator:
      for case in range(1, 6):
         spec = case_spec(case)
         standard = standard_by_case[case]
         primitive = np.asarray((spec["left"], spec["right"]), dtype=float)
         primitive = primitive[:, (0, 2, 3, 4, 1, 5, 6, 7)].T
         input_state, input_flux = primitive_state_and_flux(
            primitive, float(spec["gamma"])
         )
         exact_flux = exact_interface_flux(spec)
         standard_evaluation = evaluator.evaluate(spec, *standard)
         rows.append(result_row(
            case=case,
            variant="standard",
            standard=standard,
            evaluation=standard_evaluation,
            optimizer=None,
            exact_flux=exact_flux,
            input_state=input_state,
            input_flux=input_flux,
         ))
         for variant in ("unconstrained", "signal_bounded"):
            evaluation, optimizer = fit_fan(
               evaluator, spec, standard, variant
            )
            rows.append(result_row(
               case=case,
               variant=variant,
               standard=standard,
               evaluation=evaluation,
               optimizer=optimizer,
               exact_flux=exact_flux,
               input_state=input_state,
               input_flux=input_flux,
            ))
            print(
               f"B{case} {variant}: speeds="
               f"({evaluation.speeds[0]:.8g}, {evaluation.speeds[4]:.8g}) "
               f"contact={evaluation.contact_residual:.3e} "
               f"p-rms={np.sqrt(np.mean(evaluation.pressure_log_residual**2)):.3e} "
               f"bounded={evaluation.speeds[0] <= standard[0] + 1.0e-12 and evaluation.speeds[4] >= standard[1] - 1.0e-12}"
            )
   rows.sort(key=lambda row: (int(row["case"]),
                              ("standard", "unconstrained", "signal_bounded").index(str(row["variant"]))))
   RESULTS.parent.mkdir(parents=True, exist_ok=True)
   write_tsv(RESULTS, rows, list(rows[0]))
   if not args.skip_plot:
      make_plot(rows)
   print(f"wrote {repo_relative(RESULTS)}")
   if not args.skip_plot:
      print(f"wrote {repo_relative(PLOT)}")
   return 0


if __name__ == "__main__":
   raise SystemExit(main())
