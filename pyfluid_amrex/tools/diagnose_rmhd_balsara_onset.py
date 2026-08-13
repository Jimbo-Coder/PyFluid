#!/usr/bin/env python3
"""Locate accepted-step onset of the two distinct difficult SRMHD symptoms.

This is a production-path diagnostic, not a reconstruction prototype.  It
keeps Balsara 4 ringing separate from Balsara 3 recovery robustness, writes
every accepted state, and compares each state with a cell-averaged conserved
reference at the same self-similar time.  Exact data are used only offline.

The accepted-state trace localizes the first surviving artifact and existing
guard/recovery event.  Optional debug-stage output then compares identical
SSPRK3 candidates; it localizes a stage boundary but does not by itself assign
the effect to reconstruction, the Riemann flux, or their interaction.
"""

from __future__ import annotations

import argparse
import dataclasses
import re
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from plotfile_reader import (  # noqa: E402
   parse_plot_bounds,
   parse_plot_header_1d,
   plotfile_step,
   plotfile_timeline,
   read_composite_state_1d,
)
from scan_rmhd_balsara_reconstruction import COMPONENT_NAMES  # noqa: E402
from validate_rmhd import (  # noqa: E402
   EXACT_BALSARA_CASES,
   EXACT_BALSARA_ROOT,
   INPUTS,
   RUN_ROOT,
   exact_rmhd_cell_averages,
   exact_rmhd_conserved,
   read_last_exact_balsara_profile,
   run_case,
)
from validation_common import ROOT, read_tsv, repo_relative, write_tsv  # noqa: E402


@dataclasses.dataclass(frozen=True)
class TraceProfile:
   name: str
   symptom: str
   case: int
   cells: int
   reconstruction: str
   guard: str
   phase: str = "accepted_timeline"
   timestep: str = "ssprk3"
   max_step: int | None = None
   riemann: str = "hll"
   candidate_riemann: str = "same"
   candidate_dmp: bool = True
   debug_stage_output: bool = False


TRACE_PROFILES = (
   TraceProfile(
      "b4_ppm4_guarded",
      "ringing",
      4,
      256,
      "ppm4",
      "pwc",
   ),
   TraceProfile(
      "b4_ppm4_unguarded",
      "ringing_control",
      4,
      256,
      "ppm4",
      "off",
   ),
   TraceProfile(
      "b4_pwc",
      "ringing_control",
      4,
      256,
      "pwc",
      "off",
   ),
   TraceProfile(
      "b4_mp5_guarded",
      "ringing_control",
      4,
      256,
      "mp5",
      "pwc",
   ),
   TraceProfile(
      "b4_ppm4_guarded_recovery_only",
      "ringing_guard_attribution",
      4,
      256,
      "ppm4",
      "pwc",
      candidate_dmp=False,
   ),
   TraceProfile(
      "b4_ppm4_guarded_hlld_full",
      "ringing_riemann_attribution",
      4,
      256,
      "ppm4",
      "pwc",
      riemann="hlld",
   ),
   TraceProfile(
      "b3_mp5_unguarded",
      "recovery",
      3,
      128,
      "mp5",
      "off",
   ),
   TraceProfile(
      "b3_mp5_guarded",
      "recovery_control",
      3,
      128,
      "mp5",
      "pwc",
   ),
   TraceProfile(
      "b3_pwc",
      "recovery_control",
      3,
      128,
      "pwc",
      "off",
   ),
   TraceProfile(
      "b3_mp5_guarded_recovery_only",
      "recovery_guard_attribution",
      3,
      128,
      "mp5",
      "pwc",
      candidate_dmp=False,
   ),
   TraceProfile(
      "b3_mp5_guarded_hlld_full",
      "recovery_riemann_attribution",
      3,
      128,
      "mp5",
      "pwc",
      riemann="hlld",
   ),
   TraceProfile(
      "b4_stage1_ppm4_unguarded",
      "ringing_stage_replay",
      4,
      256,
      "ppm4",
      "off",
      phase="stage1_replay",
      timestep="euler",
      max_step=1,
   ),
   TraceProfile(
      "b4_stage1_ppm4_guarded",
      "ringing_stage_replay",
      4,
      256,
      "ppm4",
      "pwc",
      phase="stage1_replay",
      timestep="euler",
      max_step=1,
   ),
   TraceProfile(
      "b4_stage1_ppm4_guarded_llf",
      "ringing_stage_replay",
      4,
      256,
      "ppm4",
      "pwc",
      phase="stage1_replay",
      timestep="euler",
      max_step=1,
      candidate_riemann="llf",
   ),
   TraceProfile(
      "b4_stage1_pwc",
      "ringing_stage_replay",
      4,
      256,
      "pwc",
      "off",
      phase="stage1_replay",
      timestep="euler",
      max_step=1,
   ),
   TraceProfile(
      "b3_stage1_mp5_unguarded",
      "recovery_stage_replay",
      3,
      128,
      "mp5",
      "off",
      phase="stage1_replay",
      timestep="euler",
      max_step=1,
   ),
   TraceProfile(
      "b3_stage1_mp5_guarded",
      "recovery_stage_replay",
      3,
      128,
      "mp5",
      "pwc",
      phase="stage1_replay",
      timestep="euler",
      max_step=1,
   ),
   TraceProfile(
      "b4_ssprk3_stage_trace_ppm4_guarded",
      "ringing_stage_trace",
      4,
      256,
      "ppm4",
      "pwc",
      phase="ssprk3_stage_trace",
      max_step=1,
      debug_stage_output=True,
   ),
   TraceProfile(
      "b4_ssprk3_stage_trace_ppm4_unguarded",
      "ringing_stage_trace_control",
      4,
      256,
      "ppm4",
      "off",
      phase="ssprk3_stage_trace",
      max_step=4,
      debug_stage_output=True,
   ),
   TraceProfile(
      "b4_ssprk3_stage_trace_ppm4_guarded_llf",
      "ringing_stage_trace_control",
      4,
      256,
      "ppm4",
      "pwc",
      phase="ssprk3_stage_trace",
      max_step=1,
      candidate_riemann="llf",
      debug_stage_output=True,
   ),
   TraceProfile(
      "b4_ssprk3_stage_trace_pwc",
      "ringing_stage_trace_control",
      4,
      256,
      "pwc",
      "off",
      phase="ssprk3_stage_trace",
      max_step=1,
      debug_stage_output=True,
   ),
   TraceProfile(
      "b4_ssprk3_stage_trace_mp5_guarded",
      "ringing_stage_trace_control",
      4,
      256,
      "mp5",
      "pwc",
      phase="ssprk3_stage_trace",
      max_step=1,
      debug_stage_output=True,
   ),
   TraceProfile(
      "b4_ssprk3_stage_trace_ppm4_guarded_recovery_only",
      "ringing_guard_attribution",
      4,
      256,
      "ppm4",
      "pwc",
      phase="ssprk3_stage_trace",
      max_step=1,
      candidate_dmp=False,
      debug_stage_output=True,
   ),
   TraceProfile(
      "b4_ssprk3_stage_trace_ppm4_unguarded_hlld",
      "ringing_riemann_attribution",
      4,
      256,
      "ppm4",
      "off",
      phase="ssprk3_stage_trace",
      max_step=1,
      riemann="hlld",
      debug_stage_output=True,
   ),
   TraceProfile(
      "b4_ssprk3_stage_trace_ppm4_guarded_hlld",
      "ringing_riemann_attribution",
      4,
      256,
      "ppm4",
      "pwc",
      phase="ssprk3_stage_trace",
      max_step=1,
      riemann="hlld",
      debug_stage_output=True,
   ),
   TraceProfile(
      "b3_ssprk3_stage_trace_mp5_unguarded",
      "recovery_stage_trace",
      3,
      128,
      "mp5",
      "off",
      phase="ssprk3_stage_trace",
      max_step=2,
      debug_stage_output=True,
   ),
   TraceProfile(
      "b3_ssprk3_stage_trace_mp5_guarded",
      "recovery_stage_trace_control",
      3,
      128,
      "mp5",
      "pwc",
      phase="ssprk3_stage_trace",
      max_step=2,
      debug_stage_output=True,
   ),
   TraceProfile(
      "b3_ssprk3_stage_trace_pwc",
      "recovery_stage_trace_control",
      3,
      128,
      "pwc",
      "off",
      phase="ssprk3_stage_trace",
      max_step=2,
      debug_stage_output=True,
   ),
   TraceProfile(
      "b3_ssprk3_stage_trace_mp5_guarded_recovery_only",
      "recovery_guard_attribution",
      3,
      128,
      "mp5",
      "pwc",
      phase="ssprk3_stage_trace",
      max_step=2,
      candidate_dmp=False,
      debug_stage_output=True,
   ),
   TraceProfile(
      "b3_ssprk3_stage_trace_mp5_unguarded_hlld",
      "recovery_riemann_attribution",
      3,
      128,
      "mp5",
      "off",
      phase="ssprk3_stage_trace",
      max_step=2,
      riemann="hlld",
      debug_stage_output=True,
   ),
   TraceProfile(
      "b3_ssprk3_stage_trace_mp5_guarded_hlld",
      "recovery_riemann_attribution",
      3,
      128,
      "mp5",
      "pwc",
      phase="ssprk3_stage_trace",
      max_step=2,
      riemann="hlld",
      debug_stage_output=True,
   ),
)
PROFILE_BY_NAME = {profile.name: profile for profile in TRACE_PROFILES}
RESULTS = RUN_ROOT / "balsara_onset_trace.tsv"
SUMMARY = RUN_ROOT / "balsara_onset_summary.tsv"
PLOT = ROOT / "plots/current/scans/rmhd_balsara_onset.png"
STAGE_PLOT = ROOT / "plots/current/scans/rmhd_balsara_stage_onset.png"
PROFILE_PLOT = ROOT / "plots/current/scans/rmhd_balsara_profiles.png"
STAGE_PROFILE_PLOT = (
   ROOT / "plots/current/scans/rmhd_balsara_stage_profiles.png"
)
HYBRID_TIMELINE_PLOT = (
   ROOT / "plots/current/scans/rmhd_balsara_hybrid_timelines.png"
)
HYBRID_PROFILE_EVOLUTION_PLOT = (
   ROOT / "plots/current/scans/rmhd_balsara_hybrid_profile_evolution.png"
)

RECOVERY_EVENT = re.compile(
   r"WARNING: relativistic recovery repair at step=(?P<step>\d+) "
   r"time=(?P<time>\S+) "
   r"c2p_fallbacks=(?P<c2p>\d+) "
   r"atmosphere_resets=(?P<reset>\d+) "
   r"rmhd_pwl_corrections=(?P<pwl>\d+) "
   r"rmhd_pwc_corrections=(?P<pwc>\d+)"
)
STAGE_NAMES = (
   "ssprk3_stage1_candidate",
   "ssprk3_stage2_candidate",
   "ssprk3_stage3_candidate",
   "ssprk3_accepted",
)
STAGE_ORDER = {name: index for index, name in enumerate(STAGE_NAMES)}


def case_spec(case: int) -> dict[str, object]:
   return next(
      spec for spec in EXACT_BALSARA_CASES if int(spec["case"]) == case
   )


def initial_exact_profile(spec: dict[str, object]) -> np.ndarray:
   """Return a repeated-coordinate exact-table state for the t=0 jump.

   The exact Balsara table stores total pressure, whereas the input catalog
   stores gas pressure.  Preserve that contract here just as
   :func:`exact_rmhd_conserved` expects.
   """

   rows: list[tuple[float, ...]] = []
   for coordinate, state in (
      (-1.0, spec["left"]),
      (0.0, spec["left"]),
      (0.0, spec["right"]),
      (1.0, spec["right"]),
   ):
      rho, pressure, v1, v2, v3, b1, b2, b3 = state
      velocity_squared = v1 * v1 + v2 * v2 + v3 * v3
      inverse_lorentz_squared = 1.0 - velocity_squared
      magnetic_squared = b1 * b1 + b2 * b2 + b3 * b3
      magnetic_velocity = b1 * v1 + b2 * v2 + b3 * v3
      comoving_magnetic_squared = (
         magnetic_squared * inverse_lorentz_squared
         + magnetic_velocity * magnetic_velocity
      )
      total_pressure = pressure + 0.5 * comoving_magnetic_squared
      rows.append((
         coordinate,
         rho,
         total_pressure,
         v1,
         v2,
         v3,
         b2,
         b3,
      ))
   return np.asarray(rows, dtype=float)


def exact_average_at_time(
   exact_profile: np.ndarray,
   spec: dict[str, object],
   time: float,
   cell_edges: np.ndarray,
) -> np.ndarray:
   if time <= 16.0 * np.finfo(float).eps:
      timed_profile = initial_exact_profile(spec)
   else:
      timed_profile = exact_profile.copy()
      timed_profile[:, 0] *= time / float(spec["time"])
   return exact_rmhd_cell_averages(
      timed_profile,
      normal_magnetic=float(spec["left"][5]),
      gamma=float(spec["gamma"]),
      cell_edges=cell_edges,
   )


def spatial_profile(
   path: Path,
   case: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
   """Read one conserved profile and its time-matched FV exact reference."""

   spec = case_spec(case)
   names, output_time = parse_plot_header_1d(path)
   state = read_composite_state_1d(path, names)
   numerical = np.vstack([state[names.index(name)] for name in COMPONENT_NAMES])
   prob_lo, prob_hi = parse_plot_bounds(path)
   cell_edges = np.linspace(prob_lo[0], prob_hi[0], numerical.shape[1] + 1)
   cell_centers = 0.5 * (cell_edges[:-1] + cell_edges[1:])
   exact_profile = read_last_exact_balsara_profile(
      EXACT_BALSARA_ROOT / f"solution_b{case}.dat"
   )
   exact_average = exact_average_at_time(
      exact_profile, spec, output_time, cell_edges
   )
   return cell_centers, numerical, exact_average, output_time


def latest_state_path(run_name: str) -> Path | None:
   samples = plotfile_timeline(RUN_ROOT / run_name, require=False)
   return samples[-1].path if samples else None


def closest_state_path(run_name: str, target_time: float) -> Path | None:
   samples = plotfile_timeline(RUN_ROOT / run_name, require=False)
   if not samples:
      return None
   return min(samples, key=lambda sample: abs(sample.time - target_time)).path


def debug_stage_path(
   run_name: str,
   step: int,
   stage: str,
) -> Path | None:
   for sample_step, sample_stage, path in stage_output_paths(
      RUN_ROOT / run_name
   ):
      if sample_step == step and sample_stage == stage:
         return path
   return None


def plateau_zigzag_metrics(
   numerical: np.ndarray,
   exact_average: np.ndarray,
   scales: np.ndarray,
) -> tuple[float, int, str]:
   """Measure slope reversals only where the exact FV state is a plateau."""

   numerical_slopes = np.diff(numerical, axis=1)
   exact_slopes = np.diff(exact_average, axis=1)
   maximum = 0.0
   count = 0
   maximum_component = ""
   for component, name in enumerate(COMPONENT_NAMES):
      left = numerical_slopes[component, :-1]
      right = numerical_slopes[component, 1:]
      exact_left = exact_slopes[component, :-1]
      exact_right = exact_slopes[component, 1:]
      plateau = np.maximum(np.abs(exact_left), np.abs(exact_right)) <= (
         1.0e-9 * scales[component]
      )
      amplitude = np.minimum(np.abs(left), np.abs(right)) / scales[component]
      reversal = (left * right < 0.0) & plateau & (amplitude > 1.0e-12)
      count += int(np.count_nonzero(reversal))
      if np.any(reversal):
         component_maximum = float(np.max(amplitude[reversal]))
         if component_maximum > maximum:
            maximum = component_maximum
            maximum_component = name
   return maximum, count, maximum_component


def accepted_state_metrics(
   numerical: np.ndarray,
   exact_average: np.ndarray,
   exact_point_state: np.ndarray,
) -> dict[str, object]:
   scales = np.maximum(
      np.ptp(exact_point_state, axis=1),
      np.max(np.abs(exact_point_state), axis=1),
   )
   scales = np.maximum(scales, 1.0e-14)
   component_l1 = np.mean(np.abs(numerical - exact_average), axis=1) / scales

   exact_min = np.min(exact_point_state, axis=1)
   exact_max = np.max(exact_point_state, axis=1)
   overshoot = np.maximum(
      np.maximum(exact_min[:, np.newaxis] - numerical, 0.0),
      np.maximum(numerical - exact_max[:, np.newaxis], 0.0),
   ) / scales[:, np.newaxis]
   maximum_overshoot_index = np.unravel_index(
      int(np.argmax(overshoot)), overshoot.shape
   )

   numerical_tv = np.sum(np.abs(np.diff(numerical, axis=1)), axis=1)
   exact_tv = np.sum(np.abs(np.diff(exact_average, axis=1)), axis=1)
   active = exact_tv > 1.0e-12 * scales
   tv_ratio = np.ones_like(exact_tv)
   tv_ratio[active] = numerical_tv[active] / exact_tv[active]
   maximum_tv_component = int(np.argmax(tv_ratio))

   zigzag, zigzag_count, zigzag_component = plateau_zigzag_metrics(
      numerical, exact_average, scales
   )
   return {
      "maximum_normalized_component_l1": float(np.max(component_l1)),
      "maximum_l1_component": COMPONENT_NAMES[int(np.argmax(component_l1))],
      "maximum_normalized_global_overshoot": float(np.max(overshoot)),
      "maximum_overshoot_component": COMPONENT_NAMES[
         maximum_overshoot_index[0]
      ],
      "maximum_component_tv_ratio": float(tv_ratio[maximum_tv_component]),
      "maximum_tv_component": COMPONENT_NAMES[maximum_tv_component],
      "maximum_plateau_zigzag": zigzag,
      "plateau_zigzag_count": zigzag_count,
      "maximum_zigzag_component": zigzag_component,
   }


def recovery_events(run_dir: Path) -> dict[int, dict[str, int]]:
   events: dict[int, dict[str, int]] = {}
   for match in RECOVERY_EVENT.finditer((run_dir / "run.log").read_text()):
      step = int(match.group("step"))
      values = events.setdefault(
         step,
         {"c2p_fallbacks": 0, "atmosphere_resets": 0,
          "rmhd_pwl_corrections": 0, "rmhd_pwc_corrections": 0},
      )
      values["c2p_fallbacks"] += int(match.group("c2p"))
      values["atmosphere_resets"] += int(match.group("reset"))
      values["rmhd_pwl_corrections"] += int(match.group("pwl"))
      values["rmhd_pwc_corrections"] += int(match.group("pwc"))
   return events


def stage_output_paths(run_dir: Path) -> list[tuple[int, str, Path]]:
   debug_dir = run_dir / "debugStage"
   paths: list[tuple[int, str, Path]] = []
   for path in debug_dir.glob("*GF*"):
      stage_name = next(
         (name for name in STAGE_NAMES if path.name.startswith(f"{name}GF")),
         None,
      )
      if stage_name is not None:
         paths.append((plotfile_step(path), stage_name, path))
   return sorted(paths, key=lambda item: (item[0], STAGE_ORDER[item[1]]))


def plotfile_metric_row(
   path: Path,
   *,
   profile: TraceProfile,
   spec: dict[str, object],
   exact_profile: np.ndarray,
   exact_point_state: np.ndarray,
   stage: str,
) -> dict[str, object]:
   names, output_time = parse_plot_header_1d(path)
   state = read_composite_state_1d(path, names)
   numerical = np.vstack([state[names.index(name)] for name in COMPONENT_NAMES])
   prob_lo, prob_hi = parse_plot_bounds(path)
   cell_edges = np.linspace(prob_lo[0], prob_hi[0], numerical.shape[1] + 1)
   exact_average = exact_average_at_time(
      exact_profile, spec, output_time, cell_edges
   )
   return {
      "profile": profile.name,
      "phase": profile.phase,
      "symptom": profile.symptom,
      "case": profile.case,
      "cells": profile.cells,
      "reconstruction": profile.reconstruction,
      "guard": profile.guard,
      "timestep": profile.timestep,
      "riemann": profile.riemann,
      "candidate_riemann": profile.candidate_riemann,
      "candidate_dmp": int(profile.candidate_dmp),
      "step": plotfile_step(path),
      "stage": stage,
      "time": output_time,
      **accepted_state_metrics(numerical, exact_average, exact_point_state),
   }


def trace_run(profile: TraceProfile, args: argparse.Namespace) -> list[dict[str, object]]:
   spec = case_spec(profile.case)
   overrides = {
      "geometry.prob_lo": "-1.0",
      "geometry.prob_hi": "1.0",
      "amr.n_cell": str(profile.cells),
      "amr.max_grid_size": str(profile.cells),
      "pyfluid.gamma": repr(spec["gamma"]),
      "pyfluid.final_time": repr(spec["time"]),
      "pyfluid.max_step": str(
         profile.max_step if profile.max_step is not None else args.max_step
      ),
      "pyfluid.ng": "3",
      "pyfluid.cfl": "0.9",
      "pyfluid.recon": profile.reconstruction,
      "pyfluid.riemann": profile.riemann,
      "pyfluid.timestep": profile.timestep,
      "pyfluid.rmhd_candidate_guard": profile.guard,
      "pyfluid.rmhd_candidate_riemann": profile.candidate_riemann,
      "pyfluid.rmhd_candidate_dmp": "1" if profile.candidate_dmp else "0",
      "pyfluid.rmhd_candidate_dmp_factor": repr(args.dmp_factor),
      "pyfluid.physical_check_int": "1",
      "pyfluid.debug_stage_output": "1" if profile.debug_stage_output else "0",
      "pyfluid.scalar_int": "1",
      "pyfluid.plot_int": "1",
      "pyfluid.profile": "off",
   }
   for side, state in (("left", spec["left"]), ("right", spec["right"])):
      for name, value in zip(
         ("rho", "pressure", "v1", "v2", "v3", "b1", "b2", "b3"),
         state,
         strict=True,
      ):
         overrides[f"pyfluid.{name}_{side}"] = repr(value)

   run_name = f"onset_{profile.name}_n{profile.cells}"
   run_dir, _ = run_case(
      run_name,
      INPUTS / "rmhd_balsara1_1d.in",
      overrides,
      args,
   )
   events = recovery_events(run_dir)
   exact_profile = read_last_exact_balsara_profile(
      EXACT_BALSARA_ROOT / f"solution_b{profile.case}.dat"
   )
   exact_point_state = exact_rmhd_conserved(
      exact_profile,
      normal_magnetic=float(spec["left"][5]),
      gamma=float(spec["gamma"]),
   )

   rows: list[dict[str, object]] = []
   cumulative = {
      "c2p_fallbacks": 0,
      "atmosphere_resets": 0,
      "rmhd_pwl_corrections": 0,
      "rmhd_pwc_corrections": 0,
   }
   if profile.debug_stage_output:
      samples = stage_output_paths(run_dir)
      if not samples:
         raise RuntimeError(f"{run_name} did not write debug SSPRK3 stages")
   else:
      samples = [
         (sample.step, "accepted", sample.path)
         for sample in plotfile_timeline(run_dir, require=True)
      ]
   for sample_step, stage, path in samples:
      row = plotfile_metric_row(
         path,
         profile=profile,
         spec=spec,
         exact_profile=exact_profile,
         exact_point_state=exact_point_state,
         stage=stage,
      )
      report_events = stage in {"accepted", "ssprk3_accepted"}
      step_events = events.get(sample_step, {}) if report_events else {}
      for name in cumulative:
         cumulative[name] += int(step_events.get(name, 0))
      rows.append({
         **row,
         **{name: int(step_events.get(name, 0)) for name in cumulative},
         **{f"cumulative_{name}": value for name, value in cumulative.items()},
         "run_dir": repo_relative(run_dir),
      })
   return rows


def first_step(
   rows: list[dict[str, object]],
   key: str,
   threshold: float = 0.0,
) -> int | str:
   for row in rows:
      if float(row[key]) > threshold:
         return int(row["step"])
   return ""


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
   summaries: list[dict[str, object]] = []
   for profile_name in dict.fromkeys(str(row["profile"]) for row in rows):
      selected = [row for row in rows if row["profile"] == profile_name]
      final = selected[-1]
      summaries.append({
         "profile": profile_name,
         "phase": final["phase"],
         "symptom": final["symptom"],
         "case": final["case"],
         "cells": final["cells"],
         "reconstruction": final["reconstruction"],
         "guard": final["guard"],
         "timestep": final["timestep"],
         "riemann": final["riemann"],
         "candidate_riemann": final["candidate_riemann"],
         "candidate_dmp": final["candidate_dmp"],
         "final_stage": final["stage"],
         "steps": final["step"],
         "final_time": final["time"],
         "first_global_overshoot_step": first_step(
            selected, "maximum_normalized_global_overshoot", 1.0e-10
         ),
         "first_plateau_zigzag_step": first_step(
            selected, "maximum_plateau_zigzag", 1.0e-8
         ),
         "first_pwc_correction_step": first_step(
            selected, "rmhd_pwc_corrections"
         ),
         "first_c2p_fallback_step": first_step(selected, "c2p_fallbacks"),
         "first_atmosphere_reset_step": first_step(
            selected, "atmosphere_resets"
         ),
         "final_maximum_l1": final["maximum_normalized_component_l1"],
         "final_maximum_overshoot": final[
            "maximum_normalized_global_overshoot"
         ],
         "final_maximum_tv_ratio": final["maximum_component_tv_ratio"],
         "final_maximum_plateau_zigzag": final["maximum_plateau_zigzag"],
         "total_c2p_fallbacks": final["cumulative_c2p_fallbacks"],
         "total_atmosphere_resets": final["cumulative_atmosphere_resets"],
         "total_pwc_corrections": final["cumulative_rmhd_pwc_corrections"],
         "run_dir": final["run_dir"],
      })
   return summaries


def plot_trace(rows: list[dict[str, object]]) -> None:
   figure, axes = plt.subplots(
      2, 2, figsize=(11.0, 7.2), constrained_layout=True
   )
   metrics = (
      ("maximum_normalized_component_l1", "maximum normalized L1"),
      ("maximum_normalized_global_overshoot", "global overshoot"),
      ("maximum_component_tv_ratio", "maximum TV ratio"),
      ("maximum_plateau_zigzag", "plateau zig-zag amplitude"),
   )
   for profile_name in dict.fromkeys(str(row["profile"]) for row in rows):
      selected = [row for row in rows if row["profile"] == profile_name]
      if selected[0]["phase"] != "accepted_timeline":
         continue
      times = [float(row["time"]) for row in selected]
      for axis, (key, label) in zip(axes.flat, metrics, strict=True):
         values = np.asarray([float(row[key]) for row in selected])
         if key != "maximum_component_tv_ratio":
            values = np.maximum(values, 1.0e-16)
            axis.set_yscale("log")
         axis.plot(times, values, label=profile_name)
         axis.set_xlabel("accepted-state time")
         axis.set_ylabel(label)
         axis.grid(alpha=0.22)
   handles, labels = axes[0, 0].get_legend_handles_labels()
   if handles:
      axes[0, 0].legend(handles, labels, frameon=False, fontsize=8)
   figure.suptitle(
      "SRMHD symptom onset — exact data used only for accepted-state scoring"
   )
   PLOT.parent.mkdir(parents=True, exist_ok=True)
   figure.savefig(PLOT, dpi=180)
   plt.close(figure)


def plot_stage_trace(rows: list[dict[str, object]]) -> None:
   stage_rows = [
      row for row in rows if row["phase"] == "ssprk3_stage_trace"
   ]
   if not stage_rows:
      return
   figure, axes = plt.subplots(
      2, 2, figsize=(11.5, 7.6), constrained_layout=True
   )
   for row_index, case in enumerate((4, 3)):
      case_rows = [row for row in stage_rows if int(row["case"]) == case]
      if not case_rows:
         for axis in axes[row_index]:
            axis.set_visible(False)
         continue
      target_step = max(int(row["step"]) for row in case_rows)
      profile_names = list(dict.fromkeys(str(row["profile"]) for row in case_rows))
      for profile_name in profile_names:
         selected = [
            row for row in case_rows
            if row["profile"] == profile_name
            and int(row["step"]) == target_step
         ]
         selected.sort(key=lambda row: STAGE_ORDER[str(row["stage"])])
         label = profile_name.replace(f"b{case}_ssprk3_stage_trace_", "")
         stages = [STAGE_ORDER[str(row["stage"])] for row in selected]
         overshoot = np.maximum(
            [float(row["maximum_normalized_global_overshoot"]) for row in selected],
            1.0e-16,
         )
         tv_ratio = [float(row["maximum_component_tv_ratio"]) for row in selected]
         axes[row_index, 0].plot(stages, overshoot, marker="o", label=label)
         axes[row_index, 1].plot(stages, tv_ratio, marker="o", label=label)
      axes[row_index, 0].set_yscale("log")
      axes[row_index, 0].set_ylabel(f"Balsara {case} global overshoot")
      axes[row_index, 1].set_ylabel(f"Balsara {case} maximum TV ratio")
      for axis in axes[row_index]:
         axis.set_xticks(range(len(STAGE_NAMES)))
         axis.set_xticklabels(("stage 1", "stage 2", "stage 3", "accepted"))
         axis.grid(alpha=0.22)
         axis.set_title(f"accepted step {target_step}")
      axes[row_index, 1].legend(frameon=False, fontsize=7)
   figure.suptitle("SRMHD SSPRK3 causal stage trace at CFL 0.9")
   STAGE_PLOT.parent.mkdir(parents=True, exist_ok=True)
   figure.savefig(STAGE_PLOT, dpi=180)
   plt.close(figure)


def plot_hybrid_timelines(rows: list[dict[str, object]]) -> bool:
   """Compare exact/A/B/A-B metrics across the complete accepted evolution."""

   case_profiles = (
      (
         4,
         (
            ("b4_ppm4_unguarded", "PPM4 raw", "tab:red"),
            ("b4_pwc", "global PWC", "tab:green"),
            ("b4_ppm4_guarded", "PPM4 + PWC", "tab:blue"),
         ),
      ),
      (
         3,
         (
            ("b3_mp5_unguarded", "MP5 raw", "tab:red"),
            ("b3_pwc", "global PWC", "tab:green"),
            ("b3_mp5_guarded", "MP5 + PWC", "tab:blue"),
         ),
      ),
   )
   available = {str(row["profile"]) for row in rows}
   required = {
      profile_name
      for _, profiles in case_profiles
      for profile_name, _, _ in profiles
   }
   if not required.issubset(available):
      return False

   metrics = (
      (
         "maximum_normalized_component_l1",
         "maximum normalized L1 error vs exact",
         None,
         True,
      ),
      (
         "maximum_normalized_global_overshoot",
         "global overshoot beyond exact range",
         0.0,
         True,
      ),
      (
         "maximum_component_tv_ratio",
         "maximum TV ratio (exact = 1)",
         1.0,
         False,
      ),
   )
   figure, axes = plt.subplots(
      2, 3, figsize=(12.8, 7.0), constrained_layout=True
   )
   for row_index, (case, profiles) in enumerate(case_profiles):
      for column, (metric, ylabel, exact_value, logarithmic) in enumerate(metrics):
         axis = axes[row_index, column]
         logarithmic_floor = 1.0e-8
         if exact_value is not None:
            baseline = logarithmic_floor if logarithmic else exact_value
            axis.axhline(
               baseline,
               color="black",
               linewidth=1.2,
               linestyle="--",
               label=f"exact reference ({exact_value:g})",
            )
         for profile_index, (profile_name, label, color) in enumerate(profiles):
            selected = [
               row for row in rows
               if row["profile"] == profile_name
               and row["phase"] == "accepted_timeline"
            ]
            selected.sort(key=lambda row: float(row["time"]))
            times = np.asarray([float(row["time"]) for row in selected])
            values = np.asarray([float(row[metric]) for row in selected])
            plotted_values = (
               np.maximum(values, logarithmic_floor)
               if logarithmic and exact_value is not None
               else values
            )
            axis.plot(
               times,
               plotted_values,
               color=color,
               linewidth=1.35,
               marker="o",
               markersize=2.1,
               markevery=max(1, len(times) // 10),
               label=label,
            )
            axis.annotate(
               f"{values[-1]:.3g}",
               (times[-1], plotted_values[-1]),
               xytext=(4, (-9, 0, 9)[profile_index]),
               textcoords="offset points",
               color=color,
               fontsize=7,
            )
         if logarithmic:
            axis.set_yscale("log")
            if exact_value is not None:
               axis.set_ylim(bottom=0.8 * logarithmic_floor)
         axis.set_xlabel("accepted-state time")
         axis.set_ylabel(ylabel)
         axis.set_title(f"Balsara {case}")
         axis.grid(alpha=0.2, which="both")
         axis.margins(x=0.08)
      axes[row_index, 0].legend(frameon=False, fontsize=7)
   figure.suptitle(
      "SRMHD exact/A/B/A-B comparison across the full accepted evolution"
   )
   HYBRID_TIMELINE_PLOT.parent.mkdir(parents=True, exist_ok=True)
   figure.savefig(HYBRID_TIMELINE_PLOT, dpi=190)
   plt.close(figure)
   return True


def plot_final_profiles() -> bool:
   """Plot human-readable final profiles without changing FV scoring."""

   cases = (
      (
         4,
         "Balsara 4",
         (
            ("onset_b4_ppm4_unguarded_n256", "PPM4 raw", "tab:red"),
            ("onset_b4_pwc_n256", "global PWC", "tab:green"),
            ("onset_b4_ppm4_guarded_n256", "PPM4 + PWC", "tab:blue"),
         ),
      ),
      (
         3,
         "Balsara 3",
         (
            ("onset_b3_mp5_unguarded_n128", "MP5 raw", "tab:red"),
            ("onset_b3_pwc_n128", "global PWC", "tab:green"),
            ("onset_b3_mp5_guarded_n128", "MP5 + PWC", "tab:blue"),
         ),
      ),
   )
   component_specs = (
      ("dens", r"$D$"),
      ("mom1", r"$S_1$"),
      ("mom2", r"$S_2$"),
      ("b2", r"$B_2$"),
   )
   paths = {
      run_name: latest_state_path(run_name)
      for _, _, runs in cases
      for run_name, _, _ in runs
   }
   if any(path is None for path in paths.values()):
      return False

   figure, axes = plt.subplots(
      2, len(component_specs), figsize=(13.5, 6.4), constrained_layout=True
   )
   for row_index, (case, case_label, runs) in enumerate(cases):
      exact_drawn = False
      output_time = 0.0
      for run_name, label, color in runs:
         path = paths[run_name]
         assert path is not None
         x, numerical, exact_average, output_time = spatial_profile(path, case)
         for column, (component, component_label) in enumerate(component_specs):
            axis = axes[row_index, column]
            component_index = COMPONENT_NAMES.index(component)
            if not exact_drawn:
               axis.plot(
                  x,
                  exact_average[component_index],
                  color="black",
                  linewidth=1.35,
                  drawstyle="steps-mid",
                  label="exact FV average",
                  zorder=4,
               )
            axis.plot(
               x,
               numerical[component_index],
               color=color,
               linewidth=0.8,
               marker=".",
               markersize=2.0,
               label=label,
               alpha=0.9,
            )
            axis.set_xlim(-0.55, 0.55)
            axis.set_xlabel("x")
            axis.set_ylabel(component_label)
            axis.grid(alpha=0.2)
         exact_drawn = True
      axes[row_index, 0].set_title(f"{case_label}, t={output_time:.3g}")
      axes[row_index, 0].legend(frameon=False, fontsize=7)
   figure.suptitle(
      "SRMHD final conserved profiles — numerical cell averages vs exact FV averages"
   )
   PROFILE_PLOT.parent.mkdir(parents=True, exist_ok=True)
   figure.savefig(PROFILE_PLOT, dpi=190)
   plt.close(figure)
   return True


def plot_hybrid_profile_evolution() -> bool:
   """Compare density and symptom-bearing A/B profiles through full evolution."""

   target_times = (0.07, 0.20, 0.40)
   case_specs = (
      (
         4,
         "Balsara 4",
         (
            ("onset_b4_ppm4_unguarded_n256", "A: PPM4 raw", "tab:red"),
            ("onset_b4_pwc_n256", "B: global PWC", "tab:green"),
            (
               "onset_b4_ppm4_guarded_n256",
               "A/B: PPM4 + PWC guard",
               "tab:blue",
            ),
         ),
      ),
      (
         3,
         "Balsara 3",
         (
            ("onset_b3_mp5_unguarded_n128", "A: MP5 raw", "tab:red"),
            ("onset_b3_pwc_n128", "B: global PWC", "tab:green"),
            (
               "onset_b3_mp5_guarded_n128",
               "A/B: MP5 + PWC guard",
               "tab:blue",
            ),
         ),
      ),
   )
   profile_rows = (
      (4, "Balsara 4", "dens", r"$D$"),
      (4, "Balsara 4", "mom2", r"$S_2$"),
      (3, "Balsara 3", "dens", r"$D$"),
      (3, "Balsara 3", "mom1", r"$S_1$"),
   )
   runs_by_case = {case: runs for case, _, runs in case_specs}
   paths = {
      (run_name, target_time): closest_state_path(run_name, target_time)
      for _, _, runs in case_specs
      for run_name, _, _ in runs
      for target_time in target_times
   }
   if any(path is None for path in paths.values()):
      return False

   figure, axes = plt.subplots(
      len(profile_rows),
      len(target_times),
      figsize=(13.4, 12.4),
      constrained_layout=True,
   )
   for row_index, (case, case_label, component, component_label) in enumerate(
      profile_rows
   ):
      runs = runs_by_case[case]
      component_index = COMPONENT_NAMES.index(component)
      for column, target_time in enumerate(target_times):
         axis = axes[row_index, column]
         profiles = []
         for run_name, label, color in runs:
            path = paths[(run_name, target_time)]
            assert path is not None
            x, numerical, exact_average, output_time = spatial_profile(path, case)
            profiles.append(
               (x, numerical, exact_average, output_time, label, color)
            )

         reference = profiles[-1]
         reference_x, _, reference_exact, reference_time, _, _ = reference
         axis.plot(
            reference_x / reference_time,
            reference_exact[component_index],
            color="black",
            linewidth=1.45,
            drawstyle="steps-mid",
            label="exact FV average",
            zorder=4,
         )
         for x, numerical, _, output_time, label, color in profiles:
            axis.plot(
               x / output_time,
               numerical[component_index],
               color=color,
               linewidth=0.95,
               marker=".",
               markersize=2.1,
               label=label,
               alpha=0.9,
            )
         output_times = [profile[3] for profile in profiles]
         time_spread = max(output_times) - min(output_times)
         axis.set_title(
            f"{case_label}, t={reference_time:.4g}"
            f" (spread {time_spread:.1e})"
         )
         axis.set_xlim(-1.27, 1.27)
         axis.set_xlabel(r"self-similar coordinate $x/t$")
         axis.set_ylabel(f"{case_label}\n{component_label}")
         axis.grid(alpha=0.2)
      if component == "dens":
         axes[row_index, 0].legend(frameon=False, fontsize=7)
   figure.suptitle(
      "SRMHD density and symptom-bearing profiles — exact/A/B/A-B"
   )
   HYBRID_PROFILE_EVOLUTION_PLOT.parent.mkdir(parents=True, exist_ok=True)
   figure.savefig(HYBRID_PROFILE_EVOLUTION_PLOT, dpi=190)
   plt.close(figure)
   return True


def plot_stage_profiles() -> bool:
   """Show the spatial candidates that establish the causal stage boundary."""

   b4_runs = (
      (
         "onset_b4_ssprk3_stage_trace_ppm4_unguarded_n256",
         "PPM4 raw",
         "tab:red",
      ),
      (
         "onset_b4_ssprk3_stage_trace_ppm4_guarded_n256",
         "PPM4 + PWC",
         "tab:blue",
      ),
      (
         "onset_b4_ssprk3_stage_trace_pwc_n256",
         "global PWC",
         "tab:green",
      ),
   )
   b4_stages = (
      "ssprk3_stage1_candidate",
      "ssprk3_stage2_candidate",
      "ssprk3_stage3_candidate",
   )
   b3_runs = (
      (
         "onset_b3_ssprk3_stage_trace_mp5_unguarded_n128",
         "MP5 raw",
         "tab:red",
      ),
      (
         "onset_b3_ssprk3_stage_trace_pwc_n128",
         "global PWC",
         "tab:green",
      ),
      (
         "onset_b3_ssprk3_stage_trace_mp5_guarded_n128",
         "MP5 + PWC",
         "tab:blue",
      ),
   )
   paths: dict[tuple[str, int, str], Path | None] = {}
   for run_name, _, _ in b4_runs:
      for stage in b4_stages:
         paths[(run_name, 1, stage)] = debug_stage_path(run_name, 1, stage)
   for run_name, _, _ in b3_runs:
      paths[(run_name, 2, "ssprk3_stage1_candidate")] = debug_stage_path(
         run_name, 2, "ssprk3_stage1_candidate"
      )
   if any(path is None for path in paths.values()):
      return False

   figure, axes = plt.subplots(2, 3, figsize=(12.2, 7.0), constrained_layout=True)
   mom2_index = COMPONENT_NAMES.index("mom2")
   for column, stage in enumerate(b4_stages):
      exact_drawn = False
      stage_time = 0.0
      for run_name, label, color in b4_runs:
         path = paths[(run_name, 1, stage)]
         assert path is not None
         x, numerical, exact_average, stage_time = spatial_profile(path, 4)
         if not exact_drawn:
            axes[0, column].plot(
               x,
               exact_average[mom2_index],
               color="black",
               linewidth=1.35,
               drawstyle="steps-mid",
               label="exact FV average",
               zorder=4,
            )
         axes[0, column].plot(
            x,
            numerical[mom2_index],
            color=color,
            linewidth=0.85,
            marker="o",
            markersize=2.5,
            label=label,
            alpha=0.88,
         )
         exact_drawn = True
      axes[0, column].set_title(
         f"Balsara 4 {stage.replace('ssprk3_', '').replace('_candidate', '')}"
         f", t={stage_time:.4g}"
      )
      axes[0, column].set_xlim(-0.065, 0.065)
      axes[0, column].set_ylabel(r"$S_2$")
      axes[0, column].set_xlabel("x")
      axes[0, column].grid(alpha=0.2)
   axes[0, 2].legend(frameon=False, fontsize=6.8)

   for column, (component, component_label) in enumerate(
      (("dens", r"$D$"), ("mom1", r"$S_1$"), ("tau", r"$\tau$"))
   ):
      component_index = COMPONENT_NAMES.index(component)
      exact_drawn = False
      stage_time = 0.0
      for run_name, label, color in b3_runs:
         path = paths[(run_name, 2, "ssprk3_stage1_candidate")]
         assert path is not None
         x, numerical, exact_average, stage_time = spatial_profile(path, 3)
         if not exact_drawn:
            axes[1, column].plot(
               x,
               exact_average[component_index],
               color="black",
               linewidth=1.35,
               drawstyle="steps-mid",
               label="exact FV average",
               zorder=4,
            )
         axes[1, column].plot(
            x,
            numerical[component_index],
            color=color,
            linewidth=0.85,
            marker="o",
            markersize=3.0,
            label=label,
            alpha=0.88,
         )
         exact_drawn = True
      axes[1, column].set_title(
         f"Balsara 3 step 2 stage 1, t={stage_time:.4g}"
      )
      axes[1, column].set_xlim(-0.14, 0.14)
      axes[1, column].set_ylabel(component_label)
      axes[1, column].set_xlabel("x")
      axes[1, column].grid(alpha=0.2)
   axes[1, 0].legend(frameon=False, fontsize=7)
   figure.suptitle(
      "SRMHD causal candidate profiles — conserved cell averages, CFL 0.9"
   )
   STAGE_PROFILE_PLOT.parent.mkdir(parents=True, exist_ok=True)
   figure.savefig(STAGE_PROFILE_PLOT, dpi=190)
   plt.close(figure)
   return True


def main() -> int:
   parser = argparse.ArgumentParser(description=__doc__)
   parser.add_argument(
      "--profile",
      dest="profiles",
      action="append",
      choices=tuple(PROFILE_BY_NAME),
      help="trace profile; repeatable (default: timelines and stage-1 replays)",
   )
   parser.add_argument("--max-step", type=int, default=10000)
   parser.add_argument("--dmp-factor", type=float, default=1.05)
   parser.add_argument("--ranks", type=int, default=1)
   parser.add_argument("--timeout", type=int, default=300)
   parser.add_argument("--force", action="store_true")
   parser.add_argument("--plot-only", action="store_true")
   args = parser.parse_args()
   selected_profiles = [
      PROFILE_BY_NAME[name]
      for name in dict.fromkeys(
         args.profiles or [profile.name for profile in TRACE_PROFILES]
      )
   ]

   if args.plot_only:
      rows = read_tsv(RESULTS)
   else:
      rows = []
      for profile in selected_profiles:
         rows.extend(trace_run(profile, args))
      write_tsv(
         RESULTS,
         rows,
         [
            "profile", "phase", "symptom", "case", "cells",
            "reconstruction", "guard", "timestep", "riemann",
            "candidate_riemann", "candidate_dmp",
            "step", "stage", "time",
            "maximum_normalized_component_l1", "maximum_l1_component",
            "maximum_normalized_global_overshoot",
            "maximum_overshoot_component", "maximum_component_tv_ratio",
            "maximum_tv_component", "maximum_plateau_zigzag",
            "plateau_zigzag_count", "maximum_zigzag_component",
            "c2p_fallbacks", "atmosphere_resets", "rmhd_pwl_corrections",
            "rmhd_pwc_corrections", "cumulative_c2p_fallbacks",
            "cumulative_atmosphere_resets",
            "cumulative_rmhd_pwl_corrections",
            "cumulative_rmhd_pwc_corrections", "run_dir",
         ],
      )

   summary_rows = summarize(rows)
   write_tsv(
      SUMMARY,
      summary_rows,
      [
         "profile", "phase", "symptom", "case", "cells",
         "reconstruction", "guard", "timestep", "riemann",
         "candidate_riemann", "candidate_dmp",
         "final_stage", "steps", "final_time",
         "first_global_overshoot_step",
         "first_plateau_zigzag_step", "first_pwc_correction_step",
         "first_c2p_fallback_step", "first_atmosphere_reset_step",
         "final_maximum_l1", "final_maximum_overshoot",
         "final_maximum_tv_ratio", "final_maximum_plateau_zigzag",
         "total_c2p_fallbacks", "total_atmosphere_resets",
         "total_pwc_corrections", "run_dir",
      ],
   )
   plot_trace(rows)
   plot_stage_trace(rows)
   wrote_hybrid_timeline_plot = plot_hybrid_timelines(rows)
   wrote_profile_plot = plot_final_profiles()
   wrote_hybrid_profile_evolution_plot = plot_hybrid_profile_evolution()
   wrote_stage_profile_plot = plot_stage_profiles()
   for row in summary_rows:
      print(
         f"{row['profile']}: final L1={float(row['final_maximum_l1']):.6e} "
         f"overshoot={float(row['final_maximum_overshoot']):.6e} "
         f"TV={float(row['final_maximum_tv_ratio']):.6f} "
         f"zigzag={float(row['final_maximum_plateau_zigzag']):.6e} "
         f"first overshoot/zigzag/PWC/C2P/reset="
         f"{row['first_global_overshoot_step']}/"
         f"{row['first_plateau_zigzag_step']}/"
         f"{row['first_pwc_correction_step']}/"
         f"{row['first_c2p_fallback_step']}/"
         f"{row['first_atmosphere_reset_step']}"
      )
   print(f"wrote {repo_relative(RESULTS)}")
   print(f"wrote {repo_relative(SUMMARY)}")
   print(f"wrote {repo_relative(PLOT)}")
   if any(row["phase"] == "ssprk3_stage_trace" for row in rows):
      print(f"wrote {repo_relative(STAGE_PLOT)}")
   if wrote_hybrid_timeline_plot:
      print(f"wrote {repo_relative(HYBRID_TIMELINE_PLOT)}")
   if wrote_profile_plot:
      print(f"wrote {repo_relative(PROFILE_PLOT)}")
   if wrote_hybrid_profile_evolution_plot:
      print(f"wrote {repo_relative(HYBRID_PROFILE_EVOLUTION_PLOT)}")
   if wrote_stage_profile_plot:
      print(f"wrote {repo_relative(STAGE_PROFILE_PLOT)}")
   return 0


if __name__ == "__main__":
   raise SystemExit(main())
