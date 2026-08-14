#!/usr/bin/env python3
"""Small shared helpers for AMReX validation scripts."""

from __future__ import annotations

import csv
import math
import os
import re
import signal
import subprocess
import tempfile
import time
import tomllib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from causal_validation import infer_localized_causal_model, rows_before_boundary_contact
from machine_common import MachineProfile, load_machine, mpi_prefix


ROOT = Path(__file__).resolve().parents[2]
BUILD = ROOT / "build" / "pyfluid_amrex"


def method_catalog_names(group: str) -> tuple[str, ...]:
   """Return one method family's names from the declarative schema."""
   with (ROOT / "pyfluid_amrex/schema/methods.toml").open("rb") as handle:
      catalog = tomllib.load(handle)
   entries = catalog.get(group)
   if not isinstance(entries, list) or not entries:
      raise RuntimeError(f"methods.toml has no nonempty {group!r} catalog")
   names = tuple(entry.get("name") for entry in entries)
   if any(not isinstance(name, str) or not name for name in names) \
         or len(set(names)) != len(names):
      raise RuntimeError(f"methods.toml has invalid {group!r} names")
   return names


@dataclass(frozen=True)
class ScalarPolicy:
   name: str
   mass_tol: float | None = 1.0e-8
   energy_tol: float | None = 1.0e-8
   momentum_tol: float | None = 1.0e-8
   divb_tol: float | None = 1.0e-10
   floors: bool = True
   require_performance: bool = True


@dataclass(frozen=True)
class AmrexRunResult:
   command: list[str]
   returncode: int
   seconds: float
   stdout: str
   timed_out: bool = False


MPI_FATAL_MARKERS = (
   "amrex::Abort::",
   "MPI_ABORT was invoked",
   "libc++abi: terminating due to uncaught exception",
   "SIGABRT",
   "SIGSEGV",
   "Segmentation fault",
   "Floating point exception",
)


def _signal_process_group(process: subprocess.Popen[bytes], signum: int) -> None:
   try:
      if os.name == "posix":
         os.killpg(process.pid, signum)
      elif process.poll() is None:
         process.send_signal(signum)
   except ProcessLookupError:
      pass


def _terminate_process_group(process: subprocess.Popen[bytes]) -> None:
   """Terminate the launcher and every rank in its dedicated process group."""
   _signal_process_group(process, signal.SIGTERM)
   try:
      process.wait(timeout=3)
   except subprocess.TimeoutExpired:
      pass
   # The launcher can exit while failed ranks remain in the same group.
   _signal_process_group(process, signal.SIGKILL)
   try:
      process.wait(timeout=3)
   except subprocess.TimeoutExpired:
      process.kill()
      process.wait()


def amrex_executable(dim: int) -> Path:
   if dim not in (1, 2, 3):
      raise ValueError(f"AMReX dimension must be 1, 2, or 3, got {dim}")
   return BUILD / f"pyfluid_amrex_{dim}d"


def build_cached_target(target: str, *, parallel: int = 4) -> Path:
   """Bring one configured CMake target up to date or fail with its log."""
   completed = subprocess.run(
      (
         "cmake",
         "--build",
         str(BUILD),
         "--target",
         target,
         "--parallel",
         str(parallel),
      ),
      cwd=ROOT,
      check=False,
      capture_output=True,
      text=True,
   )
   if completed.returncode != 0:
      raise RuntimeError(
         f"failed to build {target}:\n{completed.stdout}\n{completed.stderr}"
      )
   return BUILD / target


def _mpi_prefix(ranks: int, machine: str | Path | MachineProfile | None = None) -> list[str]:
   """Return the canonical launcher prefix; callers execute through ``run_mpi``."""
   return mpi_prefix(ranks, machine)


def mpi_command(
   executable: Path,
   arguments: Sequence[str | Path] = (),
   *,
   ranks: int = 3,
   machine: str | Path | MachineProfile | None = None,
) -> list[str]:
   """Build the canonical local MPI command used by all AMReX tools."""
   program = [str(executable), *(str(argument) for argument in arguments)]
   return [*_mpi_prefix(ranks, machine), *program]


def run_mpi(
   executable: Path,
   arguments: Sequence[str | Path] = (),
   *,
   ranks: int = 3,
   timeout: int | None = None,
   extra_env: dict[str, str] | None = None,
   cwd: Path = ROOT,
   machine: str | Path | MachineProfile | None = None,
) -> AmrexRunResult:
   """Run one local MPI program with the repository's standard process contract."""
   profile = load_machine(machine)
   command = mpi_command(executable, arguments, ranks=ranks, machine=profile)
   env = dict(os.environ)
   for name, value in profile.environment.items():
      env.setdefault(name, value)
   if extra_env:
      env.update(extra_env)
   start = time.perf_counter()
   timed_out = False
   fatal_output = False
   process: subprocess.Popen[bytes] | None = None
   try:
      with tempfile.TemporaryDirectory(prefix="pyfluid-mpi-") as temporary:
         output_path = Path(temporary) / "output.log"
         with output_path.open("wb") as output_stream:
            process = subprocess.Popen(
               command,
               cwd=cwd,
               env=env,
               stdout=output_stream,
               stderr=subprocess.STDOUT,
               start_new_session=os.name == "posix",
            )
            with output_path.open("rb") as output_reader:
               scan_tail = ""
               fatal_seen_at: float | None = None
               while process.poll() is None:
                  chunk = output_reader.read()
                  if chunk:
                     scan_tail = (
                        scan_tail + chunk.decode(errors="replace")
                     )[-8192:]
                     if fatal_seen_at is None and any(
                        marker in scan_tail for marker in MPI_FATAL_MARKERS
                     ):
                        fatal_seen_at = time.perf_counter()
                  elapsed = time.perf_counter() - start
                  if timeout is not None and elapsed >= timeout:
                     timed_out = True
                     _terminate_process_group(process)
                     break
                  if fatal_seen_at is not None \
                        and time.perf_counter() - fatal_seen_at >= 0.5:
                     fatal_output = True
                     _terminate_process_group(process)
                     break
                  time.sleep(0.05)
            returncode = process.wait()
         stdout = output_path.read_text(errors="replace")
   except BaseException:
      if process is not None:
         _terminate_process_group(process)
      raise

   if timed_out or fatal_output:
      if stdout and not stdout.endswith("\n"):
         stdout += "\n"
      if timed_out:
         stdout += f"TIMEOUT after {timeout} seconds; MPI process group terminated\n"
         returncode = -999
      else:
         stdout += "MPI process group terminated after fatal rank output\n"
   elif returncode != 0 and process is not None:
      # A launcher may return before all descendants have been reaped.
      _terminate_process_group(process)
   return AmrexRunResult(
      command=command,
      returncode=returncode,
      seconds=time.perf_counter() - start,
      stdout=stdout,
      timed_out=timed_out,
   )


def select_named_cases(
   cases: Sequence[Any],
   only: Sequence[str] | None,
   *,
   known_cases: Sequence[Any] | None = None,
   filtered_context: str = "selected suite(s)",
) -> list[Any]:
   """Filter dataclass-style cases by their ``name`` attribute."""
   if not only:
      return list(cases)

   wanted = set(only)
   known = known_cases if known_cases is not None else cases
   known_names = {case.name for case in known}
   unknown = sorted(wanted - known_names)
   if unknown:
      raise ValueError(f"unknown case(s): {', '.join(unknown)}")

   selected = [case for case in cases if case.name in wanted]
   selected_names = {case.name for case in selected}
   filtered = sorted(wanted - selected_names)
   if filtered:
      raise ValueError(f"case(s) not in {filtered_context}: {', '.join(filtered)}")
   return selected


def filter_cases_by_attribute(
   cases: Sequence[Any],
   requested: Sequence[str] | None,
   *,
   attribute: str,
   label: str,
) -> list[Any]:
   """Filter dataclass-style cases by an attribute such as coordinate or branch."""
   if not requested or "all" in requested:
      return list(cases)
   wanted = set(requested)
   known = {getattr(case, attribute) for case in cases}
   unknown = sorted(wanted - known)
   if unknown:
      raise ValueError(f"unknown {label}(s): {', '.join(unknown)}")
   return [case for case in cases if getattr(case, attribute) in wanted]


def run_amrex(
   *,
   dim: int,
   input_path: Path,
   ranks: int = 3,
   executable: Path | None = None,
   timeout: int | None = None,
   extra_env: dict[str, str] | None = None,
   machine: str | Path | MachineProfile | None = None,
) -> AmrexRunResult:
   executable_path = executable if executable is not None else amrex_executable(dim)
   return run_mpi(
      executable_path,
      [input_path],
      ranks=ranks,
      timeout=timeout,
      extra_env=extra_env,
      machine=machine,
   )


def materialize_run_input(
   base_input: Path,
   output_path: Path,
   parameters: Mapping[str, str],
   *,
   remove_parameters: Sequence[str] = (),
) -> Path:
   """Write one concrete AMReX input file for a generated validation case."""
   values = read_key_values(base_input)
   values.update(parameters)
   for name in remove_parameters:
      values.pop(name, None)
   output_path.parent.mkdir(parents=True, exist_ok=True)
   with output_path.open("w") as stream:
      for key in sorted(values):
         stream.write(f"{key} = {values[key]}\n")
   return output_path


def parameters_from_assignments(assignments: Sequence[str]) -> dict[str, str]:
   """Convert test-case ``key=value`` declarations before writing a parfile."""
   parameters: dict[str, str] = {}
   for assignment in assignments:
      if "=" not in assignment:
         raise ValueError(f"invalid test parameter declaration: {assignment!r}")
      key, value = assignment.split("=", 1)
      key = key.strip()
      if not key:
         raise ValueError(f"empty test parameter name: {assignment!r}")
      parameters[key] = value.strip()
   return parameters


def scalar_checks(
   name: str = "custom",
   *,
   mass: bool = True,
   energy: bool = True,
   momentum: bool = True,
   divb: bool = True,
   floors: bool = True,
   performance: bool = True,
   mass_tol: float = 1.0e-8,
   energy_tol: float = 1.0e-8,
   momentum_tol: float = 1.0e-8,
   divb_tol: float = 1.0e-10,
) -> ScalarPolicy:
   return ScalarPolicy(
      name=name,
      mass_tol=mass_tol if mass else None,
      energy_tol=energy_tol if energy else None,
      momentum_tol=momentum_tol if momentum else None,
      divb_tol=divb_tol if divb else None,
      floors=floors,
      require_performance=performance,
   )


CLOSED_POLICY = scalar_checks("closed")
OPEN_POLICY = scalar_checks(
   "open",
   mass=False,
   energy=False,
   momentum=False,
)
OPEN_MHD_POLICY = scalar_checks(
   "open_mhd",
   mass=False,
   energy=False,
   momentum=False,
   divb_tol=1.0e-8,
)
DRIVEN_POLICY = scalar_checks(
   "driven",
   energy=False,
   momentum=False,
)
DYNAMIC_SPACETIME_POLICY = scalar_checks(
   "dynamic_spacetime",
   energy=False,
   momentum=False,
)
WALL_POLICY = scalar_checks(
   "wall",
   momentum=False,
)
DIFFUSIVE_POLICY = scalar_checks(
   "diffusive",
)
DIFFUSIVE_WALL_POLICY = scalar_checks(
   "diffusive_wall",
   momentum=False,
)
SMOKE_POLICY = scalar_checks(
   "smoke",
   mass=False,
   energy=False,
   momentum=False,
   divb=False,
   performance=False,
)
ELLIPTIC_POLICY = scalar_checks(
   "elliptic",
   performance=False,
)
BAROTROPIC_POLICY = scalar_checks(
   "barotropic",
   energy=False,
)
BAROTROPIC_DIFFUSIVE_POLICY = scalar_checks(
   "barotropic_diffusive",
   energy=False,
)
BAROTROPIC_WALL_POLICY = scalar_checks(
   "barotropic_wall",
   energy=False,
   momentum=False,
)

SCALAR_POLICIES = {
   policy.name: policy
   for policy in (
      CLOSED_POLICY,
      OPEN_POLICY,
      OPEN_MHD_POLICY,
      DRIVEN_POLICY,
      DYNAMIC_SPACETIME_POLICY,
      WALL_POLICY,
      DIFFUSIVE_POLICY,
      DIFFUSIVE_WALL_POLICY,
      SMOKE_POLICY,
      ELLIPTIC_POLICY,
   )
}

RUN_SUMMARY_RE = re.compile(
   r"PyFluid AMReX (?P<physics>\w+) complete:.*?"
   r"\bbc=(?P<bc>\w+).*?"
   r"\bdiffusion=(?P<diffusion>\w+).*?"
   r"\bviscosity=(?P<viscosity>[^\s]+).*?"
   r"\bresistivity=(?P<resistivity>[^\s]+).*?"
   r"\bct=(?P<ct>\w+)",
)

VALIDATION_HISTORY_FIELDS = [
   "timestamp",
   "kind",
   "suite",
   "step",
   "case",
   "label",
   "status",
   "reason",
   "returncode",
   "seconds",
   "dim",
   "input",
   "policy",
   "ranks",
   "cfl",
   "steps",
   "time",
   "min_rho",
   "min_pressure",
   "mass_rel_drift",
   "energy_rel_drift",
   "gravity_total_energy_rel_drift",
   "passive_mass_rel_drift",
   "momentum_abs_drift",
   "divb_l1",
   "external_budget_valid",
   "external_mass_rel_error",
   "external_energy_rel_error",
   "causal_boundary_contact_time",
   "causal_precontact_rows",
   "causal_mass_rel_drift",
   "causal_energy_rel_drift",
   "causal_momentum_abs_drift",
   "avg_mcells_per_s",
   "avg_sim_time_per_s",
   "amr_active_finest_level",
   "amr_finest_headroom",
   "amr_total_budget_fraction",
   "amr_peak_level_budget_fraction",
   "diffusion_residual",
   "diffusion_iterations",
   "log",
   "tail",
   "description",
]

PERFORMANCE_HISTORY_FIELDS = [
   "timestamp",
   "kind",
   "case",
   "label",
   "status",
   "ranks",
   "repeats",
   "mcells_s",
   "layout_mcells_s",
   "sim_time_per_s",
   "baseline_mcells_s",
   "baseline_label",
   "ratio",
   "warn_fraction",
   "returncode",
   "execution_mode",
   "mpi_processes",
   "speedup_vs_1rank",
   "efficiency_vs_1rank",
   "sim_time_speedup_vs_1rank",
   "sim_time_efficiency_vs_1rank",
   "dominant_stage",
   "dominant_stage_fraction",
   "timestep_fraction",
   "flux_fraction",
   "run_dir",
]


def repo_relative(path: Path) -> str:
   return str(path.resolve().relative_to(ROOT))


def read_tsv(path: Path, *, require_complete: bool = False) -> list[dict[str, str]]:
   """Read a TSV file, optionally rejecting incomplete data rows.

   History files intentionally contain blank optional columns. Runtime scalar
   output does not: an incomplete scalar row means a failed or interrupted
   writer and must not be interpreted as simulation data.
   """
   if not path.exists():
      return []
   rows: list[str] = []
   line_numbers: list[int] = []
   with path.open(newline="") as f:
      for line_number, raw in enumerate(f, start=1):
         line = raw.strip()
         if not line or line.startswith("#"):
            continue
         rows.append(line)
         line_numbers.append(line_number)
   if not rows:
      return []
   reader = csv.DictReader(rows, delimiter="\t")
   fieldnames = reader.fieldnames or []
   parsed_rows: list[dict[str, str]] = []
   for row_number, row in enumerate(reader, start=1):
      if require_complete and any(row.get(field) in (None, "") for field in fieldnames):
         source_line = line_numbers[row_number] if row_number < len(line_numbers) else "unknown"
         raise ValueError(f"{path}: incomplete TSV row at line {source_line}")
      parsed_rows.append(row)
   return parsed_rows


def read_float_tsv(path: Path, *, stop_on_incomplete: bool = False) -> list[dict[str, float]]:
   raw_rows = read_tsv(path, require_complete=not stop_on_incomplete)
   rows: list[dict[str, float]] = []
   for row in raw_rows:
      if any(value in (None, "") for value in row.values()):
         if stop_on_incomplete:
            break
         raise ValueError(f"{path}: incomplete TSV row")
      rows.append({key: float(value) for key, value in row.items()})
   return rows


def read_scalar_rows(output_dir: Path) -> list[dict[str, str]]:
   return read_tsv(output_dir / "scalars.tsv", require_complete=True)


def next_numbered_path(directory: Path, stem: str, suffix: str = ".png") -> Path:
   directory.mkdir(parents=True, exist_ok=True)
   index = 1
   while True:
      path = directory / f"{stem}_{index:03d}{suffix}"
      if not path.exists():
         return path
      index += 1


def read_key_values(path: Path | None) -> dict[str, str]:
   values: dict[str, str] = {}
   if path is None or not path.exists():
      return values
   for raw in path.read_text().splitlines():
      line = raw.split("#", 1)[0].strip()
      if not line or "=" not in line:
         continue
      key, value = line.split("=", 1)
      values[key.strip()] = value.strip()
   return values


def find_input_file(run_dir: Path, explicit: Path | None = None) -> Path | None:
   if explicit is not None:
      return explicit
   candidates = [
      run_dir / "input.in",
      run_dir / "run.in",
      run_dir / f"{run_dir.name}.in",
      run_dir.parent / "inputs" / f"{run_dir.name}.in",
   ]
   for candidate in candidates:
      if candidate.exists():
         return candidate
   return None


def read_run_summary(run_dir: Path) -> dict[str, str]:
   log = run_dir / "run.log"
   if not log.exists():
      return {}
   match = RUN_SUMMARY_RE.search(log.read_text(errors="replace"))
   return match.groupdict() if match else {}


def parse_float(value: object, default: float = math.nan) -> float:
   try:
      return float(value)  # type: ignore[arg-type]
   except (TypeError, ValueError):
      return default


def finite_positive(value: float) -> bool:
   return math.isfinite(value) and value > 0.0


def average_positive(rows: Iterable[dict[str, str]], key: str) -> float:
   values = [parse_float(row.get(key)) for row in rows]
   positives = [value for value in values if finite_positive(value)]
   if not positives:
      return math.nan
   return sum(positives) / len(positives)


def momentum_abs_drift(rows: list[dict[str, str]]) -> float:
   if not rows:
      return math.nan
   first = rows[0]
   max_drift = 0.0
   for row in rows:
      total = 0.0
      for key in ("mom1", "mom2", "mom3"):
         delta = parse_float(row.get(key), 0.0) - parse_float(first.get(key), 0.0)
         total += delta * delta
      drift = math.sqrt(total)
      if not math.isfinite(drift):
         return math.nan
      max_drift = max(max_drift, drift)
   return max_drift


def max_abs_column(rows: Iterable[dict[str, str]], key: str) -> float:
   values = [abs(parse_float(row.get(key))) for row in rows]
   finite_values = [value for value in values if math.isfinite(value)]
   if len(finite_values) != len(values):
      return math.nan
   return max(finite_values, default=math.nan)


def has_gravity_total_energy(rows: list[dict[str, str]]) -> bool:
   if not rows or "gravity_total_energy_rel_drift" not in rows[0]:
      return False
   for row in rows:
      if abs(parse_float(row.get("gravity_potential_energy"), 0.0)) > 0.0:
         return True
   return False


def min_column(rows: Iterable[dict[str, str]], key: str) -> float:
   values = [parse_float(row.get(key)) for row in rows]
   finite_values = [value for value in values if math.isfinite(value)]
   if len(finite_values) != len(values):
      return math.nan
   return min(finite_values, default=math.nan)


def enabled_number(value: str | None) -> bool:
   if value is None:
      return False
   try:
      return abs(float(value)) > 0.0
   except ValueError:
      return value.lower() not in ("none", "off", "false", "0")


def explicit_scalar_policy(params: dict[str, str]) -> ScalarPolicy | None:
   policy_name = params.get("pyfluid.validation_policy")
   if not policy_name:
      return None
   policy_name = policy_name.strip().lower()
   if policy_name not in SCALAR_POLICIES:
      raise ValueError(f"unknown validation policy {policy_name!r}; choose one of {', '.join(SCALAR_POLICIES)}")
   return SCALAR_POLICIES[policy_name]


def infer_scalar_policy(
   params: dict[str, str],
   summary: dict[str, str],
   rows: list[dict[str, str]],
) -> tuple[ScalarPolicy, str]:
   explicit = explicit_scalar_policy(params)
   if explicit is not None:
      return explicit, "explicit validation policy"

   bc_values = []
   for key, value in params.items():
      if key == "pyfluid.bc" or key.startswith("pyfluid.bc_"):
         bc_values.append(value.strip().lower())
   if summary.get("bc"):
      bc_values.append(summary["bc"].strip().lower())
   if any(value in ("inflow", "outflow", "diode") for value in bc_values):
      return OPEN_POLICY, "open/inflow/diode boundary"
   has_reflecting_wall = any(value == "reflecting" for value in bc_values)

   solid_motion = params.get("pyfluid.solid_motion", "").strip().lower()
   wall_work = parse_float(rows[-1].get("wall_work"), 0.0) if rows else 0.0
   if solid_motion and solid_motion not in ("none", "off", "static"):
      return DRIVEN_POLICY, "moving/driven solid"
   if abs(wall_work) > 0.0:
      return DRIVEN_POLICY, "nonzero wall work"

   if params.get("spacetime.evolution", "none").strip().lower() == "evolved":
      return DYNAMIC_SPACETIME_POLICY, "fluid evolution in a time-dependent spacetime"

   eos = params.get("pyfluid.eos", summary.get("eos", "")).strip().lower()
   is_barotropic = eos == "polytrope"
   diffusion = params.get("pyfluid.diffusion", summary.get("diffusion", "")).strip().lower()
   viscosity = params.get("pyfluid.viscosity", summary.get("viscosity"))
   resistivity = params.get("pyfluid.resistivity", summary.get("resistivity"))
   if diffusion not in ("", "none", "off") or enabled_number(viscosity) or enabled_number(resistivity):
      if is_barotropic:
         if has_reflecting_wall:
            return BAROTROPIC_WALL_POLICY, "barotropic EOS with diffusion and reflecting wall"
         return BAROTROPIC_DIFFUSIVE_POLICY, "barotropic EOS with diffusion/resistivity"
      if has_reflecting_wall:
         return DIFFUSIVE_WALL_POLICY, "diffusion/resistivity with reflecting wall"
      return DIFFUSIVE_POLICY, "diffusion/resistivity enabled"

   if has_reflecting_wall:
      if is_barotropic:
         return BAROTROPIC_WALL_POLICY, "barotropic EOS with reflecting wall"
      return WALL_POLICY, "reflecting wall"

   if is_barotropic:
      return BAROTROPIC_POLICY, "barotropic EOS"

   return CLOSED_POLICY, "closed/default"


def infer_mhd_run(params: dict[str, str], summary: dict[str, str], rows: list[dict[str, str]]) -> bool:
   physics = (params.get("pyfluid.physics") or summary.get("physics") or "").strip().lower()
   if physics == "mhd":
      return True
   if (params.get("pyfluid.ct") or summary.get("ct") or "").strip().lower() == "on":
      return True
   if params.get("pyfluid.initial_data", "").lower().find("mag") >= 0:
      return True
   if rows:
      magnetic_energy = parse_float(
         rows[-1].get("newtonian_magnetic_energy"), 0.0
      ) + parse_float(rows[-1].get("rmhd_electromagnetic_energy"), 0.0)
      divb = parse_float(rows[-1].get("divb_l1"), 0.0)
      return abs(magnetic_energy) > 0.0 or abs(divb) > 0.0
   return False


def check_scalars(
   rows: list[dict[str, str]],
   *,
   mhd: bool = False,
   mass: bool = True,
   energy: bool = True,
   momentum: bool = True,
   divb: bool = True,
   floors: bool = True,
   performance: bool = True,
   mass_tol: float = 1.0e-8,
   energy_tol: float = 1.0e-8,
   momentum_tol: float = 1.0e-8,
   divb_tol: float = 1.0e-10,
) -> list[str]:
   if not rows:
      return ["missing scalars.tsv rows"]
   failures: list[str] = []

   if floors:
      min_rho_values = [parse_float(row.get("min_rho")) for row in rows]
      min_pressure_values = [parse_float(row.get("min_pressure")) for row in rows]
      if not min_rho_values or any(not finite_positive(value) for value in min_rho_values):
         failures.append("min_rho has a non-positive or non-finite row")
      if not min_pressure_values or any(not finite_positive(value) for value in min_pressure_values):
         failures.append("min_pressure has a non-positive or non-finite row")

   if mass:
      mass_drift = max_abs_column(rows, "mass_rel_drift")
      if not math.isfinite(mass_drift) or mass_drift > mass_tol:
         failures.append(f"mass_rel_drift={mass_drift:.3e} > {mass_tol:.3e}")
      passive_mass = max_abs_column(rows, "passive_mass")
      if math.isfinite(passive_mass) and passive_mass > 0.0:
         passive_drift = max_abs_column(rows, "passive_mass_rel_drift")
         if not math.isfinite(passive_drift) or passive_drift > mass_tol:
            failures.append(f"passive_mass_rel_drift={passive_drift:.3e} > {mass_tol:.3e}")

   if energy:
      energy_key = "gravity_total_energy_rel_drift" if has_gravity_total_energy(rows) else "energy_rel_drift"
      energy_drift = max_abs_column(rows, energy_key)
      if not math.isfinite(energy_drift) or energy_drift > energy_tol:
         failures.append(f"{energy_key}={energy_drift:.3e} > {energy_tol:.3e}")

   if momentum:
      momentum_drift = momentum_abs_drift(rows)
      if not math.isfinite(momentum_drift) or momentum_drift > momentum_tol:
         failures.append(f"momentum_abs_drift={momentum_drift:.3e} > {momentum_tol:.3e}")

   if mhd and divb:
      divb_drift = max_abs_column(rows, "divb_l1")
      if not math.isfinite(divb_drift) or divb_drift > divb_tol:
         failures.append(f"divb_l1={divb_drift:.3e} > {divb_tol:.3e}")

   if performance:
      mcells = average_positive(rows, "mcells_per_s")
      sim_rate = average_positive(rows, "sim_time_per_s")
      if not finite_positive(mcells):
         failures.append("missing positive mcells_per_s")
      if not finite_positive(sim_rate):
         failures.append("missing positive sim_time_per_s")

   return failures


def check_scalar_rows(rows: list[dict[str, str]], policy: ScalarPolicy, *, mhd: bool) -> list[str]:
   failures = check_scalars(
      rows,
      mhd=mhd,
      mass=policy.mass_tol is not None,
      energy=policy.energy_tol is not None,
      momentum=policy.momentum_tol is not None,
      divb=policy.divb_tol is not None,
      floors=policy.floors,
      performance=policy.require_performance,
      mass_tol=policy.mass_tol or 1.0e-8,
      energy_tol=policy.energy_tol or 1.0e-8,
      momentum_tol=policy.momentum_tol or 1.0e-8,
      divb_tol=policy.divb_tol or 1.0e-10,
   )
   failures.extend(
      check_external_conservation(
         rows,
         mass=policy.mass_tol is not None,
         energy=policy.energy_tol is not None,
      )
   )
   return failures


def check_external_conservation(
   rows: list[dict[str, str]],
   *,
   mass: bool = True,
   energy: bool = True,
   tolerance: float = 2.0e-10,
) -> list[str]:
   """Check the solver's discrete physical-boundary/source conservation ledger."""
   if not rows or "external_budget_valid" not in rows[0]:
      return []
   valid_rows = [row for row in rows if parse_float(row.get("external_budget_valid"), 0.0) > 0.5]
   if not valid_rows:
      return []

   failures: list[str] = []
   columns = []
   if mass:
      columns.append("external_mass_rel_error")
   if energy:
      columns.append("external_energy_rel_error")
   for column in columns:
      error = max_abs_column(valid_rows, column)
      if not math.isfinite(error) or error > tolerance:
         failures.append(f"{column}={error:.3e} > {tolerance:.3e}")
   return failures


def check_causal_precontact_scalars(
   rows: list[dict[str, str]],
   params: dict[str, str],
   policy: ScalarPolicy,
) -> tuple[list[str], dict[str, object]]:
   metrics: dict[str, object] = {
      "causal_boundary_contact_time": math.nan,
      "causal_precontact_rows": 0,
      "causal_mass_rel_drift": math.nan,
      "causal_energy_rel_drift": math.nan,
      "causal_momentum_abs_drift": math.nan,
   }
   if policy.name not in (OPEN_POLICY.name, OPEN_MHD_POLICY.name):
      return [], metrics

   model = infer_localized_causal_model(params)
   if model is None or not math.isfinite(model.boundary_contact_time):
      return [], metrics

   precontact = rows_before_boundary_contact(rows, model)
   evolved = [row for row in precontact if parse_float(row.get("time"), 0.0) > 0.0]
   metrics.update({
      "causal_boundary_contact_time": model.boundary_contact_time,
      "causal_precontact_rows": len(evolved),
      "causal_mass_rel_drift": max_abs_column(precontact, "mass_rel_drift"),
      "causal_energy_rel_drift": max_abs_column(precontact, "energy_rel_drift"),
      "causal_momentum_abs_drift": momentum_abs_drift(precontact),
   })
   if not evolved:
      return [], metrics

   max_level = int(parse_float(params.get("amr.max_level"), 0.0))
   tolerance = 1.0e-11 if max_level > 0 else 1.0e-12
   # Curvilinear equations evolve local orthonormal momentum components with
   # geometric source terms. Runtime diagnostics report their physical
   # Cartesian integral, but that integral is not a telescoping finite-volume
   # invariant of this formulation and therefore converges rather than
   # remaining at roundoff. Keep reporting it; reserve the roundoff gate for
   # Cartesian conservative momentum.
   check_momentum = params.get("mesh.coordinate", "cartesian") == "cartesian"
   failures = check_scalars(
      precontact,
      mass=True,
      energy=True,
      momentum=check_momentum,
      divb=False,
      floors=False,
      performance=False,
      mass_tol=tolerance,
      energy_tol=tolerance,
      momentum_tol=tolerance,
   )
   return [f"before causal boundary contact: {failure}" for failure in failures], metrics


def scalar_validation_row(
   run_dir: Path,
   *,
   case: str | None = None,
   input_file: Path | None = None,
   policy: ScalarPolicy | str | None = None,
   checks: ScalarPolicy | str | None = None,
   mhd: bool | None = None,
   dim: int | str | None = None,
   ranks: int | str | None = None,
   base_input: str = "",
   kind: str = "scalar_sanity",
) -> dict[str, object]:
   run_dir = run_dir.resolve()
   rows = read_scalar_rows(run_dir)
   found_input = find_input_file(run_dir, input_file)
   params = read_key_values(found_input)
   summary = read_run_summary(run_dir)

   selected_checks = checks if checks is not None else policy
   if selected_checks is None or selected_checks == "auto":
      scalar_policy, policy_reason = infer_scalar_policy(params, summary, rows)
   elif isinstance(selected_checks, ScalarPolicy):
      scalar_policy = selected_checks
      policy_reason = "declared validation policy"
   else:
      scalar_policy = SCALAR_POLICIES[selected_checks]
      policy_reason = "declared validation policy"

   mhd_enabled = infer_mhd_run(params, summary, rows) if mhd is None else mhd
   failures = check_scalar_rows(rows, scalar_policy, mhd=mhd_enabled)
   causal_failures, causal_metrics = check_causal_precontact_scalars(rows, params, scalar_policy)
   failures.extend(causal_failures)
   if math.isfinite(parse_float(causal_metrics["causal_boundary_contact_time"])):
      policy_reason += "; closed scalar budgets enforced before causal boundary contact"
   final = rows[-1] if rows else {}
   return {
      "kind": kind,
      "case": case or run_dir.name,
      "label": case or run_dir.name,
      "status": "PASS" if not failures else "FAIL",
      "reason": "; ".join(failures),
      "dim": "" if dim is None else str(dim),
      "input": base_input or (repo_relative(found_input) if found_input else ""),
      "policy": scalar_policy.name,
      "description": policy_reason,
      "run_dir": repo_relative(run_dir),
      "ranks": "" if ranks is None else str(ranks),
      "steps": final.get("step", ""),
      "time": final.get("time", ""),
      "min_rho": min_column(rows, "min_rho"),
      "min_pressure": min_column(rows, "min_pressure"),
      "mass_rel_drift": max_abs_column(rows, "mass_rel_drift"),
      "energy_rel_drift": max_abs_column(rows, "energy_rel_drift"),
      "gravity_total_energy_rel_drift": max_abs_column(rows, "gravity_total_energy_rel_drift"),
      "passive_mass_rel_drift": max_abs_column(rows, "passive_mass_rel_drift"),
      "momentum_abs_drift": momentum_abs_drift(rows),
      "divb_l1": max_abs_column(rows, "divb_l1") if mhd_enabled else 0.0,
      "external_budget_valid": int(parse_float(final.get("external_budget_valid"), 0.0) > 0.5),
      "external_mass_rel_error": max_abs_column(rows, "external_mass_rel_error"),
      "external_energy_rel_error": max_abs_column(rows, "external_energy_rel_error"),
      **causal_metrics,
      "avg_mcells_per_s": average_positive(rows, "mcells_per_s"),
      "avg_sim_time_per_s": average_positive(rows, "sim_time_per_s"),
      "amr_active_finest_level": final.get("amr_active_finest_level", ""),
      "amr_finest_headroom": final.get("amr_finest_headroom", ""),
      "amr_total_budget_fraction": final.get("amr_total_budget_fraction", ""),
      "amr_peak_level_budget_fraction": final.get("amr_peak_level_budget_fraction", ""),
      "diffusion_residual": final.get("diffusion_residual", ""),
      "diffusion_iterations": final.get("diffusion_iterations", ""),
   }


_TSV_CONTROL_TRANSLATION = str.maketrans({
   "\0": r"\0",
   "\t": r"\t",
   "\r": r"\r",
   "\n": r"\n",
})


def sanitize_tsv_row(row: dict[str, object]) -> dict[str, object]:
   return {
      key: value.translate(_TSV_CONTROL_TRANSLATION)
      if isinstance(value, str)
      else value
      for key, value in row.items()
   }


def write_tsv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
   path.parent.mkdir(parents=True, exist_ok=True)
   with path.open("w", newline="") as f:
      writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t", extrasaction="ignore")
      writer.writeheader()
      writer.writerows(sanitize_tsv_row(row) for row in rows)


def recent_history_path(path: Path) -> Path:
   return path.with_name(f"{path.stem}_recent{path.suffix}")


def recent_history_rows(rows: list[dict[str, str]], keep_runs: int) -> list[dict[str, str]]:
   if keep_runs <= 0:
      return []
   kept_timestamps: set[str] = set()
   for row in reversed(rows):
      timestamp = row.get("timestamp", "")
      if timestamp and timestamp not in kept_timestamps:
         kept_timestamps.add(timestamp)
         if len(kept_timestamps) >= keep_runs:
            break
   return [row for row in rows if row.get("timestamp", "") in kept_timestamps]


def write_recent_history(path: Path, fields: list[str], keep_runs: int) -> None:
   rows = read_tsv(path)
   write_tsv(recent_history_path(path), recent_history_rows(rows, keep_runs), fields)


def append_history(path: Path, rows: list[dict[str, object]], fields: list[str], *, recent_runs: int) -> None:
   path.parent.mkdir(parents=True, exist_ok=True)
   if path.exists():
      with path.open(newline="") as f:
         reader = csv.reader(f, delimiter="\t")
         header = next(reader, [])
      if header != fields:
         stamp = datetime.now().isoformat(timespec="seconds").replace(":", "-")
         archive = path.with_name(f"{path.stem}.legacy_{stamp}{path.suffix}")
         path.rename(archive)
   exists = path.exists()
   with path.open("a", newline="") as f:
      writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t", extrasaction="ignore")
      if not exists:
         writer.writeheader()
      stamp = datetime.now().isoformat(timespec="milliseconds")
      for row in rows:
         values = sanitize_tsv_row(row)
         values.setdefault("timestamp", stamp)
         writer.writerow(values)
   write_recent_history(path, fields, recent_runs)


def append_validation_history(path: Path, rows: list[dict[str, object]]) -> None:
   append_history(path, rows, VALIDATION_HISTORY_FIELDS, recent_runs=10)


def append_performance_history(path: Path, rows: list[dict[str, object]]) -> None:
   append_history(path, rows, PERFORMANCE_HISTORY_FIELDS, recent_runs=40)
