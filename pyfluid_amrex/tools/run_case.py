#!/usr/bin/env python3
"""Run one AMReX case, then validate and report it."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

from postprocess_common import physics_validation_rows, write_physics_validation
from input_parameters import read_key_values
from validation_common import (
   ROOT, SCALAR_POLICIES, amrex_executable, repo_relative, run_amrex,
)


def infer_dim(params: dict[str, str]) -> int:
   ncell = params.get("amr.n_cell", "").split()
   if len(ncell) in (1, 2, 3):
      return len(ncell)
   raise ValueError("run input must define one-, two-, or three-dimensional amr.n_cell")


def output_dir_from_params(params: dict[str, str]) -> Path:
   try:
      path = Path(params["pyfluid.output_dir"])
   except KeyError as exc:
      raise ValueError("run input must define pyfluid.output_dir") from exc
   return path if path.is_absolute() else ROOT / path


def path_uses_directory(path: Path, directory: Path) -> bool:
   """Include symlink entries inside the directory, even if targets are outside."""
   resolved = directory.resolve()
   return path.resolve().is_relative_to(resolved) or any(
      parent.resolve() == resolved for parent in path.absolute().parents
   )


def safe_clean_run_dir(path: Path, *, dependencies: tuple[Path, ...] = ()) -> None:
   resolved = path.resolve()
   runs_root = (ROOT / "runs").resolve()
   if not resolved.is_relative_to(runs_root) or resolved == runs_root:
      raise ValueError(f"refusing to clean non-run output directory: {resolved}")
   for dependency in dependencies:
      if path_uses_directory(dependency, resolved) or resolved.is_relative_to(dependency.resolve()):
         raise ValueError(
            f"refusing to clean {resolved}: overlaps run dependency {dependency}; "
            "select a separate output directory or use --keep-existing"
         )
   if resolved.exists():
      shutil.rmtree(resolved)


def write_run_log(run_dir: Path, command: list[str], stdout: str, seconds: float) -> None:
   run_dir.mkdir(parents=True, exist_ok=True)
   with (run_dir / "run.log").open("w") as f:
      f.write("$ " + " ".join(command) + "\n")
      f.write(f"# wall_seconds = {seconds:.6g}\n")
      f.write(stdout)
      if stdout and not stdout.endswith("\n"):
         f.write("\n")


def call_report(args: argparse.Namespace, run_dir: Path, input_file: Path) -> int:
   command = [
      sys.executable,
      str(ROOT / "pyfluid_amrex/tools/report_run.py"),
      str(run_dir),
      "--input",
      str(input_file),
      "--output",
      str(run_dir / "report.tsv"),
      "--policy",
      args.policy,
      "--mhd",
      args.mhd,
   ]
   if args.skip_plots:
      command.append("--skip-plots")
   if args.gf_plots:
      command.append("--gf-plots")
   if args.numbered_plots:
      command.append("--numbered-plots")
   if args.show_boxes:
      command.append("--show-boxes")
   if args.components:
      command.extend(["--components", args.components])
   if args.aux_components:
      command.extend(["--aux-components", args.aux_components])
   completed = subprocess.run(command, cwd=ROOT, text=True, check=False)
   return completed.returncode


def call_initial_data_report(run_dir: Path, input_file: Path) -> int:
   command = [
      sys.executable,
      str(ROOT / "pyfluid_amrex/tools/initial_data_report.py"),
      str(run_dir),
      "--input",
      str(input_file),
   ]
   completed = subprocess.run(command, cwd=ROOT, text=True, check=False)
   return completed.returncode


def run_physics_hooks(run_dir: Path, input_file: Path) -> bool:
   rows = physics_validation_rows(run_dir, input_file=input_file)
   if not rows:
      return True
   write_physics_validation(run_dir / "physics_postprocess.tsv", rows)
   ok = True
   for row in rows:
      ok = ok and row["status"] == "PASS"
      reason = f" ({row['reason']})" if row.get("reason") else ""
      print(
         f"{row['status']} physics:{row['hook']} "
         f"{row['metric']}={float(row['value']):.3e} "
         f"tol={float(row['tolerance']):.3e}{reason}"
      )
   print(f"wrote {repo_relative(run_dir / 'physics_postprocess.tsv')}")
   return ok


def main() -> int:
   parser = argparse.ArgumentParser(description=__doc__)
   parser.add_argument("input", type=Path)
   parser.add_argument("--ranks", type=int, default=3)
   parser.add_argument("--machine", help="Machine profile name or TOML path; defaults to local or PYFLUID_MACHINE.")
   parser.add_argument("--keep-existing", action="store_true", help="Do not clean the output directory before running.")
   parser.add_argument("--timeout", type=int)
   parser.add_argument("--policy", choices=("auto", *SCALAR_POLICIES.keys()), default="auto")
   parser.add_argument("--mhd", choices=("auto", "yes", "no"), default="auto")
   parser.add_argument("--physics-hooks", choices=("auto", "off"), default="auto")
   parser.add_argument("--skip-plots", action="store_true")
   parser.add_argument("--gf-plots", action="store_true", help="Also generate level-2 state/aux GF snapshot plots.")
   parser.add_argument("--numbered-plots", action="store_true")
   parser.add_argument("--show-boxes", action="store_true")
   parser.add_argument("--components", default="")
   parser.add_argument("--aux-components", default="")
   args = parser.parse_args()

   input_path = args.input if args.input.is_absolute() else ROOT / args.input
   if not input_path.is_file():
      raise FileNotFoundError(f"run input does not exist: {input_path}")
   params = read_key_values(input_path)
   if "pyfluid.run_name" not in params:
      raise ValueError("run input must define pyfluid.run_name")
   dim = infer_dim(params)
   run_dir = output_dir_from_params(params)
   executable = amrex_executable(dim, args.machine)
   if not executable.is_file() or not os.access(executable, os.X_OK):
      raise FileNotFoundError(f"AMReX executable is missing or not executable: {executable}")
   # These are read dependencies, never disposable output. Resolve relative
   # paths against the same working directory used by run_amrex.
   dependencies = [executable]
   for key in ("pyfluid.restart_file", "initial_data.input_file", "pyfluid.eos_table"):
      value = params.get(key, "")
      if value:
         dependency = Path(value)
         dependency = dependency if dependency.is_absolute() else ROOT / dependency
         if not dependency.exists():
            raise FileNotFoundError(f"{key} does not exist: {dependency}")
         dependencies.append(dependency)
   saved_input: bytes | None = None
   if (
      not args.keep_existing
      and path_uses_directory(input_path, run_dir)
   ):
      saved_input = input_path.read_bytes()
   if not args.keep_existing:
      safe_clean_run_dir(run_dir, dependencies=tuple(dependencies))
   if saved_input is not None:
      run_dir.mkdir(parents=True, exist_ok=True)
      input_path = run_dir / "run.in"
      input_path.write_bytes(saved_input)

   result = run_amrex(
      dim=dim,
      input_path=input_path,
      ranks=args.ranks,
      timeout=args.timeout,
      machine=args.machine,
      executable=executable,
   )
   write_run_log(run_dir, result.command, result.stdout, result.seconds)
   resolved_input = run_dir / "run.in"
   print(
      f"{'PASS' if result.returncode == 0 else 'FAIL'} run: "
      f"dim={dim} ranks={args.ranks} seconds={result.seconds:.3f} dir={repo_relative(run_dir)}",
      flush=True,
   )
   if result.returncode != 0:
      return result.returncode
   if not resolved_input.exists():
      raise RuntimeError(f"solver did not write canonical run configuration: {resolved_input}")

   resolved_params = read_key_values(resolved_input)
   if resolved_params.get("initial_data.solve_only", "0") in ("1", "true", "True"):
      return call_initial_data_report(run_dir, resolved_input)

   report_status = call_report(args, run_dir, resolved_input)
   if report_status != 0:
      return report_status

   if args.physics_hooks == "auto" and not run_physics_hooks(run_dir, resolved_input):
      return 1
   return 0


if __name__ == "__main__":
   raise SystemExit(main())
