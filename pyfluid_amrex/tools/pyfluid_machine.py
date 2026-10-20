#!/usr/bin/env python3
"""Configure, build, run, or submit PyFluid through one machine profile."""

from __future__ import annotations

import argparse
import shlex
import subprocess
from pathlib import Path

from machine_common import ROOT, expanded_strings, load_machine
from validation_common import read_key_values


RUNSCRIPT_TEMPLATE = ROOT / "pyfluid_amrex/machines/run.sh.in"


def require_build_value(profile, key: str):
   if key not in profile.build:
      raise ValueError(f"{profile.path}: [build].{key} is required")
   return profile.build[key]


def configure(profile) -> int:
   source_dir = profile.expand(str(require_build_value(profile, "source_dir")))
   build_dir = profile.expand(str(require_build_value(profile, "build_dir")))
   cmake_args = expanded_strings(profile, profile.build.get("cmake_args", ()))
   preset = profile.build.get("configure_preset")
   preset_args = ["--preset", str(preset)] if preset else []
   return subprocess.run(
      ["cmake", "-S", source_dir, "-B", build_dir, *preset_args, *cmake_args],
      cwd=ROOT,
   ).returncode


def build(profile, dimensions: list[int]) -> int:
   build_dir = profile.expand(str(require_build_value(profile, "build_dir")))
   jobs = int(profile.build.get("jobs", 1))
   targets = [f"pyfluid_amrex_{dim}d" for dim in dimensions]
   return subprocess.run(["cmake", "--build", build_dir, "-j", str(jobs), "--target", *targets], cwd=ROOT).returncode


def run_command(profile, input_path: Path, ranks: int) -> list[str]:
   return [
      "uv",
      "run",
      "python",
      str(ROOT / "pyfluid_amrex/tools/run_case.py"),
      str(input_path),
      "--ranks",
      str(ranks),
      "--machine",
      str(profile.path),
   ]


def input_dimension(input_path: Path) -> int:
   n_cell = read_key_values(input_path).get("amr.n_cell", "").split()
   if len(n_cell) not in (1, 2, 3):
      raise ValueError(f"{input_path}: amr.n_cell must have one, two, or three entries")
   return len(n_cell)


def direct_solver_command(profile, input_path: Path, ranks: int) -> list[str]:
   build_dir = Path(profile.expand(str(require_build_value(profile, "build_dir"))))
   executable = build_dir / f"pyfluid_amrex_{input_dimension(input_path)}d"
   launcher = [profile.expand(token).format(ranks=ranks) for token in profile.mpi_launcher]
   if ranks == 1 and profile.direct_one_rank:
      launcher = []
   return [*launcher, str(executable), str(input_path)]


def submission_run_dir(input_path: Path) -> Path:
   value = read_key_values(input_path).get("pyfluid.output_dir")
   if value is None:
      raise ValueError(f"{input_path}: pyfluid.output_dir is required")
   path = Path(value)
   return path if path.is_absolute() else ROOT / path


def write_submission(profile, input_path: Path, ranks: int) -> tuple[Path, list[str]]:
   header = expanded_strings(profile, profile.submit.get("header", ("#!/bin/sh",)))
   command = direct_solver_command(profile, input_path, ranks)
   run_name = input_path.stem
   script = ROOT / "runs/submissions" / f"{run_name}_{profile.name}.sh"
   script.parent.mkdir(parents=True, exist_ok=True)
   environment = "\n".join(
      f"export {name}={shlex.quote(profile.expand(value))}"
      for name, value in profile.environment.items()
   )
   replacements = {
      "@header@": "\n".join(header),
      "@work_dir@": shlex.quote(profile.expand(str(profile.run.get("work_dir", "{root}")))),
      "@environment@": environment,
      "@run_dir@": shlex.quote(str(submission_run_dir(input_path))),
      "@command@": shlex.join(command),
   }
   contents = RUNSCRIPT_TEMPLATE.read_text()
   for placeholder, value in replacements.items():
      contents = contents.replace(placeholder, value)
   script.write_text(contents)
   script.chmod(0o755)
   submit_command = expanded_strings(profile, profile.submit.get("command", ()))
   return script, [*submit_command, str(script)]


def main() -> int:
   parser = argparse.ArgumentParser(description=__doc__)
   parser.add_argument("--machine", help="Profile name or TOML path; defaults to local or PYFLUID_MACHINE.")
   subparsers = parser.add_subparsers(dest="action", required=True)
   subparsers.add_parser("show")
   subparsers.add_parser("configure")
   build_parser = subparsers.add_parser("build")
   build_parser.add_argument("--dim", choices=("1", "2", "3", "all"))
   run_parser = subparsers.add_parser("run")
   run_parser.add_argument("input", type=Path)
   run_parser.add_argument("--ranks", type=int)
   submit_parser = subparsers.add_parser("submit")
   submit_parser.add_argument("input", type=Path)
   submit_parser.add_argument("--ranks", type=int)
   submit_parser.add_argument("--dry-run", action="store_true")
   args = parser.parse_args()

   profile = load_machine(args.machine)
   if args.action == "show":
      print(f"name={profile.name}")
      print(f"path={profile.path}")
      print(f"mpi_launcher={shlex.join(profile.mpi_launcher)}")
      print(f"build_dir={profile.expand(str(profile.build.get('build_dir', '')))}")
      print(f"submit_command={shlex.join(str(value) for value in profile.submit.get('command', ())) or '<disabled>'}")
      return 0
   if args.action == "configure":
      return configure(profile)
   if args.action == "build":
      if args.dim is None:
         dimensions = [int(dim) for dim in profile.build.get("dimensions", [1, 2, 3])]
      else:
         dimensions = [1, 2, 3] if args.dim == "all" else [int(args.dim)]
      if not dimensions or any(dim not in (1, 2, 3) for dim in dimensions):
         raise ValueError(f"{profile.path}: build.dimensions must select 1, 2, or 3")
      return build(profile, dimensions)

   input_path = args.input if args.input.is_absolute() else ROOT / args.input
   input_path = input_path.resolve()
   ranks = args.ranks or profile.default_ranks
   if args.action == "run":
      return subprocess.run(run_command(profile, input_path, ranks), cwd=ROOT).returncode

   script, submit_command = write_submission(profile, input_path, ranks)
   print(f"wrote {script.relative_to(ROOT)}")
   if args.dry_run:
      print(shlex.join(submit_command) if len(submit_command) > 1 else "scheduler submission disabled")
      return 0
   if len(submit_command) <= 1:
      raise ValueError(f"{profile.path}: [submit].command is required for non-dry-run submission")
   return subprocess.run(submit_command, cwd=ROOT).returncode


if __name__ == "__main__":
   raise SystemExit(main())
