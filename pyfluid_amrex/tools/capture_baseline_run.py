#!/usr/bin/env python3
"""Run one AMReX validation baseline and preserve only that run's outputs.

Rolling histories under ``runs/current`` remain the normal development log.
This tool is for deliberately freezing one clean suite/benchmark pass when the
current infrastructure is known good.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

from validation_common import ROOT, amrex_executable


DEFAULT_ARCHIVE_ROOT = ROOT / "runs" / "baselines"
SUITE_RUN_ROOT = ROOT / "runs" / "current" / "amrex_standard_gate"
PERF_RUN_ROOT = ROOT / "runs" / "current" / "pyfluid_amrex_performance"
RANK_SCALING_ROOT = ROOT / "runs" / "current" / "pyfluid_amrex_rank_scaling"
SUBCYCLING_BENEFIT_ROOT = ROOT / "runs" / "current" / "subcycling_benefit"
AMR_SUMMARY = ROOT / "runs" / "current" / "amr_validation_summary.tsv"


def clean_label(label: str) -> str:
   cleaned = []
   for char in label.strip().lower():
      if char.isalnum():
         cleaned.append(char)
      elif char in ("-", "_", "."):
         cleaned.append(char)
      elif char.isspace():
         cleaned.append("_")
   value = "".join(cleaned).strip("._-")
   return value or "baseline"


def git_value(args: list[str]) -> str:
   completed = subprocess.run(
      ["git", *args],
      cwd=ROOT,
      text=True,
      stdout=subprocess.PIPE,
      stderr=subprocess.DEVNULL,
      check=False,
   )
   return completed.stdout.strip() if completed.returncode == 0 else ""


def run_logged(command: list[str], log_path: Path) -> int:
   log_path.parent.mkdir(parents=True, exist_ok=True)
   with log_path.open("w") as log:
      completed = subprocess.run(
         command,
         cwd=ROOT,
         text=True,
         stdout=log,
         stderr=subprocess.STDOUT,
         check=False,
      )
   return completed.returncode


def file_sha256(path: Path) -> str:
   digest = hashlib.sha256()
   with path.open("rb") as handle:
      for chunk in iter(lambda: handle.read(1024 * 1024), b""):
         digest.update(chunk)
   return digest.hexdigest()


def write_identity(archive: Path) -> None:
   source_hashes = {
      str(path.relative_to(ROOT)): file_sha256(path)
      for path in sorted((ROOT / "pyfluid_amrex").rglob("*"))
      if path.is_file() and not {"__pycache__", ".uv-cache"}.intersection(path.parts)
   }
   executables = {}
   for dim in (1, 2, 3):
      path = amrex_executable(dim)
      if path.is_file():
         key = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
         executables[key] = file_sha256(path)
   identity = {"source_sha256": source_hashes, "executable_sha256": executables}
   (archive / "identity.json").write_text(json.dumps(identity, indent=2) + "\n")


def copy_if_exists(source: Path, target: Path, copied: list[tuple[Path, Path]]) -> None:
   if not source.exists():
      return
   target.parent.mkdir(parents=True, exist_ok=True)
   if source.is_dir():
      if target.exists():
         shutil.rmtree(target)
      shutil.copytree(source, target)
   else:
      shutil.copy2(source, target)
   copied.append((source, target))


def row_count(path: Path) -> int | str:
   if path.suffix.lower() not in (".tsv", ".csv"):
      return ""
   delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
   try:
      with path.open(newline="") as f:
         reader = csv.reader(f, delimiter=delimiter)
         next(reader, None)
         return sum(1 for _ in reader)
   except OSError:
      return ""


def summary_steps(summary_path: Path) -> set[str]:
   if not summary_path.exists():
      return set()
   with summary_path.open(newline="") as f:
      reader = csv.DictReader(f, delimiter="\t")
      return {row.get("step", "") for row in reader if row.get("step")}


def write_manifest(
   archive: Path,
   *,
   label: str,
   level: str,
   suite_command: list[str],
   suite_returncode: int,
   performance_command: list[str] | None,
   performance_returncode: int | None,
   copied: list[tuple[Path, Path]],
) -> None:
   created = datetime.now().isoformat(timespec="seconds")
   branch = git_value(["branch", "--show-current"])
   head = git_value(["rev-parse", "HEAD"])
   status = git_value(["status", "--short"])

   lines = [
      f"created = {created}",
      f"label = {label}",
      f"suite_level = {level}",
      f"suite_returncode = {suite_returncode}",
      f"suite_command = {' '.join(suite_command)}",
      f"performance_returncode = {'' if performance_returncode is None else performance_returncode}",
      f"performance_command = {'' if performance_command is None else ' '.join(performance_command)}",
      f"git_branch = {branch}",
      f"git_head = {head}",
      "git_status_short =",
      status,
      "",
      "This directory is one deliberately frozen baseline run.",
      "It contains that suite run's summary/logs and optional benchmark result only.",
      "Dependency manifests and source/executable hashes record this snapshot.",
      "It does not copy accumulated validation_history.tsv/performance_history.tsv.",
   ]
   (archive / "manifest.txt").write_text("\n".join(lines) + "\n")

   with (archive / "manifest.tsv").open("w", newline="") as f:
      fields = ["source", "archive", "bytes", "rows"]
      writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
      writer.writeheader()
      for source, target in copied:
         if target.is_dir():
            for child in sorted(path for path in target.rglob("*") if path.is_file()):
               writer.writerow({
                  "source": str((source / child.relative_to(target)).relative_to(ROOT)),
                  "archive": str(child.relative_to(ROOT)),
                  "bytes": child.stat().st_size,
                  "rows": row_count(child),
               })
         else:
            writer.writerow({
               "source": str(source.relative_to(ROOT)),
               "archive": str(target.relative_to(ROOT)),
               "bytes": target.stat().st_size,
               "rows": row_count(target),
            })


def main() -> int:
   parser = argparse.ArgumentParser(description=__doc__)
   parser.add_argument("--label", default="amrex_baseline")
   parser.add_argument("--level", choices=["quick", "standard", "full", "stability"], default="standard")
   parser.add_argument("--kind", action="append", help="Forwarded to validate_amrex_suite.py; may repeat.")
   parser.add_argument("--only", action="append", help="Forwarded to validate_amrex_suite.py; may repeat.")
   parser.add_argument("--archive-root", type=Path, default=DEFAULT_ARCHIVE_ROOT)
   parser.add_argument(
      "--with-performance",
      action=argparse.BooleanOptionalAction,
      default=None,
      help="Also run benchmark_pyfluid.py once and save that run's result. Defaults to on except for --level full, whose suite already includes performance steps.",
   )
   parser.add_argument("--allow-benchmark-warn", action=argparse.BooleanOptionalAction, default=False)
   args = parser.parse_args()

   timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
   label = clean_label(args.label)
   archive = args.archive_root.resolve() / f"{timestamp}_{label}"
   archive.mkdir(parents=True, exist_ok=False)

   suite_command = [
      "uv",
      "run",
      "python",
      "pyfluid_amrex/tools/validate_amrex_suite.py",
      "--level",
      args.level,
   ]
   for kind in args.kind or ():
      suite_command.extend(["--kind", kind])
   for only in args.only or ():
      suite_command.extend(["--only", only])

   print(f"baseline={archive.relative_to(ROOT)}")
   print(f"running suite: {' '.join(suite_command)}", flush=True)
   # Invalid arguments can fail before the suite resets its working output.
   # Never attach an earlier run's summary or logs to a fresh baseline.
   if SUITE_RUN_ROOT.exists():
      shutil.rmtree(SUITE_RUN_ROOT)
   suite_returncode = run_logged(suite_command, archive / "suite.stdout.log")

   copied: list[tuple[Path, Path]] = []
   for name in ("amrex.toml", "metal.toml"):
      copy_if_exists(
         ROOT / "pyfluid_amrex" / "dependencies" / name,
         archive / "dependencies" / name,
         copied,
      )
   suite_summary = SUITE_RUN_ROOT / "summary.tsv"
   copy_if_exists(suite_summary, archive / "validation_summary.tsv", copied)
   copy_if_exists(SUITE_RUN_ROOT / "logs", archive / "validation_logs", copied)
   steps = summary_steps(suite_summary)
   if "performance" in steps:
      copy_if_exists(PERF_RUN_ROOT / "results.tsv", archive / "performance_results.tsv", copied)
   if "rank_scaling" in steps:
      copy_if_exists(RANK_SCALING_ROOT / "results.tsv", archive / "rank_scaling_results.tsv", copied)
   if "subcycling_benefit" in steps:
      copy_if_exists(SUBCYCLING_BENEFIT_ROOT / "subcycling_benefit.tsv", archive / "subcycling_benefit.tsv", copied)
   if "amr_validation" in steps:
      copy_if_exists(AMR_SUMMARY, archive / "amr_validation_summary.tsv", copied)

   performance_command: list[str] | None = None
   performance_returncode: int | None = None
   with_performance = args.with_performance if args.with_performance is not None else args.level != "full"
   if with_performance and suite_returncode == 0:
      performance_command = [
         "uv",
         "run",
         "python",
         "pyfluid_amrex/tools/benchmark_pyfluid.py",
         "--force",
      ]
      if args.allow_benchmark_warn:
         performance_command.append("--allow-warn")
      print(f"running performance: {' '.join(performance_command)}", flush=True)
      if PERF_RUN_ROOT.exists():
         shutil.rmtree(PERF_RUN_ROOT)
      performance_returncode = run_logged(performance_command, archive / "performance.stdout.log")
      copy_if_exists(PERF_RUN_ROOT / "results.tsv", archive / "performance_results.tsv", copied)
   elif with_performance:
      print("skipping performance: validation suite failed", flush=True)

   if (archive / "performance_results.tsv").exists():
      for validation in sorted(PERF_RUN_ROOT.glob("*/validation.tsv")):
         case_dir = validation.parent
         for name in ("validation.tsv", "run.in", "run.log", "scalars.tsv"):
            copy_if_exists(
               case_dir / name,
               archive / "performance_cases" / case_dir.name / name,
               copied,
            )

   write_identity(archive)

   write_manifest(
      archive,
      label=label,
      level=args.level,
      suite_command=suite_command,
      suite_returncode=suite_returncode,
      performance_command=performance_command,
      performance_returncode=performance_returncode,
      copied=copied,
   )

   print(f"wrote {archive.relative_to(ROOT) / 'manifest.tsv'}")
   if suite_returncode != 0:
      return suite_returncode
   if performance_returncode not in (None, 0):
      return performance_returncode
   return 0


if __name__ == "__main__":
   raise SystemExit(main())
