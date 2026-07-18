#!/usr/bin/env python3
"""Build one CMake target and append compact compile/cache timing history."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

from validation_common import ROOT, append_history


DEFAULT_BUILD_DIR = ROOT / "build" / "pyfluid_amrex"
DEFAULT_HISTORY = ROOT / "runs" / "current" / "compile_history.tsv"
DEFAULT_CCACHE_DIR = ROOT / ".ccache" / "pyfluid_amrex"
HISTORY_FIELDS = [
   "timestamp",
   "label",
   "target",
   "jobs",
   "elapsed_s",
   "status",
   "revision",
   "dirty",
   "cache_hits",
   "cache_misses",
   "cache_hit_fraction",
   "cache_size_mib",
]


def command_output(command: list[str]) -> str:
   result = subprocess.run(
      command,
      cwd=ROOT,
      check=False,
      capture_output=True,
      text=True,
   )
   return result.stdout.strip() if result.returncode == 0 else ""


def ccache_stats() -> dict[str, int]:
   ccache = shutil.which("ccache")
   if ccache is None:
      return {}
   environment = os.environ.copy()
   environment.setdefault("CCACHE_DIR", str(DEFAULT_CCACHE_DIR))
   result = subprocess.run(
      [ccache, "--print-stats", "--format", "json"],
      cwd=ROOT,
      check=False,
      capture_output=True,
      text=True,
      env=environment,
   )
   if result.returncode != 0:
      return {}
   try:
      return {key: int(value) for key, value in json.loads(result.stdout).items()}
   except (TypeError, ValueError):
      return {}


def stat_delta(before: dict[str, int], after: dict[str, int], key: str) -> int:
   return max(after.get(key, 0) - before.get(key, 0), 0)


def main() -> int:
   parser = argparse.ArgumentParser(description=__doc__)
   parser.add_argument("--target", default="pyfluid_amrex_production")
   parser.add_argument("--jobs", type=int, default=6)
   parser.add_argument("--label", default="")
   parser.add_argument("--build-dir", type=Path, default=DEFAULT_BUILD_DIR)
   parser.add_argument("--history", type=Path, default=DEFAULT_HISTORY)
   args = parser.parse_args()

   if args.jobs < 1:
      raise SystemExit("--jobs must be positive")

   build_dir = args.build_dir if args.build_dir.is_absolute() else ROOT / args.build_dir
   command = [
      "cmake",
      "--build",
      str(build_dir),
      "--target",
      args.target,
      "--parallel",
      str(args.jobs),
   ]
   before = ccache_stats()
   start = time.perf_counter()
   result = subprocess.run(command, cwd=ROOT, check=False)
   elapsed = time.perf_counter() - start
   after = ccache_stats()

   hits = (
      stat_delta(before, after, "direct_cache_hit")
      + stat_delta(before, after, "preprocessed_cache_hit")
   )
   misses = stat_delta(before, after, "cache_miss")
   cache_attempts = hits + misses
   row = {
      "label": args.label,
      "target": args.target,
      "jobs": args.jobs,
      "elapsed_s": f"{elapsed:.6f}",
      "status": "pass" if result.returncode == 0 else "fail",
      "revision": command_output(["git", "rev-parse", "--short", "HEAD"]),
      "dirty": int(bool(command_output(["git", "status", "--porcelain"]))),
      "cache_hits": hits,
      "cache_misses": misses,
      "cache_hit_fraction": f"{hits / cache_attempts:.6f}" if cache_attempts else "",
      "cache_size_mib": f"{after.get('cache_size_kibibyte', 0) / 1024.0:.3f}",
   }
   append_history(args.history, [row], HISTORY_FIELDS, recent_runs=40)
   print(
      f"compile target={args.target} status={row['status']} elapsed={elapsed:.2f}s "
      f"cache_hits={hits} cache_misses={misses} history={args.history}"
   )
   return result.returncode


if __name__ == "__main__":
   raise SystemExit(main())
