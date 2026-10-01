#!/usr/bin/env python3
"""Build one CMake target and append compact compile/cache timing history.

Forced recaches are printed separately: ccache excludes them from hit/miss
counters, so zero hits/misses alone does not prove that a build was a no-op.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

from validation_common import ROOT, append_history, run_bounded_command


DEFAULT_BUILD_DIR = ROOT / "build" / "pyfluid_amrex"
DEFAULT_HISTORY = ROOT / "runs" / "current" / "compile_history.tsv"
DEFAULT_CCACHE_DIR = ROOT / ".ccache" / "pyfluid_amrex"
# Development goals on the maintained Mac mini, not promises for every host
# or method. Selection is explicit: this tool never cleans or forces a recache.
COMPILE_BUDGETS = {"no-op": 1.0, "local-edit": 20.0, "shared-edit": 60.0,
                   "cold-selected": 300.0}
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


def compile_status(returncode: int, elapsed: float, budget: float | None) -> str:
   if returncode:
      return "fail"
   return "slow" if budget is not None and elapsed > budget else "pass"


def main() -> int:
   parser = argparse.ArgumentParser(description=__doc__)
   parser.add_argument("--target", default="pyfluid_amrex_production")
   parser.add_argument("--jobs", type=int, default=6)
   parser.add_argument("--label", default="")
   parser.add_argument("--build-dir", type=Path, default=DEFAULT_BUILD_DIR)
   parser.add_argument("--history", type=Path, default=DEFAULT_HISTORY)
   budgets = parser.add_mutually_exclusive_group()
   budgets.add_argument("--budget", choices=COMPILE_BUDGETS,
                        help="enforce an explicitly selected development time budget")
   budgets.add_argument("--max-seconds", type=float,
                        help="explicit budget for a different host or target scope")
   parser.add_argument("--allow-slow", action="store_true",
                       help="record a slow result without failing; compiler errors still fail")
   parser.add_argument("--timeout", type=float, default=300.0,
                       help="hard build deadline in seconds (default: 300); independent of the timing goal")
   args = parser.parse_args()

   if args.jobs < 1:
      raise SystemExit("--jobs must be positive")
   if not 0.0 < args.timeout < float("inf"):
      raise SystemExit("--timeout must be finite and positive")
   budget = COMPILE_BUDGETS.get(args.budget, args.max_seconds)
   if budget is not None and (not 0.0 < budget < float("inf")):
      raise SystemExit("--max-seconds must be finite and positive")

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
   timed_out = False
   try:
      result = run_bounded_command(command, timeout=args.timeout)
      returncode = result.returncode
   except subprocess.TimeoutExpired:
      timed_out = True
      returncode = 124
   elapsed = time.perf_counter() - start
   after = ccache_stats()

   hits = (
      stat_delta(before, after, "direct_cache_hit")
      + stat_delta(before, after, "preprocessed_cache_hit")
   )
   misses = stat_delta(before, after, "cache_miss")
   recaches = stat_delta(before, after, "recache")
   cache_attempts = hits + misses
   row = {
      "timestamp": datetime.now().isoformat(timespec="milliseconds"),
      "label": args.label,
      "target": args.target,
      "jobs": args.jobs,
      "elapsed_s": f"{elapsed:.6f}",
      "status": "timeout" if timed_out else compile_status(returncode, elapsed, budget),
      "revision": command_output(["git", "rev-parse", "--short", "HEAD"]),
      "dirty": int(bool(command_output(["git", "status", "--porcelain"]))),
      "cache_hits": hits,
      "cache_misses": misses,
      "cache_hit_fraction": f"{hits / cache_attempts:.6f}" if cache_attempts else "",
      "cache_size_mib": f"{after.get('cache_size_kibibyte', 0) / 1024.0:.3f}",
   }
   append_history(args.history, [row], HISTORY_FIELDS, recent_runs=40)
   # Keep the established rolling TSV schema; the latest compact record adds
   # provenance and recache/budget details without archiving existing history.
   args.history.with_suffix(".latest.json").write_text(json.dumps({
      **row, "forced_recompiles": recaches, "budget_s": budget,
      "budget_class": args.budget, "allow_slow": args.allow_slow,
      "timeout_s": args.timeout, "timed_out": timed_out,
      "build_dir": str(build_dir.resolve()), "command": command,
      "cache_stats_available": bool(before) and bool(after),
   }, indent=2) + "\n")
   print(
      f"compile target={args.target} status={row['status']} elapsed={elapsed:.2f}s "
      f"cache_hits={hits} cache_misses={misses} forced_recompiles={recaches} "
      f"history={args.history}"
   )
   if budget is not None:
      print(f"compile budget={budget:g}s status={row['status']}")
   return returncode or (2 if row["status"] == "slow" and not args.allow_slow else 0)


if __name__ == "__main__":
   raise SystemExit(main())
