#!/usr/bin/env python3
"""Check spherical-origin topology, tensor signs and partial-hierarchy donors."""

from __future__ import annotations

import argparse

from validation_common import (
    ROOT,
    append_validation_history,
    build_cached_target,
    repo_relative,
    run_mpi,
    write_tsv,
)


RUN_ROOT = ROOT / "runs/current/origin_boundary"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ranks", type=int, default=3)
    parser.add_argument("--dims", nargs="+", type=int, choices=(2, 3), default=(2, 3))
    args = parser.parse_args()
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    rows = []
    for dimension in sorted(set(args.dims)):
        target = f"pyfluid_amrex_origin_boundary_check_{dimension}d"
        executable = build_cached_target(target)
        result = run_mpi(executable, ranks=args.ranks, timeout=30)
        log = RUN_ROOT / f"{dimension}d.log"
        log.write_text(result.stdout)
        passed = result.returncode == 0 and "PASS spherical-origin topology check" in result.stdout
        row = {
            "kind": "feature", "suite": "origin_boundary", "case": target,
            "status": "PASS" if passed else "FAIL", "ranks": args.ranks,
            "policy": "boundary_topology", "seconds": result.seconds,
            "returncode": result.returncode, "log": repo_relative(log),
        }
        rows.append(row)
        print(f"{row['status']} {target}: {repo_relative(log)}", flush=True)
        if not passed:
            print(result.stdout, flush=True)
    write_tsv(RUN_ROOT / "summary.tsv", rows, list(rows[0]))
    append_validation_history(ROOT / "runs/current/validation_history.tsv", rows)
    return 0 if all(row["status"] == "PASS" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
