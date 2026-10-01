#!/usr/bin/env python3
"""Replay selected CMake compile commands uncached, outside the working build.

Owners run sequentially with their actual production flags. This measures
individual translation units, not generation, linking or a parallel clean
build. --trace supports Clang (including MPI wrappers) and NVIDIA NVCC.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import subprocess
import time

from validation_common import ROOT, run_bounded_command


def compile_arguments(entry: dict) -> list[str]:
    return entry.get("arguments") or shlex.split(entry["command"])


def isolated_command(entry: dict, output: Path, trace: Path | None = None) -> list[str]:
    """Retarget only compiler outputs; never overwrite CMake objects/depfiles."""
    arguments = compile_arguments(entry)
    if not {"-c", "--compile"}.intersection(arguments) \
            or any(arg.startswith("@") for arg in arguments):
        raise ValueError("expected a compile-only command without response files")
    if Path(arguments[0]).name in {"ccache", "sccache", "cmake"}:
        raise ValueError("compile database must contain the compiler, not a launcher")
    paired = {"-o", "-MF", "-MT", "-MQ", "-MJ", "--serialize-diagnostics",
              "--output-file", "--dependency-output",
              "--fdevice-time-trace", "-fdevice-time-trace"}
    stripped = {"-MD", "-MMD", "-MP", "-MG", "-ftime-trace",
                "--generate-dependencies-with-compile",
                "--generate-nonsystem-dependencies-with-compile",
                "--generate-dependency-targets"}
    result = []
    index = 0
    while index < len(arguments):
        argument = arguments[index]
        if argument in {"--keep", "-keep", "--keep-dir", "-keep-dir",
                        "--options-file", "-optf", "--time", "-time"} or argument.startswith(
            ("--keep-dir=", "-keep-dir=", "--options-file=", "-optf=",
             "--time=", "-time=")
        ):
            raise ValueError(f"unsupported output-producing option: {argument}")
        if argument in paired:
            if index + 1 == len(arguments):
                raise ValueError(f"missing value after {argument}")
            index += 2
            continue
        # NVCC's optimization report is not a joined -o output filename.
        if argument == "-opt-info" or argument.startswith("-opt-info="):
            result.append(argument)
            index += 1
            continue
        if argument in stripped or argument.startswith("-ftime-trace=") \
                or argument.startswith("-ftime-trace-granularity=") \
                or argument.startswith(("--fdevice-time-trace=", "-fdevice-time-trace=")) \
                or argument.startswith(("--output-file=", "--dependency-output=")) \
                or argument.startswith("--serialize-diagnostics=") \
                or any(argument.startswith(flag) and argument != flag
                       for flag in ("-o", "-MF", "-MT", "-MQ", "-MJ")):
            index += 1
            continue
        if argument.startswith(("-Wp,", "-save-temps", "-fprofile", "-fcoverage",
                                "-fsave-optimization-record", "-foptimization-record")) \
                or argument in {"--coverage", "-ftest-coverage", "-Xclang"}:
            raise ValueError(f"unsupported output-producing option: {argument}")
        result.append(argument)
        index += 1
    result += ["-o", str(output)]
    if trace is not None:
        if Path(arguments[0]).name in {"nvcc", "nvcc.exe"}:
            result += [f"--fdevice-time-trace={trace}"]
        else:
            result += [f"-ftime-trace={trace}", "-ftime-trace-granularity=10000"]
    return result


def trace_totals(path: Path) -> dict[str, float]:
    trace = json.loads(path.read_text())
    events = trace["traceEvents"]
    if "isNVRTC" in trace.get("otherData", {}):
        # NVCC records nested events and may run tools/threads concurrently.
        # Report each tool's elapsed span, never a sum of its nested events.
        processes = {event["pid"]: event["args"]["name"] for event in events
                     if event.get("name") == "process_name"}
        totals = {}
        for process, name in processes.items():
            spans = [event for event in events
                     if event.get("pid") == process and event.get("ph") == "X"]
            if spans:
                start = min(event["ts"] for event in spans)
                end = max(event["ts"] + event["dur"] for event in spans)
                totals[f"{name} (process {process})"] = (end - start) / 1e6
        return totals
    # Totals are inclusive and overlap. Never sum frontend/instantiation or
    # backend/optimizer subphases to manufacture an elapsed-time estimate.
    names = {"ExecuteCompiler", "Frontend", "Backend", "Optimizer",
             "CodeGenPasses", "InstantiateFunction", "InstantiateClass"}
    return {
        event["name"].removeprefix("Total "): event["dur"] / 1e6
        for event in events
        if event.get("name", "").startswith("Total ")
        and event["name"].removeprefix("Total ") in names
    }


def select_entry(entries: list[dict], source: Path, dimension: int) -> dict:
    selected = [entry for entry in entries
                if (Path(entry["directory"]) / entry["file"]).resolve() == source.resolve()
                and f"-DAMREX_SPACEDIM={dimension}" in compile_arguments(entry)]
    if len(selected) != 1:
        raise ValueError(f"expected one {dimension}D compile owner for {source}, "
                         f"found {len(selected)}; configure the selected build first")
    return selected[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, action="append", required=True,
                        help="source path relative to pyfluid_amrex; repeat for a scan")
    parser.add_argument("--dimension", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--build-dir", type=Path, default=ROOT / "build/pyfluid_amrex")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "runs/current/compile_profiles")
    parser.add_argument("--trace", action="store_true", help="collect Clang or NVCC phase timings")
    parser.add_argument("--timeout", type=float, default=300.0,
                        help="hard deadline per source in seconds (default: 300); stop the scan on expiry")
    args = parser.parse_args()
    if not 0.0 < args.timeout < float("inf"):
        parser.error("--timeout must be finite and positive")
    entries = json.loads((args.build_dir / "compile_commands.json").read_text())
    output_dir = args.output_dir.resolve()
    sources = [(ROOT / "pyfluid_amrex" / source).resolve() for source in args.source]
    try:
        selected = [(source, select_entry(entries, source, args.dimension))
                    for source in dict.fromkeys(sources)]
        for source, _ in selected:
            source.relative_to(ROOT / "pyfluid_amrex")
    except ValueError as error:
        parser.error(str(error))
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True, check=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=ROOT,
                               capture_output=True, text=True, check=True).stdout)
    environment = os.environ.copy()
    environment["CCACHE_DISABLE"] = "1"
    rows = []
    for source, entry in selected:
        stem = output_dir / f"{args.dimension}d" / source.relative_to(ROOT / "pyfluid_amrex")
        stem.parent.mkdir(parents=True, exist_ok=True)
        output = stem.with_suffix(".o")
        trace = stem.with_suffix(".trace.json") if args.trace else None
        command = isolated_command(entry, output, trace)
        # A failed rerun must not report a stale object or trace as its result.
        output.unlink(missing_ok=True)
        if trace is not None:
            trace.unlink(missing_ok=True)
        started = datetime.now(timezone.utc).isoformat()
        start = time.perf_counter()
        timed_out = False
        try:
            with stem.with_suffix(".log").open("w") as log:
                try:
                    result = run_bounded_command(
                        command, timeout=args.timeout, cwd=Path(entry["directory"]),
                        env=environment, stdout=log,
                    )
                    returncode = result.returncode
                except subprocess.TimeoutExpired:
                    timed_out = True
                    returncode = 124
                    log.write(f"\nCompile exceeded {args.timeout:g}s; process group terminated.\n")
        except BaseException:
            output.unlink(missing_ok=True)
            raise
        elapsed = time.perf_counter() - start
        row = {
            "source": str(source.relative_to(ROOT)), "dimension": args.dimension,
            "timestamp": started, "elapsed_s": elapsed, "returncode": returncode,
            "timeout_s": args.timeout, "timed_out": timed_out,
            "object_bytes": output.stat().st_size if output.exists() else 0,
            "trace_seconds_inclusive": trace_totals(trace)
            if returncode == 0 and trace and trace.exists() else {},
            "compile_command": command, "directory": entry["directory"],
            "command_sha256": hashlib.sha256(json.dumps(compile_arguments(entry)).encode()).hexdigest(),
            "build_dir": str(args.build_dir.resolve()), "revision": revision, "dirty": dirty,
            "host": platform.platform(), "cache": "disabled",
        }
        output.unlink(missing_ok=True)  # Keep compact evidence, not another object cache.
        rows.append(row)
        (output_dir / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
        print(f"{row['source']} {elapsed:.3f}s exit={returncode} timeout={timed_out} "
              f"object={row['object_bytes']} bytes phases={row['trace_seconds_inclusive']}", flush=True)
        if returncode:
            return returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
