#!/usr/bin/env python3
"""Run one AMReX case and render standard movie outputs.

This is intentionally a thin wrapper around the existing run/report and movie
tools so movie workflows stay one-command and run-local.
"""

from __future__ import annotations

import argparse
import datetime as dt
import subprocess
import sys
from pathlib import Path

from validation_common import ROOT, read_key_values

RUN_CASE = ROOT / "pyfluid_amrex/tools/run_case.py"
MOVIE = ROOT / "pyfluid_amrex/tools/plot_amrex_boxed_movie.py"


def run_command(command: list[str]) -> None:
    completed = subprocess.run(command, cwd=ROOT, check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def append_report(run_dir: Path, rows: list[tuple[str, str]]) -> None:
    report = run_dir / "movie_report.tsv"
    new_file = not report.exists()
    with report.open("a", encoding="utf-8") as f:
        if new_file:
            f.write("timestamp\tkind\tpath\n")
        stamp = dt.datetime.now(dt.UTC).isoformat(timespec="seconds")
        for kind, path in rows:
            f.write(f"{stamp}\t{kind}\t{path}\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--ranks", type=int, default=3)
    parser.add_argument("--components", default="pressure", help="Comma-separated components to render")
    parser.add_argument("--title", default="", help="Movie title; default is a compact run/config title")
    parser.add_argument("--curvilinear-view", default="auto", choices=("auto", "native", "r-theta", "r-phi", "multipatch"))
    parser.add_argument("--box-levels", default="", help="Comma-separated AMR levels to outline")
    parser.add_argument("--box-alpha", type=float, default=0.42)
    parser.add_argument("--box-linewidth", type=float, default=0.5)
    parser.add_argument("--target-frames", type=int, default=120)
    parser.add_argument("--duration-ms", type=int, default=70)
    parser.add_argument("--max-pixels", type=int, default=900)
    parser.add_argument("--boxes-only", action="store_true")
    parser.add_argument("--nobox-only", action="store_true")
    args = parser.parse_args()

    input_path = args.input if args.input.is_absolute() else ROOT / args.input
    parameters = read_key_values(input_path)
    try:
        run_name = parameters["pyfluid.run_name"]
        output_dir_value = parameters["pyfluid.output_dir"]
    except KeyError as exc:
        raise SystemExit(f"movie input must define {exc.args[0]}") from exc
    run_dir = ROOT / output_dir_value
    plot_dir = run_dir / "stateGF"

    run_command([
        sys.executable,
        str(RUN_CASE),
        str(input_path),
        "--ranks",
        str(args.ranks),
    ])

    title = args.title
    if not title:
        title = f"{run_name} AMR movie"

    rendered: list[tuple[str, str]] = []
    components = [item.strip() for item in args.components.split(",") if item.strip()]
    for component in components:
        variants: list[tuple[str, list[str]]] = []
        if not args.boxes_only:
            variants.append(("nobox", ["--hide-boxes"]))
        if not args.nobox_only:
            box_args: list[str] = []
            if args.box_levels:
                box_args.extend(["--box-levels", args.box_levels])
            box_args.extend(["--box-alpha", str(args.box_alpha), "--box-linewidth", str(args.box_linewidth)])
            variants.append(("boxes", box_args))

        for tag, variant_args in variants:
            gif_name = f"{run_dir.name}_{component}_{tag}.gif"
            run_command([
                sys.executable,
                str(MOVIE),
                "--plot-dir",
                str(plot_dir),
                "--output-dir",
                str(run_dir),
                "--gif",
                str(run_dir / gif_name),
                "--component",
                component,
                "--title",
                title,
                "--curvilinear-view",
                args.curvilinear_view,
                "--duration-ms",
                str(args.duration_ms),
                "--target-frames",
                str(args.target_frames),
                "--max-pixels",
                str(args.max_pixels),
                *variant_args,
            ])
            rendered.append((f"{component}_{tag}", str(run_dir / gif_name)))

    append_report(run_dir, [("run_dir", str(run_dir)), *rendered])
    print(f"PASS movie_case run_dir={run_dir}")
    for kind, path in rendered:
        print(f"{kind}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
