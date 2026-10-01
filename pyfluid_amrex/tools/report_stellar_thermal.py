#!/usr/bin/env python3
"""Report thermal pressure excess in a single-level, single-polytrope TOV run.

Uses the solver-authored thermal_entropy_proxy, not a second primitive recovery.
For equal cold/thermal gamma, K_th/K_cold = p_th/p_cold. This is not temperature.
Regions use the initial coordinate surface radius and coordinate-volume means;
they are fixed Eulerian shells, not tracked fluid elements or proper-mass means.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from plotfile_reader import (
    level_count, parse_plot_bounds, parse_plot_header, read_level_state,
    sorted_aux_plotfiles,
)
from validation_common import read_key_values, write_tsv


def thermal_surface_rows(
    run_dir: Path, surface_radius: float, initial_run_dir: Path | None = None,
) -> list[dict]:
    params = read_key_values(run_dir / "run.in")
    if (params.get("pyfluid.eos") != "hybrid_polytrope"
            or float(params["pyfluid.polytropic_gamma"]) != float(params["pyfluid.thermal_gamma"])):
        raise ValueError("p_th/p_cold from K_th requires hybrid_polytrope with equal gammas")
    if params.get("pyfluid.ic") != "tov" or params.get("mesh.coordinate", "cartesian") != "cartesian":
        raise ValueError("this fixed spherical-shell probe expects Cartesian TOV initial data")
    cold_k = float(params["pyfluid.polytropic_k"])
    if not np.isfinite(surface_radius) or surface_radius <= 0 or cold_k <= 0:
        raise ValueError("surface radius and cold polytropic K must be positive")
    rows = []
    baseline = None
    baseline_bounds = None
    plotfiles = sorted_aux_plotfiles(run_dir, require=True)
    if initial_run_dir is not None:
        initial_params = read_key_values(initial_run_dir / "run.in")
        model_prefixes = ("geometry.", "mesh.", "initial_data.atmosphere",
                          "pyfluid.tov_", "pyfluid.polytropic", "pyfluid.thermal_gamma",
                          "pyfluid.eos", "pyfluid.rho_floor", "pyfluid.pressure_floor")
        for key in set(params) | set(initial_params):
            if not key.startswith(model_prefixes):
                continue
            if initial_params.get(key) != params.get(key):
                raise ValueError(f"initial reference differs in {key}")
        plotfiles = sorted_aux_plotfiles(initial_run_dir, require=True) + plotfiles
    for plotfile in plotfiles:
        names, ndim, time = parse_plot_header(plotfile)
        if "thermal_entropy_proxy" not in names:
            continue
        if level_count(plotfile) != 1:
            raise ValueError("this shell probe requires a uniform single-level output")
        ratio = read_level_state(plotfile, names, ndim, 0)[names.index("thermal_entropy_proxy")] / cold_k
        bounds = parse_plot_bounds(plotfile)
        if baseline is None:
            if time != 0.0:
                raise ValueError("initial t=0 thermal output is required")
            baseline = ratio.copy()
            baseline_bounds = bounds
        if ratio.shape != baseline.shape or bounds != baseline_bounds:
            raise ValueError("mesh changed within the fixed-shell comparison")
        if not np.all(np.isfinite(ratio)):
            raise ValueError("nonfinite thermal field")
        axes = [np.linspace(lo, hi, count, endpoint=False) + (hi-lo)/(2*count)
                for lo, hi, count in zip(*bounds, ratio.shape)]
        radius = np.sqrt(sum(axis**2 for axis in np.meshgrid(*axes, indexing="ij")))
        for region, lower, upper in (("core", 0, 0.8), ("surface", 0.8, 1.0),
                                     ("exterior", 1.0, 1.2)):
            mask = (radius >= lower*surface_radius) & (radius < upper*surface_radius)
            if not np.any(mask):
                raise ValueError(f"mesh does not sample {region}")
            values = ratio[mask]
            change = (ratio-baseline)[mask]
            rows.append({
                "time": time, "region": region, "cells": int(np.count_nonzero(mask)),
                "thermal_pressure_ratio_mean": float(np.mean(values)),
                "thermal_pressure_ratio_p95": float(np.quantile(values, .95)),
                "thermal_pressure_ratio_min": float(np.min(values)),
                "thermal_pressure_ratio_max": float(np.max(values)),
                "thermal_pressure_ratio_change_mean": float(np.mean(change)),
                "thermal_pressure_ratio_change_max": float(np.max(change)),
            })
    if not rows:
        raise ValueError("no thermal_entropy_proxy output found")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--surface-radius", type=float, required=True,
                        help="initial coordinate (isotropic) radius, not areal radius")
    parser.add_argument("--initial-run-dir", type=Path,
                        help="matching t=0-only run when evolution cadence skips t=0")
    args = parser.parse_args()
    rows = thermal_surface_rows(args.run_dir, args.surface_radius, args.initial_run_dir)
    output = args.run_dir / "stellar_thermal.tsv"
    write_tsv(output, rows, list(rows[0]))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
