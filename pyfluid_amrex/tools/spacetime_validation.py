"""Shared spacetime validation runs and independent analytic postprocessing.

These expressions do not call the generated evolution equations. Components
are measured in the caller's orthonormal frame, including in a curved chart.
"""

from __future__ import annotations

import argparse
import math
import re
import shutil
from pathlib import Path

import numpy as np

from plotfile_reader import plotfile_step
from validation_common import (
    ROOT, materialize_run_input, read_key_values, repo_relative,
    run_amrex, scalar_validation_row,
)

RUN_ROOT = ROOT / "runs/current/z4c_validation"
GAUGE_WAVE_INPUT = ROOT / "pyfluid_amrex/inputs/z4c_gauge_wave_1d.in"


def gauge_wave_exact(
    x: np.ndarray,
    time: float,
    *,
    amplitude: float,
    wave_number: float,
    normal: tuple[np.ndarray | float, np.ndarray | float, np.ndarray | float],
) -> dict[str, np.ndarray]:
    """Conformal variables for ds²=H(-dt²+dx²)+dy²+dz².

    ``normal`` contains the local components of the Cartesian x unit vector.
    Lambda is the contracted connection *difference* from flat space, so it
    transforms as a vector. K_hat equals physical K on this exact Theta=0
    solution. This oracle applies to Z4c in Cartesian coordinates and fCCZ4
    in any static orthonormal frame of the same flat reference geometry.
    """
    frequency = 2.0 * math.pi * wave_number
    phase = frequency * (x - time)
    metric_factor = 1.0 - amplitude * np.sin(phase)
    lapse = np.sqrt(metric_factor)
    chi = metric_factor ** (-1.0 / 3.0)
    physical_curvature = -0.5 * amplitude * frequency * np.cos(phase) / lapse
    trace_curvature = physical_curvature / metric_factor
    transverse_curvature = -chi * trace_curvature / 3.0
    longitudinal_curvature = chi * (
        physical_curvature - metric_factor * trace_curvature / 3.0
    )
    connection = (
        -2.0 * amplitude * frequency * np.cos(phase)
        / (3.0 * metric_factor ** (5.0 / 3.0))
    )
    exact = {
        "chi": chi,
        "lapse": lapse,
        "k_hat": trace_curvature,
        "theta": np.zeros_like(x),
    }
    for first in range(3):
        exact[f"gamma_tilde{first + 1}"] = connection * normal[first]
        exact[f"shift{first + 1}"] = np.zeros_like(x)
        for second in range(first, 3):
            suffix = f"{first + 1}{second + 1}"
            projection = normal[first] * normal[second]
            diagonal = float(first == second)
            exact[f"gamma_tilde{suffix}"] = (
                chi * diagonal + chi * (metric_factor - 1.0) * projection
            )
            exact[f"a_tilde{suffix}"] = (
                transverse_curvature * diagonal
                + (longitudinal_curvature - transverse_curvature) * projection
            )
    return exact


def validation_run_root(args: argparse.Namespace) -> Path:
    formulation = getattr(args, "evolution_formulation", "z4c")
    root = RUN_ROOT if formulation == "z4c" else RUN_ROOT / formulation
    order = getattr(args, "derivative_order", 4)
    root = root if order == 4 else root / f"order_{order}"
    prolongation = getattr(args, "prolongation_order", None)
    return root if prolongation is None else root / f"prolongation_{prolongation}"


def run_dimension_case(
    name: str,
    base_input: Path,
    overrides: dict[str, str],
    args: argparse.Namespace,
    *,
    dim: int,
    checks: object,
    mhd: bool,
    kind: str = "analytic",
    remove_parameters: tuple[str, ...] = (),
    dependencies: tuple[Path, ...] = (),
) -> tuple[Path, dict[str, object]]:
    evolution_formulation = getattr(args, "evolution_formulation", "z4c")
    run_dir = validation_run_root(args) / name
    run_log = run_dir / "run.log"
    zero_step_completed = (
        overrides.get("pyfluid.max_step") == "0"
        and run_log.is_file()
        and re.search(
            r"^AMReX \([^\n]+\) finalized$",
            run_log.read_text(),
            re.MULTILINE,
        ) is not None
    )
    dependency_changed = bool(dependencies) and (
        not run_log.is_file()
        or any(
            path.stat().st_mtime > run_log.stat().st_mtime
            for path in dependencies
        )
    )
    run_needed = (
        args.force
        or dependency_changed
        or (
            latest_spacetime_plotfile(run_dir) is None
            and not zero_step_completed
        )
    )
    if run_needed and run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    base_evolution = read_key_values(base_input).get("spacetime.evolution", "fixed")
    evolution = overrides.get("spacetime.evolution", base_evolution)
    accuracy_overrides = {
        "spacetime.derivative_order": str(getattr(args, "derivative_order", 4)),
    }
    prolongation = getattr(args, "prolongation_order", None)
    if prolongation is not None:
        accuracy_overrides["spacetime.prolongation_order"] = str(prolongation)
    input_path = materialize_run_input(
        base_input,
        run_dir / "input.in",
        {
            "pyfluid.run_name": name,
            "pyfluid.output_dir": repo_relative(run_dir),
            "spacetime.formulation": evolution_formulation if evolution == "evolved" else "z4c",
            **accuracy_overrides,
            "pyfluid.aux_gfs": "none",
            "pyfluid.aux_gf_compute_int": "-1",
            "pyfluid.aux_gf_int": "-1",
            **overrides,
        },
        remove_parameters=remove_parameters,
    )
    if run_needed:
        completed = run_amrex(
            dim=dim,
            input_path=input_path,
            ranks=args.ranks,
            timeout=args.timeout,
        )
        (run_dir / "run.log").write_text(completed.stdout)
        if completed.returncode != 0:
            raise RuntimeError(f"{name} failed; see {repo_relative(run_dir / 'run.log')}")
    scalar = scalar_validation_row(
        run_dir,
        case=name,
        input_file=run_dir / "run.in",
        checks=checks,
        mhd=mhd,
        dim=dim,
        ranks=args.ranks,
        kind=kind,
    )
    if scalar["status"] != "PASS":
        raise RuntimeError(f"{name} scalar validation failed: {scalar['reason']}")
    return run_dir, scalar


def spacetime_plotfiles(run_dir: Path) -> list[Path]:
    files = list((run_dir / "stateGF").glob("spacetime_state*.h5"))
    return sorted(files, key=plotfile_step)


def latest_spacetime_plotfile(run_dir: Path) -> Path | None:
    files = spacetime_plotfiles(run_dir)
    return files[-1] if files else None


def latest_fluid_plotfile(run_dir: Path) -> Path | None:
    files = list((run_dir / "stateGF").glob("stateGF*.h5"))
    return max(files, key=plotfile_step) if files else None
