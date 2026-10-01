#!/usr/bin/env python3
"""Validate equilibrium perturbations, cold closure, and artifact semantics."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import re
import shutil

from validation_common import (
    ROOT,
    materialize_run_input,
    read_tsv,
    repo_relative,
    run_amrex,
    write_tsv,
)


BASE_INPUT = ROOT / "pyfluid_amrex/inputs/tov_direct_id_3d.in"
TORUS_INPUT = (
    ROOT / "pyfluid_amrex/inputs/kerr_schild_q_power_torus_evolution_3d.in"
)
RUN_ROOT = ROOT / "runs/current/initial_data_perturbation_validation"


def run_case(
    name: str,
    overrides: dict[str, str],
    ranks: int,
    timeout: int | None,
    *,
    remove_parameters: tuple[str, ...] = ("initial_data.output_file",),
    base_input: Path = BASE_INPUT,
    n_cell: str = "16 16 16",
) -> tuple[Path, list[dict[str, str]], list[dict[str, str]], str]:
    run_dir = RUN_ROOT / name
    run_dir.mkdir(parents=True, exist_ok=True)
    input_path = materialize_run_input(
        base_input,
        RUN_ROOT / f"{name}.in",
        {
            "pyfluid.run_name": name,
            "pyfluid.output_dir": repo_relative(run_dir),
            "amr.n_cell": n_cell,
            "amr.max_grid_size": "8",
            "initial_data.solve_only": "0",
            "pyfluid.max_step": "0",
            "pyfluid.scalar_int": "0",
            "pyfluid.plot_int": "-1",
            "diagnostics.matter_mode_m_max": "2",
            "diagnostics.matter_mode_origin1": "0.0",
            "diagnostics.matter_mode_origin2": "0.0",
            "diagnostics.matter_mode_origin3": "0.0",
            "diagnostics.matter_mode_axis1": "0.0",
            "diagnostics.matter_mode_axis2": "0.0",
            "diagnostics.matter_mode_axis3": "1.0",
            **overrides,
        },
        remove_parameters=remove_parameters,
    )
    completed = run_amrex(
        dim=3,
        input_path=input_path,
        ranks=ranks,
        timeout=timeout,
    )
    (run_dir / "run.log").write_text(completed.stdout)
    if completed.returncode != 0:
        raise RuntimeError(
            f"initial-data perturbation run failed; see {run_dir / 'run.log'}"
        )
    return (
        run_dir,
        read_tsv(run_dir / "scalars.tsv", require_complete=True),
        read_tsv(run_dir / "multipoles.tsv", require_complete=True),
        completed.stdout,
    )


def modes(rows: list[dict[str, str]]) -> dict[int, complex]:
    return {
        int(row["m"]): complex(float(row["real"]), float(row["imag"]))
        for row in rows
        if int(row["step"]) == 0 and row["field"] == "rest_mass"
    }


def validate(args: argparse.Namespace) -> Path:
    if args.force and RUN_ROOT.exists():
        shutil.rmtree(RUN_ROOT)
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    equilibrium_artifact = RUN_ROOT / "equilibrium_physical_id"

    baseline_dir, baseline_scalars, baseline_rows, _ = run_case(
        "baseline",
        {"initial_data.output_file": repo_relative(equilibrium_artifact)},
        args.ranks,
        args.timeout,
        remove_parameters=(),
    )
    if not equilibrium_artifact.is_dir():
        raise RuntimeError("baseline did not write its reusable physical-ID artifact")
    baseline = modes(baseline_rows)
    if set(baseline) != {0, 1, 2}:
        raise RuntimeError(f"unexpected baseline mode set {sorted(baseline)}")
    baseline_mass = baseline[0].real
    if baseline_mass <= 0.0 or abs(baseline[1]) > 2.0e-13 * baseline_mass:
        raise RuntimeError(f"baseline TOV is not axisymmetric: C1={baseline[1]}")

    pressure_dir, pressure_scalars, pressure_rows, pressure_log = run_case(
        "cold_pressure_scale",
        {
            "initial_data.input_file": repo_relative(equilibrium_artifact),
            "initial_data.perturbation": "cold_pressure_scale",
            "initial_data.perturbation_pressure_scale": "0.995",
        },
        args.ranks,
        args.timeout,
    )
    pressure = modes(pressure_rows)
    if not (0.99 * baseline_mass < pressure[0].real < baseline_mass):
        raise RuntimeError(
            "cold-pressure perturbation did not produce the expected small "
            f"mass decrease: baseline={baseline_mass}, perturbed={pressure[0].real}"
        )
    if "constraint_polish=none" not in pressure_log:
        raise RuntimeError("pressure perturbation did not report its constraint status")

    radial_dir, radial_scalars, radial_rows, radial_log = run_case(
        "density_radial_font2002",
        {
            "initial_data.input_file": repo_relative(equilibrium_artifact),
            "initial_data.perturbation": "density_radial_font2002",
            "initial_data.perturbation_amplitude": "0.01",
        },
        args.ranks,
        args.timeout,
    )
    radial = modes(radial_rows)
    if not (baseline_mass < radial[0].real < 1.02 * baseline_mass):
        raise RuntimeError(
            "radial density perturbation did not produce the expected small "
            f"mass increase: baseline={baseline_mass}, perturbed={radial[0].real}"
        )
    if abs(radial[1]) > 2.0e-13 * radial[0].real:
        raise RuntimeError(
            "radial density perturbation broke spherical symmetry: "
            f"C1={radial[1]}"
        )
    radial_amplitude_match = re.search(
        r"kind=density_radial_font2002 "
        r"maximum_fractional_density_change=([0-9.eE+-]+)",
        radial_log,
    )
    measured_radial_amplitude = (
        float(radial_amplitude_match.group(1))
        if radial_amplitude_match is not None else math.nan
    )
    if not (0.009 < measured_radial_amplitude <= 0.01) \
            or "constraint_polish=none" not in radial_log:
        raise RuntimeError(
            "radial density perturbation did not report its amplitude and "
            "constraint status"
        )

    density_artifact = RUN_ROOT / "density_mode_requested_output"
    density_dir, density_scalars, density_rows, density_log = run_case(
        "density_m1",
        {
            "initial_data.input_file": repo_relative(equilibrium_artifact),
            "initial_data.output_file": repo_relative(density_artifact),
            "initial_data.perturbation": "density_azimuthal_espino2019",
            "initial_data.perturbation_amplitude": "0.005",
            "initial_data.perturbation_mode": "1",
            "initial_data.perturbation_phase": "0.0",
        },
        args.ranks,
        args.timeout,
        remove_parameters=(),
    )
    density = modes(density_rows)
    normalized_m1 = abs(density[1]) / density[0].real
    if normalized_m1 <= 1.0e-4:
        raise RuntimeError(f"m=1 seed did not excite C1: |C1|/C0={normalized_m1}")
    if "constraint_polish=none" not in density_log:
        raise RuntimeError("density perturbation did not report its constraint status")

    reload_dir, reload_scalars, reload_rows, _ = run_case(
        "requested_output_reload",
        {
            "initial_data.input_file": repo_relative(density_artifact),
            "initial_data.perturbation": "none",
        },
        args.ranks,
        args.timeout,
    )
    reloaded = modes(reload_rows)
    artifact_mode_error = abs(reloaded[1] - baseline[1]) / baseline_mass
    artifact_mass_error = abs(reloaded[0].real - baseline_mass) / baseline_mass
    if artifact_mode_error > 2.0e-13 or artifact_mass_error > 2.0e-13:
        raise RuntimeError(
            "evolution-start perturbation leaked into the reusable equilibrium "
            f"artifact: mode_error={artifact_mode_error:.3e}, "
            f"mass_error={artifact_mass_error:.3e}"
        )

    disk_dir, disk_scalars, disk_rows, disk_log = run_case(
        "equatorially_reflected_disk_density_m1",
        {
            "geometry.prob_lo": "-32.0 -32.0 0.0",
            "geometry.prob_hi": "32.0 32.0 32.0",
            "amr.max_level": "0",
            "amr.max_grid_size": "16",
            # This coarse t=0 fixture tests disk modes, not a resolved horizon.
            # Keep a regular complete grid instead of inheriting live excision.
            "spacetime.black_hole_interior": "filled",
            "spacetime.black_hole_matter": "evolve",
            "horizon.enabled": "0",
            "initial_data.perturbation": "density_azimuthal_espino2019",
            "initial_data.perturbation_amplitude": "0.01",
            "initial_data.perturbation_mode": "1",
            "initial_data.perturbation_phase": "0.0",
            "initial_data.perturbation_radius": "26.0",
            "pyfluid.bc_zlo": "symmetry",
            "pyfluid.scalar_diagnostics": "standard,modes",
        },
        args.ranks,
        args.timeout,
        remove_parameters=(
            "amr.ref_ratio",
            "initial_data.input_file",
            "initial_data.output_file",
        ),
        base_input=TORUS_INPUT,
        n_cell="32 32 16",
    )
    disk = modes(disk_rows)
    disk_normalized_m1 = abs(disk[1]) / disk[0].real
    if disk_normalized_m1 <= 1.0e-4:
        raise RuntimeError(
            "disk m=1 seed did not excite C1: "
            f"|C1|/C0={disk_normalized_m1}"
        )
    if (
        "kind=density_azimuthal_espino2019" not in disk_log
        or "constraint_polish=none" not in disk_log
    ):
        raise RuntimeError(
            "disk density perturbation did not report its kind and "
            "constraint status"
        )

    result_path = RUN_ROOT / "initial_data_perturbations.tsv"
    write_tsv(
        result_path,
        [{
            "status": "PASS",
            "ranks": args.ranks,
            "baseline_mass": baseline_mass,
            "pressure_scaled_mass_ratio": pressure[0].real / baseline_mass,
            "radial_density_mass_ratio": radial[0].real / baseline_mass,
            "density_m1_normalized_amplitude": normalized_m1,
            "disk_density_m1_normalized_amplitude": disk_normalized_m1,
            "artifact_mass_relative_error": artifact_mass_error,
            "artifact_m1_relative_error": artifact_mode_error,
            "baseline_run_dir": repo_relative(baseline_dir),
            "pressure_run_dir": repo_relative(pressure_dir),
            "radial_run_dir": repo_relative(radial_dir),
            "density_run_dir": repo_relative(density_dir),
            "reload_run_dir": repo_relative(reload_dir),
            "disk_run_dir": repo_relative(disk_dir),
            "baseline_scalar_rows": len(baseline_scalars),
            "pressure_scalar_rows": len(pressure_scalars),
            "radial_scalar_rows": len(radial_scalars),
            "density_scalar_rows": len(density_scalars),
            "reload_scalar_rows": len(reload_scalars),
            "disk_scalar_rows": len(disk_scalars),
        }],
        [
            "status",
            "ranks",
            "baseline_mass",
            "pressure_scaled_mass_ratio",
            "radial_density_mass_ratio",
            "density_m1_normalized_amplitude",
            "disk_density_m1_normalized_amplitude",
            "artifact_mass_relative_error",
            "artifact_m1_relative_error",
            "baseline_run_dir",
            "pressure_run_dir",
            "radial_run_dir",
            "density_run_dir",
            "reload_run_dir",
            "disk_run_dir",
            "baseline_scalar_rows",
            "pressure_scalar_rows",
            "radial_scalar_rows",
            "density_scalar_rows",
            "reload_scalar_rows",
            "disk_scalar_rows",
        ],
    )
    return result_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--ranks", type=int, default=3)
    parser.add_argument("--timeout", type=int)
    args = parser.parse_args()
    result = validate(args)
    print(f"PASS initial-data perturbations: {repo_relative(result)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
