#!/usr/bin/env python3
"""Validate paired stable-TOV and collapse-to-horizon evolutions."""

from __future__ import annotations

import argparse
import math
import re
import shutil
from pathlib import Path

from physical_id_metadata import read_physical_id_metadata
from validation_common import (
    OPEN_POLICY,
    ROOT,
    append_validation_history,
    materialize_run_input,
    read_tsv,
    repo_relative,
    run_amrex,
    scalar_validation_row,
    write_tsv,
)


BASE_INPUT = ROOT / "pyfluid_amrex/inputs/tov_collapse_3d_amr.in"
SOURCE_INPUT = (
    ROOT / "pyfluid_amrex/inputs/tov_collapse_xcts_spherical_id_2d.in"
)
RUN_ROOT = ROOT / "runs/current/tov_collapse"
RESULTS = RUN_ROOT / "tov_collapse_validation.tsv"
HISTORY = ROOT / "runs/current/validation_history.tsv"
RECOVERY = re.compile(
    r"Relativistic recovery c2p_fallbacks=(\d+) atmosphere_resets=(\d+)"
)
CONSTRAINT_FIELDS = (
    "z4c_hamiltonian_linf",
    "z4c_hamiltonian_l2",
    "z4c_momentum_linf",
    "z4c_z_linf",
    "z4c_theta_linf",
    "z4c_log_conformal_determinant_linf",
    "z4c_trace_a_tilde_linf",
)
STABLE_CENTRAL_DENSITY = "1.28e-3"
COLLAPSE_CENTRAL_DENSITY = "8.0e-3"
COLLAPSE_EQUILIBRIUM_ADM_MASS = 1.44729432922
COLLAPSE_EQUILIBRIUM_BARYONIC_MASS = 1.53498577255


def source_run_name(central_density: str) -> str:
    return (
        "tov_collapse_control_stable_xcts_spherical_id_2d"
        if central_density == STABLE_CENTRAL_DENSITY
        else "tov_collapse_xcts_spherical_id_2d"
    )


def prepare_source_artifact(
    central_density: str,
    *,
    ranks: int,
    timeout: int | None,
    force: bool,
    machine: str | None,
) -> Path:
    name = source_run_name(central_density)
    run_dir = RUN_ROOT / name
    artifact = run_dir / "physical_id"
    required = (
        artifact / "Header",
        artifact / "Header.meta",
        artifact / "physical_fluid.h5",
        artifact / "adm_geometry.h5",
    )
    if not force and all(path.is_file() for path in required):
        validate_source_artifact(artifact, central_density)
        return artifact
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    input_path = materialize_run_input(
        SOURCE_INPUT,
        run_dir / "input.in",
        {
            "pyfluid.run_name": name,
            "pyfluid.output_dir": repo_relative(run_dir),
            "pyfluid.tov_central_density": central_density,
            "initial_data.output_file": repo_relative(artifact),
            "initial_data.perturbation": (
                "none"
                if central_density == STABLE_CENTRAL_DENSITY
                else "density_radial_font2002"
            ),
        },
    )
    completed = run_amrex(
        dim=2,
        input_path=input_path,
        ranks=ranks,
        timeout=timeout,
        machine=machine,
    )
    (run_dir / "run.log").write_text(completed.stdout)
    if completed.returncode != 0:
        raise RuntimeError(
            f"{name} source solve failed; see "
            f"{repo_relative(run_dir / 'run.log')}"
        )
    if not all(path.is_file() for path in required):
        raise RuntimeError(f"{name} did not produce a complete physical ID")
    if "Evolution preflight C2P/P2C:" not in completed.stdout:
        raise RuntimeError(f"{name} omitted its C2P/P2C handoff preflight")
    validate_source_artifact(artifact, central_density)
    return artifact


def validate_source_artifact(artifact: Path, central_density: str) -> None:
    metadata = read_physical_id_metadata(artifact)
    required = ("adm_mass", "baryonic_mass", "maximum_density")
    if any(key not in metadata for key in required):
        raise RuntimeError(
            f"{repo_relative(artifact)} lacks collapse-source diagnostics"
        )
    values = {key: float(metadata[key]) for key in required}
    if not all(math.isfinite(value) and value > 0.0 for value in values.values()):
        raise RuntimeError(
            f"{repo_relative(artifact)} has invalid collapse-source diagnostics"
        )
    if central_density != COLLAPSE_CENTRAL_DENSITY:
        return
    if values["adm_mass"] <= COLLAPSE_EQUILIBRIUM_ADM_MASS \
            or values["baryonic_mass"] <= COLLAPSE_EQUILIBRIUM_BARYONIC_MASS:
        raise RuntimeError(
            "controlled-collapse source did not retain the positive Font "
            "density perturbation through its constraint polish: "
            f"ADM={values['adm_mass']:.12g} "
            f"Mb={values['baryonic_mass']:.12g}"
        )


def case_overrides(
    name: str,
    central_density: str,
    level: str,
    source_artifact: Path,
) -> dict[str, str]:
    overrides = {
        "pyfluid.run_name": name,
        "pyfluid.output_dir": repo_relative(RUN_ROOT / name),
        "pyfluid.tov_central_density": central_density,
        "initial_data.input_file": repo_relative(source_artifact),
    }
    if "control_stable" in name:
        overrides["initial_data.perturbation"] = "none"
    if level == "smoke":
        overrides.update(
            {
                "geometry.prob_hi": "24.0 24.0 24.0",
                "amr.n_cell": "24 24 24",
                "amr.max_level": "2",
                "amr.ref_ratio": "2 2",
                "amr.initial_refine_levels": "1",
                "amr.initial_refine_fixed_levels": "0",
                "amr.min_feature_length": "0.5",
                "amr.level_cell_budget": "250000",
                "amr.total_cell_budget": "500000",
                "pyfluid.final_time": "-1",
                "pyfluid.max_step": "2",
                "pyfluid.plot_int": "-1",
                "pyfluid.aux_gf_int": "-1",
                "pyfluid.checkpoint_int": "-1",
            }
        )
    elif level == "pilot":
        overrides.update(
            {
                "geometry.prob_hi": "32.0 32.0 32.0",
                "amr.n_cell": "32 32 32",
                "amr.max_level": "3",
                "amr.ref_ratio": "2 2 2",
                "amr.initial_refine_levels": "2",
                "amr.initial_refine_fixed_levels": "1",
                "amr.min_feature_length": "0.25",
                "amr.initial_refine_box_hi": "8.0 8.0 8.0",
                "amr.level_cell_budget": "250000",
                "amr.total_cell_budget": "750000",
                "pyfluid.final_time": "75.0",
                "pyfluid.max_step": "1500",
                "horizon.radial_points": "48",
                "horizon.polar_points": "12",
                "pyfluid.plot_int": "-1",
                "pyfluid.aux_gf_int": "-1",
                "pyfluid.checkpoint_int": "256",
            }
        )
    return overrides


def run_case(
    name: str,
    central_density: str,
    *,
    level: str,
    ranks: int,
    timeout: int | None,
    force: bool,
    machine: str | None,
) -> tuple[dict[str, object], dict[str, object]]:
    source_artifact = prepare_source_artifact(
        central_density,
        ranks=ranks,
        timeout=timeout,
        force=force,
        machine=machine,
    )
    run_dir = RUN_ROOT / name
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    remove_parameters = (
        ("pyfluid.aux_gfs",) if level in ("smoke", "pilot") else ()
    )
    input_path = materialize_run_input(
        BASE_INPUT,
        run_dir / "input.in",
        case_overrides(name, central_density, level, source_artifact),
        remove_parameters=remove_parameters,
    )
    completed = run_amrex(
        dim=3,
        input_path=input_path,
        ranks=ranks,
        timeout=timeout,
        machine=machine,
    )
    (run_dir / "run.log").write_text(completed.stdout)
    if completed.returncode != 0:
        raise RuntimeError(
            f"{name} failed; see {repo_relative(run_dir / 'run.log')}"
        )
    scalar = scalar_validation_row(
        run_dir,
        case=name,
        input_file=run_dir / "run.in",
        checks=OPEN_POLICY,
        mhd=False,
        dim=3,
        ranks=ranks,
        kind="collapse",
    )
    if scalar["status"] != "PASS":
        raise RuntimeError(f"{name} scalar validation failed: {scalar['reason']}")
    rows = read_tsv(run_dir / "scalars.tsv", require_complete=True)
    if not rows:
        raise RuntimeError(f"{name} did not write scalar diagnostics")
    for scalar_row in rows:
        for field in CONSTRAINT_FIELDS:
            if field not in scalar_row or not math.isfinite(
                float(scalar_row[field])
            ):
                raise RuntimeError(f"{name} has nonfinite {field}")
    recovery = RECOVERY.findall(completed.stdout)
    fallbacks, resets = (
        tuple(int(value) for value in recovery[-1]) if recovery else (-1, -1)
    )
    if fallbacks != 0 or resets != 0:
        raise RuntimeError(
            f"{name} used C2P fallback/reset: {fallbacks}/{resets}"
        )
    found_rows = [
        row for row in rows if float(row["black_hole_horizon_found"]) == 1.0
    ]
    if level == "smoke" and found_rows:
        raise RuntimeError(f"{name} fabricated a horizon during its smoke run")
    if level in ("pilot", "formation") \
            and "target_unstable" in name and not found_rows:
        raise RuntimeError(f"{name} did not form an apparent horizon")
    if level in ("pilot", "formation") \
            and "control_stable" in name and found_rows:
        raise RuntimeError(f"{name} stable control formed an apparent horizon")
    formation_time = (
        float(found_rows[0]["time"]) if found_rows else math.nan
    )
    final = rows[-1]
    initial = rows[0]
    result = {
        "case": name,
        "status": "PASS",
        "level": level,
        "steps": int(final["step"]),
        "final_time": float(final["time"]),
        "central_density": float(central_density),
        "initial_max_rho": float(initial["max_rho"]),
        "final_max_rho": float(final["max_rho"]),
        "final_min_lapse": float(final["min_lapse"]),
        "horizon_found": int(bool(found_rows)),
        "horizon_formation_time": formation_time,
        "final_horizon_mass": float(
            final["black_hole_horizon_coordinate_christodoulou_mass"]
        ),
        "final_hamiltonian_l2": float(final["z4c_hamiltonian_l2"]),
        "c2p_fallbacks": fallbacks,
        "atmosphere_resets": resets,
        "run_dir": repo_relative(run_dir),
    }
    return result, scalar


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--level", choices=("smoke", "pilot", "formation"), default="smoke"
    )
    parser.add_argument(
        "--only", choices=("stable", "unstable", "both"), default="both"
    )
    parser.add_argument("--ranks", type=int, default=3)
    parser.add_argument("--machine")
    parser.add_argument("--timeout", type=int)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    if args.force and RESULTS.exists():
        RESULTS.unlink()

    selected: list[tuple[str, str]] = []
    if args.only in ("stable", "both"):
        selected.append(("tov_collapse_control_stable", STABLE_CENTRAL_DENSITY))
    if args.only in ("unstable", "both"):
        selected.append(("tov_collapse_target_unstable", COLLAPSE_CENTRAL_DENSITY))

    results: list[dict[str, object]] = []
    scalar_rows: list[dict[str, object]] = []
    for base_name, central_density in selected:
        name = f"{base_name}_{args.level}"
        result, scalar = run_case(
            name,
            central_density,
            level=args.level,
            ranks=args.ranks,
            timeout=args.timeout,
            force=args.force,
            machine=args.machine,
        )
        results.append(result)
        scalar_rows.append(scalar)
        print(
            f"PASS {name}: t={result['final_time']:.6g} "
            f"max_rho={result['final_max_rho']:.6e} "
            f"min_lapse={result['final_min_lapse']:.6e} "
            f"horizon={result['horizon_found']}"
        )
    write_tsv(RESULTS, results, list(results[0]))
    append_validation_history(HISTORY, scalar_rows)
    print(f"wrote {repo_relative(RESULTS)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
