#!/usr/bin/env python3
"""Validate Kerr-Schild full-grid evolution and reflection contracts."""

from __future__ import annotations

import argparse
import math
import re
import shutil
from pathlib import Path

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


BASE_INPUT = ROOT / "pyfluid_amrex/inputs/kerr_schild_filled_evolution_3d.in"
RUN_ROOT = ROOT / "runs/current/black_hole_validation"
RESULTS = RUN_ROOT / "black_hole_validation.tsv"
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
    "z4c_hamiltonian_bh_exterior_linf",
    "z4c_hamiltonian_bh_exterior_l2",
    "z4c_momentum_bh_exterior_linf",
)


def run_case(
    name: str,
    overrides: dict[str, str],
    *,
    ranks: int,
    timeout: int | None,
) -> tuple[dict[str, object], dict[str, object]]:
    run_dir = RUN_ROOT / name
    run_dir.mkdir(parents=True, exist_ok=True)
    input_path = materialize_run_input(
        BASE_INPUT,
        run_dir / "input.in",
        {
            "pyfluid.run_name": name,
            "pyfluid.output_dir": repo_relative(run_dir),
            **overrides,
        },
    )
    completed = run_amrex(
        dim=3,
        input_path=input_path,
        ranks=ranks,
        timeout=timeout,
    )
    (run_dir / "run.log").write_text(completed.stdout)
    if completed.returncode != 0:
        raise RuntimeError(f"{name} failed; see {repo_relative(run_dir / 'run.log')}")
    scalar = scalar_validation_row(
        run_dir,
        case=name,
        input_file=run_dir / "run.in",
        checks=OPEN_POLICY,
        mhd=False,
        dim=3,
        ranks=ranks,
        kind="black_hole",
    )
    if scalar["status"] != "PASS":
        raise RuntimeError(f"{name} scalar validation failed: {scalar['reason']}")
    rows = read_tsv(run_dir / "scalars.tsv", require_complete=True)
    if not rows or int(rows[-1]["step"]) != int(overrides.get("pyfluid.max_step", "3")):
        raise RuntimeError(f"{name} has an incomplete scalar history")
    for row in rows:
        for field in CONSTRAINT_FIELDS:
            if field not in row or not math.isfinite(float(row[field])):
                raise RuntimeError(f"{name} has nonfinite {field}")
    recovery = RECOVERY.findall(completed.stdout)
    c2p_fallbacks, atmosphere_resets = (
        tuple(int(value) for value in recovery[-1]) if recovery else (-1, -1)
    )
    if c2p_fallbacks != 0 or atmosphere_resets != 0:
        raise RuntimeError(
            f"{name} used C2P fallback/reset: {c2p_fallbacks}/{atmosphere_resets}"
        )
    final = rows[-1]
    return {
        "case": name,
        "status": "PASS",
        "steps": int(final["step"]),
        "hamiltonian_l2": float(final["z4c_hamiltonian_l2"]),
        "hamiltonian_linf": float(final["z4c_hamiltonian_linf"]),
        "momentum_linf": float(final["z4c_momentum_linf"]),
        "exterior_hamiltonian_l2": float(
            final.get("z4c_hamiltonian_bh_exterior_l2", math.nan)
        ),
        "exterior_hamiltonian_linf": float(
            final.get("z4c_hamiltonian_bh_exterior_linf", math.nan)
        ),
        "exterior_momentum_linf": float(
            final.get("z4c_momentum_bh_exterior_linf", math.nan)
        ),
        "horizon_radius": float(final.get("black_hole_horizon_radius", math.nan)),
        "min_lapse": float(final.get("min_lapse", math.nan)),
        "max_rho": float(final.get("max_rho", math.nan)),
        "symmetry_lapse_gap": math.nan,
        "symmetry_gap_ratio": math.nan,
        "c2p_fallbacks": c2p_fallbacks,
        "atmosphere_resets": atmosphere_resets,
        "run_dir": repo_relative(run_dir),
    }, scalar


def require_matched_symmetry_extrema(
    reference: dict[str, object],
    reduced: dict[str, object],
    label: str,
) -> None:
    for field in (
        "hamiltonian_linf",
        "momentum_linf",
        "exterior_hamiltonian_linf",
        "exterior_momentum_linf",
    ):
        reference_value = float(reference[field])
        reduced_value = float(reduced[field])
        scale = max(1.0, abs(reference_value), abs(reduced_value))
        if abs(reference_value - reduced_value) > 2.0e-12 * scale:
            raise RuntimeError(
                f"{label} changed {field}: "
                f"full={reference_value:.16e}, reduced={reduced_value:.16e}"
            )


def validate_negative_reflection(*, ranks: int, timeout: int | None) -> dict[str, object]:
    name = "kerr_z_spin_invalid_x_reflection"
    run_dir = RUN_ROOT / name
    run_dir.mkdir(parents=True, exist_ok=True)
    input_path = materialize_run_input(
        BASE_INPUT,
        run_dir / "input.in",
        {
            "pyfluid.run_name": name,
            "pyfluid.output_dir": repo_relative(run_dir),
            "geometry.prob_lo": "0.0 -8.0 -8.0",
            "geometry.prob_hi": "8.0 8.0 8.0",
            "amr.n_cell": "16 32 32",
            "pyfluid.bc_xlo": "symmetry",
            "pyfluid.max_step": "1",
            "pyfluid.plot_int": "-1",
        },
    )
    completed = run_amrex(
        dim=3,
        input_path=input_path,
        ranks=ranks,
        timeout=timeout,
    )
    (run_dir / "run.log").write_text(completed.stdout)
    expected = "a Kerr spin is an axial vector"
    if completed.returncode == 0 or expected not in completed.stdout:
        raise RuntimeError("invalid Kerr reflection did not fail at configuration")
    return {
        "case": name,
        "status": "PASS",
        "steps": 0,
        "hamiltonian_l2": math.nan,
        "hamiltonian_linf": math.nan,
        "momentum_linf": math.nan,
        "exterior_hamiltonian_l2": math.nan,
        "exterior_hamiltonian_linf": math.nan,
        "exterior_momentum_linf": math.nan,
        "horizon_radius": math.nan,
        "min_lapse": math.nan,
        "max_rho": math.nan,
        "symmetry_lapse_gap": math.nan,
        "symmetry_gap_ratio": math.nan,
        "c2p_fallbacks": 0,
        "atmosphere_resets": 0,
        "run_dir": repo_relative(run_dir),
    }


def validate_unimplemented_excision(
    *, ranks: int, timeout: int | None
) -> dict[str, object]:
    name = "kerr_schild_excision_unimplemented"
    run_dir = RUN_ROOT / name
    run_dir.mkdir(parents=True, exist_ok=True)
    input_path = materialize_run_input(
        BASE_INPUT,
        run_dir / "input.in",
        {
            "pyfluid.run_name": name,
            "pyfluid.output_dir": repo_relative(run_dir),
            "spacetime.black_hole_interior": "excision",
            "spacetime.black_hole_fill_radius": "-1.0",
            "pyfluid.max_step": "1",
            "pyfluid.plot_int": "-1",
        },
    )
    completed = run_amrex(
        dim=3,
        input_path=input_path,
        ranks=ranks,
        timeout=timeout,
    )
    (run_dir / "run.log").write_text(completed.stdout)
    expected = "Kerr-Schild excision is a reserved"
    if completed.returncode == 0 or expected not in completed.stdout:
        raise RuntimeError("unimplemented Kerr excision did not fail honestly")
    return {
        "case": name,
        "status": "PASS",
        "steps": 0,
        "hamiltonian_l2": math.nan,
        "hamiltonian_linf": math.nan,
        "momentum_linf": math.nan,
        "exterior_hamiltonian_l2": math.nan,
        "exterior_hamiltonian_linf": math.nan,
        "exterior_momentum_linf": math.nan,
        "horizon_radius": math.nan,
        "min_lapse": math.nan,
        "max_rho": math.nan,
        "symmetry_lapse_gap": math.nan,
        "symmetry_gap_ratio": math.nan,
        "c2p_fallbacks": 0,
        "atmosphere_resets": 0,
        "run_dir": repo_relative(run_dir),
    }


def validate_incompatible_physical_id(
    physical_id: Path,
    *,
    ranks: int,
    timeout: int | None,
) -> dict[str, object]:
    name = "kerr_schild_incompatible_physical_id"
    run_dir = RUN_ROOT / name
    run_dir.mkdir(parents=True, exist_ok=True)
    input_path = materialize_run_input(
        BASE_INPUT,
        run_dir / "input.in",
        {
            "pyfluid.run_name": name,
            "pyfluid.output_dir": repo_relative(run_dir),
            "initial_data.input_file": repo_relative(physical_id),
            "spacetime.dimensionless_spin3": "0.4",
            "pyfluid.max_step": "1",
            "pyfluid.plot_int": "-1",
        },
    )
    completed = run_amrex(
        dim=3,
        input_path=input_path,
        ranks=ranks,
        timeout=timeout,
    )
    (run_dir / "run.log").write_text(completed.stdout)
    expected = "physical-ID Kerr-Schild parameters are incompatible"
    if completed.returncode == 0 or expected not in completed.stdout:
        raise RuntimeError("incompatible Kerr physical ID did not fail honestly")
    return {
        "case": name,
        "status": "PASS",
        "steps": 0,
        "hamiltonian_l2": math.nan,
        "hamiltonian_linf": math.nan,
        "momentum_linf": math.nan,
        "exterior_hamiltonian_l2": math.nan,
        "exterior_hamiltonian_linf": math.nan,
        "exterior_momentum_linf": math.nan,
        "horizon_radius": math.nan,
        "min_lapse": math.nan,
        "max_rho": math.nan,
        "symmetry_lapse_gap": math.nan,
        "symmetry_gap_ratio": math.nan,
        "c2p_fallbacks": 0,
        "atmosphere_resets": 0,
        "run_dir": repo_relative(run_dir),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ranks", type=int, default=3)
    parser.add_argument("--timeout", type=int)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.force and RUN_ROOT.exists():
        shutil.rmtree(RUN_ROOT)
    RUN_ROOT.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, object]] = []
    scalar_rows: list[dict[str, object]] = []
    physical_id = RUN_ROOT / "kerr_schild_filled_full" / "physical_id"
    full, full_scalar = run_case(
        "kerr_schild_filled_full",
        {
            "initial_data.output_file": repo_relative(physical_id),
            "pyfluid.max_step": "3",
        },
        ranks=args.ranks,
        timeout=args.timeout,
    )
    results.append(full)
    scalar_rows.append(full_scalar)

    loaded, loaded_scalar = run_case(
        "kerr_schild_filled_changed_grid",
        {
            "initial_data.input_file": repo_relative(physical_id),
            "amr.n_cell": "24 24 24",
            "amr.max_grid_size": "12",
            "amr.blocking_factor": "4",
            "pyfluid.max_step": "1",
            "pyfluid.plot_int": "-1",
        },
        ranks=args.ranks,
        timeout=args.timeout,
    )
    results.append(loaded)
    scalar_rows.append(loaded_scalar)

    fixed, fixed_scalar = run_case(
        "kerr_schild_fixed_exterior",
        {
            "spacetime.evolution": "fixed",
            "spacetime.matter_coupling": "coupled",
            "spacetime.fixed_metric": "kerr_schild",
            "spacetime.origin1": "-16.0",
            "spacetime.dimensionless_spin1": "0.2",
            "spacetime.dimensionless_spin2": "-0.1",
            "spacetime.dimensionless_spin3": "0.3",
            "spacetime.black_hole_interior": "exterior",
            "spacetime.black_hole_fill_radius": "-1.0",
            "pyfluid.max_step": "1",
            "pyfluid.plot_int": "-1",
        },
        ranks=args.ranks,
        timeout=args.timeout,
    )
    results.append(fixed)
    scalar_rows.append(fixed_scalar)

    moving_puncture, moving_puncture_scalar = run_case(
        "kerr_schild_moving_puncture_step16",
        {
            "pyfluid.max_step": "16",
            "pyfluid.plot_int": "-1",
        },
        ranks=args.ranks,
        timeout=args.timeout,
    )
    if float(moving_puncture["min_lapse"]) < 1.0e-10:
        raise RuntimeError("Kerr moving-puncture lapse crossed its configured floor")
    results.append(moving_puncture)
    scalar_rows.append(moving_puncture_scalar)

    schwarzschild_common = {
        "spacetime.dimensionless_spin3": "0.0",
        "pyfluid.final_time": "0.03",
        "pyfluid.max_step": "1",
        "pyfluid.plot_int": "-1",
    }
    schwarzschild_full, schwarzschild_full_scalar = run_case(
        "schwarzschild_filled_full_n32",
        schwarzschild_common,
        ranks=args.ranks,
        timeout=args.timeout,
    )
    schwarzschild_octant, schwarzschild_octant_scalar = run_case(
        "schwarzschild_filled_octant_n32",
        {
            **schwarzschild_common,
            "geometry.prob_lo": "0.0 0.0 0.0",
            "geometry.prob_hi": "8.0 8.0 8.0",
            "amr.n_cell": "16 16 16",
            "pyfluid.bc_xlo": "symmetry",
            "pyfluid.bc_ylo": "symmetry",
            "pyfluid.bc_zlo": "symmetry",
        },
        ranks=args.ranks,
        timeout=args.timeout,
    )
    octant_gap = abs(
        float(schwarzschild_full["min_lapse"])
        - float(schwarzschild_octant["min_lapse"])
    )
    octant_scale = max(
        1.0,
        abs(float(schwarzschild_full["min_lapse"])),
        abs(float(schwarzschild_octant["min_lapse"])),
    )
    if octant_gap > 1.0e-10 * octant_scale:
        raise RuntimeError(
            "Schwarzschild octant reflection changed the matched-grid "
            f"minimum lapse by {octant_gap:.3e}"
        )
    require_matched_symmetry_extrema(
        schwarzschild_full,
        schwarzschild_octant,
        "Schwarzschild octant reflection",
    )
    for row in (schwarzschild_full, schwarzschild_octant):
        row["symmetry_lapse_gap"] = octant_gap
        row["symmetry_gap_ratio"] = 0.0
    results.extend((schwarzschild_full, schwarzschild_octant))
    scalar_rows.extend((schwarzschild_full_scalar, schwarzschild_octant_scalar))

    symmetry_common = {
        "pyfluid.final_time": "0.03",
        "pyfluid.max_step": "1",
        "pyfluid.plot_int": "-1",
    }
    coarse_full, coarse_full_scalar = run_case(
        "kerr_schild_symmetry_full_n32",
        symmetry_common,
        ranks=args.ranks,
        timeout=args.timeout,
    )
    coarse_half, coarse_half_scalar = run_case(
        "kerr_schild_symmetry_half_n32",
        {
            **symmetry_common,
            "geometry.prob_lo": "-8.0 -8.0 0.0",
            "geometry.prob_hi": "8.0 8.0 8.0",
            "amr.n_cell": "32 32 16",
            "pyfluid.bc_zlo": "symmetry",
        },
        ranks=args.ranks,
        timeout=args.timeout,
    )
    fine_full, fine_full_scalar = run_case(
        "kerr_schild_symmetry_full_n64",
        {
            **symmetry_common,
            "amr.n_cell": "64 64 64",
            "amr.max_grid_size": "32",
        },
        ranks=args.ranks,
        timeout=args.timeout,
    )
    fine_half, fine_half_scalar = run_case(
        "kerr_schild_symmetry_half_n64",
        {
            **symmetry_common,
            "geometry.prob_lo": "-8.0 -8.0 0.0",
            "geometry.prob_hi": "8.0 8.0 8.0",
            "amr.n_cell": "64 64 32",
            "amr.max_grid_size": "32",
            "pyfluid.bc_zlo": "symmetry",
        },
        ranks=args.ranks,
        timeout=args.timeout,
    )
    coarse_gap = abs(
        float(coarse_full["min_lapse"]) - float(coarse_half["min_lapse"])
    )
    fine_gap = abs(
        float(fine_full["min_lapse"]) - float(fine_half["min_lapse"])
    )
    gap_ratio = fine_gap / max(coarse_gap, float.fromhex("0x1p-1022"))
    fine_scale = max(
        1.0, abs(float(fine_full["min_lapse"])), abs(float(fine_half["min_lapse"]))
    )
    if fine_gap > 1.0e-4 * fine_scale or gap_ratio > 0.35:
        raise RuntimeError(
            "Kerr equatorial-reflection mismatch does not converge: "
            f"coarse={coarse_gap:.3e}, fine={fine_gap:.3e}, "
            f"ratio={gap_ratio:.3f}"
        )
    require_matched_symmetry_extrema(
        coarse_full, coarse_half, "coarse Kerr equatorial reflection"
    )
    require_matched_symmetry_extrema(
        fine_full, fine_half, "fine Kerr equatorial reflection"
    )
    for field in (
        "exterior_hamiltonian_linf",
        "exterior_hamiltonian_l2",
        "exterior_momentum_linf",
    ):
        coarse_value = float(coarse_full[field])
        fine_value = float(fine_full[field])
        if fine_value > 0.4 * coarse_value:
            raise RuntimeError(
                f"Kerr exterior {field} does not improve from N=32 to N=64: "
                f"coarse={coarse_value:.6e}, fine={fine_value:.6e}"
            )
    for row in (coarse_full, coarse_half):
        row["symmetry_lapse_gap"] = coarse_gap
        row["symmetry_gap_ratio"] = gap_ratio
    for row in (fine_full, fine_half):
        row["symmetry_lapse_gap"] = fine_gap
        row["symmetry_gap_ratio"] = gap_ratio
    results.extend((coarse_full, coarse_half, fine_full, fine_half))
    scalar_rows.extend(
        (
            coarse_full_scalar,
            coarse_half_scalar,
            fine_full_scalar,
            fine_half_scalar,
        )
    )
    results.append(
        validate_negative_reflection(ranks=args.ranks, timeout=args.timeout)
    )
    results.append(
        validate_unimplemented_excision(ranks=args.ranks, timeout=args.timeout)
    )
    results.append(
        validate_incompatible_physical_id(
            physical_id,
            ranks=args.ranks,
            timeout=args.timeout,
        )
    )

    fields = (
        "case",
        "status",
        "steps",
        "hamiltonian_l2",
        "hamiltonian_linf",
        "momentum_linf",
        "exterior_hamiltonian_l2",
        "exterior_hamiltonian_linf",
        "exterior_momentum_linf",
        "horizon_radius",
        "min_lapse",
        "max_rho",
        "symmetry_lapse_gap",
        "symmetry_gap_ratio",
        "c2p_fallbacks",
        "atmosphere_resets",
        "run_dir",
    )
    write_tsv(RESULTS, results, fields)
    append_validation_history(HISTORY, scalar_rows)
    for row in results:
        print(f"PASS {row['case']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
