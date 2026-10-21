#!/usr/bin/env python3
"""Check relativistic diode fluxes and profile-invariant flux dispatch."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np

from validate_black_hole import COUPLED_EXCISION_INPUT, EXCISION_INPUT, RUN_ROOT, run_case
from plotfile_reader import latest_plotfile, parse_plot_header, read_level_state
from validation_common import (
    OPEN_MHD_POLICY,
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


BASE_INPUT = ROOT / "pyfluid_amrex/inputs/kerr_schild_bounded_fixed_evolution_3d.in"
RHD_DISPATCH_INPUT = ROOT / "pyfluid_amrex/inputs/rhd_uniform_1d.in"
RMHD_DISPATCH_INPUT = ROOT / "pyfluid_amrex/inputs/rmhd_uniform_1d.in"
PERFORMANCE_SCALARS = frozenset(
    ("mcells_per_s", "layout_mcells_per_s", "sim_time_per_s")
)


def _dispatch_scalar_history(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [
        {key: value for key, value in row.items() if key not in PERFORMANCE_SCALARS}
        for row in rows
    ]


def _run_dispatch_case(
    physics: str,
    detail: str,
    *,
    ranks: int,
    timeout: int,
    executable: Path | None,
    output_root: Path,
) -> tuple[dict[str, object], dict[str, object], list[dict[str, str]], np.ndarray]:
    name = f"diode_dispatch_{physics}_{detail}"
    run_dir = output_root / name
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    base_input = RHD_DISPATCH_INPUT if physics == "rhd" else RMHD_DISPATCH_INPUT
    riemann = "hll" if physics == "rhd" else "llf"
    input_path = materialize_run_input(
        base_input,
        run_dir / "input.in",
        {
            "geometry.is_periodic": "0",
            "spacetime.initial_data": "minkowski",
            "spacetime.fixed_metric": "minkowski",
            "pyfluid.run_name": name,
            "pyfluid.output_dir": repo_relative(run_dir),
            "pyfluid.validation_policy": "open_mhd" if physics == "rmhd" else "open",
            "pyfluid.bc": "diode",
            "pyfluid.riemann": riemann,
            "pyfluid.profile": "on",
            "pyfluid.profile_detail": detail,
            "pyfluid.final_time": "-1",
            "pyfluid.max_step": "2",
            "pyfluid.scalar_int": "1",
            "pyfluid.plot_int": "2",
            "pyfluid.checkpoint_int": "-1",
        },
    )
    completed = run_amrex(
        dim=1,
        input_path=input_path,
        ranks=ranks,
        executable=executable,
        timeout=timeout,
    )
    (run_dir / "run.log").write_text(completed.stdout)
    if completed.returncode != 0:
        raise RuntimeError(f"{name} failed; see {repo_relative(run_dir / 'run.log')}")
    rows = read_tsv(run_dir / "scalars.tsv", require_complete=True)
    if len(rows) != 3 or int(rows[-1]["step"]) != 2:
        raise RuntimeError(f"{name} has an incomplete scalar history")
    plotfile = latest_plotfile(run_dir, require=True)
    names, ndim, _time = parse_plot_header(plotfile)
    state = read_level_state(plotfile, names, ndim, 0)
    if not np.all(np.isfinite(state)):
        raise RuntimeError(f"{name} has nonfinite output state")
    scalar = scalar_validation_row(
        run_dir,
        case=name,
        input_file=run_dir / "run.in",
        checks=OPEN_MHD_POLICY if physics == "rmhd" else OPEN_POLICY,
        mhd=physics == "rmhd",
        dim=1,
        ranks=ranks,
        kind="relativistic_diode_dispatch",
    )
    if scalar["status"] != "PASS":
        raise RuntimeError(f"{name} scalar validation failed: {scalar['reason']}")
    boundary = [float(row["boundary_mass_change"]) for row in rows]
    result: dict[str, object] = {
        "case": name,
        "physics": physics,
        "profile_detail": detail,
        "riemann": riemann,
        "status": "PASS",
        "boundary_mass_change": boundary[-1],
        "minimum_boundary_mass_change": min(boundary),
        "state_values": state.size,
        "run_dir": repo_relative(run_dir),
    }
    return result, scalar, rows, state


def validate_profile_dispatch(
    *,
    ranks: int,
    timeout: int,
    executable: Path | None,
    output_root: Path,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    results: list[dict[str, object]] = []
    scalar_rows: list[dict[str, object]] = []
    failures: list[str] = []
    for physics in ("rhd", "rmhd"):
        failure_count = len(failures)
        runs = {}
        for detail in ("basic", "flux"):
            result, scalar, rows, state = _run_dispatch_case(
                physics,
                detail,
                ranks=ranks,
                timeout=timeout,
                executable=executable,
                output_root=output_root,
            )
            results.append(result)
            scalar_rows.append(scalar)
            runs[detail] = (rows, state)

        basic_rows, basic_state = runs["basic"]
        flux_rows, flux_state = runs["flux"]
        basic_boundary = [row["boundary_mass_change"] for row in basic_rows]
        flux_boundary = [row["boundary_mass_change"] for row in flux_rows]
        if basic_boundary != flux_boundary:
            failures.append(
                f"{physics} boundary transport budget history changed with profiling: "
                f"basic={basic_boundary}, flux={flux_boundary}"
            )
        if _dispatch_scalar_history(basic_rows) != _dispatch_scalar_history(flux_rows):
            failures.append(f"{physics} scalar history changed with profiling")
        if not np.array_equal(basic_state, flux_state):
            max_difference = float(np.max(np.abs(basic_state - flux_state)))
            failures.append(
                f"{physics} final state changed with profiling: max_abs={max_difference:.17g}"
            )
        for detail, rows in (("basic", basic_rows), ("flux", flux_rows)):
            boundary = [float(row["boundary_mass_change"]) for row in rows]
            if max(boundary) > 0.0:
                failures.append(
                    f"{physics}/{detail} admitted baryon mass through a diode: {boundary}"
                )
            if min(boundary) >= 0.0:
                failures.append(
                    f"{physics}/{detail} did not exercise outgoing diode transport: {boundary}"
                )
        status = "PASS" if len(failures) == failure_count else "FAIL"
        print(
            f"{status} diode dispatch {physics}: "
            f"boundary_mass_change={float(basic_boundary[-1]):.6e}",
            flush=True,
        )

    if failures:
        for result in results:
            result["status"] = "FAIL"
            result["reason"] = "; ".join(failures)
    write_tsv(
        output_root / "diode_profile_dispatch.tsv",
        results,
        list(dict.fromkeys(key for row in results for key in row)),
    )
    if failures:
        raise RuntimeError("; ".join(failures))
    return results, scalar_rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ranks", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--dispatch-only", action="store_true")
    parser.add_argument("--executable", type=Path, help="1D profiling-dispatch executable")
    parser.add_argument("--output-root", type=Path, default=RUN_ROOT,
                        help="output root for profiling-dispatch controls")
    args = parser.parse_args()
    output_root = args.output_root
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    output_root.mkdir(parents=True, exist_ok=True)
    executable = args.executable
    if executable is not None and not executable.is_absolute():
        executable = ROOT / executable
    _dispatch_results, dispatch_scalars = validate_profile_dispatch(
        ranks=args.ranks,
        timeout=args.timeout,
        executable=executable,
        output_root=output_root,
    )
    if args.dispatch_only:
        append_validation_history(
            ROOT / "runs/current/validation_history.tsv", dispatch_scalars
        )
        return 0
    magnetic = {
        "pyfluid.physics": "rmhd", "pyfluid.ct": "on",
        "pyfluid.ic": "sod_plane",
        "pyfluid.shock_nx": "0", "pyfluid.shock_ny": "0",
        "pyfluid.shock_nz": "1",
        "pyfluid.rho_left": "1e-4", "pyfluid.rho_right": "1e-4",
        "pyfluid.pressure_left": "1e-6", "pyfluid.pressure_right": "1e-6",
        "pyfluid.b3_left": "1e-6", "pyfluid.b3_right": "1e-6",
        "pyfluid.b3_0": "1e-6",
    }
    cases = {
        "outflow_control": {"pyfluid.bc": "outflow"},
        "rhd_fixed": {},
        "rhd_fixed_ssprk3": {
            "pyfluid.recon": "mp5",
            "pyfluid.riemann": "llf",
            "pyfluid.timestep": "ssprk3",
        },
        # The separate coarse filled/evolved MP5 probe fails with either
        # outflow or diode (Cycle 367). Isolate the boundary regression using
        # the already validated evolved-excision interior, not a hidden floor
        # change or reduced CFL. Its ledger includes the inner surface too.
        "rhd_coupled_excision": {},
        "rmhd_flat": {
            **magnetic,
            "spacetime.initial_data": "minkowski",
            "spacetime.fixed_metric": "minkowski",
            "spacetime.black_hole_matter": "evolve",
            "spacetime.dimensionless_spin3": "0",
            "spacetime.black_hole_interior": "exterior",
            "spacetime.black_hole_fill_radius": "-1",
            "horizon.enabled": "0",
        },
        "rmhd_hlld_excision": {
            **magnetic,
            "pyfluid.recon": "mp5", "pyfluid.riemann": "hlld",
            # Exercise the candidate path too, not only a high-order flux.
            "pyfluid.rmhd_candidate_guard": "pwl-pwc",
            "pyfluid.rmhd_candidate_dmp": "1",
            "pyfluid.rmhd_candidate_dmp_factor": "1",
        },
    }
    results, scalar_rows = [], list(dispatch_scalars)
    for label, parameters in cases.items():
        name = "diode_" + label
        base_input = {
            "rhd_coupled_excision": COUPLED_EXCISION_INPUT,
            "rmhd_hlld_excision": EXCISION_INPUT,
        }.get(label, BASE_INPUT)
        result, scalar = run_case(
            name, {"pyfluid.max_step": "3", "pyfluid.plot_int": "-1",
                   "pyfluid.bc": "diode", **parameters},
            ranks=args.ranks, timeout=args.timeout,
            base_input=base_input,
            require_bh_exterior_constraints=label != "rmhd_flat",
        )
        records = read_tsv(RUN_ROOT / name / "scalars.tsv", require_complete=True)
        boundary = [float(record["boundary_mass_change"]) for record in records]
        result["boundary_mass_change"] = boundary[-1]
        result["external_mass_rel_error"] = scalar["external_mass_rel_error"]
        if label == "outflow_control":
            if boundary[-1] <= 1e-8:
                raise RuntimeError("outflow control did not demonstrate shift-driven mass influx")
        elif max(boundary) > 1e-14:
            raise RuntimeError(f"{name} admitted baryon mass through its numerical boundary flux")
        if float(scalar["external_mass_rel_error"]) > 1e-12:
            raise RuntimeError(f"{name} failed baryon closure")
        if result["interior_resets"] != 0 or result["interior_bounds"] != 0:
            raise RuntimeError(f"{name} needed an interior intervention")
        if label == "rmhd_hlld_excision":
            log_lines = (RUN_ROOT / name / "run.log").read_text().splitlines()
            recovery_line = next(line for line in reversed(log_lines)
                                 if line.startswith("Relativistic recovery "))
            counts = dict(token.split("=", 1) for token in recovery_line.split()[2:])
            corrections = sum(int(counts[key]) for key in
                              ("rmhd_pwl_corrections", "rmhd_pwc_corrections"))
            result["candidate_corrections"] = corrections
            if corrections == 0:
                raise RuntimeError("diode RMHD control did not exercise candidate correction")
        if label.startswith("rmhd"):
            result["divb_linf"] = records[-1]["divb_linf"]
            if max(abs(float(record["divb_linf"])) for record in records) > 1e-14:
                raise RuntimeError("diode RMHD control lost its divergence-free CT field")
        results.append(result)
        scalar_rows.append(scalar)
        write_tsv(RUN_ROOT / "diode_validation.tsv", results,
                  list(dict.fromkeys(key for row in results for key in row)))
        print(f"PASS {name}: boundary_mass_change={boundary[-1]:.6e}", flush=True)
    append_validation_history(ROOT / "runs/current/validation_history.tsv", scalar_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
