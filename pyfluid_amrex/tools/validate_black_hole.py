#!/usr/bin/env python3
"""Validate Kerr-Schild full-grid evolution and reflection contracts."""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np

from plotfile_reader import (
    level_count,
    parse_plot_header,
    read_aux_field,
    read_level_state,
)
from physical_id_metadata import read_physical_id_metadata
from validation_common import (
    OPEN_POLICY,
    ROOT,
    append_validation_history,
    build_cached_target,
    materialize_run_input,
    read_tsv,
    repo_relative,
    run_amrex,
    scalar_validation_row,
    write_tsv,
)


BASE_INPUT = ROOT / "pyfluid_amrex/inputs/kerr_schild_filled_evolution_3d.in"
SPHERICAL_XCTS_INPUT = (
    ROOT / "pyfluid_amrex/inputs/black_hole_xcts_spherical_id_2d.in"
)
SPHERICAL_XCTS_3D_INPUT = (
    ROOT / "pyfluid_amrex/inputs/black_hole_xcts_spherical_id_3d.in"
)
RUN_ROOT = ROOT / "runs/current/black_hole_validation"
RESULTS = RUN_ROOT / "black_hole_validation.tsv"
HISTORY = ROOT / "runs/current/validation_history.tsv"
FIELD_CATALOG = ROOT / "pyfluid_amrex/generated/field_catalog.json"
RECOVERY = re.compile(
    r"Relativistic recovery c2p_fallbacks=(\d+) atmosphere_resets=(\d+)"
)
XCTS_SOLVE = re.compile(
    r"xcts initial data: matter_iterations=(\d+) .*?"
    r"coupled_iterations=(\d+) "
    r"hamiltonian_residual=([0-9.eE+-]+) "
    r"lapse_residual=([0-9.eE+-]+) "
    r"momentum_residual=([0-9.eE+-]+)"
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


read_artifact_metadata = read_physical_id_metadata


def adm_component_indices() -> dict[str, int]:
    catalog = json.loads(FIELD_CATALOG.read_text())
    components = [
        component["name"]
        for field in catalog["fields"]
        if field["role"] == "InitialDataState"
        and field["sector"] == "Spacetime"
        for component in field["components"]
    ]
    return {name: index for index, name in enumerate(components)}


def run_spherical_xcts_case(
    name: str,
    overrides: dict[str, str],
    *,
    ranks: int,
    timeout: int | None,
) -> tuple[Path, Path, str]:
    run_dir = RUN_ROOT / name
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    artifact = run_dir / "physical_id"
    input_path = materialize_run_input(
        SPHERICAL_XCTS_INPUT,
        run_dir / "input.in",
        {
            "pyfluid.run_name": name,
            "pyfluid.output_dir": repo_relative(run_dir),
            "initial_data.output_file": repo_relative(artifact),
            **overrides,
        },
    )
    completed = run_amrex(
        dim=2,
        input_path=input_path,
        ranks=ranks,
        timeout=timeout,
    )
    (run_dir / "run.log").write_text(completed.stdout)
    if completed.returncode != 0:
        raise RuntimeError(
            f"{name} failed; see {repo_relative(run_dir / 'run.log')}"
        )
    if not artifact.is_dir():
        raise RuntimeError(f"{name} did not write its physical-ID artifact")
    recovery = RECOVERY.findall(completed.stdout)
    if not recovery or tuple(int(value) for value in recovery[-1]) != (0, 0):
        raise RuntimeError(f"{name} used relativistic recovery during ID")
    solve = XCTS_SOLVE.search(completed.stdout)
    if solve is None:
        raise RuntimeError(f"{name} did not report its XCTS solve")
    residual = max(abs(float(solve.group(index))) for index in (3, 4, 5))
    if residual > 1.0e-10:
        raise RuntimeError(f"{name} elliptic residual is {residual:.3e}")
    return run_dir, artifact, completed.stdout


def validate_spherical_xcts_black_hole(
    *, ranks: int, timeout: int | None
) -> dict[str, object]:
    _, coarse_artifact, _ = run_spherical_xcts_case(
        "schwarzschild_xcts_spherical_shell_n128",
        {
            "amr.n_cell": "128 16",
            "amr.max_grid_size": "64",
        },
        ranks=ranks,
        timeout=timeout,
    )
    run_dir, artifact, log_text = run_spherical_xcts_case(
        "schwarzschild_xcts_spherical_shell_changed_n256",
        {"initial_data.input_file": repo_relative(coarse_artifact)},
        ranks=ranks,
        timeout=timeout,
    )
    if "re-solving black-hole physical initial data on the changed r theta shell" \
            not in log_text:
        raise RuntimeError("changed-grid black-hole load skipped its XCTS solve")

    source_metadata = read_artifact_metadata(coarse_artifact)
    metadata = read_artifact_metadata(artifact)
    if metadata.get("source_ic") != "black_hole" \
            or metadata.get("spacetime_initial_data") != "xcts" \
            or metadata.get("mesh_coordinate") != "spherical" \
            or metadata.get("mesh_axes") != "r theta":
        raise RuntimeError("spherical black-hole artifact lost its owner identity")
    radial_key = "coordinate_faces_0_0"
    if metadata.get(radial_key) == source_metadata.get(radial_key):
        raise RuntimeError("changed-grid black-hole re-solve retained the source map")
    radial_faces = np.asarray(
        [float(value) for value in metadata[radial_key].split(",")]
    )

    geometry_file = artifact / "adm_geometry.h5"
    names, dimension, _ = parse_plot_header(geometry_file)
    state = read_level_state(geometry_file, names, dimension, 0)
    if dimension != 2 or state.shape != (16, 257, 33):
        raise RuntimeError(
            f"unexpected spherical XCTS ADM shape {state.shape} in {dimension}D"
        )
    if not np.all(np.isfinite(state)):
        raise RuntimeError("spherical XCTS ADM artifact contains nonfinite values")
    component = adm_component_indices()
    if len(component) != state.shape[0]:
        raise RuntimeError("ADM artifact disagrees with the generated field catalog")

    mass = float(metadata["black_hole_mass"])
    isotropic_radius = 0.5 * mass
    conformal_factor = 1.0 + isotropic_radius / radial_faces
    exact_metric = conformal_factor**4
    exact_lapse = (
        (1.0 - isotropic_radius / radial_faces) / conformal_factor
    )
    diagonal = state[
        [component[name] for name in ("gamma11", "gamma22", "gamma33")]
    ]
    metric_relative_error = float(
        np.max(
            np.abs(diagonal - exact_metric[None, :, None])
            / exact_metric[None, :, None]
        )
    )
    lapse_linf_error = float(
        np.max(np.abs(state[component["lapse"]] - exact_lapse[:, None]))
    )
    zero_components = state[
        [
            component[name]
            for name in (
                "gamma12",
                "gamma13",
                "gamma23",
                "k11",
                "k12",
                "k13",
                "k22",
                "k23",
                "k33",
                "shift1",
                "shift2",
                "shift3",
            )
        ]
    ]
    zero_field_linf = float(np.max(np.abs(zero_components)))
    theta_invariance = max(
        float(np.max(np.ptp(component, axis=1))) for component in state
    )
    if metric_relative_error > 2.0e-3 \
            or lapse_linf_error > 2.0e-5 \
            or zero_field_linf > 1.0e-14 \
            or theta_invariance > 2.0e-12:
        raise RuntimeError(
            "spherical XCTS artifact failed its analytic Schwarzschild check: "
            f"metric={metric_relative_error:.3e}, "
            f"lapse={lapse_linf_error:.3e}, "
            f"zero={zero_field_linf:.3e}, "
            f"theta={theta_invariance:.3e}"
        )
    if state[component["lapse"], 0, :].any():
        raise RuntimeError("spherical XCTS horizon lapse is not exactly zero")

    solve = XCTS_SOLVE.search(log_text)
    assert solve is not None
    residual = max(abs(float(solve.group(index))) for index in (3, 4, 5))
    artifact_bytes = sum(
        path.stat().st_size for path in artifact.rglob("*") if path.is_file()
    )
    return {
        "case": "schwarzschild_xcts_spherical_shell_changed_n256",
        "status": "PASS",
        "steps": 0,
        "c2p_fallbacks": 0,
        "atmosphere_resets": 0,
        "xcts_metric_relative_error": metric_relative_error,
        "xcts_lapse_linf_error": lapse_linf_error,
        "xcts_zero_field_linf": zero_field_linf,
        "xcts_theta_invariance": theta_invariance,
        "xcts_elliptic_residual": residual,
        "xcts_artifact_bytes": artifact_bytes,
        "run_dir": repo_relative(run_dir),
    }


def validate_spherical_xcts_black_hole_3d(
    *, ranks: int, timeout: int | None
) -> dict[str, object]:
    name = "schwarzschild_xcts_spherical_shell_3d_nphi4"
    run_dir = RUN_ROOT / name
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    artifact = run_dir / "physical_id"
    input_path = materialize_run_input(
        SPHERICAL_XCTS_3D_INPUT,
        run_dir / "input.in",
        {
            "pyfluid.run_name": name,
            "pyfluid.output_dir": repo_relative(run_dir),
            "initial_data.output_file": repo_relative(artifact),
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
        raise RuntimeError(
            f"{name} failed; see {repo_relative(run_dir / 'run.log')}"
        )
    recovery = RECOVERY.findall(completed.stdout)
    if not recovery or tuple(int(value) for value in recovery[-1]) != (0, 0):
        raise RuntimeError(f"{name} used relativistic recovery during ID")
    solve = XCTS_SOLVE.search(completed.stdout)
    if solve is None:
        raise RuntimeError(f"{name} did not report its XCTS solve")
    residual = max(abs(float(solve.group(index))) for index in (3, 4, 5))
    if residual > 3.0e-8:
        raise RuntimeError(f"{name} elliptic residual is {residual:.3e}")

    source_artifact = artifact
    _, reduced_artifact, _ = run_spherical_xcts_case(
        "schwarzschild_xcts_spherical_shell_2d_matched_96x24",
        {
            "geometry.prob_hi": "24.0 1.5707963267948966",
            "amr.n_cell": "96 24",
            "amr.max_grid_size": "32",
            "amr.blocking_factor": "1",
            "initial_data.adaptive_shell_width_ratio": "8.0",
        },
        ranks=ranks,
        timeout=timeout,
    )
    maximum_reduced_difference = 0.0
    maximum_phi_variation = 0.0
    for field_name in ("physical_fluid.h5", "adm_geometry.h5"):
        reduced_names, reduced_dimension, _ = parse_plot_header(
            reduced_artifact / field_name
        )
        full_names, full_dimension, _ = parse_plot_header(
            source_artifact / field_name
        )
        reduced_state = read_level_state(
            reduced_artifact / field_name,
            reduced_names,
            reduced_dimension,
            0,
        )
        full_state = read_level_state(
            source_artifact / field_name,
            full_names,
            full_dimension,
            0,
        )
        scale = max(float(np.max(np.abs(reduced_state))), 1.0e-14)
        maximum_reduced_difference = max(
            maximum_reduced_difference,
            float(np.max(np.abs(full_state[..., 0] - reduced_state))) / scale,
        )
        maximum_phi_variation = max(
            maximum_phi_variation,
            float(
                np.max(
                    np.abs(
                        full_state
                        - np.repeat(
                            full_state[..., :1], full_state.shape[-1], axis=-1
                        )
                    )
                )
            ) / scale,
        )
    if maximum_reduced_difference > 3.0e-12 \
            or maximum_phi_variation > 5.0e-14:
        raise RuntimeError(
            "matched 2D and full-storage spherical XCTS black-hole shells "
            "are not coordinate equivalent: "
            f"fields={maximum_reduced_difference:.3e}, "
            f"phi={maximum_phi_variation:.3e}"
        )

    name = "schwarzschild_xcts_spherical_shell_3d_changed_nphi4"
    run_dir = RUN_ROOT / name
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    artifact = run_dir / "physical_id"
    input_path = materialize_run_input(
        SPHERICAL_XCTS_3D_INPUT,
        run_dir / "input.in",
        {
            "pyfluid.run_name": name,
            "pyfluid.output_dir": repo_relative(run_dir),
            "amr.n_cell": "128 32 4",
            "amr.max_grid_size": "64",
            "initial_data.input_file": repo_relative(source_artifact),
            "initial_data.output_file": repo_relative(artifact),
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
        raise RuntimeError(
            f"{name} failed; see {repo_relative(run_dir / 'run.log')}"
        )
    if "re-solving black-hole physical initial data on the changed " \
            "r theta phi shell" not in completed.stdout:
        raise RuntimeError("changed 3D spherical black-hole load skipped XCTS")
    solve = XCTS_SOLVE.search(completed.stdout)
    if solve is None:
        raise RuntimeError(f"{name} did not report its XCTS re-solve")
    residual = max(abs(float(solve.group(index))) for index in (3, 4, 5))
    if residual > 3.0e-8:
        raise RuntimeError(f"{name} elliptic residual is {residual:.3e}")

    metadata = read_artifact_metadata(artifact)
    if (
        metadata.get("source_ic") != "black_hole"
        or metadata.get("spacetime_initial_data") != "xcts"
        or metadata.get("mesh_coordinate") != "spherical"
        or metadata.get("mesh_axes") != "r theta phi"
        or metadata.get("periodicity") != "0,0,1"
    ):
        raise RuntimeError("3D spherical black-hole artifact lost owner identity")
    radial_faces = np.asarray(
        [float(value) for value in metadata["coordinate_faces_0_0"].split(",")]
    )
    geometry_file = artifact / "adm_geometry.h5"
    names, dimension, _ = parse_plot_header(geometry_file)
    state = read_level_state(geometry_file, names, dimension, 0)
    if dimension != 3 or state.shape != (16, 129, 33, 5):
        raise RuntimeError(
            f"unexpected 3D spherical XCTS ADM shape {state.shape} in {dimension}D"
        )
    if not np.all(np.isfinite(state)):
        raise RuntimeError("3D spherical XCTS ADM artifact contains nonfinite values")
    field_component = adm_component_indices()
    mass = float(metadata["black_hole_mass"])
    isotropic_radius = 0.5 * mass
    conformal_factor = 1.0 + isotropic_radius / radial_faces
    exact_metric = conformal_factor**4
    exact_lapse = (1.0 - isotropic_radius / radial_faces) / conformal_factor
    diagonal = state[
        [field_component[name] for name in ("gamma11", "gamma22", "gamma33")]
    ]
    metric_relative_error = float(
        np.max(
            np.abs(diagonal - exact_metric[None, :, None, None])
            / exact_metric[None, :, None, None]
        )
    )
    lapse_linf_error = float(
        np.max(
            np.abs(
                state[field_component["lapse"]]
                - exact_lapse[:, None, None]
            )
        )
    )
    zero_components = state[
        [
            field_component[name]
            for name in (
                "gamma12",
                "gamma13",
                "gamma23",
                "k11",
                "k12",
                "k13",
                "k22",
                "k23",
                "k33",
                "shift1",
                "shift2",
                "shift3",
            )
        ]
    ]
    zero_field_linf = float(np.max(np.abs(zero_components)))
    angular_invariance = max(
        float(np.max(np.ptp(values, axis=(1, 2)))) for values in state
    )
    if (
        metric_relative_error > 2.0e-2
        or lapse_linf_error > 3.0e-3
        or zero_field_linf > 1.0e-14
        or angular_invariance > 2.0e-10
    ):
        raise RuntimeError(
            "3D spherical XCTS artifact failed its analytic Schwarzschild "
            f"check: metric={metric_relative_error:.3e}, "
            f"lapse={lapse_linf_error:.3e}, zero={zero_field_linf:.3e}, "
            f"angular={angular_invariance:.3e}"
        )
    if state[field_component["lapse"], 0].any():
        raise RuntimeError("3D spherical XCTS horizon lapse is not exactly zero")

    artifact_bytes = sum(
        path.stat().st_size for path in artifact.rglob("*") if path.is_file()
    )
    return {
        "case": name,
        "status": "PASS",
        "steps": 0,
        "c2p_fallbacks": 0,
        "atmosphere_resets": 0,
        "xcts_metric_relative_error": metric_relative_error,
        "xcts_lapse_linf_error": lapse_linf_error,
        "xcts_zero_field_linf": zero_field_linf,
        "xcts_theta_invariance": angular_invariance,
        "xcts_reduced_field_difference": maximum_reduced_difference,
        "xcts_phi_variation": maximum_phi_variation,
        "xcts_elliptic_residual": residual,
        "xcts_artifact_bytes": artifact_bytes,
        "run_dir": repo_relative(run_dir),
    }


def validate_spherical_xcts_rejection(
    name: str,
    overrides: dict[str, str],
    expected: str,
    *,
    ranks: int,
    timeout: int | None,
) -> dict[str, object]:
    run_dir = RUN_ROOT / name
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    input_path = materialize_run_input(
        SPHERICAL_XCTS_INPUT,
        run_dir / "input.in",
        {
            "pyfluid.run_name": name,
            "pyfluid.output_dir": repo_relative(run_dir),
            "initial_data.output_file": repo_relative(run_dir / "physical_id"),
            **overrides,
        },
    )
    completed = run_amrex(
        dim=2,
        input_path=input_path,
        ranks=ranks,
        timeout=timeout,
    )
    (run_dir / "run.log").write_text(completed.stdout)
    if completed.returncode == 0 or expected not in completed.stdout:
        raise RuntimeError(f"invalid spherical XCTS case {name} was accepted")
    return {
        "case": name,
        "status": "PASS",
        "steps": 0,
        "c2p_fallbacks": 0,
        "atmosphere_resets": 0,
        "run_dir": repo_relative(run_dir),
    }


def validate_apparent_horizon_core() -> None:
    target = "pyfluid_amrex_apparent_horizon_check"
    binary = build_cached_target(target)
    completed = subprocess.run(
        (str(binary),),
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"{target} failed:\n{completed.stdout}\n{completed.stderr}"
        )
    if "PASS apparent-horizon finder check" not in completed.stdout:
        raise RuntimeError(f"{target} did not report its PASS marker")


def validate_apparent_horizon_run(
    *, ranks: int, timeout: int | None
) -> tuple[dict[str, object], dict[str, object]]:
    name = "kerr_schild_apparent_horizon"
    row, scalar = run_case(
        name,
        {
            "pyfluid.max_step": "2",
            "pyfluid.plot_int": "-1",
            "pyfluid.aux_gf_int": "1",
            "pyfluid.aux_gfs": "apparent_horizon_mask,excision_mask",
            "pyfluid.checkpoint_int": "2",
            "amr.max_level": "1",
            "amr.ref_ratio": "2",
            "amr.regrid_int": "1",
            "amr.n_error_buf": "2",
            "horizon.enabled": "1",
            "horizon.find_int": "1",
            "horizon.radial_points": "48",
            "horizon.polar_points": "16",
            "horizon.coarsest_polar_points": "8",
            "horizon.theta_linf_times_mass_tolerance": "5.0e-4",
            "horizon.theta_l2_times_mass_tolerance": "1.0e-2",
            "horizon.refine_band_cells": "2.0",
        },
        ranks=ranks,
        timeout=timeout,
    )
    run_dir = RUN_ROOT / name
    horizon_rows = read_tsv(run_dir / "apparent_horizon.tsv")
    if len(horizon_rows) != 3:
        raise RuntimeError(f"{name} did not persist all three tracked surfaces")
    initial = horizon_rows[0]
    spin = 0.5
    exact_area = 8.0 * math.pi * (1.0 + math.sqrt(1.0 - spin * spin))
    area_error = abs(float(initial["area"]) - exact_area) / exact_area
    if area_error > 3.0e-2:
        raise RuntimeError(
            f"{name} Kerr horizon area error is {area_error:.3e}"
        )
    theta_linf = float(initial["theta_linf_times_mass"])
    if theta_linf > 5.0e-4:
        raise RuntimeError(
            f"{name} did not satisfy its expansion tolerance: {theta_linf:.3e}"
        )
    shape_rows = read_tsv(run_dir / "apparent_horizon" / "ah00000000.tsv")
    if len(shape_rows) != 16 * 32:
        raise RuntimeError(f"{name} persisted an incomplete horizon shape")

    checkpoint = run_dir / "checkpoints" / "chk00002"
    restart_input = materialize_run_input(
        BASE_INPUT,
        run_dir / "restart_input.in",
        {
            "pyfluid.run_name": name,
            "pyfluid.output_dir": repo_relative(run_dir),
            "pyfluid.restart_file": repo_relative(checkpoint),
            "pyfluid.max_step": "3",
            "pyfluid.plot_int": "-1",
            "pyfluid.aux_gf_int": "1",
            "pyfluid.aux_gfs": "apparent_horizon_mask,excision_mask",
            "pyfluid.checkpoint_int": "-1",
            "amr.max_level": "1",
            "amr.ref_ratio": "2",
            "amr.regrid_int": "1",
            "amr.n_error_buf": "2",
            "horizon.enabled": "1",
            "horizon.find_int": "3",
            "horizon.radial_points": "48",
            "horizon.polar_points": "16",
            "horizon.coarsest_polar_points": "8",
            "horizon.theta_linf_times_mass_tolerance": "5.0e-4",
            "horizon.theta_l2_times_mass_tolerance": "1.0e-2",
            "horizon.refine_band_cells": "2.0",
        },
    )
    restarted = run_amrex(
        dim=3,
        input_path=restart_input,
        ranks=ranks,
        timeout=timeout,
    )
    (run_dir / "restart.log").write_text(restarted.stdout)
    if restarted.returncode != 0:
        raise RuntimeError(f"{name} restart failed; see {run_dir / 'restart.log'}")
    if "restored apparent-horizon surface from step 2" not in restarted.stdout:
        raise RuntimeError(f"{name} restart did not restore its tracked surface")
    restarted_scalars = read_tsv(
        run_dir / "scalars.tsv", require_complete=True
    )
    if not restarted_scalars or int(restarted_scalars[-1]["step"]) != 3:
        raise RuntimeError(f"{name} restart has an incomplete scalar history")
    for scalar_row in restarted_scalars:
        for field in CONSTRAINT_FIELDS:
            if field not in scalar_row or not math.isfinite(float(scalar_row[field])):
                raise RuntimeError(f"{name} restart has nonfinite {field}")
    recovery = RECOVERY.findall(restarted.stdout)
    restarted_fallbacks, restarted_resets = (
        tuple(int(value) for value in recovery[-1]) if recovery else (-1, -1)
    )
    if restarted_fallbacks != 0 or restarted_resets != 0:
        raise RuntimeError(
            f"{name} restart used C2P fallback/reset: "
            f"{restarted_fallbacks}/{restarted_resets}"
        )
    restarted_horizon_rows = read_tsv(run_dir / "apparent_horizon.tsv")
    if len(restarted_horizon_rows) != 4 or int(
        restarted_horizon_rows[-1]["step"]
    ) != 3:
        raise RuntimeError(f"{name} restart did not continue its horizon history")

    aux_dir = run_dir / "auxGF"
    horizon_mask, _ = read_aux_field(
        aux_dir, "apparent_horizon_mask", step=3
    )
    excision_mask, _ = read_aux_field(aux_dir, "excision_mask", step=3)
    horizon_cells = int(horizon_mask.sum())
    excision_cells = int(excision_mask.sum())
    if horizon_cells <= 0 or excision_cells <= 0:
        raise RuntimeError(f"{name} emitted an empty horizon/excision mask")
    if (excision_mask > horizon_mask).any():
        raise RuntimeError(f"{name} excision mask is not nested inside the horizon")
    aux_plotfile = next(aux_dir.glob("auxGF00003.h5"))
    if level_count(aux_plotfile) < 2:
        raise RuntimeError(f"{name} horizon band did not create an AMR level")
    fine_horizon_mask, _ = read_aux_field(
        aux_dir, "apparent_horizon_mask", step=3, level=1
    )
    fine_excision_mask, _ = read_aux_field(
        aux_dir, "excision_mask", step=3, level=1
    )
    if fine_horizon_mask.sum() <= 0 or fine_excision_mask.sum() <= 0:
        raise RuntimeError(f"{name} emitted empty fine-level masks")
    if (fine_excision_mask > fine_horizon_mask).any():
        raise RuntimeError(f"{name} fine excision mask is not nested")
    row["ah_theta_linf_times_mass"] = theta_linf
    row["ah_area_relative_error"] = area_error
    row["ah_horizon_cells"] = horizon_cells
    row["ah_excision_cells"] = excision_cells
    return row, scalar


def run_case(
    name: str,
    overrides: dict[str, str],
    *,
    ranks: int,
    timeout: int | None,
) -> tuple[dict[str, object], dict[str, object]]:
    run_dir = RUN_ROOT / name
    if run_dir.exists():
        shutil.rmtree(run_dir)
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
    results.append(
        validate_spherical_xcts_black_hole(
            ranks=args.ranks, timeout=args.timeout
        )
    )
    results.append(
        validate_spherical_xcts_black_hole_3d(
            ranks=args.ranks, timeout=args.timeout
        )
    )
    results.extend(
        validate_spherical_xcts_rejection(
            name,
            overrides,
            expected,
            ranks=args.ranks,
            timeout=args.timeout,
        )
        for name, overrides, expected in (
            (
                "schwarzschild_xcts_invalid_inner_boundary",
                {"geometry.prob_lo": "0.4 0.0"},
                "vacuum Schwarzschild horizon at r=M/2",
            ),
            (
                "schwarzschild_xcts_invalid_origin",
                {"spacetime.origin1": "1.0"},
                "origin-centered solve-only spherical r-theta or axisymmetric "
                "r-theta-phi XCTS shell",
            ),
        )
    )
    validate_apparent_horizon_core()
    horizon, horizon_scalar = validate_apparent_horizon_run(
        ranks=args.ranks, timeout=args.timeout
    )
    results.append(horizon)
    scalar_rows.append(horizon_scalar)
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
        "ah_theta_linf_times_mass",
        "ah_area_relative_error",
        "ah_horizon_cells",
        "ah_excision_cells",
        "xcts_metric_relative_error",
        "xcts_lapse_linf_error",
        "xcts_zero_field_linf",
        "xcts_theta_invariance",
        "xcts_reduced_field_difference",
        "xcts_phi_variation",
        "xcts_elliptic_residual",
        "xcts_artifact_bytes",
        "run_dir",
    )
    write_tsv(RESULTS, results, fields)
    append_validation_history(HISTORY, scalar_rows)
    for row in results:
        print(f"PASS {row['case']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
