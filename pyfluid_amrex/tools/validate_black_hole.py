#!/usr/bin/env python3
"""Validate black-hole ID, gauge, horizon, and reflection contracts."""

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
STATIC_TRUMPET_INPUT = (
    ROOT / "pyfluid_amrex/inputs/static_trumpet_evolution_3d.in"
)
SPHERICAL_XCTS_INPUT = (
    ROOT / "pyfluid_amrex/inputs/black_hole_xcts_spherical_id_2d.in"
)
SPHERICAL_XCTS_3D_INPUT = (
    ROOT / "pyfluid_amrex/inputs/black_hole_xcts_spherical_id_3d.in"
)
SPHERICAL_SPINNING_XCTS_3D_INPUT = (
    ROOT / "pyfluid_amrex/inputs/black_hole_xcts_spinning_spherical_id_3d.in"
)
SPINNING_XCTS_CARTESIAN_INPUT = (
    ROOT
    / "pyfluid_amrex/inputs/black_hole_xcts_spinning_from_spherical_id_3d.in"
)
SPINNING_XCTS_EVOLUTION_INPUT = (
    ROOT / "pyfluid_amrex/inputs/black_hole_xcts_spinning_evolution_3d.in"
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


def validate_schwarzschild_throat_diagnostics(
    metadata: dict[str, str],
) -> tuple[float, float, float, float]:
    mass = float(metadata["black_hole_mass"])
    exact_area = 16.0 * math.pi * mass * mass
    exact_circumference = 4.0 * math.pi * mass
    area_error = abs(float(metadata["black_hole_throat_area"]) - exact_area) \
        / exact_area
    circumference_error = max(
        abs(
            float(metadata["black_hole_throat_equatorial_circumference"])
            - exact_circumference
        ),
        abs(
            float(metadata["black_hole_throat_polar_circumference"])
            - exact_circumference
        ),
    ) / exact_circumference
    mass_error = abs(float(metadata["black_hole_throat_kerr_mass"]) - mass) \
        / mass
    spin_error = abs(float(metadata["black_hole_throat_kerr_spin"]))
    angular_momentum_error = abs(
        float(metadata["black_hole_throat_angular_momentum"])
    )
    christodoulou_mass_error = abs(
        float(metadata["black_hole_throat_christodoulou_mass"]) - mass
    ) / mass
    christodoulou_spin_error = abs(
        float(metadata["black_hole_throat_christodoulou_spin"])
    )
    irreducible_mass_error = abs(
        float(metadata["black_hole_irreducible_mass"]) - mass
    ) / mass
    if area_error > 1.0e-2 \
            or circumference_error > 5.0e-3 \
            or mass_error > 5.0e-3 \
            or spin_error > 6.0e-2 \
            or angular_momentum_error > 1.0e-12 \
            or christodoulou_mass_error > 5.0e-3 \
            or christodoulou_spin_error > 1.0e-12 \
            or irreducible_mass_error > 5.0e-3:
        raise RuntimeError(
            "spherical XCTS throat diagnostics failed the Schwarzschild "
            f"check: area={area_error:.3e}, "
            f"circumference={circumference_error:.3e}, "
            f"mass={mass_error:.3e}, spin={spin_error:.3e}, "
            f"J_H={angular_momentum_error:.3e}, "
            f"Christodoulou_mass={christodoulou_mass_error:.3e}, "
            f"Christodoulou_spin={christodoulou_spin_error:.3e}, "
            f"irreducible_mass={irreducible_mass_error:.3e}"
        )
    return area_error, circumference_error, mass_error, spin_error


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
    _, _, exact_layout_log = run_spherical_xcts_case(
        "schwarzschild_xcts_spherical_shell_exact_layout_resolve",
        {
            "amr.n_cell": "128 16",
            "amr.max_grid_size": "64",
            "initial_data.input_file": repo_relative(coarse_artifact),
        },
        ranks=ranks,
        timeout=timeout,
    )
    resolve_message = (
        "re-solving black-hole physical initial data on the configured "
        "r theta shell"
    )
    if resolve_message not in exact_layout_log:
        raise RuntimeError(
            "exact-layout solve-only black-hole load skipped its XCTS solve"
        )

    run_dir, artifact, log_text = run_spherical_xcts_case(
        "schwarzschild_xcts_spherical_shell_changed_n256",
        {"initial_data.input_file": repo_relative(coarse_artifact)},
        ranks=ranks,
        timeout=timeout,
    )
    if resolve_message not in log_text:
        raise RuntimeError("changed-grid black-hole load skipped its XCTS solve")

    source_metadata = read_artifact_metadata(coarse_artifact)
    metadata = read_artifact_metadata(artifact)
    (
        throat_area_error,
        throat_circumference_error,
        throat_mass_error,
        throat_spin_error,
    ) = validate_schwarzschild_throat_diagnostics(metadata)
    coarse_throat_errors = validate_schwarzschild_throat_diagnostics(
        source_metadata
    )
    fine_throat_errors = (
        throat_area_error,
        throat_circumference_error,
        throat_mass_error,
        throat_spin_error,
    )
    if any(
        fine_error > 0.55 * coarse_error
        for fine_error, coarse_error in zip(
            fine_throat_errors, coarse_throat_errors, strict=True
        )
    ):
        raise RuntimeError(
            "spherical XCTS throat diagnostics did not improve under radial "
            f"refinement: coarse={coarse_throat_errors}, "
            f"fine={fine_throat_errors}"
        )
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
        "xcts_throat_area_relative_error": throat_area_error,
        "xcts_throat_circumference_relative_error": throat_circumference_error,
        "xcts_throat_kerr_mass_relative_error": throat_mass_error,
        "xcts_throat_kerr_spin_absolute_error": throat_spin_error,
        "run_dir": repo_relative(run_dir),
    }


def validate_spherical_xcts_spinning_black_hole(
    *, ranks: int, timeout: int | None
) -> dict[str, object]:
    target_spin = 0.3
    run_dir, artifact, log_text = run_spherical_xcts_case(
        "spinning_xcts_spherical_shell_chi0p3",
        {"spacetime.dimensionless_spin3": str(target_spin)},
        ranks=ranks,
        timeout=timeout,
    )
    metadata = read_artifact_metadata(artifact)
    irreducible_mass = float(metadata["black_hole_irreducible_mass"])
    angular_momentum = float(
        metadata["black_hole_throat_angular_momentum"]
    )
    horizon_mass = float(
        metadata["black_hole_throat_christodoulou_mass"]
    )
    measured_spin = float(
        metadata["black_hole_throat_christodoulou_spin"]
    )
    expected_mass = math.sqrt(
        irreducible_mass * irreducible_mass
        + angular_momentum * angular_momentum
        / (4.0 * irreducible_mass * irreducible_mass)
    )
    expected_spin = angular_momentum / (horizon_mass * horizon_mass)
    spin_error = abs(measured_spin - target_spin)
    identity_error = max(
        abs(horizon_mass - expected_mass) / expected_mass,
        abs(measured_spin - expected_spin),
    )
    if spin_error > 1.0e-4 or identity_error > 2.0e-13:
        raise RuntimeError(
            "spinning spherical XCTS throat failed its physical-spin check: "
            f"target={target_spin:.16e}, measured={measured_spin:.16e}, "
            f"identity={identity_error:.3e}"
        )

    geometry_file = artifact / "adm_geometry.h5"
    names, dimension, _ = parse_plot_header(geometry_file)
    state = read_level_state(geometry_file, names, dimension, 0)
    component = adm_component_indices()
    shift_linf = float(np.max(np.abs(state[component["shift3"]])))
    extrinsic_linf = float(
        np.max(
            np.abs(
                state[
                    [
                        component["k13"],
                        component["k23"],
                        component["k33"],
                    ]
                ]
            )
        )
    )
    if shift_linf <= 1.0e-4 or extrinsic_linf <= 1.0e-5:
        raise RuntimeError(
            "spinning spherical XCTS artifact lost its rotational fields: "
            f"shift={shift_linf:.3e}, K={extrinsic_linf:.3e}"
        )

    full_name = "spinning_xcts_spherical_shell_chi0p3_3d"
    full_run_dir = RUN_ROOT / full_name
    if full_run_dir.exists():
        shutil.rmtree(full_run_dir)
    full_run_dir.mkdir(parents=True)
    full_artifact = full_run_dir / "physical_id"
    full_input = materialize_run_input(
        SPHERICAL_SPINNING_XCTS_3D_INPUT,
        full_run_dir / "input.in",
        {
            "pyfluid.run_name": full_name,
            "pyfluid.output_dir": repo_relative(full_run_dir),
            "initial_data.output_file": repo_relative(full_artifact),
        },
    )
    completed = run_amrex(
        dim=3,
        input_path=full_input,
        ranks=ranks,
        timeout=timeout,
    )
    (full_run_dir / "run.log").write_text(completed.stdout)
    if completed.returncode != 0 or not full_artifact.is_dir():
        raise RuntimeError(
            f"{full_name} failed; see "
            f"{repo_relative(full_run_dir / 'run.log')}"
        )
    recovery = RECOVERY.findall(completed.stdout)
    if not recovery or tuple(int(value) for value in recovery[-1]) != (0, 0):
        raise RuntimeError(f"{full_name} used relativistic recovery during ID")
    full_solve = XCTS_SOLVE.search(completed.stdout)
    if full_solve is None:
        raise RuntimeError(f"{full_name} did not report its XCTS solve")
    full_residual = max(
        abs(float(full_solve.group(index))) for index in (3, 4, 5)
    )
    if full_residual > 1.0e-10:
        raise RuntimeError(f"{full_name} elliptic residual is {full_residual:.3e}")

    full_metadata = read_artifact_metadata(full_artifact)
    full_spin = float(
        full_metadata["black_hole_throat_christodoulou_spin"]
    )
    if abs(full_spin - target_spin) > 1.0e-4:
        raise RuntimeError(
            f"full-storage spinning XCTS spin is {full_spin:.16e}"
        )
    full_names, full_dimension, _ = parse_plot_header(
        full_artifact / "adm_geometry.h5"
    )
    full_state = read_level_state(
        full_artifact / "adm_geometry.h5",
        full_names,
        full_dimension,
        0,
    )
    field_scale = max(float(np.max(np.abs(state))), 1.0e-14)
    reduced_difference = float(
        np.max(np.abs(full_state[..., 0] - state)) / field_scale
    )
    phi_variation = float(
        np.max(
            np.abs(
                full_state
                - np.repeat(
                    full_state[..., :1], full_state.shape[-1], axis=-1
                )
            )
        ) / field_scale
    )
    if reduced_difference > 5.0e-12 or phi_variation > 5.0e-14:
        raise RuntimeError(
            "matched reduced/full-storage spinning XCTS fields disagree: "
            f"fields={reduced_difference:.3e}, phi={phi_variation:.3e}"
        )

    high_spin_target = 0.675
    _, high_spin_artifact, _ = run_spherical_xcts_case(
        "spinning_xcts_spherical_shell_chi0p675",
        {"spacetime.dimensionless_spin3": str(high_spin_target)},
        ranks=ranks,
        timeout=timeout,
    )
    high_spin_metadata = read_artifact_metadata(high_spin_artifact)
    high_spin_measured = float(
        high_spin_metadata["black_hole_throat_christodoulou_spin"]
    )
    high_spin_error = abs(high_spin_measured - high_spin_target)
    if high_spin_error > 1.0e-4:
        raise RuntimeError(
            "high-spin spherical XCTS target mismatch: "
            f"target={high_spin_target:.16e}, "
            f"measured={high_spin_measured:.16e}"
        )
    solve = XCTS_SOLVE.search(log_text)
    assert solve is not None
    residual = max(abs(float(solve.group(index))) for index in (3, 4, 5))
    return {
        "case": "spinning_xcts_spherical_shell_chi0p3",
        "status": "PASS",
        "steps": 0,
        "c2p_fallbacks": 0,
        "atmosphere_resets": 0,
        "xcts_target_spin": target_spin,
        "xcts_measured_spin": measured_spin,
        "xcts_spin_absolute_error": spin_error,
        "xcts_horizon_angular_momentum": angular_momentum,
        "xcts_christodoulou_mass": horizon_mass,
        "xcts_christodoulou_identity_error": identity_error,
        "xcts_shift_linf": shift_linf,
        "xcts_extrinsic_curvature_linf": extrinsic_linf,
        "xcts_elliptic_residual": residual,
        "xcts_3d_measured_spin": full_spin,
        "xcts_3d_elliptic_residual": full_residual,
        "xcts_3d_reduced_difference": reduced_difference,
        "xcts_3d_phi_variation": phi_variation,
        "xcts_high_spin_target": high_spin_target,
        "xcts_high_spin_measured": high_spin_measured,
        "xcts_high_spin_absolute_error": high_spin_error,
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
    source_metadata = read_artifact_metadata(source_artifact)
    reduced_metadata = read_artifact_metadata(reduced_artifact)
    for field in (
        "black_hole_throat_area",
        "black_hole_irreducible_mass",
        "black_hole_throat_equatorial_circumference",
        "black_hole_throat_polar_circumference",
        "black_hole_throat_kerr_mass",
        "black_hole_throat_kerr_spin",
        "black_hole_throat_angular_momentum",
        "black_hole_throat_christodoulou_mass",
        "black_hole_throat_christodoulou_spin",
    ):
        source_value = float(source_metadata[field])
        reduced_value = float(reduced_metadata[field])
        if abs(source_value - reduced_value) > 5.0e-13 * max(
            1.0, abs(reduced_value)
        ):
            raise RuntimeError(
                "matched 2D and full-storage spherical XCTS throat "
                f"diagnostics disagree for {field}: "
                f"3D={source_value:.16e}, 2D={reduced_value:.16e}"
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
    if "re-solving black-hole physical initial data on the configured " \
            "r theta phi shell" not in completed.stdout:
        raise RuntimeError("changed 3D spherical black-hole load skipped XCTS")
    solve = XCTS_SOLVE.search(completed.stdout)
    if solve is None:
        raise RuntimeError(f"{name} did not report its XCTS re-solve")
    residual = max(abs(float(solve.group(index))) for index in (3, 4, 5))
    if residual > 3.0e-8:
        raise RuntimeError(f"{name} elliptic residual is {residual:.3e}")

    metadata = read_artifact_metadata(artifact)
    (
        throat_area_error,
        throat_circumference_error,
        throat_mass_error,
        throat_spin_error,
    ) = validate_schwarzschild_throat_diagnostics(metadata)
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
        "xcts_throat_area_relative_error": throat_area_error,
        "xcts_throat_circumference_relative_error": throat_circumference_error,
        "xcts_throat_kerr_mass_relative_error": throat_mass_error,
        "xcts_throat_kerr_spin_absolute_error": throat_spin_error,
        "run_dir": repo_relative(run_dir),
    }


def validate_spinning_xcts_cartesian_handoff(
    *, ranks: int, timeout: int | None
) -> tuple[dict[str, object], dict[str, object]]:
    source_artifact = (
        RUN_ROOT / "spinning_xcts_spherical_shell_chi0p3" / "physical_id"
    )
    if not source_artifact.is_dir():
        raise RuntimeError("spinning XCTS handoff source artifact is missing")

    materialized_name = "spinning_xcts_cartesian_materialization"
    materialized_dir = RUN_ROOT / materialized_name
    if materialized_dir.exists():
        shutil.rmtree(materialized_dir)
    materialized_dir.mkdir(parents=True)
    materialized_artifact = materialized_dir / "physical_id"
    materialized_input = materialize_run_input(
        SPINNING_XCTS_CARTESIAN_INPUT,
        materialized_dir / "input.in",
        {
            "pyfluid.run_name": materialized_name,
            "pyfluid.output_dir": repo_relative(materialized_dir),
            "initial_data.input_file": repo_relative(source_artifact),
            "initial_data.output_file": repo_relative(materialized_artifact),
        },
    )
    completed = run_amrex(
        dim=3,
        input_path=materialized_input,
        ranks=ranks,
        timeout=timeout,
    )
    (materialized_dir / "run.log").write_text(completed.stdout)
    if completed.returncode != 0 or not materialized_artifact.is_dir():
        raise RuntimeError(
            f"{materialized_name} failed; see "
            f"{repo_relative(materialized_dir / 'run.log')}"
        )
    recovery = RECOVERY.findall(completed.stdout)
    if not recovery or tuple(int(value) for value in recovery[-1]) != (0, 0):
        raise RuntimeError(f"{materialized_name} used relativistic recovery")
    materialized_metadata = read_artifact_metadata(materialized_artifact)
    if materialized_metadata.get("mesh_coordinate") != "cartesian" \
            or materialized_metadata.get("mesh_axes") != "x y z" \
            or abs(float(materialized_metadata["black_hole_spin3"]) - 0.3) \
            > 1.0e-14:
        raise RuntimeError(
            f"{materialized_name} lost its Cartesian/spin identity"
        )

    evolved_name = "spinning_xcts_cartesian_evolution_step1"
    evolved, evolved_scalar = run_case(
        evolved_name,
        {
            "initial_data.input_file": repo_relative(materialized_artifact),
            "pyfluid.max_step": "1",
            "pyfluid.plot_int": "-1",
        },
        ranks=ranks,
        timeout=timeout,
        base_input=SPINNING_XCTS_EVOLUTION_INPUT,
        require_bh_exterior_constraints=False,
    )
    if int(evolved["steps"]) != 1:
        raise RuntimeError(f"{evolved_name} did not complete one step")
    evolved["case"] = evolved_name
    evolved["xcts_cartesian_artifact_bytes"] = sum(
        path.stat().st_size
        for path in materialized_artifact.rglob("*")
        if path.is_file()
    )
    evolved["xcts_cartesian_source_spin"] = float(
        materialized_metadata["black_hole_throat_christodoulou_spin"]
    )
    return evolved, evolved_scalar


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
            "pyfluid.scalar_diagnostics": "standard,horizon",
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
    horizon_mass_error = abs(float(initial["kerr_horizon_mass"]) - 1.0)
    horizon_spin_error = abs(
        float(initial["kerr_dimensionless_spin"]) - spin
    )
    if horizon_mass_error > 2.0e-2 or horizon_spin_error > 5.0e-2:
        raise RuntimeError(
            f"{name} Kerr horizon mass/spin errors are "
            f"{horizon_mass_error:.3e}/{horizon_spin_error:.3e}"
        )
    coordinate_spin_error = abs(
        float(initial["coordinate_christodoulou_dimensionless_spin3"])
        - spin
    )
    transverse_coordinate_spin = math.hypot(
        float(initial["coordinate_christodoulou_dimensionless_spin1"]),
        float(initial["coordinate_christodoulou_dimensionless_spin2"]),
    )
    if coordinate_spin_error > 2.0e-2 or transverse_coordinate_spin > 1.0e-10:
        raise RuntimeError(
            f"{name} coordinate spin errors are "
            f"{coordinate_spin_error:.3e}/{transverse_coordinate_spin:.3e}"
        )
    scalar_rows = read_tsv(run_dir / "scalars.tsv", require_complete=True)
    initial_scalar = next(
        scalar_row
        for scalar_row in scalar_rows
        if int(float(scalar_row["step"])) == 0
    )
    scalar_spin = float(
        initial_scalar[
            "black_hole_horizon_coordinate_christodoulou_spin3"
        ]
    )
    if abs(
        scalar_spin
        - float(initial["coordinate_christodoulou_dimensionless_spin3"])
    ) > 1.0e-13:
        raise RuntimeError(
            f"{name} scalar catalog disagrees with its horizon tracker"
        )
    if not (
        float(initial["polar_circumference"])
        < float(initial["equatorial_circumference"])
    ):
        raise RuntimeError(f"{name} Kerr horizon circumferences are not ordered")
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
            "pyfluid.scalar_diagnostics": "standard,horizon",
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
    row["ah_kerr_mass_relative_error"] = horizon_mass_error
    row["ah_kerr_spin_absolute_error"] = horizon_spin_error
    row["ah_coordinate_spin_absolute_error"] = coordinate_spin_error
    row["ah_transverse_coordinate_spin"] = transverse_coordinate_spin
    row["ah_horizon_cells"] = horizon_cells
    row["ah_excision_cells"] = excision_cells
    return row, scalar


def validate_reflected_apparent_horizon(
    *, ranks: int, timeout: int | None
) -> tuple[dict[str, object], dict[str, object]]:
    name = "kerr_schild_apparent_horizon_equatorial_half"
    row, scalar = run_case(
        name,
        {
            "geometry.prob_lo": "-8.0 -8.0 0.0",
            "geometry.prob_hi": "8.0 8.0 8.0",
            "amr.n_cell": "32 32 16",
            "amr.max_grid_size": "16",
            "pyfluid.bc_zlo": "symmetry",
            "pyfluid.max_step": "1",
            "pyfluid.plot_int": "-1",
            "horizon.enabled": "1",
            "horizon.find_int": "8",
            "horizon.radial_points": "48",
            "horizon.polar_points": "16",
            "horizon.coarsest_polar_points": "8",
            "horizon.theta_linf_times_mass_tolerance": "5.0e-4",
            "horizon.theta_l2_times_mass_tolerance": "1.0e-2",
        },
        ranks=ranks,
        timeout=timeout,
    )
    horizon_rows = read_tsv(RUN_ROOT / name / "apparent_horizon.tsv")
    if len(horizon_rows) != 1 or int(horizon_rows[0]["step"]) != 0:
        raise RuntimeError(
            f"{name} did not persist exactly its initial reflected surface"
        )
    initial = horizon_rows[0]
    spin = 0.5
    exact_area = 8.0 * math.pi * (1.0 + math.sqrt(1.0 - spin * spin))
    area_error = abs(float(initial["area"]) - exact_area) / exact_area
    theta_linf = float(initial["theta_linf_times_mass"])
    if area_error > 3.0e-2 or theta_linf > 5.0e-4:
        raise RuntimeError(
            f"{name} reflected surface mismatch: area={area_error:.3e}, "
            f"theta={theta_linf:.3e}"
        )
    reference_rows = read_tsv(
        RUN_ROOT / "kerr_schild_apparent_horizon" / "apparent_horizon.tsv"
    )
    if not reference_rows or int(reference_rows[0]["step"]) != 0:
        raise RuntimeError(f"{name} is missing its matched full-domain reference")
    reference = reference_rows[0]
    for field in (
        "theta_linf_times_mass",
        "theta_l2_times_mass",
        "area",
        "irreducible_mass",
        "equatorial_circumference",
        "polar_circumference",
        "kerr_horizon_mass",
        "kerr_dimensionless_spin",
        "coordinate_angular_momentum1",
        "coordinate_angular_momentum2",
        "coordinate_angular_momentum3",
        "coordinate_angular_momentum_magnitude",
        "coordinate_christodoulou_mass",
        "coordinate_christodoulou_dimensionless_spin1",
        "coordinate_christodoulou_dimensionless_spin2",
        "coordinate_christodoulou_dimensionless_spin3",
        "coordinate_christodoulou_dimensionless_spin_magnitude",
        "minimum_radius",
        "maximum_radius",
        "mean_radius",
    ):
        reflected_value = float(initial[field])
        reference_value = float(reference[field])
        scale = max(1.0, abs(reference_value))
        if abs(reflected_value - reference_value) > 5.0e-12 * scale:
            raise RuntimeError(
                f"{name} changed {field} across reflection: "
                f"full={reference_value:.16e}, "
                f"half={reflected_value:.16e}"
            )
    row["ah_theta_linf_times_mass"] = theta_linf
    row["ah_area_relative_error"] = area_error
    return row, scalar


def run_case(
    name: str,
    overrides: dict[str, str],
    *,
    ranks: int,
    timeout: int | None,
    base_input: Path = BASE_INPUT,
    require_bh_exterior_constraints: bool = True,
) -> tuple[dict[str, object], dict[str, object]]:
    run_dir = RUN_ROOT / name
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    input_path = materialize_run_input(
        base_input,
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
    constraint_fields = CONSTRAINT_FIELDS \
        if require_bh_exterior_constraints else CONSTRAINT_FIELDS[:-3]
    for row in rows:
        for field in constraint_fields:
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


def horizon_endpoint_metrics(name: str) -> tuple[float, float, float]:
    rows = read_tsv(RUN_ROOT / name / "apparent_horizon.tsv")
    if len(rows) < 2:
        raise RuntimeError(f"{name} did not persist a horizon history")
    exact_area = 16.0 * math.pi
    initial_area_error = abs(float(rows[0]["area"]) - exact_area) / exact_area
    final_area_error = abs(float(rows[-1]["area"]) - exact_area) / exact_area
    initial_radius = float(rows[0]["mean_radius"])
    radius_change = (
        abs(float(rows[-1]["mean_radius"]) - initial_radius)
        / initial_radius
    )
    return initial_area_error, final_area_error, radius_change


def validate_static_trumpet(
    *, ranks: int, timeout: int | None
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    scalar_rows: list[dict[str, object]] = []
    common = {
        "pyfluid.max_step": "16",
        "pyfluid.plot_int": "-1",
    }
    moving, moving_scalar = run_case(
        "static_trumpet_moving_puncture_step16",
        common,
        ranks=ranks,
        timeout=timeout,
        base_input=STATIC_TRUMPET_INPUT,
    )
    matched, matched_scalar = run_case(
        "static_trumpet_matched_gauge_step16",
        {
            **common,
            "spacetime.lapse_gauge": "trumpet",
            "spacetime.shift_gauge": "frozen",
        },
        ranks=ranks,
        timeout=timeout,
        base_input=STATIC_TRUMPET_INPUT,
    )
    for row in (moving, matched):
        initial_error, final_error, radius_change = horizon_endpoint_metrics(
            str(row["case"])
        )
        row["ah_initial_area_relative_error"] = initial_error
        row["ah_final_area_relative_error"] = final_error
        row["ah_mean_radius_relative_change"] = radius_change
        if initial_error > 5.0e-2:
            raise RuntimeError(
                f"{row['case']} initial horizon area error is {initial_error:.3e}"
            )
    if float(moving["ah_final_area_relative_error"]) > 2.0e-2:
        raise RuntimeError(
            "static-trumpet moving-puncture horizon area exceeds its "
            "short-run regression bound"
        )
    if float(matched["ah_mean_radius_relative_change"]) > 5.0e-2:
        raise RuntimeError(
            "matched static-trumpet gauge does not retain its coordinate "
            "horizon on the short control"
        )
    rows.extend((moving, matched))
    scalar_rows.extend((moving_scalar, matched_scalar))

    physical_id = RUN_ROOT / "static_trumpet_full" / "physical_id"
    full, full_scalar = run_case(
        "static_trumpet_full",
        {
            "initial_data.output_file": repo_relative(physical_id),
            "horizon.enabled": "0",
            "pyfluid.max_step": "1",
            "pyfluid.plot_int": "-1",
        },
        ranks=ranks,
        timeout=timeout,
        base_input=STATIC_TRUMPET_INPUT,
    )
    if not physical_id.is_dir():
        raise RuntimeError("static trumpet did not write its physical-ID artifact")
    loaded, loaded_scalar = run_case(
        "static_trumpet_changed_grid",
        {
            "initial_data.input_file": repo_relative(physical_id),
            "amr.n_cell": "24 24 24",
            "amr.max_grid_size": "12",
            "amr.blocking_factor": "4",
            "horizon.enabled": "0",
            "pyfluid.max_step": "1",
            "pyfluid.plot_int": "-1",
        },
        ranks=ranks,
        timeout=timeout,
        base_input=STATIC_TRUMPET_INPUT,
    )
    octant, octant_scalar = run_case(
        "static_trumpet_octant",
        {
            "geometry.prob_lo": "0.0 0.0 0.0",
            "geometry.prob_hi": "8.0 8.0 8.0",
            "amr.n_cell": "16 16 16",
            "pyfluid.bc_xlo": "symmetry",
            "pyfluid.bc_ylo": "symmetry",
            "pyfluid.bc_zlo": "symmetry",
            "horizon.enabled": "0",
            "pyfluid.max_step": "1",
            "pyfluid.plot_int": "-1",
        },
        ranks=ranks,
        timeout=timeout,
        base_input=STATIC_TRUMPET_INPUT,
    )
    require_matched_symmetry_extrema(
        full, octant, "static-trumpet octant reflection"
    )
    rows.extend((full, loaded, octant))
    scalar_rows.extend((full_scalar, loaded_scalar, octant_scalar))
    return rows, scalar_rows


def validate_static_trumpet_rejection(
    name: str,
    overrides: dict[str, str],
    *,
    ranks: int,
    timeout: int | None,
) -> dict[str, object]:
    run_dir = RUN_ROOT / name
    run_dir.mkdir(parents=True, exist_ok=True)
    input_path = materialize_run_input(
        STATIC_TRUMPET_INPUT,
        run_dir / "input.in",
        {
            "pyfluid.run_name": name,
            "pyfluid.output_dir": repo_relative(run_dir),
            "horizon.enabled": "0",
            "pyfluid.max_step": "1",
            "pyfluid.plot_int": "-1",
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
    expected = "static-trumpet spacetime requires"
    if completed.returncode == 0 or expected not in completed.stdout:
        raise RuntimeError(f"{name} did not fail its static-trumpet contract")
    return {
        "case": name,
        "status": "PASS",
        "steps": 0,
        "c2p_fallbacks": 0,
        "atmosphere_resets": 0,
        "run_dir": repo_relative(run_dir),
    }


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
        validate_spherical_xcts_spinning_black_hole(
            ranks=args.ranks, timeout=args.timeout
        )
    )
    handoff, handoff_scalar = validate_spinning_xcts_cartesian_handoff(
        ranks=args.ranks, timeout=args.timeout
    )
    results.append(handoff)
    scalar_rows.append(handoff_scalar)
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
                "quasi-equilibrium horizon at r=M/2",
            ),
            (
                "schwarzschild_xcts_invalid_origin",
                {"spacetime.origin1": "1.0"},
                "origin-centered native spherical XCTS shell solve",
            ),
        )
    )
    validate_apparent_horizon_core()
    horizon, horizon_scalar = validate_apparent_horizon_run(
        ranks=args.ranks, timeout=args.timeout
    )
    results.append(horizon)
    scalar_rows.append(horizon_scalar)
    reflected_horizon, reflected_horizon_scalar = (
        validate_reflected_apparent_horizon(
            ranks=args.ranks, timeout=args.timeout
        )
    )
    results.append(reflected_horizon)
    scalar_rows.append(reflected_horizon_scalar)
    static_rows, static_scalar_rows = validate_static_trumpet(
        ranks=args.ranks, timeout=args.timeout
    )
    results.extend(static_rows)
    scalar_rows.extend(static_scalar_rows)
    results.extend(
        validate_static_trumpet_rejection(
            name,
            overrides,
            ranks=args.ranks,
            timeout=args.timeout,
        )
        for name, overrides in (
            (
                "static_trumpet_nonzero_spin_rejected",
                {"spacetime.dimensionless_spin3": "0.1"},
            ),
            (
                "static_trumpet_filled_interior_rejected",
                {
                    "spacetime.black_hole_interior": "filled",
                    "spacetime.black_hole_fill_radius": "0.5",
                },
            ),
        )
    )
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
        "ah_initial_area_relative_error",
        "ah_final_area_relative_error",
        "ah_mean_radius_relative_change",
        "ah_horizon_cells",
        "ah_excision_cells",
        "ah_kerr_mass_relative_error",
        "ah_kerr_spin_absolute_error",
        "ah_coordinate_spin_absolute_error",
        "ah_transverse_coordinate_spin",
        "xcts_metric_relative_error",
        "xcts_lapse_linf_error",
        "xcts_zero_field_linf",
        "xcts_theta_invariance",
        "xcts_reduced_field_difference",
        "xcts_phi_variation",
        "xcts_elliptic_residual",
        "xcts_artifact_bytes",
        "xcts_throat_area_relative_error",
        "xcts_throat_circumference_relative_error",
        "xcts_throat_kerr_mass_relative_error",
        "xcts_throat_kerr_spin_absolute_error",
        "xcts_target_spin",
        "xcts_measured_spin",
        "xcts_spin_absolute_error",
        "xcts_horizon_angular_momentum",
        "xcts_christodoulou_mass",
        "xcts_christodoulou_identity_error",
        "xcts_shift_linf",
        "xcts_extrinsic_curvature_linf",
        "xcts_3d_measured_spin",
        "xcts_3d_elliptic_residual",
        "xcts_3d_reduced_difference",
        "xcts_3d_phi_variation",
        "xcts_high_spin_target",
        "xcts_high_spin_measured",
        "xcts_high_spin_absolute_error",
        "xcts_cartesian_artifact_bytes",
        "xcts_cartesian_source_spin",
        "run_dir",
    )
    write_tsv(RESULTS, results, fields)
    append_validation_history(HISTORY, scalar_rows)
    for row in results:
        print(f"PASS {row['case']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
