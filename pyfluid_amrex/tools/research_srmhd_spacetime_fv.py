#!/usr/bin/env python3
"""Carry the coherent primitive space-time FV prototype into flat SRMHD.

The standalone adapter keeps the production C2P and HLLD pointwise physics as
an executable oracle while testing a new fully discrete operator around it:
finite-volume conserved reconstruction, one center C2P, primitive CWENO-Z,
a weak local space-time predictor, one time-integrated shared flux, and the
same MP5/PWL/PWC conservative correction hierarchy used in the lower systems.
It is intentionally slow research code, not an AMReX implementation.
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
from scipy.linalg import schur

sys.path.insert(0, str(Path(__file__).resolve().parent))

import research_contact_correction as contact_correction  # noqa: E402
import research_physical_wave_subcell_srmhd as srmhd  # noqa: E402
import research_primitive_spacetime_fv as spacetime  # noqa: E402
from validate_rmhd import (  # noqa: E402
    EXACT_BALSARA_CASES,
    EXACT_BALSARA_ROOT,
    exact_rmhd_cell_averages,
    exact_rmhd_conserved,
    read_last_exact_balsara_profile,
)


Array = np.ndarray
ROOT = Path(__file__).resolve().parents[2]
ORACLE = ROOT / "build/pyfluid_amrex/pyfluid_amrex_rmhd_riemann_check"
REDUCED = np.asarray((0, 1, 2, 3, 4, 6, 7))
CONSERVED_NAMES = (
    "dens",
    "mom1",
    "mom2",
    "mom3",
    "tau",
    "b1",
    "b2",
    "b3",
)


@dataclasses.dataclass(frozen=True)
class Recovery:
    primitive: Array
    success: Array


@dataclasses.dataclass(frozen=True)
class StepResult:
    state: Array | None
    maximum_speed: float
    cell_orders: Array
    riemann_fallbacks: int = 0
    physical_front_faces: int = 0


@dataclasses.dataclass
class EvolutionStats:
    steps: int = 0
    cfl_retries: int = 0
    high_cells: int = 0
    mp5_cells: int = 0
    pwl_cells: int = 0
    pwc_cells: int = 0
    riemann_fallbacks: int = 0
    physical_front_faces: int = 0
    minimum_order: int = 3


@dataclasses.dataclass(frozen=True)
class SmoothResult:
    method: str
    cells: int
    cfl: float
    completed: bool
    maximum_l1: float
    observed_order: float
    maximum_overshoot: float
    maximum_tv_ratio: float
    mass_error: float
    momentum_error: float
    energy_error: float
    normal_b_spread: float
    steps: int
    cfl_retries: int
    mp5_cell_steps: int
    pwl_cell_steps: int
    pwc_cell_steps: int
    riemann_fallbacks: int
    physical_front_faces: int


@dataclasses.dataclass(frozen=True)
class ShockResult:
    method: str
    case: int
    cells: int
    cfl: float
    completed: bool
    maximum_l1: float
    maximum_l1_component: str
    density_l1: float
    maximum_overshoot: float
    maximum_overshoot_component: str
    maximum_tv_ratio: float
    maximum_tv_component: str
    maximum_inactive_error: float
    maximum_integral_error: float
    normal_b_spread: float
    steps: int
    cfl_retries: int
    mp5_cell_steps: int
    pwl_cell_steps: int
    pwc_cell_steps: int
    riemann_fallbacks: int
    physical_front_faces: int


@dataclasses.dataclass(frozen=True)
class FrontResult:
    case: str
    method: str
    cells: int
    cfl: float
    completed: bool
    maximum_l1: float
    maximum_overshoot: float
    maximum_tv_ratio: float
    mass_error: float
    momentum_error: float
    energy_error: float
    normal_b_spread: float
    steps: int
    mp5_cell_steps: int
    pwl_cell_steps: int
    pwc_cell_steps: int
    riemann_fallbacks: int
    physical_front_faces: int


@dataclasses.dataclass(frozen=True)
class BasisAuditResult:
    sample: str
    states: int
    direct_eigen_bases: int
    schur_bases: int
    rejected_bases: int
    maximum_residual: float


def _run_oracle(mode: str, rows: list[str]) -> Array:
    if not ORACLE.is_file():
        raise FileNotFoundError(
            f"missing {ORACLE}; build pyfluid_amrex_rmhd_riemann_check"
        )
    completed = subprocess.run(
        [str(ORACLE), mode],
        input="".join(rows),
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"SRMHD oracle failed ({mode}):\n{completed.stderr}"
        )
    table = np.loadtxt(completed.stdout.splitlines(), ndmin=2)
    expected = np.arange(table.shape[0], dtype=int)
    if not np.array_equal(table[:, 0].astype(int), expected):
        raise RuntimeError(f"SRMHD oracle reordered rows in {mode}")
    return table


def recover_conserved(state: Array, gamma: float) -> Recovery:
    values = np.asarray(state, dtype=float)
    rows = [
        f"{index} {gamma:.17g} "
        + " ".join(f"{value:.17g}" for value in row)
        + "\n"
        for index, row in enumerate(values)
    ]
    table = _run_oracle("--c2p-table", rows)
    # Oracle columns: rho, eps, p, v1, v2, v3, b1, b2, b3 are embedded in
    # the complete diagnostic row.  Return the project primitive order.
    primitive = table[:, (7, 10, 11, 12, 9, 13, 14, 15)]
    return Recovery(primitive=primitive, success=table[:, 1].astype(bool))


def state_and_flux(primitive: Array, gamma: float) -> tuple[Array, Array]:
    values = np.asarray(primitive, dtype=float)
    oracle_order = (0, 4, 1, 2, 3, 5, 6, 7)
    rows = [
        f"{index} {gamma:.17g} "
        + " ".join(f"{row[item]:.17g}" for item in oracle_order)
        + "\n"
        for index, row in enumerate(values)
    ]
    table = _run_oracle("--primitive-table", rows)
    return table[:, 1:9], table[:, 9:17]


def hlld_fluxes(
    left: Array,
    right: Array,
    gamma: float,
) -> tuple[Array, Array]:
    left_values = np.asarray(left, dtype=float)
    right_values = np.asarray(right, dtype=float)
    oracle_order = np.asarray((0, 4, 1, 2, 3, 5, 6, 7))
    rows = []
    for index, (left_row, right_row) in enumerate(
        zip(left_values, right_values, strict=True)
    ):
        values = np.concatenate(
            (left_row[oracle_order], right_row[oracle_order])
        )
        rows.append(
            f"{index} {gamma:.17g} "
            + " ".join(f"{value:.17g}" for value in values)
            + "\n"
        )
    table = _run_oracle("--flux-table", rows)
    return table[:, 18:26], table[:, 26].astype(bool)


def _total_energy_state_and_flux(
    primitive: Array,
    gamma: float,
) -> tuple[Array, Array]:
    state, flux = _total_energy_state_and_flux_array(
        primitive[None, :], gamma
    )
    return state[0], flux[0]


def _total_energy_state_and_flux_array(
    primitive: Array,
    gamma: float,
) -> tuple[Array, Array]:
    """Vectorized flat-SRMHD state and x-flux in total-energy form."""

    values = np.asarray(
        primitive, dtype=np.result_type(primitive, float)
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        rho = values[..., 0]
        velocity = values[..., 1:4]
        pressure = values[..., 4]
        magnetic = values[..., 5:8]
        velocity2 = np.sum(velocity * velocity, axis=-1)
        lorentz = 1.0 / np.sqrt(1.0 - velocity2)
        enthalpy = 1.0 + gamma * pressure / ((gamma - 1.0) * rho)
        magnetic2 = np.sum(magnetic * magnetic, axis=-1)
        magnetic_velocity = np.sum(magnetic * velocity, axis=-1)
        alpha_b0 = lorentz * magnetic_velocity
        comoving = magnetic / lorentz[..., None] + (
            alpha_b0[..., None] * velocity
        )
        comoving2 = (
            magnetic2 / (lorentz * lorentz) + magnetic_velocity**2
        )
        total_pressure = pressure + 0.5 * comoving2
        z_value = rho * enthalpy * lorentz * lorentz
        dens = rho * lorentz
        momentum = (
            (z_value + magnetic2)[..., None] * velocity
            - magnetic_velocity[..., None] * magnetic
        )
        tau = (
            z_value
            + magnetic2
            - pressure
            - 0.5 * comoving2
            - dens
        )
        state = np.concatenate(
            (
                dens[..., None],
                momentum,
                tau[..., None],
                magnetic,
            ),
            axis=-1,
        )

    flux = np.empty_like(state)
    normal_velocity = velocity[..., 0]
    normal_magnetic = magnetic[..., 0]
    with np.errstate(divide="ignore", invalid="ignore"):
        flux[..., 0] = state[..., 0] * normal_velocity
        flux[..., 1:4] = (
            state[..., 1:4] * normal_velocity[..., None]
            - comoving * (normal_magnetic / lorentz)[..., None]
        )
        flux[..., 1] += total_pressure
        flux[..., 4] = (
            state[..., 4] * normal_velocity
            + total_pressure * normal_velocity
            - alpha_b0 * normal_magnetic / lorentz
        )
        flux[..., 5:8] = (
            magnetic * normal_velocity[..., None]
            - velocity * normal_magnetic[..., None]
        )

    # The primitive Jacobian is unchanged by the linear E=tau+D row
    # transformation.  Using total energy here avoids subtractive derivatives
    # while the finite-volume state and all face fluxes retain production tau.
    state = state.copy()
    state[..., 4] += state[..., 0]
    flux[..., 4] += flux[..., 0]
    return state, flux


def reduced_primitive_jacobian(primitive: Array, gamma: float) -> Array:
    """Numerically form ``dU/dV^-1 dF/dV`` with constant normal B."""

    return reduced_primitive_jacobians(primitive[None, :], gamma)[0]


def reduced_primitive_jacobians(primitive: Array, gamma: float) -> Array:
    """Vectorized reduced primitive Jacobians with constant normal B."""

    values = np.asarray(primitive, dtype=float)
    original_shape = values.shape[:-1]
    flat = values.reshape(-1, 8)
    count = len(flat)
    valid = (
        np.all(np.isfinite(flat), axis=1)
        & (flat[:, 0] > 0.0)
        & (flat[:, 4] > 0.0)
        & (np.sum(flat[:, 1:4] ** 2, axis=1) < 1.0)
    )
    # The flat-SRMHD map above is analytic in every primitive variable, so a
    # complex-step derivative avoids subtractive cancellation in extreme
    # states and near degenerate wave families.
    steps = 1.0e-30 * np.maximum(np.abs(flat[:, REDUCED]), 1.0)
    plus = np.repeat(
        flat[:, None, :].astype(complex), len(REDUCED), axis=1
    )
    rows = np.arange(count)[:, None]
    columns = np.arange(len(REDUCED))[None, :]
    components = REDUCED[None, :]
    plus[rows, columns, components] += 1.0j * steps
    plus_state, plus_flux = _total_energy_state_and_flux_array(plus, gamma)
    # Derivative arrays enter as [state, input, output]; transpose to
    # the conventional [state, output, input] Jacobian ordering.
    conserved_jacobian = np.swapaxes(
        np.imag(plus_state[..., REDUCED]) / steps[..., None],
        -1,
        -2,
    )
    flux_jacobian = np.swapaxes(
        np.imag(plus_flux[..., REDUCED]) / steps[..., None],
        -1,
        -2,
    )
    result = np.full((count, 7, 7), np.nan)
    try:
        result[valid] = np.linalg.solve(
            conserved_jacobian[valid], flux_jacobian[valid]
        )
    except np.linalg.LinAlgError:
        for index in np.flatnonzero(valid):
            try:
                result[index] = np.linalg.solve(
                    conserved_jacobian[index], flux_jacobian[index]
                )
            except np.linalg.LinAlgError:
                pass
    return result.reshape(original_shape + (7, 7))


def point_primitives(
    state: Array,
    gamma: float,
    periodic: bool,
    parameters: spacetime.PredictorParameters,
) -> tuple[Array, Array]:
    conserved_polynomial = spacetime.cweno_z_coefficients(
        state,
        periodic,
        parameters,
        point_samples=False,
    )
    raw_point_state = conserved_polynomial[:, 0]
    point_state = raw_point_state.copy()
    recovery = recover_conserved(point_state, gamma)
    success = recovery.success
    point_primitive = recovery.primitive
    scale = np.ones(len(state))
    for _ in range(32):
        if np.all(success):
            return point_primitive, scale
        scale[~success] *= 0.5
        point_state[~success] = state[~success] + scale[
            ~success, np.newaxis
        ] * (raw_point_state[~success] - state[~success])
        recovery = recover_conserved(point_state, gamma)
        success = recovery.success
        point_primitive = recovery.primitive
    base = recover_conserved(state, gamma)
    point_primitive[~success] = base.primitive[~success]
    scale[~success] = 0.0
    return point_primitive, scale


def component_mp5_parabolic_coefficients(
    primitive: Array,
    gamma: float,
    periodic: bool,
) -> Array:
    result = np.zeros((len(primitive), 5, 8))
    srmhd.component_mp5.periodic = periodic
    for cell in range(len(primitive)):
        _left_neighbor, left_trace = srmhd.component_mp5(
            primitive, cell - 1, gamma
        )
        right_trace, _right_neighbor = srmhd.component_mp5(
            primitive, cell, gamma
        )
        curvature = 3.0 * (
            left_trace + right_trace - 2.0 * primitive[cell]
        )
        result[cell, 0] = primitive[cell] - curvature / 12.0
        result[cell, 1] = right_trace - left_trace
        result[cell, 2] = curvature
        result[cell, 1:, 5] = 0.0
        result[cell, 0, 5] = primitive[cell, 5]
    return result


def _characteristic_basis(
    primitive: Array,
    gamma: float,
    jacobian: Array | None = None,
) -> tuple[Array, Array] | None:
    """Return a well-conditioned real numerical SRMHD eigenbasis."""

    matrix = (
        reduced_primitive_jacobian(primitive, gamma)
        if jacobian is None
        else jacobian
    )
    if not np.all(np.isfinite(matrix)):
        return None
    eigenvalues, right = np.linalg.eig(matrix)
    scale = max(float(np.max(np.abs(eigenvalues))), 1.0)
    if np.max(np.abs(np.imag(eigenvalues))) > 1.0e-8 * scale:
        return None
    if np.max(np.abs(np.imag(right))) <= 1.0e-7:
        right = np.real(right)
        order = np.argsort(np.real(eigenvalues))
        right = right[:, order]
        if np.all(np.isfinite(right)) and np.linalg.cond(right) <= 1.0e8:
            try:
                return np.linalg.inv(right), right
            except np.linalg.LinAlgError:
                pass

    # Individual RMHD eigenvectors are not a stable coordinate system when
    # wave speeds coalesce.  The orthogonal real-Schur vectors span the same
    # invariant subspaces without amplifying roundoff; exact family ordering
    # is immaterial inside a degenerate block because the predictor evolves
    # the complete primitive Jacobian rather than seven decoupled equations.
    triangular, right_schur = schur(matrix, output="real")
    if (
        not np.all(np.isfinite(triangular))
        or not np.all(np.isfinite(right_schur))
    ):
        return None
    residual = np.linalg.norm(
        matrix @ right_schur - right_schur @ triangular
    ) / max(float(np.linalg.norm(matrix)), 1.0)
    if residual > 1.0e-10:
        return None
    return right_schur.T, right_schur


def _contact_characteristic_fraction(
    left_primitive: Array,
    right_primitive: Array,
    gamma: float,
) -> float:
    """Return the fraction of a local jump carried by the material field."""

    center = 0.5 * (left_primitive + right_primitive)
    matrix = reduced_primitive_jacobian(center, gamma)
    eigenvalues, right_basis = np.linalg.eig(matrix)
    if (
        not np.all(np.isfinite(eigenvalues))
        or not np.all(np.isfinite(right_basis))
        or np.max(np.abs(np.imag(eigenvalues))) > 1.0e-8
        or np.max(np.abs(np.imag(right_basis))) > 1.0e-7
    ):
        return 0.0
    eigenvalues = np.real(eigenvalues)
    right_basis = np.real(right_basis)
    if np.linalg.cond(right_basis) > 1.0e8:
        return 0.0
    try:
        amplitudes = np.linalg.solve(
            right_basis,
            (right_primitive - left_primitive)[REDUCED],
        )
    except np.linalg.LinAlgError:
        return 0.0
    contact = int(np.argmin(np.abs(eigenvalues - center[1])))
    total = float(np.sum(np.abs(amplitudes)))
    if total <= 256.0 * np.finfo(float).eps:
        return 0.0
    return float(np.clip(abs(amplitudes[contact]) / total, 0.0, 1.0))


def _magnetic_activation(
    first: Array,
    second: Array,
    gamma: float,
) -> float:
    center = 0.5 * (first + second)
    rho = float(center[0])
    velocity = center[1:4]
    pressure = float(center[4])
    magnetic = center[5:8]
    lorentz2 = 1.0 / (1.0 - float(np.dot(velocity, velocity)))
    magnetic_velocity = float(np.dot(magnetic, velocity))
    comoving_magnetic2 = float(np.dot(magnetic, magnetic)) / lorentz2
    comoving_magnetic2 += magnetic_velocity * magnetic_velocity
    enthalpy = 1.0 + gamma * pressure / ((gamma - 1.0) * rho)
    return comoving_magnetic2 / (rho * enthalpy + comoving_magnetic2)


def characteristic_cweno_z_coefficients(
    primitive: Array,
    gamma: float,
    periodic: bool,
    parameters: spacetime.PredictorParameters,
    *,
    point_samples: bool,
) -> Array:
    """Build cell-local SRMHD CWENO-Z polynomials wave by wave."""

    component_result = spacetime.cweno_z_coefficients(
        primitive,
        periodic,
        parameters,
        point_samples=point_samples,
    )
    result = component_result.copy()
    central_weight = parameters.central_linear_weight
    side_weight = (1.0 - central_weight) / 3.0
    linear_weights = np.asarray(
        (central_weight, side_weight, side_weight, side_weight)
    )
    epsilon = 128.0 * np.finfo(float).eps
    jacobians = reduced_primitive_jacobians(primitive, gamma)
    for cell in range(len(primitive)):
        basis = _characteristic_basis(
            primitive[cell], gamma, jacobians[cell]
        )
        if basis is None:
            continue
        left_basis, right_basis = basis
        five_primitive = spacetime._stencil(
            primitive, cell, spacetime.QUARTIC_OFFSETS, periodic
        )
        five = five_primitive[:, REDUCED] @ left_basis.T
        optimal = (
            spacetime.POINT_QUARTIC_INVERSE
            if point_samples
            else spacetime.QUARTIC_INVERSE
        ) @ five
        quadratic_inverses = (
            spacetime.POINT_QUADRATIC_INVERSES
            if point_samples
            else spacetime.QUADRATIC_INVERSES
        )
        lower = []
        for offsets, inverse in zip(
            spacetime.QUADRATIC_OFFSETS,
            quadratic_inverses,
            strict=True,
        ):
            stencil = spacetime._stencil(
                primitive, cell, offsets, periodic
            )
            waves = stencil[:, REDUCED] @ left_basis.T
            coefficients = np.zeros((5, 7))
            coefficients[:3] = inverse @ waves
            lower.append(coefficients)
        candidates = np.empty((4, 5, 7))
        candidates[1:] = lower
        candidates[0] = (
            optimal - side_weight * (lower[0] + lower[1] + lower[2])
        ) / central_weight
        indicators = np.asarray(
            [
                spacetime._candidate_smoothness(candidate)
                for candidate in candidates
            ]
        )
        variation = np.ptp(five, axis=0)
        amplitude = np.max(np.abs(five), axis=0)
        constant = variation <= (
            256.0 * np.finfo(float).eps * np.maximum(amplitude, 1.0)
        )
        activity = variation / np.maximum(amplitude, 1.0)
        activity_scale = float(np.max(activity))
        inactive = constant | (
            activity
            <= parameters.sensor_relative_activity_floor * activity_scale
        )
        scales = np.maximum(
            variation,
            64.0
            * np.finfo(float).eps
            * np.maximum(amplitude, 1.0),
        )
        normalized = indicators / (scales * scales)
        normalized[:, inactive] = 0.0
        tau = np.abs(normalized[1] - normalized[3])
        alpha = linear_weights[:, None] * (
            1.0
            + (tau[None, :] / (epsilon + normalized))
            ** parameters.wenoz_exponent
        )
        weights = alpha / np.sum(alpha, axis=0, keepdims=True)
        wave_coefficients = np.einsum(
            "sc,spc->pc", weights, candidates
        )
        if np.any(inactive):
            wave_coefficients[:, inactive] = 0.0
            wave_coefficients[0, inactive] = five[2, inactive]
        reduced_coefficients = wave_coefficients @ right_basis.T
        if point_samples:
            reduced_coefficients[0] = primitive[cell, REDUCED]
        else:
            reduced_coefficients[0] += primitive[cell, REDUCED] - (
                spacetime._polynomial_average(reduced_coefficients)
            )
        result[cell] = 0.0
        result[cell][:, REDUCED] = reduced_coefficients
        result[cell, 0, 5] = primitive[cell, 5]
    return result


def characteristic_reconstruction_pair(
    primitive: Array,
    left_cell: int,
    gamma: float,
    periodic: bool,
    reconstruction: Callable[..., float],
    density_pair: Callable[[Array], tuple[float, float]] | None = None,
) -> tuple[Array, Array]:
    """Reconstruct one interface in a frozen numerical SRMHD basis."""

    stencil = np.asarray(
        [
            spacetime._stencil(
                primitive,
                left_cell,
                (offset,),
                periodic,
            )[0]
            for offset in range(-2, 4)
        ]
    )
    center = 0.5 * (stencil[2] + stencil[3])
    center[5] = 0.5 * (stencil[2, 5] + stencil[3, 5])
    basis = _characteristic_basis(center, gamma)
    if basis is None:
        left = np.empty(8)
        right = np.empty(8)
        for component in range(8):
            left[component], right[component] = (
                srmhd.base.reconstruct_scalar_pair(
                    stencil[:, component], reconstruction
                )
            )
    else:
        left_basis, right_basis = basis
        waves = stencil[:, REDUCED] @ left_basis.T
        left_waves = np.empty(7)
        right_waves = np.empty(7)
        for wave in range(7):
            left_waves[wave], right_waves[wave] = (
                srmhd.base.reconstruct_scalar_pair(
                    waves[:, wave], reconstruction
                )
            )
        left = stencil[2].copy()
        right = stencil[3].copy()
        left[REDUCED] = left_waves @ right_basis.T
        right[REDUCED] = right_waves @ right_basis.T
    if density_pair is not None:
        density_left, density_right = density_pair(stencil)
        left_weight, right_weight = contact_correction.density_trace_weights(
            stencil,
            lambda first, second: _contact_characteristic_fraction(
                first, second, gamma
            ),
            activation_weight=lambda first, second: _magnetic_activation(
                first, second, gamma
            ),
        )
        left[0] += left_weight * (density_left - left[0])
        right[0] += right_weight * (density_right - right[0])
    left[5] = right[5] = center[5]

    weight = 1.0
    for _ in range(53):
        pair = np.asarray(
            (
                stencil[2] + weight * (left - stencil[2]),
                stencil[3] + weight * (right - stencil[3]),
            )
        )
        if srmhd.primitive_is_admissible(pair):
            break
        weight *= 0.5
    return (
        stencil[2] + weight * (left - stencil[2]),
        stencil[3] + weight * (right - stencil[3]),
    )


def characteristic_parabolic_coefficients(
    primitive: Array,
    gamma: float,
    periodic: bool,
    reconstruction: Callable[..., float],
    density_pair: Callable[[Array], tuple[float, float]] | None = None,
) -> Array:
    result = np.zeros((len(primitive), 5, 8))
    for cell in range(len(primitive)):
        _left_neighbor, left_trace = characteristic_reconstruction_pair(
            primitive,
            cell - 1,
            gamma,
            periodic,
            reconstruction,
            density_pair,
        )
        right_trace, _right_neighbor = characteristic_reconstruction_pair(
            primitive,
            cell,
            gamma,
            periodic,
            reconstruction,
            density_pair,
        )
        curvature = 3.0 * (
            left_trace + right_trace - 2.0 * primitive[cell]
        )
        result[cell, 0] = primitive[cell] - curvature / 12.0
        result[cell, 1] = right_trace - left_trace
        result[cell, 2] = curvature
        result[cell, 1:, 5] = 0.0
        result[cell, 0, 5] = primitive[cell, 5]
    return result


def _admissible_polynomial(
    coefficients: Array,
    centers: Array,
) -> tuple[Array, Array]:
    result = coefficients.copy()
    valid = np.ones(len(result), dtype=bool)
    for cell in range(len(result)):
        original = result[cell].copy()
        theta = 1.0
        accepted = False
        for _ in range(54):
            trial = original.copy()
            trial[0] = centers[cell] + theta * (
                original[0] - centers[cell]
            )
            trial[1:] *= theta
            trial[1:, 5] = 0.0
            trial[0, 5] = centers[cell, 5]
            nodal = spacetime.SPACE_EVALUATION @ trial
            if srmhd.primitive_is_admissible(nodal):
                result[cell] = trial
                accepted = True
                break
            theta *= 0.5
        if not accepted:
            result[cell] = 0.0
            result[cell, 0] = centers[cell]
            valid[cell] = srmhd.primitive_is_admissible(
                centers[cell : cell + 1]
            )
    return result, valid


def primitive_predictor_rhs(
    values: Array,
    derivative_matrix: Array,
    dx: float,
    gamma: float,
) -> Array:
    derivatives = np.einsum(
        "ab,nbc->nac", derivative_matrix, values
    ) / dx
    jacobians = reduced_primitive_jacobians(values, gamma)
    result = np.full_like(values, np.nan)
    result[..., REDUCED] = -np.einsum(
        "...ij,...j->...i",
        jacobians,
        derivatives[..., REDUCED],
    )
    result[..., 5] = 0.0
    return result


def _stage_validity(stages: tuple[Array, ...]) -> Array:
    valid = np.ones(stages[0].shape[0], dtype=bool)
    for stage in stages:
        valid &= np.asarray(
            [srmhd.primitive_is_admissible(cell) for cell in stage]
        )
    return valid


def lsdg_predict(
    coefficients: Array,
    centers: Array,
    dt: float,
    dx: float,
    gamma: float,
    iterations: int,
    tolerance: float,
) -> spacetime.LocalPrediction:
    positive, polynomial_valid = _admissible_polynomial(
        coefficients, centers
    )
    initial = np.einsum(
        "qp,npc->nqc", spacetime.GAUSS_SPACE_EVALUATION, positive
    )
    values = np.repeat(initial[:, :, np.newaxis, :], 5, axis=2)
    residual = np.full(len(values), math.inf)
    for _ in range(iterations):
        nonconservative_product = np.empty_like(values)
        for time_node in range(5):
            rhs = primitive_predictor_rhs(
                values[:, :, time_node],
                spacetime.GAUSS_SPACE_DERIVATIVE,
                dx,
                gamma,
            )
            nonconservative_product[:, :, time_node] = -dt * rhs
        weak_rhs = (
            initial[:, :, np.newaxis, :]
            * spacetime.GAUSS_TIME_INITIAL[
                np.newaxis, np.newaxis, :, np.newaxis
            ]
            - nonconservative_product
            * spacetime.GAUSS_WEIGHTS_01[
                np.newaxis, np.newaxis, :, np.newaxis
            ]
        )
        next_values = np.einsum(
            "lk,nakc->nalc", spacetime.GAUSS_TIME_WEAK_INVERSE, weak_rhs
        )
        residual = np.max(
            np.abs(next_values - values)
            / np.maximum(1.0, np.abs(next_values)),
            axis=(1, 2, 3),
        )
        values = next_values
    left = np.einsum(
        "q,nqtc->ntc", spacetime.GAUSS_SPACE_LEFT, values
    )
    right = np.einsum(
        "q,nqtc->ntc", spacetime.GAUSS_SPACE_RIGHT, values
    )
    boundary_stages = tuple(
        np.stack((left[:, time_node], right[:, time_node]), axis=1)
        for time_node in range(5)
    )
    valid = polynomial_valid & _stage_validity(boundary_stages)
    valid &= residual <= tolerance
    valid &= np.all(np.isfinite(values), axis=(1, 2, 3))
    maximum_speed = 0.0
    for stage in boundary_stages:
        for primitive in stage.reshape(-1, 8):
            if not srmhd.primitive_is_admissible(primitive[None, :]):
                continue
            maximum_speed = max(
                maximum_speed,
                *(abs(speed) for speed in srmhd.signal_speeds(primitive, gamma)),
            )
    return spacetime.LocalPrediction(
        boundary_stages=boundary_stages,
        time_weights=spacetime.GAUSS_WEIGHTS_01,
        valid_cells=valid,
        maximum_speed=maximum_speed,
    )


def time_integrated_fluxes(
    predictions: list[spacetime.LocalPrediction],
    gamma: float,
    periodic: bool,
    boundary_primitives: tuple[Array, Array] | None,
) -> list[tuple[Array, Array, Array, Array]]:
    cells = len(predictions[0].boundary_stages[0])
    fluxes = [np.zeros((cells + 1, 8)) for _ in predictions]
    valid = [np.ones(cells + 1, dtype=bool) for _ in predictions]
    speeds = [np.zeros(cells + 1) for _ in predictions]
    fallback_counts = [
        np.zeros(cells + 1, dtype=int) for _ in predictions
    ]
    left_batch = []
    right_batch = []
    owners: list[tuple[int, int, float]] = []
    for prediction_index, prediction in enumerate(predictions):
        for weight, stage in zip(
            prediction.time_weights,
            prediction.boundary_stages,
            strict=True,
        ):
            for face in range(cells + 1):
                left, right = spacetime._boundary_traces(
                    stage, face, periodic, boundary_primitives
                )
                if not srmhd.primitive_is_admissible(
                    np.asarray((left, right))
                ):
                    valid[prediction_index][face] = False
                    continue
                left_batch.append(left)
                right_batch.append(right)
                owners.append((prediction_index, face, float(weight)))
                for primitive in (left, right):
                    speeds[prediction_index][face] = max(
                        speeds[prediction_index][face],
                        *(
                            abs(speed)
                            for speed in srmhd.signal_speeds(primitive, gamma)
                        ),
                    )
    if owners:
        face_fluxes, fallbacks = hlld_fluxes(
            np.asarray(left_batch), np.asarray(right_batch), gamma
        )
        for owner, face_flux, used_fallback in zip(
            owners, face_fluxes, fallbacks, strict=True
        ):
            prediction_index, face, weight = owner
            fluxes[prediction_index][face] += weight * face_flux
            fallback_counts[prediction_index][face] += int(used_fallback)
    result = []
    for prediction_index in range(len(predictions)):
        if periodic:
            fluxes[prediction_index][-1] = fluxes[prediction_index][0]
            valid[prediction_index][-1] = valid[prediction_index][0]
            speeds[prediction_index][-1] = speeds[prediction_index][0]
            fallback_counts[prediction_index][-1] = fallback_counts[
                prediction_index
            ][0]
        result.append(
            (
                fluxes[prediction_index],
                valid[prediction_index],
                speeds[prediction_index],
                fallback_counts[prediction_index],
            )
        )
    return result


def spacetime_step(
    state: Array,
    dt: float,
    dx: float,
    gamma: float,
    periodic: bool,
    parameters: spacetime.PredictorParameters,
    boundary_primitives: tuple[Array, Array] | None = None,
    *,
    contact_correction_enabled: bool = False,
) -> StepResult:
    recovery = recover_conserved(state, gamma)
    primitive = recovery.primitive
    if not np.all(recovery.success) or not srmhd.primitive_is_admissible(
        primitive
    ):
        return StepResult(None, math.inf, np.zeros(len(state), dtype=int))
    reconstruction_primitive, _center_scales = point_primitives(
        state, gamma, periodic, parameters
    )
    if parameters.characteristic_reconstruction:
        mp5_coefficients = characteristic_parabolic_coefficients(
            primitive,
            gamma,
            periodic,
            srmhd.base.mp5_left,
            (
                contact_correction.ppm3_contact_density_pair
                if contact_correction_enabled
                else None
            ),
        )
        high_coefficients = (
            mp5_coefficients.copy()
            if parameters.mp5_parabolic_primary
            else characteristic_cweno_z_coefficients(
                reconstruction_primitive,
                gamma,
                periodic,
                parameters,
                point_samples=True,
            )
        )
    else:
        mp5_coefficients = component_mp5_parabolic_coefficients(
            primitive, gamma, periodic
        )
        high_coefficients = (
            mp5_coefficients.copy()
            if parameters.mp5_parabolic_primary
            else spacetime.cweno_z_coefficients(
                reconstruction_primitive,
                periodic,
                parameters,
                point_samples=True,
            )
        )
    high_primitive = (
        primitive
        if parameters.mp5_parabolic_primary
        else reconstruction_primitive
    )
    high_coefficients[:, 1:, 5] = 0.0
    high_coefficients[:, 0, 5] = reconstruction_primitive[:, 5]
    coefficient_sets = (
        spacetime.pwc_coefficients(primitive),
        spacetime.pwl_coefficients(primitive, periodic),
        mp5_coefficients,
        high_coefficients,
    )
    for coefficients in coefficient_sets:
        coefficients[:, 1:, 5] = 0.0
        coefficients[:, 0, 5] = primitive[:, 5]
    center_sets = (
        primitive,
        primitive,
        primitive,
        high_primitive,
    )
    predictions = [
        lsdg_predict(
            coefficients,
            centers,
            dt,
            dx,
            gamma,
            parameters.picard_iterations,
            parameters.picard_tolerance,
        )
        for coefficients, centers in zip(
            coefficient_sets, center_sets, strict=True
        )
    ]
    flux_data = time_integrated_fluxes(
        predictions, gamma, periodic, boundary_primitives
    )
    fallback_speed = max(
        float(np.max(flux_result[2])) for flux_result in flux_data
    )
    physical_overrides = (
        srmhd.physical_flux_overrides(
            primitive, dx, dt, gamma, periodic
        )
        if parameters.physical_fronts
        else {}
    )
    cell_orders = np.full(len(state), 3, dtype=int)
    for cell in range(len(state)):
        while (
            cell_orders[cell] > 0
            and not predictions[cell_orders[cell]].valid_cells[cell]
        ):
            cell_orders[cell] -= 1
    for _ in range(parameters.maximum_correction_iterations):
        face_orders = spacetime._face_orders(cell_orders, periodic)
        selected_fluxes = np.empty_like(flux_data[0][0])
        selected_speeds = np.empty(len(face_orders))
        for face, requested_order in enumerate(face_orders):
            order = int(requested_order)
            while order > 0 and not flux_data[order][1][face]:
                order -= 1
            if not flux_data[order][1][face]:
                return StepResult(
                    None, fallback_speed, cell_orders
                )
            selected_fluxes[face] = flux_data[order][0][face]
            selected_speeds[face] = flux_data[order][2][face]
        selected_fallbacks = np.asarray(
            [
                flux_data[int(order)][3][face]
                for face, order in enumerate(face_orders)
            ]
        )
        effective_physical_faces: set[int] = set()
        for face, flux in physical_overrides.items():
            difference = float(np.linalg.norm(flux - selected_fluxes[face]))
            flux_scale = max(
                float(np.linalg.norm(flux)),
                float(np.linalg.norm(selected_fluxes[face])),
                1.0,
            )
            if difference > 256.0 * np.finfo(float).eps * flux_scale:
                effective_physical_faces.add(
                    face % len(state) if periodic else face
                )
            selected_fluxes[face] = flux
            selected_fallbacks[face] = 0
        maximum_speed = float(np.max(selected_speeds))
        candidate = state - (dt / dx) * (
            selected_fluxes[1:] - selected_fluxes[:-1]
        )
        candidate_recovery = recover_conserved(candidate, gamma)
        candidate_primitive = candidate_recovery.primitive
        physical = candidate_recovery.success.copy()
        if np.all(physical):
            physical &= np.asarray(
                [
                    srmhd.primitive_is_admissible(row[None, :])
                    for row in candidate_primitive
                ]
            )
        normal_b_scale = max(float(np.max(np.abs(candidate[:, 5]))), 1.0)
        normal_b_valid = (
            np.ptp(candidate[:, 5]) <= 1.0e-12 * normal_b_scale
        )
        bad = ~physical
        if np.all(physical) and normal_b_valid:
            bad |= spacetime._relaxed_dmp_failures(
                primitive,
                candidate_primitive,
                cell_orders,
                periodic,
                parameters,
            )
        elif not normal_b_valid:
            bad[:] = True
        reducible = bad & (cell_orders > 0)
        if not np.any(bad) or not np.any(reducible):
            if np.all(physical) and normal_b_valid:
                fallback_faces = (
                    selected_fallbacks[:-1]
                    if periodic
                    else selected_fallbacks
                )
                return StepResult(
                    candidate,
                    maximum_speed,
                    cell_orders,
                    int(np.sum(fallback_faces)),
                    len(effective_physical_faces),
                )
            return StepResult(
                None, maximum_speed, cell_orders
            )
        cell_orders[reducible] -= 1
    return StepResult(None, fallback_speed, cell_orders, riemann_fallbacks)


def evolve_spacetime(
    primitive: Array,
    final_time: float,
    cfl: float,
    gamma: float,
    parameters: spacetime.PredictorParameters,
    periodic: bool = True,
    boundary_primitives: tuple[Array, Array] | None = None,
    domain_length: float = 1.0,
    accepted_callback: (
        Callable[[float, float, Array, StepResult], None] | None
    ) = None,
    *,
    contact_correction_enabled: bool = False,
) -> tuple[Array, bool, EvolutionStats]:
    state, _flux = state_and_flux(primitive, gamma)
    dx = domain_length / len(state)
    time = 0.0
    stats = EvolutionStats()
    speed_estimate = 0.0
    while time < final_time:
        recovery = recover_conserved(state, gamma)
        current = recovery.primitive
        if not np.all(recovery.success):
            return current, False, stats
        cell_speed = max(
            abs(speed)
            for row in current
            for speed in srmhd.signal_speeds(row, gamma)
        )
        dt = min(
            final_time - time,
            cfl * dx / max(cell_speed, speed_estimate),
        )
        accepted: StepResult | None = None
        for _ in range(12):
            trial = spacetime_step(
                state,
                dt,
                dx,
                gamma,
                periodic,
                parameters,
                boundary_primitives,
                contact_correction_enabled=contact_correction_enabled,
            )
            if (
                trial.state is not None
                and np.isfinite(trial.maximum_speed)
                and dt * trial.maximum_speed
                <= cfl * dx * (1.0 + 1.0e-12)
            ):
                accepted = trial
                break
            if np.isfinite(trial.maximum_speed) and trial.maximum_speed > 0.0:
                speed_dt = 0.99 * cfl * dx / trial.maximum_speed
                dt = (
                    speed_dt
                    if trial.state is not None
                    else min(0.5 * dt, speed_dt)
                )
            else:
                dt *= 0.5
            stats.cfl_retries += 1
        if accepted is None or accepted.state is None:
            return current, False, stats
        state = accepted.state
        if accepted_callback is not None:
            accepted_callback(time + dt, dt, state, accepted)
        speed_estimate = accepted.maximum_speed
        time += dt
        stats.steps += 1
        stats.high_cells += int(np.count_nonzero(accepted.cell_orders == 3))
        stats.mp5_cells += int(np.count_nonzero(accepted.cell_orders == 2))
        stats.pwl_cells += int(np.count_nonzero(accepted.cell_orders == 1))
        stats.pwc_cells += int(np.count_nonzero(accepted.cell_orders == 0))
        stats.riemann_fallbacks += accepted.riemann_fallbacks
        stats.physical_front_faces += accepted.physical_front_faces
        stats.minimum_order = min(
            stats.minimum_order, int(np.min(accepted.cell_orders))
        )
    final_recovery = recover_conserved(state, gamma)
    return (
        final_recovery.primitive,
        bool(np.all(final_recovery.success)),
        stats,
    )


def component_mp5_faces(
    primitive: Array,
    gamma: float,
    periodic: bool,
    boundary_primitives: tuple[Array, Array] | None = None,
) -> tuple[Array, Array]:
    srmhd.component_mp5.periodic = periodic
    left = np.empty((len(primitive) + 1, 8))
    right = np.empty_like(left)
    for face in range(len(left)):
        left[face], right[face] = srmhd.component_mp5(
            primitive, face - 1, gamma
        )
    if periodic:
        left[-1] = left[0]
        right[-1] = right[0]
    elif boundary_primitives is not None:
        left[0] = right[0] = boundary_primitives[0]
        left[-1] = right[-1] = boundary_primitives[1]
    return left, right


def evolve_mp5_ssprk3(
    primitive: Array,
    final_time: float,
    cfl: float,
    gamma: float,
    periodic: bool,
    boundary_primitives: tuple[Array, Array] | None = None,
    domain_length: float = 1.0,
) -> tuple[Array, bool, EvolutionStats]:
    state, _flux = state_and_flux(primitive, gamma)
    dx = domain_length / len(state)
    time = 0.0
    stats = EvolutionStats()

    def rhs(stage_state: Array) -> tuple[Array | None, float, int]:
        recovery = recover_conserved(stage_state, gamma)
        if not np.all(recovery.success):
            return None, math.inf, 0
        left, right = component_mp5_faces(
            recovery.primitive,
            gamma,
            periodic,
            boundary_primitives,
        )
        flux, fallbacks = hlld_fluxes(left, right, gamma)
        speed = max(
            abs(signal)
            for row in recovery.primitive
            for signal in srmhd.signal_speeds(row, gamma)
        )
        counted_fallbacks = fallbacks[:-1] if periodic else fallbacks
        return (
            -(flux[1:] - flux[:-1]) / dx,
            speed,
            int(np.count_nonzero(counted_fallbacks)),
        )

    while time < final_time:
        rhs0, speed0, fallback0 = rhs(state)
        if rhs0 is None:
            return recover_conserved(state, gamma).primitive, False, stats
        dt = min(final_time - time, cfl * dx / speed0)
        accepted = False
        for _ in range(12):
            stage1 = state + dt * rhs0
            rhs1, speed1, fallback1 = rhs(stage1)
            if rhs1 is None:
                dt *= 0.5
                continue
            stage2 = 0.75 * state + 0.25 * (stage1 + dt * rhs1)
            rhs2, speed2, fallback2 = rhs(stage2)
            if rhs2 is None:
                dt *= 0.5
                continue
            reduced = min(
                dt,
                final_time - time,
                cfl * dx / max(speed0, speed1, speed2),
            )
            if reduced < dt * (1.0 - 1.0e-13):
                dt = 0.99 * reduced
                continue
            candidate = (
                state / 3.0 + 2.0 * (stage2 + dt * rhs2) / 3.0
            )
            recovery = recover_conserved(candidate, gamma)
            if not np.all(recovery.success):
                dt *= 0.5
                continue
            state = candidate
            stats.riemann_fallbacks += fallback0 + fallback1 + fallback2
            accepted = True
            break
        if not accepted:
            return recover_conserved(state, gamma).primitive, False, stats
        time += dt
        stats.steps += 1
    recovery = recover_conserved(state, gamma)
    return recovery.primitive, bool(np.all(recovery.success)), stats


def entropy_cell_averages(
    cells: int,
    time: float,
    gamma: float,
    amplitude: float = 0.1,
    magnetic: Array | None = None,
) -> Array:
    velocity = np.asarray((0.3, 0.05, 0.0))
    magnetic_field = (
        np.asarray((0.5, 0.2, -0.1))
        if magnetic is None
        else np.asarray(magnetic, dtype=float)
    )
    centers = (np.arange(cells) + 0.5) / cells
    cell_width = 1.0 / cells
    averaging_factor = math.sin(math.pi * cell_width) / (
        math.pi * cell_width
    )
    density = 1.0 + amplitude * averaging_factor * np.sin(
        2.0 * math.pi * (centers - velocity[0] * time)
    )
    primitive = np.column_stack(
        (
            density,
            np.full(cells, velocity[0]),
            np.full(cells, velocity[1]),
            np.full(cells, velocity[2]),
            np.ones(cells),
            np.full(cells, magnetic_field[0]),
            np.full(cells, magnetic_field[1]),
            np.full(cells, magnetic_field[2]),
        )
    )
    state, _flux = state_and_flux(primitive, gamma)
    return recover_conserved(state, gamma).primitive


def _result(
    method: str,
    cells: int,
    cfl: float,
    evolved: Array,
    exact: Array,
    completed: bool,
    gamma: float,
    stats: EvolutionStats | None,
) -> SmoothResult:
    component_l1 = []
    overshoots = []
    tv_ratios = []
    for component in range(8):
        scale = max(float(np.ptp(exact[:, component])), 1.0)
        component_l1.append(
            float(np.mean(np.abs(evolved[:, component] - exact[:, component])))
            / scale
        )
        overshoots.append(
            max(
                0.0,
                float(np.max(evolved[:, component]) - np.max(exact[:, component])),
                float(np.min(exact[:, component]) - np.min(evolved[:, component])),
            )
            / scale
        )
        exact_tv = spacetime.total_variation(exact[:, component])
        error_tv = spacetime.total_variation(
            evolved[:, component] - exact[:, component]
        )
        tv_ratios.append(
            spacetime.total_variation(evolved[:, component]) / exact_tv
            if exact_tv > 1.0e-12 * scale
            else 1.0 + error_tv / scale
        )
    exact_state, _flux = state_and_flux(exact, gamma)
    evolved_state, _flux = state_and_flux(evolved, gamma)
    integral_error = np.abs(
        np.mean(evolved_state, axis=0) - np.mean(exact_state, axis=0)
    )
    return SmoothResult(
        method=method,
        cells=cells,
        cfl=cfl,
        completed=completed,
        maximum_l1=max(component_l1),
        observed_order=math.nan,
        maximum_overshoot=max(overshoots),
        maximum_tv_ratio=max(tv_ratios),
        mass_error=float(integral_error[0]),
        momentum_error=float(np.max(integral_error[1:4])),
        energy_error=float(integral_error[4]),
        normal_b_spread=float(np.ptp(evolved[:, 5])),
        steps=0 if stats is None else stats.steps,
        cfl_retries=0 if stats is None else stats.cfl_retries,
        mp5_cell_steps=0 if stats is None else stats.mp5_cells,
        pwl_cell_steps=0 if stats is None else stats.pwl_cells,
        pwc_cell_steps=0 if stats is None else stats.pwc_cells,
        riemann_fallbacks=(
            0 if stats is None else stats.riemann_fallbacks
        ),
        physical_front_faces=(
            0 if stats is None else stats.physical_front_faces
        ),
    )


def run_entropy(
    cells_values: tuple[int, ...],
    cfl: float,
    gamma: float,
    parameters: spacetime.PredictorParameters,
    *,
    magnetic: Array | None = None,
    label: str = "",
) -> list[SmoothResult]:
    rows = []
    previous: dict[str, tuple[int, float]] = {}
    for cells in cells_values:
        initial = entropy_cell_averages(
            cells, 0.0, gamma, magnetic=magnetic
        )
        exact = entropy_cell_averages(
            cells, 1.0, gamma, magnetic=magnetic
        )
        baseline, baseline_completed, baseline_stats = evolve_mp5_ssprk3(
            initial, 1.0, cfl, gamma, periodic=True
        )
        candidate, candidate_completed, stats = evolve_spacetime(
            initial, 1.0, cfl, gamma, parameters, periodic=True
        )
        current = (
            _result(
                f"{label}component_mp5_ssprk3_hlld",
                cells,
                cfl,
                baseline,
                exact,
                baseline_completed,
                gamma,
                baseline_stats,
            ),
            _result(
                f"{label}{_candidate_name(parameters)}",
                cells,
                cfl,
                candidate,
                exact,
                candidate_completed,
                gamma,
                stats,
            ),
        )
        for row in current:
            prior = previous.get(row.method)
            order = math.nan
            if prior is not None and row.maximum_l1 > 0.0:
                previous_cells, previous_error = prior
                order = math.log(previous_error / row.maximum_l1) / math.log(
                    cells / previous_cells
                )
            rows.append(dataclasses.replace(row, observed_order=order))
            previous[row.method] = (cells, row.maximum_l1)
    return rows


def linear_wave_cell_averages(
    cells: int,
    time: float,
    gamma: float,
    wave: int,
    amplitude: float = 1.0e-6,
) -> tuple[Array, float]:
    base = np.asarray((1.0, 0.2, 0.05, -0.03, 1.0, 0.5, 0.2, -0.1))
    jacobian = reduced_primitive_jacobian(base, gamma)
    eigenvalues, right = np.linalg.eig(jacobian)
    order = np.argsort(np.real(eigenvalues))
    eigenvalues = np.real(eigenvalues[order])
    direction = np.real(right[:, order[wave]])
    direction /= max(float(np.max(np.abs(direction))), 1.0e-14)
    full_direction = np.zeros(8)
    full_direction[REDUCED] = direction
    centers = (np.arange(cells) + 0.5) / cells
    cell_width = 1.0 / cells
    averaging_factor = math.sin(math.pi * cell_width) / (
        math.pi * cell_width
    )
    phase = 2.0 * math.pi * (
        centers - float(eigenvalues[wave]) * time
    )
    primitive = base + (
        amplitude * averaging_factor * np.sin(phase)
    )[:, None] * full_direction
    state, _flux = state_and_flux(primitive, gamma)
    return recover_conserved(state, gamma).primitive, float(eigenvalues[wave])


def run_linear_waves(
    cells_values: tuple[int, ...],
    cfl: float,
    gamma: float,
    parameters: spacetime.PredictorParameters,
) -> list[tuple[str, list[SmoothResult]]]:
    wave_names = (
        "fast_minus",
        "alfven_minus",
        "slow_minus",
        "entropy",
        "slow_plus",
        "alfven_plus",
        "fast_plus",
    )
    all_rows = []
    for wave, wave_name in enumerate(wave_names):
        rows = []
        previous: dict[str, tuple[int, float]] = {}
        for cells in cells_values:
            initial, speed = linear_wave_cell_averages(
                cells, 0.0, gamma, wave
            )
            exact, _speed = linear_wave_cell_averages(
                cells, 1.0, gamma, wave
            )
            baseline, baseline_completed, baseline_stats = (
                evolve_mp5_ssprk3(
                    initial, 1.0, cfl, gamma, periodic=True
                )
            )
            candidate, candidate_completed, stats = evolve_spacetime(
                initial, 1.0, cfl, gamma, parameters, periodic=True
            )
            current = (
                _result(
                    f"{wave_name}:{speed:+.6f}:component_mp5_ssprk3_hlld",
                    cells,
                    cfl,
                    baseline,
                    exact,
                    baseline_completed,
                    gamma,
                    baseline_stats,
                ),
                _result(
                    f"{wave_name}:{speed:+.6f}:{_candidate_name(parameters)}",
                    cells,
                    cfl,
                    candidate,
                    exact,
                    candidate_completed,
                    gamma,
                    stats,
                ),
            )
            for row in current:
                method = row.method.rsplit(":", maxsplit=1)[-1]
                prior = previous.get(method)
                order_value = math.nan
                if prior is not None and row.maximum_l1 > 0.0:
                    prior_cells, prior_error = prior
                    order_value = math.log(
                        prior_error / row.maximum_l1
                    ) / math.log(cells / prior_cells)
                rows.append(
                    dataclasses.replace(row, observed_order=order_value)
                )
                previous[method] = (cells, row.maximum_l1)
        all_rows.append((wave_name, rows))
    return all_rows


def print_rows(rows: list[SmoothResult]) -> None:
    if not rows:
        return
    fields = tuple(field.name for field in dataclasses.fields(rows[0]))
    print("\t".join(fields))
    for row in rows:
        values = []
        for field in fields:
            value = getattr(row, field)
            values.append(f"{value:.12e}" if isinstance(value, float) else str(value))
        print("\t".join(values))


def _candidate_name(parameters: spacetime.PredictorParameters) -> str:
    prefix = (
        "characteristic_cweno_z"
        if parameters.characteristic_reconstruction
        else "component_cweno_z"
    )
    return f"{prefix}_aderprim_lsdg_hlld"


def _balsara_spec(case: int) -> dict[str, object]:
    for spec in EXACT_BALSARA_CASES:
        if int(spec["case"]) == case:
            return spec
    raise ValueError(f"unknown Balsara case {case}")


def _balsara_initial(
    spec: dict[str, object],
    cells: int,
) -> tuple[Array, tuple[Array, Array]]:
    left_catalog = np.asarray(spec["left"], dtype=float)
    right_catalog = np.asarray(spec["right"], dtype=float)
    order = np.asarray((0, 2, 3, 4, 1, 5, 6, 7))
    left = left_catalog[order]
    right = right_catalog[order]
    centers = -1.0 + (np.arange(cells) + 0.5) * (2.0 / cells)
    initial = np.where((centers < 0.0)[:, None], left, right)
    return initial, (left, right)


def _shock_result(
    method: str,
    case: int,
    cells: int,
    cfl: float,
    evolved: Array,
    completed: bool,
    gamma: float,
    exact_profile: Array,
    exact_average: Array,
    stats: EvolutionStats | None,
) -> ShockResult:
    numerical, _flux = state_and_flux(evolved, gamma)
    numerical = numerical.T
    exact_point = exact_rmhd_conserved(
        exact_profile,
        normal_magnetic=float(exact_average[5, 0]),
        gamma=gamma,
    )
    scales = np.maximum(
        np.ptp(exact_point, axis=1),
        np.max(np.abs(exact_point), axis=1),
    )
    scales = np.maximum(scales, 1.0)
    inactive = np.ptp(exact_point, axis=1) <= (
        256.0
        * np.finfo(float).eps
        * np.maximum(np.max(np.abs(exact_point), axis=1), 1.0)
    )
    component_l1 = np.mean(np.abs(numerical - exact_average), axis=1) / scales
    overshoot = np.maximum.reduce(
        (
            np.zeros(8),
            np.max(numerical, axis=1) - np.max(exact_point, axis=1),
            np.min(exact_point, axis=1) - np.min(numerical, axis=1),
        )
    ) / scales
    tv_ratios = np.empty(8)
    for component in range(8):
        exact_tv = spacetime.total_variation(exact_average[component])
        numerical_tv = spacetime.total_variation(numerical[component])
        if exact_tv > 1.0e-12 * scales[component]:
            tv_ratios[component] = numerical_tv / exact_tv
        else:
            tv_ratios[component] = 1.0 + numerical_tv / scales[component]
    integral_error = np.abs(
        np.mean(numerical, axis=1) - np.mean(exact_average, axis=1)
    ) / scales
    return ShockResult(
        method=method,
        case=case,
        cells=cells,
        cfl=cfl,
        completed=completed,
        maximum_l1=float(np.max(component_l1)),
        maximum_l1_component=CONSERVED_NAMES[int(np.argmax(component_l1))],
        density_l1=float(component_l1[0]),
        maximum_overshoot=float(np.max(overshoot)),
        maximum_overshoot_component=CONSERVED_NAMES[
            int(np.argmax(overshoot))
        ],
        maximum_tv_ratio=float(np.max(tv_ratios)),
        maximum_tv_component=CONSERVED_NAMES[int(np.argmax(tv_ratios))],
        maximum_inactive_error=(
            float(np.max(np.abs(numerical[inactive] - exact_average[inactive])))
            if np.any(inactive)
            else 0.0
        ),
        maximum_integral_error=float(np.max(integral_error)),
        normal_b_spread=float(np.ptp(numerical[5])),
        steps=0 if stats is None else stats.steps,
        cfl_retries=0 if stats is None else stats.cfl_retries,
        mp5_cell_steps=0 if stats is None else stats.mp5_cells,
        pwl_cell_steps=0 if stats is None else stats.pwl_cells,
        pwc_cell_steps=0 if stats is None else stats.pwc_cells,
        riemann_fallbacks=(
            0 if stats is None else stats.riemann_fallbacks
        ),
        physical_front_faces=(
            0 if stats is None else stats.physical_front_faces
        ),
    )


def run_balsara(
    cases: tuple[int, ...],
    cells_values: tuple[int, ...],
    cfl: float,
    parameters: spacetime.PredictorParameters,
) -> list[ShockResult]:
    rows = []
    for case in cases:
        spec = _balsara_spec(case)
        gamma = float(spec["gamma"])
        final_time = float(spec["time"])
        exact_profile = read_last_exact_balsara_profile(
            EXACT_BALSARA_ROOT / f"solution_b{case}.dat"
        )
        for cells in cells_values:
            initial, boundaries = _balsara_initial(spec, cells)
            edges = np.linspace(-1.0, 1.0, cells + 1)
            exact_average = exact_rmhd_cell_averages(
                exact_profile,
                normal_magnetic=float(boundaries[0][5]),
                gamma=gamma,
                cell_edges=edges,
            )
            baseline, baseline_completed, baseline_stats = evolve_mp5_ssprk3(
                initial,
                final_time,
                cfl,
                gamma,
                periodic=False,
                boundary_primitives=boundaries,
                domain_length=2.0,
            )
            candidate, candidate_completed, stats = evolve_spacetime(
                initial,
                final_time,
                cfl,
                gamma,
                parameters,
                periodic=False,
                boundary_primitives=boundaries,
                domain_length=2.0,
            )
            rows.append(
                _shock_result(
                    "component_mp5_ssprk3_hlld",
                    case,
                    cells,
                    cfl,
                    baseline,
                    baseline_completed,
                    gamma,
                    exact_profile,
                    exact_average,
                    baseline_stats,
                )
            )
            rows.append(
                _shock_result(
                    _candidate_name(parameters),
                    case,
                    cells,
                    cfl,
                    candidate,
                    candidate_completed,
                    gamma,
                    exact_profile,
                    exact_average,
                    stats,
                )
            )
    return rows


def _front_result(
    case: str,
    method: str,
    cells: int,
    cfl: float,
    evolved: Array,
    exact: Array,
    completed: bool,
    gamma: float,
    stats: EvolutionStats,
) -> FrontResult:
    numerical_state, _flux = state_and_flux(evolved, gamma)
    exact_state, _flux = state_and_flux(exact, gamma)
    scales = np.maximum(np.ptp(exact_state, axis=0), 1.0)
    component_l1 = np.mean(
        np.abs(numerical_state - exact_state), axis=0
    ) / scales
    overshoot = np.maximum.reduce(
        (
            np.zeros(8),
            np.max(numerical_state, axis=0) - np.max(exact_state, axis=0),
            np.min(exact_state, axis=0) - np.min(numerical_state, axis=0),
        )
    ) / scales
    tv_ratios = []
    for component in range(8):
        exact_tv = spacetime.total_variation(exact_state[:, component])
        numerical_tv = spacetime.total_variation(
            numerical_state[:, component]
        )
        tv_ratios.append(
            numerical_tv / exact_tv
            if exact_tv > 1.0e-12 * scales[component]
            else 1.0 + numerical_tv / scales[component]
        )
    integral_error = np.abs(
        np.mean(numerical_state, axis=0) - np.mean(exact_state, axis=0)
    )
    return FrontResult(
        case=case,
        method=method,
        cells=cells,
        cfl=cfl,
        completed=completed,
        maximum_l1=float(np.max(component_l1)),
        maximum_overshoot=float(np.max(overshoot)),
        maximum_tv_ratio=float(np.max(tv_ratios)),
        mass_error=float(integral_error[0]),
        momentum_error=float(np.max(integral_error[1:4])),
        energy_error=float(integral_error[4]),
        normal_b_spread=float(np.ptp(numerical_state[:, 5])),
        steps=stats.steps,
        mp5_cell_steps=stats.mp5_cells,
        pwl_cell_steps=stats.pwl_cells,
        pwc_cell_steps=stats.pwc_cells,
        riemann_fallbacks=stats.riemann_fallbacks,
        physical_front_faces=stats.physical_front_faces,
    )


def run_exact_fronts(
    cells_values: tuple[int, ...],
    cfl: float,
    gamma: float,
    parameters: spacetime.PredictorParameters,
) -> list[FrontResult]:
    contact_left = np.asarray((2.0, 0.5, 0.1, -0.1, 1.0, 0.5, 0.3, -0.2))
    contact_right = contact_left.copy()
    contact_right[0] = 1.0
    shock_left, shock_right, shock_speed = srmhd.oblique_shock_states(gamma)
    cases = (
        (
            "moving_rmhd_contact",
            contact_left,
            contact_right,
            0.5,
            0.6,
        ),
        (
            "moving_oblique_rmhd_shock",
            shock_left,
            shock_right,
            shock_speed,
            0.6,
        ),
    )
    rows = []
    for cells in cells_values:
        for case, left, right, speed, final_time in cases:
            initial_position = 0.217
            initial = srmhd.discontinuity_averages(
                cells, initial_position, left, right, gamma
            )
            exact = srmhd.discontinuity_averages(
                cells,
                initial_position + speed * final_time,
                left,
                right,
                gamma,
            )
            for physical_fronts in (False, True):
                method_parameters = dataclasses.replace(
                    parameters, physical_fronts=physical_fronts
                )
                evolved, completed, stats = evolve_spacetime(
                    initial,
                    final_time,
                    cfl,
                    gamma,
                    method_parameters,
                    periodic=False,
                    boundary_primitives=(left, right),
                )
                rows.append(
                    _front_result(
                        case,
                        _candidate_name(method_parameters)
                        + ("_physical_front" if physical_fronts else ""),
                        cells,
                        cfl,
                        evolved,
                        exact,
                        completed,
                        gamma,
                        stats,
                    )
                )
    return rows


def run_basis_audit(
    states: int,
    gamma: float,
    seed: int = 273,
) -> list[BasisAuditResult]:
    rng = np.random.default_rng(seed)
    rho = 10.0 ** rng.uniform(-6.0, 3.0, states)
    pressure = 10.0 ** rng.uniform(-8.0, 4.0, states)
    velocity_direction = rng.normal(size=(states, 3))
    velocity_direction /= np.linalg.norm(
        velocity_direction, axis=1
    )[:, None]
    velocity = velocity_direction * rng.uniform(0.0, 0.999, states)[:, None]
    magnetic_direction = rng.normal(size=(states, 3))
    magnetic_direction /= np.linalg.norm(
        magnetic_direction, axis=1
    )[:, None]
    magnetic = magnetic_direction * (
        10.0 ** rng.uniform(-6.0, 3.0, states)
    )[:, None]
    broad = np.column_stack((rho, velocity, pressure, magnetic))
    samples = [("broad", broad)]
    bn_zero = broad.copy()
    bn_zero[:, 5] = 0.0
    samples.append(("bn_zero", bn_zero))
    b_zero = broad.copy()
    b_zero[:, 5:8] = 0.0
    samples.append(("b_zero", b_zero))

    rows = []
    for sample_name, primitive in samples:
        jacobians = reduced_primitive_jacobians(primitive, gamma)
        direct_count = 0
        schur_count = 0
        rejected_count = 0
        maximum_residual = 0.0
        for state, matrix in zip(primitive, jacobians, strict=True):
            eigenvalues, eigenvectors = np.linalg.eig(matrix)
            eigenvalue_scale = max(
                float(np.max(np.abs(eigenvalues))), 1.0
            )
            direct = (
                np.max(np.abs(np.imag(eigenvalues)))
                <= 1.0e-8 * eigenvalue_scale
                and np.max(np.abs(np.imag(eigenvectors))) <= 1.0e-7
                and np.linalg.cond(np.real(eigenvectors)) <= 1.0e8
            )
            basis = _characteristic_basis(state, gamma, matrix)
            if basis is None:
                rejected_count += 1
                continue
            left, right = basis
            if direct:
                direct_count += 1
            else:
                schur_count += 1
            projected = left @ matrix @ right
            residual = np.linalg.norm(
                matrix @ right - right @ projected
            ) / max(float(np.linalg.norm(matrix)), 1.0)
            maximum_residual = max(maximum_residual, float(residual))
        rows.append(
            BasisAuditResult(
                sample=sample_name,
                states=states,
                direct_eigen_bases=direct_count,
                schur_bases=schur_count,
                rejected_bases=rejected_count,
                maximum_residual=maximum_residual,
            )
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", type=int, nargs="+", default=(8, 16))
    parser.add_argument("--cfl", type=float, default=0.9)
    parser.add_argument("--gamma", type=float, default=5.0 / 3.0)
    parser.add_argument("--picard-iterations", type=int, default=8)
    parser.add_argument("--physical-fronts", action="store_true")
    parser.add_argument("--component-reconstruction", action="store_true")
    parser.add_argument("--dmp-relative-tolerance", type=float, default=0.01)
    parser.add_argument("--skip-entropy", action="store_true")
    parser.add_argument("--balsara-cases", type=int, nargs="+", default=())
    parser.add_argument("--balsara-cells", type=int, nargs="+", default=(64,))
    parser.add_argument("--exact-fronts", action="store_true")
    parser.add_argument("--front-cells", type=int, nargs="+", default=(16,))
    parser.add_argument("--linear-waves", action="store_true")
    parser.add_argument("--degenerate-entropy", action="store_true")
    parser.add_argument("--basis-audit", type=int, default=0)
    args = parser.parse_args()
    if any(cells < 8 for cells in args.cells):
        raise ValueError("all grids need at least eight cells")
    if args.cfl <= 0.0 or args.gamma <= 1.0:
        raise ValueError("CFL and gamma are outside their domains")
    if args.dmp_relative_tolerance < 0.0:
        raise ValueError("the DMP tolerance must be nonnegative")
    if args.basis_audit < 0:
        raise ValueError("the basis audit size must be nonnegative")
    parameters = spacetime.PredictorParameters(
        predictor_kind="lsdg",
        picard_iterations=args.picard_iterations,
        characteristic_reconstruction=not args.component_reconstruction,
        anchor_mp5_traces=False,
        sensor_relative_activity_floor=1.0e-7,
        physical_fronts=args.physical_fronts,
        dmp_relative_tolerance=args.dmp_relative_tolerance,
    )
    if not args.skip_entropy:
        print("[cell-averaged SRMHD entropy wave]")
        print_rows(
            run_entropy(
                tuple(sorted(set(args.cells))),
                args.cfl,
                args.gamma,
                parameters,
            )
        )
    if args.balsara_cases:
        if any(case not in range(1, 6) for case in args.balsara_cases):
            raise ValueError("Balsara cases are numbered 1 through 5")
        if any(cells < 8 for cells in args.balsara_cells):
            raise ValueError("all Balsara grids need at least eight cells")
        print("[exact finite-volume Balsara shock tubes]")
        print_rows(
            run_balsara(
                tuple(dict.fromkeys(args.balsara_cases)),
                tuple(sorted(set(args.balsara_cells))),
                args.cfl,
                parameters,
            )
        )
    if args.exact_fronts:
        if any(cells < 8 for cells in args.front_cells):
            raise ValueError("all physical-front grids need at least eight cells")
        print("[isolated exact SRMHD physical fronts]")
        print_rows(
            run_exact_fronts(
                tuple(sorted(set(args.front_cells))),
                args.cfl,
                args.gamma,
                parameters,
            )
        )
    if args.linear_waves:
        for wave_name, rows in run_linear_waves(
            tuple(sorted(set(args.cells))),
            args.cfl,
            args.gamma,
            parameters,
        ):
            print(f"[linearized SRMHD {wave_name} wave]")
            print_rows(rows)
    if args.degenerate_entropy:
        for label, magnetic in (
            ("bn_zero:", np.asarray((0.0, 0.2, -0.1))),
            ("b_zero:", np.zeros(3)),
        ):
            print(f"[cell-averaged SRMHD entropy wave {label[:-1]}]")
            print_rows(
                run_entropy(
                    tuple(sorted(set(args.cells))),
                    args.cfl,
                    args.gamma,
                    parameters,
                    magnetic=magnetic,
                    label=label,
                )
            )
    if args.basis_audit:
        print("[SRMHD characteristic/Schur basis audit]")
        print_rows(run_basis_audit(args.basis_audit, args.gamma))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
