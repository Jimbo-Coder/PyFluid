#!/usr/bin/env python3
"""Prototype a coupled finite-volume-consistent reconstruction hierarchy.

This is a research harness, not production solver code.  The candidate has
three deliberately separate pieces:

1. Build one admissible polynomial of the *conserved cell averages* inside
   each cell.  Recover primitives at three Gauss points and quadrature-average
   them.  This supplies an actual finite-volume primitive average instead of
   interpreting ``C2P(Ubar)`` as one.
2. Reconstruct those primitive averages with one shared, continuously varying
   PPM4 -> MP5 hierarchy.  The same cell order is used for every component and
   the smaller order of the two cells sharing a face is used on both traces.
3. In SRMHD, test every concrete SSPRK3 candidate against recovery and a
   relaxed density/energy DMP.  Replace both shared faces adjoining a troubled
   cell with PWC fluxes, growing the conservative mask only when a corrected
   neighbor would otherwise be unrecoverable.

The conserved polynomial and the face traces are scaled toward known-valid
cell states when necessary.  An atmosphere reset is not part of this
reconstruction hierarchy.

The implementation is intentionally system agnostic.  Small adapters below
exercise Newtonian HD, Newtonian MHD, and SRMHD using the existing research
harnesses and the production SRMHD C2P/Riemann oracle.  It must earn a clear
cross-system improvement here before any AMReX integration is considered.
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from diagnose_rmhd_balsara_onset import (  # noqa: E402
    case_spec,
    exact_average_at_time,
)
from diagnose_rmhd_operator import (  # noqa: E402
    _limited_face as rmhd_limited_face,
    exact_primitives_at,
    primitive_state_and_flux,
    reconstruct_faces as reconstruct_rmhd_faces,
    recover_conserved,
    riemann_fluxes,
    rhs_from_faces,
    select_riemann_flux,
)
from research_mhd_wave_reconstruction import (  # noqa: E402
    REDUCED as MHD_REDUCED,
    balanced_metric_nonlinear_mp5 as mhd_balanced_mp5,
    brio_wu_results,
    characteristic_basis as mhd_characteristic_basis,
    characteristic_mp5 as mhd_characteristic_mp5,
    circular_alfven_result,
    conserved_array as mhd_conserved_array,
    from_characteristics as mhd_from_characteristics,
    mhd_hlld_flux,
    primitive_from_conserved as mhd_primitive_from_conserved,
    to_characteristics as mhd_to_characteristics,
)
from research_modal_reconstruction import (  # noqa: E402
    mc_slope,
    mp5_left,
    ppm4_left,
)
from research_wave_reconstruction import (  # noqa: E402
    SHOCK_CASES,
    characteristic_basis as hydro_characteristic_basis,
    characteristic_mp5 as hydro_characteristic_mp5,
    component_reconstruction,
    conserved_from_primitive as hydro_conserved_from_primitive,
    entropy_wave_result,
    evolve as evolve_hydro,
    from_characteristics as hydro_from_characteristics,
    hllc_flux,
    primitive_from_conserved as hydro_primitive_from_conserved,
    shock_result,
    to_characteristics as hydro_to_characteristics,
)
from validate_rmhd import (  # noqa: E402
    EXACT_BALSARA_ROOT,
    read_last_exact_balsara_profile,
)


Array = np.ndarray
BatchRecovery = Callable[[Array, float], tuple[Array, Array]]
ConservedMap = Callable[[Array, float], Array]


@dataclasses.dataclass(frozen=True)
class CoupledFvParameters:
    """Dimensionless controls for the continuous polynomial hierarchy."""

    high_mode_threshold: float = 0.10
    nonlinear_mode_threshold: float = 0.35
    transition_exponent: int = 4
    quadrature_reduction: float = 0.5
    maximum_admissibility_iterations: int = 32
    primitive_quadrature: bool = True
    rmhd_characteristic: bool = True
    characteristic_conditioning_halo: int = 3
    characteristic_quadrature_threshold: float = 1.0
    smooth_mode_plateau: float = 0.025
    candidate_dmp_factor: float = 1.05


@dataclasses.dataclass(frozen=True)
class PreparedState:
    primitive_average: Array
    base_primitive: Array
    theta_middle: Array
    theta_high: Array
    quadrature_scale: Array
    face_high_scale: Array
    quadrature_reductions: int


@dataclasses.dataclass(frozen=True)
class RmhdResult:
    case: int
    method: str
    cells: int
    final_time: float
    completed: bool
    failed_stage: str
    maximum_l1: float
    maximum_linf: float
    maximum_overshoot: float
    maximum_tv_ratio: float
    riemann_fallbacks: int
    stage_fallbacks: int
    corrected_faces: int
    minimum_flux_weight: float
    quadrature_reductions: int


@dataclasses.dataclass(frozen=True)
class LimitedStage:
    """Result of one conservative high/low flux-corrected RK stage."""

    candidate: Array | None
    failed_reason: str
    riemann_fallbacks: int
    quadrature_reductions: int
    corrected_faces: int
    minimum_flux_weight: float


def _cell_average_matrix(offsets: Array, degree: int) -> Array:
    matrix = np.empty((len(offsets), degree + 1))
    for row, offset in enumerate(offsets):
        lower = float(offset) - 0.5
        upper = float(offset) + 0.5
        for power in range(degree + 1):
            matrix[row, power] = (
                upper ** (power + 1) - lower ** (power + 1)
            ) / (power + 1)
    return matrix


GAUSS_POINTS = np.array(
    (-0.5 * math.sqrt(3.0 / 5.0), 0.0, 0.5 * math.sqrt(3.0 / 5.0))
)
GAUSS_WEIGHTS = np.array((5.0 / 18.0, 4.0 / 9.0, 5.0 / 18.0))
QUADRATIC_OFFSETS = np.arange(-1, 2, dtype=float)
QUARTIC_OFFSETS = np.arange(-2, 3, dtype=float)
QUADRATIC_EVALUATION = np.vander(
    GAUSS_POINTS,
    3,
    increasing=True,
) @ np.linalg.inv(_cell_average_matrix(QUADRATIC_OFFSETS, 2))
QUARTIC_EVALUATION = np.vander(
    GAUSS_POINTS,
    5,
    increasing=True,
) @ np.linalg.inv(_cell_average_matrix(QUARTIC_OFFSETS, 4))
DISCRETE_MODAL_BASIS = np.linalg.qr(
    np.vander(QUARTIC_OFFSETS, 5, increasing=True)
)[0]


def _cell_index(index: int, cells: int, periodic: bool) -> int:
    if periodic:
        return index % cells
    return min(cells - 1, max(0, index))


def _stencil(
    values: Array,
    center: int,
    offsets: range | tuple[int, ...],
    periodic: bool,
) -> Array:
    return np.asarray(
        [
            values[_cell_index(center + offset, len(values), periodic)]
            for offset in offsets
        ]
    )


def _transition_weight(
    ratio: Array,
    threshold: float,
    exponent: int,
    smooth_plateau: float,
) -> Array:
    scaled = np.maximum(ratio, 0.0) / threshold
    return np.where(
        scaled <= smooth_plateau,
        1.0,
        1.0 / (1.0 + scaled**exponent),
    )


def modal_sensor_weights(
    values: Array,
    periodic: bool,
    parameters: CoupledFvParameters,
) -> tuple[Array, Array]:
    """Return shared nonlinear and top-difference weights per cell."""

    cells = len(values)
    nonlinear_weight = np.ones(cells)
    difference_weight = np.ones(cells)
    epsilon = 128.0 * np.finfo(float).eps
    for cell in range(cells):
        stencil = _stencil(values, cell, range(-2, 3), periodic)
        coefficients = DISCRETE_MODAL_BASIS.T @ stencil
        lower_power = np.sqrt(np.sum(coefficients[1:3] ** 2, axis=0))
        upper_power = np.sqrt(np.sum(coefficients[3:5] ** 2, axis=0))
        high_denominator = np.sqrt(
            np.sum(coefficients[1:4] ** 2, axis=0)
        )
        amplitude = np.max(np.abs(stencil), axis=0)
        variation = np.sqrt(np.sum(coefficients[1:5] ** 2, axis=0))
        active = variation > epsilon * np.maximum(1.0, amplitude)
        nonlinear_ratio = np.zeros_like(variation)
        high_ratio = np.zeros_like(variation)
        nonlinear_ratio[active] = upper_power[active] / np.maximum(
            lower_power[active],
            epsilon * np.maximum(1.0, amplitude[active]),
        )
        high_ratio[active] = np.abs(coefficients[4, active]) / np.maximum(
            high_denominator[active],
            epsilon * np.maximum(1.0, amplitude[active]),
        )
        state_nonlinear_ratio = float(np.max(nonlinear_ratio))
        state_high_ratio = float(np.max(high_ratio))
        nonlinear_weight[cell] = float(
            _transition_weight(
                np.asarray(state_nonlinear_ratio),
                parameters.nonlinear_mode_threshold,
                parameters.transition_exponent,
                parameters.smooth_mode_plateau,
            )
        )
        difference_weight[cell] = float(
            _transition_weight(
                np.asarray(state_high_ratio),
                parameters.high_mode_threshold,
                parameters.transition_exponent,
                parameters.smooth_mode_plateau,
            )
        )
    return nonlinear_weight, difference_weight


def coupled_modal_weights(
    values: Array,
    periodic: bool,
    parameters: CoupledFvParameters,
) -> tuple[Array, Array]:
    """Return one middle/high-order weight per cell for a vector state."""

    nonlinear, difference = modal_sensor_weights(
        values,
        periodic,
        parameters,
    )
    return nonlinear, np.minimum(nonlinear, difference)


def _polynomial_nodes(
    conserved: Array,
    theta_middle: Array,
    theta_high: Array,
    periodic: bool,
) -> Array:
    """Evaluate the coupled PWL/quadratic/quartic cell polynomial."""

    cells, components = conserved.shape
    result = np.empty((cells, len(GAUSS_POINTS), components))
    for cell in range(cells):
        center = conserved[cell]
        three = _stencil(conserved, cell, range(-1, 2), periodic)
        five = _stencil(conserved, cell, range(-2, 3), periodic)
        slopes = np.asarray(
            [
                mc_slope(three[0, component], center[component], three[2, component])
                for component in range(components)
            ]
        )
        linear = center[np.newaxis, :] + GAUSS_POINTS[:, np.newaxis] * slopes
        quadratic = QUADRATIC_EVALUATION @ three
        quartic = QUARTIC_EVALUATION @ five
        result[cell] = (
            linear
            + theta_middle[cell] * (quadratic - linear)
            + theta_high[cell] * (quartic - quadratic)
        )
    return result


def finite_volume_primitive_averages(
    conserved: Array,
    gamma: float,
    recover: BatchRecovery,
    periodic: bool,
    parameters: CoupledFvParameters,
) -> PreparedState:
    """Recover quadrature-consistent primitive cell averages.

    A single scale per cell contracts every quadrature state toward the valid
    conserved average.  Because each candidate polynomial has the correct
    central-cell average, this scaling preserves that average exactly.
    """

    base_primitive, base_success = recover(conserved, gamma)
    if not np.all(base_success):
        bad = np.flatnonzero(~base_success)
        raise RuntimeError(f"input cell averages are inadmissible: {bad[:12]}")
    theta_middle, theta_high = coupled_modal_weights(
        conserved,
        periodic,
        parameters,
    )
    if not parameters.primitive_quadrature:
        primitive_theta_middle, primitive_theta_high = coupled_modal_weights(
            base_primitive,
            periodic,
            parameters,
        )
        return PreparedState(
            primitive_average=base_primitive,
            base_primitive=base_primitive,
            theta_middle=np.minimum(theta_middle, primitive_theta_middle),
            theta_high=np.minimum(theta_high, primitive_theta_high),
            quadrature_scale=np.ones(len(conserved)),
            face_high_scale=np.ones(len(conserved)),
            quadrature_reductions=0,
        )
    raw_nodes = _polynomial_nodes(
        conserved,
        theta_middle,
        theta_high,
        periodic,
    )
    quadrature_scale = np.ones(len(conserved))
    primitive_nodes = np.empty_like(raw_nodes)
    reductions = 0
    for _ in range(parameters.maximum_admissibility_iterations + 1):
        nodes = conserved[:, np.newaxis, :] + quadrature_scale[
            :, np.newaxis, np.newaxis
        ] * (raw_nodes - conserved[:, np.newaxis, :])
        recovered, success = recover(
            nodes.reshape(-1, conserved.shape[1]),
            gamma,
        )
        success_by_cell = success.reshape(len(conserved), len(GAUSS_POINTS))
        bad_cells = ~np.all(success_by_cell, axis=1)
        if not np.any(bad_cells):
            primitive_nodes = recovered.reshape(raw_nodes.shape)
            break
        quadrature_scale[bad_cells] *= parameters.quadrature_reduction
        reductions += int(np.count_nonzero(bad_cells))
    else:
        bad = np.flatnonzero(bad_cells)
        raise RuntimeError(
            "quadrature admissibility scaling did not converge in cells "
            f"{bad[:12].tolist()}"
        )
    # A polynomial that had to be contracted onto the boundary of the
    # admissible set is a poor nonlinear quadrature model even when its final
    # nodes technically recover.  In ultra-relativistic states a tiny
    # conserved perturbation at that boundary can imply an O(1) primitive
    # perturbation.  Retain the finite-volume correction only where the raw
    # polynomial was admissible; otherwise use the known-valid cell-average
    # primitive for this representation step.  This is distinct from the
    # continuous face-order hierarchy below.
    admissible_quadrature_scale = quadrature_scale.copy()
    reduced_cells = admissible_quadrature_scale < 1.0
    if np.any(reduced_cells):
        representation_scale = admissible_quadrature_scale.copy()
        representation_scale[reduced_cells] = 0.0
        nodes = conserved[:, np.newaxis, :] + representation_scale[
            :, np.newaxis, np.newaxis
        ] * (raw_nodes - conserved[:, np.newaxis, :])
        recovered, success = recover(
            nodes.reshape(-1, conserved.shape[1]),
            gamma,
        )
        if not np.all(success):
            bad = np.flatnonzero(~success)
            raise RuntimeError(
                "cell-average primitive recovery failed after quadrature "
                f"reset: {bad[:12].tolist()}"
            )
        primitive_nodes = recovered.reshape(raw_nodes.shape)
    primitive_average = np.einsum(
        "q,nqc->nc",
        GAUSS_WEIGHTS,
        primitive_nodes,
    )
    primitive_theta_middle, primitive_theta_high = coupled_modal_weights(
        primitive_average,
        periodic,
        parameters,
    )
    return PreparedState(
        primitive_average=primitive_average,
        base_primitive=base_primitive,
        theta_middle=np.minimum(theta_middle, primitive_theta_middle),
        theta_high=np.minimum(theta_high, primitive_theta_high),
        quadrature_scale=admissible_quadrature_scale,
        face_high_scale=np.where(
            admissible_quadrature_scale < 1.0,
            0.0,
            1.0,
        ),
        quadrature_reductions=reductions,
    )


def _scalar_trace_hierarchy(
    values: Array,
    theta_high: float,
) -> float:
    middle = ppm4_left(*values)
    high = mp5_left(*values)
    return middle + theta_high * (high - middle)


def reconstruct_prepared_face(
    prepared: PreparedState,
    left_cell: int,
    periodic: bool,
    encode: Callable[[Array], Array],
    decode: Callable[[Array], Array],
    project: Callable[[Array, Array], Array],
    normal_magnetic_component: int | None = None,
) -> tuple[Array, Array]:
    cells = len(prepared.primitive_average)
    left_index = _cell_index(left_cell, cells, periodic)
    right_index = _cell_index(left_cell + 1, cells, periodic)
    theta_middle = min(
        prepared.theta_middle[left_index],
        prepared.theta_middle[right_index],
    )
    theta_high = min(
        prepared.theta_high[left_index],
        prepared.theta_high[right_index],
        prepared.face_high_scale[left_index],
        prepared.face_high_scale[right_index],
        theta_middle,
    )
    variables = encode(prepared.primitive_average)
    left_stencil = _stencil(variables, left_cell, range(-2, 4), periodic)
    right_stencil = left_stencil[::-1]
    left_variables = np.asarray(
        [
            _scalar_trace_hierarchy(
                left_stencil[:, component],
                theta_high,
            )
            for component in range(variables.shape[1])
        ]
    )
    right_variables = np.asarray(
        [
            _scalar_trace_hierarchy(
                right_stencil[:, component],
                theta_high,
            )
            for component in range(variables.shape[1])
        ]
    )
    left = decode(left_variables)
    right = decode(right_variables)
    left = project(left, prepared.base_primitive[left_index])
    right = project(right, prepared.base_primitive[right_index])
    if normal_magnetic_component is not None:
        normal_b = 0.5 * (
            prepared.base_primitive[left_index, normal_magnetic_component]
            + prepared.base_primitive[right_index, normal_magnetic_component]
        )
        left[normal_magnetic_component] = normal_b
        right[normal_magnetic_component] = normal_b
    return left, right


def _identity(values: Array) -> Array:
    return np.asarray(values)


def _newtonian_project(
    trace: Array,
    center: Array,
    pressure_component: int,
) -> Array:
    if np.all(np.isfinite(trace)) and trace[0] > 0.0 and trace[pressure_component] > 0.0:
        return trace
    theta = 1.0
    for component in (0, pressure_component):
        floor = 1.0e-13 * max(1.0, center[component])
        correction = trace[component] - center[component]
        if not np.isfinite(trace[component]):
            theta = 0.0
        elif correction < 0.0 and trace[component] < floor:
            theta = min(theta, (center[component] - floor) / (-correction))
    if not np.all(np.isfinite(trace)):
        theta = 0.0
    return center + max(0.0, min(1.0, theta)) * (trace - center)


class CoupledFvReconstruction:
    """Cache one prepared vector state for the existing HD/MHD harnesses."""

    def __init__(
        self,
        conserved_from_primitive: ConservedMap,
        primitive_from_conserved: Callable[[Array, float], Array],
        pressure_component: int,
        parameters: CoupledFvParameters,
        normal_magnetic_component: int | None = None,
    ) -> None:
        self.conserved_from_primitive = conserved_from_primitive
        self.primitive_from_conserved = primitive_from_conserved
        self.pressure_component = pressure_component
        self.parameters = parameters
        self.normal_magnetic_component = normal_magnetic_component
        self.periodic = False
        self._source: Array | None = None
        self._gamma = math.nan
        self._prepared: PreparedState | None = None
        self.total_quadrature_reductions = 0

    def _recover(self, state: Array, gamma: float) -> tuple[Array, Array]:
        primitive = self.primitive_from_conserved(state, gamma)
        success = (
            np.all(np.isfinite(primitive), axis=1)
            & (primitive[:, 0] > 0.0)
            & (primitive[:, self.pressure_component] > 0.0)
        )
        return primitive, success

    def _prepare(self, primitive: Array, gamma: float) -> PreparedState:
        if self._source is primitive and self._gamma == gamma:
            assert self._prepared is not None
            return self._prepared
        conserved = self.conserved_from_primitive(primitive, gamma)
        prepared = finite_volume_primitive_averages(
            conserved,
            gamma,
            self._recover,
            self.periodic,
            self.parameters,
        )
        self._source = primitive
        self._gamma = gamma
        self._prepared = prepared
        self.total_quadrature_reductions += prepared.quadrature_reductions
        return prepared

    def __call__(
        self,
        primitive: Array,
        left_cell: int,
        gamma: float,
    ) -> tuple[Array, Array]:
        prepared = self._prepare(primitive, gamma)
        return reconstruct_prepared_face(
            prepared,
            left_cell,
            self.periodic,
            _identity,
            _identity,
            lambda trace, center: _newtonian_project(
                trace,
                center,
                self.pressure_component,
            ),
            self.normal_magnetic_component,
        )


class CoupledHydroCharacteristic(CoupledFvReconstruction):
    def __call__(
        self,
        primitive: Array,
        left_cell: int,
        gamma: float,
    ) -> tuple[Array, Array]:
        prepared = self._prepare(primitive, gamma)
        cells = len(prepared.primitive_average)
        left_index = _cell_index(left_cell, cells, self.periodic)
        right_index = _cell_index(left_cell + 1, cells, self.periodic)
        stencil = _stencil(
            prepared.primitive_average,
            left_cell,
            range(-2, 4),
            self.periodic,
        )
        rho_ref, sound = hydro_characteristic_basis(
            stencil[2], stencil[3], gamma
        )
        waves = hydro_to_characteristics(stencil, rho_ref, sound)
        theta_middle = min(
            prepared.theta_middle[left_index],
            prepared.theta_middle[right_index],
        )
        theta_high = min(
            prepared.theta_high[left_index],
            prepared.theta_high[right_index],
            prepared.face_high_scale[left_index],
            prepared.face_high_scale[right_index],
            theta_middle,
        )
        left_waves = np.asarray(
            [
                _scalar_trace_hierarchy(
                    waves[:, wave],
                    theta_high,
                )
                for wave in range(3)
            ]
        )
        right_waves = np.asarray(
            [
                _scalar_trace_hierarchy(
                    waves[::-1, wave],
                    theta_high,
                )
                for wave in range(3)
            ]
        )
        left = hydro_from_characteristics(left_waves, rho_ref, sound)
        right = hydro_from_characteristics(right_waves, rho_ref, sound)
        return _newtonian_project(
            left, prepared.base_primitive[left_index], 2
        ), _newtonian_project(
            right, prepared.base_primitive[right_index], 2
        )


class CoupledMhdCharacteristic(CoupledFvReconstruction):
    def __call__(
        self,
        primitive: Array,
        left_cell: int,
        gamma: float,
    ) -> tuple[Array, Array]:
        prepared = self._prepare(primitive, gamma)
        cells = len(prepared.primitive_average)
        left_index = _cell_index(left_cell, cells, self.periodic)
        right_index = _cell_index(left_cell + 1, cells, self.periodic)
        stencil = _stencil(
            prepared.primitive_average,
            left_cell,
            range(-2, 4),
            self.periodic,
        )
        normal_b = 0.5 * (stencil[2, 5] + stencil[3, 5])
        basis = mhd_characteristic_basis(
            stencil[2], stencil[3], gamma, normal_b
        )
        waves = mhd_to_characteristics(stencil[:, MHD_REDUCED], basis)
        theta_middle = min(
            prepared.theta_middle[left_index],
            prepared.theta_middle[right_index],
        )
        theta_high = min(
            prepared.theta_high[left_index],
            prepared.theta_high[right_index],
            prepared.face_high_scale[left_index],
            prepared.face_high_scale[right_index],
            theta_middle,
        )
        left_waves = np.asarray(
            [
                _scalar_trace_hierarchy(
                    waves[:, wave],
                    theta_high,
                )
                for wave in range(7)
            ]
        )
        right_waves = np.asarray(
            [
                _scalar_trace_hierarchy(
                    waves[::-1, wave],
                    theta_high,
                )
                for wave in range(7)
            ]
        )
        left_reduced = mhd_from_characteristics(left_waves, basis)
        right_reduced = mhd_from_characteristics(right_waves, basis)
        left = np.insert(left_reduced, 5, normal_b)
        right = np.insert(right_reduced, 5, normal_b)
        return _newtonian_project(
            left, prepared.base_primitive[left_index], 4
        ), _newtonian_project(
            right, prepared.base_primitive[right_index], 4
        )


def hydro_candidate(parameters: CoupledFvParameters) -> CoupledFvReconstruction:
    return CoupledHydroCharacteristic(
        hydro_conserved_from_primitive,
        hydro_primitive_from_conserved,
        pressure_component=2,
        parameters=parameters,
    )


def mhd_candidate(parameters: CoupledFvParameters) -> CoupledFvReconstruction:
    return CoupledMhdCharacteristic(
        mhd_conserved_array,
        mhd_primitive_from_conserved,
        pressure_component=4,
        parameters=parameters,
        normal_magnetic_component=5,
    )


def _rmhd_recover(state: Array, gamma: float) -> tuple[Array, Array]:
    recovery = recover_conserved(state.T, gamma)
    return recovery.primitive.T, recovery.success


def _encode_rmhd(primitive: Array) -> Array:
    velocity_squared = np.sum(primitive[:, 1:4] ** 2, axis=1)
    lorentz = 1.0 / np.sqrt(1.0 - velocity_squared)
    values = primitive.copy()
    values[:, 1:4] *= lorentz[:, np.newaxis]
    return values


def _decode_rmhd(values: Array) -> Array:
    lorentz = math.sqrt(1.0 + float(np.dot(values[1:4], values[1:4])))
    primitive = values.copy()
    primitive[1:4] /= lorentz
    return primitive


def _decode_rmhd_batch(values: Array) -> Array:
    primitive = values.copy()
    lorentz = np.sqrt(1.0 + np.sum(values[:, 1:4] ** 2, axis=1))
    primitive[:, 1:4] /= lorentz[:, np.newaxis]
    return primitive


def prepare_rmhd(
    state: Array,
    gamma: float,
    periodic: bool,
    parameters: CoupledFvParameters,
) -> PreparedState:
    prepared = finite_volume_primitive_averages(
        state.T,
        gamma,
        _rmhd_recover,
        periodic,
        parameters,
    )
    # If the quartic conserved polynomial cannot be recovered at its
    # quadrature nodes, neither its nonlinear primitive average nor a local
    # characteristic basis formed from that representation is trustworthy.
    # Suppress both high-order corrections throughout their stencil halo;
    # this conditioning statement does not depend on a problem-specific
    # Lorentz-factor threshold.
    ill_conditioned = (
        prepared.quadrature_scale
        < parameters.characteristic_quadrature_threshold
    )
    conditioned_halo = ill_conditioned.copy()
    for offset in range(1, parameters.characteristic_conditioning_halo + 1):
        conditioned_halo[offset:] |= ill_conditioned[:-offset]
        conditioned_halo[:-offset] |= ill_conditioned[offset:]
    return dataclasses.replace(
        prepared,
        face_high_scale=np.where(conditioned_halo, 0.0, 1.0),
    )


def rmhd_faces(
    prepared: PreparedState,
    gamma: float,
    periodic: bool,
) -> tuple[Array, Array]:
    cells = len(prepared.primitive_average)
    left = np.empty((8, cells + 1))
    right = np.empty((8, cells + 1))
    for face in range(cells + 1):
        left_face, right_face = reconstruct_prepared_face(
            prepared,
            face - 1,
            periodic,
            _encode_rmhd,
            _decode_rmhd,
            lambda trace, center: rmhd_limited_face(
                trace.copy(), center, gamma
            ),
            normal_magnetic_component=5,
        )
        left[:, face] = left_face
        right[:, face] = right_face
    return left, right


RMHD_REDUCED = np.array((0, 1, 2, 3, 4, 6, 7))


def _rmhd_characteristic_matrices(
    prepared: PreparedState,
    gamma: float,
    periodic: bool,
) -> tuple[Array, Array, Array]:
    """Numerically form the local primitive-variable SRMHD eigensystem."""

    cells = len(prepared.primitive_average)
    encoded = _encode_rmhd(prepared.primitive_average)
    reference = np.empty((cells + 1, len(RMHD_REDUCED)))
    normal_b = np.empty(cells + 1)
    for face in range(cells + 1):
        left_index = _cell_index(face - 1, cells, periodic)
        right_index = _cell_index(face, cells, periodic)
        reference[face] = 0.5 * (
            encoded[left_index, RMHD_REDUCED]
            + encoded[right_index, RMHD_REDUCED]
        )
        normal_b[face] = 0.5 * (
            prepared.base_primitive[left_index, 5]
            + prepared.base_primitive[right_index, 5]
        )

    variables = len(RMHD_REDUCED)
    perturbations = np.empty((2 * variables * (cells + 1), 8))
    steps = np.empty((cells + 1, variables))
    for face in range(cells + 1):
        for variable in range(variables):
            step = 2.0e-6 * max(1.0, abs(reference[face, variable]))
            if variable in (0, 4):
                step = min(step, 0.2 * reference[face, variable])
            steps[face, variable] = step
            for side, sign in enumerate((-1.0, 1.0)):
                row = (face * variables + variable) * 2 + side
                perturbed = reference[face].copy()
                perturbed[variable] += sign * step
                encoded_full = np.insert(perturbed, 5, normal_b[face])
                perturbations[row] = _decode_rmhd(encoded_full)

    conserved, flux = primitive_state_and_flux(perturbations.T, gamma)
    conserved = conserved[RMHD_REDUCED].T.reshape(
        cells + 1, variables, 2, variables
    )
    flux = flux[RMHD_REDUCED].T.reshape(
        cells + 1, variables, 2, variables
    )
    right_matrices = np.empty((cells + 1, variables, variables))
    left_matrices = np.empty_like(right_matrices)
    valid = np.ones(cells + 1, dtype=bool)
    for face in range(cells + 1):
        conserved_jacobian = np.empty((variables, variables))
        flux_jacobian = np.empty((variables, variables))
        for variable in range(variables):
            denominator = 2.0 * steps[face, variable]
            conserved_jacobian[:, variable] = (
                conserved[face, variable, 1]
                - conserved[face, variable, 0]
            ) / denominator
            flux_jacobian[:, variable] = (
                flux[face, variable, 1] - flux[face, variable, 0]
            ) / denominator
        try:
            primitive_jacobian = np.linalg.solve(
                conserved_jacobian,
                flux_jacobian,
            )
            eigenvalues, right = np.linalg.eig(primitive_jacobian)
            if np.max(np.abs(np.imag(eigenvalues))) > 2.0e-6:
                raise np.linalg.LinAlgError("complex SRMHD characteristics")
            if np.max(np.abs(np.imag(right))) > 2.0e-5:
                raise np.linalg.LinAlgError("complex SRMHD eigenvectors")
            order = np.argsort(np.real(eigenvalues))
            right = np.real(right[:, order])
            right /= np.linalg.norm(right, axis=0)[np.newaxis, :]
            if np.linalg.cond(right) > 1.0e9:
                raise np.linalg.LinAlgError("ill-conditioned SRMHD basis")
            left = np.linalg.inv(right)
        except np.linalg.LinAlgError:
            valid[face] = False
            right = np.eye(variables)
            left = np.eye(variables)
        right_matrices[face] = right
        left_matrices[face] = left
    return right_matrices, left_matrices, valid


def rmhd_characteristic_faces(
    prepared: PreparedState,
    gamma: float,
    periodic: bool,
    parameters: CoupledFvParameters,
) -> tuple[Array, Array, int]:
    cells = len(prepared.primitive_average)
    encoded = _encode_rmhd(prepared.primitive_average)
    right_matrices, left_matrices, valid = _rmhd_characteristic_matrices(
        prepared,
        gamma,
        periodic,
    )
    component_left, component_right = rmhd_faces(
        prepared,
        gamma,
        periodic,
    )
    left_faces = component_left.copy()
    right_faces = component_right.copy()
    fallback_count = int(np.count_nonzero(~valid))
    for face in range(cells + 1):
        if not valid[face]:
            continue
        left_cell = face - 1
        left_index = _cell_index(left_cell, cells, periodic)
        right_index = _cell_index(face, cells, periodic)
        if min(
            prepared.face_high_scale[left_index],
            prepared.face_high_scale[right_index],
        ) <= 0.0:
            # The same near-causal conditioning test that suppresses the MP5
            # correction also suppresses the numerically conditioned
            # characteristic basis.  Retain the already-computed component
            # PPM trace on this face.
            continue
        reference = 0.5 * (
            encoded[left_index, RMHD_REDUCED]
            + encoded[right_index, RMHD_REDUCED]
        )
        stencil = _stencil(encoded, left_cell, range(-2, 4), periodic)
        reduced = stencil[:, RMHD_REDUCED]
        waves = (
            left_matrices[face] @ (reduced - reference).T
        ).T
        theta_middle = min(
            prepared.theta_middle[left_index],
            prepared.theta_middle[right_index],
        )
        theta_high = min(
            prepared.theta_high[left_index],
            prepared.theta_high[right_index],
            prepared.face_high_scale[left_index],
            prepared.face_high_scale[right_index],
            theta_middle,
        )
        left_waves = np.asarray(
            [
                _scalar_trace_hierarchy(
                    waves[:, wave],
                    theta_high,
                )
                for wave in range(len(RMHD_REDUCED))
            ]
        )
        right_waves = np.asarray(
            [
                _scalar_trace_hierarchy(
                    waves[::-1, wave],
                    theta_high,
                )
                for wave in range(len(RMHD_REDUCED))
            ]
        )
        left_reduced = reference + right_matrices[face] @ left_waves
        right_reduced = reference + right_matrices[face] @ right_waves
        normal_b = 0.5 * (
            prepared.base_primitive[left_index, 5]
            + prepared.base_primitive[right_index, 5]
        )
        left_encoded = np.insert(left_reduced, 5, normal_b)
        right_encoded = np.insert(right_reduced, 5, normal_b)
        left_face = rmhd_limited_face(
            _decode_rmhd(left_encoded),
            prepared.base_primitive[left_index],
            gamma,
        )
        right_face = rmhd_limited_face(
            _decode_rmhd(right_encoded),
            prepared.base_primitive[right_index],
            gamma,
        )
        left_faces[:, face] = left_face
        right_faces[:, face] = right_face
    return left_faces, right_faces, fallback_count


def rmhd_flux(
    state: Array,
    gamma: float,
    riemann: str,
    parameters: CoupledFvParameters,
) -> tuple[Array | None, int, int]:
    try:
        prepared = prepare_rmhd(state, gamma, False, parameters)
    except RuntimeError:
        return None, 0, 0
    if parameters.rmhd_characteristic:
        left, right, characteristic_fallbacks = rmhd_characteristic_faces(
            prepared,
            gamma,
            False,
            parameters,
        )
    else:
        left, right = rmhd_faces(
            prepared,
            gamma,
            False,
        )
        characteristic_fallbacks = 0
    flux, fallbacks = select_riemann_flux(
        riemann_fluxes(left, right, gamma),
        riemann,
    )
    return (
        flux,
        fallbacks + characteristic_fallbacks,
        prepared.quadrature_reductions,
    )


def rmhd_rhs(
    state: Array,
    gamma: float,
    dx: float,
    riemann: str,
    parameters: CoupledFvParameters,
) -> tuple[Array | None, int, int]:
    flux, fallbacks, reductions = rmhd_flux(
        state,
        gamma,
        riemann,
        parameters,
    )
    if flux is None:
        return None, fallbacks, reductions
    return -(flux[:, 1:] - flux[:, :-1]) / dx, fallbacks, reductions


def rmhd_control_flux(
    state: Array,
    gamma: float,
    riemann: str,
    reconstruction: str,
) -> tuple[Array | None, int, int]:
    recovery = recover_conserved(state, gamma)
    if not np.all(recovery.success):
        return None, 0, 0
    left, right = reconstruct_rmhd_faces(
        recovery.primitive,
        reconstruction,
        gamma,
    )
    flux, fallbacks = select_riemann_flux(
        riemann_fluxes(left, right, gamma),
        riemann,
    )
    return flux, fallbacks, 0


def rmhd_control_rhs(
    state: Array,
    gamma: float,
    dx: float,
    riemann: str,
    reconstruction: str,
) -> tuple[Array | None, int, int]:
    flux, fallbacks, reductions = rmhd_control_flux(
        state,
        gamma,
        riemann,
        reconstruction,
    )
    if flux is None:
        return None, fallbacks, reductions
    return -(flux[:, 1:] - flux[:, :-1]) / dx, fallbacks, reductions


def _rmhd_metrics(
    case: int,
    method: str,
    state: Array,
    exact: Array,
    completed: bool,
    failed_stage: str,
    riemann_fallbacks: int,
    stage_fallbacks: int,
    corrected_faces: int,
    minimum_flux_weight: float,
    quadrature_reductions: int,
) -> RmhdResult:
    scale = np.maximum(np.max(np.abs(exact), axis=1), np.ptp(exact, axis=1))
    scale = np.maximum(scale, 1.0e-14)
    error = np.abs(state - exact) / scale[:, np.newaxis]
    component_l1 = np.mean(error, axis=1)
    component_linf = np.max(error, axis=1)
    overshoot = np.maximum(
        np.max(state, axis=1) - np.max(exact, axis=1),
        np.min(exact, axis=1) - np.min(state, axis=1),
    ) / scale
    numerical_tv = np.sum(np.abs(np.diff(state, axis=1)), axis=1)
    exact_tv = np.sum(np.abs(np.diff(exact, axis=1)), axis=1)
    active = exact_tv > 1.0e-14 * scale
    tv_ratio = np.ones_like(exact_tv)
    tv_ratio[active] = numerical_tv[active] / exact_tv[active]
    return RmhdResult(
        case=case,
        method=method,
        cells=state.shape[1],
        final_time=math.nan,
        completed=completed,
        failed_stage=failed_stage,
        maximum_l1=float(np.max(component_l1)),
        maximum_linf=float(np.max(component_linf)),
        maximum_overshoot=max(0.0, float(np.max(overshoot))),
        maximum_tv_ratio=float(np.max(tv_ratio)),
        riemann_fallbacks=riemann_fallbacks,
        stage_fallbacks=stage_fallbacks,
        corrected_faces=corrected_faces,
        minimum_flux_weight=minimum_flux_weight,
        quadrature_reductions=quadrature_reductions,
    )


def _rmhd_candidate_bad_cells(
    candidate: Array,
    bound_state: Array,
    gamma: float,
    dmp_factor: float,
    *,
    enforce_dmp: bool,
) -> Array:
    """Return cells failing recovery or the relaxed density/energy DMP."""

    bad = ~recover_conserved(candidate, gamma).success
    if enforce_dmp:
        padded = np.pad(bound_state, ((0, 0), (1, 1)), mode="edge")
        for component in (0, 4):
            neighborhood = np.stack(
                (
                    padded[component, :-2],
                    padded[component, 1:-1],
                    padded[component, 2:],
                )
            )
            lower = np.min(neighborhood, axis=0)
            upper = np.max(neighborhood, axis=0)
            bad |= candidate[component] < lower / dmp_factor
            bad |= candidate[component] > dmp_factor * upper
    return np.flatnonzero(bad)


def _rmhd_candidate_guarded_stage(
    affine_base: Array,
    stage_state: Array,
    rk_weight: float,
    dt: float,
    dx: float,
    gamma: float,
    evaluate_high_flux: Callable[[Array], tuple[Array | None, int, int]],
    evaluate_pwc_flux: Callable[[Array], tuple[Array | None, int, int]],
    dmp_factor: float,
) -> LimitedStage:
    """Apply the production-style a-posteriori PWC candidate guard.

    The high-order RK candidate is tested against recovery and the same relaxed
    density/energy DMP used by the production RMHD guard.  Every face adjacent
    to a troubled cell is replaced by the shared PWC flux.  The cell mask then
    grows monotonically only when that shared-face replacement makes a neighbor
    unrecoverable.
    """

    high_flux, fallbacks, reductions = evaluate_high_flux(stage_state)
    if high_flux is None:
        return LimitedStage(None, "high_flux", fallbacks, reductions, 0, 1.0)
    high_rhs = -(high_flux[:, 1:] - high_flux[:, :-1]) / dx
    high_candidate = affine_base + rk_weight * dt * high_rhs

    failed_cells = _rmhd_candidate_bad_cells(
        high_candidate,
        stage_state,
        gamma,
        dmp_factor,
        enforce_dmp=True,
    )
    if failed_cells.size == 0:
        return LimitedStage(high_candidate, "", fallbacks, reductions, 0, 1.0)

    pwc_flux, pwc_fallbacks, _ = evaluate_pwc_flux(stage_state)
    fallbacks += pwc_fallbacks
    if pwc_flux is None:
        return LimitedStage(None, "pwc_flux", fallbacks, reductions, 0, 1.0)

    troubled = np.zeros(stage_state.shape[1], dtype=bool)
    troubled[failed_cells] = True
    for _ in range(stage_state.shape[1] + 1):
        corrected_faces = np.zeros(stage_state.shape[1] + 1, dtype=bool)
        corrected_faces[:-1] |= troubled
        corrected_faces[1:] |= troubled
        selected_flux = high_flux.copy()
        selected_flux[:, corrected_faces] = pwc_flux[:, corrected_faces]
        rhs = -(selected_flux[:, 1:] - selected_flux[:, :-1]) / dx
        candidate = affine_base + rk_weight * dt * rhs
        newly_bad = _rmhd_candidate_bad_cells(
            candidate,
            stage_state,
            gamma,
            dmp_factor,
            enforce_dmp=False,
        )
        newly_bad = newly_bad[~troubled[newly_bad]]
        if newly_bad.size == 0:
            return LimitedStage(
                candidate,
                "",
                fallbacks,
                reductions,
                int(np.count_nonzero(corrected_faces)),
                0.0,
            )
        troubled[newly_bad] = True
    return LimitedStage(None, "closure", fallbacks, reductions, 0, 0.0)


def _rmhd_candidate_guarded_ssprk3_step(
    state: Array,
    dt: float,
    dx: float,
    gamma: float,
    evaluate_high_flux: Callable[[Array], tuple[Array | None, int, int]],
    evaluate_pwc_flux: Callable[[Array], tuple[Array | None, int, int]],
    dmp_factor: float,
) -> tuple[Array | None, int, int, int, float, int, str]:
    """Apply the same candidate guard to every concrete SSPRK3 stage."""

    stages: list[LimitedStage] = []
    first = _rmhd_candidate_guarded_stage(
        state,
        state,
        1.0,
        dt,
        dx,
        gamma,
        evaluate_high_flux,
        evaluate_pwc_flux,
        dmp_factor,
    )
    stages.append(first)
    if first.candidate is None:
        return _candidate_guard_failure(stages, "stage1")

    second = _rmhd_candidate_guarded_stage(
        0.75 * state + 0.25 * first.candidate,
        first.candidate,
        0.25,
        dt,
        dx,
        gamma,
        evaluate_high_flux,
        evaluate_pwc_flux,
        dmp_factor,
    )
    stages.append(second)
    if second.candidate is None:
        return _candidate_guard_failure(stages, "stage2")

    third = _rmhd_candidate_guarded_stage(
        (1.0 / 3.0) * state + (2.0 / 3.0) * second.candidate,
        second.candidate,
        2.0 / 3.0,
        dt,
        dx,
        gamma,
        evaluate_high_flux,
        evaluate_pwc_flux,
        dmp_factor,
    )
    stages.append(third)
    if third.candidate is None:
        return _candidate_guard_failure(stages, "stage3")
    corrected_faces = sum(stage.corrected_faces for stage in stages)
    return (
        third.candidate,
        sum(stage.riemann_fallbacks for stage in stages),
        sum(stage.quadrature_reductions for stage in stages),
        corrected_faces,
        0.0 if corrected_faces else 1.0,
        sum(stage.corrected_faces > 0 for stage in stages),
        "",
    )


def _candidate_guard_failure(
    stages: list[LimitedStage],
    stage_name: str,
) -> tuple[None, int, int, int, float, int, str]:
    corrected_faces = sum(stage.corrected_faces for stage in stages)
    return (
        None,
        sum(stage.riemann_fallbacks for stage in stages),
        sum(stage.quadrature_reductions for stage in stages),
        corrected_faces,
        0.0 if corrected_faces else 1.0,
        sum(stage.corrected_faces > 0 for stage in stages),
        f"{stage_name}_{stages[-1].failed_reason}",
    )


def _rmhd_ssprk3_step(
    state: Array,
    dt: float,
    gamma: float,
    evaluate_rhs: Callable[[Array], tuple[Array | None, int, int]],
) -> tuple[Array | None, int, int, str]:
    """Attempt one complete SSPRK3 step with one spatial operator."""

    fallback_count = 0
    reduction_count = 0
    first_rhs, fallbacks, reductions = evaluate_rhs(state)
    fallback_count += fallbacks
    reduction_count += reductions
    if first_rhs is None:
        return None, fallback_count, reduction_count, "stage1_rhs"
    first = state + dt * first_rhs
    if not np.all(recover_conserved(first, gamma).success):
        return None, fallback_count, reduction_count, "stage1_candidate"

    second_rhs, fallbacks, reductions = evaluate_rhs(first)
    fallback_count += fallbacks
    reduction_count += reductions
    if second_rhs is None:
        return None, fallback_count, reduction_count, "stage2_rhs"
    second = 0.75 * state + 0.25 * (first + dt * second_rhs)
    if not np.all(recover_conserved(second, gamma).success):
        return None, fallback_count, reduction_count, "stage2_candidate"

    third_rhs, fallbacks, reductions = evaluate_rhs(second)
    fallback_count += fallbacks
    reduction_count += reductions
    if third_rhs is None:
        return None, fallback_count, reduction_count, "stage3_rhs"
    candidate = (1.0 / 3.0) * state + (2.0 / 3.0) * (
        second + dt * third_rhs
    )
    if not np.all(recover_conserved(candidate, gamma).success):
        return None, fallback_count, reduction_count, "accepted_candidate"
    return candidate, fallback_count, reduction_count, ""


def evolve_rmhd_candidate(
    case: int,
    cells: int,
    final_time: float,
    cfl: float,
    riemann: str,
    parameters: CoupledFvParameters,
) -> tuple[RmhdResult, Array, Array]:
    spec = case_spec(case)
    gamma = float(spec["gamma"])
    profile = read_last_exact_balsara_profile(
        EXACT_BALSARA_ROOT / f"solution_b{case}.dat"
    )
    edges = np.linspace(-1.0, 1.0, cells + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    dx = 2.0 / cells
    initial_primitive = exact_primitives_at(profile, spec, 0.0, centers)
    state, _ = primitive_state_and_flux(initial_primitive, gamma)
    time = 0.0
    failed_stage = ""
    fallback_count = 0
    stage_fallback_count = 0
    corrected_face_count = 0
    minimum_flux_weight = 1.0
    reduction_count = 0
    while time < final_time - 32.0 * np.finfo(float).eps:
        dt = min(final_time - time, cfl * dx)
        (
            candidate,
            fallbacks,
            reductions,
            corrected_faces,
            step_minimum_flux_weight,
            replay_attempts,
            candidate_failure,
        ) = _rmhd_candidate_guarded_ssprk3_step(
                state,
                dt,
                dx,
                gamma,
                lambda stage_state: rmhd_flux(
                    stage_state,
                    gamma,
                    riemann,
                    parameters,
                ),
                lambda stage_state: rmhd_control_flux(
                    stage_state,
                    gamma,
                    riemann,
                    "pwc",
                ),
                parameters.candidate_dmp_factor,
            )
        fallback_count += fallbacks
        reduction_count += reductions
        stage_fallback_count += replay_attempts
        corrected_face_count += corrected_faces
        minimum_flux_weight = min(
            minimum_flux_weight,
            step_minimum_flux_weight,
        )
        if candidate is None:
            low_failure = ""
            for low_reconstruction in ("ppm4", "pwl", "pwc"):
                stage_fallback_count += 1
                candidate, fallbacks, reductions, low_failure = (
                    _rmhd_ssprk3_step(
                        state,
                        dt,
                        gamma,
                        lambda stage_state, method=low_reconstruction: (
                            rmhd_control_rhs(
                                stage_state,
                                gamma,
                                dx,
                                riemann,
                                method,
                            )
                        ),
                    )
                )
                fallback_count += fallbacks
                reduction_count += reductions
                if candidate is not None:
                    break
            else:
                failed_stage = (
                    f"candidate:{candidate_failure};low:{low_failure}"
                )
                break
        state = candidate
        time += dt
    exact = exact_average_at_time(profile, spec, time, edges)
    result = _rmhd_metrics(
        case,
        (
            f"coupled_fv5_characteristic_{riemann}"
            if parameters.rmhd_characteristic
            else f"coupled_fv5_{riemann}"
        ),
        state,
        exact,
        not failed_stage,
        failed_stage,
        fallback_count,
        stage_fallback_count,
        corrected_face_count,
        minimum_flux_weight,
        reduction_count,
    )
    result = dataclasses.replace(result, final_time=time)
    return result, state, exact


def evolve_rmhd_control(
    case: int,
    cells: int,
    final_time: float,
    cfl: float,
    reconstruction: str,
    riemann: str,
) -> tuple[RmhdResult, Array, Array]:
    spec = case_spec(case)
    gamma = float(spec["gamma"])
    profile = read_last_exact_balsara_profile(
        EXACT_BALSARA_ROOT / f"solution_b{case}.dat"
    )
    edges = np.linspace(-1.0, 1.0, cells + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    dx = 2.0 / cells
    initial_primitive = exact_primitives_at(profile, spec, 0.0, centers)
    state, _ = primitive_state_and_flux(initial_primitive, gamma)
    time = 0.0
    failed_stage = ""
    fallback_count = 0
    while time < final_time - 32.0 * np.finfo(float).eps:
        dt = min(final_time - time, cfl * dx)
        first_rhs, fallbacks, _ = rmhd_control_rhs(
            state, gamma, dx, riemann, reconstruction
        )
        fallback_count += fallbacks
        if first_rhs is None:
            failed_stage = "stage1"
            break
        first = state + dt * first_rhs
        second_rhs, fallbacks, _ = rmhd_control_rhs(
            first, gamma, dx, riemann, reconstruction
        )
        fallback_count += fallbacks
        if second_rhs is None:
            failed_stage = "stage2"
            break
        second = 0.75 * state + 0.25 * (first + dt * second_rhs)
        third_rhs, fallbacks, _ = rmhd_control_rhs(
            second, gamma, dx, riemann, reconstruction
        )
        fallback_count += fallbacks
        if third_rhs is None:
            failed_stage = "stage3"
            break
        state = (1.0 / 3.0) * state + (2.0 / 3.0) * (
            second + dt * third_rhs
        )
        recovery = recover_conserved(state, gamma)
        if not np.all(recovery.success):
            failed_stage = "accepted"
            break
        time += dt
    exact = exact_average_at_time(profile, spec, time, edges)
    result = _rmhd_metrics(
        case,
        f"{reconstruction}_{riemann}",
        state,
        exact,
        not failed_stage,
        failed_stage,
        fallback_count,
        0,
        0,
        1.0,
        0,
    )
    result = dataclasses.replace(result, final_time=time)
    return result, state, exact


def print_dataclass_rows(rows: list[object]) -> None:
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


def run_hydro(
    cells_values: tuple[int, ...],
    cfl: float,
    parameters: CoupledFvParameters,
) -> None:
    gamma = 1.4
    methods = {
        "pwl": component_reconstruction(
            lambda _vmm, vm, v0, vp, _vpp, _vppp: (
                v0 + 0.5 * mc_slope(vm, v0, vp)
            )
        ),
        "ppm4": component_reconstruction(ppm4_left),
        "mp5_component": component_reconstruction(mp5_left),
        "mp5_characteristic": hydro_characteristic_mp5,
        "coupled_fv5": hydro_candidate(parameters),
    }
    rows = []
    for cells in cells_values:
        for name, reconstruct in methods.items():
            rows.append(
                entropy_wave_result(
                    name,
                    cells,
                    cfl,
                    gamma,
                    reconstruct,
                    hllc_flux,
                )
            )
            x = (np.arange(cells) + 0.5) / cells
            for case in SHOCK_CASES:
                primitive = np.empty((cells, 3))
                primitive[x < 0.5] = case.left
                primitive[x >= 0.5] = case.right
                evolved, completed = evolve_hydro(
                    primitive,
                    case.final_time,
                    cfl,
                    gamma,
                    reconstruct,
                    hllc_flux,
                    periodic=False,
                )
                rows.append(
                    shock_result(
                        case,
                        name,
                        cells,
                        evolved,
                        completed,
                        gamma,
                    )
                )
    print("\n[Newtonian HD]")
    print_dataclass_rows(rows)


def run_mhd(
    cells_values: tuple[int, ...],
    cfl: float,
    reference_factor: int,
    parameters: CoupledFvParameters,
) -> None:
    methods = {
        "mp5_characteristic": mhd_characteristic_mp5,
        "balanced_metric_mp5": mhd_balanced_mp5,
        "coupled_fv5": mhd_candidate(parameters),
    }
    rows = []
    for cells in cells_values:
        for name, reconstruct in methods.items():
            rows.append(
                circular_alfven_result(
                    name,
                    cells,
                    cfl,
                    reconstruct,
                    mhd_hlld_flux,
                )
            )
        rows.extend(
            brio_wu_results(
                methods,
                {"hlld": mhd_hlld_flux},
                cells,
                cfl,
                reference_factor,
            )
        )
    print("\n[Newtonian MHD]")
    print_dataclass_rows(rows)


def run_rmhd(
    cells: int,
    cfl: float,
    final_mode: str,
    parameters: CoupledFvParameters,
    requested_cases: tuple[int, ...],
    profile_output: Path | None,
) -> None:
    rows = []
    profiles: dict[str, Array] = {
        "x": (np.arange(cells) + 0.5) * (2.0 / cells) - 1.0,
    }
    cases = requested_cases
    for case in cases:
        print(
            f"running SRMHD Balsara {case} ({final_mode}, N={cells})",
            flush=True,
        )
        spec = case_spec(case)
        final_time = (
            4.0 * cfl * (2.0 / cells)
            if final_mode == "short"
            else float(spec["time"])
        )
        for reconstruction in ("ppm4", "mp5"):
            result, state, exact = evolve_rmhd_control(
                case,
                cells,
                final_time,
                cfl,
                reconstruction,
                "hlld",
            )
            rows.append(result)
            profiles[f"b{case}_{reconstruction}"] = state
            profiles[f"b{case}_exact"] = exact
        result, state, exact = evolve_rmhd_candidate(
            case,
            cells,
            final_time,
            cfl,
            "hlld",
            parameters,
        )
        rows.append(result)
        profiles[f"b{case}_candidate"] = state
        profiles[f"b{case}_exact"] = exact
    print("\n[SRMHD]")
    print_dataclass_rows(rows)
    if profile_output is not None:
        profile_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(profile_output, **profiles)
        print(f"wrote {profile_output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--systems",
        nargs="+",
        choices=("hd", "mhd", "rmhd"),
        default=("hd", "mhd", "rmhd"),
    )
    parser.add_argument("--cells", type=int, nargs="+", default=(32, 64))
    parser.add_argument("--cfl", type=float, default=0.4)
    parser.add_argument("--reference-factor", type=int, default=4)
    parser.add_argument("--rmhd-mode", choices=("short", "full"), default="short")
    parser.add_argument(
        "--rmhd-cases", type=int, nargs="+", default=(1, 2, 3, 4, 5)
    )
    parser.add_argument("--high-threshold", type=float, default=0.10)
    parser.add_argument("--nonlinear-threshold", type=float, default=0.35)
    parser.add_argument("--transition-exponent", type=int, default=4)
    parser.add_argument("--no-fv-primitive", action="store_true")
    parser.add_argument(
        "--rmhd-characteristic",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--characteristic-conditioning-halo", type=int, default=3
    )
    parser.add_argument(
        "--characteristic-quadrature-threshold", type=float, default=1.0
    )
    parser.add_argument("--smooth-mode-plateau", type=float, default=0.025)
    parser.add_argument("--candidate-dmp-factor", type=float, default=1.05)
    parser.add_argument("--profile-output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.cfl <= 0.0:
        raise ValueError("--cfl must be positive")
    parameters = CoupledFvParameters(
        high_mode_threshold=args.high_threshold,
        nonlinear_mode_threshold=args.nonlinear_threshold,
        transition_exponent=args.transition_exponent,
        primitive_quadrature=not args.no_fv_primitive,
        rmhd_characteristic=args.rmhd_characteristic,
        characteristic_conditioning_halo=args.characteristic_conditioning_halo,
        characteristic_quadrature_threshold=(
            args.characteristic_quadrature_threshold
        ),
        smooth_mode_plateau=args.smooth_mode_plateau,
        candidate_dmp_factor=args.candidate_dmp_factor,
    )
    cells_values = tuple(args.cells)
    if "hd" in args.systems:
        run_hydro(cells_values, args.cfl, parameters)
    if "mhd" in args.systems:
        run_mhd(cells_values, args.cfl, args.reference_factor, parameters)
    if "rmhd" in args.systems:
        run_rmhd(
            max(cells_values),
            args.cfl,
            args.rmhd_mode,
            parameters,
            tuple(args.rmhd_cases),
            args.profile_output,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
