#!/usr/bin/env python3
"""Falsify subcell BVD reconstructions before production integration.

This standalone Newtonian Euler harness compares characteristic MP5 with two
local cascade-BVD precursors.  Each precursor offers a high-order polynomial
and two THINC profiles (beta 1.1 and 1.6) for every cell:

* ``wavewise`` selects a candidate independently in each characteristic wave;
* ``shared`` selects one candidate for the complete characteristic state.

The selection minimizes the two face jumps adjacent to a cell.  It follows the
P4-THINC-BVD construction closely, but uses a local, order-independent
selection instead of the paper's mutating three-cell sweep.  Consequently this
is a research precursor, not a reproduction claim.  ``mp5`` and ``p4`` can
both be used as the smooth candidate.

The shared-state form is the more conservative design hypothesis for eventual
SRHD/SRMHD work: the characteristic map and admissible-set projection can
change while the reconstruction-selection contract remains the same.
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import research_modal_reconstruction as modal  # noqa: E402
import research_wave_reconstruction as base  # noqa: E402
from exact_reference import cell_average_function  # noqa: E402


Array = np.ndarray
CellFaces = tuple[float, float]
Candidate = str

THINC_SMOOTH = "thinc_1p1"
THINC_SHARP = "thinc_1p6"
THINC_MHD_SMOOTH = "thinc_1p05"
THINC_MHD_SHARP = "thinc_1p2"
HIGH_ORDER = "high"

EXTRA_SHOCK_CASES = (
    base.ShockCase(
        "reverse_blast",
        (1.0, 0.0, 0.01),
        (1.0, 0.0, 100.0),
        0.012,
    ),
    base.ShockCase(
        "colliding_shocks",
        (5.99924, 19.5975, 460.894),
        (5.99242, -6.19633, 46.095),
        0.006,
    ),
)
ALL_SHOCK_CASES = (*base.SHOCK_CASES, *EXTRA_SHOCK_CASES)


def p4_cell_faces(values: Array, cell: int) -> CellFaces:
    """Return left/right P4 traces whose cell average is values[cell]."""
    left = (
        values[cell + 2] / 30.0
        - 13.0 * values[cell + 1] / 60.0
        + 47.0 * values[cell] / 60.0
        + 9.0 * values[cell - 1] / 20.0
        - values[cell - 2] / 20.0
    )
    right = (
        values[cell - 2] / 30.0
        - 13.0 * values[cell - 1] / 60.0
        + 47.0 * values[cell] / 60.0
        + 9.0 * values[cell + 1] / 20.0
        - values[cell + 2] / 20.0
    )
    return float(left), float(right)


def mp5_cell_faces(values: Array, cell: int) -> CellFaces:
    """Return MP5 traces on both faces of one cell."""
    right = base.mp5_left(
        values[cell - 2],
        values[cell - 1],
        values[cell],
        values[cell + 1],
        values[cell + 2],
        values[cell + 3],
    )
    left = base.mp5_left(
        values[cell + 2],
        values[cell + 1],
        values[cell],
        values[cell - 1],
        values[cell - 2],
        values[cell - 3],
    )
    return float(left), float(right)


def scale_invariant_mp5_left(
    vmm: float,
    vm: float,
    v0: float,
    vp: float,
    vpp: float,
    _vppp: float = 0.0,
) -> float:
    """MP5 with a homogeneous unlimited-candidate roundoff test."""
    alpha = 4.0
    epsilon = 1.0e-12
    unlimited = modal.unlimited_fifth_left(vmm, vm, v0, vp, vpp)
    monotone = v0 + modal.minmod(vp - v0, alpha * (v0 - vm))
    scale = max(abs(vmm), abs(vm), abs(v0), abs(vp), abs(vpp))
    accept_unlimited = (unlimited - v0) * (
        unlimited - monotone
    ) <= epsilon * scale * scale

    dm = vmm - 2.0 * vm + v0
    d0 = vm - 2.0 * v0 + vp
    dp = v0 - 2.0 * vp + vpp
    d_m4_left = modal.minmod4(
        4.0 * dm - d0,
        4.0 * d0 - dm,
        dm,
        d0,
    )
    d_m4_right = modal.minmod4(
        4.0 * d0 - dp,
        4.0 * dp - d0,
        d0,
        dp,
    )
    upper_limit = v0 + alpha * (v0 - vm)
    average = 0.5 * (v0 + vp)
    median = average - 0.5 * d_m4_right
    large_curvature = (
        v0 + 0.5 * (v0 - vm) + (4.0 / 3.0) * d_m4_left
    )
    lower_bound = max(
        min(v0, vp, median),
        min(v0, upper_limit, large_curvature),
    )
    upper_bound = min(
        max(v0, vp, median),
        max(v0, upper_limit, large_curvature),
    )
    limited = unlimited + modal.minmod(
        lower_bound - unlimited,
        upper_bound - unlimited,
    )
    return unlimited if accept_unlimited else limited


def scale_invariant_mp5_cell_faces(
    values: Array,
    cell: int,
) -> CellFaces:
    right = scale_invariant_mp5_left(
        values[cell - 2],
        values[cell - 1],
        values[cell],
        values[cell + 1],
        values[cell + 2],
        values[cell + 3],
    )
    left = scale_invariant_mp5_left(
        values[cell + 2],
        values[cell + 1],
        values[cell],
        values[cell - 1],
        values[cell - 2],
        values[cell - 3],
    )
    return float(left), float(right)


def thinc_cell_faces(
    values: Array,
    cell: int,
    beta: float,
) -> CellFaces:
    """Return conservative THINC traces for a monotone three-cell stencil."""
    previous = float(values[cell - 1])
    average = float(values[cell])
    following = float(values[cell + 1])
    if (following - average) * (average - previous) <= 0.0:
        return average, average

    minimum = min(previous, following)
    amplitude = max(previous, following) - minimum
    scale = max(
        np.finfo(float).tiny,
        abs(previous),
        abs(average),
        abs(following),
    )
    if amplitude <= 64.0 * np.finfo(float).eps * scale:
        return average, average

    normalized_average = min(
        1.0 - 8.0 * np.finfo(float).eps,
        max(
            8.0 * np.finfo(float).eps,
            (average - minimum) / amplitude,
        ),
    )
    orientation = 1.0 if following > previous else -1.0
    exponential = math.exp(
        orientation * beta * (2.0 * normalized_average - 1.0)
    )
    auxiliary = (exponential - math.cosh(beta)) / math.sinh(beta)
    tanh_beta = math.tanh(beta)
    denominator = 1.0 + auxiliary * tanh_beta
    if abs(denominator) <= 64.0 * np.finfo(float).eps:
        return average, average

    left = minimum + 0.5 * amplitude * (
        1.0 + orientation * auxiliary
    )
    right = minimum + 0.5 * amplitude * (
        1.0
        + orientation
        * (tanh_beta + auxiliary)
        / denominator
    )
    lower = minimum - 64.0 * np.finfo(float).eps * scale
    upper = minimum + amplitude + 64.0 * np.finfo(float).eps * scale
    if not (
        np.isfinite(left)
        and np.isfinite(right)
        and lower <= left <= upper
        and lower <= right <= upper
    ):
        return average, average
    return float(left), float(right)


def candidate_faces(
    values: Array,
    cell: int,
    candidate: Candidate,
    high_order: str,
) -> CellFaces:
    if candidate == HIGH_ORDER:
        if high_order == "p4":
            return p4_cell_faces(values, cell)
        if high_order == "mp5":
            return mp5_cell_faces(values, cell)
        if high_order == "mp5_si":
            return scale_invariant_mp5_cell_faces(values, cell)
        raise ValueError(f"unknown high-order candidate {high_order!r}")
    if candidate == THINC_SMOOTH:
        return thinc_cell_faces(values, cell, 1.1)
    if candidate == THINC_SHARP:
        return thinc_cell_faces(values, cell, 1.6)
    if candidate == THINC_MHD_SMOOTH:
        return thinc_cell_faces(values, cell, 1.05)
    if candidate == THINC_MHD_SHARP:
        return thinc_cell_faces(values, cell, 1.2)
    raise ValueError(f"unknown candidate {candidate!r}")


def boundary_variation(
    values: Array,
    cell: int,
    candidates: tuple[Candidate, Candidate, Candidate],
    high_order: str,
) -> float:
    previous = candidate_faces(
        values,
        cell - 1,
        candidates[0],
        high_order,
    )
    current = candidate_faces(
        values,
        cell,
        candidates[1],
        high_order,
    )
    following = candidate_faces(
        values,
        cell + 1,
        candidates[2],
        high_order,
    )
    return abs(previous[1] - current[0]) + abs(
        current[1] - following[0]
    )


def scalar_stage_one(
    values: Array,
    cell: int,
    high_order: str,
    thinc_smooth: Candidate = THINC_SMOOTH,
) -> Candidate:
    high_variation = boundary_variation(
        values,
        cell,
        (HIGH_ORDER,) * 3,
        high_order,
    )
    thinc_variation = boundary_variation(
        values,
        cell,
        (thinc_smooth,) * 3,
        high_order,
    )
    tolerance = 64.0 * np.finfo(float).eps * max(
        np.finfo(float).tiny,
        high_variation,
        thinc_variation,
    )
    if thinc_variation < high_variation - tolerance:
        return thinc_smooth
    return HIGH_ORDER


def scalar_candidate(
    values: Array,
    cell: int,
    high_order: str,
    allow_sharp: bool = True,
    thinc_candidates: tuple[Candidate, Candidate] = (
        THINC_SMOOTH,
        THINC_SHARP,
    ),
) -> Candidate:
    stage_one = tuple(
        scalar_stage_one(
            values,
            neighbor,
            high_order,
            thinc_smooth=thinc_candidates[0],
        )
        for neighbor in range(cell - 1, cell + 2)
    )
    if not allow_sharp:
        return stage_one[1]
    base_variation = boundary_variation(
        values,
        cell,
        stage_one,
        high_order,
    )
    sharp_variation = boundary_variation(
        values,
        cell,
        (stage_one[0], thinc_candidates[1], stage_one[2]),
        high_order,
    )
    tolerance = 64.0 * np.finfo(float).eps * max(
        np.finfo(float).tiny,
        base_variation,
        sharp_variation,
    )
    if sharp_variation < base_variation - tolerance:
        return thinc_candidates[1]
    return stage_one[1]


def wave_scale(values: Array, cell: int) -> float:
    local = values[cell - 3 : cell + 4]
    return max(
        float(np.max(local) - np.min(local)),
        1.0e-12
        * max(
            np.finfo(float).tiny,
            float(np.max(np.abs(local))),
        ),
    )


def aggregate_boundary_variation(
    waves: Array,
    cell: int,
    candidates: tuple[Candidate, Candidate, Candidate],
    high_order: str,
) -> float:
    return sum(
        boundary_variation(
            waves[:, wave],
            cell,
            candidates,
            high_order,
        )
        / wave_scale(waves[:, wave], cell)
        for wave in range(waves.shape[1])
    )


def shared_stage_one(
    waves: Array,
    cell: int,
    high_order: str,
    thinc_smooth: Candidate = THINC_SMOOTH,
) -> Candidate:
    high_variation = aggregate_boundary_variation(
        waves,
        cell,
        (HIGH_ORDER,) * 3,
        high_order,
    )
    thinc_variation = aggregate_boundary_variation(
        waves,
        cell,
        (thinc_smooth,) * 3,
        high_order,
    )
    tolerance = 64.0 * np.finfo(float).eps * max(
        np.finfo(float).tiny,
        high_variation,
        thinc_variation,
    )
    if thinc_variation < high_variation - tolerance:
        return thinc_smooth
    return HIGH_ORDER


def shared_candidate(
    waves: Array,
    cell: int,
    high_order: str,
    allow_sharp: bool = True,
    thinc_candidates: tuple[Candidate, Candidate] = (
        THINC_SMOOTH,
        THINC_SHARP,
    ),
) -> Candidate:
    stage_one = tuple(
        shared_stage_one(
            waves,
            neighbor,
            high_order,
            thinc_smooth=thinc_candidates[0],
        )
        for neighbor in range(cell - 1, cell + 2)
    )
    if not allow_sharp:
        return stage_one[1]
    base_variation = aggregate_boundary_variation(
        waves,
        cell,
        stage_one,
        high_order,
    )
    sharp_variation = aggregate_boundary_variation(
        waves,
        cell,
        (stage_one[0], thinc_candidates[1], stage_one[2]),
        high_order,
    )
    tolerance = 64.0 * np.finfo(float).eps * max(
        np.finfo(float).tiny,
        base_variation,
        sharp_variation,
    )
    if sharp_variation < base_variation - tolerance:
        return thinc_candidates[1]
    return stage_one[1]


def bvd_reconstruction(
    high_order: str,
    selection: str,
) -> base.FaceReconstruction:
    """Construct a characteristic local cascade-BVD face operator."""

    def reconstruct(
        primitive: Array,
        left_cell: int,
        gamma: float,
    ) -> tuple[Array, Array]:
        periodic = bool(reconstruct.periodic)
        stencil = np.array(
            [
                base.stencil_value(
                    primitive,
                    left_cell + offset,
                    periodic,
                )
                for offset in range(-7, 9)
            ]
        )
        center = 7
        rho_ref, sound = base.characteristic_basis(
            stencil[center],
            stencil[center + 1],
            gamma,
        )
        waves = base.to_characteristics(stencil, rho_ref, sound)
        left_waves = np.empty(3)
        right_waves = np.empty(3)

        if selection == "shared":
            left_candidate = shared_candidate(
                waves,
                center,
                high_order,
            )
            right_candidate = shared_candidate(
                waves,
                center + 1,
                high_order,
            )

        for wave in range(3):
            values = waves[:, wave]
            if selection == "wavewise":
                left_candidate = scalar_candidate(
                    values,
                    center,
                    high_order,
                )
                right_candidate = scalar_candidate(
                    values,
                    center + 1,
                    high_order,
                )
            elif selection != "shared":
                raise ValueError(f"unknown BVD selection {selection!r}")
            left_waves[wave] = candidate_faces(
                values,
                center,
                left_candidate,
                high_order,
            )[1]
            right_waves[wave] = candidate_faces(
                values,
                center + 1,
                right_candidate,
                high_order,
            )[0]

        left_trace = base.from_characteristics(
            left_waves,
            rho_ref,
            sound,
        )
        right_trace = base.from_characteristics(
            right_waves,
            rho_ref,
            sound,
        )
        common_weight = min(
            relative_positive_trace_weight(
                left_trace,
                stencil[center],
                (0, 2),
            ),
            relative_positive_trace_weight(
                right_trace,
                stencil[center + 1],
                (0, 2),
            ),
        )
        return (
            stencil[center]
            + common_weight * (left_trace - stencil[center]),
            stencil[center + 1]
            + common_weight
            * (right_trace - stencil[center + 1]),
        )

    reconstruct.periodic = False
    return reconstruct


def physical_subcell_hybrid(
    primitive: Array,
    left_cell: int,
    gamma: float,
) -> tuple[Array, Array]:
    """Use subcell BVD on the contact and balanced MP5 on acoustic waves."""
    periodic = bool(physical_subcell_hybrid.periodic)
    stencil = np.array(
        [
            base.stencil_value(
                primitive,
                left_cell + offset,
                periodic,
            )
            for offset in range(-7, 9)
        ]
    )
    center = 7
    rho_ref, sound = base.characteristic_basis(
        stencil[center],
        stencil[center + 1],
        gamma,
    )
    waves = base.to_characteristics(stencil, rho_ref, sound)
    left_waves = np.empty(3)
    right_waves = np.empty(3)

    contact_values = waves[:, 0]
    left_candidate = scalar_candidate(
        contact_values,
        center,
        "mp5",
    )
    right_candidate = scalar_candidate(
        contact_values,
        center + 1,
        "mp5",
    )
    left_waves[0] = candidate_faces(
        contact_values,
        center,
        left_candidate,
        "mp5",
    )[1]
    right_waves[0] = candidate_faces(
        contact_values,
        center + 1,
        right_candidate,
        "mp5",
    )[0]
    for wave in (1, 2):
        values = waves[center - 2 : center + 4, wave]
        (
            left_waves[wave],
            right_waves[wave],
            _,
            _,
        ) = base.hierarchical_scalar_pair(values, "balanced")

    left_trace = base.from_characteristics(
        left_waves,
        rho_ref,
        sound,
    )
    right_trace = base.from_characteristics(
        right_waves,
        rho_ref,
        sound,
    )
    weight = min(
        relative_positive_trace_weight(
            left_trace,
            stencil[center],
            (0, 2),
        ),
        relative_positive_trace_weight(
            right_trace,
            stencil[center + 1],
            (0, 2),
        ),
    )
    cell_jump = waves[center + 1] - waves[center]

    def physical_pair(
        candidate_weight: float,
    ) -> tuple[Array, Array]:
        return (
            stencil[center]
            + candidate_weight * (left_trace - stencil[center]),
            stencil[center + 1]
            + candidate_weight
            * (right_trace - stencil[center + 1]),
        )

    def pair_feasible(candidate_weight: float) -> bool:
        left, right = physical_pair(candidate_weight)
        if not base.primitive_is_admissible(np.array((left, right))):
            return False
        trace_waves = base.to_characteristics(
            np.array((left, right)),
            rho_ref,
            sound,
        )
        trace_jump = trace_waves[1] - trace_waves[0]
        for wave in (1, 2):
            lower = min(0.0, cell_jump[wave])
            upper = max(0.0, cell_jump[wave])
            tolerance = 128.0 * np.finfo(float).eps * max(
                1.0,
                abs(lower),
                abs(upper),
                abs(trace_jump[wave]),
            )
            if not (
                lower - tolerance
                <= trace_jump[wave]
                <= upper + tolerance
            ):
                return False
        return True

    if not pair_feasible(weight):
        lower_weight = 0.0
        upper_weight = weight
        for _ in range(52):
            middle_weight = 0.5 * (lower_weight + upper_weight)
            if pair_feasible(middle_weight):
                lower_weight = middle_weight
            else:
                upper_weight = middle_weight
        weight = lower_weight
    return physical_pair(weight)


physical_subcell_hybrid.periodic = False


def transported_contact_error(
    cells: int,
    cfl: float,
    gamma: float,
    reconstruct: base.FaceReconstruction,
) -> tuple[bool, float, float, float]:
    cell_edges = np.linspace(0.0, 1.0, cells + 1)
    density = cell_average_function(
        lambda points: np.where(
            (points >= 0.2) & (points < 0.6),
            2.0,
            1.0,
        ),
        cell_edges,
        breakpoints=(0.2, 0.6),
        quadrature_order=2,
    )
    primitive = np.column_stack(
        (density, np.full(cells, 0.5), np.ones(cells))
    )
    evolved, completed = base.evolve(
        primitive,
        final_time=1.0,
        cfl=cfl,
        gamma=gamma,
        reconstruct=reconstruct,
        riemann_flux=base.hllc_flux,
        periodic=True,
    )
    exact = cell_average_function(
        lambda points: np.where(
            (np.mod(points - 0.5, 1.0) >= 0.2)
            & (np.mod(points - 0.5, 1.0) < 0.6),
            2.0,
            1.0,
        ),
        cell_edges,
        breakpoints=(0.1, 0.7),
        quadrature_order=2,
    )
    l1 = float(np.mean(np.abs(evolved[:, 0] - exact)))
    overshoot = max(
        0.0,
        float(np.max(evolved[:, 0]) - 2.0),
        float(1.0 - np.min(evolved[:, 0])),
    )
    tv_ratio = base.total_variation(evolved[:, 0]) / 2.0
    return completed, l1, overshoot, tv_ratio


def thinc_conservation_error(samples: int, seed: int) -> float:
    rng = np.random.default_rng(seed)
    maximum_error = 0.0
    quadrature_x, quadrature_weight = np.polynomial.legendre.leggauss(32)
    unit_x = 0.5 * (quadrature_x + 1.0)
    unit_weight = 0.5 * quadrature_weight
    for _ in range(samples):
        endpoints = np.sort(rng.uniform(-10.0, 10.0, size=2))
        average = rng.uniform(endpoints[0], endpoints[1])
        if rng.random() < 0.5:
            values = np.array((endpoints[0], average, endpoints[1]))
        else:
            values = np.array((endpoints[1], average, endpoints[0]))
        for beta in (1.1, 1.6):
            left, right = thinc_cell_faces(values, 1, beta)
            amplitude = abs(values[2] - values[0])
            if amplitude == 0.0 or left == right:
                reconstructed_average = left
            else:
                minimum = min(values[0], values[2])
                orientation = 1.0 if values[2] > values[0] else -1.0
                left_fraction = (
                    2.0 * (left - minimum) / amplitude - 1.0
                ) / orientation
                center = -math.atanh(
                    max(
                        -1.0 + 4.0 * np.finfo(float).eps,
                        min(
                            1.0 - 4.0 * np.finfo(float).eps,
                            left_fraction,
                        ),
                    )
                ) / beta
                profile = minimum + 0.5 * amplitude * (
                    1.0
                    + orientation
                    * np.tanh(beta * (unit_x - center))
                )
                reconstructed_average = float(
                    np.dot(unit_weight, profile)
                )
            error = abs(reconstructed_average - average) / max(
                1.0,
                abs(average),
            )
            maximum_error = max(maximum_error, error)
    return maximum_error


def relative_positive_trace_weight(
    trace: Array,
    cell: Array,
    positive_components: tuple[int, ...],
) -> float:
    """Scale a trace using floors homogeneous in the adjacent state."""
    weight = 1.0
    for component in positive_components:
        floor = max(
            np.finfo(float).tiny,
            1.0e-13 * cell[component],
        )
        correction = trace[component] - cell[component]
        if trace[component] < floor and correction < 0.0:
            weight = min(
                weight,
                (cell[component] - floor) / (-correction),
            )
    return max(0.0, min(1.0, float(weight)))


def local_bounds_weight(
    trace: Array,
    cell: Array,
    neighborhood: Array,
) -> float:
    """Return one state-coherent scale enforcing local primitive bounds."""
    lower = np.min(neighborhood, axis=0)
    upper = np.max(neighborhood, axis=0)
    weight = 1.0
    for component in range(len(cell)):
        correction = trace[component] - cell[component]
        if correction > 0.0 and trace[component] > upper[component]:
            weight = min(
                weight,
                (upper[component] - cell[component]) / correction,
            )
        elif correction < 0.0 and trace[component] < lower[component]:
            weight = min(
                weight,
                (cell[component] - lower[component]) / (-correction),
            )
    return max(0.0, min(1.0, float(weight)))


def scalar_bvd_blend_weight(
    values: Array,
    cell: int,
    candidate: Candidate,
    high_order: str,
) -> float:
    """Map the relative BVD gain to a continuous high-to-subcell blend."""
    if candidate == HIGH_ORDER:
        return 0.0
    high_variation = boundary_variation(
        values,
        cell,
        (HIGH_ORDER,) * 3,
        high_order,
    )
    candidate_variation = boundary_variation(
        values,
        cell,
        (candidate,) * 3,
        high_order,
    )
    scale = max(high_variation, np.finfo(float).tiny)
    return max(
        0.0,
        min(1.0, (high_variation - candidate_variation) / scale),
    )


def shared_bvd_blend_weight(
    waves: Array,
    cell: int,
    candidate: Candidate,
    high_order: str,
) -> float:
    """Whole-state analogue of :func:`scalar_bvd_blend_weight`."""
    if candidate == HIGH_ORDER:
        return 0.0
    high_variation = aggregate_boundary_variation(
        waves,
        cell,
        (HIGH_ORDER,) * 3,
        high_order,
    )
    candidate_variation = aggregate_boundary_variation(
        waves,
        cell,
        (candidate,) * 3,
        high_order,
    )
    scale = max(high_variation, np.finfo(float).tiny)
    return max(
        0.0,
        min(1.0, (high_variation - candidate_variation) / scale),
    )


def random_admissibility_scan(
    methods: dict[str, base.FaceReconstruction],
    samples: int,
    gamma: float,
    seed: int,
) -> dict[str, tuple[int, float, float]]:
    rng = np.random.default_rng(seed)
    failures = {name: 0 for name in methods}
    minimum_rho = {name: math.inf for name in methods}
    minimum_pressure = {name: math.inf for name in methods}
    for _ in range(samples):
        stencil = np.column_stack(
            (
                np.exp(rng.uniform(math.log(1.0e-4), math.log(1.0e4), 24)),
                rng.uniform(-30.0, 30.0, 24),
                np.exp(rng.uniform(math.log(1.0e-5), math.log(1.0e5), 24)),
            )
        )
        for name, reconstruct in methods.items():
            reconstruct.periodic = False
            left, right = reconstruct(stencil, 11, gamma)
            minimum_rho[name] = min(
                minimum_rho[name],
                float(left[0]),
                float(right[0]),
            )
            minimum_pressure[name] = min(
                minimum_pressure[name],
                float(left[2]),
                float(right[2]),
            )
            if not base.primitive_is_admissible(
                np.array((left, right))
            ):
                failures[name] += 1
    return {
        name: (
            failures[name],
            minimum_rho[name],
            minimum_pressure[name],
        )
        for name in methods
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", nargs="+", type=int, default=(64, 128))
    parser.add_argument("--cases", nargs="+", default=("all",))
    parser.add_argument(
        "--methods",
        nargs="+",
        default=(
            "mp5_characteristic",
            "balanced_metric_mp5",
            "physical_subcell_hybrid",
            "mp5_thinc_wavewise",
            "mp5_thinc_shared",
        ),
    )
    parser.add_argument("--cfl", type=float, default=0.4)
    parser.add_argument("--gamma", type=float, default=1.4)
    parser.add_argument("--random-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    methods: dict[str, base.FaceReconstruction] = {
        "mp5_characteristic": base.characteristic_mp5,
        "balanced_metric_mp5": base.balanced_metric_nonlinear_mp5,
        "physical_subcell_hybrid": physical_subcell_hybrid,
        "p4_thinc_wavewise": bvd_reconstruction("p4", "wavewise"),
        "p4_thinc_shared": bvd_reconstruction("p4", "shared"),
        "mp5_thinc_wavewise": bvd_reconstruction("mp5", "wavewise"),
        "mp5_thinc_shared": bvd_reconstruction("mp5", "shared"),
        "mp5_si_thinc_wavewise": bvd_reconstruction(
            "mp5_si",
            "wavewise",
        ),
        "mp5_si_thinc_shared": bvd_reconstruction(
            "mp5_si",
            "shared",
        ),
    }
    if args.methods != ["all"] and args.methods != ("all",):
        unknown_methods = sorted(set(args.methods) - set(methods))
        if unknown_methods:
            parser.error(
                f"unknown methods: {', '.join(unknown_methods)}"
            )
        methods = {name: methods[name] for name in args.methods}
    cases = {
        "entropy_wave",
        "contact",
        *(case.name for case in ALL_SHOCK_CASES),
    }
    requested_cases = cases if args.cases == ["all"] else set(args.cases)
    unknown = sorted(requested_cases - cases)
    if unknown:
        parser.error(f"unknown cases: {', '.join(unknown)}")

    results: list[base.Result] = []
    contact_rows: list[tuple[str, int, bool, float, float, float]] = []
    for cells in args.cells:
        for name, reconstruct in methods.items():
            if "entropy_wave" in requested_cases:
                results.append(
                    base.entropy_wave_result(
                        name,
                        cells,
                        args.cfl,
                        args.gamma,
                        reconstruct,
                        base.hllc_flux,
                    )
                )
            if "contact" in requested_cases:
                contact_rows.append(
                    (
                        name,
                        cells,
                        *transported_contact_error(
                            cells,
                            args.cfl,
                            args.gamma,
                            reconstruct,
                        ),
                    )
                )
            for case in ALL_SHOCK_CASES:
                if case.name not in requested_cases:
                    continue
                x = (np.arange(cells) + 0.5) / cells
                primitive = np.empty((cells, 3))
                primitive[x < 0.5] = case.left
                primitive[x >= 0.5] = case.right
                evolved, completed = base.evolve(
                    primitive,
                    case.final_time,
                    args.cfl,
                    args.gamma,
                    reconstruct,
                    base.hllc_flux,
                    periodic=False,
                )
                results.append(
                    base.shock_result(
                        case,
                        name,
                        cells,
                        evolved,
                        completed,
                        args.gamma,
                    )
                )

    base.print_results(results)
    print(
        "contact_method\tcells\tcompleted\tl1\tovershoot\ttv_ratio"
    )
    for row in contact_rows:
        print(
            f"{row[0]}\t{row[1]}\t{row[2]}\t"
            f"{row[3]:.12e}\t{row[4]:.12e}\t{row[5]:.12e}"
        )
    print(
        "thinc_conservation_max_error\t"
        f"{thinc_conservation_error(1000, args.seed):.12e}"
    )
    for name, values in random_admissibility_scan(
        methods,
        args.random_samples,
        args.gamma,
        args.seed,
    ).items():
        print(
            f"admissibility\t{name}\t{values[0]}\t"
            f"{values[1]:.12e}\t{values[2]:.12e}"
        )

    if args.output is not None:
        base.write_results(args.output, results)
        contact_path = args.output.with_name(
            f"{args.output.stem}_contacts.tsv"
        )
        contact_path.parent.mkdir(parents=True, exist_ok=True)
        with contact_path.open("w") as stream:
            stream.write(
                "method\tcells\tcompleted\tl1\tovershoot\ttv_ratio\n"
            )
            for row in contact_rows:
                stream.write(
                    f"{row[0]}\t{row[1]}\t{row[2]}\t"
                    f"{row[3]:.17e}\t{row[4]:.17e}\t"
                    f"{row[5]:.17e}\n"
                )


if __name__ == "__main__":
    main()
