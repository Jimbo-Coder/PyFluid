#!/usr/bin/env python3
"""Carry the local MP5-THINC BVD precursor into Newtonian ideal MHD.

The normal magnetic field remains one face value and is never reconstructed
independently on the two sides.  The seven reduced characteristic fields use
either independent candidate choices (``wavewise``) or one shared whole-state
choice (``shared``).  This is a one-dimensional research harness, not a CT,
multidimensional, relativistic, or production implementation.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import research_mhd_wave_reconstruction as mhd  # noqa: E402
import research_subcell_reconstruction as subcell  # noqa: E402
from research_constrained_riemann import mhd_hlld_flux  # noqa: E402
from research_wave_reconstruction import stencil_value  # noqa: E402


Array = np.ndarray


def dai_woodward_initial(cells: int) -> Array:
    """Return the standard seven-wave Dai-Woodward shock tube."""
    x = (np.arange(cells) + 0.5) / cells
    primitive = np.empty((cells, 8))
    left = x < 0.5
    magnetic_scale = math.sqrt(4.0 * math.pi)
    primitive[:, 0] = np.where(left, 1.08, 1.0)
    primitive[:, 1] = np.where(left, 1.2, 0.0)
    primitive[:, 2] = np.where(left, 0.01, 0.0)
    primitive[:, 3] = np.where(left, 0.5, 0.0)
    primitive[:, 4] = np.where(left, 0.95, 1.0)
    primitive[:, 5] = 2.0 / magnetic_scale
    primitive[:, 6] = np.where(left, 3.6, 4.0) / magnetic_scale
    primitive[:, 7] = 2.0 / magnetic_scale
    return primitive


def reference_results(
    case_name: str,
    initial: Callable[[int], Array],
    gamma: float,
    final_time: float,
    methods: dict[str, mhd.FaceReconstruction],
    cells: int,
    cfl: float,
    reference_factor: int,
) -> list[mhd.Result]:
    reference, reference_completed = mhd.evolve(
        initial(cells * reference_factor),
        final_time,
        cfl,
        gamma,
        mhd.characteristic_mp5,
        mhd_hlld_flux,
        periodic=False,
    )
    if not reference_completed:
        raise RuntimeError(f"fine {case_name} reference failed")
    exact = mhd.coarsen_conserved(reference, cells, gamma)
    results = []
    for name, reconstruct in methods.items():
        evolved, completed = mhd.evolve(
            initial(cells),
            final_time,
            cfl,
            gamma,
            reconstruct,
            mhd_hlld_flux,
            periodic=False,
        )
        results.append(
            mhd.result_from_exact(
                f"{case_name}_reference",
                name,
                evolved,
                exact,
                completed,
                gamma,
            )
        )
    return results


def bvd_mhd_reconstruction(
    selection: str,
    high_order: str = "mp5",
    thinc_waves: frozenset[int] | None = None,
    allow_sharp: bool = True,
    local_bounds: bool = False,
    continuous: bool = False,
    reject_unbounded: bool = False,
    thinc_candidates: tuple[str, str] = (
        subcell.THINC_SMOOTH,
        subcell.THINC_SHARP,
    ),
) -> mhd.FaceReconstruction:
    """Construct a high-order/THINC characteristic MHD face operator."""

    def reconstruct(
        primitive: Array,
        left_cell: int,
        gamma: float,
    ) -> tuple[Array, Array]:
        periodic = bool(reconstruct.periodic)
        stencil = np.array(
            [
                stencil_value(
                    primitive,
                    left_cell + offset,
                    periodic,
                )
                for offset in range(-7, 9)
            ]
        )
        center = 7
        normal_b = 0.5 * (
            stencil[center, 5] + stencil[center + 1, 5]
        )
        basis = mhd.characteristic_basis(
            stencil[center],
            stencil[center + 1],
            gamma,
            normal_b,
        )
        reduced = stencil[:, mhd.REDUCED]
        waves = mhd.to_characteristics(reduced, basis)
        left_waves = np.empty(7)
        right_waves = np.empty(7)
        used_subcell = False

        active_waves = (
            frozenset(range(7)) if thinc_waves is None else thinc_waves
        )
        if selection == "shared":
            active_indices = sorted(active_waves)
            active_values = waves[:, active_indices]
            left_candidate = subcell.shared_candidate(
                active_values,
                center,
                high_order,
                allow_sharp=allow_sharp,
                thinc_candidates=thinc_candidates,
            )
            right_candidate = subcell.shared_candidate(
                active_values,
                center + 1,
                high_order,
                allow_sharp=allow_sharp,
                thinc_candidates=thinc_candidates,
            )
            left_blend = subcell.shared_bvd_blend_weight(
                active_values,
                center,
                left_candidate,
                high_order,
            )
            right_blend = subcell.shared_bvd_blend_weight(
                active_values,
                center + 1,
                right_candidate,
                high_order,
            )

        for wave in range(7):
            values = waves[:, wave]
            if wave not in active_waves:
                left_candidate = subcell.HIGH_ORDER
                right_candidate = subcell.HIGH_ORDER
            elif selection == "wavewise":
                left_candidate = subcell.scalar_candidate(
                    values,
                    center,
                    high_order,
                    allow_sharp=allow_sharp,
                    thinc_candidates=thinc_candidates,
                )
                right_candidate = subcell.scalar_candidate(
                    values,
                    center + 1,
                    high_order,
                    allow_sharp=allow_sharp,
                    thinc_candidates=thinc_candidates,
                )
            elif selection != "shared":
                raise ValueError(f"unknown BVD selection {selection!r}")
            if selection == "wavewise":
                left_blend = subcell.scalar_bvd_blend_weight(
                    values,
                    center,
                    left_candidate,
                    high_order,
                )
                right_blend = subcell.scalar_bvd_blend_weight(
                    values,
                    center + 1,
                    right_candidate,
                    high_order,
                )
            used_subcell = used_subcell or left_blend > 0.0 or right_blend > 0.0
            left_subcell = subcell.candidate_faces(
                values,
                center,
                left_candidate,
                high_order,
            )[1]
            right_subcell = subcell.candidate_faces(
                values,
                center + 1,
                right_candidate,
                high_order,
            )[0]
            if continuous:
                left_high = subcell.candidate_faces(
                    values,
                    center,
                    subcell.HIGH_ORDER,
                    high_order,
                )[1]
                right_high = subcell.candidate_faces(
                    values,
                    center + 1,
                    subcell.HIGH_ORDER,
                    high_order,
                )[0]
                left_waves[wave] = left_high + left_blend * (
                    left_subcell - left_high
                )
                right_waves[wave] = right_high + right_blend * (
                    right_subcell - right_high
                )
            else:
                left_waves[wave] = left_subcell
                right_waves[wave] = right_subcell

        left_reduced = mhd.from_characteristics(left_waves, basis)
        right_reduced = mhd.from_characteristics(right_waves, basis)
        if reject_unbounded and used_subcell:
            left_bound_weight = subcell.local_bounds_weight(
                left_reduced,
                reduced[center],
                reduced[center - 1 : center + 2],
            )
            right_bound_weight = subcell.local_bounds_weight(
                right_reduced,
                reduced[center + 1],
                reduced[center : center + 3],
            )
            if min(left_bound_weight, right_bound_weight) < (
                1.0 - 128.0 * np.finfo(float).eps
            ):
                for wave in range(7):
                    values = waves[:, wave]
                    left_waves[wave] = subcell.candidate_faces(
                        values,
                        center,
                        subcell.HIGH_ORDER,
                        high_order,
                    )[1]
                    right_waves[wave] = subcell.candidate_faces(
                        values,
                        center + 1,
                        subcell.HIGH_ORDER,
                        high_order,
                    )[0]
                left_reduced = mhd.from_characteristics(
                    left_waves,
                    basis,
                )
                right_reduced = mhd.from_characteristics(
                    right_waves,
                    basis,
                )
                used_subcell = False
        common_weight = min(
            subcell.relative_positive_trace_weight(
                left_reduced,
                reduced[center],
                (0, 4),
            ),
            subcell.relative_positive_trace_weight(
                right_reduced,
                reduced[center + 1],
                (0, 4),
            ),
        )
        if local_bounds and used_subcell:
            common_weight = min(
                common_weight,
                subcell.local_bounds_weight(
                    left_reduced,
                    reduced[center],
                    reduced[center - 1 : center + 2],
                ),
                subcell.local_bounds_weight(
                    right_reduced,
                    reduced[center + 1],
                    reduced[center : center + 3],
                ),
            )
        left_reduced = reduced[center] + common_weight * (
            left_reduced - reduced[center]
        )
        right_reduced = reduced[center + 1] + common_weight * (
            right_reduced - reduced[center + 1]
        )
        return (
            np.insert(left_reduced, 5, normal_b),
            np.insert(right_reduced, 5, normal_b),
        )

    reconstruct.periodic = False
    return reconstruct


def _physical_subcell_mhd_hybrid(
    primitive: Array,
    left_cell: int,
    gamma: float,
    thinc_candidates: tuple[str, str],
    periodic: bool,
    purity_power: float | None = None,
    adaptive_beta_power: float | None = None,
) -> tuple[Array, Array]:
    """Use THINC on linearly-degenerate and balanced MP5 on MHD waves."""
    stencil = np.array(
        [
            stencil_value(
                primitive,
                left_cell + offset,
                periodic,
            )
            for offset in range(-7, 9)
        ]
    )
    center = 7
    normal_b = 0.5 * (
        stencil[center, 5] + stencil[center + 1, 5]
    )
    basis = mhd.characteristic_basis(
        stencil[center],
        stencil[center + 1],
        gamma,
        normal_b,
    )
    reduced = stencil[:, mhd.REDUCED]
    waves = mhd.to_characteristics(reduced, basis)
    left_waves = np.empty(7)
    right_waves = np.empty(7)
    for wave in range(7):
        values = waves[:, wave]
        if wave in mhd.NONLINEAR_WAVES:
            (
                left_waves[wave],
                right_waves[wave],
                _,
                _,
            ) = mhd.hierarchical_scalar_pair(
                values[center - 2 : center + 4],
                "balanced",
            )
        else:
            left_candidate = subcell.scalar_candidate(
                values,
                center,
                "mp5",
                thinc_candidates=thinc_candidates,
            )
            right_candidate = subcell.scalar_candidate(
                values,
                center + 1,
                "mp5",
                thinc_candidates=thinc_candidates,
            )
            left_subcell = subcell.candidate_faces(
                values,
                center,
                left_candidate,
                "mp5",
            )[1]
            right_subcell = subcell.candidate_faces(
                values,
                center + 1,
                right_candidate,
                "mp5",
            )[0]
            def wave_activity(
                target_wave: int,
                cell: int,
            ) -> float:
                local = waves[cell - 2 : cell + 3, target_wave]
                normalization = basis.rho if target_wave == 3 else 1.0
                return float(
                    np.sum(np.abs(np.diff(local))) / normalization
                )

            left_linear_activity = wave_activity(wave, center)
            right_linear_activity = wave_activity(
                wave,
                center + 1,
            )
            left_nonlinear_activity = sum(
                wave_activity(nonlinear_wave, center)
                for nonlinear_wave in mhd.NONLINEAR_WAVES
            )
            right_nonlinear_activity = sum(
                wave_activity(nonlinear_wave, center + 1)
                for nonlinear_wave in mhd.NONLINEAR_WAVES
            )
            tiny = np.finfo(float).tiny
            left_purity = left_linear_activity / max(
                left_linear_activity + left_nonlinear_activity,
                tiny,
            )
            right_purity = right_linear_activity / max(
                right_linear_activity + right_nonlinear_activity,
                tiny,
            )
            if adaptive_beta_power is not None:
                left_soft_candidate = subcell.scalar_candidate(
                    values,
                    center,
                    "mp5",
                    thinc_candidates=(
                        subcell.THINC_MHD_SMOOTH,
                        subcell.THINC_MHD_SHARP,
                    ),
                )
                right_soft_candidate = subcell.scalar_candidate(
                    values,
                    center + 1,
                    "mp5",
                    thinc_candidates=(
                        subcell.THINC_MHD_SMOOTH,
                        subcell.THINC_MHD_SHARP,
                    ),
                )
                left_soft = subcell.candidate_faces(
                    values,
                    center,
                    left_soft_candidate,
                    "mp5",
                )[1]
                right_soft = subcell.candidate_faces(
                    values,
                    center + 1,
                    right_soft_candidate,
                    "mp5",
                )[0]
                left_subcell = left_soft + (
                    left_purity**adaptive_beta_power
                    * (left_subcell - left_soft)
                )
                right_subcell = right_soft + (
                    right_purity**adaptive_beta_power
                    * (right_subcell - right_soft)
                )

            if purity_power is None:
                left_waves[wave] = left_subcell
                right_waves[wave] = right_subcell
            else:
                left_high = subcell.candidate_faces(
                    values,
                    center,
                    subcell.HIGH_ORDER,
                    "mp5",
                )[1]
                right_high = subcell.candidate_faces(
                    values,
                    center + 1,
                    subcell.HIGH_ORDER,
                    "mp5",
                )[0]
                left_waves[wave] = left_high + left_purity**purity_power * (
                    left_subcell - left_high
                )
                right_waves[wave] = (
                    right_high
                    + right_purity**purity_power
                    * (right_subcell - right_high)
                )

    left_reduced = mhd.from_characteristics(left_waves, basis)
    right_reduced = mhd.from_characteristics(right_waves, basis)
    weight = min(
        subcell.relative_positive_trace_weight(
            left_reduced,
            reduced[center],
            (0, 4),
        ),
        subcell.relative_positive_trace_weight(
            right_reduced,
            reduced[center + 1],
            (0, 4),
        ),
    )
    cell_jump = waves[center + 1] - waves[center]

    def physical_pair(
        candidate_weight: float,
    ) -> tuple[Array, Array]:
        return (
            reduced[center]
            + candidate_weight * (left_reduced - reduced[center]),
            reduced[center + 1]
            + candidate_weight
            * (right_reduced - reduced[center + 1]),
        )

    def pair_feasible(candidate_weight: float) -> bool:
        left, right = physical_pair(candidate_weight)
        if not mhd.primitive_is_admissible(np.array((left, right))):
            return False
        trace_waves = mhd.to_characteristics(
            np.array((left, right)),
            basis,
        )
        trace_jump = trace_waves[1] - trace_waves[0]
        for wave in mhd.NONLINEAR_WAVES:
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
    left_reduced, right_reduced = physical_pair(weight)
    return (
        np.insert(left_reduced, 5, normal_b),
        np.insert(right_reduced, 5, normal_b),
    )


def physical_subcell_mhd_hybrid(
    primitive: Array,
    left_cell: int,
    gamma: float,
) -> tuple[Array, Array]:
    return _physical_subcell_mhd_hybrid(
        primitive,
        left_cell,
        gamma,
        (
            subcell.THINC_SMOOTH,
            subcell.THINC_SHARP,
        ),
        bool(physical_subcell_mhd_hybrid.periodic),
        None,
    )


physical_subcell_mhd_hybrid.periodic = False


def physical_subcell_mhd_soft_hybrid(
    primitive: Array,
    left_cell: int,
    gamma: float,
) -> tuple[Array, Array]:
    return _physical_subcell_mhd_hybrid(
        primitive,
        left_cell,
        gamma,
        (
            subcell.THINC_MHD_SMOOTH,
            subcell.THINC_MHD_SHARP,
        ),
        bool(physical_subcell_mhd_soft_hybrid.periodic),
        None,
    )


physical_subcell_mhd_soft_hybrid.periodic = False


def physical_subcell_mhd_purity_hybrid(
    primitive: Array,
    left_cell: int,
    gamma: float,
) -> tuple[Array, Array]:
    return _physical_subcell_mhd_hybrid(
        primitive,
        left_cell,
        gamma,
        (
            subcell.THINC_SMOOTH,
            subcell.THINC_SHARP,
        ),
        bool(physical_subcell_mhd_purity_hybrid.periodic),
        2.0,
    )


physical_subcell_mhd_purity_hybrid.periodic = False


def physical_subcell_mhd_purity1_hybrid(
    primitive: Array,
    left_cell: int,
    gamma: float,
) -> tuple[Array, Array]:
    return _physical_subcell_mhd_hybrid(
        primitive,
        left_cell,
        gamma,
        (
            subcell.THINC_SMOOTH,
            subcell.THINC_SHARP,
        ),
        bool(physical_subcell_mhd_purity1_hybrid.periodic),
        1.0,
    )


physical_subcell_mhd_purity1_hybrid.periodic = False


def physical_subcell_mhd_purity_half_hybrid(
    primitive: Array,
    left_cell: int,
    gamma: float,
) -> tuple[Array, Array]:
    return _physical_subcell_mhd_hybrid(
        primitive,
        left_cell,
        gamma,
        (
            subcell.THINC_SMOOTH,
            subcell.THINC_SHARP,
        ),
        bool(physical_subcell_mhd_purity_half_hybrid.periodic),
        0.5,
    )


physical_subcell_mhd_purity_half_hybrid.periodic = False


def physical_subcell_mhd_adaptive_hybrid(
    primitive: Array,
    left_cell: int,
    gamma: float,
) -> tuple[Array, Array]:
    return _physical_subcell_mhd_hybrid(
        primitive,
        left_cell,
        gamma,
        (
            subcell.THINC_SMOOTH,
            subcell.THINC_SHARP,
        ),
        bool(physical_subcell_mhd_adaptive_hybrid.periodic),
        None,
        0.5,
    )


physical_subcell_mhd_adaptive_hybrid.periodic = False


def random_admissibility_scan(
    methods: dict[str, mhd.FaceReconstruction],
    samples: int,
    gamma: float,
    seed: int,
) -> dict[str, tuple[int, float, float, float]]:
    rng = np.random.default_rng(seed)
    failures = {name: 0 for name in methods}
    minimum_rho = {name: math.inf for name in methods}
    minimum_pressure = {name: math.inf for name in methods}
    maximum_normal_b_jump = {name: 0.0 for name in methods}
    for _ in range(samples):
        stencil = np.empty((24, 8))
        stencil[:, 0] = np.exp(
            rng.uniform(math.log(1.0e-3), math.log(1.0e3), 24)
        )
        stencil[:, 1:4] = rng.uniform(-10.0, 10.0, (24, 3))
        stencil[:, 4] = np.exp(
            rng.uniform(math.log(1.0e-4), math.log(1.0e4), 24)
        )
        stencil[:, 5:8] = rng.uniform(-10.0, 10.0, (24, 3))
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
                float(left[4]),
                float(right[4]),
            )
            maximum_normal_b_jump[name] = max(
                maximum_normal_b_jump[name],
                abs(float(left[5] - right[5])),
            )
            if not mhd.primitive_is_admissible(
                np.array((left, right))
            ):
                failures[name] += 1
    return {
        name: (
            failures[name],
            minimum_rho[name],
            minimum_pressure[name],
            maximum_normal_b_jump[name],
        )
        for name in methods
    }


def scaling_invariance_scan(
    methods: dict[str, mhd.FaceReconstruction],
    samples: int,
    gamma: float,
    seed: int,
) -> dict[str, float]:
    """Check MHD homogeneity under rho,p,B rescaling."""
    rng = np.random.default_rng(seed)
    maximum_error = {name: 0.0 for name in methods}
    for _ in range(samples):
        stencil = np.empty((24, 8))
        stencil[:, 0] = np.exp(rng.uniform(-2.0, 2.0, 24))
        stencil[:, 1:4] = rng.uniform(-2.0, 2.0, (24, 3))
        stencil[:, 4] = np.exp(rng.uniform(-2.0, 2.0, 24))
        stencil[:, 5:8] = rng.uniform(-2.0, 2.0, (24, 3))
        for name, reconstruct in methods.items():
            reconstruct.periodic = False
            reference = reconstruct(stencil, 11, gamma)
            for factor in (1.0e-8, 1.0e8):
                component_scale = np.array(
                    (
                        factor,
                        1.0,
                        1.0,
                        1.0,
                        factor,
                        math.sqrt(factor),
                        math.sqrt(factor),
                        math.sqrt(factor),
                    )
                )
                scaled_stencil = stencil * component_scale
                scaled = reconstruct(scaled_stencil, 11, gamma)
                expected = np.array(reference) * component_scale
                denominator = np.maximum(
                    np.max(np.abs(expected), axis=0),
                    1.0e-12 * component_scale,
                )
                error = float(
                    np.max(np.abs(np.array(scaled) - expected) / denominator)
                )
                maximum_error[name] = max(maximum_error[name], error)
    return maximum_error


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", type=int, nargs="+", default=(32,))
    parser.add_argument(
        "--cases",
        nargs="+",
        default=("circular_alfven", "brio_wu"),
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=(
            "mp5_characteristic",
            "balanced_metric_mp5",
            "physical_subcell_hybrid",
            "physical_subcell_soft_hybrid",
            "physical_subcell_purity_half_hybrid",
            "physical_subcell_adaptive_hybrid",
            "mp5_thinc_wavewise",
        ),
    )
    parser.add_argument("--cfl", type=float, default=0.4)
    parser.add_argument("--reference-factor", type=int, default=4)
    parser.add_argument("--random-samples", type=int, default=2000)
    parser.add_argument("--scale-samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    methods: dict[str, mhd.FaceReconstruction] = {
        "mp5_characteristic": mhd.characteristic_mp5,
        "balanced_metric_mp5": mhd.balanced_metric_nonlinear_mp5,
        "physical_subcell_hybrid": physical_subcell_mhd_hybrid,
        "physical_subcell_soft_hybrid": (
            physical_subcell_mhd_soft_hybrid
        ),
        "physical_subcell_purity_hybrid": (
            physical_subcell_mhd_purity_hybrid
        ),
        "physical_subcell_purity1_hybrid": (
            physical_subcell_mhd_purity1_hybrid
        ),
        "physical_subcell_purity_half_hybrid": (
            physical_subcell_mhd_purity_half_hybrid
        ),
        "physical_subcell_adaptive_hybrid": (
            physical_subcell_mhd_adaptive_hybrid
        ),
        "mp5_thinc_wavewise": bvd_mhd_reconstruction("wavewise"),
        "mp5_thinc_shared": bvd_mhd_reconstruction("shared"),
        "mp5_si_thinc_wavewise": bvd_mhd_reconstruction(
            "wavewise",
            high_order="mp5_si",
        ),
        "mp5_si_thinc_shared": bvd_mhd_reconstruction(
            "shared",
            high_order="mp5_si",
        ),
        "mp5_thinc_bounded_wavewise": bvd_mhd_reconstruction(
            "wavewise",
            local_bounds=True,
        ),
        "mp5_thinc_bounded_shared": bvd_mhd_reconstruction(
            "shared",
            local_bounds=True,
        ),
        "mp5_thinc_continuous_wavewise": bvd_mhd_reconstruction(
            "wavewise",
            continuous=True,
        ),
        "mp5_thinc_continuous_shared": bvd_mhd_reconstruction(
            "shared",
            continuous=True,
        ),
        "mp5_thinc_admissible_wavewise": bvd_mhd_reconstruction(
            "wavewise",
            reject_unbounded=True,
        ),
        "mp5_thinc_admissible_shared": bvd_mhd_reconstruction(
            "shared",
            reject_unbounded=True,
        ),
        "mp5_thinc_mhd_wavewise": bvd_mhd_reconstruction(
            "wavewise",
            thinc_candidates=(
                subcell.THINC_MHD_SMOOTH,
                subcell.THINC_MHD_SHARP,
            ),
        ),
        "mp5_thinc_mhd_shared": bvd_mhd_reconstruction(
            "shared",
            thinc_candidates=(
                subcell.THINC_MHD_SMOOTH,
                subcell.THINC_MHD_SHARP,
            ),
        ),
        "mp5_thinc_soft_wavewise": bvd_mhd_reconstruction(
            "wavewise",
            allow_sharp=False,
        ),
        "mp5_thinc_soft_shared": bvd_mhd_reconstruction(
            "shared",
            allow_sharp=False,
        ),
        "mp5_thinc_ld_wavewise": bvd_mhd_reconstruction(
            "wavewise",
            thinc_waves=frozenset((1, 3, 5)),
        ),
        "mp5_thinc_ld_shared": bvd_mhd_reconstruction(
            "shared",
            thinc_waves=frozenset((1, 3, 5)),
        ),
    }
    if args.methods != ["all"] and args.methods != ("all",):
        unknown_methods = sorted(set(args.methods) - set(methods))
        if unknown_methods:
            parser.error(
                f"unknown methods: {', '.join(unknown_methods)}"
            )
        methods = {name: methods[name] for name in args.methods}
    unknown_cases = sorted(
        set(args.cases) - {"circular_alfven", "brio_wu", "dai_woodward"}
    )
    if unknown_cases:
        parser.error(f"unknown cases: {', '.join(unknown_cases)}")

    results: list[mhd.Result] = []
    for cells in args.cells:
        if "circular_alfven" in args.cases:
            for name, reconstruct in methods.items():
                results.append(
                    mhd.circular_alfven_result(
                        name,
                        cells,
                        args.cfl,
                        reconstruct,
                        mhd_hlld_flux,
                    )
                )
        if "brio_wu" in args.cases:
            results.extend(
                mhd.brio_wu_results(
                    methods,
                    {"hlld": mhd_hlld_flux},
                    cells,
                    args.cfl,
                    args.reference_factor,
                )
            )
        if "dai_woodward" in args.cases:
            results.extend(
                reference_results(
                    "dai_woodward",
                    dai_woodward_initial,
                    5.0 / 3.0,
                    0.2,
                    methods,
                    cells,
                    args.cfl,
                    args.reference_factor,
                )
            )
    mhd.print_results(results)
    for name, values in random_admissibility_scan(
        methods,
        args.random_samples,
        5.0 / 3.0,
        args.seed,
    ).items():
        print(
            f"admissibility\t{name}\t{values[0]}\t"
            f"{values[1]:.12e}\t{values[2]:.12e}\t"
            f"{values[3]:.12e}"
        )
    if args.scale_samples > 0:
        for name, error in scaling_invariance_scan(
            methods,
            args.scale_samples,
            5.0 / 3.0,
            args.seed + 1,
        ).items():
            print(f"scaling_invariance\t{name}\t{error:.12e}")
    if args.output is not None:
        mhd.write_results(args.output, results)
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
