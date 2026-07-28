#!/usr/bin/env python3
"""Stress final paired-trace invariants for research reconstructions.

This is a broad log-uniform falsification harness, not a production gate.  It
checks the traces returned after whole-state positivity scaling, because
checking only the pre-scaling wave coefficients can miss a broken final
left/right jump contract.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import research_mhd_wave_reconstruction as mhd  # noqa: E402
import research_wave_reconstruction as hydro  # noqa: E402


Array = np.ndarray


@dataclasses.dataclass(frozen=True)
class Summary:
    physics: str
    objective: str
    samples: int
    nonfinite_or_nonpositive: int
    jump_violations: int
    maximum_normalized_jump_violation: float


def normalized_interval_violation(value: float, jump: float) -> float:
    lower = min(0.0, jump)
    upper = max(0.0, jump)
    return max(0.0, lower - value, value - upper) / max(1.0, abs(jump))


def random_hydro_stencil(rng: np.random.Generator) -> Array:
    stencil = np.empty((6, 3))
    stencil[:, 0] = 10.0 ** rng.uniform(-6.0, 6.0, 6)
    stencil[:, 1] = rng.normal(0.0, 20.0, 6)
    stencil[:, 2] = 10.0 ** rng.uniform(-8.0, 8.0, 6)
    return stencil


def random_mhd_stencil(rng: np.random.Generator) -> Array:
    stencil = np.empty((6, 8))
    stencil[:, 0] = 10.0 ** rng.uniform(-6.0, 6.0, 6)
    stencil[:, 1:4] = rng.normal(0.0, 10.0, (6, 3))
    stencil[:, 4] = 10.0 ** rng.uniform(-8.0, 8.0, 6)
    normal_b = np.copysign(
        10.0 ** rng.uniform(-5.0, 4.0),
        rng.uniform(-1.0, 1.0),
    )
    stencil[:, 5] = normal_b
    stencil[:, 6:8] = (
        rng.normal(0.0, 1.0, (6, 2))
        * 10.0 ** rng.uniform(-5.0, 4.0, (6, 1))
    )
    return stencil


def check_hydro(
    rng: np.random.Generator,
    samples: int,
    objective: str,
) -> Summary:
    bad_states = 0
    violations = 0
    maximum_violation = 0.0
    for _ in range(samples):
        stencil = random_hydro_stencil(rng)
        left, right = hydro.hierarchical_wave_reconstruction(
            stencil,
            2,
            1.4,
            False,
            objective,
            False,
        )
        rho_ref, sound = hydro.characteristic_basis(
            stencil[2],
            stencil[3],
            1.4,
        )
        cell_waves = hydro.to_characteristics(
            stencil[[2, 3]],
            rho_ref,
            sound,
        )
        trace_waves = hydro.to_characteristics(
            np.array((left, right)),
            rho_ref,
            sound,
        )
        cell_jump = cell_waves[1] - cell_waves[0]
        trace_jump = trace_waves[1] - trace_waves[0]
        for wave in (1, 2):
            violation = normalized_interval_violation(
                trace_jump[wave],
                cell_jump[wave],
            )
            maximum_violation = max(maximum_violation, violation)
            violations += violation > 1.0e-10
        if (
            not np.all(np.isfinite(left))
            or not np.all(np.isfinite(right))
            or left[0] <= 0.0
            or left[2] <= 0.0
            or right[0] <= 0.0
            or right[2] <= 0.0
        ):
            bad_states += 1
    return Summary(
        "hydro",
        objective,
        samples,
        bad_states,
        violations,
        maximum_violation,
    )


def check_mhd(
    rng: np.random.Generator,
    samples: int,
    objective: str,
) -> Summary:
    bad_states = 0
    violations = 0
    maximum_violation = 0.0
    reduced_indices = list(mhd.REDUCED)
    for _ in range(samples):
        stencil = random_mhd_stencil(rng)
        normal_b = float(stencil[0, 5])
        left, right = mhd.reconstruct_characteristic(
            stencil,
            2,
            5.0 / 3.0,
            objective,
            False,
        )
        basis = mhd.characteristic_basis(
            stencil[2],
            stencil[3],
            5.0 / 3.0,
            normal_b,
        )
        cell_waves = mhd.to_characteristics(
            stencil[:, reduced_indices],
            basis,
        )[[2, 3]]
        trace_waves = mhd.to_characteristics(
            np.array(
                (
                    left[reduced_indices],
                    right[reduced_indices],
                )
            ),
            basis,
        )
        cell_jump = cell_waves[1] - cell_waves[0]
        trace_jump = trace_waves[1] - trace_waves[0]
        for wave in mhd.NONLINEAR_WAVES:
            violation = normalized_interval_violation(
                trace_jump[wave],
                cell_jump[wave],
            )
            maximum_violation = max(maximum_violation, violation)
            violations += violation > 1.0e-10
        if (
            not np.all(np.isfinite(left))
            or not np.all(np.isfinite(right))
            or left[0] <= 0.0
            or left[4] <= 0.0
            or right[0] <= 0.0
            or right[4] <= 0.0
        ):
            bad_states += 1
    return Summary(
        "mhd",
        objective,
        samples,
        bad_states,
        violations,
        maximum_violation,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=243)
    parser.add_argument(
        "--objectives",
        nargs="+",
        choices=("coefficient", "trace", "balanced"),
        default=("coefficient", "balanced"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "runs/current/research/reconstruction_invariants.tsv"
        ),
    )
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    rows: list[Summary] = []
    for objective in args.objectives:
        rows.append(check_hydro(rng, args.samples, objective))
        rows.append(check_mhd(rng, args.samples, objective))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[field.name for field in dataclasses.fields(Summary)],
            delimiter="\t",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(dataclasses.asdict(row))
    for row in rows:
        print(
            f"{row.physics} {row.objective}: samples={row.samples} "
            f"bad_states={row.nonfinite_or_nonpositive} "
            f"jump_violations={row.jump_violations} "
            f"max={row.maximum_normalized_jump_violation:.3e}"
        )
    print(f"wrote {args.output}")
    if any(
        row.nonfinite_or_nonpositive != 0
        or row.jump_violations != 0
        or row.maximum_normalized_jump_violation > 1.0e-10
        for row in rows
    ):
        raise RuntimeError(f"paired-trace invariant failure: {rows}")


if __name__ == "__main__":
    main()
