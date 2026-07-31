#!/usr/bin/env python3
"""Falsify the cross-regime physical-wave detector on unrelated states.

Every sampled average is constructed as an exact convex combination of two
admissible conserved endpoint states.  That makes the mixture test maximally
easy to pass.  The endpoints are otherwise unrelated, so the complete
Rankine--Hugoniot residual must reject them.  Exact moving contacts provide the
positive control in NRHD, NRMHD, SRHD, and flat-spacetime SRMHD.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import research_physical_wave_subcell as nrhd  # noqa: E402
import research_physical_wave_subcell_mhd as nrmhd  # noqa: E402
import research_physical_wave_subcell_srhd as srhd  # noqa: E402
import research_physical_wave_subcell_srmhd as srmhd  # noqa: E402


Array = np.ndarray
PairFactory = Callable[[np.random.Generator], tuple[Array, Array]]
StateMap = Callable[[Array, float], Array]
PrimitiveMap = Callable[[Array, float], Array]
Admissibility = Callable[[Array], bool]
WaveFit = Callable[[Array, Array, Array, float], object | None]


@dataclasses.dataclass(frozen=True)
class AuditResult:
    system: str
    requested_pairs: int
    recovered_mixtures: int
    false_accepts: int
    contact_controls: int
    accepted_contacts: int


def log_uniform(
    rng: np.random.Generator,
    lower_exponent: float = -1.0,
    upper_exponent: float = 1.0,
) -> float:
    return float(10.0 ** rng.uniform(lower_exponent, upper_exponent))


def subluminal_velocity(
    rng: np.random.Generator,
    maximum_speed: float = 0.75,
) -> Array:
    direction = rng.normal(size=3)
    direction /= np.linalg.norm(direction)
    return direction * rng.uniform(0.0, maximum_speed)


def nrhd_pair(rng: np.random.Generator) -> tuple[Array, Array]:
    def state() -> Array:
        return np.array(
            (
                log_uniform(rng),
                rng.uniform(-2.0, 2.0),
                log_uniform(rng),
            )
        )

    return state(), state()


def nrmhd_pair(rng: np.random.Generator) -> tuple[Array, Array]:
    normal_b = rng.uniform(-1.0, 1.0)

    def state() -> Array:
        return np.concatenate(
            (
                [log_uniform(rng)],
                rng.uniform(-1.5, 1.5, size=3),
                [log_uniform(rng), normal_b],
                rng.uniform(-1.5, 1.5, size=2),
            )
        )

    return state(), state()


def srhd_pair(rng: np.random.Generator) -> tuple[Array, Array]:
    def state() -> Array:
        return np.array(
            (
                log_uniform(rng),
                rng.uniform(-0.8, 0.8),
                log_uniform(rng),
            )
        )

    return state(), state()


def srmhd_pair(rng: np.random.Generator) -> tuple[Array, Array]:
    normal_b = rng.uniform(-0.8, 0.8)

    def state() -> Array:
        return np.concatenate(
            (
                [log_uniform(rng, -0.5, 0.5)],
                subluminal_velocity(rng),
                [log_uniform(rng, -0.5, 0.5), normal_b],
                rng.uniform(-0.8, 0.8, size=2),
            )
        )

    return state(), state()


def audit_system(
    *,
    name: str,
    samples: int,
    rng: np.random.Generator,
    pair_factory: PairFactory,
    conserved: StateMap,
    primitive: PrimitiveMap,
    admissible: Admissibility,
    fit: WaveFit,
    contact_left: Array,
    contact_right: Array,
    gamma: float,
) -> AuditResult:
    recovered = 0
    false_accepts = 0
    for _ in range(samples):
        left, right = pair_factory(rng)
        left_fraction = rng.uniform(0.02, 0.98)
        endpoints = conserved(np.array((left, right)), gamma)
        average_state = (
            left_fraction * endpoints[0]
            + (1.0 - left_fraction) * endpoints[1]
        )
        average = primitive(average_state[None, :], gamma)[0]
        if not admissible(average[None, :]):
            continue
        recovered += 1
        if fit(average, left, right, gamma) is not None:
            false_accepts += 1

    contact_controls = 16
    accepted_contacts = 0
    contact_states = conserved(
        np.array((contact_left, contact_right)),
        gamma,
    )
    for left_fraction in np.linspace(0.05, 0.95, contact_controls):
        average_state = (
            left_fraction * contact_states[0]
            + (1.0 - left_fraction) * contact_states[1]
        )
        average = primitive(average_state[None, :], gamma)[0]
        if (
            admissible(average[None, :])
            and fit(average, contact_left, contact_right, gamma) is not None
        ):
            accepted_contacts += 1

    if recovered < max(1, int(0.8 * samples)):
        raise RuntimeError(
            f"{name} primitive inversion recovered only "
            f"{recovered}/{samples} random mixtures"
        )
    if false_accepts:
        raise RuntimeError(
            f"{name} detector accepted {false_accepts}/{recovered} "
            "unrelated endpoint pairs"
        )
    if accepted_contacts != contact_controls:
        raise RuntimeError(
            f"{name} detector accepted only "
            f"{accepted_contacts}/{contact_controls} exact contacts"
        )
    return AuditResult(
        system=name,
        requested_pairs=samples,
        recovered_mixtures=recovered,
        false_accepts=false_accepts,
        contact_controls=contact_controls,
        accepted_contacts=accepted_contacts,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=248)
    parser.add_argument("--gamma", type=float, default=5.0 / 3.0)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)

    nrhd_contact_left = np.array((2.0, 0.4, 1.0))
    nrhd_contact_right = np.array((1.0, 0.4, 1.0))
    nrmhd_contact_left = np.array(
        (2.0, 0.4, 0.1, -0.1, 1.0, 0.5, 0.3, -0.2)
    )
    nrmhd_contact_right = nrmhd_contact_left.copy()
    nrmhd_contact_right[0] = 1.0
    srhd_contact_left = np.array((2.0, 0.5, 1.0))
    srhd_contact_right = np.array((1.0, 0.5, 1.0))
    srmhd_contact_left = np.array(
        (2.0, 0.5, 0.1, -0.1, 1.0, 0.5, 0.3, -0.2)
    )
    srmhd_contact_right = srmhd_contact_left.copy()
    srmhd_contact_right[0] = 1.0

    results = (
        audit_system(
            name="nrhd",
            samples=args.samples,
            rng=rng,
            pair_factory=nrhd_pair,
            conserved=nrhd.base.conserved_from_primitive,
            primitive=nrhd.base.primitive_from_conserved,
            admissible=nrhd.base.primitive_is_admissible,
            fit=nrhd.fit_physical_wave,
            contact_left=nrhd_contact_left,
            contact_right=nrhd_contact_right,
            gamma=args.gamma,
        ),
        audit_system(
            name="nrmhd",
            samples=args.samples,
            rng=rng,
            pair_factory=nrmhd_pair,
            conserved=nrmhd.mhd.conserved_array,
            primitive=nrmhd.mhd.primitive_from_conserved,
            admissible=nrmhd.mhd.primitive_is_admissible,
            fit=nrmhd.fit_physical_mhd_wave,
            contact_left=nrmhd_contact_left,
            contact_right=nrmhd_contact_right,
            gamma=args.gamma,
        ),
        audit_system(
            name="srhd",
            samples=args.samples,
            rng=rng,
            pair_factory=srhd_pair,
            conserved=srhd.conserved_array,
            primitive=srhd.primitive_from_conserved,
            admissible=srhd.primitive_is_admissible,
            fit=srhd.fit_physical_wave,
            contact_left=srhd_contact_left,
            contact_right=srhd_contact_right,
            gamma=args.gamma,
        ),
        audit_system(
            name="srmhd",
            samples=args.samples,
            rng=rng,
            pair_factory=srmhd_pair,
            conserved=srmhd.conserved_array,
            primitive=srmhd.primitive_from_conserved,
            admissible=srmhd.primitive_is_admissible,
            fit=srmhd.fit_physical_wave,
            contact_left=srmhd_contact_left,
            contact_right=srmhd_contact_right,
            gamma=args.gamma,
        ),
    )
    fields = tuple(field.name for field in dataclasses.fields(AuditResult))
    print("\t".join(fields))
    for result in results:
        print("\t".join(str(getattr(result, field)) for field in fields))


if __name__ == "__main__":
    main()
