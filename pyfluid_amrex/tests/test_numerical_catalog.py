from pathlib import Path
import sys

import sympy as sp


CODEGEN = Path(__file__).resolve().parents[1] / "codegen"
sys.path.insert(0, str(CODEGEN))

from pyfluid_codegen.differential_equations import (  # noqa: E402
    average_stencils,
    centered_stencils,
    centering_stencils,
    ko_stencils,
    one_sided_stencils,
    upwind_stencils,
)
from pyfluid_codegen.numerical_catalog import SPATIAL_ORDERS  # noqa: E402


def _monomial_sample(offset: int, degree: int) -> sp.Rational:
    return sp.Rational(offset) ** degree


def _monomial_average(offset: int, degree: int) -> sp.Rational:
    half = sp.Rational(1, 2)
    return ((offset + half) ** (degree + 1)
            - (offset - half) ** (degree + 1)) / (degree + 1)


def test_every_stencil_family_uses_the_one_order_catalog() -> None:
    assert tuple(stencil.order for stencil in centered_stencils(1)) == SPATIAL_ORDERS
    assert tuple(pair[0].order for pair in one_sided_stencils()) == SPATIAL_ORDERS
    assert tuple(pair[0].order for pair in upwind_stencils()) == SPATIAL_ORDERS
    assert tuple(stencil.order for stencil in ko_stencils()) == SPATIAL_ORDERS
    assert tuple(stencil.order for stencil in centering_stencils()) == SPATIAL_ORDERS


def test_derivative_moments_and_dissipation_sign() -> None:
    for derivative in (1, 2):
        families = [*centered_stencils(derivative)]
        families.extend(stencil for pair in one_sided_stencils(derivative) for stencil in pair)
        if derivative == 1:
            families.extend(stencil for pair in upwind_stencils() for stencil in pair)
        for stencil in families:
            for degree in range(stencil.order + derivative):
                moment = sum(weight * offset**degree for offset, weight
                             in zip(stencil.offsets, stencil.weights, strict=True))
                expected = sp.factorial(derivative) if degree == derivative else 0
                assert moment == expected, (stencil.order, derivative, degree)
    for stencil in ko_stencils():
        assert sum(stencil.weights) == 0
        nyquist = sum(weight * (-1)**abs(offset) for offset, weight
                      in zip(stencil.offsets, stencil.weights, strict=True))
        assert nyquist < 0


def test_centering_weights_reproduce_staggered_polynomials() -> None:
    for stencil in centering_stencils():
        for degree in range(stencil.order):
            actual = sum(
                weight * _monomial_sample(offset, degree)
                for offset, weight in zip(stencil.offsets, stencil.weights, strict=True)
            )
            assert sp.simplify(actual - sp.Rational(1, 2) ** degree) == 0


def test_point_average_conversions_reproduce_polynomial_functionals() -> None:
    for to_average in (True, False):
        for stencil in average_stencils(to_average=to_average):
            # Symmetry makes the one extra odd moment exact.
            for degree in range(stencil.order):
                source = _monomial_sample if to_average else _monomial_average
                target = _monomial_average if to_average else _monomial_sample
                actual = sum(
                    weight * source(offset, degree)
                    for offset, weight in zip(
                        stencil.offsets, stencil.weights, strict=True
                    )
                )
                assert sp.simplify(actual - target(0, degree)) == 0


def test_runge_kutta_checks_reject_nonlinear_order_loss() -> None:
    from fractions import Fraction
    from pyfluid_codegen.time_integrator_catalog import TIME_INTEGRATORS, validate_order

    method = next(item for item in TIME_INTEGRATORS if item.name == "rk4")
    # Preserve stage times and final quadrature moments, but break a nonlinear
    # rooted-tree condition. A sum-of-weights check would accept this tableau.
    a = [list(row) for row in method.a]
    a[3] = [Fraction(0), Fraction(1, 2), Fraction(1, 2), Fraction(0)]
    assert tuple(map(sum, a)) == method.c
    try:
        validate_order(a, method.b, method.c, method.order)
    except ValueError as error:
        assert "order-4" in str(error)
    else:
        raise AssertionError("nonlinear fourth-order defect was accepted")


def test_projection_weights_follow_the_accepted_update() -> None:
    from fractions import Fraction
    from pyfluid_codegen.time_integrator_catalog import TIME_INTEGRATORS

    methods = {method.name: method for method in TIME_INTEGRATORS}
    assert methods["ssprk3"].projection_weights == (1, Fraction(1, 6), Fraction(2, 3))
    assert methods["rk4"].projection_weights == (1, 0, 0, 0)
    assert methods["midpoint"].projection_weights == (1, 0)
