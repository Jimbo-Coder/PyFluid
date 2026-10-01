"""Reference-metric fCCZ4 on a static flat geometric reference metric.

Equations follow Mewes et al., arXiv:2002.06225v2, Section II.1.
All tensor indices, including derivative indices, are in the local reference
orthonormal basis. Derivatives are reference-covariant. Physical K is used,
and chi=exp(-4 phi). The constraint damping convention absorbs 1/alpha into
kappa_1, as in that paper. No coordinate, patch, or finite-difference policy
is selected here.
"""

from dataclasses import dataclass

import sympy as sp

from .reference_metric_evolution import conformal_metric_rhs
from .spacetime_geometry import reference_connection_expressions
from .tensor_jets import ScalarJet, VectorJet, SymmetricTensorJet


@dataclass(frozen=True)
class Fccz4State:
    metric: SymmetricTensorJet
    a_tilde: SymmetricTensorJet
    chi: ScalarJet
    trace_k: ScalarJet
    theta: ScalarJet
    connection: VectorJet
    lapse: ScalarJet
    shift: VectorJet


@dataclass(frozen=True)
class Fccz4Matter:
    energy: sp.Expr
    momentum: sp.Matrix  # Covariant Eulerian momentum S_i.
    stress: sp.Matrix  # Covariant physical stress S_ij.


@dataclass(frozen=True)
class Fccz4Damping:
    kappa1: sp.Expr
    kappa2: sp.Expr
    kappa3: sp.Expr = sp.Integer(1)


@dataclass(frozen=True)
class Fccz4Geometry:
    inverse: sp.Matrix
    delta: tuple
    contracted_delta: sp.Matrix
    physical_ricci: sp.Matrix
    z_covariant: sp.Matrix
    z_contravariant: sp.Matrix
    ricci_z4: sp.Matrix
    lapse_hessian: sp.Matrix
    shift_divergence: sp.Expr
    d_shift_divergence: sp.Matrix


@dataclass(frozen=True)
class Fccz4Rhs:
    chi: sp.Expr
    metric: sp.Matrix
    a_tilde: sp.Matrix
    trace_k: sp.Expr
    theta: sp.Expr
    connection: sp.Matrix


@dataclass(frozen=True)
class Fccz4GaugeRhs:
    lapse: sp.Expr
    shift: sp.Matrix
    driver: sp.Matrix


def moving_puncture_rhs(state: Fccz4State, connection_rhs: sp.Matrix,
                       driver: VectorJet, eta: sp.Expr) -> Fccz4GaugeRhs:
    """Advective 1+log (Bona-Masso f=2/alpha) and covariant Gamma driver.

    The auxiliary driver obeys the two-equation convention beta_dot=B+advection,
    B_dot=3/4 (Lambda_dot-advection)+advection(B)-eta B. The 3/4 factor belongs
    in B_dot in this convention, not in both equations.
    """
    lapse = (state.shift.value.dot(state.lapse.first)
             - 2 * state.lapse.value * (state.trace_k.value - 2 * state.theta.value))
    shift = driver.value + state.shift.first.T * state.shift.value
    driver_rhs = (sp.Rational(3, 4) * (connection_rhs - state.connection.first.T * state.shift.value)
                  + driver.first.T * state.shift.value - eta * driver.value)
    return Fccz4GaugeRhs(lapse, shift, driver_rhs)


def contract(inverse: sp.Matrix, tensor: sp.Matrix) -> sp.Expr:
    return sp.trace(inverse * tensor)


def scalar_hessian(jet: ScalarJet, delta) -> sp.Matrix:
    return sp.Matrix(3, 3, lambda i, j: jet.second[i, j] - sum(
        delta[k][i][j] * jet.first[k] for k in range(3)
    ))


def fccz4_geometry(state: Fccz4State) -> Fccz4Geometry:
    metric = state.metric.value
    inverse = metric.inv()
    first = state.metric.first
    second = state.metric.second
    delta, contracted_delta = reference_connection_expressions(state.metric)
    contracted_delta = sp.Matrix(contracted_delta)
    d_inverse = tuple(-inverse * derivative * inverse for derivative in first)

    def connection_derivative(d, i, j, k):
        return sum(
            (d_inverse[d][i, l] * (first[j][l, k] + first[k][l, j] - first[l][j, k])
             + inverse[i, l] * (second[d][j][l, k] + second[d][k][l, j] - second[d][l][j, k])) / 2
            for l in range(3)
        )

    d_delta = tuple(tuple(tuple(tuple(connection_derivative(d, i, j, k)
                                     for k in range(3)) for j in range(3))
                         for i in range(3)) for d in range(3))
    d_contracted = sp.Matrix(3, 3, lambda d, i: sum(
        d_inverse[d][j, k] * delta[i][j][k] + inverse[j, k] * d_delta[d][i][j][k]
        for j in range(3) for k in range(3)
    ))
    ricci = sp.Matrix(3, 3, lambda i, j: sum(
        d_delta[k][k][i][j] - d_delta[j][k][i][k]
        + sum(delta[k][k][l] * delta[l][i][j] - delta[k][j][l] * delta[l][i][k]
              for l in range(3)) for k in range(3)
    ))

    chi = state.chi.value
    d_chi = state.chi.first
    hessian_chi = scalar_hessian(state.chi, delta)
    physical_ricci = (
        ricci + (hessian_chi + metric * contract(inverse, hessian_chi)) / (2 * chi)
        - (d_chi * d_chi.T + 3 * metric * (d_chi.dot(inverse * d_chi))) / (4 * chi**2)
    )
    physical_delta = tuple(tuple(tuple(
        delta[k][i][j] - (
            int(k == i) * d_chi[j] + int(k == j) * d_chi[i]
            - metric[i, j] * (inverse * d_chi)[k]
        ) / (2 * chi) for j in range(3)) for i in range(3)) for k in range(3))
    difference = state.connection.value - contracted_delta
    z_covariant = metric * difference / 2
    z_contravariant = chi * difference / 2
    d_z = sp.Matrix(3, 3, lambda d, i: (
        first[d] * difference / 2
        + metric * (state.connection.first.row(d).T - d_contracted.row(d).T) / 2
    )[i] - sum(physical_delta[k][d][i] * z_covariant[k] for k in range(3)))
    ricci_z4 = physical_ricci + d_z + d_z.T

    trace_delta = sp.Matrix([sum(delta[j][j][k] for j in range(3)) for k in range(3)])
    divergence = sp.trace(state.shift.first) + trace_delta.dot(state.shift.value)
    d_divergence = sp.Matrix([sum(
        state.shift.second[d][j, j]
        + sum(d_delta[d][j][j][k] * state.shift.value[k]
              + delta[j][j][k] * state.shift.first[d, k] for k in range(3))
        for j in range(3)
    ) for d in range(3)])
    return Fccz4Geometry(
        inverse, delta, contracted_delta, physical_ricci,
        z_covariant, z_contravariant, ricci_z4,
        scalar_hessian(state.lapse, physical_delta), divergence, d_divergence,
    )


def fccz4_rhs(state: Fccz4State, matter: Fccz4Matter, damping: Fccz4Damping,
             geometry: Fccz4Geometry | None = None) -> Fccz4Rhs:
    geometry = geometry if geometry is not None else fccz4_geometry(state)
    inverse = geometry.inverse
    metric = state.metric.value
    a_tilde = state.a_tilde.value
    a_upper = inverse * a_tilde * inverse
    a_squared = contract(a_upper, a_tilde)
    chi, lapse = state.chi.value, state.lapse.value
    trace_k, theta = state.trace_k.value, state.theta.value
    shift = state.shift.value
    d_shift = state.shift.first
    divergence = geometry.shift_divergence
    z = geometry.z_contravariant
    ricci_z4_scalar = chi * contract(inverse, geometry.ricci_z4)
    stress_trace = chi * contract(inverse, matter.stress)
    pi = sp.pi

    metric_rhs = conformal_metric_rhs(
        state.metric, a_tilde, chi, state.chi.first, lapse, trace_k, shift, d_shift,
    )
    curvature_source = -geometry.lapse_hessian + lapse * (geometry.ricci_z4 - 8 * pi * matter.stress)
    trace_free_source = curvature_source - metric * contract(inverse, curvature_source) / 3
    a_rhs = (d_shift * a_tilde + a_tilde * d_shift.T
             - sp.Rational(2, 3) * a_tilde * divergence
             - 2 * lapse * a_tilde * inverse * a_tilde
             + lapse * a_tilde * (trace_k - 2 * theta) + chi * trace_free_source)
    for k in range(3):
        a_rhs += shift[k] * state.a_tilde.first[k]

    k_rhs = (shift.dot(state.trace_k.first) - chi * contract(inverse, geometry.lapse_hessian)
             + lapse * (ricci_z4_scalar + trace_k**2 - 2 * theta * trace_k)
             - 3 * damping.kappa1 * (1 + damping.kappa2) * theta
             + 4 * pi * lapse * (stress_trace - 3 * matter.energy))
    theta_rhs = (shift.dot(state.theta.first)
                 + lapse / 2 * (ricci_z4_scalar - a_squared + sp.Rational(2, 3) * trace_k**2
                                - 2 * theta * trace_k)
                 - z.dot(state.lapse.first) - damping.kappa1 * (2 + damping.kappa2) * theta
                 - 8 * pi * lapse * matter.energy)

    d_phi = -state.chi.first / (4 * chi)
    connection_rhs = state.connection.first.T * shift - d_shift.T * state.connection.value
    for i in range(3):
        connection_rhs[i] += sum(
            inverse[j, k] * state.shift.second[j][k, i]
            + 2 * lapse * a_upper[j, k] * geometry.delta[i][j][k]
            for j in range(3) for k in range(3)
        )
        connection_rhs[i] += (
            sp.Rational(2, 3) * geometry.contracted_delta[i] * divergence
            + (inverse * geometry.d_shift_divergence)[i] / 3
            - 2 * (a_upper * (state.lapse.first - 6 * lapse * d_phi))[i]
            - sp.Rational(4, 3) * lapse * (inverse * state.trace_k.first)[i]
            + 2 * (inverse * (lapse * state.theta.first - theta * state.lapse.first))[i]
            - sp.Rational(4, 3) * lapse * trace_k * z[i] / chi
            - 2 * damping.kappa1 * z[i] / chi
            + 2 * damping.kappa3 / chi * (
                sp.Rational(2, 3) * z[i] * sp.trace(d_shift) - (d_shift.T * z)[i])
            - 16 * pi * lapse * (inverse * matter.momentum)[i]
        )
    return Fccz4Rhs(metric_rhs.chi, metric_rhs.metric, a_rhs, k_rhs, theta_rhs, connection_rhs)


@dataclass(frozen=True)
class Fccz4Constraints:
    hamiltonian: sp.Expr
    momentum: sp.Matrix
    z_covariant: sp.Matrix
    theta: sp.Expr
    log_determinant: sp.Expr
    trace_a: sp.Expr


def fccz4_constraints(state: Fccz4State, matter: Fccz4Matter,
                      geometry: Fccz4Geometry | None = None) -> Fccz4Constraints:
    """Physical ADM constraints, independent of the constraint-added RHS."""
    geometry = geometry if geometry is not None else fccz4_geometry(state)
    inverse = geometry.inverse
    metric = state.metric.value
    chi = state.chi.value
    gradient = state.chi.first
    a_mixed = inverse * state.a_tilde.value
    d_inverse = tuple(-inverse * derivative * inverse for derivative in state.metric.first)
    physical_delta = tuple(tuple(tuple(
        geometry.delta[k][i][j] - (
            int(k == i) * gradient[j] + int(k == j) * gradient[i]
            - metric[i, j] * (inverse * gradient)[k]
        ) / (2 * chi) for j in range(3)) for i in range(3)) for k in range(3))
    momentum = sp.Matrix([sum(
        (d_inverse[j] * state.a_tilde.value + inverse * state.a_tilde.first[j])[j, i]
        + sum(physical_delta[j][j][k] * a_mixed[k, i]
              - physical_delta[k][j][i] * a_mixed[j, k] for k in range(3))
        for j in range(3)
    ) - sp.Rational(2, 3) * state.trace_k.first[i] - 8 * sp.pi * matter.momentum[i]
        for i in range(3)])
    hamiltonian = (chi * contract(inverse, geometry.physical_ricci)
                   - sp.trace(a_mixed * a_mixed)
                   + sp.Rational(2, 3) * state.trace_k.value**2 - 16 * sp.pi * matter.energy)
    return Fccz4Constraints(hamiltonian, momentum, geometry.z_covariant,
                            state.theta.value, sp.log(metric.det()), sp.trace(a_mixed))
