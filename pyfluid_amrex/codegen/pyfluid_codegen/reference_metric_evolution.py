"""Reference-metric conformal evolution in a local orthonormal basis.

SphericalNR (arXiv:2002.06225v2), Section II.1 metric/phi equations and
Eq. (33), with chi=exp(-4 phi).
Inputs carry reference-covariant derivatives, including their derivative
indices. No patch map, coordinate family, or storage dimension enters these
equations. The reference metric and local basis are time independent.
"""

from dataclasses import dataclass

import sympy as sp

from .kernel_equations import StructKernel
from .tensor_jets import SYMMETRIC_COMPONENTS, SymmetricTensorJet, symbolic_tensor_jet


@dataclass(frozen=True)
class ConformalMetricRhs:
    chi: sp.Expr
    metric: sp.Matrix


def conformal_metric_rhs(
    metric: SymmetricTensorJet,
    a_tilde: sp.Matrix,
    chi: sp.Expr,
    d_chi: sp.Matrix,
    lapse: sp.Expr,
    trace_k: sp.Expr,
    shift: sp.Matrix,
    d_shift: sp.Matrix,
) -> ConformalMetricRhs:
    """Coordinate-time RHS; d_shift[derivative, vector component].

    The input metric is bar-gamma, not its deviation from the reference.
    Since the reference is static, its deviation has the same RHS. trace_k
    is physical K, not the Cartesian Z4c variable K_hat=K-2 Theta.
    The conformal divergence retains derivatives of det(bar-gamma)/det(hat-gamma)
    so this expression is also meaningful off the determinant constraint.
    """
    inverse = metric.value.inv()
    divergence = sp.trace(d_shift)
    for k in range(3):
        divergence += shift[k] * sp.trace(inverse * metric.first[k]) / 2

    lie_metric = d_shift * metric.value + metric.value * d_shift.T
    for k in range(3):
        lie_metric += shift[k] * metric.first[k]
    metric_rhs = lie_metric - sp.Rational(2, 3) * metric.value * divergence - 2 * lapse * a_tilde
    chi_rhs = (shift.dot(d_chi)
               + sp.Rational(2, 3) * chi * (lapse * trace_k - divergence))
    return ConformalMetricRhs(chi_rhs, metric_rhs)


def reference_metric_evolution_kernels():
    metric = symbolic_tensor_jet()
    a_tilde = sp.Matrix(3, 3, lambda i, j: sp.Symbol(
        f"a_tilde[{SYMMETRIC_COMPONENTS.index(tuple(sorted((i, j))))}]"
    ))
    shift = sp.Matrix([sp.Symbol(f"shift.value[{i}]") for i in range(3)])
    d_shift = sp.Matrix(3, 3, lambda i, j: sp.Symbol(f"shift.first[{i}][{j}]"))
    d_chi = sp.Matrix([sp.Symbol(f"d_chi[{i}]") for i in range(3)])
    result = conformal_metric_rhs(
        metric, a_tilde, sp.Symbol("chi"), d_chi,
        sp.Symbol("lapse"), sp.Symbol("trace_k"), shift, d_shift,
    )
    return (StructKernel(
        name="reference_conformal_metric_rhs",
        return_type="ConformalMetricRhs",
        arguments=(("const SymmetricTensorJet&", "jet"),
                   ("const amrex::Real*", "a_tilde"), ("amrex::Real", "chi"),
                   ("const amrex::Real*", "d_chi"), ("amrex::Real", "lapse"),
                   ("amrex::Real", "trace_k"), ("const ReferenceVectorGradient&", "shift")),
        outputs=("chi", *(f"metric{i+1}{j+1}" for i, j in SYMMETRIC_COMPONENTS)),
        expressions=(result.chi, *(result.metric[i, j] for i, j in SYMMETRIC_COMPONENTS)),
    ),)
