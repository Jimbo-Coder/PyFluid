"""Tensor-basis derivatives shared by reference-metric spacetime equations.

The input is a symmetric tensor in the orthonormal reference basis, together
with coordinate partial derivatives of those rescaled components. Basis
factors are differentiated analytically, never by a mesh stencil. A symmetry
reduction zeros field derivatives, not derivatives of the basis factors.

This is geometry, not a Z4c-to-fCCZ4 formulation conversion. It applies to
either a physical metric or a conformal metric with its own determinant rule.
All expressions are for regular charts; pole/origin limits need separate
boundary and discretization treatment.
"""

import sympy as sp

from .coordinates import OrthogonalFrame, orthogonal_frames
from .kernel_equations import StructKernel
from .metric_differential import metric_christoffel
from .reference_metric import ReferenceMetricGeometry
from .tensor_jets import (
    SYMMETRIC_COMPONENTS, ScalarJet, VectorJet, SymmetricTensorJet, symbolic_tensor_jet,
    symbolic_scalar_jet, symbolic_vector_jet,
)


def reference_covariant_scalar_jet(frame: OrthogonalFrame, jet: ScalarJet) -> ScalarJet:
    geometry = ReferenceMetricGeometry(frame)
    scales = frame.scale_factors
    first = sp.Matrix([jet.first[i] / scales[i] for i in range(3)])
    second = sp.Matrix(3, 3, lambda i, j: (
        jet.second[i, j] - sum(geometry.christoffel(k, i, j) * jet.first[k] for k in range(3))
    ) / (scales[i] * scales[j]))
    return ScalarJet(jet.value, first, second)


def reference_covariant_vector_jet(frame: OrthogonalFrame, jet: VectorJet) -> VectorJet:
    """Unit-vector components and their coordinate partial jets to hat-D jets."""
    geometry = ReferenceMetricGeometry(frame)
    scales, coordinates = frame.scale_factors, frame.coordinates
    connection = geometry.christoffel
    vector = sp.Matrix([jet.value[i] / scales[i] for i in range(3)])
    partial = sp.Matrix(3, 3, lambda d, i:
        jet.first[d, i] / scales[i] + jet.value[i] * sp.diff(1 / scales[i], coordinates[d]))
    first = sp.Matrix(3, 3, lambda d, i: partial[d, i]
                      + sum(connection(i, d, k) * vector[k] for k in range(3)))

    def second_component(d, e, i):
        result = (jet.second[d][e, i] / scales[i]
                  + jet.first[d, i] * sp.diff(1 / scales[i], coordinates[e])
                  + jet.first[e, i] * sp.diff(1 / scales[i], coordinates[d])
                  + jet.value[i] * sp.diff(1 / scales[i], coordinates[d], coordinates[e]))
        for k in range(3):
            result += (sp.diff(connection(i, e, k), coordinates[d]) * vector[k]
                       + connection(i, e, k) * partial[d, k]
                       + connection(i, d, k) * first[e, k]
                       - connection(k, d, e) * first[k, i])
        return sp.trigsimp(sp.simplify(result * scales[i] / (scales[d] * scales[e])), method="fu")
    return VectorJet(
        jet.value,
        sp.Matrix(3, 3, lambda d, i: sp.trigsimp(sp.simplify(
            first[d, i] * scales[i] / scales[d]), method="fu")),
        tuple(sp.Matrix(3, 3, lambda e, i: second_component(d, e, i)) for d in range(3)),
    )


def coordinate_metric_jet(
    frame: OrthogonalFrame, rescaled: SymmetricTensorJet,
) -> SymmetricTensorJet:
    """Apply g_ij = h_i h_j g_(i)(j), including both product derivatives."""
    coordinates = frame.coordinates
    scale = sp.Matrix(3, 3, lambda i, j: frame.scale_factors[i] * frame.scale_factors[j])
    first_scale = tuple(scale.diff(q) for q in coordinates)
    def product(left, right):
        return left.multiply_elementwise(right)
    first = tuple(
        product(scale, rescaled.first[d]) + product(first_scale[d], rescaled.value)
        for d in range(3)
    )
    second = tuple(tuple(
        product(scale, rescaled.second[d][e])
        + product(first_scale[d], rescaled.first[e])
        + product(first_scale[e], rescaled.first[d])
        + product(first_scale[d].diff(coordinates[e]), rescaled.value)
        for e in range(3)
    ) for d in range(3))
    return SymmetricTensorJet(product(scale, rescaled.value), first, second)


def reference_covariant_metric_jet(
    frame: OrthogonalFrame, rescaled: SymmetricTensorJet,
) -> SymmetricTensorJet:
    """Return hat-D_a g_bc and hat-D_a hat-D_b g_cd in the unit basis.

    The second derivative has four covariant indices. In particular the
    connection acts on the first derivative's direction as well as on the
    two metric indices. This term survives field symmetry reductions.
    """
    geometry = ReferenceMetricGeometry(frame)
    connection = geometry.christoffel
    jet = coordinate_metric_jet(frame, rescaled)
    first = tuple(sp.Matrix(3, 3, lambda i, j:
        jet.first[d][i, j] - sum(
            connection(k, d, i) * jet.value[k, j]
            + connection(k, d, j) * jet.value[i, k] for k in range(3)
        )) for d in range(3))

    def second_component(d, e, i, j):
        result = jet.second[d][e][i, j]
        for k in range(3):
            result -= (
                sp.diff(connection(k, e, i), frame.coordinates[d]) * jet.value[k, j]
                + connection(k, e, i) * jet.first[d][k, j]
                + sp.diff(connection(k, e, j), frame.coordinates[d]) * jet.value[i, k]
                + connection(k, e, j) * jet.first[d][i, k]
                + connection(k, d, e) * first[k][i, j]
                + connection(k, d, i) * first[e][k, j]
                + connection(k, d, j) * first[e][i, k]
            )
        return result

    # Jets enter linearly. Expansion plus targeted trig identities avoids
    # general multivariate factorization over every independent jet symbol.
    scales = frame.scale_factors
    return SymmetricTensorJet(
        rescaled.value,
        tuple(sp.Matrix(3, 3, lambda i, j: sp.trigsimp(sp.expand(
            first[d][i, j] / (scales[d] * scales[i] * scales[j])
        ), method="fu")) for d in range(3)),
        tuple(tuple(sp.Matrix(3, 3, lambda i, j: sp.trigsimp(sp.expand(
            second_component(d, e, i, j)
            / (scales[d] * scales[e] * scales[i] * scales[j])
        ), method="fu")) for e in range(3)) for d in range(3)),
    )


def spacetime_geometry_kernels():
    """Emit the same full tensor operation for each registered orthogonal frame."""
    kernels = []
    operations = (
        ("coordinate", coordinate_metric_jet),
        ("reference", reference_covariant_metric_jet),
    )
    for frame in orthogonal_frames():
        for operation, owner in operations:
            result = owner(frame, symbolic_tensor_jet())
            coordinates = dict(zip(frame.coordinates, sp.symbols("q1 q2 q3"), strict=True))
            matrices = [result.value, *result.first]
            matrices.extend(matrix for row in result.second for matrix in row)
            expressions = tuple(
                sp.simplify(matrix[i, j]).subs(coordinates)
                for matrix in matrices for i, j in SYMMETRIC_COMPONENTS
            )
            kernels.append(StructKernel(
                name=f"{frame.name}_{operation}_metric_jet",
                return_type="SymmetricTensorJet",
                arguments=(("amrex::Real", "q1"), ("amrex::Real", "q2"),
                           ("amrex::Real", "q3"), ("const SymmetricTensorJet&", "jet")),
                outputs=tuple(f"component_{n}" for n in range(len(expressions))),
                expressions=expressions,
            ))
    kernels.append(reference_connection_kernel())
    for frame in orthogonal_frames():
        coordinates = dict(zip(frame.coordinates, sp.symbols("q1 q2 q3"), strict=True))
        if frame.cartesian_position is None:
            raise ValueError(f"{frame.name} needs a Cartesian embedding to generate its frame")
        jacobian = sp.Matrix(frame.cartesian_position).jacobian(frame.coordinates)
        basis = sp.diag(*(1 / h for h in frame.scale_factors)) * jacobian.T
        kernels.append(StructKernel(
            name=f"{frame.name}_reference_basis", return_type="ReferenceBasis",
            arguments=(("amrex::Real", "q1"), ("amrex::Real", "q2"), ("amrex::Real", "q3")),
            outputs=tuple(f"component_{i}" for i in range(9)),
            expressions=tuple(sp.simplify(value).subs(coordinates) for value in basis),
        ))
        scalar = reference_covariant_scalar_jet(frame, symbolic_scalar_jet("jet"))
        vector = reference_covariant_vector_jet(frame, symbolic_vector_jet("jet"))
        scalar_values = (scalar.value, *scalar.first, *scalar.second)
        vector_values = (*vector.value, *vector.first, *(value for matrix in vector.second for value in matrix))
        for name, return_type, values in (
            ("scalar", "ReferenceScalarJet", scalar_values),
            ("vector", "ReferenceVectorJet", vector_values),
        ):
            kernels.append(StructKernel(
                name=f"{frame.name}_reference_{name}_jet", return_type=return_type,
                arguments=(("amrex::Real", "q1"), ("amrex::Real", "q2"), ("amrex::Real", "q3"),
                           (f"const {return_type}&", "jet")),
                outputs=tuple(f"component_{i}" for i in range(len(values))),
                expressions=tuple(sp.simplify(value).subs(coordinates) for value in values),
            ))
    return tuple(kernels)


def reference_connection_expressions(jet: SymmetricTensorJet):
    """Delta = Gamma(metric) - Gamma(reference), SphericalNR Eq. (21).

    All indices are in the orthonormal reference basis; jet.first contains
    reference-covariant derivatives, not coordinate partial derivatives.
    The same metric contraction as the coordinate Christoffel formula applies
    because the connection difference is a tensor.
    """
    inverse = jet.value.inv()
    delta = metric_christoffel(inverse, list(jet.first))
    contracted = tuple(sum(
        inverse[j, k] * delta[i][j][k] for j in range(3) for k in range(3)
    ) for i in range(3))
    return delta, contracted


def reference_connection_kernel():
    delta, contracted = reference_connection_expressions(symbolic_tensor_jet())
    expressions = tuple(value for plane in delta for row in plane for value in row) + contracted
    return StructKernel(
        name="reference_metric_connection",
        return_type="ReferenceMetricConnection",
        arguments=(("const SymmetricTensorJet&", "jet"),),
        outputs=tuple(f"component_{n}" for n in range(len(expressions))),
        expressions=expressions,
    )
