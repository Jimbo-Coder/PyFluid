"""C++ point-data bindings for the coordinate-independent fCCZ4 equations."""

import sympy as sp

from .fccz4_equations import (
    Fccz4State, Fccz4Geometry, Fccz4Damping, Fccz4Matter, fccz4_geometry, fccz4_rhs,
    moving_puncture_rhs, fccz4_constraints,
)
from .kernel_equations import StructKernel
from .tensor_jets import (SYMMETRIC_COMPONENTS, symbolic_scalar_jet, symbolic_vector_jet, symbolic_tensor_jet)


def symmetric_symbols(prefix):
    return sp.Matrix(3, 3, lambda i, j: sp.Symbol(
        f"{prefix}[{SYMMETRIC_COMPONENTS.index(tuple(sorted((i, j))))}]"
    ))


def vector_symbols(prefix):
    return sp.Matrix([sp.Symbol(f"{prefix}[{i}]") for i in range(3)])


def point_symbols():
    return Fccz4State(
        symbolic_tensor_jet("state.metric"), symbolic_tensor_jet("state.a_tilde"),
        symbolic_scalar_jet("state.chi"), symbolic_scalar_jet("state.trace_k"),
        symbolic_scalar_jet("state.theta"), symbolic_vector_jet("state.connection"),
        symbolic_scalar_jet("state.lapse"), symbolic_vector_jet("state.shift"),
    )


def geometry_symbols():
    return Fccz4Geometry(
        symmetric_symbols("geometry.inverse"),
        tuple(tuple(tuple(sp.Symbol(f"geometry.delta[{i}][{j}][{k}]") for k in range(3))
                    for j in range(3)) for i in range(3)),
        vector_symbols("geometry.contracted_delta"), symmetric_symbols("geometry.physical_ricci"),
        vector_symbols("geometry.z_covariant"), vector_symbols("geometry.z_contravariant"),
        symmetric_symbols("geometry.ricci_z4"), symmetric_symbols("geometry.lapse_hessian"),
        sp.Symbol("geometry.shift_divergence"), vector_symbols("geometry.d_shift_divergence"),
    )


def symmetric_values(matrix):
    return tuple(matrix[i, j] for i, j in SYMMETRIC_COMPONENTS)


def fccz4_geometry_kernel():
    geometry = fccz4_geometry(point_symbols())
    values = (
        *symmetric_values(geometry.inverse),
        *(value for plane in geometry.delta for row in plane for value in row),
        *geometry.contracted_delta, *symmetric_values(geometry.physical_ricci),
        *geometry.z_covariant, *geometry.z_contravariant, *symmetric_values(geometry.ricci_z4),
        *symmetric_values(geometry.lapse_hessian), geometry.shift_divergence, *geometry.d_shift_divergence,
    )
    return StructKernel("fccz4_geometry", "Fccz4Geometry", (("const Fccz4Point&", "state"),),
                        tuple(f"component_{i}" for i in range(len(values))), values)


def fccz4_rhs_kernel():
    matter = Fccz4Matter(
        sp.Symbol("matter.rho"), sp.Matrix([sp.Symbol(f"matter.momentum{i+1}") for i in range(3)]),
        sp.Matrix(3, 3, lambda i, j: sp.Symbol(f"matter.stress{min(i,j)+1}{max(i,j)+1}")),
    )
    damping = Fccz4Damping(*(sp.Symbol(f"damping.kappa{i}") for i in (1, 2, 3)))
    rhs = fccz4_rhs(point_symbols(), matter, damping, geometry_symbols())
    values = (rhs.chi, *symmetric_values(rhs.metric), *symmetric_values(rhs.a_tilde),
              rhs.trace_k, rhs.theta, *rhs.connection)
    return StructKernel(
        "fccz4_rhs", "Fccz4Rhs",
        (("const Fccz4Point&", "state"), ("const EulerianMatter&", "matter"),
         ("const Fccz4Damping&", "damping"), ("const Fccz4Geometry&", "geometry")),
        tuple(f"component_{i}" for i in range(len(values))), values,
    )


def fccz4_gauge_kernel():
    rhs = moving_puncture_rhs(point_symbols(), vector_symbols("rhs.connection"),
                             symbolic_vector_jet("driver"), sp.Symbol("eta"))
    values = (rhs.lapse, *rhs.shift, *rhs.driver)
    return StructKernel(
        "fccz4_moving_puncture_rhs", "Fccz4GaugeRhs",
        (("const Fccz4Point&", "state"), ("const Fccz4Rhs&", "rhs"),
         ("const ReferenceVectorJet&", "driver"), ("amrex::Real", "eta")),
        tuple(f"component_{i}" for i in range(len(values))), values,
    )


def fccz4_constraints_kernel():
    matter = Fccz4Matter(
        sp.Symbol("matter.rho"),
        sp.Matrix([sp.Symbol(f"matter.momentum{i+1}") for i in range(3)]),
        sp.zeros(3),
    )
    constraints = fccz4_constraints(point_symbols(), matter, geometry_symbols())
    values = (constraints.hamiltonian, *constraints.momentum, *constraints.z_covariant,
              constraints.theta, constraints.log_determinant, constraints.trace_a)
    return StructKernel(
        "fccz4_constraints", "Fccz4Constraints",
        (("const Fccz4Point&", "state"), ("const EulerianMatter&", "matter"),
         ("const Fccz4Geometry&", "geometry")),
        tuple(f"component_{i}" for i in range(len(values))), values,
    )
