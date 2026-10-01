"""Independent exact-spacetime and matter checks for the generated formulation."""

from pathlib import Path
import sys
import unittest

import sympy as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "codegen"))
from pyfluid_codegen.coordinates import cartesian_frame, spherical_frame
from pyfluid_codegen.fccz4_equations import (
    Fccz4Damping, Fccz4Matter, Fccz4State, fccz4_rhs, fccz4_constraints, fccz4_geometry,
)
from pyfluid_codegen.fccz4_kernels import point_symbols
from pyfluid_codegen.z4c_equations import (
    z4c_connections_kernel, z4c_diagnostic_geometry_kernel, z4c_constraints_kernel,
)
from pyfluid_codegen.spacetime_geometry import (
    reference_covariant_metric_jet, reference_covariant_scalar_jet, reference_covariant_vector_jet,
)
from pyfluid_codegen.tensor_jets import ScalarJet, VectorJet, SymmetricTensorJet


def sample_state(frame, values, point):
    coordinates = frame.coordinates

    def evaluate(value):
        if isinstance(value, sp.MatrixBase):
            return value.subs(point).applyfunc(sp.simplify)
        return sp.simplify(sp.sympify(value).subs(point))

    def scalar(value):
        value = sp.sympify(value)
        partial = ScalarJet(value, sp.Matrix([sp.diff(value, q) for q in coordinates]),
                            sp.hessian(value, coordinates))
        result = reference_covariant_scalar_jet(frame, partial)
        return ScalarJet(evaluate(result.value), evaluate(result.first), evaluate(result.second))

    def tensor(value):
        partial = SymmetricTensorJet.differentiate(value, coordinates)
        result = reference_covariant_metric_jet(frame, partial)
        return SymmetricTensorJet(evaluate(result.value), tuple(evaluate(m) for m in result.first),
                                  tuple(tuple(evaluate(m) for m in row) for row in result.second))

    def vector(value):
        first = sp.Matrix(3, 3, lambda d, i: sp.diff(value[i], coordinates[d]))
        partial = VectorJet(value, first, tuple(first.diff(q) for q in coordinates))
        result = reference_covariant_vector_jet(frame, partial)
        return VectorJet(evaluate(result.value), evaluate(result.first), tuple(evaluate(m) for m in result.second))

    return Fccz4State(
        tensor(values.get("metric", sp.eye(3))), tensor(values.get("a_tilde", sp.zeros(3))),
        scalar(values.get("chi", 1)), scalar(values.get("trace_k", 0)), scalar(values.get("theta", 0)),
        vector(values.get("connection", sp.zeros(3, 1))), scalar(values.get("lapse", 1)),
        vector(values.get("shift", sp.zeros(3, 1))),
    )


VACUUM = Fccz4Matter(sp.Integer(0), sp.zeros(3, 1), sp.zeros(3))
DAMPING = Fccz4Damping(sp.Rational(1, 10), sp.Integer(0))


class Fccz4EquationTests(unittest.TestCase):
    def test_constraint_derivative_dependencies(self):
        # The runtime diagnostic gather omits gauge and transport stencils.
        # Check the symbolic source, including intermediate geometry, so a
        # future equation change cannot silently consume an unfilled input.
        cartesian_symbols = set()
        for kernel in (z4c_connections_kernel(), z4c_diagnostic_geometry_kernel(),
                       z4c_constraints_kernel()):
            for expression in (*kernel.expressions, *(value for _, value in kernel.bindings)):
                cartesian_symbols.update(expression.free_symbols)
        cartesian_members = {
            str(symbol).split("[", 1)[0] for symbol in cartesian_symbols
            if str(symbol).startswith("derivatives.")
        }
        self.assertEqual(cartesian_members, {
            "derivatives.d_chi", "derivatives.dd_chi",
            "derivatives.d_gamma_tilde", "derivatives.dd_gamma_tilde",
            "derivatives.d_a_tilde", "derivatives.d_k_hat", "derivatives.d_theta",
        })

        # Inline the reference geometry through the same symbolic equations,
        # without the evolution-only Ricci-Z4, lapse Hessian or shift outputs.
        point = point_symbols()
        geometry = fccz4_geometry(point)
        constraints = fccz4_constraints(point, VACUUM, geometry)
        expressions = (constraints.hamiltonian, *constraints.momentum,
                       *constraints.z_covariant, constraints.theta,
                       constraints.log_determinant, constraints.trace_a,
                       *geometry.z_contravariant)
        reference_members = {
            str(symbol).split("[", 1)[0]
            for expression in expressions for symbol in expression.free_symbols
            if ".first[" in str(symbol) or ".second[" in str(symbol)
        }
        self.assertEqual(reference_members, {
            "state.metric.first", "state.metric.second", "state.chi.first",
            "state.chi.second", "state.a_tilde.first", "state.trace_k.first",
        })

    def assert_rhs(self, actual, expected):
        for name in ("chi", "metric", "a_tilde", "trace_k", "theta", "connection"):
            value = getattr(actual, name)
            reference = expected.get(name, sp.zeros(*value.shape) if isinstance(value, sp.MatrixBase) else 0)
            difference = value - reference
            if isinstance(difference, sp.MatrixBase):
                self.assertEqual(difference.applyfunc(sp.simplify), sp.zeros(*difference.shape), name)
            else:
                self.assertEqual(sp.simplify(difference), 0, name)

    def test_static_schwarzschild_and_homogeneous_matter(self):
        frame = spherical_frame()
        radius, theta, phi = frame.coordinates
        psi = 1 + 1 / (2 * radius)
        values = {"chi": psi**-4, "lapse": (1 - 1 / (2 * radius)) / psi}
        for radial_point in (sp.Rational(3, 4), sp.Integer(2), sp.Integer(10)):
            state = sample_state(frame, values, {radius: radial_point, theta: sp.pi / 3, phi: sp.pi / 5})
            self.assert_rhs(fccz4_rhs(state, VACUUM, DAMPING), {})
            constraints = fccz4_constraints(state, VACUUM)
            self.assertEqual(sp.simplify(constraints.hamiltonian), 0)
            self.assertEqual(constraints.momentum.applyfunc(sp.simplify), sp.zeros(3, 1))

        # Flat FLRW at a=1, H=1/5. Friedmann and continuity equations give
        # Kdot=12 pi (rho+p), with zero Z4 constraints for dust and radiation.
        hubble = sp.Rational(1, 5)
        energy = 3 * hubble**2 / (8 * sp.pi)
        for pressure in (0, energy / 3):
            state = sample_state(frame, {"trace_k": -3 * hubble},
                                 {radius: 2, theta: sp.pi / 3, phi: sp.pi / 5})
            matter = Fccz4Matter(energy, sp.zeros(3, 1), pressure * sp.eye(3))
            self.assertEqual(sp.simplify(fccz4_constraints(state, matter).hamiltonian), 0)
            self.assert_rhs(fccz4_rhs(state, matter, DAMPING),
                            {"chi": -2 * hubble, "trace_k": 12 * sp.pi * (energy + pressure)})

    def test_full_tensor_frame_covariance(self):
        # Spatially constant Cartesian tensors have nonzero raw spherical
        # derivatives. Their reference derivatives vanish; no numerical
        # differentiation or generated C++ supplies this independent oracle.
        cartesian = cartesian_frame()
        spherical = spherical_frame()
        r, theta, phi = spherical.coordinates
        jacobian = sp.Matrix(spherical.cartesian_position).jacobian(spherical.coordinates)
        rotation = sp.diag(*(1 / h for h in spherical.scale_factors)) * jacobian.T
        sample = {r: 2, theta: sp.pi / 3, phi: sp.pi / 4}
        q = rotation.subs(sample).applyfunc(sp.simplify)
        values = {
            "metric": sp.diag(2, 1, sp.Rational(1, 2)),
            "a_tilde": sp.Matrix([[1, 2, 3], [2, -1, 1], [3, 1, sp.Rational(1, 4)]]) / 10,
            "chi": sp.Rational(3, 4), "trace_k": sp.Rational(1, 3),
            "theta": sp.Rational(1, 20), "lapse": sp.Rational(4, 5),
            "shift": sp.Matrix([1, 2, 3]) / 10,
            "connection": sp.Matrix([2, -1, 3]) / 20,
        }
        transformed = dict(values)
        for name in ("metric", "a_tilde"):
            transformed[name] = rotation * values[name] * rotation.T
        for name in ("shift", "connection"):
            transformed[name] = rotation * values[name]
        first = sample_state(cartesian, values, dict.fromkeys(cartesian.coordinates, 0))
        second = sample_state(spherical, transformed, sample)
        matter = Fccz4Matter(sp.Rational(1, 10), sp.Matrix([1, -2, 1]) / 100,
                            sp.diag(1, 2, 3) / 100)
        rotated_matter = Fccz4Matter(matter.energy, q * matter.momentum, q * matter.stress * q.T)
        expected = fccz4_rhs(first, matter, DAMPING)
        self.assert_rhs(fccz4_rhs(second, rotated_matter, DAMPING), {
            "chi": expected.chi, "trace_k": expected.trace_k, "theta": expected.theta,
            "metric": q * expected.metric * q.T,
            "a_tilde": q * expected.a_tilde * q.T,
            "connection": q * expected.connection,
        })
        constraints = fccz4_constraints(second, rotated_matter)
        self.assertEqual((constraints.momentum + 8 * sp.pi * rotated_matter.momentum)
                         .applyfunc(sp.simplify), sp.zeros(3, 1))

    def test_gauge_wave_and_constraint_damping(self):
        frame = cartesian_frame()
        x, y, z = frame.coordinates
        time = sp.Symbol("time", real=True)
        h = 1 - sp.sin(2 * (x - time)) / 10
        lapse = sp.sqrt(h)
        chi = h**(-sp.Rational(1, 3))
        physical = sp.diag(h, 1, 1)
        extrinsic = -physical.diff(time) / (2 * lapse)
        trace_k = sp.trace(physical.inv() * extrinsic)
        values = {
            "chi": chi, "metric": chi * physical, "trace_k": trace_k, "lapse": lapse,
            "a_tilde": chi * (extrinsic - physical * trace_k / 3),
            "connection": sp.Matrix([-sp.diff(h**(-sp.Rational(2, 3)), x), 0, 0]),
        }
        point = {x: 0, y: 0, z: 0, time: 0}
        expected = {name: value.diff(time).subs(point) for name, value in values.items() if name != "lapse"}
        self.assert_rhs(fccz4_rhs(sample_state(frame, values, point), VACUUM, DAMPING), expected)

        theta = sp.Rational(1, 13)
        connection = sp.Matrix([1, -2, 3]) / 7
        state = sample_state(frame, {"theta": theta, "connection": connection}, point)
        self.assert_rhs(fccz4_rhs(state, VACUUM, DAMPING), {
            "trace_k": -3 * DAMPING.kappa1 * theta,
            "theta": -2 * DAMPING.kappa1 * theta,
            "connection": -DAMPING.kappa1 * connection,
        })


if __name__ == "__main__":
    unittest.main()
