"""Continuum coordinate contracts, independent of ID/evolution setup.

Cartesian embeddings and the chain rule are the independent oracle. Production
scale-factor operators must agree for arbitrary local field values/derivatives
(jets), not only selected physical solutions. These tests do not certify a
discretization, coordinate singularity, boundary closure, or nonlinear solve.
"""

from pathlib import Path
import sys
import unittest

import sympy as sp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "codegen"))
from pyfluid_codegen.coordinates import orthogonal_frames
from pyfluid_codegen.reference_metric import ReferenceMetricGeometry
from pyfluid_codegen.reference_metric_evolution import conformal_metric_rhs
from pyfluid_codegen.spacetime_geometry import (
    SymmetricTensorJet, coordinate_metric_jet, reference_covariant_metric_jet,
    reference_connection_expressions,
)
from pyfluid_codegen.tensor_algebra import (
    OrthogonalOperators, ScalarField, Tensor, Vector, VectorField,
)


def reduced(expression):
    return sp.trigsimp(sp.expand(expression))


def cartesian_embedding(frame):
    """Elementary geometric definitions, deliberately not scale-factor formulas."""
    first, second, third = frame.coordinates
    if frame.name == "cartesian":
        return sp.Matrix([first, second, third])
    if frame.name == "cylindrical":
        return sp.Matrix([first * sp.cos(second), first * sp.sin(second), third])
    if frame.name == "spherical":
        return sp.Matrix([
            first * sp.sin(second) * sp.cos(third),
            first * sp.sin(second) * sp.sin(third),
            first * sp.cos(second),
        ])
    raise AssertionError(
        f"new coordinate family needs an independent embedding: {frame.name}"
    )


def tensor(matrix):
    return Tensor.from_rows(tuple(tuple(row) for row in matrix.tolist()))


def coordinate_contract(frame):
    coordinates = frame.coordinates
    jacobian = cartesian_embedding(frame).jacobian(coordinates)
    scales = sp.diag(*frame.scale_factors)
    basis = jacobian * scales.inv()
    # Valid on a regular right-handed chart (r>0, 0<theta<pi in spherical).
    inverse_jacobian = scales.inv() * basis.T
    operators = OrthogonalOperators(frame.scale_factors, coordinates)

    scalar_gradient = sp.Matrix(sp.symbols("f0:3"))
    scalar_hessian = sp.Matrix(
        3, 3, lambda i, j: sp.Symbol(f"f{min(i, j)}{max(i, j)}")
    )
    scalar = ScalarField(
        sp.Symbol("f"), Vector(*scalar_gradient), tensor(scalar_hessian)
    )
    vector = sp.Matrix(sp.symbols("v0:3"))
    vector_derivatives = sp.Matrix(3, 3, lambda i, j: sp.Symbol(f"v{i}_{j}"))
    vector_field = VectorField(Vector(*vector), tensor(vector_derivatives))
    # d(E v)/dq includes moving-basis derivatives even on a symmetry-reduced
    # chart where some derivatives of the orthonormal components vanish.
    cartesian_vector_derivatives = sp.Matrix.hstack(*(
        basis.diff(axis) * vector + basis * vector_derivatives[:, direction]
        for direction, axis in enumerate(coordinates)
    )) * inverse_jacobian
    expected_vector_gradient = basis.T * cartesian_vector_derivatives.T * basis
    cartesian_curl = sp.Matrix([
        cartesian_vector_derivatives[2, 1] - cartesian_vector_derivatives[1, 2],
        cartesian_vector_derivatives[0, 2] - cartesian_vector_derivatives[2, 0],
        cartesian_vector_derivatives[1, 0] - cartesian_vector_derivatives[0, 1],
    ])
    expected_laplacian = sum(
        inverse_jacobian[b, j] * (
            scalar_hessian[a, b] * inverse_jacobian[a, j]
            + scalar_gradient[a] * inverse_jacobian[a, j].diff(coordinates[b])
        )
        for j in range(3) for a in range(3) for b in range(3)
    )

    tensor_values = sp.Matrix(3, 3, lambda i, j: sp.Symbol(f"t{i}{j}"))
    tensor_derivatives = [
        sp.Matrix(3, 3, lambda i, j: sp.Symbol(f"t{i}{j}_{axis}"))
        for axis in range(3)
    ]
    cartesian_tensor_derivatives = [
        basis.diff(axis) * tensor_values * basis.T
        + basis * tensor_derivatives[direction] * basis.T
        + basis * tensor_values * basis.T.diff(axis)
        for direction, axis in enumerate(coordinates)
    ]
    cartesian_tensor_divergence = sp.Matrix([
        sum(inverse_jacobian[a, j] * cartesian_tensor_derivatives[a][j, i]
            for a in range(3) for j in range(3))
        for i in range(3)
    ])
    volume = frame.volume_scale
    finite_volume_tensor_divergence = sp.Matrix([
        sum((volume / frame.scale_factors[a]).diff(coordinates[a])
            * tensor_values[a, i] / volume
            + tensor_derivatives[a][a, i] / frame.scale_factors[a]
            for a in range(3))
        for i in range(3)
    ])
    connection = sp.Matrix(
        operators.tensor_divergence_connection(tensor(tensor_values)).as_tuple()
    )
    reference_mixed_divergence = sp.Matrix(
        ReferenceMetricGeometry(frame).orthonormal_mixed_divergence_from_derivatives(
            tensor(tensor_values.T),
            tuple(tensor(derivative.T) for derivative in tensor_derivatives),
        ).as_tuple()
    )
    actual_vector_gradient = sp.Matrix(operators.gradient(vector_field).as_rows())
    residuals = {
        "metric": jacobian.T * jacobian - scales * scales,
        "orthonormal_basis": basis.T * basis - sp.eye(3),
        "orientation_volume": sp.Matrix([jacobian.det() - volume]),
        "scalar_gradient": sp.Matrix(operators.gradient(scalar).as_tuple())
            - basis.T * inverse_jacobian.T * scalar_gradient,
        "vector_gradient": actual_vector_gradient - expected_vector_gradient,
        "divergence": sp.Matrix([operators.divergence(vector_field)
                                  - sp.trace(cartesian_vector_derivatives)]),
        "curl": sp.Matrix(operators.curl(vector_field).as_tuple())
            - basis.T * cartesian_curl,
        "laplacian": sp.Matrix([operators.laplacian(scalar) - expected_laplacian]),
        "tensor_divergence": finite_volume_tensor_divergence + connection
            - basis.T * cartesian_tensor_divergence,
        # The independent Cartesian-chain-rule check above establishes the
        # physical tensor divergence.  The mixed Valencia convention stores
        # the transpose: flux[component, normal].  This second identity makes
        # that otherwise easy-to-hide index ordering explicit.
        "reference_mixed_divergence": reference_mixed_divergence
            - finite_volume_tensor_divergence - connection,
    }
    # Arbitrary jets include every choice of zero inactive derivatives. Do not
    # count substitution into an already-zero identity as extra layout tests:
    # executable axis mappings/storage are tested by dimensional consistency.
    return {name: value.applyfunc(reduced) for name, value in residuals.items()}


class CoordinateContractTests(unittest.TestCase):
    def test_conformal_kinematics_adm_and_frame_covariance(self):
        metric = sp.Matrix([[3, 1, 0], [1, 2, 1], [0, 1, 4]])
        derivatives = tuple(sp.Matrix([
            [d + 1, 2 - d, 1], [2 - d, d - 1, 2], [1, 2, 3 - d],
        ]) for d in range(3))
        zeros = tuple(tuple(sp.zeros(3) for _ in range(3)) for _ in range(3))
        jet = SymmetricTensorJet(metric, derivatives, zeros)
        a_tilde = sp.diag(1, 2, -1)
        a_tilde -= metric * sp.trace(metric.inv() * a_tilde) / 3
        chi, lapse, trace_k = sp.Rational(2, 3), sp.Rational(4, 5), sp.Rational(1, 7)
        d_chi, shift = sp.Matrix([1, -2, 3]), sp.Matrix([2, 3, -1])
        d_shift = sp.Matrix([[1, 2, 3], [-1, 4, 2], [2, 1, -3]])
        rhs = conformal_metric_rhs(jet, a_tilde, chi, d_chi, lapse, trace_k, shift, d_shift)
        self.assertEqual(sp.simplify(sp.trace(metric.inv() * rhs.metric)), 0)
        # Independent ADM metric equation: dt gamma = Lie_beta gamma - 2 alpha K.
        physical = metric / chi
        extrinsic = (a_tilde + metric * trace_k / 3) / chi
        lie_physical = d_shift * physical + physical * d_shift.T
        for k in range(3):
            lie_physical += shift[k] * (derivatives[k] / chi - metric * d_chi[k] / chi**2)
        actual = rhs.metric / chi - metric * rhs.chi / chi**2
        self.assertEqual((actual - lie_physical + 2 * lapse * extrinsic).applyfunc(sp.simplify), sp.zeros(3))

        # A proper rotation mixing all three axes. Reference-covariant derivative
        # indices transform too; raw coordinate-partial jets cannot use this rule.
        rotation = sp.Matrix([[1, -2, 2], [2, 2, 1], [-2, 1, 2]]) / 3
        self.assertEqual(rotation * rotation.T, sp.eye(3))
        rotated_first = tuple(sum(
            (rotation[d, k] * rotation * derivatives[k] * rotation.T for k in range(3)),
            sp.zeros(3),
        ) for d in range(3))
        rotated = conformal_metric_rhs(
            SymmetricTensorJet(rotation * metric * rotation.T, rotated_first, zeros),
            rotation * a_tilde * rotation.T, chi, rotation * d_chi, lapse, trace_k,
            rotation * shift, rotation * d_shift * rotation.T,
        )
        self.assertEqual(sp.simplify(rotated.chi - rhs.chi), 0)
        self.assertEqual((rotated.metric - rotation * rhs.metric * rotation.T).applyfunc(sp.simplify), sp.zeros(3))

    def test_rescaled_spacetime_metric_curvature(self):
        # An independent conformal-flat identity gives R=-8 psi^-5 Laplacian(psi).
        # Include nonzero curvature so an accidentally zero Ricci cannot pass.
        from pyfluid_codegen.metric_differential import metric_christoffel, metric_ricci_tensor

        for frame in orthogonal_frames():
            if frame.name == "cartesian":
                radius_squared = sum(q**2 for q in frame.coordinates)
            elif frame.name == "cylindrical":
                radius_squared = frame.coordinates[0]**2 + frame.coordinates[2]**2
            else:
                radius_squared = frame.coordinates[0]**2
            for psi, scalar_curvature in (
                (sp.Integer(1), 0),
                (1 / sp.sqrt(1 + radius_squared / 4), 6),
            ):
                rescaled = SymmetricTensorJet.differentiate(sp.eye(3) * psi**4, frame.coordinates)
                jet = coordinate_metric_jet(frame, rescaled)
                inverse = jet.value.inv()
                connection = metric_christoffel(inverse, list(jet.first))
                ricci = metric_ricci_tensor(inverse, list(jet.first),
                                     [list(row) for row in jet.second], connection)
                scalar = sum(inverse[i, j] * ricci[i, j] for i in range(3) for j in range(3))
                self.assertEqual(sp.trigsimp(sp.simplify(scalar)), scalar_curvature)
                if psi == 1:
                    reference = ReferenceMetricGeometry(frame)
                    for i in range(3):
                        for j in range(3):
                            self.assertEqual(sp.trigsimp(sp.simplify(ricci[i, j])), 0)
                            for k in range(3):
                                self.assertEqual(sp.simplify(connection[i][j][k]
                                                            - reference.christoffel(i, j, k)), 0)

    def test_rescaled_spacetime_metric_reduced_jets(self):
        frame = next(frame for frame in orthogonal_frames() if frame.name == "spherical")
        radius, theta, phi = frame.coordinates
        # r-theta and r-phi field reductions retain off-diagonal tensor entries
        # and all geometric theta derivatives. Direct differentiation is the
        # oracle for the supplied-jet product rule, including mixed derivatives.
        for active in ((radius,), (radius, theta), (radius, phi), (radius, theta, phi)):
            functions = [sp.Function(f"g{n}")(*active) for n in range(6)]
            a, b, c, d, e, f = functions
            rescaled = sp.Matrix([[a, b, c], [b, d, e], [c, e, f]])
            result = coordinate_metric_jet(frame, SymmetricTensorJet.differentiate(rescaled, frame.coordinates))
            scales = sp.diag(*frame.scale_factors)
            exact = SymmetricTensorJet.differentiate(scales * rescaled * scales, frame.coordinates)
            for actual, expected in zip(
                (result.value, *result.first, *(m for row in result.second for m in row)),
                (exact.value, *exact.first, *(m for row in exact.second for m in row)), strict=True,
            ):
                self.assertEqual((actual - expected).applyfunc(sp.expand), sp.zeros(3))

        # A scalar times the reference metric has covariant derivative
        # (D f) hat-g and Hessian (D D f) hat-g. In particular angular
        # Hessians do not vanish merely because f has radial dependence.
        field = radius**3
        result = reference_covariant_metric_jet(
            frame, SymmetricTensorJet.differentiate(field * sp.eye(3), frame.coordinates),
        )
        # For g=f(r) hat-g, Delta^i_jk = (delta^i_k d_j f +
        # delta^i_j d_k f - delta_jk d^i f)/(2f), in the unit basis.
        delta, contracted = reference_connection_expressions(result)
        for i in range(3):
            self.assertEqual(sp.simplify(contracted[i]), -3 / (2 * radius**4) if i == 0 else 0)
            for j in range(3):
                for k in range(3):
                    expected = sp.Rational(3, 2) / radius * (
                        int(i == k and j == 0) + int(i == j and k == 0) - int(j == k and i == 0)
                    )
                    self.assertEqual(sp.simplify(delta[i][j][k] - expected), 0)
        for d in range(3):
            expected_first = 3 * radius**2 if d == 0 else 0
            self.assertEqual(result.first[d], expected_first * sp.eye(3))
            for e in range(3):
                expected_second = ((6 if d == 0 else 3) * radius) if d == e else 0
                self.assertEqual(result.second[d][e], expected_second * sp.eye(3))

    def test_reference_derivatives_of_constant_cartesian_tensor(self):
        # All six components contribute after rotation; this catches connection
        # index/sign errors that isotropic metrics cannot detect.
        constant = sp.Matrix([[2, 1, 3], [1, 5, 2], [3, 2, 7]])
        for frame in orthogonal_frames():
            basis = cartesian_embedding(frame).jacobian(frame.coordinates) * sp.diag(
                *(1 / scale for scale in frame.scale_factors)
            )
            rescaled = (basis.T * constant * basis).applyfunc(sp.trigsimp)
            analytic = SymmetricTensorJet.differentiate(rescaled, frame.coordinates)
            point = dict(zip(frame.coordinates, (sp.Integer(2), sp.pi / 3, sp.pi / 4), strict=True))
            def evaluate(matrix):
                return matrix.subs(point).applyfunc(sp.simplify)
            evaluated = SymmetricTensorJet(
                evaluate(analytic.value), tuple(evaluate(m) for m in analytic.first),
                tuple(tuple(evaluate(m) for m in row) for row in analytic.second),
            )
            result = reference_covariant_metric_jet(frame, evaluated)
            for matrix in (*result.first, *(m for row in result.second for m in row)):
                self.assertEqual(matrix.subs(point).applyfunc(sp.simplify), sp.zeros(3))

    def assert_zero_matrix(self, matrix, label):
        for index, entry in enumerate(matrix):
            residual = reduced(entry)
            self.assertEqual(residual, 0, f"{label} entry={index}: {residual}")

    def test_continuum_operators_against_cartesian_chain_rule(self):
        for frame in orthogonal_frames():
            residuals = coordinate_contract(frame)
            for name, residual in residuals.items():
                with self.subTest(coordinate=frame.name, operator=name):
                    self.assert_zero_matrix(residual, f"{frame.name} {name}")

    def test_rigid_rotation_has_vorticity_but_no_strain(self):
        frame = next(
            frame for frame in orthogonal_frames() if frame.name == "cylindrical"
        )
        radius, _, _ = frame.coordinates
        omega = sp.Symbol("omega", nonzero=True)
        values = sp.Matrix([0, omega * radius, 0])
        field = VectorField(Vector(*values), tensor(values.jacobian(frame.coordinates)))
        operators = OrthogonalOperators(frame.scale_factors, frame.coordinates)
        expected = sp.Matrix([[0, omega, 0], [-omega, 0, 0], [0, 0, 0]])
        self.assert_zero_matrix(
            sp.Matrix(operators.gradient(field).as_rows()) - expected,
            "rigid rotation gradient",
        )
        self.assert_zero_matrix(
            sp.Matrix(operators.strain_rate(field).as_rows()), "rigid rotation strain"
        )
        self.assertEqual(operators.curl(field), Vector(0, 0, 2 * omega))

    def test_reference_metric_and_basis_roundtrip(self):
        upper = Vector(*sp.symbols("u0:3"))
        lower = Vector(*sp.symbols("l0:3"))
        mixed = Tensor.from_rows(tuple(
            tuple(sp.Symbol(f"a{component}{normal}") for normal in range(3))
            for component in range(3)
        ))
        for frame in orthogonal_frames():
            geometry = ReferenceMetricGeometry(frame)
            jacobian = cartesian_embedding(frame).jacobian(frame.coordinates)
            with self.subTest(coordinate=frame.name, quantity="metric"):
                self.assert_zero_matrix(
                    sp.Matrix(geometry.covariant_metric.as_rows())
                    - jacobian.T * jacobian,
                    f"{frame.name} reference metric",
                )
                self.assertEqual(
                    reduced(geometry.sqrt_determinant - frame.volume_scale), 0
                )
            with self.subTest(coordinate=frame.name, quantity="basis"):
                self.assertEqual(
                    geometry.coordinate_vector_to_orthonormal(
                        geometry.orthonormal_vector_to_coordinate(
                            upper, covariant=False
                        ),
                        covariant=False,
                    ),
                    upper,
                )
                self.assertEqual(
                    geometry.coordinate_vector_to_orthonormal(
                        geometry.orthonormal_vector_to_coordinate(
                            lower, covariant=True
                        ),
                        covariant=True,
                    ),
                    lower,
                )
                self.assertEqual(
                    geometry.coordinate_mixed_to_orthonormal(
                        geometry.orthonormal_mixed_to_coordinate(mixed)
                    ),
                    mixed,
                )

    def test_reference_christoffels(self):
        expected_nonzero = {
            "cartesian": {},
            "cylindrical": {
                (0, 1, 1): lambda r, _: -r,
                (1, 0, 1): lambda r, _: 1 / r,
                (1, 1, 0): lambda r, _: 1 / r,
            },
            "spherical": {
                (0, 1, 1): lambda r, theta: -r,
                (0, 2, 2): lambda r, theta: -r * sp.sin(theta) ** 2,
                (1, 0, 1): lambda r, theta: 1 / r,
                (1, 1, 0): lambda r, theta: 1 / r,
                (1, 2, 2): lambda r, theta: -sp.sin(theta) * sp.cos(theta),
                (2, 0, 2): lambda r, theta: 1 / r,
                (2, 2, 0): lambda r, theta: 1 / r,
                (2, 1, 2): lambda r, theta: 1 / sp.tan(theta),
                (2, 2, 1): lambda r, theta: 1 / sp.tan(theta),
            },
        }
        for frame in orthogonal_frames():
            geometry = ReferenceMetricGeometry(frame)
            radius = frame.coordinates[0]
            theta = frame.coordinates[1]
            expected = expected_nonzero[frame.name]
            for upper in range(3):
                for lower_first in range(3):
                    for lower_second in range(3):
                        index = (upper, lower_first, lower_second)
                        value = geometry.christoffel(*index)
                        target = expected[index](radius, theta) if index in expected else 0
                        with self.subTest(coordinate=frame.name, index=index):
                            self.assertEqual(reduced(value - target), 0)

    def test_reference_vector_divergence_matches_shared_operator(self):
        for frame in orthogonal_frames():
            coordinates = frame.coordinates
            components = Vector(*(
                sp.Function(f"{frame.name}_v{component}")(*coordinates)
                for component in range(3)
            ))
            derivatives = Tensor.from_rows(tuple(
                tuple(sp.diff(components.component(component), coordinate)
                      for coordinate in coordinates)
                for component in range(3)
            ))
            shared = OrthogonalOperators(
                frame.scale_factors, coordinates
            ).divergence(VectorField(components, derivatives))
            reference = ReferenceMetricGeometry(
                frame
            ).orthonormal_vector_divergence(components)
            with self.subTest(coordinate=frame.name):
                self.assertEqual(reduced(reference - shared), 0)

    def test_isotropic_pressure_is_analytically_well_balanced(self):
        for frame in orthogonal_frames():
            pressure = sp.Function(f"{frame.name}_pressure")(*frame.coordinates)
            flux = Tensor.identity(3, pressure)
            geometry = ReferenceMetricGeometry(frame)
            divergence = geometry.orthonormal_mixed_divergence(flux)
            transport = geometry.orthonormal_mixed_transport_divergence(flux)
            expected = Vector(*(
                sp.diff(pressure, frame.coordinates[component])
                / frame.scale_factors[component]
                for component in range(3)
            ))
            with self.subTest(coordinate=frame.name):
                self.assertEqual(divergence, expected)
                self.assertEqual(transport, expected)
                self.assertEqual(
                    geometry.orthonormal_mixed_geometry_divergence(flux),
                    Vector.zero(3),
                )

    def test_reference_basis_correction_uses_mixed_index_order(self):
        flux = Tensor.from_rows(tuple(
            tuple(sp.Symbol(f"f{component}{normal}") for normal in range(3))
            for component in range(3)
        ))
        for frame in orthogonal_frames():
            geometry = ReferenceMetricGeometry(frame)
            correction = geometry.orthonormal_mixed_basis_divergence(flux)
            legacy_transposed = OrthogonalOperators(
                frame.scale_factors, frame.coordinates
            ).tensor_divergence_connection(flux.transpose())
            with self.subTest(coordinate=frame.name):
                self.assertEqual(correction, legacy_transposed)

        spherical = ReferenceMetricGeometry(next(
            frame for frame in orthogonal_frames() if frame.name == "spherical"
        ))
        radius, theta, _ = spherical.coordinates
        pressure = sp.Symbol("pressure")
        pressure_correction = spherical.orthonormal_mixed_basis_divergence(
            Tensor.identity(3, pressure)
        )
        self.assertEqual(
            pressure_correction,
            Vector(-2 * pressure / radius, -pressure / (radius * sp.tan(theta)), 0),
        )
        self.assertEqual(
            spherical.orthonormal_mixed_basis_divergence(
                Tensor.identity(3, pressure) - Tensor.identity(3, pressure)
            ),
            Vector.zero(3),
        )

    def test_symmetry_reduction_retains_inactive_stress_geometry(self):
        frame = next(
            frame for frame in orthogonal_frames() if frame.name == "spherical"
        )
        radius = frame.coordinates[0]
        radial_flux = sp.Function("radial_flux")(radius)
        geometry = ReferenceMetricGeometry(frame)
        self.assertEqual(
            reduced(
                geometry.orthonormal_vector_divergence(
                    Vector(radial_flux, 0, 0)
                )
                - sp.diff(radial_flux, radius)
                - 2 * radial_flux / radius
            ),
            0,
        )
        pressure = sp.Function("spherical_pressure")(radius)
        self.assertEqual(
            geometry.orthonormal_mixed_divergence(
                Tensor.identity(3, pressure)
            ),
            Vector(sp.diff(pressure, radius), 0, 0),
        )


if __name__ == "__main__":
    unittest.main()
