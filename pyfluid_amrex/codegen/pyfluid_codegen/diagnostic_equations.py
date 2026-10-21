from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from .kernel_equations import StructKernel
from .tensor_algebra import Vector, symmetric_tensor3_symbols
from .z4c_equations import (
   DIMENSION,
   Z4cPhysicalGeometryExpressions,
   z4c_physical_christoffel,
   z4c_physical_geometry_expressions,
)


@dataclass(frozen=True)
class DiagnosticEquationRecord:
   identifier: str
   title: str
   latex: str


DIAGNOSTIC_EQUATION_RECORDS = (
   DiagnosticEquationRecord(
      "rest-mass",
      "Total rest mass",
      r"M_0 = \int_\Omega D\,d^3x",
   ),
   DiagnosticEquationRecord(
      "baryon-flux",
      "Valencia baryon flux through a spatial surface",
      r"\mathcal F_0^i=D(\alpha v^i-\beta^i),\qquad"
      r"\dot M_{0,\rm in}=-\oint_S\mathcal F_0^i\,dS_i",
   ),
   DiagnosticEquationRecord(
      "adm-mass-surface",
      "ADM mass surface integral",
      r"M_{\rm ADM}^{S}=\frac{1}{16\pi}\oint_{\partial\Omega}"
      r"\sqrt{\gamma}\,\gamma^{ij}\gamma^{kl}"
      r"(\partial_j\gamma_{ik}-\partial_k\gamma_{ij})\,dS_l",
   ),
   DiagnosticEquationRecord(
      "adm-mass-volume",
      "Constraint-assisted ADM mass volume integral",
      r"M_{\rm ADM}^{V}=\int_\Omega\sqrt{\gamma}\left[\rho+\frac{1}{16\pi}"
      r"\left(K_{ij}K^{ij}-K^2-\mathcal G\right)\right]d^3x,\qquad"
      r"\mathcal G=\gamma^{ij}\left(\Gamma^k{}_{i\ell}\Gamma^\ell{}_{jk}"
      r"-\Gamma^k{}_{ij}\Gamma^\ell{}_{k\ell}\right)",
   ),
   DiagnosticEquationRecord(
      "adm-linear-momentum-surface",
      "ADM linear momentum surface integral",
      r"P_i^{S}=\frac{1}{8\pi}\oint_{\partial\Omega}"
      r"\sqrt{\gamma}(K^l{}_i-\delta^l{}_iK)\,dS_l",
   ),
   DiagnosticEquationRecord(
      "adm-linear-momentum-volume",
      "Constraint-assisted ADM linear momentum volume integral",
      r"P_a^{V}=\int_\Omega\sqrt{\gamma}\left[S_i\xi_a^i+\frac{1}{16\pi}"
      r"(K^{ij}-\gamma^{ij}K)\mathcal L_{\xi_a}\gamma_{ij}\right]d^3x,"
      r"\qquad \xi_a^i=\delta_a^i",
   ),
   DiagnosticEquationRecord(
      "adm-angular-momentum-surface",
      "ADM angular momentum surface integral",
      r"J_i^{S}=\frac{1}{8\pi}\oint_{\partial\Omega}"
      r"\sqrt{\gamma}(K^l{}_k-\delta^l{}_kK)\phi_i^k\,dS_l,\qquad"
      r"\phi_i^k=\epsilon_{ijk}x^j",
   ),
   DiagnosticEquationRecord(
      "adm-angular-momentum-volume",
      "Constraint-assisted ADM angular momentum volume integral",
      r"J_a^{V}=\int_\Omega\sqrt{\gamma}\left[S_i\phi_a^i+\frac{1}{16\pi}"
      r"(K^{ij}-\gamma^{ij}K)\mathcal L_{\phi_a}\gamma_{ij}\right]d^3x,"
      r"\qquad \phi_a^i=\epsilon_{aji}x^j",
   ),
   DiagnosticEquationRecord(
      "proper-mass-volume",
      "Proper matter-energy volume integral",
      r"M_{\rm proper}^{V}=\int_\Omega\sqrt{\gamma}\,E\,d^3x",
   ),
   DiagnosticEquationRecord(
      "komar-mass-surface",
      "Komar mass surface integral",
      r"M_{\rm K}^{S}=\frac{1}{4\pi}\oint_{\partial\Omega}"
      r"\left(D^i\alpha-K^i{}_j\beta^j\right)dS_i",
   ),
   DiagnosticEquationRecord(
      "komar-mass-volume",
      "Stationary Komar mass volume integral",
      r"M_{\rm K}^{V}=\int_\Omega\sqrt{\gamma}"
      r"\left[\alpha(E+S)-2\beta^iS_i\right]d^3x,\qquad"
      r"S=\gamma^{ij}S_{ij}",
   ),
   DiagnosticEquationRecord(
      "hamiltonian-norms",
      "Hamiltonian constraint norms",
      r"\|\mathcal H\|_2=\sqrt{V^{-1}\int_\Omega\mathcal H^2dV},\qquad"
      r"\|\mathcal H\|_\infty=\max_\Omega|\mathcal H|",
   ),
   DiagnosticEquationRecord(
      "rmhd-runtime-invariants",
      "Relativistic magnetofluid invariants",
      r"b^2=b^\mu b_\mu,\qquad \sigma_h=\frac{b^2}{\rho h},\qquad"
      r"\beta_{\rm plasma}=\frac{2P}{b^2},\qquad W=(1-v_iv^i)^{-1/2}",
   ),
   DiagnosticEquationRecord(
      "rmhd-electromagnetic-energy",
      "Eulerian electromagnetic energy",
      r"E_{\rm EM}=\int_\Omega\sqrt{\gamma}\,n_\mu n_\nu"
      r"\left(T_{\rm RMHD}^{\mu\nu}-T_{\rm RHD}^{\mu\nu}\right)d^3x",
   ),
   DiagnosticEquationRecord(
      "weyl-radiation",
      "Electric and magnetic Weyl radiation",
      r"E_{ij}=\left[R_{ij}+KK_{ij}-K_{ik}K^k{}_j"
      r"-4\pi\left(S_{ij}+\gamma_{ij}(E-S)\right)\right]^{\rm TF},\qquad"
      r"B_{ij}=\left[\epsilon_i{}^{kl}D_kK_{lj}\right]^{\rm STF},\qquad"
      r"\Psi_4=(E_{ij}-iB_{ij})\bar m^i\bar m^j",
   ),
   DiagnosticEquationRecord(
      "matter-energy-flux",
      "Finite-radius matter energy flux",
      r"\mathcal F_{\rm matter}=\oint \alpha\sqrt{\gamma}\,J^i_{(t)}dS_i,\qquad"
      r"J^i_{(t)}=-E\beta^i+\alpha S^i+\frac{\beta^i}{\alpha}S_j\beta^j"
      r"-S^i{}_j\beta^j",
   ),
)


def conserved_diagnostic_integrands_kernel() -> StructKernel:
   symbols = sp.symbols(
      "rho mom1 mom2 mom3 energy b1 b2 b3 x1 x2 x3 "
      "gravity_phi gravity_density_offset gravity_energy_factor passive_density divb"
   )
   (
      rho, mom1, mom2, mom3, energy, b1, b2, b3, x1, x2, x3,
      gravity_phi, gravity_density_offset, gravity_energy_factor,
      passive_density, divb,
   ) = symbols
   momentum = Vector(mom1, mom2, mom3)
   magnetic = Vector(b1, b2, b3)
   position = Vector(x1, x2, x3)
   angular_momentum = position.cross(momentum)
   return StructKernel(
      name="conserved_diagnostic_integrands",
      return_type="ConservedDiagnosticIntegrands",
      arguments=tuple(("amrex::Real", str(symbol)) for symbol in symbols),
      outputs=(
         "mass", "momentum1", "momentum2", "momentum3", "energy",
         "angular_momentum1", "angular_momentum2", "angular_momentum3",
         "magnetic_energy", "gravity_potential_energy", "passive_mass", "divb",
      ),
      expressions=(
         rho,
         *momentum,
         energy,
         *angular_momentum,
         sp.Rational(1, 2) * magnetic.norm_squared(),
         gravity_energy_factor * (rho - gravity_density_offset) * gravity_phi,
         passive_density,
         sp.Abs(divb),
      ),
   )


def baryon_flux_integrands_kernel() -> StructKernel:
   densitized_density = sp.Symbol("densitized_density")
   velocity = Vector(*(
      sp.Symbol(f"velocity{component + 1}")
      for component in range(DIMENSION)
   ))
   shift = Vector(*(
      sp.Symbol(f"metric.shift{component + 1}")
      for component in range(DIMENSION)
   ))
   lapse = sp.Symbol("metric.lapse", positive=True)
   flux = densitized_density * (lapse * velocity - shift)
   return StructKernel(
      name="baryon_flux_integrands",
      return_type="BaryonFluxIntegrands",
      arguments=(
         ("amrex::Real", "densitized_density"),
         ("amrex::Real", "velocity1"),
         ("amrex::Real", "velocity2"),
         ("amrex::Real", "velocity3"),
         ("const SpacetimeMetric&", "metric"),
      ),
      outputs=("flux1", "flux2", "flux3"),
      expressions=tuple(flux),
   )


def _levi_civita(first: int, second: int, third: int) -> int:
   return int(sp.LeviCivita(first, second, third))


def _adm_mass_flux(geometry: Z4cPhysicalGeometryExpressions) -> Vector:
   sqrt_det_gamma = sp.sqrt(geometry.gamma.det())
   return Vector(*(
      sqrt_det_gamma * sum(
         geometry.inverse_gamma[first, second]
         * geometry.inverse_gamma[third, surface]
         * (
            geometry.d_gamma[second][first, third]
            - geometry.d_gamma[third][first, second]
         )
         for first in range(DIMENSION)
         for second in range(DIMENSION)
         for third in range(DIMENSION)
      ) / (16 * sp.pi)
      for surface in range(DIMENSION)
   ))


def _rotation_vector(axis: int, position: Vector) -> Vector:
   return Vector(*(
      sum(
         _levi_civita(axis, position_axis, component)
         * position.component(position_axis)
         for position_axis in range(DIMENSION)
      )
      for component in range(DIMENSION)
   ))


def _translation_vector(axis: int) -> Vector:
   return Vector(*(sp.Integer(axis == component) for component in range(DIMENSION)))


def _lie_metric(
   vector: Vector,
   vector_derivative: tuple[tuple[sp.Expr, ...], ...],
   geometry: Z4cPhysicalGeometryExpressions,
) -> sp.Matrix:
   return sp.Matrix(DIMENSION, DIMENSION, lambda first, second: sum(
      vector.component(direction) * geometry.d_gamma[direction][first, second]
      + geometry.gamma[direction, second] * vector_derivative[direction][first]
      + geometry.gamma[first, direction] * vector_derivative[direction][second]
      for direction in range(DIMENSION)
   ))


def adm_flux_integrands_kernel() -> StructKernel:
   geometry = z4c_physical_geometry_expressions()
   position = Vector(*(sp.Symbol(f"position.{axis}") for axis in ("x", "y", "z")))
   mass_flux = _adm_mass_flux(geometry)
   lapse_gradient = Vector(*(
      sp.Symbol(f"derivatives.d_lapse[{direction}]")
      for direction in range(DIMENSION)
   ))
   shift = Vector(*(
      sp.Symbol(f"state.shift{component + 1}")
      for component in range(DIMENSION)
   ))
   raised_lapse_gradient = Vector(*(
      sum(
         geometry.inverse_gamma[upper, lower] * lapse_gradient.component(lower)
         for lower in range(DIMENSION)
      )
      for upper in range(DIMENSION)
   ))
   raised_curvature_shift = Vector(*(
      sum(
         geometry.inverse_gamma[upper, first]
         * geometry.extrinsic_curvature[first, second]
         * shift.component(second)
         for first in range(DIMENSION)
         for second in range(DIMENSION)
      )
      for upper in range(DIMENSION)
   ))
   sqrt_det_gamma = sp.sqrt(geometry.gamma.det())
   komar_mass_flux = sqrt_det_gamma * (
      raised_lapse_gradient - raised_curvature_shift
   ) / (4 * sp.pi)
   mixed_curvature = geometry.inverse_gamma * geometry.extrinsic_curvature
   trace_adjusted = mixed_curvature - sp.eye(DIMENSION) * geometry.trace_extrinsic_curvature
   linear_momentum_fluxes = tuple(
      Vector(*(
         sqrt_det_gamma * trace_adjusted[surface_axis, axis] / (8 * sp.pi)
         for surface_axis in range(DIMENSION)
      ))
      for axis in range(DIMENSION)
   )
   angular_momentum_fluxes = tuple(
      Vector(*(
         sqrt_det_gamma
         * sum(
            _rotation_vector(axis, position).component(curvature_axis)
            * trace_adjusted[surface_axis, curvature_axis]
            for curvature_axis in range(DIMENSION)
         )
         / (8 * sp.pi)
         for surface_axis in range(DIMENSION)
      ))
      for axis in range(DIMENSION)
   )
   return StructKernel(
      name="adm_flux_integrands",
      return_type="AdmFluxIntegrands",
      arguments=(
         ("const Z4cState&", "state"),
         ("const Z4cSpatialDerivatives&", "derivatives"),
         ("const Vector3&", "position"),
      ),
      outputs=(
         "mass1", "mass2", "mass3",
         "komar_mass1", "komar_mass2", "komar_mass3",
         "linear_momentum11", "linear_momentum12", "linear_momentum13",
         "linear_momentum21", "linear_momentum22", "linear_momentum23",
         "linear_momentum31", "linear_momentum32", "linear_momentum33",
         "angular_momentum11", "angular_momentum12", "angular_momentum13",
         "angular_momentum21", "angular_momentum22", "angular_momentum23",
         "angular_momentum31", "angular_momentum32", "angular_momentum33",
      ),
      expressions=(
         *mass_flux,
         *komar_mass_flux,
         *(component for flux in linear_momentum_fluxes for component in flux),
         *(component for flux in angular_momentum_fluxes for component in flux),
      ),
   )


def adm_constraint_volume_integrands_kernel() -> StructKernel:
   geometry = z4c_physical_geometry_expressions()
   christoffel = z4c_physical_christoffel(geometry)
   position = Vector(*(sp.Symbol(f"position.{axis}") for axis in ("x", "y", "z")))
   matter_density = sp.Symbol("matter.rho")
   matter_momentum = Vector(*(
      sp.Symbol(f"matter.momentum{component + 1}")
      for component in range(DIMENSION)
   ))
   matter_stress = sp.Matrix((
      (
         sp.Symbol("matter.stress11"),
         sp.Symbol("matter.stress12"),
         sp.Symbol("matter.stress13"),
      ),
      (
         sp.Symbol("matter.stress12"),
         sp.Symbol("matter.stress22"),
         sp.Symbol("matter.stress23"),
      ),
      (
         sp.Symbol("matter.stress13"),
         sp.Symbol("matter.stress23"),
         sp.Symbol("matter.stress33"),
      ),
   ))
   shift = Vector(*(
      sp.Symbol(f"state.shift{component + 1}")
      for component in range(DIMENSION)
   ))

   gamma_gamma = sum(
      geometry.inverse_gamma[first, second]
      * (
         christoffel[upper][first][contracted]
         * christoffel[contracted][second][upper]
         - christoffel[upper][first][second]
         * christoffel[contracted][upper][contracted]
      )
      for first in range(DIMENSION)
      for second in range(DIMENSION)
      for upper in range(DIMENSION)
      for contracted in range(DIMENSION)
   )


   curvature_up = (
      geometry.inverse_gamma
      * geometry.extrinsic_curvature
      * geometry.inverse_gamma
   )
   curvature_squared = sum(
      geometry.extrinsic_curvature[first, second] * curvature_up[first, second]
      for first in range(DIMENSION)
      for second in range(DIMENSION)
   )
   trace_adjusted_up = (
      curvature_up
      - geometry.inverse_gamma * geometry.trace_extrinsic_curvature
   )
   sqrt_det_gamma = sp.sqrt(geometry.gamma.det())
   mass = sqrt_det_gamma * (
      matter_density
      + (
         curvature_squared
         - geometry.trace_extrinsic_curvature**2
         - gamma_gamma
      ) / (16 * sp.pi)
   )
   proper_mass = sqrt_det_gamma * matter_density
   stress_trace = sum(
      geometry.inverse_gamma[first, second] * matter_stress[first, second]
      for first in range(DIMENSION)
      for second in range(DIMENSION)
   )
   komar_mass = sqrt_det_gamma * (
      sp.Symbol("state.lapse") * (matter_density + stress_trace)
      - 2 * shift.dot(matter_momentum)
   )

   linear_momentum = []
   angular_momentum = []
   for axis in range(DIMENSION):
      translation = _translation_vector(axis)
      zero_derivative = tuple(
         tuple(sp.Integer(0) for _ in range(DIMENSION))
         for _ in range(DIMENSION)
      )
      translation_lie_metric = _lie_metric(
         translation, zero_derivative, geometry
      )
      linear_momentum.append(
         sqrt_det_gamma * (
            translation.dot(matter_momentum)
            + sum(
               trace_adjusted_up[first, second]
               * translation_lie_metric[first, second]
               for first in range(DIMENSION)
               for second in range(DIMENSION)
            ) / (16 * sp.pi)
         )
      )
      rotation = _rotation_vector(axis, position)
      rotation_derivative = tuple(
         tuple(_levi_civita(axis, derivative, component) for derivative in range(DIMENSION))
         for component in range(DIMENSION)
      )
      lie_metric = _lie_metric(rotation, rotation_derivative, geometry)
      angular_momentum.append(
         sqrt_det_gamma * (
            rotation.dot(matter_momentum)
            + sum(
               trace_adjusted_up[first, second] * lie_metric[first, second]
               for first in range(DIMENSION)
               for second in range(DIMENSION)
            ) / (16 * sp.pi)
         )
      )

   return StructKernel(
      name="adm_constraint_volume_integrands",
      return_type="AdmVolumeIntegrands",
      arguments=(
         ("const Z4cState&", "state"),
         ("const Z4cSpatialDerivatives&", "derivatives"),
         ("const Z4cMatter&", "matter"),
         ("const Vector3&", "position"),
      ),
      outputs=(
         "mass",
         "proper_mass",
         "komar_mass",
         "linear_momentum1",
         "linear_momentum2",
         "linear_momentum3",
         "angular_momentum1",
         "angular_momentum2",
         "angular_momentum3",
      ),
      expressions=(
         mass, proper_mass, komar_mass, *linear_momentum, *angular_momentum
      ),
   )


def _physical_trace_free(
   tensor: sp.Matrix,
   metric: sp.Matrix,
   inverse_metric: sp.Matrix,
) -> sp.Matrix:
   trace = sum(
      inverse_metric[first, second] * tensor[first, second]
      for first in range(DIMENSION)
      for second in range(DIMENSION)
   )
   return tensor - metric * trace / DIMENSION


def _physical_christoffel_argument() -> list[list[list[sp.Expr]]]:
   return [
      [
         [
            sp.Symbol(
               f"connections.christoffel{upper + 1}_"
               f"{min(first, second) + 1}{max(first, second) + 1}"
            )
            for second in range(DIMENSION)
         ]
         for first in range(DIMENSION)
      ]
      for upper in range(DIMENSION)
   ]


def weyl_radiation_integrands_kernel() -> StructKernel:
   geometry = z4c_physical_geometry_expressions()
   metric = symmetric_tensor3_symbols("metric.gamma")
   inverse_metric = symmetric_tensor3_symbols("metric.inv_gamma")
   ricci = symmetric_tensor3_symbols("diagnostic_geometry.ricci")
   stress = symmetric_tensor3_symbols("matter.stress")
   stress_trace = sum(
      inverse_metric[first, second] * stress[first, second]
      for first in range(DIMENSION)
      for second in range(DIMENSION)
   )
   curvature = geometry.extrinsic_curvature
   electric_raw = (
      ricci
      + geometry.trace_extrinsic_curvature * curvature
      - curvature * inverse_metric * curvature
      - 4 * sp.pi * (
         stress + metric * (sp.Symbol("matter.rho") - stress_trace)
      )
   )
   electric_symmetric = (electric_raw + electric_raw.transpose()) / 2
   electric = _physical_trace_free(electric_symmetric, metric, inverse_metric)

   christoffel = _physical_christoffel_argument()
   covariant_curvature_derivative = [
      sp.Matrix(
         DIMENSION,
         DIMENSION,
         lambda first, second: geometry.d_extrinsic_curvature[direction][first, second]
         - sum(
            christoffel[contracted][direction][first] * curvature[contracted, second]
            + christoffel[contracted][direction][second] * curvature[first, contracted]
            for contracted in range(DIMENSION)
         ),
      )
      for direction in range(DIMENSION)
   ]
   sqrt_det_gamma = sp.Symbol("metric.sqrt_det_gamma", positive=True)
   magnetic_raw = sp.Matrix(
      DIMENSION,
      DIMENSION,
      lambda first, second: sum(
         metric[first, lowered]
         * _levi_civita(lowered, derivative, curvature_axis)
         * covariant_curvature_derivative[derivative][curvature_axis, second]
         / sqrt_det_gamma
         for lowered in range(DIMENSION)
         for derivative in range(DIMENSION)
         for curvature_axis in range(DIMENSION)
      ),
   )
   magnetic_symmetric = (magnetic_raw + magnetic_raw.transpose()) / 2
   magnetic = _physical_trace_free(magnetic_symmetric, metric, inverse_metric)

   symmetric_components = ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))
   return StructKernel(
      name="weyl_radiation_integrands",
      return_type="WeylRadiationIntegrands",
      arguments=(
         ("const Z4cState&", "state"),
         ("const Z4cSpatialDerivatives&", "derivatives"),
         ("const Z4cConformalMetric&", "metric"),
         ("const Z4cConnections&", "connections"),
         ("const Z4cDiagnosticGeometry&", "diagnostic_geometry"),
         ("const Z4cMatter&", "matter"),
      ),
      outputs=(
         "electric11", "electric12", "electric13", "electric22", "electric23", "electric33",
         "magnetic11", "magnetic12", "magnetic13", "magnetic22", "magnetic23", "magnetic33",
      ),
      expressions=(
         *(electric[first, second] for first, second in symmetric_components),
         *(magnetic[first, second] for first, second in symmetric_components),
      ),
   )


def psi4_from_weyl_kernel() -> StructKernel:
   metric = symmetric_tensor3_symbols("metric.gamma")
   inverse_metric = symmetric_tensor3_symbols("metric.inv_gamma")
   electric = symmetric_tensor3_symbols("weyl.electric")
   magnetic = symmetric_tensor3_symbols("weyl.magnetic")
   sqrt_det_gamma = sp.Symbol("metric.sqrt_det_gamma", positive=True)
   bindings: list[tuple[sp.Symbol, sp.Expr]] = []

   def bind(name: str, expression: sp.Expr) -> sp.Symbol:
      symbol = sp.Symbol(name)
      bindings.append((symbol, expression))
      return symbol

   def bind_vector(name: str, value: Vector) -> Vector:
      return Vector(*(
         bind(f"{name}{component + 1}", value.component(component))
         for component in range(DIMENSION)
      ))

   position = Vector(*(sp.Symbol(f"position.{axis}") for axis in ("x", "y", "z")))
   azimuthal = Vector(-position.component(1), position.component(0), 0)
   radial = position
   polar_covector = azimuthal.cross(radial)
   polar = bind_vector("polar", Vector(*(
      sqrt_det_gamma * sum(
         inverse_metric[upper, lower] * polar_covector.component(lower)
         for lower in range(DIMENSION)
      )
      for upper in range(DIMENSION)
   )))

   def metric_dot(left: Vector, right: Vector) -> sp.Expr:
      return sum(
         metric[first, second] * left.component(first) * right.component(second)
         for first in range(DIMENSION)
         for second in range(DIMENSION)
      )

   # Keep Gram-Schmidt stages explicit. Expanding each normalization into
   # both Weyl contractions makes algebraic CSE revisit the same large trees.
   # Ordered bindings also preserve each normalization as a numerical step.
   azimuthal_norm = bind("azimuthal_norm", sp.sqrt(metric_dot(azimuthal, azimuthal)))
   azimuthal_unit = bind_vector("azimuthal_unit", azimuthal / azimuthal_norm)
   radial_projection = bind("radial_projection", metric_dot(azimuthal_unit, radial))
   radial_orthogonal = bind_vector(
      "radial_orthogonal", radial - azimuthal_unit * radial_projection
   )
   radial_norm = bind(
      "radial_norm", sp.sqrt(metric_dot(radial_orthogonal, radial_orthogonal))
   )
   radial_unit = bind_vector("radial_unit", radial_orthogonal / radial_norm)
   polar_azimuthal_projection = bind(
      "polar_azimuthal_projection", metric_dot(azimuthal_unit, polar)
   )
   polar_radial_projection = bind(
      "polar_radial_projection", metric_dot(radial_unit, polar)
   )
   polar_orthogonal = bind_vector("polar_orthogonal", (
      polar
      - azimuthal_unit * polar_azimuthal_projection
      - radial_unit * polar_radial_projection
   ))
   polar_norm = bind(
      "polar_norm", sp.sqrt(metric_dot(polar_orthogonal, polar_orthogonal))
   )
   polar_unit = bind_vector("polar_unit", polar_orthogonal / polar_norm)

   def contract(tensor: sp.Matrix, left: Vector, right: Vector) -> sp.Expr:
      return sum(
         tensor[first, second] * left.component(first) * right.component(second)
         for first in range(DIMENSION)
         for second in range(DIMENSION)
      )

   electric_polar = contract(electric, polar_unit, polar_unit)
   electric_azimuthal = contract(electric, azimuthal_unit, azimuthal_unit)
   electric_cross = contract(electric, polar_unit, azimuthal_unit)
   magnetic_polar = contract(magnetic, polar_unit, polar_unit)
   magnetic_azimuthal = contract(magnetic, azimuthal_unit, azimuthal_unit)
   magnetic_cross = contract(magnetic, polar_unit, azimuthal_unit)
   psi4_real = (electric_polar - electric_azimuthal) / 2 - magnetic_cross
   psi4_imaginary = -electric_cross - (magnetic_polar - magnetic_azimuthal) / 2
   return StructKernel(
      name="psi4_from_weyl",
      return_type="Psi4Integrands",
      arguments=(
         ("const Z4cConformalMetric&", "metric"),
         ("const WeylRadiationIntegrands&", "weyl"),
         ("const Vector3&", "position"),
      ),
      outputs=("psi4_real", "psi4_imaginary"),
      expressions=(psi4_real, psi4_imaginary),
      bindings=tuple(bindings),
   )


def matter_energy_flux_integrands_kernel() -> StructKernel:
   inverse_metric = symmetric_tensor3_symbols("metric.inv_gamma")
   stress = symmetric_tensor3_symbols("matter.stress")
   momentum = Vector(*(
      sp.Symbol(f"matter.momentum{component + 1}")
      for component in range(DIMENSION)
   ))
   shift = Vector(*(
      sp.Symbol(f"state.shift{component + 1}")
      for component in range(DIMENSION)
   ))
   raised_momentum = Vector(*(
      sum(
         inverse_metric[upper, lower] * momentum.component(lower)
         for lower in range(DIMENSION)
      )
      for upper in range(DIMENSION)
   ))
   mixed_stress = inverse_metric * stress
   lapse = sp.Symbol("state.lapse", positive=True)
   momentum_shift = momentum.dot(shift)
   current = Vector(*(
      -sp.Symbol("matter.rho") * shift.component(upper)
      + lapse * raised_momentum.component(upper)
      + shift.component(upper) * momentum_shift / lapse
      - sum(
         mixed_stress[upper, lower] * shift.component(lower)
         for lower in range(DIMENSION)
      )
      for upper in range(DIMENSION)
   ))
   densitized_flux = current * (
      lapse * sp.Symbol("metric.sqrt_det_gamma", positive=True)
   )
   return StructKernel(
      name="matter_energy_flux_integrands",
      return_type="MatterEnergyFluxIntegrands",
      arguments=(
         ("const Z4cState&", "state"),
         ("const Z4cConformalMetric&", "metric"),
         ("const Z4cMatter&", "matter"),
      ),
      outputs=("flux1", "flux2", "flux3"),
      expressions=tuple(densitized_flux),
   )


def diagnostic_kernels() -> tuple[StructKernel, ...]:
   return (
      conserved_diagnostic_integrands_kernel(),
      baryon_flux_integrands_kernel(),
      adm_flux_integrands_kernel(),
      adm_constraint_volume_integrands_kernel(),
      weyl_radiation_integrands_kernel(),
      psi4_from_weyl_kernel(),
      matter_energy_flux_integrands_kernel(),
   )
