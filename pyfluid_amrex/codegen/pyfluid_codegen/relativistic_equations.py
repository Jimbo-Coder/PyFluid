from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from .eos_equations import ideal_gas_equation_of_state
from .field_catalog import (
   RELATIVISTIC_HYDRO_COMPONENT_NAMES,
   RELATIVISTIC_MHD_COMPONENT_NAMES,
)
from .kernel_equations import ScalarKernel, StructKernel, differentiate_with_bindings
from .physical_system import (
   ValenciaHydrodynamics,
   ValenciaMagnetohydrodynamics,
   ValenciaMHDPrimitive,
   ValenciaMetric,
   ValenciaMetricDerivatives,
   ValenciaPrimitive,
   eulerian_matter_projection,
   valencia_geometric_source,
)
from .tensor_algebra import (
   IndexPosition,
   MetricTensor,
   Tensor,
   Tensor3,
   Vector,
   Vector3,
)


def _symbols() -> tuple[ValenciaPrimitive, ValenciaMetric]:
   primitive = ValenciaPrimitive(
      density=sp.Symbol("primitive.rho", positive=True),
      specific_internal_energy=sp.Symbol("primitive.eps", nonnegative=True),
      pressure=sp.Symbol("primitive.pressure", positive=True),
      velocity=Vector(*sp.symbols("primitive.v1 primitive.v2 primitive.v3")),
   )
   covariant = Tensor.from_rows((
      (sp.Symbol("metric.gamma11"), sp.Symbol("metric.gamma12"), sp.Symbol("metric.gamma13")),
      (sp.Symbol("metric.gamma12"), sp.Symbol("metric.gamma22"), sp.Symbol("metric.gamma23")),
      (sp.Symbol("metric.gamma13"), sp.Symbol("metric.gamma23"), sp.Symbol("metric.gamma33")),
   ))
   contravariant = Tensor.from_rows((
      (sp.Symbol("metric.inv_gamma11"), sp.Symbol("metric.inv_gamma12"), sp.Symbol("metric.inv_gamma13")),
      (sp.Symbol("metric.inv_gamma12"), sp.Symbol("metric.inv_gamma22"), sp.Symbol("metric.inv_gamma23")),
      (sp.Symbol("metric.inv_gamma13"), sp.Symbol("metric.inv_gamma23"), sp.Symbol("metric.inv_gamma33")),
   ))
   metric = ValenciaMetric(
      lapse=sp.Symbol("metric.lapse", positive=True),
      shift=Vector(*sp.symbols("metric.shift1 metric.shift2 metric.shift3")),
      spatial_metric=MetricTensor(covariant, contravariant),
      sqrt_spatial_determinant=sp.Symbol("metric.sqrt_det_gamma", positive=True),
   )
   return primitive, metric


def _rmhd_symbols() -> tuple[ValenciaMHDPrimitive, ValenciaMetric]:
   hydrodynamic, metric = _symbols()
   return ValenciaMHDPrimitive(
      density=hydrodynamic.density,
      specific_internal_energy=hydrodynamic.specific_internal_energy,
      pressure=hydrodynamic.pressure,
      velocity=hydrodynamic.velocity,
      magnetic=Vector(*sp.symbols("primitive.b1 primitive.b2 primitive.b3")),
   ), metric


def _metric_derivative_symbols() -> ValenciaMetricDerivatives:
   lapse_gradient = Vector3(
      *sp.symbols("derivatives.d1_lapse derivatives.d2_lapse derivatives.d3_lapse"),
      position=IndexPosition.COVARIANT,
   )
   shift_gradients = tuple(
      Vector3(
         *(
            sp.Symbol(f"derivatives.d{direction}_shift{component}")
            for direction in range(1, 4)
         ),
         position=IndexPosition.COVARIANT,
      )
      for component in range(1, 4)
   )
   spatial_metric_gradients = tuple(
      Tensor3(
         (
            (
               sp.Symbol(f"derivatives.d{direction}_gamma11"),
               sp.Symbol(f"derivatives.d{direction}_gamma12"),
               sp.Symbol(f"derivatives.d{direction}_gamma13"),
            ),
            (
               sp.Symbol(f"derivatives.d{direction}_gamma12"),
               sp.Symbol(f"derivatives.d{direction}_gamma22"),
               sp.Symbol(f"derivatives.d{direction}_gamma23"),
            ),
            (
               sp.Symbol(f"derivatives.d{direction}_gamma13"),
               sp.Symbol(f"derivatives.d{direction}_gamma23"),
               sp.Symbol(f"derivatives.d{direction}_gamma33"),
            ),
         ),
         (IndexPosition.COVARIANT, IndexPosition.COVARIANT),
      )
      for direction in range(1, 4)
   )
   extrinsic_curvature = Tensor3(
      (
         (
            sp.Symbol("derivatives.extrinsic11"),
            sp.Symbol("derivatives.extrinsic12"),
            sp.Symbol("derivatives.extrinsic13"),
         ),
         (
            sp.Symbol("derivatives.extrinsic12"),
            sp.Symbol("derivatives.extrinsic22"),
            sp.Symbol("derivatives.extrinsic23"),
         ),
         (
            sp.Symbol("derivatives.extrinsic13"),
            sp.Symbol("derivatives.extrinsic23"),
            sp.Symbol("derivatives.extrinsic33"),
         ),
      ),
      (IndexPosition.COVARIANT, IndexPosition.COVARIANT),
   )
   return ValenciaMetricDerivatives(
      lapse_gradient=lapse_gradient,
      shift_gradients=shift_gradients,
      spatial_metric_gradients=spatial_metric_gradients,
      extrinsic_curvature=extrinsic_curvature,
   )


@dataclass(frozen=True)
class _RMHDRecoverySymbols:
   density: sp.Symbol
   momentum: Vector
   energy: sp.Symbol
   magnetic: Vector
   momentum_squared: sp.Symbol
   magnetic_squared: sp.Symbol
   magnetic_momentum: sp.Symbol


def _rmhd_recovery_symbols() -> _RMHDRecoverySymbols:
   return _RMHDRecoverySymbols(
      density=sp.Symbol("recovery.dens", positive=True),
      momentum=Vector(*sp.symbols("recovery.mom1 recovery.mom2 recovery.mom3")),
      energy=sp.Symbol("recovery.tau", real=True),
      magnetic=Vector(*sp.symbols("recovery.b1 recovery.b2 recovery.b3")),
      momentum_squared=sp.Symbol("recovery.momentum_squared", nonnegative=True),
      magnetic_squared=sp.Symbol("recovery.magnetic_squared", nonnegative=True),
      magnetic_momentum=sp.Symbol("recovery.magnetic_momentum", real=True),
   )


def rmhd_recovery_state_kernel() -> StructKernel:
   _, metric = _rmhd_symbols()
   inverse_sqrt_det = 1 / metric.sqrt_spatial_determinant
   momentum_covariant = Vector3(
      *(
         sp.Symbol(f"conserved.mom{index}") * inverse_sqrt_det
         for index in range(1, 4)
      ),
      position=IndexPosition.COVARIANT,
   )
   momentum_contravariant = momentum_covariant.raise_index(
      metric.spatial_metric
   )
   magnetic_contravariant = Vector3(
      *(
         sp.Symbol(f"conserved.b{index}") * inverse_sqrt_det
         for index in range(1, 4)
      ),
      position=IndexPosition.CONTRAVARIANT,
   )
   magnetic_covariant = magnetic_contravariant.lower_index(
      metric.spatial_metric
   )
   return StructKernel(
      name="rmhd_recovery_state",
      return_type="RMHDRecoveryState",
      arguments=(
         ("const RMHDConserved&", "conserved"),
         ("const SpacetimeMetric&", "metric"),
      ),
      outputs=(
         *RELATIVISTIC_MHD_COMPONENT_NAMES,
         "momentum_squared",
         "magnetic_squared",
         "magnetic_momentum",
      ),
      expressions=(
         sp.Symbol("conserved.dens") * inverse_sqrt_det,
         *momentum_contravariant.as_tuple(),
         sp.Symbol("conserved.tau") * inverse_sqrt_det,
         *magnetic_contravariant.as_tuple(),
         momentum_covariant.contract(momentum_contravariant),
         magnetic_covariant.contract(magnetic_contravariant),
         momentum_covariant.contract(magnetic_contravariant),
      ),
   )


def rhd_recovery_state_kernel() -> StructKernel:
   _, metric = _symbols()
   inverse_sqrt_det = 1 / metric.sqrt_spatial_determinant
   momentum_covariant = Vector3(
      *(
         sp.Symbol(f"conserved.mom{index}") * inverse_sqrt_det
         for index in range(1, 4)
      ),
      position=IndexPosition.COVARIANT,
   )
   momentum_contravariant = momentum_covariant.raise_index(
      metric.spatial_metric
   )
   return StructKernel(
      name="rhd_recovery_state",
      return_type="RHDRecoveryState",
      arguments=(
         ("const RHDConserved&", "conserved"),
         ("const SpacetimeMetric&", "metric"),
      ),
      outputs=(*RELATIVISTIC_HYDRO_COMPONENT_NAMES, "momentum_squared"),
      expressions=(
         sp.Symbol("conserved.dens") * inverse_sqrt_det,
         *momentum_contravariant.as_tuple(),
         sp.Symbol("conserved.tau") * inverse_sqrt_det,
         momentum_covariant.contract(momentum_contravariant),
      ),
   )


def rhd_primitive_to_conserved_kernel() -> StructKernel:
   primitive, metric = _symbols()
   conserved = ValenciaHydrodynamics().primitive_to_conserved(primitive, metric)
   return StructKernel(
      name="rhd_primitive_to_conserved",
      return_type="RHDConserved",
      arguments=(("const RHDPrimitive&", "primitive"), ("const SpacetimeMetric&", "metric")),
      outputs=RELATIVISTIC_HYDRO_COMPONENT_NAMES,
      expressions=(
         conserved.density,
         *conserved.momentum.as_tuple(),
         conserved.energy,
      ),
   )


def rhd_flux_kernel(direction: int) -> StructKernel:
   if direction < 0 or direction >= 3:
      raise ValueError("RHD flux direction must be in [0, 3)")
   primitive, metric = _symbols()
   flux = ValenciaHydrodynamics().conservation_law(primitive, metric).flux_along(direction)
   density_flux = flux[0]
   momentum_flux = flux[1]
   energy_flux = flux[2]
   if not isinstance(density_flux, sp.Expr) or not isinstance(momentum_flux, Vector) or not isinstance(energy_flux, sp.Expr):
      raise TypeError("Valencia conservation law produced an invalid flux layout")
   return StructKernel(
      name=f"rhd_flux_dir{direction}",
      return_type="RHDConserved",
      arguments=(("const RHDPrimitive&", "primitive"), ("const SpacetimeMetric&", "metric")),
      outputs=RELATIVISTIC_HYDRO_COMPONENT_NAMES,
      expressions=(density_flux, *momentum_flux.as_tuple(), energy_flux),
   )


def rmhd_primitive_to_conserved_kernel() -> StructKernel:
   primitive, metric = _rmhd_symbols()
   conserved = ValenciaMagnetohydrodynamics().primitive_to_conserved(
      primitive, metric
   )
   return StructKernel(
      name="rmhd_primitive_to_conserved",
      return_type="RMHDConserved",
      arguments=(("const RMHDPrimitive&", "primitive"), ("const SpacetimeMetric&", "metric")),
      outputs=RELATIVISTIC_MHD_COMPONENT_NAMES,
      expressions=(
         conserved.density,
         *conserved.momentum.as_tuple(),
         conserved.energy,
         *conserved.magnetic.as_tuple(),
      ),
   )


def rmhd_comoving_magnetic_squared_kernel() -> ScalarKernel:
   primitive, metric = _rmhd_symbols()
   return ScalarKernel(
      name="rmhd_comoving_magnetic_squared",
      arguments=(
         ("const RMHDPrimitive&", "primitive"),
         ("const SpacetimeMetric&", "metric"),
      ),
      expression=ValenciaMagnetohydrodynamics().comoving_magnetic_squared(
         primitive, metric
      ),
   )


def rmhd_flux_kernel(direction: int) -> StructKernel:
   if direction < 0 or direction >= 3:
      raise ValueError("RMHD flux direction must be in [0, 3)")
   primitive, metric = _rmhd_symbols()
   flux = ValenciaMagnetohydrodynamics().conservation_law(
      primitive, metric
   ).flux_along(direction)
   density_flux, momentum_flux, energy_flux, magnetic_flux = flux
   if (
      not isinstance(density_flux, sp.Expr)
      or not isinstance(momentum_flux, Vector)
      or not isinstance(energy_flux, sp.Expr)
      or not isinstance(magnetic_flux, Vector)
   ):
      raise TypeError("Valencia RMHD conservation law produced an invalid flux layout")
   return StructKernel(
      name=f"rmhd_flux_dir{direction}",
      return_type="RMHDConserved",
      arguments=(("const RMHDPrimitive&", "primitive"), ("const SpacetimeMetric&", "metric")),
      outputs=RELATIVISTIC_MHD_COMPONENT_NAMES,
      expressions=(
         density_flux,
         *momentum_flux.as_tuple(),
         energy_flux,
         *magnetic_flux.as_tuple(),
      ),
   )


def _stress_energy_kernel(
   name: str,
   primitive_type: str,
   stress_energy: Tensor,
) -> StructKernel:
   components = (
      (0, 0),
      (0, 1),
      (0, 2),
      (0, 3),
      (1, 1),
      (1, 2),
      (1, 3),
      (2, 2),
      (2, 3),
      (3, 3),
   )
   return StructKernel(
      name=name,
      return_type="StressEnergyTensor",
      arguments=(
         (f"const {primitive_type}&", "primitive"),
         ("const SpacetimeMetric&", "metric"),
      ),
      outputs=tuple(f"t{first}{second}" for first, second in components),
      expressions=tuple(
         stress_energy.component(first, second) for first, second in components
      ),
   )


def rhd_stress_energy_kernel() -> StructKernel:
   primitive, metric = _symbols()
   stress_energy = ValenciaHydrodynamics().stress_energy(primitive, metric)
   return _stress_energy_kernel(
      "rhd_stress_energy", "RHDPrimitive", stress_energy.values
   )


def rmhd_stress_energy_kernel() -> StructKernel:
   primitive, metric = _rmhd_symbols()
   stress_energy = ValenciaMagnetohydrodynamics().stress_energy(primitive, metric)
   return _stress_energy_kernel(
      "rmhd_stress_energy", "RMHDPrimitive", stress_energy.values
   )


def _eulerian_matter_kernel(
   name: str,
   primitive_type: str,
   stress_energy: Tensor,
   metric: ValenciaMetric,
) -> StructKernel:
   matter = eulerian_matter_projection(stress_energy, metric)
   return StructKernel(
      name=name,
      return_type="EulerianMatter",
      arguments=(
         (f"const {primitive_type}&", "primitive"),
         ("const SpacetimeMetric&", "metric"),
      ),
      outputs=(
         "rho",
         "momentum1",
         "momentum2",
         "momentum3",
         "stress11",
         "stress12",
         "stress13",
         "stress22",
         "stress23",
         "stress33",
      ),
      expressions=(
         matter.energy,
         *matter.momentum.as_tuple(),
         matter.stress.component(0, 0),
         matter.stress.component(0, 1),
         matter.stress.component(0, 2),
         matter.stress.component(1, 1),
         matter.stress.component(1, 2),
         matter.stress.component(2, 2),
      ),
   )


def rhd_eulerian_matter_kernel() -> StructKernel:
   primitive, metric = _symbols()
   return _eulerian_matter_kernel(
      "rhd_eulerian_matter",
      "RHDPrimitive",
      ValenciaHydrodynamics().stress_energy(primitive, metric),
      metric,
   )


def rmhd_eulerian_matter_kernel() -> StructKernel:
   primitive, metric = _rmhd_symbols()
   return _eulerian_matter_kernel(
      "rmhd_eulerian_matter",
      "RMHDPrimitive",
      ValenciaMagnetohydrodynamics().stress_energy(primitive, metric),
      metric,
   )


def rhd_diagnostic_integrands_kernel() -> StructKernel:
   primitive, metric = _symbols()
   velocity_squared = metric.spatial_metric.norm_squared(primitive.velocity)
   return StructKernel(
      name="rhd_diagnostic_integrands",
      return_type="RelativisticFluidDiagnosticIntegrands",
      arguments=(
         ("const RHDPrimitive&", "primitive"),
         ("const SpacetimeMetric&", "metric"),
      ),
      outputs=("lorentz_factor",),
      expressions=(1 / sp.sqrt(1 - velocity_squared),),
   )


def rmhd_diagnostic_integrands_kernel() -> StructKernel:
   primitive, metric = _rmhd_symbols()
   system = ValenciaMagnetohydrodynamics()
   hydro_primitive = ValenciaPrimitive(
      density=primitive.density,
      specific_internal_energy=primitive.specific_internal_energy,
      pressure=primitive.pressure,
      velocity=primitive.velocity,
   )
   magnetic_squared = system.comoving_magnetic_squared(primitive, metric)
   specific_enthalpy = (
      1 + primitive.specific_internal_energy
      + primitive.pressure / primitive.density
   )
   velocity_squared = metric.spatial_metric.norm_squared(primitive.velocity)
   magnetofluid_energy = eulerian_matter_projection(
      system.stress_energy(primitive, metric), metric
   ).energy
   fluid_energy = eulerian_matter_projection(
      ValenciaHydrodynamics().stress_energy(hydro_primitive, metric), metric
   ).energy
   return StructKernel(
      name="rmhd_diagnostic_integrands",
      return_type="RMHDDiagnosticIntegrands",
      arguments=(
         ("const RMHDPrimitive&", "primitive"),
         ("const SpacetimeMetric&", "metric"),
      ),
      outputs=(
         "electromagnetic_energy",
         "b_squared",
         "enthalpy_magnetization",
         "plasma_beta",
         "lorentz_factor",
      ),
      expressions=(
         metric.sqrt_spatial_determinant * (magnetofluid_energy - fluid_energy),
         magnetic_squared,
         magnetic_squared / (primitive.density * specific_enthalpy),
         2 * primitive.pressure / magnetic_squared,
         1 / sp.sqrt(1 - velocity_squared),
      ),
   )


def rhd_geometric_source_kernel() -> StructKernel:
   primitive, metric = _symbols()
   system = ValenciaHydrodynamics()
   conserved = system.primitive_to_conserved(primitive, metric)
   source = valencia_geometric_source(
      conserved,
      system.stress_energy(primitive, metric),
      metric,
      _metric_derivative_symbols(),
   )
   return StructKernel(
      name="rhd_geometric_source",
      return_type="RHDConserved",
      arguments=(
         ("const RHDPrimitive&", "primitive"),
         ("const SpacetimeMetric&", "metric"),
         ("const SpacetimeMetricDerivatives&", "derivatives"),
      ),
      outputs=RELATIVISTIC_HYDRO_COMPONENT_NAMES,
      expressions=(sp.Integer(0), *source.momentum.as_tuple(), source.energy),
   )


def rmhd_geometric_source_kernel() -> StructKernel:
   primitive, metric = _rmhd_symbols()
   system = ValenciaMagnetohydrodynamics()
   conserved = system.primitive_to_conserved(primitive, metric)
   source = valencia_geometric_source(
      conserved,
      system.stress_energy(primitive, metric),
      metric,
      _metric_derivative_symbols(),
   )
   return StructKernel(
      name="rmhd_geometric_source",
      return_type="RMHDConserved",
      arguments=(
         ("const RMHDPrimitive&", "primitive"),
         ("const SpacetimeMetric&", "metric"),
         ("const SpacetimeMetricDerivatives&", "derivatives"),
      ),
      outputs=RELATIVISTIC_MHD_COMPONENT_NAMES,
      expressions=(
         sp.Integer(0),
         *source.momentum.as_tuple(),
         source.energy,
         sp.Integer(0),
         sp.Integer(0),
         sp.Integer(0),
      ),
   )


def _relativistic_wave_speeds(
   velocity: Vector,
   characteristic_speed_squared: sp.Expr,
   direction: int,
   metric: ValenciaMetric,
) -> tuple[sp.Expr, sp.Expr]:
   velocity_squared = metric.spatial_metric.norm_squared(velocity)
   normal_velocity = velocity.as_tuple()[direction]
   radicand = (1 - velocity_squared) * (
      metric.spatial_metric.contravariant.component(direction, direction)
      * (1 - velocity_squared * characteristic_speed_squared)
      - normal_velocity**2 * (1 - characteristic_speed_squared)
   )
   denominator = 1 - velocity_squared * characteristic_speed_squared
   characteristic_speed = sp.sqrt(characteristic_speed_squared)
   common = normal_velocity * (1 - characteristic_speed_squared)
   spread = characteristic_speed * sp.sqrt(sp.Max(0, radicand))
   shift = metric.shift.as_tuple()[direction]
   return (
      metric.lapse * (common - spread) / denominator - shift,
      metric.lapse * (common + spread) / denominator - shift,
   )


def srhd_wave_speed_kernel(direction: int) -> StructKernel:
   if direction < 0 or direction >= 3:
      raise ValueError("SRHD wave-speed direction must be in [0, 3)")
   primitive, metric = _symbols()
   sound_speed_squared = sp.Symbol("sound_speed_squared", nonnegative=True)
   minus, plus = _relativistic_wave_speeds(
      primitive.velocity, sound_speed_squared, direction, metric
   )
   return StructKernel(
      name=f"rhd_wave_speeds_dir{direction}",
      return_type="RHDWaveSpeeds",
      arguments=(
         ("const RHDPrimitive&", "primitive"),
         ("const SpacetimeMetric&", "metric"),
         ("amrex::Real", "sound_speed_squared"),
      ),
      outputs=("minus", "plus"),
      expressions=(minus, plus),
   )


def srmhd_wave_speed_kernel(direction: int) -> StructKernel:
   if direction < 0 or direction >= 3:
      raise ValueError("SRMHD wave-speed direction must be in [0, 3)")
   primitive, metric = _rmhd_symbols()
   sound_speed_squared = sp.Symbol(
      "sound_speed_squared", nonnegative=True
   )
   specific_enthalpy = sp.Symbol("specific_enthalpy", positive=True)
   velocity_squared = metric.spatial_metric.norm_squared(primitive.velocity)
   lorentz_factor_squared = 1 / (1 - velocity_squared)
   magnetic_squared = metric.spatial_metric.norm_squared(primitive.magnetic)
   magnetic_velocity = primitive.magnetic.dot(
      metric.spatial_metric.lower(primitive.velocity)
   )
   comoving_magnetic_squared = (
      magnetic_squared / lorentz_factor_squared + magnetic_velocity**2
   )
   alfven_speed_squared = comoving_magnetic_squared / (
      primitive.density * specific_enthalpy + comoving_magnetic_squared
   )
   fast_speed_squared = (
      sound_speed_squared
      + alfven_speed_squared
      - sound_speed_squared * alfven_speed_squared
   )
   minus, plus = _relativistic_wave_speeds(
      primitive.velocity, fast_speed_squared, direction, metric
   )
   return StructKernel(
      name=f"srmhd_wave_speeds_dir{direction}",
      return_type="RMHDWaveSpeeds",
      arguments=(
         ("const RMHDPrimitive&", "primitive"),
         ("const SpacetimeMetric&", "metric"),
         ("amrex::Real", "sound_speed_squared"),
         ("amrex::Real", "specific_enthalpy"),
      ),
      outputs=("minus", "plus"),
      expressions=(minus, plus),
   )


def _srhd_palenzuela_expressions() -> tuple[
   tuple[sp.Symbol, ...],
   sp.Expr,
   tuple[sp.Expr, ...],
   tuple[tuple[sp.Symbol, sp.Expr], ...],
]:
   dens = sp.Symbol("recovery.dens", positive=True)
   momentum = Vector(*sp.symbols("recovery.mom1 recovery.mom2 recovery.mom3"))
   tau = sp.Symbol("recovery.tau", real=True)
   x, gamma = sp.symbols("x gamma", positive=True)
   momentum_squared = sp.Symbol("recovery.momentum_squared", nonnegative=True)
   # Preserve dimensionless normalization before CSE. Expanding these ratios
   # into powers of D creates scale-dependent cancellation and underflow.
   q = sp.Symbol("energy_ratio", real=True)
   r = sp.Symbol("momentum_ratio_squared", nonnegative=True)
   velocity_squared = sp.Symbol("velocity_squared", nonnegative=True)
   inverse_lorentz = sp.Symbol("inverse_lorentz", positive=True)
   lorentz_factor = sp.Symbol("lorentz_factor", positive=True)
   lorentz_minus_one = sp.Symbol("lorentz_minus_one", nonnegative=True)
   bindings = (
      (q, tau / dens),
      (r, momentum_squared / dens**2),
      (velocity_squared, r / x**2),
      (inverse_lorentz, sp.sqrt(1 - velocity_squared)),
      (lorentz_factor, 1 / inverse_lorentz),
      (lorentz_minus_one,
       velocity_squared / (inverse_lorentz * (1 + inverse_lorentz))),
   )
   density = dens * inverse_lorentz
   # Rationalize W-1. At rest this gives eps=q exactly, including cold matter;
   # subtracting nearly equal density-dependent terms can make eps negative.
   specific_internal_energy = (
      lorentz_minus_one + lorentz_factor * (q - r / x)
   )
   ideal_gas = ideal_gas_equation_of_state()
   pressure = ideal_gas.pressure_from_specific_internal_energy(
      specific_internal_energy
   ).subs({ideal_gas.density: density, ideal_gas.parameters[0]: gamma})
   specific_enthalpy = ideal_gas.relativistic_specific_enthalpy(
      specific_internal_energy
   ).subs({ideal_gas.density: density, ideal_gas.parameters[0]: gamma})
   residual = x - specific_enthalpy * lorentz_factor
   primitive = (
      density,
      specific_internal_energy,
      pressure,
      *(momentum / (dens * x)).as_tuple(),
   )
   return (x, gamma), residual, primitive, bindings


def srhd_palenzuela_kernels() -> tuple[ScalarKernel | StructKernel, ...]:
   arguments, residual, primitive, bindings = _srhd_palenzuela_expressions()
   scalar_arguments = (
      ("const RHDRecoveryState&", "recovery"),
      *(str(symbol) for symbol in arguments),
   )
   return (
      ScalarKernel(
         name="rhd_specific_internal_energy_from_palenzuela",
         arguments=(("const RHDRecoveryState&", "recovery"), ("amrex::Real", "x")),
         expression=primitive[1],
         bindings=bindings,
      ),
      ScalarKernel(
         name="srhd_palenzuela_residual",
         arguments=scalar_arguments,
         expression=residual,
         bindings=bindings,
      ),
      ScalarKernel(
         name="srhd_palenzuela_residual_derivative",
         arguments=scalar_arguments,
         expression=differentiate_with_bindings(residual, arguments[-2], bindings),
         bindings=bindings,
      ),
      StructKernel(
         name="srhd_primitive_from_palenzuela",
         return_type="RHDPrimitive",
         arguments=(
            ("const RHDRecoveryState&", "recovery"),
            ("amrex::Real", "x"),
            ("amrex::Real", "gamma"),
         ),
         outputs=("rho", "eps", "pressure", "v1", "v2", "v3"),
         expressions=primitive,
         bindings=bindings,
      ),
   )


def _srmhd_palenzuela_expressions() -> tuple[
   tuple[sp.Symbol, ...],
   sp.Expr,
   tuple[sp.Expr, ...],
   tuple[sp.Expr, ...],
]:
   recovery = _rmhd_recovery_symbols()
   dens = recovery.density
   momentum = recovery.momentum
   tau = recovery.energy
   magnetic = recovery.magnetic
   x, gamma = sp.symbols("x gamma", positive=True)
   momentum_squared = recovery.momentum_squared
   magnetic_squared = recovery.magnetic_squared
   magnetic_momentum = recovery.magnetic_momentum
   q = tau / dens
   r = momentum_squared / dens**2
   s = magnetic_squared / dens
   t = magnetic_momentum / dens ** sp.Rational(3, 2)
   inverse_lorentz_squared = 1 - (
      x**2 * r + (2 * x + s) * t**2
   ) / (x**2 * (x + s)**2)
   lorentz_factor = 1 / sp.sqrt(inverse_lorentz_squared)
   density = dens / lorentz_factor
   specific_internal_energy = (
      lorentz_factor
      - 1
      + (1 - lorentz_factor**2) * x / lorentz_factor
      + lorentz_factor * (
         q - s + t**2 / (2 * x**2) + s / (2 * lorentz_factor**2)
      )
   )
   ideal_gas = ideal_gas_equation_of_state()
   pressure = ideal_gas.pressure_from_specific_internal_energy(
      specific_internal_energy
   ).subs({ideal_gas.density: density, ideal_gas.parameters[0]: gamma})
   specific_enthalpy = ideal_gas.relativistic_specific_enthalpy(
      specific_internal_energy
   ).subs({ideal_gas.density: density, ideal_gas.parameters[0]: gamma})
   residual = x - specific_enthalpy * lorentz_factor
   z = x * dens
   velocity = momentum / (z + magnetic_squared) + (
      magnetic_momentum * magnetic / (z * (z + magnetic_squared))
   )
   primitive = (
      density,
      specific_internal_energy,
      pressure,
      *velocity.as_tuple(),
      *magnetic.as_tuple(),
   )
   kinematics = (
      density,
      specific_internal_energy,
      lorentz_factor,
      *velocity.as_tuple(),
      *magnetic.as_tuple(),
   )
   return (x, gamma), residual, primitive, kinematics


def srmhd_palenzuela_kernels() -> tuple[ScalarKernel | StructKernel, ...]:
   arguments, residual, primitive, kinematics = (
      _srmhd_palenzuela_expressions()
   )
   scalar_arguments = (
      ("const RMHDRecoveryState&", "recovery"),
      *(str(symbol) for symbol in arguments),
   )
   return (
      ScalarKernel(
         name="srmhd_palenzuela_residual",
         arguments=scalar_arguments,
         expression=residual,
      ),
      ScalarKernel(
         name="srmhd_palenzuela_residual_derivative",
         arguments=scalar_arguments,
         expression=sp.diff(residual, arguments[-2]),
      ),
      StructKernel(
         name="srmhd_primitive_from_palenzuela",
         return_type="RMHDPrimitive",
         arguments=(
            ("const RMHDRecoveryState&", "recovery"),
            ("amrex::Real", "x"),
            ("amrex::Real", "gamma"),
         ),
         outputs=("rho", "eps", "pressure", "v1", "v2", "v3", "b1", "b2", "b3"),
         expressions=primitive,
      ),
      StructKernel(
         name="srmhd_palenzuela_kinematics",
         return_type="RMHDRecoveryKinematics",
         arguments=(
            ("const RMHDRecoveryState&", "recovery"),
            ("amrex::Real", "x"),
         ),
         outputs=(
            "rho", "eps", "lorentz_factor",
            "v1", "v2", "v3", "b1", "b2", "b3",
         ),
         expressions=kinematics,
      ),
   )


def _srmhd_entropy_expressions(
   root_kind: str,
) -> tuple[tuple[sp.Symbol, ...], sp.Expr, tuple[sp.Expr, ...]]:
   recovery = _rmhd_recovery_symbols()
   dens = recovery.density
   momentum = recovery.momentum
   magnetic = recovery.magnetic
   conserved_entropy = sp.Symbol("conserved_entropy", positive=True)
   gamma = sp.Symbol("gamma", positive=True)
   momentum_squared = recovery.momentum_squared
   magnetic_squared = recovery.magnetic_squared
   magnetic_momentum = recovery.magnetic_momentum
   if root_kind == "enthalpy_lorentz":
      root = sp.Symbol("x", positive=True)
      normalized_momentum_squared = momentum_squared / dens**2
      normalized_magnetic_squared = magnetic_squared / dens
      normalized_magnetic_momentum = magnetic_momentum / dens ** sp.Rational(3, 2)
      inverse_lorentz_squared = 1 - (
         root**2 * normalized_momentum_squared
         + (2 * root + normalized_magnetic_squared)
         * normalized_magnetic_momentum**2
      ) / (
         root**2 * (root + normalized_magnetic_squared) ** 2
      )
      lorentz_factor = 1 / sp.sqrt(inverse_lorentz_squared)
      density = dens / lorentz_factor
      entropy_function = conserved_entropy / lorentz_factor
      pressure = entropy_function * density ** (gamma - 1)
      specific_internal_energy = pressure / ((gamma - 1) * density)
      specific_enthalpy = 1 + gamma * specific_internal_energy
      residual = root - specific_enthalpy * lorentz_factor
      z = root * dens
   elif root_kind == "density":
      root = sp.Symbol("density", positive=True)
      density = root
      lorentz_factor = dens / density
      velocity_squared = 1 - 1 / lorentz_factor**2
      entropy_function = conserved_entropy / lorentz_factor
      pressure = entropy_function * density ** (gamma - 1)
      specific_internal_energy = pressure / ((gamma - 1) * density)
      specific_enthalpy = 1 + gamma * specific_internal_energy
      z = density * specific_enthalpy * lorentz_factor**2
      residual = (
         momentum_squared
         - velocity_squared * (magnetic_squared + z) ** 2
         + magnetic_momentum**2 * (magnetic_squared + 2 * z) / z**2
      )
   else:
      raise ValueError(f"unknown entropy C2P root kind: {root_kind}")
   velocity = momentum / (z + magnetic_squared) + (
      magnetic_momentum * magnetic / (z * (z + magnetic_squared))
   )
   primitive = (
      density,
      specific_internal_energy,
      pressure,
      *velocity.as_tuple(),
      *magnetic.as_tuple(),
   )
   return (conserved_entropy, root, gamma), residual, primitive


def _srmhd_entropy_kernels(
   name: str,
   root_kind: str,
) -> tuple[ScalarKernel | StructKernel, ...]:
   arguments, residual, primitive = _srmhd_entropy_expressions(root_kind)
   scalar_arguments = (
      ("const RMHDRecoveryState&", "recovery"),
      *(str(symbol) for symbol in arguments),
   )
   root = arguments[-2]
   return (
      ScalarKernel(
         name=f"srmhd_{name}_residual",
         arguments=scalar_arguments,
         expression=residual,
      ),
      ScalarKernel(
         name=f"srmhd_{name}_residual_derivative",
         arguments=scalar_arguments,
         expression=sp.diff(residual, root),
      ),
      StructKernel(
         name=f"srmhd_primitive_from_{name}",
         return_type="RMHDPrimitive",
         arguments=(
            ("const RMHDRecoveryState&", "recovery"),
            ("amrex::Real", "conserved_entropy"),
            ("amrex::Real", str(root)),
            ("amrex::Real", "gamma"),
         ),
         outputs=("rho", "eps", "pressure", "v1", "v2", "v3", "b1", "b2", "b3"),
         expressions=primitive,
      ),
   )


def srmhd_entropy_kernels() -> tuple[ScalarKernel | StructKernel, ...]:
   return (
      *_srmhd_entropy_kernels("palenzuela_entropy", "enthalpy_lorentz"),
      *_srmhd_entropy_kernels("noble_entropy", "density"),
   )


def _srmhd_cold_expressions() -> tuple[
   tuple[sp.Symbol, ...], sp.Expr, tuple[sp.Expr, ...]
]:
   recovery = _rmhd_recovery_symbols()
   dens = recovery.density
   momentum = recovery.momentum
   magnetic = recovery.magnetic
   density, polytropic_k, polytropic_gamma = sp.symbols(
      "density polytropic_k polytropic_gamma", positive=True
   )
   momentum_squared = recovery.momentum_squared
   magnetic_squared = recovery.magnetic_squared
   magnetic_momentum = recovery.magnetic_momentum
   lorentz_factor = dens / density
   velocity_squared = 1 - 1 / lorentz_factor**2
   pressure = polytropic_k * density**polytropic_gamma
   specific_internal_energy = pressure / (
      density * (polytropic_gamma - 1)
   )
   specific_enthalpy = 1 + specific_internal_energy + pressure / density
   z = density * specific_enthalpy * lorentz_factor**2
   residual = (
      momentum_squared
      - velocity_squared * (magnetic_squared + z) ** 2
      + magnetic_momentum**2 * (magnetic_squared + 2 * z) / z**2
   )
   velocity = momentum / (z + magnetic_squared) + (
      magnetic_momentum * magnetic / (z * (z + magnetic_squared))
   )
   primitive = (
      density,
      specific_internal_energy,
      pressure,
      *velocity.as_tuple(),
      *magnetic.as_tuple(),
   )
   return (density, polytropic_k, polytropic_gamma), residual, primitive


def srmhd_cold_kernels() -> tuple[ScalarKernel | StructKernel, ...]:
   arguments, residual, primitive = _srmhd_cold_expressions()
   scalar_arguments = (
      ("const RMHDRecoveryState&", "recovery"),
      *(str(symbol) for symbol in arguments),
   )
   density = arguments[-3]
   return (
      ScalarKernel(
         name="srmhd_font_cold_residual",
         arguments=scalar_arguments,
         expression=residual,
      ),
      ScalarKernel(
         name="srmhd_font_cold_residual_derivative",
         arguments=scalar_arguments,
         expression=sp.diff(residual, density),
      ),
      StructKernel(
         name="srmhd_primitive_from_font_cold",
         return_type="RMHDPrimitive",
         arguments=(
            ("const RMHDRecoveryState&", "recovery"),
            ("amrex::Real", "density"),
            ("amrex::Real", "polytropic_k"),
            ("amrex::Real", "polytropic_gamma"),
         ),
         outputs=("rho", "eps", "pressure", "v1", "v2", "v3", "b1", "b2", "b3"),
         expressions=primitive,
      ),
   )


def _srmhd_noble_2d_expressions() -> tuple[
   tuple[sp.Symbol, ...],
   tuple[sp.Expr, ...],
   tuple[sp.Expr, ...],
]:
   """Return Noble's seeded two-dimensional (Z, v^2) ideal-MHD system."""
   recovery = _rmhd_recovery_symbols()
   dens = recovery.density
   momentum = recovery.momentum
   tau = recovery.energy
   magnetic = recovery.magnetic
   z, velocity_squared, gamma = sp.symbols(
      "z velocity_squared gamma", positive=True
   )
   momentum_squared = recovery.momentum_squared
   magnetic_squared = recovery.magnetic_squared
   magnetic_momentum = recovery.magnetic_momentum
   inverse_lorentz = sp.sqrt(1 - velocity_squared)
   pressure = (gamma - 1) / gamma * (
      z * (1 - velocity_squared) - dens * inverse_lorentz
   )
   residual1 = (
      momentum_squared
      - velocity_squared * (magnetic_squared + z) ** 2
      + magnetic_momentum**2 * (magnetic_squared + 2 * z) / z**2
   )
   residual2 = (
      tau
      + dens
      - sp.Rational(1, 2) * magnetic_squared * (1 + velocity_squared)
      + sp.Rational(1, 2) * magnetic_momentum**2 / z**2
      - z
      + pressure
   )
   lorentz_factor = 1 / inverse_lorentz
   density = dens / lorentz_factor
   specific_internal_energy = (
      z * (1 - velocity_squared) / density - 1
   ) / gamma
   velocity = momentum / (z + magnetic_squared) + (
      magnetic_momentum * magnetic / (z * (z + magnetic_squared))
   )
   primitive = (
      density,
      specific_internal_energy,
      pressure,
      *velocity.as_tuple(),
      *magnetic.as_tuple(),
   )
   return (z, velocity_squared, gamma), (residual1, residual2), primitive


def srmhd_noble_2d_kernels() -> tuple[StructKernel, ...]:
   arguments, residuals, primitive = _srmhd_noble_2d_expressions()
   scalar_arguments = (
      ("const RMHDRecoveryState&", "recovery"),
      *(("amrex::Real", str(symbol)) for symbol in arguments),
   )
   z = arguments[-3]
   velocity_squared = arguments[-2]
   residual1, residual2 = residuals
   return (
      StructKernel(
         name="srmhd_noble_2d_system",
         return_type="RootSystem2",
         arguments=scalar_arguments,
         outputs=(
            "residual1",
            "residual2",
            "jacobian11",
            "jacobian12",
            "jacobian21",
            "jacobian22",
         ),
         expressions=(
            residual1,
            residual2,
            sp.diff(residual1, z),
            sp.diff(residual1, velocity_squared),
            sp.diff(residual2, z),
            sp.diff(residual2, velocity_squared),
         ),
      ),
      StructKernel(
         name="srmhd_primitive_from_noble_2d",
         return_type="RMHDPrimitive",
         arguments=(
            ("const RMHDRecoveryState&", "recovery"),
            ("amrex::Real", "z"),
            ("amrex::Real", "velocity_squared"),
            ("amrex::Real", "gamma"),
         ),
         outputs=("rho", "eps", "pressure", "v1", "v2", "v3", "b1", "b2", "b3"),
         expressions=primitive,
      ),
   )
def _srmhd_reprimand_expressions() -> tuple[
   tuple[sp.Symbol, ...],
   sp.Expr,
   tuple[sp.Expr, ...],
   sp.Expr,
   sp.Expr,
   sp.Expr,
   tuple[sp.Expr, ...],
]:
   """Return Kastaun's seed-free ideal-MHD master root."""
   recovery = _rmhd_recovery_symbols()
   dens = recovery.density
   momentum = recovery.momentum
   tau = recovery.energy
   magnetic = recovery.magnetic
   mu, gamma = sp.symbols("mu gamma", positive=True)
   normalized_momentum = momentum / dens
   normalized_magnetic = magnetic / sp.sqrt(dens)
   momentum_squared = recovery.momentum_squared / dens**2
   magnetic_squared = recovery.magnetic_squared / dens
   magnetic_momentum = recovery.magnetic_momentum / dens ** sp.Rational(3, 2)
   magnetic_momentum_squared = magnetic_momentum**2
   perpendicular_product = (
      momentum_squared * magnetic_squared - magnetic_momentum_squared
   )
   x = 1 / (1 + mu * magnetic_squared)
   fluid_momentum_squared = x * (
      momentum_squared * x
      + mu * (x + 1) * magnetic_momentum_squared
   )
   fluid_energy = tau / dens - sp.Rational(1, 2) * (
      magnetic_squared + (mu * x) ** 2 * perpendicular_product
   )
   velocity_squared_raw = mu**2 * fluid_momentum_squared
   # RePrimAnd bounds every trial state by the maximum Lorentz factor implied
   # by the minimum EOS enthalpy. For the ideal-gas closure h_min = 1.
   velocity_squared_limit = momentum_squared / (1 + momentum_squared)
   velocity_squared = sp.Min(velocity_squared_raw, velocity_squared_limit)
   lorentz_factor = 1 / sp.sqrt(1 - velocity_squared)
   density = dens / lorentz_factor
   specific_internal_energy = lorentz_factor * (
      fluid_energy
      - mu
      * fluid_momentum_squared
      * (1 - mu * lorentz_factor / (1 + lorentz_factor))
   )
   # Substitute the ideal-gas closure before forming the master root. Calling
   # the generic simplifier on this already-large rational expression is both
   # unnecessary and prohibitively expensive.
   pressure = (gamma - 1) * density * specific_internal_energy
   pressure_ratio = pressure / (density * (1 + specific_internal_energy))
   specific_enthalpy = 1 + gamma * specific_internal_energy
   enthalpy_lorentz = sp.Max(
      (1 + pressure_ratio)
      * (1 + fluid_energy - mu * fluid_momentum_squared),
      specific_enthalpy / lorentz_factor,
   )
   updated_mu = 1 / (
      enthalpy_lorentz + mu * fluid_momentum_squared
   )
   residual = mu - updated_mu
   minimum_enthalpy_lorentz = sp.sqrt(1 + fluid_momentum_squared)
   upper_residual = mu * minimum_enthalpy_lorentz - 1
   velocity = mu * x * (
      normalized_momentum + magnetic_momentum * mu * normalized_magnetic
   )
   primitive = (
      density,
      specific_internal_energy,
      pressure,
      *velocity.as_tuple(),
      *magnetic.as_tuple(),
   )
   kinematics = (
      density,
      specific_internal_energy,
      lorentz_factor,
      fluid_momentum_squared,
      fluid_energy,
      *velocity.as_tuple(),
      *magnetic.as_tuple(),
   )
   return (
      (mu, gamma),
      residual,
      primitive,
      upper_residual,
      sp.diff(upper_residual, mu),
      lorentz_factor,
      kinematics,
   )


def srmhd_reprimand_kernels() -> tuple[ScalarKernel | StructKernel, ...]:
   (
      arguments,
      residual,
      primitive,
      upper_residual,
      upper_derivative,
      lorentz_factor,
      kinematics,
   ) = _srmhd_reprimand_expressions()
   scalar_arguments = (
      ("const RMHDRecoveryState&", "recovery"),
      *(str(symbol) for symbol in arguments),
   )
   return (
      ScalarKernel(
         name="srmhd_reprimand_residual",
         arguments=scalar_arguments,
         expression=residual,
      ),
      ScalarKernel(
         name="srmhd_reprimand_upper_residual",
         arguments=scalar_arguments,
         expression=upper_residual,
      ),
      ScalarKernel(
         name="srmhd_reprimand_upper_residual_derivative",
         arguments=scalar_arguments,
         expression=upper_derivative,
      ),
      ScalarKernel(
         name="srmhd_reprimand_lorentz_factor",
         arguments=scalar_arguments,
         expression=lorentz_factor,
      ),
      StructKernel(
         name="srmhd_primitive_from_reprimand",
         return_type="RMHDPrimitive",
         arguments=(
            ("const RMHDRecoveryState&", "recovery"),
            ("amrex::Real", "mu"),
            ("amrex::Real", "gamma"),
         ),
         outputs=("rho", "eps", "pressure", "v1", "v2", "v3", "b1", "b2", "b3"),
         expressions=primitive,
      ),
      StructKernel(
         name="srmhd_reprimand_kinematics",
         return_type="RMHDRePrimAndKinematics",
         arguments=(
            ("const RMHDRecoveryState&", "recovery"),
            ("amrex::Real", "mu"),
         ),
         outputs=(
            "rho", "eps", "lorentz_factor",
            "fluid_momentum_squared", "fluid_energy",
            "v1", "v2", "v3", "b1", "b2", "b3",
         ),
         expressions=kinematics,
      ),
   )


def relativistic_core_kernels() -> tuple[ScalarKernel | StructKernel, ...]:
   return (
      rhd_primitive_to_conserved_kernel(),
      rhd_recovery_state_kernel(),
      rmhd_primitive_to_conserved_kernel(),
      rmhd_comoving_magnetic_squared_kernel(),
      rmhd_recovery_state_kernel(),
   )


def relativistic_flux_kernels() -> tuple[ScalarKernel | StructKernel, ...]:
   return (
      *(rhd_flux_kernel(direction) for direction in range(3)),
      *(srhd_wave_speed_kernel(direction) for direction in range(3)),
      *(rmhd_flux_kernel(direction) for direction in range(3)),
      *(srmhd_wave_speed_kernel(direction) for direction in range(3)),
   )


def relativistic_source_kernels() -> tuple[ScalarKernel | StructKernel, ...]:
   return (
      rhd_stress_energy_kernel(),
      rhd_eulerian_matter_kernel(),
      rhd_diagnostic_integrands_kernel(),
      rhd_geometric_source_kernel(),
      rmhd_stress_energy_kernel(),
      rmhd_eulerian_matter_kernel(),
      rmhd_diagnostic_integrands_kernel(),
      rmhd_geometric_source_kernel(),
   )


def rhd_c2p_kernels() -> tuple[ScalarKernel | StructKernel, ...]:
   return srhd_palenzuela_kernels()


def rmhd_c2p_kernels() -> tuple[ScalarKernel | StructKernel, ...]:
   return (
      *srmhd_palenzuela_kernels(),
      *srmhd_entropy_kernels(),
      *srmhd_cold_kernels(),
      *srmhd_noble_2d_kernels(),
      *srmhd_reprimand_kernels(),
   )


def relativistic_kernel_groups() -> dict[str, tuple[ScalarKernel | StructKernel, ...]]:
   return {
      "core": relativistic_core_kernels(),
      "flux": relativistic_flux_kernels(),
      "sources": relativistic_source_kernels(),
      "c2p_rhd": rhd_c2p_kernels(),
      "c2p_rmhd": rmhd_c2p_kernels(),
   }


def relativistic_kernels() -> tuple[ScalarKernel | StructKernel, ...]:
   return tuple(
      kernel
      for kernels in relativistic_kernel_groups().values()
      for kernel in kernels
   )
