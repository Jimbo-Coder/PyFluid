from __future__ import annotations

import sympy as sp

from .kernel_equations import ScalarKernel, StructKernel
from .tensor_algebra import Tensor


def ctt_conformal_geometry_kernels() -> tuple[ScalarKernel | StructKernel, ...]:
   conformal_factor, energy_density, conformal_extrinsic_squared, mean_curvature_squared = (
      sp.symbols(
         "conformal_factor energy_density conformal_extrinsic_squared "
         "mean_curvature_squared",
         real=True,
      )
   )
   vector_gradient = Tensor.from_rows(tuple(
      tuple(sp.Symbol(f"d_vector{component + 1}_{direction + 1}", real=True)
            for direction in range(3))
      for component in range(3)
   ))
   conformal_extrinsic = (vector_gradient + vector_gradient.transpose()).deviatoric()
   components = tuple(
      conformal_extrinsic.component(first, second)
      for first, second in ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))
   )
   squared = conformal_extrinsic.double_contract(conformal_extrinsic)
   matter_source = (
      sp.Rational(1, 12) * mean_curvature_squared * conformal_factor**5
      - 2 * sp.pi * energy_density * conformal_factor**5
      - sp.Rational(1, 8) * conformal_extrinsic_squared / conformal_factor**7
   )
   residual = sp.Symbol("laplacian", real=True) - matter_source
   derivative = sp.diff(residual, conformal_factor)
   gradient_arguments = tuple(
      ("amrex::Real", f"d_vector{component + 1}_{direction + 1}")
      for component in range(3)
      for direction in range(3)
   )
   return (
      StructKernel(
         name="conformal_longitudinal_tensor",
         return_type="ConformalTraceFreeTensor",
         arguments=gradient_arguments,
         outputs=("a11", "a12", "a13", "a22", "a23", "a33", "squared"),
         expressions=(*components, squared),
      ),
      ScalarKernel(
         name="conformal_hamiltonian_source",
         arguments=(
            "conformal_factor",
            "energy_density",
            "conformal_extrinsic_squared",
            "mean_curvature_squared",
         ),
         expression=matter_source,
      ),
      ScalarKernel(
         name="conformal_hamiltonian_residual",
         arguments=(
            "conformal_factor",
            "laplacian",
            "energy_density",
            "conformal_extrinsic_squared",
            "mean_curvature_squared",
         ),
         expression=residual,
      ),
      ScalarKernel(
         name="conformal_hamiltonian_local_jacobian",
         arguments=(
            "conformal_factor",
            "energy_density",
            "conformal_extrinsic_squared",
            "mean_curvature_squared",
         ),
         expression=derivative,
      ),
   )


def xcts_kernels() -> tuple[ScalarKernel, ...]:
   conformal_factor, lapse_times_conformal_factor = sp.symbols(
      "conformal_factor lapse_times_conformal_factor", positive=True
   )
   energy_density, stress_trace, momentum_density = sp.symbols(
      "energy_density stress_trace momentum_density", real=True
   )
   conformal_extrinsic_squared = sp.Symbol(
      "conformal_extrinsic_squared", nonnegative=True
   )
   conformal_gradient = sp.symbols(
      "conformal_gradient1 conformal_gradient2 conformal_gradient3", real=True
   )
   lapse_conformal_gradient = sp.symbols(
      "lapse_conformal_gradient1 lapse_conformal_gradient2 "
      "lapse_conformal_gradient3",
      real=True,
   )
   longitudinal_row = sp.symbols(
      "conformal_extrinsic_i1 conformal_extrinsic_i2 "
      "conformal_extrinsic_i3",
      real=True,
   )

   conformal_lapse = lapse_times_conformal_factor / conformal_factor
   extrinsic_scale = (
      conformal_factor**7 / (2 * lapse_times_conformal_factor)
   )
   logarithmic_weight_gradient = tuple(
      7 * conformal_gradient[direction] / conformal_factor
      - lapse_conformal_gradient[direction] / lapse_times_conformal_factor
      for direction in range(3)
   )
   longitudinal_scale = 1 / extrinsic_scale
   # momentum_density is the covariant Eulerian S_i.  Raising it with
   # gamma^ij = psi^-4 delta^ij reduces 16 pi alpha psi^4 S^i to
   # 16 pi (alpha psi) S_i / psi.
   shift_source = (
      16
      * sp.pi
      * lapse_times_conformal_factor
      * momentum_density
      / conformal_factor
      - longitudinal_scale
      * sum(
         logarithmic_weight_gradient[direction] * longitudinal_row[direction]
         for direction in range(3)
      )
   )
   lapse_source = lapse_times_conformal_factor * (
      sp.Rational(7, 8)
      * conformal_extrinsic_squared
      / conformal_factor**8
      + 2
      * sp.pi
      * conformal_factor**4
      * (energy_density + 2 * stress_trace)
   )
   return (
      ScalarKernel(
         name="xcts_conformal_extrinsic_scale",
         arguments=("conformal_factor", "lapse_times_conformal_factor"),
         expression=extrinsic_scale,
      ),
      ScalarKernel(
         name="xcts_shift_source",
         arguments=(
            "conformal_factor",
            "lapse_times_conformal_factor",
            "momentum_density",
            "conformal_gradient1",
            "conformal_gradient2",
            "conformal_gradient3",
            "lapse_conformal_gradient1",
            "lapse_conformal_gradient2",
            "lapse_conformal_gradient3",
            "conformal_extrinsic_i1",
            "conformal_extrinsic_i2",
            "conformal_extrinsic_i3",
         ),
         expression=shift_source,
      ),
      ScalarKernel(
         name="xcts_lapse_conformal_source",
         arguments=(
            "conformal_factor",
            "lapse_times_conformal_factor",
            "energy_density",
            "stress_trace",
            "conformal_extrinsic_squared",
         ),
         expression=lapse_source,
      ),
      ScalarKernel(
         name="xcts_lapse_from_conformal_unknowns",
         arguments=("conformal_factor", "lapse_times_conformal_factor"),
         expression=conformal_lapse,
      ),
   )


def rotating_star_kernels() -> tuple[ScalarKernel, ...]:
   injection_energy, lapse = sp.symbols(
      "injection_energy lapse", positive=True
   )
   velocity_squared = sp.Symbol("velocity_squared", nonnegative=True)
   log_specific_enthalpy = (
      sp.log(injection_energy)
      - sp.log(lapse)
      - sp.Rational(1, 2) * sp.log(1 - velocity_squared)
   )
   omega, omega_center, differential_length, rotation_power = sp.symbols(
      "omega omega_center differential_length rotation_power", positive=True
   )
   j_constant = differential_length**2 * (omega_center - omega)
   q_power = differential_length**2 * omega * (
      (omega_center / omega) ** rotation_power - 1
   )
   j_constant_integral = -sp.Rational(1, 2) * differential_length**2 * (
      omega - omega_center
   ) ** 2
   q_power_integral = differential_length**2 * (
      omega_center**rotation_power
      * (omega ** (2 - rotation_power) - omega_center ** (2 - rotation_power))
      / (2 - rotation_power)
      - (omega**2 - omega_center**2) / 2
   )
   q_power_two_integral = differential_length**2 * (
      omega_center**2 * sp.log(omega / omega_center)
      - (omega**2 - omega_center**2) / 2
   )
   specific_angular_momentum, differential_length_b = sp.symbols(
      "specific_angular_momentum differential_length_b", positive=True
   )
   uryu_x = specific_angular_momentum / (
      differential_length**2 * omega_center
   )
   uryu_kappa = differential_length**2 / differential_length_b**2
   uryu_plus_13_omega = omega_center * (
      1 + uryu_kappa * uryu_x
   ) / (1 + uryu_x**4)
   uryu_plus_13_integral_primitive = sp.Rational(1, 4) * (
      2 * uryu_kappa * sp.atan(uryu_x**2)
      - sp.sqrt(2) * (
         sp.atan(1 - sp.sqrt(2) * uryu_x)
         - sp.atan(1 + sp.sqrt(2) * uryu_x)
      )
      + sp.sqrt(2) * sp.atanh(
         sp.sqrt(2) * uryu_x / (1 + uryu_x**2)
      )
   )
   uryu_plus_13_integral = (
      specific_angular_momentum * uryu_plus_13_omega
      - differential_length**2
      * omega_center**2
      * uryu_plus_13_integral_primitive
   )
   uryu_j_envelope_omega = omega_center * (
      1 + uryu_kappa * uryu_x
   ) * (1 - uryu_x)
   uryu_j_envelope_integral = differential_length**2 * omega_center**2 * (
      (uryu_kappa - 1) * uryu_x**2 / 2
      - 2 * uryu_kappa * uryu_x**3 / 3
   )
   return (
      ScalarKernel(
         name="uniform_rotation_log_specific_enthalpy",
         arguments=("injection_energy", "lapse", "velocity_squared"),
         expression=log_specific_enthalpy,
      ),
      ScalarKernel(
         name="j_constant_specific_angular_momentum",
         arguments=("omega", "omega_center", "differential_length"),
         expression=j_constant,
      ),
      ScalarKernel(
         name="q_power_specific_angular_momentum",
         arguments=(
            "omega",
            "omega_center",
            "differential_length",
            "rotation_power",
         ),
         expression=q_power,
      ),
      ScalarKernel(
         name="j_constant_rotation_integral",
         arguments=("omega", "omega_center", "differential_length"),
         expression=j_constant_integral,
      ),
      ScalarKernel(
         name="q_power_rotation_integral",
         arguments=(
            "omega",
            "omega_center",
            "differential_length",
            "rotation_power",
         ),
         expression=q_power_integral,
      ),
      ScalarKernel(
         name="q_power_two_rotation_integral",
         arguments=("omega", "omega_center", "differential_length"),
         expression=q_power_two_integral,
      ),
      ScalarKernel(
         name="uryu_plus_13_angular_velocity",
         arguments=(
            "specific_angular_momentum",
            "omega_center",
            "differential_length",
            "differential_length_b",
         ),
         expression=uryu_plus_13_omega,
      ),
      ScalarKernel(
         name="uryu_plus_13_angular_velocity_derivative",
         arguments=(
            "specific_angular_momentum",
            "omega_center",
            "differential_length",
            "differential_length_b",
         ),
         expression=sp.diff(
            uryu_plus_13_omega, specific_angular_momentum
         ),
      ),
      ScalarKernel(
         name="uryu_plus_13_rotation_integral",
         arguments=(
            "specific_angular_momentum",
            "omega_center",
            "differential_length",
            "differential_length_b",
         ),
         expression=uryu_plus_13_integral,
      ),
      ScalarKernel(
         name="uryu_j_envelope_p1_angular_velocity",
         arguments=(
            "specific_angular_momentum",
            "omega_center",
            "differential_length",
            "differential_length_b",
         ),
         expression=uryu_j_envelope_omega,
      ),
      ScalarKernel(
         name="uryu_j_envelope_p1_angular_velocity_derivative",
         arguments=(
            "specific_angular_momentum",
            "omega_center",
            "differential_length",
            "differential_length_b",
         ),
         expression=sp.diff(
            uryu_j_envelope_omega, specific_angular_momentum
         ),
      ),
      ScalarKernel(
         name="uryu_j_envelope_p1_rotation_integral",
         arguments=(
            "specific_angular_momentum",
            "omega_center",
            "differential_length",
            "differential_length_b",
         ),
         expression=uryu_j_envelope_integral,
      ),
   )


def equilibrium_torus_kernels() -> tuple[ScalarKernel, ...]:
   metric_tt, metric_t_phi, metric_phi_phi, specific_angular_momentum = (
      sp.symbols(
         "metric_tt metric_t_phi metric_phi_phi specific_angular_momentum",
         real=True,
      )
   )
   angular_velocity = -(
      metric_t_phi + specific_angular_momentum * metric_tt
   ) / (
      metric_phi_phi + specific_angular_momentum * metric_t_phi
   )
   angular_velocity_argument = sp.Symbol("angular_velocity", real=True)
   normalization_squared = -(
      metric_tt
      + 2 * angular_velocity_argument * metric_t_phi
      + angular_velocity_argument**2 * metric_phi_phi
   )
   normalization_squared_argument = sp.Symbol(
      "normalization_squared", positive=True
   )
   minus_covariant_time_velocity = -(
      metric_tt + angular_velocity_argument * metric_t_phi
   ) / sp.sqrt(normalization_squared_argument)
   inner_minus_covariant_time_velocity, local_minus_covariant_time_velocity = (
      sp.symbols(
         "inner_minus_covariant_time_velocity "
         "local_minus_covariant_time_velocity",
         positive=True,
      )
   )
   return (
      ScalarKernel(
         name="constant_l_torus_angular_velocity",
         arguments=(
            "metric_tt",
            "metric_t_phi",
            "metric_phi_phi",
            "specific_angular_momentum",
         ),
         expression=angular_velocity,
      ),
      ScalarKernel(
         name="constant_l_torus_normalization_squared",
         arguments=(
            "metric_tt",
            "metric_t_phi",
            "metric_phi_phi",
            "angular_velocity",
         ),
         expression=normalization_squared,
      ),
      ScalarKernel(
         name="constant_l_torus_minus_covariant_time_velocity",
         arguments=(
            "metric_tt",
            "metric_t_phi",
            "angular_velocity",
            "normalization_squared",
         ),
         expression=minus_covariant_time_velocity,
      ),
      ScalarKernel(
         name="constant_l_torus_log_specific_enthalpy",
         arguments=(
            "inner_minus_covariant_time_velocity",
            "local_minus_covariant_time_velocity",
         ),
         expression=sp.log(inner_minus_covariant_time_velocity)
         - sp.log(local_minus_covariant_time_velocity),
      ),
   )


def ctt_momentum_kernels() -> tuple[ScalarKernel, ...]:
   conformal_factor, momentum_density, mean_curvature_gradient = sp.symbols(
      "conformal_factor momentum_density mean_curvature_gradient", real=True
   )
   return (
      ScalarKernel(
         name="conformal_momentum_source",
         arguments=(
            "conformal_factor",
            "momentum_density",
            "mean_curvature_gradient",
         ),
         expression=(
            sp.Rational(2, 3)
            * conformal_factor**6
            * mean_curvature_gradient
            + 8 * sp.pi * conformal_factor**10 * momentum_density
         ),
      ),
   )


def cttk_kernels() -> tuple[ScalarKernel, ...]:
   conformal_factor, laplacian, energy_density, conformal_extrinsic_squared = (
      sp.symbols(
         "conformal_factor laplacian energy_density conformal_extrinsic_squared",
         real=True,
      )
   )
   mean_curvature_squared = (
      12 * laplacian / conformal_factor**5
      + sp.Rational(3, 2)
      * conformal_extrinsic_squared
      / conformal_factor**12
      + 24 * sp.pi * energy_density
   )
   return (
      ScalarKernel(
         name="cttk_mean_curvature_squared",
         arguments=(
            "conformal_factor",
            "laplacian",
            "energy_density",
            "conformal_extrinsic_squared",
         ),
         expression=mean_curvature_squared,
      ),
      ScalarKernel(
         name="cttk_hybrid_mean_curvature_squared",
         arguments=("energy_density",),
         expression=24 * sp.pi * energy_density,
      ),
   )


def ctt_canonical_state_kernels() -> tuple[StructKernel, ...]:
   conformal_factor = sp.Symbol("conformal_factor", positive=True)
   conformal_extrinsic = tuple(
      sp.Symbol(f"conformal_extrinsic_{component}", real=True)
      for component in ("11", "12", "13", "22", "23", "33")
   )
   mean_curvature = sp.Symbol("mean_curvature", real=True)
   lapse = sp.Symbol("lapse", positive=True)
   shift = tuple(
      sp.Symbol(f"shift{component}", real=True) for component in range(1, 4)
   )
   inverse_conformal_factor_six = conformal_factor**-6
   return (
      StructKernel(
         name="ctt_canonical_z4c_state",
         return_type="Z4cState",
         arguments=(
            ("amrex::Real", "conformal_factor"),
            *(("amrex::Real", f"conformal_extrinsic_{component}")
              for component in ("11", "12", "13", "22", "23", "33")),
            ("amrex::Real", "mean_curvature"),
            ("amrex::Real", "lapse"),
            *(("amrex::Real", f"shift{component}") for component in range(1, 4)),
         ),
         outputs=(
            "chi",
            "gamma_tilde11",
            "gamma_tilde12",
            "gamma_tilde13",
            "gamma_tilde22",
            "gamma_tilde23",
            "gamma_tilde33",
            "k_hat",
            "a_tilde11",
            "a_tilde12",
            "a_tilde13",
            "a_tilde22",
            "a_tilde23",
            "a_tilde33",
            "gamma_tilde1",
            "gamma_tilde2",
            "gamma_tilde3",
            "theta",
            "lapse",
            "shift1",
            "shift2",
            "shift3",
         ),
         expressions=(
            conformal_factor**-4,
            sp.Integer(1), sp.Integer(0), sp.Integer(0),
            sp.Integer(1), sp.Integer(0), sp.Integer(1),
            mean_curvature,
            *(component * inverse_conformal_factor_six
              for component in conformal_extrinsic),
            sp.Integer(0), sp.Integer(0), sp.Integer(0),
            sp.Integer(0),
            lapse,
            *shift,
         ),
      ),
   )


def manufactured_ctt_momentum_kernels() -> tuple[ScalarKernel, ...]:
   x, y, z, amplitude = sp.symbols("x y z amplitude", real=True)
   coordinates = (x, y, z)
   vector_potential = sp.Matrix((
      amplitude * sp.prod(sp.sin(sp.pi * coordinate) for coordinate in coordinates),
      0,
      0,
   ))
   divergence = sum(
      sp.diff(vector_potential[direction], coordinates[direction])
      for direction in range(3)
   )
   vector_laplacian = sp.Matrix([
      sum(
         sp.diff(vector_potential[component], coordinate, 2)
         for coordinate in coordinates
      ) + sp.diff(divergence, coordinates[component]) / 3
      for component in range(3)
   ])
   kernels: list[ScalarKernel] = []
   for component in range(3):
      kernels.extend((
         ScalarKernel(
            name=f"ctt_manufactured_vector_potential_{component + 1}",
            arguments=("x", "y", "z", "amplitude"),
            expression=vector_potential[component],
         ),
         ScalarKernel(
            name=f"ctt_manufactured_momentum_source_{component + 1}",
            arguments=("x", "y", "z", "amplitude"),
            expression=vector_laplacian[component],
         ),
      ))
   return tuple(kernels)


def manufactured_coupled_ctt_kernels() -> tuple[ScalarKernel, ...]:
   x, y, z, conformal_amplitude, vector_amplitude = sp.symbols(
      "x y z conformal_amplitude vector_amplitude", real=True
   )
   coordinates = (x, y, z)
   profile = sp.prod(sp.sin(sp.pi * coordinate) for coordinate in coordinates)
   conformal_factor = 1 + conformal_amplitude * profile
   vector_potential = (
      vector_amplitude * profile,
      sp.Integer(0),
      sp.Integer(0),
   )
   vector_gradient = Tensor.from_rows(tuple(
      tuple(sp.diff(vector_potential[component], coordinates[direction])
            for direction in range(3))
      for component in range(3)
   ))
   conformal_extrinsic = (vector_gradient + vector_gradient.transpose()).deviatoric()
   conformal_extrinsic_squared = conformal_extrinsic.double_contract(
      conformal_extrinsic
   )
   conformal_laplacian = sum(
      sp.diff(conformal_factor, coordinate, 2) for coordinate in coordinates
   )
   energy_density = (
      -conformal_laplacian
      - conformal_extrinsic_squared / (8 * conformal_factor**7)
   ) / (2 * sp.pi * conformal_factor**5)
   divergence = sum(
      sp.diff(vector_potential[direction], coordinates[direction])
      for direction in range(3)
   )
   vector_laplacian = tuple(
      sum(
         sp.diff(vector_potential[component], coordinate, 2)
         for coordinate in coordinates
      ) + sp.diff(divergence, coordinates[component]) / 3
      for component in range(3)
   )
   kernels = [
      ScalarKernel(
         name="ctt_manufactured_coupled_energy_density",
         arguments=("x", "y", "z", "conformal_amplitude", "vector_amplitude"),
         expression=energy_density,
      ),
   ]
   for component in range(3):
      kernels.append(ScalarKernel(
         name=f"ctt_manufactured_coupled_momentum_density_{component + 1}",
         arguments=("x", "y", "z", "conformal_amplitude", "vector_amplitude"),
         expression=(
            vector_laplacian[component]
            / (8 * sp.pi * conformal_factor**10)
         ),
      ))
   return tuple(kernels)


def manufactured_cttk_kernels() -> tuple[ScalarKernel, ...]:
   x, y, z, mean_curvature_background, mean_curvature_amplitude, vector_amplitude = (
      sp.symbols(
         "x y z mean_curvature_background mean_curvature_amplitude "
         "vector_amplitude",
         real=True,
      )
   )
   coordinates = (x, y, z)
   profile = sp.prod(sp.sin(2 * sp.pi * coordinate) for coordinate in coordinates)
   mean_curvature = mean_curvature_background + mean_curvature_amplitude * profile
   vector_potential = (
      vector_amplitude * profile,
      sp.Integer(0),
      sp.Integer(0),
   )
   vector_gradient = Tensor.from_rows(tuple(
      tuple(sp.diff(vector_potential[component], coordinates[direction])
            for direction in range(3))
      for component in range(3)
   ))
   conformal_extrinsic = (vector_gradient + vector_gradient.transpose()).deviatoric()
   conformal_extrinsic_squared = conformal_extrinsic.double_contract(
      conformal_extrinsic
   )
   energy_density = (
      mean_curvature**2
      - sp.Rational(3, 2) * conformal_extrinsic_squared
   ) / (24 * sp.pi)
   divergence = sum(
      sp.diff(vector_potential[direction], coordinates[direction])
      for direction in range(3)
   )
   vector_laplacian = tuple(
      sum(
         sp.diff(vector_potential[component], coordinate, 2)
         for coordinate in coordinates
      ) + sp.diff(divergence, coordinates[component]) / 3
      for component in range(3)
   )
   kernels: list[ScalarKernel] = [
      ScalarKernel(
         name="cttk_manufactured_mean_curvature",
         arguments=(
            "x",
            "y",
            "z",
            "mean_curvature_background",
            "mean_curvature_amplitude",
         ),
         expression=mean_curvature,
      ),
      ScalarKernel(
         name="cttk_manufactured_energy_density",
         arguments=(
            "x",
            "y",
            "z",
            "mean_curvature_background",
            "mean_curvature_amplitude",
            "vector_amplitude",
         ),
         expression=energy_density,
      ),
   ]
   for component in range(3):
      kernels.extend((
         ScalarKernel(
            name=f"cttk_manufactured_vector_potential_{component + 1}",
            arguments=("x", "y", "z", "vector_amplitude"),
            expression=vector_potential[component],
         ),
         ScalarKernel(
            name=f"cttk_manufactured_momentum_density_{component + 1}",
            arguments=(
               "x",
               "y",
               "z",
               "mean_curvature_background",
               "mean_curvature_amplitude",
               "vector_amplitude",
            ),
            expression=(
               vector_laplacian[component]
               - sp.Rational(2, 3)
               * sp.diff(mean_curvature, coordinates[component])
            ) / (8 * sp.pi),
         ),
      ))
   return tuple(kernels)


def manufactured_ctt_kernels() -> tuple[ScalarKernel, ...]:
   coordinates = sp.symbols("x y z", real=True)
   amplitude = sp.symbols("amplitude", real=True)
   kernels: list[ScalarKernel] = []
   for dimension in range(1, 4):
      profile = sp.prod(
         sp.sin(sp.pi * coordinates[direction])
         for direction in range(dimension)
      )
      conformal_factor = 1 + amplitude * profile
      energy_density = (
         dimension * sp.pi * amplitude * profile / (2 * conformal_factor**5)
      )
      kernels.extend((
         ScalarKernel(
            name=f"ctt_manufactured_conformal_factor_{dimension}d",
            arguments=("x", "y", "z", "amplitude"),
            expression=conformal_factor,
         ),
         ScalarKernel(
            name=f"ctt_manufactured_energy_density_{dimension}d",
            arguments=("x", "y", "z", "amplitude"),
            expression=energy_density,
         ),
      ))
   return tuple(kernels)


def initial_data_kernels() -> tuple[ScalarKernel | StructKernel, ...]:
   return (
      *ctt_conformal_geometry_kernels(),
      *xcts_kernels(),
      *rotating_star_kernels(),
      *equilibrium_torus_kernels(),
      *ctt_momentum_kernels(),
      *cttk_kernels(),
      *ctt_canonical_state_kernels(),
      *manufactured_ctt_kernels(),
      *manufactured_ctt_momentum_kernels(),
      *manufactured_coupled_ctt_kernels(),
      *manufactured_cttk_kernels(),
      *manufactured_poisson_kernels(),
   )


def manufactured_poisson_kernels() -> tuple[ScalarKernel, ...]:
   coordinates = sp.symbols("x y z", real=True)
   kernels: list[ScalarKernel] = []
   for dimension in range(1, 4):
      solution = sp.prod(
         sp.sin(sp.pi * coordinates[direction])
         for direction in range(dimension)
      )
      kernels.extend((
         ScalarKernel(
            name=f"initial_data_poisson_solution_{dimension}d",
            arguments=("x", "y", "z"),
            expression=solution,
         ),
         ScalarKernel(
            name=f"initial_data_poisson_rhs_{dimension}d",
            arguments=("x", "y", "z"),
            expression=-dimension * sp.pi**2 * solution,
         ),
      ))
   return tuple(kernels)
