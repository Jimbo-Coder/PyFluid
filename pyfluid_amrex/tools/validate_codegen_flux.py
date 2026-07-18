#!/usr/bin/env python3
from __future__ import annotations

import math
import random
import sys
import sympy as sp
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CODEGEN = ROOT / "pyfluid_amrex" / "codegen"
FLUX_HEADER = ROOT / "pyfluid_amrex" / "include" / "pyfluid_amrex" / "kernels_generated" / "ideal_mhd_flux_cartesian.hpp"
EOS_HEADER = ROOT / "pyfluid_amrex" / "include" / "pyfluid_amrex" / "kernels_generated" / "eos.hpp"
OPERATOR_HEADER = ROOT / "pyfluid_amrex" / "include" / "pyfluid_amrex" / "kernels_generated" / "scalar_ops.hpp"
RIEMANN_HEADER = ROOT / "pyfluid_amrex" / "include" / "pyfluid_amrex" / "kernels_generated" / "riemann_ops.hpp"
VECTOR_HEADER = ROOT / "pyfluid_amrex" / "include" / "pyfluid_amrex" / "kernels_generated" / "vector_ops.hpp"
METRIC_HEADER = ROOT / "pyfluid_amrex" / "include" / "pyfluid_amrex" / "kernels_generated" / "coordinate_metrics.hpp"
CHARACTERISTIC_HEADER = (
   ROOT / "pyfluid_amrex" / "include" / "pyfluid_amrex" / "kernels_generated" / "characteristic_transforms.hpp"
)
RELATIVISTIC_HEADER = (
   ROOT / "pyfluid_amrex" / "include" / "pyfluid_amrex" / "kernels_generated" / "relativistic_fluid.hpp"
)
RELATIVISTIC_HEADER_DIR = RELATIVISTIC_HEADER.parent / "relativistic"
Z4C_HEADER = (
   ROOT / "pyfluid_amrex" / "include" / "pyfluid_amrex" / "kernels_generated" / "z4c.hpp"
)
Z4C_HEADER_DIR = Z4C_HEADER.parent / "z4c"
INITIAL_DATA_HEADER = (
   ROOT / "pyfluid_amrex" / "include" / "pyfluid_amrex" / "kernels_generated" / "initial_data.hpp"
)
sys.path.insert(0, str(CODEGEN))

from pyfluid_codegen.emit_cpp import (  # noqa: E402
   render_characteristic_header,
   render_eos_header,
   render_flux_header,
   render_initial_data_header,
   render_metric_header,
   render_operator_header,
   render_relativistic_header,
   render_relativistic_headers,
   render_riemann_header,
   render_vector_header,
   render_z4c_header,
   render_z4c_headers,
)
from pyfluid_codegen.characteristic_equations import characteristic_kernels  # noqa: E402
from pyfluid_codegen.eos_equations import analytic_equations_of_state, eos_kernels  # noqa: E402
from pyfluid_codegen.coordinates import orthogonal_frames  # noqa: E402
from pyfluid_codegen.flux_equations import hydro_flux_equations, mhd_flux_equations  # noqa: E402
from pyfluid_codegen.metric_equations import metric_kernels  # noqa: E402
from pyfluid_codegen.operator_equations import (  # noqa: E402
   diffusive_flux_equation,
   explicit_diffusion_update_equation,
   finite_volume_laplacian_face_weight_equation,
   implicit_relaxed_equation,
   implicit_residual_equation,
   laplacian_equation,
)
from pyfluid_codegen.riemann_equations import riemann_flux_kernels  # noqa: E402
from pyfluid_codegen.relativistic_equations import relativistic_kernels  # noqa: E402
from pyfluid_codegen.physical_system import (  # noqa: E402
   Hydrodynamics,
   Magnetohydrodynamics,
   NewtonianViscosity,
)
from pyfluid_codegen.tensor_algebra import (  # noqa: E402
   IndexPosition,
   IndexedTensor,
   MetricTensor,
   OrthogonalOperators,
   ScalarField,
   Tensor,
   Vector,
   Vector4,
   VectorField,
)
from pyfluid_codegen.vector_equations import vector_operator_kernels  # noqa: E402


def hydro_reference(values: dict[str, float], direction: int) -> list[float]:
   rho = values["state.rho"]
   mom = [values["state.mom1"], values["state.mom2"], values["state.mom3"]]
   energy = values["state.energy"]
   pressure = values["pressure"]
   vn = mom[direction] / rho
   return [
      rho * vn,
      mom[0] * vn + (pressure if direction == 0 else 0.0),
      mom[1] * vn + (pressure if direction == 1 else 0.0),
      mom[2] * vn + (pressure if direction == 2 else 0.0),
      (energy + pressure) * vn,
   ]


def mhd_reference(values: dict[str, float], direction: int) -> list[float]:
   rho = values["state.rho"]
   mom = [values["state.mom1"], values["state.mom2"], values["state.mom3"]]
   energy = values["state.energy"]
   pressure = values["pressure"]
   b = [values["state.b1"], values["state.b2"], values["state.b3"]]
   v = [component / rho for component in mom]
   vn = v[direction]
   bn = b[direction]
   b2 = b[0] * b[0] + b[1] * b[1] + b[2] * b[2]
   ptot = pressure + 0.5 * b2
   vdotb = v[0] * b[0] + v[1] * b[1] + v[2] * b[2]
   out = [
      rho * vn,
      mom[0] * vn - b[0] * bn,
      mom[1] * vn - b[1] * bn,
      mom[2] * vn - b[2] * bn,
      (energy + ptot) * vn - bn * vdotb,
      b[0] * vn - bn * v[0],
      b[1] * vn - bn * v[1],
      b[2] * vn - bn * v[2],
   ]
   out[1 + direction] += ptot
   out[5 + direction] = 0.0
   return out


def evaluate(expressions, values: dict[str, float]) -> list[float]:
   substitutions = {
      symbol: values[str(symbol)]
      for expr in expressions
      for symbol in expr.free_symbols
      if str(symbol) in values
   }
   return [float(expr.subs(substitutions)) for expr in expressions]


def compile_evaluator(expressions):
   symbols = sorted(
      {symbol for expression in expressions for symbol in expression.free_symbols},
      key=str,
   )
   function = sp.lambdify(symbols, expressions, modules="math", cse=True)

   def evaluate_compiled(values: dict[str, float]) -> list[float]:
      return [float(value) for value in function(*(values[str(symbol)] for symbol in symbols))]

   return evaluate_compiled


def assert_close(actual: list[float], expected: list[float], label: str) -> None:
   for index, (a, e) in enumerate(zip(actual, expected, strict=True)):
      scale = max(1.0, abs(e))
      if not math.isfinite(a) or abs(a - e) > 1.0e-12 * scale:
         raise AssertionError(f"{label}[{index}] actual={a:.17e} expected={e:.17e}")


def invert_symmetric_3x3(matrix: list[list[float]]) -> tuple[list[list[float]], float]:
   a, b, c = matrix[0]
   _, d, e = matrix[1]
   _, _, f = matrix[2]
   cofactor11 = d * f - e * e
   cofactor12 = c * e - b * f
   cofactor13 = b * e - c * d
   cofactor22 = a * f - c * c
   cofactor23 = b * c - a * e
   cofactor33 = a * d - b * b
   determinant = a * cofactor11 + b * cofactor12 + c * cofactor13
   inverse = [
      [cofactor11 / determinant, cofactor12 / determinant, cofactor13 / determinant],
      [cofactor12 / determinant, cofactor22 / determinant, cofactor23 / determinant],
      [cofactor13 / determinant, cofactor23 / determinant, cofactor33 / determinant],
   ]
   return inverse, determinant


def random_relativistic_values(rng: random.Random) -> dict[str, float]:
   lower = [
      [rng.uniform(0.7, 1.4), 0.0, 0.0],
      [rng.uniform(-0.25, 0.25), rng.uniform(0.7, 1.4), 0.0],
      [rng.uniform(-0.25, 0.25), rng.uniform(-0.25, 0.25), rng.uniform(0.7, 1.4)],
   ]
   gamma = [[sum(lower[i][k] * lower[j][k] for k in range(3)) for j in range(3)] for i in range(3)]
   inverse, determinant = invert_symmetric_3x3(gamma)
   velocity = [rng.uniform(-0.45, 0.45) for _ in range(3)]
   velocity_squared = sum(gamma[i][j] * velocity[i] * velocity[j] for i in range(3) for j in range(3))
   if velocity_squared >= 0.75:
      scale = math.sqrt(0.65 / velocity_squared)
      velocity = [scale * value for value in velocity]
   values = {
      "primitive.rho": rng.uniform(0.1, 8.0),
      "primitive.eps": rng.uniform(0.0, 2.0),
      "primitive.pressure": rng.uniform(0.01, 4.0),
      "primitive.v1": velocity[0],
      "primitive.v2": velocity[1],
      "primitive.v3": velocity[2],
      "primitive.b1": rng.uniform(-2.0, 2.0),
      "primitive.b2": rng.uniform(-2.0, 2.0),
      "primitive.b3": rng.uniform(-2.0, 2.0),
      "gamma": 5.0 / 3.0,
      "metric.lapse": rng.uniform(0.3, 1.4),
      "metric.shift1": rng.uniform(-0.3, 0.3),
      "metric.shift2": rng.uniform(-0.3, 0.3),
      "metric.shift3": rng.uniform(-0.3, 0.3),
      "metric.sqrt_det_gamma": math.sqrt(determinant),
   }
   for i in range(3):
      for j in range(i, 3):
         values[f"metric.gamma{i + 1}{j + 1}"] = gamma[i][j]
         values[f"metric.inv_gamma{i + 1}{j + 1}"] = inverse[i][j]
   for direction in range(1, 4):
      values[f"derivatives.d{direction}_lapse"] = rng.uniform(-0.2, 0.2)
      for component in range(1, 4):
         values[f"derivatives.d{direction}_shift{component}"] = rng.uniform(-0.2, 0.2)
      for first in range(1, 4):
         for second in range(first, 4):
            values[f"derivatives.d{direction}_gamma{first}{second}"] = rng.uniform(-0.2, 0.2)
   for first in range(1, 4):
      for second in range(first, 4):
         values[f"derivatives.extrinsic{first}{second}"] = rng.uniform(-0.2, 0.2)
   return values


def relativistic_wave_speed_reference(
   velocity: list[float],
   characteristic_speed_squared: float,
   direction: int,
) -> list[float]:
   velocity_squared = sum(component * component for component in velocity)
   normal_velocity = velocity[direction]
   radicand = (1.0 - velocity_squared) * (
      1.0
      - velocity_squared * characteristic_speed_squared
      - normal_velocity**2 * (1.0 - characteristic_speed_squared)
   )
   denominator = 1.0 - velocity_squared * characteristic_speed_squared
   common = normal_velocity * (1.0 - characteristic_speed_squared)
   spread = math.sqrt(characteristic_speed_squared * max(0.0, radicand))
   return [(common - spread) / denominator, (common + spread) / denominator]


def srhd_wave_speed_reference(values: dict[str, float], direction: int) -> list[float]:
   gamma = values["gamma"]
   eps = values["primitive.eps"]
   pressure = (gamma - 1.0) * values["primitive.rho"] * eps
   enthalpy = 1.0 + eps + pressure / values["primitive.rho"]
   sound_speed_squared = gamma * pressure / (values["primitive.rho"] * enthalpy)
   velocity = [values[f"primitive.v{component + 1}"] for component in range(3)]
   return relativistic_wave_speed_reference(velocity, sound_speed_squared, direction)


def relativistic_reference(values: dict[str, float]) -> tuple[list[float], list[list[float]]]:
   rho = values["primitive.rho"]
   eps = values["primitive.eps"]
   pressure = values["primitive.pressure"]
   velocity = [values[f"primitive.v{i + 1}"] for i in range(3)]
   gamma = [[values[f"metric.gamma{min(i, j) + 1}{max(i, j) + 1}"] for j in range(3)] for i in range(3)]
   covariant_velocity = [sum(gamma[i][j] * velocity[j] for j in range(3)) for i in range(3)]
   velocity_squared = sum(velocity[i] * covariant_velocity[i] for i in range(3))
   lorentz_factor = 1.0 / math.sqrt(1.0 - velocity_squared)
   enthalpy = 1.0 + eps + pressure / rho
   sqrt_det_gamma = values["metric.sqrt_det_gamma"]
   dens = sqrt_det_gamma * rho * lorentz_factor
   momentum = [
      sqrt_det_gamma * rho * enthalpy * lorentz_factor**2 * component
      for component in covariant_velocity
   ]
   tau = sqrt_det_gamma * (rho * enthalpy * lorentz_factor**2 - pressure - rho * lorentz_factor)
   conserved = [dens, *momentum, tau]
   lapse = values["metric.lapse"]
   transport = [lapse * velocity[i] - values[f"metric.shift{i + 1}"] for i in range(3)]
   pressure_transport = lapse * sqrt_det_gamma * pressure
   fluxes = []
   for direction in range(3):
      fluxes.append([
         dens * transport[direction],
         *(momentum[component] * transport[direction] + (pressure_transport if component == direction else 0.0)
           for component in range(3)),
         tau * transport[direction] + pressure_transport * velocity[direction],
      ])
   return conserved, fluxes


def relativistic_mhd_reference(
   values: dict[str, float],
) -> tuple[list[float], list[list[float]]]:
   rho = values["primitive.rho"]
   eps = values["primitive.eps"]
   pressure = values["primitive.pressure"]
   velocity = [values[f"primitive.v{i + 1}"] for i in range(3)]
   magnetic = [values[f"primitive.b{i + 1}"] for i in range(3)]
   metric = [[values[f"metric.gamma{min(i, j) + 1}{max(i, j) + 1}"] for j in range(3)] for i in range(3)]
   velocity_covariant = [sum(metric[i][j] * velocity[j] for j in range(3)) for i in range(3)]
   magnetic_covariant = [sum(metric[i][j] * magnetic[j] for j in range(3)) for i in range(3)]
   velocity_squared = sum(velocity[i] * velocity_covariant[i] for i in range(3))
   magnetic_squared = sum(magnetic[i] * magnetic_covariant[i] for i in range(3))
   magnetic_velocity = sum(magnetic[i] * velocity_covariant[i] for i in range(3))
   lorentz_factor = 1.0 / math.sqrt(1.0 - velocity_squared)
   alpha_b0 = lorentz_factor * magnetic_velocity
   comoving_covariant = [
      magnetic_covariant[i] / lorentz_factor + alpha_b0 * velocity_covariant[i]
      for i in range(3)
   ]
   comoving_squared = (magnetic_squared + alpha_b0**2) / lorentz_factor**2
   enthalpy = 1.0 + eps + pressure / rho
   sqrt_det_gamma = values["metric.sqrt_det_gamma"]
   dens = sqrt_det_gamma * rho * lorentz_factor
   fluid_enthalpy = sqrt_det_gamma * rho * enthalpy * lorentz_factor**2
   total_enthalpy = fluid_enthalpy + sqrt_det_gamma * (
      magnetic_squared + alpha_b0**2
   )
   momentum = [
      total_enthalpy * velocity_covariant[i]
      - sqrt_det_gamma * alpha_b0 * comoving_covariant[i]
      for i in range(3)
   ]
   total_pressure = pressure + 0.5 * comoving_squared
   tau = fluid_enthalpy - dens + sqrt_det_gamma * (
      magnetic_squared - total_pressure
   )
   densitized_magnetic = [sqrt_det_gamma * component for component in magnetic]
   conserved = [dens, *momentum, tau, *densitized_magnetic]
   lapse = values["metric.lapse"]
   transport = [lapse * velocity[i] - values[f"metric.shift{i + 1}"] for i in range(3)]
   fluxes = []
   for direction in range(3):
      momentum_flux = [
         momentum[component] * transport[direction]
         + lapse
         * sqrt_det_gamma
         * (
            (total_pressure if component == direction else 0.0)
            - comoving_covariant[component] * magnetic[direction] / lorentz_factor
         )
         for component in range(3)
      ]
      energy_flux = (
         tau * transport[direction]
         + lapse
         * sqrt_det_gamma
         * (
            total_pressure * velocity[direction]
            - alpha_b0 * magnetic[direction] / lorentz_factor
         )
      )
      magnetic_flux = [
         densitized_magnetic[component] * transport[direction]
         - transport[component] * densitized_magnetic[direction]
         for component in range(3)
      ]
      fluxes.append([
         dens * transport[direction],
         *momentum_flux,
         energy_flux,
         *magnetic_flux,
      ])
   return conserved, fluxes


def relativistic_stress_energy_reference(
   values: dict[str, float],
   magnetized: bool,
) -> list[float]:
   rho = values["primitive.rho"]
   eps = values["primitive.eps"]
   pressure = values["primitive.pressure"]
   lapse = values["metric.lapse"]
   shift = [values[f"metric.shift{i + 1}"] for i in range(3)]
   velocity = [values[f"primitive.v{i + 1}"] for i in range(3)]
   gamma_covariant = [
      [values[f"metric.gamma{min(i, j) + 1}{max(i, j) + 1}"] for j in range(3)]
      for i in range(3)
   ]
   gamma_contravariant = [
      [
         values[f"metric.inv_gamma{min(i, j) + 1}{max(i, j) + 1}"]
         for j in range(3)
      ]
      for i in range(3)
   ]
   velocity_covariant = [
      sum(gamma_covariant[i][j] * velocity[j] for j in range(3))
      for i in range(3)
   ]
   velocity_squared = sum(
      velocity[i] * velocity_covariant[i] for i in range(3)
   )
   lorentz_factor = 1.0 / math.sqrt(1.0 - velocity_squared)
   four_velocity = [
      lorentz_factor / lapse,
      *(
         lorentz_factor * (velocity[i] - shift[i] / lapse)
         for i in range(3)
      ),
   ]
   inverse_lapse_squared = 1.0 / lapse**2
   spacetime_inverse = [
      [-inverse_lapse_squared, *(value * inverse_lapse_squared for value in shift)],
      *[
         [
            shift[i] * inverse_lapse_squared,
            *(
               gamma_contravariant[i][j]
               - shift[i] * shift[j] * inverse_lapse_squared
               for j in range(3)
            ),
         ]
         for i in range(3)
      ],
   ]
   specific_enthalpy = 1.0 + eps + pressure / rho
   comoving_magnetic = [0.0] * 4
   comoving_magnetic_squared = 0.0
   if magnetized:
      magnetic = [values[f"primitive.b{i + 1}"] for i in range(3)]
      magnetic_covariant = [
         sum(gamma_covariant[i][j] * magnetic[j] for j in range(3))
         for i in range(3)
      ]
      alpha_b0 = lorentz_factor * sum(
         magnetic[i] * velocity_covariant[i] for i in range(3)
      )
      comoving_magnetic = [
         alpha_b0 / lapse,
         *(
            magnetic[i] / lorentz_factor
            + alpha_b0 * (velocity[i] - shift[i] / lapse)
            for i in range(3)
         ),
      ]
      comoving_magnetic_squared = (
         sum(magnetic[i] * magnetic_covariant[i] for i in range(3))
         + alpha_b0**2
      ) / lorentz_factor**2
   total_enthalpy = rho * specific_enthalpy + comoving_magnetic_squared
   total_pressure = pressure + 0.5 * comoving_magnetic_squared
   tensor = [
      [
         total_enthalpy * four_velocity[i] * four_velocity[j]
         + total_pressure * spacetime_inverse[i][j]
         - comoving_magnetic[i] * comoving_magnetic[j]
         for j in range(4)
      ]
      for i in range(4)
   ]
   return [
      tensor[0][0],
      tensor[0][1],
      tensor[0][2],
      tensor[0][3],
      tensor[1][1],
      tensor[1][2],
      tensor[1][3],
      tensor[2][2],
      tensor[2][3],
      tensor[3][3],
   ]


def geometric_source_reference(
   values: dict[str, float],
   magnetized: bool,
) -> list[float]:
   conserved = (
      relativistic_mhd_reference(values)[0]
      if magnetized
      else relativistic_reference(values)[0]
   )
   packed_stress = relativistic_stress_energy_reference(values, magnetized)
   stress = [
      [packed_stress[0], packed_stress[1], packed_stress[2], packed_stress[3]],
      [packed_stress[1], packed_stress[4], packed_stress[5], packed_stress[6]],
      [packed_stress[2], packed_stress[5], packed_stress[7], packed_stress[8]],
      [packed_stress[3], packed_stress[6], packed_stress[8], packed_stress[9]],
   ]
   shift = [values[f"metric.shift{component + 1}"] for component in range(3)]
   spatial_stress = [
      [
         stress[first + 1][second + 1]
         + shift[first] * stress[0][second + 1]
         + shift[second] * stress[first + 1][0]
         + shift[first] * shift[second] * stress[0][0]
         for second in range(3)
      ]
      for first in range(3)
   ]
   sqrt_det = values["metric.sqrt_det_gamma"]
   momentum_covariant = [value / sqrt_det for value in conserved[1:4]]
   inverse_metric = [
      [
         values[f"metric.inv_gamma{min(first, second) + 1}{max(first, second) + 1}"]
         for second in range(3)
      ]
      for first in range(3)
   ]
   momentum_contravariant = [
      sum(inverse_metric[first][second] * momentum_covariant[second] for second in range(3))
      for first in range(3)
   ]
   eulerian_energy = (conserved[0] + conserved[4]) / sqrt_det
   momentum_source = []
   for direction in range(3):
      shift_term = sum(
         momentum_covariant[component]
         * values[f"derivatives.d{direction + 1}_shift{component + 1}"]
         for component in range(3)
      )
      metric_term = sum(
         spatial_stress[first][second]
         * values[
            f"derivatives.d{direction + 1}_gamma{min(first, second) + 1}{max(first, second) + 1}"
         ]
         for first in range(3)
         for second in range(3)
      )
      momentum_source.append(
         sqrt_det
         * (
            -eulerian_energy * values[f"derivatives.d{direction + 1}_lapse"]
            + shift_term
            + 0.5 * values["metric.lapse"] * metric_term
         )
      )
   extrinsic_term = sum(
      spatial_stress[first][second]
      * values[f"derivatives.extrinsic{min(first, second) + 1}{max(first, second) + 1}"]
      for first in range(3)
      for second in range(3)
   )
   lapse_term = sum(
      momentum_contravariant[direction]
      * values[f"derivatives.d{direction + 1}_lapse"]
      for direction in range(3)
   )
   source = [
      0.0,
      *momentum_source,
      sqrt_det * (values["metric.lapse"] * extrinsic_term - lapse_term),
   ]
   return source + ([0.0, 0.0, 0.0] if magnetized else [])


def srmhd_wave_speed_reference(values: dict[str, float], direction: int) -> list[float]:
   gamma = values["gamma"]
   rho = values["primitive.rho"]
   eps = values["primitive.eps"]
   pressure = (gamma - 1.0) * rho * eps
   enthalpy = 1.0 + eps + pressure / rho
   velocity = [values[f"primitive.v{i + 1}"] for i in range(3)]
   magnetic = [values[f"primitive.b{i + 1}"] for i in range(3)]
   velocity_squared = sum(component * component for component in velocity)
   lorentz_factor_squared = 1.0 / (1.0 - velocity_squared)
   magnetic_squared = sum(component * component for component in magnetic)
   magnetic_velocity = sum(magnetic[i] * velocity[i] for i in range(3))
   comoving_squared = magnetic_squared / lorentz_factor_squared + magnetic_velocity**2
   sound_squared = gamma * pressure / (rho * enthalpy)
   alfven_squared = comoving_squared / (rho * enthalpy + comoving_squared)
   fast_squared = sound_squared + alfven_squared - sound_squared * alfven_squared
   return relativistic_wave_speed_reference(velocity, fast_squared, direction)


def validate_relativistic_contract(rng: random.Random) -> None:
   kernels = {kernel.name: kernel for kernel in relativistic_kernels()}
   evaluated_kernel_names = {
      "rhd_primitive_to_conserved",
      "rhd_stress_energy",
      "rhd_geometric_source",
      "rmhd_primitive_to_conserved",
      "rmhd_stress_energy",
      "rmhd_geometric_source",
      *(f"rhd_flux_dir{direction}" for direction in range(3)),
      *(f"rmhd_flux_dir{direction}" for direction in range(3)),
      *(f"rhd_wave_speeds_dir{direction}" for direction in range(3)),
      *(f"srmhd_wave_speeds_dir{direction}" for direction in range(3)),
   }
   evaluators = {
      name: compile_evaluator(kernels[name].expressions)
      for name in evaluated_kernel_names
   }
   for _ in range(96):
      values = random_relativistic_values(rng)
      expected_conserved, expected_fluxes = relativistic_reference(values)
      assert_close(
         evaluators["rhd_primitive_to_conserved"](values),
         expected_conserved,
         "RHD p2c",
      )
      expected_rmhd, expected_rmhd_fluxes = relativistic_mhd_reference(values)
      assert_close(
         evaluators["rmhd_primitive_to_conserved"](values),
         expected_rmhd,
         "RMHD p2c",
      )
      assert_close(
         evaluators["rhd_stress_energy"](values),
         relativistic_stress_energy_reference(values, False),
         "RHD stress-energy",
      )
      assert_close(
         evaluators["rmhd_stress_energy"](values),
         relativistic_stress_energy_reference(values, True),
         "RMHD stress-energy",
      )
      assert_close(
         evaluators["rhd_geometric_source"](values),
         geometric_source_reference(values, False),
         "RHD geometric source",
      )
      assert_close(
         evaluators["rmhd_geometric_source"](values),
         geometric_source_reference(values, True),
         "RMHD geometric source",
      )
      for direction in range(3):
         assert_close(
            evaluators[f"rhd_flux_dir{direction}"](values),
            expected_fluxes[direction],
            f"RHD flux dir={direction}",
         )
         assert_close(
            evaluators[f"rmhd_flux_dir{direction}"](values),
            expected_rmhd_fluxes[direction],
            f"RMHD flux dir={direction}",
         )
         minkowski_values = dict(values)
         minkowski_values["metric.lapse"] = 1.0
         minkowski_values["metric.shift1"] = 0.0
         minkowski_values["metric.shift2"] = 0.0
         minkowski_values["metric.shift3"] = 0.0
         minkowski_values["metric.sqrt_det_gamma"] = 1.0
         for i in range(3):
            for j in range(i, 3):
               minkowski_values[f"metric.gamma{i + 1}{j + 1}"] = 1.0 if i == j else 0.0
               minkowski_values[f"metric.inv_gamma{i + 1}{j + 1}"] = (
                  1.0 if i == j else 0.0
               )
         velocity_squared = sum(minkowski_values[f"primitive.v{i + 1}"] ** 2 for i in range(3))
         if velocity_squared >= 0.95:
            velocity_scale = math.sqrt(0.8 / velocity_squared)
            for i in range(3):
               minkowski_values[f"primitive.v{i + 1}"] *= velocity_scale
         rho = minkowski_values["primitive.rho"]
         eps = minkowski_values["primitive.eps"]
         gamma = minkowski_values["gamma"]
         pressure = (gamma - 1.0) * rho * eps
         specific_enthalpy = 1.0 + eps + pressure / rho
         minkowski_values["sound_speed_squared"] = (
            gamma * pressure / (rho * specific_enthalpy)
         )
         assert_close(
            evaluators[f"rhd_wave_speeds_dir{direction}"](minkowski_values),
            srhd_wave_speed_reference(minkowski_values, direction),
            f"SRHD wave speeds dir={direction}",
         )
         assert_close(
            evaluators[f"srmhd_wave_speeds_dir{direction}"](minkowski_values),
            srmhd_wave_speed_reference(minkowski_values, direction),
            f"SRMHD wave speeds dir={direction}",
         )

      zero_field = dict(values)
      zero_field.update({"primitive.b1": 0.0, "primitive.b2": 0.0, "primitive.b3": 0.0})
      assert_close(
         evaluators["rmhd_primitive_to_conserved"](zero_field)[:5],
         evaluators["rhd_primitive_to_conserved"](zero_field),
         "RMHD zero-field conserved reduction",
      )
      assert_close(
         evaluators["rmhd_stress_energy"](zero_field),
         evaluators["rhd_stress_energy"](zero_field),
         "RMHD zero-field stress-energy reduction",
      )
      for direction in range(3):
         assert_close(
            evaluators[f"rmhd_flux_dir{direction}"](zero_field)[:5],
            evaluators[f"rhd_flux_dir{direction}"](zero_field),
            f"RMHD zero-field flux reduction dir={direction}",
         )

      cyclic = (1, 2, 0)
      rotated_values = dict(minkowski_values)
      for field in ("v", "b"):
         for component, source in enumerate(cyclic, start=1):
            rotated_values[f"primitive.{field}{component}"] = (
               minkowski_values[f"primitive.{field}{source + 1}"]
            )
      original_conserved = evaluators["rmhd_primitive_to_conserved"](
         minkowski_values
      )
      rotated_conserved = evaluators["rmhd_primitive_to_conserved"](
         rotated_values
      )
      expected_rotated_conserved = [
         original_conserved[0],
         *(original_conserved[1 + source] for source in cyclic),
         original_conserved[4],
         *(original_conserved[5 + source] for source in cyclic),
      ]
      assert_close(
         rotated_conserved,
         expected_rotated_conserved,
         "RMHD cyclic conserved covariance",
      )
      original_flux = evaluators["rmhd_flux_dir1"](minkowski_values)
      rotated_flux = evaluators["rmhd_flux_dir0"](rotated_values)
      expected_rotated_flux = [
         original_flux[0],
         *(original_flux[1 + source] for source in cyclic),
         original_flux[4],
         *(original_flux[5 + source] for source in cyclic),
      ]
      assert_close(
         rotated_flux,
         expected_rotated_flux,
         "RMHD cyclic flux covariance",
      )

      gamma = [[values[f"metric.gamma{min(i, j) + 1}{max(i, j) + 1}"] for j in range(3)] for i in range(3)]
      inverse = [[values[f"metric.inv_gamma{min(i, j) + 1}{max(i, j) + 1}"] for j in range(3)] for i in range(3)]
      identity_error = max(
         abs(sum(gamma[i][k] * inverse[k][j] for k in range(3)) - (1.0 if i == j else 0.0))
         for i in range(3)
         for j in range(3)
      )
      if identity_error > 2.0e-12:
         raise AssertionError(f"RHD metric inverse residual={identity_error:.3e}")

   base_velocity = (0.31, -0.17, 0.09)
   newtonian_values = random_relativistic_values(rng)
   errors = []
   for scale in (1.0e-2, 5.0e-3):
      values = dict(newtonian_values)
      values.update({
         "metric.lapse": 1.0,
         "metric.shift1": 0.0,
         "metric.shift2": 0.0,
         "metric.shift3": 0.0,
         "metric.gamma11": 1.0,
         "metric.gamma12": 0.0,
         "metric.gamma13": 0.0,
         "metric.gamma22": 1.0,
         "metric.gamma23": 0.0,
         "metric.gamma33": 1.0,
         "metric.inv_gamma11": 1.0,
         "metric.inv_gamma12": 0.0,
         "metric.inv_gamma13": 0.0,
         "metric.inv_gamma22": 1.0,
         "metric.inv_gamma23": 0.0,
         "metric.inv_gamma33": 1.0,
         "metric.sqrt_det_gamma": 1.0,
         "primitive.eps": 0.7 * scale**2,
         "primitive.pressure": values["primitive.rho"] * 0.2 * scale**2,
         "primitive.v1": scale * base_velocity[0],
         "primitive.v2": scale * base_velocity[1],
         "primitive.v3": scale * base_velocity[2],
      })
      actual = evaluators["rhd_primitive_to_conserved"](values)
      rho = values["primitive.rho"]
      kinetic = 0.5 * rho * sum(component**2 for component in base_velocity) * scale**2
      newtonian_energy = rho * values["primitive.eps"] + kinetic
      errors.append(abs(actual[4] - newtonian_energy))
   if not errors[1] < 0.07 * errors[0]:
      raise AssertionError(f"RHD Newtonian-limit error did not converge at fourth order: {errors}")


def validate_symbolic_operator_contract() -> None:
   vector = Vector(*sp.symbols("a1 a2 a3"))
   gradient = Vector(*sp.symbols("df_dq1 df_dq2 df_dq3"))
   derivatives = Tensor.from_rows(
      (
         sp.symbols("da1_dq1 da1_dq2 da1_dq3"),
         sp.symbols("da2_dq1 da2_dq2 da2_dq3"),
         sp.symbols("da3_dq1 da3_dq2 da3_dq3"),
      )
   )
   hessian = Tensor.from_rows(
      (
         sp.symbols("d2f_dq1dq1 d2f_dq1dq2 d2f_dq1dq3"),
         sp.symbols("d2f_dq2dq1 d2f_dq2dq2 d2f_dq2dq3"),
         sp.symbols("d2f_dq3dq1 d2f_dq3dq2 d2f_dq3dq3"),
      )
   )
   vector_field = VectorField(vector, derivatives, Vector(*sp.symbols("dt_a1 dt_a2 dt_a3")))
   scalar_field = ScalarField(sp.Symbol("f"), gradient, hessian, sp.Symbol("dt_f"))
   viscosity = sp.Symbol("mu")
   for frame in orthogonal_frames():
      operators = OrthogonalOperators(frame.scale_factors, frame.coordinates)
      vector_gradient = operators.gradient(vector_field)
      if not isinstance(vector_gradient, Tensor):
         raise AssertionError(f"{frame.name} vector gradient did not produce a tensor")
      if sp.simplify(operators.divergence(vector_field) - vector_gradient.trace()) != 0:
         raise AssertionError(f"{frame.name} divergence does not match trace(grad)")
      stress = NewtonianViscosity(viscosity).stress(vector_field, operators)
      if stress != stress.transpose():
         raise AssertionError(f"{frame.name} Newtonian stress is not symmetric")
      if sp.simplify(stress.trace()) != 0:
         raise AssertionError(f"{frame.name} Newtonian stress is not traceless")

   cartesian = OrthogonalOperators(orthogonal_frames()[0].scale_factors, orthogonal_frames()[0].coordinates)
   if cartesian.gradient(scalar_field) != gradient:
      raise AssertionError("Cartesian scalar gradient changed its components")
   expected_laplacian = sum(hessian.component(axis, axis) for axis in range(3))
   if sp.simplify(cartesian.laplacian(scalar_field) - expected_laplacian) != 0:
      raise AssertionError("Cartesian scalar Laplacian does not match the Hessian trace")
   expected_curl = Vector(
      derivatives.component(2, 1) - derivatives.component(1, 2),
      derivatives.component(0, 2) - derivatives.component(2, 0),
      derivatives.component(1, 0) - derivatives.component(0, 1),
   )
   if cartesian.curl(vector_field) != expected_curl:
      raise AssertionError("Cartesian curl does not match the antisymmetric vector gradient")
   if scalar_field.dt() != sp.Symbol("dt_f") or vector_field.dt() != Vector(*sp.symbols("dt_a1 dt_a2 dt_a3")):
      raise AssertionError("field time derivatives changed their symbolic values")
   vector4 = Vector4(
      *sp.symbols("u0 u1 u2 u3"), position=IndexPosition.CONTRAVARIANT
   )
   metric4 = MetricTensor(Tensor.identity(4), Tensor.identity(4))
   covector4 = vector4.lower_index(metric4)
   tensor4 = vector4.outer(covector4)
   if vector4.dimension != 4 or tensor4.shape != (4, 4):
      raise AssertionError("rank-generic algebra does not support four-dimensional values")
   if vector4.contract(covector4) != sum(value**2 for value in vector4.as_tuple()):
      raise AssertionError("indexed vector contraction failed")
   if tensor4.trace() != vector4.contract(covector4):
      raise AssertionError("mixed-index tensor trace failed")
   raised_tensor4 = tensor4.raise_index(metric4, 1)
   if not isinstance(raised_tensor4, IndexedTensor):
      raise AssertionError("indexed tensor raise operation failed")
   if sp.simplify(Tensor.identity(4).deviatoric().trace()) != 0:
      raise AssertionError("four-dimensional deviatoric projection is not traceless")


def validate_symbolic_eos_contract() -> None:
   pressure = sp.Symbol("pressure", positive=True)
   equations_of_state = analytic_equations_of_state()
   for eos in equations_of_state:
      sound_speed_squared = eos.sound_speed_squared()
      if sp.simplify(sound_speed_squared - sp.diff(eos.pressure(), eos.density)) == 0:
         if not eos.is_barotropic and sp.diff(eos.pressure(), eos.internal_energy_density) != 0:
            raise AssertionError(f"{eos.name} lost its adiabatic energy contribution")
      if eos.is_barotropic:
         energy_density = eos.internal_energy_from_pressure(pressure)
         specific_energy = energy_density / eos.density
         if sp.simplify(
            eos.density**2 * sp.diff(specific_energy, eos.density) - eos.pressure()
         ) != 0:
            raise AssertionError(f"{eos.name} violates the barotropic first-law identity")
      else:
         recovered_pressure = eos.pressure().subs(
            eos.internal_energy_density,
            eos.internal_energy_from_pressure(pressure),
         )
         if sp.simplify(recovered_pressure - pressure) != 0:
            raise AssertionError(f"{eos.name} pressure/internal-energy closure is not invertible")
   ideal_gas, polytrope, hybrid_polytrope = equations_of_state
   if Hydrodynamics(ideal_gas).is_barotropic or not Hydrodynamics(polytrope).is_barotropic:
      raise AssertionError("the physical system did not derive its barotropic capability from the EOS")
   if Hydrodynamics(hybrid_polytrope).is_magnetized or not Magnetohydrodynamics(ideal_gas).is_magnetized:
      raise AssertionError("the physical system did not derive its magnetized capability from its equations")


def random_eos_values(rng: random.Random) -> dict[str, float]:
   return {
      "rho": rng.uniform(0.2, 5.0),
      "internal_energy_density": rng.uniform(0.1, 20.0),
      "specific_internal_energy": rng.uniform(0.01, 20.0),
      "pressure": rng.uniform(0.05, 5.0),
      "gamma": rng.uniform(1.1, 2.1),
      "polytropic_k": rng.uniform(0.1, 3.0),
      "polytropic_gamma": rng.uniform(1.1, 2.1),
      "thermal_gamma": rng.uniform(1.1, 2.1),
      "cold_specific_energy_offset": rng.uniform(0.0, 0.2),
   }


def eos_reference(name: str, values: dict[str, float]) -> float:
   rho = values["rho"]
   internal = values["internal_energy_density"]
   specific_internal = values["specific_internal_energy"]
   pressure = values["pressure"]
   gamma = values["gamma"]
   polytropic_k = values["polytropic_k"]
   polytropic_gamma = values["polytropic_gamma"]
   thermal_gamma = values["thermal_gamma"]
   cold_specific_energy_offset = values["cold_specific_energy_offset"]
   cold_pressure = polytropic_k * rho**polytropic_gamma
   cold_energy = (
      cold_specific_energy_offset * rho
      + cold_pressure / (polytropic_gamma - 1.0)
   )
   cold_specific_energy = cold_energy / rho
   hybrid_pressure = cold_pressure + (thermal_gamma - 1.0) * (
      internal - cold_energy
   )
   relativistic_hybrid_pressure = cold_pressure + (thermal_gamma - 1.0) * rho * (
      specific_internal - cold_specific_energy
   )
   references = {
      "ideal_gas_pressure_from_density_energy": (gamma - 1.0) * internal,
      "ideal_gas_internal_energy_density_from_pressure": pressure / (gamma - 1.0),
      "ideal_gas_sound_speed_squared": gamma * pressure / rho,
      "polytrope_cold_pressure": cold_pressure,
      "polytrope_cold_energy_density": cold_energy,
      "polytrope_sound_speed_squared": polytropic_gamma * cold_pressure / rho,
      "hybrid_polytrope_pressure_from_density_energy": (
         hybrid_pressure
      ),
      "hybrid_polytrope_internal_energy_density_from_pressure": (
         cold_energy + (pressure - cold_pressure) / (thermal_gamma - 1.0)
      ),
      "hybrid_polytrope_sound_speed_squared": (
         polytropic_gamma * cold_pressure
         + thermal_gamma * (pressure - cold_pressure)
      ) / rho,
      "relativistic_ideal_gas_pressure": (
         (gamma - 1.0) * rho * specific_internal
      ),
      "relativistic_ideal_gas_specific_enthalpy": 1.0 + gamma * specific_internal,
      "relativistic_ideal_gas_sound_speed_squared": (
         gamma * (gamma - 1.0) * specific_internal
         / (1.0 + gamma * specific_internal)
      ),
      "relativistic_polytrope_pressure": cold_pressure,
      "relativistic_polytrope_specific_enthalpy": (
         1.0 + specific_internal + cold_pressure / rho
      ),
      "relativistic_polytrope_sound_speed_squared": (
         polytropic_gamma * cold_pressure
         / (rho * (1.0 + specific_internal) + cold_pressure)
      ),
      "relativistic_hybrid_polytrope_pressure": relativistic_hybrid_pressure,
      "relativistic_hybrid_polytrope_specific_enthalpy": (
         1.0 + specific_internal + relativistic_hybrid_pressure / rho
      ),
      "relativistic_hybrid_polytrope_sound_speed_squared": (
         polytropic_gamma * cold_pressure
         + thermal_gamma * (relativistic_hybrid_pressure - cold_pressure)
      ) / (
         rho * (1.0 + specific_internal) + relativistic_hybrid_pressure
      ),
   }
   return references[name]


def random_state(rng: random.Random) -> dict[str, float]:
   rho = rng.uniform(0.2, 5.0)
   return {
      "state.rho": rho,
      "state.mom1": rng.uniform(-2.0, 2.0),
      "state.mom2": rng.uniform(-2.0, 2.0),
      "state.mom3": rng.uniform(-2.0, 2.0),
      "state.energy": rng.uniform(1.0, 20.0),
      "pressure": rng.uniform(0.05, 5.0),
      "state.b1": rng.uniform(-1.0, 1.0),
      "state.b2": rng.uniform(-1.0, 1.0),
      "state.b3": rng.uniform(-1.0, 1.0),
   }


def random_operator_values(rng: random.Random) -> dict[str, float]:
   return {
      "base": rng.uniform(-3.0, 3.0),
      "center": rng.uniform(-3.0, 3.0),
      "xm": rng.uniform(-3.0, 3.0),
      "xp": rng.uniform(-3.0, 3.0),
      "ym": rng.uniform(-3.0, 3.0),
      "yp": rng.uniform(-3.0, 3.0),
      "zm": rng.uniform(-3.0, 3.0),
      "zp": rng.uniform(-3.0, 3.0),
      "inv_dx2": rng.uniform(0.1, 20.0),
      "inv_dy2": rng.uniform(0.1, 20.0),
      "inv_dz2": rng.uniform(0.1, 20.0),
      "cx": rng.uniform(0.0, 5.0),
      "cy": rng.uniform(0.0, 5.0),
      "cz": rng.uniform(0.0, 5.0),
      "coeff_dt": rng.uniform(-0.5, 0.5),
      "lap": rng.uniform(-10.0, 10.0),
      "left": rng.uniform(-3.0, 3.0),
      "right": rng.uniform(-3.0, 3.0),
      "coefficient": rng.uniform(0.0, 5.0),
      "inv_spacing": rng.uniform(0.1, 20.0),
      "face_area": rng.uniform(0.1, 20.0),
      "volume": rng.uniform(0.1, 20.0),
      "spacing": rng.uniform(0.1, 20.0),
   }


def random_riemann_values(rng: random.Random) -> dict[str, float]:
   values: dict[str, float] = {
      "wavespeed": rng.uniform(0.01, 8.0),
      "s_left": rng.uniform(-8.0, -0.01),
      "s_right": rng.uniform(0.01, 8.0),
   }
   for prefix in ("flux_l", "flux_r", "state_l", "state_r"):
      values[f"{prefix}.rho"] = rng.uniform(0.1, 10.0)
      values[f"{prefix}.mom1"] = rng.uniform(-5.0, 5.0)
      values[f"{prefix}.mom2"] = rng.uniform(-5.0, 5.0)
      values[f"{prefix}.mom3"] = rng.uniform(-5.0, 5.0)
      values[f"{prefix}.energy"] = rng.uniform(0.1, 20.0)
      values[f"{prefix}.dens"] = rng.uniform(0.1, 10.0)
      values[f"{prefix}.tau"] = rng.uniform(0.1, 20.0)
      values[f"{prefix}.b1"] = rng.uniform(-3.0, 3.0)
      values[f"{prefix}.b2"] = rng.uniform(-3.0, 3.0)
      values[f"{prefix}.b3"] = rng.uniform(-3.0, 3.0)
   return values


def random_characteristic_values(rng: random.Random) -> dict[str, float]:
   rho = rng.uniform(0.2, 5.0)
   sound_speed = rng.uniform(0.2, 3.0)
   alpha_angle = rng.uniform(0.0, 0.5 * math.pi)
   beta_angle = rng.uniform(-math.pi, math.pi)
   alpha_f = math.cos(alpha_angle)
   alpha_s = math.sin(alpha_angle)
   beta_t1 = math.cos(beta_angle)
   beta_t2 = math.sin(beta_angle)
   fast_speed = rng.uniform(sound_speed, 4.0 * sound_speed)
   slow_speed = rng.uniform(0.05 * sound_speed, sound_speed)
   values = {
      "rho_ref": rho,
      "sound_speed": sound_speed,
      "basis.inv_rho": 1.0 / rho,
      "basis.inv_sqrt_rho": 1.0 / math.sqrt(rho),
      "basis.inv_sound_speed": 1.0 / sound_speed,
      "basis.sound2": sound_speed * sound_speed,
      "basis.inv_sound2": 1.0 / (sound_speed * sound_speed),
      "primitive.rho": rng.uniform(0.2, 5.0),
      "primitive.vn": rng.uniform(-3.0, 3.0),
      "primitive.vt1": rng.uniform(-3.0, 3.0),
      "primitive.vt2": rng.uniform(-3.0, 3.0),
      "primitive.pressure": rng.uniform(0.05, 8.0),
      "primitive.bt1": rng.uniform(-3.0, 3.0),
      "primitive.bt2": rng.uniform(-3.0, 3.0),
      "basis.rho": rho,
      "basis.sqrt_rho": math.sqrt(rho),
      "basis.sound_speed": sound_speed,
      "basis.fast_speed": fast_speed,
      "basis.slow_speed": slow_speed,
      "basis.alpha_f": alpha_f,
      "basis.alpha_s": alpha_s,
      "basis.beta_t1": beta_t1,
      "basis.beta_t2": beta_t2,
      "basis.normal_sign": -1.0 if rng.random() < 0.5 else 1.0,
      "basis.velocity_denom": (
         alpha_f * alpha_f * fast_speed * fast_speed
         + alpha_s * alpha_s * slow_speed * slow_speed
      ),
   }
   values["basis.inv_velocity_denom"] = 1.0 / values["basis.velocity_denom"]
   return values


def characteristic_projection_reference(name: str, values: dict[str, float]) -> list[float]:
   rho = values["primitive.rho"]
   vn = values["primitive.vn"]
   vt1 = values["primitive.vt1"]
   vt2 = values["primitive.vt2"]
   pressure = values["primitive.pressure"]
   sound_speed = values["sound_speed"] if name.startswith("hydro_") else values["basis.sound_speed"]
   inv_sound2 = 1.0 / (sound_speed * sound_speed)
   if name == "hydro_characteristics":
      rho_ref = values["rho_ref"]
      return [
         rho - pressure * inv_sound2,
         vt1,
         vt2,
         0.5 * (pressure * inv_sound2 - rho_ref * vn / sound_speed),
         0.5 * (pressure * inv_sound2 + rho_ref * vn / sound_speed),
      ]

   alpha_f = values["basis.alpha_f"]
   alpha_s = values["basis.alpha_s"]
   beta_t1 = values["basis.beta_t1"]
   beta_t2 = values["basis.beta_t2"]
   sqrt_rho = values["basis.sqrt_rho"]
   basis_rho = values["basis.rho"]
   normal_sign = values["basis.normal_sign"]
   fast_speed = values["basis.fast_speed"]
   slow_speed = values["basis.slow_speed"]
   velocity_denom = values["basis.velocity_denom"]
   bt1 = values["primitive.bt1"]
   bt2 = values["primitive.bt2"]
   compressive = pressure * inv_sound2 / basis_rho
   b_parallel = (beta_t1 * bt1 + beta_t2 * bt2) / (sound_speed * sqrt_rho)
   fast_sum = alpha_f * compressive + alpha_s * b_parallel
   slow_sum = alpha_s * compressive - alpha_f * b_parallel
   v_parallel = normal_sign * (beta_t1 * vt1 + beta_t2 * vt2)
   fast_diff = (alpha_f * fast_speed * vn - alpha_s * slow_speed * v_parallel) / velocity_denom
   slow_diff = (alpha_s * slow_speed * vn + alpha_f * fast_speed * v_parallel) / velocity_denom
   alfven_sum = (beta_t2 * bt1 - beta_t1 * bt2) / sqrt_rho
   alfven_diff = normal_sign * (beta_t2 * vt1 - beta_t1 * vt2)
   return [
      0.5 * (fast_sum - fast_diff),
      0.5 * (alfven_sum + alfven_diff),
      0.5 * (slow_sum - slow_diff),
      rho - pressure * inv_sound2,
      0.5 * (slow_sum + slow_diff),
      0.5 * (alfven_sum - alfven_diff),
      0.5 * (fast_sum + fast_diff),
   ]


def validate_characteristic_round_trip(rng: random.Random) -> None:
   kernels = {kernel.name: kernel for kernel in characteristic_kernels()}
   for _ in range(128):
      values = random_characteristic_values(rng)
      for projection_name, recovery_name, primitive_fields in (
         (
            "hydro_characteristics",
            "hydro_normal_primitive_from_characteristics",
            ("rho", "vn", "vt1", "vt2", "pressure"),
         ),
         (
            "mhd_characteristics",
            "mhd_reduced_from_characteristics",
            ("rho", "vn", "vt1", "vt2", "pressure", "bt1", "bt2"),
         ),
      ):
         projection = kernels[projection_name]
         projected = evaluate(projection.expressions, values)
         assert_close(
            projected,
            characteristic_projection_reference(projection_name, values),
            projection_name,
         )
         recovery_values = dict(values)
         recovery_values.update(
            {
               f"waves.{field}": value
               for field, value in zip(projection.outputs, projected, strict=True)
            }
         )
         recovered = evaluate(kernels[recovery_name].expressions, recovery_values)
         expected = [values[f"primitive.{field}"] for field in primitive_fields]
         assert_close(recovered, expected, f"{projection_name} round trip")


def riemann_reference(name: str, output: str, values: dict[str, float]) -> float:
   flux_l = values[f"flux_l.{output}"]
   flux_r = values[f"flux_r.{output}"]
   state_l = values[f"state_l.{output}"]
   state_r = values[f"state_r.{output}"]
   if name.endswith("_llf_flux"):
      return 0.5 * (flux_l + flux_r) - 0.5 * values["wavespeed"] * (state_r - state_l)
   if name.endswith("_hll_flux"):
      s_left = values["s_left"]
      s_right = values["s_right"]
      return (
         s_right * flux_l
         - s_left * flux_r
         + s_left * s_right * (state_r - state_l)
      ) / (s_right - s_left)
   raise AssertionError(f"missing Riemann reference for {name}")


def random_vector_values(rng: random.Random) -> dict[str, float]:
   return {
      "daz_dq2": rng.uniform(-5.0, 5.0),
      "day_dq3": rng.uniform(-5.0, 5.0),
      "dax_dq3": rng.uniform(-5.0, 5.0),
      "daz_dq1": rng.uniform(-5.0, 5.0),
      "day_dq1": rng.uniform(-5.0, 5.0),
      "dax_dq2": rng.uniform(-5.0, 5.0),
      "dax_dq1": rng.uniform(-5.0, 5.0),
      "day_dq2": rng.uniform(-5.0, 5.0),
      "daz_dq3": rng.uniform(-5.0, 5.0),
      "inv_h2h3": rng.uniform(0.1, 4.0),
      "inv_h3h1": rng.uniform(0.1, 4.0),
      "inv_h1h2": rng.uniform(0.1, 4.0),
      "inv_volume_scale": rng.uniform(0.1, 4.0),
      "d_h3_a3_dq2": rng.uniform(-5.0, 5.0),
      "d_h2_a2_dq3": rng.uniform(-5.0, 5.0),
      "d_h1_a1_dq3": rng.uniform(-5.0, 5.0),
      "d_h3_a3_dq1": rng.uniform(-5.0, 5.0),
      "d_h2_a2_dq1": rng.uniform(-5.0, 5.0),
      "d_h1_a1_dq2": rng.uniform(-5.0, 5.0),
      "d_h2h3_a1_dq1": rng.uniform(-5.0, 5.0),
      "d_h3h1_a2_dq2": rng.uniform(-5.0, 5.0),
      "d_h1h2_a3_dq3": rng.uniform(-5.0, 5.0),
      "vx": rng.uniform(-3.0, 3.0),
      "vy": rng.uniform(-3.0, 3.0),
      "vz": rng.uniform(-3.0, 3.0),
      "bx": rng.uniform(-2.0, 2.0),
      "by": rng.uniform(-2.0, 2.0),
      "bz": rng.uniform(-2.0, 2.0),
      "eta": rng.uniform(0.0, 2.0),
      "j": rng.uniform(-3.0, 3.0),
      "ideal": rng.uniform(-4.0, 4.0),
      "h_tangent": rng.uniform(0.1, 5.0),
      "d_scaled_b": rng.uniform(-4.0, 4.0),
      "denominator": rng.uniform(0.1, 5.0),
      "pressure": rng.uniform(0.01, 10.0),
      "floor": 1.0e-12,
      "face_b": rng.uniform(-2.0, 2.0),
      "face_area": rng.uniform(0.1, 5.0),
      "face_flux": rng.uniform(-5.0, 5.0),
      "edge_e": rng.uniform(-3.0, 3.0),
      "edge_length": rng.uniform(0.1, 5.0),
      "dt": rng.uniform(0.001, 0.5),
      "ez_hi": rng.uniform(-3.0, 3.0),
      "ez_lo": rng.uniform(-3.0, 3.0),
      "ex_hi": rng.uniform(-3.0, 3.0),
      "ex_lo": rng.uniform(-3.0, 3.0),
      "ey_hi": rng.uniform(-3.0, 3.0),
      "ey_lo": rng.uniform(-3.0, 3.0),
      "flux_x_hi": rng.uniform(-3.0, 3.0),
      "flux_x_lo": rng.uniform(-3.0, 3.0),
      "flux_y_hi": rng.uniform(-3.0, 3.0),
      "flux_y_lo": rng.uniform(-3.0, 3.0),
      "flux_z_hi": rng.uniform(-3.0, 3.0),
      "flux_z_lo": rng.uniform(-3.0, 3.0),
      "volume": rng.uniform(0.1, 5.0),
      "mu": rng.uniform(0.001, 3.0),
      "q1": rng.uniform(0.4, 4.0),
      "q2": rng.uniform(0.3, math.pi - 0.3),
      "q3": rng.uniform(-2.0, 2.0),
      "v1": rng.uniform(-3.0, 3.0),
      "v2": rng.uniform(-3.0, 3.0),
      "v3": rng.uniform(-3.0, 3.0),
      "dv1_dq1": rng.uniform(-4.0, 4.0),
      "dv1_dq2": rng.uniform(-4.0, 4.0),
      "dv1_dq3": rng.uniform(-4.0, 4.0),
      "dv2_dq1": rng.uniform(-4.0, 4.0),
      "dv2_dq2": rng.uniform(-4.0, 4.0),
      "dv2_dq3": rng.uniform(-4.0, 4.0),
      "dv3_dq1": rng.uniform(-4.0, 4.0),
      "dv3_dq2": rng.uniform(-4.0, 4.0),
      "dv3_dq3": rng.uniform(-4.0, 4.0),
      "t11": rng.uniform(-3.0, 3.0),
      "t12": rng.uniform(-3.0, 3.0),
      "t13": rng.uniform(-3.0, 3.0),
      "t21": rng.uniform(-3.0, 3.0),
      "t22": rng.uniform(-3.0, 3.0),
      "t23": rng.uniform(-3.0, 3.0),
      "t31": rng.uniform(-3.0, 3.0),
      "t32": rng.uniform(-3.0, 3.0),
      "t33": rng.uniform(-3.0, 3.0),
   }


def random_metric_values(rng: random.Random) -> dict[str, float]:
   rlo = rng.uniform(0.001, 8.0)
   rhi = rlo + rng.uniform(0.001, 2.0)
   theta_lo = rng.uniform(0.05, math.pi - 0.5)
   theta_hi = theta_lo + rng.uniform(0.01, 0.4)
   theta_hi = min(theta_hi, math.pi - 0.05)
   theta = rng.uniform(theta_lo, theta_hi)
   radius = rng.uniform(0.001, 10.0)
   return {
      "dx": rng.uniform(0.001, 3.0),
      "dy": rng.uniform(0.001, 3.0),
      "dz": rng.uniform(0.001, 3.0),
      "r": radius,
      "theta": theta,
      "phi": rng.uniform(-math.pi, math.pi),
      "rlo": rlo,
      "rhi": rhi,
      "theta_lo": theta_lo,
      "theta_hi": theta_hi,
      "z_width": rng.uniform(0.001, 3.0),
      "phi_width": rng.uniform(0.001, 2.0 * math.pi),
      "rho": rng.uniform(0.01, 5.0),
      "vr": rng.uniform(-2.0, 2.0),
      "vtheta": rng.uniform(-2.0, 2.0),
      "vphi": rng.uniform(-2.0, 2.0),
      "vz": rng.uniform(-2.0, 2.0),
      "br": rng.uniform(-2.0, 2.0),
      "btheta": rng.uniform(-2.0, 2.0),
      "bphi": rng.uniform(-2.0, 2.0),
      "bz": rng.uniform(-2.0, 2.0),
      "inv_r": 1.0 / radius,
      "cot_theta": math.cos(theta) / math.sin(theta),
   }


def laplacian_reference(values: dict[str, float], dim: int) -> float:
   value = (values["xm"] - 2.0 * values["center"] + values["xp"]) * values["inv_dx2"]
   if dim >= 2:
      value += (values["ym"] - 2.0 * values["center"] + values["yp"]) * values["inv_dy2"]
   if dim == 3:
      value += (values["zm"] - 2.0 * values["center"] + values["zp"]) * values["inv_dz2"]
   return value


def implicit_relaxed_reference(values: dict[str, float], dim: int) -> float:
   numerator = values["base"] + values["cx"] * (values["xm"] + values["xp"])
   denominator = 1.0 + 2.0 * values["cx"]
   if dim >= 2:
      numerator += values["cy"] * (values["ym"] + values["yp"])
      denominator += 2.0 * values["cy"]
   if dim == 3:
      numerator += values["cz"] * (values["zm"] + values["zp"])
      denominator += 2.0 * values["cz"]
   return numerator / denominator


def implicit_residual_reference(values: dict[str, float], dim: int) -> float:
   value = (
      values["center"]
      - values["base"]
      + values["cx"] * (2.0 * values["center"] - values["xm"] - values["xp"])
   )
   if dim >= 2:
      value += values["cy"] * (2.0 * values["center"] - values["ym"] - values["yp"])
   if dim == 3:
      value += values["cz"] * (2.0 * values["center"] - values["zm"] - values["zp"])
   return value


def orthogonal_scale_data(coordinate: str, values: dict[str, float]) -> tuple[list[float], list[list[float]]]:
   q1 = values["q1"]
   q2 = values["q2"]
   if coordinate == "cylindrical":
      return [1.0, q1, 1.0], [
         [0.0, 0.0, 0.0],
         [1.0, 0.0, 0.0],
         [0.0, 0.0, 0.0],
      ]
   if coordinate == "spherical":
      return [1.0, q1, q1 * math.sin(q2)], [
         [0.0, 0.0, 0.0],
         [1.0, 0.0, 0.0],
         [math.sin(q2), q1 * math.cos(q2), 0.0],
      ]
   return [1.0, 1.0, 1.0], [
      [0.0, 0.0, 0.0],
      [0.0, 0.0, 0.0],
      [0.0, 0.0, 0.0],
   ]


def viscous_stress_reference(coordinate: str, normal: int, component: int, values: dict[str, float]) -> float:
   h, dh = orthogonal_scale_data(coordinate, values)
   velocity = [values["v1"], values["v2"], values["v3"]]
   dv = [
      [values["dv1_dq1"], values["dv1_dq2"], values["dv1_dq3"]],
      [values["dv2_dq1"], values["dv2_dq2"], values["dv2_dq3"]],
      [values["dv3_dq1"], values["dv3_dq2"], values["dv3_dq3"]],
   ]
   volume = h[0] * h[1] * h[2]
   divergence = 0.0
   for axis in range(3):
      area = volume / h[axis]
      d_area = 0.0
      for scale_axis in range(3):
         if scale_axis != axis:
            product = 1.0
            for other in range(3):
               if other != axis and other != scale_axis:
                  product *= h[other]
            d_area += product * dh[scale_axis][axis]
      divergence += area * dv[axis][axis] + velocity[axis] * d_area
   divergence /= volume

   strain = [[0.0 for _ in range(3)] for _ in range(3)]
   for i in range(3):
      diagonal = dv[i][i] / h[i]
      for k in range(3):
         if k != i:
            diagonal += velocity[k] * dh[i][k] / (h[i] * h[k])
      strain[i][i] = diagonal
      for j in range(i + 1, 3):
         term_ij = h[j] / h[i] * (dv[j][i] / h[j] - velocity[j] * dh[j][i] / (h[j] ** 2))
         term_ji = h[i] / h[j] * (dv[i][j] / h[i] - velocity[i] * dh[i][j] / (h[i] ** 2))
         strain[i][j] = 0.5 * (term_ij + term_ji)
         strain[j][i] = strain[i][j]
   trace = divergence / 3.0 if normal == component else 0.0
   return 2.0 * values["mu"] * (strain[normal][component] - trace)


def tensor_connection_reference(coordinate: str, component: int, values: dict[str, float]) -> float:
   h, dh = orthogonal_scale_data(coordinate, values)
   tensor = [
      [values["t11"], values["t12"], values["t13"]],
      [values["t21"], values["t22"], values["t23"]],
      [values["t31"], values["t32"], values["t33"]],
   ]
   source = 0.0
   for normal in range(3):
      source += tensor[normal][component] * dh[component][normal] / (h[component] * h[normal])
      source -= tensor[normal][normal] * dh[normal][component] / (h[component] * h[normal])
   return source


def vector_reference(name: str, values: dict[str, float]) -> float:
   if name == "curl_x_cartesian":
      return values["daz_dq2"] - values["day_dq3"]
   if name == "curl_y_cartesian":
      return values["dax_dq3"] - values["daz_dq1"]
   if name == "curl_z_cartesian":
      return values["day_dq1"] - values["dax_dq2"]
   if name == "divergence_cartesian":
      return values["dax_dq1"] + values["day_dq2"] + values["daz_dq3"]
   if name == "curl_orthogonal_1":
      return values["inv_h2h3"] * (values["d_h3_a3_dq2"] - values["d_h2_a2_dq3"])
   if name == "curl_orthogonal_2":
      return values["inv_h3h1"] * (values["d_h1_a1_dq3"] - values["d_h3_a3_dq1"])
   if name == "curl_orthogonal_3":
      return values["inv_h1h2"] * (values["d_h2_a2_dq1"] - values["d_h1_a1_dq2"])
   if name == "divergence_orthogonal":
      return values["inv_volume_scale"] * (
         values["d_h2h3_a1_dq1"] + values["d_h3h1_a2_dq2"] + values["d_h1h2_a3_dq3"]
      )
   for coordinate in ("cartesian", "cylindrical", "spherical"):
      prefix = f"{coordinate}_viscous_stress_"
      if name.startswith(prefix):
         indices = name.removeprefix(prefix)
         return viscous_stress_reference(coordinate, int(indices[0]) - 1, int(indices[1]) - 1, values)
      prefix = f"{coordinate}_tensor_divergence_connection_"
      if name.startswith(prefix):
         component = int(name.removeprefix(prefix)) - 1
         return tensor_connection_reference(coordinate, component, values)
   if name == "ideal_emf_x":
      return values["vz"] * values["by"] - values["vy"] * values["bz"]
   if name == "ideal_emf_y":
      return values["vx"] * values["bz"] - values["vz"] * values["bx"]
   if name == "ideal_emf_z":
      return values["vy"] * values["bx"] - values["vx"] * values["by"]
   if name == "resistive_emf":
      return values["eta"] * values["j"]
   if name == "total_emf":
      return values["ideal"] + values["eta"] * values["j"]
   if name == "resistive_emf_scaled_gradient":
      return values["eta"] * values["h_tangent"] * values["d_scaled_b"] / values["denominator"]
   if name == "magnetic_energy_density_from_components":
      return 0.5 * (values["bx"] ** 2 + values["by"] ** 2 + values["bz"] ** 2)
   if name == "plasma_beta_from_pressure":
      return values["pressure"] / max(
         values["floor"],
         0.5 * (values["bx"] ** 2 + values["by"] ** 2 + values["bz"] ** 2),
      )
   if name == "magnetic_face_flux":
      return values["face_b"] * values["face_area"]
   if name == "magnetic_face_component":
      return values["face_flux"] / values["face_area"]
   if name == "emf_line_integral":
      return values["edge_e"] * values["edge_length"]
   if name == "ct_flux_update_2d_dir0":
      return values["face_flux"] - values["dt"] * (values["ez_hi"] - values["ez_lo"])
   if name == "ct_flux_update_2d_dir1":
      return values["face_flux"] + values["dt"] * (values["ez_hi"] - values["ez_lo"])
   if name == "ct_flux_update_3d_dir0":
      return values["face_flux"] - values["dt"] * (
         (values["ez_hi"] - values["ez_lo"]) - (values["ey_hi"] - values["ey_lo"])
      )
   if name == "ct_flux_update_3d_dir1":
      return values["face_flux"] - values["dt"] * (
         (values["ex_hi"] - values["ex_lo"]) - (values["ez_hi"] - values["ez_lo"])
      )
   if name == "ct_flux_update_3d_dir2":
      return values["face_flux"] - values["dt"] * (
         (values["ey_hi"] - values["ey_lo"]) - (values["ex_hi"] - values["ex_lo"])
      )
   if name == "finite_volume_divergence_1d":
      return (values["flux_x_hi"] - values["flux_x_lo"]) / values["volume"]
   if name == "finite_volume_divergence_2d":
      return (
         (values["flux_x_hi"] - values["flux_x_lo"])
         + (values["flux_y_hi"] - values["flux_y_lo"])
      ) / values["volume"]
   if name == "finite_volume_divergence_3d":
      return (
         (values["flux_x_hi"] - values["flux_x_lo"])
         + (values["flux_y_hi"] - values["flux_y_lo"])
         + (values["flux_z_hi"] - values["flux_z_lo"])
      ) / values["volume"]
   raise AssertionError(f"missing vector reference for {name}")


def metric_reference(name: str, values: dict[str, float]) -> float:
   radial_shell_cyl = (values["rhi"] ** 2 - values["rlo"] ** 2) / 2.0
   radial_shell_sph = (values["rhi"] ** 3 - values["rlo"] ** 3) / 3.0
   polar_weight = math.cos(values["theta_lo"]) - math.cos(values["theta_hi"])
   theta_source_denom = (
      math.cos(values["theta_hi"]) * values["rhi"] ** 3
      - math.cos(values["theta_hi"]) * values["rlo"] ** 3
      - values["rhi"] ** 3 * math.cos(values["theta_lo"])
      + math.cos(values["theta_lo"]) * values["rlo"] ** 3
   )
   theta_pressure_factor = (
      -1.5
      * (values["rhi"] ** 2 - values["rlo"] ** 2)
      * (math.sin(values["theta_hi"]) - math.sin(values["theta_lo"]))
      / theta_source_denom
   )
   if name == "cartesian_cell_volume":
      return values["dx"] * values["dy"] * values["dz"]
   if name == "cartesian_face_area_x":
      return values["dy"] * values["dz"]
   if name == "cartesian_face_area_y":
      return values["dx"] * values["dz"]
   if name == "cartesian_face_area_z":
      return values["dx"] * values["dy"]
   if name == "cartesian_edge_length_x":
      return values["dx"]
   if name == "cartesian_edge_length_y":
      return values["dy"]
   if name == "cartesian_edge_length_z":
      return values["dz"]
   if name.startswith("cartesian_scale_") or name == "cartesian_volume_scale":
      return 1.0
   if name == "cylindrical_scale_r":
      return 1.0
   if name == "cylindrical_scale_phi":
      return values["r"]
   if name == "cylindrical_scale_z":
      return 1.0
   if name == "cylindrical_volume_scale":
      return values["r"]
   if name == "cylindrical_radial_shell_volume_scale":
      return (values["rhi"] ** 2 - values["rlo"] ** 2) / 2.0
   if name == "cylindrical_radial_face_area_scale":
      return values["r"]
   if name == "cylindrical_radial_pressure_source_factor":
      shell_volume_scale = (values["rhi"] ** 2 - values["rlo"] ** 2) / 2.0
      return (values["rhi"] - values["rlo"]) / shell_volume_scale
   if name == "spherical_scale_r":
      return 1.0
   if name == "spherical_scale_theta":
      return values["r"]
   if name == "spherical_scale_phi":
      return values["r"] * math.sin(values["theta"])
   if name == "spherical_volume_scale":
      return values["r"] ** 2 * math.sin(values["theta"])
   if name == "spherical_radial_shell_volume_scale":
      return (values["rhi"] ** 3 - values["rlo"] ** 3) / 3.0
   if name == "spherical_radial_face_area_scale":
      return values["r"] ** 2
   if name == "spherical_radial_pressure_source_factor":
      shell_volume_scale = (values["rhi"] ** 3 - values["rlo"] ** 3) / 3.0
      return (values["rhi"] ** 2 - values["rlo"] ** 2) / shell_volume_scale
   if name == "spherical_polar_weight":
      return polar_weight
   if name == "cylindrical_rz_cell_volume_scale":
      return radial_shell_cyl * values["z_width"]
   if name == "cylindrical_rz_radial_face_area_scale":
      return values["r"] * values["z_width"]
   if name == "cylindrical_rz_radial_pressure_source_factor":
      return 2.0 / (values["rhi"] + values["rlo"])
   if name == "cylindrical_rz_axial_face_area_scale":
      return radial_shell_cyl
   if name == "spherical_rtheta_cell_volume_scale":
      return radial_shell_sph * polar_weight
   if name == "spherical_rtheta_radial_face_area_scale":
      return values["r"] ** 2 * polar_weight
   if name == "spherical_rtheta_radial_pressure_source_factor":
      return 3.0 * (values["rhi"] + values["rlo"]) / (
         values["rhi"] ** 2 + values["rhi"] * values["rlo"] + values["rlo"] ** 2
      )
   if name == "spherical_rtheta_theta_face_area_scale":
      return radial_shell_cyl * math.sin(values["theta"])
   if name == "spherical_rtheta_theta_pressure_source_factor":
      return theta_pressure_factor
   if name == "spherical_rphi_cell_volume_scale":
      return radial_shell_sph * math.sin(values["theta"]) * values["phi_width"]
   if name == "spherical_rphi_radial_face_area_scale":
      return values["r"] ** 2 * math.sin(values["theta"]) * values["phi_width"]
   if name == "spherical_rphi_radial_pressure_source_factor":
      return 3.0 * (values["rhi"] ** 2 - values["rlo"] ** 2) / (
         values["rhi"] ** 3 - values["rlo"] ** 3
      )
   if name == "spherical_rphi_azimuthal_face_area_scale":
      return radial_shell_cyl
   if name == "cylindrical_phiz_cell_volume_scale":
      return values["r"] * values["z_width"] * values["phi_width"]
   if name == "cylindrical_phiz_axial_face_area_scale":
      return values["r"] * values["phi_width"]
   if name == "cylindrical_phiz_azimuthal_face_area_scale":
      return values["z_width"]
   if name == "spherical_thetaphi_cell_volume_scale":
      return values["r"] ** 2 * polar_weight * values["phi_width"]
   if name == "spherical_thetaphi_theta_face_area_scale":
      return values["r"] * math.sin(values["theta"]) * values["phi_width"]
   if name == "spherical_thetaphi_theta_pressure_source_factor":
      return (math.sin(values["theta_hi"]) - math.sin(values["theta_lo"])) / (
         values["r"] * polar_weight
      )
   if name == "spherical_thetaphi_azimuthal_face_area_scale":
      return values["r"] * (values["theta_hi"] - values["theta_lo"])
   if name == "cylindrical_rphiz_cell_volume_scale":
      return radial_shell_cyl * values["z_width"] * values["phi_width"]
   if name == "cylindrical_rphiz_radial_face_area_scale":
      return values["r"] * values["z_width"] * values["phi_width"]
   if name == "cylindrical_rphiz_radial_pressure_source_factor":
      return 2.0 / (values["rhi"] + values["rlo"])
   if name == "cylindrical_rphiz_axial_face_area_scale":
      return radial_shell_cyl * values["phi_width"]
   if name == "cylindrical_rphiz_azimuthal_face_area_scale":
      return values["z_width"] * (values["rhi"] - values["rlo"])
   if name == "spherical_rthetaphi_cell_volume_scale":
      return radial_shell_sph * polar_weight * values["phi_width"]
   if name == "spherical_rthetaphi_radial_face_area_scale":
      return values["r"] ** 2 * polar_weight * values["phi_width"]
   if name == "spherical_rthetaphi_radial_pressure_source_factor":
      return 3.0 * (values["rhi"] + values["rlo"]) / (
         values["rhi"] ** 2 + values["rhi"] * values["rlo"] + values["rlo"] ** 2
      )
   if name == "spherical_rthetaphi_theta_face_area_scale":
      return radial_shell_cyl * math.sin(values["theta"]) * values["phi_width"]
   if name == "spherical_rthetaphi_theta_pressure_source_factor":
      return theta_pressure_factor
   if name == "spherical_rthetaphi_azimuthal_face_area_scale":
      return radial_shell_cyl * (values["theta_hi"] - values["theta_lo"])
   if name == "cylindrical_rphiz_radial_edge_length_scale":
      return values["rhi"] - values["rlo"]
   if name == "cylindrical_rphiz_axial_edge_length_scale":
      return values["z_width"]
   if name == "cylindrical_rphiz_azimuthal_edge_length_scale":
      return values["r"] * values["phi_width"]
   if name == "spherical_rthetaphi_radial_edge_length_scale":
      return values["rhi"] - values["rlo"]
   if name == "spherical_rthetaphi_theta_edge_length_scale":
      return values["r"] * (values["theta_hi"] - values["theta_lo"])
   if name == "spherical_rthetaphi_azimuthal_edge_length_scale":
      return values["r"] * math.sin(values["theta"]) * values["phi_width"]
   if name == "cylindrical_momentum_curvature_source_mom1":
      return values["inv_r"] * values["rho"] * values["vphi"] ** 2
   if name == "cylindrical_momentum_curvature_source_mom2":
      return -values["inv_r"] * values["rho"] * values["vphi"] * values["vr"]
   if name == "cylindrical_momentum_curvature_source_mom3":
      return 0.0
   if name == "spherical_momentum_curvature_source_mom1":
      return values["inv_r"] * values["rho"] * (values["vphi"] ** 2 + values["vtheta"] ** 2)
   if name == "spherical_momentum_curvature_source_mom2":
      return values["inv_r"] * values["rho"] * (
         values["cot_theta"] * values["vphi"] ** 2 - values["vr"] * values["vtheta"]
      )
   if name == "spherical_momentum_curvature_source_mom3":
      return -values["inv_r"] * values["rho"] * values["vphi"] * (
         values["cot_theta"] * values["vtheta"] + values["vr"]
      )
   if name == "cylindrical_mhd_stress_curvature_source_mom1":
      return values["inv_r"] * (values["rho"] * values["vphi"] ** 2 - values["bphi"] ** 2)
   if name == "cylindrical_mhd_stress_curvature_source_mom2":
      return values["inv_r"] * (values["bphi"] * values["br"] - values["rho"] * values["vphi"] * values["vr"])
   if name == "cylindrical_mhd_stress_curvature_source_mom3":
      return 0.0
   if name == "spherical_mhd_stress_curvature_source_mom1":
      return values["inv_r"] * (
         values["rho"] * (values["vtheta"] ** 2 + values["vphi"] ** 2)
         - values["btheta"] ** 2
         - values["bphi"] ** 2
      )
   if name == "spherical_mhd_stress_curvature_source_mom2":
      return values["inv_r"] * (
         values["cot_theta"] * (values["rho"] * values["vphi"] ** 2 - values["bphi"] ** 2)
         - (values["rho"] * values["vr"] * values["vtheta"] - values["br"] * values["btheta"])
      )
   if name == "spherical_mhd_stress_curvature_source_mom3":
      return -values["inv_r"] * (
         (values["rho"] * values["vr"] * values["vphi"] - values["br"] * values["bphi"])
         + values["cot_theta"] * (
            values["rho"] * values["vtheta"] * values["vphi"] - values["btheta"] * values["bphi"]
         )
      )
   raise AssertionError(f"missing metric reference for {name}")


def main() -> None:
   if not FLUX_HEADER.exists():
      raise SystemExit(f"missing generated header: {FLUX_HEADER}")
   if not EOS_HEADER.exists():
      raise SystemExit(f"missing generated header: {EOS_HEADER}")
   if not OPERATOR_HEADER.exists():
      raise SystemExit(f"missing generated header: {OPERATOR_HEADER}")
   if not RIEMANN_HEADER.exists():
      raise SystemExit(f"missing generated header: {RIEMANN_HEADER}")
   if not VECTOR_HEADER.exists():
      raise SystemExit(f"missing generated header: {VECTOR_HEADER}")
   if not METRIC_HEADER.exists():
      raise SystemExit(f"missing generated header: {METRIC_HEADER}")
   if not CHARACTERISTIC_HEADER.exists():
      raise SystemExit(f"missing generated header: {CHARACTERISTIC_HEADER}")
   if not RELATIVISTIC_HEADER.exists():
      raise SystemExit(f"missing generated header: {RELATIVISTIC_HEADER}")
   if not Z4C_HEADER.exists():
      raise SystemExit(f"missing generated header: {Z4C_HEADER}")
   if not INITIAL_DATA_HEADER.exists():
      raise SystemExit(f"missing generated header: {INITIAL_DATA_HEADER}")
   rendered_flux = render_flux_header()
   stored_flux = FLUX_HEADER.read_text(encoding="utf-8")
   if rendered_flux != stored_flux:
      raise SystemExit(f"generated header is stale: rerun {CODEGEN / 'generate_kernels.py'}")
   if render_eos_header() != EOS_HEADER.read_text(encoding="utf-8"):
      raise SystemExit(f"generated EOS header is stale: rerun {CODEGEN / 'generate_kernels.py'}")
   rendered_operator = render_operator_header()
   stored_operator = OPERATOR_HEADER.read_text(encoding="utf-8")
   if rendered_operator != stored_operator:
      raise SystemExit(f"generated operator header is stale: rerun {CODEGEN / 'generate_kernels.py'}")
   rendered_riemann = render_riemann_header()
   stored_riemann = RIEMANN_HEADER.read_text(encoding="utf-8")
   if rendered_riemann != stored_riemann:
      raise SystemExit(f"generated Riemann header is stale: rerun {CODEGEN / 'generate_kernels.py'}")
   rendered_vector = render_vector_header()
   stored_vector = VECTOR_HEADER.read_text(encoding="utf-8")
   if rendered_vector != stored_vector:
      raise SystemExit(f"generated vector header is stale: rerun {CODEGEN / 'generate_kernels.py'}")
   rendered_metric = render_metric_header()
   stored_metric = METRIC_HEADER.read_text(encoding="utf-8")
   if rendered_metric != stored_metric:
      raise SystemExit(f"generated metric header is stale: rerun {CODEGEN / 'generate_kernels.py'}")
   rendered_characteristic = render_characteristic_header()
   stored_characteristic = CHARACTERISTIC_HEADER.read_text(encoding="utf-8")
   if rendered_characteristic != stored_characteristic:
      raise SystemExit(f"generated characteristic header is stale: rerun {CODEGEN / 'generate_kernels.py'}")
   rendered_relativistic = render_relativistic_header()
   stored_relativistic = RELATIVISTIC_HEADER.read_text(encoding="utf-8")
   if rendered_relativistic != stored_relativistic:
      raise SystemExit(f"generated relativistic header is stale: rerun {CODEGEN / 'generate_kernels.py'}")
   for name, rendered in render_relativistic_headers().items():
      path = RELATIVISTIC_HEADER_DIR / f"{name}.hpp"
      if not path.exists() or rendered != path.read_text(encoding="utf-8"):
         raise SystemExit(f"generated relativistic group is stale: {path}")
   if render_z4c_header() != Z4C_HEADER.read_text(encoding="utf-8"):
      raise SystemExit(f"generated Z4c header is stale: rerun {CODEGEN / 'generate_kernels.py'}")
   for name, rendered in render_z4c_headers().items():
      path = Z4C_HEADER_DIR / f"{name}.hpp"
      if not path.exists() or rendered != path.read_text(encoding="utf-8"):
         raise SystemExit(f"generated Z4c group is stale: {path}")
   if render_initial_data_header() != INITIAL_DATA_HEADER.read_text(encoding="utf-8"):
      raise SystemExit(
         f"generated initial-data header is stale: rerun {CODEGEN / 'generate_kernels.py'}"
      )
   validate_symbolic_operator_contract()
   validate_symbolic_eos_contract()
   rng = random.Random(20260628)
   validate_relativistic_contract(rng)
   validate_characteristic_round_trip(rng)
   riemann_kernels = tuple(riemann_flux_kernels())
   vector_kernels = tuple(vector_operator_kernels())
   metric_kernels_cached = tuple(metric_kernels())
   eos_kernels_cached = tuple(eos_kernels())
   for _ in range(128):
      values = random_state(rng)
      for direction in range(3):
         hydro_actual = evaluate(hydro_flux_equations(direction).expressions, values)
         hydro_expected = hydro_reference(values, direction)
         assert_close(hydro_actual, hydro_expected, f"hydro dir={direction}")
         mhd_actual = evaluate(mhd_flux_equations(direction).expressions, values)
         mhd_expected = mhd_reference(values, direction)
         assert_close(mhd_actual, mhd_expected, f"mhd dir={direction}")
      operator_values = random_operator_values(rng)
      update_actual = evaluate((explicit_diffusion_update_equation().expression,), operator_values)
      update_expected = [operator_values["center"] + operator_values["coeff_dt"] * operator_values["lap"]]
      assert_close(update_actual, update_expected, "explicit diffusion update")
      flux_actual = evaluate((diffusive_flux_equation().expression,), operator_values)
      flux_expected = [
         -operator_values["coefficient"]
         * (operator_values["right"] - operator_values["left"])
         * operator_values["inv_spacing"]
      ]
      assert_close(flux_actual, flux_expected, "diffusive flux")
      weight_actual = evaluate((finite_volume_laplacian_face_weight_equation().expression,), operator_values)
      weight_expected = [
         operator_values["face_area"] / (operator_values["volume"] * operator_values["spacing"])
      ]
      assert_close(weight_actual, weight_expected, "finite-volume Laplacian face weight")
      for dim in (1, 2, 3):
         lap_actual = evaluate((laplacian_equation(dim).expression,), operator_values)
         lap_expected = [laplacian_reference(operator_values, dim)]
         assert_close(lap_actual, lap_expected, f"laplacian dim={dim}")
         relaxed_actual = evaluate((implicit_relaxed_equation(dim).expression,), operator_values)
         relaxed_expected = [implicit_relaxed_reference(operator_values, dim)]
         assert_close(relaxed_actual, relaxed_expected, f"implicit relaxed dim={dim}")
         residual_actual = evaluate((implicit_residual_equation(dim).expression,), operator_values)
         residual_expected = [implicit_residual_reference(operator_values, dim)]
         assert_close(residual_actual, residual_expected, f"implicit residual dim={dim}")
      riemann_values = random_riemann_values(rng)
      for kernel in riemann_kernels:
         riemann_actual = evaluate(kernel.expressions, riemann_values)
         riemann_expected = [
            riemann_reference(kernel.name, output, riemann_values)
            for output in kernel.outputs
         ]
         assert_close(riemann_actual, riemann_expected, kernel.name)
      vector_values = random_vector_values(rng)
      for kernel in vector_kernels:
         vector_actual = evaluate((kernel.expression,), vector_values)
         vector_expected = [vector_reference(kernel.name, vector_values)]
         assert_close(vector_actual, vector_expected, kernel.name)
      metric_values = random_metric_values(rng)
      for kernel in metric_kernels_cached:
         metric_actual = evaluate((kernel.expression,), metric_values)
         metric_expected = [metric_reference(kernel.name, metric_values)]
         assert_close(metric_actual, metric_expected, kernel.name)
      eos_values = random_eos_values(rng)
      for kernel in eos_kernels_cached:
         eos_actual = evaluate((kernel.expression,), eos_values)
         eos_expected = [eos_reference(kernel.name, eos_values)]
         assert_close(eos_actual, eos_expected, kernel.name)
   print("PASS codegen_flux_riemann_eos_characteristics_relativistic_operators_vector_ops_and_metrics")


if __name__ == "__main__":
   main()
