from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import sympy as sp
from sympy.printing.c import C99CodePrinter

from .characteristic_equations import characteristic_kernels
from .differential_equations import centered_stencils, ko_stencils, upwind_stencils
from .eos_equations import eos_kernels
from .flux_equations import hydro_flux_equations, mhd_flux_equations
from .initial_data_equations import initial_data_kernels
from .kernel_equations import ScalarKernel, StructKernel
from .metric_equations import metric_kernels
from .operator_equations import scalar_operator_kernels
from .patch_map_equations import patch_map_kernels
from .relativistic_equations import relativistic_kernel_groups
from .riemann_equations import riemann_flux_kernels
from .vector_equations import vector_operator_kernels
from .z4c_equations import z4c_kernel_groups


@dataclass(frozen=True)
class CsePolicy:
   hard_drop_ops: int = 0
   soft_drop_ops: int = 3


class PyFluidCxxPrinter(C99CodePrinter):
   def _print_Pi(self, expr: sp.NumberSymbol) -> str:
      del expr
      return "pyfluid_amrex::constants::pi"

   def _print_Float(self, expr: sp.Float) -> str:
      return f"{super()._print_Float(expr)}_rt"

   def _print_Rational(self, expr: sp.Rational) -> str:
      return f"{expr.p}.0_rt/{expr.q}.0_rt"

   def _print_Max(self, expr: sp.Max) -> str:
      args = [self._print(arg) for arg in expr.args]
      value = args[0]
      for arg in args[1:]:
         value = f"(({value}) > ({arg}) ? ({value}) : ({arg}))"
      return value

   def _print_Function(self, expr: sp.Function) -> str:
      function_name = expr.func.__name__
      args = ", ".join(self._print(arg) for arg in expr.args)
      return super()._print_Function(expr)

   def _print_Pow(self, expr: sp.Pow) -> str:
      if expr.exp == 2:
         base = self._print(expr.base)
         return f"(({base})*({base}))"
      if expr.exp == 3:
         base = self._print(expr.base)
         return f"(({base})*({base})*({base}))"
      if expr.exp == -1:
         base = self._print(expr.base)
         return f"1.0_rt/({base})"
      if expr.exp == -2:
         base = self._print(expr.base)
         return f"1.0_rt/(({base})*({base}))"
      if expr.exp == -3:
         base = self._print(expr.base)
         return f"1.0_rt/(({base})*({base})*({base}))"
      return super()._print_Pow(expr)

   def _print_sin(self, expr: sp.Function) -> str:
      return f"std::sin({self._print(expr.args[0])})"

   def _print_cos(self, expr: sp.Function) -> str:
      return f"std::cos({self._print(expr.args[0])})"

   def _print_tan(self, expr: sp.Function) -> str:
      return f"std::tan({self._print(expr.args[0])})"

   def _print_atan(self, expr: sp.Function) -> str:
      return f"std::atan({self._print(expr.args[0])})"

   def _print_Abs(self, expr: sp.Abs) -> str:
      return f"amrex::Math::abs({self._print(expr.args[0])})"


def _symbol_count(exprs: list[sp.Expr], symbol: sp.Symbol) -> int:
   count = 0
   for expr in exprs:
      count += sum(1 for node in sp.preorder_traversal(expr) if node == symbol)
   return count


def _optimize(expressions: tuple[sp.Expr, ...], policy: CsePolicy) -> tuple[list[tuple[sp.Symbol, sp.Expr]], list[sp.Expr]]:
   replacements, reduced = sp.cse(
      expressions,
      symbols=sp.numbered_symbols("tmp"),
      optimizations="basic",
   )
   replacements = list(replacements)
   reduced = list(reduced)
   changed = True
   while changed:
      changed = False
      kept: list[tuple[sp.Symbol, sp.Expr]] = []
      for index, (symbol, expr) in enumerate(replacements):
         later_exprs = [rhs for _, rhs in replacements[index + 1 :]] + reduced
         use_count = _symbol_count(later_exprs, symbol)
         op_count = int(sp.count_ops(expr))
         hard_drop = op_count <= policy.hard_drop_ops
         soft_drop = use_count <= 1 and op_count <= policy.soft_drop_ops
         if hard_drop or soft_drop:
            replacement = {symbol: expr}
            replacements = kept + [
               (later_symbol, later_expr.xreplace(replacement))
               for later_symbol, later_expr in replacements[index + 1 :]
            ]
            reduced = [out.xreplace(replacement) for out in reduced]
            changed = True
            break
         kept.append((symbol, expr))
   return replacements, reduced


def _ccode(expr: sp.Expr) -> str:
   return PyFluidCxxPrinter().doprint(expr)


def _emit_dispatch(name: str, return_type: str, callee_prefix: str) -> str:
   return "\n".join(
      [
         "template <int Direction>",
         "AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE",
         f"{return_type} {name}(const {return_type}& state, amrex::Real pressure) noexcept",
         "{",
         "   if constexpr (Direction == 0) {",
         f"      return {callee_prefix}_dir0(state, pressure);",
         "   } else if constexpr (Direction == 1) {",
         f"      return {callee_prefix}_dir1(state, pressure);",
         "   } else {",
         f"      return {callee_prefix}_dir2(state, pressure);",
         "   }",
         "}",
      ]
   )


def _emit_scalar_kernel(kernel: ScalarKernel, policy: CsePolicy) -> str:
   replacements, outputs = _optimize((kernel.expression,), policy)
   args = ", ".join(
      f"amrex::Real {argument}"
      if isinstance(argument, str)
      else f"{argument[0]} {argument[1]}"
      for argument in kernel.arguments
   )
   lines = [
      "AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE",
      f"amrex::Real {kernel.name}({args}) noexcept",
      "{",
   ]
   for symbol, expr in replacements:
      lines.append(f"   const amrex::Real {symbol} = {_ccode(expr)};")
   lines.append(f"   return {_ccode(outputs[0])};")
   lines.append("}")
   return "\n".join(lines)


def _emit_scalar_dim_dispatch(name: str, callee_prefix: str, arguments: tuple[str, ...]) -> str:
   args = ", ".join(f"amrex::Real {argument}" for argument in arguments)
   call_args = ", ".join(arguments)
   return "\n".join(
      [
         "template <int Dim>",
         "AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE",
         f"amrex::Real {name}({args}) noexcept",
         "{",
         "   if constexpr (Dim == 1) {",
         f"      return {callee_prefix}_1d({call_args});",
         "   } else if constexpr (Dim == 2) {",
         f"      return {callee_prefix}_2d({call_args});",
         "   } else {",
         f"      return {callee_prefix}_3d({call_args});",
         "   }",
         "}",
      ]
   )


def _emit_struct_kernel(kernel: StructKernel, policy: CsePolicy) -> str:
   replacements, outputs = _optimize(kernel.expressions, policy)
   args = ", ".join(f"{argument_type} {name}" for argument_type, name in kernel.arguments)
   lines = [
      "AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE",
      f"{kernel.return_type} {kernel.name}({args}) noexcept",
      "{",
   ]
   for symbol, expr in replacements:
      lines.append(f"   const amrex::Real {symbol} = {_ccode(expr)};")
   lines.append("   return {")
   for index, expr in enumerate(outputs):
      comma = "," if index + 1 < len(outputs) else ""
      lines.append(f"      {_ccode(expr)}{comma}")
   lines.append("   };")
   lines.append("}")
   return "\n".join(lines)


def _emit_kernel_group(
   kernels: tuple[ScalarKernel | StructKernel, ...],
   policy: CsePolicy,
) -> list[str]:
   return [
      _emit_scalar_kernel(kernel, policy)
      if isinstance(kernel, ScalarKernel)
      else _emit_struct_kernel(kernel, policy)
      for kernel in kernels
   ]


def _render_generated_group(body: str, includes: tuple[str, ...]) -> str:
   return "\n".join(
      [
         "#pragma once",
         "",
         "// Generated by pyfluid_amrex/codegen/generate_kernels.py. Do not edit by hand.",
         *includes,
         "",
         "namespace pyfluid_amrex::generated {",
         "",
         "using namespace amrex::literals;",
         "",
         body,
         "",
         "} // namespace pyfluid_amrex::generated",
         "",
      ]
   )


def render_flux_header(policy: CsePolicy = CsePolicy()) -> str:
   kernels: list[str] = []
   for direction in range(3):
      kernels.append(_emit_struct_kernel(hydro_flux_equations(direction), policy))
   for direction in range(3):
      kernels.append(_emit_struct_kernel(mhd_flux_equations(direction), policy))
   kernels.append(_emit_dispatch("hydro_flux_orthonormal", "HydroState", "hydro_flux_orthonormal"))
   kernels.append(_emit_dispatch("mhd_flux_orthonormal", "MHDState", "mhd_flux_orthonormal"))
   kernels.append(_emit_dispatch("hydro_flux_cartesian", "HydroState", "hydro_flux_orthonormal"))
   kernels.append(_emit_dispatch("mhd_flux_cartesian", "MHDState", "mhd_flux_orthonormal"))
   body = "\n\n".join(kernels)
   return "\n".join(
      [
         "#pragma once",
         "",
         "// Generated by pyfluid_amrex/codegen/generate_kernels.py. Do not edit by hand.",
         '#include "pyfluid_amrex/constants.hpp"',
         "",
         '#include "pyfluid_amrex/relativistic_state.hpp"',
         '#include "pyfluid_amrex/state.hpp"',
         "",
         "namespace pyfluid_amrex::generated {",
         "",
         "using namespace amrex::literals;",
         "",
         body,
         "",
         "} // namespace pyfluid_amrex::generated",
         "",
      ]
   )


def render_eos_header(policy: CsePolicy = CsePolicy()) -> str:
   body = "\n\n".join(_emit_scalar_kernel(kernel, policy) for kernel in eos_kernels())
   return "\n".join(
      [
         "#pragma once",
         "",
         "// Generated by pyfluid_amrex/codegen/generate_kernels.py. Do not edit by hand.",
         '#include "pyfluid_amrex/constants.hpp"',
         "",
         "#include <AMReX_GpuQualifiers.H>",
         "#include <AMReX_REAL.H>",
         "",
         "#include <cmath>",
         "",
         "namespace pyfluid_amrex::generated {",
         "",
         "using namespace amrex::literals;",
         "",
         body,
         "",
         "} // namespace pyfluid_amrex::generated",
         "",
      ]
   )


def render_riemann_header(policy: CsePolicy = CsePolicy()) -> str:
   kernels = [_emit_struct_kernel(kernel, policy) for kernel in riemann_flux_kernels()]
   body = "\n\n".join(kernels)
   return "\n".join(
      [
         "#pragma once",
         "",
         "// Generated by pyfluid_amrex/codegen/generate_kernels.py. Do not edit by hand.",
         '#include "pyfluid_amrex/constants.hpp"',
         "",
         '#include "pyfluid_amrex/state.hpp"',
         "",
         "namespace pyfluid_amrex::generated {",
         "",
         "using namespace amrex::literals;",
         "",
         body,
         "",
         "} // namespace pyfluid_amrex::generated",
         "",
      ]
   )


def _relativistic_flux_wrappers() -> tuple[str, ...]:
   return (
      "\n".join(
         [
            "template <int Direction>",
            "AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE",
            "RHDConserved rhd_flux(const RHDPrimitive& primitive, const SpacetimeMetric& metric) noexcept",
            "{",
            "   static_assert(Direction >= 0 && Direction < 3);",
            "   if constexpr (Direction == 0) {",
            "      return rhd_flux_dir0(primitive, metric);",
            "   } else if constexpr (Direction == 1) {",
            "      return rhd_flux_dir1(primitive, metric);",
            "   } else {",
            "      return rhd_flux_dir2(primitive, metric);",
            "   }",
            "}",
         ]
      ),
      "\n".join(
         [
            "template <int Direction>",
            "AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE",
            "RMHDConserved rmhd_flux(const RMHDPrimitive& primitive, const SpacetimeMetric& metric) noexcept",
            "{",
            "   static_assert(Direction >= 0 && Direction < 3);",
            "   if constexpr (Direction == 0) {",
            "      return rmhd_flux_dir0(primitive, metric);",
            "   } else if constexpr (Direction == 1) {",
            "      return rmhd_flux_dir1(primitive, metric);",
            "   } else {",
            "      return rmhd_flux_dir2(primitive, metric);",
            "   }",
            "}",
         ]
      ),
      "\n".join(
         [
            "template <int Direction>",
            "AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE",
            "RMHDWaveSpeeds srmhd_wave_speeds(const RMHDPrimitive& primitive, const SpacetimeMetric& metric, amrex::Real gamma) noexcept",
            "{",
            "   static_assert(Direction >= 0 && Direction < 3);",
            "   if constexpr (Direction == 0) {",
            "      return srmhd_wave_speeds_dir0(primitive, metric, gamma);",
            "   } else if constexpr (Direction == 1) {",
            "      return srmhd_wave_speeds_dir1(primitive, metric, gamma);",
            "   } else {",
            "      return srmhd_wave_speeds_dir2(primitive, metric, gamma);",
            "   }",
            "}",
         ]
      ),
      "\n".join(
         [
            "template <int Direction>",
            "AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE",
            "RHDWaveSpeeds rhd_wave_speeds(const RHDPrimitive& primitive, const SpacetimeMetric& metric, amrex::Real sound_speed_squared) noexcept",
            "{",
            "   static_assert(Direction >= 0 && Direction < 3);",
            "   if constexpr (Direction == 0) {",
            "      return rhd_wave_speeds_dir0(primitive, metric, sound_speed_squared);",
            "   } else if constexpr (Direction == 1) {",
            "      return rhd_wave_speeds_dir1(primitive, metric, sound_speed_squared);",
            "   } else {",
            "      return rhd_wave_speeds_dir2(primitive, metric, sound_speed_squared);",
            "   }",
            "}",
         ]
      ),
   )


def render_relativistic_headers(policy: CsePolicy = CsePolicy()) -> dict[str, str]:
   groups = relativistic_kernel_groups()
   common_includes = (
      '#include "pyfluid_amrex/constants.hpp"',
      '#include "pyfluid_amrex/relativistic_state.hpp"',
      "#include <cmath>",
   )
   rendered = {
      name: _render_generated_group(
         "\n\n".join(_emit_kernel_group(kernels, policy)),
         common_includes
         + (('#include "pyfluid_amrex/c2p_root.hpp"',) if name == "c2p_rmhd" else ()),
      )
      for name, kernels in groups.items()
   }
   flux_kernels = _emit_kernel_group(groups["flux"], policy)
   flux_kernels.extend(_relativistic_flux_wrappers())
   rendered["flux"] = _render_generated_group(
      "\n\n".join(flux_kernels), common_includes
   )
   return rendered


def render_relativistic_header(policy: CsePolicy = CsePolicy()) -> str:
   del policy
   includes = (
      '#include "pyfluid_amrex/kernels_generated/relativistic/core.hpp"',
      '#include "pyfluid_amrex/kernels_generated/relativistic/flux.hpp"',
      '#include "pyfluid_amrex/kernels_generated/relativistic/sources.hpp"',
      '#include "pyfluid_amrex/kernels_generated/relativistic/c2p_rhd.hpp"',
      '#include "pyfluid_amrex/kernels_generated/relativistic/c2p_rmhd.hpp"',
   )
   return "\n".join(("#pragma once", "", *includes, ""))


def render_initial_data_header(policy: CsePolicy = CsePolicy()) -> str:
   body = "\n\n".join(
      _emit_scalar_kernel(kernel, policy)
      if isinstance(kernel, ScalarKernel)
      else _emit_struct_kernel(kernel, policy)
      for kernel in initial_data_kernels()
   )
   return "\n".join(
      [
         "#pragma once",
         "",
         "// Generated by pyfluid_amrex/codegen/generate_kernels.py. Do not edit by hand.",
         '#include "pyfluid_amrex/constants.hpp"',
         '#include "pyfluid_amrex/initial_data/ctt.hpp"',
         '#include "pyfluid_amrex/z4c_state.hpp"',
         "",
         "namespace pyfluid_amrex::generated {",
         "",
         "using namespace amrex::literals;",
         "",
         body,
         "",
         "} // namespace pyfluid_amrex::generated",
         "",
      ]
   )


def _z4c_constraint_wrappers() -> str:
   return "\n".join(
      [
         "AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE",
         "Z4cConstraints evaluate_z4c_constraints(",
         "   const Z4cState& state,",
         "   const Z4cSpatialDerivatives& derivatives,",
         "   const Z4cMatter& matter",
         ") noexcept",
         "{",
         "   const Z4cConformalMetric metric = z4c_conformal_metric(state);",
         "   const Z4cConnections connections = z4c_connections(state, derivatives, metric);",
         "   const Z4cDiagnosticGeometry diagnostic_geometry = z4c_diagnostic_geometry(",
         "      state, derivatives, metric, connections",
         "   );",
         "   return z4c_constraints_from_geometry(",
         "      state, derivatives, metric, connections, diagnostic_geometry, matter",
         "   );",
         "}",
         "",
         "AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE",
         "Z4cConstraints evaluate_vacuum_z4c_constraints(",
         "   const Z4cState& state,",
         "   const Z4cSpatialDerivatives& derivatives",
         ") noexcept",
         "{",
         "   return evaluate_z4c_constraints(state, derivatives, vacuum_z4c_matter());",
         "}",
      ]
   )


def _z4c_evolution_wrappers() -> str:
   return "\n".join(
      [
         "",
         "template <Z4cLapseGauge LapseGauge, Z4cShiftGauge ShiftGauge>",
         "AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE",
         "Z4cRhs evaluate_z4c_rhs(",
         "   const Z4cState& state,",
         "   const Z4cSpatialDerivatives& derivatives,",
         "   const Z4cMatter& matter,",
         "   amrex::Real time,",
         "   const Z4cEvolutionParameters& parameters",
         ") noexcept",
         "{",
         "   const Z4cConformalMetric metric = z4c_conformal_metric(state);",
         "   const Z4cConnections connections = z4c_connections(state, derivatives, metric);",
         "   const Z4cEvolutionGeometry evolution_geometry = z4c_evolution_geometry(",
         "      state, derivatives, metric, connections",
         "   );",
         "   const Z4cGeometricRhs geometry = z4c_geometric_rhs(",
         "      state,",
         "      derivatives,",
         "      metric,",
         "      connections,",
         "      evolution_geometry,",
         "      matter,",
         "      parameters.kappa1,",
         "      parameters.kappa2",
         "   );",
         "   amrex::Real lapse = 0.0_rt;",
         "   if constexpr (LapseGauge == Z4cLapseGauge::Harmonic) {",
         "      lapse = z4c_harmonic_lapse_rhs(state, derivatives);",
         "   } else if constexpr (LapseGauge == Z4cLapseGauge::OnePlusLog) {",
         "      lapse = z4c_one_plus_log_lapse_rhs(state, derivatives);",
         "   } else if constexpr (LapseGauge == Z4cLapseGauge::SlowStart) {",
         "      lapse = z4c_slow_start_lapse_rhs(",
         "         state,",
         "         derivatives,",
         "         time,",
         "         parameters.slow_start_strength,",
         "         parameters.slow_start_width",
         "      );",
         "   }",
         "   Z4cShiftRhs shift{};",
         "   if constexpr (ShiftGauge == Z4cShiftGauge::GammaDriver) {",
         "      shift = z4c_gamma_driver_shift_rhs(",
         "         state, derivatives, parameters.shift_mu, parameters.shift_eta",
         "      );",
         "   }",
         "   return {geometry, lapse, shift};",
         "}",
         "",
         "template <Z4cLapseGauge LapseGauge, Z4cShiftGauge ShiftGauge>",
         "AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE",
         "Z4cRhs evaluate_vacuum_z4c_rhs(",
         "   const Z4cState& state,",
         "   const Z4cSpatialDerivatives& derivatives,",
         "   amrex::Real time,",
         "   const Z4cEvolutionParameters& parameters",
         ") noexcept",
         "{",
         "   return evaluate_z4c_rhs<LapseGauge, ShiftGauge>(",
         "      state, derivatives, vacuum_z4c_matter(), time, parameters",
         "   );",
         "}",
      ]
   )


def render_z4c_headers(policy: CsePolicy = CsePolicy()) -> dict[str, str]:
   groups = z4c_kernel_groups()
   common_includes = (
      '#include "pyfluid_amrex/constants.hpp"',
      '#include "pyfluid_amrex/z4c_state.hpp"',
      "#include <AMReX_Math.H>",
      "#include <cmath>",
   )
   rendered = {
      name: _render_generated_group(
         "\n\n".join(_emit_kernel_group(kernels, policy)), common_includes
      )
      for name, kernels in groups.items()
   }
   rendered["constraints"] = _render_generated_group(
      "\n\n".join(
         (
            *_emit_kernel_group(groups["constraints"], policy),
            _z4c_constraint_wrappers(),
         )
      ),
      common_includes
      + ('#include "pyfluid_amrex/kernels_generated/z4c/geometry.hpp"',),
   )
   rendered["evolution"] = _render_generated_group(
      "\n\n".join(
         (
            *_emit_kernel_group(groups["evolution"], policy),
            _z4c_evolution_wrappers(),
         )
      ),
      common_includes
      + (
         '#include "pyfluid_amrex/kernels_generated/z4c/geometry.hpp"',
         '#include "pyfluid_amrex/kernels_generated/z4c/gauge.hpp"',
      ),
   )
   return rendered


def render_z4c_header(policy: CsePolicy = CsePolicy()) -> str:
   del policy
   includes = (
      '#include "pyfluid_amrex/kernels_generated/z4c/geometry.hpp"',
      '#include "pyfluid_amrex/kernels_generated/z4c/constraints.hpp"',
      '#include "pyfluid_amrex/kernels_generated/z4c/gauge.hpp"',
      '#include "pyfluid_amrex/kernels_generated/z4c/evolution.hpp"',
   )
   return "\n".join(("#pragma once", "", *includes, ""))


def render_characteristic_header(policy: CsePolicy = CsePolicy()) -> str:
   body = "\n\n".join(_emit_struct_kernel(kernel, policy) for kernel in characteristic_kernels())
   return "\n".join(
      [
         "#pragma once",
         "",
         "// Generated by pyfluid_amrex/codegen/generate_kernels.py. Do not edit by hand.",
         '#include "pyfluid_amrex/constants.hpp"',
         "",
         '#include "pyfluid_amrex/characteristics.hpp"',
         "",
         "namespace pyfluid_amrex::generated {",
         "",
         "using namespace amrex::literals;",
         "",
         body,
         "",
         "} // namespace pyfluid_amrex::generated",
         "",
      ]
   )


def render_operator_header(policy: CsePolicy = CsePolicy()) -> str:
   kernels = [_emit_scalar_kernel(kernel, policy) for kernel in scalar_operator_kernels()]
   kernels.append(
      _emit_scalar_dim_dispatch(
         "laplacian_cartesian",
         "laplacian_cartesian",
         ("center", "xm", "xp", "ym", "yp", "zm", "zp", "inv_dx2", "inv_dy2", "inv_dz2"),
      )
   )
   kernels.append(
      _emit_scalar_dim_dispatch(
         "implicit_relaxed_cartesian",
         "implicit_relaxed_cartesian",
         ("base", "xm", "xp", "ym", "yp", "zm", "zp", "cx", "cy", "cz"),
      )
   )
   kernels.append(
      _emit_scalar_dim_dispatch(
         "implicit_residual_cartesian",
         "implicit_residual_cartesian",
         ("base", "center", "xm", "xp", "ym", "yp", "zm", "zp", "cx", "cy", "cz"),
      )
   )
   body = "\n\n".join(kernels)
   return "\n".join(
      [
         "#pragma once",
         "",
         "// Generated by pyfluid_amrex/codegen/generate_kernels.py. Do not edit by hand.",
         '#include "pyfluid_amrex/constants.hpp"',
         "",
         "#include <AMReX_GpuQualifiers.H>",
         "#include <AMReX_REAL.H>",
         "",
         "namespace pyfluid_amrex::generated {",
         "",
         "using namespace amrex::literals;",
         "",
         body,
         "",
         "} // namespace pyfluid_amrex::generated",
         "",
      ]
   )


def _fd_weight(weight: sp.Rational) -> str:
   if weight == 0:
      return "0.0_rt"
   if weight.q == 1:
      return f"{weight.p}.0_rt"
   return f"({weight.p}.0_rt/{weight.q}.0_rt)"


def _fd_expression(offsets: tuple[int, ...], weights: tuple[sp.Rational, ...]) -> str:
   terms = [
      f"{_fd_weight(weight)} * sample_direction<Direction>(field, i, j, k, component, {offset})"
      for offset, weight in zip(offsets, weights, strict=True)
      if weight != 0
   ]
   return " + ".join(terms).replace("+ -", "- ")


def _mixed_fd_expression(
   offsets: tuple[int, ...],
   weights: tuple[sp.Rational, ...],
) -> str:
   terms = []
   for offset1, weight1 in zip(offsets, weights, strict=True):
      for offset2, weight2 in zip(offsets, weights, strict=True):
         weight = sp.Rational(weight1 * weight2)
         if weight == 0:
            continue
         terms.append(
            f"{_fd_weight(weight)} * sample_directions<Direction1, Direction2>("
            f"field, i, j, k, component, {offset1}, {offset2})"
         )
   return " + ".join(terms).replace("+ -", "- ")


def _fd_dispatch(name: str, stencils: tuple, scale: str) -> str:
   lines = [
      "template <int Order, int Direction, typename Array>",
      "AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE",
      f"amrex::Real {name}(Array const& field, int i, int j, int k, int component, amrex::Real {scale}) noexcept",
      "{",
      "   static_assert(Order == 2 || Order == 4 || Order == 6 || Order == 8);",
   ]
   for index, stencil in enumerate(stencils):
      prefix = "if" if index == 0 else "else if"
      expression = _fd_expression(stencil.offsets, stencil.weights)
      lines.extend(
         [
            f"   {prefix} constexpr (Order == {stencil.order}) {{",
            f"      return {scale} * ({expression});",
            "   }",
         ]
      )
   lines.extend(["   return 0.0_rt;", "}"])
   return "\n".join(lines)


def render_differential_header() -> str:
   first = _fd_dispatch("centered_first", centered_stencils(1), "inv_spacing")
   second = _fd_dispatch("centered_second", centered_stencils(2), "inv_spacing_squared")
   ko = _fd_dispatch("kreiss_oliger", ko_stencils(), "inv_spacing")

   mixed_lines = [
      "template <int Order, int Direction1, int Direction2, typename Array>",
      "AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE",
      "amrex::Real centered_mixed(Array const& field, int i, int j, int k, int component, amrex::Real inv_spacing_product) noexcept",
      "{",
      "   static_assert(Order == 2 || Order == 4 || Order == 6 || Order == 8);",
      "   static_assert(Direction1 != Direction2);",
   ]
   for index, stencil in enumerate(centered_stencils(1)):
      prefix = "if" if index == 0 else "else if"
      expression = _mixed_fd_expression(stencil.offsets, stencil.weights)
      mixed_lines.extend(
         [
            f"   {prefix} constexpr (Order == {stencil.order}) {{",
            f"      return inv_spacing_product * ({expression});",
            "   }",
         ]
      )
   mixed_lines.extend(["   return 0.0_rt;", "}"])

   upwind_lines = [
      "template <int Order, int Direction, typename Array>",
      "AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE",
      "amrex::Real upwind_first(Array const& field, int i, int j, int k, int component, amrex::Real speed, amrex::Real inv_spacing) noexcept",
      "{",
      "   static_assert(Order == 2 || Order == 4 || Order == 6 || Order == 8);",
   ]
   for index, (backward, forward) in enumerate(upwind_stencils()):
      prefix = "if" if index == 0 else "else if"
      backward_expression = _fd_expression(backward.offsets, backward.weights)
      forward_expression = _fd_expression(forward.offsets, forward.weights)
      upwind_lines.extend(
         [
            f"   {prefix} constexpr (Order == {backward.order}) {{",
            f"      const amrex::Real derivative = speed >= 0.0_rt ? ({backward_expression}) : ({forward_expression});",
            "      return inv_spacing * derivative;",
            "   }",
         ]
      )
   upwind_lines.extend(["   return 0.0_rt;", "}"])

   return "\n".join(
      [
         "#pragma once",
         "",
         "// Generated by pyfluid_amrex/codegen/generate_kernels.py. Do not edit by hand.",
         '#include "pyfluid_amrex/constants.hpp"',
         "",
         "#include <AMReX_GpuQualifiers.H>",
         "#include <AMReX_REAL.H>",
         "",
         "namespace pyfluid_amrex::generated {",
         "",
         "using namespace amrex::literals;",
         "",
         "template <int Direction, typename Array>",
         "AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE",
         "amrex::Real sample_direction(Array const& field, int i, int j, int k, int component, int offset) noexcept",
         "{",
         "   static_assert(Direction >= 0 && Direction < 3);",
         "   if constexpr (Direction == 0) return field(i + offset, j, k, component);",
         "   if constexpr (Direction == 1) return field(i, j + offset, k, component);",
         "   return field(i, j, k + offset, component);",
         "}",
         "",
         "template <int Direction1, int Direction2, typename Array>",
         "AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE",
         "amrex::Real sample_directions(Array const& field, int i, int j, int k, int component, int offset1, int offset2) noexcept",
         "{",
         "   static_assert(Direction1 >= 0 && Direction1 < 3);",
         "   static_assert(Direction2 >= 0 && Direction2 < 3);",
         "   if constexpr (Direction1 == 0) i += offset1;",
         "   if constexpr (Direction1 == 1) j += offset1;",
         "   if constexpr (Direction1 == 2) k += offset1;",
         "   if constexpr (Direction2 == 0) i += offset2;",
         "   if constexpr (Direction2 == 1) j += offset2;",
         "   if constexpr (Direction2 == 2) k += offset2;",
         "   return field(i, j, k, component);",
         "}",
         "",
         first,
         "",
         second,
         "",
         "\n".join(mixed_lines),
         "",
         "\n".join(upwind_lines),
         "",
         ko,
         "",
         "template <int Direction, typename Array>",
         "AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE",
         "amrex::Real cell_to_face(Array const& field, int i, int j, int k, int component) noexcept",
         "{",
         "   return 0.5_rt * (sample_direction<Direction>(field, i, j, k, component, -1) + field(i, j, k, component));",
         "}",
         "",
         "template <int Direction, typename Array>",
         "AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE",
         "amrex::Real face_to_cell(Array const& field, int i, int j, int k, int component) noexcept",
         "{",
         "   return 0.5_rt * (field(i, j, k, component) + sample_direction<Direction>(field, i, j, k, component, 1));",
         "}",
         "",
         "template <int Order>",
         "AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE",
         "amrex::Real centering_transfer_weight(int ordinal) noexcept",
         "{",
         "   static_assert(Order == 2 || Order == 4 || Order == 6 || Order == 8);",
         "   if constexpr (Order == 2) return 0.5_rt;",
         "   if constexpr (Order == 4) {",
         "      constexpr amrex::Real weights[4] = {-0.0625_rt, 0.5625_rt, 0.5625_rt, -0.0625_rt};",
         "      return weights[ordinal];",
         "   }",
         "   if constexpr (Order == 6) {",
         "      constexpr amrex::Real weights[6] = {3.0_rt / 256.0_rt, -25.0_rt / 256.0_rt, 150.0_rt / 256.0_rt, 150.0_rt / 256.0_rt, -25.0_rt / 256.0_rt, 3.0_rt / 256.0_rt};",
         "      return weights[ordinal];",
         "   }",
         "   constexpr amrex::Real weights[8] = {-5.0_rt / 2048.0_rt, 49.0_rt / 2048.0_rt, -245.0_rt / 2048.0_rt, 1225.0_rt / 2048.0_rt, 1225.0_rt / 2048.0_rt, -245.0_rt / 2048.0_rt, 49.0_rt / 2048.0_rt, -5.0_rt / 2048.0_rt};",
         "   return weights[ordinal];",
         "}",
         "",
         "template <int Order>",
         "AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE",
         "int centering_transfer_count(int source_mask, int destination_mask, int direction) noexcept",
         "{",
         "   return ((source_mask ^ destination_mask) & (1 << direction)) != 0 ? Order : 1;",
         "}",
         "",
         "template <int Order>",
         "AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE",
         "int centering_transfer_offset(int source_mask, int destination_mask, int direction, int ordinal) noexcept",
         "{",
         "   if (((source_mask ^ destination_mask) & (1 << direction)) == 0) return 0;",
         "   const bool source_is_nodal = (source_mask & (1 << direction)) != 0;",
         "   return (source_is_nodal ? 1 - Order / 2 : -Order / 2) + ordinal;",
         "}",
         "",
         "template <int Order>",
         "AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE",
         "amrex::Real centering_transfer_axis_weight(int source_mask, int destination_mask, int direction, int ordinal) noexcept",
         "{",
         "   return ((source_mask ^ destination_mask) & (1 << direction)) != 0",
         "      ? centering_transfer_weight<Order>(ordinal)",
         "      : 1.0_rt;",
         "}",
         "",
         "template <int Order, int Dim, typename Array>",
         "AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE",
         "amrex::Real topology_centering_transfer(Array const& field, int i, int j, int k, int component, int source_mask, int destination_mask) noexcept",
         "{",
         "   static_assert(Order == 2 || Order == 4 || Order == 6 || Order == 8);",
         "   static_assert(Dim >= 1 && Dim <= 3);",
         "   source_mask &= (1 << Dim) - 1;",
         "   destination_mask &= (1 << Dim) - 1;",
         "   amrex::Real result = 0.0_rt;",
         "   const int count1 = centering_transfer_count<Order>(source_mask, destination_mask, 0);",
         "   for (int ordinal1 = 0; ordinal1 < count1; ++ordinal1) {",
         "      const int offset1 = centering_transfer_offset<Order>(source_mask, destination_mask, 0, ordinal1);",
         "      const amrex::Real weight1 = centering_transfer_axis_weight<Order>(source_mask, destination_mask, 0, ordinal1);",
         "      if constexpr (Dim == 1) {",
         "         result += weight1 * field(i + offset1, j, k, component);",
         "      } else {",
         "         const int count2 = centering_transfer_count<Order>(source_mask, destination_mask, 1);",
         "         for (int ordinal2 = 0; ordinal2 < count2; ++ordinal2) {",
         "            const int offset2 = centering_transfer_offset<Order>(source_mask, destination_mask, 1, ordinal2);",
         "            const amrex::Real weight12 = weight1 * centering_transfer_axis_weight<Order>(source_mask, destination_mask, 1, ordinal2);",
         "            if constexpr (Dim == 2) {",
         "               result += weight12 * field(i + offset1, j + offset2, k, component);",
         "            } else {",
         "               const int count3 = centering_transfer_count<Order>(source_mask, destination_mask, 2);",
         "               for (int ordinal3 = 0; ordinal3 < count3; ++ordinal3) {",
         "                  const int offset3 = centering_transfer_offset<Order>(source_mask, destination_mask, 2, ordinal3);",
         "                  result += weight12 * centering_transfer_axis_weight<Order>(source_mask, destination_mask, 2, ordinal3) * field(i + offset1, j + offset2, k + offset3, component);",
         "               }",
         "            }",
         "         }",
         "      }",
         "   }",
         "   return result;",
         "}",
         "",
         "template <int Order, int Dim, bool CellToVertex, typename Array>",
         "AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE",
         "amrex::Real centering_transfer(Array const& field, int i, int j, int k, int component) noexcept",
         "{",
         "   constexpr int vertex_mask = (1 << Dim) - 1;",
         "   constexpr int source_mask = CellToVertex ? 0 : vertex_mask;",
         "   constexpr int destination_mask = CellToVertex ? vertex_mask : 0;",
         "   return topology_centering_transfer<Order, Dim>(",
         "      field, i, j, k, component, source_mask, destination_mask",
         "   );",
         "}",
         "",
         "template <int Order, int Direction, bool CellToFace, typename Array>",
         "AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE",
         "amrex::Real centering_transfer_1d(Array const& field, int i, int j, int k, int component) noexcept",
         "{",
         "   static_assert(Order == 2 || Order == 4 || Order == 6 || Order == 8);",
         "   static_assert(Direction >= 0 && Direction < 3);",
         "   constexpr int first_offset = CellToFace ? -Order / 2 : 1 - Order / 2;",
         "   amrex::Real result = 0.0_rt;",
         "   for (int ordinal = 0; ordinal < Order; ++ordinal) {",
         "      result += centering_transfer_weight<Order>(ordinal) * sample_direction<Direction>(field, i, j, k, component, first_offset + ordinal);",
         "   }",
         "   return result;",
         "}",
         "",
         "template <int Order, int Direction, typename Array>",
         "AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE",
         "amrex::Real cell_to_face_ordered(Array const& field, int i, int j, int k, int component) noexcept",
         "{",
         "   return centering_transfer_1d<Order, Direction, true>(field, i, j, k, component);",
         "}",
         "",
         "template <int Order, int Direction, typename Array>",
         "AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE",
         "amrex::Real face_to_cell_ordered(Array const& field, int i, int j, int k, int component) noexcept",
         "{",
         "   return centering_transfer_1d<Order, Direction, false>(field, i, j, k, component);",
         "}",
         "",
         "template <int Order, int Dim, typename Array>",
         "AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE",
         "amrex::Real vertex_to_cell(Array const& field, int i, int j, int k, int component) noexcept",
         "{",
         "   return centering_transfer<Order, Dim, false>(field, i, j, k, component);",
         "}",
         "",
         "template <int Order, int Dim, typename Array>",
         "AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE",
         "amrex::Real cell_to_vertex(Array const& field, int i, int j, int k, int component) noexcept",
         "{",
         "   return centering_transfer<Order, Dim, true>(field, i, j, k, component);",
         "}",
         "",
         "} // namespace pyfluid_amrex::generated",
         "",
      ]
   )


def render_vector_header(policy: CsePolicy = CsePolicy()) -> str:
   kernels = [_emit_scalar_kernel(kernel, policy) for kernel in vector_operator_kernels()]
   body = "\n\n".join(kernels)
   return "\n".join(
      [
         "#pragma once",
         "",
         "// Generated by pyfluid_amrex/codegen/generate_kernels.py. Do not edit by hand.",
         '#include "pyfluid_amrex/constants.hpp"',
         "",
         "#include <AMReX_GpuQualifiers.H>",
         "#include <AMReX_REAL.H>",
         "#include <AMReX_Math.H>",
         "",
         "#include <cmath>",
         "",
         "namespace pyfluid_amrex::generated {",
         "",
         "using namespace amrex::literals;",
         "",
         body,
         "",
         "} // namespace pyfluid_amrex::generated",
         "",
      ]
   )


def render_metric_header(policy: CsePolicy = CsePolicy()) -> str:
   kernels = [_emit_scalar_kernel(kernel, policy) for kernel in metric_kernels()]
   body = "\n\n".join(kernels)
   return "\n".join(
      [
         "#pragma once",
         "",
         "// Generated by pyfluid_amrex/codegen/generate_kernels.py. Do not edit by hand.",
         '#include "pyfluid_amrex/constants.hpp"',
         "",
         "#include <AMReX_GpuQualifiers.H>",
         "#include <AMReX_REAL.H>",
         "",
         "#include <cmath>",
         "",
         "namespace pyfluid_amrex::generated {",
         "",
         "using namespace amrex::literals;",
         "",
         body,
         "",
         "} // namespace pyfluid_amrex::generated",
         "",
      ]
   )


def render_patch_map_header(policy: CsePolicy = CsePolicy()) -> str:
   kernels = [_emit_struct_kernel(kernel, policy) for kernel in patch_map_kernels()]
   body = "\n\n".join(kernels)
   return "\n".join(
      [
         "#pragma once",
         "",
         "// Generated by pyfluid_amrex/codegen/generate_kernels.py. Do not edit by hand.",
         '#include "pyfluid_amrex/constants.hpp"',
         "",
         "#include <AMReX_GpuQualifiers.H>",
         "#include <AMReX_REAL.H>",
         "#include <AMReX_Math.H>",
         "",
         "#include <cmath>",
         "",
         "namespace pyfluid_amrex::generated {",
         "",
         "using namespace amrex::literals;",
         "",
         "struct PatchPoint {",
         "   amrex::Real q1;",
         "   amrex::Real q2;",
         "   amrex::Real q3;",
         "};",
         "",
         "struct PatchJacobian {",
         "   amrex::Real j11;",
         "   amrex::Real j12;",
         "   amrex::Real j13;",
         "   amrex::Real j21;",
         "   amrex::Real j22;",
         "   amrex::Real j23;",
         "   amrex::Real j31;",
         "   amrex::Real j32;",
         "   amrex::Real j33;",
         "};",
         "",
         body,
         "",
         "} // namespace pyfluid_amrex::generated",
         "",
      ]
   )


def _write_if_changed(path: Path, content: str) -> None:
   path.parent.mkdir(parents=True, exist_ok=True)
   if path.exists() and path.read_text(encoding="utf-8") == content:
      return
   path.write_text(content, encoding="utf-8")


def write_flux_header(path: Path, policy: CsePolicy = CsePolicy()) -> None:
   _write_if_changed(path, render_flux_header(policy))


def write_eos_header(path: Path, policy: CsePolicy = CsePolicy()) -> None:
   _write_if_changed(path, render_eos_header(policy))


def write_riemann_header(path: Path, policy: CsePolicy = CsePolicy()) -> None:
   _write_if_changed(path, render_riemann_header(policy))


def write_relativistic_header(path: Path, policy: CsePolicy = CsePolicy()) -> None:
   for name, content in render_relativistic_headers(policy).items():
      _write_if_changed(path.parent / "relativistic" / f"{name}.hpp", content)
   _write_if_changed(path, render_relativistic_header(policy))


def write_initial_data_header(path: Path, policy: CsePolicy = CsePolicy()) -> None:
   _write_if_changed(path, render_initial_data_header(policy))


def write_z4c_header(path: Path, policy: CsePolicy = CsePolicy()) -> None:
   for name, content in render_z4c_headers(policy).items():
      _write_if_changed(path.parent / "z4c" / f"{name}.hpp", content)
   _write_if_changed(path, render_z4c_header(policy))


def write_characteristic_header(path: Path, policy: CsePolicy = CsePolicy()) -> None:
   _write_if_changed(path, render_characteristic_header(policy))


def write_operator_header(path: Path, policy: CsePolicy = CsePolicy()) -> None:
   _write_if_changed(path, render_operator_header(policy))


def write_differential_header(path: Path, policy: CsePolicy = CsePolicy()) -> None:
   del policy
   _write_if_changed(path, render_differential_header())


def write_vector_header(path: Path, policy: CsePolicy = CsePolicy()) -> None:
   _write_if_changed(path, render_vector_header(policy))


def write_metric_header(path: Path, policy: CsePolicy = CsePolicy()) -> None:
   _write_if_changed(path, render_metric_header(policy))


def write_patch_map_header(path: Path, policy: CsePolicy = CsePolicy()) -> None:
   _write_if_changed(path, render_patch_map_header(policy))
