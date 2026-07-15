from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import sympy as sp
from sympy.printing.c import C99CodePrinter

from .characteristic_equations import characteristic_kernels
from .eos_equations import eos_kernels
from .flux_equations import hydro_flux_equations, mhd_flux_equations
from .kernel_equations import ScalarKernel, StructKernel
from .metric_equations import metric_kernels
from .operator_equations import scalar_operator_kernels
from .patch_map_equations import patch_map_kernels
from .relativistic_equations import relativistic_kernels
from .riemann_equations import riemann_flux_kernels
from .vector_equations import vector_operator_kernels


@dataclass(frozen=True)
class CsePolicy:
   hard_drop_ops: int = 0
   soft_drop_ops: int = 3


class PyFluidCxxPrinter(C99CodePrinter):
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


def render_relativistic_header(policy: CsePolicy = CsePolicy()) -> str:
   kernels = [
      _emit_scalar_kernel(kernel, policy)
      if isinstance(kernel, ScalarKernel)
      else _emit_struct_kernel(kernel, policy)
      for kernel in relativistic_kernels()
   ]
   kernels.append(
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
      )
   )
   kernels.append(
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
      )
   )
   kernels.append(
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
      )
   )
   kernels.append(
      "\n".join(
         [
            "template <int Direction>",
            "AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE",
            "RHDWaveSpeeds srhd_wave_speeds(const RHDPrimitive& primitive, const SpacetimeMetric& metric, amrex::Real gamma) noexcept",
            "{",
            "   static_assert(Direction >= 0 && Direction < 3);",
            "   if constexpr (Direction == 0) {",
            "      return srhd_wave_speeds_dir0(primitive, metric, gamma);",
            "   } else if constexpr (Direction == 1) {",
            "      return srhd_wave_speeds_dir1(primitive, metric, gamma);",
            "   } else {",
            "      return srhd_wave_speeds_dir2(primitive, metric, gamma);",
            "   }",
            "}",
         ]
      )
   )
   body = "\n\n".join(kernels)
   return "\n".join(
      [
         "#pragma once",
         "",
         "// Generated by pyfluid_amrex/codegen/generate_kernels.py. Do not edit by hand.",
         "",
         '#include "pyfluid_amrex/c2p_root.hpp"',
         '#include "pyfluid_amrex/relativistic_state.hpp"',
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


def render_characteristic_header(policy: CsePolicy = CsePolicy()) -> str:
   body = "\n\n".join(_emit_struct_kernel(kernel, policy) for kernel in characteristic_kernels())
   return "\n".join(
      [
         "#pragma once",
         "",
         "// Generated by pyfluid_amrex/codegen/generate_kernels.py. Do not edit by hand.",
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


def render_vector_header(policy: CsePolicy = CsePolicy()) -> str:
   kernels = [_emit_scalar_kernel(kernel, policy) for kernel in vector_operator_kernels()]
   body = "\n\n".join(kernels)
   return "\n".join(
      [
         "#pragma once",
         "",
         "// Generated by pyfluid_amrex/codegen/generate_kernels.py. Do not edit by hand.",
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
   _write_if_changed(path, render_relativistic_header(policy))


def write_characteristic_header(path: Path, policy: CsePolicy = CsePolicy()) -> None:
   _write_if_changed(path, render_characteristic_header(policy))


def write_operator_header(path: Path, policy: CsePolicy = CsePolicy()) -> None:
   _write_if_changed(path, render_operator_header(policy))


def write_vector_header(path: Path, policy: CsePolicy = CsePolicy()) -> None:
   _write_if_changed(path, render_vector_header(policy))


def write_metric_header(path: Path, policy: CsePolicy = CsePolicy()) -> None:
   _write_if_changed(path, render_metric_header(policy))


def write_patch_map_header(path: Path, policy: CsePolicy = CsePolicy()) -> None:
   _write_if_changed(path, render_patch_map_header(policy))
