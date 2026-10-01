"""C++ value descriptors for the checked exact time-integrator catalog."""
from fractions import Fraction

from .time_integrator_catalog import TIME_INTEGRATORS


def literal(value: Fraction) -> str:
    if value.denominator == 1:
        return f"{value.numerator}.0_rt"
    return f"({value.numerator}.0_rt / {value.denominator}.0_rt)"


def render_time_integrator_ids_header() -> str:
    lines = ["// Generated from schema/time_integrators.toml. Do not edit.",
             "#pragma once", "namespace pyfluid_amrex {", "enum TimeIntegratorCode : int {"]
    lines.extend(f"   {method.enumerator} = {method.code}," for method in TIME_INTEGRATORS)
    lines.extend(["};", "} // namespace pyfluid_amrex\n"])
    return "\n".join(lines)


def render_time_integrator_header() -> str:
    lines = ["// Generated from schema/time_integrators.toml. Do not edit.",
             "#pragma once", "#include <AMReX_REAL.H>", "#include <array>", "#include <string_view>",
             '#include "pyfluid_amrex/generated/time_integrator_ids.hpp"', "",
             "namespace pyfluid_amrex {", "using namespace amrex::literals;"]
    lines.extend(["struct TimeIntegratorDescriptor {",
                  "   std::string_view name; int code; int order; int stages;", "};",
                  f"inline constexpr std::array<TimeIntegratorDescriptor, {len(TIME_INTEGRATORS)}> TimeIntegratorCatalog{{{{"])
    lines.extend(f'   {{"{method.name}", {method.enumerator}, {method.order}, {len(method.b)}}},'
                 for method in TIME_INTEGRATORS)
    lines.extend(["}};", "struct RungeKuttaScale {",
                  "   amrex::Real numerator;", "   amrex::Real denominator;",
                  "   constexpr amrex::Real apply(amrex::Real dt) const noexcept",
                  "   {", "      return numerator * dt / denominator;", "   }", "};",
                  "struct RungeKuttaStage {", "   RungeKuttaScale time;", "   RungeKuttaScale output_time;",
                  "   amrex::Real base_weight;", "   amrex::Real input_weight;", "   RungeKuttaScale rhs_weight;",
                  "   RungeKuttaScale final_rhs_weight;", "   amrex::Real projection_weight;",
                  "};", "",
                  "template <int Method>", "constexpr auto const& time_integrator_stages() noexcept;", ""])
    for method in TIME_INTEGRATORS:
        name = method.enumerator + "Stages"
        lines.append(f"inline constexpr std::array<RungeKuttaStage, {len(method.b)}> {name}{{{{")
        for index in range(len(method.b)):
            base = Fraction(0) if index == 0 else method.state[index][0]
            current = Fraction(1) if index == 0 else method.state[index][index]
            values = (method.c[index], method.output_times[index], base, current,
                      method.rhs[index][index], method.b[index], method.projection_weights[index])
            entries = [
                ("{" + f"{value.numerator}.0_rt, {value.denominator}.0_rt" + "}")
                if position in (0, 1, 4, 5) else literal(value)
                for position, value in enumerate(values)
            ]
            lines.append("   {" + ", ".join(entries) + "},")
        lines.extend(["}};", "template <>",
                      f"constexpr auto const& time_integrator_stages<{method.enumerator}>() noexcept",
                      "{", f"   return {name};", "}", ""])
    lines.append("} // namespace pyfluid_amrex\n")
    return "\n".join(lines)
