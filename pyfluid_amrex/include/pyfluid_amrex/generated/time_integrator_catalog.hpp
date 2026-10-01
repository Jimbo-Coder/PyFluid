// Generated from schema/time_integrators.toml. Do not edit.
#pragma once
#include <AMReX_REAL.H>
#include <array>
#include <string_view>
#include "pyfluid_amrex/generated/time_integrator_ids.hpp"

namespace pyfluid_amrex {
using namespace amrex::literals;
struct TimeIntegratorDescriptor {
   std::string_view name; int code; int order; int stages;
};
inline constexpr std::array<TimeIntegratorDescriptor, 5> TimeIntegratorCatalog{{
   {"euler", TimeEuler, 1, 1},
   {"corrector", TimeCorrector, 2, 2},
   {"midpoint", TimeMidpoint, 2, 2},
   {"ssprk3", TimeSSPRK3, 3, 3},
   {"rk4", TimeRK4, 4, 4},
}};
struct RungeKuttaScale {
   amrex::Real numerator;
   amrex::Real denominator;
   constexpr amrex::Real apply(amrex::Real dt) const noexcept
   {
      return numerator * dt / denominator;
   }
};
struct RungeKuttaStage {
   RungeKuttaScale time;
   RungeKuttaScale output_time;
   amrex::Real base_weight;
   amrex::Real input_weight;
   RungeKuttaScale rhs_weight;
   RungeKuttaScale final_rhs_weight;
   amrex::Real projection_weight;
};

template <int Method>
constexpr auto const& time_integrator_stages() noexcept;

inline constexpr std::array<RungeKuttaStage, 1> TimeEulerStages{{
   {{0.0_rt, 1.0_rt}, {1.0_rt, 1.0_rt}, 0.0_rt, 1.0_rt, {1.0_rt, 1.0_rt}, {1.0_rt, 1.0_rt}, 1.0_rt},
}};
template <>
constexpr auto const& time_integrator_stages<TimeEuler>() noexcept
{
   return TimeEulerStages;
}

inline constexpr std::array<RungeKuttaStage, 2> TimeCorrectorStages{{
   {{0.0_rt, 1.0_rt}, {1.0_rt, 1.0_rt}, 0.0_rt, 1.0_rt, {1.0_rt, 1.0_rt}, {1.0_rt, 2.0_rt}, 1.0_rt},
   {{1.0_rt, 1.0_rt}, {1.0_rt, 1.0_rt}, (1.0_rt / 2.0_rt), (1.0_rt / 2.0_rt), {1.0_rt, 2.0_rt}, {1.0_rt, 2.0_rt}, (1.0_rt / 2.0_rt)},
}};
template <>
constexpr auto const& time_integrator_stages<TimeCorrector>() noexcept
{
   return TimeCorrectorStages;
}

inline constexpr std::array<RungeKuttaStage, 2> TimeMidpointStages{{
   {{0.0_rt, 1.0_rt}, {1.0_rt, 2.0_rt}, 0.0_rt, 1.0_rt, {1.0_rt, 2.0_rt}, {0.0_rt, 1.0_rt}, 1.0_rt},
   {{1.0_rt, 2.0_rt}, {1.0_rt, 1.0_rt}, 1.0_rt, 0.0_rt, {1.0_rt, 1.0_rt}, {1.0_rt, 1.0_rt}, 0.0_rt},
}};
template <>
constexpr auto const& time_integrator_stages<TimeMidpoint>() noexcept
{
   return TimeMidpointStages;
}

inline constexpr std::array<RungeKuttaStage, 3> TimeSSPRK3Stages{{
   {{0.0_rt, 1.0_rt}, {1.0_rt, 1.0_rt}, 0.0_rt, 1.0_rt, {1.0_rt, 1.0_rt}, {1.0_rt, 6.0_rt}, 1.0_rt},
   {{1.0_rt, 1.0_rt}, {1.0_rt, 2.0_rt}, (3.0_rt / 4.0_rt), (1.0_rt / 4.0_rt), {1.0_rt, 4.0_rt}, {1.0_rt, 6.0_rt}, (1.0_rt / 6.0_rt)},
   {{1.0_rt, 2.0_rt}, {1.0_rt, 1.0_rt}, (1.0_rt / 3.0_rt), (2.0_rt / 3.0_rt), {2.0_rt, 3.0_rt}, {2.0_rt, 3.0_rt}, (2.0_rt / 3.0_rt)},
}};
template <>
constexpr auto const& time_integrator_stages<TimeSSPRK3>() noexcept
{
   return TimeSSPRK3Stages;
}

inline constexpr std::array<RungeKuttaStage, 4> TimeRK4Stages{{
   {{0.0_rt, 1.0_rt}, {1.0_rt, 2.0_rt}, 0.0_rt, 1.0_rt, {1.0_rt, 2.0_rt}, {1.0_rt, 6.0_rt}, 1.0_rt},
   {{1.0_rt, 2.0_rt}, {1.0_rt, 2.0_rt}, 1.0_rt, 0.0_rt, {1.0_rt, 2.0_rt}, {1.0_rt, 3.0_rt}, 0.0_rt},
   {{1.0_rt, 2.0_rt}, {1.0_rt, 1.0_rt}, 1.0_rt, 0.0_rt, {1.0_rt, 1.0_rt}, {1.0_rt, 3.0_rt}, 0.0_rt},
   {{1.0_rt, 1.0_rt}, {1.0_rt, 1.0_rt}, 1.0_rt, 0.0_rt, {1.0_rt, 6.0_rt}, {1.0_rt, 6.0_rt}, 0.0_rt},
}};
template <>
constexpr auto const& time_integrator_stages<TimeRK4>() noexcept
{
   return TimeRK4Stages;
}

} // namespace pyfluid_amrex
