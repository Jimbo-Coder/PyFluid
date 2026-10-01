// Generated from schema/time_integrators.toml. Do not edit.
#pragma once
namespace pyfluid_amrex {
enum TimeIntegratorCode : int {
   TimeEuler = 0,
   TimeCorrector = 1,
   TimeMidpoint = 2,
   TimeSSPRK3 = 3,
   TimeRK4 = 4,
};
} // namespace pyfluid_amrex
