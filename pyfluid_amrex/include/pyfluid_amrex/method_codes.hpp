#pragma once

#include "pyfluid_amrex/generated/method_catalog.hpp"

namespace pyfluid_amrex {

enum RiemannMethodCode : int {
   RiemannLLF = 0,
   RiemannHLL = 1,
   RiemannHLLE = 2,
   RiemannHLLC = 3,
   RiemannHLLD = 4
};

constexpr bool is_five_point_reconstruction(int method) noexcept
{
   return reconstruction_stencil_radius(method) == 2;
}

constexpr bool is_characteristic_reconstruction(int method) noexcept
{
   return reconstruction_is_characteristic(method);
}

enum TimeIntegratorCode : int {
   TimeEuler = 0,
   TimeCorrector = 1,
   TimeMidpoint = 2,
   TimeSSPRK3 = 3,
   TimeRK4 = 4
};

enum DiffusionMethodCode : int {
   DiffusionNone = 0,
   DiffusionExplicit = 1,
   DiffusionImplicit = 2
};

enum EOSMethodCode : int {
   EOSIdealGas = 0,
   EOSPolytrope = 1,
   EOSHybridPolytrope = 2,
   EOSCompose = 3,
   EOSRuntime = -1
};

} // namespace pyfluid_amrex
