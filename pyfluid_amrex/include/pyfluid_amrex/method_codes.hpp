#pragma once

namespace pyfluid_amrex {

enum RiemannMethodCode : int {
   RiemannLLF = 0,
   RiemannHLL = 1,
   RiemannHLLE = 2,
   RiemannHLLC = 3,
   RiemannHLLD = 4
};

enum ReconstructionMethodCode : int {
   ReconPWC = 0,
   ReconPWL = 1,
   ReconWENOZ = 2,
   ReconWENOZChar = 3,
   ReconMP5 = 4,
   ReconMP5Char = 5,
   ReconTENO5 = 6,
   ReconTENO5Char = 7
};

constexpr bool is_five_point_reconstruction(int method) noexcept
{
   return method == ReconWENOZ || method == ReconWENOZChar
      || method == ReconMP5 || method == ReconMP5Char
      || method == ReconTENO5 || method == ReconTENO5Char;
}

constexpr bool is_characteristic_reconstruction(int method) noexcept
{
   return method == ReconWENOZChar || method == ReconMP5Char || method == ReconTENO5Char;
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
   EOSHybridPolytrope = 2
};

} // namespace pyfluid_amrex
