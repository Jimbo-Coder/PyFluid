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
   ReconWENOZChar = 3
};

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
