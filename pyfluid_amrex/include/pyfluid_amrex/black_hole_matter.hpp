#pragma once

#include <AMReX_REAL.H>

namespace pyfluid_amrex {

using namespace amrex::literals;

// Runtime policy for matter strictly inside a tracked black-hole surface.
// This is deliberately orthogonal to the initial-data interior completion and
// to fixed versus evolved spacetime, and therefore is not a template axis.
enum class BlackHoleMatterTreatment : int {
  Evolve = 0,
  Bounded = 1,
  Excision = 2,
};

struct BlackHoleMatterParameters {
  BlackHoleMatterTreatment treatment = BlackHoleMatterTreatment::Evolve;
  amrex::Real maximum_lorentz_factor = 10.0_rt;
  amrex::Real maximum_specific_internal_energy = 100.0_rt;
  amrex::Real maximum_magnetization = 100.0_rt;
  amrex::Real maximum_density = -1.0_rt;
};

} // namespace pyfluid_amrex
