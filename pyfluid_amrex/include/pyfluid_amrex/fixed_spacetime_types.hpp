#pragma once

#include <AMReX_REAL.H>

namespace pyfluid_amrex {

enum class FixedSpacetimeMetric : int {
   Minkowski = 0,
   Rindler = 1,
   SchwarzschildIsotropic = 2,
   KerrSchild = 3,
};

enum class BlackHoleInterior : int {
   Exterior = 0,
   Filled = 1,
   Excision = 2,
};

struct FixedSpacetimeParameters {
   FixedSpacetimeMetric metric = FixedSpacetimeMetric::Minkowski;
   amrex::Real acceleration1 = 0.0;
   amrex::Real acceleration2 = 0.0;
   amrex::Real acceleration3 = 0.0;
   amrex::Real origin1 = 0.0;
   amrex::Real origin2 = 0.0;
   amrex::Real origin3 = 0.0;
   amrex::Real mass = 1.0;
   amrex::Real dimensionless_spin1 = 0.0;
   amrex::Real dimensionless_spin2 = 0.0;
   amrex::Real dimensionless_spin3 = 0.0;
   BlackHoleInterior black_hole_interior = BlackHoleInterior::Exterior;
   amrex::Real black_hole_fill_radius = -1.0;
};

} // namespace pyfluid_amrex
