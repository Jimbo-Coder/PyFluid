#pragma once

#include <AMReX_REAL.H>

#include <complex>
#include <iosfwd>
#include <optional>
#include <string>
#include <vector>

#include "pyfluid_amrex/surface_geometry.hpp"

namespace pyfluid_amrex {

struct MultipoleMode {
   std::string field{};
   std::string basis{};
   int spin_weight = 0;
   std::optional<int> ell{};
   int m = 0;
   std::complex<amrex::Real> value{};
};

struct MultipoleDiagnostics {
   amrex::Real radius = 0.0;
   std::vector<MultipoleMode> modes{};
};

std::complex<amrex::Real> spin_weighted_spherical_harmonic(
   int spin_weight,
   int ell,
   int m,
   amrex::Real theta,
   amrex::Real phi
);

std::vector<MultipoleMode> decompose_spin_weighted_field(
   ParametricSurface const& surface,
   std::vector<amrex::Real> const& real_values,
   std::vector<amrex::Real> const& imaginary_values,
   std::string field,
   int spin_weight,
   int ell_min,
   int ell_max
);

std::vector<MultipoleMode> decompose_azimuthal_field(
   ParametricSurface const& surface,
   std::vector<amrex::Real> const& values,
   std::string field,
   int m_max
);

amrex::Real surface_complex_l2(
   ParametricSurface const& surface,
   std::vector<amrex::Real> const& real_values,
   std::vector<amrex::Real> const& imaginary_values
);

void write_multipole_header(
   std::ofstream& output,
   int step,
   amrex::Real time
);

void append_multipole_rows(
   std::ofstream& output,
   int step,
   amrex::Real time,
   amrex::Real radius,
   std::vector<MultipoleMode> const& modes
);

} // namespace pyfluid_amrex
