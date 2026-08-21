#pragma once

#include <AMReX_Array.H>
#include <AMReX_REAL.H>

#include <array>
#include <cstddef>
#include <optional>
#include <string>
#include <vector>

#include "pyfluid_amrex/relativistic_state.hpp"

namespace pyfluid_amrex {

// gamma_ij, gamma^ij, d_k gamma_ij, and K_ij.  Lapse and shift do not enter
// the MOTS expansion, so they are deliberately absent from this interpolation
// contract.
constexpr int ApparentHorizonGeometryComponents = 36;

enum ApparentHorizonGeometryComponent : int {
   AhGamma11 = 0,
   AhGamma12,
   AhGamma13,
   AhGamma22,
   AhGamma23,
   AhGamma33,
   AhInverseGamma11,
   AhInverseGamma12,
   AhInverseGamma13,
   AhInverseGamma22,
   AhInverseGamma23,
   AhInverseGamma33,
   AhD1Gamma11,
   AhD1Gamma12,
   AhD1Gamma13,
   AhD1Gamma22,
   AhD1Gamma23,
   AhD1Gamma33,
   AhD2Gamma11,
   AhD2Gamma12,
   AhD2Gamma13,
   AhD2Gamma22,
   AhD2Gamma23,
   AhD2Gamma33,
   AhD3Gamma11,
   AhD3Gamma12,
   AhD3Gamma13,
   AhD3Gamma22,
   AhD3Gamma23,
   AhD3Gamma33,
   AhExtrinsic11,
   AhExtrinsic12,
   AhExtrinsic13,
   AhExtrinsic22,
   AhExtrinsic23,
   AhExtrinsic33,
};

using ApparentHorizonGeometrySample =
   std::array<amrex::Real, ApparentHorizonGeometryComponents>;

ApparentHorizonGeometrySample pack_apparent_horizon_geometry(
   SpacetimeMetric const& metric,
   SpacetimeMetricDerivatives const& derivatives
);

void unpack_apparent_horizon_geometry(
   ApparentHorizonGeometrySample const& sample,
   SpacetimeMetric& metric,
   SpacetimeMetricDerivatives& derivatives
);

// A host-owned spherical interpolation volume.  Geometry is sampled on a
// cell-centered angular grid and arbitrary strictly increasing radial spokes.
// The AMR adapter constructs this once per horizon search; pseudo-time stages
// then use only inexpensive one-dimensional/bilinear interpolation here.
class ApparentHorizonGeometryGrid {
public:
   ApparentHorizonGeometryGrid(
      std::array<amrex::Real, 3> center,
      std::vector<amrex::Real> radial_coordinates,
      int num_theta,
      int num_phi,
      std::vector<ApparentHorizonGeometrySample> samples
   );

   ApparentHorizonGeometrySample interpolate(
      amrex::Real radius,
      amrex::Real theta,
      amrex::Real phi
   ) const;

   std::array<amrex::Real, 3> const& center() const noexcept
   {
      return center_;
   }

   std::vector<amrex::Real> const& radial_coordinates() const noexcept
   {
      return radial_coordinates_;
   }

   int num_theta() const noexcept
   {
      return num_theta_;
   }

   int num_phi() const noexcept
   {
      return num_phi_;
   }

private:
   std::size_t index(int radial, int theta, int phi) const noexcept;
   ApparentHorizonGeometrySample interpolate_angular(
      int radial,
      amrex::Real theta,
      amrex::Real phi
   ) const;

   std::array<amrex::Real, 3> center_{};
   std::vector<amrex::Real> radial_coordinates_;
   int num_theta_ = 0;
   int num_phi_ = 0;
   std::vector<ApparentHorizonGeometrySample> samples_;
};

class ApparentHorizonSurface {
public:
   ApparentHorizonSurface() = default;
   ApparentHorizonSurface(
      std::array<amrex::Real, 3> center,
      int num_theta,
      int num_phi,
      std::vector<amrex::Real> radius
   );

   amrex::Real radius_at(amrex::Real theta, amrex::Real phi) const;
   amrex::Real signed_radial_distance(
      amrex::Real x,
      amrex::Real y,
      amrex::Real z
   ) const;

   std::array<amrex::Real, 3> const& center() const noexcept
   {
      return center_;
   }

   int num_theta() const noexcept
   {
      return num_theta_;
   }

   int num_phi() const noexcept
   {
      return num_phi_;
   }

   std::vector<amrex::Real> const& radius() const noexcept
   {
      return radius_;
   }

private:
   std::array<amrex::Real, 3> center_{};
   int num_theta_ = 0;
   int num_phi_ = 0;
   std::vector<amrex::Real> radius_;
};

struct ApparentHorizonFinderParameters {
   amrex::Real mass_scale = 1.0;
   amrex::Real cfl = 0.98;
   amrex::Real eta_times_mass = 7.0;
   amrex::Real theta_linf_times_mass_tolerance = 1.0e-5;
   amrex::Real theta_l2_times_mass_tolerance = 1.0e-2;
   int max_iterations_per_level = 10000;
   int coarsest_num_theta = 8;
   int target_num_theta = 32;
   bool over_relaxation = true;
   int over_relaxation_interval = 64;
   amrex::Real over_relaxation_growth = 1.2;
   amrex::Real over_relaxation_minimum_improvement = 0.01;
};

struct ApparentHorizonResult {
   bool converged = false;
   std::string failure_reason;
   ApparentHorizonSurface surface;
   int iterations = 0;
   amrex::Real pseudo_time = 0.0;
   amrex::Real theta_linf_times_mass = 0.0;
   amrex::Real theta_l2_times_mass = 0.0;
   amrex::Real area = 0.0;
   amrex::Real irreducible_mass = 0.0;
   std::array<amrex::Real, 3> centroid{};
   amrex::Real minimum_radius = 0.0;
   amrex::Real maximum_radius = 0.0;
   amrex::Real mean_radius = 0.0;
};

// BHaHAHA-style nonlinear hyperbolic relaxation.  The implementation uses
// SSPRK3, scale-aware stopping criteria, pole-parity angular derivatives,
// coarse-to-fine continuation, and optional residual-minimizing
// over-relaxation.  It is an independent implementation of the published
// method and consumes only the shared ADM 3+1 geometry contract.
ApparentHorizonResult find_apparent_horizon(
   ApparentHorizonGeometryGrid const& geometry,
   amrex::Real initial_radius,
   ApparentHorizonFinderParameters const& parameters,
   std::optional<ApparentHorizonSurface> const& initial_surface = std::nullopt
);

} // namespace pyfluid_amrex
