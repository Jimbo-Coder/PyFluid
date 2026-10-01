#pragma once

#include <AMReX_GpuContainers.H>
#include <AMReX_REAL.H>

#include <utility>
#include <vector>

namespace pyfluid_amrex {

using namespace amrex::literals;

struct SurfaceQuadraturePoint {
   amrex::GpuArray<amrex::Real, 3> position{};
   amrex::GpuArray<amrex::Real, 3> radial_unit{};
   amrex::GpuArray<amrex::Real, 3> area_vector{};
   amrex::Real theta = 0.0_rt;
   amrex::Real phi = 0.0_rt;
   amrex::Real solid_angle_weight = 0.0_rt;
};

class ParametricSurface {
public:
   static ParametricSurface sphere(
      amrex::GpuArray<amrex::Real, 3> center,
      amrex::Real radius,
      int polar_points
   );

   // Construct a star-shaped S2 surface from cell-centered samples
   // r(theta_i, phi_j). The ordering matches ApparentHorizonSurface.
   static ParametricSurface radial_graph(
      amrex::GpuArray<amrex::Real, 3> center,
      int num_theta,
      int num_phi,
      std::vector<amrex::Real> radius
   );

   std::vector<SurfaceQuadraturePoint> const& points() const noexcept
   {
      return points_;
   }

   amrex::GpuArray<amrex::Real, 3> const& center() const noexcept
   {
      return center_;
   }

   amrex::Real radius() const noexcept
   {
      return radius_;
   }

   // Each integrand contributes three contiguous Cartesian flux components
   // per quadrature point. Contracting with dS_i returns one surface charge.
   std::vector<amrex::Real> integrate_normal_fluxes(
      std::vector<amrex::Real> const& sampled_fluxes,
      int num_integrands
   ) const;

private:
   ParametricSurface(
      amrex::GpuArray<amrex::Real, 3> center,
      amrex::Real radius,
      std::vector<SurfaceQuadraturePoint> points
   )
      : center_(center), radius_(radius), points_(std::move(points))
   {
   }

   amrex::GpuArray<amrex::Real, 3> center_{};
   amrex::Real radius_ = 0.0_rt;
   std::vector<SurfaceQuadraturePoint> points_;
};

} // namespace pyfluid_amrex
