#include "pyfluid_amrex/apparent_horizon_finder.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

#include "relativity/apparent_horizon_angular.hpp"

namespace pyfluid_amrex {

ApparentHorizonSurface::ApparentHorizonSurface(
   std::array<amrex::Real, 3> center,
   int num_theta,
   int num_phi,
   std::vector<amrex::Real> radius
)
   : center_(center),
     num_theta_(num_theta),
     num_phi_(num_phi),
     radius_(std::move(radius))
{
   const bool finite_positive_radius = std::all_of(
      radius_.begin(), radius_.end(), [](amrex::Real value) {
         return std::isfinite(value) && value > 0.0;
      }
   );
   if (num_theta_ < 2 || num_phi_ < 2 || num_phi_ % 2 != 0
       || radius_.size() != static_cast<std::size_t>(num_theta_ * num_phi_)
       || !finite_positive_radius) {
      throw std::invalid_argument("invalid apparent-horizon radial surface");
   }
}

amrex::Real ApparentHorizonSurface::radius_at(
   amrex::Real theta,
   amrex::Real phi
) const
{
   if (radius_.empty()) {
      throw std::logic_error("apparent-horizon surface is empty");
   }
   return apparent_horizon_detail::interpolate_surface(
      radius_, num_theta_, num_phi_, theta, phi
   );
}

amrex::Real ApparentHorizonSurface::signed_radial_distance(
   amrex::Real x,
   amrex::Real y,
   amrex::Real z
) const
{
   const amrex::Real relative_x = x - center_[0];
   const amrex::Real relative_y = y - center_[1];
   const amrex::Real relative_z = z - center_[2];
   const amrex::Real coordinate_radius = std::sqrt(
      relative_x * relative_x + relative_y * relative_y
      + relative_z * relative_z
   );
   if (coordinate_radius == 0.0) {
      return -radius_at(0.0, 0.0);
   }
   const amrex::Real theta = std::acos(std::clamp(
      relative_z / coordinate_radius, amrex::Real(-1.0), amrex::Real(1.0)
   ));
   const amrex::Real phi = std::atan2(relative_y, relative_x);
   return coordinate_radius - radius_at(theta, phi);
}

} // namespace pyfluid_amrex
