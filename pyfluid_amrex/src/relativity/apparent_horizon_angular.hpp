#pragma once

#include <AMReX_REAL.H>

#include <cmath>
#include <cstddef>
#include <numbers>
#include <vector>

namespace pyfluid_amrex::apparent_horizon_detail {

constexpr amrex::Real pi = std::numbers::pi_v<amrex::Real>;

inline int wrap_index(int index, int count) noexcept
{
   index %= count;
   return index < 0 ? index + count : index;
}

struct AngularIndex {
   int theta = 0;
   int phi_shift = 0;
};

inline AngularIndex map_theta_index(
   int index,
   int num_theta,
   int num_phi
) noexcept
{
   int shift = 0;
   while (index < 0 || index >= num_theta) {
      if (index < 0) {
         index = -index - 1;
         shift += num_phi / 2;
      }
      if (index >= num_theta) {
         index = 2 * num_theta - index - 1;
         shift += num_phi / 2;
      }
   }
   return {index, shift};
}

inline void canonicalize_angles(
   amrex::Real& theta,
   amrex::Real& phi
) noexcept
{
   while (theta < 0.0 || theta > pi) {
      if (theta < 0.0) {
         theta = -theta;
         phi += pi;
      }
      if (theta > pi) {
         theta = 2.0 * pi - theta;
         phi += pi;
      }
   }
   phi = std::fmod(phi, 2.0 * pi);
   if (phi < 0.0) {
      phi += 2.0 * pi;
   }
}

inline std::size_t surface_index(int theta, int phi, int num_phi) noexcept
{
   return static_cast<std::size_t>(theta) * num_phi + phi;
}

inline amrex::Real surface_value(
   std::vector<amrex::Real> const& values,
   int theta,
   int phi,
   int num_theta,
   int num_phi
) noexcept
{
   const AngularIndex mapped = map_theta_index(theta, num_theta, num_phi);
   return values[surface_index(
      mapped.theta, wrap_index(phi + mapped.phi_shift, num_phi), num_phi
   )];
}

inline amrex::Real interpolate_surface(
   std::vector<amrex::Real> const& values,
   int num_theta,
   int num_phi,
   amrex::Real theta,
   amrex::Real phi
)
{
   canonicalize_angles(theta, phi);
   const amrex::Real dtheta = pi / num_theta;
   const amrex::Real dphi = 2.0 * pi / num_phi;
   const amrex::Real theta_index = theta / dtheta - 0.5;
   const amrex::Real phi_index = phi / dphi - 0.5;
   const int theta_lower = static_cast<int>(std::floor(theta_index));
   const int phi_lower = static_cast<int>(std::floor(phi_index));
   const amrex::Real theta_fraction = theta_index - theta_lower;
   const amrex::Real phi_fraction = phi_index - phi_lower;
   amrex::Real result = 0.0;
   for (int theta_offset = 0; theta_offset < 2; ++theta_offset) {
      const amrex::Real theta_weight = theta_offset == 0
         ? 1.0 - theta_fraction : theta_fraction;
      const AngularIndex mapped = map_theta_index(
         theta_lower + theta_offset, num_theta, num_phi
      );
      for (int phi_offset = 0; phi_offset < 2; ++phi_offset) {
         const amrex::Real phi_weight = phi_offset == 0
            ? 1.0 - phi_fraction : phi_fraction;
         const int mapped_phi = wrap_index(
            phi_lower + phi_offset + mapped.phi_shift, num_phi
         );
         result += theta_weight * phi_weight
            * values[surface_index(mapped.theta, mapped_phi, num_phi)];
      }
   }
   return result;
}

} // namespace pyfluid_amrex::apparent_horizon_detail
