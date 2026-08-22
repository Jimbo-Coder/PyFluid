#pragma once

#include <AMReX_GpuContainers.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_REAL.H>

#include <cmath>

namespace pyfluid_amrex {

struct AzimuthalFrame {
   amrex::GpuArray<amrex::Real, 3> origin{};
   amrex::GpuArray<amrex::Real, 3> axis{{0.0, 0.0, 1.0}};
   amrex::GpuArray<amrex::Real, 3> radial_basis{{1.0, 0.0, 0.0}};
   amrex::GpuArray<amrex::Real, 3> azimuthal_basis{{0.0, 1.0, 0.0}};
};

struct AzimuthalCoordinate {
   amrex::Real cylindrical_radius = 0.0;
   amrex::Real phi = 0.0;
   amrex::Real axial_coordinate = 0.0;
};

inline AzimuthalFrame make_azimuthal_frame(
   amrex::GpuArray<amrex::Real, 3> const& origin,
   amrex::GpuArray<amrex::Real, 3> axis
)
{
   const amrex::Real axis_norm = std::sqrt(
      axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]
   );
   for (amrex::Real& component : axis) {
      component /= axis_norm;
   }
   const amrex::GpuArray<amrex::Real, 3> seed =
      std::abs(axis[2]) < 0.9
      ? amrex::GpuArray<amrex::Real, 3>{{0.0, 0.0, 1.0}}
      : amrex::GpuArray<amrex::Real, 3>{{1.0, 0.0, 0.0}};
   const amrex::Real seed_axis =
      seed[0] * axis[0] + seed[1] * axis[1] + seed[2] * axis[2];
   amrex::GpuArray<amrex::Real, 3> radial_basis{{
      seed[0] - seed_axis * axis[0],
      seed[1] - seed_axis * axis[1],
      seed[2] - seed_axis * axis[2],
   }};
   const amrex::Real radial_norm = std::sqrt(
      radial_basis[0] * radial_basis[0]
      + radial_basis[1] * radial_basis[1]
      + radial_basis[2] * radial_basis[2]
   );
   for (amrex::Real& component : radial_basis) {
      component /= radial_norm;
   }
   return {
      origin,
      axis,
      radial_basis,
      {{
         axis[1] * radial_basis[2] - axis[2] * radial_basis[1],
         axis[2] * radial_basis[0] - axis[0] * radial_basis[2],
         axis[0] * radial_basis[1] - axis[1] * radial_basis[0],
      }},
   };
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
AzimuthalCoordinate azimuthal_coordinate(
   AzimuthalFrame const& frame,
   amrex::GpuArray<amrex::Real, 3> const& physical
) noexcept
{
   const amrex::Real relative1 = physical[0] - frame.origin[0];
   const amrex::Real relative2 = physical[1] - frame.origin[1];
   const amrex::Real relative3 = physical[2] - frame.origin[2];
   const amrex::Real projected1 =
      relative1 * frame.radial_basis[0]
      + relative2 * frame.radial_basis[1]
      + relative3 * frame.radial_basis[2];
   const amrex::Real projected2 =
      relative1 * frame.azimuthal_basis[0]
      + relative2 * frame.azimuthal_basis[1]
      + relative3 * frame.azimuthal_basis[2];
   return {
      std::sqrt(projected1 * projected1 + projected2 * projected2),
      std::atan2(projected2, projected1),
      relative1 * frame.axis[0] + relative2 * frame.axis[1]
         + relative3 * frame.axis[2],
   };
}

} // namespace pyfluid_amrex
