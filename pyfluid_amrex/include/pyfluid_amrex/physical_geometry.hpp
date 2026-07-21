#pragma once

#include <AMReX_GpuQualifiers.H>
#include <AMReX_REAL.H>

#include <cmath>

namespace pyfluid_amrex {

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real cartesian_radius_squared(
   amrex::Real x,
   amrex::Real y,
   amrex::Real z
) noexcept
{
   return x * x + y * y + z * z;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real cartesian_radius(
   amrex::Real x,
   amrex::Real y,
   amrex::Real z
) noexcept
{
   return std::sqrt(cartesian_radius_squared(x, y, z));
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real cartesian_cylindrical_radius_squared(
   amrex::Real x,
   amrex::Real y
) noexcept
{
   return x * x + y * y;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real cartesian_cylindrical_radius(
   amrex::Real x,
   amrex::Real y
) noexcept
{
   return std::sqrt(cartesian_cylindrical_radius_squared(x, y));
}

} // namespace pyfluid_amrex
