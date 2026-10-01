#pragma once

#include "pyfluid_amrex/sampled_sphere.hpp"

namespace pyfluid_amrex::apparent_horizon_detail {

using sampled_sphere_detail::AngularIndex;
using sampled_sphere_detail::canonicalize_angles;
using sampled_sphere_detail::interpolate_surface;
using sampled_sphere_detail::map_theta_index;
using sampled_sphere_detail::pi;
using sampled_sphere_detail::surface_index;
using sampled_sphere_detail::surface_value;
using sampled_sphere_detail::wrap_index;

} // namespace pyfluid_amrex::apparent_horizon_detail
