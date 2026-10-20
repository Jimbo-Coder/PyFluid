#pragma once

#include <AMReX_Geometry.H>

#include <limits>

#include "pyfluid_amrex/config.hpp"
#include "pyfluid_amrex/method_codes.hpp"
#include "pyfluid_amrex/metric_provider.hpp"

namespace pyfluid_amrex {

using namespace amrex::literals;

// Timestep policy does not depend on the viscous stress/RHS implementation.
inline amrex::Real diffusion_inverse_mesh_spacing2_sum(amrex::Geometry const& geom, RunConfig const& cfg)
{
   const int coordinate = mesh_coordinate_code(cfg.mesh_coordinate);
   const MeshAxes axes = mesh_axes_layout(cfg);
   const auto dx = geom.CellSizeArray();
   const auto problo = geom.ProbLoArray();
   amrex::Real value = 0.0_rt;
   const amrex::Real spacing0 = mesh_face_gradient_spacing(coordinate, axes, problo, dx, 0, 0, 0, 0);
   value += 1.0_rt / (spacing0 * spacing0);
#if (AMREX_SPACEDIM >= 2)
   const amrex::Real spacing1 = mesh_face_gradient_spacing(coordinate, axes, problo, dx, 1, 0, 0, 0);
   value += 1.0_rt / (spacing1 * spacing1);
#endif
#if (AMREX_SPACEDIM == 3)
   const amrex::Real spacing2 = mesh_face_gradient_spacing(coordinate, axes, problo, dx, 2, 0, 0, 0);
   value += 1.0_rt / (spacing2 * spacing2);
#endif
   return value;
}

inline amrex::Real max_diffusion_coefficient(RunConfig const& cfg)
{
   amrex::Real coefficient = cfg.viscosity;
   if (resistivity_enabled(cfg)) {
      coefficient = amrex::max(coefficient, cfg.resistivity);
   }
   return coefficient;
}

inline amrex::Real diffusion_dt_limit(
   amrex::Geometry const& geom, RunConfig const& cfg, int diffusion_method
)
{
   if (diffusion_method != DiffusionExplicit) {
      return std::numeric_limits<amrex::Real>::max();
   }
   const amrex::Real coefficient = max_diffusion_coefficient(cfg);
   if (coefficient <= 0.0_rt) {
      return std::numeric_limits<amrex::Real>::max();
   }
   return cfg.diffusion_cfl / (2.0_rt * coefficient * diffusion_inverse_mesh_spacing2_sum(geom, cfg));
}

} // namespace pyfluid_amrex
