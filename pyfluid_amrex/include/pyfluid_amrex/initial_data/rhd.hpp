#pragma once

#include <AMReX_Geometry.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_MultiFab.H>

#include <cmath>

#include "pyfluid_amrex/config.hpp"
#include "pyfluid_amrex/initial_data/sampling.hpp"
#include "pyfluid_amrex/kernels_generated/relativistic_fluid.hpp"
#include "pyfluid_amrex/relativistic_state.hpp"
#include "pyfluid_amrex/state.hpp"

namespace pyfluid_amrex {

struct RelativisticInitialDataParameters {
   amrex::Real rho0;
   amrex::Real pressure0;
   amrex::Real amplitude;
   amrex::Real wave_number;
   amrex::Real v1_0;
   amrex::Real v2_0;
   amrex::Real v3_0;
   amrex::Real b1_0;
   amrex::Real b2_0;
   amrex::Real b3_0;
   amrex::Real shock_x0;
   amrex::Real shock_y0;
   amrex::Real shock_z0;
   amrex::Real shock_nx;
   amrex::Real shock_ny;
   amrex::Real shock_nz;
   amrex::Real rho_left;
   amrex::Real pressure_left;
   amrex::Real v1_left;
   amrex::Real v2_left;
   amrex::Real v3_left;
   amrex::Real b1_left;
   amrex::Real b2_left;
   amrex::Real b3_left;
   amrex::Real rho_right;
   amrex::Real pressure_right;
   amrex::Real v1_right;
   amrex::Real v2_right;
   amrex::Real v3_right;
   amrex::Real b1_right;
   amrex::Real b2_right;
   amrex::Real b3_right;
   amrex::Real gamma;
   amrex::Real rho_floor;
   amrex::Real pressure_floor;
};

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
RHDPrimitive ideal_gas_rhd_primitive(
   amrex::Real rho,
   amrex::Real pressure,
   amrex::Real v1,
   amrex::Real v2,
   amrex::Real v3,
   amrex::Real gamma
) noexcept
{
   return {rho, pressure / ((gamma - 1.0) * rho), pressure, v1, v2, v3};
}

inline void initialize_relativistic_state(
   amrex::MultiFab& state,
   amrex::Geometry const& geom,
   RunConfig const& cfg
)
{
   const auto problo = geom.ProbLoArray();
   const auto probhi = geom.ProbHiArray();
   const auto dx = geom.CellSizeArray();
   const MeshAxes axes = mesh_axes_layout(cfg);
   const FixedSpacetimeParameters spacetime = fixed_spacetime_parameters(cfg);
   validate_fixed_spacetime_domain(spacetime, geom);
   const PhysicalSystem system = physical_system(cfg);
   const bool magnetized = physical_system_has_magnetic_field(system);
   const bool uniform = cfg.ic == "uniform";
   const bool sine_wave = cfg.ic == "sin_wave";
   const bool fixed_metric_hse = cfg.ic == "fixed_metric_hse";
   const bool periodic_plane = cfg.ic == "sod_periodic_plane";
   const amrex::Real two_pi = 2.0 * 3.141592653589793238462643383279502884;
   const RelativisticInitialDataParameters params{
      cfg.rho0,
      cfg.pressure0,
      cfg.amplitude,
      cfg.wave_number,
      cfg.v1_0,
      cfg.v2_0,
      cfg.v3_0,
      cfg.b1_0,
      cfg.b2_0,
      cfg.b3_0,
      cfg.shock_x0,
      cfg.shock_y0,
      cfg.shock_z0,
      cfg.shock_nx,
      cfg.shock_ny,
      cfg.shock_nz,
      cfg.rho_left,
      cfg.pressure_left,
      cfg.v1_left,
      cfg.v2_left,
      cfg.v3_left,
      std::isfinite(cfg.b1_left) ? cfg.b1_left : cfg.b1_0,
      std::isfinite(cfg.b2_left) ? cfg.b2_left : cfg.b2_0,
      std::isfinite(cfg.b3_left) ? cfg.b3_left : cfg.b3_0,
      cfg.rho_right,
      cfg.pressure_right,
      cfg.v1_right,
      cfg.v2_right,
      cfg.v3_right,
      std::isfinite(cfg.b1_right) ? cfg.b1_right : cfg.b1_0,
      std::isfinite(cfg.b2_right) ? cfg.b2_right : cfg.b2_0,
      std::isfinite(cfg.b3_right) ? cfg.b3_right : cfg.b3_0,
      cfg.gamma,
      cfg.rho_floor,
      cfg.pressure_floor,
   };
   const FixedSpacetimeGeometry reference_geometry = fixed_spacetime_geometry(
      spacetime,
      0.5 * (problo[0] + probhi[0]),
      AMREX_SPACEDIM >= 2 ? 0.5 * (problo[1] + probhi[1]) : 0.0,
      AMREX_SPACEDIM == 3 ? 0.5 * (problo[2] + probhi[2]) : 0.0,
      0.0
   );
   const amrex::Real reference_lapse = reference_geometry.metric.lapse;

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
   for (amrex::MFIter mfi(state, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const amrex::Box& box = mfi.tilebox();
      const auto conserved = state.array(mfi);
      amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
         const InitialDataSamplePoint sample = cell_center_initial_data_sample_point(
            MeshCartesian, axes, problo, dx, i, j, k
         );
         const FixedSpacetimeGeometry local_geometry = fixed_spacetime_geometry(
            spacetime,
            sample.physical.q1,
            AMREX_SPACEDIM >= 2 ? sample.physical.q2 : 0.0,
            AMREX_SPACEDIM == 3 ? sample.physical.q3 : 0.0,
            0.0
         );
         amrex::Real rho = params.rho0;
         amrex::Real pressure = params.pressure0;
         amrex::Real v1 = params.v1_0;
         amrex::Real v2 = params.v2_0;
         amrex::Real v3 = params.v3_0;
         amrex::Real b1 = params.b1_0;
         amrex::Real b2 = params.b2_0;
         amrex::Real b3 = params.b3_0;

         if (fixed_metric_hse) {
            const amrex::Real reference_enthalpy = 1.0
               + params.gamma * params.pressure0
               / ((params.gamma - 1.0) * params.rho0);
            const amrex::Real enthalpy = reference_enthalpy * reference_lapse
               / local_geometry.metric.lapse;
            const amrex::Real thermodynamic_ratio = (enthalpy - 1.0)
               / (reference_enthalpy - 1.0);
            rho = params.rho0 * std::pow(
               thermodynamic_ratio, 1.0 / (params.gamma - 1.0)
            );
            pressure = params.pressure0 * std::pow(rho / params.rho0, params.gamma);
            v1 = 0.0;
            v2 = 0.0;
            v3 = 0.0;
         } else if (sine_wave) {
            const amrex::Real length = probhi[0] - problo[0];
            const amrex::Real phase = two_pi * params.wave_number
               * (sample.physical.q1 - problo[0]) / length;
            rho *= 1.0 + params.amplitude * std::sin(phase);
         } else if (!uniform) {
            const amrex::Real y = AMREX_SPACEDIM >= 2
               ? sample.physical.q2
               : params.shock_y0;
            const amrex::Real z = AMREX_SPACEDIM == 3
               ? sample.physical.q3
               : params.shock_z0;
            const amrex::Real signed_distance =
               params.shock_nx * (sample.physical.q1 - params.shock_x0)
               + params.shock_ny * (y - params.shock_y0)
               + params.shock_nz * (z - params.shock_z0);
            const bool left = periodic_plane
               ? std::sin(two_pi * signed_distance) < 0.0
               : signed_distance < 0.0;
            rho = left ? params.rho_left : params.rho_right;
            pressure = left ? params.pressure_left : params.pressure_right;
            v1 = left ? params.v1_left : params.v1_right;
            v2 = left ? params.v2_left : params.v2_right;
            v3 = left ? params.v3_left : params.v3_right;
            b1 = left ? params.b1_left : params.b1_right;
            b2 = left ? params.b2_left : params.b2_right;
            b3 = left ? params.b3_left : params.b3_right;
         }

         rho = amrex::max(rho, params.rho_floor);
         pressure = amrex::max(pressure, params.pressure_floor);
         if (magnetized) {
            const RMHDPrimitive primitive{
               rho,
               pressure / ((params.gamma - 1.0) * rho),
               pressure,
               v1,
               v2,
               v3,
               b1,
               b2,
               b3,
            };
            store_rmhd_conserved(
               conserved,
               i,
               j,
               k,
               generated::rmhd_primitive_to_conserved(
                  primitive, local_geometry.metric
               )
            );
         } else {
            const RHDPrimitive primitive = ideal_gas_rhd_primitive(
               rho, pressure, v1, v2, v3, params.gamma
            );
            store_rhd_conserved(
               conserved,
               i,
               j,
               k,
               generated::rhd_primitive_to_conserved(
                  primitive, local_geometry.metric
               )
            );
         }
      });
   }
}

} // namespace pyfluid_amrex
