#include "amr/amr_solver_base.hpp"

#include <AMReX_GpuLaunch.H>
#include <AMReX_ParallelDescriptor.H>

#include <algorithm>
#include <cmath>
#include <numbers>
#include <vector>

#include "pyfluid_amrex/apparent_horizon_finder.hpp"
#include "pyfluid_amrex/apparent_horizon_tracker.hpp"
#include "pyfluid_amrex/hierarchy_interpolation.hpp"
#include "pyfluid_amrex/kernels_generated/z4c.hpp"
#include "pyfluid_amrex/z4c_derivatives.hpp"
#include "pyfluid_amrex/z4c_state_access.hpp"

namespace pyfluid_amrex {

void AmrSolverBase::update_apparent_horizon(int step, amrex::Real time)
{
   apparent_horizon_tracker_->restore_latest_surface(step, cfg_.output_dir);
   if (!apparent_horizon_tracker_->search_due(step)) {
      return;
   }
#if (AMREX_SPACEDIM != 3)
   amrex::Abort("apparent-horizon finding requires the 3D executable");
#else
   if (!spacetime_state_.active()) {
      amrex::Abort("apparent-horizon finding requires stored spacetime state");
   }
   fill_all_spacetime_boundaries(spacetime_state_.state(), time);

   LevelData geometry_hierarchy(finestLevel() + 1);
   for (int lev = 0; lev <= finestLevel(); ++lev) {
      amrex::MultiFab const& state = *spacetime_state_.state()[lev];
      auto geometry = std::make_unique<amrex::MultiFab>(
         state.boxArray(),
         state.DistributionMap(),
         ApparentHorizonGeometryComponents,
         1
      );
      geometry->setVal(0.0);
      const amrex::Box nodal_domain = amrex::convert(
         Geom(lev).Domain(), amrex::IntVect::TheNodeVector()
      );
      const auto spacing = Geom(lev).CellSizeArray();
      const amrex::GpuArray<amrex::Real, 3> inverse_spacing{{
         1.0 / spacing[0], 1.0 / spacing[1], 1.0 / spacing[2]
      }};
      const amrex::GpuArray<int, 3> domain_lower{{
         nodal_domain.smallEnd(0),
         nodal_domain.smallEnd(1),
         nodal_domain.smallEnd(2),
      }};
      const amrex::GpuArray<int, 3> domain_upper{{
         nodal_domain.bigEnd(0),
         nodal_domain.bigEnd(1),
         nodal_domain.bigEnd(2),
      }};
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
      for (amrex::MFIter mfi(state, amrex::TilingIfNotGPU());
           mfi.isValid(); ++mfi) {
         const amrex::Box box = mfi.tilebox();
         const auto input = state.const_array(mfi);
         const auto output = geometry->array(mfi);
         amrex::ParallelFor(
            box,
            [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
               const Z4cSpatialDerivatives z4c_derivatives =
                  gather_z4c_geometry_derivatives<4, 3>(
                     input,
                     i,
                     j,
                     k,
                     inverse_spacing,
                     domain_lower,
                     domain_upper
                  );
               const Z4cState z4c_state = load_z4c_state(input, i, j, k);
               const SpacetimeMetric metric =
                  generated::z4c_spacetime_metric(z4c_state);
               const SpacetimeMetricDerivatives derivatives =
                  generated::z4c_spacetime_metric_derivatives(
                     z4c_state, z4c_derivatives
                  );
               output(i, j, k, AhGamma11) = metric.gamma11;
               output(i, j, k, AhGamma12) = metric.gamma12;
               output(i, j, k, AhGamma13) = metric.gamma13;
               output(i, j, k, AhGamma22) = metric.gamma22;
               output(i, j, k, AhGamma23) = metric.gamma23;
               output(i, j, k, AhGamma33) = metric.gamma33;
               output(i, j, k, AhInverseGamma11) = metric.inv_gamma11;
               output(i, j, k, AhInverseGamma12) = metric.inv_gamma12;
               output(i, j, k, AhInverseGamma13) = metric.inv_gamma13;
               output(i, j, k, AhInverseGamma22) = metric.inv_gamma22;
               output(i, j, k, AhInverseGamma23) = metric.inv_gamma23;
               output(i, j, k, AhInverseGamma33) = metric.inv_gamma33;
               output(i, j, k, AhD1Gamma11) = derivatives.d1_gamma11;
               output(i, j, k, AhD1Gamma12) = derivatives.d1_gamma12;
               output(i, j, k, AhD1Gamma13) = derivatives.d1_gamma13;
               output(i, j, k, AhD1Gamma22) = derivatives.d1_gamma22;
               output(i, j, k, AhD1Gamma23) = derivatives.d1_gamma23;
               output(i, j, k, AhD1Gamma33) = derivatives.d1_gamma33;
               output(i, j, k, AhD2Gamma11) = derivatives.d2_gamma11;
               output(i, j, k, AhD2Gamma12) = derivatives.d2_gamma12;
               output(i, j, k, AhD2Gamma13) = derivatives.d2_gamma13;
               output(i, j, k, AhD2Gamma22) = derivatives.d2_gamma22;
               output(i, j, k, AhD2Gamma23) = derivatives.d2_gamma23;
               output(i, j, k, AhD2Gamma33) = derivatives.d2_gamma33;
               output(i, j, k, AhD3Gamma11) = derivatives.d3_gamma11;
               output(i, j, k, AhD3Gamma12) = derivatives.d3_gamma12;
               output(i, j, k, AhD3Gamma13) = derivatives.d3_gamma13;
               output(i, j, k, AhD3Gamma22) = derivatives.d3_gamma22;
               output(i, j, k, AhD3Gamma23) = derivatives.d3_gamma23;
               output(i, j, k, AhD3Gamma33) = derivatives.d3_gamma33;
               output(i, j, k, AhExtrinsic11) = derivatives.extrinsic11;
               output(i, j, k, AhExtrinsic12) = derivatives.extrinsic12;
               output(i, j, k, AhExtrinsic13) = derivatives.extrinsic13;
               output(i, j, k, AhExtrinsic22) = derivatives.extrinsic22;
               output(i, j, k, AhExtrinsic23) = derivatives.extrinsic23;
               output(i, j, k, AhExtrinsic33) = derivatives.extrinsic33;
            }
         );
      }
      geometry_hierarchy[lev] = std::move(geometry);
   }
   amrex::Vector<amrex::MultiFab*> geometry_fields;
   geometry_fields.reserve(geometry_hierarchy.size());
   for (auto const& geometry : geometry_hierarchy) {
      geometry_fields.push_back(geometry.get());
   }
   amrex::FillBoundary(geometry_fields, Geom(0).periodicity());

   const int radial_points = apparent_horizon_tracker_->radial_points();
   const int polar_points = apparent_horizon_tracker_->polar_points();
   const int azimuthal_points = 2 * polar_points;
   const auto center = apparent_horizon_tracker_->center();
   const amrex::Real minimum_radius =
      apparent_horizon_tracker_->minimum_search_radius();
   const amrex::Real maximum_radius =
      apparent_horizon_tracker_->maximum_search_radius();
   std::vector<amrex::Real> radial_coordinates(radial_points);
   std::vector<amrex::GpuArray<amrex::Real, 3>> positions;
   positions.reserve(
      static_cast<std::size_t>(radial_points)
      * polar_points * azimuthal_points
   );
   const amrex::Real dtheta = std::numbers::pi_v<amrex::Real>
      / polar_points;
   const amrex::Real dphi = 2.0 * std::numbers::pi_v<amrex::Real>
      / azimuthal_points;
   for (int radial_index = 0; radial_index < radial_points; ++radial_index) {
      const amrex::Real radius = minimum_radius
         + (maximum_radius - minimum_radius) * radial_index
            / (radial_points - 1);
      radial_coordinates[radial_index] = radius;
      for (int theta_index = 0; theta_index < polar_points; ++theta_index) {
         const amrex::Real theta = (theta_index + 0.5) * dtheta;
         for (int phi_index = 0;
              phi_index < azimuthal_points;
              ++phi_index) {
            const amrex::Real phi = (phi_index + 0.5) * dphi;
            positions.push_back({{
               center[0] + radius * std::sin(theta) * std::cos(phi),
               center[1] + radius * std::sin(theta) * std::sin(phi),
               center[2] + radius * std::cos(theta),
            }});
         }
      }
   }
   std::vector<amrex::Geometry> hierarchy_geometry;
   std::vector<amrex::BoxArray> hierarchy_grids;
   std::vector<amrex::DistributionMapping> hierarchy_distribution;
   for (int lev = 0; lev <= finestLevel(); ++lev) {
      hierarchy_geometry.push_back(Geom(lev));
      hierarchy_grids.push_back(boxArray(lev));
      hierarchy_distribution.push_back(DistributionMap(lev));
   }
   const HierarchyInterpolationPlan interpolation_plan(
      std::move(positions),
      mesh_coordinate_kind(),
      active_mesh_axes(),
      hierarchy_geometry,
      hierarchy_grids,
      hierarchy_distribution,
      amrex::IndexType::TheNodeType()
   );
   const std::vector<amrex::Real> values = interpolation_plan.interpolate(
      geometry_hierarchy, 0, ApparentHorizonGeometryComponents
   );
   ApparentHorizonResult result;
   std::vector<amrex::Real> surface_radius(
      static_cast<std::size_t>(polar_points) * azimuthal_points
   );
   if (amrex::ParallelDescriptor::IOProcessor()) {
      std::vector<ApparentHorizonGeometrySample> samples(
         values.size() / ApparentHorizonGeometryComponents
      );
      for (std::size_t point = 0; point < samples.size(); ++point) {
         std::copy_n(
            values.begin() + point * ApparentHorizonGeometryComponents,
            ApparentHorizonGeometryComponents,
            samples[point].begin()
         );
      }
      const ApparentHorizonGeometryGrid finder_geometry(
         center,
         std::move(radial_coordinates),
         polar_points,
         azimuthal_points,
         std::move(samples)
      );
      result = find_apparent_horizon(
         finder_geometry,
         apparent_horizon_tracker_->initial_radius(),
         apparent_horizon_tracker_->finder_parameters(),
         apparent_horizon_tracker_->previous_surface()
      );
      if (result.converged) {
         surface_radius = result.surface.radius();
      } else {
         amrex::Print()
            << "apparent-horizon find failed at step " << step << ": "
            << result.failure_reason << '\n';
      }
   }
   const int root = amrex::ParallelDescriptor::IOProcessorNumber();
   std::array<int, 2> integer_packet{{
      result.converged ? 1 : 0,
      result.iterations,
   }};
   std::array<amrex::Real, 11> real_packet{{
      result.pseudo_time,
      result.theta_linf_times_mass,
      result.theta_l2_times_mass,
      result.area,
      result.irreducible_mass,
      result.centroid[0],
      result.centroid[1],
      result.centroid[2],
      result.minimum_radius,
      result.maximum_radius,
      result.mean_radius,
   }};
   amrex::ParallelDescriptor::Bcast(
      integer_packet.data(), integer_packet.size(), root
   );
   amrex::ParallelDescriptor::Bcast(
      real_packet.data(), real_packet.size(), root
   );
   if (integer_packet[0] == 0) {
      amrex::Abort("apparent-horizon find failed; see the IO-rank diagnostic");
   }
   amrex::ParallelDescriptor::Bcast(
      surface_radius.data(), surface_radius.size(), root
   );
   if (!amrex::ParallelDescriptor::IOProcessor()) {
      result.converged = true;
      result.iterations = integer_packet[1];
      result.pseudo_time = real_packet[0];
      result.theta_linf_times_mass = real_packet[1];
      result.theta_l2_times_mass = real_packet[2];
      result.area = real_packet[3];
      result.irreducible_mass = real_packet[4];
      result.centroid = {{real_packet[5], real_packet[6], real_packet[7]}};
      result.minimum_radius = real_packet[8];
      result.maximum_radius = real_packet[9];
      result.mean_radius = real_packet[10];
      result.surface = ApparentHorizonSurface(
         center,
         polar_points,
         azimuthal_points,
         std::move(surface_radius)
      );
   }
   apparent_horizon_tracker_->accept_result(
      step, time, std::move(result), cfg_.output_dir
   );
   const ApparentHorizonResult* accepted = apparent_horizon_tracker_->result();
   amrex::Print()
      << "apparent horizon: step=" << step
      << " iterations=" << accepted->iterations
      << " MTheta_Linf=" << accepted->theta_linf_times_mass
      << " MTheta_L2=" << accepted->theta_l2_times_mass
      << " area=" << accepted->area
      << " irreducible_mass=" << accepted->irreducible_mass
      << " centroid=(" << accepted->centroid[0] << ','
      << accepted->centroid[1] << ',' << accepted->centroid[2] << ")\n";
#endif
}

amrex::iMultiFab const* AmrSolverBase::apparent_horizon_mask_ptr(int lev)
{
   return apparent_horizon_tracker_->apparent_horizon_mask(
      lev, *fluid_state_.state()[lev], Geom(lev)
   );
}

amrex::iMultiFab const* AmrSolverBase::excision_mask_ptr(int lev)
{
   return apparent_horizon_tracker_->excision_mask(
      lev, *fluid_state_.state()[lev], Geom(lev)
   );
}

} // namespace pyfluid_amrex
