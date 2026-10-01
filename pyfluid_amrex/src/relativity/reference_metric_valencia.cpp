#include "pyfluid_amrex/reference_metric_valencia.hpp"

#include <AMReX_GpuLaunch.H>

#include "pyfluid_amrex/flux_divergence.hpp"
#include "pyfluid_amrex/reference_metric_valencia_pointwise.hpp"
#include "pyfluid_amrex/reference_metric_spacetime.hpp"
#include "pyfluid_amrex/relativistic_primitive.hpp"
#include "pyfluid_amrex/state.hpp"
#include "pyfluid_amrex/z4c_state_access.hpp"
#include "pyfluid_amrex/kernels_generated/z4c/geometry.hpp"

namespace pyfluid_amrex {

using namespace amrex::literals;

void compute_rhd_reference_metric_rhs(
   amrex::MultiFab& rhs,
   std::array<amrex::MultiFab*, AMREX_SPACEDIM> const& fluxes,
   amrex::MultiFab const& primitive,
   amrex::Geometry const& geometry,
   int coordinate,
   MeshAxes const& axes,
   EOSParams const& eos,
   FixedSpacetimeParameters const& fixed_spacetime,
   amrex::Real time,
   amrex::MultiFab const* cell_spacetime
)
{
   const auto problo = geometry.ProbLoArray();
   const auto dx = geometry.CellSizeArray();
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
   {
   for (amrex::MFIter mfi(rhs, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const amrex::Box& bx = mfi.tilebox();
      const auto flux0 = fluxes[0]->const_array(mfi);
#if (AMREX_SPACEDIM >= 2)
      const auto flux1 = fluxes[1]->const_array(mfi);
#endif
#if (AMREX_SPACEDIM == 3)
      const auto flux2 = fluxes[2]->const_array(mfi);
#endif
      const auto cached = primitive.const_array(mfi);
      const auto stored_metric = cell_spacetime != nullptr
         ? cell_spacetime->const_array(mfi) : amrex::Array4<amrex::Real const>{};
      const auto output = rhs.array(mfi);
      amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
         const amrex::Real volume = mesh_cell_volume_scale(
            coordinate, axes, problo, dx, i, j, k
         );
         const amrex::Real area0_lo = mesh_face_area_scale(
            0, coordinate, axes, problo, dx, i, j, k
         );
         const amrex::Real area0_hi = mesh_face_area_scale(
            0, coordinate, axes, problo, dx, i + 1, j, k
         );
#if (AMREX_SPACEDIM >= 2)
         const amrex::Real area1_lo = mesh_face_area_scale(
            1, coordinate, axes, problo, dx, i, j, k
         );
         const amrex::Real area1_hi = mesh_face_area_scale(
            1, coordinate, axes, problo, dx, i, j + 1, k
         );
#endif
#if (AMREX_SPACEDIM == 3)
         const amrex::Real area2_lo = mesh_face_area_scale(
            2, coordinate, axes, problo, dx, i, j, k
         );
         const amrex::Real area2_hi = mesh_face_area_scale(
            2, coordinate, axes, problo, dx, i, j, k + 1
         );
#endif
         const RHDPrimitive local = rhd_primitive_at<EOSRuntime>(
            cached, i, j, k, eos
         );
         const auto location = mesh_coordinate_triplet_at_location(
            coordinate, axes, problo, dx, i, j, k, false, false, false
         );
         const SpacetimeMetric local_metric =
            stored_metric
            ? generated::z4c_spacetime_metric(load_z4c_state(stored_metric, i, j, k))
            : reference_metric_fixed_spacetime_geometry(
                 fixed_spacetime, coordinate, axes, location, time).metric;
         const auto basis_divergence =
            rhd_reference_metric_momentum_basis_divergence(
               local, local_metric, coordinate, axes, location
            );
         const amrex::Real pressure_flux =
            local_metric.lapse * local_metric.sqrt_det_gamma * local.pressure;

         for (int component = 0; component < NVAR_HYDRO; ++component) {
            amrex::Real integral0_lo = area0_lo * flux0(i, j, k, component);
            amrex::Real integral0_hi = area0_hi * flux0(i + 1, j, k, component);
            amrex::Real integral1_lo = 0.0_rt;
            amrex::Real integral1_hi = 0.0_rt;
            amrex::Real integral2_lo = 0.0_rt;
            amrex::Real integral2_hi = 0.0_rt;
#if (AMREX_SPACEDIM >= 2)
            integral1_lo = area1_lo * flux1(i, j, k, component);
            integral1_hi = area1_hi * flux1(i, j + 1, k, component);
#endif
#if (AMREX_SPACEDIM == 3)
            integral2_lo = area2_lo * flux2(i, j, k, component);
            integral2_hi = area2_hi * flux2(i, j, k + 1, component);
#endif
            if (component >= MOM1 && component <= MOM3) {
               const int axis = mesh_vector_axis_for_component(
                  coordinate, axes, component - MOM1
               );
               const int direction = mesh_direction_for_axis(axes, axis);
               if (direction == 0) {
                  integral0_lo -= pressure_flux * area0_lo;
                  integral0_hi -= pressure_flux * area0_hi;
               }
#if (AMREX_SPACEDIM >= 2)
               if (direction == 1) {
                  integral1_lo -= pressure_flux * area1_lo;
                  integral1_hi -= pressure_flux * area1_hi;
               }
#endif
#if (AMREX_SPACEDIM == 3)
               if (direction == 2) {
                  integral2_lo -= pressure_flux * area2_lo;
                  integral2_hi -= pressure_flux * area2_hi;
               }
#endif
            }
            amrex::Real value = -finite_volume_divergence_from_face_integrals(
               integral0_hi,
               integral0_lo,
               integral1_hi,
               integral1_lo,
               integral2_hi,
               integral2_lo,
               volume
            );
            if (component >= MOM1 && component <= MOM3) {
               value -= basis_divergence[component - MOM1];
            }
            output(i, j, k, component) = value;
         }
      });
   }
   }
}

} // namespace pyfluid_amrex
