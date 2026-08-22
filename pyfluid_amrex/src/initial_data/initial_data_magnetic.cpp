#include "pyfluid_amrex/initial_data/magnetic.hpp"

#include <AMReX.H>
#include <AMReX_ParallelDescriptor.H>

#include <array>
#include <cmath>

#include "pyfluid_amrex/config.hpp"
#include "pyfluid_amrex/kernels_generated/relativistic/core.hpp"
#include "pyfluid_amrex/relativistic_primitive.hpp"
#include "pyfluid_amrex/z4c_state_access.hpp"

namespace pyfluid_amrex {

PressurePoloidalMagneticSeedSummary apply_pressure_poloidal_magnetic_seed(
   FaceMagneticField& face_magnetic,
   amrex::MultiFab& primitive,
   amrex::MultiFab const& cell_spacetime,
   amrex::Geometry const& geometry,
   RunConfig const& config,
   EOSParams const& eos,
   PressurePoloidalMagneticSeedParameters const& parameters,
   PressurePoloidalMagneticSeedWorkspace& workspace
)
{
#if (AMREX_SPACEDIM != 3)
   amrex::ignore_unused(
      face_magnetic,
      primitive,
      cell_spacetime,
      geometry,
      config,
      eos,
      parameters,
      workspace
   );
   amrex::Abort("pressure-poloidal magnetic seed requires 3D");
   return {};
#else
   if (primitive.nComp() < NVAR_MHD
       || primitive.boxArray() != cell_spacetime.boxArray()) {
      amrex::Abort("pressure-poloidal magnetic seed received incompatible fields");
   }
   const amrex::Real axis_norm = std::sqrt(
      parameters.axis[0] * parameters.axis[0]
      + parameters.axis[1] * parameters.axis[1]
      + parameters.axis[2] * parameters.axis[2]
   );
   if (!(axis_norm > 0.0) || !std::isfinite(axis_norm)) {
      amrex::Abort("pressure-poloidal magnetic seed axis is invalid");
   }
   const std::array<amrex::Real, 3> axis{{
      parameters.axis[0] / axis_norm,
      parameters.axis[1] / axis_norm,
      parameters.axis[2] / axis_norm,
   }};

   if (!workspace.pressure) {
      workspace.pressure = std::make_unique<amrex::MultiFab>(
         primitive.boxArray(), primitive.DistributionMap(), 1, 1
      );
      workspace.comoving_field_squared = std::make_unique<amrex::MultiFab>(
         primitive.boxArray(), primitive.DistributionMap(), 1, 0
      );
      workspace.divergence = std::make_unique<amrex::MultiFab>(
         primitive.boxArray(), primitive.DistributionMap(), 1, 0
      );
      allocate_edge_electric_field(
         workspace.vector_potential,
         primitive.boxArray(),
         primitive.DistributionMap()
      );
   }

   workspace.pressure->setVal(0.0);
   amrex::MultiFab::Copy(
      *workspace.pressure, primitive, ENERGY, 0, 1, 0
   );
   workspace.pressure->FillBoundary(geometry.periodicity());
   amrex::Real maximum_pressure = workspace.pressure->norm0(0, 0, true);
   amrex::ParallelDescriptor::ReduceRealMax(maximum_pressure);
   if (!(maximum_pressure > 0.0) || !std::isfinite(maximum_pressure)) {
      amrex::Abort("pressure-poloidal magnetic seed has no positive pressure support");
   }

   const auto lower = geometry.ProbLoArray();
   const auto spacing = geometry.CellSizeArray();
   const amrex::Real inverse_maximum_pressure = 1.0 / maximum_pressure;
   const amrex::Real cutoff = parameters.pressure_cutoff_fraction;
   const amrex::Real exponent = parameters.exponent;
   const amrex::Real axis1 = axis[0];
   const amrex::Real axis2 = axis[1];
   const amrex::Real axis3 = axis[2];
   const amrex::Real center1 = parameters.center[0];
   const amrex::Real center2 = parameters.center[1];
   const amrex::Real center3 = parameters.center[2];
   set_edge_field_value(workspace.vector_potential, 0.0);
   for (int component = 0; component < 3; ++component) {
      amrex::MultiFab& edge = *workspace.vector_potential.e[component];
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
      {
      for (amrex::MFIter mfi(edge, amrex::TilingIfNotGPU());
           mfi.isValid(); ++mfi) {
         const amrex::Box box = mfi.tilebox();
         const auto pressure = workspace.pressure->const_array(mfi);
         const auto potential = edge.array(mfi);
         const int vector_component = component;
         amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(
            int i, int j, int k
         ) noexcept {
            amrex::Real edge_pressure = 0.0;
            if (vector_component == 0) {
               edge_pressure = 0.25 * (
                  pressure(i, j - 1, k - 1, 0)
                  + pressure(i, j, k - 1, 0)
                  + pressure(i, j - 1, k, 0)
                  + pressure(i, j, k, 0)
               );
            } else if (vector_component == 1) {
               edge_pressure = 0.25 * (
                  pressure(i - 1, j, k - 1, 0)
                  + pressure(i, j, k - 1, 0)
                  + pressure(i - 1, j, k, 0)
                  + pressure(i, j, k, 0)
               );
            } else {
               edge_pressure = 0.25 * (
                  pressure(i - 1, j - 1, k, 0)
                  + pressure(i, j - 1, k, 0)
                  + pressure(i - 1, j, k, 0)
                  + pressure(i, j, k, 0)
               );
            }
            const amrex::Real x = lower[0]
               + (static_cast<amrex::Real>(i)
                  + (vector_component == 0 ? 0.5 : 0.0)) * spacing[0]
               - center1;
            const amrex::Real y = lower[1]
               + (static_cast<amrex::Real>(j)
                  + (vector_component == 1 ? 0.5 : 0.0)) * spacing[1]
               - center2;
            const amrex::Real z = lower[2]
               + (static_cast<amrex::Real>(k)
                  + (vector_component == 2 ? 0.5 : 0.0)) * spacing[2]
               - center3;
            const amrex::Real tangent1 = axis2 * z - axis3 * y;
            const amrex::Real tangent2 = axis3 * x - axis1 * z;
            const amrex::Real tangent3 = axis1 * y - axis2 * x;
            const amrex::Real tangent = vector_component == 0
               ? tangent1 : (vector_component == 1 ? tangent2 : tangent3);
            const amrex::Real support = amrex::max(
               edge_pressure * inverse_maximum_pressure - cutoff,
               amrex::Real(0.0)
            );
            potential(i, j, k, 0) = tangent * std::pow(support, exponent);
         });
      }
      }
   }
   enforce_edge_electric_boundary(
      workspace.vector_potential, geometry, config
   );
   for (int direction = 0; direction < 3; ++direction) {
      face_magnetic.b[direction]->setVal(0.0);
   }
   // The CT update is B <- B - dt curl(E); dt=-1 applies B=curl(A).
   apply_edge_electric_curl(
      face_magnetic, workspace.vector_potential, geometry, -1.0, config
   );
   apply_face_b_boundary(face_magnetic, geometry, config);
   sync_cell_b_from_faces(primitive, face_magnetic, geometry, false);

   workspace.comoving_field_squared->setVal(0.0);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
   {
   for (amrex::MFIter mfi(
           *workspace.comoving_field_squared, amrex::TilingIfNotGPU()
        ); mfi.isValid(); ++mfi) {
      const amrex::Box box = mfi.tilebox();
      const auto cached = primitive.const_array(mfi);
      const auto spacetime = cell_spacetime.const_array(mfi);
      const auto output = workspace.comoving_field_squared->array(mfi);
      amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(
         int i, int j, int k
      ) noexcept {
         const SpacetimeMetric metric = generated::z4c_spacetime_metric(
            load_z4c_state(spacetime, i, j, k)
         );
         const RMHDPrimitive point = rmhd_primitive_at<EOSRuntime>(
            cached, i, j, k, eos
         );
         output(i, j, k, 0) = generated::rmhd_comoving_magnetic_squared(
            point, metric
         );
      });
   }
   }
   amrex::Real maximum_comoving_squared =
      workspace.comoving_field_squared->norm0(0, 0, true);
   amrex::ParallelDescriptor::ReduceRealMax(maximum_comoving_squared);
   if (!(maximum_comoving_squared > 0.0)
       || !std::isfinite(maximum_comoving_squared)) {
      amrex::Abort("pressure-poloidal magnetic seed produced no finite field");
   }
   const amrex::Real unit_maximum = std::sqrt(maximum_comoving_squared);
   const amrex::Real normalization =
      parameters.maximum_comoving_field / unit_maximum;
   for (int direction = 0; direction < 3; ++direction) {
      face_magnetic.b[direction]->mult(
         normalization, 0, 1, face_magnetic.b[direction]->nGrow()
      );
   }
   sync_cell_b_from_faces(primitive, face_magnetic, geometry, false);

   fill_face_divergence(
      *workspace.divergence, face_magnetic, geometry, false
   );
   amrex::Real divergence_linf = workspace.divergence->norm0(0, 0, true);
   amrex::ParallelDescriptor::ReduceRealMax(divergence_linf);
   return {
      maximum_pressure,
      unit_maximum,
      normalization,
      parameters.maximum_comoving_field,
      divergence_linf,
   };
#endif
}

} // namespace pyfluid_amrex
