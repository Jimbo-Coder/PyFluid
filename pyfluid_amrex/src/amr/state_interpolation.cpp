#include "amr/state_interpolation.hpp"

#include <AMReX_Box.H>
#include <AMReX_GpuAtomic.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_GpuElixir.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_Math.H>

#include "pyfluid_amrex/coordinate_locations.hpp"
#include "pyfluid_amrex/fixed_spacetime.hpp"
#include "pyfluid_amrex/metric_provider.hpp"
#include "pyfluid_amrex/rhd_c2p.hpp"
#include "pyfluid_amrex/rmhd_c2p.hpp"
#include "pyfluid_amrex/state.hpp"

namespace pyfluid_amrex {
namespace {

constexpr int AdmissibilityBisectionIterations = 24;

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool relativistic_child_is_admissible(
   PhysicalSystem physical_system,
   int eos_method,
   EOSParams const& eos,
   amrex::Array4<amrex::Real const> const& coarse,
   int coarse_component,
   amrex::Array4<amrex::Real const> const& fine,
   int fine_component,
   int coarse_i,
   int coarse_j,
   int coarse_k,
   int fine_i,
   int fine_j,
   int fine_k,
   amrex::Real theta,
   SpacetimeMetric const& metric
) noexcept
{
   const auto blended = [=] AMREX_GPU_HOST_DEVICE(int component) noexcept {
      const amrex::Real parent = coarse(
         coarse_i, coarse_j, coarse_k, coarse_component + component
      );
      return parent + theta * (
         fine(fine_i, fine_j, fine_k, fine_component + component) - parent
      );
   };

   if (physical_system == PhysicalSystem::RMHD) {
      const RMHDConserved conserved{
         blended(RHO),
         blended(MOM1),
         blended(MOM2),
         blended(MOM3),
         blended(ENERGY),
         blended(B1),
         blended(B2),
         blended(B3),
      };
      const RMHDRecoveryResult recovered = recover_srmhd(
         conserved, metric, eos
      );
      return recovered.succeeded()
         && recovered.primitive.rho >= eos.rho_floor
         && recovered.primitive.pressure >= eos.pressure_floor;
   }

   const RHDConserved conserved{
      blended(RHO),
      blended(MOM1),
      blended(MOM2),
      blended(MOM3),
      blended(ENERGY),
   };
   const RHDRecoveryResult recovered = recover_rhd_runtime(
      eos_method, conserved, metric, eos
   );
   return recovered.attempt.succeeded()
      && recovered.primitive.rho >= eos.rho_floor
      && recovered.primitive.pressure >= eos.pressure_floor;
}

} // namespace

void MeshBasisStateInterpolater::configure(
   amrex::Interpolater* base,
   int coordinate,
   MeshAxes const& axes,
   bool transform_state_vectors,
   PhysicalSystem physical_system,
   int eos_method,
   EOSParams const& eos,
   FixedSpacetimeParameters const& fixed_spacetime,
   bool preserve_relativistic_admissibility
)
{
   base_ = base;
   coordinate_ = coordinate;
   axes_ = axes;
   transform_state_vectors_ = transform_state_vectors;
   physical_system_ = physical_system;
   eos_method_ = eos_method;
   eos_ = eos;
   fixed_spacetime_ = fixed_spacetime;
   preserve_relativistic_admissibility_ =
      preserve_relativistic_admissibility;
}

amrex::Box MeshBasisStateInterpolater::CoarseBox(
   amrex::Box const& fine,
   int ratio
)
{
   return base().CoarseBox(fine, ratio);
}

amrex::Box MeshBasisStateInterpolater::CoarseBox(
   amrex::Box const& fine,
   amrex::IntVect const& ratio
)
{
   return base().CoarseBox(fine, ratio);
}

void MeshBasisStateInterpolater::interp(
   amrex::FArrayBox const& coarse,
   int coarse_component,
   amrex::FArrayBox& fine,
   int fine_component,
   int num_components,
   amrex::Box const& fine_region,
   amrex::IntVect const& ratio,
   amrex::Geometry const& coarse_geometry,
   amrex::Geometry const& fine_geometry,
   amrex::Vector<amrex::BCRec> const& boundary_conditions,
   int actual_component,
   int actual_state,
   amrex::RunOn run_on
)
{
   if (!transform_state_vectors_ || coordinate_ == MeshCartesian) {
      if (preserve_relativistic_admissibility_) {
         // AMReX may split a requested FillPatch region through a coarse
         // parent's child family.  Interpolate and limit a private,
         // sibling-aligned region so the result cannot depend on FAB or
         // rank decomposition, then copy back only the requested cells.
         const amrex::Box sibling_region = amrex::refine(
            amrex::coarsen(fine_region, ratio), ratio
         );
         if (sibling_region == fine_region) {
            base().interp(
               coarse,
               coarse_component,
               fine,
               fine_component,
               num_components,
               fine_region,
               ratio,
               coarse_geometry,
               fine_geometry,
               boundary_conditions,
               actual_component,
               actual_state,
               run_on
            );
            limit_relativistic_child_deviations(
               coarse,
               coarse_component,
               fine,
               fine_component,
               num_components,
               fine_region,
               ratio,
               fine_geometry,
               actual_component,
               run_on
            );
            return;
         }
         AMREX_ALWAYS_ASSERT(
            coarse.box().contains(base().CoarseBox(sibling_region, ratio))
         );
         amrex::FArrayBox sibling_fine(sibling_region, num_components);
         amrex::Elixir sibling_fine_elixir;
         if (run_on == amrex::RunOn::Device && amrex::Gpu::inLaunchRegion()) {
            // The final device copy is asynchronous. Defer scratch release on
            // its stream; the CPU path keeps ordinary stack ownership.
            sibling_fine_elixir = sibling_fine.elixir();
         }
         base().interp(
            coarse,
            coarse_component,
            sibling_fine,
            0,
            num_components,
            sibling_region,
            ratio,
            coarse_geometry,
            fine_geometry,
            boundary_conditions,
            actual_component,
            actual_state,
            run_on
         );
         limit_relativistic_child_deviations(
            coarse,
            coarse_component,
            sibling_fine,
            0,
            num_components,
            sibling_region,
            ratio,
            fine_geometry,
            actual_component,
            run_on
         );
         if (run_on == amrex::RunOn::Device) {
            fine.copy<amrex::RunOn::Device>(
               sibling_fine,
               fine_region,
               0,
               fine_region,
               fine_component,
               num_components
            );
         } else {
            fine.copy<amrex::RunOn::Host>(
               sibling_fine,
               fine_region,
               0,
               fine_region,
               fine_component,
               num_components
            );
         }
         return;
      }
      base().interp(
         coarse,
         coarse_component,
         fine,
         fine_component,
         num_components,
         fine_region,
         ratio,
         coarse_geometry,
         fine_geometry,
         boundary_conditions,
         actual_component,
         actual_state,
         run_on
      );
      return;
   }

   amrex::FArrayBox cartesian_coarse(coarse.box(), num_components);
   amrex::Elixir cartesian_coarse_elixir;
   if (run_on == amrex::RunOn::Device && amrex::Gpu::inLaunchRegion()) {
      // Both basis transforms and interpolation are asynchronous on device.
      // Retain each scratch FAB until the current stream has consumed it.
      cartesian_coarse_elixir = cartesian_coarse.elixir();
   }
   transform_state_vectors(
      coarse,
      coarse_component,
      cartesian_coarse,
      0,
      num_components,
      coarse.box(),
      coarse_geometry,
      true
   );
   amrex::FArrayBox cartesian_fine(fine.box(), num_components);
   amrex::Elixir cartesian_fine_elixir;
   if (run_on == amrex::RunOn::Device && amrex::Gpu::inLaunchRegion()) {
      cartesian_fine_elixir = cartesian_fine.elixir();
   }
   base().interp(
      cartesian_coarse,
      0,
      cartesian_fine,
      0,
      num_components,
      fine_region,
      ratio,
      coarse_geometry,
      fine_geometry,
      boundary_conditions,
      actual_component,
      actual_state,
      run_on
   );
   transform_state_vectors(
      cartesian_fine,
      0,
      fine,
      fine_component,
      num_components,
      fine_region,
      fine_geometry,
      false
   );
}

void MeshBasisStateInterpolater::transform_state_vectors(
   amrex::FArrayBox const& source,
   int source_component,
   amrex::FArrayBox& destination,
   int destination_component,
   int num_components,
   amrex::Box const& box,
   amrex::Geometry const& geometry,
   bool to_cartesian
) const
{
   const auto source_array = source.const_array();
   const auto destination_array = destination.array();
   const auto problo = geometry.ProbLoArray();
   const auto dx = geometry.CellSizeArray();
   const int coordinate = coordinate_;
   const MeshAxes axes = axes_;
   amrex::ParallelFor(
      box,
      num_components,
      [=] AMREX_GPU_DEVICE(
         int i,
         int j,
         int k,
         int component
      ) noexcept {
         const int state_component = component;
         const int vector_begin =
            state_component >= MOM1 && state_component <= MOM3
               ? MOM1
               : (state_component >= B1 && state_component <= B3 ? B1 : -1);
         if (vector_begin < 0 || vector_begin + 2 >= num_components) {
            destination_array(i, j, k, destination_component + component) =
               source_array(i, j, k, source_component + component);
            return;
         }

         const StateVector3 source_vector{
            source_array(i, j, k, source_component + vector_begin),
            source_array(i, j, k, source_component + vector_begin + 1),
            source_array(i, j, k, source_component + vector_begin + 2),
         };
         const auto q = mesh_coordinate_triplet_at_location(
            coordinate, axes, problo, dx, i, j, k, false, false, false
         );
         const StateVector3 transformed = to_cartesian
            ? mesh_basis_vector_to_cartesian(
                 coordinate, axes, q, source_vector
              )
            : cartesian_vector_to_mesh_basis(
                 coordinate, axes, q, source_vector
              );
         const int vector_component = state_component - vector_begin;
         destination_array(i, j, k, destination_component + component) =
            vector_component == 0
               ? transformed.x
               : (vector_component == 1 ? transformed.y : transformed.z);
      }
   );
}

void MeshBasisStateInterpolater::limit_relativistic_child_deviations(
   amrex::FArrayBox const& coarse,
   int coarse_component,
   amrex::FArrayBox& fine,
   int fine_component,
   int num_components,
   amrex::Box const& fine_region,
   amrex::IntVect const& ratio,
   amrex::Geometry const& fine_geometry,
   int actual_component,
   amrex::RunOn run_on
) const
{
   const int physical_components = base_component_count(physical_system_);
   if (!physical_system_is_relativistic(physical_system_)
       || actual_component != RHO
       || num_components < physical_components) {
      return;
   }

   AMREX_ALWAYS_ASSERT(coarse.box().contains(amrex::coarsen(fine_region, ratio)));
   AMREX_ALWAYS_ASSERT(fine.box().contains(fine_region));
   for (int direction = 0; direction < AMREX_SPACEDIM; ++direction) {
      AMREX_ALWAYS_ASSERT(ratio[direction] > 0);
   }

   const auto coarse_values = coarse.const_array();
   const auto fine_values = fine.array();
   const auto original_fine_values = fine.const_array();
   const amrex::Box coarse_region = amrex::coarsen(fine_region, ratio);
   const amrex::Dim3 fine_lo = amrex::lbound(fine_region);
   const amrex::Dim3 fine_hi = amrex::ubound(fine_region);
   const auto lower = fine_geometry.ProbLoArray();
   const auto spacing = fine_geometry.CellSizeArray();
   const int ratio1 = ratio[0];
#if (AMREX_SPACEDIM >= 2)
   const int ratio2 = ratio[1];
#else
   constexpr int ratio2 = 1;
#endif
#if (AMREX_SPACEDIM == 3)
   const int ratio3 = ratio[2];
#else
   constexpr int ratio3 = 1;
#endif
   const int children_per_parent = ratio1 * ratio2 * ratio3;
   const PhysicalSystem physical_system = physical_system_;
   const int eos_method = eos_method_;
   const EOSParams eos = eos_;
   const FixedSpacetimeParameters fixed_spacetime = fixed_spacetime_;
   int host_transfer_failure = 0;
   amrex::Gpu::DeviceScalar<int> device_transfer_failure{0};
   const bool run_on_device =
      run_on == amrex::RunOn::Device && amrex::Gpu::inLaunchRegion();
   int* transfer_failure_ptr = run_on_device
      ? device_transfer_failure.dataPtr()
      : &host_transfer_failure;

   AMREX_HOST_DEVICE_PARALLEL_FOR_3D_FLAG(
      run_on,
      coarse_region,
      coarse_i,
      coarse_j,
      coarse_k,
   {
      const auto visit_children = [=] AMREX_GPU_HOST_DEVICE(
         amrex::Real theta,
         bool test_admissibility
      ) noexcept {
         bool result = true;
         for (int child = 0; child < children_per_parent; ++child) {
            int remainder = child;
            const int offset1 = remainder % ratio1;
            remainder /= ratio1;
            const int offset2 = remainder % ratio2;
            const int offset3 = remainder / ratio2;
            const int fine_i = coarse_i * ratio1 + offset1;
            const int fine_j = coarse_j * ratio2 + offset2;
            const int fine_k = coarse_k * ratio3 + offset3;
            if (fine_i < fine_lo.x || fine_i > fine_hi.x
                || fine_j < fine_lo.y || fine_j > fine_hi.y
                || fine_k < fine_lo.z || fine_k > fine_hi.z) {
               continue;
            }
            if (test_admissibility) {
               const SpacetimeMetric metric =
                  fixed_spacetime_geometry_at_cell<AMREX_SPACEDIM>(
                     fixed_spacetime,
                     lower,
                     spacing,
                     fine_i,
                     fine_j,
                     fine_k,
                     0.0
                  ).metric;
               result = result && relativistic_child_is_admissible(
                  physical_system,
                  eos_method,
                  eos,
                  coarse_values,
                  coarse_component,
                  original_fine_values,
                  fine_component,
                  coarse_i,
                  coarse_j,
                  coarse_k,
                  fine_i,
                  fine_j,
                  fine_k,
                  theta,
                  metric
               );
            } else {
               for (int component = 0; component < num_components; ++component) {
                  const amrex::Real parent = coarse_values(
                     coarse_i,
                     coarse_j,
                     coarse_k,
                     coarse_component + component
                  );
                  fine_values(
                     fine_i, fine_j, fine_k, fine_component + component
                  ) = parent + theta * (
                     original_fine_values(
                        fine_i,
                        fine_j,
                        fine_k,
                        fine_component + component
                     ) - parent
                  );
               }
            }
         }
         return result;
      };

      bool has_child_deviation = false;
      bool finite_children = true;
      for (int child = 0; child < children_per_parent; ++child) {
         int remainder = child;
         const int offset1 = remainder % ratio1;
         remainder /= ratio1;
         const int offset2 = remainder % ratio2;
         const int offset3 = remainder / ratio2;
         const int fine_i = coarse_i * ratio1 + offset1;
         const int fine_j = coarse_j * ratio2 + offset2;
         const int fine_k = coarse_k * ratio3 + offset3;
         if (fine_i < fine_lo.x || fine_i > fine_hi.x
             || fine_j < fine_lo.y || fine_j > fine_hi.y
             || fine_k < fine_lo.z || fine_k > fine_hi.z) {
            continue;
         }
         for (int component = 0; component < num_components; ++component) {
            finite_children = finite_children && amrex::Math::isfinite(
               original_fine_values(
                  fine_i, fine_j, fine_k, fine_component + component
               )
            );
            if (component < physical_components) {
               has_child_deviation = has_child_deviation || (
                  original_fine_values(
                     fine_i, fine_j, fine_k, fine_component + component
                  ) != coarse_values(
                     coarse_i,
                     coarse_j,
                     coarse_k,
                     coarse_component + component
                  )
               );
            }
         }
      }
      // A nonfinite transfer is not a lower-order admissibility event. Leave
      // it visible as a transfer failure instead of silently replacing it
      // with the coarse parent.
      if (!finite_children) {
         amrex::Gpu::Atomic::Max(transfer_failure_ptr, 1);
         return;
      }
      if (!has_child_deviation || visit_children(1.0, true)) {
         return;
      }

      // The accepted coarse state is the conservative low-order member of the
      // interpolation family. A common theta keeps every child on its segment
      // from that parent and preserves the child average whenever the base
      // interpolater is conservative.
      if (!visit_children(0.0, true)) {
         amrex::Gpu::Atomic::Max(transfer_failure_ptr, 2);
         return;
      }
      amrex::Real lower_theta = 0.0;
      amrex::Real upper_theta = 1.0;
      for (int iteration = 0;
           iteration < AdmissibilityBisectionIterations;
           ++iteration) {
         const amrex::Real candidate_theta = 0.5 * (
            lower_theta + upper_theta
         );
         if (visit_children(candidate_theta, true)) {
            lower_theta = candidate_theta;
         } else {
            upper_theta = candidate_theta;
         }
      }
      visit_children(lower_theta, false);
   });

   const int failure = run_on_device
      ? device_transfer_failure.dataValue()
      : host_transfer_failure;
   if (failure == 1) {
      amrex::Abort(
         "relativistic coarse-fine interpolation produced nonfinite state"
      );
   }
   if (failure == 2) {
      amrex::Abort(
         "relativistic coarse-fine interpolation has a non-admissible "
         "coarse parent under the fine-cell metric"
      );
   }
}

} // namespace pyfluid_amrex
