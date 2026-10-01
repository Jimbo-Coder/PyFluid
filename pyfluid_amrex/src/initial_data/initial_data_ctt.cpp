#include "pyfluid_amrex/initial_data/ctt.hpp"

#include <AMReX.H>
#include <AMReX_GpuAsyncArray.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_ParallelReduce.H>

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>

#include "pyfluid_amrex/constants.hpp"
#include "pyfluid_amrex/eos/equation_of_state.hpp"
#include "pyfluid_amrex/kernels_generated/differential_ops.hpp"
#include "pyfluid_amrex/kernels_generated/initial_data.hpp"

namespace pyfluid_amrex {

using namespace amrex::literals;

namespace {

enum class ConformalScalarOperator {
   cartesian,
   axisymmetric_spherical,
   spherical_axisymmetric_storage,
   spherical,
};

amrex::Vector<GridFieldSymmetryKind> conformal_vector_symmetries()
{
   return {
      GridFieldSymmetryKind::Vector1,
      GridFieldSymmetryKind::Vector2,
      GridFieldSymmetryKind::Vector3,
   };
}

amrex::Box cartesian_first_derivative_domain(
   InitialDataEllipticHierarchy const& hierarchy,
   int level,
   amrex::MultiFab const& field
)
{
   amrex::Box domain = amrex::convert(
      hierarchy.geometry[level].Domain(), field.ixType()
   );
   for (int direction = 0; direction < AMREX_SPACEDIM; ++direction) {
      // Cell-centered elliptic fields have face-BC ghosts. At physical outer
      // vertices, generic nodal FillPatch instead constant-extrapolates; those
      // ghosts are not derivative data. Use the solved boundary value and an
      // interior one-sided stencil there. True symmetries retain centered
      // differences through explicitly mirrored/periodic ghosts.
      const bool use_ghosts = field.ixType().cellCentered()
         || hierarchy.geometry[level].isPeriodic(direction);
      if (use_ghosts || hierarchy.reflection_lo[direction] != 0) {
         domain.growLo(direction, 1);
      }
      if (use_ghosts || hierarchy.reflection_hi[direction] != 0) {
         domain.growHi(direction, 1);
      }
   }
   return domain;
}

template <int Direction>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
amrex::Real cartesian_first_derivative(
   amrex::Array4<amrex::Real const> const& field,
   int i, int j, int k, int component,
   amrex::Dim3 const& domain_lo,
   amrex::Dim3 const& domain_hi,
   amrex::Real inverse_spacing
) noexcept
{
   const int index = Direction == 0 ? i : (Direction == 1 ? j : k);
   const int lower = Direction == 0 ? domain_lo.x
      : (Direction == 1 ? domain_lo.y : domain_lo.z);
   const int upper = Direction == 0 ? domain_hi.x
      : (Direction == 1 ? domain_hi.y : domain_hi.z);
   if (upper == lower) {
      return 0.0_rt;
   }
   if (upper == lower + 1) {
      // A one-cell nodal line defines only a linear interpolant. Invariant
      // dimensional-degeneracy data therefore give exactly zero.
      return inverse_spacing * (
         generated::sample_direction<Direction>(
            field, i, j, k, component, upper - index)
         - generated::sample_direction<Direction>(
            field, i, j, k, component, lower - index)
      );
   }
   return generated::domain_first<2, Direction>(
      field, i, j, k, component, index, lower, upper, inverse_spacing
   );
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void fill_reduced_cartesian_first_derivatives(
   amrex::Array4<amrex::Real const> const& field,
   int i,
   int j,
   int k,
   amrex::GpuArray<int, AMREX_SPACEDIM> const& physical_axes,
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& inverse_spacing,
   amrex::Dim3 const& domain_lo,
   amrex::Dim3 const& domain_hi,
   amrex::GpuArray<amrex::Real, 9>& derivatives
) noexcept
{
   for (int entry = 0; entry < 9; ++entry) {
      derivatives[entry] = 0.0_rt;
   }
   for (int component = 0; component < 3; ++component) {
      derivatives[3 * component + physical_axes[0]] =
         cartesian_first_derivative<0>(
            field, i, j, k, component, domain_lo, domain_hi, inverse_spacing[0]
         );
#if (AMREX_SPACEDIM >= 2)
      derivatives[3 * component + physical_axes[1]] =
         cartesian_first_derivative<1>(
            field, i, j, k, component, domain_lo, domain_hi, inverse_spacing[1]
         );
#endif
#if (AMREX_SPACEDIM == 3)
      derivatives[3 * component + physical_axes[2]] =
         cartesian_first_derivative<2>(
            field, i, j, k, component, domain_lo, domain_hi, inverse_spacing[2]
         );
#endif
   }
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
amrex::Real reduced_cartesian_scalar_gradient_component(
   amrex::Array4<amrex::Real const> const& field,
   int i,
   int j,
   int k,
   int component,
   amrex::GpuArray<int, AMREX_SPACEDIM> const& physical_axes,
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& inverse_spacing,
   amrex::Dim3 const& domain_lo,
   amrex::Dim3 const& domain_hi
) noexcept
{
   amrex::Real derivative = 0.0_rt;
   if (physical_axes[0] == component) {
      derivative = cartesian_first_derivative<0>(
         field, i, j, k, 0, domain_lo, domain_hi, inverse_spacing[0]
      );
   }
#if (AMREX_SPACEDIM >= 2)
   if (physical_axes[1] == component) {
      derivative = cartesian_first_derivative<1>(
         field, i, j, k, 0, domain_lo, domain_hi, inverse_spacing[1]
      );
   }
#endif
#if (AMREX_SPACEDIM == 3)
   if (physical_axes[2] == component) {
      derivative = cartesian_first_derivative<2>(
         field, i, j, k, 0, domain_lo, domain_hi, inverse_spacing[2]
      );
   }
#endif
   return derivative;
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
amrex::Real reduced_cartesian_scalar_laplacian(
   amrex::Array4<amrex::Real const> const& field,
   int i,
   int j,
   int k,
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& inverse_spacing_squared
) noexcept
{
   amrex::Real result = generated::centered_second<2, 0>(
      field, i, j, k, 0, inverse_spacing_squared[0]
   );
#if (AMREX_SPACEDIM >= 2)
   result += generated::centered_second<2, 1>(
      field, i, j, k, 0, inverse_spacing_squared[1]
   );
#endif
#if (AMREX_SPACEDIM == 3)
   result += generated::centered_second<2, 2>(
      field, i, j, k, 0, inverse_spacing_squared[2]
   );
#endif
   return result;
}

// Consume the linear-solve candidates as update scratch. Callers no longer
// need their old values; scalar fixed points repopulate them next iteration.
// Preserve subtraction/offset/scaling order, including the scalar solve for
// f - 1, and update one ghost layer while measuring only valid points.
amrex::Real apply_conformal_relaxed_update(
   amrex::Vector<amrex::MultiFab*> const& field,
   amrex::Vector<amrex::MultiFab*> const& candidate,
   amrex::Real relaxation,
   amrex::Real candidate_offset = 0.0_rt
)
{
   amrex::Real norms[2]{0.0_rt, 0.0_rt};
   for (int level = 0; level < static_cast<int>(field.size()); ++level) {
      auto& solution = *field[level];
      auto& update = *candidate[level];
      const int num_components = solution.nComp();
      amrex::MultiFab::Subtract(update, solution, 0, 0, num_components, 1);
      if (candidate_offset != 0.0_rt) {
         update.plus(candidate_offset, 0, num_components, 1);
      }
      update.mult(relaxation, 0, num_components, 1);
      amrex::MultiFab::Add(solution, update, 0, 0, num_components, 1);
      norms[0] = std::max(norms[0], update.norm0(
         0, num_components, amrex::IntVect(0), true
      ));
      norms[1] = std::max(norms[1], solution.norm0(
         0, num_components, amrex::IntVect(0), true
      ));
   }
   amrex::ParallelAllReduce::Max(
      norms, 2, amrex::ParallelContext::CommunicatorSub()
   );
   return norms[0] / std::max(amrex::Real(1.0_rt), norms[1]);
}

template <typename FillSource>
CttHamiltonianResult solve_conformal_scalar_fixed_point(
   InitialDataEllipticHierarchy const& hierarchy,
   amrex::Vector<amrex::MultiFab*> const& field,
   InitialDataEllipticBoundaryConditions const& correction_boundary,
   CttHamiltonianOptions const& options,
   ConformalScalarOperator scalar_operator,
   FillSource&& fill_source
)
{
   const int num_levels = static_cast<int>(hierarchy.geometry.size());
   amrex::Vector<std::unique_ptr<amrex::MultiFab>> rhs_storage(num_levels);
   amrex::Vector<std::unique_ptr<amrex::MultiFab>> candidate_storage(num_levels);
   amrex::Vector<amrex::MultiFab const*> rhs(num_levels);
   amrex::Vector<amrex::MultiFab*> candidate(num_levels);
   for (int level = 0; level < num_levels; ++level) {
      rhs_storage[level] = std::make_unique<amrex::MultiFab>(
         field[level]->boxArray(), field[level]->DistributionMap(), 1, 0
      );
      candidate_storage[level] = std::make_unique<amrex::MultiFab>(
         field[level]->boxArray(), field[level]->DistributionMap(), 1, 1
      );
      rhs[level] = rhs_storage[level].get();
      candidate[level] = candidate_storage[level].get();
   }

   CttHamiltonianResult result;
   for (int iteration = 1; iteration <= options.max_nonlinear_iterations; ++iteration) {
      for (int level = 0; level < num_levels; ++level) {
         amrex::MultiFab::Copy(
            *candidate_storage[level], *field[level], 0, 0, 1, 1
         );
         candidate_storage[level]->plus(-1.0_rt, 0, 1, 1);
         fill_source(level, *rhs_storage[level]);
      }

      InitialDataEllipticResult linear;
      if (scalar_operator == ConformalScalarOperator::axisymmetric_spherical) {
         linear = solve_initial_data_axisymmetric_spherical_poisson(
            hierarchy, candidate, rhs, correction_boundary, options.linear
         );
      } else if (
         scalar_operator
         == ConformalScalarOperator::spherical_axisymmetric_storage
      ) {
         linear = solve_initial_data_spherical_axisymmetric_poisson(
            hierarchy, candidate, rhs, correction_boundary, options.linear
         );
      } else if (scalar_operator == ConformalScalarOperator::spherical) {
         linear = solve_initial_data_spherical_poisson(
            hierarchy, candidate, rhs, correction_boundary, options.linear
         );
      } else {
         linear = solve_initial_data_poisson(
            hierarchy, candidate, rhs, correction_boundary, options.linear
         );
      }
      result.nonlinear_iterations = iteration;
      result.relative_update = apply_conformal_relaxed_update(
         field, candidate, options.relaxation, 1.0_rt
      );
      result.linear_residual = linear.residual;
      if (!std::isfinite(result.relative_update)) {
         amrex::Abort("conformal scalar solve produced a nonfinite update");
      }
      if (result.relative_update <= options.relative_update_tolerance) {
         result.converged = true;
         break;
      }
   }
   return result;
}

CttHamiltonianResult solve_ctt_hamiltonian(
   InitialDataEllipticHierarchy const& hierarchy,
   amrex::Vector<amrex::MultiFab*> const& conformal_factor,
   amrex::Vector<amrex::MultiFab const*> const& energy_density,
   amrex::Vector<amrex::MultiFab const*> const& conformal_extrinsic_curvature,
   amrex::Vector<amrex::MultiFab const*> const& mean_curvature_squared,
   InitialDataEllipticBoundaryConditions const& conformal_correction_boundary,
   CttHamiltonianOptions const& options,
   ConformalScalarOperator scalar_operator
)
{
   const bool has_extrinsic_curvature = !conformal_extrinsic_curvature.empty();
   const bool has_mean_curvature = !mean_curvature_squared.empty();
   const int num_levels = static_cast<int>(hierarchy.geometry.size());
   if (
      num_levels == 0
      || static_cast<int>(conformal_factor.size()) != num_levels
      || static_cast<int>(energy_density.size()) != num_levels
      || (has_extrinsic_curvature
          && static_cast<int>(conformal_extrinsic_curvature.size()) != num_levels)
      || (has_mean_curvature
          && static_cast<int>(mean_curvature_squared.size()) != num_levels)
      || options.max_nonlinear_iterations <= 0
      || !(options.relative_update_tolerance > 0.0_rt)
      || !(options.relaxation > 0.0_rt && options.relaxation <= 1.0_rt)
   ) {
      amrex::Abort("invalid CTT Hamiltonian solve configuration");
   }

   for (int level = 0; level < num_levels; ++level) {
      if (conformal_factor[level] == nullptr || energy_density[level] == nullptr) {
         amrex::Abort("CTT Hamiltonian solve received null level data");
      }
      if (conformal_factor[level]->nComp() != 1
          || energy_density[level]->nComp() != 1
          || conformal_factor[level]->nGrow() < 1
          || energy_density[level]->boxArray()
             != conformal_factor[level]->boxArray()
          || (has_extrinsic_curvature
              && (conformal_extrinsic_curvature[level] == nullptr
                  || conformal_extrinsic_curvature[level]->nComp()
                     != CONFORMAL_A_NUM_COMPONENTS
                  || conformal_extrinsic_curvature[level]->boxArray()
                     != conformal_factor[level]->boxArray()))
          || (has_mean_curvature
              && (mean_curvature_squared[level] == nullptr
                  || mean_curvature_squared[level]->nComp() != 1
                  || mean_curvature_squared[level]->boxArray()
                     != conformal_factor[level]->boxArray()))) {
         amrex::Abort("invalid CTT Hamiltonian field layout");
      }
   }

   return solve_conformal_scalar_fixed_point(
      hierarchy,
      conformal_factor,
      conformal_correction_boundary,
      options,
      scalar_operator,
      [&](int level, amrex::MultiFab& rhs) {
         for (amrex::MFIter mfi(rhs, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            const amrex::Box box = mfi.tilebox();
            const auto psi = conformal_factor[level]->const_array(mfi);
            const auto matter = energy_density[level]->const_array(mfi);
            const auto extrinsic = has_extrinsic_curvature
               ? conformal_extrinsic_curvature[level]->const_array(mfi)
               : amrex::Array4<amrex::Real const>{};
            const auto curvature_squared = has_mean_curvature
               ? mean_curvature_squared[level]->const_array(mfi)
               : amrex::Array4<amrex::Real const>{};
            const auto source = rhs.array(mfi);
            amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
               const amrex::Real extrinsic_squared = has_extrinsic_curvature
                  ? extrinsic(i, j, k, CONFORMAL_A_SQUARED)
                  : 0.0_rt;
               const amrex::Real k_squared = has_mean_curvature
                  ? curvature_squared(i, j, k, 0)
                  : 0.0_rt;
               source(i, j, k, 0) = generated::conformal_hamiltonian_source(
                  psi(i, j, k, 0),
                  matter(i, j, k, 0),
                  extrinsic_squared,
                  k_squared
               );
            });
         }
      }
   );
}

template <bool XctsScaling>
void fill_conformal_extrinsic_curvature(
   InitialDataEllipticHierarchy const& hierarchy,
   amrex::Vector<amrex::MultiFab*> const& vector_field,
   amrex::Vector<amrex::MultiFab const*> const& conformal_factor,
   amrex::Vector<amrex::MultiFab const*> const& lapse_times_conformal_factor,
   amrex::Vector<amrex::MultiFab*> const& conformal_extrinsic_curvature
)
{
   const int num_levels = static_cast<int>(hierarchy.geometry.size());
   fill_initial_data_ghosts(
      hierarchy,
      vector_field,
      initial_data_boundary_records(
         hierarchy,
         conformal_vector_symmetries(),
         amrex::BCType::reflect_odd
      )
   );
   for (int level = 0; level < num_levels; ++level) {
      const auto spacing = hierarchy.geometry[level].CellSizeArray();
      const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> inv_spacing{
         AMREX_D_DECL(
            1.0_rt / spacing[0],
            1.0_rt / spacing[1],
            1.0_rt / spacing[2]
         )
      };
      const auto physical_axes = hierarchy.cartesian_physical_axes;
      const amrex::Box derivative_domain = cartesian_first_derivative_domain(
         hierarchy, level, *vector_field[level]
      );
      const auto derivative_lo = amrex::lbound(derivative_domain);
      const auto derivative_hi = amrex::ubound(derivative_domain);
      for (amrex::MFIter mfi(
              *conformal_extrinsic_curvature[level], amrex::TilingIfNotGPU()
           ); mfi.isValid(); ++mfi) {
         const amrex::Box box = mfi.tilebox();
         const auto vector = vector_field[level]->const_array(mfi);
         const auto psi = XctsScaling
            ? conformal_factor[level]->const_array(mfi)
            : amrex::Array4<amrex::Real const>{};
         const auto lapse_psi = XctsScaling
            ? lapse_times_conformal_factor[level]->const_array(mfi)
            : amrex::Array4<amrex::Real const>{};
         const auto extrinsic = conformal_extrinsic_curvature[level]->array(mfi);
         amrex::ParallelFor(box,
            [vector, physical_axes, inv_spacing, derivative_lo, derivative_hi,
             psi, lapse_psi, extrinsic]
            AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            amrex::GpuArray<amrex::Real, 9> derivatives{};
            fill_reduced_cartesian_first_derivatives(
               vector,
               i,
               j,
               k,
               physical_axes,
               inv_spacing,
               derivative_lo,
               derivative_hi,
               derivatives
            );
            ConformalTraceFreeTensor point =
               generated::conformal_longitudinal_tensor(
                  derivatives[0], derivatives[1], derivatives[2],
                  derivatives[3], derivatives[4], derivatives[5],
                  derivatives[6], derivatives[7], derivatives[8]
               );
            if constexpr (XctsScaling) {
               const amrex::Real scale = generated::xcts_conformal_extrinsic_scale(
                  psi(i, j, k, 0), lapse_psi(i, j, k, 0)
               );
               point.a11 *= scale;
               point.a12 *= scale;
               point.a13 *= scale;
               point.a22 *= scale;
               point.a23 *= scale;
               point.a33 *= scale;
               point.squared *= scale * scale;
            }
            extrinsic(i, j, k, CONFORMAL_A11) = point.a11;
            extrinsic(i, j, k, CONFORMAL_A12) = point.a12;
            extrinsic(i, j, k, CONFORMAL_A13) = point.a13;
            extrinsic(i, j, k, CONFORMAL_A22) = point.a22;
            extrinsic(i, j, k, CONFORMAL_A23) = point.a23;
            extrinsic(i, j, k, CONFORMAL_A33) = point.a33;
            extrinsic(i, j, k, CONFORMAL_A_SQUARED) = point.squared;
         });
      }
   }
}

struct ConformalMomentumResult {
   amrex::Real relative_update = 0.0_rt;
   amrex::Real linear_residual = 0.0_rt;
};

template <typename FillSource>
ConformalMomentumResult solve_conformal_vector_fixed_point(
   InitialDataEllipticHierarchy const& hierarchy,
   amrex::Vector<amrex::MultiFab*> const& vector_field,
   InitialDataEllipticBoundaryConditions const& boundary,
   InitialDataEllipticOptions const& linear_options,
   amrex::Real relaxation,
   FillSource&& fill_source
)
{
   const int num_levels = static_cast<int>(hierarchy.geometry.size());
   amrex::Vector<std::unique_ptr<amrex::MultiFab>> source_storage(num_levels);
   amrex::Vector<std::unique_ptr<amrex::MultiFab>> candidate_storage(num_levels);
   amrex::Vector<amrex::MultiFab const*> source(num_levels);
   amrex::Vector<amrex::MultiFab*> candidate(num_levels);
   for (int level = 0; level < num_levels; ++level) {
      source_storage[level] = std::make_unique<amrex::MultiFab>(
         vector_field[level]->boxArray(),
         vector_field[level]->DistributionMap(),
         3,
         0
      );
      candidate_storage[level] = std::make_unique<amrex::MultiFab>(
         vector_field[level]->boxArray(),
         vector_field[level]->DistributionMap(),
         3,
         1
      );
      amrex::MultiFab::Copy(
         *candidate_storage[level], *vector_field[level], 0, 0, 3, 1
      );
      source[level] = source_storage[level].get();
      candidate[level] = candidate_storage[level].get();
      fill_source(level, *source_storage[level]);
   }

   const InitialDataEllipticResult linear =
      solve_initial_data_conformal_vector_laplacian(
         hierarchy,
         candidate,
         source,
         boundary,
         linear_options
      );
   return {
      apply_conformal_relaxed_update(vector_field, candidate, relaxation),
      linear.residual,
   };
}

ConformalMomentumResult solve_conformal_momentum_iteration(
   InitialDataEllipticHierarchy const& hierarchy,
   amrex::Vector<amrex::MultiFab const*> const& conformal_factor,
   amrex::Vector<amrex::MultiFab*> const& vector_potential,
   amrex::Vector<amrex::MultiFab const*> const& mean_curvature,
   amrex::Vector<amrex::MultiFab const*> const& momentum_density,
   InitialDataEllipticBoundaryConditions const& vector_potential_boundary,
   InitialDataEllipticOptions const& linear_options,
   amrex::Real relaxation
)
{
   const bool variable_mean_curvature = !mean_curvature.empty();
   return solve_conformal_vector_fixed_point(
      hierarchy,
      vector_potential,
      vector_potential_boundary,
      linear_options,
      relaxation,
      [&](int level, amrex::MultiFab& source) {
         const auto spacing = hierarchy.geometry[level].CellSizeArray();
         const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> inv_spacing{
            AMREX_D_DECL(
               1.0_rt / spacing[0],
               1.0_rt / spacing[1],
               1.0_rt / spacing[2]
            )
         };
         const auto physical_axes = hierarchy.cartesian_physical_axes;
         const amrex::Box derivative_domain = cartesian_first_derivative_domain(
            hierarchy, level, *conformal_factor[level]
         );
         const auto derivative_lo = amrex::lbound(derivative_domain);
         const auto derivative_hi = amrex::ubound(derivative_domain);
         for (amrex::MFIter mfi(source, amrex::TilingIfNotGPU());
              mfi.isValid(); ++mfi) {
            const amrex::Box box = mfi.tilebox();
            const auto psi = conformal_factor[level]->const_array(mfi);
            const auto momentum = momentum_density[level]->const_array(mfi);
            const auto curvature = variable_mean_curvature
               ? mean_curvature[level]->const_array(mfi)
               : amrex::Array4<amrex::Real const>{};
            const auto momentum_source = source.array(mfi);
            amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(
               int i, int j, int k
            ) noexcept {
               for (int component = 0; component < 3; ++component) {
                  amrex::Real curvature_gradient = 0.0_rt;
                  if (variable_mean_curvature) {
                     curvature_gradient =
                        reduced_cartesian_scalar_gradient_component(
                           curvature,
                           i,
                           j,
                           k,
                           component,
                           physical_axes,
                           inv_spacing,
                           derivative_lo,
                           derivative_hi
                        );
                  }
                  momentum_source(i, j, k, component) =
                     generated::conformal_momentum_source(
                        psi(i, j, k, 0),
                        momentum(i, j, k, component),
                        curvature_gradient
                     );
               }
            });
         }
      }
   );
}

struct CttkCurvatureUpdate {
   amrex::Real relative_update = 0.0_rt;
   amrex::Real minimum_squared = 0.0_rt;
};

enum class CttkHamiltonianPolicy {
   pure,
   hybrid,
};

void validate_cttk_configuration(
   InitialDataEllipticHierarchy const& hierarchy,
   amrex::Vector<amrex::MultiFab const*> const& conformal_factor,
   amrex::Vector<amrex::MultiFab*> const& mean_curvature,
   amrex::Vector<amrex::MultiFab*> const& vector_potential,
   amrex::Vector<amrex::MultiFab*> const& conformal_extrinsic_curvature,
   amrex::Vector<amrex::MultiFab const*> const& energy_density,
   amrex::Vector<amrex::MultiFab const*> const& momentum_density,
   CttkOptions const& options
)
{
   const int num_levels = static_cast<int>(hierarchy.geometry.size());
   const int branch_sign = static_cast<int>(options.mean_curvature_branch);
   if (num_levels == 0
       || static_cast<int>(conformal_factor.size()) != num_levels
       || static_cast<int>(mean_curvature.size()) != num_levels
       || static_cast<int>(vector_potential.size()) != num_levels
       || static_cast<int>(conformal_extrinsic_curvature.size()) != num_levels
       || static_cast<int>(energy_density.size()) != num_levels
       || static_cast<int>(momentum_density.size()) != num_levels
       || options.max_coupled_iterations <= 0
       || !(options.relative_update_tolerance > 0.0_rt)
       || !(options.relaxation > 0.0_rt && options.relaxation <= 1.0_rt)
       || (branch_sign != -1 && branch_sign != 1)) {
      amrex::Abort("invalid CTTK configuration");
   }
   for (int level = 0; level < num_levels; ++level) {
      if (conformal_factor[level] == nullptr
          || mean_curvature[level] == nullptr
          || vector_potential[level] == nullptr
          || conformal_extrinsic_curvature[level] == nullptr
          || energy_density[level] == nullptr
          || momentum_density[level] == nullptr
          || conformal_factor[level]->nComp() != 1
          || conformal_factor[level]->nGrow() < 1
          || mean_curvature[level]->nComp() != 1
          || mean_curvature[level]->nGrow() < 1
          || vector_potential[level]->nComp() != 3
          || vector_potential[level]->nGrow() < 1
          || conformal_extrinsic_curvature[level]->nComp()
             != CONFORMAL_A_NUM_COMPONENTS
          || energy_density[level]->nComp() != 1
          || momentum_density[level]->nComp() != 3
          || conformal_factor[level]->boxArray()
             != mean_curvature[level]->boxArray()
          || conformal_factor[level]->boxArray()
             != vector_potential[level]->boxArray()
          || conformal_factor[level]->boxArray()
             != conformal_extrinsic_curvature[level]->boxArray()
          || conformal_factor[level]->boxArray()
             != energy_density[level]->boxArray()
          || conformal_factor[level]->boxArray()
             != momentum_density[level]->boxArray()) {
         amrex::Abort("invalid CTTK field layout");
      }
   }
}

CttkCurvatureUpdate update_cttk_mean_curvature(
   InitialDataEllipticHierarchy const& hierarchy,
   amrex::Vector<amrex::MultiFab const*> const& conformal_factor,
   amrex::Vector<amrex::MultiFab*> const& mean_curvature,
   amrex::Vector<amrex::MultiFab const*> const& energy_density,
   amrex::Vector<amrex::MultiFab const*> const& conformal_extrinsic_curvature,
   amrex::Vector<amrex::MultiFab*> const& mean_curvature_squared,
   CttkHamiltonianPolicy policy,
   CttkMeanCurvatureBranch branch
)
{
   const int num_levels = static_cast<int>(hierarchy.geometry.size());
   const amrex::Real sign = static_cast<int>(branch);
   amrex::Real minimum_squared = std::numeric_limits<amrex::Real>::max();
   amrex::Real update_norm = 0.0_rt;
   amrex::Real solution_norm = 0.0_rt;
   for (int level = 0; level < num_levels; ++level) {
      amrex::MultiFab previous(
         mean_curvature[level]->boxArray(),
         mean_curvature[level]->DistributionMap(),
         1,
         0
      );
      amrex::MultiFab::Copy(previous, *mean_curvature[level], 0, 0, 1, 0);
      const auto spacing = hierarchy.geometry[level].CellSizeArray();
      const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>
         inv_spacing_squared{
            AMREX_D_DECL(
               1.0_rt / (spacing[0] * spacing[0]),
               1.0_rt / (spacing[1] * spacing[1]),
               1.0_rt / (spacing[2] * spacing[2])
            )
         };
      for (amrex::MFIter mfi(
              *mean_curvature_squared[level], amrex::TilingIfNotGPU()
           );
           mfi.isValid(); ++mfi) {
         const amrex::Box box = mfi.tilebox();
         const auto psi = conformal_factor[level]->const_array(mfi);
         const auto energy = energy_density[level]->const_array(mfi);
         const auto extrinsic =
            conformal_extrinsic_curvature[level]->const_array(mfi);
         const auto curvature_squared = mean_curvature_squared[level]->array(mfi);
         amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            if (policy == CttkHamiltonianPolicy::hybrid) {
               curvature_squared(i, j, k, 0) =
                  generated::cttk_hybrid_mean_curvature_squared(
                     energy(i, j, k, 0)
                  );
            } else {
               const amrex::Real laplacian =
                  reduced_cartesian_scalar_laplacian(
                     psi, i, j, k, inv_spacing_squared
                  );
               curvature_squared(i, j, k, 0) =
                  generated::cttk_mean_curvature_squared(
                     psi(i, j, k, 0),
                     laplacian,
                     energy(i, j, k, 0),
                     extrinsic(i, j, k, CONFORMAL_A_SQUARED)
                  );
            }
         });
      }
      minimum_squared = std::min(
         minimum_squared, mean_curvature_squared[level]->min(0, 0, false)
      );
      if (minimum_squared < 0.0_rt) {
         amrex::Abort(
            "CTTK Hamiltonian constraint produced negative mean-curvature squared"
         );
      }
      for (amrex::MFIter mfi(
              *mean_curvature[level], amrex::TilingIfNotGPU()
           ); mfi.isValid(); ++mfi) {
         const amrex::Box box = mfi.tilebox();
         const auto curvature_squared =
            mean_curvature_squared[level]->const_array(mfi);
         const auto curvature = mean_curvature[level]->array(mfi);
         amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            curvature(i, j, k, 0) = sign * std::sqrt(
               curvature_squared(i, j, k, 0)
            );
         });
      }
      amrex::MultiFab::Subtract(previous, *mean_curvature[level], 0, 0, 1, 0);
      update_norm = std::max(update_norm, previous.norm0(0, 0, false));
      solution_norm = std::max(
         solution_norm, mean_curvature[level]->norm0(0, 0, false)
      );
   }
   fill_initial_data_ghosts(
      hierarchy,
      mean_curvature,
      initial_data_boundary_records(
         hierarchy,
         {GridFieldSymmetryKind::Scalar},
         amrex::BCType::reflect_odd
      )
   );
   return {
      update_norm / std::max(amrex::Real(1.0_rt), solution_norm),
      minimum_squared,
   };
}

void validate_xcts_configuration(
   InitialDataEllipticHierarchy const& hierarchy,
   amrex::Vector<amrex::MultiFab*> const& conformal_factor,
   amrex::Vector<amrex::MultiFab*> const& lapse_times_conformal_factor,
   amrex::Vector<amrex::MultiFab*> const& shift,
   amrex::Vector<amrex::MultiFab*> const& conformal_extrinsic_curvature,
   amrex::Vector<amrex::MultiFab const*> const& energy_density,
   amrex::Vector<amrex::MultiFab const*> const& momentum_density,
   amrex::Vector<amrex::MultiFab const*> const& stress_trace,
   XctsOptions const& options
)
{
   const int num_levels = static_cast<int>(hierarchy.geometry.size());
   if (num_levels == 0
       || static_cast<int>(conformal_factor.size()) != num_levels
       || static_cast<int>(lapse_times_conformal_factor.size()) != num_levels
       || static_cast<int>(shift.size()) != num_levels
       || static_cast<int>(conformal_extrinsic_curvature.size()) != num_levels
       || static_cast<int>(energy_density.size()) != num_levels
       || static_cast<int>(momentum_density.size()) != num_levels
       || static_cast<int>(stress_trace.size()) != num_levels
       || options.max_coupled_iterations <= 0
       || !(options.relative_update_tolerance > 0.0_rt)
       || !(options.relaxation > 0.0_rt && options.relaxation <= 1.0_rt)) {
      amrex::Abort("invalid XCTS configuration");
   }
   for (int level = 0; level < num_levels; ++level) {
      if (conformal_factor[level] == nullptr
          || lapse_times_conformal_factor[level] == nullptr
          || shift[level] == nullptr
          || conformal_extrinsic_curvature[level] == nullptr
          || energy_density[level] == nullptr
          || momentum_density[level] == nullptr
          || stress_trace[level] == nullptr) {
         amrex::Abort("XCTS solve received null level data");
      }
      const amrex::BoxArray& grids = conformal_factor[level]->boxArray();
      if (conformal_factor[level]->nComp() != 1
          || conformal_factor[level]->nGrow() < 1
          || lapse_times_conformal_factor[level]->nComp() != 1
          || lapse_times_conformal_factor[level]->nGrow() < 1
          || shift[level]->nComp() != 3
          || shift[level]->nGrow() < 1
          || conformal_extrinsic_curvature[level]->nComp()
             != CONFORMAL_A_NUM_COMPONENTS
          || energy_density[level]->nComp() != 1
          || momentum_density[level]->nComp() != 3
          || stress_trace[level]->nComp() != 1
          || lapse_times_conformal_factor[level]->boxArray() != grids
          || shift[level]->boxArray() != grids
          || conformal_extrinsic_curvature[level]->boxArray() != grids
          || energy_density[level]->boxArray() != grids
          || momentum_density[level]->boxArray() != grids
          || stress_trace[level]->boxArray() != grids) {
         amrex::Abort("invalid XCTS field layout");
      }
   }
}

CttHamiltonianResult solve_xcts_lapse_iteration(
   InitialDataEllipticHierarchy const& hierarchy,
   amrex::Vector<amrex::MultiFab const*> const& conformal_factor,
   amrex::Vector<amrex::MultiFab*> const& lapse_times_conformal_factor,
   amrex::Vector<amrex::MultiFab const*> const& conformal_extrinsic_curvature,
   amrex::Vector<amrex::MultiFab const*> const& energy_density,
   amrex::Vector<amrex::MultiFab const*> const& stress_trace,
   InitialDataEllipticBoundaryConditions const& lapse_correction_boundary,
   CttHamiltonianOptions const& options
)
{
   return solve_conformal_scalar_fixed_point(
      hierarchy,
      lapse_times_conformal_factor,
      lapse_correction_boundary,
      options,
      ConformalScalarOperator::cartesian,
      [&](int level, amrex::MultiFab& source) {
         for (amrex::MFIter mfi(source, amrex::TilingIfNotGPU());
              mfi.isValid(); ++mfi) {
            const amrex::Box box = mfi.tilebox();
            const auto psi = conformal_factor[level]->const_array(mfi);
            const auto lapse_psi =
               lapse_times_conformal_factor[level]->const_array(mfi);
            const auto extrinsic =
               conformal_extrinsic_curvature[level]->const_array(mfi);
            const auto energy = energy_density[level]->const_array(mfi);
            const auto stress = stress_trace[level]->const_array(mfi);
            const auto rhs = source.array(mfi);
            amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(
               int i, int j, int k
            ) noexcept {
               rhs(i, j, k, 0) = generated::xcts_lapse_conformal_source(
                  psi(i, j, k, 0),
                  lapse_psi(i, j, k, 0),
                  energy(i, j, k, 0),
                  stress(i, j, k, 0),
                  extrinsic(i, j, k, CONFORMAL_A_SQUARED)
               );
            });
         }
      }
   );
}

ConformalMomentumResult solve_xcts_shift_iteration(
   InitialDataEllipticHierarchy const& hierarchy,
   amrex::Vector<amrex::MultiFab const*> const& conformal_factor,
   amrex::Vector<amrex::MultiFab const*> const& lapse_times_conformal_factor,
   amrex::Vector<amrex::MultiFab*> const& shift,
   amrex::Vector<amrex::MultiFab const*> const& conformal_extrinsic_curvature,
   amrex::Vector<amrex::MultiFab const*> const& momentum_density,
   InitialDataEllipticBoundaryConditions const& shift_boundary,
   InitialDataEllipticOptions const& linear_options,
   amrex::Real relaxation
)
{
   return solve_conformal_vector_fixed_point(
      hierarchy,
      shift,
      shift_boundary,
      linear_options,
      relaxation,
      [&](int level, amrex::MultiFab& source) {
         const auto spacing = hierarchy.geometry[level].CellSizeArray();
         const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> inv_spacing{
            AMREX_D_DECL(
               1.0_rt / spacing[0],
               1.0_rt / spacing[1],
               1.0_rt / spacing[2]
            )
         };
         const auto physical_axes = hierarchy.cartesian_physical_axes;
         const amrex::Box derivative_domain = cartesian_first_derivative_domain(
            hierarchy, level, *conformal_factor[level]
         );
         const auto derivative_lo = amrex::lbound(derivative_domain);
         const auto derivative_hi = amrex::ubound(derivative_domain);
         for (amrex::MFIter mfi(source, amrex::TilingIfNotGPU());
              mfi.isValid(); ++mfi) {
            const amrex::Box box = mfi.tilebox();
            const auto psi = conformal_factor[level]->const_array(mfi);
            const auto lapse_psi =
               lapse_times_conformal_factor[level]->const_array(mfi);
            const auto momentum = momentum_density[level]->const_array(mfi);
            const auto extrinsic =
               conformal_extrinsic_curvature[level]->const_array(mfi);
            const auto rhs = source.array(mfi);
            amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(
               int i, int j, int k
            ) noexcept {
               const amrex::Real d_psi[3] = {
                  reduced_cartesian_scalar_gradient_component(
                     psi, i, j, k, 0, physical_axes, inv_spacing,
                     derivative_lo, derivative_hi
                  ),
                  reduced_cartesian_scalar_gradient_component(
                     psi, i, j, k, 1, physical_axes, inv_spacing,
                     derivative_lo, derivative_hi
                  ),
                  reduced_cartesian_scalar_gradient_component(
                     psi, i, j, k, 2, physical_axes, inv_spacing,
                     derivative_lo, derivative_hi
                  ),
               };
               const amrex::Real d_lapse_psi[3] = {
                  reduced_cartesian_scalar_gradient_component(
                     lapse_psi, i, j, k, 0, physical_axes, inv_spacing,
                     derivative_lo, derivative_hi
                  ),
                  reduced_cartesian_scalar_gradient_component(
                     lapse_psi, i, j, k, 1, physical_axes, inv_spacing,
                     derivative_lo, derivative_hi
                  ),
                  reduced_cartesian_scalar_gradient_component(
                     lapse_psi, i, j, k, 2, physical_axes, inv_spacing,
                     derivative_lo, derivative_hi
                  ),
               };
               for (int component = 0; component < 3; ++component) {
                  const int a_i1 = component == 0
                     ? CONFORMAL_A11
                     : (component == 1 ? CONFORMAL_A12 : CONFORMAL_A13);
                  const int a_i2 = component == 0
                     ? CONFORMAL_A12
                     : (component == 1 ? CONFORMAL_A22 : CONFORMAL_A23);
                  const int a_i3 = component == 0
                     ? CONFORMAL_A13
                     : (component == 1 ? CONFORMAL_A23 : CONFORMAL_A33);
                  rhs(i, j, k, component) = generated::xcts_shift_source(
                     psi(i, j, k, 0),
                     lapse_psi(i, j, k, 0),
                     momentum(i, j, k, component),
                     d_psi[0],
                     d_psi[1],
                     d_psi[2],
                     d_lapse_psi[0],
                     d_lapse_psi[1],
                     d_lapse_psi[2],
                     extrinsic(i, j, k, a_i1),
                     extrinsic(i, j, k, a_i2),
                     extrinsic(i, j, k, a_i3)
                  );
               }
            });
         }
      }
   );
}
#if (AMREX_SPACEDIM >= 2)
void validate_axisymmetric_xcts_configuration(
   InitialDataEllipticHierarchy const& hierarchy,
   amrex::Vector<amrex::MultiFab*> const& conformal_factor,
   amrex::Vector<amrex::MultiFab*> const& lapse_times_conformal_factor,
   amrex::Vector<amrex::MultiFab*> const& azimuthal_shift,
   amrex::Vector<amrex::MultiFab*> const& conformal_extrinsic_curvature,
   amrex::Vector<amrex::MultiFab const*> const& energy_density,
   amrex::Vector<amrex::MultiFab const*> const& azimuthal_momentum_density,
   amrex::Vector<amrex::MultiFab const*> const& stress_trace,
   XctsOptions const& options
)
{
   const int num_levels = static_cast<int>(hierarchy.geometry.size());
   if (num_levels != 1
       || static_cast<int>(conformal_factor.size()) != num_levels
       || static_cast<int>(lapse_times_conformal_factor.size()) != num_levels
       || static_cast<int>(azimuthal_shift.size()) != num_levels
       || static_cast<int>(conformal_extrinsic_curvature.size()) != num_levels
       || static_cast<int>(energy_density.size()) != num_levels
       || static_cast<int>(azimuthal_momentum_density.size()) != num_levels
       || static_cast<int>(stress_trace.size()) != num_levels
       || options.max_coupled_iterations <= 0
       || !(options.relative_update_tolerance > 0.0_rt)
       || !(options.relaxation > 0.0_rt && options.relaxation <= 1.0_rt)) {
      amrex::Abort("invalid axisymmetric spherical XCTS configuration");
   }
   const amrex::Box domain = hierarchy.geometry[0].Domain();
   if (domain.length(0) < 3 || domain.length(1) < 2) {
      amrex::Abort("axisymmetric spherical XCTS grid is too small");
   }
#if (AMREX_SPACEDIM == 3)
   const auto problo = hierarchy.geometry[0].ProbLoArray();
   const auto probhi = hierarchy.geometry[0].ProbHiArray();
   const amrex::Real phi_extent = probhi[2] - problo[2];
   if (!hierarchy.geometry[0].isPeriodic(2)
       || std::abs(phi_extent - 2.0_rt * constants::pi)
          > 64.0_rt * std::numeric_limits<amrex::Real>::epsilon()
             * std::max(amrex::Real(1.0_rt), std::abs(phi_extent))) {
      amrex::Abort(
         "3D axisymmetric spherical XCTS requires a periodic full-2pi phi axis"
      );
   }
#endif
   const amrex::BoxArray& grids = hierarchy.grids[0];
   if (grids.ixType() != amrex::IndexType::TheCellType()) {
      amrex::Abort("axisymmetric spherical XCTS fields must be cell-centered");
   }
   auto valid_scalar = [&](amrex::MultiFab const* field, int grow) {
      return field != nullptr && field->nComp() == 1
         && field->nGrow() >= grow && field->boxArray() == grids;
   };
   if (!valid_scalar(conformal_factor[0], 1)
       || !valid_scalar(lapse_times_conformal_factor[0], 1)
       || !valid_scalar(azimuthal_shift[0], 1)
       || conformal_extrinsic_curvature[0] == nullptr
       || conformal_extrinsic_curvature[0]->nComp()
          != CONFORMAL_A_NUM_COMPONENTS
       || conformal_extrinsic_curvature[0]->boxArray() != grids
       || !valid_scalar(energy_density[0], 0)
       || !valid_scalar(azimuthal_momentum_density[0], 0)
       || !valid_scalar(stress_trace[0], 0)) {
      amrex::Abort("invalid axisymmetric spherical XCTS field layout");
   }
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
amrex::Real axisymmetric_cell_derivative(
   amrex::Array4<amrex::Real const> const& field,
   int i,
   int j,
   int k,
   int component,
   int direction,
   amrex::Dim3 domain_lo,
   amrex::Dim3 domain_hi,
   amrex::Real inverse_angular_spacing,
   amrex::Real radial_lower_face,
   amrex::Real const* radial_centers
) noexcept
{
   if (direction == 0) {
      int lower_i = i - 1;
      int upper_i = i + 1;
      const bool radial_origin = radial_lower_face == 0.0_rt;
      if (i == domain_lo.x && !radial_origin) {
         lower_i = i + 1;
         upper_i = i + 2;
      } else if (i == domain_hi.x) {
         lower_i = i - 2;
         upper_i = i - 1;
      }
      const amrex::Real center = radial_centers[i - domain_lo.x];
      const amrex::Real lower_coordinate = lower_i < domain_lo.x
         ? 2.0_rt * radial_lower_face
            - radial_centers[domain_lo.x - lower_i - 1]
         : radial_centers[lower_i - domain_lo.x];
      const amrex::Real upper_coordinate =
         radial_centers[upper_i - domain_lo.x];
      const amrex::Real lower_value = field(
         lower_i, j, k, component
      );
      const amrex::Real center_value = field(i, j, k, component);
      const amrex::Real upper_value = field(
         upper_i, j, k, component
      );
      return lower_value * (center - upper_coordinate) / (
               (lower_coordinate - center)
               * (lower_coordinate - upper_coordinate)
            )
         + center_value * (
               2.0_rt * center - lower_coordinate - upper_coordinate
            ) / (
               (center - lower_coordinate)
               * (center - upper_coordinate)
            )
         + upper_value * (center - lower_coordinate) / (
               (upper_coordinate - lower_coordinate)
               * (upper_coordinate - center)
            );
   }
   return 0.5_rt * inverse_angular_spacing * (
      field(i, j + 1, k, component)
      - field(i, j - 1, k, component)
   );
}

void fill_axisymmetric_xcts_extrinsic_curvature(
   InitialDataEllipticHierarchy const& hierarchy,
   amrex::Vector<amrex::MultiFab*> const& azimuthal_shift,
   amrex::Vector<amrex::MultiFab const*> const& conformal_factor,
   amrex::Vector<amrex::MultiFab const*> const& lapse_times_conformal_factor,
   amrex::Vector<amrex::MultiFab*> const& conformal_extrinsic_curvature,
   InitialDataEllipticBoundaryConditions const& azimuthal_shift_boundary
)
{
   fill_initial_data_ghosts(
      hierarchy,
      azimuthal_shift,
      initial_data_boundary_records(hierarchy, azimuthal_shift_boundary)
   );
   for (int level = 0;
        level < static_cast<int>(hierarchy.geometry.size());
        ++level) {
      const auto problo = hierarchy.geometry[level].ProbLoArray();
      const auto spacing = hierarchy.geometry[level].CellSizeArray();
      const amrex::Dim3 domain_lo = amrex::lbound(
         hierarchy.geometry[level].Domain()
      );
      const amrex::Dim3 domain_hi = amrex::ubound(
         hierarchy.geometry[level].Domain()
      );
      const amrex::Real inverse_dtheta = 1.0_rt / spacing[1];
      const InitialDataCoordinateLine radial_coordinates =
         hierarchy.coordinate_line(level, 0);
      const amrex::Real radial_lower_face = radial_coordinates.faces.front();
      amrex::Gpu::AsyncArray<amrex::Real> radial_center_storage(
         radial_coordinates.centers.data(), radial_coordinates.centers.size()
      );
      const amrex::Real* radial_center = radial_center_storage.data();
      for (amrex::MFIter mfi(
              *conformal_extrinsic_curvature[level],
              amrex::TilingIfNotGPU()
           ); mfi.isValid(); ++mfi) {
         const amrex::Box box = mfi.tilebox();
         const auto beta = azimuthal_shift[level]->const_array(mfi);
         const auto psi = conformal_factor[level]->const_array(mfi);
         const auto lapse_psi =
            lapse_times_conformal_factor[level]->const_array(mfi);
         const auto extrinsic =
            conformal_extrinsic_curvature[level]->array(mfi);
         amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(
            int i, int j, int k
         ) noexcept {
            const amrex::Real radius = radial_center[i - domain_lo.x];
            const amrex::Real theta = problo[1]
               + (j - domain_lo.y + 0.5_rt) * spacing[1];
            const amrex::Real value = beta(i, j, k, 0);
            const amrex::Real radial_derivative =
               axisymmetric_cell_derivative(
                  beta, i, j, k, 0, 0, domain_lo, domain_hi,
                  inverse_dtheta, radial_lower_face, radial_center
               );
            const amrex::Real angular_derivative =
               axisymmetric_cell_derivative(
                  beta, i, j, k, 0, 1, domain_lo, domain_hi,
                  inverse_dtheta, radial_lower_face, radial_center
               );
            const amrex::Real longitudinal_r_phi =
               radial_derivative - value / radius;
            const amrex::Real longitudinal_theta_phi = (
               angular_derivative - value / std::tan(theta)
            ) / radius;
            const amrex::Real scale =
               generated::xcts_conformal_extrinsic_scale(
                  psi(i, j, k, 0), lapse_psi(i, j, k, 0)
               );
            const amrex::Real a_r_phi = scale * longitudinal_r_phi;
            const amrex::Real a_theta_phi =
               scale * longitudinal_theta_phi;
            extrinsic(i, j, k, CONFORMAL_A11) = 0.0_rt;
            extrinsic(i, j, k, CONFORMAL_A12) = 0.0_rt;
            extrinsic(i, j, k, CONFORMAL_A13) = a_r_phi;
            extrinsic(i, j, k, CONFORMAL_A22) = 0.0_rt;
            extrinsic(i, j, k, CONFORMAL_A23) = a_theta_phi;
            extrinsic(i, j, k, CONFORMAL_A33) = 0.0_rt;
            extrinsic(i, j, k, CONFORMAL_A_SQUARED) = 2.0_rt * (
               a_r_phi * a_r_phi + a_theta_phi * a_theta_phi
            );
         });
      }
   }
}

CttHamiltonianResult solve_axisymmetric_xcts_lapse_iteration(
   InitialDataEllipticHierarchy const& hierarchy,
   amrex::Vector<amrex::MultiFab const*> const& conformal_factor,
   amrex::Vector<amrex::MultiFab*> const& lapse_times_conformal_factor,
   amrex::Vector<amrex::MultiFab const*> const& conformal_extrinsic_curvature,
   amrex::Vector<amrex::MultiFab const*> const& energy_density,
   amrex::Vector<amrex::MultiFab const*> const& stress_trace,
   InitialDataEllipticBoundaryConditions const& lapse_correction_boundary,
   CttHamiltonianOptions const& options
)
{
   return solve_conformal_scalar_fixed_point(
      hierarchy,
      lapse_times_conformal_factor,
      lapse_correction_boundary,
      options,
#if (AMREX_SPACEDIM == 2)
      ConformalScalarOperator::axisymmetric_spherical,
#else
      ConformalScalarOperator::spherical_axisymmetric_storage,
#endif
      [&](int level, amrex::MultiFab& source) {
         for (amrex::MFIter mfi(source, amrex::TilingIfNotGPU());
              mfi.isValid(); ++mfi) {
            const amrex::Box box = mfi.tilebox();
            const auto psi = conformal_factor[level]->const_array(mfi);
            const auto lapse_psi =
               lapse_times_conformal_factor[level]->const_array(mfi);
            const auto extrinsic =
               conformal_extrinsic_curvature[level]->const_array(mfi);
            const auto energy = energy_density[level]->const_array(mfi);
            const auto stress = stress_trace[level]->const_array(mfi);
            const auto rhs = source.array(mfi);
            amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(
               int i, int j, int k
            ) noexcept {
               rhs(i, j, k, 0) = generated::xcts_lapse_conformal_source(
                  psi(i, j, k, 0),
                  lapse_psi(i, j, k, 0),
                  energy(i, j, k, 0),
                  stress(i, j, k, 0),
                  extrinsic(i, j, k, CONFORMAL_A_SQUARED)
               );
            });
         }
      }
   );
}

ConformalMomentumResult solve_axisymmetric_xcts_shift_iteration(
   InitialDataEllipticHierarchy const& hierarchy,
   amrex::Vector<amrex::MultiFab*> const& conformal_factor,
   amrex::Vector<amrex::MultiFab*> const& lapse_times_conformal_factor,
   amrex::Vector<amrex::MultiFab*> const& azimuthal_shift,
   amrex::Vector<amrex::MultiFab const*> const& conformal_extrinsic_curvature,
   amrex::Vector<amrex::MultiFab const*> const& azimuthal_momentum_density,
   InitialDataEllipticBoundaryConditions const& conformal_boundary,
   InitialDataEllipticBoundaryConditions const& lapse_boundary,
   InitialDataEllipticBoundaryConditions const& shift_boundary,
   InitialDataEllipticOptions const& linear_options,
   amrex::Real relaxation
)
{
   fill_initial_data_ghosts(
      hierarchy,
      conformal_factor,
      initial_data_boundary_records(hierarchy, conformal_boundary)
   );
   fill_initial_data_ghosts(
      hierarchy,
      lapse_times_conformal_factor,
      initial_data_boundary_records(hierarchy, lapse_boundary)
   );
   amrex::Vector<std::unique_ptr<amrex::MultiFab>> source_storage(1);
   amrex::Vector<std::unique_ptr<amrex::MultiFab>> candidate_storage(1);
   source_storage[0] = std::make_unique<amrex::MultiFab>(
      azimuthal_shift[0]->boxArray(),
      azimuthal_shift[0]->DistributionMap(),
      1,
      0
   );
   candidate_storage[0] = std::make_unique<amrex::MultiFab>(
      azimuthal_shift[0]->boxArray(),
      azimuthal_shift[0]->DistributionMap(),
      1,
      1
   );
   const auto problo = hierarchy.geometry[0].ProbLoArray();
   const auto spacing = hierarchy.geometry[0].CellSizeArray();
   const amrex::Dim3 domain_lo = amrex::lbound(
      hierarchy.geometry[0].Domain()
   );
   const amrex::Dim3 domain_hi = amrex::ubound(
      hierarchy.geometry[0].Domain()
   );
   const amrex::Real inverse_dtheta = 1.0_rt / spacing[1];
   const InitialDataCoordinateLine radial_coordinates =
      hierarchy.coordinate_line(0, 0);
   const amrex::Real radial_lower_face = radial_coordinates.faces.front();
   amrex::Gpu::AsyncArray<amrex::Real> radial_center_storage(
      radial_coordinates.centers.data(), radial_coordinates.centers.size()
   );
   const amrex::Real* radial_center = radial_center_storage.data();
   for (amrex::MFIter mfi(
           *source_storage[0], amrex::TilingIfNotGPU()
        ); mfi.isValid(); ++mfi) {
      const amrex::Box box = mfi.tilebox();
      const auto psi = conformal_factor[0]->const_array(mfi);
      const auto lapse_psi =
         lapse_times_conformal_factor[0]->const_array(mfi);
      const auto extrinsic =
         conformal_extrinsic_curvature[0]->const_array(mfi);
      const auto momentum = azimuthal_momentum_density[0]->const_array(mfi);
      const auto source = source_storage[0]->array(mfi);
      amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(
         int i, int j, int k
      ) noexcept {
         const amrex::Real radius = radial_center[i - domain_lo.x];
         const amrex::Real d_psi_r = axisymmetric_cell_derivative(
            psi, i, j, k, 0, 0, domain_lo, domain_hi,
            inverse_dtheta, radial_lower_face, radial_center
         );
         const amrex::Real d_psi_theta = axisymmetric_cell_derivative(
            psi, i, j, k, 0, 1, domain_lo, domain_hi,
            inverse_dtheta, radial_lower_face, radial_center
         ) / radius;
         const amrex::Real d_lapse_psi_r = axisymmetric_cell_derivative(
            lapse_psi, i, j, k, 0, 0, domain_lo, domain_hi,
            inverse_dtheta, radial_lower_face, radial_center
         );
         const amrex::Real d_lapse_psi_theta = axisymmetric_cell_derivative(
            lapse_psi, i, j, k, 0, 1, domain_lo, domain_hi,
            inverse_dtheta, radial_lower_face, radial_center
         ) / radius;
         source(i, j, k, 0) = generated::xcts_shift_source(
            psi(i, j, k, 0),
            lapse_psi(i, j, k, 0),
            momentum(i, j, k, 0),
            d_psi_r,
            d_psi_theta,
            0.0_rt,
            d_lapse_psi_r,
            d_lapse_psi_theta,
            0.0_rt,
            extrinsic(i, j, k, CONFORMAL_A13),
            extrinsic(i, j, k, CONFORMAL_A23),
            0.0_rt
         );
      });
   }
   amrex::MultiFab::Copy(
      *candidate_storage[0], *azimuthal_shift[0], 0, 0, 1, 1
   );
   amrex::Vector<amrex::MultiFab*> candidate{candidate_storage[0].get()};
   amrex::Vector<amrex::MultiFab const*> source{source_storage[0].get()};
   const InitialDataEllipticResult linear =
#if (AMREX_SPACEDIM == 2)
      solve_initial_data_axisymmetric_spherical_toroidal_vector_poisson(
         hierarchy, candidate, source, shift_boundary, linear_options
      );
#else
      solve_initial_data_spherical_axisymmetric_toroidal_vector_poisson(
         hierarchy, candidate, source, shift_boundary, linear_options
      );
#endif
   return {
      apply_conformal_relaxed_update(azimuthal_shift, candidate, relaxation),
      linear.residual,
   };
}
#endif

} // namespace

CttHamiltonianResult solve_ctt_time_symmetric_hamiltonian(
   InitialDataEllipticHierarchy const& hierarchy,
   amrex::Vector<amrex::MultiFab*> const& conformal_factor,
   amrex::Vector<amrex::MultiFab const*> const& energy_density,
   InitialDataEllipticBoundaryConditions const& conformal_correction_boundary,
   CttHamiltonianOptions const& options
)
{
   return solve_ctt_hamiltonian(
      hierarchy,
      conformal_factor,
      energy_density,
      {},
      {},
      conformal_correction_boundary,
      options,
      ConformalScalarOperator::cartesian
   );
}

CttResult solve_ctt_constraints(
   InitialDataEllipticHierarchy const& hierarchy,
   amrex::Vector<amrex::MultiFab*> const& conformal_factor,
   amrex::Vector<amrex::MultiFab*> const& vector_potential,
   amrex::Vector<amrex::MultiFab*> const& conformal_extrinsic_curvature,
   amrex::Vector<amrex::MultiFab const*> const& energy_density,
   amrex::Vector<amrex::MultiFab const*> const& momentum_density,
   InitialDataEllipticBoundaryConditions const& conformal_correction_boundary,
   InitialDataEllipticBoundaryConditions const& vector_potential_boundary,
   CttOptions const& options
)
{
   const int num_levels = static_cast<int>(hierarchy.geometry.size());
   if (
      num_levels == 0
      || static_cast<int>(conformal_factor.size()) != num_levels
      || static_cast<int>(vector_potential.size()) != num_levels
      || static_cast<int>(conformal_extrinsic_curvature.size()) != num_levels
      || static_cast<int>(energy_density.size()) != num_levels
      || static_cast<int>(momentum_density.size()) != num_levels
      || options.max_coupled_iterations <= 0
      || !(options.relative_update_tolerance > 0.0_rt)
      || !(options.relaxation > 0.0_rt && options.relaxation <= 1.0_rt)
   ) {
      amrex::Abort("invalid coupled CTT configuration");
   }

   amrex::Vector<amrex::MultiFab const*> conformal_factor_const(num_levels);
   amrex::Vector<amrex::MultiFab const*> extrinsic_const(num_levels);
   for (int level = 0; level < num_levels; ++level) {
      if (conformal_factor[level] == nullptr
          || vector_potential[level] == nullptr
          || conformal_extrinsic_curvature[level] == nullptr
          || energy_density[level] == nullptr
          || momentum_density[level] == nullptr
          || conformal_factor[level]->nComp() != 1
          || vector_potential[level]->nComp() != 3
          || vector_potential[level]->nGrow() < 1
          || conformal_factor[level]->boxArray()
             != vector_potential[level]->boxArray()
          || conformal_extrinsic_curvature[level]->nComp()
             != CONFORMAL_A_NUM_COMPONENTS
          || conformal_extrinsic_curvature[level]->boxArray()
             != vector_potential[level]->boxArray()
          || energy_density[level]->nComp() != 1
          || energy_density[level]->boxArray()
             != vector_potential[level]->boxArray()
          || momentum_density[level]->nComp() != 3
          || momentum_density[level]->boxArray()
             != vector_potential[level]->boxArray()) {
         amrex::Abort("invalid coupled CTT field layout");
      }
      conformal_factor_const[level] = conformal_factor[level];
      extrinsic_const[level] = conformal_extrinsic_curvature[level];
   }

   CttResult result;
   CttHamiltonianOptions hamiltonian_options;
   hamiltonian_options.max_nonlinear_iterations = options.max_coupled_iterations;
   hamiltonian_options.relative_update_tolerance = options.relative_update_tolerance;
   hamiltonian_options.relaxation = options.relaxation;
   hamiltonian_options.linear = options.linear;
   for (int iteration = 1; iteration <= options.max_coupled_iterations; ++iteration) {
      const ConformalMomentumResult momentum =
         solve_conformal_momentum_iteration(
            hierarchy,
            conformal_factor_const,
            vector_potential,
            {},
            momentum_density,
            vector_potential_boundary,
            options.linear,
            options.relaxation
         );
      fill_conformal_extrinsic_curvature<false>(
         hierarchy, vector_potential, {}, {}, conformal_extrinsic_curvature
      );
      const CttHamiltonianResult hamiltonian = solve_ctt_hamiltonian(
         hierarchy,
         conformal_factor,
         energy_density,
         extrinsic_const,
         {},
         conformal_correction_boundary,
         hamiltonian_options,
         ConformalScalarOperator::cartesian
      );
      result.coupled_iterations = iteration;
      result.relative_update = std::max(
         momentum.relative_update, hamiltonian.relative_update
      );
      result.hamiltonian_residual = hamiltonian.linear_residual;
      result.momentum_residual = momentum.linear_residual;
      if (!std::isfinite(result.relative_update)) {
         amrex::Abort("coupled CTT solve produced a nonfinite update");
      }
      if (hamiltonian.converged
          && result.relative_update <= options.relative_update_tolerance) {
         result.converged = true;
         break;
      }
   }
   fill_conformal_extrinsic_curvature<false>(
      hierarchy, vector_potential, {}, {}, conformal_extrinsic_curvature
   );
   return result;
}

CttkResult solve_cttk_constraints(
   InitialDataEllipticHierarchy const& hierarchy,
   amrex::Vector<amrex::MultiFab const*> const& conformal_factor,
   amrex::Vector<amrex::MultiFab*> const& mean_curvature,
   amrex::Vector<amrex::MultiFab*> const& vector_potential,
   amrex::Vector<amrex::MultiFab*> const& conformal_extrinsic_curvature,
   amrex::Vector<amrex::MultiFab const*> const& energy_density,
   amrex::Vector<amrex::MultiFab const*> const& momentum_density,
   InitialDataEllipticBoundaryConditions const& vector_potential_boundary,
   CttkOptions const& options
)
{
   validate_cttk_configuration(
      hierarchy,
      conformal_factor,
      mean_curvature,
      vector_potential,
      conformal_extrinsic_curvature,
      energy_density,
      momentum_density,
      options
   );
   const int num_levels = static_cast<int>(hierarchy.geometry.size());

   amrex::Vector<std::unique_ptr<amrex::MultiFab>>
      mean_curvature_squared_storage(num_levels);
   amrex::Vector<amrex::MultiFab*> mean_curvature_squared(num_levels);
   amrex::Vector<amrex::MultiFab const*> mean_curvature_const(num_levels);
   amrex::Vector<amrex::MultiFab const*> extrinsic_const(num_levels);
   for (int level = 0; level < num_levels; ++level) {
      mean_curvature_const[level] = mean_curvature[level];
      extrinsic_const[level] = conformal_extrinsic_curvature[level];
      mean_curvature_squared_storage[level] = std::make_unique<amrex::MultiFab>(
         conformal_factor[level]->boxArray(),
         conformal_factor[level]->DistributionMap(),
         1,
         0
      );
      mean_curvature_squared[level] = mean_curvature_squared_storage[level].get();
   }

   CttkResult result;
   for (int iteration = 1; iteration <= options.max_coupled_iterations; ++iteration) {
      fill_conformal_extrinsic_curvature<false>(
         hierarchy, vector_potential, {}, {}, conformal_extrinsic_curvature
      );
      const CttkCurvatureUpdate curvature = update_cttk_mean_curvature(
         hierarchy,
         conformal_factor,
         mean_curvature,
         energy_density,
         extrinsic_const,
         mean_curvature_squared,
         CttkHamiltonianPolicy::pure,
         options.mean_curvature_branch
      );
      const ConformalMomentumResult momentum =
         solve_conformal_momentum_iteration(
            hierarchy,
            conformal_factor,
            vector_potential,
            mean_curvature_const,
            momentum_density,
            vector_potential_boundary,
            options.linear,
            options.relaxation
         );
      result.coupled_iterations = iteration;
      result.relative_update = std::max(
         curvature.relative_update, momentum.relative_update
      );
      result.momentum_residual = momentum.linear_residual;
      result.minimum_mean_curvature_squared = curvature.minimum_squared;
      if (!std::isfinite(result.relative_update)) {
         amrex::Abort("CTTK solve produced a nonfinite update");
      }
      if (result.relative_update <= options.relative_update_tolerance) {
         fill_conformal_extrinsic_curvature<false>(
            hierarchy, vector_potential, {}, {}, conformal_extrinsic_curvature
         );
         const CttkCurvatureUpdate final_curvature = update_cttk_mean_curvature(
            hierarchy,
            conformal_factor,
            mean_curvature,
            energy_density,
            extrinsic_const,
            mean_curvature_squared,
            CttkHamiltonianPolicy::pure,
            options.mean_curvature_branch
         );
         result.relative_update = std::max(
            result.relative_update, final_curvature.relative_update
         );
         result.minimum_mean_curvature_squared =
            final_curvature.minimum_squared;
         if (result.relative_update <= options.relative_update_tolerance) {
            result.converged = true;
            break;
         }
      }
   }
   return result;
}

CttkResult solve_cttk_hybrid_constraints(
   InitialDataEllipticHierarchy const& hierarchy,
   amrex::Vector<amrex::MultiFab*> const& conformal_factor,
   amrex::Vector<amrex::MultiFab*> const& mean_curvature,
   amrex::Vector<amrex::MultiFab*> const& vector_potential,
   amrex::Vector<amrex::MultiFab*> const& conformal_extrinsic_curvature,
   amrex::Vector<amrex::MultiFab const*> const& energy_density,
   amrex::Vector<amrex::MultiFab const*> const& momentum_density,
   InitialDataEllipticBoundaryConditions const& conformal_correction_boundary,
   InitialDataEllipticBoundaryConditions const& vector_potential_boundary,
   CttkOptions const& options
)
{
   const int num_levels = static_cast<int>(hierarchy.geometry.size());
   amrex::Vector<amrex::MultiFab const*> conformal_factor_const(num_levels);
   for (int level = 0; level < num_levels; ++level) {
      conformal_factor_const[level] = conformal_factor[level];
   }
   validate_cttk_configuration(
      hierarchy,
      conformal_factor_const,
      mean_curvature,
      vector_potential,
      conformal_extrinsic_curvature,
      energy_density,
      momentum_density,
      options
   );

   amrex::Vector<std::unique_ptr<amrex::MultiFab>>
      mean_curvature_squared_storage(num_levels);
   amrex::Vector<amrex::MultiFab*> mean_curvature_squared(num_levels);
   amrex::Vector<amrex::MultiFab const*> mean_curvature_squared_const(num_levels);
   amrex::Vector<amrex::MultiFab const*> mean_curvature_const(num_levels);
   amrex::Vector<amrex::MultiFab const*> extrinsic_const(num_levels);
   for (int level = 0; level < num_levels; ++level) {
      mean_curvature_squared_storage[level] = std::make_unique<amrex::MultiFab>(
         conformal_factor[level]->boxArray(),
         conformal_factor[level]->DistributionMap(),
         1,
         0
      );
      mean_curvature_squared[level] = mean_curvature_squared_storage[level].get();
      mean_curvature_squared_const[level] =
         mean_curvature_squared_storage[level].get();
      mean_curvature_const[level] = mean_curvature[level];
      extrinsic_const[level] = conformal_extrinsic_curvature[level];
   }

   CttHamiltonianOptions hamiltonian_options;
   hamiltonian_options.max_nonlinear_iterations = options.max_coupled_iterations;
   hamiltonian_options.relative_update_tolerance = options.relative_update_tolerance;
   hamiltonian_options.relaxation = options.relaxation;
   hamiltonian_options.linear = options.linear;

   CttkResult result;
   for (int iteration = 1; iteration <= options.max_coupled_iterations; ++iteration) {
      const CttkCurvatureUpdate curvature = update_cttk_mean_curvature(
         hierarchy,
         conformal_factor_const,
         mean_curvature,
         energy_density,
         extrinsic_const,
         mean_curvature_squared,
         CttkHamiltonianPolicy::hybrid,
         options.mean_curvature_branch
      );
      const ConformalMomentumResult momentum =
         solve_conformal_momentum_iteration(
            hierarchy,
            conformal_factor_const,
            vector_potential,
            mean_curvature_const,
            momentum_density,
            vector_potential_boundary,
            options.linear,
            options.relaxation
         );
      fill_conformal_extrinsic_curvature<false>(
         hierarchy, vector_potential, {}, {}, conformal_extrinsic_curvature
      );
      const CttHamiltonianResult hamiltonian = solve_ctt_hamiltonian(
         hierarchy,
         conformal_factor,
         energy_density,
         extrinsic_const,
         mean_curvature_squared_const,
         conformal_correction_boundary,
         hamiltonian_options,
         ConformalScalarOperator::cartesian
      );
      result.coupled_iterations = iteration;
      result.relative_update = std::max({
         curvature.relative_update,
         momentum.relative_update,
         hamiltonian.relative_update,
      });
      result.hamiltonian_residual = hamiltonian.linear_residual;
      result.momentum_residual = momentum.linear_residual;
      result.minimum_mean_curvature_squared = curvature.minimum_squared;
      if (!std::isfinite(result.relative_update)) {
         amrex::Abort("hybrid CTTK solve produced a nonfinite update");
      }
      if (hamiltonian.converged
          && result.relative_update <= options.relative_update_tolerance) {
         result.converged = true;
         break;
      }
   }
   fill_conformal_extrinsic_curvature<false>(
      hierarchy, vector_potential, {}, {}, conformal_extrinsic_curvature
   );
   update_cttk_mean_curvature(
      hierarchy,
      conformal_factor_const,
      mean_curvature,
      energy_density,
      extrinsic_const,
      mean_curvature_squared,
      CttkHamiltonianPolicy::hybrid,
      options.mean_curvature_branch
   );
   return result;
}

XctsResult solve_xcts_constraints(
   InitialDataEllipticHierarchy const& hierarchy,
   amrex::Vector<amrex::MultiFab*> const& conformal_factor,
   amrex::Vector<amrex::MultiFab*> const& lapse_times_conformal_factor,
   amrex::Vector<amrex::MultiFab*> const& shift,
   amrex::Vector<amrex::MultiFab*> const& conformal_extrinsic_curvature,
   amrex::Vector<amrex::MultiFab const*> const& energy_density,
   amrex::Vector<amrex::MultiFab const*> const& momentum_density,
   amrex::Vector<amrex::MultiFab const*> const& stress_trace,
   InitialDataEllipticBoundaryConditions const& conformal_correction_boundary,
   InitialDataEllipticBoundaryConditions const& lapse_correction_boundary,
   InitialDataEllipticBoundaryConditions const& shift_boundary,
   XctsOptions const& options
)
{
   validate_xcts_configuration(
      hierarchy,
      conformal_factor,
      lapse_times_conformal_factor,
      shift,
      conformal_extrinsic_curvature,
      energy_density,
      momentum_density,
      stress_trace,
      options
   );
   const int num_levels = static_cast<int>(hierarchy.geometry.size());
   amrex::Vector<amrex::MultiFab const*> conformal_const(num_levels);
   amrex::Vector<amrex::MultiFab const*> lapse_conformal_const(num_levels);
   amrex::Vector<amrex::MultiFab const*> extrinsic_const(num_levels);
   for (int level = 0; level < num_levels; ++level) {
      conformal_const[level] = conformal_factor[level];
      lapse_conformal_const[level] = lapse_times_conformal_factor[level];
      extrinsic_const[level] = conformal_extrinsic_curvature[level];
   }

   CttHamiltonianOptions scalar_options;
   scalar_options.max_nonlinear_iterations = options.max_coupled_iterations;
   scalar_options.relative_update_tolerance = options.relative_update_tolerance;
   scalar_options.relaxation = options.relaxation;
   scalar_options.linear = options.linear;

   XctsResult result;
   for (int iteration = 1; iteration <= options.max_coupled_iterations; ++iteration) {
      fill_conformal_extrinsic_curvature<true>(
         hierarchy,
         shift,
         conformal_const,
         lapse_conformal_const,
         conformal_extrinsic_curvature
      );
      const CttHamiltonianResult hamiltonian = solve_ctt_hamiltonian(
         hierarchy,
         conformal_factor,
         energy_density,
         extrinsic_const,
         {},
         conformal_correction_boundary,
         scalar_options,
         ConformalScalarOperator::cartesian
      );
      const CttHamiltonianResult lapse = solve_xcts_lapse_iteration(
         hierarchy,
         conformal_const,
         lapse_times_conformal_factor,
         extrinsic_const,
         energy_density,
         stress_trace,
         lapse_correction_boundary,
         scalar_options
      );
      for (int level = 0; level < num_levels; ++level) {
         if (!(conformal_factor[level]->min(0, 0, false) > 0.0_rt)
             || !(lapse_times_conformal_factor[level]->min(0, 0, false) > 0.0_rt)) {
            amrex::Abort("XCTS solve produced a nonpositive conformal unknown");
         }
      }
      fill_conformal_extrinsic_curvature<true>(
         hierarchy,
         shift,
         conformal_const,
         lapse_conformal_const,
         conformal_extrinsic_curvature
      );
      const ConformalMomentumResult momentum = solve_xcts_shift_iteration(
         hierarchy,
         conformal_const,
         lapse_conformal_const,
         shift,
         extrinsic_const,
         momentum_density,
         shift_boundary,
         options.linear,
         options.relaxation
      );
      result.coupled_iterations = iteration;
      result.relative_update = std::max({
         hamiltonian.relative_update,
         lapse.relative_update,
         momentum.relative_update,
      });
      result.hamiltonian_residual = hamiltonian.linear_residual;
      result.lapse_residual = lapse.linear_residual;
      result.momentum_residual = momentum.linear_residual;
      if (!std::isfinite(result.relative_update)) {
         amrex::Abort("XCTS solve produced a nonfinite update");
      }
      if (hamiltonian.converged
          && lapse.converged
          && result.relative_update <= options.relative_update_tolerance) {
         result.converged = true;
         break;
      }
   }
   fill_conformal_extrinsic_curvature<true>(
      hierarchy,
      shift,
      conformal_const,
      lapse_conformal_const,
      conformal_extrinsic_curvature
   );
   return result;
}

XctsResult solve_axisymmetric_spherical_xcts_constraints(
   InitialDataEllipticHierarchy const& hierarchy,
   amrex::Vector<amrex::MultiFab*> const& conformal_factor,
   amrex::Vector<amrex::MultiFab*> const& lapse_times_conformal_factor,
   amrex::Vector<amrex::MultiFab*> const& azimuthal_shift,
   amrex::Vector<amrex::MultiFab*> const& conformal_extrinsic_curvature,
   amrex::Vector<amrex::MultiFab const*> const& energy_density,
   amrex::Vector<amrex::MultiFab const*> const& azimuthal_momentum_density,
   amrex::Vector<amrex::MultiFab const*> const& stress_trace,
   InitialDataEllipticBoundaryConditions const& conformal_correction_boundary,
   InitialDataEllipticBoundaryConditions const& lapse_correction_boundary,
   InitialDataEllipticBoundaryConditions const& azimuthal_shift_boundary,
   XctsOptions const& options
)
{
#if (AMREX_SPACEDIM == 1)
   amrex::ignore_unused(
      hierarchy,
      conformal_factor,
      lapse_times_conformal_factor,
      azimuthal_shift,
      conformal_extrinsic_curvature,
      energy_density,
      azimuthal_momentum_density,
      stress_trace,
      conformal_correction_boundary,
      lapse_correction_boundary,
      azimuthal_shift_boundary,
      options
   );
   amrex::Abort(
      "axisymmetric spherical XCTS requires r-theta or r-theta-phi"
   );
   return {};
#else
   validate_axisymmetric_xcts_configuration(
      hierarchy,
      conformal_factor,
      lapse_times_conformal_factor,
      azimuthal_shift,
      conformal_extrinsic_curvature,
      energy_density,
      azimuthal_momentum_density,
      stress_trace,
      options
   );
   const int num_levels = static_cast<int>(hierarchy.geometry.size());
   amrex::Vector<amrex::MultiFab const*> conformal_const(num_levels);
   amrex::Vector<amrex::MultiFab const*> lapse_conformal_const(num_levels);
   amrex::Vector<amrex::MultiFab const*> extrinsic_const(num_levels);
   for (int level = 0; level < num_levels; ++level) {
      conformal_const[level] = conformal_factor[level];
      lapse_conformal_const[level] = lapse_times_conformal_factor[level];
      extrinsic_const[level] = conformal_extrinsic_curvature[level];
   }

   CttHamiltonianOptions scalar_options;
   scalar_options.max_nonlinear_iterations = options.max_coupled_iterations;
   scalar_options.relative_update_tolerance = options.relative_update_tolerance;
   scalar_options.relaxation = options.relaxation;
   scalar_options.linear = options.linear;

   XctsResult result;
   for (int iteration = 1;
        iteration <= options.max_coupled_iterations;
        ++iteration) {
      fill_axisymmetric_xcts_extrinsic_curvature(
         hierarchy,
         azimuthal_shift,
         conformal_const,
         lapse_conformal_const,
         conformal_extrinsic_curvature,
         azimuthal_shift_boundary
      );
      const CttHamiltonianResult hamiltonian = solve_ctt_hamiltonian(
         hierarchy,
         conformal_factor,
         energy_density,
         extrinsic_const,
         {},
         conformal_correction_boundary,
         scalar_options,
#if (AMREX_SPACEDIM == 2)
         ConformalScalarOperator::axisymmetric_spherical
#else
         ConformalScalarOperator::spherical_axisymmetric_storage
#endif
      );
      const CttHamiltonianResult lapse =
         solve_axisymmetric_xcts_lapse_iteration(
            hierarchy,
            conformal_const,
            lapse_times_conformal_factor,
            extrinsic_const,
            energy_density,
            stress_trace,
            lapse_correction_boundary,
            scalar_options
         );
      for (int level = 0; level < num_levels; ++level) {
         if (!(conformal_factor[level]->min(0, 0, false) > 0.0_rt)
             || !(lapse_times_conformal_factor[level]->min(0, 0, false)
                  > 0.0_rt)) {
            amrex::Abort(
               "axisymmetric XCTS produced a nonpositive conformal unknown"
            );
         }
      }
      fill_axisymmetric_xcts_extrinsic_curvature(
         hierarchy,
         azimuthal_shift,
         conformal_const,
         lapse_conformal_const,
         conformal_extrinsic_curvature,
         azimuthal_shift_boundary
      );
      const ConformalMomentumResult momentum =
         solve_axisymmetric_xcts_shift_iteration(
            hierarchy,
            conformal_factor,
            lapse_times_conformal_factor,
            azimuthal_shift,
            extrinsic_const,
            azimuthal_momentum_density,
            conformal_correction_boundary,
            lapse_correction_boundary,
            azimuthal_shift_boundary,
            options.linear,
            options.relaxation
         );
      result.coupled_iterations = iteration;
      result.relative_update = std::max({
         hamiltonian.relative_update,
         lapse.relative_update,
         momentum.relative_update,
      });
      result.hamiltonian_residual = hamiltonian.linear_residual;
      result.lapse_residual = lapse.linear_residual;
      result.momentum_residual = momentum.linear_residual;
      if (!std::isfinite(result.relative_update)) {
         amrex::Abort("axisymmetric XCTS produced a nonfinite update");
      }
      if (hamiltonian.converged
          && lapse.converged
          && result.relative_update <= options.relative_update_tolerance) {
         result.converged = true;
         break;
      }
   }
   fill_axisymmetric_xcts_extrinsic_curvature(
      hierarchy,
      azimuthal_shift,
      conformal_const,
      lapse_conformal_const,
      conformal_extrinsic_curvature,
      azimuthal_shift_boundary
   );
   return result;
#endif
}

XctsResult solve_static_spherical_xcts_constraints(
   InitialDataEllipticHierarchy const& hierarchy,
   amrex::Vector<amrex::MultiFab*> const& conformal_factor,
   amrex::Vector<amrex::MultiFab*> const& lapse_times_conformal_factor,
   amrex::Vector<amrex::MultiFab*> const& azimuthal_shift,
   amrex::Vector<amrex::MultiFab*> const& conformal_extrinsic_curvature,
   amrex::Vector<amrex::MultiFab const*> const& energy_density,
   amrex::Vector<amrex::MultiFab const*> const& azimuthal_momentum_density,
   amrex::Vector<amrex::MultiFab const*> const& stress_trace,
   InitialDataEllipticBoundaryConditions const& conformal_correction_boundary,
   InitialDataEllipticBoundaryConditions const& lapse_correction_boundary,
   amrex::MultiFab const* conformal_baryon_density,
   amrex::MultiFab const* specific_internal_energy,
   EOSParams const* eos,
   XctsOptions const& options
)
{
#if (AMREX_SPACEDIM == 1)
   amrex::ignore_unused(
      hierarchy,
      conformal_factor,
      lapse_times_conformal_factor,
      azimuthal_shift,
      conformal_extrinsic_curvature,
      energy_density,
      azimuthal_momentum_density,
      stress_trace,
      conformal_correction_boundary,
      lapse_correction_boundary,
      conformal_baryon_density,
      specific_internal_energy,
      eos,
      options
   );
   amrex::Abort("static spherical XCTS requires r-theta storage");
   return {};
#else
   validate_axisymmetric_xcts_configuration(
      hierarchy,
      conformal_factor,
      lapse_times_conformal_factor,
      azimuthal_shift,
      conformal_extrinsic_curvature,
      energy_density,
      azimuthal_momentum_density,
      stress_trace,
      options
   );
   if (hierarchy.geometry.size() != 1
       || azimuthal_momentum_density[0]->norm0(0, 0, false) > 1.0e-14_rt) {
      amrex::Abort(
         "static spherical XCTS requires one level and zero momentum density"
      );
   }
   const bool preserve_baryon_density =
      conformal_baryon_density != nullptr;
   if (preserve_baryon_density != (specific_internal_energy != nullptr)
       || preserve_baryon_density != (eos != nullptr)
       || (preserve_baryon_density
           && conformal_baryon_density->boxArray()
              != conformal_factor[0]->boxArray())
       || (preserve_baryon_density
           && specific_internal_energy->boxArray()
              != conformal_factor[0]->boxArray())) {
      amrex::Abort(
         "static spherical XCTS conformal baryon data are inconsistent"
      );
   }
   const EOSParams active_eos = preserve_baryon_density
      ? *eos : EOSParams{};

   amrex::MultiFab hamiltonian_rhs(
      conformal_factor[0]->boxArray(),
      conformal_factor[0]->DistributionMap(),
      1,
      0
   );
   amrex::MultiFab hamiltonian_reaction(
      conformal_factor[0]->boxArray(),
      conformal_factor[0]->DistributionMap(),
      1,
      0
   );
   amrex::MultiFab lapse_rhs(
      conformal_factor[0]->boxArray(),
      conformal_factor[0]->DistributionMap(),
      1,
      0
   );
   amrex::MultiFab lapse_reaction(
      conformal_factor[0]->boxArray(),
      conformal_factor[0]->DistributionMap(),
      1,
      0
   );
   amrex::MultiFab conformal_candidate(
      conformal_factor[0]->boxArray(),
      conformal_factor[0]->DistributionMap(),
      1,
      1
   );
   amrex::MultiFab lapse_candidate(
      conformal_factor[0]->boxArray(),
      conformal_factor[0]->DistributionMap(),
      1,
      1
   );
   amrex::MultiFab update(
      conformal_factor[0]->boxArray(),
      conformal_factor[0]->DistributionMap(),
      1,
      1
   );
   amrex::MultiFab conformal_previous(
      conformal_factor[0]->boxArray(),
      conformal_factor[0]->DistributionMap(),
      1,
      1
   );
   amrex::MultiFab nonlinear_candidate(
      conformal_factor[0]->boxArray(),
      conformal_factor[0]->DistributionMap(),
      1,
      1
   );
   amrex::MultiFab nonlinear_defect(
      conformal_factor[0]->boxArray(),
      conformal_factor[0]->DistributionMap(),
      1,
      0
   );

   amrex::Vector<amrex::MultiFab*> conformal_candidate_view{
      &conformal_candidate
   };
   amrex::Vector<amrex::MultiFab*> lapse_candidate_view{&lapse_candidate};
   amrex::Vector<amrex::MultiFab const*> hamiltonian_rhs_view{
      &hamiltonian_rhs
   };
   amrex::Vector<amrex::MultiFab const*> hamiltonian_reaction_view{
      &hamiltonian_reaction
   };
   amrex::Vector<amrex::MultiFab const*> lapse_rhs_view{&lapse_rhs};
   amrex::Vector<amrex::MultiFab const*> lapse_reaction_view{
      &lapse_reaction
   };
   amrex::Vector<amrex::MultiFab*> nonlinear_candidate_view{
      &nonlinear_candidate
   };

   azimuthal_shift[0]->setVal(0.0_rt);
   conformal_extrinsic_curvature[0]->setVal(0.0_rt);
   const auto hamiltonian_fixed_point_defect = [&]() {
      for (amrex::MFIter mfi(
              hamiltonian_rhs, amrex::TilingIfNotGPU()
           ); mfi.isValid(); ++mfi) {
         const amrex::Box box = mfi.tilebox();
         const auto psi = conformal_factor[0]->const_array(mfi);
         const auto energy = energy_density[0]->const_array(mfi);
         const auto baryon = preserve_baryon_density
            ? conformal_baryon_density->const_array(mfi)
            : amrex::Array4<amrex::Real const>{};
         const auto epsilon = preserve_baryon_density
            ? specific_internal_energy->const_array(mfi)
            : amrex::Array4<amrex::Real const>{};
         const auto rhs = hamiltonian_rhs.array(mfi);
         amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(
            int i, int j, int k
         ) noexcept {
            const amrex::Real psi_value = psi(i, j, k, 0);
            amrex::Real energy_value = energy(i, j, k, 0);
            if (preserve_baryon_density
                && baryon(i, j, k, 0) > 0.0_rt) {
               const amrex::Real psi_squared = psi_value * psi_value;
               const amrex::Real rho = baryon(i, j, k, 0) / (
                  psi_squared * psi_squared * psi_squared
               );
               energy_value = rho * (1.0_rt + epsilon(i, j, k, 0));
            }
            rhs(i, j, k, 0) = generated::conformal_hamiltonian_source(
               psi_value, energy_value, 0.0_rt, 0.0_rt
            );
         });
      }
      solve_initial_data_axisymmetric_spherical_poisson(
         hierarchy,
         nonlinear_candidate_view,
         hamiltonian_rhs_view,
         conformal_correction_boundary,
         options.linear
      );
      nonlinear_candidate.plus(1.0_rt, 0, 1, 1);
      amrex::MultiFab::Copy(
         nonlinear_defect, nonlinear_candidate, 0, 0, 1, 0
      );
      amrex::MultiFab::Subtract(
         nonlinear_defect, *conformal_factor[0], 0, 0, 1, 0
      );
      return nonlinear_defect.norm0(0, 0, false) / std::max(
         amrex::Real(1.0_rt), conformal_factor[0]->norm0(0, 0, false)
      );
   };
   XctsResult result;
   for (int iteration = 1;
        iteration <= options.max_coupled_iterations;
        ++iteration) {
      const amrex::Real old_hamiltonian_defect =
         hamiltonian_fixed_point_defect();
      for (amrex::MFIter mfi(
              hamiltonian_rhs, amrex::TilingIfNotGPU()
           ); mfi.isValid(); ++mfi) {
         const amrex::Box box = mfi.tilebox();
         const auto psi = conformal_factor[0]->const_array(mfi);
         const auto energy = energy_density[0]->const_array(mfi);
         const auto baryon = preserve_baryon_density
            ? conformal_baryon_density->const_array(mfi)
            : amrex::Array4<amrex::Real const>{};
         const auto epsilon = preserve_baryon_density
            ? specific_internal_energy->const_array(mfi)
            : amrex::Array4<amrex::Real const>{};
         const auto rhs = hamiltonian_rhs.array(mfi);
         const auto reaction = hamiltonian_reaction.array(mfi);
         amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(
            int i, int j, int k
         ) noexcept {
            const amrex::Real psi_value = psi(i, j, k, 0);
            amrex::Real energy_value = energy(i, j, k, 0);
            amrex::Real local_jacobian =
               generated::conformal_hamiltonian_local_jacobian(
                  psi_value, energy_value, 0.0_rt, 0.0_rt
               );
            if (preserve_baryon_density
                && baryon(i, j, k, 0) > 0.0_rt) {
               const amrex::Real psi_squared = psi_value * psi_value;
               const amrex::Real rho = baryon(i, j, k, 0) / (
                  psi_squared * psi_squared * psi_squared
               );
               energy_value = rho * (1.0_rt + epsilon(i, j, k, 0));
               // At fixed q=sqrt(gamma)rho=psi^6 rho and fixed epsilon,
               // E=rho(1+epsilon) scales as psi^-6.  Thus
               // -d[-2 pi psi^5 E]/dpsi = -2 pi psi^4 E.
               local_jacobian = -2.0_rt * constants::pi
                  * amrex::Math::powi<4>(psi_value) * energy_value;
            }
            const amrex::Real source =
               generated::conformal_hamiltonian_source(
                  psi_value, energy_value, 0.0_rt, 0.0_rt
               );
            // Solve directly for the new correction c=psi_new-1:
            // (Delta + J)c = source + J(psi_old-1).
            reaction(i, j, k, 0) = local_jacobian;
            rhs(i, j, k, 0) = source
               + local_jacobian * (psi_value - 1.0_rt);
         });
      }
      const InitialDataEllipticResult hamiltonian =
         solve_initial_data_axisymmetric_spherical_helmholtz(
            hierarchy,
            conformal_candidate_view,
            hamiltonian_rhs_view,
            hamiltonian_reaction_view,
            conformal_correction_boundary,
            options.linear
         );
      if (!std::isfinite(hamiltonian.residual)) {
         amrex::Abort("static spherical XCTS Newton solve is nonfinite");
      }
      conformal_candidate.plus(1.0_rt, 0, 1, 1);
      amrex::MultiFab::Copy(
         conformal_previous, *conformal_factor[0], 0, 0, 1, 1
      );
      amrex::MultiFab::Copy(update, conformal_candidate, 0, 0, 1, 1);
      amrex::MultiFab::Subtract(
         update, *conformal_factor[0], 0, 0, 1, 1
      );
      const amrex::Real full_update_norm = update.norm0(0, 0, false);
      amrex::Real step_fraction = options.relaxation;
      amrex::Real hamiltonian_defect =
         std::numeric_limits<amrex::Real>::infinity();
      bool accepted_step = false;
      constexpr amrex::Real minimum_step_fraction = 1.0_rt / 1048576.0_rt;
      while (step_fraction >= minimum_step_fraction) {
         amrex::MultiFab::Copy(
            *conformal_factor[0], conformal_previous, 0, 0, 1, 1
         );
         amrex::MultiFab::Saxpy(
            *conformal_factor[0], step_fraction, update, 0, 0, 1, 1
         );
         if (conformal_factor[0]->min(0, 0, false) > 0.0_rt) {
            hamiltonian_defect = hamiltonian_fixed_point_defect();
            const amrex::Real sufficient_defect = std::max(
               options.relative_update_tolerance,
               old_hamiltonian_defect * (1.0_rt - 1.0e-4_rt * step_fraction)
            );
            if (hamiltonian_defect <= sufficient_defect) {
               accepted_step = true;
               break;
            }
         }
         step_fraction *= 0.5_rt;
      }
      if (!accepted_step) {
         amrex::Abort(
            "static spherical XCTS Newton line search did not reduce the "
            "Hamiltonian defect"
         );
      }
      const amrex::Real conformal_update =
         step_fraction * full_update_norm
         / std::max(
            amrex::Real(1.0_rt),
            conformal_factor[0]->norm0(0, 0, false)
         );

      for (amrex::MFIter mfi(lapse_rhs, amrex::TilingIfNotGPU());
           mfi.isValid(); ++mfi) {
         const amrex::Box box = mfi.tilebox();
         const auto psi = conformal_factor[0]->const_array(mfi);
         const auto energy = energy_density[0]->const_array(mfi);
         const auto stress = stress_trace[0]->const_array(mfi);
         const auto baryon = preserve_baryon_density
            ? conformal_baryon_density->const_array(mfi)
            : amrex::Array4<amrex::Real const>{};
         const auto epsilon = preserve_baryon_density
            ? specific_internal_energy->const_array(mfi)
            : amrex::Array4<amrex::Real const>{};
         const auto rhs = lapse_rhs.array(mfi);
         const auto reaction = lapse_reaction.array(mfi);
         amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(
            int i, int j, int k
         ) noexcept {
            const amrex::Real psi_value = psi(i, j, k, 0);
            amrex::Real energy_value = energy(i, j, k, 0);
            amrex::Real stress_value = stress(i, j, k, 0);
            if (preserve_baryon_density
                && baryon(i, j, k, 0) > 0.0_rt) {
               const amrex::Real psi_squared = psi_value * psi_value;
               const amrex::Real rho = baryon(i, j, k, 0) / (
                  psi_squared * psi_squared * psi_squared
               );
               const amrex::Real specific_energy = epsilon(i, j, k, 0);
               energy_value = rho * (1.0_rt + specific_energy);
               stress_value = 3.0_rt * relativistic_pressure<EOSRuntime>(
                  rho, specific_energy, active_eos
               );
            }
            const amrex::Real coefficient = 2.0_rt * constants::pi
               * amrex::Math::powi<4>(psi_value)
               * (energy_value + 2.0_rt * stress_value);
            // Delta(alpha psi)-q(alpha psi)=0.  For c=alpha psi-1,
            // (Delta-q)c=q.
            reaction(i, j, k, 0) = -coefficient;
            rhs(i, j, k, 0) = coefficient;
         });
      }
      const InitialDataEllipticResult lapse =
         solve_initial_data_axisymmetric_spherical_helmholtz(
            hierarchy,
            lapse_candidate_view,
            lapse_rhs_view,
            lapse_reaction_view,
            lapse_correction_boundary,
            options.linear
         );
      lapse_candidate.plus(1.0_rt, 0, 1, 1);
      amrex::MultiFab::Copy(update, lapse_candidate, 0, 0, 1, 1);
      amrex::MultiFab::Subtract(
         update, *lapse_times_conformal_factor[0], 0, 0, 1, 1
      );
      update.mult(options.relaxation, 0, 1, 1);
      amrex::MultiFab::Add(
         *lapse_times_conformal_factor[0], update, 0, 0, 1, 1
      );
      const amrex::Real lapse_update = update.norm0(0, 0, false)
         / std::max(
            amrex::Real(1.0_rt),
            lapse_times_conformal_factor[0]->norm0(0, 0, false)
         );

      result.coupled_iterations = iteration;
      result.relative_update = std::max(conformal_update, lapse_update);
      result.hamiltonian_residual = hamiltonian_defect;
      result.lapse_residual = lapse.residual;
      result.momentum_residual = 0.0_rt;
      if (!(conformal_factor[0]->min(0, 0, false) > 0.0_rt)
          || !(lapse_times_conformal_factor[0]->min(0, 0, false) > 0.0_rt)
          || !std::isfinite(result.relative_update)) {
         amrex::Abort(
            "static spherical XCTS produced a nonpositive or nonfinite iterate"
         );
      }
      if (result.relative_update <= options.relative_update_tolerance
          && hamiltonian_defect <= options.relative_update_tolerance) {
         result.converged = true;
         break;
      }
   }
   // The separable direct solve owns valid cells only.  Vertex ADM
   // reconstruction reads the adjacent cell ghosts at the radial and angular
   // boundaries, so synchronize the accepted nonlinear fields before handing
   // them to the common cell-to-vertex materializer.  Leaving the old seed in
   // these ghosts produces an angular-axis defect despite a converged radial
   // solution.
   fill_initial_data_ghosts(
      hierarchy,
      conformal_factor,
      initial_data_boundary_records(
         hierarchy, conformal_correction_boundary
      )
   );
   fill_initial_data_ghosts(
      hierarchy,
      lapse_times_conformal_factor,
      initial_data_boundary_records(
         hierarchy, lapse_correction_boundary
      )
   );
   return result;
#endif
}

} // namespace pyfluid_amrex
