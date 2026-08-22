#include "pyfluid_amrex/initial_data/ctt.hpp"

#include <AMReX.H>
#include <AMReX_GpuAsyncArray.H>
#include <AMReX_GpuLaunch.H>

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>

#include "pyfluid_amrex/constants.hpp"
#include "pyfluid_amrex/kernels_generated/differential_ops.hpp"
#include "pyfluid_amrex/kernels_generated/initial_data.hpp"

namespace pyfluid_amrex {

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
         candidate_storage[level]->plus(-1.0, 0, 1, 1);
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
      amrex::Real update_norm = 0.0;
      amrex::Real solution_norm = 0.0;
      for (int level = 0; level < num_levels; ++level) {
         amrex::MultiFab update(
            field[level]->boxArray(), field[level]->DistributionMap(), 1, 1
         );
         amrex::MultiFab::Copy(update, *candidate_storage[level], 0, 0, 1, 1);
         amrex::MultiFab::Subtract(update, *field[level], 0, 0, 1, 1);
         update.plus(1.0, 0, 1, 1);
         update.mult(options.relaxation, 0, 1, 1);
         amrex::MultiFab::Add(*field[level], update, 0, 0, 1, 1);
         update_norm = std::max(update_norm, update.norm0(0, 0, false));
         solution_norm = std::max(
            solution_norm, field[level]->norm0(0, 0, false)
         );
      }
      result.nonlinear_iterations = iteration;
      result.relative_update = update_norm / std::max(amrex::Real(1.0), solution_norm);
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
      || !(options.relative_update_tolerance > 0.0)
      || !(options.relaxation > 0.0 && options.relaxation <= 1.0)
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
                  : 0.0;
               const amrex::Real k_squared = has_mean_curvature
                  ? curvature_squared(i, j, k, 0)
                  : 0.0;
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

#if (AMREX_SPACEDIM == 3)
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
      const amrex::GpuArray<amrex::Real, 3> inv_spacing{{
         1.0 / spacing[0], 1.0 / spacing[1], 1.0 / spacing[2]
      }};
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
         amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            ConformalTraceFreeTensor point =
               generated::conformal_longitudinal_tensor(
                  generated::centered_first<2, 0>(vector, i, j, k, 0, inv_spacing[0]),
                  generated::centered_first<2, 1>(vector, i, j, k, 0, inv_spacing[1]),
                  generated::centered_first<2, 2>(vector, i, j, k, 0, inv_spacing[2]),
                  generated::centered_first<2, 0>(vector, i, j, k, 1, inv_spacing[0]),
                  generated::centered_first<2, 1>(vector, i, j, k, 1, inv_spacing[1]),
                  generated::centered_first<2, 2>(vector, i, j, k, 1, inv_spacing[2]),
                  generated::centered_first<2, 0>(vector, i, j, k, 2, inv_spacing[0]),
                  generated::centered_first<2, 1>(vector, i, j, k, 2, inv_spacing[1]),
                  generated::centered_first<2, 2>(vector, i, j, k, 2, inv_spacing[2])
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
   amrex::Real relative_update = 0.0;
   amrex::Real linear_residual = 0.0;
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
   amrex::Real update_norm = 0.0;
   amrex::Real solution_norm = 0.0;
   for (int level = 0; level < num_levels; ++level) {
      amrex::MultiFab update(
         vector_field[level]->boxArray(),
         vector_field[level]->DistributionMap(),
         3,
         1
      );
      amrex::MultiFab::Copy(update, *candidate_storage[level], 0, 0, 3, 1);
      amrex::MultiFab::Subtract(update, *vector_field[level], 0, 0, 3, 1);
      update.mult(relaxation, 0, 3, 1);
      amrex::MultiFab::Add(*vector_field[level], update, 0, 0, 3, 1);
      for (int component = 0; component < 3; ++component) {
         update_norm = std::max(
            update_norm, update.norm0(component, 0, false)
         );
         solution_norm = std::max(
            solution_norm,
            vector_field[level]->norm0(component, 0, false)
         );
      }
   }
   return {
      update_norm / std::max(amrex::Real(1.0), solution_norm),
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
         const amrex::GpuArray<amrex::Real, 3> inv_spacing{{
            1.0 / spacing[0], 1.0 / spacing[1], 1.0 / spacing[2]
         }};
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
                  amrex::Real curvature_gradient = 0.0;
                  if (variable_mean_curvature) {
                     if (component == 0) {
                        curvature_gradient = generated::centered_first<2, 0>(
                           curvature, i, j, k, 0, inv_spacing[0]
                        );
                     } else if (component == 1) {
                        curvature_gradient = generated::centered_first<2, 1>(
                           curvature, i, j, k, 0, inv_spacing[1]
                        );
                     } else {
                        curvature_gradient = generated::centered_first<2, 2>(
                           curvature, i, j, k, 0, inv_spacing[2]
                        );
                     }
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
   amrex::Real relative_update = 0.0;
   amrex::Real minimum_squared = 0.0;
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
       || !(options.relative_update_tolerance > 0.0)
       || !(options.relaxation > 0.0 && options.relaxation <= 1.0)
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
   amrex::Real update_norm = 0.0;
   amrex::Real solution_norm = 0.0;
   for (int level = 0; level < num_levels; ++level) {
      amrex::MultiFab previous(
         mean_curvature[level]->boxArray(),
         mean_curvature[level]->DistributionMap(),
         1,
         0
      );
      amrex::MultiFab::Copy(previous, *mean_curvature[level], 0, 0, 1, 0);
      const auto spacing = hierarchy.geometry[level].CellSizeArray();
      const amrex::GpuArray<amrex::Real, 3> inv_spacing_squared{{
         1.0 / (spacing[0] * spacing[0]),
         1.0 / (spacing[1] * spacing[1]),
         1.0 / (spacing[2] * spacing[2])
      }};
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
                  generated::centered_second<2, 0>(
                     psi, i, j, k, 0, inv_spacing_squared[0]
                  )
                  + generated::centered_second<2, 1>(
                     psi, i, j, k, 0, inv_spacing_squared[1]
                  )
                  + generated::centered_second<2, 2>(
                     psi, i, j, k, 0, inv_spacing_squared[2]
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
      if (minimum_squared < 0.0) {
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
      update_norm / std::max(amrex::Real(1.0), solution_norm),
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
       || !(options.relative_update_tolerance > 0.0)
       || !(options.relaxation > 0.0 && options.relaxation <= 1.0)) {
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
         const amrex::GpuArray<amrex::Real, 3> inv_spacing{{
            1.0 / spacing[0], 1.0 / spacing[1], 1.0 / spacing[2]
         }};
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
                  generated::centered_first<2, 0>(
                     psi, i, j, k, 0, inv_spacing[0]
                  ),
                  generated::centered_first<2, 1>(
                     psi, i, j, k, 0, inv_spacing[1]
                  ),
                  generated::centered_first<2, 2>(
                     psi, i, j, k, 0, inv_spacing[2]
                  ),
               };
               const amrex::Real d_lapse_psi[3] = {
                  generated::centered_first<2, 0>(
                     lapse_psi, i, j, k, 0, inv_spacing[0]
                  ),
                  generated::centered_first<2, 1>(
                     lapse_psi, i, j, k, 0, inv_spacing[1]
                  ),
                  generated::centered_first<2, 2>(
                     lapse_psi, i, j, k, 0, inv_spacing[2]
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
#endif

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
       || !(options.relative_update_tolerance > 0.0)
       || !(options.relaxation > 0.0 && options.relaxation <= 1.0)) {
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
       || std::abs(phi_extent - 2.0 * constants::pi)
          > 64.0 * std::numeric_limits<amrex::Real>::epsilon()
             * std::max(amrex::Real(1.0), std::abs(phi_extent))) {
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
      const bool radial_origin = radial_lower_face == 0.0;
      if (i == domain_lo.x && !radial_origin) {
         lower_i = i + 1;
         upper_i = i + 2;
      } else if (i == domain_hi.x) {
         lower_i = i - 2;
         upper_i = i - 1;
      }
      const amrex::Real center = radial_centers[i - domain_lo.x];
      const amrex::Real lower_coordinate = lower_i < domain_lo.x
         ? 2.0 * radial_lower_face
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
               2.0 * center - lower_coordinate - upper_coordinate
            ) / (
               (center - lower_coordinate)
               * (center - upper_coordinate)
            )
         + upper_value * (center - lower_coordinate) / (
               (upper_coordinate - lower_coordinate)
               * (upper_coordinate - center)
            );
   }
   return 0.5 * inverse_angular_spacing * (
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
      const amrex::Real inverse_dtheta = 1.0 / spacing[1];
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
               + (j - domain_lo.y + 0.5) * spacing[1];
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
            extrinsic(i, j, k, CONFORMAL_A11) = 0.0;
            extrinsic(i, j, k, CONFORMAL_A12) = 0.0;
            extrinsic(i, j, k, CONFORMAL_A13) = a_r_phi;
            extrinsic(i, j, k, CONFORMAL_A22) = 0.0;
            extrinsic(i, j, k, CONFORMAL_A23) = a_theta_phi;
            extrinsic(i, j, k, CONFORMAL_A33) = 0.0;
            extrinsic(i, j, k, CONFORMAL_A_SQUARED) = 2.0 * (
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

struct AxisymmetricMomentumResult {
   amrex::Real relative_update = 0.0;
   amrex::Real linear_residual = 0.0;
};

AxisymmetricMomentumResult solve_axisymmetric_xcts_shift_iteration(
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
   const amrex::Real inverse_dtheta = 1.0 / spacing[1];
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
            0.0,
            d_lapse_psi_r,
            d_lapse_psi_theta,
            0.0,
            extrinsic(i, j, k, CONFORMAL_A13),
            extrinsic(i, j, k, CONFORMAL_A23),
            0.0
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
   amrex::MultiFab update(
      azimuthal_shift[0]->boxArray(),
      azimuthal_shift[0]->DistributionMap(),
      1,
      1
   );
   amrex::MultiFab::Copy(update, *candidate_storage[0], 0, 0, 1, 1);
   amrex::MultiFab::Subtract(update, *azimuthal_shift[0], 0, 0, 1, 1);
   update.mult(relaxation, 0, 1, 1);
   amrex::MultiFab::Add(*azimuthal_shift[0], update, 0, 0, 1, 1);
   return {
      update.norm0(0, 0, false) / std::max(
         amrex::Real(1.0), azimuthal_shift[0]->norm0(0, 0, false)
      ),
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
#if (AMREX_SPACEDIM != 3)
   amrex::ignore_unused(
      hierarchy,
      conformal_factor,
      vector_potential,
      conformal_extrinsic_curvature,
      energy_density,
      momentum_density,
      conformal_correction_boundary,
      vector_potential_boundary,
      options
   );
   amrex::Abort("coupled CTT momentum constraints require the 3D executable");
   return {};
#else
   const int num_levels = static_cast<int>(hierarchy.geometry.size());
   if (
      num_levels == 0
      || static_cast<int>(conformal_factor.size()) != num_levels
      || static_cast<int>(vector_potential.size()) != num_levels
      || static_cast<int>(conformal_extrinsic_curvature.size()) != num_levels
      || static_cast<int>(energy_density.size()) != num_levels
      || static_cast<int>(momentum_density.size()) != num_levels
      || options.max_coupled_iterations <= 0
      || !(options.relative_update_tolerance > 0.0)
      || !(options.relaxation > 0.0 && options.relaxation <= 1.0)
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
#endif
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
#if (AMREX_SPACEDIM != 3)
   amrex::ignore_unused(
      hierarchy,
      conformal_factor,
      mean_curvature,
      vector_potential,
      conformal_extrinsic_curvature,
      energy_density,
      momentum_density,
      vector_potential_boundary,
      options
   );
   amrex::Abort("CTTK constraints require the 3D executable");
   return {};
#else
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
#endif
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
#if (AMREX_SPACEDIM != 3)
   amrex::ignore_unused(
      hierarchy,
      conformal_factor,
      mean_curvature,
      vector_potential,
      conformal_extrinsic_curvature,
      energy_density,
      momentum_density,
      conformal_correction_boundary,
      vector_potential_boundary,
      options
   );
   amrex::Abort("hybrid CTTK constraints require the 3D executable");
   return {};
#else
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
#endif
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
#if (AMREX_SPACEDIM != 3)
   amrex::ignore_unused(
      hierarchy,
      conformal_factor,
      lapse_times_conformal_factor,
      shift,
      conformal_extrinsic_curvature,
      energy_density,
      momentum_density,
      stress_trace,
      conformal_correction_boundary,
      lapse_correction_boundary,
      shift_boundary,
      options
   );
   amrex::Abort("XCTS constraints require the 3D executable");
   return {};
#else
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
         if (!(conformal_factor[level]->min(0, 0, false) > 0.0)
             || !(lapse_times_conformal_factor[level]->min(0, 0, false) > 0.0)) {
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
#endif
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
         if (!(conformal_factor[level]->min(0, 0, false) > 0.0)
             || !(lapse_times_conformal_factor[level]->min(0, 0, false)
                  > 0.0)) {
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
      const AxisymmetricMomentumResult momentum =
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

} // namespace pyfluid_amrex
