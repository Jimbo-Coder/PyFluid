#include "pyfluid_amrex/initial_data/elliptic.hpp"

#include <AMReX.H>
#include <AMReX_GMRES.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_FillPatchUtil.H>
#include <AMReX_LO_BCTYPES.H>
#include <AMReX_MLABecLaplacian.H>
#include <AMReX_MLMG.H>
#include <AMReX_MLNodeLaplacian.H>
#include <AMReX_MLTensorOp.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_PhysBCFunct.H>

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <vector>

#include "pyfluid_amrex/initial_data/centering.hpp"
#include "pyfluid_amrex/kernels_generated/differential_ops.hpp"

namespace pyfluid_amrex {
namespace {

using InitialDataBoundaryFunction = amrex::GpuBndryFuncFab<amrex::FabFillNoOp>;
using InitialDataPhysicalBoundary = amrex::PhysBCFunct<InitialDataBoundaryFunction>;

using BoundaryTypeArray = amrex::Array<amrex::LinOpBCType, AMREX_SPACEDIM>;

enum class BoundaryRepresentation {
   dirichlet,
   robin,
};

amrex::BoxArray elliptic_box_array(
   InitialDataEllipticHierarchy const& hierarchy,
   int level,
   InitialDataEllipticCentering centering
)
{
   if (centering == InitialDataEllipticCentering::vertex) {
      return amrex::convert(
         hierarchy.grids[level], amrex::IntVect::TheNodeVector()
      );
   }
   return hierarchy.grids[level];
}

struct LoweredBoundaryConditions {
   BoundaryRepresentation representation = BoundaryRepresentation::robin;
   BoundaryTypeArray domain_types;
   amrex::Vector<std::unique_ptr<amrex::MultiFab>> level_data;
};

BoundaryTypeArray domain_boundary_types(
   amrex::Geometry const& geometry,
   BoundaryRepresentation representation
)
{
   BoundaryTypeArray types;
   for (int direction = 0; direction < AMREX_SPACEDIM; ++direction) {
      types[direction] = geometry.isPeriodic(direction)
         ? amrex::LinOpBCType::Periodic
         : representation == BoundaryRepresentation::dirichlet
            ? amrex::LinOpBCType::Dirichlet
            : amrex::LinOpBCType::Robin;
   }
   return types;
}

int validate_elliptic_solve(
   InitialDataEllipticHierarchy const& hierarchy,
   amrex::Vector<amrex::MultiFab*> const& solution,
   amrex::Vector<amrex::MultiFab const*> const& rhs,
   InitialDataEllipticBoundaryConditions const& boundary,
   InitialDataEllipticOptions const& options,
   int num_components
)
{
   const int num_levels = static_cast<int>(hierarchy.geometry.size());
   if (
      num_levels == 0
      || static_cast<int>(hierarchy.grids.size()) != num_levels
      || static_cast<int>(hierarchy.distribution.size()) != num_levels
      || static_cast<int>(hierarchy.refinement_ratio.size()) != num_levels - 1
      || static_cast<int>(solution.size()) != num_levels
      || static_cast<int>(rhs.size()) != num_levels
      || boundary.num_levels() != num_levels
      || boundary.num_components != num_components
   ) {
      amrex::Abort("initial-data elliptic hierarchy arrays must have the same nonzero size");
   }
   if (hierarchy.cell_interpolater == nullptr) {
      amrex::Abort("initial-data hierarchy requires a cell interpolater");
   }
   if (
      options.max_order < 2
      || options.max_iterations <= 0
      || !(options.relative_tolerance > 0.0)
      || options.absolute_tolerance < 0.0
   ) {
      amrex::Abort("invalid initial-data elliptic solver options");
   }
   for (int level = 0; level < num_levels; ++level) {
      const amrex::BoxArray expected_grids = elliptic_box_array(
         hierarchy, level, boundary.centering()
      );
      if (solution[level] == nullptr || rhs[level] == nullptr
          || solution[level]->nComp() != num_components
          || rhs[level]->nComp() != num_components
          || solution[level]->boxArray() != expected_grids
          || rhs[level]->boxArray() != expected_grids
          || boundary.coefficient_a(level).nComp() != num_components
          || boundary.coefficient_b(level).nComp() != num_components
          || boundary.coefficient_c(level).nComp() != num_components
          || boundary.coefficient_a(level).boxArray() != expected_grids
          || boundary.coefficient_b(level).boxArray() != expected_grids
          || boundary.coefficient_c(level).boxArray() != expected_grids
          || boundary.coefficient_a(level).nGrow() < 1
          || boundary.coefficient_b(level).nGrow() < 1
          || boundary.coefficient_c(level).nGrow() < 1) {
         amrex::Abort("initial-data elliptic solve received incompatible level data");
      }
      if (solution[level]->contains_nan(0, num_components, 0)) {
         amrex::Abort("initial-data elliptic solution contains nonfinite valid data");
      }
      if (rhs[level]->contains_nan(0, num_components, 0)) {
         amrex::Abort("initial-data elliptic right-hand side contains nonfinite valid data");
      }
      if (boundary.coefficient_a(level).contains_nan(0, num_components, 1)
          || boundary.coefficient_b(level).contains_nan(0, num_components, 1)
          || boundary.coefficient_c(level).contains_nan(0, num_components, 1)) {
         amrex::Abort("initial-data elliptic boundary contains nonfinite coefficients");
      }
   }
   return num_levels;
}

amrex::Vector<std::unique_ptr<amrex::MultiFab>> coordinate_to_normal_b(
   InitialDataEllipticHierarchy const& hierarchy,
   InitialDataEllipticBoundaryConditions const& boundary
)
{
   const int num_levels = boundary.num_levels();
   amrex::Vector<std::unique_ptr<amrex::MultiFab>> normal_b(num_levels);
   for (int level = 0; level < num_levels; ++level) {
      normal_b[level] = std::make_unique<amrex::MultiFab>(
         hierarchy.grids[level],
         hierarchy.distribution[level],
         boundary.num_components,
         1
      );
      amrex::MultiFab::Copy(
         *normal_b[level],
         boundary.coefficient_b(level),
         0,
         0,
         boundary.num_components,
         1
      );
      const amrex::Dim3 domain_lo = amrex::lbound(hierarchy.geometry[level].Domain());
      const amrex::Dim3 domain_hi = amrex::ubound(hierarchy.geometry[level].Domain());
      amrex::GpuArray<int, AMREX_SPACEDIM> periodic;
      for (int direction = 0; direction < AMREX_SPACEDIM; ++direction) {
         periodic[direction] = hierarchy.geometry[level].isPeriodic(direction);
      }
      for (amrex::MFIter mfi(*normal_b[level]); mfi.isValid(); ++mfi) {
         const amrex::Box box = mfi.fabbox();
         const auto coefficient = normal_b[level]->array(mfi);
         const int num_components = boundary.num_components;
         amrex::ParallelFor(
            box,
            num_components,
            [=] AMREX_GPU_DEVICE(int i, int j, int k, int component) noexcept {
               bool outside_low = false;
               bool has_physical_face = false;
               if (!periodic[0] && (i < domain_lo.x || i > domain_hi.x)) {
                  outside_low = i < domain_lo.x;
                  has_physical_face = true;
               }
#if (AMREX_SPACEDIM >= 2)
               else if (!periodic[1] && (j < domain_lo.y || j > domain_hi.y)) {
                  outside_low = j < domain_lo.y;
                  has_physical_face = true;
               }
#endif
#if (AMREX_SPACEDIM == 3)
               else if (!periodic[2] && (k < domain_lo.z || k > domain_hi.z)) {
                  outside_low = k < domain_lo.z;
                  has_physical_face = true;
               }
#endif
               if (has_physical_face && outside_low) {
                  coefficient(i, j, k, component) *= -1.0;
               }
            }
         );
      }
   }
   return normal_b;
}

bool has_zero_derivative_coefficient(
   InitialDataEllipticBoundaryConditions const& boundary
)
{
   for (int level = 0; level < boundary.num_levels(); ++level) {
      for (int component = 0; component < boundary.num_components; ++component) {
         if (boundary.coefficient_b(level).norm0(component, 1, false) != 0.0) {
            return false;
         }
      }
   }
   return true;
}

amrex::Vector<std::unique_ptr<amrex::MultiFab>> dirichlet_values(
   InitialDataEllipticHierarchy const& hierarchy,
   InitialDataEllipticBoundaryConditions const& boundary
)
{
   const int num_levels = boundary.num_levels();
   amrex::Vector<std::unique_ptr<amrex::MultiFab>> values(num_levels);
   for (int level = 0; level < num_levels; ++level) {
      values[level] = std::make_unique<amrex::MultiFab>(
         hierarchy.grids[level],
         hierarchy.distribution[level],
         boundary.num_components,
         1
      );
      values[level]->setVal(0.0);
      const amrex::Dim3 domain_lo = amrex::lbound(hierarchy.geometry[level].Domain());
      const amrex::Dim3 domain_hi = amrex::ubound(hierarchy.geometry[level].Domain());
      for (amrex::MFIter mfi(*values[level]); mfi.isValid(); ++mfi) {
         const amrex::Box box = mfi.fabbox();
         const auto coefficient_a = boundary.coefficient_a(level).const_array(mfi);
         const auto coefficient_c = boundary.coefficient_c(level).const_array(mfi);
         const auto value = values[level]->array(mfi);
         const int num_components = boundary.num_components;
         amrex::ParallelFor(
            box,
            num_components,
            [=] AMREX_GPU_DEVICE(int i, int j, int k, int component) noexcept {
               const bool outside_domain = i < domain_lo.x || i > domain_hi.x
#if (AMREX_SPACEDIM >= 2)
                  || j < domain_lo.y || j > domain_hi.y
#endif
#if (AMREX_SPACEDIM == 3)
                  || k < domain_lo.z || k > domain_hi.z
#endif
                  ;
               if (outside_domain) {
                  const amrex::Real a = coefficient_a(i, j, k, component);
                  value(i, j, k, component) = a != 0.0
                     ? coefficient_c(i, j, k, component) / a
                     : std::numeric_limits<amrex::Real>::quiet_NaN();
               }
            }
         );
      }
      if (values[level]->contains_nan(0, boundary.num_components, 1)) {
         amrex::Abort("Dirichlet coefficients require nonzero A on physical boundaries");
      }
   }
   return values;
}

LoweredBoundaryConditions lower_boundary_conditions(
   InitialDataEllipticHierarchy const& hierarchy,
   InitialDataEllipticBoundaryConditions const& boundary
)
{
   LoweredBoundaryConditions lowered;
   lowered.representation = has_zero_derivative_coefficient(boundary)
      ? BoundaryRepresentation::dirichlet
      : BoundaryRepresentation::robin;
   lowered.domain_types = domain_boundary_types(
      hierarchy.geometry[0], lowered.representation
   );
   lowered.level_data = lowered.representation == BoundaryRepresentation::dirichlet
      ? dirichlet_values(hierarchy, boundary)
      : coordinate_to_normal_b(hierarchy, boundary);
   return lowered;
}

template <typename LinearOperator>
void set_boundary_conditions(
   LinearOperator& linear_operator,
   InitialDataEllipticBoundaryConditions const& boundary,
   LoweredBoundaryConditions const& lowered
)
{
   linear_operator.setDomainBC(lowered.domain_types, lowered.domain_types);
   for (int level = 0; level < boundary.num_levels(); ++level) {
      if (lowered.representation == BoundaryRepresentation::dirichlet) {
         linear_operator.setLevelBC(level, lowered.level_data[level].get());
      } else {
         linear_operator.setLevelBC(
            level,
            nullptr,
            &boundary.coefficient_a(level),
            lowered.level_data[level].get(),
            &boundary.coefficient_c(level)
         );
      }
   }
}

amrex::Vector<std::unique_ptr<amrex::MultiFab>> negative_fields(
   InitialDataEllipticHierarchy const& hierarchy,
   amrex::Vector<amrex::MultiFab const*> const& fields,
   int num_components
)
{
   const int num_levels = static_cast<int>(fields.size());
   amrex::Vector<std::unique_ptr<amrex::MultiFab>> result(num_levels);
   for (int level = 0; level < num_levels; ++level) {
      result[level] = std::make_unique<amrex::MultiFab>(
         fields[level]->boxArray(), fields[level]->DistributionMap(),
         num_components,
         0
      );
      amrex::MultiFab::Copy(
         *result[level], *fields[level], 0, 0, num_components, 0
      );
      result[level]->mult(-1.0, 0, num_components, 0);
   }
   return result;
}

void configure_mlmg(amrex::MLMG& mlmg, InitialDataEllipticOptions const& options)
{
   mlmg.setMaxIter(options.max_iterations);
   mlmg.setMaxFmgIter(options.max_fmg_iterations);
   mlmg.setVerbose(options.verbose);
   mlmg.setBottomVerbose(options.verbose);
}

InitialDataEllipticResult elliptic_result(amrex::Real residual)
{
   if (!std::isfinite(residual)) {
      amrex::Abort("initial-data elliptic solve produced a nonfinite residual");
   }
   return {residual, 0.0, 0};
}

amrex::Vector<amrex::BCRec> nodal_solution_boundary_records(
   InitialDataEllipticHierarchy const& hierarchy,
   int num_components
)
{
   amrex::Vector<amrex::BCRec> records(num_components);
   for (amrex::BCRec& record : records) {
      for (int direction = 0; direction < AMREX_SPACEDIM; ++direction) {
         const int type = hierarchy.geometry[0].isPeriodic(direction)
            ? amrex::BCType::int_dir
            : amrex::BCType::foextrap;
         record.setLo(direction, type);
         record.setHi(direction, type);
      }
   }
   return records;
}

void synchronize_nodal_solution(
   InitialDataEllipticHierarchy const& hierarchy,
   amrex::Vector<amrex::MultiFab*> const& solution
)
{
   if (solution.front()->nGrow() == 0) {
      return;
   }
   detail::fill_initial_data_vertex_ghosts(
      hierarchy,
      solution,
      nodal_solution_boundary_records(hierarchy, solution.front()->nComp())
   );
}

void set_nodal_dirichlet_values(
   InitialDataEllipticHierarchy const& hierarchy,
   amrex::Vector<amrex::MultiFab*> const& solution,
   InitialDataEllipticBoundaryConditions const& boundary
)
{
   for (int level = 0; level < boundary.num_levels(); ++level) {
      const amrex::Box nodal_domain = amrex::surroundingNodes(
         hierarchy.geometry[level].Domain()
      );
      const auto periodic = hierarchy.geometry[level].isPeriodicArray();
      const int num_components = boundary.num_components;
      for (amrex::MFIter mfi(*solution[level], amrex::TilingIfNotGPU());
           mfi.isValid(); ++mfi) {
         const amrex::Box box = mfi.tilebox();
         const auto coefficient_a = boundary.coefficient_a(level).const_array(mfi);
         const auto coefficient_c = boundary.coefficient_c(level).const_array(mfi);
         const auto values = solution[level]->array(mfi);
         amrex::ParallelFor(
            box,
            num_components,
            [=] AMREX_GPU_DEVICE(int i, int j, int k, int component) noexcept {
               const bool physical_boundary =
                  (!periodic[0]
                   && (i == nodal_domain.smallEnd(0)
                       || i == nodal_domain.bigEnd(0)))
#if (AMREX_SPACEDIM >= 2)
                  || (!periodic[1]
                      && (j == nodal_domain.smallEnd(1)
                          || j == nodal_domain.bigEnd(1)))
#endif
#if (AMREX_SPACEDIM == 3)
                  || (!periodic[2]
                      && (k == nodal_domain.smallEnd(2)
                          || k == nodal_domain.bigEnd(2)))
#endif
                  ;
               if (physical_boundary) {
                  int boundary_i = i;
                  int boundary_j = j;
                  int boundary_k = k;
                  if (!periodic[0] && i == nodal_domain.smallEnd(0)) {
                     --boundary_i;
                  } else if (!periodic[0] && i == nodal_domain.bigEnd(0)) {
                     ++boundary_i;
                  }
#if (AMREX_SPACEDIM >= 2)
                  else if (!periodic[1] && j == nodal_domain.smallEnd(1)) {
                     --boundary_j;
                  } else if (!periodic[1] && j == nodal_domain.bigEnd(1)) {
                     ++boundary_j;
                  }
#endif
#if (AMREX_SPACEDIM == 3)
                  else if (!periodic[2] && k == nodal_domain.smallEnd(2)) {
                     --boundary_k;
                  } else if (!periodic[2] && k == nodal_domain.bigEnd(2)) {
                     ++boundary_k;
                  }
#endif
                  const amrex::Real a = coefficient_a(
                     boundary_i, boundary_j, boundary_k, component
                  );
                  values(i, j, k, component) = a != 0.0
                     ? coefficient_c(
                          boundary_i, boundary_j, boundary_k, component
                       ) / a
                     : std::numeric_limits<amrex::Real>::quiet_NaN();
               }
            }
         );
      }
      if (solution[level]->contains_nan(0, num_components, 0)) {
         amrex::Abort("nodal Dirichlet coefficients require nonzero A");
      }
      solution[level]->FillBoundary(hierarchy.geometry[level].periodicity());
   }
}

amrex::Real compute_nodal_robin_residual(
   InitialDataEllipticHierarchy const& hierarchy,
   amrex::Vector<amrex::MultiFab const*> const& solution,
   amrex::Vector<amrex::MultiFab*> const& residual,
   InitialDataEllipticBoundaryConditions const& boundary,
   bool include_inhomogeneous_term = true
)
{
   const int num_levels = boundary.num_levels();
   amrex::Real max_residual = 0.0;
   for (int level = 0; level < num_levels; ++level) {
      residual[level]->setVal(0.0);

      const amrex::Box nodal_domain = amrex::surroundingNodes(
         hierarchy.geometry[level].Domain()
      );
      const auto periodic = hierarchy.geometry[level].isPeriodicArray();
      const auto dx = hierarchy.geometry[level].CellSizeArray();
      for (amrex::MFIter mfi(*residual[level], amrex::TilingIfNotGPU());
           mfi.isValid(); ++mfi) {
         const amrex::Box box = mfi.tilebox();
         const auto coefficient_a = boundary.coefficient_a(level).const_array(mfi);
         const auto coefficient_b = boundary.coefficient_b(level).const_array(mfi);
         const auto coefficient_c = boundary.coefficient_c(level).const_array(mfi);
         const auto current_solution = solution[level]->const_array(mfi);
         const auto boundary_residual = residual[level]->array(mfi);
         amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            amrex::Real candidate_sum = 0.0;
            int candidate_count = 0;
            auto add_face = [=, &candidate_sum, &candidate_count] AMREX_GPU_DEVICE(
               int boundary_i,
               int boundary_j,
               int boundary_k,
               int direction,
               int interior_sign
            ) noexcept {
               const amrex::Real a = coefficient_a(
                  boundary_i, boundary_j, boundary_k, 0
               );
               const amrex::Real b = coefficient_b(
                  boundary_i, boundary_j, boundary_k, 0
               );
               const amrex::Real c = include_inhomogeneous_term
                  ? coefficient_c(boundary_i, boundary_j, boundary_k, 0)
                  : 0.0;
               int near_i = i;
               int near_j = j;
               int near_k = k;
               int far_i = i;
               int far_j = j;
               int far_k = k;
               if (direction == 0) {
                  near_i += interior_sign;
                  far_i += 2 * interior_sign;
               }
#if (AMREX_SPACEDIM >= 2)
               else if (direction == 1) {
                  near_j += interior_sign;
                  far_j += 2 * interior_sign;
               }
#endif
#if (AMREX_SPACEDIM == 3)
               else {
                  near_k += interior_sign;
                  far_k += 2 * interior_sign;
               }
#endif
               const amrex::Real derivative_sign = -interior_sign;
               const amrex::Real derivative_scale = derivative_sign * b
                  / (2.0 * dx[direction]);
               const amrex::Real denominator = a + 3.0 * derivative_scale;
               const amrex::Real numerator = c
                  + 4.0 * derivative_scale
                     * current_solution(near_i, near_j, near_k, 0)
                  - derivative_scale
                     * current_solution(far_i, far_j, far_k, 0);
               candidate_sum += denominator != 0.0
                  ? numerator / denominator
                  : std::numeric_limits<amrex::Real>::quiet_NaN();
               ++candidate_count;
            };

            if (!periodic[0] && i == nodal_domain.smallEnd(0)) {
               add_face(i - 1, j, k, 0, 1);
            }
            if (!periodic[0] && i == nodal_domain.bigEnd(0)) {
               add_face(i + 1, j, k, 0, -1);
            }
#if (AMREX_SPACEDIM >= 2)
            if (!periodic[1] && j == nodal_domain.smallEnd(1)) {
               add_face(i, j - 1, k, 1, 1);
            }
            if (!periodic[1] && j == nodal_domain.bigEnd(1)) {
               add_face(i, j + 1, k, 1, -1);
            }
#endif
#if (AMREX_SPACEDIM == 3)
            if (!periodic[2] && k == nodal_domain.smallEnd(2)) {
               add_face(i, j, k - 1, 2, 1);
            }
            if (!periodic[2] && k == nodal_domain.bigEnd(2)) {
               add_face(i, j, k + 1, 2, -1);
            }
#endif
            if (candidate_count > 0) {
               const amrex::Real candidate = candidate_sum / candidate_count;
               boundary_residual(i, j, k, 0) =
                  candidate - current_solution(i, j, k, 0);
            }
         });
      }
      if (residual[level]->contains_nan(0, 1, 0)) {
         amrex::Abort(
            "nodal Robin boundary produced a singular boundary stencil"
         );
      }
      max_residual = std::max(
         max_residual, residual[level]->norm0(0, 0, false)
      );
   }
   return max_residual;
}

struct NodalBoundaryVector {
   amrex::Vector<amrex::MultiFab> levels;
};

class NodalRobinSchurOperator {
public:
   using RT = amrex::Real;

   NodalRobinSchurOperator(
      InitialDataEllipticHierarchy const& hierarchy,
      amrex::Vector<amrex::MultiFab*> const& layout,
      amrex::Vector<amrex::MultiFab const*> const& physical_rhs,
      InitialDataEllipticBoundaryConditions const& boundary,
      InitialDataEllipticOptions const& options,
      amrex::MLMG& mlmg
   )
      : hierarchy_(hierarchy),
        layout_(layout),
        physical_rhs_(physical_rhs),
        boundary_(boundary),
        options_(options),
        mlmg_(mlmg),
        homogeneous_solution_(make_vector()),
        base_solution_(make_vector())
   {
      for (int level = 0; level < static_cast<int>(layout_.size()); ++level) {
         amrex::MultiFab::Copy(
            base_solution_.levels[level], *layout_[level], 0, 0, 1, 0
         );
      }
      setToZero(homogeneous_solution_);
   }

   NodalBoundaryVector makeVecRHS() const { return make_vector(); }
   NodalBoundaryVector makeVecLHS() const { return make_vector(); }

   void apply(NodalBoundaryVector& lhs, NodalBoundaryVector const& rhs)
   {
      assign(homogeneous_solution_, base_solution_);
      increment(homogeneous_solution_, rhs, 1.0);
      const int num_levels = static_cast<int>(layout_.size());
      amrex::Vector<amrex::MultiFab*> solve_fields(num_levels);
      amrex::Vector<amrex::MultiFab const*> solve_rhs(num_levels);
      for (int level = 0; level < num_levels; ++level) {
         solve_fields[level] = &homogeneous_solution_.levels[level];
         solve_rhs[level] = physical_rhs_[level];
      }
      last_residual_ = mlmg_.solve(
         solve_fields,
         solve_rhs,
         options_.relative_tolerance,
         options_.absolute_tolerance
      );
      increment(homogeneous_solution_, base_solution_, -1.0);

      amrex::Vector<amrex::MultiFab const*> const_solution(num_levels);
      amrex::Vector<amrex::MultiFab*> residual(num_levels);
      for (int level = 0; level < num_levels; ++level) {
         const_solution[level] = &homogeneous_solution_.levels[level];
         residual[level] = &lhs.levels[level];
      }
      compute_nodal_robin_residual(
         hierarchy_, const_solution, residual, boundary_, false
      );
      scale(lhs, -1.0);
   }

   void add_last_homogeneous_response(
      amrex::Vector<amrex::MultiFab*> const& solution
   ) const
   {
      for (int level = 0; level < static_cast<int>(solution.size()); ++level) {
         amrex::MultiFab::Add(
            *solution[level], homogeneous_solution_.levels[level], 0, 0, 1, 0
         );
      }
   }

   RT last_residual() const noexcept { return last_residual_; }

   void assign(NodalBoundaryVector& lhs, NodalBoundaryVector const& rhs) const
   {
      for (int level = 0; level < static_cast<int>(lhs.levels.size()); ++level) {
         amrex::MultiFab::Copy(
            lhs.levels[level], rhs.levels[level], 0, 0, 1, 0
         );
      }
   }

   RT dotProduct(
      NodalBoundaryVector const& first,
      NodalBoundaryVector const& second
   ) const
   {
      RT result = 0.0;
      for (int level = 0; level < static_cast<int>(first.levels.size()); ++level) {
         result += amrex::MultiFab::Dot(
            first.levels[level], 0, second.levels[level], 0, 1, 0, false
         );
      }
      return result;
   }

   void increment(
      NodalBoundaryVector& lhs,
      NodalBoundaryVector const& rhs,
      RT coefficient
   ) const
   {
      for (int level = 0; level < static_cast<int>(lhs.levels.size()); ++level) {
         amrex::MultiFab::Saxpy(
            lhs.levels[level], coefficient, rhs.levels[level], 0, 0, 1, 0
         );
      }
   }

   void linComb(
      NodalBoundaryVector& lhs,
      RT first_coefficient,
      NodalBoundaryVector const& first,
      RT second_coefficient,
      NodalBoundaryVector const& second
   ) const
   {
      for (int level = 0; level < static_cast<int>(lhs.levels.size()); ++level) {
         amrex::MultiFab::LinComb(
            lhs.levels[level],
            first_coefficient,
            first.levels[level],
            0,
            second_coefficient,
            second.levels[level],
            0,
            0,
            1,
            0
         );
      }
   }

   RT norm2(NodalBoundaryVector const& vector) const
   {
      return std::sqrt(std::max(RT(0.0), dotProduct(vector, vector)));
   }

   void precond(
      NodalBoundaryVector& lhs,
      NodalBoundaryVector const& rhs
   ) const
   {
      assign(lhs, rhs);
   }

   void scale(NodalBoundaryVector& vector, RT coefficient) const
   {
      for (amrex::MultiFab& level : vector.levels) {
         level.mult(coefficient, 0, 1, 0);
      }
   }

   void setToZero(NodalBoundaryVector& vector) const
   {
      for (amrex::MultiFab& level : vector.levels) {
         level.setVal(0.0);
      }
   }

private:
   NodalBoundaryVector make_vector() const
   {
      NodalBoundaryVector result;
      result.levels.reserve(layout_.size());
      for (amrex::MultiFab const* level_layout : layout_) {
         result.levels.emplace_back(
            level_layout->boxArray(), level_layout->DistributionMap(), 1, 0
         );
         result.levels.back().setVal(0.0);
      }
      return result;
   }

   InitialDataEllipticHierarchy const& hierarchy_;
   amrex::Vector<amrex::MultiFab*> const& layout_;
   amrex::Vector<amrex::MultiFab const*> const& physical_rhs_;
   InitialDataEllipticBoundaryConditions const& boundary_;
   InitialDataEllipticOptions const& options_;
   amrex::MLMG& mlmg_;
   NodalBoundaryVector homogeneous_solution_;
   NodalBoundaryVector base_solution_;
   RT last_residual_ = 0.0;
};

InitialDataEllipticResult solve_initial_data_nodal_poisson(
   InitialDataEllipticHierarchy const& hierarchy,
   amrex::Vector<amrex::MultiFab*> const& solution,
   amrex::Vector<amrex::MultiFab const*> const& rhs,
   InitialDataEllipticBoundaryConditions const& boundary,
   InitialDataEllipticOptions const& options
)
{
   amrex::LPInfo info;
   if (has_zero_derivative_coefficient(boundary)) {
      set_nodal_dirichlet_values(hierarchy, solution, boundary);
      amrex::MLNodeLaplacian poisson(
         hierarchy.geometry,
         hierarchy.grids,
         hierarchy.distribution,
         info,
         {},
         1.0
      );
      const BoundaryTypeArray domain_types = domain_boundary_types(
         hierarchy.geometry[0], BoundaryRepresentation::dirichlet
      );
      poisson.setDomainBC(domain_types, domain_types);

      amrex::MLMG mlmg(poisson);
      configure_mlmg(mlmg, options);
      const amrex::Real residual = mlmg.solve(
         solution,
         rhs,
         options.relative_tolerance,
         options.absolute_tolerance
      );
      synchronize_nodal_solution(hierarchy, solution);
      return elliptic_result(residual);
   }

   amrex::MLNodeLaplacian poisson(
      hierarchy.geometry,
      hierarchy.grids,
      hierarchy.distribution,
      info,
      {},
      1.0
   );
   const BoundaryTypeArray domain_types = domain_boundary_types(
      hierarchy.geometry[0], BoundaryRepresentation::dirichlet
   );
   poisson.setDomainBC(domain_types, domain_types);

   amrex::MLMG mlmg(poisson);
   configure_mlmg(mlmg, options);
   amrex::Real residual = mlmg.solve(
      solution, rhs, options.relative_tolerance, options.absolute_tolerance
   );
   const int num_levels = boundary.num_levels();
   NodalRobinSchurOperator boundary_operator(
      hierarchy, solution, rhs, boundary, options, mlmg
   );
   NodalBoundaryVector boundary_rhs = boundary_operator.makeVecRHS();
   amrex::Vector<amrex::MultiFab const*> const_solution(num_levels);
   amrex::Vector<amrex::MultiFab*> boundary_residual_fields(num_levels);
   amrex::Real solution_scale = 1.0;
   for (int level = 0; level < num_levels; ++level) {
      const_solution[level] = solution[level];
      boundary_residual_fields[level] = &boundary_rhs.levels[level];
      solution_scale = std::max(
         solution_scale, solution[level]->norm0(0, 0, false)
      );
   }
   const amrex::Real initial_boundary_residual = compute_nodal_robin_residual(
      hierarchy, const_solution, boundary_residual_fields, boundary
   );
   const amrex::Real boundary_tolerance = std::max(
      options.absolute_tolerance,
      options.relative_tolerance * solution_scale
   );
   if (initial_boundary_residual <= boundary_tolerance) {
      synchronize_nodal_solution(hierarchy, solution);
      return {residual, initial_boundary_residual, 0};
   }

   NodalBoundaryVector boundary_solution = boundary_operator.makeVecLHS();
   amrex::GMRES<NodalBoundaryVector, NodalRobinSchurOperator> gmres;
   gmres.define(boundary_operator);
   gmres.setRestartLength(16);
   gmres.setMaxIters(options.max_iterations);
   gmres.setVerbose(options.verbose);
   gmres.solve(
      boundary_solution,
      boundary_rhs,
      0.0,
      boundary_tolerance,
      options.max_iterations
   );
   if (gmres.getStatus() != 0) {
      amrex::Print() << "nodal Robin GMRES residual="
                     << gmres.getResidualNorm() << " tolerance="
                     << boundary_tolerance << '\n';
      amrex::Abort("nodal Robin boundary iteration did not converge");
   }

   NodalBoundaryVector schur_check = boundary_operator.makeVecRHS();
   boundary_operator.apply(schur_check, boundary_solution);
   boundary_operator.increment(schur_check, boundary_rhs, -1.0);
   const amrex::Real schur_residual = boundary_operator.norm2(schur_check);
   if (schur_residual > boundary_tolerance) {
      amrex::Print() << "nodal Robin Schur residual=" << schur_residual
                     << " tolerance=" << boundary_tolerance << '\n';
      amrex::Abort("nodal Robin boundary solve failed its Schur residual check");
   }
   residual = std::max(residual, boundary_operator.last_residual());
   boundary_operator.add_last_homogeneous_response(solution);
   for (int level = 0; level < num_levels; ++level) {
      const_solution[level] = solution[level];
   }
   const amrex::Real final_boundary_residual = compute_nodal_robin_residual(
      hierarchy, const_solution, boundary_residual_fields, boundary
   );
   if (final_boundary_residual > boundary_tolerance) {
      amrex::Print() << "nodal Robin final boundary residual="
                     << final_boundary_residual << " tolerance="
                     << boundary_tolerance << " Schur residual="
                     << schur_residual << '\n';
      amrex::Abort("nodal Robin boundary solve failed its residual check");
   }
   synchronize_nodal_solution(hierarchy, solution);
   return {residual, final_boundary_residual, gmres.getNumIters()};
}

InitialDataEllipticResult solve_initial_data_nodal_conformal_vector_laplacian(
   InitialDataEllipticHierarchy const& hierarchy,
   amrex::Vector<amrex::MultiFab*> const& vector_potential,
   amrex::Vector<amrex::MultiFab const*> const& source,
   InitialDataEllipticBoundaryConditions const& boundary,
   InitialDataEllipticOptions const& options
)
{
#if (AMREX_SPACEDIM != 3)
   amrex::ignore_unused(
      hierarchy, vector_potential, source, boundary, options
   );
   amrex::Abort("the CTT conformal-vector operator requires three dimensions");
   return {};
#else
   const int num_levels = boundary.num_levels();
   amrex::Vector<std::unique_ptr<amrex::MultiFab>> corrected_storage(num_levels);
   amrex::Vector<std::unique_ptr<amrex::MultiFab>> candidate_storage(num_levels);
   amrex::Vector<std::unique_ptr<amrex::MultiFab>> scalar_solution_storage(num_levels);
   amrex::Vector<std::unique_ptr<amrex::MultiFab>> scalar_rhs_storage(num_levels);
   amrex::Vector<amrex::MultiFab*> candidate(num_levels);
   amrex::Vector<amrex::MultiFab*> scalar_solution(num_levels);
   amrex::Vector<amrex::MultiFab const*> scalar_rhs(num_levels);
   for (int level = 0; level < num_levels; ++level) {
      corrected_storage[level] = std::make_unique<amrex::MultiFab>(
         vector_potential[level]->boxArray(),
         vector_potential[level]->DistributionMap(),
         3,
         0
      );
      candidate_storage[level] = std::make_unique<amrex::MultiFab>(
         vector_potential[level]->boxArray(),
         vector_potential[level]->DistributionMap(),
         3,
         1
      );
      scalar_solution_storage[level] = std::make_unique<amrex::MultiFab>(
         vector_potential[level]->boxArray(),
         vector_potential[level]->DistributionMap(),
         1,
         1
      );
      scalar_rhs_storage[level] = std::make_unique<amrex::MultiFab>(
         vector_potential[level]->boxArray(),
         vector_potential[level]->DistributionMap(),
         1,
         0
      );
      candidate[level] = candidate_storage[level].get();
      scalar_solution[level] = scalar_solution_storage[level].get();
      scalar_rhs[level] = scalar_rhs_storage[level].get();
   }

   amrex::Vector<amrex::BCRec> boundary_records(3);
   for (amrex::BCRec& record : boundary_records) {
      for (int direction = 0; direction < 3; ++direction) {
         const int type = hierarchy.geometry[0].isPeriodic(direction)
            ? amrex::BCType::int_dir
            : amrex::BCType::foextrap;
         record.setLo(direction, type);
         record.setHi(direction, type);
      }
   }

   InitialDataEllipticResult result;
   constexpr amrex::Real conformal_vector_relaxation = 0.5;
   for (int iteration = 0; iteration < options.max_iterations; ++iteration) {
      detail::fill_initial_data_vertex_ghosts(
         hierarchy, vector_potential, boundary_records
      );
      for (int level = 0; level < num_levels; ++level) {
         amrex::MultiFab::Copy(
            *corrected_storage[level], *source[level], 0, 0, 3, 0
         );
         const auto spacing = hierarchy.geometry[level].CellSizeArray();
         const amrex::GpuArray<amrex::Real, 3> inv_spacing{{
            1.0 / spacing[0], 1.0 / spacing[1], 1.0 / spacing[2]
         }};
         const amrex::Box derivative_domain = amrex::grow(
            amrex::surroundingNodes(hierarchy.geometry[level].Domain()), -1
         );
         for (amrex::MFIter mfi(
                 *corrected_storage[level], amrex::TilingIfNotGPU()
              ); mfi.isValid(); ++mfi) {
            const amrex::Box box = mfi.tilebox() & derivative_domain;
            if (!box.ok()) {
               continue;
            }
            const auto vector = vector_potential[level]->const_array(mfi);
            const auto corrected = corrected_storage[level]->array(mfi);
            amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
               const amrex::Real grad_div_1 =
                  generated::centered_second<2, 0>(
                     vector, i, j, k, 0, inv_spacing[0] * inv_spacing[0]
                  )
                  + generated::centered_mixed<2, 0, 1>(
                     vector, i, j, k, 1, inv_spacing[0] * inv_spacing[1]
                  )
                  + generated::centered_mixed<2, 0, 2>(
                     vector, i, j, k, 2, inv_spacing[0] * inv_spacing[2]
                  );
               const amrex::Real grad_div_2 =
                  generated::centered_mixed<2, 0, 1>(
                     vector, i, j, k, 0, inv_spacing[0] * inv_spacing[1]
                  )
                  + generated::centered_second<2, 1>(
                     vector, i, j, k, 1, inv_spacing[1] * inv_spacing[1]
                  )
                  + generated::centered_mixed<2, 1, 2>(
                     vector, i, j, k, 2, inv_spacing[1] * inv_spacing[2]
                  );
               const amrex::Real grad_div_3 =
                  generated::centered_mixed<2, 0, 2>(
                     vector, i, j, k, 0, inv_spacing[0] * inv_spacing[2]
                  )
                  + generated::centered_mixed<2, 1, 2>(
                     vector, i, j, k, 1, inv_spacing[1] * inv_spacing[2]
                  )
                  + generated::centered_second<2, 2>(
                     vector, i, j, k, 2, inv_spacing[2] * inv_spacing[2]
                  );
               corrected(i, j, k, 0) -= grad_div_1 / 3.0;
               corrected(i, j, k, 1) -= grad_div_2 / 3.0;
               corrected(i, j, k, 2) -= grad_div_3 / 3.0;
            });
         }
      }

      result.residual = 0.0;
      result.boundary_residual = 0.0;
      result.boundary_iterations = 0;
      for (int component = 0; component < 3; ++component) {
         for (int level = 0; level < num_levels; ++level) {
            amrex::MultiFab::Copy(
               *scalar_solution[level],
               *vector_potential[level],
               component,
               0,
               1,
               1
            );
            amrex::MultiFab::Copy(
               *scalar_rhs_storage[level],
               *corrected_storage[level],
               component,
               0,
               1,
               0
            );
         }
         const InitialDataEllipticResult component_result =
            solve_initial_data_poisson(
               hierarchy,
               scalar_solution,
               scalar_rhs,
               boundary.component(component),
               options
            );
         result.residual = std::max(
            result.residual, component_result.residual
         );
         result.boundary_residual = std::max(
            result.boundary_residual, component_result.boundary_residual
         );
         result.boundary_iterations = std::max(
            result.boundary_iterations, component_result.boundary_iterations
         );
         for (int level = 0; level < num_levels; ++level) {
            amrex::MultiFab::Copy(
               *candidate_storage[level],
               *scalar_solution[level],
               0,
               component,
               1,
               1
            );
         }
      }
      for (int level = num_levels - 1; level > 0; --level) {
         amrex::average_down_nodal(
            *candidate_storage[level],
            *candidate_storage[level - 1],
            hierarchy.refinement_ratio[level - 1]
         );
      }
      detail::fill_initial_data_vertex_ghosts(
         hierarchy, candidate, boundary_records
      );

      amrex::Real update_norm = 0.0;
      amrex::Real solution_norm = 1.0;
      for (int level = 0; level < num_levels; ++level) {
         amrex::MultiFab update(
            vector_potential[level]->boxArray(),
            vector_potential[level]->DistributionMap(),
            3,
            0
         );
         amrex::MultiFab::Copy(
            update, *candidate_storage[level], 0, 0, 3, 0
         );
         amrex::MultiFab::Subtract(
            update, *vector_potential[level], 0, 0, 3, 0
         );
         amrex::MultiFab::Saxpy(
            *vector_potential[level],
            conformal_vector_relaxation,
            update,
            0,
            0,
            3,
            0
         );
         update.mult(conformal_vector_relaxation, 0, 3, 0);
         for (int component = 0; component < 3; ++component) {
            update_norm = std::max(
               update_norm, update.norm0(component, 0, false)
            );
            solution_norm = std::max(
               solution_norm,
               vector_potential[level]->norm0(component, 0, false)
            );
         }
      }
      if (update_norm <= options.relative_tolerance * solution_norm) {
         result.residual = update_norm / solution_norm;
         return result;
      }
   }
   amrex::Abort("nodal CTT conformal-vector iteration did not converge");
   return {};
#endif
}

} // namespace

void fill_initial_data_ghosts(
   InitialDataEllipticHierarchy const& hierarchy,
   amrex::Vector<amrex::MultiFab*> const& fields,
   amrex::Vector<amrex::BCRec> const& boundary_conditions
)
{
   if (!fields.empty() && fields.front() != nullptr
       && fields.front()->ixType().nodeCentered()) {
      detail::fill_initial_data_vertex_ghosts(
         hierarchy, fields, boundary_conditions
      );
      return;
   }
   const int num_levels = static_cast<int>(hierarchy.geometry.size());
   if (num_levels == 0
       || static_cast<int>(hierarchy.grids.size()) != num_levels
       || static_cast<int>(hierarchy.distribution.size()) != num_levels
       || static_cast<int>(hierarchy.refinement_ratio.size()) != num_levels - 1
       || static_cast<int>(fields.size()) != num_levels
       || hierarchy.cell_interpolater == nullptr) {
      amrex::Abort("invalid initial-data hierarchy ghost fill");
   }
   const int num_components = fields.front() != nullptr
      ? fields.front()->nComp()
      : 0;
   if (num_components <= 0
       || static_cast<int>(boundary_conditions.size()) != num_components) {
      amrex::Abort("invalid initial-data boundary component layout");
   }

   InitialDataBoundaryFunction boundary_function(amrex::FabFillNoOp{});
   for (int level = 0; level < num_levels; ++level) {
      if (fields[level] == nullptr
          || fields[level]->nComp() != num_components
          || fields[level]->nGrow() < 1
          || fields[level]->boxArray() != hierarchy.grids[level]) {
         amrex::Abort("incompatible initial-data field hierarchy");
      }
      InitialDataPhysicalBoundary fine_boundary(
         hierarchy.geometry[level], boundary_conditions, boundary_function
      );
      amrex::Vector<amrex::MultiFab*> fine_data{fields[level]};
      amrex::Vector<amrex::Real> fine_time{0.0};
      if (level == 0) {
         amrex::FillPatchSingleLevel(
            *fields[level],
            fields[level]->nGrowVect(),
            0.0,
            fine_data,
            fine_time,
            0,
            0,
            num_components,
            hierarchy.geometry[level],
            fine_boundary,
            0
         );
         continue;
      }

      InitialDataPhysicalBoundary coarse_boundary(
         hierarchy.geometry[level - 1], boundary_conditions, boundary_function
      );
      amrex::Vector<amrex::MultiFab*> coarse_data{fields[level - 1]};
      amrex::Vector<amrex::Real> coarse_time{0.0};
      amrex::FillPatchTwoLevels(
         *fields[level],
         fields[level]->nGrowVect(),
         0.0,
         coarse_data,
         coarse_time,
         fine_data,
         fine_time,
         0,
         0,
         num_components,
         hierarchy.geometry[level - 1],
         hierarchy.geometry[level],
         coarse_boundary,
         0,
         fine_boundary,
         0,
         hierarchy.refinement_ratio[level - 1],
         hierarchy.cell_interpolater,
         boundary_conditions,
         0
      );
   }
}

InitialDataEllipticBoundaryConditions InitialDataEllipticBoundaryConditions::constant(
   InitialDataEllipticHierarchy const& hierarchy,
   int num_components,
   amrex::Real coefficient_a,
   amrex::Real coefficient_b,
   amrex::Real coefficient_c,
   InitialDataEllipticCentering centering
)
{
   const int num_levels = static_cast<int>(hierarchy.geometry.size());
   if (
      num_levels == 0
      || static_cast<int>(hierarchy.grids.size()) != num_levels
      || static_cast<int>(hierarchy.distribution.size()) != num_levels
      || num_components <= 0
   ) {
      amrex::Abort("invalid hierarchy for initial-data elliptic boundary coefficients");
   }
   InitialDataEllipticBoundaryConditions boundary;
   boundary.num_components = num_components;
   boundary.centering_ = centering;
   boundary.coefficient_a_.resize(num_levels);
   boundary.coefficient_b_.resize(num_levels);
   boundary.coefficient_c_.resize(num_levels);
   for (int level = 0; level < num_levels; ++level) {
      boundary.coefficient_a_[level] = std::make_unique<amrex::MultiFab>(
         elliptic_box_array(hierarchy, level, centering),
         hierarchy.distribution[level], num_components, 1
      );
      boundary.coefficient_b_[level] = std::make_unique<amrex::MultiFab>(
         elliptic_box_array(hierarchy, level, centering),
         hierarchy.distribution[level], num_components, 1
      );
      boundary.coefficient_c_[level] = std::make_unique<amrex::MultiFab>(
         elliptic_box_array(hierarchy, level, centering),
         hierarchy.distribution[level], num_components, 1
      );
      boundary.coefficient_a_[level]->setVal(coefficient_a);
      boundary.coefficient_b_[level]->setVal(coefficient_b);
      boundary.coefficient_c_[level]->setVal(coefficient_c);
   }
   return boundary;
}

amrex::MultiFab& InitialDataEllipticBoundaryConditions::coefficient_a(
   int level
) noexcept
{
   return *coefficient_a_[level];
}

amrex::MultiFab& InitialDataEllipticBoundaryConditions::coefficient_b(
   int level
) noexcept
{
   return *coefficient_b_[level];
}

amrex::MultiFab& InitialDataEllipticBoundaryConditions::coefficient_c(
   int level
) noexcept
{
   return *coefficient_c_[level];
}

amrex::MultiFab const& InitialDataEllipticBoundaryConditions::coefficient_a(
   int level
) const noexcept
{
   return *coefficient_a_[level];
}

amrex::MultiFab const& InitialDataEllipticBoundaryConditions::coefficient_b(
   int level
) const noexcept
{
   return *coefficient_b_[level];
}

amrex::MultiFab const& InitialDataEllipticBoundaryConditions::coefficient_c(
   int level
) const noexcept
{
   return *coefficient_c_[level];
}

int InitialDataEllipticBoundaryConditions::num_levels() const noexcept
{
   return static_cast<int>(coefficient_a_.size());
}

InitialDataEllipticCentering
InitialDataEllipticBoundaryConditions::centering() const noexcept
{
   return centering_;
}

InitialDataEllipticBoundaryConditions
InitialDataEllipticBoundaryConditions::component(int component) const
{
   if (component < 0 || component >= num_components) {
      amrex::Abort("invalid initial-data elliptic boundary component");
   }
   InitialDataEllipticBoundaryConditions result;
   result.num_components = 1;
   result.centering_ = centering_;
   const int num_levels = this->num_levels();
   result.coefficient_a_.resize(num_levels);
   result.coefficient_b_.resize(num_levels);
   result.coefficient_c_.resize(num_levels);
   for (int level = 0; level < num_levels; ++level) {
      auto make_component = [component](amrex::MultiFab const& source) {
         auto destination = std::make_unique<amrex::MultiFab>(
            source.boxArray(), source.DistributionMap(), 1, source.nGrow()
         );
         amrex::MultiFab::Copy(
            *destination, source, component, 0, 1, source.nGrow()
         );
         return destination;
      };
      result.coefficient_a_[level] = make_component(coefficient_a(level));
      result.coefficient_b_[level] = make_component(coefficient_b(level));
      result.coefficient_c_[level] = make_component(coefficient_c(level));
   }
   return result;
}

void set_centered_power_law_robin_boundary(
   InitialDataEllipticHierarchy const& hierarchy,
   InitialDataEllipticBoundaryConditions& boundary,
   amrex::Vector<amrex::Real> const& asymptotic_values,
   amrex::Real falloff_power
)
{
   if (
      !(falloff_power > 0.0)
      || boundary.num_levels() != static_cast<int>(hierarchy.geometry.size())
      || static_cast<int>(asymptotic_values.size()) != boundary.num_components
   ) {
      amrex::Abort("invalid centered power-law Robin boundary configuration");
   }
   for (int level = 0; level < boundary.num_levels(); ++level) {
      const bool vertex_centered =
         boundary.centering() == InitialDataEllipticCentering::vertex;
      const auto problo = hierarchy.geometry[level].ProbLoArray();
      const auto probhi = hierarchy.geometry[level].ProbHiArray();
      const auto dx = hierarchy.geometry[level].CellSizeArray();
      amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> center;
      amrex::GpuArray<int, AMREX_SPACEDIM> periodic;
      for (int direction = 0; direction < AMREX_SPACEDIM; ++direction) {
         center[direction] = 0.5 * (problo[direction] + probhi[direction]);
         periodic[direction] = hierarchy.geometry[level].isPeriodic(direction);
      }
      const amrex::Box coefficient_domain = vertex_centered
         ? amrex::surroundingNodes(hierarchy.geometry[level].Domain())
         : hierarchy.geometry[level].Domain();
      const amrex::Dim3 domain_lo = amrex::lbound(coefficient_domain);
      const amrex::Dim3 domain_hi = amrex::ubound(coefficient_domain);
      const amrex::Real coordinate_offset = vertex_centered ? 0.0 : 0.5;
      for (int component = 0; component < boundary.num_components; ++component) {
         const amrex::Real asymptotic_value = asymptotic_values[component];
         for (amrex::MFIter mfi(boundary.coefficient_a(level)); mfi.isValid(); ++mfi) {
            const amrex::Box box = mfi.fabbox();
            const auto coefficient_a = boundary.coefficient_a(level).array(mfi);
            const auto coefficient_b = boundary.coefficient_b(level).array(mfi);
            const auto coefficient_c = boundary.coefficient_c(level).array(mfi);
            amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
               int face_direction = -1;
               amrex::Real position_1 = problo[0]
                  + (i + coordinate_offset) * dx[0] - center[0];
#if (AMREX_SPACEDIM >= 2)
               amrex::Real position_2 = problo[1]
                  + (j + coordinate_offset) * dx[1] - center[1];
#else
               amrex::Real position_2 = 0.0;
#endif
#if (AMREX_SPACEDIM == 3)
               amrex::Real position_3 = problo[2]
                  + (k + coordinate_offset) * dx[2] - center[2];
#else
               amrex::Real position_3 = 0.0;
#endif
               if (!periodic[0] && i < domain_lo.x) {
                  face_direction = 0;
                  position_1 = problo[0] - center[0];
               } else if (!periodic[0] && i > domain_hi.x) {
                  face_direction = 0;
                  position_1 = probhi[0] - center[0];
               }
#if (AMREX_SPACEDIM >= 2)
               else if (!periodic[1] && j < domain_lo.y) {
                  face_direction = 1;
                  position_2 = problo[1] - center[1];
               } else if (!periodic[1] && j > domain_hi.y) {
                  face_direction = 1;
                  position_2 = probhi[1] - center[1];
               }
#endif
#if (AMREX_SPACEDIM == 3)
               else if (!periodic[2] && k < domain_lo.z) {
                  face_direction = 2;
                  position_3 = problo[2] - center[2];
               } else if (!periodic[2] && k > domain_hi.z) {
                  face_direction = 2;
                  position_3 = probhi[2] - center[2];
               }
#endif
               if (face_direction >= 0) {
                  const amrex::Real radius_squared =
                     position_1 * position_1
                     + position_2 * position_2
                     + position_3 * position_3;
                  amrex::Real face_position = position_1;
#if (AMREX_SPACEDIM >= 2)
                  if (face_direction == 1) {
                     face_position = position_2;
                  }
#endif
#if (AMREX_SPACEDIM == 3)
                  if (face_direction == 2) {
                     face_position = position_3;
                  }
#endif
                  coefficient_a(i, j, k, component) = falloff_power;
                  coefficient_b(i, j, k, component) = face_position != 0.0
                     ? radius_squared / face_position
                     : std::numeric_limits<amrex::Real>::quiet_NaN();
                  coefficient_c(i, j, k, component) =
                     falloff_power * asymptotic_value;
               }
            });
         }
      }
      if (boundary.coefficient_b(level).contains_nan(
             0, boundary.num_components, 1
          )) {
         amrex::Abort(
            "centered power-law Robin boundary requires faces away from its center"
         );
      }
   }
}

InitialDataEllipticResult solve_initial_data_poisson(
   InitialDataEllipticHierarchy const& hierarchy,
   amrex::Vector<amrex::MultiFab*> const& solution,
   amrex::Vector<amrex::MultiFab const*> const& rhs,
   InitialDataEllipticBoundaryConditions const& boundary,
   InitialDataEllipticOptions const& options
)
{
   const int num_levels = validate_elliptic_solve(
      hierarchy, solution, rhs, boundary, options, 1
   );
   if (boundary.centering() == InitialDataEllipticCentering::vertex) {
      return solve_initial_data_nodal_poisson(
         hierarchy, solution, rhs, boundary, options
      );
   }

   amrex::LPInfo info;
   amrex::MLABecLaplacian poisson(
      hierarchy.geometry,
      hierarchy.grids,
      hierarchy.distribution,
      info
   );
   poisson.setMaxOrder(options.max_order);
   poisson.setScalars(0.0, -1.0);
   const LoweredBoundaryConditions lowered = lower_boundary_conditions(
      hierarchy, boundary
   );
   for (int level = 0; level < num_levels; ++level) {
      poisson.setBCoeffs(level, 1.0);
   }
   set_boundary_conditions(poisson, boundary, lowered);

   amrex::MLMG mlmg(poisson);
   configure_mlmg(mlmg, options);
   return elliptic_result(mlmg.solve(
      solution,
      rhs,
      options.relative_tolerance,
      options.absolute_tolerance
   ));
}

InitialDataEllipticResult solve_initial_data_conformal_vector_laplacian(
   InitialDataEllipticHierarchy const& hierarchy,
   amrex::Vector<amrex::MultiFab*> const& vector_potential,
   amrex::Vector<amrex::MultiFab const*> const& source,
   InitialDataEllipticBoundaryConditions const& boundary,
   InitialDataEllipticOptions const& options
)
{
   const int num_levels = validate_elliptic_solve(
      hierarchy,
      vector_potential,
      source,
      boundary,
      options,
      AMREX_SPACEDIM
   );
   if (boundary.centering() == InitialDataEllipticCentering::vertex) {
      return solve_initial_data_nodal_conformal_vector_laplacian(
         hierarchy, vector_potential, source, boundary, options
      );
   }
   auto negative_source_storage = negative_fields(
      hierarchy, source, AMREX_SPACEDIM
   );
   amrex::Vector<amrex::MultiFab const*> negative_source(num_levels);
   for (int level = 0; level < num_levels; ++level) {
      negative_source[level] = negative_source_storage[level].get();
   }

   amrex::LPInfo info;
   amrex::MLTensorOp vector_laplacian(
      hierarchy.geometry,
      hierarchy.grids,
      hierarchy.distribution,
      info
   );
   vector_laplacian.setMaxOrder(options.max_order);
   vector_laplacian.setScalars(0.0, 1.0);
   const LoweredBoundaryConditions lowered = lower_boundary_conditions(
      hierarchy, boundary
   );
   if (lowered.representation == BoundaryRepresentation::robin) {
      amrex::Abort(
         "AMReX MLTensorOp does not consistently support Robin tensor cross terms"
      );
   }
   for (int level = 0; level < num_levels; ++level) {
      vector_laplacian.setShearViscosity(level, 1.0);
   }
   set_boundary_conditions(vector_laplacian, boundary, lowered);

   amrex::MLMG mlmg(vector_laplacian);
   configure_mlmg(mlmg, options);
   return elliptic_result(mlmg.solve(
      vector_potential,
      negative_source,
      options.relative_tolerance,
      options.absolute_tolerance
   ));
}

} // namespace pyfluid_amrex
