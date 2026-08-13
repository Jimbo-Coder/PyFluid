#include "pyfluid_amrex/initial_data/elliptic.hpp"

#include <AMReX.H>
#include <AMReX_BCUtil.H>
#include <AMReX_GMRES.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_FillPatchUtil.H>
#include <AMReX_FabArrayUtility.H>
#include <AMReX_LO_BCTYPES.H>
#include <AMReX_MLABecLaplacian.H>
#include <AMReX_MLMG.H>
#include <AMReX_MLNodeLaplacian.H>
#include <AMReX_MLTensorOp.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_PhysBCFunct.H>
#include <AMReX_ParallelContext.H>
#include <AMReX_ParallelReduce.H>

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <vector>

#include "pyfluid_amrex/constants.hpp"
#include "pyfluid_amrex/initial_data/centering.hpp"
#include "pyfluid_amrex/kernels_generated/differential_ops.hpp"

namespace pyfluid_amrex {

amrex::Vector<amrex::Real> InitialDataEllipticHierarchy::coordinate_faces(
   int level,
   int direction
) const
{
   if (level < 0 || level >= static_cast<int>(geometry.size())
       || direction < 0 || direction >= AMREX_SPACEDIM) {
      amrex::Abort("invalid initial-data coordinate-face request");
   }
   const amrex::Box domain = geometry[level].Domain();
   const int cells = domain.length(direction);
   if (!physical_coordinate_faces.empty()) {
      if (physical_coordinate_faces.size() != geometry.size()) {
         amrex::Abort(
            "initial-data physical-coordinate hierarchy has the wrong size"
         );
      }
      auto const& mapped = physical_coordinate_faces[level][direction];
      if (!mapped.empty()) {
         if (static_cast<int>(mapped.size()) != cells + 1) {
            amrex::Abort(
               "initial-data physical face count does not match its index domain"
            );
         }
         for (int face = 0; face < cells; ++face) {
            if (!(mapped[face + 1] > mapped[face])
                || !std::isfinite(mapped[face])) {
               amrex::Abort(
                  "initial-data physical coordinate faces must be finite and increasing"
               );
            }
         }
         if (!std::isfinite(mapped.back())) {
            amrex::Abort("initial-data physical coordinate face is not finite");
         }
         return mapped;
      }
   }

   const auto problo = geometry[level].ProbLoArray();
   const auto spacing = geometry[level].CellSizeArray();
   amrex::Vector<amrex::Real> uniform(cells + 1);
   for (int face = 0; face <= cells; ++face) {
      uniform[face] = problo[direction] + face * spacing[direction];
   }
   return uniform;
}

amrex::Vector<amrex::Real> surface_focused_radial_faces(
   amrex::Real surface_radius,
   amrex::Real outer_radius,
   int surface_cells,
   int exterior_cells
)
{
   if (!(surface_radius > 0.0) || !(outer_radius > surface_radius)
       || surface_cells <= 0 || exterior_cells <= 0) {
      amrex::Abort("invalid surface-focused radial-grid parameters");
   }
   const amrex::Real inner_spacing = surface_radius / surface_cells;
   const amrex::Real exterior_extent = outer_radius - surface_radius;
   const amrex::Real uniform_extent = exterior_cells * inner_spacing;
   const amrex::Real scale = std::max(outer_radius, amrex::Real(1.0));
   if (exterior_extent + 64.0 * std::numeric_limits<amrex::Real>::epsilon()
          * scale < uniform_extent) {
      amrex::Abort(
         "surface-focused exterior cannot become finer than the source region"
      );
   }

   const auto geometric_extent = [&](amrex::Real ratio) {
      amrex::Real interval = inner_spacing;
      amrex::Real extent = 0.0;
      for (int cell = 0; cell < exterior_cells; ++cell) {
         interval *= ratio;
         extent += interval;
      }
      return extent;
   };
   amrex::Real ratio = 1.0;
   if (exterior_extent > uniform_extent
       + 64.0 * std::numeric_limits<amrex::Real>::epsilon() * scale) {
      amrex::Real lower = 1.0;
      amrex::Real upper = 2.0;
      while (geometric_extent(upper) < exterior_extent) {
         upper *= 2.0;
         if (!std::isfinite(upper)) {
            amrex::Abort("surface-focused radial ratio did not bracket");
         }
      }
      for (int iteration = 0; iteration < 100; ++iteration) {
         const amrex::Real midpoint = 0.5 * (lower + upper);
         if (geometric_extent(midpoint) < exterior_extent) {
            lower = midpoint;
         } else {
            upper = midpoint;
         }
      }
      ratio = 0.5 * (lower + upper);
   }

   amrex::Vector<amrex::Real> faces(
      surface_cells + exterior_cells + 1, 0.0
   );
   for (int face = 1; face <= surface_cells; ++face) {
      faces[face] = face * inner_spacing;
   }
   amrex::Real interval = inner_spacing;
   for (int cell = 0; cell < exterior_cells; ++cell) {
      interval *= ratio;
      faces[surface_cells + cell + 1] =
         faces[surface_cells + cell] + interval;
   }
   faces.back() = outer_radius;
   return faces;
}

namespace {

using InitialDataBoundaryFunction = amrex::GpuBndryFuncFab<amrex::FabFillNoOp>;
using InitialDataPhysicalBoundary = amrex::PhysBCFunct<InitialDataBoundaryFunction>;

using BoundaryTypeArray = amrex::Array<amrex::LinOpBCType, AMREX_SPACEDIM>;

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
   bool has_robin = false;
   amrex::Vector<BoundaryTypeArray> lower_types;
   amrex::Vector<BoundaryTypeArray> upper_types;
   amrex::Vector<std::unique_ptr<amrex::MultiFab>> level_data;
};

amrex::LinOpBCType linop_boundary_type(
   InitialDataEllipticBoundaryType type,
   bool robin_as_dirichlet = false
)
{
   switch (type) {
   case InitialDataEllipticBoundaryType::periodic:
      return amrex::LinOpBCType::Periodic;
   case InitialDataEllipticBoundaryType::dirichlet:
      return amrex::LinOpBCType::Dirichlet;
   case InitialDataEllipticBoundaryType::neumann:
      return amrex::LinOpBCType::Neumann;
   case InitialDataEllipticBoundaryType::robin:
      return robin_as_dirichlet
         ? amrex::LinOpBCType::Dirichlet
         : amrex::LinOpBCType::Robin;
   }
   return amrex::LinOpBCType::Dirichlet;
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
      options.boundary_interpolation_order < 2
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

bool has_stored_robin_boundary(
   InitialDataEllipticBoundaryConditions const& boundary
)
{
   for (int component = 0; component < boundary.num_components; ++component) {
      for (int direction = 0; direction < AMREX_SPACEDIM; ++direction) {
         if (boundary.lower_type(component, direction)
                == InitialDataEllipticBoundaryType::robin
             || boundary.upper_type(component, direction)
                == InitialDataEllipticBoundaryType::robin) {
            return true;
         }
      }
   }
   return false;
}

bool has_legacy_coefficient_robin_boundary(
   InitialDataEllipticBoundaryConditions const& boundary
)
{
   if (has_stored_robin_boundary(boundary)) {
      return false;
   }
   for (int component = 0; component < boundary.num_components; ++component) {
      for (int direction = 0; direction < AMREX_SPACEDIM; ++direction) {
         if (boundary.lower_reflection_parity(component, direction) != 0
             || boundary.upper_reflection_parity(component, direction) != 0) {
            return false;
         }
      }
   }
   for (int level = 0; level < boundary.num_levels(); ++level) {
      for (int component = 0; component < boundary.num_components; ++component) {
         if (boundary.coefficient_b(level).norm0(component, 1, false) != 0.0) {
            return true;
         }
      }
   }
   return false;
}

InitialDataEllipticBoundaryType resolved_boundary_type(
   InitialDataEllipticBoundaryConditions const& boundary,
   int component,
   int direction,
   bool upper
)
{
   const auto stored = upper
      ? boundary.upper_type(component, direction)
      : boundary.lower_type(component, direction);
   const int reflection = upper
      ? boundary.upper_reflection_parity(component, direction)
      : boundary.lower_reflection_parity(component, direction);
   if (reflection == 0
       && stored == InitialDataEllipticBoundaryType::dirichlet
       && has_legacy_coefficient_robin_boundary(boundary)) {
      return InitialDataEllipticBoundaryType::robin;
   }
   return stored;
}

bool has_robin_boundary(
   InitialDataEllipticBoundaryConditions const& boundary
)
{
   return has_stored_robin_boundary(boundary)
      || has_legacy_coefficient_robin_boundary(boundary);
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
      const auto periodic = hierarchy.geometry[level].isPeriodicArray();
      for (int component = 0;
           component < boundary.num_components;
           ++component) {
         amrex::GpuArray<int, AMREX_SPACEDIM> lower_dirichlet{};
         amrex::GpuArray<int, AMREX_SPACEDIM> upper_dirichlet{};
         for (int direction = 0; direction < AMREX_SPACEDIM; ++direction) {
            lower_dirichlet[direction] =
               resolved_boundary_type(boundary, component, direction, false)
               == InitialDataEllipticBoundaryType::dirichlet;
            upper_dirichlet[direction] =
               resolved_boundary_type(boundary, component, direction, true)
               == InitialDataEllipticBoundaryType::dirichlet;
         }
         for (amrex::MFIter mfi(*values[level]); mfi.isValid(); ++mfi) {
            const amrex::Box box = mfi.fabbox();
            const auto coefficient_a =
               boundary.coefficient_a(level).const_array(mfi);
            const auto coefficient_c =
               boundary.coefficient_c(level).const_array(mfi);
            const auto value = values[level]->array(mfi);
            amrex::ParallelFor(
               box,
               [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                  bool dirichlet_face = false;
                  if (!periodic[0] && i < domain_lo.x) {
                     dirichlet_face = lower_dirichlet[0] != 0;
                  } else if (!periodic[0] && i > domain_hi.x) {
                     dirichlet_face = upper_dirichlet[0] != 0;
                  }
#if (AMREX_SPACEDIM >= 2)
                  else if (!periodic[1] && j < domain_lo.y) {
                     dirichlet_face = lower_dirichlet[1] != 0;
                  } else if (!periodic[1] && j > domain_hi.y) {
                     dirichlet_face = upper_dirichlet[1] != 0;
                  }
#endif
#if (AMREX_SPACEDIM == 3)
                  else if (!periodic[2] && k < domain_lo.z) {
                     dirichlet_face = lower_dirichlet[2] != 0;
                  } else if (!periodic[2] && k > domain_hi.z) {
                     dirichlet_face = upper_dirichlet[2] != 0;
                  }
#endif
                  if (dirichlet_face) {
                     const amrex::Real a = coefficient_a(
                        i, j, k, component
                     );
                     value(i, j, k, component) = a != 0.0
                        ? coefficient_c(i, j, k, component) / a
                        : std::numeric_limits<amrex::Real>::quiet_NaN();
                  }
               }
            );
         }
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
   lowered.has_robin = has_robin_boundary(boundary);
   lowered.lower_types.resize(boundary.num_components);
   lowered.upper_types.resize(boundary.num_components);
   for (int component = 0; component < boundary.num_components; ++component) {
      for (int direction = 0; direction < AMREX_SPACEDIM; ++direction) {
         lowered.lower_types[component][direction] = linop_boundary_type(
            resolved_boundary_type(boundary, component, direction, false)
         );
         lowered.upper_types[component][direction] = linop_boundary_type(
            resolved_boundary_type(boundary, component, direction, true)
         );
      }
   }
   lowered.level_data = !lowered.has_robin
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
   linear_operator.setDomainBC(lowered.lower_types, lowered.upper_types);
   for (int level = 0; level < boundary.num_levels(); ++level) {
      if (!lowered.has_robin) {
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
   InitialDataEllipticBoundaryConditions const& boundary
)
{
   amrex::Vector<amrex::BCRec> records(boundary.num_components);
   for (int component = 0; component < boundary.num_components; ++component) {
      for (int direction = 0; direction < AMREX_SPACEDIM; ++direction) {
         auto record_type = [](InitialDataEllipticBoundaryType type) {
            if (type == InitialDataEllipticBoundaryType::periodic) {
               return amrex::BCType::int_dir;
            }
            if (type == InitialDataEllipticBoundaryType::neumann) {
               return amrex::BCType::reflect_even;
            }
            return amrex::BCType::foextrap;
         };
         const int lower_parity =
            boundary.lower_reflection_parity(component, direction);
         const int upper_parity =
            boundary.upper_reflection_parity(component, direction);
         records[component].setLo(
            direction,
            lower_parity < 0 ? amrex::BCType::reflect_odd
            : lower_parity > 0 ? amrex::BCType::reflect_even
                               : record_type(boundary.lower_type(component, direction))
         );
         records[component].setHi(
            direction,
            upper_parity < 0 ? amrex::BCType::reflect_odd
            : upper_parity > 0 ? amrex::BCType::reflect_even
                               : record_type(boundary.upper_type(component, direction))
         );
      }
   }
   return records;
}

void synchronize_nodal_solution(
   InitialDataEllipticHierarchy const& hierarchy,
   amrex::Vector<amrex::MultiFab*> const& solution,
   InitialDataEllipticBoundaryConditions const& boundary
)
{
   if (solution.front()->nGrow() == 0) {
      return;
   }
   detail::fill_initial_data_vertex_ghosts(
      hierarchy,
      solution,
      nodal_solution_boundary_records(hierarchy, boundary)
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
      amrex::GpuArray<int, AMREX_SPACEDIM> lower_fixed{};
      amrex::GpuArray<int, AMREX_SPACEDIM> upper_fixed{};
      for (int direction = 0; direction < AMREX_SPACEDIM; ++direction) {
         const auto upper = resolved_boundary_type(
            boundary, 0, direction, true
         );
         const auto resolved_lower = resolved_boundary_type(
            boundary, 0, direction, false
         );
         lower_fixed[direction] =
            resolved_lower == InitialDataEllipticBoundaryType::dirichlet
            || resolved_lower == InitialDataEllipticBoundaryType::robin;
         upper_fixed[direction] =
            upper == InitialDataEllipticBoundaryType::dirichlet
            || upper == InitialDataEllipticBoundaryType::robin;
      }
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
                  (!periodic[0] && lower_fixed[0] != 0
                   && i == nodal_domain.smallEnd(0))
                  || (!periodic[0] && upper_fixed[0] != 0
                      && i == nodal_domain.bigEnd(0))
                  ||
                  (
#if (AMREX_SPACEDIM >= 2)
                   (!periodic[1] && lower_fixed[1] != 0
                    && j == nodal_domain.smallEnd(1))
                   || (!periodic[1] && upper_fixed[1] != 0
                       && j == nodal_domain.bigEnd(1))
#else
                   false
#endif
                  )
#if (AMREX_SPACEDIM == 3)
                  || (!periodic[2] && lower_fixed[2] != 0
                      && k == nodal_domain.smallEnd(2))
                  || (!periodic[2] && upper_fixed[2] != 0
                      && k == nodal_domain.bigEnd(2))
#endif
                  ;
               if (physical_boundary) {
                  int boundary_i = i;
                  int boundary_j = j;
                  int boundary_k = k;
                  if (!periodic[0] && lower_fixed[0] != 0
                      && i == nodal_domain.smallEnd(0)) {
                     --boundary_i;
                  } else if (!periodic[0] && upper_fixed[0] != 0
                             && i == nodal_domain.bigEnd(0)) {
                     ++boundary_i;
                  }
#if (AMREX_SPACEDIM >= 2)
                  else if (!periodic[1] && lower_fixed[1] != 0
                           && j == nodal_domain.smallEnd(1)) {
                     --boundary_j;
                  } else if (!periodic[1] && upper_fixed[1] != 0
                             && j == nodal_domain.bigEnd(1)) {
                     ++boundary_j;
                  }
#endif
#if (AMREX_SPACEDIM == 3)
                  else if (!periodic[2] && lower_fixed[2] != 0
                           && k == nodal_domain.smallEnd(2)) {
                     --boundary_k;
                  } else if (!periodic[2] && upper_fixed[2] != 0
                             && k == nodal_domain.bigEnd(2)) {
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
      amrex::GpuArray<int, AMREX_SPACEDIM> lower_robin{};
      amrex::GpuArray<int, AMREX_SPACEDIM> upper_robin{};
      for (int direction = 0; direction < AMREX_SPACEDIM; ++direction) {
         lower_robin[direction] = resolved_boundary_type(
            boundary, 0, direction, false
         )
            == InitialDataEllipticBoundaryType::robin;
         upper_robin[direction] = resolved_boundary_type(
            boundary, 0, direction, true
         )
            == InitialDataEllipticBoundaryType::robin;
      }
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

            if (!periodic[0] && lower_robin[0] != 0
                && i == nodal_domain.smallEnd(0)) {
               add_face(i - 1, j, k, 0, 1);
            }
            if (!periodic[0] && upper_robin[0] != 0
                && i == nodal_domain.bigEnd(0)) {
               add_face(i + 1, j, k, 0, -1);
            }
#if (AMREX_SPACEDIM >= 2)
            if (!periodic[1] && lower_robin[1] != 0
                && j == nodal_domain.smallEnd(1)) {
               add_face(i, j - 1, k, 1, 1);
            }
            if (!periodic[1] && upper_robin[1] != 0
                && j == nodal_domain.bigEnd(1)) {
               add_face(i, j + 1, k, 1, -1);
            }
#endif
#if (AMREX_SPACEDIM == 3)
            if (!periodic[2] && lower_robin[2] != 0
                && k == nodal_domain.smallEnd(2)) {
               add_face(i, j, k - 1, 2, 1);
            }
            if (!periodic[2] && upper_robin[2] != 0
                && k == nodal_domain.bigEnd(2)) {
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
   BoundaryTypeArray base_lower;
   BoundaryTypeArray base_upper;
   for (int direction = 0; direction < AMREX_SPACEDIM; ++direction) {
      base_lower[direction] = linop_boundary_type(
         resolved_boundary_type(boundary, 0, direction, false), true
      );
      base_upper[direction] = linop_boundary_type(
         resolved_boundary_type(boundary, 0, direction, true), true
      );
   }
   if (!has_robin_boundary(boundary)) {
      set_nodal_dirichlet_values(hierarchy, solution, boundary);
      amrex::MLNodeLaplacian poisson(
         hierarchy.geometry,
         hierarchy.grids,
         hierarchy.distribution,
         info,
         {},
         1.0
      );
      poisson.setDomainBC(base_lower, base_upper);

      amrex::MLMG mlmg(poisson);
      configure_mlmg(mlmg, options);
      const amrex::Real residual = mlmg.solve(
         solution,
         rhs,
         options.relative_tolerance,
         options.absolute_tolerance
      );
      synchronize_nodal_solution(hierarchy, solution, boundary);
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
   poisson.setDomainBC(base_lower, base_upper);

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
      synchronize_nodal_solution(hierarchy, solution, boundary);
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
   synchronize_nodal_solution(hierarchy, solution, boundary);
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

   const amrex::Vector<amrex::BCRec> boundary_records =
      nodal_solution_boundary_records(hierarchy, boundary);

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

amrex::Vector<amrex::BCRec> initial_data_boundary_records(
   InitialDataEllipticHierarchy const& hierarchy,
   amrex::Vector<GridFieldSymmetryKind> const& component_symmetries,
   int nonsymmetry_type
)
{
   if (component_symmetries.empty()) {
      amrex::Abort("initial-data boundary records require components");
   }
   amrex::Vector<amrex::BCRec> records(component_symmetries.size());
   for (int component = 0;
        component < static_cast<int>(component_symmetries.size());
        ++component) {
      for (int direction = 0; direction < AMREX_SPACEDIM; ++direction) {
         const int parity = grid_field_reflection_parity(
            component_symmetries[component], direction
         );
         const int reflection_type = parity < 0
            ? amrex::BCType::reflect_odd
            : amrex::BCType::reflect_even;
         const int periodic_type = hierarchy.geometry[0].isPeriodic(direction)
            ? amrex::BCType::int_dir
            : nonsymmetry_type;
         records[component].setLo(
            direction,
            hierarchy.reflection_lo[direction] != 0
               ? reflection_type
               : periodic_type
         );
         records[component].setHi(
            direction,
            hierarchy.reflection_hi[direction] != 0
               ? reflection_type
               : periodic_type
         );
      }
   }
   return records;
}

amrex::Vector<amrex::BCRec> initial_data_boundary_records(
   InitialDataEllipticHierarchy const& hierarchy,
   InitialDataEllipticBoundaryConditions const& boundary,
   int nonsymmetry_type
)
{
   if (boundary.num_components <= 0
       || boundary.num_levels() != static_cast<int>(hierarchy.geometry.size())) {
      amrex::Abort("invalid explicit initial-data boundary records");
   }
   amrex::Vector<amrex::BCRec> records(boundary.num_components);
   for (int component = 0;
        component < boundary.num_components;
        ++component) {
      for (int direction = 0; direction < AMREX_SPACEDIM; ++direction) {
         const int fallback = hierarchy.geometry[0].isPeriodic(direction)
            ? amrex::BCType::int_dir
            : nonsymmetry_type;
         const int lower_parity = boundary.lower_reflection_parity(
            component, direction
         );
         const int upper_parity = boundary.upper_reflection_parity(
            component, direction
         );
         records[component].setLo(
            direction,
            hierarchy.reflection_lo[direction] != 0
               ? (lower_parity < 0
                  ? amrex::BCType::reflect_odd
                  : amrex::BCType::reflect_even)
               : fallback
         );
         records[component].setHi(
            direction,
            hierarchy.reflection_hi[direction] != 0
               ? (upper_parity < 0
                  ? amrex::BCType::reflect_odd
                  : amrex::BCType::reflect_even)
               : fallback
         );
      }
   }
   return records;
}

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
   amrex::Vector<amrex::BCRec> reflection_boundaries = boundary_conditions;
   bool has_reflection = false;
   for (amrex::BCRec& record : reflection_boundaries) {
      for (int direction = 0; direction < AMREX_SPACEDIM; ++direction) {
         if (hierarchy.reflection_lo[direction] == 0) {
            record.setLo(direction, amrex::BCType::int_dir);
         } else {
            has_reflection = true;
         }
         if (hierarchy.reflection_hi[direction] == 0) {
            record.setHi(direction, amrex::BCType::int_dir);
         } else {
            has_reflection = true;
         }
      }
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
         if (has_reflection) {
            amrex::FillDomainBoundary(
               *fields[level], hierarchy.geometry[level], reflection_boundaries
            );
         }
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
      if (has_reflection) {
            amrex::FillDomainBoundary(
               *fields[level], hierarchy.geometry[level], reflection_boundaries
            );
      }
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
   const int side_entries = num_components * AMREX_SPACEDIM;
   boundary.lower_types_.resize(side_entries);
   boundary.upper_types_.resize(side_entries);
   boundary.lower_reflection_parity_.assign(side_entries, 0);
   boundary.upper_reflection_parity_.assign(side_entries, 0);
   for (int component = 0; component < num_components; ++component) {
      for (int direction = 0; direction < AMREX_SPACEDIM; ++direction) {
         const auto type = hierarchy.geometry[0].isPeriodic(direction)
            ? InitialDataEllipticBoundaryType::periodic
            : coefficient_b == 0.0
               ? InitialDataEllipticBoundaryType::dirichlet
               : InitialDataEllipticBoundaryType::robin;
         const int side = component * AMREX_SPACEDIM + direction;
         boundary.lower_types_[side] = type;
         boundary.upper_types_[side] = type;
      }
   }
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

InitialDataEllipticBoundaryType
InitialDataEllipticBoundaryConditions::lower_type(
   int component, int direction
) const noexcept
{
   return lower_types_[component * AMREX_SPACEDIM + direction];
}

InitialDataEllipticBoundaryType
InitialDataEllipticBoundaryConditions::upper_type(
   int component, int direction
) const noexcept
{
   return upper_types_[component * AMREX_SPACEDIM + direction];
}

int InitialDataEllipticBoundaryConditions::lower_reflection_parity(
   int component, int direction
) const noexcept
{
   return lower_reflection_parity_[component * AMREX_SPACEDIM + direction];
}

int InitialDataEllipticBoundaryConditions::upper_reflection_parity(
   int component, int direction
) const noexcept
{
   return upper_reflection_parity_[component * AMREX_SPACEDIM + direction];
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
   const int source_offset = component * AMREX_SPACEDIM;
   result.lower_types_.assign(
      lower_types_.begin() + source_offset,
      lower_types_.begin() + source_offset + AMREX_SPACEDIM
   );
   result.upper_types_.assign(
      upper_types_.begin() + source_offset,
      upper_types_.begin() + source_offset + AMREX_SPACEDIM
   );
   result.lower_reflection_parity_.assign(
      lower_reflection_parity_.begin() + source_offset,
      lower_reflection_parity_.begin() + source_offset + AMREX_SPACEDIM
   );
   result.upper_reflection_parity_.assign(
      upper_reflection_parity_.begin() + source_offset,
      upper_reflection_parity_.begin() + source_offset + AMREX_SPACEDIM
   );
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

void set_initial_data_reflection_boundaries(
   InitialDataEllipticHierarchy const& hierarchy,
   InitialDataEllipticBoundaryConditions& boundary,
   amrex::Vector<GridFieldSymmetryKind> const& component_symmetries
)
{
   if (boundary.num_levels() != static_cast<int>(hierarchy.geometry.size())
       || static_cast<int>(component_symmetries.size())
          != boundary.num_components) {
      amrex::Abort("invalid initial-data reflection boundary configuration");
   }
   for (int component = 0; component < boundary.num_components; ++component) {
      for (int direction = 0; direction < AMREX_SPACEDIM; ++direction) {
         const int side = component * AMREX_SPACEDIM + direction;
         const int parity = grid_field_reflection_parity(
            component_symmetries[component], direction
         );
         if (hierarchy.reflection_lo[direction] != 0) {
            boundary.lower_types_[side] = parity < 0
               ? InitialDataEllipticBoundaryType::dirichlet
               : InitialDataEllipticBoundaryType::neumann;
            boundary.lower_reflection_parity_[side] = parity;
         }
         if (hierarchy.reflection_hi[direction] != 0) {
            boundary.upper_types_[side] = parity < 0
               ? InitialDataEllipticBoundaryType::dirichlet
               : InitialDataEllipticBoundaryType::neumann;
            boundary.upper_reflection_parity_[side] = parity;
         }
      }
   }

   for (int level = 0; level < boundary.num_levels(); ++level) {
      const amrex::Box coefficient_domain =
         boundary.centering() == InitialDataEllipticCentering::vertex
         ? amrex::surroundingNodes(hierarchy.geometry[level].Domain())
         : hierarchy.geometry[level].Domain();
      const amrex::Dim3 domain_lo = amrex::lbound(coefficient_domain);
      const amrex::Dim3 domain_hi = amrex::ubound(coefficient_domain);
      const auto reflection_lo = hierarchy.reflection_lo;
      const auto reflection_hi = hierarchy.reflection_hi;
      for (int component = 0; component < boundary.num_components; ++component) {
         amrex::GpuArray<int, AMREX_SPACEDIM> parity{};
         for (int direction = 0; direction < AMREX_SPACEDIM; ++direction) {
            parity[direction] = grid_field_reflection_parity(
               component_symmetries[component], direction
            );
         }
         for (amrex::MFIter mfi(boundary.coefficient_a(level));
              mfi.isValid(); ++mfi) {
            const amrex::Box box = mfi.fabbox();
            const auto coefficient_a = boundary.coefficient_a(level).array(mfi);
            const auto coefficient_b = boundary.coefficient_b(level).array(mfi);
            const auto coefficient_c = boundary.coefficient_c(level).array(mfi);
            amrex::ParallelFor(
               box,
               [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                  int direction = -1;
                  if (reflection_lo[0] != 0 && i < domain_lo.x) {
                     direction = 0;
                  } else if (reflection_hi[0] != 0 && i > domain_hi.x) {
                     direction = 0;
                  }
#if (AMREX_SPACEDIM >= 2)
                  else if (reflection_lo[1] != 0 && j < domain_lo.y) {
                     direction = 1;
                  } else if (reflection_hi[1] != 0 && j > domain_hi.y) {
                     direction = 1;
                  }
#endif
#if (AMREX_SPACEDIM == 3)
                  else if (reflection_lo[2] != 0 && k < domain_lo.z) {
                     direction = 2;
                  } else if (reflection_hi[2] != 0 && k > domain_hi.z) {
                     direction = 2;
                  }
#endif
                  if (direction >= 0) {
                     const bool odd = parity[direction] < 0;
                     coefficient_a(i, j, k, component) = odd ? 1.0 : 0.0;
                     coefficient_b(i, j, k, component) = odd ? 0.0 : 1.0;
                     coefficient_c(i, j, k, component) = 0.0;
                  }
               }
            );
         }
      }
   }
}

void set_axisymmetric_spherical_toroidal_regularity(
   InitialDataEllipticHierarchy const& hierarchy,
   InitialDataEllipticBoundaryConditions& boundary,
   int component
)
{
#if (AMREX_SPACEDIM != 2)
   amrex::ignore_unused(hierarchy, boundary, component);
   amrex::Abort("spherical toroidal regularity requires two dimensions");
#else
   if (boundary.num_levels() != static_cast<int>(hierarchy.geometry.size())
       || component < 0 || component >= boundary.num_components
       || boundary.centering() != InitialDataEllipticCentering::cell) {
      amrex::Abort("invalid spherical toroidal regularity configuration");
   }
   auto near = [](amrex::Real first, amrex::Real second) {
      const amrex::Real scale = std::max({
         amrex::Real(1.0), std::abs(first), std::abs(second)
      });
      return std::abs(first - second)
         <= 64.0 * std::numeric_limits<amrex::Real>::epsilon() * scale;
   };
   const auto first_problo = hierarchy.geometry.front().ProbLoArray();
   const auto first_probhi = hierarchy.geometry.front().ProbHiArray();
   const amrex::Real half_pi = 0.5 * constants::pi;
   const bool lower_axis = near(first_problo[1], 0.0);
   const bool lower_equator = near(first_problo[1], half_pi);
   const bool upper_axis = near(first_probhi[1], constants::pi);
   const bool upper_equator = near(first_probhi[1], half_pi);
   if (!near(first_problo[0], 0.0)
       || !(lower_axis || lower_equator)
       || !(upper_axis || upper_equator)
       || hierarchy.reflection_lo[0] == 0
       || hierarchy.reflection_lo[1] == 0
       || hierarchy.reflection_hi[1] == 0) {
      amrex::Abort("invalid r-theta domain for toroidal regularity");
   }

   const int radial_side = component * AMREX_SPACEDIM;
   const int angular_side = radial_side + 1;
   boundary.lower_types_[radial_side] =
      InitialDataEllipticBoundaryType::dirichlet;
   boundary.lower_reflection_parity_[radial_side] = -1;
   boundary.lower_types_[angular_side] = lower_axis
      ? InitialDataEllipticBoundaryType::dirichlet
      : InitialDataEllipticBoundaryType::neumann;
   boundary.lower_reflection_parity_[angular_side] = lower_axis ? -1 : 1;
   boundary.upper_types_[angular_side] = upper_axis
      ? InitialDataEllipticBoundaryType::dirichlet
      : InitialDataEllipticBoundaryType::neumann;
   boundary.upper_reflection_parity_[angular_side] = upper_axis ? -1 : 1;

   for (int level = 0; level < boundary.num_levels(); ++level) {
      const auto problo = hierarchy.geometry[level].ProbLoArray();
      const auto probhi = hierarchy.geometry[level].ProbHiArray();
      if (!near(problo[0], 0.0)
          || !near(problo[1], first_problo[1])
          || !near(probhi[1], first_probhi[1])) {
         amrex::Abort("inconsistent r-theta hierarchy geometry");
      }
      const amrex::Box domain = hierarchy.geometry[level].Domain();
      const amrex::Dim3 domain_lo = amrex::lbound(domain);
      const amrex::Dim3 domain_hi = amrex::ubound(domain);
      const int selected_component = component;
      const int lower_angular_is_axis = lower_axis ? 1 : 0;
      const int upper_angular_is_axis = upper_axis ? 1 : 0;
      for (amrex::MFIter mfi(boundary.coefficient_a(level));
           mfi.isValid(); ++mfi) {
         const amrex::Box box = mfi.fabbox();
         const auto a = boundary.coefficient_a(level).array(mfi);
         const auto b = boundary.coefficient_b(level).array(mfi);
         const auto c = boundary.coefficient_c(level).array(mfi);
         amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(
            int i, int j, int k
         ) noexcept {
            int odd = -1;
            if (i < domain_lo.x) {
               odd = 1;
            } else if (j < domain_lo.y) {
               odd = lower_angular_is_axis;
            } else if (j > domain_hi.y) {
               odd = upper_angular_is_axis;
            }
            if (odd >= 0) {
               a(i, j, k, selected_component) = odd != 0 ? 1.0 : 0.0;
               b(i, j, k, selected_component) = odd != 0 ? 0.0 : 1.0;
               c(i, j, k, selected_component) = 0.0;
            }
         });
      }
   }
#endif
}

void set_axisymmetric_spherical_radial_robin_boundary(
   InitialDataEllipticHierarchy const& hierarchy,
   InitialDataEllipticBoundaryConditions& boundary,
   amrex::Vector<amrex::Real> const& asymptotic_values,
   amrex::Real falloff_power
)
{
#if (AMREX_SPACEDIM != 2)
   amrex::ignore_unused(
      hierarchy, boundary, asymptotic_values, falloff_power
   );
   amrex::Abort("spherical radial Robin data require two dimensions");
#else
   if (boundary.centering() != InitialDataEllipticCentering::cell
       || boundary.num_levels() != static_cast<int>(hierarchy.geometry.size())
       || static_cast<int>(asymptotic_values.size())
          != boundary.num_components
       || !(falloff_power > 0.0)) {
      amrex::Abort("invalid spherical radial Robin boundary configuration");
   }
   for (int component = 0; component < boundary.num_components; ++component) {
      const int side = component * AMREX_SPACEDIM;
      boundary.upper_types_[side] = InitialDataEllipticBoundaryType::robin;
      boundary.upper_reflection_parity_[side] = 1;
   }
   for (int level = 0; level < boundary.num_levels(); ++level) {
      const auto problo = hierarchy.geometry[level].ProbLoArray();
      const auto probhi = hierarchy.geometry[level].ProbHiArray();
      if (hierarchy.geometry[level].isPeriodic(0)
          || std::abs(problo[0])
             > 64.0 * std::numeric_limits<amrex::Real>::epsilon()
          || !(probhi[0] > 0.0)) {
         amrex::Abort("spherical radial Robin data require r in [0,R]");
      }
      const amrex::Dim3 domain_hi = amrex::ubound(
         hierarchy.geometry[level].Domain()
      );
      const amrex::Real coefficient = falloff_power / probhi[0];
      for (int component = 0;
           component < boundary.num_components;
           ++component) {
         const amrex::Real asymptotic = asymptotic_values[component];
         for (amrex::MFIter mfi(boundary.coefficient_a(level));
              mfi.isValid(); ++mfi) {
            const amrex::Box box = mfi.fabbox();
            const auto a = boundary.coefficient_a(level).array(mfi);
            const auto b = boundary.coefficient_b(level).array(mfi);
            const auto c = boundary.coefficient_c(level).array(mfi);
            amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(
               int i, int j, int k
            ) noexcept {
               if (i > domain_hi.x) {
                  a(i, j, k, component) = coefficient;
                  b(i, j, k, component) = 1.0;
                  c(i, j, k, component) = coefficient * asymptotic;
               }
            });
         }
      }
   }
#endif
}

void set_centered_power_law_robin_boundary(
   InitialDataEllipticHierarchy const& hierarchy,
   InitialDataEllipticBoundaryConditions& boundary,
   amrex::Vector<amrex::Real> const& asymptotic_values,
   amrex::Real falloff_power,
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const* physical_center
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
      amrex::GpuArray<int, AMREX_SPACEDIM> robin_lo{};
      amrex::GpuArray<int, AMREX_SPACEDIM> robin_hi{};
      for (int direction = 0; direction < AMREX_SPACEDIM; ++direction) {
         center[direction] = physical_center != nullptr
            ? (*physical_center)[direction]
            : 0.5 * (problo[direction] + probhi[direction]);
         periodic[direction] = hierarchy.geometry[level].isPeriodic(direction);
      }
      const amrex::Box coefficient_domain = vertex_centered
         ? amrex::surroundingNodes(hierarchy.geometry[level].Domain())
         : hierarchy.geometry[level].Domain();
      const amrex::Dim3 domain_lo = amrex::lbound(coefficient_domain);
      const amrex::Dim3 domain_hi = amrex::ubound(coefficient_domain);
      const amrex::Real coordinate_offset = vertex_centered ? 0.0 : 0.5;
      for (int component = 0; component < boundary.num_components; ++component) {
         for (int direction = 0; direction < AMREX_SPACEDIM; ++direction) {
            const int side = component * AMREX_SPACEDIM + direction;
            if (!periodic[direction]
                && boundary.lower_reflection_parity_[side] == 0) {
               boundary.lower_types_[side] =
                  InitialDataEllipticBoundaryType::robin;
               robin_lo[direction] = 1;
            }
            if (!periodic[direction]
                && boundary.upper_reflection_parity_[side] == 0) {
               boundary.upper_types_[side] =
                  InitialDataEllipticBoundaryType::robin;
               robin_hi[direction] = 1;
            }
         }
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
               if (robin_lo[0] != 0 && i < domain_lo.x) {
                  face_direction = 0;
                  position_1 = problo[0] - center[0];
               } else if (robin_hi[0] != 0 && i > domain_hi.x) {
                  face_direction = 0;
                  position_1 = probhi[0] - center[0];
               }
#if (AMREX_SPACEDIM >= 2)
               else if (robin_lo[1] != 0 && j < domain_lo.y) {
                  face_direction = 1;
                  position_2 = problo[1] - center[1];
               } else if (robin_hi[1] != 0 && j > domain_hi.y) {
                  face_direction = 1;
                  position_2 = probhi[1] - center[1];
               }
#endif
#if (AMREX_SPACEDIM == 3)
               else if (robin_lo[2] != 0 && k < domain_lo.z) {
                  face_direction = 2;
                  position_3 = problo[2] - center[2];
               } else if (robin_hi[2] != 0 && k > domain_hi.z) {
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
   poisson.setMaxOrder(options.boundary_interpolation_order);
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

namespace {

enum class AxisymmetricSphericalMode {
   scalar,
   toroidal_vector,
};

} // namespace

#if (AMREX_SPACEDIM == 2)
namespace {

struct SymmetricTridiagonalEigensystem {
   std::vector<amrex::Real> eigenvalues;
   // Mode-major eigenvectors: vectors[mode * size + row].
   std::vector<amrex::Real> vectors;
};

SymmetricTridiagonalEigensystem symmetric_tridiagonal_eigensystem(
   std::vector<amrex::Real> diagonal,
   std::vector<amrex::Real> off_diagonal
)
{
   // Implicit QL iteration keeps the self-adjoint angular solve independent
   // of an optional platform LAPACK while retaining an orthonormal basis.
   const int size = static_cast<int>(diagonal.size());
   if (size < 1 || static_cast<int>(off_diagonal.size()) != size) {
      amrex::Abort("invalid symmetric tridiagonal eigensystem");
   }
   std::vector<amrex::Real> vectors(
      static_cast<std::size_t>(size) * size, 0.0
   );
   for (int index = 0; index < size; ++index) {
      vectors[static_cast<std::size_t>(index) * size + index] = 1.0;
   }

   off_diagonal[size - 1] = 0.0;
   amrex::Real accumulated_shift = 0.0;
   amrex::Real running_scale = 0.0;
   const amrex::Real epsilon = std::numeric_limits<amrex::Real>::epsilon();
   for (int lower = 0; lower < size; ++lower) {
      running_scale = std::max(
         running_scale,
         std::abs(diagonal[lower]) + std::abs(off_diagonal[lower])
      );
      int upper = lower;
      while (upper < size
             && std::abs(off_diagonal[upper]) > epsilon * running_scale) {
         ++upper;
      }
      if (upper > lower) {
         int iterations = 0;
         do {
            if (++iterations > 64) {
               amrex::Abort(
                  "symmetric tridiagonal eigensystem did not converge"
               );
            }
            amrex::Real pivot = diagonal[lower];
            amrex::Real ratio = (
               diagonal[lower + 1] - pivot
            ) / (2.0 * off_diagonal[lower]);
            amrex::Real radius = std::hypot(ratio, 1.0);
            if (ratio < 0.0) {
               radius = -radius;
            }
            diagonal[lower] = off_diagonal[lower] / (ratio + radius);
            diagonal[lower + 1] = off_diagonal[lower] * (ratio + radius);
            const amrex::Real next_diagonal = diagonal[lower + 1];
            const amrex::Real shift = pivot - diagonal[lower];
            for (int index = lower + 2; index < size; ++index) {
               diagonal[index] -= shift;
            }
            accumulated_shift += shift;

            pivot = diagonal[upper];
            amrex::Real cosine = 1.0;
            amrex::Real previous_cosine = cosine;
            amrex::Real previous_previous_cosine = cosine;
            amrex::Real sine = 0.0;
            amrex::Real previous_sine = 0.0;
            const amrex::Real next_off_diagonal = off_diagonal[lower + 1];
            for (int index = upper - 1; index >= lower; --index) {
               previous_previous_cosine = previous_cosine;
               previous_cosine = cosine;
               previous_sine = sine;
               const amrex::Real product = cosine * off_diagonal[index];
               const amrex::Real diagonal_product = cosine * pivot;
               radius = std::hypot(pivot, off_diagonal[index]);
               off_diagonal[index + 1] = sine * radius;
               sine = off_diagonal[index] / radius;
               cosine = pivot / radius;
               pivot = cosine * diagonal[index] - sine * product;
               diagonal[index + 1] = diagonal_product + sine * (
                  cosine * product + sine * diagonal[index]
               );
               for (int row = 0; row < size; ++row) {
                  const std::size_t upper_offset =
                     static_cast<std::size_t>(row) * size + index + 1;
                  const std::size_t lower_offset = upper_offset - 1;
                  const amrex::Real upper_value = vectors[upper_offset];
                  vectors[upper_offset] = sine * vectors[lower_offset]
                     + cosine * upper_value;
                  vectors[lower_offset] = cosine * vectors[lower_offset]
                     - sine * upper_value;
               }
            }
            pivot = -sine * previous_sine * previous_previous_cosine
               * next_off_diagonal * off_diagonal[lower] / next_diagonal;
            off_diagonal[lower] = sine * pivot;
            diagonal[lower] = cosine * pivot;
         } while (
            std::abs(off_diagonal[lower]) > epsilon * running_scale
         );
      }
      diagonal[lower] += accumulated_shift;
      off_diagonal[lower] = 0.0;
   }

   for (int first = 0; first < size - 1; ++first) {
      int smallest = first;
      for (int second = first + 1; second < size; ++second) {
         if (diagonal[second] < diagonal[smallest]) {
            smallest = second;
         }
      }
      if (smallest != first) {
         std::swap(diagonal[first], diagonal[smallest]);
         for (int row = 0; row < size; ++row) {
            std::swap(
               vectors[static_cast<std::size_t>(row) * size + first],
               vectors[static_cast<std::size_t>(row) * size + smallest]
            );
         }
      }
   }
   std::vector<amrex::Real> mode_vectors(vectors.size());
   for (int mode = 0; mode < size; ++mode) {
      for (int row = 0; row < size; ++row) {
         mode_vectors[static_cast<std::size_t>(mode) * size + row] =
            vectors[static_cast<std::size_t>(row) * size + mode];
      }
   }
   return {std::move(diagonal), std::move(mode_vectors)};
}

amrex::MultiFab host_copy(amrex::MultiFab const& source)
{
   amrex::MultiFab result(
      source.boxArray(), source.DistributionMap(), source.nComp(),
      source.nGrowVect(),
      amrex::MFInfo().SetArena(amrex::The_Pinned_Arena())
   );
   amrex::dtoh_memcpy(result, source);
   amrex::Gpu::streamSynchronize();
   return result;
}

void copy_host_to_device(
   amrex::MultiFab& destination,
   amrex::MultiFab const& source
)
{
   amrex::htod_memcpy(destination, source);
   amrex::Gpu::streamSynchronize();
}

} // namespace
#endif

namespace {

InitialDataEllipticResult solve_axisymmetric_spherical_separable(
   InitialDataEllipticHierarchy const& hierarchy,
   amrex::Vector<amrex::MultiFab*> const& solution,
   amrex::Vector<amrex::MultiFab const*> const& rhs,
   InitialDataEllipticBoundaryConditions const& boundary,
   InitialDataEllipticOptions const& options,
   AxisymmetricSphericalMode mode
)
{
#if (AMREX_SPACEDIM != 2)
   amrex::ignore_unused(hierarchy, solution, rhs, boundary, options, mode);
   amrex::Abort(
      "the axisymmetric spherical elliptic operator requires a 2D r-theta hierarchy"
   );
   return {};
#else
   const int num_levels = validate_elliptic_solve(
      hierarchy, solution, rhs, boundary, options, 1
   );
   if (boundary.centering() != InitialDataEllipticCentering::cell) {
      amrex::Abort(
         "the axisymmetric spherical elliptic operator requires cell-centered fields"
      );
   }

   auto near = [](amrex::Real first, amrex::Real second) {
      const amrex::Real scale = std::max({
         amrex::Real(1.0), std::abs(first), std::abs(second)
      });
      return std::abs(first - second)
         <= 64.0 * std::numeric_limits<amrex::Real>::epsilon() * scale;
   };
   if (num_levels != 1) {
      amrex::Abort(
         "the tensor-product spherical elliptic solver requires one uniform level"
      );
   }
   amrex::Geometry const& geometry = hierarchy.geometry[0];
   const auto problo = geometry.ProbLoArray();
   const auto probhi = geometry.ProbHiArray();
   const auto dx = geometry.CellSizeArray();
   const amrex::Box domain = geometry.Domain();
   const amrex::Vector<amrex::Real> radial_faces =
      hierarchy.coordinate_faces(0, 0);
   const amrex::Vector<amrex::Real> angular_faces =
      hierarchy.coordinate_faces(0, 1);
   const amrex::Real half_pi = 0.5 * constants::pi;
   const bool lower_axis = near(problo[1], 0.0);
   const bool upper_axis = near(probhi[1], constants::pi);
   const bool standard_theta_domain =
      (lower_axis
       && (near(probhi[1], half_pi) || near(probhi[1], constants::pi)))
      || (near(problo[1], half_pi) && near(probhi[1], constants::pi));
   if ((geometry.Coord() != amrex::CoordSys::cartesian
        && geometry.Coord() != amrex::CoordSys::SPHERICAL)
       || geometry.isPeriodic(0) || geometry.isPeriodic(1)
       || !near(radial_faces.front(), 0.0)
       || !near(radial_faces.back(), probhi[0])
       || !(probhi[0] > 0.0)
       || !standard_theta_domain) {
      amrex::Abort(
         "axisymmetric spherical elliptic coordinates require a nonperiodic "
         "physical (r,theta) grid with r in [0,R] and theta in [0,pi/2], "
         "[pi/2,pi], or [0,pi]"
      );
   }
   const bool toroidal =
      mode == AxisymmetricSphericalMode::toroidal_vector;
   const InitialDataEllipticBoundaryType origin_type = toroidal
      ? InitialDataEllipticBoundaryType::dirichlet
      : InitialDataEllipticBoundaryType::neumann;
   const InitialDataEllipticBoundaryType lower_angular_type =
      toroidal && lower_axis
      ? InitialDataEllipticBoundaryType::dirichlet
      : InitialDataEllipticBoundaryType::neumann;
   const InitialDataEllipticBoundaryType upper_angular_type =
      toroidal && upper_axis
      ? InitialDataEllipticBoundaryType::dirichlet
      : InitialDataEllipticBoundaryType::neumann;
   if (resolved_boundary_type(boundary, 0, 0, false) != origin_type
       || resolved_boundary_type(boundary, 0, 1, false)
          != lower_angular_type
       || resolved_boundary_type(boundary, 0, 1, true)
          != upper_angular_type) {
      amrex::Abort(toroidal
         ? "the spherical toroidal solve requires odd origin/axis and even "
           "equator regularity"
         : "the spherical scalar solve requires even origin, axis, and "
           "equator regularity");
   }
   const InitialDataEllipticBoundaryType radial_outer_type =
      resolved_boundary_type(boundary, 0, 0, true);
   if (radial_outer_type != InitialDataEllipticBoundaryType::dirichlet
       && radial_outer_type != InitialDataEllipticBoundaryType::robin) {
      amrex::Abort(
         "the spherical scalar solve requires Dirichlet or Robin radial closure"
      );
   }

   const int radial_cells = domain.length(0);
   const int angular_cells = domain.length(1);
   const int radial_begin = domain.smallEnd(0);
   const int angular_begin = domain.smallEnd(1);
   const int radial_end = domain.bigEnd(0);
   for (int face = 0; face <= angular_cells; ++face) {
      if (!near(angular_faces[face], problo[1] + face * dx[1])) {
         amrex::Abort(
            "the separable spherical solver currently requires uniform theta"
         );
      }
   }
   std::vector<amrex::Real> radial_centers(radial_cells);
   std::vector<amrex::Real> radial_widths(radial_cells);
   for (int radial_index = 0;
        radial_index < radial_cells;
        ++radial_index) {
      radial_centers[radial_index] = 0.5 * (
         radial_faces[radial_index] + radial_faces[radial_index + 1]
      );
      radial_widths[radial_index] =
         radial_faces[radial_index + 1] - radial_faces[radial_index];
   }
   constexpr int plane_index = 0;
   const std::size_t total_cells =
      static_cast<std::size_t>(radial_cells) * angular_cells;

   amrex::MultiFab host_rhs = host_copy(*rhs[0]);
   amrex::MultiFab host_coefficient_a = host_copy(boundary.coefficient_a(0));
   amrex::MultiFab host_coefficient_b = host_copy(boundary.coefficient_b(0));
   amrex::MultiFab host_coefficient_c = host_copy(boundary.coefficient_c(0));
   std::vector<amrex::Real> weighted_rhs(total_cells, 0.0);
   std::vector<amrex::Real> outer_a(angular_cells, 0.0);
   std::vector<amrex::Real> outer_b(angular_cells, 0.0);
   std::vector<amrex::Real> outer_c(angular_cells, 0.0);
   std::vector<int> outer_counts(angular_cells, 0);
   for (amrex::MFIter mfi(host_rhs); mfi.isValid(); ++mfi) {
      const amrex::Box box = mfi.validbox();
      const auto source = host_rhs.const_array(mfi);
      for (int j = box.smallEnd(1); j <= box.bigEnd(1); ++j) {
         const int angular_index = j - angular_begin;
         for (int i = box.smallEnd(0); i <= box.bigEnd(0); ++i) {
            const int radial_index = i - radial_begin;
            const amrex::Real radius = radial_centers[radial_index];
            weighted_rhs[
               static_cast<std::size_t>(radial_index) * angular_cells
               + angular_index
            ] = radius * radius * source(i, j, plane_index, 0);
         }
      }
   }
   for (amrex::MFIter mfi(host_coefficient_a); mfi.isValid(); ++mfi) {
      const amrex::Box valid_box = mfi.validbox();
      if (valid_box.bigEnd(0) != radial_end) {
         continue;
      }
      const auto coefficient_a = host_coefficient_a.const_array(mfi);
      const auto coefficient_b = host_coefficient_b.const_array(mfi);
      const auto coefficient_c = host_coefficient_c.const_array(mfi);
      const int ghost_i = radial_end + 1;
      for (int j = valid_box.smallEnd(1); j <= valid_box.bigEnd(1); ++j) {
         const int angular_index = j - angular_begin;
         outer_a[angular_index] = coefficient_a(
            ghost_i, j, plane_index, 0
         );
         outer_b[angular_index] = coefficient_b(
            ghost_i, j, plane_index, 0
         );
         outer_c[angular_index] = coefficient_c(
            ghost_i, j, plane_index, 0
         );
         outer_counts[angular_index] = 1;
      }
   }
   const auto communicator = amrex::ParallelContext::CommunicatorSub();
   amrex::ParallelAllReduce::Sum(
      weighted_rhs.data(), static_cast<int>(weighted_rhs.size()), communicator
   );
   amrex::ParallelAllReduce::Sum(
      outer_a.data(), angular_cells, communicator
   );
   amrex::ParallelAllReduce::Sum(
      outer_b.data(), angular_cells, communicator
   );
   amrex::ParallelAllReduce::Sum(
      outer_c.data(), angular_cells, communicator
   );
   amrex::ParallelAllReduce::Sum(
      outer_counts.data(), angular_cells, communicator
   );

   std::vector<amrex::Real> outer_gamma(angular_cells);
   amrex::Real outer_delta = 0.0;
   for (int angular_index = 0;
        angular_index < angular_cells;
        ++angular_index) {
      if (outer_counts[angular_index] != 1) {
         amrex::Abort("radial boundary data do not have one unique owner");
      }
      const amrex::Real a = outer_a[angular_index];
      const amrex::Real b = radial_outer_type
            == InitialDataEllipticBoundaryType::dirichlet
         ? 0.0
         : outer_b[angular_index];
      const amrex::Real outer_width = radial_widths.back();
      const amrex::Real denominator = 0.5 * a + b / outer_width;
      if (denominator == 0.0 || !std::isfinite(denominator)) {
         amrex::Abort("singular radial elliptic boundary coefficients");
      }
      const amrex::Real delta =
         (b / outer_width - 0.5 * a) / denominator;
      if (angular_index == 0) {
         outer_delta = delta;
      } else if (!near(delta, outer_delta)) {
         amrex::Abort(
            "the tensor-product spherical solve requires an angularly uniform "
            "radial boundary operator"
         );
      }
      outer_gamma[angular_index] = outer_c[angular_index] / denominator;
   }

   const amrex::Real angular_scale = 1.0 / (dx[1] * dx[1]);
   const amrex::Real angular_spin_squared =
      mode == AxisymmetricSphericalMode::toroidal_vector ? 1.0 : 0.0;
   std::vector<amrex::Real> radial_lower(radial_cells, 0.0);
   std::vector<amrex::Real> radial_diagonal(radial_cells, 0.0);
   std::vector<amrex::Real> radial_upper(radial_cells, 0.0);
   for (int radial_index = 0; radial_index < radial_cells; ++radial_index) {
      if (radial_index > 0) {
         const amrex::Real face_radius = radial_faces[radial_index];
         const amrex::Real center_distance =
            radial_centers[radial_index]
            - radial_centers[radial_index - 1];
         radial_lower[radial_index] = face_radius * face_radius / (
            radial_widths[radial_index] * center_distance
         );
      }
      if (radial_index + 1 < radial_cells) {
         const amrex::Real face_radius = radial_faces[radial_index + 1];
         const amrex::Real center_distance =
            radial_centers[radial_index + 1]
            - radial_centers[radial_index];
         radial_upper[radial_index] = face_radius * face_radius / (
            radial_widths[radial_index] * center_distance
         );
      }
      radial_diagonal[radial_index] = -(
         radial_lower[radial_index] + radial_upper[radial_index]
      );
   }
   const amrex::Real outer_face_coefficient =
      radial_faces.back() * radial_faces.back() / (
         radial_widths.back() * radial_widths.back()
      );
   radial_diagonal.back() += outer_face_coefficient * (
      outer_delta - 1.0
   );
   for (int angular_index = 0;
        angular_index < angular_cells;
        ++angular_index) {
      weighted_rhs[
         static_cast<std::size_t>(radial_cells - 1) * angular_cells
         + angular_index
      ] -= outer_face_coefficient * outer_gamma[angular_index];
   }

   std::vector<amrex::Real> angular_weight(angular_cells, 0.0);
   std::vector<amrex::Real> angular_diagonal(angular_cells, 0.0);
   std::vector<amrex::Real> angular_off_diagonal(angular_cells, 0.0);
   for (int angular_index = 0;
        angular_index < angular_cells;
        ++angular_index) {
      const amrex::Real theta = problo[1]
         + (angular_index + 0.5) * dx[1];
      angular_weight[angular_index] = std::sqrt(std::sin(theta));
      if (!(angular_weight[angular_index] > 0.0)) {
         amrex::Abort("invalid spherical angular cell weight");
      }
      angular_diagonal[angular_index] -= angular_spin_squared / (
         angular_weight[angular_index] * angular_weight[angular_index]
         * angular_weight[angular_index] * angular_weight[angular_index]
      );
   }
   for (int angular_index = 0;
        angular_index + 1 < angular_cells;
        ++angular_index) {
      const amrex::Real face_theta = problo[1]
         + (angular_index + 1) * dx[1];
      const amrex::Real face_coefficient =
         std::sin(face_theta) * angular_scale;
      angular_off_diagonal[angular_index] = face_coefficient / (
         angular_weight[angular_index]
         * angular_weight[angular_index + 1]
      );
      angular_diagonal[angular_index] -= face_coefficient / (
         angular_weight[angular_index] * angular_weight[angular_index]
      );
      angular_diagonal[angular_index + 1] -= face_coefficient / (
         angular_weight[angular_index + 1]
         * angular_weight[angular_index + 1]
      );
   }
   const SymmetricTridiagonalEigensystem angular_eigensystem =
      symmetric_tridiagonal_eigensystem(
         std::move(angular_diagonal), std::move(angular_off_diagonal)
      );

   const int rank = amrex::ParallelContext::MyProcSub();
   const int ranks = amrex::ParallelContext::NProcsSub();
   std::vector<amrex::Real> mode_solution(
      static_cast<std::size_t>(angular_cells) * radial_cells, 0.0
   );
   std::vector<amrex::Real> forward_upper(radial_cells, 0.0);
   std::vector<amrex::Real> forward_rhs(radial_cells, 0.0);
   for (int mode = rank; mode < angular_cells; mode += ranks) {
      const amrex::Real* eigenvector =
         angular_eigensystem.vectors.data()
         + static_cast<std::size_t>(mode) * angular_cells;
      amrex::Real* radial_mode = mode_solution.data()
         + static_cast<std::size_t>(mode) * radial_cells;
      for (int radial_index = 0;
           radial_index < radial_cells;
           ++radial_index) {
         amrex::Real projected_rhs = 0.0;
         const amrex::Real* radial_rhs = weighted_rhs.data()
            + static_cast<std::size_t>(radial_index) * angular_cells;
         for (int angular_index = 0;
              angular_index < angular_cells;
              ++angular_index) {
            projected_rhs += eigenvector[angular_index]
               * angular_weight[angular_index]
               * radial_rhs[angular_index];
         }
         radial_mode[radial_index] = projected_rhs;
      }

      amrex::Real denominator = radial_diagonal[0]
         + angular_eigensystem.eigenvalues[mode];
      if (denominator == 0.0 || !std::isfinite(denominator)) {
         amrex::Abort("singular spherical radial mode");
      }
      forward_upper[0] = radial_upper[0] / denominator;
      forward_rhs[0] = radial_mode[0] / denominator;
      for (int radial_index = 1;
           radial_index < radial_cells;
           ++radial_index) {
         denominator = radial_diagonal[radial_index]
            + angular_eigensystem.eigenvalues[mode]
            - radial_lower[radial_index]
               * forward_upper[radial_index - 1];
         if (denominator == 0.0 || !std::isfinite(denominator)) {
            amrex::Abort("singular spherical radial elimination");
         }
         forward_upper[radial_index] = radial_index + 1 < radial_cells
            ? radial_upper[radial_index] / denominator
            : 0.0;
         forward_rhs[radial_index] = (
            radial_mode[radial_index]
            - radial_lower[radial_index] * forward_rhs[radial_index - 1]
         ) / denominator;
      }
      radial_mode[radial_cells - 1] = forward_rhs[radial_cells - 1];
      for (int radial_index = radial_cells - 2;
           radial_index >= 0;
           --radial_index) {
         radial_mode[radial_index] = forward_rhs[radial_index]
            - forward_upper[radial_index]
               * radial_mode[radial_index + 1];
      }
   }

   std::vector<amrex::Real> solution_values(total_cells, 0.0);
   for (int mode = rank; mode < angular_cells; mode += ranks) {
      const amrex::Real* eigenvector =
         angular_eigensystem.vectors.data()
         + static_cast<std::size_t>(mode) * angular_cells;
      const amrex::Real* radial_mode = mode_solution.data()
         + static_cast<std::size_t>(mode) * radial_cells;
      for (int radial_index = 0;
           radial_index < radial_cells;
           ++radial_index) {
         amrex::Real* radial_solution = solution_values.data()
            + static_cast<std::size_t>(radial_index) * angular_cells;
         for (int angular_index = 0;
              angular_index < angular_cells;
              ++angular_index) {
            radial_solution[angular_index] += eigenvector[angular_index]
               * radial_mode[radial_index]
               / angular_weight[angular_index];
         }
      }
   }
   amrex::ParallelAllReduce::Sum(
      solution_values.data(), static_cast<int>(solution_values.size()),
      communicator
   );

   amrex::Real residual = 0.0;
   for (int radial_index = 0;
        radial_index < radial_cells;
        ++radial_index) {
      for (int angular_index = 0;
           angular_index < angular_cells;
           ++angular_index) {
         const std::size_t offset =
            static_cast<std::size_t>(radial_index) * angular_cells
            + angular_index;
         amrex::Real applied = radial_diagonal[radial_index]
            * solution_values[offset];
         if (radial_index > 0) {
            applied += radial_lower[radial_index]
               * solution_values[offset - angular_cells];
         }
         if (radial_index + 1 < radial_cells) {
            applied += radial_upper[radial_index]
               * solution_values[offset + angular_cells];
         }
         if (angular_index > 0) {
            const amrex::Real theta = problo[1]
               + (angular_index + 0.5) * dx[1];
            const amrex::Real face_theta = problo[1]
               + angular_index * dx[1];
            const amrex::Real coefficient =
               std::sin(face_theta) / std::sin(theta) * angular_scale;
            applied += coefficient * (
               solution_values[offset - 1] - solution_values[offset]
            );
         }
         if (angular_index + 1 < angular_cells) {
            const amrex::Real theta = problo[1]
               + (angular_index + 0.5) * dx[1];
            const amrex::Real face_theta = problo[1]
               + (angular_index + 1) * dx[1];
            const amrex::Real coefficient =
               std::sin(face_theta) / std::sin(theta) * angular_scale;
            applied += coefficient * (
               solution_values[offset + 1] - solution_values[offset]
            );
         }
         const amrex::Real theta = problo[1]
            + (angular_index + 0.5) * dx[1];
         applied -= angular_spin_squared / (
            std::sin(theta) * std::sin(theta)
         ) * solution_values[offset];
         residual = std::max(
            residual, std::abs(applied - weighted_rhs[offset])
         );
      }
   }

   amrex::MultiFab host_solution = host_copy(*solution[0]);
   for (amrex::MFIter mfi(host_solution); mfi.isValid(); ++mfi) {
      const amrex::Box box = mfi.validbox();
      const auto destination = host_solution.array(mfi);
      for (int j = box.smallEnd(1); j <= box.bigEnd(1); ++j) {
         const int angular_index = j - angular_begin;
         for (int i = box.smallEnd(0); i <= box.bigEnd(0); ++i) {
            const int radial_index = i - radial_begin;
            destination(i, j, plane_index, 0) = solution_values[
               static_cast<std::size_t>(radial_index) * angular_cells
               + angular_index
            ];
         }
      }
   }
   copy_host_to_device(*solution[0], host_solution);
   return elliptic_result(residual);
#endif
}

} // namespace

InitialDataEllipticResult solve_initial_data_axisymmetric_spherical_poisson(
   InitialDataEllipticHierarchy const& hierarchy,
   amrex::Vector<amrex::MultiFab*> const& solution,
   amrex::Vector<amrex::MultiFab const*> const& rhs,
   InitialDataEllipticBoundaryConditions const& boundary,
   InitialDataEllipticOptions const& options
)
{
   return solve_axisymmetric_spherical_separable(
      hierarchy,
      solution,
      rhs,
      boundary,
      options,
      AxisymmetricSphericalMode::scalar
   );
}

InitialDataEllipticResult
solve_initial_data_axisymmetric_spherical_toroidal_vector_poisson(
   InitialDataEllipticHierarchy const& hierarchy,
   amrex::Vector<amrex::MultiFab*> const& azimuthal_solution,
   amrex::Vector<amrex::MultiFab const*> const& azimuthal_source,
   InitialDataEllipticBoundaryConditions const& boundary,
   InitialDataEllipticOptions const& options
)
{
   return solve_axisymmetric_spherical_separable(
      hierarchy,
      azimuthal_solution,
      azimuthal_source,
      boundary,
      options,
      AxisymmetricSphericalMode::toroidal_vector
   );
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
   vector_laplacian.setMaxOrder(options.boundary_interpolation_order);
   vector_laplacian.setScalars(0.0, 1.0);
   const LoweredBoundaryConditions lowered = lower_boundary_conditions(
      hierarchy, boundary
   );
   if (lowered.has_robin) {
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
