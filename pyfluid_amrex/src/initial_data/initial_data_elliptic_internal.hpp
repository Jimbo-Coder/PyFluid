#pragma once

#include <AMReX.H>
#include <AMReX_LO_BCTYPES.H>
#include <AMReX_MLMG.H>

#include <memory>

#include "pyfluid_amrex/initial_data/elliptic.hpp"

namespace pyfluid_amrex::detail {

using BoundaryTypeArray =
   amrex::Array<amrex::LinOpBCType, AMREX_SPACEDIM>;

struct LoweredBoundaryConditions {
   bool has_robin = false;
   amrex::Vector<BoundaryTypeArray> lower_types;
   amrex::Vector<BoundaryTypeArray> upper_types;
   amrex::Vector<std::unique_ptr<amrex::MultiFab>> level_data;
   amrex::Vector<std::unique_ptr<amrex::MultiFab>> robin_normal_b;
};

int validate_elliptic_solve(
   InitialDataEllipticHierarchy const& hierarchy,
   amrex::Vector<amrex::MultiFab*> const& solution,
   amrex::Vector<amrex::MultiFab const*> const& rhs,
   InitialDataEllipticBoundaryConditions const& boundary,
   InitialDataEllipticOptions const& options,
   int num_components
);

InitialDataEllipticBoundaryType resolved_boundary_type(
   InitialDataEllipticBoundaryConditions const& boundary,
   int component,
   int direction,
   bool upper
);

LoweredBoundaryConditions lower_boundary_conditions(
   InitialDataEllipticHierarchy const& hierarchy,
   InitialDataEllipticBoundaryConditions const& boundary
);

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
            lowered.level_data[level].get(),
            &boundary.coefficient_a(level),
            lowered.robin_normal_b[level].get(),
            &boundary.coefficient_c(level)
         );
      }
   }
}

void configure_mlmg(
   amrex::MLMG& mlmg,
   InitialDataEllipticOptions const& options
);

InitialDataEllipticResult elliptic_result(amrex::Real residual);

} // namespace pyfluid_amrex::detail
