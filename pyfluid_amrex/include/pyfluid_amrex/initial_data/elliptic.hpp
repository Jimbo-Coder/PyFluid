#pragma once

#include <AMReX_Array.H>
#include <AMReX_BCRec.H>
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Geometry.H>
#include <AMReX_Interpolater.H>
#include <AMReX_MultiFab.H>
#include <AMReX_Vector.H>

#include <memory>

namespace pyfluid_amrex {

enum class InitialDataEllipticCentering {
   cell,
   vertex,
};

struct InitialDataEllipticHierarchy {
   amrex::Vector<amrex::Geometry> geometry;
   amrex::Vector<amrex::BoxArray> grids;
   amrex::Vector<amrex::DistributionMapping> distribution;
   amrex::Vector<amrex::IntVect> refinement_ratio;
   amrex::Interpolater* cell_interpolater = &amrex::cell_cons_interp;
};

struct InitialDataEllipticOptions {
   int max_order = 4;
   int max_iterations = 200;
   int max_fmg_iterations = 0;
   int verbose = 0;
   amrex::Real relative_tolerance = 1.0e-12;
   amrex::Real absolute_tolerance = 0.0;
};

struct InitialDataEllipticResult {
   amrex::Real residual = 0.0;
   amrex::Real boundary_residual = 0.0;
   int boundary_iterations = 0;
};

struct InitialDataEllipticBoundaryConditions {
   // On each nonperiodic coordinate face: A f + B d_d f = C. B multiplies
   // the increasing-coordinate derivative; the backend handles face normals.
   static InitialDataEllipticBoundaryConditions constant(
      InitialDataEllipticHierarchy const& hierarchy,
      int num_components,
      amrex::Real coefficient_a,
      amrex::Real coefficient_b,
      amrex::Real coefficient_c,
      InitialDataEllipticCentering centering =
         InitialDataEllipticCentering::cell
   );

   amrex::MultiFab& coefficient_a(int level) noexcept;
   amrex::MultiFab& coefficient_b(int level) noexcept;
   amrex::MultiFab& coefficient_c(int level) noexcept;
   amrex::MultiFab const& coefficient_a(int level) const noexcept;
   amrex::MultiFab const& coefficient_b(int level) const noexcept;
   amrex::MultiFab const& coefficient_c(int level) const noexcept;
   int num_levels() const noexcept;
   InitialDataEllipticCentering centering() const noexcept;
   InitialDataEllipticBoundaryConditions component(int component) const;

   int num_components = 0;

private:
   amrex::Vector<std::unique_ptr<amrex::MultiFab>> coefficient_a_;
   amrex::Vector<std::unique_ptr<amrex::MultiFab>> coefficient_b_;
   amrex::Vector<std::unique_ptr<amrex::MultiFab>> coefficient_c_;
   InitialDataEllipticCentering centering_ = InitialDataEllipticCentering::cell;
};

// Populate the common coefficient fields for
// f = f_inf + amplitude/r^p; the unknown amplitude cancels.
void set_centered_power_law_robin_boundary(
   InitialDataEllipticHierarchy const& hierarchy,
   InitialDataEllipticBoundaryConditions& boundary,
   amrex::Vector<amrex::Real> const& asymptotic_values,
   amrex::Real falloff_power = 1.0
);

// Fill same-level, coarse-fine, periodic, and physical ghosts according to
// the centering carried by the field hierarchy.
void fill_initial_data_ghosts(
   InitialDataEllipticHierarchy const& hierarchy,
   amrex::Vector<amrex::MultiFab*> const& fields,
   amrex::Vector<amrex::BCRec> const& boundary_conditions
);

InitialDataEllipticResult solve_initial_data_poisson(
   InitialDataEllipticHierarchy const& hierarchy,
   amrex::Vector<amrex::MultiFab*> const& solution,
   amrex::Vector<amrex::MultiFab const*> const& rhs,
   InitialDataEllipticBoundaryConditions const& boundary,
   InitialDataEllipticOptions const& options = {}
);

InitialDataEllipticResult solve_initial_data_conformal_vector_laplacian(
   InitialDataEllipticHierarchy const& hierarchy,
   amrex::Vector<amrex::MultiFab*> const& vector_potential,
   amrex::Vector<amrex::MultiFab const*> const& source,
   InitialDataEllipticBoundaryConditions const& boundary,
   InitialDataEllipticOptions const& options = {}
);

} // namespace pyfluid_amrex
