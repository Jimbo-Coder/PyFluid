#pragma once

#include <AMReX_Array.H>
#include <AMReX_BCRec.H>
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Geometry.H>
#include <AMReX_Interpolater.H>
#include <AMReX_MultiFab.H>
#include <AMReX_Vector.H>

#include <array>
#include <memory>

#include "pyfluid_amrex/fields.hpp"

namespace pyfluid_amrex {

using namespace amrex::literals;

enum class InitialDataEllipticCentering {
   cell,
   vertex,
};

struct InitialDataCoordinateLine {
   amrex::Vector<amrex::Real> faces;
   amrex::Vector<amrex::Real> centers;
   amrex::Vector<amrex::Real> widths;
   bool uniform = true;
};

struct InitialDataEllipticHierarchy {
   amrex::Vector<amrex::Geometry> geometry;
   amrex::Vector<amrex::BoxArray> grids;
   amrex::Vector<amrex::DistributionMapping> distribution;
   amrex::Vector<amrex::IntVect> refinement_ratio;
   amrex::Interpolater* cell_interpolater = &amrex::cell_cons_interp;
   amrex::GpuArray<int, AMREX_SPACEDIM> reflection_lo{};
   amrex::GpuArray<int, AMREX_SPACEDIM> reflection_hi{};

   // Runtime grid directions map to these canonical Cartesian physical axes
   // (x=0, y=1, z=2).  The default is x, x-y, or x-y-z according to the
   // executable dimension.  Reduced conformal-constraint operators retain
   // all three physical vector/tensor components while setting derivatives
   // along omitted axes to zero.
   amrex::GpuArray<int, AMREX_SPACEDIM> cartesian_physical_axes{
      AMREX_D_DECL(0, 1, 2)
   };

   // Optional physical face coordinates for a solver-private mapped grid.
   // Empty directions use the uniform AMReX Geometry coordinates.  These
   // coordinates describe elliptic unknowns only; portable ID fields are
   // materialized onto their destination chart before evolution state exists.
   amrex::Vector<
      std::array<amrex::Vector<amrex::Real>, AMREX_SPACEDIM>
   > physical_coordinate_faces;

   amrex::Vector<amrex::Real> coordinate_faces(
      int level, int direction
   ) const;
   InitialDataCoordinateLine coordinate_line(
      int level, int direction
   ) const;
};

// Uniform spacing through a compact source region followed by geometrically
// increasing radial intervals to the gravity boundary.  The returned faces
// include both endpoints.  TOV and RNS can share this mapping; matter support
// remains a field property rather than a second coordinate system.
amrex::Vector<amrex::Real> surface_focused_radial_faces(
   amrex::Real surface_radius,
   amrex::Real outer_radius,
   int surface_cells,
   int exterior_cells
);

// Allocate most radial cells to the complete stellar source region inferred
// from a spherical seed, then stretch smoothly through the source-free
// gravitational exterior.  This is a solver-private ID coordinate, not an
// evolution AMR hierarchy.
amrex::Vector<amrex::Real> stellar_adaptive_radial_faces(
   amrex::Real seed_surface_radius,
   amrex::Real outer_radius,
   int radial_cells,
   amrex::Real source_radius_factor,
   amrex::Real exterior_fraction
);

// Resolve a finite inner boundary while spending progressively fewer radial
// cells in the smooth exterior. The final cell is
// outer_to_inner_cell_width_ratio times wider than the first; one recovers a
// uniform shell. This is suitable for apparent-horizon and excision surfaces,
// not for a grid containing a regular radial origin.
amrex::Vector<amrex::Real> stretched_radial_shell_faces(
   amrex::Real inner_radius,
   amrex::Real outer_radius,
   int radial_cells,
   amrex::Real outer_to_inner_cell_width_ratio
);

enum class InitialDataEllipticBoundaryType {
   periodic,
   dirichlet,
   neumann,
   robin,
};

struct InitialDataEllipticOptions {
   // AMReX uses this only for coarse-fine boundary interpolation.
   // The cell-centered and nodal interior operators remain second order.
   int boundary_interpolation_order = 4;
   int max_iterations = 200;
   int max_fmg_iterations = 0;
   int verbose = 0;
   amrex::Real relative_tolerance = 1.0e-12_rt;
   amrex::Real absolute_tolerance = 0.0_rt;
};

struct InitialDataEllipticResult {
   amrex::Real residual = 0.0_rt;
   amrex::Real boundary_residual = 0.0_rt;
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
   InitialDataEllipticBoundaryConditions components(
      amrex::Vector<int> const& components
   ) const;
   InitialDataEllipticBoundaryType lower_type(
      int component, int direction
   ) const noexcept;
   InitialDataEllipticBoundaryType upper_type(
      int component, int direction
   ) const noexcept;
   int lower_reflection_parity(int component, int direction) const noexcept;
   int upper_reflection_parity(int component, int direction) const noexcept;

   int num_components = 0;

private:
   amrex::Vector<std::unique_ptr<amrex::MultiFab>> coefficient_a_;
   amrex::Vector<std::unique_ptr<amrex::MultiFab>> coefficient_b_;
   amrex::Vector<std::unique_ptr<amrex::MultiFab>> coefficient_c_;
   amrex::Vector<InitialDataEllipticBoundaryType> lower_types_;
   amrex::Vector<InitialDataEllipticBoundaryType> upper_types_;
   amrex::Vector<int> lower_reflection_parity_;
   amrex::Vector<int> upper_reflection_parity_;
   InitialDataEllipticCentering centering_ = InitialDataEllipticCentering::cell;

   friend void set_initial_data_reflection_boundaries(
      InitialDataEllipticHierarchy const&,
      InitialDataEllipticBoundaryConditions&,
      amrex::Vector<GridFieldSymmetryKind> const&
   );
   friend void set_centered_power_law_robin_boundary(
      InitialDataEllipticHierarchy const&,
      InitialDataEllipticBoundaryConditions&,
      amrex::Vector<amrex::Real> const&,
      amrex::Real,
      amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const*
   );
   friend void set_axisymmetric_spherical_toroidal_regularity(
      InitialDataEllipticHierarchy const&,
      InitialDataEllipticBoundaryConditions&,
      int
   );
   friend void set_axisymmetric_spherical_radial_robin_boundary(
      InitialDataEllipticHierarchy const&,
      InitialDataEllipticBoundaryConditions&,
      amrex::Vector<amrex::Real> const&,
      amrex::Real
   );
   friend void set_axisymmetric_spherical_rotating_horizon_boundary(
      InitialDataEllipticHierarchy const&,
      InitialDataEllipticBoundaryConditions&,
      amrex::Real,
      amrex::Real,
      int
   );
   friend void set_spherical_radial_robin_boundary(
      InitialDataEllipticHierarchy const&,
      InitialDataEllipticBoundaryConditions&,
      amrex::Vector<amrex::Real> const&,
      amrex::Real
   );
};

// Apply homogeneous reflection conditions on hierarchy faces marked as
// reflection planes. Scalar/tensor component parity comes from the GF catalog.
void set_initial_data_reflection_boundaries(
   InitialDataEllipticHierarchy const& hierarchy,
   InitialDataEllipticBoundaryConditions& boundary,
   amrex::Vector<GridFieldSymmetryKind> const& component_symmetries
);

// Mark one scalar storage component as an orthonormal azimuthal vector on an
// r-theta hierarchy: odd at a radial origin and the rotation axis, even at the
// equator. On a shell, both radial conditions remain owned by the caller.
void set_axisymmetric_spherical_toroidal_regularity(
   InitialDataEllipticHierarchy const& hierarchy,
   InitialDataEllipticBoundaryConditions& boundary,
   int component = 0
);

// Apply f = f_inf + amplitude/r^p at the radial outer face of a physical
// r-theta hierarchy without interpreting theta as a Cartesian distance.
void set_axisymmetric_spherical_radial_robin_boundary(
   InitialDataEllipticHierarchy const& hierarchy,
   InitialDataEllipticBoundaryConditions& boundary,
   amrex::Vector<amrex::Real> const& asymptotic_values,
   amrex::Real falloff_power
);

// Impose the Cook-Pfeiffer tangential shift on a spherical excision surface.
// The stored scalar is the orthonormal azimuthal component, so the rotational
// conformal Killing vector Omega_H (partial_phi)^i has
// beta_hat_phi = Omega_H r_H sin(theta). The normal shift remains owned by the
// lapse/slicing choice; this helper changes only the shell-inner tangential
// Dirichlet data.
void set_axisymmetric_spherical_rotating_horizon_boundary(
   InitialDataEllipticHierarchy const& hierarchy,
   InitialDataEllipticBoundaryConditions& boundary,
   amrex::Real horizon_radius,
   amrex::Real horizon_rotation_parameter,
   int component = 0
);

// Apply the same asymptotic radial condition on an r-theta-phi hierarchy.
// The coefficient contract remains expressed using the physical increasing-r
// derivative; a mapped spherical backend converts it to its logical radial
// derivative internally.
void set_spherical_radial_robin_boundary(
   InitialDataEllipticHierarchy const& hierarchy,
   InitialDataEllipticBoundaryConditions& boundary,
   amrex::Vector<amrex::Real> const& asymptotic_values,
   amrex::Real falloff_power
);

// Populate the common coefficient fields for
// f = f_inf + amplitude/r^p; the unknown amplitude cancels.
void set_centered_power_law_robin_boundary(
   InitialDataEllipticHierarchy const& hierarchy,
   InitialDataEllipticBoundaryConditions& boundary,
   amrex::Vector<amrex::Real> const& asymptotic_values,
   amrex::Real falloff_power = 1.0_rt,
   amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const* physical_center = nullptr
);

amrex::Vector<amrex::BCRec> initial_data_boundary_records(
   InitialDataEllipticHierarchy const& hierarchy,
   amrex::Vector<GridFieldSymmetryKind> const& component_symmetries,
   int nonsymmetry_type = amrex::BCType::foextrap
);

// Build reflection records from the explicit component boundary metadata.
// This is needed by reduced-coordinate components whose parity cannot be
// inferred from a Cartesian tensor component alone.
amrex::Vector<amrex::BCRec> initial_data_boundary_records(
   InitialDataEllipticHierarchy const& hierarchy,
   InitialDataEllipticBoundaryConditions const& boundary,
   int nonsymmetry_type = amrex::BCType::foextrap
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

// Solve the axisymmetric physical scalar equation
//
//   (1/r^2) d_r(r^2 d_r u)
//     + (1/(r^2 sin(theta))) d_theta(sin(theta) d_theta u) = rhs
//
// on a two-dimensional logical Cartesian hierarchy whose ordered physical
// coordinates are (r, theta). The unknown and physical rhs are cell-centered,
// so the coordinate origin and axis are boundary faces rather than duplicated
// degrees of freedom. Multiplying the equation by r^2 sin(theta) gives the
// regular conservative operator
//
//   sin(theta) d_r(r^2 d_r u)
//     + d_theta(sin(theta) d_theta u) = r^2 sin(theta) rhs.
//
// Callers supply origin or shell-inner, axis/equator, and outer conditions
// through the ordinary elliptic boundary contract. The tensor-product backend
// requires a shell's A/B inner operator to be independent of theta;
// inhomogeneous C data may vary over the inner face.
InitialDataEllipticResult solve_initial_data_axisymmetric_spherical_poisson(
   InitialDataEllipticHierarchy const& hierarchy,
   amrex::Vector<amrex::MultiFab*> const& solution,
   amrex::Vector<amrex::MultiFab const*> const& rhs,
   InitialDataEllipticBoundaryConditions const& boundary,
   InitialDataEllipticOptions const& options = {}
);

// Solve
//
//   Laplacian(u) + reaction(r) u = rhs
//
// with the same mapped-radial, axisymmetric spherical operator.  The reaction
// coefficient must be spherically symmetric.  Keeping it in the existing
// separable operator permits robust Newton steps for static spherical XCTS
// without introducing a second elliptic discretization.
InitialDataEllipticResult solve_initial_data_axisymmetric_spherical_helmholtz(
   InitialDataEllipticHierarchy const& hierarchy,
   amrex::Vector<amrex::MultiFab*> const& solution,
   amrex::Vector<amrex::MultiFab const*> const& rhs,
   amrex::Vector<amrex::MultiFab const*> const& reaction,
   InitialDataEllipticBoundaryConditions const& boundary,
   InitialDataEllipticOptions const& options = {}
);

// Solve the full physical scalar equation
//
//   (1/r^2) d_r(r^2 d_r u)
//     + (1/(r^2 sin(theta))) d_theta(sin(theta) d_theta u)
//     + (1/(r^2 sin(theta)^2)) d_phi^2 u = rhs
//
// on a cell-centered three-dimensional logical hierarchy ordered as
// (r,theta,phi). Phi is periodic, theta spans [0,pi] or one equator-reflected
// half, and radial coordinates may use explicit mapped faces. The backend
// solves the Jacobian-weighted conservative equation, so the coordinate
// origin and axes remain boundary faces rather than duplicated unknowns.
InitialDataEllipticResult solve_initial_data_spherical_poisson(
   InitialDataEllipticHierarchy const& hierarchy,
   amrex::Vector<amrex::MultiFab*> const& solution,
   amrex::Vector<amrex::MultiFab const*> const& rhs,
   InitialDataEllipticBoundaryConditions const& boundary,
   InitialDataEllipticOptions const& options = {}
);

// Solve the phi-independent scalar equation on a full (r,theta,phi) storage
// hierarchy. This is the native three-dimensional storage owner for an
// axisymmetric physical problem: each periodic phi plane carries the same
// r-theta operator and no unused nonaxisymmetric mode is introduced into the
// linear solve. The separate full-spherical operator above remains the owner
// for genuinely phi-dependent fields.
InitialDataEllipticResult solve_initial_data_spherical_axisymmetric_poisson(
   InitialDataEllipticHierarchy const& hierarchy,
   amrex::Vector<amrex::MultiFab*> const& solution,
   amrex::Vector<amrex::MultiFab const*> const& rhs,
   InitialDataEllipticBoundaryConditions const& boundary,
   InitialDataEllipticOptions const& options = {}
);

// Solve the same full spherical operator for an axisymmetric orthonormal
// azimuthal vector component. For d_phi v_hat_phi = 0 the conformal vector
// Laplacian reduces to the scalar spherical Laplacian minus
// v_hat_phi/(r^2 sin(theta)^2). The storage may retain any periodic phi
// extent, but the caller owns and validates axisymmetry.
InitialDataEllipticResult
solve_initial_data_spherical_toroidal_vector_poisson(
   InitialDataEllipticHierarchy const& hierarchy,
   amrex::Vector<amrex::MultiFab*> const& azimuthal_solution,
   amrex::Vector<amrex::MultiFab const*> const& azimuthal_source,
   InitialDataEllipticBoundaryConditions const& boundary,
   InitialDataEllipticOptions const& options = {}
);

InitialDataEllipticResult
solve_initial_data_spherical_axisymmetric_toroidal_vector_poisson(
   InitialDataEllipticHierarchy const& hierarchy,
   amrex::Vector<amrex::MultiFab*> const& azimuthal_solution,
   amrex::Vector<amrex::MultiFab const*> const& azimuthal_source,
   InitialDataEllipticBoundaryConditions const& boundary,
   InitialDataEllipticOptions const& options = {}
);

// Solve the axisymmetric physical azimuthal-vector equation
//
//   (1/r^2) d_r(r^2 d_r v_hat_phi)
//     + (1/(r^2 sin(theta))) d_theta(sin(theta) d_theta v_hat_phi)
//     - v_hat_phi/(r^2 sin(theta)^2) = source_hat_phi,
//
// where v_hat_phi is the orthonormal azimuthal component. A pure axisymmetric
// azimuthal vector is divergence-free, so this is also its flat conformal-
// vector Laplacian. Finite regular solutions vanish at the origin and axis;
// the singular angular term enforces that behavior on the cell-centered
// r-theta grid, while the equator has even reflection regularity.
InitialDataEllipticResult
solve_initial_data_axisymmetric_spherical_toroidal_vector_poisson(
   InitialDataEllipticHierarchy const& hierarchy,
   amrex::Vector<amrex::MultiFab*> const& azimuthal_solution,
   amrex::Vector<amrex::MultiFab const*> const& azimuthal_source,
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
