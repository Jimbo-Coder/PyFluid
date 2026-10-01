#pragma once

#include <AMReX_MultiFab.H>
#include <AMReX_Vector.H>

#include <functional>

#include "pyfluid_amrex/config.hpp"
#include "pyfluid_amrex/eos/types.hpp"
#include "pyfluid_amrex/initial_data/ctt.hpp"
#include "pyfluid_amrex/initial_data/elliptic.hpp"
#include "pyfluid_amrex/initial_data/equilibrium_torus.hpp"
#include "pyfluid_amrex/initial_data/provider.hpp"
#include "pyfluid_amrex/initial_data/rotating_star.hpp"
#include "pyfluid_amrex/physical_system.hpp"

namespace pyfluid_amrex {

using namespace amrex::literals;

enum class ConformalConstraintFormulation {
   ctt,
   cttk,
   xcts,
};

struct ConformalConstraintSummary {
   int coupled_iterations = 0;
   amrex::Real relative_update = 0.0_rt;
   amrex::Real hamiltonian_residual = 0.0_rt;
   amrex::Real lapse_residual = 0.0_rt;
   amrex::Real momentum_residual = 0.0_rt;
   bool converged = false;
};

struct ConformalBlackHoleParameters {
   bool active = false;
   amrex::Real mass = 1.0_rt;
   // Signed physical horizon-spin target along the native spherical axis.
   // The XCTS horizon rotation parameter is calibrated to this quasi-local
   // Christodoulou spin; it is deliberately not exposed as a second knob.
   amrex::Real dimensionless_spin = 0.0_rt;
};

struct ConformalBlackHoleSummary {
   amrex::Real throat_area = -1.0_rt;
   amrex::Real irreducible_mass = -1.0_rt;
   amrex::Real equatorial_circumference = -1.0_rt;
   amrex::Real polar_circumference = -1.0_rt;
   amrex::Real kerr_horizon_mass = -1.0_rt;
   amrex::Real kerr_dimensionless_spin = -1.0_rt;
   amrex::Real angular_momentum = 0.0_rt;
   amrex::Real christodoulou_mass = -1.0_rt;
   amrex::Real christodoulou_dimensionless_spin = 0.0_rt;
};

struct ConformalMaterializationResult {
   int matter_iterations = 0;
   amrex::Real relative_matter_update = 0.0_rt;
   ConformalConstraintSummary constraints{};
   ConformalBlackHoleSummary black_hole{};
   RotatingStarUpdateSummary rotating_star{};
   EquilibriumTorusUpdateSummary equilibrium_torus{};
};

using ConformalContinuationObserver = std::function<void(
   int continuation_step,
   int continuation_steps,
   amrex::Real stellar_angular_velocity,
   amrex::Real disk_angular_velocity,
   ConformalBlackHoleSummary const& black_hole,
   RotatingStarUpdateSummary const& rotating_star,
   EquilibriumTorusUpdateSummary const& equilibrium_torus
)>;

// Optional runtime hook invoked after each matter update and before the
// corresponding constraint-source assembly. Keeping it outside the compiled
// formulation/EoS products lets magnetic-ID ownership remain opt-in.
using ConformalMatterObserver = std::function<void()>;

// Optional runtime owner for a coupled stellar matter update.  It replaces
// the ordinary circular-fluid update so a Grad-Shafranov field and magnetic
// Bernoulli term can advance in the same fixed point without adding a
// formulation/EoS template product.  The boolean is false on the protected
// first spherical-seed constraint sweep.
struct ConformalStellarMatterUpdateResult {
   RotatingStarUpdateSummary rotating_star{};
   amrex::Real relative_update = 0.0_rt;
};

using ConformalStellarMatterUpdater = std::function<
   ConformalStellarMatterUpdateResult(
   RotatingStarParameters const&,
   bool update_fluid
)>;

ConformalMaterializationResult materialize_conformal_initial_data(
   InitialDataEllipticHierarchy const& hierarchy,
   amrex::Vector<amrex::MultiFab*> const& primitive_state,
   amrex::Vector<amrex::MultiFab*> const& spacetime_state,
   amrex::Vector<amrex::MultiFab*> const& cell_spacetime,
   PhysicalSystem system,
   int eos_method,
   EOSParams const& eos,
   InitialDataFieldSelection const& fields,
   int coordinate_kind,
   ConformalConstraintFormulation formulation,
   ConformalBlackHoleParameters const& black_hole,
   RotatingStarParameters const& rotating_star,
   EquilibriumTorusParameters const& equilibrium_torus,
   bool seed_from_spacetime = false,
   ConformalContinuationObserver const& continuation_observer = {},
   ConformalMatterObserver const& matter_observer = {},
   ConformalStellarMatterUpdater const& stellar_matter_updater = {},
   bool preserve_conformal_baryon_density = false
);

} // namespace pyfluid_amrex
