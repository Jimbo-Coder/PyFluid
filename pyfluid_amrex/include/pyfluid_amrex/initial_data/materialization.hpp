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

enum class ConformalConstraintFormulation {
   ctt,
   cttk,
   xcts,
};

struct ConformalConstraintSummary {
   int coupled_iterations = 0;
   amrex::Real relative_update = 0.0;
   amrex::Real hamiltonian_residual = 0.0;
   amrex::Real lapse_residual = 0.0;
   amrex::Real momentum_residual = 0.0;
   bool converged = false;
};

struct ConformalBlackHoleParameters {
   bool active = false;
   amrex::Real mass = 1.0;
};

struct ConformalMaterializationResult {
   int matter_iterations = 0;
   amrex::Real relative_matter_update = 0.0;
   ConformalConstraintSummary constraints{};
   RotatingStarUpdateSummary rotating_star{};
   EquilibriumTorusUpdateSummary equilibrium_torus{};
};

using ConformalContinuationObserver = std::function<void(
   int continuation_step,
   int continuation_steps,
   amrex::Real angular_velocity,
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
   amrex::Real relative_update = 0.0;
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
   ConformalStellarMatterUpdater const& stellar_matter_updater = {}
);

} // namespace pyfluid_amrex
