#pragma once

#include <AMReX_MultiFab.H>
#include <AMReX_Vector.H>

#include <functional>

#include "pyfluid_amrex/config.hpp"
#include "pyfluid_amrex/eos/types.hpp"
#include "pyfluid_amrex/initial_data/ctt.hpp"
#include "pyfluid_amrex/initial_data/elliptic.hpp"
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

struct ConformalMaterializationResult {
   int matter_iterations = 0;
   amrex::Real relative_matter_update = 0.0;
   ConformalConstraintSummary constraints{};
   RotatingStarUpdateSummary rotating_star{};
};

using ConformalContinuationObserver = std::function<void(
   int continuation_step,
   int continuation_steps,
   amrex::Real angular_velocity,
   RotatingStarUpdateSummary const& rotating_star
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
   ConformalConstraintFormulation formulation,
   RotatingStarParameters const& rotating_star,
   bool seed_from_spacetime = false,
   ConformalContinuationObserver const& continuation_observer = {}
);

} // namespace pyfluid_amrex
