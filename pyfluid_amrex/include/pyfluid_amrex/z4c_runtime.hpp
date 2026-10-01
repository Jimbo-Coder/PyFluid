#pragma once

#include <AMReX_BCRec.H>
#include <AMReX_Geometry.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_MultiFab.H>
#include <AMReX_Vector.H>

#include "pyfluid_amrex/config.hpp"
#include "pyfluid_amrex/excision_geometry.hpp"
#include "pyfluid_amrex/z4c_state.hpp"

namespace pyfluid_amrex {

using namespace amrex::literals;

struct RelativisticInitialDataView;

void debug_check_spacetime_metric(
   amrex::MultiFab const& state, amrex::IntVect const& ghost_width,
   bool enabled, bool apply_floors_next = false);

enum class Z4cInitialData : int {
   Minkowski,
   GaugeWave,
   LinearWave,
   RobustStability,
};

struct Z4cInitialDataParameters {
   Z4cInitialData kind = Z4cInitialData::Minkowski;
   amrex::Real amplitude = 0.0_rt;
   amrex::Real wave_number = 1.0_rt;
};

Z4cInitialDataParameters z4c_initial_data_parameters(RunConfig const& cfg);
Z4cEvolutionParameters z4c_evolution_parameters(RunConfig const& cfg);
Z4cLapseGauge z4c_lapse_gauge(RunConfig const& cfg);
Z4cShiftGauge z4c_shift_gauge(RunConfig const& cfg);
amrex::Vector<amrex::BCRec> z4c_boundary_records(
   RunConfig const& cfg,
   amrex::Geometry const& geometry
);

// Stage-local outgoing-wave closure, separate from FillPatch ghost data.
// Leaves periodic/reflection boundaries and the bulk PDE unchanged.
void apply_z4c_outer_boundary_rhs(
   amrex::MultiFab const& state,
   amrex::MultiFab& rhs,
   amrex::Geometry const& geometry,
   RunConfig const& cfg
);

void initialize_z4c_state(
   amrex::MultiFab& state,
   amrex::Geometry const& geometry,
   RunConfig const& cfg,
   amrex::Real time,
   RelativisticInitialDataView const& initial_data
);

void initialize_z4c_cell_state(
   amrex::MultiFab& state,
   amrex::Geometry const& geometry,
   RunConfig const& cfg,
   amrex::Real time,
   RelativisticInitialDataView const& initial_data
);

void complete_spacetime_connections(
   amrex::MultiFab& state, amrex::Geometry const& geometry, RunConfig const& config
);

void complete_z4c_contracted_connections(
   amrex::MultiFab& state,
   amrex::Geometry const& geometry,
   SphericalExcisionRegion const& excision_region = {},
   int derivative_order = 4
);

struct Z4cFillPatchBoundary {
   int derivative_order = 4;
   AMREX_GPU_DEVICE
   void operator()(
      amrex::IntVect const& index,
      amrex::Array4<amrex::Real> const& values,
      int destination_component,
      int num_components,
      amrex::GeometryData const& geometry,
      amrex::Real,
      amrex::BCRec const* boundary_records,
      int,
      int
   ) const noexcept;
};

} // namespace pyfluid_amrex
