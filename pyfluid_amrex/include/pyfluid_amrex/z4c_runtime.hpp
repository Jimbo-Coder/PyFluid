#pragma once

#include <AMReX_BCRec.H>
#include <AMReX_Geometry.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_MultiFab.H>
#include <AMReX_Vector.H>

#include "pyfluid_amrex/config.hpp"
#include "pyfluid_amrex/initial_data/tov.hpp"
#include "pyfluid_amrex/z4c_state.hpp"

namespace pyfluid_amrex {

enum class Z4cInitialData : int {
   Minkowski,
   GaugeWave,
   LinearWave,
   RobustStability,
};

struct Z4cInitialDataParameters {
   Z4cInitialData kind = Z4cInitialData::Minkowski;
   amrex::Real amplitude = 0.0;
   amrex::Real wave_number = 1.0;
};

Z4cInitialDataParameters z4c_initial_data_parameters(RunConfig const& cfg);
Z4cEvolutionParameters z4c_evolution_parameters(RunConfig const& cfg);
Z4cLapseGauge z4c_lapse_gauge(RunConfig const& cfg);
Z4cShiftGauge z4c_shift_gauge(RunConfig const& cfg);
amrex::Vector<amrex::BCRec> z4c_boundary_records(
   RunConfig const& cfg,
   amrex::Geometry const& geometry
);

void initialize_z4c_state(
   amrex::MultiFab& state,
   amrex::Geometry const& geometry,
   RunConfig const& cfg,
   amrex::Real time,
   TovInitialDataView const& tov_initial_data = {}
);

void complete_z4c_contracted_connections(
   amrex::MultiFab& state,
   amrex::Geometry const& geometry
);

struct Z4cFillPatchBoundary {
   AMREX_GPU_DEVICE
   void operator()(
      amrex::IntVect const&,
      amrex::Array4<amrex::Real> const&,
      int,
      int,
      amrex::GeometryData const&,
      amrex::Real,
      amrex::BCRec const*,
      int,
      int
   ) const noexcept
   {}
};

} // namespace pyfluid_amrex
