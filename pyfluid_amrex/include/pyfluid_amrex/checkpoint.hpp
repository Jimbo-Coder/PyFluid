#pragma once

#include <AMReX_BoxArray.H>
#include <AMReX_Geometry.H>
#include <AMReX_IndexType.H>
#include <AMReX_MultiFab.H>
#include <AMReX_Vector.H>

#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "pyfluid_amrex/ct.hpp"
#include "pyfluid_amrex/config.hpp"

namespace pyfluid_amrex {

struct CheckpointFieldInfo {
   std::string name;
   int num_components = 0;
   amrex::IndexType index_type;
};

struct CheckpointHeader {
   int finest_level = 0;
   int state_components = 0;
   amrex::Real reference_mass = std::numeric_limits<amrex::Real>::quiet_NaN();
   amrex::Real reference_energy = std::numeric_limits<amrex::Real>::quiet_NaN();
   amrex::Real reference_max_rho = std::numeric_limits<amrex::Real>::quiet_NaN();
   amrex::Long aligned_regrid_ticks = 0;
   amrex::Long aligned_base_regrid_tick = -1;
   amrex::Vector<int> level_steps;
   amrex::Vector<amrex::Real> dt;
   amrex::Vector<amrex::Real> time;
   amrex::Vector<amrex::BoxArray> box_arrays;
   std::vector<CheckpointFieldInfo> fields;
};

struct CheckpointFieldSet {
   std::string name;
   std::vector<std::unique_ptr<amrex::MultiFab>> const* levels = nullptr;
};

void write_checkpoint(
   std::vector<std::unique_ptr<amrex::MultiFab>> const& state,
   amrex::Vector<int> const& level_steps,
   amrex::Vector<amrex::Real> const& level_times,
   amrex::Vector<amrex::Real> const& level_dts,
   amrex::Vector<amrex::Geometry> const& geometry,
   amrex::Vector<amrex::IntVect> const& refinement_ratios,
   int finest_level,
   std::string const& checkpoint_dir,
   int step,
   RunConfig const& cfg = RunConfig{},
   std::vector<FaceMagneticField> const* face_b = nullptr,
   std::vector<CheckpointFieldSet> const& extra_fields = {},
   amrex::Real reference_mass = std::numeric_limits<amrex::Real>::quiet_NaN(),
   amrex::Real reference_energy = std::numeric_limits<amrex::Real>::quiet_NaN(),
   amrex::Real reference_max_rho = std::numeric_limits<amrex::Real>::quiet_NaN(),
   amrex::Long aligned_regrid_ticks = 0,
   amrex::Long aligned_base_regrid_tick = 0,
   std::string const& checkpoint_compression = "best"
);

CheckpointHeader read_checkpoint_header(std::string const& checkpoint_name);
std::string checkpoint_field_path(std::string const& checkpoint_name, std::string const& field_name);
bool checkpoint_field_exists(std::string const& checkpoint_name, std::string const& field_name);
void read_checkpoint_field(
   amrex::MultiFab& field,
   std::string const& checkpoint_name,
   int level,
   std::string const& field_name
);
void validate_checkpoint_compatibility(
   std::string const& checkpoint_name,
   RunConfig const& cfg,
   int state_components,
   amrex::Geometry const& base_geometry,
   amrex::Vector<amrex::IntVect> const& refinement_ratios,
   int finest_level
);

} // namespace pyfluid_amrex
