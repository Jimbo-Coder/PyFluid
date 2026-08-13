#pragma once

#include <AMReX_Geometry.H>
#include <AMReX_BoxArray.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_IntVect.H>
#include <AMReX_MultiFab.H>
#include <AMReX_Vector.H>

#include <array>
#include <string>
#include <vector>

namespace pyfluid_amrex {

struct MultilevelHdf5FieldLayout {
   int num_components = 0;
   amrex::Vector<amrex::BoxArray> boxes;
   amrex::Vector<amrex::Box> domains;
   amrex::Vector<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>> prob_lo;
   amrex::Vector<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>> prob_hi;
};

// A dimension-neutral view of the same AMReX HDF5 field payload.  Checkpoints
// continue to use the native-dimension reader above; portable physical initial
// data use this representation when the source and target executables have
// different dimensions.
struct RuntimeDimensionHdf5Box {
   std::array<int, 3> lower{};
   std::array<int, 3> upper{};
   std::vector<amrex::Real> data;
};

struct RuntimeDimensionHdf5Level {
   std::array<int, 3> domain_lower{};
   std::array<int, 3> domain_upper{};
   std::array<amrex::Real, 3> prob_lo{};
   std::array<amrex::Real, 3> prob_hi{};
   std::array<std::vector<amrex::Real>, 3> coordinate_faces;
   std::vector<RuntimeDimensionHdf5Box> boxes;
};

struct RuntimeDimensionHdf5Field {
   int dimension = 0;
   int num_components = 0;
   std::vector<RuntimeDimensionHdf5Level> levels;
};

void write_multilevel_hdf5_field(
   amrex::Vector<amrex::MultiFab const*> const& fields,
   amrex::Vector<amrex::Geometry> const& geometry,
   amrex::Vector<int> const& level_steps,
   amrex::Vector<amrex::IntVect> const& refinement_ratios,
   int finest_level,
   amrex::Real time,
   std::string const& root,
   std::string const& field_name,
   std::string const& compression,
   std::string const& format_label
);

bool multilevel_hdf5_field_exists(
   std::string const& root,
   std::string const& field_name
);

void read_multilevel_hdf5_field(
   amrex::MultiFab& field,
   std::string const& root,
   int level,
   std::string const& field_name
);

MultilevelHdf5FieldLayout read_multilevel_hdf5_field_layout(
   std::string const& root,
   std::string const& field_name
);

RuntimeDimensionHdf5Field read_runtime_dimension_hdf5_field(
   std::string const& root,
   std::string const& field_name
);

} // namespace pyfluid_amrex
