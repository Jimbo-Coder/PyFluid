#pragma once

#include <AMReX_Geometry.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_IntVect.H>
#include <AMReX_MultiFab.H>
#include <AMReX_Vector.H>

#include <array>
#include <string>

#include "pyfluid_amrex/config.hpp"
#include "pyfluid_amrex/physical_system.hpp"

namespace pyfluid_amrex {

struct InitialAtmosphereParameters;

struct PhysicalInitialDataMetadata {
   std::string source_ic;
   std::string spacetime_initial_data;
   std::string rotation_law;
   amrex::Real angular_velocity = 0.0;
   amrex::Real matter_outer_radius = -1.0;
   amrex::GpuArray<amrex::Real, 3> rotation_center{};
   bool exact_layout = false;
   bool used_exact_cache = false;
};

using ConstPhysicalIdField = amrex::Vector<amrex::MultiFab const*>;
using MutablePhysicalIdField = amrex::Vector<amrex::MultiFab*>;
using ConstPhysicalIdFaceFields =
   std::array<ConstPhysicalIdField, AMREX_SPACEDIM>;
using MutablePhysicalIdFaceFields =
   std::array<MutablePhysicalIdField, AMREX_SPACEDIM>;
using PhysicalIdCoordinateFaces = amrex::Vector<
   std::array<amrex::Vector<amrex::Real>, AMREX_SPACEDIM>
>;

void write_physical_initial_data(
   std::string const& path,
   RunConfig const& cfg,
   PhysicalSystem system,
   amrex::Vector<amrex::Geometry> const& geometry,
   amrex::Vector<amrex::IntVect> const& refinement_ratios,
   ConstPhysicalIdField const& primitive,
   ConstPhysicalIdField const& evolution_fluid,
   ConstPhysicalIdField const& evolution_spacetime,
   ConstPhysicalIdFaceFields const& face_magnetic,
   PhysicalIdCoordinateFaces const& coordinate_faces = {}
);

PhysicalInitialDataMetadata read_physical_initial_data(
   std::string const& path,
   RunConfig const& cfg,
   PhysicalSystem system,
   InitialAtmosphereParameters const& atmosphere,
   amrex::Vector<amrex::Geometry> const& geometry,
   amrex::Vector<amrex::IntVect> const& refinement_ratios,
   MutablePhysicalIdField const& primitive,
   MutablePhysicalIdField const& evolution_fluid,
   MutablePhysicalIdField const& evolution_spacetime,
   MutablePhysicalIdFaceFields const& face_magnetic,
   PhysicalIdCoordinateFaces* coordinate_faces = nullptr
);

void materialize_physical_initial_data_fluid(
   RunConfig const& cfg,
   PhysicalSystem system,
   amrex::Vector<amrex::Geometry> const& geometry,
   ConstPhysicalIdField const& primitive,
   ConstPhysicalIdField const& cell_spacetime,
   MutablePhysicalIdField const& evolution_fluid
);

} // namespace pyfluid_amrex
