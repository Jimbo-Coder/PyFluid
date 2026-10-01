#pragma once

#include <AMReX_Geometry.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_IntVect.H>
#include <AMReX_MultiFab.H>
#include <AMReX_Vector.H>

#include <array>
#include <string>
#include <vector>

#include "pyfluid_amrex/config.hpp"
#include "pyfluid_amrex/physical_system.hpp"

namespace pyfluid_amrex {

using namespace amrex::literals;

struct InitialAtmosphereParameters;

// Runtime composition metadata is deliberately non-templated.  It describes
// which single-object owners contributed to an accepted artifact; it does not
// imply that their final GR fields may be superposed without a global solve.
struct PhysicalIdComponentMetadata {
   std::string role;
   std::string kind;
   std::string gravity;
   amrex::GpuArray<amrex::Real, 3> origin{{0.0_rt, 0.0_rt, 0.0_rt}};
   amrex::GpuArray<amrex::Real, 3> axis{{0.0_rt, 0.0_rt, 1.0_rt}};
};

struct PhysicalInitialDataMetadata {
   std::vector<PhysicalIdComponentMetadata> components;
   std::string source_ic;
   std::string relativistic_exterior = "atmosphere";
   std::string spacetime_initial_data;
   amrex::Real black_hole_mass = -1.0_rt;
   amrex::GpuArray<amrex::Real, 3> black_hole_spin{{0.0_rt, 0.0_rt, 0.0_rt}};
   amrex::GpuArray<amrex::Real, 3> black_hole_origin{{0.0_rt, 0.0_rt, 0.0_rt}};
   std::string black_hole_interior;
   amrex::Real black_hole_fill_radius = -1.0_rt;
   amrex::Real black_hole_throat_area = -1.0_rt;
   amrex::Real black_hole_irreducible_mass = -1.0_rt;
   amrex::Real black_hole_throat_equatorial_circumference = -1.0_rt;
   amrex::Real black_hole_throat_polar_circumference = -1.0_rt;
   amrex::Real black_hole_throat_kerr_mass = -1.0_rt;
   amrex::Real black_hole_throat_kerr_spin = -1.0_rt;
   amrex::Real black_hole_throat_angular_momentum = 0.0_rt;
   amrex::Real black_hole_throat_christodoulou_mass = -1.0_rt;
   amrex::Real black_hole_throat_christodoulou_spin = 0.0_rt;
   std::string disk;
   std::string disk_gravity;
   std::string disk_rotation_law;
   amrex::Real disk_specific_angular_momentum = 0.0_rt;
   amrex::Real disk_angular_velocity = 0.0_rt;
   amrex::Real disk_differential_rotation_length = 10.0_rt;
   amrex::Real disk_rotation_power_q = 1.0_rt;
   amrex::Real disk_inner_radius = -1.0_rt;
   amrex::GpuArray<amrex::Real, 3> disk_axis{{0.0_rt, 0.0_rt, 1.0_rt}};
   amrex::Real disk_maximum_density = -1.0_rt;
   amrex::Real disk_maximum_velocity = -1.0_rt;
   amrex::Real disk_resolved_inner_radius = -1.0_rt;
   amrex::Real disk_outer_radius = -1.0_rt;
   amrex::Real disk_density_maximum_radius = -1.0_rt;
   amrex::Real disk_maximum_log_specific_enthalpy = -1.0_rt;
   amrex::Real disk_baryonic_mass = -1.0_rt;
   amrex::Real disk_angular_momentum = 0.0_rt;
   std::string rotation_law;
   amrex::Real angular_velocity = 0.0_rt;
   amrex::Real differential_rotation_length = 0.0_rt;
   amrex::Real differential_rotation_length_b = 0.0_rt;
   amrex::Real rotation_power_q = 1.0_rt;
   amrex::Real maximum_density = -1.0_rt;
   amrex::Real maximum_velocity = -1.0_rt;
   amrex::Real injection_energy = -1.0_rt;
   amrex::Real equatorial_radius = -1.0_rt;
   amrex::Real polar_radius = -1.0_rt;
   amrex::Real equatorial_log_enthalpy_gradient = 0.0_rt;
   amrex::Real polar_log_enthalpy_gradient = 0.0_rt;
   amrex::Real mass_shedding_indicator = -1.0_rt;
   amrex::Real adm_mass = -1.0_rt;
   amrex::Real baryonic_mass = -1.0_rt;
   amrex::Real angular_momentum = 0.0_rt;
   amrex::Real dimensionless_spin = 0.0_rt;
   amrex::Real matter_outer_radius = -1.0_rt;
   amrex::GpuArray<amrex::Real, 3> rotation_axis{{0.0_rt, 0.0_rt, 1.0_rt}};
   amrex::GpuArray<amrex::Real, 3> rotation_center{};
   std::string magnetic_seed;
   amrex::Real magnetic_seed_pressure_cutoff = 0.0_rt;
   amrex::Real magnetic_seed_exponent = 0.0_rt;
   amrex::Real magnetic_seed_maximum_comoving_field = 0.0_rt;
   std::string magnetic_equilibrium;
   std::string magnetic_force_balance;
   std::string magnetic_face_representation;
   amrex::Real magnetic_equilibrium_poloidal_magnetization = 0.0_rt;
   amrex::Real magnetic_equilibrium_nonlinear_poloidal = 0.0_rt;
   amrex::Real magnetic_equilibrium_relaxation = 0.0_rt;
   bool used_rigid_transform = false;
   bool exact_layout = false;
   bool used_exact_cache = false;
};

// A saved self-gravitating disk may seed the next member of the same XCTS
// sequence by changing only the nonconstant rotation law's angular-velocity
// scale. Evolution loads and changes to the law/EoS/support contract remain
// exact-identity operations.
bool physical_id_disk_continuation_requested(
   PhysicalInitialDataMetadata const& metadata,
   RunConfig const& cfg
);

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
   PhysicalIdCoordinateFaces const& coordinate_faces,
   PhysicalInitialDataMetadata const& artifact_metadata
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

// Restore the portable primitive core of an already validated exact-layout
// artifact before a lossless solve-only reserialization.
void restore_physical_initial_data_primitive(
   std::string const& path,
   MutablePhysicalIdField const& primitive
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
