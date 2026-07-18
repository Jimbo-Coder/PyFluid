#pragma once

#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>

namespace pyfluid_amrex {

struct FaceMagneticField;
struct RelativisticInitialDataView;
struct RunConfig;

void initialize_state(
   amrex::MultiFab& state,
   amrex::Geometry const& geometry,
   RunConfig const& config,
   RelativisticInitialDataView const& initial_data
);

void initialize_relativistic_primitive_cache_from_provider(
   amrex::MultiFab& primitive,
   amrex::Geometry const& geometry,
   RunConfig const& config,
   RelativisticInitialDataView const& initial_data
);

void initialize_magnetic_blast_noise_face_b(
   FaceMagneticField& face_b,
   amrex::Geometry const& geometry,
   RunConfig const& config
);
void initialize_resistive_current_sheet_face_b(
   FaceMagneticField& face_b,
   amrex::Geometry const& geometry,
   RunConfig const& config
);
void initialize_reconnecting_shear_layer_face_b(
   FaceMagneticField& face_b,
   amrex::Geometry const& geometry,
   RunConfig const& config
);
void initialize_orszag_tang_face_b(
   FaceMagneticField& face_b,
   amrex::Geometry const& geometry,
   RunConfig const& config
);
void initialize_sod_plane_face_b(
   FaceMagneticField& face_b,
   amrex::Geometry const& geometry,
   RunConfig const& config
);

} // namespace pyfluid_amrex
