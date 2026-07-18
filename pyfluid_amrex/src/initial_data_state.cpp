#include "pyfluid_amrex/initial_data.hpp"

#include "initial_data_state_impl.hpp"

namespace pyfluid_amrex {

void initialize_state(
   amrex::MultiFab& state,
   amrex::Geometry const& geometry,
   RunConfig const& config,
   RelativisticInitialDataView const& initial_data
)
{
   initialize_state_impl(state, geometry, config, initial_data);
}

void initialize_magnetic_blast_noise_face_b(
   FaceMagneticField& face_b,
   amrex::Geometry const& geometry,
   RunConfig const& config
)
{
   initialize_magnetic_blast_noise_face_b_impl(face_b, geometry, config);
}

void initialize_resistive_current_sheet_face_b(
   FaceMagneticField& face_b,
   amrex::Geometry const& geometry,
   RunConfig const& config
)
{
   initialize_resistive_current_sheet_face_b_impl(face_b, geometry, config);
}

void initialize_reconnecting_shear_layer_face_b(
   FaceMagneticField& face_b,
   amrex::Geometry const& geometry,
   RunConfig const& config
)
{
   initialize_reconnecting_shear_layer_face_b_impl(face_b, geometry, config);
}

void initialize_orszag_tang_face_b(
   FaceMagneticField& face_b,
   amrex::Geometry const& geometry,
   RunConfig const& config
)
{
   initialize_orszag_tang_face_b_impl(face_b, geometry, config);
}

void initialize_sod_plane_face_b(
   FaceMagneticField& face_b,
   amrex::Geometry const& geometry,
   RunConfig const& config
)
{
   initialize_sod_plane_face_b_impl(face_b, geometry, config);
}

} // namespace pyfluid_amrex
