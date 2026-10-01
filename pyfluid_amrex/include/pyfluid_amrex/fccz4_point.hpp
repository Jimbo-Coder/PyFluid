#pragma once

#include "pyfluid_amrex/metric_jet.hpp"
#include "pyfluid_amrex/relativistic_state.hpp"

namespace pyfluid_amrex {

// Transient point data. Derivatives are reference-covariant in a static local
// unit basis. These structs do not define checkpoint or grid-field identities.
struct Fccz4Point {
   SymmetricTensorJet metric;
   SymmetricTensorJet a_tilde;
   ReferenceScalarJet chi;
   ReferenceScalarJet trace_k;
   ReferenceScalarJet theta;
   ReferenceVectorJet connection;
   ReferenceScalarJet lapse;
   ReferenceVectorJet shift;
};

struct Fccz4Damping {
   amrex::Real kappa1;
   amrex::Real kappa2;
   amrex::Real kappa3;
};

struct Fccz4Geometry {
   amrex::Real inverse[6];
   amrex::Real delta[3][3][3];
   amrex::Real contracted_delta[3];
   amrex::Real physical_ricci[6];
   amrex::Real z_covariant[3];
   amrex::Real z_contravariant[3];
   amrex::Real ricci_z4[6];
   amrex::Real lapse_hessian[6];
   amrex::Real shift_divergence;
   amrex::Real d_shift_divergence[3];
};

struct Fccz4Rhs {
   amrex::Real chi;
   amrex::Real metric[6];
   amrex::Real a_tilde[6];
   amrex::Real trace_k;
   amrex::Real theta;
   amrex::Real connection[3];
};

struct Fccz4Constraints {
   amrex::Real hamiltonian;
   amrex::Real momentum[3];
   amrex::Real z_covariant[3];
   amrex::Real theta;
   amrex::Real log_determinant;
   amrex::Real trace_a;
};

struct Fccz4GaugeRhs {
   amrex::Real lapse;
   amrex::Real shift[3];
   amrex::Real driver[3];
};

} // namespace pyfluid_amrex
