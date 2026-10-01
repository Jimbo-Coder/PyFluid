#pragma once

#include <AMReX_REAL.H>

namespace pyfluid_amrex {

// Temporary pointwise tensor/derivative data, not an evolved grid function.
// Tensor order: 11,12,13,22,23,33. Derivative directions use the canonical
// coordinate order. All three tensor directions survive symmetry reductions.
struct SymmetricTensorJet {
   amrex::Real value[6];
   amrex::Real first[3][6];
   amrex::Real second[3][3][6];
};

// Connection difference is a tensor: upper index first, then two lower
// indices. Unlike raw Christoffels, these are valid unit-basis components.
struct ReferenceMetricConnection {
   amrex::Real delta[3][3][3];
   amrex::Real contracted[3];
};

struct ReferenceBasis {
   amrex::Real to_local[3][3];
};

// Components and reference-covariant derivative in the local unit basis.
// first[derivative][vector component]; every index participates in a frame change.
struct ReferenceVectorGradient {
   amrex::Real value[3];
   amrex::Real first[3][3];
};

struct ConformalMetricRhs {
   amrex::Real chi;
   amrex::Real metric[6];
};

struct ReferenceScalarJet {
   amrex::Real value;
   amrex::Real first[3];
   amrex::Real second[3][3];
};

struct ReferenceVectorJet {
   amrex::Real value[3];
   amrex::Real first[3][3];
   amrex::Real second[3][3][3];
};

} // namespace pyfluid_amrex
