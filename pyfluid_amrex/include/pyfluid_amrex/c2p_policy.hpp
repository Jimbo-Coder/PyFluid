#pragma once

#include <AMReX_REAL.H>

#include <array>
#include <cstdint>
#include <limits>

namespace pyfluid_amrex {

enum class C2PMethod : std::uint8_t {
   Noble2D,
   Palenzuela1D,
   Noble1DEntropy,
   Palenzuela1DEntropy,
   Font1D,
   RePrimAnd,
};

enum class C2PStatus : std::uint8_t {
   Success,
   InvalidConservedState,
   InvalidEquationOfState,
   NoBracket,
   NonfiniteResidual,
   DidNotConverge,
   UnphysicalPrimitiveState,
   ConservedResidualMismatch,
};

struct C2PAttemptResult {
   C2PMethod method = C2PMethod::Noble2D;
   C2PStatus status = C2PStatus::DidNotConverge;
   int iterations = 0;
   amrex::Real residual = 0.0;
   amrex::Real conserved_residual = 0.0;

   [[nodiscard]] constexpr bool succeeded() const noexcept
   {
      return status == C2PStatus::Success;
   }
};

struct C2PRecoveryOptions {
   int max_iterations = 80;
   amrex::Real relative_tolerance =
      512.0 * std::numeric_limits<amrex::Real>::epsilon();
   amrex::Real absolute_tolerance =
      512.0 * std::numeric_limits<amrex::Real>::epsilon();
   amrex::Real conserved_tolerance = 5.0e-10;
};

struct RMHDEntropyInput {
   // sqrt(gamma) W P / rho^(Gamma - 1), matching the Valencia entropy
   // conservation law used by GRHayL/AsterX.
   amrex::Real conserved_entropy = 0.0;
};

struct ColdPolytropeEquationOfState {
   amrex::Real polytropic_k = 1.0;
   amrex::Real polytropic_gamma = 2.0;
};

struct C2PMethodCapabilities {
   bool implemented = false;
   bool requires_seed = false;
   bool requires_entropy = false;
   bool requires_cold_equation_of_state = false;
};

[[nodiscard]] constexpr C2PMethodCapabilities c2p_method_capabilities(
   C2PMethod method
) noexcept
{
   switch (method) {
   case C2PMethod::Palenzuela1D:
   case C2PMethod::RePrimAnd:
      return {true, false, false, false};
   case C2PMethod::Noble2D:
      return {true, true, false, false};
   case C2PMethod::Noble1DEntropy:
      return {true, true, true, false};
   case C2PMethod::Palenzuela1DEntropy:
      return {true, false, true, false};
   case C2PMethod::Font1D:
      return {true, false, false, true};
   }
   return {};
}

// The ideal-energy state carries no evolved entropy and has no cold-EOS
// projection. RePrimAnd is both faster and at least as robust on the maintained
// direct candidate corpora; Palenzuela remains an independent final fallback.
inline constexpr std::array<C2PMethod, 2> ideal_rmhd_c2p_policy{
   C2PMethod::RePrimAnd,
   C2PMethod::Palenzuela1D,
};

inline constexpr std::array<C2PMethod, 3> seeded_ideal_rmhd_c2p_policy{
   C2PMethod::Noble2D,
   C2PMethod::RePrimAnd,
   C2PMethod::Palenzuela1D,
};

} // namespace pyfluid_amrex
