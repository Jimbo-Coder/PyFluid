#pragma once

#include <AMReX_GpuContainers.H>
#include <AMReX_REAL.H>

#include <memory>
#include <string>
#include <vector>

#include "pyfluid_amrex/eos/types.hpp"

namespace pyfluid_amrex::eos {

using namespace amrex::literals;

class TabulatedEOSStorage {
public:
   TabulatedEOSStorage(
      std::string source_directory,
      std::vector<amrex::Real> log_density,
      std::vector<amrex::Real> log_pressure,
      std::vector<amrex::Real> log_specific_enthalpy,
      amrex::Real low_density_gamma,
      amrex::Real high_density_sound_speed_squared,
      amrex::Real baryon_mass_mev,
      int source_points
   );

   [[nodiscard]] TabulatedEOSView view() const noexcept;
   [[nodiscard]] std::string const& backend() const noexcept;
   [[nodiscard]] std::string const& fingerprint() const noexcept;
   [[nodiscard]] std::string const& source_directory() const noexcept;
   [[nodiscard]] amrex::Real baryon_mass_mev() const noexcept;
   [[nodiscard]] int source_points() const noexcept;
   [[nodiscard]] int retained_points() const noexcept;
   [[nodiscard]] bool was_truncated_for_causality() const noexcept;

private:
   std::string backend_;
   std::string fingerprint_;
   std::string source_directory_;
   amrex::Gpu::ManagedVector<amrex::Real> log_density_;
   amrex::Gpu::ManagedVector<amrex::Real> log_pressure_;
   amrex::Gpu::ManagedVector<amrex::Real> log_specific_enthalpy_;
   amrex::Real low_density_gamma_ = 0.0_rt;
   amrex::Real high_density_sound_speed_squared_ = 0.0_rt;
   amrex::Real baryon_mass_mev_ = 0.0_rt;
   int source_points_ = 0;
};

std::shared_ptr<TabulatedEOSStorage const> load_compose_cold_table(
   std::string const& source_directory
);

} // namespace pyfluid_amrex::eos
