#pragma once

namespace pyfluid_amrex {

enum class TemporalCadenceUnit {
   Steps,
   Disabled,
};

struct TemporalCadence {
   TemporalCadenceUnit unit = TemporalCadenceUnit::Disabled;
   int interval = -1;

   constexpr bool should_record(int step, bool reached_final) const noexcept
   {
      return unit == TemporalCadenceUnit::Steps
         && interval > 0
         && step > 0
         && (step % interval == 0 || reached_final);
   }

   constexpr bool should_record_initial() const noexcept
   {
      return unit == TemporalCadenceUnit::Steps && interval == 0;
   }
};

} // namespace pyfluid_amrex
