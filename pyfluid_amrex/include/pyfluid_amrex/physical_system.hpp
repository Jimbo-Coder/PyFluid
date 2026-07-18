#pragma once

#include <cstdint>

namespace pyfluid_amrex {

enum class PhysicalSystem : std::uint8_t {
   HD = 0,
   MHD = 1,
   RHD = 2,
   RMHD = 3,
};

enum class SpacetimeEvolution : std::uint8_t {
   None = 0,
   Fixed = 1,
   Evolved = 2,
};

enum class EvolutionSector : std::uint8_t {
   None = 0,
   Fluid = 1U << 0U,
   Spacetime = 1U << 1U,
};

struct PhysicalSystemCapabilities {
   bool relativistic = false;
   bool magnetized = false;
   bool has_spacetime_sector = false;
   bool supports_ct = false;
};

constexpr EvolutionSector operator|(EvolutionSector lhs, EvolutionSector rhs) noexcept
{
   return static_cast<EvolutionSector>(
      static_cast<std::uint8_t>(lhs) | static_cast<std::uint8_t>(rhs)
   );
}

constexpr bool has_evolution_sector(EvolutionSector sectors, EvolutionSector sector) noexcept
{
   return (static_cast<std::uint8_t>(sectors) & static_cast<std::uint8_t>(sector)) != 0;
}

constexpr PhysicalSystemCapabilities physical_system_capabilities(PhysicalSystem system) noexcept
{
   switch (system) {
   case PhysicalSystem::HD:
      return {};
   case PhysicalSystem::MHD:
      return {false, true, false, true};
   case PhysicalSystem::RHD:
      return {true, false, true, false};
   case PhysicalSystem::RMHD:
      return {true, true, true, true};
   }
   return {};
}

constexpr bool physical_system_is_relativistic(PhysicalSystem system) noexcept
{
   return physical_system_capabilities(system).relativistic;
}

constexpr bool physical_system_has_magnetic_field(PhysicalSystem system) noexcept
{
   return physical_system_capabilities(system).magnetized;
}

constexpr bool physical_system_has_spacetime_sector(PhysicalSystem system) noexcept
{
   return physical_system_capabilities(system).has_spacetime_sector;
}

constexpr bool physical_system_supports_ct(PhysicalSystem system) noexcept
{
   return physical_system_capabilities(system).supports_ct;
}

constexpr bool valid_spacetime_evolution(
   PhysicalSystem system,
   SpacetimeEvolution spacetime_evolution
) noexcept
{
   if (physical_system_is_relativistic(system)) {
      return spacetime_evolution != SpacetimeEvolution::None;
   }
   return spacetime_evolution == SpacetimeEvolution::None;
}

constexpr bool spacetime_evolves(SpacetimeEvolution spacetime_evolution) noexcept
{
   return spacetime_evolution == SpacetimeEvolution::Evolved;
}

constexpr EvolutionSector evolution_sectors(
   PhysicalSystem system,
   SpacetimeEvolution spacetime_evolution
) noexcept
{
   EvolutionSector sectors = EvolutionSector::Fluid;
   if (spacetime_evolves(spacetime_evolution)
       && physical_system_has_spacetime_sector(system)) {
      sectors = sectors | EvolutionSector::Spacetime;
   }
   return sectors;
}

static_assert(static_cast<std::uint8_t>(PhysicalSystem::HD) == 0);
static_assert(static_cast<std::uint8_t>(PhysicalSystem::MHD) == 1);
static_assert(static_cast<std::uint8_t>(PhysicalSystem::RHD) == 2);
static_assert(static_cast<std::uint8_t>(PhysicalSystem::RMHD) == 3);
static_assert(valid_spacetime_evolution(PhysicalSystem::HD, SpacetimeEvolution::None));
static_assert(!valid_spacetime_evolution(PhysicalSystem::HD, SpacetimeEvolution::Fixed));
static_assert(!valid_spacetime_evolution(PhysicalSystem::RHD, SpacetimeEvolution::None));
static_assert(valid_spacetime_evolution(PhysicalSystem::RHD, SpacetimeEvolution::Fixed));
static_assert(valid_spacetime_evolution(PhysicalSystem::RMHD, SpacetimeEvolution::Evolved));
static_assert(has_evolution_sector(
   evolution_sectors(PhysicalSystem::RMHD, SpacetimeEvolution::Fixed),
   EvolutionSector::Fluid
));
static_assert(!has_evolution_sector(
   evolution_sectors(PhysicalSystem::RMHD, SpacetimeEvolution::Fixed),
   EvolutionSector::Spacetime
));
static_assert(has_evolution_sector(
   evolution_sectors(PhysicalSystem::RMHD, SpacetimeEvolution::Evolved),
   EvolutionSector::Spacetime
));

} // namespace pyfluid_amrex
