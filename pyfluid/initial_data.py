from dataclasses import dataclass


@dataclass(frozen=True)
class InitialCondition:
   name: str
   magnetized: bool = False
   perturbed: bool = True


_MAGNETIZED_IDS = {8, 10, 11, 12, 13, 14, 15}

_INITIAL_CONDITION_NAMES = {
   0: "Sod.Shock(L)",
   1: "Sod.Shock(R)",
   2: "Sod.Shock(B)",
   3: "Sod.Shock(T)",
   4: "Square.Pulse",
   5: "Gaussian.Pulse",
   6: "Linear.Wave",
   7: "Kelvin-Helmholtz instability",
   8: "magnetized Kelvin-Helmholtz instability",
   9: "Background",
   10: "Balsara.1A",
   11: "Balsara.1B",
   12: "Balsara.1C",
   13: "Balsara.1D",
   14: "Balsara.1E",
   15: "field-loop advection",
}

INITIAL_CONDITIONS = {
   ici: InitialCondition(name=name, magnetized=ici in _MAGNETIZED_IDS, perturbed=ici != 9)
   for ici, name in _INITIAL_CONDITION_NAMES.items()
}
SUPPORTED_INITIAL_CONDITIONS = tuple(sorted(INITIAL_CONDITIONS))
MAGNETIC_INITIAL_CONDITIONS = tuple(ici for ici in SUPPORTED_INITIAL_CONDITIONS if INITIAL_CONDITIONS[ici].magnetized)
BALSARA_INITIAL_CONDITIONS = (10, 11, 12, 13, 14)
PLOT_BALSARA_INITIAL_CONDITIONS = (9, 10, 11, 12, 13)

# The default perturbation amplitude is tuned for KH-like setups and is too
# large for a linear wave smoke test.
SAFE_INITIAL_CONDITION_OVERRIDES = {
   6: {"delrho_factor": 0.01},
}


def ic_has_magnetic_fields(ici):
   return initial_condition_info(ici).magnetized


def ic_has_perturbation(ici):
   return initial_condition_info(ici).perturbed


def initial_condition_info(ici):
   try:
      return INITIAL_CONDITIONS[int(ici)]
   except KeyError as exc:
      raise ValueError(f"Initial condition not recognized: {ici}") from exc


def initial_condition_name(ici):
   return initial_condition_info(ici).name
