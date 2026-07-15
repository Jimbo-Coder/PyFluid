"""Machine-profile loading shared by local validation and HPC helpers."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
MACHINE_ROOT = ROOT / "pyfluid_amrex/machines"
DEFAULT_MACHINE = MACHINE_ROOT / "local.toml"


@dataclass(frozen=True)
class MachineProfile:
   name: str
   path: Path
   mpi_launcher: tuple[str, ...]
   direct_one_rank: bool
   default_ranks: int
   environment: Mapping[str, str]
   build: Mapping[str, object]
   run: Mapping[str, object]
   submit: Mapping[str, object]

   def expand(self, value: str) -> str:
      return value.replace("{root}", str(ROOT)).replace("{machine}", self.name)


def machine_path(selection: str | Path | None = None) -> Path:
   selected = selection or os.environ.get("PYFLUID_MACHINE")
   if selected is None:
      return DEFAULT_MACHINE
   candidate = Path(selected).expanduser()
   if not candidate.is_absolute() and candidate.parent == Path("."):
      named = MACHINE_ROOT / f"{candidate.name.removesuffix('.toml')}.toml"
      if named.exists():
         return named
   return candidate if candidate.is_absolute() else ROOT / candidate


def load_machine(selection: str | Path | MachineProfile | None = None) -> MachineProfile:
   if isinstance(selection, MachineProfile):
      return selection
   path = machine_path(selection).resolve()
   if not path.is_file():
      raise FileNotFoundError(f"machine profile does not exist: {path}")
   with path.open("rb") as stream:
      data = tomllib.load(stream)
   launcher = tuple(str(item) for item in data.get("mpi_launcher", ()))
   if not launcher:
      raise ValueError(f"{path}: mpi_launcher must not be empty")
   return MachineProfile(
      name=str(data.get("name", path.stem)),
      path=path,
      mpi_launcher=launcher,
      direct_one_rank=bool(data.get("direct_one_rank", True)),
      default_ranks=int(data.get("default_ranks", 3)),
      environment={str(key): str(value) for key, value in data.get("environment", {}).items()},
      build=data.get("build", {}),
      run=data.get("run", {}),
      submit=data.get("submit", {}),
   )


def mpi_prefix(ranks: int, machine: str | Path | MachineProfile | None = None) -> list[str]:
   if ranks < 1:
      raise ValueError(f"MPI rank count must be positive, got {ranks}")
   profile = load_machine(machine)
   if ranks == 1 and profile.direct_one_rank:
      return []
   return [profile.expand(token).format(ranks=ranks) for token in profile.mpi_launcher]


def expanded_strings(profile: MachineProfile, values: Sequence[object]) -> list[str]:
   return [profile.expand(str(value)) for value in values]
