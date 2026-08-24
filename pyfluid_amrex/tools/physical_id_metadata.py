"""Strict reader for the generated physical-ID artifact manifest contract."""

from __future__ import annotations

import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CATALOG_PATH = (
   ROOT / "pyfluid_amrex" / "generated" / "physical_id_catalog.json"
)


def _scalar_valid(value: str, integer: bool) -> bool:
   try:
      if integer:
         int(value)
      else:
         return math.isfinite(float(value))
   except ValueError:
      return False
   return True


def _value_valid(value: str, value_type: str) -> bool:
   if value_type == "string":
      return True
   if value_type == "integer":
      return _scalar_valid(value, True)
   if value_type == "real":
      return _scalar_valid(value, False)
   if value_type == "boolean":
      return value in ("0", "1")
   if value_type in ("integer_vector", "real_vector"):
      return not value or all(
         _scalar_valid(item, value_type == "integer_vector")
         for item in value.split(",")
      )
   raise ValueError(f"unsupported physical-ID catalog type: {value_type}")


def _dynamic_key_matches(key: str, entry: dict[str, object]) -> bool:
   prefix = str(entry["prefix"])
   suffix = str(entry["suffix"])
   if not key.startswith(prefix) or not key.endswith(suffix):
      return False
   stop = len(key) - len(suffix) if suffix else len(key)
   indices = key[len(prefix):stop]
   parts = indices.split("_")
   return (
      len(parts) == int(entry["index_count"])
      and all(part.isdigit() for part in parts)
   )


def _validate_component_records(
   metadata: dict[str, str], dynamic: tuple[dict[str, object], ...]
) -> None:
   component_entries = tuple(
      entry for entry in dynamic if entry["prefix"] == "component_"
   )
   actual = {
      key for key in metadata
      if any(_dynamic_key_matches(key, entry) for entry in component_entries)
   }
   count_value = metadata.get("component_count")
   if count_value is None:
      if actual:
         raise ValueError("component records require component_count")
      return
   try:
      count = int(count_value)
   except ValueError as error:
      raise ValueError("invalid physical-ID component_count") from error
   if count < 0:
      raise ValueError("invalid physical-ID component_count")
   expected = {
      f"component_{index}{entry['suffix']}"
      for index in range(count)
      for entry in component_entries
   }
   if actual != expected:
      missing = sorted(expected - actual)
      extra = sorted(actual - expected)
      detail = []
      if missing:
         detail.append("missing=" + ",".join(missing))
      if extra:
         detail.append("extra=" + ",".join(extra))
      raise ValueError("invalid physical-ID component records: " + ";".join(detail))


def read_physical_id_metadata(artifact: Path) -> dict[str, str]:
   metadata_path = artifact / "Header.meta"
   metadata: dict[str, str] = {}
   for line in metadata_path.read_text(errors="replace").splitlines():
      if not line.startswith("#") or "=" not in line:
         continue
      key, value = line[1:].split("=", 1)
      metadata[key.strip()] = value.strip()
   catalog = json.loads(CATALOG_PATH.read_text())
   declared = {entry["name"]: entry for entry in catalog["metadata_keys"]}
   dynamic = tuple(catalog["dynamic_keys"])
   unknown = []
   invalid = []
   for key, value in metadata.items():
      entry = declared.get(key)
      if entry is None:
         entry = next(
            (candidate for candidate in dynamic
             if _dynamic_key_matches(key, candidate)),
            None,
         )
      if entry is None:
         unknown.append(key)
      elif not _value_valid(value, entry["value_type"]):
         invalid.append(key)
   unknown.sort()
   invalid.sort()
   if unknown:
      raise ValueError("unknown artifact metadata: " + ",".join(unknown))
   if invalid:
      raise ValueError(
         "artifact metadata has invalid value types: " + ",".join(invalid)
      )
   _validate_component_records(metadata, dynamic)
   if metadata.get("schema_version") == "3":
      missing = sorted(
         key for key, entry in declared.items()
         if entry["required"] and key not in metadata
      )
      if missing:
         raise ValueError("missing artifact metadata: " + ",".join(missing))
   return metadata


def physical_id_metadata_real(
   metadata: dict[str, str], key: str, fallback: float
) -> float:
   try:
      return float(metadata[key])
   except (KeyError, ValueError):
      return fallback


def physical_id_component_labels(metadata: dict[str, str]) -> tuple[str, ...]:
   try:
      count = int(metadata.get("component_count", "0"))
   except ValueError as error:
      raise ValueError("invalid physical-ID component_count") from error
   if count < 0:
      raise ValueError("invalid physical-ID component_count")
   labels = []
   for index in range(count):
      values = tuple(
         metadata.get(f"component_{index}_{field}", "")
         for field in ("role", "kind", "gravity")
      )
      if any(not value for value in values):
         raise ValueError(f"incomplete physical-ID component_{index} record")
      labels.append(":".join(values))
   return tuple(labels)
