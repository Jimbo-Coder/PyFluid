"""Shared freshness and publication for CMake-owned generated source families.

CMake supplies the complete dependency/output lists. This module deliberately
does not import the symbolic generator or declare a second family catalog.
"""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
from importlib.metadata import version as package_version
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Callable, Iterator, Mapping, Sequence


def generator_environment() -> dict:
    """Describe the running generator without importing symbolic packages."""
    return {
        "python_implementation": sys.implementation.name,
        "python_version": list(sys.version_info),
        "implementation_version": list(sys.implementation.version),
        "sympy": package_version("sympy"),
        "mpmath": package_version("mpmath"),
    }


def _hash_files(paths: Sequence[Path]) -> dict[str, str]:
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


@contextmanager
def generation_lock(path: Path, timeout: float) -> Iterator[None]:
    """Lock one family across build profiles; process death releases the lock."""
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("codegen lock timeout must be finite and positive")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        if os.name == "nt":
            import msvcrt

            stream.seek(0, os.SEEK_END)
            if stream.tell() == 0:
                stream.write(b"\0")
                stream.flush()

            def acquire():
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)

            def release():
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            def acquire():
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)

            def release():
                fcntl.flock(stream, fcntl.LOCK_UN)

        deadline = time.monotonic() + timeout
        while True:
            try:
                acquire()
                break
            except (BlockingIOError, PermissionError):
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"timed out waiting for codegen family lock: {path}")
                time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
        try:
            yield
        finally:
            release()


def _read_manifest(path: Path) -> dict | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _publish_manifest(path: Path, manifest: dict) -> None:
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=path.name + ".", delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(manifest, stream, sort_keys=True, indent=2)
        stream.write("\n")
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def generate_if_needed(
    manifest_path: Path,
    dependencies: Sequence[Path],
    outputs: Sequence[Path],
    options: Mapping,
    generate: Callable[[dict[Path, Path]], None],
    *,
    lock_timeout: float = 300.0,
) -> bool:
    """Generate once under a shared lock; preserve unchanged output timestamps.

The callback receives original-to-staged output paths. Only complete verified
outputs are published, atomically per file, followed by the atomic manifest.
"""
    manifest_path = manifest_path.resolve()
    dependencies = sorted({path.resolve() for path in dependencies})
    outputs = sorted({path.resolve() for path in outputs})
    if not dependencies or not outputs:
        raise ValueError("cached generation requires explicit dependencies and outputs")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with generation_lock(manifest_path.with_suffix(".lock"), lock_timeout):
        # Recheck only after locking: another profile may have just generated it.
        inputs = _hash_files(dependencies)
        environment = generator_environment()
        identity = {"version": 2, "inputs": inputs, "options": dict(options),
                    "environment": environment}
        previous = _read_manifest(manifest_path)
        if previous is not None and all(previous.get(key) == value for key, value in identity.items()):
            try:
                current_outputs = _hash_files(outputs)
            except FileNotFoundError:
                current_outputs = None
            if previous.get("outputs") == current_outputs:
                return False

        common_root = Path(os.path.commonpath([str(path.parent) for path in outputs]))
        with tempfile.TemporaryDirectory(prefix="generate-", dir=manifest_path.parent) as temporary:
            staging = Path(temporary)
            replacements = {path: staging / path.relative_to(common_root) for path in outputs}
            generate(replacements)
            generated = {path for path in staging.rglob("*") if path.is_file()}
            expected = set(replacements.values())
            if generated != expected:
                missing = sorted(str(path.relative_to(staging)) for path in expected - generated)
                unexpected = sorted(str(path.relative_to(staging)) for path in generated - expected)
                raise RuntimeError(f"codegen output declaration mismatch: missing={missing}; unexpected={unexpected}")
            if _hash_files(dependencies) != inputs:
                raise RuntimeError("codegen dependencies changed during generation; retry the build")
            if generator_environment() != environment:
                raise RuntimeError("codegen environment changed during generation; retry the build")
            # Every file is complete before publication. A failed generation
            # leaves the prior outputs and provenance intact.
            for destination, source in replacements.items():
                if destination.is_file() and destination.read_bytes() == source.read_bytes():
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                os.replace(source, destination)
            _publish_manifest(manifest_path, {**identity, "outputs": _hash_files(outputs)})
        return True
