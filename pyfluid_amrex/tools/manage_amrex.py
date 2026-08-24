#!/usr/bin/env python3
"""Check or safely update the repository-local AMReX installation."""

from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
import tomllib
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEPS_DIR = ROOT / ".deps"
MANIFEST = ROOT / "pyfluid_amrex/dependencies/amrex.toml"
ACTIVE_INSTALL = DEPS_DIR / "amrex-mpi-lsolvers-hdf5"
PYFLUID_SOURCE = ROOT / "pyfluid_amrex"
PYFLUID_BUILD = ROOT / "build/pyfluid_amrex"
RELEASE_RE = re.compile(r"^[0-9]{2}\.[0-9]{2}(?:\.[0-9]+)?$")


@dataclass(frozen=True)
class AmrexPin:
    repository: str
    version: str
    commit: str


@dataclass(frozen=True)
class ActivationState:
    kind: str
    link_target: str | None = None
    backup: Path | None = None


def run(command: list[str], *, cwd: Path = ROOT) -> None:
    print(f"+ {shlex.join(command)}", flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def captured(command: list[str], *, cwd: Path = ROOT) -> str:
    result = subprocess.run(
        command,
        cwd=cwd,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout


def load_pin() -> AmrexPin:
    data = tomllib.loads(MANIFEST.read_text())
    if data.get("schema_version") != 1:
        raise ValueError(f"{MANIFEST}: unsupported schema_version")
    pin = AmrexPin(
        repository=str(data.get("repository", "")),
        version=str(data.get("version", "")),
        commit=str(data.get("commit", "")),
    )
    if not pin.repository or not RELEASE_RE.fullmatch(pin.version):
        raise ValueError(f"{MANIFEST}: invalid repository or release version")
    if not re.fullmatch(r"[0-9a-f]{40}", pin.commit):
        raise ValueError(f"{MANIFEST}: commit must be a full Git object ID")
    return pin


def write_pin(pin: AmrexPin) -> None:
    contents = (
        "schema_version = 1\n"
        f'repository = "{pin.repository}"\n'
        f'version = "{pin.version}"\n'
        f'commit = "{pin.commit}"\n'
    )
    temporary = MANIFEST.with_suffix(".toml.next")
    temporary.write_text(contents)
    os.replace(temporary, MANIFEST)


def version_key(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in version.split("."))


def remote_releases(repository: str) -> dict[str, str]:
    output = captured(["git", "ls-remote", "--tags", repository])
    tag_objects: dict[str, str] = {}
    release_commits: dict[str, str] = {}
    for line in output.splitlines():
        object_id, ref = line.split(maxsplit=1)
        prefix = "refs/tags/"
        if not ref.startswith(prefix):
            continue
        tag = ref[len(prefix) :]
        peeled = tag.endswith("^{}")
        if peeled:
            tag = tag[:-3]
        if not RELEASE_RE.fullmatch(tag):
            continue
        if peeled:
            release_commits[tag] = object_id
        else:
            tag_objects[tag] = object_id
    for tag, object_id in tag_objects.items():
        release_commits.setdefault(tag, object_id)
    if not release_commits:
        raise RuntimeError(f"no release tags found at {repository}")
    return release_commits


def installed_version(prefix: Path) -> str | None:
    version_header = prefix / "include/AMReX_Version.H"
    if not version_header.is_file():
        return None
    match = re.search(
        r'^#define AMREX_GIT_VERSION "([^"]+)"$',
        version_header.read_text(),
        flags=re.MULTILINE,
    )
    return match.group(1) if match else None


def install_contract_errors(prefix: Path) -> list[str]:
    config = prefix / "lib/cmake/AMReX/AMReXConfig.cmake"
    if not config.is_file():
        return [f"missing {config}"]
    text = config.read_text()
    expected = {
        "AMReX_MPI_FOUND": "ON",
        "AMReX_HDF5_FOUND": "ON",
        "AMReX_LSOLVERS_FOUND": "ON",
    }
    errors = []
    for name, value in expected.items():
        if not re.search(rf"set\({name}\s+{value}\)", text):
            errors.append(f"{name} is not {value}")
    if not re.search(r"set\(AMReX_SPACEDIM\s+1;2;3\)", text):
        errors.append("AMReX_SPACEDIM is not 1;2;3")
    for dimension in (1, 2, 3):
        library = prefix / f"lib/libamrex_{dimension}d.a"
        if not library.is_file():
            errors.append(f"missing {library}")
    return errors


def managed_source_commit(version: str) -> str | None:
    source = source_path(version)
    if not (source / ".git").is_dir():
        return None
    return captured(["git", "rev-parse", "HEAD"], cwd=source).strip()


def managed_source_dirty(version: str) -> bool | None:
    source = source_path(version)
    if not (source / ".git").is_dir():
        return None
    return bool(captured(["git", "status", "--porcelain"], cwd=source).strip())


def print_status(pin: AmrexPin, releases: dict[str, str] | None) -> bool:
    installed = installed_version(ACTIVE_INSTALL)
    errors = install_contract_errors(ACTIVE_INSTALL)
    source_commit = managed_source_commit(pin.version)
    source_dirty = managed_source_dirty(pin.version)
    print(f"manifest_version={pin.version}")
    print(f"manifest_commit={pin.commit}")
    print(f"source_commit={source_commit or '<unmanaged-or-missing>'}")
    print(f"source_dirty={source_dirty if source_dirty is not None else '<unknown>'}")
    print(f"installed_version={installed or '<missing>'}")
    if ACTIVE_INSTALL.is_symlink():
        print(f"active_install={ACTIVE_INSTALL} -> {os.readlink(ACTIVE_INSTALL)}")
    else:
        print(f"active_install={ACTIVE_INSTALL}")
    latest = None
    if releases is not None:
        latest = max(releases, key=version_key)
        print(f"latest_release={latest}")
        print(f"latest_commit={releases[latest]}")
    for error in errors:
        print(f"contract_error={error}")
    if source_commit != pin.commit:
        print("contract_error=managed source does not match manifest commit")
    expected_install = install_path(pin.version).resolve()
    active_target_ok = (
        ACTIVE_INSTALL.is_symlink() and ACTIVE_INSTALL.resolve() == expected_install
    )
    if not active_target_ok:
        print("contract_error=active install is not the pinned versioned install")
    if source_dirty:
        print("contract_error=managed source has local changes")
    current = (
        installed == pin.version
        and source_commit == pin.commit
        and source_dirty is False
        and active_target_ok
        and not errors
    )
    pinned_latest = latest is None or (
        pin.version == latest and pin.commit == releases[latest]
    )
    status = "current" if current and pinned_latest else "update_available"
    print(f"status={status}")
    return status == "current"


def source_path(version: str) -> Path:
    return DEPS_DIR / f"amrex-{version}"


def build_path(version: str) -> Path:
    return DEPS_DIR / f"build-amrex-{version}-mpi-lsolvers-hdf5"


def install_path(version: str) -> Path:
    return DEPS_DIR / f"amrex-{version}-mpi-lsolvers-hdf5"


def prepare_source(pin: AmrexPin) -> Path:
    source = source_path(pin.version)
    if source.exists():
        if not (source / ".git").is_dir():
            raise RuntimeError(
                f"{source} exists but is not the managed Git checkout; "
                "move it aside or use the already installed release"
            )
    else:
        run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "--branch",
                pin.version,
                pin.repository,
                str(source),
            ]
        )
    checkout_commit = captured(["git", "rev-parse", "HEAD"], cwd=source).strip()
    if checkout_commit != pin.commit:
        raise RuntimeError(
            f"{source}: expected {pin.commit}, found {checkout_commit}"
        )
    if captured(["git", "status", "--porcelain"], cwd=source).strip():
        raise RuntimeError(f"{source}: managed source has local changes")
    return source


def build_amrex(pin: AmrexPin, jobs: int) -> Path:
    source = prepare_source(pin)
    build = build_path(pin.version)
    install = install_path(pin.version)
    hdf5_root = Path("/opt/homebrew/opt/hdf5-mpi")
    compilers = {
        "C": Path("/opt/homebrew/bin/mpicc"),
        "CXX": Path("/opt/homebrew/bin/mpicxx"),
    }
    for name, compiler in compilers.items():
        if not compiler.exists():
            raise RuntimeError(f"missing {name} compiler wrapper: {compiler}")
    if not hdf5_root.exists():
        raise RuntimeError(f"missing Homebrew parallel HDF5: {hdf5_root}")
    run(
        [
            "cmake",
            "-S",
            str(source),
            "-B",
            str(build),
            f"-DCMAKE_INSTALL_PREFIX={install}",
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DCMAKE_C_COMPILER={compilers['C']}",
            f"-DCMAKE_CXX_COMPILER={compilers['CXX']}",
            f"-DCMAKE_PREFIX_PATH={hdf5_root}",
            f"-DHDF5_ROOT={hdf5_root}",
            "-DAMReX_SPACEDIM=1;2;3",
            "-DAMReX_MPI=ON",
            "-DAMReX_OMP=OFF",
            "-DAMReX_FORTRAN=OFF",
            "-DAMReX_PARTICLES=OFF",
            "-DAMReX_LINEAR_SOLVERS=ON",
            "-DAMReX_LINEAR_SOLVERS_INCFLO=ON",
            "-DAMReX_LINEAR_SOLVERS_EM=ON",
            "-DAMReX_HDF5=ON",
        ]
    )
    run(["cmake", "--build", str(build), "-j", str(jobs)])
    run(["cmake", "--install", str(build)])
    errors = install_contract_errors(install)
    if installed_version(install) != pin.version:
        errors.append(
            f"installed version is {installed_version(install)!r}, "
            f"expected {pin.version}"
        )
    if errors:
        raise RuntimeError("invalid candidate AMReX install: " + "; ".join(errors))
    return install


def atomic_symlink(target: Path | str, link: Path) -> None:
    temporary = link.with_name(f".{link.name}.next")
    if temporary.is_symlink():
        temporary.unlink()
    elif temporary.exists():
        raise RuntimeError(f"refusing to replace unexpected {temporary}")
    os.symlink(str(target), temporary)
    os.replace(temporary, link)


def activate_install(candidate: Path) -> ActivationState:
    relative_candidate = Path(os.path.relpath(candidate, ACTIVE_INSTALL.parent))
    if ACTIVE_INSTALL.is_symlink():
        state = ActivationState("symlink", link_target=os.readlink(ACTIVE_INSTALL))
        atomic_symlink(relative_candidate, ACTIVE_INSTALL)
        return state
    if ACTIVE_INSTALL.exists():
        old_version = installed_version(ACTIVE_INSTALL)
        if old_version is None:
            raise RuntimeError(f"cannot identify legacy install {ACTIVE_INSTALL}")
        backup = install_path(old_version)
        if backup.exists():
            raise RuntimeError(f"refusing to overwrite preserved install {backup}")
        ACTIVE_INSTALL.rename(backup)
        try:
            atomic_symlink(relative_candidate, ACTIVE_INSTALL)
        except Exception:
            backup.rename(ACTIVE_INSTALL)
            raise
        return ActivationState("directory", backup=backup)
    atomic_symlink(relative_candidate, ACTIVE_INSTALL)
    return ActivationState("missing")


def restore_install(state: ActivationState) -> None:
    if state.kind == "symlink":
        if state.link_target is None:
            raise RuntimeError("missing previous AMReX link target")
        atomic_symlink(state.link_target, ACTIVE_INSTALL)
        return
    if ACTIVE_INSTALL.is_symlink():
        ACTIVE_INSTALL.unlink()
    if state.kind == "directory":
        if state.backup is None or not state.backup.exists():
            raise RuntimeError("missing preserved AMReX install during rollback")
        state.backup.rename(ACTIVE_INSTALL)


def rebuild_pyfluid(jobs: int) -> None:
    run(["cmake", "--preset", "full-release"], cwd=PYFLUID_SOURCE)
    run(
        [
            "cmake",
            "--build",
            str(PYFLUID_BUILD),
            "-j",
            str(jobs),
            "--target",
            "pyfluid_amrex_production",
        ]
    )


def update(pin: AmrexPin, target: AmrexPin, jobs: int) -> None:
    candidate = build_amrex(target, jobs)
    previous = activate_install(candidate)
    try:
        rebuild_pyfluid(jobs)
    except Exception:
        print("PyFluid rebuild failed; restoring the previous AMReX install.")
        restore_install(previous)
        try:
            rebuild_pyfluid(jobs)
        except Exception as rollback_error:
            raise RuntimeError(
                "AMReX activation and the restored PyFluid rebuild both failed"
            ) from rollback_error
        raise
    write_pin(target)
    print(f"updated AMReX {pin.version} -> {target.version}")
    print("Run validate_amrex_suite.py --level quick before committing the pin.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--update",
        action="store_true",
        help="build, verify, activate, and pin an official stable release",
    )
    parser.add_argument(
        "--version",
        help="release to install; with --update only (defaults to latest)",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="check the manifest and local install without querying GitHub",
    )
    parser.add_argument("--jobs", type=int, default=3)
    args = parser.parse_args()
    if args.version and not args.update:
        parser.error("--version requires --update")
    if args.offline and args.update:
        parser.error("--offline cannot be combined with --update")
    if args.jobs < 1:
        parser.error("--jobs must be positive")

    pin = load_pin()
    releases = None if args.offline else remote_releases(pin.repository)
    if not args.update:
        return 0 if print_status(pin, releases) else 1
    if releases is None:
        raise RuntimeError("release metadata is required for an update")
    version = args.version or max(releases, key=version_key)
    if version not in releases:
        raise RuntimeError(f"official stable AMReX release not found: {version}")
    target = AmrexPin(pin.repository, version, releases[version])
    if target == pin and print_status(pin, releases):
        print("AMReX is already current; no update performed.")
        return 0
    update(pin, target, args.jobs)
    return 0 if print_status(target, releases) else 1


if __name__ == "__main__":
    raise SystemExit(main())
