#!/usr/bin/env python3
"""Check or build the pinned, isolated experimental AdaptiveCpp/Metal route.

No dependency checkout is reset or cleaned. Existing source must match the
declared pristine commit or the exact reviewed patch result before use.
Homebrew packages are prerequisites; this tool does not install system tools.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import platform
import re
import subprocess
import tempfile
import tomllib
import urllib.request
import zipfile

from manage_amrex import (
    DEPS_DIR, ROOT, atomic_symlink, captured, install_contract_errors, load_pin, run,
)


MANIFEST = ROOT / "pyfluid_amrex/dependencies/metal.toml"
TOOLCHAIN_SOURCE = DEPS_DIR / "adaptivecpp-metal-source"
TOOLCHAIN_BUILD = DEPS_DIR / "build-adaptivecpp-metal"
TOOLCHAIN_INSTALL = DEPS_DIR / "adaptivecpp-metal"
REFERENCE_SOURCE = DEPS_DIR / "warpx-metal-reference"


def source_diff(path: Path) -> bytes:
    diff = subprocess.check_output(["git", "diff", "--binary", "HEAD"], cwd=path)
    # git apply leaves newly added files untracked. Include their names and
    # contents in the identity without changing the checkout's Git index.
    added = subprocess.check_output(
        ["git", "ls-files", "--others", "--exclude-standard", "-z"], cwd=path,
    )
    for name in sorted(filter(None, added.split(b"\0"))):
        diff += b"\0untracked\0" + name + b"\0" + (path / name.decode()).read_bytes()
    return diff


def prepare_source(
    path: Path, repository: str, commit: str, patches: list[Path], expected: str,
    *, build: bool,
) -> None:
    if not path.exists():
        if not build:
            raise RuntimeError(f"missing source: {path}")
        run(["git", "clone", "--no-checkout", repository, str(path)])
        run(["git", "checkout", "--detach", commit], cwd=path)
    if captured(["git", "rev-parse", "HEAD"], cwd=path).strip() != commit:
        raise RuntimeError(f"source pin mismatch: {path}; checkout left untouched")
    diff = source_diff(path)
    if not diff and patches and build:
        # Later patches depend on earlier ones. Compose them in a temporary
        # index, then apply one checked diff without changing the real index.
        with tempfile.TemporaryDirectory(prefix="pyfluid-metal-patches-") as temporary:
            environment = {**os.environ, "GIT_INDEX_FILE": str(Path(temporary) / "index")}
            subprocess.run(["git", "read-tree", "HEAD"], cwd=path,
                           env=environment, check=True)
            for patch in patches:
                subprocess.run(["git", "apply", "--cached", str(patch)], cwd=path,
                               env=environment, check=True)
            combined = subprocess.check_output(
                ["git", "diff", "--cached", "--binary", "HEAD"], cwd=path,
                env=environment,
            )
            subprocess.run(["git", "apply", "--check"], input=combined, cwd=path, check=True)
            subprocess.run(["git", "apply"], input=combined, cwd=path, check=True)
        diff = source_diff(path)
    if hashlib.sha256(diff).hexdigest() != expected:
        raise RuntimeError(f"source differs from reviewed patch result: {path}")


def prepare_headers(config: dict, *, build: bool) -> Path:
    archive = DEPS_DIR / Path(config["url"]).name
    if not archive.exists() and build:
        with urllib.request.urlopen(config["url"]) as response:
            archive.write_bytes(response.read())
    if not archive.exists() or hashlib.sha256(archive.read_bytes()).hexdigest() != config["sha256"]:
        raise RuntimeError(f"missing or mismatched Apple header archive: {archive}")
    with zipfile.ZipFile(archive) as source:
        for entry in source.infolist():
            if entry.is_dir() or entry.filename.startswith("__MACOSX/"):
                continue
            relative = Path(entry.filename)
            if relative.is_absolute() or ".." in relative.parts or relative.parts[0] != "metal-cpp":
                raise RuntimeError(f"unexpected archive path: {entry.filename}")
            destination = DEPS_DIR / relative
            contents = source.read(entry)
            if destination.exists():
                if destination.read_bytes() != contents:
                    raise RuntimeError(f"modified Apple header: {destination}")
            elif build:
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(contents)
            else:
                raise RuntimeError(f"missing Apple header: {destination}")
    return DEPS_DIR / "metal-cpp"


def brew_prefix(package: str) -> Path:
    prefix = Path(captured(["brew", "--prefix", package]).strip())
    if not prefix.is_dir():
        raise RuntimeError(f"install prerequisite Homebrew package {package}")
    return prefix


def configure_toolchain(config: dict, headers: Path, jobs: int) -> None:
    llvm = brew_prefix("llvm@20")
    linker = brew_prefix("lld@20")
    omp = brew_prefix("libomp")
    brew_prefix("boost")
    run([
        "cmake", "-S", str(TOOLCHAIN_SOURCE), "-B", str(TOOLCHAIN_BUILD), "-G", "Ninja",
        f"-DCMAKE_INSTALL_PREFIX={TOOLCHAIN_INSTALL}",
        f"-DCMAKE_C_COMPILER={llvm}/bin/clang", f"-DCMAKE_CXX_COMPILER={llvm}/bin/clang++",
        "-DCMAKE_OSX_ARCHITECTURES=arm64", f"-DCMAKE_OSX_SYSROOT={config['sdk']}",
        f"-DLLVM_DIR={llvm}/lib/cmake/llvm", "-DACPP_COMPILER_FEATURE_PROFILE=full",
        f"-DACPP_LLD_PATH={linker}/bin/ld64.lld",
        f"-DOpenMP_C_FLAGS=-Xclang -fopenmp -I{omp}/include",
        f"-DOpenMP_CXX_FLAGS=-Xclang -fopenmp -I{omp}/include",
        "-DOpenMP_C_LIB_NAMES=omp", "-DOpenMP_CXX_LIB_NAMES=omp",
        f"-DOpenMP_omp_LIBRARY={omp}/lib/libomp.dylib", f"-DMETAL_INCLUDE_DIR={headers}",
        "-DWITH_METAL_BACKEND=ON", "-DWITH_CPU_BACKEND=ON",
        "-DWITH_CUDA_BACKEND=OFF", "-DWITH_ROCM_BACKEND=OFF",
        "-DWITH_OPENCL_BACKEND=OFF", "-DWITH_LEVEL_ZERO_BACKEND=OFF",
        "-DCMAKE_BUILD_TYPE=Release",
    ])
    run(["cmake", "--build", str(TOOLCHAIN_BUILD), "--parallel", str(jobs)])
    run(["cmake", "--install", str(TOOLCHAIN_BUILD)])


def configure_amrex(config: dict, source: Path, build_dir: Path, install: Path, jobs: int) -> None:
    hdf5 = brew_prefix("hdf5-mpi")
    run([
        "cmake", "-S", str(source), "-B", str(build_dir), "-G", "Ninja",
        f"-DCMAKE_CXX_COMPILER={TOOLCHAIN_INSTALL}/bin/acpp",
        f"-DCMAKE_INSTALL_PREFIX={install}", "-DCMAKE_BUILD_TYPE=Release",
        "-DCMAKE_OSX_ARCHITECTURES=arm64", f"-DCMAKE_OSX_SYSROOT={config['sdk']}",
        "-DAMReX_GPU_BACKEND=SYCL", "-DAMReX_PRECISION=SINGLE", "-DAMReX_SPACEDIM=1;2;3",
        "-DAMReX_MPI=ON", "-DAMReX_OMP=OFF", "-DAMReX_FORTRAN=OFF",
        "-DAMReX_PARTICLES=OFF", "-DAMReX_LSOLVERS=ON", "-DAMReX_HDF5=ON",
        f"-DHDF5_ROOT={hdf5}", f"-DCMAKE_PREFIX_PATH={hdf5}",
        "-DAMReX_SYCL_SUB_GROUP_SIZE=32", "-DAMReX_SYCL_AOT=OFF",
        "-DAMReX_SYCL_SPLIT_KERNEL=OFF", "-DAMReX_SYCL_ONEDPL=OFF",
    ])
    run(["cmake", "--build", str(build_dir), "--parallel", str(jobs)])
    run(["cmake", "--install", str(build_dir)])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", action="store_true", help="prepare sources and build; default is read-only")
    parser.add_argument("--stage", choices=("all", "toolchain", "amrex"), default="all")
    parser.add_argument("--jobs", type=int, default=2)
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        parser.error("this experimental profile requires Apple Silicon macOS")
    config = tomllib.loads(MANIFEST.read_text())
    pin = load_pin()
    if config.get("schema_version") != 1 or config["amrex_patch_base"] != pin.commit:
        parser.error("Metal patch base must match the maintained AMReX pin")
    if not Path(config["sdk"]).is_dir():
        parser.error(f"compatible SDK missing: {config['sdk']}")
    source = DEPS_DIR / f"amrex-{pin.version}-metal-source"
    build_dir = DEPS_DIR / f"build-amrex-{pin.version}-metal"
    install = DEPS_DIR / f"amrex-{pin.version}-metal"
    if args.build:
        DEPS_DIR.mkdir(exist_ok=True)
    reference = config["warpx_metal"]
    prepare_source(REFERENCE_SOURCE, reference["repository"], reference["commit"], [],
                   hashlib.sha256(b"").hexdigest(), build=args.build)
    toolchain = config["adaptivecpp"]
    prepare_source(
        TOOLCHAIN_SOURCE, toolchain["repository"], toolchain["commit"],
        [REFERENCE_SOURCE / "patches/adaptivecpp" / name for name in reference["patches"]]
        + [MANIFEST.parent / name for name in toolchain.get("patches", [])],
        toolchain["diff_sha256"], build=args.build,
    )
    prepare_source(source, pin.repository, pin.commit,
                   [MANIFEST.parent / config["amrex_patch"]], config["amrex_diff_sha256"],
                   build=args.build)
    headers = prepare_headers(config["metal_cpp"], build=args.build)
    if args.build and args.stage in ("all", "toolchain"):
        configure_toolchain(config, headers, args.jobs)
    if args.build and args.stage in ("all", "amrex"):
        configure_amrex(config, source, build_dir, install, args.jobs)
    if args.stage != "toolchain":
        errors = install_contract_errors(install, "mpi-fp32")
        package = install / "lib/cmake/AMReX/AMReXConfig.cmake"
        if package.is_file() and not re.search(r"set\(AMReX_SYCL_FOUND\s+ON\)", package.read_text()):
            errors.append("AMReX SYCL backend is not enabled")
        if errors:
            raise RuntimeError("; ".join(errors))
        if args.build:
            active = DEPS_DIR / "amrex-metal"
            if active.exists() and not active.is_symlink():
                raise RuntimeError(f"refusing to replace non-link installation: {active}")
            atomic_symlink(install.relative_to(DEPS_DIR), active)
    print("PASS pinned Metal dependency contract; GPU execution requires a separate run")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
