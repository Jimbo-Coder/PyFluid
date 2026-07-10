#!/usr/bin/env python3
"""Render one AMReX plotfile with VisIt CLI.

Run with:
   /Applications/VisIt.app/Contents/MacOS/VisIt -cli -nowin -s pyfluid_amrex/tools/visit_render_amrex.py <plotfile> <output.png> [variable]
"""

from __future__ import annotations

import os
from pathlib import Path
import sys


def database_header(path: str) -> tuple[str, int, list[str]]:
    plotfile = Path(path)
    header = plotfile / "Header" if plotfile.is_dir() else plotfile
    with header.open() as f:
        f.readline()
        ncomp = int(f.readline())
        names = [f.readline().strip() for _ in range(ncomp)]
        ndim = int(f.readline())
    return str(header), ndim, names


def visit_name(name: str) -> str:
    return f"<{name}>"


def define_derived_variable(variable: str, names: list[str]) -> None:
    if variable in names:
        return
    if variable == "pressure" and {"rho", "mom1", "mom2", "mom3", "energy"}.issubset(names):
        kinetic = (
            f"0.5*({visit_name('mom1')}*{visit_name('mom1')}"
            f"+{visit_name('mom2')}*{visit_name('mom2')}"
            f"+{visit_name('mom3')}*{visit_name('mom3')})/{visit_name('rho')}"
        )
        magnetic = "0.0"
        if {"b1", "b2", "b3"}.issubset(names):
            magnetic = (
                f"0.5*({visit_name('b1')}*{visit_name('b1')}"
                f"+{visit_name('b2')}*{visit_name('b2')}"
                f"+{visit_name('b3')}*{visit_name('b3')})"
            )
        DefineScalarExpression("pressure", f"({visit_name('energy')}-{kinetic}-{magnetic})*(5.0/3.0-1.0)")
        return
    if variable.endswith("_mag"):
        prefix = variable.removesuffix("_mag")
        components = [f"{prefix}1", f"{prefix}2", f"{prefix}3"]
        if set(components).issubset(names):
            terms = "+".join(f"{visit_name(name)}*{visit_name(name)}" for name in components)
            DefineScalarExpression(variable, f"sqrt({terms})")
            return
    raise RuntimeError(f"{variable} not found; available variables: {', '.join(names)}")


def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit("usage: visit_render_amrex.py <plotfile> <output.png> [variable]")

    database, ndim, names = database_header(sys.argv[1])
    output = sys.argv[2]
    variable = sys.argv[3] if len(sys.argv) > 3 else "rho"

    OpenDatabase(database, 0, f"Boxlib{ndim}D")
    define_derived_variable(variable, names)
    try:
        annotations = AnnotationAttributes()
        annotations.userInfoFlag = 0
        annotations.databaseInfoFlag = 0
        SetAnnotationAttributes(annotations)
    except Exception:
        pass
    AddPlot("Pseudocolor", variable)
    DrawPlots()

    attrs = SaveWindowAttributes()
    attrs.outputToCurrentDirectory = 0
    attrs.outputDirectory = os.path.dirname(os.path.abspath(output))
    attrs.fileName = os.path.splitext(os.path.basename(output))[0]
    attrs.family = 0
    attrs.format = attrs.PNG
    attrs.width = 1400
    attrs.height = 1000
    SetSaveWindowAttributes(attrs)
    SaveWindow()


main()
