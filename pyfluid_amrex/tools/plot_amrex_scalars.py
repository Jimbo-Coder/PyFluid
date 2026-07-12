#!/usr/bin/env python3
"""Plot the shared scalar report for one PyFluid AMReX run."""

from __future__ import annotations

import argparse
from pathlib import Path

from plotting_common import plot_scalar_report


def main() -> None:
   parser = argparse.ArgumentParser()
   parser.add_argument("--input", required=True, type=Path)
   parser.add_argument("--output", required=True, type=Path)
   parser.add_argument("--title", default="AMReX scalar diagnostics")
   args = parser.parse_args()
   plot_scalar_report(args.input, args.output, title=args.title)


if __name__ == "__main__":
   main()
