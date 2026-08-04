"""Command-line entry point."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from gonet_astrometry import __version__


def build_parser() -> argparse.ArgumentParser:
    """Construct the command-line parser."""

    parser = argparse.ArgumentParser(
        prog="gonet-astrometry",
        description="Astrometric calibration of wide-field GONet images.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line interface.

    Parameters
    ----------
    argv
        Optional command-line arguments excluding the executable name.

    Returns
    -------
    int
        Process exit status.
    """

    parser = build_parser()
    parser.parse_args(argv)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
