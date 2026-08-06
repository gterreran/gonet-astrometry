"""Command-line entry point."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

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

    subparsers = parser.add_subparsers(dest="command")
    portal = subparsers.add_parser(
        "portal",
        help="Launch the desktop calibration portal.",
    )
    portal.add_argument(
        "--image",
        "--input",
        dest="image",
        type=Path,
        default=None,
        help=("Optional GONet image or folder used to seed portal input " "discovery."),
    )
    portal.add_argument("--host", default="127.0.0.1")
    portal.add_argument("--port", type=int, default=8050)
    portal.add_argument("--debug", action="store_true")
    portal.add_argument(
        "--server-only",
        action="store_true",
        help="Run Dash without opening the native desktop window.",
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
    arguments = parser.parse_args(argv)

    if arguments.command == "portal":
        from gonet_astrometry.portal.app import run_portal, run_portal_server

        launch = run_portal_server if arguments.server_only else run_portal
        launch(
            initial_path=arguments.image,
            host=arguments.host,
            port=arguments.port,
            debug=arguments.debug,
        )

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
