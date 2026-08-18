"""Portal compatibility imports for shared GONet input discovery."""

from gonet_astrometry.io.discovery import (
    GONET_SUFFIXES,
    DiscoveryResult,
    discover_gonet_files,
    parse_source_paths,
)

__all__ = [
    "GONET_SUFFIXES",
    "DiscoveryResult",
    "discover_gonet_files",
    "parse_source_paths",
]
