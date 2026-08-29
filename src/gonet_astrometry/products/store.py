"""Filesystem layout for reusable GONet Astrometry workflow products."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ProductStore:
    """Paths for one reusable astrometry workflow output directory.

    Parameters
    ----------
    output_dir
        Directory containing mid-level products and, by default, reports.
    """

    output_dir: Path

    @property
    def detections_path(self) -> Path:
        """Return the portable source-detection product path."""
        return self.output_dir / "detections.npz"

    @property
    def tracks_path(self) -> Path:
        """Return the portable bootstrap-track product path."""
        return self.output_dir / "tracks.npz"

    @property
    def temporal_tracks_path(self) -> Path:
        """Return the Grid-aware spherical temporal-track product path."""
        return self.output_dir / "temporal_tracks.npz"

    @property
    def stellar_tracks_path(self) -> Path:
        """Return the merged physical stellar-track product path."""
        return self.output_dir / "stellar_tracks.npz"

    @property
    def sidereal_path(self) -> Path:
        """Return the portable Grid-calibrated sidereal solution path."""
        return self.output_dir / "sidereal_rotation.npz"

    @property
    def bright_star_catalog_path(self) -> Path:
        """Return the reusable local Bright Star Catalogue cache path."""
        return self.output_dir / "bright_star_catalog.npz"

    @property
    def orientation_path(self) -> Path:
        """Return the portable absolute camera-orientation product path."""
        return self.output_dir / "absolute_orientation.npz"

    def ensure(self) -> None:
        """Create the output directory when necessary."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
