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

    def ensure(self) -> None:
        """Create the output directory when necessary."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
