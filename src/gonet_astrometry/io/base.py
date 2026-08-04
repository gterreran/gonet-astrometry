"""Input protocol definitions."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from gonet_astrometry.models.frame import ImageFrame
from gonet_astrometry.models.grid import GridCalibration


class ImageLoader(Protocol):
    """Protocol for loading native GONet image frames."""

    def load(self, path: Path) -> ImageFrame:
        """Load one image and its astrometric metadata."""

        ...


class GridCalibrationLoader(Protocol):
    """Protocol for loading Grid calibration output."""

    def load(self, path: Path) -> GridCalibration:
        """Load one Grid calibration from disk."""

        ...
