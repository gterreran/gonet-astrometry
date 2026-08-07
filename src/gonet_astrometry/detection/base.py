"""Source-detector interface."""

from __future__ import annotations

from typing import Protocol

from gonet_astrometry.detection.preprocessing import PreparedDetectionImage
from gonet_astrometry.models.detection import DetectionCatalog
from gonet_astrometry.models.frame import ImageFrame


class SourceDetector(Protocol):
    """Protocol for interchangeable source-detection backends."""

    @property
    def name(self) -> str:
        """Return a stable detector name for provenance records."""
        ...

    def detect(self, frame_identifier: str, frame: ImageFrame) -> DetectionCatalog:
        """Prepare and detect source candidates in one native GONet image."""
        ...

    def detect_prepared(
        self,
        frame_identifier: str,
        prepared: PreparedDetectionImage,
    ) -> DetectionCatalog:
        """Detect candidates in a shared preprocessed image."""
        ...
