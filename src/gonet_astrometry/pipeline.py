"""High-level astrometric calibration pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from gonet_astrometry.detection.base import SourceDetector
from gonet_astrometry.io.base import ImageLoader
from gonet_astrometry.models.detection import DetectionCatalog
from gonet_astrometry.models.frame import ImageFrame


@dataclass(slots=True)
class AstrometryPipeline:
    """Coordinate image loading and source detection.

    This initial scaffold intentionally stops before matching and optimization.
    Those stages will be added after the image and Grid-calibration contracts
    are validated against the current GONet Wizard implementation.

    Parameters
    ----------
    image_loader
        Loader for native GONet images and metadata.
    source_detector
        Interchangeable source-detection backend.
    """

    image_loader: ImageLoader
    source_detector: SourceDetector

    def load_frames(self, paths: tuple[Path, ...]) -> dict[str, ImageFrame]:
        """Load image frames keyed by their input paths.

        Parameters
        ----------
        paths
            Paths to native GONet images.

        Returns
        -------
        dict
            Mapping from stringified path to loaded image frame.

        Raises
        ------
        ValueError
            If no input paths are supplied or a path is repeated.
        """

        if not paths:
            raise ValueError("At least one image path is required")
        identifiers = tuple(str(path) for path in paths)
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("Image paths must be unique within a session")
        return {
            identifier: self.image_loader.load(path)
            for identifier, path in zip(identifiers, paths, strict=True)
        }

    def detect_sources(
        self,
        frames: dict[str, ImageFrame],
    ) -> dict[str, DetectionCatalog]:
        """Run source detection on loaded frames."""

        return {
            identifier: self.source_detector.detect(identifier, frame)
            for identifier, frame in frames.items()
        }
