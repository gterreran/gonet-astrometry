from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from gonet_astrometry.models.detection import DetectionCatalog
from gonet_astrometry.models.frame import (
    ImageFrame,
    ImageMetadata,
    ObserverLocation,
)
from gonet_astrometry.pipeline import AstrometryPipeline


class FakeLoader:
    def load(self, path: Path) -> ImageFrame:
        metadata = ImageMetadata(
            datetime(2026, 8, 4, 3, 0, tzinfo=timezone.utc),
            1.0,
            ObserverLocation(0.0, 0.0),
            path,
        )
        return ImageFrame(np.zeros((2, 2)), metadata)


class FakeDetector:
    @property
    def name(self) -> str:
        return "fake"

    def detect(self, frame_identifier: str, frame: ImageFrame) -> DetectionCatalog:
        assert frame.shape == (2, 2)
        return DetectionCatalog(frame_identifier, (), self.name)


def test_pipeline_loads_and_detects() -> None:
    pipeline = AstrometryPipeline(FakeLoader(), FakeDetector())
    frames = pipeline.load_frames((Path("one.raw"), Path("two.raw")))
    catalogs = pipeline.detect_sources(frames)
    assert tuple(catalogs) == ("one.raw", "two.raw")
    assert all(catalog.detector_name == "fake" for catalog in catalogs.values())


def test_pipeline_rejects_empty_input() -> None:
    pipeline = AstrometryPipeline(FakeLoader(), FakeDetector())
    with pytest.raises(ValueError, match="At least one"):
        pipeline.load_frames(())


def test_pipeline_rejects_duplicate_paths() -> None:
    pipeline = AstrometryPipeline(FakeLoader(), FakeDetector())
    with pytest.raises(ValueError, match="unique"):
        pipeline.load_frames((Path("one.raw"), Path("one.raw")))
