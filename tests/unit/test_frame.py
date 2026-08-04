from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from gonet_astrometry.models.frame import (
    ImageFrame,
    ImageMetadata,
    ObserverLocation,
)


def make_metadata(duration: float = 10.0) -> ImageMetadata:
    return ImageMetadata(
        exposure_start=datetime(2026, 8, 4, 3, 0, tzinfo=timezone.utc),
        exposure_duration_s=duration,
        location=ObserverLocation(41.8781, -87.6298, 181.0),
        source_path=Path("frame.raw"),
    )


def test_exposure_midpoint_uses_duration() -> None:
    metadata = make_metadata(10.0)
    assert metadata.exposure_midpoint == datetime(
        2026,
        8,
        4,
        3,
        0,
        5,
        tzinfo=timezone.utc,
    )


def test_metadata_rejects_naive_timestamp() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        ImageMetadata(
            datetime(2026, 8, 4, 3, 0),
            1.0,
            ObserverLocation(0.0, 0.0),
        )


def test_metadata_rejects_non_positive_duration() -> None:
    with pytest.raises(ValueError, match="strictly positive"):
        make_metadata(0.0)


@pytest.mark.parametrize(
    ("latitude", "longitude", "message"),
    [(91.0, 0.0, "latitude"), (0.0, 181.0, "longitude")],
)
def test_location_rejects_invalid_coordinates(
    latitude: float,
    longitude: float,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        ObserverLocation(latitude, longitude)


def test_frame_accepts_native_two_dimensional_array() -> None:
    frame = ImageFrame(np.zeros((4, 6), dtype=np.uint16), make_metadata())
    assert frame.shape == (4, 6)
    assert frame.mask is None


def test_frame_rejects_non_two_dimensional_array() -> None:
    with pytest.raises(ValueError, match="two-dimensional"):
        ImageFrame(np.zeros((2, 3, 4)), make_metadata())


def test_frame_rejects_mask_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="must match"):
        ImageFrame(
            np.zeros((4, 6)),
            make_metadata(),
            mask=np.zeros((4, 5), dtype=bool),
        )
