from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

import pytest

from gonet_astrometry.adapters.gonet_metadata import (
    GONetMetadataError,
    image_metadata_from_wizard,
)


def test_metadata_adapter_parses_nested_gps_and_utc_timestamp() -> None:
    metadata = {
        "date_utc": "2025-08-06T06:00:46+00:00",
        "ExposureTime": Fraction(1, 2),
        "GPSInfo": {
            "GPSLatitude": (41, 52, (30, 1)),
            "GPSLatitudeRef": "N",
            "GPSLongitude": (87, 37, 30),
            "GPSLongitudeRef": "W",
            "GPSAltitude": (181, 1),
        },
    }

    result = image_metadata_from_wizard(Path("frame.jpg"), metadata)

    assert result.exposure_start == datetime(2025, 8, 6, 6, 0, 46, tzinfo=timezone.utc)
    assert result.exposure_duration_s == 0.5
    assert result.location.latitude_deg == pytest.approx(41.875)
    assert result.location.longitude_deg == pytest.approx(-87.625)
    assert result.location.elevation_m == 181.0
    assert result.source_path == Path("frame.jpg")
    assert result.sensor_orientation == "native BGGR sensor mosaic"


def test_metadata_adapter_accepts_current_wizard_parser_shape() -> None:
    metadata = {
        "unix_time": "1754460046",
        "exposure_time": 0.999981,
        "GPS": {
            "latitude": 41.88,
            "longitude": -87.63,
            "altitude": 181.0,
        },
    }

    result = image_metadata_from_wizard(Path("frame.jpg"), metadata)

    assert result.exposure_start == datetime.fromtimestamp(
        1754460046,
        tz=timezone.utc,
    )
    assert result.exposure_duration_s == pytest.approx(0.999981)
    assert result.location.latitude_deg == 41.88
    assert result.location.longitude_deg == -87.63
    assert result.location.elevation_m == 181.0


def test_metadata_adapter_prefers_filename_unix_to_naive_exif() -> None:
    metadata = {
        "DateTime": "2025:10:29 20:41:14",
        "DateTimeDigitized": "2025:10:29 20:41:14",
        "DateTimeOriginal": "2025:10:29 20:41:14",
        "exposure_time": 29.999993,
        "GPS": {
            "latitude": 54.025555555555556,
            "longitude": -9.822777777777777,
            "altitude": 41.73,
        },
    }

    result = image_metadata_from_wizard(
        Path("256_251029_204008_1761770474.jpg"),
        metadata,
    )

    assert result.exposure_start == datetime.fromtimestamp(
        1761770474,
        tz=timezone.utc,
    )
    assert result.exposure_duration_s == pytest.approx(29.999993)
    assert result.location.latitude_deg == pytest.approx(54.025555555555556)
    assert result.location.longitude_deg == pytest.approx(-9.822777777777777)
    assert result.location.elevation_m == pytest.approx(41.73)


def test_metadata_adapter_uses_filename_unix_timestamp() -> None:
    metadata = {
        "exptime": "0.999981",
        "latitude": -16.5,
        "longitude": -68.15,
        "altitude": 3650,
        "GPSAltitudeRef": 1,
    }

    result = image_metadata_from_wizard(
        Path("225_250806_060044_1754460046.jpg"),
        metadata,
    )

    assert result.exposure_start == datetime.fromtimestamp(
        1754460046,
        tz=timezone.utc,
    )
    assert result.location.elevation_m == -3650.0


def test_metadata_adapter_parses_naive_exif_with_offset() -> None:
    metadata = {
        "DateTimeOriginal": "2025:08:06 01:00:46",
        "OffsetTimeOriginal": "-05:00",
        "ExposureDurationS": 1.0,
        "GPSLatitude": 41.88,
        "GPSLongitude": -87.63,
    }

    result = image_metadata_from_wizard(Path("frame.jpg"), metadata)

    assert result.exposure_start == datetime(2025, 8, 6, 6, 0, 46, tzinfo=timezone.utc)
    assert result.location.elevation_m == 0.0


@pytest.mark.parametrize(
    ("metadata", "message"),
    [
        (
            {"exptime": 1, "latitude": 1, "longitude": 2},
            "Missing exposure timestamp",
        ),
        (
            {
                "date_utc": "2025-08-06T00:00:00+00:00",
                "latitude": 1,
                "longitude": 2,
            },
            "Missing exposure duration",
        ),
        (
            {
                "date_utc": "2025-08-06T00:00:00+00:00",
                "exptime": 1,
                "longitude": 2,
            },
            "Missing observing latitude",
        ),
    ],
)
def test_metadata_adapter_reports_missing_required_values(
    metadata: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(GONetMetadataError, match=message):
        image_metadata_from_wizard(Path("frame.jpg"), metadata)


def test_metadata_adapter_rejects_invalid_values() -> None:
    base = {
        "date_utc": "2025-08-06T00:00:00+00:00",
        "exptime": 1,
        "latitude": 1,
        "longitude": 2,
    }

    with pytest.raises(GONetMetadataError, match="strictly positive"):
        image_metadata_from_wizard(Path("frame.jpg"), {**base, "exptime": 0})

    with pytest.raises(GONetMetadataError, match="degree, minute, and second"):
        image_metadata_from_wizard(
            Path("frame.jpg"),
            {**base, "latitude": (1, 2)},
        )

    with pytest.raises(GONetMetadataError, match="timezone-naive"):
        image_metadata_from_wizard(
            Path("frame.jpg"),
            {**base, "datetimeoriginal": "2025:08:06 01:00:46", "date_utc": None},
        )

    with pytest.raises(GONetMetadataError, match="Invalid optional"):
        image_metadata_from_wizard(
            Path("frame.jpg"),
            {**base, "altitude": "high"},
        )


def test_metadata_adapter_accepts_unix_and_datetime_variants() -> None:
    common = {
        "exptime": 1,
        "latitude": 1,
        "longitude": 2,
    }

    unix_result = image_metadata_from_wizard(
        Path("frame.jpg"),
        {**common, "unix_time": 1_754_460_046},
    )
    assert unix_result.exposure_start == datetime.fromtimestamp(
        1_754_460_046,
        tz=timezone.utc,
    )

    datetime_result = image_metadata_from_wizard(
        Path("frame.jpg"),
        {
            **common,
            "exposure_start": datetime(2025, 8, 6, 6, tzinfo=timezone.utc),
        },
    )
    assert datetime_result.exposure_start.hour == 6

    z_result = image_metadata_from_wizard(
        Path("frame.jpg"),
        {**common, "date_utc": "2025-08-06T06:00:46Z"},
    )
    assert z_result.exposure_start.tzinfo is timezone.utc

    naive_utc_result = image_metadata_from_wizard(
        Path("frame.jpg"),
        {**common, "date_utc": "2025-08-06 06:00:46"},
    )
    assert naive_utc_result.exposure_start.tzinfo is timezone.utc


def test_metadata_adapter_reports_malformed_timestamps_and_offsets() -> None:
    common = {
        "exptime": 1,
        "latitude": 1,
        "longitude": 2,
    }

    with pytest.raises(GONetMetadataError, match="Invalid Unix"):
        image_metadata_from_wizard(
            Path("frame.jpg"),
            {**common, "unix_time": "not-a-time"},
        )

    with pytest.raises(GONetMetadataError, match="Could not parse exposure"):
        image_metadata_from_wizard(
            Path("frame.jpg"),
            {**common, "date_utc": "not-a-date"},
        )

    with pytest.raises(GONetMetadataError, match="Could not parse UTC offset"):
        image_metadata_from_wizard(
            Path("frame.jpg"),
            {
                **common,
                "datetimeoriginal": "2025:08:06 01:00:46",
                "offsettimeoriginal": "bad-offset",
            },
        )


def test_metadata_adapter_reports_empty_metadata() -> None:
    with pytest.raises(GONetMetadataError, match="Missing exposure timestamp"):
        image_metadata_from_wizard(Path("frame.jpg"), None)
