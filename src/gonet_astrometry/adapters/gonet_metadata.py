"""Translate GONet Wizard metadata into astrometry domain models."""

from __future__ import annotations

import re
from collections.abc import Mapping
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

from gonet_astrometry.models.frame import ImageMetadata, ObserverLocation

_UNIX_TOKEN = re.compile(r"(?:^|_)(\d{10})(?=_|\.|$)")


class GONetMetadataError(ValueError):
    """Raised when required astrometric metadata cannot be interpreted."""


def image_metadata_from_wizard(
    path: Path,
    metadata: Mapping[object, object] | None,
) -> ImageMetadata:
    """Build validated astrometric metadata from Wizard metadata.

    Parameters
    ----------
    path
        Source GONet image path. A camera-generated Unix timestamp embedded in
        the filename is preferred to timezone-naive EXIF timestamps.
    metadata
        Mapping returned by ``GONetFileRaw.meta``.

    Returns
    -------
    ImageMetadata
        Normalized exposure and observing-location metadata.

    Raises
    ------
    GONetMetadataError
        If exposure duration, timestamp, latitude, or longitude is absent or
        cannot be interpreted.

    Notes
    -----
    File creation and modification times are never used. After an explicit Unix
    value in Wizard metadata, a ten-digit Unix token embedded in the GONet
    filename is preferred to EXIF datetime fields because it is unambiguous UTC.
    """
    values = _normalized_metadata(metadata)
    exposure_start = _exposure_start(path, values)
    exposure_duration_s = _required_float(
        values,
        aliases=(
            "exposuredurations",
            "exposureduration",
            "exposuretime",
            "exptime",
        ),
        label="exposure duration",
    )
    if exposure_duration_s <= 0:
        raise GONetMetadataError("Exposure duration must be strictly positive.")

    latitude = _coordinate(
        values,
        aliases=("latitudedeg", "gpslatitude", "latitude", "gpslat"),
        reference_aliases=("gpslatituderef", "latituderef"),
        label="latitude",
    )
    longitude = _coordinate(
        values,
        aliases=("longitudedeg", "gpslongitude", "longitude", "gpslon"),
        reference_aliases=("gpslongituderef", "longituderef"),
        label="longitude",
    )
    elevation = _optional_float(
        values,
        aliases=("elevationm", "gpsaltitude", "elevation", "altitude"),
        default=0.0,
    )
    altitude_ref = _first(values, ("gpsaltituderef", "altituderef"))
    if altitude_ref is not None and str(altitude_ref).strip() in {"1", "below"}:
        elevation = -abs(elevation)

    return ImageMetadata(
        exposure_start=exposure_start,
        exposure_duration_s=exposure_duration_s,
        location=ObserverLocation(
            latitude_deg=latitude,
            longitude_deg=longitude,
            elevation_m=elevation,
        ),
        source_path=Path(path),
        sensor_orientation="native BGGR sensor mosaic",
    )


def _normalized_metadata(
    metadata: Mapping[object, object] | None,
) -> dict[str, object]:
    """Return flattened metadata indexed by normalized key names."""
    if metadata is None:
        return {}

    normalized: dict[str, object] = {}

    def visit(mapping: Mapping[object, object], prefix: str = "") -> None:
        for raw_key, value in mapping.items():
            key = _normalize_key(raw_key)
            combined = f"{prefix}{key}" if prefix else key
            if isinstance(value, Mapping):
                visit(value, combined)
                continue
            normalized.setdefault(key, value)
            normalized.setdefault(combined, value)

    visit(metadata)
    return normalized


def _normalize_key(value: object) -> str:
    """Normalize a metadata key for tolerant alias matching."""
    return "".join(
        character for character in str(value).casefold() if character.isalnum()
    )


def _exposure_start(path: Path, values: Mapping[str, object]) -> datetime:
    """Return the exposure start time in UTC."""
    unix_value = _first(values, ("unixtime", "unixtimestamp", "timestamp"))
    if unix_value is not None:
        try:
            return datetime.fromtimestamp(_as_float(unix_value), tz=timezone.utc)
        except (OverflowError, OSError, TypeError, ValueError) as exc:
            raise GONetMetadataError("Invalid Unix exposure timestamp.") from exc

    matches = _UNIX_TOKEN.findall(Path(path).name)
    if matches:
        return datetime.fromtimestamp(float(matches[-1]), tz=timezone.utc)

    timestamp_aliases = (
        "exposurestartutc",
        "datetimeutc",
        "dateutc",
        "utcdatetime",
        "exposurestart",
        "datetimeoriginal",
        "datetime",
    )
    for alias in timestamp_aliases:
        value = values.get(alias)
        if value is None:
            continue
        return _parse_datetime(value, values, explicit_utc="utc" in alias)

    raise GONetMetadataError(
        "Missing exposure timestamp in Wizard metadata and GONet filename."
    )


def _parse_datetime(
    value: object,
    values: Mapping[str, object],
    *,
    explicit_utc: bool,
) -> datetime:
    """Parse one datetime value and normalize it to UTC."""
    if isinstance(value, datetime):
        parsed = value
    else:
        text = str(value).strip()
        if text.endswith("Z"):
            text = f"{text[:-1]}+00:00"
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            try:
                parsed = datetime.strptime(text, "%Y:%m:%d %H:%M:%S")
            except ValueError as exc:
                raise GONetMetadataError(
                    f"Could not parse exposure timestamp {value!r}."
                ) from exc

    if parsed.tzinfo is None:
        offset_value = _first(
            values,
            ("offsettimeoriginal", "utcoffset", "timezoneoffset"),
        )
        if offset_value is not None:
            offset = str(offset_value).strip()
            try:
                parsed = datetime.fromisoformat(f"{parsed.isoformat()}{offset}")
            except ValueError as exc:
                raise GONetMetadataError(
                    f"Could not parse UTC offset {offset_value!r}."
                ) from exc
        elif explicit_utc:
            parsed = parsed.replace(tzinfo=timezone.utc)
        else:
            raise GONetMetadataError(
                "Exposure timestamp is timezone-naive and no UTC offset is available."
            )

    return parsed.astimezone(timezone.utc)


def _coordinate(
    values: Mapping[str, object],
    *,
    aliases: tuple[str, ...],
    reference_aliases: tuple[str, ...],
    label: str,
) -> float:
    """Return one signed decimal-degree coordinate."""
    raw_value = _first(values, aliases)
    if raw_value is None:
        raise GONetMetadataError(f"Missing observing {label} in Wizard metadata.")

    reference = _first(values, reference_aliases)
    decimal = _degrees(raw_value)
    if reference is not None and str(reference).strip().upper() in {"S", "W"}:
        decimal = -abs(decimal)
    return decimal


def _degrees(value: object) -> float:
    """Convert decimal or degree-minute-second metadata to degrees."""
    if isinstance(value, (tuple, list)):
        if len(value) != 3:
            raise GONetMetadataError(
                "GPS coordinates must contain degree, minute, and second values."
            )
        degrees, minutes, seconds = (_as_float(component) for component in value)
        sign = -1.0 if degrees < 0 else 1.0
        return sign * (abs(degrees) + minutes / 60.0 + seconds / 3600.0)
    return _as_float(value)


def _required_float(
    values: Mapping[str, object],
    *,
    aliases: tuple[str, ...],
    label: str,
) -> float:
    """Return one required floating-point metadata value."""
    value = _first(values, aliases)
    if value is None:
        raise GONetMetadataError(f"Missing {label} in Wizard metadata.")
    try:
        return _as_float(value)
    except (TypeError, ValueError, ZeroDivisionError) as exc:
        raise GONetMetadataError(f"Invalid {label}: {value!r}.") from exc


def _optional_float(
    values: Mapping[str, object],
    *,
    aliases: tuple[str, ...],
    default: float,
) -> float:
    """Return an optional floating-point metadata value."""
    value = _first(values, aliases)
    if value is None:
        return default
    try:
        return _as_float(value)
    except (TypeError, ValueError, ZeroDivisionError) as exc:
        raise GONetMetadataError(
            f"Invalid optional metadata value: {value!r}."
        ) from exc


def _first(values: Mapping[str, object], aliases: tuple[str, ...]) -> object | None:
    """Return the first present metadata alias."""
    for alias in aliases:
        if alias in values:
            return values[alias]
    return None


def _as_float(value: object) -> float:
    """Convert numeric, rational, or string metadata to ``float``."""
    if isinstance(value, Fraction):
        return float(value)
    if isinstance(value, (tuple, list)) and len(value) == 2:
        numerator, denominator = value
        return float(numerator) / float(denominator)
    if hasattr(value, "numerator") and hasattr(value, "denominator"):
        return float(value.numerator) / float(value.denominator)
    return float(value)  # type: ignore[arg-type]
