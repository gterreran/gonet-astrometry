"""Portable serialization for bootstrap image-plane star tracks."""

from __future__ import annotations

from collections.abc import Iterable
import json
from dataclasses import asdict, dataclass
from typing import Any, cast
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from gonet_astrometry.models.track import (
    StarTrack,
    TrackClass,
    TrackDiagnostics,
    TrackPoint,
)
from gonet_astrometry.products.errors import ProductFormatError, ProductMismatchError
from gonet_astrometry.tracking.config import TrackingConfig
from gonet_astrometry.tracking.image_plane import ImagePlaneTrackingResult
from gonet_astrometry.tracking.sequence import DetectionSequence

_FORMAT = "gonet-astrometry-tracks"
_VERSION = 1


@dataclass(frozen=True, slots=True)
class TrackingProduct:
    """Reusable image-plane tracking result and its provenance identifiers."""

    product_id: str
    detection_product_id: str
    tracking_config: TrackingConfig
    result: ImagePlaneTrackingResult


def save_tracking_product(path: Path, product: TrackingProduct) -> Path:
    """Write ``product`` as a compressed NPZ containing no object arrays."""
    tracks = product.result.tracks
    offsets = [0]
    points: list[TrackPoint] = []
    for track in tracks:
        points.extend(track.points)
        offsets.append(len(points))

    diagnostics = [track.diagnostics for track in tracks]
    arrays: dict[str, NDArray[np.generic]] = {
        "format": _scalar_string(_FORMAT),
        "version": np.asarray(_VERSION, dtype=np.int64),
        "product_id": _scalar_string(product.product_id),
        "detection_product_id": _scalar_string(product.detection_product_id),
        "tracking_config_json": _scalar_string(
            json.dumps(
                asdict(product.tracking_config),
                sort_keys=True,
                separators=(",", ":"),
            )
        ),
        "track_identifier": np.asarray(
            [track.identifier for track in tracks], dtype=np.int64
        ),
        "catalog_identifier": _strings(
            "" if track.catalog_identifier is None else track.catalog_identifier
            for track in tracks
        ),
        "quality": np.asarray(
            [np.nan if track.quality is None else track.quality for track in tracks],
            dtype=np.float64,
        ),
        "point_offsets": np.asarray(offsets, dtype=np.int64),
        "point_frame_identifier": _strings(point.frame_identifier for point in points),
        "point_detection_identifier": np.asarray(
            [point.detection_identifier for point in points], dtype=np.int64
        ),
        "diag_present": np.asarray(
            [item is not None for item in diagnostics], dtype=np.bool_
        ),
        "diag_duration_s": _diag_float(diagnostics, "duration_s"),
        "diag_displacement_px": _diag_float(diagnostics, "displacement_px"),
        "diag_mean_speed_px_per_minute": _diag_float(
            diagnostics, "mean_speed_px_per_minute"
        ),
        "diag_fit_rms_px": _diag_float(diagnostics, "fit_rms_px"),
        "diag_missed_frames": _diag_int(diagnostics, "missed_frames"),
        "diag_class": _strings(
            "candidate" if item is None else item.diagnostic_class
            for item in diagnostics
        ),
        "diag_detection_count": _diag_int(diagnostics, "detection_count"),
        "diag_span_epoch_count": _diag_int(diagnostics, "span_epoch_count"),
        "diag_coverage_fraction": _diag_float(diagnostics, "coverage_fraction"),
        "diag_median_interval_s": _diag_float(diagnostics, "median_interval_s"),
        "diag_max_interval_s": _diag_float(diagnostics, "max_interval_s"),
    }
    return _atomic_savez(path, arrays)


def load_tracking_product(
    path: Path,
    sequence: DetectionSequence,
    *,
    expected_product_id: str | None = None,
    expected_detection_product_id: str | None = None,
) -> TrackingProduct:
    """Load and validate a portable bootstrap tracking product."""
    try:
        with np.load(path, allow_pickle=False) as data:
            _validate_header(
                data,
                expected_product_id=expected_product_id,
                expected_detection_product_id=expected_detection_product_id,
            )
            identifiers = np.asarray(data["track_identifier"], dtype=np.int64)
            catalog_ids = _read_strings(data, "catalog_identifier")
            quality = np.asarray(data["quality"], dtype=np.float64)
            offsets = np.asarray(data["point_offsets"], dtype=np.int64)
            frame_ids = _read_strings(data, "point_frame_identifier")
            detection_ids = np.asarray(
                data["point_detection_identifier"], dtype=np.int64
            )
            present = np.asarray(data["diag_present"], dtype=np.bool_)
            count = len(identifiers)
            if len(offsets) != count + 1 or offsets[0] != 0:
                raise ProductFormatError("Invalid tracking point offsets")
            if (
                int(offsets[-1]) != len(frame_ids)
                or len(frame_ids) != len(detection_ids)
            ):
                raise ProductFormatError(
                    "Tracking point arrays have inconsistent lengths"
                )
            _require_track_lengths(data, count, catalog_ids, quality, present)
            tracks: list[StarTrack] = []
            for index in range(count):
                start = int(offsets[index])
                stop = int(offsets[index + 1])
                if stop < start:
                    raise ProductFormatError("Tracking point offsets are not monotonic")
                points = tuple(
                    TrackPoint(frame_ids[item], int(detection_ids[item]))
                    for item in range(start, stop)
                )
                diagnostics = (
                    _diagnostics_from_data(data, index) if present[index] else None
                )
                catalog_identifier = catalog_ids[index] or None
                track_quality = (
                    None if np.isnan(quality[index]) else float(quality[index])
                )
                tracks.append(
                    StarTrack(
                        identifier=int(identifiers[index]),
                        points=points,
                        catalog_identifier=catalog_identifier,
                        quality=track_quality,
                        diagnostics=diagnostics,
                    )
                )
            config_payload = json.loads(
                _read_scalar_string(data, "tracking_config_json")
            )
            if not isinstance(config_payload, dict):
                raise ProductFormatError("Tracking configuration is not a JSON object")
            config_mapping = cast(dict[str, Any], config_payload)
            product = TrackingProduct(
                product_id=_read_scalar_string(data, "product_id"),
                detection_product_id=_read_scalar_string(
                    data, "detection_product_id"
                ),
                tracking_config=TrackingConfig(**config_mapping),
                result=ImagePlaneTrackingResult(sequence, tuple(tracks)),
            )
    except ProductMismatchError:
        raise
    except (KeyError, OSError, TypeError, ValueError) as exc:
        if isinstance(exc, ProductFormatError):
            raise
        raise ProductFormatError(
            f"Could not read tracking product {path}: {exc}"
        ) from exc
    return product


def _validate_header(
    data: np.lib.npyio.NpzFile,
    *,
    expected_product_id: str | None,
    expected_detection_product_id: str | None,
) -> None:
    if _read_scalar_string(data, "format") != _FORMAT:
        raise ProductFormatError("File is not a GONet Astrometry tracking product")
    if int(np.asarray(data["version"]).item()) != _VERSION:
        raise ProductFormatError("Unsupported tracking product version")
    product_id = _read_scalar_string(data, "product_id")
    detection_id = _read_scalar_string(data, "detection_product_id")
    if expected_product_id is not None and product_id != expected_product_id:
        raise ProductMismatchError(
            "Tracking product does not match the current tracking settings"
        )
    if (
        expected_detection_product_id is not None
        and detection_id != expected_detection_product_id
    ):
        raise ProductMismatchError(
            "Tracking product belongs to a different detection product"
        )


def _diagnostics_from_data(
    data: np.lib.npyio.NpzFile,
    index: int,
) -> TrackDiagnostics:
    diagnostic_class = str(np.asarray(data["diag_class"])[index])
    if diagnostic_class not in {"candidate", "low-motion", "poor-fit"}:
        raise ProductFormatError("Invalid track diagnostic class")
    return TrackDiagnostics(
        duration_s=float(np.asarray(data["diag_duration_s"])[index]),
        displacement_px=float(np.asarray(data["diag_displacement_px"])[index]),
        mean_speed_px_per_minute=float(
            np.asarray(data["diag_mean_speed_px_per_minute"])[index]
        ),
        fit_rms_px=float(np.asarray(data["diag_fit_rms_px"])[index]),
        missed_frames=int(np.asarray(data["diag_missed_frames"])[index]),
        diagnostic_class=cast(TrackClass, diagnostic_class),
        detection_count=int(np.asarray(data["diag_detection_count"])[index]),
        span_epoch_count=int(np.asarray(data["diag_span_epoch_count"])[index]),
        coverage_fraction=float(np.asarray(data["diag_coverage_fraction"])[index]),
        median_interval_s=float(np.asarray(data["diag_median_interval_s"])[index]),
        max_interval_s=float(np.asarray(data["diag_max_interval_s"])[index]),
    )


def _require_track_lengths(
    data: np.lib.npyio.NpzFile,
    count: int,
    catalog_ids: tuple[str, ...],
    quality: NDArray[np.float64],
    present: NDArray[np.bool_],
) -> None:
    if len(catalog_ids) != count or len(quality) != count or len(present) != count:
        raise ProductFormatError("Tracking arrays have inconsistent lengths")
    for name in (
        "diag_duration_s",
        "diag_displacement_px",
        "diag_mean_speed_px_per_minute",
        "diag_fit_rms_px",
        "diag_missed_frames",
        "diag_class",
        "diag_detection_count",
        "diag_span_epoch_count",
        "diag_coverage_fraction",
        "diag_median_interval_s",
        "diag_max_interval_s",
    ):
        if len(np.asarray(data[name])) != count:
            raise ProductFormatError(f"Tracking field {name!r} has invalid length")


def _diag_float(
    values: list[TrackDiagnostics | None],
    name: str,
) -> NDArray[np.float64]:
    return np.asarray(
        [np.nan if item is None else float(getattr(item, name)) for item in values],
        dtype=np.float64,
    )


def _diag_int(values: list[TrackDiagnostics | None], name: str) -> NDArray[np.int64]:
    return np.asarray(
        [-1 if item is None else int(getattr(item, name)) for item in values],
        dtype=np.int64,
    )


def _strings(values: Iterable[object]) -> NDArray[np.str_]:
    strings = tuple(str(item) for item in values)
    width = max((len(item) for item in strings), default=1)
    return np.asarray(strings, dtype=f"<U{width}")


def _scalar_string(value: str) -> NDArray[np.str_]:
    return np.asarray(value, dtype=f"<U{max(1, len(value))}")


def _read_strings(data: np.lib.npyio.NpzFile, name: str) -> tuple[str, ...]:
    array = np.asarray(data[name])
    if array.ndim != 1:
        raise ProductFormatError(f"Product field {name!r} must be one-dimensional")
    return tuple(str(item) for item in array.tolist())


def _read_scalar_string(data: np.lib.npyio.NpzFile, name: str) -> str:
    array = np.asarray(data[name])
    if array.ndim != 0:
        raise ProductFormatError(f"Product field {name!r} must be scalar")
    return str(array.item())


def _atomic_savez(path: Path, arrays: dict[str, NDArray[np.generic]]) -> Path:
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination
