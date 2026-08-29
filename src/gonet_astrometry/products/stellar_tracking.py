"""Portable products for spherical temporal and merged stellar tracks.

The legacy :mod:`gonet_astrometry.products.tracks` product remains readable for
older image-plane runs.  New Grid-assisted workflows use two explicit products:

``temporal_tracks.npz``
    Spherical, timestamp-aware associations produced directly from detections.

``stellar_tracks.npz``
    Sidereal-consistent physical star tracks after robust fragment merging.

Both formats are pickle-free and store only plain NumPy arrays plus JSON-encoded
configuration dictionaries.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from gonet_astrometry.models.track import StarTrack, TrackDiagnostics, TrackPoint
from gonet_astrometry.products.errors import ProductFormatError, ProductMismatchError
from gonet_astrometry.solving.sidereal import SiderealFitConfig
from gonet_astrometry.solving.stellar_tracks import StellarTrackMergeConfig
from gonet_astrometry.tracking.image_plane import ImagePlaneTrackingResult
from gonet_astrometry.tracking.sequence import DetectionSequence
from gonet_astrometry.tracking.spherical import SphericalTrackingConfig

_TEMPORAL_FORMAT = "gonet-astrometry-temporal-tracks"
_STELLAR_FORMAT = "gonet-astrometry-stellar-tracks"
_VERSION = 1


@dataclass(frozen=True, slots=True)
class TemporalTrackingProduct:
    """Reusable spherical temporal-association product."""

    product_id: str
    detection_product_id: str
    grid_calibration_path: Path
    tracking_config: SphericalTrackingConfig
    result: ImagePlaneTrackingResult


@dataclass(frozen=True, slots=True)
class StellarTrackingProduct:
    """Reusable sidereal-consistent merged stellar-track product."""

    product_id: str
    temporal_tracking_product_id: str
    grid_calibration_path: Path
    merge_config: StellarTrackMergeConfig
    sidereal_config: SiderealFitConfig
    reference_time: datetime
    source_fragment_ids: tuple[tuple[int, ...], ...]
    result: ImagePlaneTrackingResult

    def __post_init__(self) -> None:
        if self.reference_time.tzinfo is None:
            raise ValueError("reference_time must be timezone-aware")
        if len(self.source_fragment_ids) != len(self.result.tracks):
            raise ValueError(
                "source_fragment_ids must align one-to-one with result.tracks"
            )


def save_temporal_tracking_product(
    path: Path,
    product: TemporalTrackingProduct,
) -> Path:
    """Write a pickle-free spherical temporal-track product."""
    arrays = _track_arrays(product.result)
    arrays.update(
        {
            "format": _scalar_string(_TEMPORAL_FORMAT),
            "version": np.asarray(_VERSION, dtype=np.int64),
            "product_id": _scalar_string(product.product_id),
            "detection_product_id": _scalar_string(product.detection_product_id),
            "grid_calibration_path": _scalar_string(
                str(product.grid_calibration_path.expanduser().resolve())
            ),
            "tracking_config_json": _scalar_string(
                _json_mapping(asdict(product.tracking_config))
            ),
        }
    )
    return _atomic_savez(path, arrays)


def load_temporal_tracking_product(
    path: Path,
    sequence: DetectionSequence,
    *,
    expected_product_id: str | None = None,
    expected_detection_product_id: str | None = None,
) -> TemporalTrackingProduct:
    """Load and provenance-check a spherical temporal-track product."""
    try:
        with np.load(path, allow_pickle=False) as data:
            _validate_header(
                data,
                expected_format=_TEMPORAL_FORMAT,
                expected_product_id=expected_product_id,
            )
            detection_product_id = _read_scalar_string(data, "detection_product_id")
            if (
                expected_detection_product_id is not None
                and detection_product_id != expected_detection_product_id
            ):
                raise ProductMismatchError(
                    "Temporal tracking product was created from a different "
                    "detection product"
                )
            result = _read_track_result(data, sequence)
            return TemporalTrackingProduct(
                product_id=_read_scalar_string(data, "product_id"),
                detection_product_id=detection_product_id,
                grid_calibration_path=Path(
                    _read_scalar_string(data, "grid_calibration_path")
                ),
                tracking_config=SphericalTrackingConfig(
                    **json.loads(_read_scalar_string(data, "tracking_config_json"))
                ),
                result=result,
            )
    except (OSError, KeyError, ValueError, TypeError) as exc:
        if isinstance(exc, (ProductFormatError, ProductMismatchError)):
            raise
        raise ProductFormatError(
            f"Could not load temporal tracking product {path}: {exc}"
        ) from exc


def save_stellar_tracking_product(
    path: Path,
    product: StellarTrackingProduct,
) -> Path:
    """Write a pickle-free merged stellar-track product."""
    arrays = _track_arrays(product.result)

    fragment_offsets = [0]
    fragment_ids: list[int] = []
    for source_ids in product.source_fragment_ids:
        fragment_ids.extend(source_ids)
        fragment_offsets.append(len(fragment_ids))

    arrays.update(
        {
            "format": _scalar_string(_STELLAR_FORMAT),
            "version": np.asarray(_VERSION, dtype=np.int64),
            "product_id": _scalar_string(product.product_id),
            "temporal_tracking_product_id": _scalar_string(
                product.temporal_tracking_product_id
            ),
            "grid_calibration_path": _scalar_string(
                str(product.grid_calibration_path.expanduser().resolve())
            ),
            "merge_config_json": _scalar_string(
                _json_mapping(asdict(product.merge_config))
            ),
            "sidereal_config_json": _scalar_string(
                _json_mapping(asdict(product.sidereal_config))
            ),
            "reference_time_iso": _scalar_string(product.reference_time.isoformat()),
            "source_fragment_offsets": np.asarray(fragment_offsets, dtype=np.int64),
            "source_fragment_identifier": np.asarray(fragment_ids, dtype=np.int64),
        }
    )
    return _atomic_savez(path, arrays)


def load_stellar_tracking_product(
    path: Path,
    sequence: DetectionSequence,
    *,
    expected_product_id: str | None = None,
    expected_temporal_tracking_product_id: str | None = None,
) -> StellarTrackingProduct:
    """Load and provenance-check a merged stellar-track product."""
    try:
        with np.load(path, allow_pickle=False) as data:
            _validate_header(
                data,
                expected_format=_STELLAR_FORMAT,
                expected_product_id=expected_product_id,
            )
            temporal_id = _read_scalar_string(data, "temporal_tracking_product_id")
            if (
                expected_temporal_tracking_product_id is not None
                and temporal_id != expected_temporal_tracking_product_id
            ):
                raise ProductMismatchError(
                    "Stellar tracking product was created from a different "
                    "temporal tracking product"
                )

            result = _read_track_result(data, sequence)
            source_fragment_ids = _read_fragment_groups(
                data,
                len(result.tracks),
            )
            return StellarTrackingProduct(
                product_id=_read_scalar_string(data, "product_id"),
                temporal_tracking_product_id=temporal_id,
                grid_calibration_path=Path(
                    _read_scalar_string(data, "grid_calibration_path")
                ),
                merge_config=StellarTrackMergeConfig(
                    **json.loads(_read_scalar_string(data, "merge_config_json"))
                ),
                sidereal_config=SiderealFitConfig(
                    **json.loads(_read_scalar_string(data, "sidereal_config_json"))
                ),
                reference_time=datetime.fromisoformat(
                    _read_scalar_string(data, "reference_time_iso")
                ),
                source_fragment_ids=source_fragment_ids,
                result=result,
            )
    except (OSError, KeyError, ValueError, TypeError) as exc:
        if isinstance(exc, (ProductFormatError, ProductMismatchError)):
            raise
        raise ProductFormatError(
            f"Could not load stellar tracking product {path}: {exc}"
        ) from exc


def _track_arrays(
    result: ImagePlaneTrackingResult,
) -> dict[str, NDArray[np.generic]]:
    tracks = result.tracks
    points: list[TrackPoint] = []
    point_offsets = [0]
    for track in tracks:
        points.extend(track.points)
        point_offsets.append(len(points))

    diagnostics = [track.diagnostics for track in tracks]
    return {
        "track_identifier": np.asarray(
            [track.identifier for track in tracks],
            dtype=np.int64,
        ),
        "catalog_identifier": _strings(
            "" if track.catalog_identifier is None else track.catalog_identifier
            for track in tracks
        ),
        "quality": np.asarray(
            [
                np.nan if track.quality is None else float(track.quality)
                for track in tracks
            ],
            dtype=np.float64,
        ),
        "point_offsets": np.asarray(point_offsets, dtype=np.int64),
        "point_frame_identifier": _strings(point.frame_identifier for point in points),
        "point_detection_identifier": np.asarray(
            [point.detection_identifier for point in points],
            dtype=np.int64,
        ),
        "diag_present": np.asarray(
            [item is not None for item in diagnostics],
            dtype=np.bool_,
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


def _read_track_result(
    data: np.lib.npyio.NpzFile,
    sequence: DetectionSequence,
) -> ImagePlaneTrackingResult:
    identifiers = np.asarray(data["track_identifier"], dtype=np.int64)
    catalog_ids = _read_strings(data, "catalog_identifier")
    quality = np.asarray(data["quality"], dtype=np.float64)
    offsets = np.asarray(data["point_offsets"], dtype=np.int64)
    frame_ids = _read_strings(data, "point_frame_identifier")
    detection_ids = np.asarray(
        data["point_detection_identifier"],
        dtype=np.int64,
    )
    present = np.asarray(data["diag_present"], dtype=np.bool_)

    count = len(identifiers)
    if len(offsets) != count + 1 or int(offsets[0]) != 0:
        raise ProductFormatError("Invalid tracking point offsets")
    if int(offsets[-1]) != len(frame_ids) or len(frame_ids) != len(detection_ids):
        raise ProductFormatError("Tracking point arrays have inconsistent lengths")
    _require_track_lengths(data, count, catalog_ids, quality, present)

    tracks: list[StarTrack] = []
    for index, identifier in enumerate(identifiers):
        start = int(offsets[index])
        stop = int(offsets[index + 1])
        points = tuple(
            TrackPoint(frame_id, int(detection_id))
            for frame_id, detection_id in zip(
                frame_ids[start:stop],
                detection_ids[start:stop],
                strict=True,
            )
        )

        diagnostics: TrackDiagnostics | None = None
        if bool(present[index]):
            diagnostics = TrackDiagnostics(
                duration_s=float(data["diag_duration_s"][index]),
                displacement_px=float(data["diag_displacement_px"][index]),
                mean_speed_px_per_minute=float(
                    data["diag_mean_speed_px_per_minute"][index]
                ),
                fit_rms_px=float(data["diag_fit_rms_px"][index]),
                missed_frames=int(data["diag_missed_frames"][index]),
                diagnostic_class=str(data["diag_class"][index]),
                detection_count=int(data["diag_detection_count"][index]),
                span_epoch_count=int(data["diag_span_epoch_count"][index]),
                coverage_fraction=float(data["diag_coverage_fraction"][index]),
                median_interval_s=float(data["diag_median_interval_s"][index]),
                max_interval_s=float(data["diag_max_interval_s"][index]),
            )

        catalog_identifier = catalog_ids[index] or None
        quality_value = float(quality[index])
        tracks.append(
            StarTrack(
                identifier=int(identifier),
                points=points,
                catalog_identifier=catalog_identifier,
                quality=(quality_value if np.isfinite(quality_value) else None),
                diagnostics=diagnostics,
            )
        )

    return ImagePlaneTrackingResult(sequence, tuple(tracks))


def _read_fragment_groups(
    data: np.lib.npyio.NpzFile,
    track_count: int,
) -> tuple[tuple[int, ...], ...]:
    offsets = np.asarray(
        data["source_fragment_offsets"],
        dtype=np.int64,
    )
    identifiers = np.asarray(
        data["source_fragment_identifier"],
        dtype=np.int64,
    )
    if len(offsets) != track_count + 1 or int(offsets[0]) != 0:
        raise ProductFormatError("Invalid source-fragment offsets")
    if int(offsets[-1]) != len(identifiers):
        raise ProductFormatError("Source-fragment arrays have inconsistent lengths")

    return tuple(
        tuple(
            int(item)
            for item in identifiers[int(offsets[index]) : int(offsets[index + 1])]
        )
        for index in range(track_count)
    )


def _validate_header(
    data: np.lib.npyio.NpzFile,
    *,
    expected_format: str,
    expected_product_id: str | None,
) -> None:
    if _read_scalar_string(data, "format") != expected_format:
        raise ProductFormatError(
            f"Unexpected product format; expected {expected_format!r}"
        )
    version = int(np.asarray(data["version"]).item())
    if version != _VERSION:
        raise ProductFormatError(
            f"Unsupported product version {version}; expected {_VERSION}"
        )
    product_id = _read_scalar_string(data, "product_id")
    if expected_product_id is not None and product_id != expected_product_id:
        raise ProductMismatchError(
            "Track product provenance does not match the current workflow"
        )


def _require_track_lengths(
    data: np.lib.npyio.NpzFile,
    count: int,
    catalog_ids: tuple[str, ...],
    quality: NDArray[np.float64],
    present: NDArray[np.bool_],
) -> None:
    if len(catalog_ids) != count or len(quality) != count or len(present) != count:
        raise ProductFormatError("Tracking metadata arrays have invalid length")
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
    diagnostics: list[TrackDiagnostics | None],
    name: str,
) -> NDArray[np.float64]:
    return np.asarray(
        [
            np.nan if item is None else float(getattr(item, name))
            for item in diagnostics
        ],
        dtype=np.float64,
    )


def _diag_int(
    diagnostics: list[TrackDiagnostics | None],
    name: str,
) -> NDArray[np.int64]:
    return np.asarray(
        [-1 if item is None else int(getattr(item, name)) for item in diagnostics],
        dtype=np.int64,
    )


def _json_mapping(value: dict[str, Any]) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _strings(values: Iterable[object]) -> NDArray[np.str_]:
    strings = tuple(str(item) for item in values)
    width = max((len(item) for item in strings), default=1)
    return np.asarray(strings, dtype=f"<U{width}")


def _scalar_string(value: str) -> NDArray[np.str_]:
    return np.asarray(value, dtype=f"<U{max(1, len(value))}")


def _read_strings(
    data: np.lib.npyio.NpzFile,
    name: str,
) -> tuple[str, ...]:
    array = np.asarray(data[name])
    if array.ndim != 1:
        raise ProductFormatError(f"Product field {name!r} must be one-dimensional")
    return tuple(str(item) for item in array.tolist())


def _read_scalar_string(
    data: np.lib.npyio.NpzFile,
    name: str,
) -> str:
    array = np.asarray(data[name])
    if array.ndim != 0:
        raise ProductFormatError(f"Product field {name!r} must be scalar")
    return str(array.item())


def _atomic_savez(
    path: Path,
    arrays: dict[str, NDArray[np.generic]],
) -> Path:
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
