"""Portable serialization for multi-image source-detection sequences."""

from __future__ import annotations

import json
from collections.abc import Iterable, Sized
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from gonet_astrometry.detection.config import DetectionConfig
from gonet_astrometry.models.detection import (
    Detection,
    DetectionCatalog,
    DetectionDiagnostics,
)
from gonet_astrometry.models.frame import ObserverLocation
from gonet_astrometry.products.errors import ProductFormatError, ProductMismatchError
from gonet_astrometry.tracking.sequence import DetectionEpoch, DetectionSequence

_FORMAT = "gonet-astrometry-detections"
_VERSION = 1


@dataclass(frozen=True, slots=True)
class DetectionProduct:
    """Reusable detection sequence plus its input-preflight provenance.

    Parameters
    ----------
    product_id
        Semantic identity computed from source-file fingerprints, detector
        configuration, algorithm, and location-preflight tolerance.
    algorithm
        Registered source-detection backend identifier.
    detection_config
        Detection and preprocessing settings used to create the catalogs.
    location_tolerance_m
        Location-group tolerance used during input preflight.
    sequence
        Chronological source catalogs and observing metadata.
    reference_path
        Source image used as the report background when the product was built.
    discovered_files
        Candidate files present before location preflight.
    zero_gps_files
        Files skipped because their parsed GPS was exactly ``0,0,0``.
    location_outlier_files
        Files skipped because they did not belong to the dominant site group.
    metadata_errors
        Files skipped because required metadata could not be interpreted.
    """

    product_id: str
    algorithm: str
    detection_config: DetectionConfig
    location_tolerance_m: float
    sequence: DetectionSequence
    reference_path: Path
    discovered_files: tuple[Path, ...]
    zero_gps_files: tuple[Path, ...] = ()
    location_outlier_files: tuple[Path, ...] = ()
    metadata_errors: tuple[tuple[Path, str], ...] = ()

    @property
    def skipped_file_count(self) -> int:
        """Return the number of files removed before source detection."""
        return (
            len(self.zero_gps_files)
            + len(self.location_outlier_files)
            + len(self.metadata_errors)
        )


def save_detection_product(path: Path, product: DetectionProduct) -> Path:
    """Write ``product`` as a compressed NPZ containing no object arrays."""
    epochs = product.sequence.epochs
    offsets = [0]
    detections: list[Detection] = []
    for epoch in epochs:
        detections.extend(epoch.catalog.detections)
        offsets.append(len(detections))

    metadata_error_paths = tuple(item[0] for item in product.metadata_errors)
    metadata_error_messages = tuple(item[1] for item in product.metadata_errors)

    arrays: dict[str, NDArray[np.generic]] = {
        "format": _scalar_string(_FORMAT),
        "version": np.asarray(_VERSION, dtype=np.int64),
        "product_id": _scalar_string(product.product_id),
        "algorithm": _scalar_string(product.algorithm),
        "detection_config_json": _scalar_string(
            json.dumps(
                asdict(product.detection_config),
                sort_keys=True,
                separators=(",", ":"),
            )
        ),
        "location_tolerance_m": np.asarray(
            product.location_tolerance_m, dtype=np.float64
        ),
        "reference_path": _scalar_string(str(product.reference_path)),
        "discovered_files": _strings(product.discovered_files),
        "zero_gps_files": _strings(product.zero_gps_files),
        "location_outlier_files": _strings(product.location_outlier_files),
        "metadata_error_paths": _strings(metadata_error_paths),
        "metadata_error_messages": _strings(metadata_error_messages),
        "frame_identifiers": _strings(epoch.frame_identifier for epoch in epochs),
        "source_paths": _strings(epoch.source_path for epoch in epochs),
        "exposure_midpoints": _strings(
            epoch.exposure_midpoint.isoformat() for epoch in epochs
        ),
        "latitude_deg": np.asarray(
            [epoch.location.latitude_deg for epoch in epochs], dtype=np.float64
        ),
        "longitude_deg": np.asarray(
            [epoch.location.longitude_deg for epoch in epochs], dtype=np.float64
        ),
        "elevation_m": np.asarray(
            [epoch.location.elevation_m for epoch in epochs], dtype=np.float64
        ),
        "image_shapes": np.asarray(
            [epoch.image_shape for epoch in epochs], dtype=np.int64
        ),
        "sensor_orientations": _strings(
            epoch.sensor_orientation for epoch in epochs
        ),
        "detector_names": _strings(epoch.catalog.detector_name for epoch in epochs),
        "catalog_offsets": np.asarray(offsets, dtype=np.int64),
        "detection_identifier": np.asarray(
            [item.identifier for item in detections], dtype=np.int64
        ),
        "x": np.asarray([item.x for item in detections], dtype=np.float64),
        "y": np.asarray([item.y for item in detections], dtype=np.float64),
        "flux": np.asarray([item.flux for item in detections], dtype=np.float64),
        "signal_to_noise": np.asarray(
            [item.signal_to_noise for item in detections], dtype=np.float64
        ),
        "x_uncertainty": np.asarray(
            [item.x_uncertainty for item in detections], dtype=np.float64
        ),
        "y_uncertainty": np.asarray(
            [item.y_uncertainty for item in detections], dtype=np.float64
        ),
        "elongation": _optional_float_array(item.elongation for item in detections),
        "flags_json": _strings(json.dumps(item.flags) for item in detections),
        "diag_peak_value": _optional_float_array(
            item.diagnostics.peak_value for item in detections
        ),
        "diag_area_pixels": _optional_int_array(
            item.diagnostics.area_pixels for item in detections
        ),
        "diag_semimajor_sigma_px": _optional_float_array(
            item.diagnostics.semimajor_sigma_px for item in detections
        ),
        "diag_semiminor_sigma_px": _optional_float_array(
            item.diagnostics.semiminor_sigma_px for item in detections
        ),
        "diag_orientation_deg": _optional_float_array(
            item.diagnostics.orientation_deg for item in detections
        ),
        "diag_ellipticity": _optional_float_array(
            item.diagnostics.ellipticity for item in detections
        ),
        "diag_backend_flags": _optional_int_array(
            item.diagnostics.backend_flags for item in detections
        ),
        "diag_sharpness": _optional_float_array(
            item.diagnostics.sharpness for item in detections
        ),
        "diag_roundness1": _optional_float_array(
            item.diagnostics.roundness1 for item in detections
        ),
        "diag_roundness2": _optional_float_array(
            item.diagnostics.roundness2 for item in detections
        ),
    }
    return _atomic_savez(path, arrays)


def load_detection_product(
    path: Path,
    *,
    expected_product_id: str | None = None,
) -> DetectionProduct:
    """Load and validate a portable source-detection product."""
    try:
        with np.load(path, allow_pickle=False) as data:
            _validate_header(data, expected_product_id)
            frame_ids = _read_strings(data, "frame_identifiers")
            source_paths = _read_strings(data, "source_paths")
            midpoints = _read_strings(data, "exposure_midpoints")
            orientations = _read_strings(data, "sensor_orientations")
            detector_names = _read_strings(data, "detector_names")
            offsets = np.asarray(data["catalog_offsets"], dtype=np.int64)
            epoch_count = len(frame_ids)
            _require_lengths(
                epoch_count,
                source_paths=source_paths,
                midpoints=midpoints,
                orientations=orientations,
                detector_names=detector_names,
                latitude_deg=data["latitude_deg"],
                longitude_deg=data["longitude_deg"],
                elevation_m=data["elevation_m"],
                image_shapes=data["image_shapes"],
            )
            if len(offsets) != epoch_count + 1 or offsets[0] != 0:
                raise ProductFormatError("Invalid detection catalog offsets")
            detection_count = int(offsets[-1])
            detection_arrays = _read_detection_arrays(data, detection_count)
            epochs: list[DetectionEpoch] = []
            for index in range(epoch_count):
                start = int(offsets[index])
                stop = int(offsets[index + 1])
                if stop < start:
                    raise ProductFormatError(
                        "Detection catalog offsets are not monotonic"
                    )
                catalog_detections = tuple(
                    _detection_from_arrays(detection_arrays, item)
                    for item in range(start, stop)
                )
                catalog = DetectionCatalog(
                    frame_identifier=frame_ids[index],
                    detections=catalog_detections,
                    detector_name=detector_names[index],
                )
                shape_row = np.asarray(data["image_shapes"])[index]
                if shape_row.shape != (2,):
                    raise ProductFormatError("Invalid image shape in detection product")
                epochs.append(
                    DetectionEpoch(
                        frame_identifier=frame_ids[index],
                        source_path=Path(source_paths[index]),
                        exposure_midpoint=datetime.fromisoformat(midpoints[index]),
                        location=ObserverLocation(
                            float(np.asarray(data["latitude_deg"])[index]),
                            float(np.asarray(data["longitude_deg"])[index]),
                            float(np.asarray(data["elevation_m"])[index]),
                        ),
                        image_shape=(int(shape_row[0]), int(shape_row[1])),
                        sensor_orientation=orientations[index],
                        catalog=catalog,
                    )
                )

            error_paths = _read_strings(data, "metadata_error_paths")
            error_messages = _read_strings(data, "metadata_error_messages")
            if len(error_paths) != len(error_messages):
                raise ProductFormatError("Metadata-error arrays have different lengths")
            config_payload = json.loads(
                _read_scalar_string(data, "detection_config_json")
            )
            if not isinstance(config_payload, dict):
                raise ProductFormatError("Detection configuration is not a JSON object")
            config_mapping = cast(dict[str, Any], config_payload)
            product = DetectionProduct(
                product_id=_read_scalar_string(data, "product_id"),
                algorithm=_read_scalar_string(data, "algorithm"),
                detection_config=DetectionConfig(**config_mapping),
                location_tolerance_m=float(
                    np.asarray(data["location_tolerance_m"]).item()
                ),
                sequence=DetectionSequence(tuple(epochs)),
                reference_path=Path(_read_scalar_string(data, "reference_path")),
                discovered_files=tuple(
                    Path(item) for item in _read_strings(data, "discovered_files")
                ),
                zero_gps_files=tuple(
                    Path(item) for item in _read_strings(data, "zero_gps_files")
                ),
                location_outlier_files=tuple(
                    Path(item)
                    for item in _read_strings(data, "location_outlier_files")
                ),
                metadata_errors=tuple(
                    (Path(item), message)
                    for item, message in zip(error_paths, error_messages, strict=True)
                ),
            )
    except ProductMismatchError:
        raise
    except (KeyError, OSError, TypeError, ValueError) as exc:
        if isinstance(exc, ProductFormatError):
            raise
        raise ProductFormatError(
            f"Could not read detection product {path}: {exc}"
        ) from exc
    return product


def _validate_header(
    data: np.lib.npyio.NpzFile,
    expected_product_id: str | None,
) -> None:
    if _read_scalar_string(data, "format") != _FORMAT:
        raise ProductFormatError("File is not a GONet Astrometry detection product")
    if int(np.asarray(data["version"]).item()) != _VERSION:
        raise ProductFormatError("Unsupported detection product version")
    product_id = _read_scalar_string(data, "product_id")
    if expected_product_id is not None and product_id != expected_product_id:
        raise ProductMismatchError(
            "Detection product does not match the current inputs or settings"
        )


def _read_detection_arrays(
    data: np.lib.npyio.NpzFile,
    count: int,
) -> dict[str, NDArray[np.generic] | tuple[str, ...]]:
    names = (
        "detection_identifier",
        "x",
        "y",
        "flux",
        "signal_to_noise",
        "x_uncertainty",
        "y_uncertainty",
        "elongation",
        "diag_peak_value",
        "diag_area_pixels",
        "diag_semimajor_sigma_px",
        "diag_semiminor_sigma_px",
        "diag_orientation_deg",
        "diag_ellipticity",
        "diag_backend_flags",
        "diag_sharpness",
        "diag_roundness1",
        "diag_roundness2",
    )
    arrays: dict[str, NDArray[np.generic] | tuple[str, ...]] = {}
    for name in names:
        value = np.asarray(data[name])
        if len(value) != count:
            raise ProductFormatError(f"Detection array {name!r} has invalid length")
        arrays[name] = value
    flags = _read_strings(data, "flags_json")
    if len(flags) != count:
        raise ProductFormatError("Detection flags array has invalid length")
    arrays["flags_json"] = flags
    return arrays


def _detection_from_arrays(
    arrays: dict[str, NDArray[np.generic] | tuple[str, ...]],
    index: int,
) -> Detection:
    def value(name: str) -> object:
        return arrays[name][index]  # type: ignore[index]

    flags_value = json.loads(str(value("flags_json")))
    if not isinstance(flags_value, list) or not all(
        isinstance(item, str) for item in flags_value
    ):
        raise ProductFormatError("Detection flags are not a JSON string list")
    diagnostics = DetectionDiagnostics(
        peak_value=_none_if_nan(value("diag_peak_value")),
        area_pixels=_none_if_negative_int(value("diag_area_pixels")),
        semimajor_sigma_px=_none_if_nan(value("diag_semimajor_sigma_px")),
        semiminor_sigma_px=_none_if_nan(value("diag_semiminor_sigma_px")),
        orientation_deg=_none_if_nan(value("diag_orientation_deg")),
        ellipticity=_none_if_nan(value("diag_ellipticity")),
        backend_flags=_none_if_negative_int(value("diag_backend_flags")),
        sharpness=_none_if_nan(value("diag_sharpness")),
        roundness1=_none_if_nan(value("diag_roundness1")),
        roundness2=_none_if_nan(value("diag_roundness2")),
    )
    return Detection(
        identifier=int(value("detection_identifier")),
        x=float(value("x")),
        y=float(value("y")),
        flux=float(value("flux")),
        signal_to_noise=float(value("signal_to_noise")),
        x_uncertainty=float(value("x_uncertainty")),
        y_uncertainty=float(value("y_uncertainty")),
        elongation=_none_if_nan(value("elongation")),
        flags=tuple(cast(list[str], flags_value)),
        diagnostics=diagnostics,
    )


def _optional_float_array(
    values: Iterable[float | None],
) -> NDArray[np.float64]:
    return np.asarray(
        [np.nan if item is None else float(item) for item in values],
        dtype=np.float64,
    )


def _optional_int_array(values: Iterable[int | None]) -> NDArray[np.int64]:
    return np.asarray(
        [-1 if item is None else int(item) for item in values],
        dtype=np.int64,
    )


def _none_if_nan(value: object) -> float | None:
    parsed = float(value)
    return None if np.isnan(parsed) else parsed


def _none_if_negative_int(value: object) -> int | None:
    parsed = int(value)
    return None if parsed < 0 else parsed


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


def _require_lengths(expected: int, **values: Sized) -> None:
    for name, value in values.items():
        if len(value) != expected:
            raise ProductFormatError(f"Product field {name!r} has invalid length")


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
