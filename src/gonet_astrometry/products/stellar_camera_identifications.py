"""Portable stellar identifications produced by a fixed stellar camera model."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from gonet_astrometry.products.errors import ProductFormatError, ProductMismatchError
from gonet_astrometry.tracking.catalog_identification import (
    StellarIdentification,
    StellarIdentificationEpoch,
    StellarIdentificationResult,
)
from gonet_astrometry.tracking.sequence import DetectionSequence
from gonet_astrometry.tracking.stellar_camera_identification import (
    StellarCameraIdentificationConfig,
)

_FORMAT = "gonet-astrometry-stellar-camera-identifications"
_VERSION = 1


@dataclass(frozen=True, slots=True)
class StellarCameraIdentificationProduct:
    """Reusable direct catalog associations for one stellar camera calibration."""

    product_id: str
    detection_product_id: str
    stellar_camera_calibration_path: Path
    catalog_path: Path
    config: StellarCameraIdentificationConfig
    result: StellarIdentificationResult


def save_stellar_camera_identification_product(
    path: Path,
    product: StellarCameraIdentificationProduct,
) -> Path:
    """Write a pickle-free stellar-camera identification NPZ."""
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    epochs = product.result.epochs
    offsets = [0]
    entries: list[StellarIdentification] = []
    for epoch in epochs:
        entries.extend(epoch.visible_stars)
        offsets.append(len(entries))
    arrays: dict[str, NDArray[np.generic]] = {
        "format": _scalar_string(_FORMAT),
        "version": np.asarray(_VERSION, dtype=np.int64),
        "product_id": _scalar_string(product.product_id),
        "detection_product_id": _scalar_string(product.detection_product_id),
        "stellar_camera_calibration_path": _scalar_string(
            str(product.stellar_camera_calibration_path)
        ),
        "catalog_path": _scalar_string(str(product.catalog_path)),
        "config_json": _scalar_string(
            json.dumps(asdict(product.config), sort_keys=True, separators=(",", ":"))
        ),
        "camera_to_enu": np.asarray(product.result.grid_to_enu, dtype=np.float64),
        "fit_pair_count": np.asarray(product.result.fit_pair_count, dtype=np.int64),
        "fit_median_residual_arcmin": np.asarray(
            product.result.fit_median_residual_arcmin, dtype=np.float64
        ),
        "fit_p90_residual_arcmin": np.asarray(
            product.result.fit_p90_residual_arcmin, dtype=np.float64
        ),
        "frame_identifiers": _strings(epoch.frame_identifier for epoch in epochs),
        "star_offsets": np.asarray(offsets, dtype=np.int64),
        "catalog_identifier": _strings(item.catalog_identifier for item in entries),
        "catalog_magnitude": np.asarray(
            [
                np.nan if item.catalog_magnitude is None else item.catalog_magnitude
                for item in entries
            ],
            dtype=np.float64,
        ),
        "azimuth_deg": np.asarray([item.azimuth_deg for item in entries]),
        "altitude_deg": np.asarray([item.altitude_deg for item in entries]),
        "predicted_x": np.asarray([item.predicted_x for item in entries]),
        "predicted_y": np.asarray([item.predicted_y for item in entries]),
        "detection_identifier": np.asarray(
            [
                -1 if item.detection_identifier is None else item.detection_identifier
                for item in entries
            ],
            dtype=np.int64,
        ),
        "residual_px": np.asarray(
            [
                np.nan if item.residual_px is None else item.residual_px
                for item in entries
            ]
        ),
        "residual_arcmin": np.asarray(
            [
                np.nan if item.residual_arcmin is None else item.residual_arcmin
                for item in entries
            ]
        ),
        "match_kind": _strings(
            "" if item.match_kind is None else item.match_kind for item in entries
        ),
    }
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(destination)
    return destination


def load_stellar_camera_identification_product(
    path: Path,
    sequence: DetectionSequence,
    *,
    expected_product_id: str | None = None,
    expected_detection_product_id: str | None = None,
) -> StellarCameraIdentificationProduct:
    """Load and validate direct stellar-camera catalog associations."""
    source = Path(path).expanduser().resolve()
    try:
        with np.load(source, allow_pickle=False) as data:
            if _read_scalar_string(data, "format") != _FORMAT:
                raise ProductFormatError(
                    "File is not a stellar-camera identification product"
                )
            if int(np.asarray(data["version"]).item()) != _VERSION:
                raise ProductFormatError(
                    "Unsupported stellar-camera identification product version"
                )
            product_id = _read_scalar_string(data, "product_id")
            detection_product_id = _read_scalar_string(data, "detection_product_id")
            if expected_product_id is not None and product_id != expected_product_id:
                raise ProductMismatchError(
                    "Stellar-camera identifications do not match current "
                    "inputs/settings"
                )
            if (
                expected_detection_product_id is not None
                and detection_product_id != expected_detection_product_id
            ):
                raise ProductMismatchError(
                    "Stellar-camera identifications reference another detection product"
                )
            config = StellarCameraIdentificationConfig(
                **json.loads(_read_scalar_string(data, "config_json"))
            )
            frame_identifiers = _read_strings(data, "frame_identifiers")
            sequence_ids = tuple(epoch.frame_identifier for epoch in sequence.epochs)
            if frame_identifiers != sequence_ids:
                raise ProductMismatchError(
                    "Stellar-camera identification epochs do not match detections"
                )
            offsets = np.asarray(data["star_offsets"], dtype=np.int64)
            identifiers = _read_strings(data, "catalog_identifier")
            count = len(identifiers)
            if offsets.shape != (len(frame_identifiers) + 1,):
                raise ProductFormatError(
                    "Stellar-camera identification offsets invalid"
                )
            if offsets[0] != 0 or offsets[-1] != count or np.any(np.diff(offsets) < 0):
                raise ProductFormatError(
                    "Stellar-camera identification offsets are inconsistent"
                )
            magnitude = np.asarray(data["catalog_magnitude"], dtype=np.float64)
            azimuth = np.asarray(data["azimuth_deg"], dtype=np.float64)
            altitude = np.asarray(data["altitude_deg"], dtype=np.float64)
            predicted_x = np.asarray(data["predicted_x"], dtype=np.float64)
            predicted_y = np.asarray(data["predicted_y"], dtype=np.float64)
            detection_identifier = np.asarray(
                data["detection_identifier"], dtype=np.int64
            )
            residual_px = np.asarray(data["residual_px"], dtype=np.float64)
            residual_arcmin = np.asarray(data["residual_arcmin"], dtype=np.float64)
            match_kind = _read_strings(data, "match_kind")
            arrays = (
                magnitude,
                azimuth,
                altitude,
                predicted_x,
                predicted_y,
                detection_identifier,
                residual_px,
                residual_arcmin,
            )
            if any(len(array) != count for array in arrays) or len(match_kind) != count:
                raise ProductFormatError(
                    "Stellar-camera identification arrays have invalid lengths"
                )

            result_epochs: list[StellarIdentificationEpoch] = []
            for epoch_index, frame_identifier in enumerate(frame_identifiers):
                start = int(offsets[epoch_index])
                stop = int(offsets[epoch_index + 1])
                valid_detection_ids = {
                    item.identifier
                    for item in sequence.epochs[epoch_index].catalog.detections
                }
                used: set[int] = set()
                entries: list[StellarIdentification] = []
                for index in range(start, stop):
                    detection_id = int(detection_identifier[index])
                    parsed = None if detection_id < 0 else detection_id
                    if parsed is not None and parsed not in valid_detection_ids:
                        raise ProductFormatError(
                            "Stellar-camera identification references missing detection"
                        )
                    if parsed is not None and parsed in used:
                        raise ProductFormatError(
                            "One detection is assigned to multiple catalog stars"
                        )
                    if parsed is not None:
                        used.add(parsed)
                    kind = match_kind[index] or None
                    if kind not in (None, "primary"):
                        raise ProductFormatError("Unknown direct stellar match kind")
                    entries.append(
                        StellarIdentification(
                            catalog_identifier=identifiers[index],
                            catalog_magnitude=(
                                None
                                if np.isnan(magnitude[index])
                                else float(magnitude[index])
                            ),
                            azimuth_deg=float(azimuth[index]),
                            altitude_deg=float(altitude[index]),
                            predicted_x=float(predicted_x[index]),
                            predicted_y=float(predicted_y[index]),
                            detection_identifier=parsed,
                            residual_px=(
                                None
                                if np.isnan(residual_px[index])
                                else float(residual_px[index])
                            ),
                            residual_arcmin=(
                                None
                                if np.isnan(residual_arcmin[index])
                                else float(residual_arcmin[index])
                            ),
                            match_kind=kind,
                        )
                    )
                result_epochs.append(
                    StellarIdentificationEpoch(frame_identifier, tuple(entries))
                )
            result = StellarIdentificationResult(
                grid_to_enu=np.asarray(data["camera_to_enu"], dtype=np.float64),
                bootstrap_frame_identifier="stellar-camera-calibration",
                epochs=tuple(result_epochs),
                fit_pair_count=int(np.asarray(data["fit_pair_count"]).item()),
                fit_median_residual_arcmin=float(
                    np.asarray(data["fit_median_residual_arcmin"]).item()
                ),
                fit_p90_residual_arcmin=float(
                    np.asarray(data["fit_p90_residual_arcmin"]).item()
                ),
            )
            calibration_path = Path(
                _read_scalar_string(data, "stellar_camera_calibration_path")
            )
            catalog_path = Path(_read_scalar_string(data, "catalog_path"))
    except (ProductFormatError, ProductMismatchError):
        raise
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ProductFormatError(
            f"Could not read stellar-camera identification product {source}: {exc}"
        ) from exc
    return StellarCameraIdentificationProduct(
        product_id=product_id,
        detection_product_id=detection_product_id,
        stellar_camera_calibration_path=calibration_path,
        catalog_path=catalog_path,
        config=config,
        result=result,
    )


def _scalar_string(value: str) -> NDArray[np.str_]:
    return np.asarray(value, dtype=f"<U{max(1, len(value))}")


def _strings(values) -> NDArray[np.str_]:
    parsed = tuple(str(value) for value in values)
    width = max((len(value) for value in parsed), default=1)
    return np.asarray(parsed, dtype=f"<U{width}")


def _read_scalar_string(data: np.lib.npyio.NpzFile, name: str) -> str:
    value = np.asarray(data[name])
    if value.ndim != 0:
        raise ProductFormatError(f"{name} must be scalar")
    return str(value.item())


def _read_strings(data: np.lib.npyio.NpzFile, name: str) -> tuple[str, ...]:
    value = np.asarray(data[name])
    if value.ndim != 1:
        raise ProductFormatError(f"{name} must be one-dimensional")
    return tuple(str(item) for item in value.tolist())
