"""Deterministic provenance identifiers for reusable workflow products."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from gonet_astrometry.detection.config import DetectionConfig
from gonet_astrometry.detection.multichannel import MultiChannelSEPConfig
from gonet_astrometry.solving.orientation import OrientationFitConfig
from gonet_astrometry.solving.sidereal import SiderealFitConfig
from gonet_astrometry.solving.stellar_tracks import StellarTrackMergeConfig
from gonet_astrometry.tracking.catalog_identification import StellarIdentificationConfig
from gonet_astrometry.tracking.config import TrackingConfig
from gonet_astrometry.tracking.spherical import SphericalTrackingConfig

DETECTION_PIPELINE_REVISION = 1
"""Manual cache revision for changes to detection semantics."""

TRACKING_PIPELINE_REVISION = 1
"""Manual cache revision for changes to bootstrap-tracking semantics."""

SIDEREAL_PIPELINE_REVISION = 2
"""Manual cache revision for Grid-calibrated sidereal-fitting semantics."""

ORIENTATION_PIPELINE_REVISION = 4
"""Manual cache revision for catalog-assisted absolute-orientation semantics."""

MULTICHANNEL_DETECTION_PIPELINE_REVISION = 2
"""Manual cache revision for independent native-channel SEP semantics."""

SPHERICAL_TRACKING_PIPELINE_REVISION = 1
"""Manual cache revision for Grid-aware spherical temporal association."""

STELLAR_TRACKING_PIPELINE_REVISION = 1
"""Manual cache revision for sidereal fragment bootstrap/merge semantics."""

STELLAR_IDENTIFICATION_PIPELINE_REVISION = 2
"""Manual cache revision for sequence-wide catalog identification semantics."""

HYBRID_FALLBACK_TRACKING_PIPELINE_REVISION = 1
"""Manual cache revision for hybrid unmatched-detection tracking semantics."""


@dataclass(frozen=True, slots=True)
class InputFingerprint:
    """Filesystem identity used to invalidate cached products safely.

    Parameters
    ----------
    path
        Absolute source path.
    size_bytes
        File size when the product identity was computed.
    mtime_ns
        Nanosecond-resolution modification time reported by the filesystem.
    """

    path: Path
    size_bytes: int
    mtime_ns: int

    def as_mapping(self) -> dict[str, object]:
        """Return a JSON-serializable representation of this fingerprint."""
        return {
            "path": str(self.path),
            "size_bytes": self.size_bytes,
            "mtime_ns": self.mtime_ns,
        }


def fingerprint_inputs(paths: Iterable[Path]) -> tuple[InputFingerprint, ...]:
    """Return deterministic stat-based fingerprints for input files.

    File contents are deliberately not hashed: reading hundreds of raw GONet
    files solely to validate a cache would defeat the purpose of the cache.
    Absolute path, size, and nanosecond modification time provide a cheap and
    conservative invalidation key.
    """
    ordered = tuple(
        sorted(
            {Path(path).expanduser().resolve() for path in paths},
            key=lambda item: str(item).casefold(),
        )
    )
    fingerprints: list[InputFingerprint] = []
    for path in ordered:
        stat = path.stat()
        fingerprints.append(
            InputFingerprint(
                path=path,
                size_bytes=stat.st_size,
                mtime_ns=stat.st_mtime_ns,
            )
        )
    return tuple(fingerprints)


def detection_product_id(
    paths: Iterable[Path],
    algorithm: str,
    config: DetectionConfig,
    *,
    location_tolerance_m: float,
) -> str:
    """Return the semantic identity of a detection-sequence product."""
    payload = {
        "kind": "detections",
        "schema_version": 1,
        "pipeline_revision": DETECTION_PIPELINE_REVISION,
        "inputs": [item.as_mapping() for item in fingerprint_inputs(paths)],
        "algorithm": algorithm,
        "detection_config": asdict(config),
        "location_tolerance_m": location_tolerance_m,
    }
    return _digest(payload)


def tracking_product_id(
    detection_id: str,
    config: TrackingConfig,
) -> str:
    """Return the semantic identity of an image-plane tracking product."""
    payload = {
        "kind": "tracks",
        "schema_version": 1,
        "pipeline_revision": TRACKING_PIPELINE_REVISION,
        "detection_product_id": detection_id,
        "tracking_config": asdict(config),
    }
    return _digest(payload)


def multichannel_detection_product_id(
    paths: Iterable[Path],
    config: DetectionConfig,
    multichannel_config: MultiChannelSEPConfig,
    grid_calibration_path: Path,
    location_tolerance_m: float,
    field_mask_path: Path | None = None,
) -> str:
    """Return the semantic identity of independent-channel detections."""
    grid = fingerprint_inputs((grid_calibration_path,))[0]
    field_mask = (
        fingerprint_inputs((field_mask_path,))[0].as_mapping()
        if field_mask_path is not None
        else None
    )
    payload = {
        "kind": "multichannel-detections",
        "schema_version": 1,
        "pipeline_revision": MULTICHANNEL_DETECTION_PIPELINE_REVISION,
        "inputs": [item.as_mapping() for item in fingerprint_inputs(paths)],
        "algorithm": "sep-independent-channels",
        "detection_config": asdict(config),
        "multichannel_config": asdict(multichannel_config),
        "grid_calibration": grid.as_mapping(),
        "field_mask": field_mask,
        "location_tolerance_m": location_tolerance_m,
    }
    return _digest(payload)


def stellar_identification_product_id(
    detection_id: str,
    grid_calibration_path: Path,
    catalog_path: Path,
    config: StellarIdentificationConfig,
) -> str:
    """Return the semantic identity of catalog stellar identifications."""
    grid = fingerprint_inputs((grid_calibration_path,))[0]
    catalog = fingerprint_inputs((catalog_path,))[0]
    payload = {
        "kind": "stellar-identifications",
        "schema_version": 1,
        "pipeline_revision": STELLAR_IDENTIFICATION_PIPELINE_REVISION,
        "detection_product_id": detection_id,
        "grid_calibration": grid.as_mapping(),
        "catalog": catalog.as_mapping(),
        "identification_config": asdict(config),
    }
    return _digest(payload)


def hybrid_fallback_tracking_product_id(
    detection_id: str,
    stellar_identification_id: str,
    grid_calibration_path: Path,
    config: SphericalTrackingConfig,
) -> str:
    """Return the identity of hybrid spherical fallback associations."""
    grid = fingerprint_inputs((grid_calibration_path,))[0]
    payload = {
        "kind": "hybrid-fallback-temporal-tracks",
        "schema_version": 1,
        "pipeline_revision": HYBRID_FALLBACK_TRACKING_PIPELINE_REVISION,
        "detection_product_id": detection_id,
        "stellar_identification_product_id": stellar_identification_id,
        "grid_calibration": grid.as_mapping(),
        "tracking_config": asdict(config),
    }
    return _digest(payload)


def spherical_tracking_product_id(
    detection_id: str,
    grid_calibration_path: Path,
    config: SphericalTrackingConfig,
) -> str:
    """Return the semantic identity of spherical temporal associations."""
    grid = fingerprint_inputs((grid_calibration_path,))[0]
    payload = {
        "kind": "spherical-temporal-tracks",
        "schema_version": 1,
        "pipeline_revision": SPHERICAL_TRACKING_PIPELINE_REVISION,
        "detection_product_id": detection_id,
        "grid_calibration": grid.as_mapping(),
        "tracking_config": asdict(config),
    }
    return _digest(payload)


def stellar_tracking_product_id(
    temporal_tracking_id: str,
    merge_config: StellarTrackMergeConfig,
    sidereal_config: SiderealFitConfig,
) -> str:
    """Return the semantic identity of merged physical stellar tracks."""
    payload = {
        "kind": "stellar-tracks",
        "schema_version": 1,
        "pipeline_revision": STELLAR_TRACKING_PIPELINE_REVISION,
        "temporal_tracking_product_id": temporal_tracking_id,
        "merge_config": asdict(merge_config),
        "sidereal_config": asdict(sidereal_config),
    }
    return _digest(payload)


def sidereal_product_id(
    tracking_id: str,
    grid_calibration_path: Path,
    config: SiderealFitConfig,
) -> str:
    """Return the semantic identity of a Grid-calibrated sidereal solution."""
    calibration = fingerprint_inputs((grid_calibration_path,))[0]
    payload = {
        "kind": "sidereal-rotation",
        "schema_version": 1,
        "pipeline_revision": SIDEREAL_PIPELINE_REVISION,
        "tracking_product_id": tracking_id,
        "grid_calibration": calibration.as_mapping(),
        "fit_config": asdict(config),
    }
    return _digest(payload)


def orientation_product_id(
    sidereal_id: str,
    catalog_path: Path,
    config: OrientationFitConfig,
) -> str:
    """Return the semantic identity of an absolute camera-orientation product."""
    catalog = fingerprint_inputs((catalog_path,))[0]
    payload = {
        "kind": "absolute-orientation",
        "schema_version": 1,
        "pipeline_revision": ORIENTATION_PIPELINE_REVISION,
        "sidereal_product_id": sidereal_id,
        "catalog": catalog.as_mapping(),
        "fit_config": asdict(config),
    }
    return _digest(payload)


def _digest(payload: dict[str, Any]) -> str:
    """Return a stable SHA-256 digest for a JSON-compatible mapping."""
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
