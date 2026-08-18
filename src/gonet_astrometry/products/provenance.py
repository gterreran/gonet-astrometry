"""Deterministic provenance identifiers for reusable workflow products."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from gonet_astrometry.detection.config import DetectionConfig
from gonet_astrometry.tracking.config import TrackingConfig

DETECTION_PIPELINE_REVISION = 1
"""Manual cache revision for changes to detection semantics."""

TRACKING_PIPELINE_REVISION = 1
"""Manual cache revision for changes to bootstrap-tracking semantics."""


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


def _digest(payload: dict[str, Any]) -> str:
    """Return a stable SHA-256 digest for a JSON-compatible mapping."""
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
