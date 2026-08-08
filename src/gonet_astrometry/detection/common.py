"""Shared helpers for converting backend measurements into detections."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import replace

import numpy as np
from numpy.typing import NDArray

from gonet_astrometry.detection.config import DetectionConfig
from gonet_astrometry.models.detection import Detection, DetectionCatalog


def empty_catalog(frame_identifier: str, detector_name: str) -> DetectionCatalog:
    """Return an empty catalog with detector provenance."""
    return DetectionCatalog(
        frame_identifier=frame_identifier,
        detections=(),
        detector_name=detector_name,
    )


def finalize_catalog(
    frame_identifier: str,
    detector_name: str,
    detections: Iterable[Detection],
    *,
    config: DetectionConfig,
) -> DetectionCatalog:
    """Sort detections by signal-to-noise and enforce ``max_sources``."""
    ordered = sorted(
        detections,
        key=lambda detection: detection.signal_to_noise,
        reverse=True,
    )
    if config.max_sources is not None:
        ordered = ordered[: config.max_sources]

    renumbered = tuple(
        replace(detection, identifier=index)
        for index, detection in enumerate(ordered, start=1)
    )
    return DetectionCatalog(
        frame_identifier=frame_identifier,
        detections=renumbered,
        detector_name=detector_name,
    )


def coordinate_uncertainty(fwhm_px: float, signal_to_noise: float) -> float:
    """Return a conservative centroid uncertainty in native pixels."""
    effective_snr = max(abs(signal_to_noise), 1.0)
    return max(0.05, fwhm_px / (2.354820045 * effective_snr))


def elongation_from_axes(major: float, minor: float) -> float | None:
    """Return the major-to-minor axis ratio when both axes are valid."""
    if not math.isfinite(major) or not math.isfinite(minor) or minor <= 0:
        return None
    return max(1.0, major / minor)


def local_source_measurements(
    data: NDArray[np.float64],
    x_peak: int,
    y_peak: int,
    *,
    radius: int,
) -> tuple[float, float, float, float | None]:
    """Measure a positive-weight centroid, flux, peak, and elongation."""
    y_min = max(0, y_peak - radius)
    y_max = min(data.shape[0], y_peak + radius + 1)
    x_min = max(0, x_peak - radius)
    x_max = min(data.shape[1], x_peak + radius + 1)

    cutout = data[y_min:y_max, x_min:x_max]
    weights = np.clip(cutout, 0.0, None)
    total = float(weights.sum())
    peak = float(data[y_peak, x_peak])
    if total <= 0:
        return float(x_peak), float(y_peak), peak, None

    yy, xx = np.indices(cutout.shape, dtype=np.float64)
    x_centroid = float((weights * (xx + x_min)).sum() / total)
    y_centroid = float((weights * (yy + y_min)).sum() / total)

    dx = xx + x_min - x_centroid
    dy = yy + y_min - y_centroid
    covariance = (
        np.array(
            [
                [(weights * dx * dx).sum(), (weights * dx * dy).sum()],
                [(weights * dx * dy).sum(), (weights * dy * dy).sum()],
            ],
            dtype=np.float64,
        )
        / total
    )
    eigenvalues = np.linalg.eigvalsh(covariance)
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    minor, major = np.sqrt(eigenvalues)
    elongation = elongation_from_axes(float(major), float(minor))
    return x_centroid, y_centroid, total, elongation


def scalar(value: object, default: float = 0.0) -> float:
    """Convert a backend scalar or scalar quantity to ``float``."""
    candidate = getattr(value, "value", value)
    try:
        result = float(candidate)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def row_scalar(
    row: object,
    names: tuple[str, ...],
    *,
    default: float = 0.0,
) -> float:
    """Read the first available numeric field from a table-like row."""
    for name in names:
        try:
            value = row[name]  # type: ignore[index]
        except (KeyError, TypeError, ValueError):
            continue
        return scalar(value, default)
    return default


def optional_row_scalar(
    row: object,
    names: tuple[str, ...],
) -> float | None:
    """Read an optional finite numeric field from a table-like row."""
    for name in names:
        try:
            value = row[name]  # type: ignore[index]
        except (KeyError, TypeError, ValueError):
            continue
        candidate = getattr(value, "value", value)
        try:
            result = float(candidate)
        except (TypeError, ValueError):
            continue
        if math.isfinite(result):
            return result
    return None
