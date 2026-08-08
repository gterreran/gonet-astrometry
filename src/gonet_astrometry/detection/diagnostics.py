"""Common source measurements and non-destructive diagnostic flags."""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import atan2, ceil, degrees, pi

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import distance_transform_edt

from gonet_astrometry.detection.config import DetectionConfig
from gonet_astrometry.detection.preprocessing import PreparedDetectionImage
from gonet_astrometry.models.detection import (
    Detection,
    DetectionCatalog,
    DetectionDiagnostics,
)


@dataclass(frozen=True, slots=True)
class LocalSourceDiagnostics:
    """Measurements derived uniformly from the prepared detection image."""

    peak_value: float | None
    area_pixels: int | None
    semimajor_sigma_px: float | None
    semiminor_sigma_px: float | None
    orientation_deg: float | None
    ellipticity: float | None
    elongation: float | None


def enrich_detection_catalog(
    catalog: DetectionCatalog,
    prepared: PreparedDetectionImage,
    config: DetectionConfig,
) -> DetectionCatalog:
    """Add common source measurements and diagnostic flags to a catalog.

    Every detector backend is measured against the same locally normalized
    Bayer image. Backend-provided values take precedence, while unavailable
    peak, area, and shape measurements are filled from a local cutout. The pass
    also identifies candidates near geometric or dynamic-mask boundaries and
    attaches descriptive shape flags. No detection is removed.

    Parameters
    ----------
    catalog
        Backend source catalog to enrich.
    prepared
        Shared mask-aware significance image used by the backend.
    config
        Detection configuration controlling source scale and thresholds.

    Returns
    -------
    DetectionCatalog
        Catalog containing enriched immutable detections.
    """
    if not catalog.detections:
        return catalog

    field_distance = distance_transform_edt(prepared.field_mask)
    bright_distance = (
        distance_transform_edt(~prepared.dynamic_mask)
        if np.any(prepared.dynamic_mask)
        else None
    )
    boundary_distance = max(1.0, config.fwhm_px)
    extended_area = max(
        4 * config.min_pixels,
        int(ceil(pi * config.fwhm_px * config.fwhm_px)),
    )
    radius = max(2, int(ceil(2.0 * config.fwhm_px)))

    enriched: list[Detection] = []
    for detection in catalog.detections:
        local = measure_local_source(
            prepared.data,
            prepared.mask,
            detection.x,
            detection.y,
            radius=radius,
            threshold_sigma=config.threshold_sigma,
        )
        diagnostics = _merge_diagnostics(detection.diagnostics, local)
        elongation = detection.elongation
        if elongation is None:
            elongation = local.elongation

        flags = list(detection.flags)
        row = int(round(detection.y))
        column = int(round(detection.x))
        if _inside(prepared.data.shape, row, column):
            if field_distance[row, column] <= boundary_distance:
                flags.append("near-field-edge")
            if (
                bright_distance is not None
                and bright_distance[row, column] <= boundary_distance
            ):
                flags.append("near-bright-mask")

        if diagnostics.backend_flags not in (None, 0):
            flags.append("backend-flagged")
        if (
            diagnostics.area_pixels is not None
            and diagnostics.area_pixels >= extended_area
        ):
            flags.append("extended")
        if elongation is not None and elongation >= 2.0:
            flags.append("elongated")
        if (
            config.bright_mask_sigma is not None
            and diagnostics.peak_value is not None
            and diagnostics.peak_value >= config.bright_mask_sigma
        ):
            flags.append("high-peak")

        enriched.append(
            replace(
                detection,
                elongation=elongation,
                flags=tuple(dict.fromkeys(flags)),
                diagnostics=diagnostics,
            )
        )

    return DetectionCatalog(
        frame_identifier=catalog.frame_identifier,
        detections=tuple(enriched),
        detector_name=catalog.detector_name,
    )


def measure_local_source(
    data: NDArray[np.float64],
    mask: NDArray[np.bool_],
    x: float,
    y: float,
    *,
    radius: int,
    threshold_sigma: float,
) -> LocalSourceDiagnostics:
    """Measure one source around a backend centroid in native coordinates."""
    x_center = int(round(x))
    y_center = int(round(y))
    if not _inside(data.shape, y_center, x_center):
        return LocalSourceDiagnostics(None, None, None, None, None, None, None)

    y_min = max(0, y_center - radius)
    y_max = min(data.shape[0], y_center + radius + 1)
    x_min = max(0, x_center - radius)
    x_max = min(data.shape[1], x_center + radius + 1)

    cutout = data[y_min:y_max, x_min:x_max]
    cutout_mask = mask[y_min:y_max, x_min:x_max]
    valid = ~cutout_mask & np.isfinite(cutout)
    if not np.any(valid):
        return LocalSourceDiagnostics(None, None, None, None, None, None, None)

    peak = float(np.max(cutout[valid]))
    footprint = valid & (cutout >= threshold_sigma)
    area = int(np.count_nonzero(footprint))
    if area == 0:
        footprint = valid & (cutout > 0.0)
        area = int(np.count_nonzero(footprint))
    area_value = area if area > 0 else None

    weights = np.where(footprint, np.clip(cutout, 0.0, None), 0.0)
    total = float(np.sum(weights))
    if total <= 0.0 or area < 2:
        return LocalSourceDiagnostics(peak, area_value, None, None, None, None, None)

    yy, xx = np.indices(cutout.shape, dtype=np.float64)
    x_centroid = float(np.sum(weights * (xx + x_min)) / total)
    y_centroid = float(np.sum(weights * (yy + y_min)) / total)
    dx = xx + x_min - x_centroid
    dy = yy + y_min - y_centroid
    covariance = (
        np.array(
            [
                [np.sum(weights * dx * dx), np.sum(weights * dx * dy)],
                [np.sum(weights * dx * dy), np.sum(weights * dy * dy)],
            ],
            dtype=np.float64,
        )
        / total
    )
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    minor = float(np.sqrt(eigenvalues[0]))
    major = float(np.sqrt(eigenvalues[1]))
    if major <= 0.0 or minor <= 0.0:
        return LocalSourceDiagnostics(peak, area_value, None, None, None, None, None)

    major_vector = eigenvectors[:, 1]
    orientation = degrees(atan2(float(major_vector[1]), float(major_vector[0])))
    elongation = max(1.0, major / minor)
    ellipticity = min(1.0, max(0.0, 1.0 - minor / major))
    return LocalSourceDiagnostics(
        peak,
        area_value,
        major,
        minor,
        orientation,
        ellipticity,
        elongation,
    )


def _merge_diagnostics(
    backend: DetectionDiagnostics,
    local: LocalSourceDiagnostics,
) -> DetectionDiagnostics:
    """Prefer backend diagnostics and fill missing values from local measures."""
    major = backend.semimajor_sigma_px or local.semimajor_sigma_px
    minor = backend.semiminor_sigma_px or local.semiminor_sigma_px
    ellipticity = backend.ellipticity
    if ellipticity is None and major is not None and minor is not None and major > 0:
        ellipticity = min(1.0, max(0.0, 1.0 - minor / major))

    return replace(
        backend,
        peak_value=(
            backend.peak_value if backend.peak_value is not None else local.peak_value
        ),
        area_pixels=(
            backend.area_pixels
            if backend.area_pixels is not None
            else local.area_pixels
        ),
        semimajor_sigma_px=major,
        semiminor_sigma_px=minor,
        orientation_deg=(
            backend.orientation_deg
            if backend.orientation_deg is not None
            else local.orientation_deg
        ),
        ellipticity=ellipticity,
    )


def _inside(shape: tuple[int, ...], row: int, column: int) -> bool:
    """Return whether a row and column lie inside a two-dimensional shape."""
    return 0 <= row < shape[0] and 0 <= column < shape[1]
