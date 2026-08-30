"""Grid-free stellar identification with a portable stellar camera calibration."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import distance_transform_edt
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

from gonet_astrometry.calibration.stellar_camera import StellarCameraCalibration
from gonet_astrometry.catalogs.base import CatalogStar
from gonet_astrometry.detection.field_mask import FieldMask
from gonet_astrometry.geometry.horizon import catalog_stars_to_enu_rays, enu_to_altaz
from gonet_astrometry.models.frame import ObserverLocation
from gonet_astrometry.tracking.catalog_identification import (
    StellarIdentification,
    StellarIdentificationEpoch,
    StellarIdentificationResult,
)
from gonet_astrometry.tracking.sequence import DetectionSequence

FloatArray = NDArray[np.float64]
CatalogRayProvider = Callable[
    [tuple[CatalogStar, ...], datetime, ObserverLocation],
    tuple[FloatArray, FloatArray],
]


@dataclass(frozen=True, slots=True)
class StellarCameraIdentificationConfig:
    """Settings for direct catalog matching through a stellar calibration."""

    limiting_magnitude: float = 4.5
    min_catalog_altitude_deg: float = 0.0
    match_radius_px: float = 5.0
    minimum_catalog_track_length: int = 2

    def __post_init__(self) -> None:
        values = (
            self.limiting_magnitude,
            self.min_catalog_altitude_deg,
            self.match_radius_px,
        )
        if not all(np.isfinite(value) for value in values):
            raise ValueError("Stellar-camera identification settings must be finite")
        if not -90.0 <= self.min_catalog_altitude_deg < 90.0:
            raise ValueError("min_catalog_altitude_deg must lie in [-90, 90)")
        if self.match_radius_px <= 0.0:
            raise ValueError("match_radius_px must be positive")
        if self.minimum_catalog_track_length < 2:
            raise ValueError("minimum_catalog_track_length must be at least 2")


class StellarCameraSequenceMatcher:
    """Match catalog stars directly to detections with fixed camera geometry."""

    def __init__(self, config: StellarCameraIdentificationConfig | None = None) -> None:
        self.config = config or StellarCameraIdentificationConfig()

    def match(
        self,
        sequence: DetectionSequence,
        calibration: StellarCameraCalibration,
        stars: tuple[CatalogStar, ...],
        *,
        field_mask: FieldMask | None = None,
        field_mask_keep_margin_px: float = 0.0,
        catalog_ray_provider: CatalogRayProvider = catalog_stars_to_enu_rays,
        show_progress: bool = False,
    ) -> StellarIdentificationResult:
        """Return per-frame stellar identities without fitting camera geometry."""
        if not sequence.epochs:
            raise ValueError(
                "Stellar-camera identification requires at least one epoch"
            )
        if sequence.epochs[0].image_shape != calibration.image_shape:
            raise ValueError(
                "Stellar camera calibration sensor shape does not match detections"
            )
        acceptance = _static_acceptance_mask(
            sequence,
            field_mask,
            field_mask_keep_margin_px,
        )
        selected_stars = tuple(
            star
            for star in stars
            if star.magnitude is None
            or star.magnitude <= self.config.limiting_magnitude
        )
        if not selected_stars:
            raise ValueError("No catalog stars satisfy the limiting magnitude")

        epochs: list[StellarIdentificationEpoch] = []
        angular_residuals: list[float] = []
        iterator = tqdm(
            sequence.epochs,
            desc="Matching stars with stellar calibration",
            unit="frame",
            dynamic_ncols=True,
            disable=not show_progress,
        )
        matched_so_far = 0
        for epoch in iterator:
            rays, altitude = catalog_ray_provider(
                selected_stars,
                epoch.exposure_midpoint,
                epoch.location,
            )
            frame = _match_epoch(
                epoch,
                selected_stars,
                np.asarray(rays, dtype=np.float64),
                np.asarray(altitude, dtype=np.float64),
                calibration,
                acceptance,
                self.config,
            )
            epochs.append(frame)
            matched_so_far += frame.matched_star_count
            angular_residuals.extend(
                item.residual_arcmin
                for item in frame.matched_stars
                if item.residual_arcmin is not None
            )
            iterator.set_postfix(matched=matched_so_far, refresh=False)

        if not angular_residuals:
            raise ValueError("Stellar-camera identification produced no matches")
        residual = np.asarray(angular_residuals, dtype=np.float64)
        return StellarIdentificationResult(
            grid_to_enu=np.asarray(calibration.camera_to_enu, dtype=np.float64),
            bootstrap_frame_identifier="stellar-camera-calibration",
            epochs=tuple(epochs),
            fit_pair_count=int(residual.size),
            fit_median_residual_arcmin=float(np.median(residual)),
            fit_p90_residual_arcmin=float(np.percentile(residual, 90.0)),
        )


def _match_epoch(
    epoch,
    stars: tuple[CatalogStar, ...],
    rays: FloatArray,
    altitude: FloatArray,
    calibration: StellarCameraCalibration,
    acceptance: NDArray[np.bool_],
    config: StellarCameraIdentificationConfig,
) -> StellarIdentificationEpoch:
    if rays.shape != (len(stars), 3) or altitude.shape != (len(stars),):
        raise ValueError("Catalog ray provider returned invalid array shapes")
    camera = rays @ np.asarray(calibration.camera_to_enu, dtype=np.float64)
    theta_deg = np.rad2deg(
        np.arctan2(np.hypot(camera[:, 0], camera[:, 1]), camera[:, 2])
    )
    keep = (
        np.isfinite(altitude)
        & (altitude >= config.min_catalog_altitude_deg)
        & np.isfinite(theta_deg)
        & (theta_deg <= calibration.calibrated_theta_max_deg + 1e-9)
    )
    candidate_indices = np.flatnonzero(keep)
    if candidate_indices.size == 0:
        return StellarIdentificationEpoch(epoch.frame_identifier, ())

    candidate_rays = rays[candidate_indices]
    x_pred, y_pred = calibration.enu_to_pixel(candidate_rays, extrapolate=False)
    rows, columns = calibration.image_shape
    inside = (
        np.isfinite(x_pred)
        & np.isfinite(y_pred)
        & (x_pred >= 0.0)
        & (x_pred < columns)
        & (y_pred >= 0.0)
        & (y_pred < rows)
    )
    accepted = np.zeros(len(candidate_indices), dtype=np.bool_)
    if np.any(inside):
        xi = np.clip(np.rint(x_pred[inside]).astype(int), 0, columns - 1)
        yi = np.clip(np.rint(y_pred[inside]).astype(int), 0, rows - 1)
        accepted[np.flatnonzero(inside)] = acceptance[yi, xi]
    retained = inside & accepted
    candidate_indices = candidate_indices[retained]
    candidate_rays = candidate_rays[retained]
    x_pred = np.asarray(x_pred[retained], dtype=np.float64)
    y_pred = np.asarray(y_pred[retained], dtype=np.float64)

    detections = epoch.catalog.detections
    detection_xy = np.asarray(
        [[item.x, item.y] for item in detections], dtype=np.float64
    ).reshape((-1, 2))
    catalog_xy = np.column_stack((x_pred, y_pred))
    cat_index, det_index, pixel_residual = _gated_assignment(
        catalog_xy,
        detection_xy,
        config.match_radius_px,
    )
    assigned: dict[int, tuple[int, float, float]] = {}
    for catalog_index, detection_index, residual_px in zip(
        cat_index, det_index, pixel_residual, strict=True
    ):
        detection = detections[int(detection_index)]
        inferred = calibration.pixel_to_enu(detection.x, detection.y, extrapolate=True)
        reference = candidate_rays[int(catalog_index)]
        dot = float(np.clip(np.dot(inferred, reference), -1.0, 1.0))
        residual_arcmin = float(np.rad2deg(np.arccos(dot)) * 60.0)
        assigned[int(catalog_index)] = (
            int(detection.identifier),
            float(residual_px),
            residual_arcmin,
        )

    entries: list[StellarIdentification] = []
    for local_index, source_index in enumerate(candidate_indices):
        star = stars[int(source_index)]
        azimuth, computed_altitude = enu_to_altaz(candidate_rays[local_index])
        match = assigned.get(local_index)
        entries.append(
            StellarIdentification(
                catalog_identifier=star.identifier,
                catalog_magnitude=star.magnitude,
                azimuth_deg=azimuth,
                altitude_deg=computed_altitude,
                predicted_x=float(x_pred[local_index]),
                predicted_y=float(y_pred[local_index]),
                detection_identifier=None if match is None else match[0],
                residual_px=None if match is None else match[1],
                residual_arcmin=None if match is None else match[2],
                match_kind=None if match is None else "primary",
            )
        )
    return StellarIdentificationEpoch(epoch.frame_identifier, tuple(entries))


def _static_acceptance_mask(
    sequence: DetectionSequence,
    field_mask: FieldMask | None,
    margin_px: float,
) -> NDArray[np.bool_]:
    image_shape = sequence.epochs[0].image_shape
    if field_mask is None:
        return np.ones(image_shape, dtype=np.bool_)
    excluded = np.asarray(field_mask.excluded, dtype=np.bool_)
    if excluded.shape != image_shape:
        raise ValueError("Field mask shape does not match detection sequence")
    allowed = ~excluded
    if margin_px <= 0.0:
        return allowed
    return allowed & (distance_transform_edt(allowed) > margin_px)


def _gated_assignment(
    catalog_xy: FloatArray,
    detection_xy: FloatArray,
    radius_px: float,
) -> tuple[NDArray[np.int64], NDArray[np.int64], FloatArray]:
    if len(catalog_xy) == 0 or len(detection_xy) == 0:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.float64),
        )
    real_cost = np.linalg.norm(
        catalog_xy[:, None, :] - detection_xy[None, :, :], axis=2
    )
    n_catalog, n_detection = real_cost.shape
    n_possible = min(n_catalog, n_detection)
    unmatched_cost = (n_possible + 1.0) * (radius_px + 1.0)
    invalid_cost = 4.0 * unmatched_cost
    size = n_catalog + n_detection
    cost = np.full((size, size), invalid_cost, dtype=np.float64)
    valid = real_cost <= radius_px
    cost[:n_catalog, :n_detection] = np.where(valid, real_cost, invalid_cost)
    cost[:n_catalog, n_detection:] = unmatched_cost
    cost[n_catalog:, :n_detection] = unmatched_cost
    cost[n_catalog:, n_detection:] = 0.0
    rows, columns = linear_sum_assignment(cost)
    keep = (rows < n_catalog) & (columns < n_detection)
    rows = rows[keep]
    columns = columns[keep]
    keep = valid[rows, columns]
    rows = rows[keep]
    columns = columns[keep]
    return (
        np.asarray(rows, dtype=np.int64),
        np.asarray(columns, dtype=np.int64),
        np.asarray(real_cost[rows, columns], dtype=np.float64),
    )
