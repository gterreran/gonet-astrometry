"""Grid-independent camera calibration from catalog-identified stars.

The Grid calibration is intentionally absent from the fitted camera geometry in
this module.  Existing Grid-assisted stellar identifications may be supplied as
bootstrap correspondences, but after the seed fit the catalog is rematched
straight to raw full-sensor detections through a direct stellar camera model.

The production intrinsic model is the compact radial ``poly3`` form selected by
held-out-star model sweeps::

    t = theta / (pi / 2)
    r = c1 * t + c3 * t**3

where ``theta`` is angular distance from the optical axis and ``r`` is full-
sensor pixel radius about ``(cx, cy)``.  A single rigid camera-to-local-ENU
rotation supplies the extrinsic attitude.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import distance_transform_edt
from scipy.optimize import least_squares, linear_sum_assignment
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from gonet_astrometry.catalogs.base import CatalogStar
from gonet_astrometry.detection.field_mask import FieldMask
from gonet_astrometry.geometry.horizon import catalog_stars_to_enu_rays, enu_to_altaz
from gonet_astrometry.models.frame import ObserverLocation
from gonet_astrometry.tracking.catalog_identification import StellarIdentificationResult
from gonet_astrometry.tracking.sequence import DetectionSequence

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
SolarAltitudeProvider = Callable[[DetectionSequence], dict[str, float]]
CatalogRayProvider = Callable[
    [tuple[CatalogStar, ...], datetime, ObserverLocation],
    tuple[FloatArray, FloatArray],
]

_THETA_SCALE = np.pi / 2.0
_COORDINATE_CONVENTION = (
    "native full sensor: x=column increasing right, y=row increasing down; "
    "camera +z=optical axis; camera +x/+y aligned with sensor x/y"
)


@dataclass(frozen=True, slots=True)
class StellarCameraCalibrationConfig:
    """Configuration for direct stellar camera calibration.

    Defaults are the empirically selected production settings from the Adler
    validation sequence.  The Grid-assisted identifications are used only as a
    conservative bootstrap.  Final intrinsic fitting uses dark-time stars at
    geometric altitude >=30 degrees.
    """

    limiting_magnitude: float = 4.5
    seed_min_altitude_deg: float = 25.0
    fit_min_altitude_deg: float = 30.0
    max_solar_altitude_deg: float = -18.0
    max_seed_identification_residual_px: float = 15.0
    min_observations_per_star: int = 5
    trusted_seed_p90_arcmin: float = 20.0
    initial_match_radius_px: float = 8.0
    final_match_radius_px: float = 5.0
    max_refit_residual_px: float = 5.0
    refinement_iterations: int = 2
    star_fold_count: int = 5
    random_seed: int = 1729
    robust_scale_px: float = 3.0
    max_nfev: int = 500
    balance_stars: bool = True

    def __post_init__(self) -> None:
        finite = (
            self.limiting_magnitude,
            self.seed_min_altitude_deg,
            self.fit_min_altitude_deg,
            self.max_solar_altitude_deg,
            self.max_seed_identification_residual_px,
            self.trusted_seed_p90_arcmin,
            self.initial_match_radius_px,
            self.final_match_radius_px,
            self.max_refit_residual_px,
            self.robust_scale_px,
        )
        if not all(np.isfinite(value) for value in finite):
            raise ValueError("Stellar-camera calibration settings must be finite")
        if not 0.0 <= self.seed_min_altitude_deg < 90.0:
            raise ValueError("seed_min_altitude_deg must lie in [0, 90)")
        if not -90.0 <= self.max_solar_altitude_deg <= 90.0:
            raise ValueError("max_solar_altitude_deg must lie in [-90, 90]")
        if not self.seed_min_altitude_deg <= self.fit_min_altitude_deg < 90.0:
            raise ValueError(
                "fit_min_altitude_deg must be >= seed_min_altitude_deg and < 90"
            )
        positive = (
            self.max_seed_identification_residual_px,
            self.trusted_seed_p90_arcmin,
            self.initial_match_radius_px,
            self.final_match_radius_px,
            self.max_refit_residual_px,
            self.robust_scale_px,
        )
        if any(value <= 0.0 for value in positive):
            raise ValueError("Stellar-camera fit radii/scales must be positive")
        if self.min_observations_per_star < 2:
            raise ValueError("min_observations_per_star must be at least 2")
        if self.refinement_iterations < 0:
            raise ValueError("refinement_iterations cannot be negative")
        if self.star_fold_count < 2:
            raise ValueError("star_fold_count must be at least 2")
        if self.max_nfev < 1:
            raise ValueError("max_nfev must be positive")


@dataclass(frozen=True, slots=True)
class StellarCameraCalibration:
    """Portable direct camera geometry fitted from stars.

    ``camera_to_enu`` maps camera-frame column vectors into local ENU.  The
    intrinsic model is radial ``poly3`` in raw full-sensor pixels and contains
    no Grid coordinates or Grid-derived distortion terms.
    """

    image_shape: tuple[int, int]
    camera_to_enu: FloatArray
    center_x_px: float
    center_y_px: float
    radial_c1_px: float
    radial_c3_px: float
    calibrated_theta_max_deg: float
    coordinate_convention: str = _COORDINATE_CONVENTION

    def __post_init__(self) -> None:
        rows, columns = self.image_shape
        if rows <= 0 or columns <= 0:
            raise ValueError("image_shape must contain positive dimensions")
        matrix = np.asarray(self.camera_to_enu, dtype=np.float64)
        if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
            raise ValueError("camera_to_enu must be a finite 3x3 matrix")
        if not np.allclose(matrix @ matrix.T, np.eye(3), atol=1e-7):
            raise ValueError("camera_to_enu must be orthonormal")
        if not np.isclose(np.linalg.det(matrix), 1.0, atol=1e-7):
            raise ValueError("camera_to_enu must be a proper rotation")
        values = (
            self.center_x_px,
            self.center_y_px,
            self.radial_c1_px,
            self.radial_c3_px,
            self.calibrated_theta_max_deg,
        )
        if not all(np.isfinite(value) for value in values):
            raise ValueError("Stellar-camera intrinsic parameters must be finite")
        if self.radial_c1_px <= 0.0:
            raise ValueError("radial_c1_px must be positive")
        if not 0.0 < self.calibrated_theta_max_deg < 180.0:
            raise ValueError("calibrated_theta_max_deg must lie in (0, 180)")
        t_max = np.deg2rad(self.calibrated_theta_max_deg) / _THETA_SCALE
        derivative = self.radial_c1_px + 3.0 * self.radial_c3_px * t_max**2
        radius = self.radial_c1_px * t_max + self.radial_c3_px * t_max**3
        if derivative <= 0.0 or radius <= 0.0:
            raise ValueError(
                "Radial poly3 mapping must remain positive and monotonic over "
                "the stellar-calibrated angular range"
            )

    @property
    def optical_axis_azimuth_deg(self) -> float:
        """Return fitted optical-axis azimuth eastward from north."""
        azimuth, _ = enu_to_altaz(np.asarray(self.camera_to_enu)[:, 2])
        return azimuth

    @property
    def optical_axis_altitude_deg(self) -> float:
        """Return fitted optical-axis altitude above the horizon."""
        _, altitude = enu_to_altaz(np.asarray(self.camera_to_enu)[:, 2])
        return altitude

    def enu_to_pixel(
        self,
        rays_enu: FloatArray,
        *,
        extrapolate: bool = False,
    ) -> tuple[FloatArray, FloatArray]:
        """Project local ENU unit directions to raw full-sensor pixels."""
        rays = np.asarray(rays_enu, dtype=np.float64)
        if rays.shape[-1:] != (3,):
            raise ValueError("rays_enu must end with a three-component vector")
        flat = rays.reshape((-1, 3))
        norm = np.linalg.norm(flat, axis=1)
        if np.any(~np.isfinite(norm)) or np.any(norm <= 0.0):
            raise ValueError("rays_enu must contain finite non-zero vectors")
        unit = flat / norm[:, None]
        camera = unit @ np.asarray(self.camera_to_enu, dtype=np.float64)
        transverse = np.hypot(camera[:, 0], camera[:, 1])
        theta = np.arctan2(transverse, camera[:, 2])
        self._validate_theta(theta, extrapolate=extrapolate)
        t = theta / _THETA_SCALE
        radius = self.radial_c1_px * t + self.radial_c3_px * t**3
        scale = np.ones_like(theta)
        nonzero = transverse > 1e-15
        scale[nonzero] = radius[nonzero] / transverse[nonzero]
        x = self.center_x_px + camera[:, 0] * scale
        y = self.center_y_px + camera[:, 1] * scale
        shape = rays.shape[:-1]
        return x.reshape(shape), y.reshape(shape)

    def pixel_to_enu(
        self,
        x: FloatArray | float,
        y: FloatArray | float,
        *,
        extrapolate: bool = False,
    ) -> FloatArray:
        """Invert raw full-sensor pixels to local ENU unit directions."""
        x_values, y_values = np.broadcast_arrays(
            np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
        )
        if np.any(~np.isfinite(x_values)) or np.any(~np.isfinite(y_values)):
            raise ValueError("Pixel coordinates must be finite")
        dx = x_values.ravel() - self.center_x_px
        dy = y_values.ravel() - self.center_y_px
        radius = np.hypot(dx, dy)
        t = self._invert_radius(radius, extrapolate=extrapolate)
        theta = t * _THETA_SCALE
        self._validate_theta(theta, extrapolate=extrapolate)
        phi = np.arctan2(dy, dx)
        sin_theta = np.sin(theta)
        camera = np.stack(
            (
                sin_theta * np.cos(phi),
                sin_theta * np.sin(phi),
                np.cos(theta),
            ),
            axis=1,
        )
        enu = camera @ np.asarray(self.camera_to_enu, dtype=np.float64).T
        return enu.reshape(x_values.shape + (3,))

    def _invert_radius(self, radius: FloatArray, *, extrapolate: bool) -> FloatArray:
        parsed = np.asarray(radius, dtype=np.float64)
        t_limit = self._radial_t_limit(extrapolate=extrapolate)
        radius_limit = self.radial_c1_px * t_limit + self.radial_c3_px * t_limit**3
        tolerance = 1e-9 * max(1.0, abs(radius_limit))
        if np.any(parsed > radius_limit + tolerance):
            scope = "invertible" if extrapolate else "stellar-calibrated"
            raise ValueError(f"Pixel radius lies outside the {scope} angular range")
        denominator = max(abs(self.radial_c1_px), 1.0)
        t = np.clip(parsed / denominator, 0.0, t_limit)
        for _ in range(20):
            value = self.radial_c1_px * t + self.radial_c3_px * t**3 - parsed
            derivative = self.radial_c1_px + 3.0 * self.radial_c3_px * t**2
            step = value / derivative
            t = np.clip(t - step, 0.0, t_limit)
        remaining = np.abs(self.radial_c1_px * t + self.radial_c3_px * t**3 - parsed)
        if np.any(remaining > max(1e-8, tolerance)):
            raise ValueError("Could not invert stellar-camera radial mapping")
        return t

    def _radial_t_limit(self, *, extrapolate: bool) -> float:
        calibrated = np.deg2rad(self.calibrated_theta_max_deg) / _THETA_SCALE
        if not extrapolate:
            return float(calibrated)
        if self.radial_c3_px >= 0.0:
            return 2.0
        turning = math.sqrt(-self.radial_c1_px / (3.0 * self.radial_c3_px))
        return float(min(2.0, turning * (1.0 - 1e-12)))

    def _validate_theta(self, theta: FloatArray, *, extrapolate: bool) -> None:
        limit = self._radial_t_limit(extrapolate=extrapolate) * _THETA_SCALE
        if np.any(theta > limit + 1e-12):
            scope = "invertible" if extrapolate else "stellar-calibrated"
            raise ValueError(
                f"Direction lies outside the {scope} angular range; "
                "extrapolation is never allowed beyond a non-monotonic radial "
                "mapping"
            )


@dataclass(frozen=True, slots=True)
class StellarCameraCalibrationFit:
    """Direct stellar-camera fit and validation summary."""

    calibration: StellarCameraCalibration
    seed_measurement_count: int
    seed_star_count: int
    trusted_seed_star_count: int
    rejected_seed_star_ids: tuple[str, ...]
    direct_match_count: int
    direct_star_count: int
    direct_match_median_px: float
    direct_match_p90_px: float
    calibration_measurement_count: int
    calibration_star_count: int
    cv_median_arcmin: float
    cv_p90_arcmin: float
    cv_p90_px: float


@dataclass(frozen=True, slots=True)
class _Dataset:
    x: FloatArray
    y: FloatArray
    enu: FloatArray
    altitude_deg: FloatArray
    star_id: NDArray[np.str_]
    frame_id: NDArray[np.str_]
    image_shape: tuple[int, int]
    initial_camera_to_enu: FloatArray
    residual_px: FloatArray

    @property
    def size(self) -> int:
        return int(self.x.size)

    @property
    def unique_star_ids(self) -> NDArray[np.str_]:
        return np.unique(self.star_id)

    def subset(self, mask: BoolArray) -> _Dataset:
        return _Dataset(
            x=self.x[mask],
            y=self.y[mask],
            enu=self.enu[mask],
            altitude_deg=self.altitude_deg[mask],
            star_id=self.star_id[mask],
            frame_id=self.frame_id[mask],
            image_shape=self.image_shape,
            initial_camera_to_enu=self.initial_camera_to_enu,
            residual_px=self.residual_px[mask],
        )


@dataclass(frozen=True, slots=True)
class _Poly3Fit:
    camera_to_enu: FloatArray
    center_x_px: float
    center_y_px: float
    radial_c1_px: float
    radial_c3_px: float


@dataclass(frozen=True, slots=True)
class _FrameCatalog:
    stars: tuple[CatalogStar, ...]
    rays_enu: FloatArray
    altitude_deg: FloatArray


@dataclass(frozen=True, slots=True)
class _FrameMatch:
    frame_identifier: str
    stars: tuple[CatalogStar, ...]
    rays_enu: FloatArray
    altitude_deg: FloatArray
    detection_identifiers: NDArray[np.int64]
    residual_px: FloatArray


class StellarCameraCalibrator:
    """Fit a direct radial camera model from stellar correspondences."""

    def __init__(
        self,
        config: StellarCameraCalibrationConfig | None = None,
        *,
        solar_altitude_provider: SolarAltitudeProvider | None = None,
        catalog_ray_provider: CatalogRayProvider | None = None,
    ) -> None:
        self.config = config or StellarCameraCalibrationConfig()
        self._solar_altitude_provider = (
            solar_altitude_provider or _geometric_solar_altitudes
        )
        self._catalog_ray_provider = catalog_ray_provider or catalog_stars_to_enu_rays

    def fit(
        self,
        sequence: DetectionSequence,
        bootstrap_identifications: StellarIdentificationResult,
        catalog_stars: tuple[CatalogStar, ...],
        *,
        field_mask: FieldMask | None = None,
        field_mask_keep_margin_px: float = 0.0,
        show_progress: bool = False,
    ) -> StellarCameraCalibrationFit:
        """Fit and validate a Grid-independent direct stellar calibration."""
        if not sequence.epochs:
            raise ValueError("Stellar camera calibration requires at least one epoch")
        if field_mask_keep_margin_px < 0.0:
            raise ValueError("field_mask_keep_margin_px cannot be negative")

        seed = _seed_dataset(sequence, bootstrap_identifications, self.config)
        if seed.unique_star_ids.size < self.config.star_fold_count:
            raise ValueError("Too few seed stars for grouped-star calibration audit")
        trusted, rejected = _trusted_seed_stars(seed, self.config)
        trusted_mask = np.asarray(
            [str(item) in trusted for item in seed.star_id], dtype=np.bool_
        )
        trusted_seed = seed.subset(trusted_mask)
        initial_fit = _fit_poly3(trusted_seed, self.config)

        selected_catalog = tuple(
            star
            for star in catalog_stars
            if star.magnitude is not None
            and star.magnitude <= self.config.limiting_magnitude
        )
        if not selected_catalog:
            raise ValueError("No catalog stars satisfy the calibration magnitude limit")
        frame_catalog = _project_catalog(
            sequence,
            selected_catalog,
            self.config.seed_min_altitude_deg,
            catalog_ray_provider=self._catalog_ray_provider,
            show_progress=show_progress,
        )
        acceptance = _static_acceptance_mask(
            sequence,
            field_mask,
            field_mask_keep_margin_px,
        )
        fit = initial_fit
        matches: tuple[_FrameMatch, ...] = ()
        for iteration in range(self.config.refinement_iterations):
            radius = (
                self.config.initial_match_radius_px
                if iteration == 0
                else self.config.final_match_radius_px
            )
            matches = _direct_match(
                sequence,
                frame_catalog,
                fit,
                acceptance,
                radius_px=radius,
                show_progress=show_progress,
                description=f"Direct stellar rematch {iteration + 1}",
            )
            refit = _dataset_from_direct_matches(
                sequence,
                matches,
                fit,
                self.config,
                min_altitude_deg=self.config.seed_min_altitude_deg,
                solar_altitude_by_frame=None,
            )
            fit = _fit_poly3(refit, self.config)

        matches = _direct_match(
            sequence,
            frame_catalog,
            fit,
            acceptance,
            radius_px=self.config.final_match_radius_px,
            show_progress=show_progress,
            description="Final direct stellar rematch",
        )

        solar_altitude = self._solar_altitude_provider(sequence)
        calibration_data = _dataset_from_direct_matches(
            sequence,
            matches,
            fit,
            self.config,
            min_altitude_deg=self.config.fit_min_altitude_deg,
            solar_altitude_by_frame=solar_altitude,
        )
        if calibration_data.unique_star_ids.size < self.config.star_fold_count:
            raise ValueError(
                "Too few dark/high-altitude stars survive stellar calibration cuts"
            )

        cv_median, cv_p90, cv_p90_px = _grouped_cv_metrics(
            calibration_data, self.config
        )
        final_fit = _fit_poly3(calibration_data, self.config)
        theta = _camera_theta(calibration_data.enu, final_fit.camera_to_enu)
        theta_max_deg = float(np.rad2deg(np.max(theta)))
        calibration = StellarCameraCalibration(
            image_shape=sequence.epochs[0].image_shape,
            camera_to_enu=final_fit.camera_to_enu,
            center_x_px=final_fit.center_x_px,
            center_y_px=final_fit.center_y_px,
            radial_c1_px=final_fit.radial_c1_px,
            radial_c3_px=final_fit.radial_c3_px,
            calibrated_theta_max_deg=theta_max_deg,
        )

        matched_residuals = np.concatenate(
            [frame.residual_px[np.isfinite(frame.residual_px)] for frame in matches]
        )
        direct_star_ids = {
            star.identifier
            for frame in matches
            for star, detection_id in zip(
                frame.stars, frame.detection_identifiers, strict=True
            )
            if detection_id >= 0
        }
        return StellarCameraCalibrationFit(
            calibration=calibration,
            seed_measurement_count=seed.size,
            seed_star_count=int(seed.unique_star_ids.size),
            trusted_seed_star_count=len(trusted),
            rejected_seed_star_ids=tuple(sorted(rejected)),
            direct_match_count=int(matched_residuals.size),
            direct_star_count=len(direct_star_ids),
            direct_match_median_px=float(np.median(matched_residuals)),
            direct_match_p90_px=float(np.percentile(matched_residuals, 90.0)),
            calibration_measurement_count=calibration_data.size,
            calibration_star_count=int(calibration_data.unique_star_ids.size),
            cv_median_arcmin=cv_median,
            cv_p90_arcmin=cv_p90,
            cv_p90_px=cv_p90_px,
        )


def _seed_dataset(
    sequence: DetectionSequence,
    identifications: StellarIdentificationResult,
    config: StellarCameraCalibrationConfig,
) -> _Dataset:
    by_frame = {epoch.frame_identifier: epoch for epoch in identifications.epochs}
    records: list[tuple[float, float, FloatArray, float, str, str, float]] = []
    for epoch in sequence.epochs:
        identification_epoch = by_frame.get(epoch.frame_identifier)
        if identification_epoch is None:
            raise ValueError(
                "Missing bootstrap stellar identifications for "
                f"{epoch.frame_identifier}"
            )
        detections = {item.identifier: item for item in epoch.catalog.detections}
        for star in identification_epoch.matched_stars:
            if star.match_kind != "primary":
                continue
            if star.residual_px is None or (
                star.residual_px > config.max_seed_identification_residual_px
            ):
                continue
            if star.altitude_deg < config.seed_min_altitude_deg:
                continue
            if star.detection_identifier is None:
                continue
            detection = detections.get(star.detection_identifier)
            if detection is None:
                continue
            ray = _altaz_to_enu(star.azimuth_deg, star.altitude_deg)
            records.append(
                (
                    detection.x,
                    detection.y,
                    ray,
                    star.altitude_deg,
                    star.catalog_identifier,
                    epoch.frame_identifier,
                    float(star.residual_px),
                )
            )
    return _records_to_dataset(
        records,
        sequence.epochs[0].image_shape,
        identifications.grid_to_enu,
        config.min_observations_per_star,
    )


def _records_to_dataset(
    records: list[tuple[float, float, FloatArray, float, str, str, float]],
    image_shape: tuple[int, int],
    initial_camera_to_enu: FloatArray,
    minimum_per_star: int,
) -> _Dataset:
    if not records:
        raise ValueError("No stellar measurements survive calibration selection")
    counts = Counter(str(item[4]) for item in records)
    retained = [item for item in records if counts[str(item[4])] >= minimum_per_star]
    if not retained:
        raise ValueError("No stellar tracks survive minimum observation count")
    return _Dataset(
        x=np.asarray([item[0] for item in retained], dtype=np.float64),
        y=np.asarray([item[1] for item in retained], dtype=np.float64),
        enu=np.asarray([item[2] for item in retained], dtype=np.float64),
        altitude_deg=np.asarray([item[3] for item in retained], dtype=np.float64),
        star_id=np.asarray([item[4] for item in retained], dtype=np.str_),
        frame_id=np.asarray([item[5] for item in retained], dtype=np.str_),
        image_shape=image_shape,
        initial_camera_to_enu=np.asarray(initial_camera_to_enu, dtype=np.float64),
        residual_px=np.asarray([item[6] for item in retained], dtype=np.float64),
    )


def _trusted_seed_stars(
    dataset: _Dataset,
    config: StellarCameraCalibrationConfig,
) -> tuple[frozenset[str], frozenset[str]]:
    folds = _star_folds(dataset.star_id, config.star_fold_count, config.random_seed)
    residual_by_star: dict[str, list[float]] = {
        str(item): [] for item in dataset.unique_star_ids
    }
    for validation in folds:
        fit = _fit_poly3(dataset.subset(~validation), config)
        sample = dataset.subset(validation)
        calibration = _calibration_from_fit(sample.image_shape, fit, 179.0)
        inferred = calibration.pixel_to_enu(sample.x, sample.y, extrapolate=True)
        angle = _angular_separation_arcmin(inferred, sample.enu)
        for star_id, residual in zip(sample.star_id, angle, strict=True):
            residual_by_star[str(star_id)].append(float(residual))
    trusted: set[str] = set()
    rejected: set[str] = set()
    for identifier, values in residual_by_star.items():
        p90 = float(np.percentile(np.asarray(values), 90.0))
        (trusted if p90 <= config.trusted_seed_p90_arcmin else rejected).add(identifier)
    return frozenset(trusted), frozenset(rejected)


def _fit_poly3(dataset: _Dataset, config: StellarCameraCalibrationConfig) -> _Poly3Fit:
    if dataset.size < 10 or dataset.unique_star_ids.size < 3:
        raise ValueError("Too few stellar measurements to fit poly3 camera model")
    initial_attitude = _sensor_aligned_attitude(dataset, dataset.initial_camera_to_enu)
    cx0 = (dataset.image_shape[1] - 1.0) / 2.0
    cy0 = (dataset.image_shape[0] - 1.0) / 2.0
    a, b = _camera_angle_plane(dataset.enu, initial_attitude)
    theta = np.hypot(a, b)
    observed_radius = np.hypot(dataset.x - cx0, dataset.y - cy0)
    t = theta / _THETA_SCALE
    design = np.column_stack((t, t**3))
    valid = np.all(np.isfinite(design), axis=1) & (theta > np.deg2rad(1.0))
    coefficients, *_ = np.linalg.lstsq(
        design[valid], observed_radius[valid], rcond=None
    )
    c1 = max(float(coefficients[0]), 0.05 * max(dataset.image_shape))
    c3 = float(coefficients[1])

    p0 = np.asarray([0.0, 0.0, 0.0, cx0, cy0, c1, c3], dtype=np.float64)
    rows, columns = dataset.image_shape
    scale = float(max(rows, columns))
    max_delta = np.deg2rad(10.0)
    lower = np.asarray(
        [
            -max_delta,
            -max_delta,
            -max_delta,
            -0.2 * columns,
            -0.2 * rows,
            0.05 * scale,
            -5.0 * scale,
        ],
        dtype=np.float64,
    )
    upper = np.asarray(
        [
            max_delta,
            max_delta,
            max_delta,
            1.2 * columns,
            1.2 * rows,
            5.0 * scale,
            5.0 * scale,
        ],
        dtype=np.float64,
    )
    weights = _star_weights(dataset.star_id) if config.balance_stars else 1.0

    def residuals(params: FloatArray) -> FloatArray:
        attitude = _apply_attitude_delta(initial_attitude, params[:3])
        pred_x, pred_y = _project_poly3(
            dataset.enu,
            attitude,
            float(params[3]),
            float(params[4]),
            float(params[5]),
            float(params[6]),
        )
        return np.concatenate(
            ((pred_x - dataset.x) * weights, (pred_y - dataset.y) * weights)
        )

    fitted = least_squares(
        residuals,
        p0,
        bounds=(lower, upper),
        loss="soft_l1",
        f_scale=config.robust_scale_px,
        x_scale="jac",
        max_nfev=config.max_nfev,
    )
    if not fitted.success and not np.isfinite(fitted.cost):
        raise RuntimeError(f"Stellar camera fit failed: {fitted.message}")
    attitude = _apply_attitude_delta(initial_attitude, fitted.x[:3])
    return _Poly3Fit(
        camera_to_enu=attitude,
        center_x_px=float(fitted.x[3]),
        center_y_px=float(fitted.x[4]),
        radial_c1_px=float(fitted.x[5]),
        radial_c3_px=float(fitted.x[6]),
    )


def _sensor_aligned_attitude(dataset: _Dataset, base: FloatArray) -> FloatArray:
    a, b = _camera_angle_plane(dataset.enu, base)
    design = np.column_stack((np.ones(dataset.size), a, b))
    target = np.column_stack((dataset.x, dataset.y))
    coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
    linear = np.asarray(coefficients[1:, :], dtype=np.float64)
    u, _, vt = np.linalg.svd(linear)
    plane_transform = np.asarray(u @ vt, dtype=np.float64)
    if np.linalg.det(plane_transform) < 0.0:
        # A reflected camera frame is incompatible with a proper 3-D attitude.
        # Flip the least-constrained SVD axis so the bootstrap remains a rotation.
        u[:, -1] *= -1.0
        plane_transform = np.asarray(u @ vt, dtype=np.float64)
    aligned_basis = np.eye(3, dtype=np.float64)
    aligned_basis[:2, :2] = plane_transform
    return np.asarray(base @ aligned_basis, dtype=np.float64)


def _camera_angle_plane(
    enu: FloatArray, camera_to_enu: FloatArray
) -> tuple[FloatArray, FloatArray]:
    camera = np.asarray(enu, dtype=np.float64) @ np.asarray(
        camera_to_enu, dtype=np.float64
    )
    transverse = np.hypot(camera[:, 0], camera[:, 1])
    theta = np.arctan2(transverse, camera[:, 2])
    scale = np.where(transverse > 1e-15, theta / transverse, 1.0)
    return camera[:, 0] * scale, camera[:, 1] * scale


def _camera_theta(enu: FloatArray, camera_to_enu: FloatArray) -> FloatArray:
    camera = np.asarray(enu, dtype=np.float64) @ np.asarray(
        camera_to_enu, dtype=np.float64
    )
    return np.arctan2(np.hypot(camera[:, 0], camera[:, 1]), camera[:, 2])


def _project_poly3(
    enu: FloatArray,
    camera_to_enu: FloatArray,
    cx: float,
    cy: float,
    c1: float,
    c3: float,
) -> tuple[FloatArray, FloatArray]:
    a, b = _camera_angle_plane(enu, camera_to_enu)
    theta = np.hypot(a, b)
    t = theta / _THETA_SCALE
    radius = c1 * t + c3 * t**3
    scale = np.ones_like(theta)
    nonzero = theta > 1e-15
    scale[nonzero] = radius[nonzero] / theta[nonzero]
    return cx + a * scale, cy + b * scale


def _apply_attitude_delta(initial: FloatArray, rotvec: FloatArray) -> FloatArray:
    delta = Rotation.from_rotvec(np.asarray(rotvec, dtype=np.float64)).as_matrix()
    return np.asarray(delta @ initial, dtype=np.float64)


def _star_weights(star_id: NDArray[np.str_]) -> FloatArray:
    _, inverse, counts = np.unique(star_id, return_inverse=True, return_counts=True)
    return 1.0 / np.sqrt(counts[inverse].astype(np.float64))


def _star_folds(
    star_id: NDArray[np.str_], fold_count: int, seed: int
) -> tuple[BoolArray, ...]:
    unique = np.unique(star_id)
    if unique.size < fold_count:
        raise ValueError("Not enough unique stars for grouped folds")
    shuffled = unique.copy()
    np.random.default_rng(seed).shuffle(shuffled)
    assignment = {
        str(identifier): index % fold_count for index, identifier in enumerate(shuffled)
    }
    return tuple(
        np.asarray(
            [assignment[str(identifier)] == fold for identifier in star_id],
            dtype=np.bool_,
        )
        for fold in range(fold_count)
    )


def _project_catalog(
    sequence: DetectionSequence,
    stars: tuple[CatalogStar, ...],
    min_altitude_deg: float,
    *,
    catalog_ray_provider: CatalogRayProvider,
    show_progress: bool,
) -> tuple[_FrameCatalog, ...]:
    result: list[_FrameCatalog] = []
    iterator = tqdm(
        sequence.epochs,
        desc="Projecting direct stellar catalog",
        unit="frame",
        dynamic_ncols=True,
        disable=not show_progress,
    )
    for epoch in iterator:
        rays, altitude = catalog_ray_provider(
            stars, epoch.exposure_midpoint, epoch.location
        )
        keep = np.isfinite(altitude) & (altitude >= min_altitude_deg)
        kept_indices = np.flatnonzero(keep)
        result.append(
            _FrameCatalog(
                stars=tuple(stars[int(index)] for index in kept_indices),
                rays_enu=np.asarray(rays[keep], dtype=np.float64),
                altitude_deg=np.asarray(altitude[keep], dtype=np.float64),
            )
        )
    return tuple(result)


def _static_acceptance_mask(
    sequence: DetectionSequence,
    field_mask: FieldMask | None,
    margin_px: float,
) -> BoolArray:
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


def _direct_match(
    sequence: DetectionSequence,
    frame_catalog: tuple[_FrameCatalog, ...],
    fit: _Poly3Fit,
    acceptance: BoolArray,
    *,
    radius_px: float,
    show_progress: bool,
    description: str,
) -> tuple[_FrameMatch, ...]:
    rows, columns = sequence.epochs[0].image_shape
    frames: list[_FrameMatch] = []
    iterator = tqdm(
        zip(sequence.epochs, frame_catalog, strict=True),
        total=len(sequence.epochs),
        desc=description,
        unit="frame",
        dynamic_ncols=True,
        disable=not show_progress,
    )
    for epoch, catalog in iterator:
        x_pred, y_pred = _project_poly3(
            catalog.rays_enu,
            fit.camera_to_enu,
            fit.center_x_px,
            fit.center_y_px,
            fit.radial_c1_px,
            fit.radial_c3_px,
        )
        inside = (
            np.isfinite(x_pred)
            & np.isfinite(y_pred)
            & (x_pred >= 0.0)
            & (x_pred < columns)
            & (y_pred >= 0.0)
            & (y_pred < rows)
        )
        accepted = np.zeros(len(x_pred), dtype=np.bool_)
        if np.any(inside):
            xi = np.clip(np.rint(x_pred[inside]).astype(int), 0, columns - 1)
            yi = np.clip(np.rint(y_pred[inside]).astype(int), 0, rows - 1)
            accepted[np.flatnonzero(inside)] = acceptance[yi, xi]
        keep = inside & accepted
        kept_stars = tuple(
            star for star, selected in zip(catalog.stars, keep, strict=True) if selected
        )
        kept_rays = np.asarray(catalog.rays_enu[keep], dtype=np.float64)
        kept_altitude = np.asarray(catalog.altitude_deg[keep], dtype=np.float64)
        catalog_xy = np.column_stack((x_pred[keep], y_pred[keep]))
        detections = epoch.catalog.detections
        detection_xy = np.asarray(
            [[item.x, item.y] for item in detections], dtype=np.float64
        ).reshape((-1, 2))
        cat_index, det_index, residual = _gated_assignment(
            catalog_xy, detection_xy, radius_px
        )
        detection_ids = np.full(len(kept_stars), -1, dtype=np.int64)
        residuals = np.full(len(kept_stars), np.nan, dtype=np.float64)
        if len(cat_index):
            detection_ids[cat_index] = np.asarray(
                [detections[int(index)].identifier for index in det_index],
                dtype=np.int64,
            )
            residuals[cat_index] = residual
        frames.append(
            _FrameMatch(
                frame_identifier=epoch.frame_identifier,
                stars=kept_stars,
                rays_enu=kept_rays,
                altitude_deg=kept_altitude,
                detection_identifiers=detection_ids,
                residual_px=residuals,
            )
        )
    return tuple(frames)


def _gated_assignment(
    catalog_xy: FloatArray, detection_xy: FloatArray, radius_px: float
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


def _dataset_from_direct_matches(
    sequence: DetectionSequence,
    matches: tuple[_FrameMatch, ...],
    fit: _Poly3Fit,
    config: StellarCameraCalibrationConfig,
    *,
    min_altitude_deg: float,
    solar_altitude_by_frame: dict[str, float] | None,
) -> _Dataset:
    records: list[tuple[float, float, FloatArray, float, str, str, float]] = []
    for epoch, frame in zip(sequence.epochs, matches, strict=True):
        detections = {item.identifier: item for item in epoch.catalog.detections}
        if solar_altitude_by_frame is not None:
            sun_altitude = solar_altitude_by_frame.get(epoch.frame_identifier)
            if sun_altitude is None:
                raise ValueError(f"Missing solar altitude for {epoch.frame_identifier}")
            if sun_altitude > config.max_solar_altitude_deg:
                continue
        for star, ray, altitude, detection_id, residual in zip(
            frame.stars,
            frame.rays_enu,
            frame.altitude_deg,
            frame.detection_identifiers,
            frame.residual_px,
            strict=True,
        ):
            if detection_id < 0 or altitude < min_altitude_deg:
                continue
            if not np.isfinite(residual) or residual > config.max_refit_residual_px:
                continue
            detection = detections.get(int(detection_id))
            if detection is None:
                continue
            records.append(
                (
                    detection.x,
                    detection.y,
                    np.asarray(ray, dtype=np.float64),
                    float(altitude),
                    star.identifier,
                    epoch.frame_identifier,
                    float(residual),
                )
            )
    return _records_to_dataset(
        records,
        sequence.epochs[0].image_shape,
        fit.camera_to_enu,
        config.min_observations_per_star,
    )


def _grouped_cv_metrics(
    dataset: _Dataset, config: StellarCameraCalibrationConfig
) -> tuple[float, float, float]:
    angular = np.full(dataset.size, np.nan, dtype=np.float64)
    pixel = np.full(dataset.size, np.nan, dtype=np.float64)
    folds = _star_folds(dataset.star_id, config.star_fold_count, config.random_seed)
    for validation in folds:
        fit = _fit_poly3(dataset.subset(~validation), config)
        sample = dataset.subset(validation)
        x_pred, y_pred = _project_poly3(
            sample.enu,
            fit.camera_to_enu,
            fit.center_x_px,
            fit.center_y_px,
            fit.radial_c1_px,
            fit.radial_c3_px,
        )
        indices = np.flatnonzero(validation)
        pixel[indices] = np.hypot(x_pred - sample.x, y_pred - sample.y)
        calibration = _calibration_from_fit(sample.image_shape, fit, 179.0)
        inferred = calibration.pixel_to_enu(sample.x, sample.y, extrapolate=True)
        angular[indices] = _angular_separation_arcmin(inferred, sample.enu)
    return (
        float(np.nanmedian(angular)),
        float(np.nanpercentile(angular, 90.0)),
        float(np.nanpercentile(pixel, 90.0)),
    )


def _calibration_from_fit(
    image_shape: tuple[int, int], fit: _Poly3Fit, theta_max_deg: float
) -> StellarCameraCalibration:
    return StellarCameraCalibration(
        image_shape=image_shape,
        camera_to_enu=fit.camera_to_enu,
        center_x_px=fit.center_x_px,
        center_y_px=fit.center_y_px,
        radial_c1_px=fit.radial_c1_px,
        radial_c3_px=fit.radial_c3_px,
        calibrated_theta_max_deg=theta_max_deg,
    )


def _angular_separation_arcmin(first: FloatArray, second: FloatArray) -> FloatArray:
    a = np.asarray(first, dtype=np.float64)
    b = np.asarray(second, dtype=np.float64)
    dots = np.sum(a * b, axis=-1)
    return np.rad2deg(np.arccos(np.clip(dots, -1.0, 1.0))) * 60.0


def _altaz_to_enu(azimuth_deg: float, altitude_deg: float) -> FloatArray:
    azimuth = np.deg2rad(azimuth_deg)
    altitude = np.deg2rad(altitude_deg)
    cosine = np.cos(altitude)
    return np.asarray(
        [cosine * np.sin(azimuth), cosine * np.cos(azimuth), np.sin(altitude)],
        dtype=np.float64,
    )


def _geometric_solar_altitudes(sequence: DetectionSequence) -> dict[str, float]:
    """Return geometric solar altitude for each exposure midpoint."""
    from astropy import units as u  # type: ignore
    from astropy.coordinates import AltAz, EarthLocation, get_sun  # type: ignore
    from astropy.time import Time  # type: ignore

    epochs = sequence.epochs
    latitude = float(np.median([epoch.location.latitude_deg for epoch in epochs]))
    longitude = float(np.median([epoch.location.longitude_deg for epoch in epochs]))
    elevation = float(np.median([epoch.location.elevation_m for epoch in epochs]))
    location = EarthLocation.from_geodetic(
        lon=longitude * u.deg,
        lat=latitude * u.deg,
        height=elevation * u.m,
    )
    times = Time([epoch.exposure_midpoint for epoch in epochs])
    frame = AltAz(obstime=times, location=location, pressure=0.0 * u.hPa)
    altitude = np.asarray(get_sun(times).transform_to(frame).alt.to_value(u.deg))
    return {
        epoch.frame_identifier: float(value)
        for epoch, value in zip(epochs, altitude, strict=True)
    }
