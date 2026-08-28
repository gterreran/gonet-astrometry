"""Catalog-assisted absolute camera orientation from the sidereal-axis solution.

The common celestial rotation axis determines two attitude degrees of freedom.
One rotation about that axis remains exactly unconstrained until an absolute
celestial direction is identified.  This module resolves that final degree of
freedom by matching de-rotated stellar track directions to a bright-star
catalog at one common observing epoch.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment

from gonet_astrometry.adapters.grid_calibration import validated_pixel_rays
from gonet_astrometry.catalogs.base import CatalogStar
from gonet_astrometry.geometry.horizon import (
    catalog_stars_to_enu_rays,
    enu_to_altaz,
    ncp_enu_vector,
)
from gonet_astrometry.models.frame import ObserverLocation
from gonet_astrometry.models.grid import GridCalibration
from gonet_astrometry.solving.sidereal import (
    SIDEREAL_RATE_RAD_PER_SECOND,
    SiderealRotationSolution,
)
from gonet_astrometry.tracking.image_plane import ImagePlaneTrackingResult

LOGGER = logging.getLogger(__name__)

CatalogRayProvider = Callable[
    [tuple[CatalogStar, ...], datetime, ObserverLocation],
    tuple[NDArray[np.float64], NDArray[np.float64]],
]


@dataclass(frozen=True, slots=True)
class OrientationFitConfig:
    """Settings for catalog-assisted absolute camera attitude recovery.

    Parameters
    ----------
    limiting_magnitude
        Bright-star magnitude limit applied before matching.
    bootstrap_limiting_magnitude
        Brighter magnitude limit used only for the initial twist bootstrap.
        Starting from the sparse bright subset reduces spurious identifications
        before the attitude is refined.
    min_catalog_altitude_deg
        Minimum geometric catalog altitude retained for matching. The initial
        orientation solution avoids the lowest sky where unmodeled atmospheric
        refraction is largest.
    min_track_points
        Minimum Grid-valid detections required to form one observed star anchor.
    min_track_duration_minutes
        Minimum elapsed time of a sidereal-consistent bootstrap track used as an
        orientation anchor.
    deduplication_radius_deg
        Angular radius used to collapse fragmented tracks that de-rotate to the
        same physical stellar direction.
    declination_tolerance_deg
        Maximum declination difference used to decide whether an observed
        anchor and catalog star can possibly represent the same object.
    consensus_bin_deg
        Coarse step used when searching the complete 0--360 degree twist range.
    consensus_tolerance_deg
        Minimum circular separation enforced between retained coarse-search
        twist candidates before local refinement.
    match_radius_deg
        Maximum angular separation accepted during global twist registration
        and subsequent one-to-one catalog matching.
    min_matches
        Minimum one-to-one catalog matches required for an absolute solution.
    max_anchors
        Maximum number of the strongest de-duplicated observed stellar anchors
        retained for catalog matching.
    inverse_reprojection_tolerance_px
        Grid forward/inverse round-trip tolerance for observed track points.
    track_validation_rms_deg
        Maximum full-track RMS accepted for a catalog identification after the
        candidate star is propagated across the entire matched bootstrap track.
    max_declination_robust_sigma_deg
        Maximum robust per-track declination scatter allowed for an orientation
        anchor.  Declination is measured independently at every Grid-calibrated
        track point relative to the fitted NCP.
    max_declination_p95_deg
        Maximum 95th-percentile absolute declination deviation allowed for an
        orientation anchor.  This suppresses tracks with a few large outliers
        that can look artificially good under a MAD-only statistic.
    """

    limiting_magnitude: float = 6.5
    bootstrap_limiting_magnitude: float = 5.0
    min_catalog_altitude_deg: float = 10.0
    min_track_points: int = 10
    min_track_duration_minutes: float = 30.0
    deduplication_radius_deg: float = 0.2
    declination_tolerance_deg: float = 0.15
    consensus_bin_deg: float = 0.25
    consensus_tolerance_deg: float = 1.0
    match_radius_deg: float = 0.15
    min_matches: int = 8
    max_anchors: int = 300
    inverse_reprojection_tolerance_px: float = 1e-2
    track_validation_rms_deg: float = 0.15
    max_declination_robust_sigma_deg: float = 0.075
    max_declination_p95_deg: float = 0.15

    def __post_init__(self) -> None:
        if not np.isfinite(self.limiting_magnitude):
            raise ValueError("limiting_magnitude must be finite")
        if not -90.0 <= self.min_catalog_altitude_deg < 90.0:
            raise ValueError("min_catalog_altitude_deg must lie in [-90, 90)")
        if not np.isfinite(self.bootstrap_limiting_magnitude):
            raise ValueError("bootstrap_limiting_magnitude must be finite")
        if self.bootstrap_limiting_magnitude > self.limiting_magnitude:
            raise ValueError(
                "bootstrap_limiting_magnitude cannot exceed limiting_magnitude"
            )
        if self.min_track_points < 3:
            raise ValueError("min_track_points must be at least 3")
        if self.min_track_duration_minutes <= 0:
            raise ValueError("min_track_duration_minutes must be positive")
        positive = (
            self.deduplication_radius_deg,
            self.declination_tolerance_deg,
            self.consensus_bin_deg,
            self.consensus_tolerance_deg,
            self.match_radius_deg,
            self.inverse_reprojection_tolerance_px,
            self.track_validation_rms_deg,
            self.max_declination_robust_sigma_deg,
            self.max_declination_p95_deg,
        )
        if any(value <= 0 for value in positive):
            raise ValueError("Orientation angular/tolerance settings must be positive")
        if self.consensus_bin_deg > 30.0:
            raise ValueError("consensus_bin_deg is unreasonably large")
        if self.min_matches < 3:
            raise ValueError("min_matches must be at least 3")
        if self.max_anchors < self.min_matches:
            raise ValueError("max_anchors must be at least min_matches")


@dataclass(frozen=True, slots=True)
class OrientationMatch:
    """One catalog identification supporting the absolute attitude solution.

    Parameters
    ----------
    track_identifier
        Bootstrap track selected as the observed stellar anchor.
    catalog_identifier
        Bright-star catalog identifier.
    residual_deg
        Full-track RMS angular separation between the observed stellar track and
        the predicted apparent motion of the matched catalog star.
    observed_declination_deg
        Declination inferred directly from the common sidereal pole.
    catalog_declination_deg
        Catalog declination.
    catalog_magnitude
        Catalog apparent magnitude when available.
    ray_grid
        De-rotated representative stellar direction in the Grid frame at the
        common reference epoch.
    """

    track_identifier: int
    catalog_identifier: str
    residual_deg: float
    observed_declination_deg: float
    catalog_declination_deg: float
    catalog_magnitude: float | None
    ray_grid: NDArray[np.float64]

    def __post_init__(self) -> None:
        ray = _unit_vector(self.ray_grid, "ray_grid")
        if self.residual_deg < 0 or not np.isfinite(self.residual_deg):
            raise ValueError(
                "Orientation match residual must be finite and non-negative"
            )
        object.__setattr__(self, "ray_grid", ray)


@dataclass(frozen=True, slots=True)
class AbsoluteOrientationSolution:
    """Absolute Grid-to-local-horizon camera attitude.

    The orientation matrix follows the convention
    ``ray_enu = grid_to_enu @ ray_grid`` where local ENU axes are east, north,
    and up.  The common sidereal-axis fit determines the NCP direction in the
    Grid frame; catalog matching resolves the otherwise unconstrained rotation
    about that pole.

    Parameters
    ----------
    grid_to_enu
        Right-handed 3x3 rotation matrix mapping Grid-frame rays into local ENU.
    reference_time
        Common epoch to which observed stellar tracks were de-rotated.
    location
        Representative observing location of the sequence.
    ncp_grid, ncp_enu
        North Celestial Pole directions in Grid and ENU coordinates.
    twist_deg
        Resolved rotation about the celestial-pole axis relative to the
        deterministic Grid/ENU equatorial bases used by the solver.
    fit_rms_deg, fit_median_deg, fit_p95_deg
        Catalog-match angular residual summary.
    anchor_count
        Number of unique de-rotated observed stellar anchors considered.
    catalog_star_count
        Number of visible bright catalog stars considered.
    matches
        One-to-one stellar identifications retained by the final fit.
    """

    grid_to_enu: NDArray[np.float64]
    reference_time: datetime
    location: ObserverLocation
    ncp_grid: NDArray[np.float64]
    ncp_enu: NDArray[np.float64]
    twist_deg: float
    fit_rms_deg: float
    fit_median_deg: float
    fit_p95_deg: float
    anchor_count: int
    catalog_star_count: int
    matches: tuple[OrientationMatch, ...]

    def __post_init__(self) -> None:
        matrix = np.asarray(self.grid_to_enu, dtype=np.float64)
        if matrix.shape != (3, 3):
            raise ValueError("grid_to_enu must have shape (3, 3)")
        if not np.all(np.isfinite(matrix)):
            raise ValueError("grid_to_enu must be finite")
        if not np.allclose(matrix @ matrix.T, np.eye(3), atol=1e-7):
            raise ValueError("grid_to_enu must be orthonormal")
        if not np.isclose(np.linalg.det(matrix), 1.0, atol=1e-7):
            raise ValueError("grid_to_enu must be a proper rotation matrix")
        if self.reference_time.tzinfo is None:
            raise ValueError("reference_time must be timezone-aware")
        if min(self.fit_rms_deg, self.fit_median_deg, self.fit_p95_deg) < 0:
            raise ValueError("Orientation residuals cannot be negative")
        if self.anchor_count < 0 or self.catalog_star_count < 0:
            raise ValueError("Orientation counts cannot be negative")
        object.__setattr__(self, "grid_to_enu", matrix.copy())
        object.__setattr__(self, "ncp_grid", _unit_vector(self.ncp_grid, "ncp_grid"))
        object.__setattr__(self, "ncp_enu", _unit_vector(self.ncp_enu, "ncp_enu"))

    @property
    def optical_axis_enu(self) -> NDArray[np.float64]:
        """Return the Grid optical axis expressed in local ENU coordinates."""
        return np.asarray(self.grid_to_enu[:, 2], dtype=np.float64)

    @property
    def optical_axis_azimuth_deg(self) -> float:
        """Return optical-axis azimuth eastward from north."""
        azimuth, _ = enu_to_altaz(self.optical_axis_enu)
        return azimuth

    @property
    def optical_axis_altitude_deg(self) -> float:
        """Return optical-axis altitude above the local horizon."""
        _, altitude = enu_to_altaz(self.optical_axis_enu)
        return altitude

    def local_direction_in_grid(
        self,
        direction_enu: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Convert one local ENU direction into the Grid frame."""
        return np.asarray(
            self.grid_to_enu.T @ _unit_vector(direction_enu, "direction_enu"),
            dtype=np.float64,
        )


@dataclass(frozen=True, slots=True)
class _ObservedAnchor:
    track_identifier: int
    ray_grid: NDArray[np.float64]
    declination_deg: float
    declination_robust_sigma_deg: float
    declination_p95_abs_deg: float
    quality: float
    track_rays: NDArray[np.float64]
    timestamps: NDArray[np.float64]

    def __post_init__(self) -> None:
        object.__setattr__(self, "ray_grid", _unit_vector(self.ray_grid, "ray_grid"))
        rays = np.asarray(self.track_rays, dtype=np.float64)
        times = np.asarray(self.timestamps, dtype=np.float64)
        if rays.ndim != 2 or rays.shape[1] != 3:
            raise ValueError("track_rays must have shape (n, 3)")
        if times.shape != (rays.shape[0],):
            raise ValueError("timestamps must align with track_rays")
        if not np.all(np.isfinite(times)):
            raise ValueError("timestamps must be finite")
        object.__setattr__(self, "track_rays", rays.copy())
        object.__setattr__(self, "timestamps", times.copy())


class AbsoluteOrientationSolver:
    """Resolve the final camera-attitude degree of freedom with catalog stars."""

    def __init__(self, config: OrientationFitConfig | None = None) -> None:
        self.config = config or OrientationFitConfig()

    def fit(
        self,
        tracking: ImagePlaneTrackingResult,
        sidereal: SiderealRotationSolution,
        calibration: GridCalibration,
        catalog_stars: tuple[CatalogStar, ...],
        *,
        catalog_ray_provider: CatalogRayProvider = catalog_stars_to_enu_rays,
    ) -> AbsoluteOrientationSolution:
        """Return the absolute Grid-to-ENU attitude from tracks and a catalog."""
        if tracking.sequence.epochs[0].image_shape != calibration.image_shape:
            raise ValueError(
                "Grid calibration sensor shape does not match the detection sequence"
            )
        reference_time = _sequence_reference_time(tracking)
        location = _representative_location(tracking)
        ncp_grid = -float(sidereal.rotation_sign) * np.asarray(
            sidereal.axis_grid, dtype=np.float64
        )
        ncp_grid = _unit_vector(ncp_grid, "ncp_grid")
        ncp_enu = ncp_enu_vector(location.latitude_deg)

        anchors = self._observed_anchors(
            tracking,
            sidereal,
            calibration,
            reference_time,
            ncp_grid,
        )
        if len(anchors) < self.config.min_matches:
            raise ValueError(
                "Not enough unique sidereal-consistent stellar anchors for "
                "absolute orientation"
            )

        bright = tuple(
            star
            for star in catalog_stars
            if star.magnitude is not None
            and star.magnitude <= self.config.limiting_magnitude
        )
        rays_enu, altitude_deg = catalog_ray_provider(bright, reference_time, location)
        rays_enu = np.asarray(rays_enu, dtype=np.float64)
        altitude_deg = np.asarray(altitude_deg, dtype=np.float64)
        if rays_enu.shape != (len(bright), 3) or altitude_deg.shape != (len(bright),):
            raise ValueError("Catalog ray provider returned inconsistent array shapes")
        visible = np.isfinite(altitude_deg) & (
            altitude_deg >= self.config.min_catalog_altitude_deg
        )
        catalog = tuple(
            star for star, keep in zip(bright, visible, strict=True) if keep
        )
        catalog_rays = rays_enu[visible]
        if len(catalog) < self.config.min_matches:
            raise ValueError("Not enough visible bright catalog stars for orientation")

        bootstrap_catalog = tuple(
            star
            for star in catalog
            if star.magnitude is not None
            and star.magnitude <= self.config.bootstrap_limiting_magnitude
        )
        if len(bootstrap_catalog) >= self.config.min_matches:
            bootstrap_mask = np.asarray(
                [
                    star.magnitude is not None
                    and star.magnitude <= self.config.bootstrap_limiting_magnitude
                    for star in catalog
                ],
                dtype=bool,
            )
            bootstrap_rays = catalog_rays[bootstrap_mask]
        else:
            bootstrap_catalog = catalog
            bootstrap_rays = catalog_rays

        grid_first, grid_second = _equatorial_basis(ncp_grid)
        enu_first, enu_second = _equatorial_basis(ncp_enu)
        observed_rays = np.stack([anchor.ray_grid for anchor in anchors], axis=0)
        observed_phase = _phase_about_axis(observed_rays, grid_first, grid_second)
        observed_dec = np.asarray(
            [anchor.declination_deg for anchor in anchors], dtype=np.float64
        )
        catalog_phase = _phase_about_axis(catalog_rays, enu_first, enu_second)
        catalog_dec = np.rad2deg(
            np.arcsin(np.clip(catalog_rays @ ncp_enu, -1.0, 1.0))
        )

        bootstrap_phase = _phase_about_axis(bootstrap_rays, enu_first, enu_second)
        bootstrap_dec = np.rad2deg(
            np.arcsin(np.clip(bootstrap_rays @ ncp_enu, -1.0, 1.0))
        )
        try:
            twist = self._consensus_twist(
                observed_phase,
                observed_dec,
                bootstrap_phase,
                bootstrap_dec,
                label=(
                    f"V<={self.config.bootstrap_limiting_magnitude:g} bootstrap"
                ),
            )
        except ValueError as exc:
            if len(bootstrap_catalog) == len(catalog):
                raise
            LOGGER.warning(
                "Bright-star orientation bootstrap failed (%s). Retrying global "
                "twist registration with the full V<=%.2f catalog.",
                exc,
                self.config.limiting_magnitude,
            )
            bootstrap_catalog = catalog
            bootstrap_rays = catalog_rays
            bootstrap_phase = catalog_phase
            bootstrap_dec = catalog_dec
            twist = self._consensus_twist(
                observed_phase,
                observed_dec,
                bootstrap_phase,
                bootstrap_dec,
                label=f"V<={self.config.limiting_magnitude:g} full catalog",
            )
        matches: list[tuple[int, int, float]] = []
        for _ in range(4):
            matrix = _orientation_matrix(
                ncp_grid,
                ncp_enu,
                grid_first,
                grid_second,
                enu_first,
                enu_second,
                twist,
            )
            transformed = (matrix @ observed_rays.T).T
            matches = _one_to_one_matches(
                transformed,
                bootstrap_rays,
                self.config.match_radius_deg,
            )
            if len(matches) < self.config.min_matches:
                raise ValueError(
                    "Catalog consensus did not produce enough one-to-one star matches"
                )
            deltas = np.asarray(
                [
                    _wrap_angle(
                        bootstrap_phase[catalog_index]
                        - observed_phase[obs_index]
                    )
                    for obs_index, catalog_index, _ in matches
                ],
                dtype=np.float64,
            )
            refined = _circular_mean(deltas)
            if abs(float(_wrap_signed_angle(refined - twist))) < np.deg2rad(1e-5):
                twist = refined
                break
            twist = refined

        matrix = _orientation_matrix(
            ncp_grid,
            ncp_enu,
            grid_first,
            grid_second,
            enu_first,
            enu_second,
            twist,
        )
        signed_axis = float(sidereal.rotation_sign) * np.asarray(
            sidereal.axis_grid, dtype=np.float64
        )
        reference_timestamp = reference_time.timestamp()
        matches = []
        for _ in range(3):
            transformed = (matrix @ observed_rays.T).T
            candidate_matches = _one_to_one_matches(
                transformed,
                catalog_rays,
                self.config.match_radius_deg,
            )
            matches = self._validate_matches(
                anchors,
                candidate_matches,
                catalog_rays,
                matrix,
                signed_axis,
                reference_timestamp,
            )
            if len(matches) < self.config.min_matches:
                raise ValueError(
                    "Absolute orientation lost catalog consensus after full-track "
                    "validation"
                )
            deltas = np.asarray(
                [
                    _wrap_angle(
                        catalog_phase[catalog_index]
                        - observed_phase[obs_index]
                    )
                    for obs_index, catalog_index, _ in matches
                ],
                dtype=np.float64,
            )
            refined = _circular_mean(deltas)
            if abs(float(_wrap_signed_angle(refined - twist))) < np.deg2rad(1e-5):
                twist = refined
                break
            twist = refined
            matrix = _orientation_matrix(
                ncp_grid,
                ncp_enu,
                grid_first,
                grid_second,
                enu_first,
                enu_second,
                twist,
            )

        transformed = (matrix @ observed_rays.T).T
        matches = self._validate_matches(
            anchors,
            _one_to_one_matches(
                transformed,
                catalog_rays,
                self.config.match_radius_deg,
            ),
            catalog_rays,
            matrix,
            signed_axis,
            reference_timestamp,
        )
        if len(matches) < self.config.min_matches:
            raise ValueError(
                "Absolute orientation lost catalog consensus after refinement"
            )

        orientation_matches: list[OrientationMatch] = []
        residuals: list[float] = []
        for observed_index, catalog_index, residual in matches:
            anchor = anchors[observed_index]
            star = catalog[catalog_index]
            residuals.append(residual)
            orientation_matches.append(
                OrientationMatch(
                    track_identifier=anchor.track_identifier,
                    catalog_identifier=star.identifier,
                    residual_deg=residual,
                    observed_declination_deg=anchor.declination_deg,
                    catalog_declination_deg=star.declination_deg,
                    catalog_magnitude=star.magnitude,
                    ray_grid=anchor.ray_grid,
                )
            )
        residual_array = np.asarray(residuals, dtype=np.float64)
        return AbsoluteOrientationSolution(
            grid_to_enu=matrix,
            reference_time=reference_time,
            location=location,
            ncp_grid=ncp_grid,
            ncp_enu=ncp_enu,
            twist_deg=float(np.mod(np.rad2deg(twist), 360.0)),
            fit_rms_deg=float(np.sqrt(np.mean(np.square(residual_array)))),
            fit_median_deg=float(np.median(residual_array)),
            fit_p95_deg=float(np.percentile(residual_array, 95.0)),
            anchor_count=len(anchors),
            catalog_star_count=len(catalog),
            matches=tuple(orientation_matches),
        )

    def _observed_anchors(
        self,
        tracking: ImagePlaneTrackingResult,
        sidereal: SiderealRotationSolution,
        calibration: GridCalibration,
        reference_time: datetime,
        ncp_grid: NDArray[np.float64],
    ) -> tuple[_ObservedAnchor, ...]:
        diagnostics = {
            item.track_identifier: item for item in sidereal.track_diagnostics
        }
        selected = []
        for track in tracking.tracks:
            diagnostic = diagnostics.get(track.identifier)
            if (
                diagnostic is None
                or diagnostic.diagnostic_class != "sidereal-consistent"
            ):
                continue
            if diagnostic.valid_point_count < self.config.min_track_points:
                continue
            if diagnostic.duration_s < 60.0 * self.config.min_track_duration_minutes:
                continue
            selected.append((track, diagnostic))
        if not selected:
            return ()

        epochs = {epoch.frame_identifier: epoch for epoch in tracking.sequence.epochs}
        detections = {
            epoch.frame_identifier: {
                detection.identifier: detection
                for detection in epoch.catalog.detections
            }
            for epoch in tracking.sequence.epochs
        }
        lengths = [len(track.points) for track, _ in selected]
        x = np.asarray(
            [
                detections[point.frame_identifier][point.detection_identifier].x
                for track, _ in selected
                for point in track.points
            ],
            dtype=np.float64,
        )
        y = np.asarray(
            [
                detections[point.frame_identifier][point.detection_identifier].y
                for track, _ in selected
                for point in track.points
            ],
            dtype=np.float64,
        )
        converted = validated_pixel_rays(
            calibration,
            x,
            y,
            reprojection_tolerance_px=self.config.inverse_reprojection_tolerance_px,
        )
        signed_axis = float(sidereal.rotation_sign) * np.asarray(
            sidereal.axis_grid, dtype=np.float64
        )
        reference_timestamp = reference_time.timestamp()
        anchors: list[_ObservedAnchor] = []
        offset = 0
        for (track, diagnostic), length in zip(selected, lengths, strict=True):
            stop = offset + length
            valid = converted.valid[offset:stop]
            rays = converted.rays[offset:stop][valid]
            timestamps = np.asarray(
                [
                    epochs[point.frame_identifier].exposure_midpoint.timestamp()
                    for point in track.points
                ],
                dtype=np.float64,
            )[valid]
            offset = stop
            if len(rays) < self.config.min_track_points:
                continue
            point_declination = np.rad2deg(
                np.arcsin(np.clip(rays @ ncp_grid, -1.0, 1.0))
            )
            declination_median = float(np.median(point_declination))
            declination_deviation = np.abs(
                point_declination - declination_median
            )
            declination_robust_sigma = float(
                1.4826 * np.median(declination_deviation)
            )
            declination_p95 = float(np.percentile(declination_deviation, 95.0))
            if (
                declination_robust_sigma
                > self.config.max_declination_robust_sigma_deg
                or declination_p95 > self.config.max_declination_p95_deg
            ):
                continue
            angles = SIDEREAL_RATE_RAD_PER_SECOND * (reference_timestamp - timestamps)
            reference_rays = _rotate_vectors(rays, signed_axis, angles)
            representative = _unit_mean(reference_rays)
            rms = diagnostic.rms_residual_deg or 0.0
            quality = (
                max(diagnostic.duration_s, 1.0)
                * np.sqrt(max(diagnostic.valid_point_count, 1))
                / max(rms, 0.01)
                / max(declination_robust_sigma, 0.01)
                / max(declination_p95, 0.02)
            )
            anchors.append(
                _ObservedAnchor(
                    track_identifier=track.identifier,
                    ray_grid=representative,
                    declination_deg=declination_median,
                    declination_robust_sigma_deg=declination_robust_sigma,
                    declination_p95_abs_deg=declination_p95,
                    quality=float(quality),
                    track_rays=rays,
                    timestamps=timestamps,
                )
            )

        anchors.sort(key=lambda item: item.quality, reverse=True)
        unique: list[_ObservedAnchor] = []
        cosine_limit = np.cos(np.deg2rad(self.config.deduplication_radius_deg))
        for anchor in anchors:
            if any(
                float(np.dot(anchor.ray_grid, other.ray_grid)) >= cosine_limit
                for other in unique
            ):
                continue
            unique.append(anchor)
            if len(unique) >= self.config.max_anchors:
                break
        return tuple(unique)

    def _consensus_twist(
        self,
        observed_phase: NDArray[np.float64],
        observed_dec: NDArray[np.float64],
        catalog_phase: NDArray[np.float64],
        catalog_dec: NDArray[np.float64],
        *,
        label: str = "catalog",
    ) -> float:
        """Return the globally registered twist around the known celestial pole.

        The pole fit reduces absolute orientation to one angular degree of
        freedom.  Rather than accumulating pairwise phase differences, this
        search evaluates the complete observed anchor pattern over the full
        0--360 degree twist range.  Declination is used only to construct a
        sparse set of physically plausible catalog candidates for each anchor.
        """
        candidate_indices = tuple(
            np.flatnonzero(
                np.abs(catalog_dec - declination)
                <= self.config.declination_tolerance_deg
            )
            for declination in observed_dec
        )
        candidate_counts = np.asarray(
            [len(item) for item in candidate_indices], dtype=np.int64
        )
        anchors_with_candidates = int(np.count_nonzero(candidate_counts))
        candidate_pair_count = int(np.sum(candidate_counts))
        LOGGER.info(
            "Orientation twist search (%s) | anchors=%d | catalog stars=%d | "
            "anchors with declination-compatible candidates=%d | candidate pairs=%d",
            label,
            len(observed_phase),
            len(catalog_phase),
            anchors_with_candidates,
            candidate_pair_count,
        )
        if candidate_pair_count < self.config.min_matches:
            raise ValueError(
                "Could not form enough declination-compatible catalog candidates "
                f"({candidate_pair_count} candidate pairs for "
                f"{anchors_with_candidates} anchors)"
            )

        # The search grid must be substantially finer than the acceptance
        # radius. Otherwise a real narrow peak can lie between neighboring
        # trial twists and disappear before the fine-refinement stage begins.
        coarse_step_deg = min(
            self.config.consensus_bin_deg,
            self.config.match_radius_deg / 3.0,
        )
        coarse_step = np.deg2rad(coarse_step_deg)
        coarse_twists = np.arange(0.0, 2.0 * np.pi, coarse_step, dtype=np.float64)
        coarse_scores = [
            (
                float(twist),
                *_twist_nearest_score(
                    float(twist),
                    observed_phase,
                    observed_dec,
                    catalog_phase,
                    catalog_dec,
                    candidate_indices,
                    self.config.match_radius_deg,
                ),
            )
            for twist in coarse_twists
        ]
        coarse_scores.sort(key=lambda item: (-item[1], item[2]))
        best_count = coarse_scores[0][1] if coarse_scores else 0
        best_median = coarse_scores[0][2] if coarse_scores else float("inf")
        LOGGER.info(
            "Orientation twist search (%s) | coarse step=%.3f deg | best "
            "simultaneous matches=%d | median residual=%.3f arcmin",
            label,
            coarse_step_deg,
            best_count,
            60.0 * best_median,
        )
        if not coarse_scores or coarse_scores[0][1] < self.config.min_matches:
            raise ValueError(
                "Global twist search did not find enough simultaneous catalog "
                f"matches (best={best_count}, required={self.config.min_matches})"
            )

        seed_separation = np.deg2rad(self.config.consensus_tolerance_deg)
        seeds: list[float] = []
        for twist, count, _ in coarse_scores:
            if count < self.config.min_matches:
                break
            if any(
                abs(_wrap_signed_angle(twist - other)) < seed_separation
                for other in seeds
            ):
                continue
            seeds.append(twist)
            if len(seeds) >= 8:
                break

        fine_step = max(np.deg2rad(0.01), coarse_step / 20.0)
        refined: list[tuple[float, int, float]] = []
        for seed in seeds:
            offsets = np.arange(
                -coarse_step,
                coarse_step + 0.5 * fine_step,
                fine_step,
                dtype=np.float64,
            )
            local_scores = [
                (
                    _wrap_angle(float(seed + offset)),
                    *_twist_nearest_score(
                        _wrap_angle(float(seed + offset)),
                        observed_phase,
                        observed_dec,
                        catalog_phase,
                        catalog_dec,
                        candidate_indices,
                        self.config.match_radius_deg,
                    ),
                )
                for offset in offsets
            ]
            local_scores.sort(key=lambda item: (-item[1], item[2]))
            best_twist = local_scores[0][0]
            assignment_count, assignment_median = _twist_assignment_score(
                best_twist,
                observed_phase,
                observed_dec,
                catalog_phase,
                catalog_dec,
                candidate_indices,
                self.config.match_radius_deg,
            )
            refined.append((best_twist, assignment_count, assignment_median))

        refined.sort(key=lambda item: (-item[1], item[2]))
        if not refined or refined[0][1] < self.config.min_matches:
            raise ValueError(
                "Global twist registration did not retain enough one-to-one matches"
            )
        return refined[0][0]

    def _validate_matches(
        self,
        anchors: tuple[_ObservedAnchor, ...],
        matches: list[tuple[int, int, float]],
        catalog_rays_enu: NDArray[np.float64],
        matrix: NDArray[np.float64],
        signed_axis: NDArray[np.float64],
        reference_timestamp: float,
    ) -> list[tuple[int, int, float]]:
        """Return candidate catalog IDs consistent with the full observed track."""
        validated: list[tuple[int, int, float]] = []
        inverse = np.asarray(matrix.T, dtype=np.float64)
        for observed_index, catalog_index, _ in matches:
            anchor = anchors[observed_index]
            catalog_reference_grid = inverse @ catalog_rays_enu[catalog_index]
            angles = SIDEREAL_RATE_RAD_PER_SECOND * (
                anchor.timestamps - reference_timestamp
            )
            repeated = np.repeat(catalog_reference_grid[None, :], len(angles), axis=0)
            predicted = _rotate_vectors(repeated, signed_axis, angles)
            cosine = np.clip(np.sum(predicted * anchor.track_rays, axis=1), -1.0, 1.0)
            separation = np.rad2deg(np.arccos(cosine))
            rms = float(np.sqrt(np.mean(np.square(separation))))
            if rms <= self.config.track_validation_rms_deg:
                validated.append((observed_index, catalog_index, rms))
        return validated


def _sequence_reference_time(tracking: ImagePlaneTrackingResult) -> datetime:
    first = tracking.sequence.epochs[0].exposure_midpoint
    last = tracking.sequence.epochs[-1].exposure_midpoint
    return first + timedelta(seconds=(last - first).total_seconds() / 2.0)


def _representative_location(tracking: ImagePlaneTrackingResult) -> ObserverLocation:
    locations = [epoch.location for epoch in tracking.sequence.epochs]
    return ObserverLocation(
        latitude_deg=float(np.median([item.latitude_deg for item in locations])),
        longitude_deg=float(np.median([item.longitude_deg for item in locations])),
        elevation_m=float(np.median([item.elevation_m for item in locations])),
    )


def _equatorial_basis(
    axis: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    preferred = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    first = preferred - axis * float(np.dot(preferred, axis))
    if float(np.linalg.norm(first)) < 1e-8:
        preferred = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
        first = preferred - axis * float(np.dot(preferred, axis))
    first /= np.linalg.norm(first)
    second = np.cross(axis, first)
    second /= np.linalg.norm(second)
    return first, second


def _phase_about_axis(
    rays: NDArray[np.float64],
    first: NDArray[np.float64],
    second: NDArray[np.float64],
) -> NDArray[np.float64]:
    return np.mod(np.arctan2(rays @ second, rays @ first), 2.0 * np.pi)


def _orientation_matrix(
    ncp_grid: NDArray[np.float64],
    ncp_enu: NDArray[np.float64],
    grid_first: NDArray[np.float64],
    grid_second: NDArray[np.float64],
    enu_first: NDArray[np.float64],
    enu_second: NDArray[np.float64],
    twist: float,
) -> NDArray[np.float64]:
    grid_basis = np.column_stack((grid_first, grid_second, ncp_grid))
    enu_basis = np.column_stack((enu_first, enu_second, ncp_enu))
    cosine = np.cos(twist)
    sine = np.sin(twist)
    rotate = np.asarray(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    matrix = enu_basis @ rotate @ grid_basis.T
    return np.asarray(matrix, dtype=np.float64)


def _twist_nearest_score(
    twist: float,
    observed_phase: NDArray[np.float64],
    observed_dec: NDArray[np.float64],
    catalog_phase: NDArray[np.float64],
    catalog_dec: NDArray[np.float64],
    candidate_indices: tuple[NDArray[np.intp], ...],
    radius_deg: float,
) -> tuple[int, float]:
    """Return simultaneous-match count and median residual for one twist.

    Catalog uniqueness is intentionally ignored in this inexpensive score.  It
    is used only to search the full one-dimensional twist space; the best
    separated peaks are subsequently checked with a one-to-one assignment.
    """
    nearest: list[float] = []
    for index, candidates in enumerate(candidate_indices):
        if candidates.size == 0:
            continue
        separation = _phase_declination_separation_deg(
            observed_phase[index] + twist,
            observed_dec[index],
            catalog_phase[candidates],
            catalog_dec[candidates],
        )
        best = float(np.min(separation))
        if best <= radius_deg:
            nearest.append(best)
    if not nearest:
        return 0, float("inf")
    return len(nearest), float(np.median(np.asarray(nearest, dtype=np.float64)))


def _twist_assignment_score(
    twist: float,
    observed_phase: NDArray[np.float64],
    observed_dec: NDArray[np.float64],
    catalog_phase: NDArray[np.float64],
    catalog_dec: NDArray[np.float64],
    candidate_indices: tuple[NDArray[np.intp], ...],
    radius_deg: float,
) -> tuple[int, float]:
    """Return one-to-one match count and median residual for one trial twist."""
    separation = np.full(
        (len(observed_phase), len(catalog_phase)),
        1e6,
        dtype=np.float64,
    )
    for index, candidates in enumerate(candidate_indices):
        if candidates.size == 0:
            continue
        separation[index, candidates] = _phase_declination_separation_deg(
            observed_phase[index] + twist,
            observed_dec[index],
            catalog_phase[candidates],
            catalog_dec[candidates],
        )
    rows, columns = linear_sum_assignment(separation)
    accepted = np.asarray(
        [
            separation[row, column]
            for row, column in zip(rows, columns, strict=True)
            if separation[row, column] <= radius_deg
        ],
        dtype=np.float64,
    )
    if accepted.size == 0:
        return 0, float("inf")
    return int(accepted.size), float(np.median(accepted))


def _phase_declination_separation_deg(
    observed_phase: float,
    observed_dec_deg: float,
    catalog_phase: NDArray[np.float64],
    catalog_dec_deg: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return exact spherical separations from pole phase and declination."""
    observed_dec = np.deg2rad(observed_dec_deg)
    catalog_dec = np.deg2rad(catalog_dec_deg)
    phase_delta = np.arctan2(
        np.sin(observed_phase - catalog_phase),
        np.cos(observed_phase - catalog_phase),
    )
    cosine = (
        np.sin(observed_dec) * np.sin(catalog_dec)
        + np.cos(observed_dec) * np.cos(catalog_dec) * np.cos(phase_delta)
    )
    return np.rad2deg(np.arccos(np.clip(cosine, -1.0, 1.0)))


def _one_to_one_matches(
    observed_enu: NDArray[np.float64],
    catalog_enu: NDArray[np.float64],
    radius_deg: float,
) -> list[tuple[int, int, float]]:
    cosine = np.clip(observed_enu @ catalog_enu.T, -1.0, 1.0)
    separation = np.rad2deg(np.arccos(cosine))
    cost = separation.copy()
    cost[cost > radius_deg] = 1e6
    rows, columns = linear_sum_assignment(cost)
    return [
        (int(row), int(column), float(separation[row, column]))
        for row, column in zip(rows, columns, strict=True)
        if separation[row, column] <= radius_deg
    ]


def _rotate_vectors(
    vectors: NDArray[np.float64],
    axis: NDArray[np.float64],
    angles: NDArray[np.float64],
) -> NDArray[np.float64]:
    cosine = np.cos(angles)[:, None]
    sine = np.sin(angles)[:, None]
    projection = (vectors @ axis)[:, None]
    return (
        vectors * cosine
        + np.cross(axis, vectors) * sine
        + axis[None, :] * projection * (1.0 - cosine)
    )


def _unit_mean(vectors: NDArray[np.float64]) -> NDArray[np.float64]:
    mean = np.mean(vectors, axis=0)
    return _unit_vector(mean, "mean ray")


def _unit_vector(value: NDArray[np.float64], name: str) -> NDArray[np.float64]:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (3,):
        raise ValueError(f"{name} must have shape (3,)")
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError(f"{name} must be finite and non-zero")
    return vector / norm


def _wrap_angle(value: float) -> float:
    return float(value % (2.0 * np.pi))


def _wrap_signed_angle(value: float) -> float:
    return float((value + np.pi) % (2.0 * np.pi) - np.pi)


def _circular_mean(values: NDArray[np.float64]) -> float:
    vector = np.mean(np.exp(1j * values))
    if abs(vector) <= 1e-12:
        raise ValueError("Catalog matches do not define a unique orientation twist")
    return float(_wrap_angle(np.angle(vector)))
