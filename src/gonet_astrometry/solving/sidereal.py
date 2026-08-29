"""Shared sidereal-rotation fitting in Grid-calibrated ray coordinates.

The bootstrap image-plane tracker intentionally knows nothing about celestial
mechanics.  This module is the first physical consistency stage: tracked pixel
positions are converted to unit Grid-frame rays and genuine stellar tracks are
required to follow one common rotation axis at the sidereal angular rate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares

from gonet_astrometry.adapters.grid_calibration import validated_pixel_rays
from gonet_astrometry.models.grid import GridCalibration
from gonet_astrometry.tracking.image_plane import ImagePlaneTrackingResult

SIDEREAL_DAY_SECONDS = 86_164.0905
"""Length of the mean sidereal day used by the first rotation model."""

SIDEREAL_RATE_RAD_PER_SECOND = 2.0 * np.pi / SIDEREAL_DAY_SECONDS
"""Fixed angular speed used by the apparent-sky rotation fit."""

SiderealTrackClass = Literal[
    "sidereal-consistent",
    "sidereal-rejected",
    "insufficient",
]


@dataclass(frozen=True, slots=True)
class SiderealFitConfig:
    """Settings for the robust common sidereal-axis fit.

    Parameters
    ----------
    min_track_points
        Minimum Grid-valid detections required for a track to constrain the
        spherical model.
    min_track_duration_minutes
        Minimum elapsed time between the first and last valid detections.
    min_track_span_deg
        Minimum observed angular displacement within the track.
    robust_scale_deg
        Soft-L1 residual scale used by the global axis optimization.
    consistent_rms_deg
        Per-track de-rotated RMS threshold used only for diagnostic
        classification after fitting.
    max_fit_tracks
        Maximum number of the most informative eligible tracks used by the
        nonlinear fit. All tracks are still scored afterward.
    max_fit_points_per_track
        Maximum evenly sampled points retained from each track in the global
        optimization.
    inverse_reprojection_tolerance_px
        Maximum Grid forward/inverse round-trip residual for accepting a pixel
        coordinate as lying inside the calibrated angular domain.
    """

    min_track_points: int = 5
    min_track_duration_minutes: float = 5.0
    min_track_span_deg: float = 0.1
    robust_scale_deg: float = 0.1
    consistent_rms_deg: float = 0.25
    max_fit_tracks: int = 1000
    max_fit_points_per_track: int = 32
    inverse_reprojection_tolerance_px: float = 1e-2

    def __post_init__(self) -> None:
        if self.min_track_points < 3:
            raise ValueError("min_track_points must be at least 3")
        if self.min_track_duration_minutes <= 0:
            raise ValueError("min_track_duration_minutes must be positive")
        if self.min_track_span_deg <= 0:
            raise ValueError("min_track_span_deg must be positive")
        if self.robust_scale_deg <= 0:
            raise ValueError("robust_scale_deg must be positive")
        if self.consistent_rms_deg <= 0:
            raise ValueError("consistent_rms_deg must be positive")
        if self.max_fit_tracks < 3:
            raise ValueError("max_fit_tracks must be at least 3")
        if self.max_fit_points_per_track < 3:
            raise ValueError("max_fit_points_per_track must be at least 3")
        if self.inverse_reprojection_tolerance_px <= 0:
            raise ValueError("inverse_reprojection_tolerance_px must be positive")


@dataclass(frozen=True, slots=True)
class SiderealTrackDiagnostics:
    """Spherical consistency measurements for one bootstrap track.

    Parameters
    ----------
    track_identifier
        Identifier of the corresponding bootstrap image-plane track.
    diagnostic_class
        Spherical classification assigned after the common-axis fit.
    total_point_count
        Number of detections in the original bootstrap track.
    valid_point_count
        Number of detections successfully converted through the Grid model.
    duration_s
        Elapsed time covered by the valid detections.
    angular_span_deg
        Largest angular separation represented by the valid ray sequence.
    rms_residual_deg, median_residual_deg, max_residual_deg
        De-rotated angular residual statistics. These are ``None`` when the
        track is insufficient to score.
    """

    track_identifier: int
    diagnostic_class: SiderealTrackClass
    total_point_count: int
    valid_point_count: int
    duration_s: float
    angular_span_deg: float
    rms_residual_deg: float | None = None
    median_residual_deg: float | None = None
    max_residual_deg: float | None = None

    @property
    def valid_fraction(self) -> float:
        """Return the fraction of track points inside Grid calibration coverage."""
        if self.total_point_count == 0:
            return 0.0
        return self.valid_point_count / self.total_point_count


@dataclass(frozen=True, slots=True)
class SiderealRotationSolution:
    """Robust apparent-sky rotation solution in Grid-frame coordinates.

    ``axis_grid`` is the representative direction of the fitted rotation-axis
    line lying in the Grid optical hemisphere (non-negative Grid ``z``).
    ``rotation_sign`` records whether apparent sky motion follows the positive
    or negative right-hand sense around that representative pole. This stage
    does not yet assign the pole an absolute north/south celestial label.

    Parameters
    ----------
    axis_grid
        Unit direction of the representative rotation pole in the Grid frame.
    fit_rms_deg, fit_median_deg, fit_p95_deg
        Angular residual summary over the points retained by the global fit.
    fitted_track_count
        Number of bootstrap tracks used by the nonlinear fit.
    fitted_point_count
        Number of sampled ray measurements used by the nonlinear fit.
    track_diagnostics
        Spherical consistency diagnostics for every bootstrap track.
    rotation_sign
        Right-hand rotation sense around ``axis_grid``.
    """

    axis_grid: NDArray[np.float64]
    fit_rms_deg: float
    fit_median_deg: float
    fit_p95_deg: float
    fitted_track_count: int
    fitted_point_count: int
    track_diagnostics: tuple[SiderealTrackDiagnostics, ...]
    rotation_sign: int = 1

    def __post_init__(self) -> None:
        axis = np.asarray(self.axis_grid, dtype=np.float64)
        if axis.shape != (3,):
            raise ValueError("axis_grid must have shape (3,)")
        norm = float(np.linalg.norm(axis))
        if not np.isfinite(norm) or not np.isclose(norm, 1.0, atol=1e-7):
            raise ValueError("axis_grid must be a finite unit vector")
        if min(self.fit_rms_deg, self.fit_median_deg, self.fit_p95_deg) < 0:
            raise ValueError("Sidereal fit residuals cannot be negative")
        if self.fitted_track_count < 0 or self.fitted_point_count < 0:
            raise ValueError("Sidereal fit counts cannot be negative")
        if self.rotation_sign not in {-1, 1}:
            raise ValueError("rotation_sign must be either -1 or +1")
        if axis[2] < -1e-12:
            raise ValueError("axis_grid must use the non-negative Grid-z direction")
        object.__setattr__(self, "axis_grid", axis.copy())

    @property
    def axis_r_deg(self) -> float:
        """Return polar radius of the apparent rotation axis in the Grid frame."""
        return float(np.rad2deg(np.arccos(np.clip(self.axis_grid[2], -1.0, 1.0))))

    @property
    def axis_theta_deg(self) -> float:
        """Return Grid-frame azimuth of the apparent rotation axis."""
        return float(
            np.mod(
                np.rad2deg(np.arctan2(self.axis_grid[1], self.axis_grid[0])),
                360.0,
            )
        )

    def diagnostic_counts(self) -> dict[SiderealTrackClass, int]:
        """Return mutually exclusive per-track spherical classifications."""
        counts: dict[SiderealTrackClass, int] = {
            "sidereal-consistent": 0,
            "sidereal-rejected": 0,
            "insufficient": 0,
        }
        for diagnostic in self.track_diagnostics:
            counts[diagnostic.diagnostic_class] += 1
        return counts


@dataclass(frozen=True, slots=True)
class _RayTrack:
    identifier: int
    times_s: NDArray[np.float64]
    rays: NDArray[np.float64]
    total_point_count: int
    duration_s: float
    angular_span_deg: float
    bootstrap_class: str

    @property
    def valid_point_count(self) -> int:
        return int(self.rays.shape[0])


class SiderealAxisFitter:
    """Fit one shared fixed-rate rotation axis to bootstrap star tracklets."""

    def __init__(self, config: SiderealFitConfig | None = None) -> None:
        self.config = config or SiderealFitConfig()

    def fit(
        self,
        tracking: ImagePlaneTrackingResult,
        calibration: GridCalibration,
    ) -> SiderealRotationSolution:
        """Convert track pixels to rays and fit the common sidereal axis."""
        if tracking.sequence.epochs[0].image_shape != calibration.image_shape:
            raise ValueError(
                "Grid calibration sensor shape does not match the detection sequence: "
                f"calibration={calibration.image_shape}, "
                f"sequence={tracking.sequence.epochs[0].image_shape}"
            )
        ray_tracks = self._build_ray_tracks(tracking, calibration)
        candidate_tracks = self._fit_candidate_tracks(ray_tracks)
        if len(candidate_tracks) < 3:
            raise ValueError(
                "At least three sufficiently long Grid-calibrated candidate tracks "
                "are required for a shared sidereal-axis fit"
            )

        seed, consensus_tracks = self._consensus_axis(candidate_tracks)
        signed_seed = self._select_rotation_sense(seed, consensus_tracks)
        preliminary_axis, _ = self._refine_axis(signed_seed, consensus_tracks)

        refit_threshold_deg = max(
            4.0 * self.config.consistent_rms_deg,
            1.0,
        )
        fit_tracks = [
            track
            for track in candidate_tracks
            if _track_rms_deg(track, preliminary_axis) <= refit_threshold_deg
        ]
        if len(fit_tracks) < 3:
            fit_tracks = consensus_tracks
        fit_tracks.sort(key=self._information_score, reverse=True)
        fit_tracks = fit_tracks[: self.config.max_fit_tracks]

        signed_axis, _ = self._refine_axis(preliminary_axis, fit_tracks)
        final_tracks = [
            track
            for track in fit_tracks
            if _track_rms_deg(track, signed_axis) <= refit_threshold_deg
        ]
        if len(final_tracks) >= 3 and len(final_tracks) < len(fit_tracks):
            signed_axis, _ = self._refine_axis(signed_axis, final_tracks)
            fit_tracks = final_tracks
        if signed_axis[2] >= 0.0:
            axis = signed_axis
            rotation_sign = 1
        else:
            axis = -signed_axis
            rotation_sign = -1

        diagnostics = tuple(
            self._diagnose_track(track, signed_axis) for track in ray_tracks
        )
        angular_fit_residuals = self._fit_angular_residuals(fit_tracks, signed_axis)
        return SiderealRotationSolution(
            axis_grid=axis,
            fit_rms_deg=float(np.sqrt(np.mean(np.square(angular_fit_residuals)))),
            fit_median_deg=float(np.median(angular_fit_residuals)),
            fit_p95_deg=float(np.percentile(angular_fit_residuals, 95.0)),
            fitted_track_count=len(fit_tracks),
            fitted_point_count=sum(
                min(track.valid_point_count, self.config.max_fit_points_per_track)
                for track in fit_tracks
            ),
            track_diagnostics=diagnostics,
            rotation_sign=rotation_sign,
        )

    def fit_candidate_count(
        self,
        tracking: ImagePlaneTrackingResult,
        calibration: GridCalibration,
    ) -> int:
        """Return the number of tracks eligible to seed a shared-axis fit.

        This performs the same Grid validation and sufficiency filtering used
        by :meth:`fit`, but does not run consensus finding or nonlinear
        optimization. It is intended for orchestration code that needs to
        decide whether a sidereal solve is meaningful for a short sequence.
        """
        if tracking.sequence.epochs[0].image_shape != calibration.image_shape:
            raise ValueError(
                "Grid calibration sensor shape does not match the detection sequence: "
                f"calibration={calibration.image_shape}, "
                f"sequence={tracking.sequence.epochs[0].image_shape}"
            )
        ray_tracks = self._build_ray_tracks(tracking, calibration)
        return len(self._fit_candidate_tracks(ray_tracks))

    def _fit_candidate_tracks(self, ray_tracks: list[_RayTrack]) -> list[_RayTrack]:
        """Return sufficiently informative tracks in fitter priority order."""
        eligible = [track for track in ray_tracks if self._is_sufficient(track)]
        candidate_tracks = [
            track for track in eligible if track.bootstrap_class == "candidate"
        ]
        if len(candidate_tracks) < 3:
            candidate_tracks = [
                track for track in eligible if track.bootstrap_class != "poor-fit"
            ]
        candidate_tracks.sort(key=self._information_score, reverse=True)
        return candidate_tracks[: self.config.max_fit_tracks]

    def _build_ray_tracks(
        self,
        tracking: ImagePlaneTrackingResult,
        calibration: GridCalibration,
    ) -> list[_RayTrack]:
        epochs = {epoch.frame_identifier: epoch for epoch in tracking.sequence.epochs}
        detections = {
            epoch.frame_identifier: {
                detection.identifier: detection
                for detection in epoch.catalog.detections
            }
            for epoch in tracking.sequence.epochs
        }
        resolved_tracks = [
            tuple(
                (
                    epochs[point.frame_identifier],
                    detections[point.frame_identifier][point.detection_identifier],
                )
                for point in track.points
            )
            for track in tracking.tracks
        ]
        lengths = [len(points) for points in resolved_tracks]
        x = np.asarray(
            [detection.x for points in resolved_tracks for _, detection in points],
            dtype=np.float64,
        )
        y = np.asarray(
            [detection.y for points in resolved_tracks for _, detection in points],
            dtype=np.float64,
        )
        converted = validated_pixel_rays(
            calibration,
            x,
            y,
            reprojection_tolerance_px=(self.config.inverse_reprojection_tolerance_px),
        )
        ray_tracks: list[_RayTrack] = []
        offset = 0
        for track, points, length in zip(
            tracking.tracks,
            resolved_tracks,
            lengths,
            strict=True,
        ):
            stop = offset + length
            valid = converted.valid[offset:stop]
            rays = converted.rays[offset:stop][valid]
            timestamps = np.asarray(
                [epoch.exposure_midpoint.timestamp() for epoch, _ in points],
                dtype=np.float64,
            )[valid]
            offset = stop
            if timestamps.size:
                times = timestamps - timestamps[0]
                duration = float(times[-1])
            else:
                times = np.empty(0, dtype=np.float64)
                duration = 0.0
            span = _angular_span_deg(rays)
            ray_tracks.append(
                _RayTrack(
                    identifier=track.identifier,
                    times_s=times,
                    rays=np.asarray(rays, dtype=np.float64),
                    total_point_count=length,
                    duration_s=duration,
                    angular_span_deg=span,
                    bootstrap_class=track.diagnostic_class,
                )
            )
        return ray_tracks

    def _is_sufficient(self, track: _RayTrack) -> bool:
        return (
            track.valid_point_count >= self.config.min_track_points
            and track.duration_s >= 60.0 * self.config.min_track_duration_minutes
            and track.angular_span_deg >= self.config.min_track_span_deg
        )

    @staticmethod
    def _information_score(track: _RayTrack) -> float:
        return float(
            max(track.duration_s, 1.0)
            * max(track.angular_span_deg, 1e-6)
            * np.sqrt(track.valid_point_count)
        )

    def _consensus_axis(
        self,
        tracks: list[_RayTrack],
    ) -> tuple[NDArray[np.float64], list[_RayTrack]]:
        estimates: list[tuple[_RayTrack, NDArray[np.float64], float]] = []
        for track in tracks:
            estimate = self._track_axis_estimate(track)
            if estimate is not None:
                normal, weight = estimate
                estimates.append((track, normal, weight))
        if len(estimates) < 3:
            raise ValueError(
                "Could not derive enough per-track rotation-axis estimates"
            )

        normals = np.stack([item[1] for item in estimates], axis=0)
        weights = np.asarray([item[2] for item in estimates], dtype=np.float64)
        dots = np.clip(np.abs(normals @ normals.T), 0.0, 1.0)
        angles_deg = np.rad2deg(np.arccos(dots))
        tolerance_deg = 12.0
        support = angles_deg <= tolerance_deg
        scores = support @ weights
        best = int(np.argmax(scores))
        inliers = support[best]

        axis = self._weighted_axis(normals[inliers], weights[inliers])
        distances = np.rad2deg(np.arccos(np.clip(np.abs(normals @ axis), 0.0, 1.0)))
        inliers = distances <= tolerance_deg
        if int(np.count_nonzero(inliers)) < 3:
            raise ValueError("Could not find a coherent common-axis track consensus")
        axis = self._weighted_axis(normals[inliers], weights[inliers])
        consensus_tracks = [
            item[0] for item, keep in zip(estimates, inliers, strict=True) if keep
        ]
        return axis, consensus_tracks

    def _track_axis_estimate(
        self,
        track: _RayTrack,
    ) -> tuple[NDArray[np.float64], float] | None:
        sampled_rays, _ = self._sample_track(track)
        centered = sampled_rays - np.mean(sampled_rays, axis=0)
        if centered.shape[0] < 3:
            return None
        _, singular, vt = np.linalg.svd(centered, full_matrices=False)
        if singular.size < 3 or singular[0] <= 0 or singular[1] <= 0:
            return None
        normal = np.asarray(vt[-1], dtype=np.float64)
        norm = float(np.linalg.norm(normal))
        if not np.isfinite(norm) or norm <= 0:
            return None
        normal /= norm
        curvature_leverage = float(singular[1] / singular[0])
        plane_error = float(singular[2] / singular[1])
        planarity_quality = 1.0 / (1.0 + (plane_error / 0.05) ** 2)
        weight = (
            max(track.angular_span_deg, 1e-3) ** 2
            * np.sqrt(track.valid_point_count)
            * max(curvature_leverage, 1e-4)
            * planarity_quality
        )
        if not np.isfinite(weight) or weight <= 0:
            return None
        return normal, float(weight)

    @staticmethod
    def _weighted_axis(
        normals: NDArray[np.float64],
        weights: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        matrix = np.einsum("i,ij,ik->jk", weights, normals, normals)
        _, vectors = np.linalg.eigh(matrix)
        axis = np.asarray(vectors[:, -1], dtype=np.float64)
        return axis / np.linalg.norm(axis)

    @staticmethod
    def _select_rotation_sense(
        axis: NDArray[np.float64],
        tracks: list[_RayTrack],
    ) -> NDArray[np.float64]:
        positive = np.asarray([_track_rms_deg(track, axis) for track in tracks])
        negative = np.asarray([_track_rms_deg(track, -axis) for track in tracks])
        return axis if np.median(positive) <= np.median(negative) else -axis

    def _refine_axis(
        self,
        seed: NDArray[np.float64],
        tracks: list[_RayTrack],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        first, second = _tangent_basis(seed)

        def axis_from_offset(offset: NDArray[np.float64]) -> NDArray[np.float64]:
            axis = seed + offset[0] * first + offset[1] * second
            return axis / np.linalg.norm(axis)

        def residual(offset: NDArray[np.float64]) -> NDArray[np.float64]:
            axis = axis_from_offset(offset)
            pieces: list[NDArray[np.float64]] = []
            for track in tracks:
                rays, times = self._sample_track(track)
                derotated = _derotate_track(rays, times, axis)
                center = _unit_mean(derotated)
                pieces.append((derotated - center) / np.sqrt(len(derotated)))
            return np.concatenate(pieces, axis=0).ravel()

        chord_scale = 2.0 * np.sin(np.deg2rad(self.config.robust_scale_deg) / 2.0)
        offset_limit = np.tan(np.deg2rad(20.0))
        optimized = least_squares(
            residual,
            x0=np.zeros(2, dtype=np.float64),
            bounds=(-offset_limit, offset_limit),
            loss="soft_l1",
            f_scale=max(chord_scale, 1e-9),
            max_nfev=100,
        )
        axis = axis_from_offset(np.asarray(optimized.x, dtype=np.float64))
        return axis, residual(np.asarray(optimized.x, dtype=np.float64))

    def _sample_track(
        self,
        track: _RayTrack,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        count = track.valid_point_count
        if count <= self.config.max_fit_points_per_track:
            return track.rays, track.times_s
        indices = np.linspace(
            0,
            count - 1,
            self.config.max_fit_points_per_track,
            dtype=np.int64,
        )
        return track.rays[indices], track.times_s[indices]

    def _diagnose_track(
        self,
        track: _RayTrack,
        axis: NDArray[np.float64],
    ) -> SiderealTrackDiagnostics:
        if not self._is_sufficient(track):
            return SiderealTrackDiagnostics(
                track_identifier=track.identifier,
                diagnostic_class="insufficient",
                total_point_count=track.total_point_count,
                valid_point_count=track.valid_point_count,
                duration_s=track.duration_s,
                angular_span_deg=track.angular_span_deg,
            )
        residuals = _track_angular_residuals_deg(track.rays, track.times_s, axis)
        rms = float(np.sqrt(np.mean(np.square(residuals))))
        diagnostic_class: SiderealTrackClass = (
            "sidereal-consistent"
            if rms <= self.config.consistent_rms_deg
            else "sidereal-rejected"
        )
        return SiderealTrackDiagnostics(
            track_identifier=track.identifier,
            diagnostic_class=diagnostic_class,
            total_point_count=track.total_point_count,
            valid_point_count=track.valid_point_count,
            duration_s=track.duration_s,
            angular_span_deg=track.angular_span_deg,
            rms_residual_deg=rms,
            median_residual_deg=float(np.median(residuals)),
            max_residual_deg=float(np.max(residuals)),
        )

    def _fit_angular_residuals(
        self,
        tracks: list[_RayTrack],
        axis: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        pieces = [
            _track_angular_residuals_deg(*self._sample_track(track), axis)
            for track in tracks
        ]
        return np.concatenate(pieces)


def _track_rms_deg(
    track: _RayTrack,
    axis: NDArray[np.float64],
) -> float:
    residuals = _track_angular_residuals_deg(track.rays, track.times_s, axis)
    return float(np.sqrt(np.mean(np.square(residuals))))


def _tangent_basis(
    axis: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    reference = (
        np.array([0.0, 0.0, 1.0], dtype=np.float64)
        if abs(float(axis[2])) < 0.9
        else np.array([1.0, 0.0, 0.0], dtype=np.float64)
    )
    first = np.cross(axis, reference)
    first /= np.linalg.norm(first)
    second = np.cross(axis, first)
    second /= np.linalg.norm(second)
    return first, second


def _rotate_about_axis(
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


def _derotate_track(
    rays: NDArray[np.float64],
    times_s: NDArray[np.float64],
    axis: NDArray[np.float64],
) -> NDArray[np.float64]:
    centered_times = times_s - float(np.mean(times_s))
    angles = -SIDEREAL_RATE_RAD_PER_SECOND * centered_times
    return _rotate_about_axis(rays, axis, angles)


def _unit_mean(vectors: NDArray[np.float64]) -> NDArray[np.float64]:
    mean = np.mean(vectors, axis=0)
    norm = float(np.linalg.norm(mean))
    if norm <= 0 or not np.isfinite(norm):
        raise ValueError("Could not determine a mean ray for a track")
    return mean / norm


def _track_angular_residuals_deg(
    rays: NDArray[np.float64],
    times_s: NDArray[np.float64],
    axis: NDArray[np.float64],
) -> NDArray[np.float64]:
    derotated = _derotate_track(rays, times_s, axis)
    center = _unit_mean(derotated)
    cosine = np.clip(derotated @ center, -1.0, 1.0)
    return np.rad2deg(np.arccos(cosine))


def _angular_span_deg(rays: NDArray[np.float64]) -> float:
    if len(rays) < 2:
        return 0.0
    reference = rays[0]
    cosine = np.clip(rays @ reference, -1.0, 1.0)
    return float(np.max(np.rad2deg(np.arccos(cosine))))
