"""Sidereal-consistent bootstrap and temporal-fragment merging.

This module promotes the validated ``robust_fit_and_merge_temporal_tracks.py``
workflow into the package while reusing :class:`SiderealAxisFitter` for the
actual common-axis optimization.

The workflow is:

1. fit a robust preliminary fixed-rate sidereal axis;
2. score every temporal fragment against that physical model;
3. de-rotate compatible fragments to one common epoch;
4. greedily merge complete-linkage groups that are spatially compatible,
   never claim two detections from the same input frame, and remain
   sidereal-consistent when combined;
5. retain groups anchored by a robust informative fragment or that become
   informative collectively;
6. refit the sidereal axis on the merged stellar tracks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
from numpy.typing import NDArray

from gonet_astrometry.adapters.grid_calibration import validated_pixel_rays
from gonet_astrometry.models.grid import GridCalibration
from gonet_astrometry.models.track import StarTrack, TrackDiagnostics, TrackPoint
from gonet_astrometry.solving.sidereal import (
    SIDEREAL_RATE_RAD_PER_SECOND,
    SiderealAxisFitter,
    SiderealFitConfig,
    SiderealRotationSolution,
)
from gonet_astrometry.tracking.image_plane import (
    ImagePlaneTrackingResult,
    ResolvedTrackPoint,
)


@dataclass(frozen=True, slots=True)
class StellarTrackMergeConfig:
    """Configuration for physically merging temporal stellar fragments.

    Defaults reproduce the validated full-night diagnostic.

    Parameters
    ----------
    min_fragment_points
        Minimum Grid-valid points required before a temporal fragment may
        participate in common-epoch merging.
    merge_radius_deg
        Maximum common-epoch angular separation between every pair of source
        fragments in one group. The default is 10 arcmin.
    merged_rms_deg
        Maximum fixed-rate sidereal RMS allowed while adding a fragment to a
        group. The default is 15 arcmin.
    """

    min_fragment_points: int = 3
    merge_radius_deg: float = 10.0 / 60.0
    merged_rms_deg: float = 15.0 / 60.0

    def __post_init__(self) -> None:
        if self.min_fragment_points < 2:
            raise ValueError("min_fragment_points must be at least 2")
        if self.merge_radius_deg <= 0.0:
            raise ValueError("merge_radius_deg must be positive")
        if self.merged_rms_deg <= 0.0:
            raise ValueError("merged_rms_deg must be positive")


@dataclass(frozen=True, slots=True)
class StellarTrackMergeResult:
    """Final merged stellar tracks plus their physical bootstrap provenance."""

    tracking: ImagePlaneTrackingResult
    source_fragment_ids: tuple[tuple[int, ...], ...]
    reference_time: datetime
    preliminary_solution: SiderealRotationSolution
    final_solution: SiderealRotationSolution

    def __post_init__(self) -> None:
        if len(self.source_fragment_ids) != len(self.tracking.tracks):
            raise ValueError(
                "source_fragment_ids must align one-to-one with final tracks"
            )
        if self.reference_time.tzinfo is None:
            raise ValueError("reference_time must be timezone-aware")


@dataclass(frozen=True, slots=True)
class _PointData:
    epoch_index: int
    resolved: ResolvedTrackPoint
    timestamp: float
    ray: NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class _Fragment:
    identifier: int
    source_fragment_ids: tuple[int, ...]
    points: tuple[_PointData, ...]

    @property
    def point_count(self) -> int:
        return len(self.points)

    @property
    def times(self) -> NDArray[np.float64]:
        return np.asarray(
            [item.timestamp for item in self.points],
            dtype=np.float64,
        )

    @property
    def rays(self) -> NDArray[np.float64]:
        return np.stack(
            [item.ray for item in self.points],
            axis=0,
        )

    @property
    def duration_minutes(self) -> float:
        if self.point_count < 2:
            return 0.0
        return float((self.points[-1].timestamp - self.points[0].timestamp) / 60.0)

    @property
    def frame_indices(self) -> tuple[int, ...]:
        return tuple(item.epoch_index for item in self.points)


class SiderealTrackMerger:
    """Bootstrap a sidereal axis, merge compatible fragments, and refit."""

    def __init__(
        self,
        merge_config: StellarTrackMergeConfig | None = None,
        sidereal_config: SiderealFitConfig | None = None,
    ) -> None:
        self.merge_config = merge_config or StellarTrackMergeConfig()
        self.sidereal_config = sidereal_config or SiderealFitConfig()

    def fit_and_merge(
        self,
        tracking: ImagePlaneTrackingResult,
        calibration: GridCalibration,
    ) -> StellarTrackMergeResult:
        """Return physically merged stellar tracks and the refined sidereal fit."""
        preliminary = SiderealAxisFitter(self.sidereal_config).fit(
            tracking,
            calibration,
        )
        signed_axis = float(preliminary.rotation_sign) * np.asarray(
            preliminary.axis_grid, dtype=np.float64
        )

        fragments = [
            fragment
            for track in tracking.tracks
            if (
                fragment := _build_fragment(
                    tracking,
                    track,
                    calibration,
                    self.sidereal_config.inverse_reprojection_tolerance_px,
                )
            )
            is not None
        ]

        robust_seed_ids = {
            diagnostic.track_identifier
            for diagnostic in preliminary.track_diagnostics
            if diagnostic.diagnostic_class == "sidereal-consistent"
        }

        candidate_fragments = [
            fragment
            for fragment in fragments
            if (
                fragment.point_count >= self.merge_config.min_fragment_points
                and _track_rms_deg(fragment, signed_axis)
                <= self.sidereal_config.consistent_rms_deg
            )
        ]
        if not candidate_fragments:
            raise ValueError(
                "No sidereal-compatible temporal fragments remain for merging"
            )

        reference_timestamp = float(
            np.median(
                np.concatenate([fragment.times for fragment in candidate_fragments])
            )
        )

        groups = _merge_consistent_fragments(
            candidate_fragments,
            signed_axis,
            reference_timestamp,
            self.merge_config,
        )
        retained = _retain_robust_groups(
            groups,
            robust_seed_ids,
            signed_axis,
            self.sidereal_config,
        )
        if len(retained) < 3:
            raise ValueError(
                "Fragment merging retained fewer than three stellar tracks"
            )

        merged_tracking, source_ids = _tracking_from_fragments(
            tracking,
            retained,
        )
        final_solution = SiderealAxisFitter(self.sidereal_config).fit(
            merged_tracking,
            calibration,
        )

        # Mirror the final physical cut of the validated diagnostic, but keep
        # the returned tracking result and final-solution identifiers aligned.
        consistent_ids = {
            diagnostic.track_identifier
            for diagnostic in final_solution.track_diagnostics
            if diagnostic.diagnostic_class == "sidereal-consistent"
        }
        if len(consistent_ids) < len(merged_tracking.tracks):
            kept_fragments = [
                fragment
                for track, fragment in zip(
                    merged_tracking.tracks,
                    retained,
                    strict=True,
                )
                if track.identifier in consistent_ids
            ]
            if len(kept_fragments) < 3:
                raise ValueError(
                    "Final sidereal consistency cut retained fewer than three stars"
                )
            merged_tracking, source_ids = _tracking_from_fragments(
                tracking,
                kept_fragments,
            )
            final_solution = SiderealAxisFitter(self.sidereal_config).fit(
                merged_tracking,
                calibration,
            )

        return StellarTrackMergeResult(
            tracking=merged_tracking,
            source_fragment_ids=source_ids,
            reference_time=datetime.fromtimestamp(
                reference_timestamp,
                tz=timezone.utc,
            ),
            preliminary_solution=preliminary,
            final_solution=final_solution,
        )


def _build_fragment(
    tracking: ImagePlaneTrackingResult,
    track: StarTrack,
    calibration: GridCalibration,
    inverse_tolerance_px: float,
) -> _Fragment | None:
    """Resolve one lightweight track and retain Grid-valid point rays."""
    resolved = tracking.resolve(track)
    if len(resolved) < 2:
        return None

    x = np.asarray(
        [item.detection.x for item in resolved],
        dtype=np.float64,
    )
    y = np.asarray(
        [item.detection.y for item in resolved],
        dtype=np.float64,
    )
    converted = validated_pixel_rays(
        calibration,
        x,
        y,
        reprojection_tolerance_px=inverse_tolerance_px,
    )

    points: list[_PointData] = []
    for index, item in enumerate(resolved):
        if not bool(converted.valid[index]):
            continue
        points.append(
            _PointData(
                epoch_index=item.epoch_index,
                resolved=item,
                timestamp=item.epoch.exposure_midpoint.timestamp(),
                ray=np.asarray(converted.rays[index], dtype=np.float64),
            )
        )

    if len(points) < 2:
        return None
    points.sort(key=lambda item: item.timestamp)
    return _Fragment(
        identifier=track.identifier,
        source_fragment_ids=(track.identifier,),
        points=tuple(points),
    )


def _normalize(vector: NDArray[np.float64]) -> NDArray[np.float64]:
    value = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(value))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("Cannot normalize a non-finite or zero vector")
    return value / norm


def _rotate(
    vectors: NDArray[np.float64],
    axis: NDArray[np.float64],
    angles: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Rotate vectors around one axis using vectorized Rodrigues rotation."""
    axis = _normalize(axis)
    vectors = np.asarray(vectors, dtype=np.float64)
    angles = np.asarray(angles, dtype=np.float64)
    cosine = np.cos(angles)[:, None]
    sine = np.sin(angles)[:, None]
    projection = (vectors @ axis)[:, None]
    return np.asarray(
        vectors * cosine
        + np.cross(axis, vectors) * sine
        + axis[None, :] * projection * (1.0 - cosine),
        dtype=np.float64,
    )


def _unit_mean(vectors: NDArray[np.float64]) -> NDArray[np.float64]:
    return _normalize(np.mean(np.asarray(vectors, dtype=np.float64), axis=0))


def _angular_separation_deg(
    left: NDArray[np.float64],
    right: NDArray[np.float64],
) -> float:
    dot = float(np.clip(np.dot(left, right), -1.0, 1.0))
    return float(np.rad2deg(np.arccos(dot)))


def _derotate_to_track_center(
    fragment: _Fragment,
    signed_axis: NDArray[np.float64],
) -> NDArray[np.float64]:
    times = fragment.times
    centered = times - float(np.mean(times))
    angles = -SIDEREAL_RATE_RAD_PER_SECOND * centered
    return _rotate(fragment.rays, signed_axis, angles)


def _track_residuals_deg(
    fragment: _Fragment,
    signed_axis: NDArray[np.float64],
) -> NDArray[np.float64]:
    derotated = _derotate_to_track_center(fragment, signed_axis)
    center = _unit_mean(derotated)
    dots = np.clip(derotated @ center, -1.0, 1.0)
    return np.rad2deg(np.arccos(dots))


def _track_rms_deg(
    fragment: _Fragment,
    signed_axis: NDArray[np.float64],
) -> float:
    residuals = _track_residuals_deg(fragment, signed_axis)
    return float(np.sqrt(np.mean(np.square(residuals))))


def _track_angular_span_deg(fragment: _Fragment) -> float:
    rays = fragment.rays
    if len(rays) < 2:
        return 0.0
    if len(rays) > 32:
        indices = np.linspace(0, len(rays) - 1, 32, dtype=np.int64)
        rays = rays[indices]
    dots = np.clip(rays @ rays.T, -1.0, 1.0)
    return float(np.max(np.rad2deg(np.arccos(dots))))


def _is_fit_sufficient(
    fragment: _Fragment,
    config: SiderealFitConfig,
) -> bool:
    return (
        fragment.point_count >= config.min_track_points
        and fragment.duration_minutes >= config.min_track_duration_minutes
        and _track_angular_span_deg(fragment) >= config.min_track_span_deg
    )


def _information_score(fragment: _Fragment) -> float:
    return (
        max(fragment.duration_minutes, 1.0e-6)
        * max(_track_angular_span_deg(fragment), 1.0e-6)
        * float(np.sqrt(fragment.point_count))
    )


def _reference_centroid(
    fragment: _Fragment,
    signed_axis: NDArray[np.float64],
    reference_timestamp: float,
) -> NDArray[np.float64]:
    angles = -SIDEREAL_RATE_RAD_PER_SECOND * (fragment.times - reference_timestamp)
    return _unit_mean(
        _rotate(
            fragment.rays,
            signed_axis,
            angles,
        )
    )


def _frames_are_disjoint(
    members: list[_Fragment],
    candidate: _Fragment,
) -> bool:
    """Require at most one detection from any input frame in a stellar group."""
    occupied = {
        frame_index for member in members for frame_index in member.frame_indices
    }
    return occupied.isdisjoint(candidate.frame_indices)


def _combine_fragments(
    identifier: int,
    members: list[_Fragment],
) -> _Fragment:
    points = [point for member in members for point in member.points]
    points.sort(key=lambda item: item.timestamp)
    return _Fragment(
        identifier=identifier,
        source_fragment_ids=tuple(
            source_id for member in members for source_id in member.source_fragment_ids
        ),
        points=tuple(points),
    )


def _merge_consistent_fragments(
    fragments: list[_Fragment],
    signed_axis: NDArray[np.float64],
    reference_timestamp: float,
    config: StellarTrackMergeConfig,
) -> list[_Fragment]:
    """Apply the validated complete-linkage common-epoch fragment merge."""
    centroids = {
        fragment.identifier: _reference_centroid(
            fragment,
            signed_axis,
            reference_timestamp,
        )
        for fragment in fragments
    }
    remaining = {fragment.identifier: fragment for fragment in fragments}
    groups: list[list[_Fragment]] = []

    while remaining:
        seed = max(
            remaining.values(),
            key=lambda fragment: (
                _information_score(fragment),
                fragment.point_count,
            ),
        )
        members = [seed]
        del remaining[seed.identifier]

        while True:
            candidates: list[tuple[float, int, _Fragment]] = []
            current_centroids = [centroids[member.identifier] for member in members]

            for candidate in remaining.values():
                if not _frames_are_disjoint(members, candidate):
                    continue

                candidate_centroid = centroids[candidate.identifier]
                separations = [
                    _angular_separation_deg(
                        candidate_centroid,
                        member_centroid,
                    )
                    for member_centroid in current_centroids
                ]
                if any(
                    separation > config.merge_radius_deg for separation in separations
                ):
                    continue

                trial = _combine_fragments(-1, members + [candidate])
                if _track_rms_deg(trial, signed_axis) > config.merged_rms_deg:
                    continue

                candidates.append(
                    (
                        max(separations),
                        candidate.identifier,
                        candidate,
                    )
                )

            if not candidates:
                break

            _, candidate_id, chosen = min(
                candidates,
                key=lambda item: (
                    item[0],
                    -item[2].point_count,
                ),
            )
            members.append(chosen)
            del remaining[candidate_id]

        groups.append(members)

    return [_combine_fragments(index, members) for index, members in enumerate(groups)]


def _retain_robust_groups(
    merged: list[_Fragment],
    robust_seed_ids: set[int],
    signed_axis: NDArray[np.float64],
    sidereal_config: SiderealFitConfig,
) -> list[_Fragment]:
    """Retain groups anchored by a robust seed or informative after merging."""
    retained: list[_Fragment] = []

    for fragment in merged:
        if _track_rms_deg(fragment, signed_axis) > sidereal_config.consistent_rms_deg:
            continue

        anchored = any(
            source_id in robust_seed_ids for source_id in fragment.source_fragment_ids
        )
        if anchored or _is_fit_sufficient(fragment, sidereal_config):
            retained.append(fragment)

    retained.sort(
        key=lambda fragment: (
            -fragment.point_count,
            fragment.identifier,
        )
    )
    return retained


def _tracking_from_fragments(
    original: ImagePlaneTrackingResult,
    fragments: list[_Fragment],
) -> tuple[
    ImagePlaneTrackingResult,
    tuple[tuple[int, ...], ...],
]:
    """Convert physical fragments back to the package's lightweight track model."""
    tracks: list[StarTrack] = []
    source_ids: list[tuple[int, ...]] = []

    for identifier, fragment in enumerate(fragments):
        diagnostics = _merged_track_diagnostics(fragment)
        tracks.append(
            StarTrack(
                identifier=identifier,
                points=tuple(
                    TrackPoint(
                        point.resolved.epoch.frame_identifier,
                        point.resolved.detection.identifier,
                    )
                    for point in fragment.points
                ),
                diagnostics=diagnostics,
            )
        )
        source_ids.append(fragment.source_fragment_ids)

    return (
        ImagePlaneTrackingResult(
            original.sequence,
            tuple(tracks),
        ),
        tuple(source_ids),
    )


def _merged_track_diagnostics(fragment: _Fragment) -> TrackDiagnostics:
    """Return generic temporal bookkeeping for one merged stellar track."""
    first = fragment.points[0]
    last = fragment.points[-1]
    duration_s = float(last.timestamp - first.timestamp)
    displacement = math.hypot(
        last.resolved.detection.x - first.resolved.detection.x,
        last.resolved.detection.y - first.resolved.detection.y,
    )
    speed = 0.0 if duration_s <= 0.0 else displacement / duration_s * 60.0

    times = fragment.times - fragment.times[0]
    xs = np.asarray(
        [item.resolved.detection.x for item in fragment.points],
        dtype=np.float64,
    )
    ys = np.asarray(
        [item.resolved.detection.y for item in fragment.points],
        dtype=np.float64,
    )
    degree = min(2, fragment.point_count - 1)
    x_coeff = np.polyfit(times, xs, degree)
    y_coeff = np.polyfit(times, ys, degree)
    x_residual = xs - np.polyval(x_coeff, times)
    y_residual = ys - np.polyval(y_coeff, times)
    fit_rms = float(np.sqrt(np.mean(np.square(x_residual) + np.square(y_residual))))

    frame_indices = fragment.frame_indices
    first_epoch = min(frame_indices)
    last_epoch = max(frame_indices)
    span_epoch_count = last_epoch - first_epoch + 1
    represented = len(set(frame_indices))
    missed_frames = span_epoch_count - represented
    intervals = np.diff(fragment.times)

    return TrackDiagnostics(
        duration_s=duration_s,
        displacement_px=float(displacement),
        mean_speed_px_per_minute=float(speed),
        fit_rms_px=fit_rms,
        missed_frames=missed_frames,
        diagnostic_class="candidate",
        detection_count=fragment.point_count,
        span_epoch_count=span_epoch_count,
        coverage_fraction=(
            represented / span_epoch_count if span_epoch_count > 0 else 0.0
        ),
        median_interval_s=(float(np.median(intervals)) if intervals.size else 0.0),
        max_interval_s=(float(np.max(intervals)) if intervals.size else 0.0),
    )
