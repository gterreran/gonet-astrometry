"""Catalog-assisted stellar identification across a detection sequence.

The Grid calibration supplies the camera's internal ray geometry but no absolute
orientation.  This module solves the remaining rigid Grid-to-local-ENU rotation
from catalog/detection correspondences and then identifies visible stars in every
frame.  The catalog is loaded once by the caller; only the apparent local
coordinates are propagated to individual exposure times.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import distance_transform_edt
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

from gonet_astrometry.adapters.grid_calibration import (
    PortableGridTransform,
    validated_pixel_rays,
)
from gonet_astrometry.catalogs.base import CatalogStar
from gonet_astrometry.detection.field_mask import FieldMask
from gonet_astrometry.geometry.horizon import catalog_stars_to_enu_rays, enu_to_altaz
from gonet_astrometry.models.frame import ObserverLocation
from gonet_astrometry.models.grid import GridCalibration
from gonet_astrometry.models.track import StarTrack, TrackDiagnostics, TrackPoint
from gonet_astrometry.tracking.image_plane import ImagePlaneTrackingResult
from gonet_astrometry.tracking.sequence import DetectionEpoch, DetectionSequence

logger = logging.getLogger(__name__)

CatalogRayProvider = Callable[
    [tuple[CatalogStar, ...], datetime, ObserverLocation],
    tuple[NDArray[np.float64], NDArray[np.float64]],
]
MatchKind = Literal["primary", "bright-rescue"]

_BASE_GRID_TO_ENU = np.asarray(
    [
        [-1.0, 0.0, 0.0],  # Grid +x / image-right -> west at zero roll
        [0.0, -1.0, 0.0],  # Grid +y / image-down  -> south
        [0.0, 0.0, 1.0],  # Grid +z              -> zenith
    ],
    dtype=np.float64,
)


@dataclass(frozen=True, slots=True)
class StellarIdentificationConfig:
    """Configuration for sequence-wide catalog identification.

    The defaults intentionally follow the catalog-backed Adler experiments:
    the production detector reaches well past V=4 in dark regions at 3.5 sigma,
    so the identification catalog extends to V=4.5 while the attitude bootstrap
    uses the denser-confidence V<=3.2 subset.
    """

    limiting_magnitude: float = 4.5
    bootstrap_limiting_magnitude: float = 3.2
    min_catalog_altitude_deg: float = 0.0
    bootstrap_rotation_step_deg: float = 2.0
    bootstrap_match_radius_px: float = 120.0
    refinement_radii_px: tuple[float, ...] = (70.0, 40.0, 25.0, 15.0)
    min_bootstrap_matches: int = 6
    bootstrap_candidate_frame_count: int = 8
    bootstrap_validation_frame_count: int = 24
    min_bootstrap_validation_frames: int = 2
    bootstrap_validation_radius_px: float = 25.0
    sequence_refinement_radius_px: float = 25.0
    sequence_refinement_iterations: int = 2
    final_match_radius_px: float = 15.0
    bright_rescue_magnitude: float = 2.5
    bright_rescue_radius_px: float = 25.0
    robust_clip_sigma: float = 4.0
    robust_clip_floor_arcmin: float = 60.0
    max_refinement_pairs: int = 20_000
    minimum_catalog_track_length: int = 2

    def __post_init__(self) -> None:
        finite = (
            self.limiting_magnitude,
            self.bootstrap_limiting_magnitude,
            self.min_catalog_altitude_deg,
            self.bootstrap_rotation_step_deg,
            self.bootstrap_match_radius_px,
            self.bootstrap_validation_radius_px,
            self.sequence_refinement_radius_px,
            self.final_match_radius_px,
            self.bright_rescue_magnitude,
            self.bright_rescue_radius_px,
            self.robust_clip_sigma,
            self.robust_clip_floor_arcmin,
        )
        if not all(np.isfinite(value) for value in finite):
            raise ValueError("Stellar-identification settings must be finite")
        if self.bootstrap_limiting_magnitude > self.limiting_magnitude:
            raise ValueError(
                "bootstrap_limiting_magnitude cannot exceed limiting_magnitude"
            )
        if not -90.0 <= self.min_catalog_altitude_deg < 90.0:
            raise ValueError("min_catalog_altitude_deg must lie in [-90, 90)")
        if not 0.0 < self.bootstrap_rotation_step_deg <= 90.0:
            raise ValueError("bootstrap_rotation_step_deg must lie in (0, 90]")
        positive = (
            self.bootstrap_match_radius_px,
            self.bootstrap_validation_radius_px,
            self.sequence_refinement_radius_px,
            self.final_match_radius_px,
            self.bright_rescue_radius_px,
            self.robust_clip_sigma,
            self.robust_clip_floor_arcmin,
        )
        if any(value <= 0.0 for value in positive):
            raise ValueError("Identification radii and robust scales must be positive")
        if not self.refinement_radii_px or any(
            value <= 0.0 for value in self.refinement_radii_px
        ):
            raise ValueError("refinement_radii_px must contain positive values")
        if self.min_bootstrap_matches < 3:
            raise ValueError("min_bootstrap_matches must be at least 3")
        if self.bootstrap_candidate_frame_count < 1:
            raise ValueError("bootstrap_candidate_frame_count must be at least 1")
        if self.bootstrap_validation_frame_count < 1:
            raise ValueError("bootstrap_validation_frame_count must be at least 1")
        if self.min_bootstrap_validation_frames < 1:
            raise ValueError("min_bootstrap_validation_frames must be at least 1")
        if self.sequence_refinement_iterations < 0:
            raise ValueError("sequence_refinement_iterations cannot be negative")
        if self.max_refinement_pairs < self.min_bootstrap_matches:
            raise ValueError("max_refinement_pairs is too small")
        if self.minimum_catalog_track_length < 2:
            raise ValueError("minimum_catalog_track_length must be at least 2")


@dataclass(frozen=True, slots=True)
class StellarIdentification:
    """One catalog star expected inside the usable field for one epoch."""

    catalog_identifier: str
    catalog_magnitude: float | None
    azimuth_deg: float
    altitude_deg: float
    predicted_x: float
    predicted_y: float
    detection_identifier: int | None = None
    residual_px: float | None = None
    residual_arcmin: float | None = None
    match_kind: MatchKind | None = None

    @property
    def matched(self) -> bool:
        """Return whether this expected star has an associated detection."""
        return self.detection_identifier is not None


@dataclass(frozen=True, slots=True)
class StellarIdentificationEpoch:
    """Catalog visibility and associations for one image."""

    frame_identifier: str
    visible_stars: tuple[StellarIdentification, ...]

    @property
    def matched_star_count(self) -> int:
        """Return the number of expected stars assigned to detections."""
        return sum(item.matched for item in self.visible_stars)

    @property
    def unmatched_star_count(self) -> int:
        """Return the number of expected stars without a detection."""
        return len(self.visible_stars) - self.matched_star_count

    @property
    def matched_detection_identifiers(self) -> tuple[int, ...]:
        """Return detection identifiers already claimed by catalog stars."""
        return tuple(
            int(item.detection_identifier)
            for item in self.visible_stars
            if item.detection_identifier is not None
        )

    @property
    def matched_stars(self) -> tuple[StellarIdentification, ...]:
        """Return expected stars assigned to detections in this epoch."""
        return tuple(item for item in self.visible_stars if item.matched)

    @property
    def unmatched_stars(self) -> tuple[StellarIdentification, ...]:
        """Return expected stars with no associated detection in this epoch."""
        return tuple(item for item in self.visible_stars if not item.matched)

    @property
    def median_residual_arcmin(self) -> float | None:
        """Return the median angular residual of matched stars when available."""
        residuals = [
            item.residual_arcmin
            for item in self.visible_stars
            if item.residual_arcmin is not None
        ]
        if not residuals:
            return None
        return float(np.median(np.asarray(residuals, dtype=np.float64)))

    def unmatched_detection_identifiers(
        self,
        epoch: DetectionEpoch,
    ) -> tuple[int, ...]:
        """Return detections in ``epoch`` that have no catalog-star assignment."""
        if epoch.frame_identifier != self.frame_identifier:
            raise ValueError("Detection epoch does not match identification epoch")
        matched = set(self.matched_detection_identifiers)
        return tuple(
            detection.identifier
            for detection in epoch.catalog.detections
            if detection.identifier not in matched
        )


@dataclass(frozen=True, slots=True)
class StellarIdentificationResult:
    """Sequence-wide fitted camera attitude and per-epoch stellar identities."""

    grid_to_enu: NDArray[np.float64]
    bootstrap_frame_identifier: str
    epochs: tuple[StellarIdentificationEpoch, ...]
    fit_pair_count: int
    fit_median_residual_arcmin: float
    fit_p90_residual_arcmin: float

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
        if self.fit_pair_count < 0:
            raise ValueError("fit_pair_count cannot be negative")
        for value in (
            self.fit_median_residual_arcmin,
            self.fit_p90_residual_arcmin,
        ):
            if value < 0.0 or not np.isfinite(value):
                raise ValueError(
                    "Fit residual summaries must be finite and non-negative"
                )
        object.__setattr__(self, "grid_to_enu", matrix.copy())

    @property
    def matched_star_count(self) -> int:
        """Return the number of matched catalog-star epochs in the sequence."""
        return sum(epoch.matched_star_count for epoch in self.epochs)

    @property
    def visible_star_count(self) -> int:
        """Return the number of expected catalog-star epochs in the sequence."""
        return sum(len(epoch.visible_stars) for epoch in self.epochs)

    @property
    def optical_axis_azimuth_deg(self) -> float:
        """Return fitted optical-axis azimuth eastward from north."""
        azimuth, _ = enu_to_altaz(self.grid_to_enu[:, 2])
        return azimuth

    @property
    def optical_axis_altitude_deg(self) -> float:
        """Return fitted optical-axis altitude above the local horizon."""
        _, altitude = enu_to_altaz(self.grid_to_enu[:, 2])
        return altitude

    def epoch_for(self, frame_identifier: str) -> StellarIdentificationEpoch | None:
        """Return the identification epoch for a frame identifier, if present."""
        for epoch in self.epochs:
            if epoch.frame_identifier == frame_identifier:
                return epoch
        return None

    def catalog_tracks(
        self,
        sequence: DetectionSequence,
        *,
        min_track_length: int = 2,
    ) -> ImagePlaneTrackingResult:
        """Build tracks by grouping detections carrying the same catalog label."""
        if min_track_length < 2:
            raise ValueError("min_track_length must be at least 2")
        epoch_lookup = {epoch.frame_identifier: epoch for epoch in sequence.epochs}
        grouped: dict[str, list[TrackPoint]] = defaultdict(list)
        for identification_epoch in self.epochs:
            if identification_epoch.frame_identifier not in epoch_lookup:
                raise ValueError(
                    "Identification result does not match detection sequence"
                )
            for star in identification_epoch.visible_stars:
                if star.detection_identifier is None:
                    continue
                grouped[star.catalog_identifier].append(
                    TrackPoint(
                        identification_epoch.frame_identifier,
                        int(star.detection_identifier),
                    )
                )

        tracks: list[StarTrack] = []
        epoch_index = {
            epoch.frame_identifier: index for index, epoch in enumerate(sequence.epochs)
        }
        detection_lookup = {
            epoch.frame_identifier: {
                detection.identifier: detection
                for detection in epoch.catalog.detections
            }
            for epoch in sequence.epochs
        }
        for catalog_identifier in sorted(grouped, key=str.casefold):
            points = sorted(
                grouped[catalog_identifier],
                key=lambda point: epoch_index[point.frame_identifier],
            )
            if len(points) < min_track_length:
                continue
            first_index = epoch_index[points[0].frame_identifier]
            last_index = epoch_index[points[-1].frame_identifier]
            first_epoch = sequence.epochs[first_index]
            last_epoch = sequence.epochs[last_index]
            duration_s = (
                last_epoch.exposure_midpoint - first_epoch.exposure_midpoint
            ).total_seconds()
            first_detection = detection_lookup[points[0].frame_identifier][
                points[0].detection_identifier
            ]
            last_detection = detection_lookup[points[-1].frame_identifier][
                points[-1].detection_identifier
            ]
            displacement = float(
                np.hypot(
                    last_detection.x - first_detection.x,
                    last_detection.y - first_detection.y,
                )
            )
            intervals = np.asarray(
                [
                    (
                        epoch_lookup[later.frame_identifier].exposure_midpoint
                        - epoch_lookup[earlier.frame_identifier].exposure_midpoint
                    ).total_seconds()
                    for earlier, later in zip(points, points[1:], strict=False)
                ],
                dtype=np.float64,
            )
            span_count = last_index - first_index + 1
            diagnostics = TrackDiagnostics(
                duration_s=float(duration_s),
                displacement_px=displacement,
                mean_speed_px_per_minute=(
                    0.0 if duration_s <= 0 else displacement / (duration_s / 60.0)
                ),
                fit_rms_px=0.0,
                missed_frames=max(0, span_count - len(points)),
                diagnostic_class="candidate",
                detection_count=len(points),
                span_epoch_count=span_count,
                coverage_fraction=len(points) / span_count,
                median_interval_s=(
                    0.0 if len(intervals) == 0 else float(np.median(intervals))
                ),
                max_interval_s=(
                    0.0 if len(intervals) == 0 else float(np.max(intervals))
                ),
            )
            tracks.append(
                StarTrack(
                    identifier=len(tracks),
                    points=tuple(points),
                    catalog_identifier=catalog_identifier,
                    quality=diagnostics.coverage_fraction,
                    diagnostics=diagnostics,
                )
            )
        return ImagePlaneTrackingResult(sequence, tuple(tracks))


@dataclass(frozen=True, slots=True)
class _FrameCatalogStar:
    catalog_index: int
    star: CatalogStar
    ray_enu: NDArray[np.float64]
    azimuth_deg: float
    altitude_deg: float


@dataclass(frozen=True, slots=True)
class _Projection:
    catalog_index: int
    x: float
    y: float


@dataclass(frozen=True, slots=True)
class _Pair:
    catalog_index: int
    detection_index: int
    residual_px: float
    kind: MatchKind = "primary"


class CatalogSequenceMatcher:
    """Fit one absolute attitude and identify catalog stars in every epoch."""

    def __init__(self, config: StellarIdentificationConfig | None = None) -> None:
        self.config = config or StellarIdentificationConfig()

    def fit_and_match(
        self,
        sequence: DetectionSequence,
        calibration: GridCalibration,
        catalog_stars: tuple[CatalogStar, ...],
        *,
        field_mask: FieldMask | None = None,
        field_mask_keep_margin_px: float = 0.0,
        catalog_ray_provider: CatalogRayProvider = catalog_stars_to_enu_rays,
        show_progress: bool = False,
    ) -> StellarIdentificationResult:
        """Solve a common Grid->ENU attitude and identify stars in every frame."""
        if sequence.epochs[0].image_shape != calibration.image_shape:
            raise ValueError(
                "Grid calibration sensor shape does not match the detection sequence"
            )
        if field_mask_keep_margin_px < 0.0:
            raise ValueError("field_mask_keep_margin_px cannot be negative")
        if field_mask is not None:
            field_mask.validate_against(
                calibration.image_shape,
                calibration.coordinate_convention,
            )

        stars = tuple(
            star
            for star in catalog_stars
            if star.magnitude is not None
            and star.magnitude <= self.config.limiting_magnitude
        )
        if len(stars) < self.config.min_bootstrap_matches:
            raise ValueError("Catalog contains too few stars for identification")

        acceptance = _static_acceptance_mask(
            calibration.image_shape,
            field_mask,
            field_mask_keep_margin_px,
        )
        projection_progress = tqdm(
            sequence.epochs,
            desc="Projecting catalog",
            unit="frame",
            dynamic_ncols=True,
            disable=None if show_progress else True,
        )
        frame_catalog = tuple(
            self._frame_catalog(epoch, stars, catalog_ray_provider)
            for epoch in projection_progress
        )
        bootstrap_epoch_index, attitude = self._bootstrap_sequence_attitude(
            sequence,
            frame_catalog,
            calibration,
            acceptance,
            show_progress=show_progress,
        )
        bootstrap_epoch = sequence.epochs[bootstrap_epoch_index]
        attitude = self._refine_sequence_attitude(
            sequence,
            frame_catalog,
            calibration,
            acceptance,
            attitude,
            show_progress=show_progress,
        )

        epoch_results: list[StellarIdentificationEpoch] = []
        all_primary_pairs: list[tuple[NDArray[np.float64], NDArray[np.float64]]] = []
        matching_progress = tqdm(
            zip(sequence.epochs, frame_catalog, strict=True),
            total=len(sequence.epochs),
            desc="Matching catalog",
            unit="frame",
            dynamic_ncols=True,
            disable=None if show_progress else True,
        )
        matched_so_far = 0
        for epoch, catalog_for_epoch in matching_progress:
            identification_epoch, pair_rays = self._identify_epoch(
                epoch,
                catalog_for_epoch,
                calibration,
                acceptance,
                attitude,
            )
            epoch_results.append(identification_epoch)
            all_primary_pairs.extend(pair_rays)
            matched_so_far += identification_epoch.matched_star_count
            matching_progress.set_postfix(matched=matched_so_far, refresh=False)

        residuals = _angular_residuals_arcmin(attitude, all_primary_pairs)
        if len(residuals) == 0:
            raise ValueError("Catalog identification produced no matched stars")
        return StellarIdentificationResult(
            grid_to_enu=attitude,
            bootstrap_frame_identifier=bootstrap_epoch.frame_identifier,
            epochs=tuple(epoch_results),
            fit_pair_count=len(all_primary_pairs),
            fit_median_residual_arcmin=float(np.median(residuals)),
            fit_p90_residual_arcmin=float(np.percentile(residuals, 90)),
        )

    def _bootstrap_sequence_attitude(
        self,
        sequence: DetectionSequence,
        frame_catalog: tuple[tuple[_FrameCatalogStar, ...], ...],
        calibration: GridCalibration,
        acceptance: NDArray[np.bool_],
        *,
        show_progress: bool = False,
    ) -> tuple[int, NDArray[np.float64]]:
        """Select a robust bootstrap from several representative epochs.

        Twilight, clouds, aircraft, and edge artifacts can make any one frame a
        poor absolute-attitude seed.  Candidate epochs are sampled across the
        full sequence while preferring typical detection counts.  Each candidate
        attitude is then validated on an independent, temporally distributed set
        of epochs.  A genuine camera attitude should reproduce the stellar
        geometry in many frames; an accidental dense-frame registration should
        not.
        """
        bright_catalog = tuple(
            tuple(
                star
                for star in stars
                if star.star.magnitude is not None
                and star.star.magnitude <= self.config.bootstrap_limiting_magnitude
            )
            for stars in frame_catalog
        )
        eligible = tuple(
            index
            for index, (epoch, stars) in enumerate(
                zip(sequence.epochs, bright_catalog, strict=True)
            )
            if len(epoch.catalog.detections) >= self.config.min_bootstrap_matches
            and len(stars) >= self.config.min_bootstrap_matches
        )
        if not eligible:
            raise ValueError(
                "No epoch has enough detections and visible bright catalog stars "
                "to bootstrap catalog identification"
            )

        candidate_indices = _representative_epoch_indices(
            sequence,
            eligible,
            self.config.bootstrap_candidate_frame_count,
        )
        validation_indices = _representative_epoch_indices(
            sequence,
            eligible,
            self.config.bootstrap_validation_frame_count,
        )
        required_validation_frames = min(
            self.config.min_bootstrap_validation_frames,
            len(validation_indices),
        )

        best_index: int | None = None
        best_matrix: NDArray[np.float64] | None = None
        best_score = (-1, -1, -np.inf)
        best_support_frames = 0
        best_total_matches = 0
        best_median_px = np.inf
        successful_candidates = 0

        candidate_progress = tqdm(
            candidate_indices,
            desc="Bootstrapping attitude",
            unit="candidate",
            dynamic_ncols=True,
            disable=None if show_progress else True,
        )
        for index in candidate_progress:
            try:
                matrix = self._bootstrap_attitude(
                    sequence.epochs[index],
                    bright_catalog[index],
                    calibration,
                    acceptance,
                )
            except ValueError:
                continue
            successful_candidates += 1
            support_frames, total_matches, median_px = self._score_attitude(
                sequence,
                bright_catalog,
                validation_indices,
                calibration,
                acceptance,
                matrix,
            )
            score = (support_frames, total_matches, -median_px)
            if score > best_score:
                best_score = score
                best_index = index
                best_matrix = matrix
                best_support_frames = support_frames
                best_total_matches = total_matches
                best_median_px = median_px

        if best_index is None or best_matrix is None:
            raise ValueError(
                "Could not bootstrap catalog attitude from any representative epoch"
            )
        if best_support_frames < required_validation_frames:
            raise ValueError(
                "Catalog attitude bootstrap failed cross-frame validation: "
                f"best candidate matched >= {self.config.min_bootstrap_matches} "
                f"bright stars in only {best_support_frames}/"
                f"{len(validation_indices)} validation frames"
            )

        logger.info(
            "Catalog attitude bootstrap | candidates %d/%d successful | "
            "validation frames %d | selected %s | support %d/%d | "
            "validation matches %d | median residual %.2f px",
            successful_candidates,
            len(candidate_indices),
            len(validation_indices),
            sequence.epochs[best_index].source_path.name,
            best_support_frames,
            len(validation_indices),
            best_total_matches,
            best_median_px,
        )
        return best_index, best_matrix

    def _score_attitude(
        self,
        sequence: DetectionSequence,
        bright_catalog: tuple[tuple[_FrameCatalogStar, ...], ...],
        validation_indices: tuple[int, ...],
        calibration: GridCalibration,
        acceptance: NDArray[np.bool_],
        attitude: NDArray[np.float64],
    ) -> tuple[int, int, float]:
        support_frames = 0
        total_matches = 0
        residuals: list[float] = []
        for index in validation_indices:
            projections = _project_stars(
                bright_catalog[index],
                attitude,
                calibration,
                acceptance,
            )
            pairs = _one_to_one_pairs(
                projections,
                _detection_xy(sequence.epochs[index]),
                self.config.bootstrap_validation_radius_px,
            )
            if len(pairs) >= self.config.min_bootstrap_matches:
                support_frames += 1
            total_matches += len(pairs)
            residuals.extend(pair.residual_px for pair in pairs)
        median_px = (
            np.inf
            if not residuals
            else float(np.median(np.asarray(residuals, dtype=np.float64)))
        )
        return support_frames, total_matches, median_px

    def _frame_catalog(
        self,
        epoch: DetectionEpoch,
        stars: tuple[CatalogStar, ...],
        provider: CatalogRayProvider,
    ) -> tuple[_FrameCatalogStar, ...]:
        rays, altitude = provider(stars, epoch.exposure_midpoint, epoch.location)
        rays = np.asarray(rays, dtype=np.float64)
        altitude = np.asarray(altitude, dtype=np.float64)
        if rays.shape != (len(stars), 3) or altitude.shape != (len(stars),):
            raise ValueError("Catalog ray provider returned invalid array shapes")
        result = []
        for index, (star, ray, alt) in enumerate(
            zip(stars, rays, altitude, strict=True)
        ):
            if float(alt) < self.config.min_catalog_altitude_deg:
                continue
            azimuth, computed_altitude = enu_to_altaz(np.asarray(ray, dtype=np.float64))
            result.append(
                _FrameCatalogStar(
                    catalog_index=index,
                    star=star,
                    ray_enu=_unit(ray),
                    azimuth_deg=azimuth,
                    altitude_deg=computed_altitude,
                )
            )
        return tuple(result)

    def _bootstrap_attitude(
        self,
        epoch: DetectionEpoch,
        stars: tuple[_FrameCatalogStar, ...],
        calibration: GridCalibration,
        acceptance: NDArray[np.bool_],
    ) -> NDArray[np.float64]:
        detection_xy = _detection_xy(epoch)
        best_matrix: NDArray[np.float64] | None = None
        best_pairs: list[_Pair] = []
        best_score = (-1, -np.inf)
        angles = np.arange(
            0.0,
            360.0,
            self.config.bootstrap_rotation_step_deg,
            dtype=np.float64,
        )
        for roll in angles:
            matrix = _zenith_attitude(float(roll))
            projections = _project_stars(stars, matrix, calibration, acceptance)
            pairs = _one_to_one_pairs(
                projections,
                detection_xy,
                self.config.bootstrap_match_radius_px,
            )
            median = (
                np.inf
                if not pairs
                else float(np.median([pair.residual_px for pair in pairs]))
            )
            score = (len(pairs), -median)
            if score > best_score:
                best_score = score
                best_matrix = matrix
                best_pairs = pairs
        if best_matrix is None or len(best_pairs) < self.config.min_bootstrap_matches:
            raise ValueError(
                "Could not bootstrap catalog attitude; not enough gated matches"
            )

        matrix = self._attitude_from_pairs(
            epoch,
            stars,
            best_pairs,
            calibration,
            best_matrix,
        )
        for radius in self.config.refinement_radii_px:
            projections = _project_stars(stars, matrix, calibration, acceptance)
            pairs = _one_to_one_pairs(projections, detection_xy, radius)
            if len(pairs) < self.config.min_bootstrap_matches:
                break
            matrix = self._attitude_from_pairs(
                epoch,
                stars,
                pairs,
                calibration,
                matrix,
            )
        return matrix

    def _attitude_from_pairs(
        self,
        epoch: DetectionEpoch,
        stars: tuple[_FrameCatalogStar, ...],
        pairs: list[_Pair],
        calibration: GridCalibration,
        previous: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        detection_xy = _detection_xy(epoch)
        catalog_by_index = {star.catalog_index: star for star in stars}
        observations = []
        for pair in pairs:
            ray = _detection_grid_ray(
                calibration,
                detection_xy[pair.detection_index, 0],
                detection_xy[pair.detection_index, 1],
            )
            if ray is None:
                continue
            observations.append((ray, catalog_by_index[pair.catalog_index].ray_enu))
        if len(observations) < self.config.min_bootstrap_matches:
            return previous
        return _robust_rotation(
            observations,
            previous,
            self.config.robust_clip_sigma,
            self.config.robust_clip_floor_arcmin,
            self.config.min_bootstrap_matches,
        )

    def _refine_sequence_attitude(
        self,
        sequence: DetectionSequence,
        frame_catalog: tuple[tuple[_FrameCatalogStar, ...], ...],
        calibration: GridCalibration,
        acceptance: NDArray[np.bool_],
        initial: NDArray[np.float64],
        *,
        show_progress: bool = False,
    ) -> NDArray[np.float64]:
        matrix = np.asarray(initial, dtype=np.float64)
        for iteration in range(self.config.sequence_refinement_iterations):
            observations: list[tuple[NDArray[np.float64], NDArray[np.float64]]] = []
            refinement_progress = tqdm(
                zip(sequence.epochs, frame_catalog, strict=True),
                total=len(sequence.epochs),
                desc=(
                    "Refining attitude "
                    f"{iteration + 1}/{self.config.sequence_refinement_iterations}"
                ),
                unit="frame",
                dynamic_ncols=True,
                disable=None if show_progress else True,
            )
            for epoch, stars in refinement_progress:
                bright = tuple(
                    star
                    for star in stars
                    if star.star.magnitude is not None
                    and star.star.magnitude <= self.config.bootstrap_limiting_magnitude
                )
                projections = _project_stars(bright, matrix, calibration, acceptance)
                pairs = _one_to_one_pairs(
                    projections,
                    _detection_xy(epoch),
                    self.config.sequence_refinement_radius_px,
                )
                by_index = {star.catalog_index: star for star in bright}
                detection_xy = _detection_xy(epoch)
                for pair in pairs:
                    ray_grid = _detection_grid_ray(
                        calibration,
                        detection_xy[pair.detection_index, 0],
                        detection_xy[pair.detection_index, 1],
                    )
                    if ray_grid is None:
                        continue
                    observations.append(
                        (ray_grid, by_index[pair.catalog_index].ray_enu)
                    )
            if len(observations) < self.config.min_bootstrap_matches:
                break
            if len(observations) > self.config.max_refinement_pairs:
                indices = np.linspace(
                    0,
                    len(observations) - 1,
                    self.config.max_refinement_pairs,
                    dtype=int,
                )
                observations = [observations[index] for index in indices]
            updated = _robust_rotation(
                observations,
                matrix,
                self.config.robust_clip_sigma,
                self.config.robust_clip_floor_arcmin,
                self.config.min_bootstrap_matches,
            )
            if _rotation_difference_deg(matrix, updated) < 1e-7:
                matrix = updated
                break
            matrix = updated
        return matrix

    def _identify_epoch(
        self,
        epoch: DetectionEpoch,
        stars: tuple[_FrameCatalogStar, ...],
        calibration: GridCalibration,
        acceptance: NDArray[np.bool_],
        attitude: NDArray[np.float64],
    ) -> tuple[
        StellarIdentificationEpoch,
        list[tuple[NDArray[np.float64], NDArray[np.float64]]],
    ]:
        detection_xy = _detection_xy(epoch)
        projections = _project_stars(stars, attitude, calibration, acceptance)
        primary = _one_to_one_pairs(
            projections,
            detection_xy,
            self.config.final_match_radius_px,
        )
        all_pairs = list(primary)
        used_catalog = {pair.catalog_index for pair in primary}
        used_detections = {pair.detection_index for pair in primary}
        rescue_stars = tuple(
            star
            for star in stars
            if star.catalog_index not in used_catalog
            and star.star.magnitude is not None
            and star.star.magnitude <= self.config.bright_rescue_magnitude
        )
        remaining_detection_indices = np.asarray(
            [
                index
                for index in range(len(detection_xy))
                if index not in used_detections
            ],
            dtype=int,
        )
        if rescue_stars and len(remaining_detection_indices):
            rescue_projections = _project_stars(
                rescue_stars,
                attitude,
                calibration,
                acceptance,
            )
            rescue_pairs = _one_to_one_pairs(
                rescue_projections,
                detection_xy[remaining_detection_indices],
                self.config.bright_rescue_radius_px,
            )
            all_pairs.extend(
                _Pair(
                    pair.catalog_index,
                    int(remaining_detection_indices[pair.detection_index]),
                    pair.residual_px,
                    "bright-rescue",
                )
                for pair in rescue_pairs
            )

        pair_by_catalog = {pair.catalog_index: pair for pair in all_pairs}
        projection_by_catalog = {
            projection.catalog_index: projection for projection in projections
        }
        star_by_index = {star.catalog_index: star for star in stars}
        entries: list[StellarIdentification] = []
        pair_rays: list[tuple[NDArray[np.float64], NDArray[np.float64]]] = []
        for catalog_index in sorted(projection_by_catalog):
            star = star_by_index[catalog_index]
            projection = projection_by_catalog[catalog_index]
            pair = pair_by_catalog.get(catalog_index)
            detection_identifier: int | None = None
            residual_px: float | None = None
            residual_arcmin: float | None = None
            kind: MatchKind | None = None
            if pair is not None:
                detection = epoch.catalog.detections[pair.detection_index]
                detection_identifier = detection.identifier
                residual_px = pair.residual_px
                ray_grid = _detection_grid_ray(calibration, detection.x, detection.y)
                if ray_grid is not None:
                    residual_arcmin = 60.0 * _angular_separation_deg(
                        attitude @ ray_grid,
                        star.ray_enu,
                    )
                    pair_rays.append((ray_grid, star.ray_enu))
                kind = pair.kind
            entries.append(
                StellarIdentification(
                    catalog_identifier=star.star.identifier,
                    catalog_magnitude=star.star.magnitude,
                    azimuth_deg=star.azimuth_deg,
                    altitude_deg=star.altitude_deg,
                    predicted_x=projection.x,
                    predicted_y=projection.y,
                    detection_identifier=detection_identifier,
                    residual_px=residual_px,
                    residual_arcmin=residual_arcmin,
                    match_kind=kind,
                )
            )
        return (
            StellarIdentificationEpoch(epoch.frame_identifier, tuple(entries)),
            pair_rays,
        )


def _representative_epoch_indices(
    sequence: DetectionSequence,
    eligible: tuple[int, ...],
    count: int,
) -> tuple[int, ...]:
    """Return temporally distributed epochs with typical detection populations."""
    if len(eligible) <= count:
        return eligible

    detection_counts = np.asarray(
        [len(sequence.epochs[index].catalog.detections) for index in eligible],
        dtype=np.float64,
    )
    typical_count = float(np.median(detection_counts))
    n_epochs = len(sequence.epochs)
    edges = np.linspace(0.0, float(n_epochs), count + 1)
    selected: list[int] = []
    eligible_set = set(eligible)
    for left, right in zip(edges[:-1], edges[1:], strict=True):
        start = int(np.floor(left))
        stop = max(start + 1, int(np.ceil(right)))
        candidates = [
            index
            for index in range(start, min(stop, n_epochs))
            if index in eligible_set
        ]
        if not candidates:
            continue
        center = 0.5 * (left + right - 1.0)
        selected.append(
            min(
                candidates,
                key=lambda index: (
                    abs(len(sequence.epochs[index].catalog.detections) - typical_count),
                    abs(index - center),
                    index,
                ),
            )
        )

    if not selected:
        return (eligible[len(eligible) // 2],)
    return tuple(dict.fromkeys(selected))


def _static_acceptance_mask(
    image_shape: tuple[int, int],
    field_mask: FieldMask | None,
    keep_margin_px: float,
) -> NDArray[np.bool_]:
    if field_mask is None:
        return np.ones(image_shape, dtype=np.bool_)
    allowed = ~np.asarray(field_mask.excluded, dtype=np.bool_)
    if keep_margin_px <= 0.0:
        return allowed
    return allowed & (distance_transform_edt(allowed) > keep_margin_px)


def _rotation_z_deg(angle_deg: float) -> NDArray[np.float64]:
    angle = np.deg2rad(angle_deg)
    c = np.cos(angle)
    s = np.sin(angle)
    return np.asarray([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _zenith_attitude(roll_deg: float) -> NDArray[np.float64]:
    return np.asarray(_BASE_GRID_TO_ENU @ _rotation_z_deg(-roll_deg), dtype=np.float64)


def _project_stars(
    stars: tuple[_FrameCatalogStar, ...],
    grid_to_enu: NDArray[np.float64],
    calibration: GridCalibration,
    acceptance: NDArray[np.bool_],
) -> list[_Projection]:
    transform = calibration.transform
    if not isinstance(transform, PortableGridTransform):
        raise TypeError("Catalog identification requires a portable Grid transform")
    height, width = calibration.image_shape
    result: list[_Projection] = []
    enu_to_grid = grid_to_enu.T
    for star in stars:
        pixel = transform.ray_to_pixel(enu_to_grid @ star.ray_enu)
        if pixel is None:
            continue
        x, y = float(pixel[0]), float(pixel[1])
        if not (0.0 <= x < width and 0.0 <= y < height):
            continue
        xi = int(np.rint(x))
        yi = int(np.rint(y))
        if xi < 0 or yi < 0 or xi >= width or yi >= height or not acceptance[yi, xi]:
            continue
        result.append(_Projection(star.catalog_index, x, y))
    return result


def _detection_xy(epoch: DetectionEpoch) -> NDArray[np.float64]:
    return np.asarray(
        [[item.x, item.y] for item in epoch.catalog.detections],
        dtype=np.float64,
    ).reshape((-1, 2))


def _one_to_one_pairs(
    projections: list[_Projection],
    detection_xy: NDArray[np.float64],
    radius_px: float,
) -> list[_Pair]:
    """Maximum-cardinality gated assignment followed by minimum separation."""
    if not projections or len(detection_xy) == 0:
        return []
    catalog_xy = np.asarray(
        [[item.x, item.y] for item in projections], dtype=np.float64
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
    result = []
    for row, column in zip(rows, columns, strict=False):
        if row >= n_catalog or column >= n_detection or not valid[row, column]:
            continue
        result.append(
            _Pair(
                projections[int(row)].catalog_index,
                int(column),
                float(real_cost[row, column]),
            )
        )
    return result


def _detection_grid_ray(
    calibration: GridCalibration,
    x: float,
    y: float,
) -> NDArray[np.float64] | None:
    converted = validated_pixel_rays(
        calibration,
        np.asarray([x], dtype=np.float64),
        np.asarray([y], dtype=np.float64),
    )
    if not bool(converted.valid[0]):
        return None
    return _unit(converted.rays[0])


def _solve_rotation(
    observations: list[tuple[NDArray[np.float64], NDArray[np.float64]]],
) -> NDArray[np.float64]:
    """Wahba/Kabsch solution mapping Grid rays to ENU rays."""
    grid = np.asarray([item[0] for item in observations], dtype=np.float64)
    enu = np.asarray([item[1] for item in observations], dtype=np.float64)
    cross_covariance = enu.T @ grid
    u, _, vt = np.linalg.svd(cross_covariance)
    correction = np.eye(3)
    correction[2, 2] = np.sign(np.linalg.det(u @ vt))
    return np.asarray(u @ correction @ vt, dtype=np.float64)


def _robust_rotation(
    observations: list[tuple[NDArray[np.float64], NDArray[np.float64]]],
    initial: NDArray[np.float64],
    clip_sigma: float,
    clip_floor_arcmin: float,
    minimum: int,
) -> NDArray[np.float64]:
    if len(observations) < minimum:
        return initial
    selected = list(observations)
    matrix = _solve_rotation(selected)
    for _ in range(5):
        residuals = _angular_residuals_arcmin(matrix, selected)
        median = float(np.median(residuals))
        mad = float(np.median(np.abs(residuals - median)))
        robust_sigma = 1.4826 * mad
        limit = max(clip_floor_arcmin, median + clip_sigma * robust_sigma)
        keep = residuals <= limit
        if int(np.count_nonzero(keep)) < minimum or np.all(keep):
            break
        selected = [item for item, retain in zip(selected, keep, strict=True) if retain]
        matrix = _solve_rotation(selected)
    return matrix


def _angular_residuals_arcmin(
    grid_to_enu: NDArray[np.float64],
    observations: list[tuple[NDArray[np.float64], NDArray[np.float64]]],
) -> NDArray[np.float64]:
    if not observations:
        return np.empty(0, dtype=np.float64)
    values = [
        60.0 * _angular_separation_deg(grid_to_enu @ grid, enu)
        for grid, enu in observations
    ]
    return np.asarray(values, dtype=np.float64)


def _angular_separation_deg(
    first: NDArray[np.float64], second: NDArray[np.float64]
) -> float:
    a = _unit(first)
    b = _unit(second)
    return float(np.rad2deg(np.arccos(np.clip(np.dot(a, b), -1.0, 1.0))))


def _rotation_difference_deg(
    first: NDArray[np.float64], second: NDArray[np.float64]
) -> float:
    delta = np.asarray(second) @ np.asarray(first).T
    cosine = np.clip((np.trace(delta) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.rad2deg(np.arccos(cosine)))


def _unit(vector: NDArray[np.float64]) -> NDArray[np.float64]:
    parsed = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(parsed))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("Direction vector must be finite and non-zero")
    return parsed / norm
