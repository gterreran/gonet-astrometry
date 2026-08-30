"""Independent native-channel SEP detection for Grid-assisted astrometry.

The full GONet Bayer mosaic is never interpolated for source finding.  Blue,
green1, green2, and red are extracted as four compact images, normalized with
independent local background/RMS models, detected independently with SEP, and
then softly associated in full-sensor coordinates.

The association is deliberately permissive: one-channel candidates are retained.
Channel multiplicity is carried as metadata for later temporal association rather
than used as a hard source-rejection criterion.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Protocol

import numpy as np
from astropy.stats import SigmaClip
from matplotlib.path import Path as MplPath
from numpy.typing import NDArray
from photutils.background import (
    Background2D,
    MADStdBackgroundRMS,
    MedianBackground,
)
from scipy.ndimage import distance_transform_edt

from gonet_astrometry.adapters.gonet_wizard import (
    GONET_CHANNELS,
    GONetChannel,
)
from gonet_astrometry.adapters.grid_calibration import PortableGridTransform
from gonet_astrometry.detection.config import DetectionConfig
from gonet_astrometry.detection.field_mask import FieldMask
from gonet_astrometry.detection.preprocessing import (
    PreparedDetectionImage,
    estimate_provisional_field_mask,
)
from gonet_astrometry.detection.sep_backend import SEPDetector
from gonet_astrometry.models.detection import Detection, DetectionCatalog
from gonet_astrometry.models.frame import ImageFrame
from gonet_astrometry.models.grid import GridCalibration

CHANNEL_OFFSETS: dict[GONetChannel, tuple[int, int]] = {
    "blue": (0, 0),
    "green1": (0, 1),
    "green2": (1, 0),
    "red": (1, 1),
}
"""Native BGGR row/column parity of each compact GONet channel."""


@dataclass(frozen=True, slots=True)
class MultiChannelSEPConfig:
    """Configuration specific to independent-channel source recovery.

    Parameters
    ----------
    field_edge_keep_margin_px
        Additional full-sensor distance required inside the automatically
        inferred illuminated footprint before a detection is retained. The
        footprint itself is controlled by :class:`DetectionConfig`, including
        ``footprint_threshold_fraction`` and ``footprint_erosion_px``. A value
        of zero accepts detections anywhere inside that conservative footprint.
        This margin is ignored when automatic footprint detection is disabled.
    field_mask_keep_margin_px
        Additional full-sensor distance required inside the optional static
        field mask before a detection is retained. Search/background estimation
        may still use the complete static mask; only final centroid acceptance is
        eroded. A value of zero accepts detections up to the static mask boundary.
    grid_search_radius_deg
        Optional Grid-calibrated angular-radius cap applied to the search field.
        ``None`` leaves the search geometry entirely image-driven.
    grid_acceptance_radius_deg
        Optional Grid-calibrated angular-radius cap applied to retained
        detections. ``None`` leaves acceptance entirely image-driven.
    grid_contour_samples
        Samples used to rasterize optional Grid-radius caps. The default matches
        the historical 75/70-degree implementation.
    channel_match_radius_px
        Maximum full-sensor centroid separation used to associate detections
        from different Bayer channels.
    background_box_size_compact_px
        ``Background2D`` box width/height in compact-channel pixels.
    minimum_channel_support
        Minimum number of native channels required for a final candidate.
        Production defaults to one: channel support is evidence, not a veto.
    """

    field_edge_keep_margin_px: float = 0.0
    field_mask_keep_margin_px: float = 0.0
    grid_search_radius_deg: float | None = None
    grid_acceptance_radius_deg: float | None = None
    grid_contour_samples: int = 2160
    channel_match_radius_px: float = 6.0
    background_box_size_compact_px: int = 128
    minimum_channel_support: int = 1

    def __post_init__(self) -> None:
        if self.field_edge_keep_margin_px < 0.0:
            raise ValueError("field_edge_keep_margin_px cannot be negative")
        if self.field_mask_keep_margin_px < 0.0:
            raise ValueError("field_mask_keep_margin_px cannot be negative")
        if (
            self.grid_search_radius_deg is not None
            and self.grid_search_radius_deg <= 0.0
        ):
            raise ValueError("grid_search_radius_deg must be positive or None")
        if (
            self.grid_acceptance_radius_deg is not None
            and self.grid_acceptance_radius_deg <= 0.0
        ):
            raise ValueError("grid_acceptance_radius_deg must be positive or None")
        if (
            self.grid_search_radius_deg is not None
            and self.grid_acceptance_radius_deg is not None
            and self.grid_acceptance_radius_deg > self.grid_search_radius_deg
        ):
            raise ValueError(
                "grid_acceptance_radius_deg cannot exceed grid_search_radius_deg"
            )
        if self.grid_contour_samples < 360:
            raise ValueError("grid_contour_samples must be at least 360")
        if self.channel_match_radius_px <= 0.0:
            raise ValueError("channel_match_radius_px must be positive")
        if self.background_box_size_compact_px < 4:
            raise ValueError("background_box_size_compact_px must be at least 4")
        if not 1 <= self.minimum_channel_support <= 4:
            raise ValueError("minimum_channel_support must lie in [1, 4]")


@dataclass(frozen=True, slots=True)
class ChannelDetection:
    """One channel-specific SEP source expressed in full-sensor coordinates."""

    channel: GONetChannel
    x: float
    y: float
    x_uncertainty: float
    y_uncertainty: float
    signal_to_noise: float
    flux: float


@dataclass(frozen=True, slots=True)
class FusedChannelDetection:
    """Soft association of one to four native-channel source detections."""

    x: float
    y: float
    x_uncertainty: float
    y_uncertainty: float
    channels: tuple[GONetChannel, ...]
    combined_signal_to_noise: float
    combined_flux: float

    @property
    def channel_support(self) -> int:
        """Return the number of represented native Bayer channels."""
        return len(self.channels)


class _PreparedDetector(Protocol):
    def detect_prepared(
        self,
        frame_identifier: str,
        prepared: PreparedDetectionImage,
    ) -> DetectionCatalog:
        """Detect sources in an already normalized compact-channel image."""
        ...


class IndependentChannelSEPDetector:
    """Run SEP independently on the four native Bayer channels and fuse softly."""

    name = "sep-independent-channels"

    def __init__(
        self,
        calibration: GridCalibration,
        detection_config: DetectionConfig | None = None,
        multichannel_config: MultiChannelSEPConfig | None = None,
        *,
        field_mask: FieldMask | None = None,
        prepared_detector: _PreparedDetector | None = None,
    ) -> None:
        self.calibration = calibration
        self.detection_config = detection_config or DetectionConfig()
        self.multichannel_config = multichannel_config or MultiChannelSEPConfig()
        self.compact_detection_config = compact_detection_config(self.detection_config)
        self._detector = (
            prepared_detector
            if prepared_detector is not None
            else SEPDetector(self.compact_detection_config)
        )

        self.field_mask = field_mask
        if self.field_mask is not None:
            self.field_mask.validate_against(
                self.calibration.image_shape,
                self.calibration.coordinate_convention,
            )

        self._grid_search_mask = self._optional_grid_radius_mask(
            self.multichannel_config.grid_search_radius_deg
        )
        self._grid_acceptance_mask = self._optional_grid_radius_mask(
            self.multichannel_config.grid_acceptance_radius_deg
        )

    def detect(
        self,
        frame_identifier: str,
        frame: ImageFrame,
    ) -> DetectionCatalog:
        """Detect and softly fuse candidates from one native full Bayer frame."""
        if frame.shape != self.calibration.image_shape:
            raise ValueError(
                "Grid calibration sensor shape does not match the image frame: "
                f"calibration={self.calibration.image_shape}, frame={frame.shape}"
            )
        if frame.shape[0] % 2 or frame.shape[1] % 2:
            raise ValueError(
                "Independent-channel detection requires even sensor dimensions"
            )

        search_full, acceptance_full = self._frame_field_masks(frame)
        by_channel: dict[
            GONetChannel,
            tuple[ChannelDetection, ...],
        ] = {}

        for channel in GONET_CHANNELS:
            row_offset, column_offset = CHANNEL_OFFSETS[channel]
            compact = np.asarray(
                frame.data[row_offset::2, column_offset::2],
                dtype=np.float64,
            )
            compact_search = np.asarray(
                search_full[row_offset::2, column_offset::2],
                dtype=np.bool_,
            )
            if compact_search.shape != compact.shape:
                raise ValueError(
                    "Automatic field mask compact sampling does not match "
                    "channel shape"
                )
            prepared = prepare_compact_channel(
                compact,
                compact_search,
                background_box_size=(
                    self.multichannel_config.background_box_size_compact_px
                ),
            )
            catalog = self._detector.detect_prepared(
                f"{frame_identifier}:{channel}",
                prepared,
            )
            accepted = filter_channel_detections(
                catalog.detections,
                channel,
                acceptance_full,
            )
            by_channel[channel] = tuple(
                compact_detection_to_full(item, channel) for item in accepted
            )

        fused = fuse_channel_detections(
            by_channel,
            match_radius_px=self.multichannel_config.channel_match_radius_px,
            min_channels=self.multichannel_config.minimum_channel_support,
        )
        detections = tuple(
            Detection(
                identifier=index,
                x=item.x,
                y=item.y,
                flux=item.combined_flux,
                signal_to_noise=item.combined_signal_to_noise,
                x_uncertainty=item.x_uncertainty,
                y_uncertainty=item.y_uncertainty,
                flags=(
                    f"channel-support:{item.channel_support}",
                    "channels:" + ",".join(item.channels),
                ),
            )
            for index, item in enumerate(fused, start=1)
        )
        return DetectionCatalog(
            frame_identifier=frame_identifier,
            detections=detections,
            detector_name=self.name,
        )

    def prepare_report_image(self, frame: ImageFrame) -> PreparedDetectionImage:
        """Return full-sensor masks matching the multichannel acceptance field.

        This product is for report rendering only. Scientific source extraction
        continues to use independently normalized compact channels.
        """
        if frame.shape != self.calibration.image_shape:
            raise ValueError(
                "Grid calibration sensor shape does not match the image frame: "
                f"calibration={self.calibration.image_shape}, frame={frame.shape}"
            )

        data = np.asarray(frame.data, dtype=np.float64)
        _, acceptance_full = self._frame_field_masks(frame)
        display = np.where(acceptance_full, data, 0.0)
        finite = data[acceptance_full]
        background = float(np.median(finite)) if finite.size else 0.0
        noise = robust_fallback_noise(finite)

        return PreparedDetectionImage(
            data=np.asarray(display, dtype=np.float64),
            mask=np.asarray(~acceptance_full, dtype=np.bool_),
            field_mask=np.asarray(acceptance_full, dtype=np.bool_),
            dynamic_mask=np.zeros(frame.shape, dtype=np.bool_),
            backgrounds=(background,) * 4,
            noises=(noise,) * 4,
        )

    def _frame_field_masks(
        self,
        frame: ImageFrame,
    ) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
        """Return per-frame search and acceptance masks in full-sensor pixels."""
        data = np.asarray(frame.data, dtype=np.float64)
        invalid = ~np.isfinite(data)
        if frame.mask is not None:
            invalid |= np.asarray(frame.mask, dtype=np.bool_)

        if self.detection_config.use_provisional_field_mask:
            automatic = estimate_provisional_field_mask(
                data,
                invalid,
                self.detection_config,
            )
        else:
            automatic = np.ones(frame.shape, dtype=np.bool_)

        search = np.asarray(automatic & ~invalid, dtype=np.bool_)

        keep_margin = self.multichannel_config.field_edge_keep_margin_px
        if self.detection_config.use_provisional_field_mask and keep_margin > 0.0:
            acceptance = distance_transform_edt(automatic) > keep_margin
        else:
            acceptance = automatic.copy()
        acceptance = np.asarray(acceptance & ~invalid, dtype=np.bool_)

        if self._grid_search_mask is not None:
            search &= self._grid_search_mask
        if self._grid_acceptance_mask is not None:
            acceptance &= self._grid_acceptance_mask

        if self.field_mask is not None:
            allowed = ~self.field_mask.excluded
            search &= allowed

            mask_keep_margin = self.multichannel_config.field_mask_keep_margin_px
            if mask_keep_margin > 0.0:
                acceptance &= distance_transform_edt(allowed) > mask_keep_margin
            else:
                acceptance &= allowed

        # A source cannot be accepted from pixels that were excluded from the
        # search/background stage, even when only one Grid-radius cap is set.
        acceptance &= search

        if not np.any(search):
            raise ValueError("Configured field masks produced an empty search field")
        if not np.any(acceptance):
            raise ValueError(
                "Configured field masks produced an empty acceptance field"
            )
        return search, acceptance

    def _optional_grid_radius_mask(
        self,
        radius_deg: float | None,
    ) -> NDArray[np.bool_] | None:
        """Rasterize one optional Grid angular-radius cap in full-sensor pixels."""
        if radius_deg is None:
            return None

        transform = self.calibration.transform
        if not isinstance(transform, PortableGridTransform):
            raise TypeError("Grid-radius field caps require a portable Grid transform")
        _, calibrated_max = transform.evaluator.calibrated_angular_range_deg
        if radius_deg > float(calibrated_max) + 1.0e-9:
            raise ValueError(
                f"Grid-radius cap {radius_deg:g} deg exceeds calibrated angular "
                f"range {float(calibrated_max):g} deg"
            )

        theta = np.linspace(
            0.0,
            360.0,
            self.multichannel_config.grid_contour_samples,
            endpoint=False,
            dtype=np.float64,
        )
        radius = np.full(theta.shape, radius_deg, dtype=np.float64)
        x, y = transform.evaluator.angle_to_pixel(
            r_deg=radius,
            theta_deg=theta,
        )
        return rasterize_inside_contour(
            self.calibration.image_shape,
            np.asarray(x, dtype=np.float64),
            np.asarray(y, dtype=np.float64),
        )


def compact_detection_config(config: DetectionConfig) -> DetectionConfig:
    """Scale full-sensor spatial detector parameters to compact sampling."""
    return replace(
        config,
        fwhm_px=max(0.5, config.fwhm_px / 2.0),
        min_separation_px=max(0.5, config.min_separation_px / 2.0),
        min_pixels=max(2, int(np.ceil(config.min_pixels / 4.0))),
        use_provisional_field_mask=False,
        bright_mask_sigma=None,
    )


def full_to_compact(
    x_full: NDArray[np.float64] | float,
    y_full: NDArray[np.float64] | float,
    channel: GONetChannel,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Convert full Bayer coordinates to one compact native channel."""
    row_offset, column_offset = CHANNEL_OFFSETS[channel]
    return (
        (np.asarray(x_full, dtype=np.float64) - column_offset) / 2.0,
        (np.asarray(y_full, dtype=np.float64) - row_offset) / 2.0,
    )


def compact_to_full(
    x_compact: NDArray[np.float64] | float,
    y_compact: NDArray[np.float64] | float,
    channel: GONetChannel,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Convert compact native-channel coordinates to full Bayer coordinates."""
    row_offset, column_offset = CHANNEL_OFFSETS[channel]
    return (
        2.0 * np.asarray(x_compact, dtype=np.float64) + column_offset,
        2.0 * np.asarray(y_compact, dtype=np.float64) + row_offset,
    )


def rasterize_inside_contour(
    shape: tuple[int, int],
    x_contour: NDArray[np.float64],
    y_contour: NDArray[np.float64],
    *,
    rows_per_chunk: int = 128,
) -> NDArray[np.bool_]:
    """Rasterize one Grid-calibrated angular contour into an image mask."""
    height, width = shape
    polygon = MplPath(
        np.column_stack((x_contour, y_contour)),
        closed=True,
    )
    inside = np.zeros(shape, dtype=np.bool_)
    x = np.arange(width, dtype=np.float64)

    for row_start in range(0, height, rows_per_chunk):
        row_stop = min(height, row_start + rows_per_chunk)
        y = np.arange(row_start, row_stop, dtype=np.float64)
        xx, yy = np.meshgrid(x, y)
        points = np.column_stack((xx.ravel(), yy.ravel()))
        inside[row_start:row_stop] = polygon.contains_points(
            points,
            radius=1.0e-9,
        ).reshape(row_stop - row_start, width)

    return inside


def robust_fallback_noise(values: NDArray[np.float64]) -> float:
    """Return a robust finite positive global fallback noise estimate."""
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 1.0

    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    noise = 1.482602218505602 * mad
    if not np.isfinite(noise) or noise <= 0.0:
        noise = float(np.std(values))
    if not np.isfinite(noise) or noise <= 0.0:
        noise = 1.0
    return noise


def prepare_compact_channel(
    data: NDArray[np.float64],
    inside_search: NDArray[np.bool_],
    *,
    background_box_size: int,
) -> PreparedDetectionImage:
    """Build an independently normalized local-significance compact channel."""
    data = np.asarray(data, dtype=np.float64)
    inside_search = np.asarray(inside_search, dtype=np.bool_)
    if data.shape != inside_search.shape:
        raise ValueError("Compact channel and search mask shapes must match")
    if not np.any(inside_search):
        raise ValueError("Compact-channel search mask contains no usable pixels")

    invalid = ~np.isfinite(data)
    mask = invalid | ~inside_search

    background = Background2D(
        data,
        box_size=(background_box_size, background_box_size),
        filter_size=(3, 3),
        mask=mask,
        sigma_clip=SigmaClip(sigma=3.0, maxiters=4),
        bkg_estimator=MedianBackground(),
        bkgrms_estimator=MADStdBackgroundRMS(),
        exclude_percentile=50.0,
    )
    background_map = np.asarray(background.background, dtype=np.float64)
    noise_map = np.asarray(background.background_rms, dtype=np.float64)

    usable_noise = np.isfinite(noise_map) & (noise_map > 0.0) & inside_search
    if np.any(usable_noise):
        fallback_noise = float(np.median(noise_map[usable_noise]))
    else:
        fallback_noise = robust_fallback_noise(data[inside_search & ~invalid])

    safe_noise = np.where(
        np.isfinite(noise_map) & (noise_map > 0.0),
        noise_map,
        fallback_noise,
    )
    significance = np.asarray(
        (data - background_map) / safe_noise,
        dtype=np.float64,
    )
    significance[mask] = 0.0

    valid_background = background_map[inside_search]
    valid_noise = safe_noise[inside_search]
    median_background = float(np.nanmedian(valid_background))
    median_noise = float(np.nanmedian(valid_noise))

    return PreparedDetectionImage(
        data=significance,
        mask=mask,
        field_mask=inside_search.copy(),
        dynamic_mask=np.zeros(data.shape, dtype=np.bool_),
        backgrounds=(median_background,) * 4,
        noises=(median_noise,) * 4,
    )


def filter_channel_detections(
    detections: tuple[Detection, ...],
    channel: GONetChannel,
    acceptance_mask_full: NDArray[np.bool_],
) -> tuple[Detection, ...]:
    """Reject channel detections outside the per-frame acceptance field."""
    if not detections:
        return detections

    accepted: list[Detection] = []
    height, width = acceptance_mask_full.shape
    for detection in detections:
        x_full, y_full = compact_to_full(detection.x, detection.y, channel)
        x_index = int(np.rint(float(x_full)))
        y_index = int(np.rint(float(y_full)))
        if not (0 <= x_index < width and 0 <= y_index < height):
            continue
        if bool(acceptance_mask_full[y_index, x_index]):
            accepted.append(detection)
    return tuple(accepted)


def compact_detection_to_full(
    detection: Detection,
    channel: GONetChannel,
) -> ChannelDetection:
    """Map one backend compact-channel detection into full sensor coordinates."""
    x_full, y_full = compact_to_full(
        detection.x,
        detection.y,
        channel,
    )
    return ChannelDetection(
        channel=channel,
        x=float(x_full),
        y=float(y_full),
        x_uncertainty=2.0 * float(detection.x_uncertainty),
        y_uncertainty=2.0 * float(detection.y_uncertainty),
        signal_to_noise=float(detection.signal_to_noise),
        flux=float(detection.flux),
    )


def weighted_coordinate(
    values: NDArray[np.float64],
    uncertainties: NDArray[np.float64],
) -> tuple[float, float]:
    """Return inverse-variance weighted coordinate and propagated uncertainty."""
    values = np.asarray(values, dtype=np.float64)
    uncertainties = np.asarray(uncertainties, dtype=np.float64)
    valid = np.isfinite(values) & np.isfinite(uncertainties) & (uncertainties > 0.0)
    if not np.any(valid):
        mean = float(np.nanmean(values))
        return mean, 1.0

    weights = 1.0 / np.square(uncertainties[valid])
    return (
        float(np.sum(values[valid] * weights) / np.sum(weights)),
        float(np.sqrt(1.0 / np.sum(weights))),
    )


def fuse_channel_detections(
    by_channel: dict[GONetChannel, tuple[ChannelDetection, ...]],
    *,
    match_radius_px: float,
    min_channels: int = 1,
) -> tuple[FusedChannelDetection, ...]:
    """Softly associate detections across native channels.

    High-S/N sources seed groups first.  At most one unused detection from each
    other channel is attached to a seed, using the nearest candidate inside the
    full-sensor match radius.  A group is consumed only when it satisfies
    ``min_channels``.
    """
    if not 1 <= min_channels <= 4:
        raise ValueError("min_channels must lie in [1, 4]")
    if match_radius_px <= 0.0:
        raise ValueError("match_radius_px must be positive")

    all_detections = [
        detection
        for channel in GONET_CHANNELS
        for detection in by_channel.get(channel, ())
    ]
    ordered = sorted(
        all_detections,
        key=lambda item: item.signal_to_noise,
        reverse=True,
    )

    indexed: dict[
        GONetChannel,
        dict[int, ChannelDetection],
    ] = {
        channel: dict(enumerate(by_channel.get(channel, ())))
        for channel in GONET_CHANNELS
    }
    reverse_index: dict[int, tuple[GONetChannel, int]] = {}
    for channel in GONET_CHANNELS:
        for index, item in indexed[channel].items():
            reverse_index[id(item)] = (channel, index)

    used: set[tuple[GONetChannel, int]] = set()
    fused: list[FusedChannelDetection] = []

    for seed in ordered:
        seed_key = reverse_index[id(seed)]
        if seed_key in used:
            continue

        members = [seed]
        member_keys = [seed_key]

        for channel in GONET_CHANNELS:
            if channel == seed.channel:
                continue

            choices: list[
                tuple[
                    float,
                    tuple[GONetChannel, int],
                    ChannelDetection,
                ]
            ] = []
            for index, candidate in indexed[channel].items():
                key = (channel, index)
                if key in used:
                    continue
                separation = float(
                    np.hypot(
                        seed.x - candidate.x,
                        seed.y - candidate.y,
                    )
                )
                if separation <= match_radius_px:
                    choices.append((separation, key, candidate))

            if choices:
                _, key, candidate = min(
                    choices,
                    key=lambda value: value[0],
                )
                members.append(candidate)
                member_keys.append(key)

        channels = tuple(
            sorted(
                {member.channel for member in members},
                key=GONET_CHANNELS.index,
            )
        )
        if len(channels) < min_channels:
            continue

        x, x_uncertainty = weighted_coordinate(
            np.asarray([item.x for item in members], dtype=np.float64),
            np.asarray(
                [item.x_uncertainty for item in members],
                dtype=np.float64,
            ),
        )
        y, y_uncertainty = weighted_coordinate(
            np.asarray([item.y for item in members], dtype=np.float64),
            np.asarray(
                [item.y_uncertainty for item in members],
                dtype=np.float64,
            ),
        )
        combined_snr = float(
            np.sqrt(np.sum(np.square([item.signal_to_noise for item in members])))
        )
        combined_flux = float(np.sum([item.flux for item in members]))

        fused.append(
            FusedChannelDetection(
                x=x,
                y=y,
                x_uncertainty=x_uncertainty,
                y_uncertainty=y_uncertainty,
                channels=channels,
                combined_signal_to_noise=combined_snr,
                combined_flux=combined_flux,
            )
        )
        used.update(member_keys)

    return tuple(
        sorted(
            fused,
            key=lambda item: item.combined_signal_to_noise,
            reverse=True,
        )
    )
