"""Command-line entry point."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from gonet_astrometry import __version__
from gonet_astrometry.adapters.gonet_wizard import GONET_CHANNELS
from gonet_astrometry.detection.config import DetectionConfig
from gonet_astrometry.detection.multichannel import MultiChannelSEPConfig
from gonet_astrometry.detection.registry import DETECTOR_SPECS
from gonet_astrometry.solving.orientation import OrientationFitConfig
from gonet_astrometry.solving.sidereal import SiderealFitConfig
from gonet_astrometry.solving.stellar_tracks import StellarTrackMergeConfig
from gonet_astrometry.tracking.catalog_identification import StellarIdentificationConfig
from gonet_astrometry.tracking.config import TrackingConfig
from gonet_astrometry.tracking.spherical import SphericalTrackingConfig

_DETECTION_DEFAULTS = DetectionConfig()
_TRACKING_DEFAULTS = TrackingConfig()
_MULTICHANNEL_DEFAULTS = MultiChannelSEPConfig()
_SPHERICAL_DEFAULTS = SphericalTrackingConfig()
_STELLAR_MERGE_DEFAULTS = StellarTrackMergeConfig()
_SIDEREAL_DEFAULTS = SiderealFitConfig()
_ORIENTATION_DEFAULTS = OrientationFitConfig()
_STELLAR_IDENTIFICATION_DEFAULTS = StellarIdentificationConfig()


def build_parser() -> argparse.ArgumentParser:
    """Construct the command-line parser."""
    parser = argparse.ArgumentParser(
        prog="gonet-astrometry",
        description="Astrometric calibration of wide-field GONet images.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    subparsers = parser.add_subparsers(dest="command")
    portal = subparsers.add_parser(
        "portal",
        help="Launch the desktop calibration portal.",
    )
    portal.add_argument(
        "--image",
        "--input",
        dest="image",
        type=Path,
        default=None,
        help=("Optional GONet image or folder used to seed portal input discovery."),
    )
    portal.add_argument("--host", default="127.0.0.1")
    portal.add_argument("--port", type=int, default=8050)
    portal.add_argument("--debug", action="store_true")
    portal.add_argument(
        "--server-only",
        action="store_true",
        help="Run Dash without opening the native desktop window.",
    )

    run = subparsers.add_parser(
        "run",
        help="Run source detection and bootstrap tracking without the GUI.",
    )
    run.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="GONet image files and/or folders to discover.",
    )
    run.add_argument(
        "--algorithm",
        required=True,
        choices=[spec.identifier for spec in DETECTOR_SPECS],
        help=(
            "Source-detection backend. With --grid-calibration the stellar "
            "pipeline currently requires 'sep' and runs it independently on "
            "the four native Bayer channels."
        ),
    )
    run.add_argument(
        "--recursive",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Search supplied folders recursively (default: enabled).",
    )
    run.add_argument(
        "--channel",
        choices=GONET_CHANNELS,
        default="green1",
        help="Compact native channel used as the PDF background.",
    )
    run.add_argument(
        "--reference-image",
        type=Path,
        default=None,
        help=(
            "Image from the retained sequence to use as the PDF background. "
            "Defaults to the detection product's original reference image. "
            "This option affects report rendering only."
        ),
    )
    run.add_argument(
        "--output-dir",
        type=Path,
        default=Path("gonet_astrometry_output"),
        help=(
            "Directory for reusable detections/tracks and, unless --output is "
            "given, the PDF report."
        ),
    )
    run.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Destination PDF diagnostic figure. Defaults to "
            "<output-dir>/tracking.pdf."
        ),
    )
    run.add_argument(
        "--overwrite-products",
        action="store_true",
        help=(
            "Recompute detections, tracks, and downstream products even when "
            "compatible products exist."
        ),
    )
    run.add_argument(
        "--workers",
        type=_positive_int,
        default=1,
        help=(
            "Worker processes for Grid-assisted per-image multichannel "
            "detection (default: %(default)s). Use 1 for the serial reference "
            "path."
        ),
    )
    run.add_argument(
        "--grid-calibration",
        type=Path,
        default=None,
        help=(
            "Portable Grid Calibration *_calibration.npz artifact. When supplied, "
            "use independent-channel SEP detection, spherical temporal tracking, "
            "sidereal fragment merging, and the final fixed-rate sidereal fit."
        ),
    )
    run.add_argument(
        "--field-mask",
        type=Path,
        default=None,
        help=(
            "Optional portable full-sensor exclusion mask. True/masked pixels "
            "are removed before multichannel background estimation and source "
            "detection. Requires --grid-calibration."
        ),
    )
    run.add_argument(
        "--detection-only",
        action="store_true",
        help=(
            "Skip spherical temporal tracking, stellar merging, sidereal fitting, "
            "and absolute orientation. If --identify-stars is supplied, catalog "
            "identification still runs after detection. Grid-assisted runs may "
            "therefore operate on a single image."
        ),
    )
    run.add_argument(
        "--overwrite-solution",
        action="store_true",
        help=(
            "Recompute the stellar fragment merge and final sidereal solution while "
            "retaining compatible detections and spherical temporal tracks."
        ),
    )
    run.add_argument(
        "--solve-orientation",
        action="store_true",
        help=(
            "Resolve the final camera-attitude degree of freedom by matching "
            "sidereal-consistent tracks to the Bright Star Catalogue."
        ),
    )
    run.add_argument(
        "--catalog-cache",
        type=Path,
        default=None,
        help=(
            "Portable Bright Star Catalogue cache. When omitted, use "
            "<output-dir>/bright_star_catalog.npz and fetch it from VizieR "
            "only if absent."
        ),
    )
    run.add_argument(
        "--overwrite-orientation",
        action="store_true",
        help=(
            "Recompute only the catalog-assisted absolute orientation while "
            "retaining compatible upstream products."
        ),
    )
    run.add_argument(
        "--identify-stars",
        action="store_true",
        help=(
            "Fit one sequence-wide Grid-to-horizon attitude and identify visible "
            "Bright Star Catalogue stars independently in every frame. This "
            "writes stellar_identifications.npz without changing the current "
            "legacy/spherical tracking path."
        ),
    )
    run.add_argument(
        "--overwrite-identifications",
        action="store_true",
        help=(
            "Recompute catalog stellar identifications while retaining compatible "
            "detections and other upstream products."
        ),
    )
    _add_detection_arguments(run)
    _add_tracking_arguments(run)
    _add_multichannel_arguments(run)
    _add_spherical_tracking_arguments(run)
    _add_stellar_merge_arguments(run)
    _add_sidereal_arguments(run)
    _add_orientation_arguments(run)
    _add_stellar_identification_arguments(run)
    return parser


def _add_detection_arguments(parser: argparse.ArgumentParser) -> None:
    """Add every :class:`DetectionConfig` setting to ``parser``."""
    parser.add_argument(
        "--threshold-sigma",
        type=float,
        default=_DETECTION_DEFAULTS.threshold_sigma,
    )
    parser.add_argument("--fwhm-px", type=float, default=_DETECTION_DEFAULTS.fwhm_px)
    parser.add_argument(
        "--min-separation-px",
        type=float,
        default=_DETECTION_DEFAULTS.min_separation_px,
    )
    parser.add_argument(
        "--min-pixels", type=int, default=_DETECTION_DEFAULTS.min_pixels
    )
    parser.add_argument(
        "--max-sources",
        type=_optional_positive_int,
        default=_DETECTION_DEFAULTS.max_sources,
        metavar="INT|none",
    )
    parser.add_argument(
        "--deblend",
        action=argparse.BooleanOptionalAction,
        default=_DETECTION_DEFAULTS.deblend,
    )
    parser.add_argument(
        "--background-box-size-px",
        type=int,
        default=_DETECTION_DEFAULTS.background_box_size_px,
    )
    parser.add_argument(
        "--background-min-valid-fraction",
        type=float,
        default=_DETECTION_DEFAULTS.background_min_valid_fraction,
    )
    parser.add_argument(
        "--background-sigma-clip",
        type=float,
        default=_DETECTION_DEFAULTS.background_sigma_clip,
    )
    parser.add_argument(
        "--background-clip-iterations",
        type=int,
        default=_DETECTION_DEFAULTS.background_clip_iterations,
    )
    parser.add_argument(
        "--use-provisional-field-mask",
        action=argparse.BooleanOptionalAction,
        default=_DETECTION_DEFAULTS.use_provisional_field_mask,
    )
    parser.add_argument(
        "--field-edge-threshold-fraction",
        "--footprint-threshold-fraction",
        dest="footprint_threshold_fraction",
        type=float,
        default=_DETECTION_DEFAULTS.footprint_threshold_fraction,
        help=(
            "Fraction of center-to-corner image contrast used to infer the "
            "illuminated fisheye edge (default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--footprint-smoothing-px",
        type=float,
        default=_DETECTION_DEFAULTS.footprint_smoothing_px,
    )
    parser.add_argument(
        "--footprint-erosion-px",
        type=float,
        default=_DETECTION_DEFAULTS.footprint_erosion_px,
    )
    parser.add_argument(
        "--bright-mask-sigma",
        type=_optional_positive_float,
        default=_DETECTION_DEFAULTS.bright_mask_sigma,
        metavar="FLOAT|none",
    )
    parser.add_argument(
        "--bright-mask-min-pixels",
        type=int,
        default=_DETECTION_DEFAULTS.bright_mask_min_pixels,
    )
    parser.add_argument(
        "--bright-mask-dilation-px",
        type=float,
        default=_DETECTION_DEFAULTS.bright_mask_dilation_px,
    )


def _add_tracking_arguments(parser: argparse.ArgumentParser) -> None:
    """Add every :class:`TrackingConfig` setting to ``parser``."""
    parser.add_argument(
        "--max-speed-px-per-minute",
        "--max-motion-px-min",
        dest="max_speed_px_per_minute",
        type=float,
        default=_TRACKING_DEFAULTS.max_speed_px_per_minute,
    )
    parser.add_argument(
        "--prediction-tolerance-px",
        type=float,
        default=_TRACKING_DEFAULTS.prediction_tolerance_px,
    )
    parser.add_argument(
        "--max-gap-minutes",
        type=float,
        default=_TRACKING_DEFAULTS.max_gap_minutes,
        help=(
            "Maximum elapsed minutes between a track's last detection and a "
            "new candidate (default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--max-gap-frames",
        "--max-missing",
        dest="max_gap_frames",
        type=_optional_nonnegative_int,
        default=_TRACKING_DEFAULTS.max_gap_frames,
        metavar="INT|none",
        help=(
            "Optional legacy missing-epoch guard; 'none' disables frame-count "
            "gating (default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--max-prediction-gap-scale",
        type=float,
        default=_TRACKING_DEFAULTS.max_prediction_gap_scale,
        help=(
            "Maximum expansion of the prediction tolerance across long time "
            "gaps (default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--min-track-length",
        type=int,
        default=_TRACKING_DEFAULTS.min_track_length,
    )
    parser.add_argument(
        "--low-motion-threshold-px-per-minute",
        type=float,
        default=_TRACKING_DEFAULTS.low_motion_threshold_px_per_minute,
    )
    parser.add_argument(
        "--poor-fit-rms-px",
        type=float,
        default=_TRACKING_DEFAULTS.poor_fit_rms_px,
    )
    parser.add_argument(
        "--location-tolerance-m",
        type=float,
        default=_TRACKING_DEFAULTS.location_tolerance_m,
    )


def _add_multichannel_arguments(parser: argparse.ArgumentParser) -> None:
    """Add independent native-channel SEP settings."""
    parser.add_argument(
        "--field-edge-keep-margin-px",
        type=float,
        default=_MULTICHANNEL_DEFAULTS.field_edge_keep_margin_px,
        help=(
            "Additional full-sensor distance required inside the automatically "
            "detected fisheye edge before retaining a source (default: "
            "%(default)s px)."
        ),
    )
    parser.add_argument(
        "--field-mask-keep-margin-px",
        type=float,
        default=_MULTICHANNEL_DEFAULTS.field_mask_keep_margin_px,
        help=(
            "Additional full-sensor distance required inside the optional static "
            "field-mask boundary before retaining a source (default: "
            "%(default)s px)."
        ),
    )
    parser.add_argument(
        "--grid-search-radius-deg",
        type=float,
        default=_MULTICHANNEL_DEFAULTS.grid_search_radius_deg,
        help=(
            "Optional Grid-calibrated angular-radius cap for background "
            "estimation and source detection. Omit to use only the image-driven "
            "field edge."
        ),
    )
    parser.add_argument(
        "--grid-acceptance-radius-deg",
        type=float,
        default=_MULTICHANNEL_DEFAULTS.grid_acceptance_radius_deg,
        help=(
            "Optional Grid-calibrated angular-radius cap for retained "
            "detections. Omit to use only the image-driven acceptance field."
        ),
    )
    parser.add_argument(
        "--multichannel-match-radius-px",
        type=float,
        default=_MULTICHANNEL_DEFAULTS.channel_match_radius_px,
    )
    parser.add_argument(
        "--multichannel-background-box-size-px",
        type=int,
        default=_MULTICHANNEL_DEFAULTS.background_box_size_compact_px,
        help="Background2D box size in compact-channel pixels.",
    )
    parser.add_argument(
        "--multichannel-min-support",
        type=int,
        default=_MULTICHANNEL_DEFAULTS.minimum_channel_support,
        help="Minimum native Bayer channels supporting a candidate (default: 1).",
    )


def _multichannel_config(arguments: argparse.Namespace) -> MultiChannelSEPConfig:
    """Construct independent-channel detection settings."""
    return MultiChannelSEPConfig(
        field_edge_keep_margin_px=arguments.field_edge_keep_margin_px,
        field_mask_keep_margin_px=arguments.field_mask_keep_margin_px,
        grid_search_radius_deg=arguments.grid_search_radius_deg,
        grid_acceptance_radius_deg=arguments.grid_acceptance_radius_deg,
        channel_match_radius_px=arguments.multichannel_match_radius_px,
        background_box_size_compact_px=(arguments.multichannel_background_box_size_px),
        minimum_channel_support=arguments.multichannel_min_support,
    )


def _add_spherical_tracking_arguments(parser: argparse.ArgumentParser) -> None:
    """Add Grid-aware spherical temporal-association settings."""
    parser.add_argument(
        "--spherical-min-track-length",
        type=int,
        default=_SPHERICAL_DEFAULTS.min_track_length,
    )
    parser.add_argument(
        "--spherical-max-gap-minutes",
        type=float,
        default=_SPHERICAL_DEFAULTS.max_gap_minutes,
    )
    parser.add_argument(
        "--spherical-single-point-max-gap-seconds",
        type=float,
        default=_SPHERICAL_DEFAULTS.single_point_max_gap_seconds,
    )
    parser.add_argument(
        "--spherical-max-initial-speed-deg-per-minute",
        type=float,
        default=_SPHERICAL_DEFAULTS.max_initial_speed_deg_per_minute,
    )
    parser.add_argument(
        "--spherical-initial-margin-arcmin",
        type=float,
        default=_SPHERICAL_DEFAULTS.initial_position_margin_arcmin,
    )
    parser.add_argument(
        "--spherical-prediction-tolerance-arcmin",
        type=float,
        default=_SPHERICAL_DEFAULTS.prediction_tolerance_arcmin,
    )
    parser.add_argument(
        "--spherical-prediction-reference-seconds",
        type=float,
        default=_SPHERICAL_DEFAULTS.prediction_reference_seconds,
    )
    parser.add_argument(
        "--spherical-max-prediction-tolerance-arcmin",
        type=float,
        default=_SPHERICAL_DEFAULTS.max_prediction_tolerance_arcmin,
    )
    parser.add_argument(
        "--spherical-velocity-fit-points",
        type=int,
        default=_SPHERICAL_DEFAULTS.velocity_fit_points,
    )
    parser.add_argument(
        "--spherical-inverse-tolerance-px",
        type=float,
        default=_SPHERICAL_DEFAULTS.inverse_reprojection_tolerance_px,
    )


def _spherical_tracking_config(
    arguments: argparse.Namespace,
) -> SphericalTrackingConfig:
    """Construct spherical temporal-association settings."""
    return SphericalTrackingConfig(
        min_track_length=arguments.spherical_min_track_length,
        max_gap_minutes=arguments.spherical_max_gap_minutes,
        single_point_max_gap_seconds=(arguments.spherical_single_point_max_gap_seconds),
        max_initial_speed_deg_per_minute=(
            arguments.spherical_max_initial_speed_deg_per_minute
        ),
        initial_position_margin_arcmin=arguments.spherical_initial_margin_arcmin,
        prediction_tolerance_arcmin=(arguments.spherical_prediction_tolerance_arcmin),
        prediction_reference_seconds=(arguments.spherical_prediction_reference_seconds),
        max_prediction_tolerance_arcmin=(
            arguments.spherical_max_prediction_tolerance_arcmin
        ),
        velocity_fit_points=arguments.spherical_velocity_fit_points,
        inverse_reprojection_tolerance_px=(arguments.spherical_inverse_tolerance_px),
    )


def _add_stellar_merge_arguments(parser: argparse.ArgumentParser) -> None:
    """Add sidereal fragment-bootstrap/merge settings."""
    parser.add_argument(
        "--stellar-min-fragment-points",
        type=int,
        default=_STELLAR_MERGE_DEFAULTS.min_fragment_points,
    )
    parser.add_argument(
        "--stellar-merge-radius-arcmin",
        type=float,
        default=60.0 * _STELLAR_MERGE_DEFAULTS.merge_radius_deg,
    )
    parser.add_argument(
        "--stellar-merged-rms-arcmin",
        type=float,
        default=60.0 * _STELLAR_MERGE_DEFAULTS.merged_rms_deg,
    )


def _stellar_merge_config(arguments: argparse.Namespace) -> StellarTrackMergeConfig:
    """Construct stellar fragment-bootstrap settings."""
    return StellarTrackMergeConfig(
        min_fragment_points=arguments.stellar_min_fragment_points,
        merge_radius_deg=arguments.stellar_merge_radius_arcmin / 60.0,
        merged_rms_deg=arguments.stellar_merged_rms_arcmin / 60.0,
    )


def _add_sidereal_arguments(parser: argparse.ArgumentParser) -> None:
    """Add shared sidereal-axis fitting parameters to ``parser``."""
    parser.add_argument(
        "--sidereal-min-track-points",
        type=int,
        default=_SIDEREAL_DEFAULTS.min_track_points,
    )
    parser.add_argument(
        "--sidereal-min-duration-minutes",
        type=float,
        default=_SIDEREAL_DEFAULTS.min_track_duration_minutes,
    )
    parser.add_argument(
        "--sidereal-min-span-deg",
        type=float,
        default=_SIDEREAL_DEFAULTS.min_track_span_deg,
    )
    parser.add_argument(
        "--sidereal-robust-scale-deg",
        type=float,
        default=_SIDEREAL_DEFAULTS.robust_scale_deg,
    )
    parser.add_argument(
        "--sidereal-consistent-rms-deg",
        type=float,
        default=_SIDEREAL_DEFAULTS.consistent_rms_deg,
    )
    parser.add_argument(
        "--sidereal-max-fit-tracks",
        type=int,
        default=_SIDEREAL_DEFAULTS.max_fit_tracks,
    )
    parser.add_argument(
        "--sidereal-max-points-per-track",
        type=int,
        default=_SIDEREAL_DEFAULTS.max_fit_points_per_track,
    )
    parser.add_argument(
        "--sidereal-inverse-tolerance-px",
        type=float,
        default=_SIDEREAL_DEFAULTS.inverse_reprojection_tolerance_px,
    )


def _sidereal_config(arguments: argparse.Namespace) -> SiderealFitConfig:
    """Construct sidereal-fit settings from parsed ``run`` arguments."""
    return SiderealFitConfig(
        min_track_points=arguments.sidereal_min_track_points,
        min_track_duration_minutes=arguments.sidereal_min_duration_minutes,
        min_track_span_deg=arguments.sidereal_min_span_deg,
        robust_scale_deg=arguments.sidereal_robust_scale_deg,
        consistent_rms_deg=arguments.sidereal_consistent_rms_deg,
        max_fit_tracks=arguments.sidereal_max_fit_tracks,
        max_fit_points_per_track=arguments.sidereal_max_points_per_track,
        inverse_reprojection_tolerance_px=arguments.sidereal_inverse_tolerance_px,
    )


def _add_stellar_identification_arguments(parser: argparse.ArgumentParser) -> None:
    """Add sequence-wide catalog-identification settings to ``parser``."""
    defaults = _STELLAR_IDENTIFICATION_DEFAULTS
    parser.add_argument(
        "--identification-limiting-magnitude",
        type=float,
        default=defaults.limiting_magnitude,
        help="Faintest catalog star considered for per-frame identification.",
    )
    parser.add_argument(
        "--identification-bootstrap-limiting-magnitude",
        type=float,
        default=defaults.bootstrap_limiting_magnitude,
        help="Faintest catalog star used to bootstrap/refine the common attitude.",
    )
    parser.add_argument(
        "--identification-min-altitude-deg",
        type=float,
        default=defaults.min_catalog_altitude_deg,
    )
    parser.add_argument(
        "--identification-rotation-step-deg",
        type=float,
        default=defaults.bootstrap_rotation_step_deg,
        help="Coarse all-sky roll search step used for automatic attitude bootstrap.",
    )
    parser.add_argument(
        "--identification-bootstrap-radius-px",
        type=float,
        default=defaults.bootstrap_match_radius_px,
    )
    parser.add_argument(
        "--identification-final-radius-px",
        type=float,
        default=defaults.final_match_radius_px,
    )
    parser.add_argument(
        "--identification-bright-rescue-magnitude",
        type=float,
        default=defaults.bright_rescue_magnitude,
    )
    parser.add_argument(
        "--identification-bright-rescue-radius-px",
        type=float,
        default=defaults.bright_rescue_radius_px,
    )
    parser.add_argument(
        "--identification-sequence-refinement-radius-px",
        type=float,
        default=defaults.sequence_refinement_radius_px,
    )
    parser.add_argument(
        "--identification-sequence-refinement-iterations",
        type=int,
        default=defaults.sequence_refinement_iterations,
    )


def _stellar_identification_config(
    arguments: argparse.Namespace,
) -> StellarIdentificationConfig:
    """Construct sequence-wide stellar-identification settings."""
    defaults = _STELLAR_IDENTIFICATION_DEFAULTS
    return StellarIdentificationConfig(
        limiting_magnitude=arguments.identification_limiting_magnitude,
        bootstrap_limiting_magnitude=(
            arguments.identification_bootstrap_limiting_magnitude
        ),
        min_catalog_altitude_deg=arguments.identification_min_altitude_deg,
        bootstrap_rotation_step_deg=arguments.identification_rotation_step_deg,
        bootstrap_match_radius_px=arguments.identification_bootstrap_radius_px,
        refinement_radii_px=defaults.refinement_radii_px,
        min_bootstrap_matches=defaults.min_bootstrap_matches,
        sequence_refinement_radius_px=(
            arguments.identification_sequence_refinement_radius_px
        ),
        sequence_refinement_iterations=(
            arguments.identification_sequence_refinement_iterations
        ),
        final_match_radius_px=arguments.identification_final_radius_px,
        bright_rescue_magnitude=(arguments.identification_bright_rescue_magnitude),
        bright_rescue_radius_px=(arguments.identification_bright_rescue_radius_px),
        robust_clip_sigma=defaults.robust_clip_sigma,
        robust_clip_floor_arcmin=defaults.robust_clip_floor_arcmin,
        max_refinement_pairs=defaults.max_refinement_pairs,
        minimum_catalog_track_length=defaults.minimum_catalog_track_length,
    )


def _add_orientation_arguments(parser: argparse.ArgumentParser) -> None:
    """Add absolute-orientation fitting parameters to ``parser``."""
    parser.add_argument(
        "--orientation-limiting-magnitude",
        type=float,
        default=_ORIENTATION_DEFAULTS.limiting_magnitude,
    )
    parser.add_argument(
        "--orientation-bootstrap-limiting-magnitude",
        type=float,
        default=_ORIENTATION_DEFAULTS.bootstrap_limiting_magnitude,
    )
    parser.add_argument(
        "--orientation-min-catalog-altitude-deg",
        type=float,
        default=_ORIENTATION_DEFAULTS.min_catalog_altitude_deg,
    )
    parser.add_argument(
        "--orientation-min-track-points",
        type=int,
        default=_ORIENTATION_DEFAULTS.min_track_points,
    )
    parser.add_argument(
        "--orientation-min-duration-minutes",
        type=float,
        default=_ORIENTATION_DEFAULTS.min_track_duration_minutes,
    )
    parser.add_argument(
        "--orientation-dedup-radius-deg",
        type=float,
        default=_ORIENTATION_DEFAULTS.deduplication_radius_deg,
    )
    parser.add_argument(
        "--orientation-declination-tolerance-deg",
        type=float,
        default=_ORIENTATION_DEFAULTS.declination_tolerance_deg,
    )
    parser.add_argument(
        "--orientation-consensus-bin-deg",
        type=float,
        default=_ORIENTATION_DEFAULTS.consensus_bin_deg,
    )
    parser.add_argument(
        "--orientation-consensus-tolerance-deg",
        type=float,
        default=_ORIENTATION_DEFAULTS.consensus_tolerance_deg,
    )
    parser.add_argument(
        "--orientation-match-radius-deg",
        type=float,
        default=_ORIENTATION_DEFAULTS.match_radius_deg,
    )
    parser.add_argument(
        "--orientation-min-matches",
        type=int,
        default=_ORIENTATION_DEFAULTS.min_matches,
    )
    parser.add_argument(
        "--orientation-max-anchors",
        type=int,
        default=_ORIENTATION_DEFAULTS.max_anchors,
    )
    parser.add_argument(
        "--orientation-inverse-tolerance-px",
        type=float,
        default=_ORIENTATION_DEFAULTS.inverse_reprojection_tolerance_px,
    )
    parser.add_argument(
        "--orientation-track-validation-rms-deg",
        type=float,
        default=_ORIENTATION_DEFAULTS.track_validation_rms_deg,
    )
    parser.add_argument(
        "--orientation-max-declination-robust-sigma-deg",
        type=float,
        default=_ORIENTATION_DEFAULTS.max_declination_robust_sigma_deg,
    )
    parser.add_argument(
        "--orientation-max-declination-p95-deg",
        type=float,
        default=_ORIENTATION_DEFAULTS.max_declination_p95_deg,
    )


def _orientation_config(arguments: argparse.Namespace) -> OrientationFitConfig:
    """Construct absolute-orientation settings from parsed arguments."""
    return OrientationFitConfig(
        limiting_magnitude=arguments.orientation_limiting_magnitude,
        bootstrap_limiting_magnitude=(
            arguments.orientation_bootstrap_limiting_magnitude
        ),
        min_catalog_altitude_deg=arguments.orientation_min_catalog_altitude_deg,
        min_track_points=arguments.orientation_min_track_points,
        min_track_duration_minutes=arguments.orientation_min_duration_minutes,
        deduplication_radius_deg=arguments.orientation_dedup_radius_deg,
        declination_tolerance_deg=(arguments.orientation_declination_tolerance_deg),
        consensus_bin_deg=arguments.orientation_consensus_bin_deg,
        consensus_tolerance_deg=arguments.orientation_consensus_tolerance_deg,
        match_radius_deg=arguments.orientation_match_radius_deg,
        min_matches=arguments.orientation_min_matches,
        max_anchors=arguments.orientation_max_anchors,
        inverse_reprojection_tolerance_px=(arguments.orientation_inverse_tolerance_px),
        track_validation_rms_deg=arguments.orientation_track_validation_rms_deg,
        max_declination_robust_sigma_deg=(
            arguments.orientation_max_declination_robust_sigma_deg
        ),
        max_declination_p95_deg=arguments.orientation_max_declination_p95_deg,
    )


def _positive_int(value: str) -> int:
    """Parse a strictly positive integer."""
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _optional_nonnegative_int(value: str) -> int | None:
    """Parse a non-negative integer or the literal ``none``."""
    if value.casefold() == "none":
        return None
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative or 'none'")
    return parsed


def _optional_positive_float(value: str) -> float | None:
    """Parse a positive float or the literal ``none``."""
    if value.casefold() == "none":
        return None
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive or 'none'")
    return parsed


def _optional_positive_int(value: str) -> int | None:
    """Parse a positive integer or the literal ``none``."""
    if value.casefold() == "none":
        return None
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive or 'none'")
    return parsed


def _detection_config(arguments: argparse.Namespace) -> DetectionConfig:
    """Construct detection settings from parsed ``run`` arguments."""
    return DetectionConfig(
        threshold_sigma=arguments.threshold_sigma,
        fwhm_px=arguments.fwhm_px,
        min_separation_px=arguments.min_separation_px,
        min_pixels=arguments.min_pixels,
        max_sources=arguments.max_sources,
        deblend=arguments.deblend,
        background_box_size_px=arguments.background_box_size_px,
        background_min_valid_fraction=arguments.background_min_valid_fraction,
        background_sigma_clip=arguments.background_sigma_clip,
        background_clip_iterations=arguments.background_clip_iterations,
        use_provisional_field_mask=arguments.use_provisional_field_mask,
        footprint_threshold_fraction=arguments.footprint_threshold_fraction,
        footprint_smoothing_px=arguments.footprint_smoothing_px,
        footprint_erosion_px=arguments.footprint_erosion_px,
        bright_mask_sigma=arguments.bright_mask_sigma,
        bright_mask_min_pixels=arguments.bright_mask_min_pixels,
        bright_mask_dilation_px=arguments.bright_mask_dilation_px,
    )


def _tracking_config(arguments: argparse.Namespace) -> TrackingConfig:
    """Construct tracking settings from parsed ``run`` arguments."""
    return TrackingConfig(
        max_speed_px_per_minute=arguments.max_speed_px_per_minute,
        prediction_tolerance_px=arguments.prediction_tolerance_px,
        max_gap_minutes=arguments.max_gap_minutes,
        max_gap_frames=arguments.max_gap_frames,
        max_prediction_gap_scale=arguments.max_prediction_gap_scale,
        min_track_length=arguments.min_track_length,
        low_motion_threshold_px_per_minute=(
            arguments.low_motion_threshold_px_per_minute
        ),
        poor_fit_rms_px=arguments.poor_fit_rms_px,
        location_tolerance_m=arguments.location_tolerance_m,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line interface.

    Parameters
    ----------
    argv
        Optional command-line arguments excluding the executable name.

    Returns
    -------
    int
        Process exit status.
    """
    parser = build_parser()
    arguments = parser.parse_args(argv)

    if arguments.command == "portal":
        from gonet_astrometry.portal.app import run_portal, run_portal_server

        launch = run_portal_server if arguments.server_only else run_portal
        launch(
            initial_path=arguments.image,
            host=arguments.host,
            port=arguments.port,
            debug=arguments.debug,
        )

    if arguments.command == "run":
        from gonet_astrometry.runner import run_cli_workflow

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)-8s | %(message)s",
            datefmt="%H:%M:%S",
        )
        output_path = (
            arguments.output
            if arguments.output is not None
            else arguments.output_dir / "tracking.pdf"
        )
        summary = run_cli_workflow(
            arguments.inputs,
            recursive=arguments.recursive,
            algorithm=arguments.algorithm,
            detection_config=_detection_config(arguments),
            tracking_config=_tracking_config(arguments),
            channel=arguments.channel,
            reference_image_path=arguments.reference_image,
            output_path=output_path,
            output_dir=arguments.output_dir,
            overwrite_products=arguments.overwrite_products,
            grid_calibration_path=arguments.grid_calibration,
            field_mask_path=arguments.field_mask,
            detection_workers=arguments.workers,
            multichannel_config=_multichannel_config(arguments),
            spherical_tracking_config=_spherical_tracking_config(arguments),
            stellar_merge_config=_stellar_merge_config(arguments),
            sidereal_config=_sidereal_config(arguments),
            overwrite_solution=arguments.overwrite_solution,
            detection_only=arguments.detection_only,
            solve_orientation=arguments.solve_orientation,
            orientation_config=_orientation_config(arguments),
            catalog_cache_path=arguments.catalog_cache,
            overwrite_orientation=arguments.overwrite_orientation,
            identify_stars=arguments.identify_stars,
            stellar_identification_config=_stellar_identification_config(arguments),
            overwrite_identifications=arguments.overwrite_identifications,
        )
        detection_state = "cached" if summary.reused_detection_product else "computed"
        tracking_state = "cached" if summary.reused_tracking_product else "computed"
        if arguments.grid_calibration is not None:
            product_parts = [f"detections {detection_state}"]
            if summary.temporal_tracking_product_path is not None:
                temporal_state = (
                    "cached" if summary.reused_temporal_tracking_product else "computed"
                )
                product_parts.append(f"temporal {temporal_state}")
            if summary.stellar_tracking_product_path is not None:
                stellar_state = (
                    "cached" if summary.reused_stellar_tracking_product else "computed"
                )
                product_parts.append(f"stellar {stellar_state}")
            if summary.stellar_identification_product_path is not None:
                identification_state = (
                    "cached"
                    if summary.reused_stellar_identification_product
                    else "computed"
                )
                product_parts.append(f"identifications {identification_state}")
            product_note = " | ".join(product_parts)
        else:
            product_note = f"detections {detection_state} | tracks {tracking_state}"

        solution_note = ""
        if summary.sidereal_product_path is not None:
            solution_state = "cached" if summary.reused_sidereal_product else "computed"
            solution_note += f" | sidereal {solution_state}"
        if summary.orientation_product_path is not None:
            orientation_state = (
                "cached" if summary.reused_orientation_product else "computed"
            )
            solution_note += f" | orientation {orientation_state}"
        print(
            f"Wrote {summary.output_path} | {summary.file_count} images | "
            f"{summary.detection_count} detections | {summary.track_count} tracks | "
            f"{summary.assigned_detection_count} assigned | "
            f"{summary.unassigned_detection_count} unassigned | "
            f"{summary.skipped_file_count} skipped before detection | "
            f"{product_note}{solution_note}"
        )

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
