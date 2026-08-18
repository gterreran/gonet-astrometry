"""Command-line entry point."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from gonet_astrometry import __version__
from gonet_astrometry.adapters.gonet_wizard import GONET_CHANNELS
from gonet_astrometry.detection.config import DetectionConfig
from gonet_astrometry.detection.registry import DETECTOR_SPECS
from gonet_astrometry.tracking.config import TrackingConfig

_DETECTION_DEFAULTS = DetectionConfig()
_TRACKING_DEFAULTS = TrackingConfig()


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
        help="Source-detection backend to run on every image.",
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
        help="Recompute detections and tracks even when compatible products exist.",
    )
    _add_detection_arguments(run)
    _add_tracking_arguments(run)
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
        "--footprint-threshold-fraction",
        type=float,
        default=_DETECTION_DEFAULTS.footprint_threshold_fraction,
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
            output_path=output_path,
            output_dir=arguments.output_dir,
            overwrite_products=arguments.overwrite_products,
        )
        detection_state = (
            "cached" if summary.reused_detection_product else "computed"
        )
        tracking_state = "cached" if summary.reused_tracking_product else "computed"
        print(
            f"Wrote {summary.output_path} | {summary.file_count} images | "
            f"{summary.detection_count} detections | {summary.track_count} tracks | "
            f"{summary.assigned_detection_count} assigned | "
            f"{summary.unassigned_detection_count} unassigned | "
            f"{summary.skipped_file_count} skipped before detection | "
            f"detections {detection_state} | tracks {tracking_state}"
        )

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
