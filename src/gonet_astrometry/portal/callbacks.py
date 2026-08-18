"""Dash callbacks for multi-file discovery and native channel display."""

from __future__ import annotations

import logging
from pathlib import Path

import plotly.graph_objects as go
from dash import Dash, Input, Output, State, html

from gonet_astrometry.adapters.gonet_wizard import (
    GONET_CHANNELS,
    GONetChannel,
    get_raw_channel,
)
from gonet_astrometry.detection.config import DetectionConfig
from gonet_astrometry.detection.preprocessing import PreparedDetectionImage
from gonet_astrometry.detection.timing import DetectionTiming
from gonet_astrometry.models.detection import DetectionCatalog
from gonet_astrometry.portal import ids
from gonet_astrometry.portal.discovery import parse_source_paths
from gonet_astrometry.portal.figures import channel_image_figure, empty_image_figure
from gonet_astrometry.portal.layout import DEFAULT_TRACKING_IMAGE_COUNT
from gonet_astrometry.portal.session import PortalSession
from gonet_astrometry.tracking.config import TrackingConfig

logger = logging.getLogger(__name__)


def discover_source_paths(
    source_value: str | None,
    recursive_values: list[str] | None,
    *,
    session: PortalSession,
    current_selection: str | None = None,
    current_tracking_selection: list[str] | None = None,
) -> tuple[
    list[dict[str, str]],
    str | None,
    list[dict[str, str]],
    list[str],
    str,
]:
    """Discover candidate images and prepare file-selector outputs.

    Parameters
    ----------
    source_value
        One or more file or directory paths entered in the sidebar.
    recursive_values
        Checklist values controlling nested folder discovery.
    session
        Server-side portal session receiving the file catalog.
    current_selection
        Existing selected display path to preserve when still available.
    current_tracking_selection
        Existing multi-image sequence selection to preserve when possible.

    Returns
    -------
    tuple
        Display dropdown options and value, tracking dropdown options and
        values, and user-facing discovery status.
    """
    source_paths = parse_source_paths(source_value)
    if not source_paths:
        message = "Enter at least one file or folder before discovery."
        logger.warning(message)
        session.discover(())
        return [], None, [], [], message

    recursive = bool(recursive_values and "recursive" in recursive_values)
    logger.info(
        "Discovering native GONet candidates from %d input path(s)%s.",
        len(source_paths),
        " recursively" if recursive else "",
    )
    result = session.discover(source_paths, recursive=recursive)
    options = [{"label": str(path), "value": str(path)} for path in result.files]
    available = {option["value"] for option in options}
    selected = current_selection if current_selection in available else None
    if selected is None and options:
        selected = options[0]["value"]

    tracking_selected = [
        value for value in (current_tracking_selection or []) if value in available
    ]
    if not tracking_selected:
        tracking_selected = [
            option["value"] for option in options[:DEFAULT_TRACKING_IMAGE_COUNT]
        ]

    if result.files:
        logger.info(result.summary)
    else:
        logger.warning(result.summary)
    return options, selected, options, tracking_selected, result.summary


def load_channel_view(
    path_value: str | None,
    channel: GONetChannel | None,
    *,
    session: PortalSession,
    show_masks: bool = True,
) -> tuple[go.Figure, str, list[html.Tr]]:
    """Load the selected file lazily and prepare image-view outputs.

    Parameters
    ----------
    path_value
        Candidate path selected in the sidebar file catalog.
    channel
        Selected native Bayer channel.
    session
        Server-side session that caches at most one native image object.
    show_masks
        Whether cached field and dynamic-mask boundaries are shown.

    Returns
    -------
    tuple
        Plotly figure, user-facing status message, and summary table rows.
    """
    normalized = (path_value or "").strip()
    if not normalized:
        message = "Discover and select a GONet image before loading."
        return empty_image_figure(message), message, []

    if channel not in GONET_CHANNELS:
        message = "Select a valid GONet channel before loading."
        logger.warning(message)
        return empty_image_figure(message), message, []

    image_path = Path(normalized).expanduser()
    cached = session.loaded_path == image_path.resolve()
    if cached:
        logger.info("Reusing cached native image %s.", image_path)
    else:
        logger.info("Loading %s through GONetFileRaw.", image_path)

    try:
        gonet_file = session.load(image_path)
        data = get_raw_channel(gonet_file, channel)
    except (OSError, RuntimeError, ValueError) as exc:
        message = f"Unable to load image: {exc}"
        logger.error(message)
        return empty_image_figure(message), message, []

    rows, columns = data.shape
    coordinate_system = (
        "expanded Bayer-plane coordinates"
        if gonet_file.is_bayer_planes
        else "compact Bayer-channel coordinates"
    )
    status = (
        f"Loaded {image_path.name}: {channel} channel, "
        f"{columns} × {rows} pixels in {coordinate_system}."
    )
    logger.info(status)
    prepared = _prepared_for_path(session, image_path) if show_masks else None
    return (
        channel_image_figure(
            data,
            channel=channel,
            source_name=str(image_path),
            detections=_catalog_for_path(session, image_path),
            field_mask=None if prepared is None else prepared.field_mask,
            dynamic_mask=None if prepared is None else prepared.dynamic_mask,
            tracking_result=session.tracking_result,
        ),
        status,
        _summary_rows(
            path=image_path,
            filename=gonet_file.filename,
            channel=channel,
            shape=(data.shape[0], data.shape[1]),
            is_bayer_planes=gonet_file.is_bayer_planes,
        ),
    )


def detect_source_view(
    path_value: str | None,
    channel: GONetChannel | None,
    detector_identifier: str | None,
    threshold_sigma: float | int | None,
    fwhm_px: float | int | None,
    min_pixels: float | int | None,
    *,
    session: PortalSession,
    show_masks: bool = True,
) -> tuple[go.Figure, str]:
    """Run the selected detector and return an overlaid channel figure.

    Parameters
    ----------
    path_value
        Selected candidate image path.
    channel
        Compact native channel used as the display background.
    detector_identifier
        Stable identifier from the detector registry.
    threshold_sigma, fwhm_px, min_pixels
        Shared detector controls from the portal sidebar.
    session
        Server-side session caching the current image, frame, and catalog.
    show_masks
        Whether preprocessing-mask boundaries are shown over the image.

    Returns
    -------
    tuple
        Updated image figure and user-facing detection status.
    """
    normalized = (path_value or "").strip()
    if not normalized:
        message = "Discover and select a GONet image before detecting sources."
        return empty_image_figure(message), message
    if channel not in GONET_CHANNELS:
        message = "Select a valid GONet channel before detecting sources."
        return empty_image_figure(message), message
    if not detector_identifier:
        message = "Select a source-detection algorithm."
        return empty_image_figure(message), message

    try:
        if threshold_sigma is None or fwhm_px is None or min_pixels is None:
            raise ValueError("all detector settings are required")
        config = DetectionConfig(
            threshold_sigma=float(threshold_sigma),
            fwhm_px=float(fwhm_px),
            min_pixels=int(min_pixels),
        )
    except (TypeError, ValueError) as exc:
        message = f"Invalid detector settings: {exc}"
        logger.warning(message)
        return empty_image_figure(message), message

    image_path = Path(normalized).expanduser()
    logger.info(
        "Running %s on %s with threshold %.2fσ and FWHM %.2f px.",
        detector_identifier,
        image_path,
        config.threshold_sigma,
        config.fwhm_px,
    )
    try:
        gonet_file = session.load(image_path)
        data = get_raw_channel(gonet_file, channel)
        catalog = session.detect(image_path, detector_identifier, config)
    except (OSError, RuntimeError, ValueError) as exc:
        message = f"Source detection failed: {exc}"
        logger.error(message)
        try:
            gonet_file = session.load(image_path)
            data = get_raw_channel(gonet_file, channel)
            figure = channel_image_figure(
                data,
                channel=channel,
                source_name=str(image_path),
                tracking_result=session.tracking_result,
            )
        except (OSError, RuntimeError, ValueError):
            figure = empty_image_figure(message)
        return figure, message

    timing = session.detection_timing
    message = (
        f"Detected {len(catalog)} source candidate"
        f"{'s' if len(catalog) != 1 else ''} with {catalog.detector_name}"
        f"{_detection_timing_status(timing)}. "
        f"{_diagnostic_status(catalog)}"
    )
    logger.info(message)
    if timing is not None:
        rate = timing.sources_per_second
        rate_text = "n/a" if rate is None else f"{rate:.1f} candidates/s"
        logger.info(
            "Detection timing | frame %.3f s | setup %.3f s | "
            "preprocess %.3f s | backend %.3f s | total %.3f s | %s",
            timing.frame_load_seconds,
            timing.detector_setup_seconds,
            timing.preprocessing_seconds,
            timing.backend_seconds,
            timing.total_seconds,
            rate_text,
        )
    prepared = session.prepared_image
    counts = catalog.diagnostic_counts()
    logger.info(
        "Detection diagnostics | compact %d | elongated %d | extended %d | "
        "mask-adjacent %d | backend-flagged %d | unclassified %d",
        counts["compact"],
        counts["elongated"],
        counts["extended"],
        counts["mask-adjacent"],
        counts["backend-flagged"],
        counts["unclassified"],
    )
    if prepared is not None:
        logger.info(
            "Detection masks | usable field %.1f%% | dynamic bright regions %.2f%%",
            100.0 * prepared.field_fraction,
            100.0 * prepared.dynamic_mask_fraction,
        )
    overlay = session.prepared_image if show_masks else None
    return (
        channel_image_figure(
            data,
            channel=channel,
            source_name=str(image_path),
            detections=catalog,
            field_mask=None if overlay is None else overlay.field_mask,
            dynamic_mask=None if overlay is None else overlay.dynamic_mask,
            tracking_result=session.tracking_result,
        ),
        message,
    )


def track_sequence_view(
    path_values: list[str] | None,
    display_path_value: str | None,
    channel: GONetChannel | None,
    detector_identifier: str | None,
    threshold_sigma: float | int | None,
    fwhm_px: float | int | None,
    min_pixels: float | int | None,
    max_speed_px_per_minute: float | int | None,
    prediction_tolerance_px: float | int | None,
    max_gap_minutes: float | int | None,
    min_track_length: float | int | None,
    *,
    session: PortalSession,
    show_masks: bool = True,
) -> tuple[go.Figure, str]:
    """Detect a selected image sequence lazily and build bootstrap tracklets."""
    if not path_values or len(path_values) < 2:
        message = "Select at least two discovered images for tracking."
        logger.warning(message)
        return empty_image_figure(message), message
    normalized_display = (display_path_value or "").strip()
    if not normalized_display:
        message = "Select a current image before displaying tracks."
        return empty_image_figure(message), message
    if channel not in GONET_CHANNELS:
        message = "Select a valid GONet channel before building tracks."
        return empty_image_figure(message), message
    if not detector_identifier:
        message = "Select a source-detection algorithm before building tracks."
        return empty_image_figure(message), message

    try:
        if threshold_sigma is None or fwhm_px is None or min_pixels is None:
            raise ValueError("all detector settings are required")
        detection_config = DetectionConfig(
            threshold_sigma=float(threshold_sigma),
            fwhm_px=float(fwhm_px),
            min_pixels=int(min_pixels),
        )
        if (
            max_speed_px_per_minute is None
            or prediction_tolerance_px is None
            or max_gap_minutes is None
            or min_track_length is None
        ):
            raise ValueError("all tracking settings are required")
        tracking_config = TrackingConfig(
            max_speed_px_per_minute=float(max_speed_px_per_minute),
            prediction_tolerance_px=float(prediction_tolerance_px),
            max_gap_minutes=float(max_gap_minutes),
            max_gap_frames=None,
            min_track_length=int(min_track_length),
        )
    except (TypeError, ValueError) as exc:
        message = f"Invalid tracking settings: {exc}"
        logger.warning(message)
        return empty_image_figure(message), message

    display_path = Path(normalized_display).expanduser()
    logger.info(
        "Building image-plane tracks from %d images with %s. "
        "Maximum motion %.1f px/min; prediction tolerance %.1f px.",
        len(path_values),
        detector_identifier,
        tracking_config.max_speed_px_per_minute,
        tracking_config.prediction_tolerance_px,
    )
    try:
        gonet_file = session.load(display_path)
        data = get_raw_channel(gonet_file, channel)
        result = session.build_tracks(
            path_values,
            detector_identifier,
            detection_config,
            tracking_config,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        message = f"Star tracking failed: {exc}"
        logger.error(message)
        return empty_image_figure(message), message

    counts = result.diagnostic_counts()
    sequence = result.sequence
    message = (
        f"Built {len(result.tracks)} tracklets from {len(sequence.epochs)} images "
        f"and {sequence.total_detections} detections: "
        f"{counts['candidate']} candidate, {counts['low-motion']} low-motion, "
        f"{counts['poor-fit']} poor-fit."
    )
    logger.info(message)
    logger.info(
        "Tracking sequence | duration %.1f min | assigned %d/%d detections | "
        "unassigned %d",
        sequence.duration_seconds / 60.0,
        result.assigned_detection_count,
        sequence.total_detections,
        result.unassigned_detection_count,
    )

    prepared = _prepared_for_path(session, display_path) if show_masks else None
    return (
        channel_image_figure(
            data,
            channel=channel,
            source_name=str(display_path),
            detections=session.catalog_for_path(display_path),
            field_mask=None if prepared is None else prepared.field_mask,
            dynamic_mask=None if prepared is None else prepared.dynamic_mask,
            tracking_result=result,
        ),
        message,
    )


def _diagnostic_status(catalog: DetectionCatalog) -> str:
    """Return a concise mutually exclusive diagnostic summary."""
    counts = catalog.diagnostic_counts()
    parts = [
        f"{counts['compact']} compact",
        f"{counts['elongated']} elongated",
        f"{counts['extended']} extended",
        f"{counts['mask-adjacent']} mask-adjacent",
        f"{counts['backend-flagged']} backend-flagged",
    ]
    if counts["unclassified"]:
        parts.append(f"{counts['unclassified']} unclassified")
    return "Diagnostics: " + ", ".join(parts) + "."


def _detection_timing_status(timing: DetectionTiming | None) -> str:
    """Return a concise detector-performance suffix for portal status text."""
    if timing is None:
        return ""

    rate = timing.sources_per_second
    rate_text = "" if rate is None else f", {rate:.1f} candidates/s"
    return (
        f" in {timing.backend_seconds:.3f} s backend time"
        f" ({timing.preprocessing_seconds:.3f} s preprocessing, "
        f"{timing.total_seconds:.3f} s total{rate_text})"
    )


def register_callbacks(app: Dash, *, session: PortalSession) -> None:
    """Register discovery and image-display callbacks.

    Parameters
    ----------
    app
        Dash application receiving the callbacks.
    session
        Server-side portal session shared by the callbacks.
    """

    @app.callback(  # type: ignore[untyped-decorator]
        Output(ids.FILE_SELECT, "options"),
        Output(ids.FILE_SELECT, "value"),
        Output(ids.TRACKING_FILES, "options"),
        Output(ids.TRACKING_FILES, "value"),
        Output(ids.DISCOVERY_STATUS, "children"),
        Input(ids.DISCOVER_FILES, "n_clicks"),
        State(ids.SOURCE_PATHS, "value"),
        State(ids.RECURSIVE_SEARCH, "value"),
        State(ids.FILE_SELECT, "value"),
        State(ids.TRACKING_FILES, "value"),
        prevent_initial_call=True,
    )
    def _discover_images(
        _n_clicks: int,
        source_value: str | None,
        recursive_values: list[str] | None,
        current_selection: str | None,
        current_tracking_selection: list[str] | None,
    ) -> tuple[
        list[dict[str, str]],
        str | None,
        list[dict[str, str]],
        list[str],
        str,
    ]:
        return discover_source_paths(
            source_value,
            recursive_values,
            session=session,
            current_selection=current_selection,
            current_tracking_selection=current_tracking_selection,
        )

    @app.callback(  # type: ignore[untyped-decorator]
        Output(ids.IMAGE_GRAPH, "figure"),
        Output(ids.STATUS, "children"),
        Output(ids.METADATA_BODY, "children"),
        Input(ids.LOAD_IMAGE, "n_clicks"),
        Input(ids.CHANNEL, "value"),
        Input(ids.SHOW_DETECTION_MASKS, "value"),
        State(ids.FILE_SELECT, "value"),
    )
    def _load_image(
        _n_clicks: int,
        channel: GONetChannel | None,
        mask_values: list[str] | None,
        path_value: str | None,
    ) -> tuple[go.Figure, str, list[html.Tr]]:
        return load_channel_view(
            path_value,
            channel,
            session=session,
            show_masks=_masks_enabled(mask_values),
        )

    @app.callback(  # type: ignore[untyped-decorator]
        Output(ids.IMAGE_GRAPH, "figure", allow_duplicate=True),
        Output(ids.DETECTION_STATUS, "children"),
        Input(ids.DETECT_SOURCES, "n_clicks"),
        State(ids.FILE_SELECT, "value"),
        State(ids.CHANNEL, "value"),
        State(ids.DETECTOR, "value"),
        State(ids.DETECTION_THRESHOLD, "value"),
        State(ids.DETECTION_FWHM, "value"),
        State(ids.DETECTION_MIN_PIXELS, "value"),
        State(ids.SHOW_DETECTION_MASKS, "value"),
        prevent_initial_call=True,
    )
    def _detect_sources(
        _n_clicks: int,
        path_value: str | None,
        channel: GONetChannel | None,
        detector_identifier: str | None,
        threshold_sigma: float | int | None,
        fwhm_px: float | int | None,
        min_pixels: float | int | None,
        mask_values: list[str] | None,
    ) -> tuple[go.Figure, str]:
        return detect_source_view(
            path_value,
            channel,
            detector_identifier,
            threshold_sigma,
            fwhm_px,
            min_pixels,
            session=session,
            show_masks=_masks_enabled(mask_values),
        )

    @app.callback(  # type: ignore[untyped-decorator]
        Output(ids.IMAGE_GRAPH, "figure", allow_duplicate=True),
        Output(ids.TRACKING_STATUS, "children"),
        Input(ids.BUILD_TRACKS, "n_clicks"),
        State(ids.TRACKING_FILES, "value"),
        State(ids.FILE_SELECT, "value"),
        State(ids.CHANNEL, "value"),
        State(ids.DETECTOR, "value"),
        State(ids.DETECTION_THRESHOLD, "value"),
        State(ids.DETECTION_FWHM, "value"),
        State(ids.DETECTION_MIN_PIXELS, "value"),
        State(ids.TRACK_MAX_SPEED, "value"),
        State(ids.TRACK_PREDICTION_TOLERANCE, "value"),
        State(ids.TRACK_MAX_GAP, "value"),
        State(ids.TRACK_MIN_LENGTH, "value"),
        State(ids.SHOW_DETECTION_MASKS, "value"),
        prevent_initial_call=True,
    )
    def _build_tracks(
        _n_clicks: int,
        path_values: list[str] | None,
        display_path_value: str | None,
        channel: GONetChannel | None,
        detector_identifier: str | None,
        threshold_sigma: float | int | None,
        fwhm_px: float | int | None,
        min_pixels: float | int | None,
        max_speed_px_per_minute: float | int | None,
        prediction_tolerance_px: float | int | None,
        max_gap_minutes: float | int | None,
        min_track_length: float | int | None,
        mask_values: list[str] | None,
    ) -> tuple[go.Figure, str]:
        return track_sequence_view(
            path_values,
            display_path_value,
            channel,
            detector_identifier,
            threshold_sigma,
            fwhm_px,
            min_pixels,
            max_speed_px_per_minute,
            prediction_tolerance_px,
            max_gap_minutes,
            min_track_length,
            session=session,
            show_masks=_masks_enabled(mask_values),
        )


def _masks_enabled(values: list[str] | None) -> bool:
    """Return whether the sidebar requests mask overlays."""
    return bool(values and "show" in values)


def _prepared_for_path(
    session: PortalSession,
    path: Path,
) -> PreparedDetectionImage | None:
    """Return cached preprocessing only when it belongs to ``path``."""
    if session.frame_path != path.resolve():
        return None
    return session.prepared_image


def _catalog_for_path(
    session: PortalSession,
    path: Path,
) -> DetectionCatalog | None:
    """Return the cached catalog only when it belongs to ``path``."""
    return session.catalog_for_path(path)


def _summary_rows(
    *,
    path: Path,
    filename: str,
    channel: GONetChannel,
    shape: tuple[int, int],
    is_bayer_planes: bool,
) -> list[html.Tr]:
    """Build summary rows for one native GONet channel."""
    rows, columns = shape
    representation = "Bayer planes" if is_bayer_planes else "Compact channel"
    values = [
        ("Path", str(path)),
        ("Wizard filename", filename),
        ("Channel", channel),
        ("Channel size", f"{columns} × {rows}"),
        ("Representation", representation),
    ]
    return [
        html.Tr(
            [
                html.Th(
                    name,
                    style={
                        "textAlign": "left",
                        "paddingRight": "0.6rem",
                        "verticalAlign": "top",
                    },
                ),
                html.Td(value, style={"wordBreak": "break-word"}),
            ]
        )
        for name, value in values
    ]
