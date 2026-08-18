"""Dash layout for the astrometry calibration workspace."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from dash import dcc, html

from gonet_astrometry.adapters.gonet_wizard import GONET_CHANNELS
from gonet_astrometry.detection.registry import detector_options
from gonet_astrometry.portal import ids
from gonet_astrometry.portal.figures import empty_image_figure

DETECTOR_OPTIONS = detector_options()
"""Registered source-detection choices exposed by the portal."""

DEFAULT_TRACKING_IMAGE_COUNT = 5
"""Maximum number of discovered images preselected for bootstrap tracking."""


def build_layout(
    initial_path: Path | None = None,
    discovered_files: Sequence[Path] = (),
) -> html.Div:
    """Build the root calibration-workspace layout.

    Parameters
    ----------
    initial_path
        Optional file or directory used to pre-populate the source field.
    discovered_files
        Candidate files already registered by the server-side session.

    Returns
    -------
    dash.html.Div
        Portal layout with a control sidebar and a main visualization area.
    """
    selected_file = str(discovered_files[0]) if discovered_files else None
    return html.Div(
        [
            _header(),
            html.Main(
                [
                    _sidebar(
                        initial_path=initial_path,
                        discovered_files=discovered_files,
                        selected_file=selected_file,
                    ),
                    html.Div(
                        [
                            _image_panel(),
                            _terminal_panel(),
                        ],
                        style={
                            "display": "grid",
                            "gridTemplateRows": "minmax(28rem, 1fr) auto",
                            "gap": "1rem",
                            "minWidth": "0",
                        },
                    ),
                ],
                style={
                    "display": "grid",
                    "gridTemplateColumns": "22rem minmax(0, 1fr)",
                    "gap": "1rem",
                    "padding": "1rem",
                    "backgroundColor": "#e5e7eb",
                    "minHeight": "calc(100vh - 5.25rem)",
                    "boxSizing": "border-box",
                },
            ),
        ],
        style={
            "fontFamily": "Arial, sans-serif",
            "color": "#111827",
            "minHeight": "100vh",
            "backgroundColor": "#e5e7eb",
        },
    )


def _header() -> html.Header:
    """Build the compact portal header."""
    return html.Header(
        [
            html.H1(
                "GONet Astrometry Calibrator",
                style={"margin": "0", "fontSize": "1.55rem"},
            ),
            html.P(
                "Multi-image stellar detection and astrometric calibration",
                style={"margin": "0.25rem 0 0", "color": "#cbd5e1"},
            ),
        ],
        style={
            "padding": "0.85rem 1.25rem",
            "backgroundColor": "#111827",
            "color": "white",
            "boxSizing": "border-box",
        },
    )


def _sidebar(
    *,
    initial_path: Path | None,
    discovered_files: Sequence[Path],
    selected_file: str | None,
) -> html.Aside:
    """Build the sidebar containing every interactive control."""
    options = [{"label": str(path), "value": str(path)} for path in discovered_files]
    tracking_values = [
        str(path) for path in discovered_files[:DEFAULT_TRACKING_IMAGE_COUNT]
    ]
    return html.Aside(
        [
            _control_group(
                "Input images",
                [
                    html.Label(
                        "Files or folders",
                        htmlFor=ids.SOURCE_PATHS,
                        style=_label_style(),
                    ),
                    dcc.Textarea(
                        id=ids.SOURCE_PATHS,
                        value=str(initial_path) if initial_path else "",
                        placeholder=(
                            "Enter one file or folder per line. Folders are scanned "
                            "for original GONet .jpg files."
                        ),
                        style={
                            "width": "100%",
                            "height": "6.5rem",
                            "resize": "vertical",
                            "boxSizing": "border-box",
                        },
                    ),
                    dcc.Checklist(
                        id=ids.RECURSIVE_SEARCH,
                        options=[
                            {
                                "label": " Search subfolders",
                                "value": "recursive",
                            }
                        ],
                        value=["recursive"],
                        style={"margin": "0.6rem 0"},
                    ),
                    html.Button(
                        "Discover images",
                        id=ids.DISCOVER_FILES,
                        n_clicks=0,
                        style=_primary_button_style(),
                    ),
                    html.Div(
                        _initial_discovery_status(discovered_files),
                        id=ids.DISCOVERY_STATUS,
                        role="status",
                        style=_status_style(),
                    ),
                    html.Label(
                        "Current image",
                        htmlFor=ids.FILE_SELECT,
                        style=_label_style(),
                    ),
                    dcc.Dropdown(
                        id=ids.FILE_SELECT,
                        options=options,
                        value=selected_file,
                        clearable=False,
                        placeholder="Discover one or more images",
                        style={"fontSize": "0.85rem"},
                    ),
                    html.Button(
                        "Load selected image",
                        id=ids.LOAD_IMAGE,
                        n_clicks=0,
                        style=_secondary_button_style(),
                    ),
                ],
            ),
            _control_group(
                "Display",
                [
                    html.Label(
                        "Native Bayer channel",
                        htmlFor=ids.CHANNEL,
                        style=_label_style(),
                    ),
                    dcc.Dropdown(
                        id=ids.CHANNEL,
                        options=[
                            {"label": channel, "value": channel}
                            for channel in GONET_CHANNELS
                        ],
                        value="green1",
                        clearable=False,
                    ),
                    html.Div(
                        "Select another channel to redraw the cached current image.",
                        style=_help_style(),
                    ),
                    dcc.Checklist(
                        id=ids.SHOW_DETECTION_MASKS,
                        options=[
                            {
                                "label": " Show field and bright-region masks",
                                "value": "show",
                            }
                        ],
                        value=["show"],
                        style={"marginTop": "0.65rem", "fontSize": "0.82rem"},
                    ),
                    html.Div(
                        "Mask boundaries appear after source detection has prepared "
                        "the full-resolution frame.",
                        style=_help_style(),
                    ),
                ],
            ),
            _control_group(
                "Star detection",
                [
                    html.Label(
                        "Detection algorithm",
                        htmlFor=ids.DETECTOR,
                        style=_label_style(),
                    ),
                    dcc.Dropdown(
                        id=ids.DETECTOR,
                        options=DETECTOR_OPTIONS,
                        value="sep",
                        clearable=False,
                        persistence=True,
                        persistence_type="session",
                    ),
                    html.Div(
                        [
                            html.Label(
                                "Threshold (σ)",
                                htmlFor=ids.DETECTION_THRESHOLD,
                                style=_label_style(),
                            ),
                            dcc.Input(
                                id=ids.DETECTION_THRESHOLD,
                                type="number",
                                min=0.1,
                                step=0.5,
                                value=5.0,
                                debounce=True,
                                style={"width": "100%", "boxSizing": "border-box"},
                            ),
                            html.Label(
                                "Approximate FWHM (sensor px)",
                                htmlFor=ids.DETECTION_FWHM,
                                style=_label_style(),
                            ),
                            dcc.Input(
                                id=ids.DETECTION_FWHM,
                                type="number",
                                min=0.5,
                                step=0.5,
                                value=3.0,
                                debounce=True,
                                style={"width": "100%", "boxSizing": "border-box"},
                            ),
                            html.Label(
                                "Minimum connected pixels",
                                htmlFor=ids.DETECTION_MIN_PIXELS,
                                style=_label_style(),
                            ),
                            dcc.Input(
                                id=ids.DETECTION_MIN_PIXELS,
                                type="number",
                                min=1,
                                step=1,
                                value=5,
                                debounce=True,
                                style={"width": "100%", "boxSizing": "border-box"},
                            ),
                        ],
                        style={
                            "display": "grid",
                            "gap": "0.4rem",
                            "marginTop": "0.65rem",
                        },
                    ),
                    html.Button(
                        "Detect sources",
                        id=ids.DETECT_SOURCES,
                        n_clicks=0,
                        style=_primary_button_style(),
                    ),
                    html.Div(
                        "Run a detector on the selected full-resolution Bayer frame.",
                        id=ids.DETECTION_STATUS,
                        role="status",
                        style=_status_style(),
                    ),
                    html.Div(
                        "All backends share Bayer-parity normalization and return "
                        "native full-sensor coordinates. Marker shapes and colors "
                        "show non-destructive diagnostic classes.",
                        style=_help_style(),
                    ),
                ],
            ),
            _control_group(
                "Star tracking",
                [
                    html.Label(
                        "Sequence images",
                        htmlFor=ids.TRACKING_FILES,
                        style=_label_style(),
                    ),
                    dcc.Dropdown(
                        id=ids.TRACKING_FILES,
                        options=options,
                        value=tracking_values,
                        multi=True,
                        placeholder="Select at least two discovered images",
                        style={"fontSize": "0.82rem"},
                    ),
                    html.Div(
                        "Up to the first five discovered images are selected by "
                        "default to avoid accidentally processing a large folder.",
                        style=_help_style(),
                    ),
                    html.Div(
                        [
                            html.Label(
                                "Maximum motion (sensor px/min)",
                                htmlFor=ids.TRACK_MAX_SPEED,
                                style=_label_style(),
                            ),
                            dcc.Input(
                                id=ids.TRACK_MAX_SPEED,
                                type="number",
                                min=0.1,
                                step=1.0,
                                value=20.0,
                                debounce=True,
                                style={"width": "100%", "boxSizing": "border-box"},
                            ),
                            html.Label(
                                "Prediction tolerance (sensor px)",
                                htmlFor=ids.TRACK_PREDICTION_TOLERANCE,
                                style=_label_style(),
                            ),
                            dcc.Input(
                                id=ids.TRACK_PREDICTION_TOLERANCE,
                                type="number",
                                min=0.1,
                                step=0.5,
                                value=6.0,
                                debounce=True,
                                style={"width": "100%", "boxSizing": "border-box"},
                            ),
                            html.Label(
                                "Maximum track gap (minutes)",
                                htmlFor=ids.TRACK_MAX_GAP,
                                style=_label_style(),
                            ),
                            dcc.Input(
                                id=ids.TRACK_MAX_GAP,
                                type="number",
                                min=0.1,
                                step=1.0,
                                value=15.0,
                                debounce=True,
                                style={"width": "100%", "boxSizing": "border-box"},
                            ),
                            html.Label(
                                "Minimum detections per track",
                                htmlFor=ids.TRACK_MIN_LENGTH,
                                style=_label_style(),
                            ),
                            dcc.Input(
                                id=ids.TRACK_MIN_LENGTH,
                                type="number",
                                min=2,
                                step=1,
                                value=3,
                                debounce=True,
                                style={"width": "100%", "boxSizing": "border-box"},
                            ),
                        ],
                        style={
                            "display": "grid",
                            "gap": "0.4rem",
                            "marginTop": "0.65rem",
                        },
                    ),
                    html.Button(
                        "Build tracks",
                        id=ids.BUILD_TRACKS,
                        n_clicks=0,
                        style=_primary_button_style(),
                    ),
                    html.Div(
                        "Process selected images sequentially and build bootstrap "
                        "image-plane tracklets.",
                        id=ids.TRACKING_STATUS,
                        role="status",
                        style=_status_style(),
                    ),
                    html.Div(
                        "This first linker uses the actual exposure timestamps for "
                        "motion, "
                        "gap handling, and prediction. It seeds, but does not replace, "
                        "the later spherical rotation-axis solution.",
                        style=_help_style(),
                    ),
                ],
            ),
            _control_group(
                "Current image",
                [
                    html.Div(
                        "Discover and select a GONet image.",
                        id=ids.STATUS,
                        role="status",
                        style=_status_style(),
                    ),
                    html.Table(
                        html.Tbody(id=ids.METADATA_BODY),
                        style={"width": "100%", "fontSize": "0.82rem"},
                    ),
                ],
            ),
            html.Button(
                "Exit",
                id=ids.BTN_EXIT,
                n_clicks=0,
                title="Close the desktop calibration portal",
                style=_exit_button_style(),
            ),
        ],
        style={
            "display": "flex",
            "flexDirection": "column",
            "gap": "0.85rem",
            "padding": "0.9rem",
            "backgroundColor": "#f8fafc",
            "borderRadius": "0.55rem",
            "boxShadow": "0 1px 4px rgba(15, 23, 42, 0.18)",
            "overflowY": "auto",
            "maxHeight": "calc(100vh - 7.25rem)",
            "boxSizing": "border-box",
        },
    )


def _control_group(title: str, children: list[object]) -> html.Section:
    """Build a visually separated sidebar control group."""
    return html.Section(
        [
            html.H2(
                title,
                style={
                    "margin": "0 0 0.7rem",
                    "fontSize": "1rem",
                    "color": "#1e293b",
                },
            ),
            *children,
        ],
        style={
            "padding": "0.8rem",
            "backgroundColor": "white",
            "border": "1px solid #dbe3ee",
            "borderRadius": "0.45rem",
        },
    )


def _image_panel() -> html.Section:
    """Build the interactive image panel."""
    return html.Section(
        dcc.Graph(
            id=ids.IMAGE_GRAPH,
            figure=empty_image_figure(),
            config={
                "displaylogo": False,
                "scrollZoom": True,
                "modeBarButtonsToRemove": ["select2d", "lasso2d"],
            },
            style={"height": "100%", "minHeight": "28rem"},
        ),
        style={
            "minWidth": "0",
            "minHeight": "0",
            "backgroundColor": "#111827",
            "borderRadius": "0.55rem",
            "overflow": "hidden",
            "boxShadow": "0 1px 4px rgba(15, 23, 42, 0.2)",
        },
    )


def _terminal_panel() -> html.Section:
    """Build the read-only activity terminal."""
    return html.Section(
        [
            html.Div(
                [
                    html.H2(
                        "Activity log",
                        style={"margin": "0", "fontSize": "1rem"},
                    ),
                    html.Span(
                        "Runtime messages and calibration progress",
                        style={"fontSize": "0.82rem", "color": "#9ca3af"},
                    ),
                ],
                style={
                    "padding": "0.7rem 1rem",
                    "borderBottom": "1px solid #374151",
                },
            ),
            html.Pre(
                "Portal activity will appear here...",
                id=ids.LOG_WINDOW,
                role="log",
                **{"aria-live": "polite"},
                style={
                    "height": "8rem",
                    "margin": "0",
                    "padding": "0.8rem 1rem",
                    "overflowY": "auto",
                    "whiteSpace": "pre-wrap",
                    "overflowWrap": "anywhere",
                    "fontFamily": "Menlo, Consolas, monospace",
                    "fontSize": "0.8rem",
                    "lineHeight": "1.45",
                    "backgroundColor": "#030712",
                    "color": "#d1d5db",
                    "userSelect": "text",
                    "WebkitUserSelect": "text",
                    "cursor": "text",
                },
            ),
            dcc.Interval(id=ids.LOG_POLL_INTERVAL, interval=500, n_intervals=0),
            html.Div(id=ids.LOG_AUTOSCROLL_DUMMY, style={"display": "none"}),
        ],
        style={
            "backgroundColor": "#111827",
            "color": "white",
            "borderRadius": "0.55rem",
            "overflow": "hidden",
            "boxShadow": "0 1px 4px rgba(15, 23, 42, 0.2)",
        },
    )


def _initial_discovery_status(discovered_files: Sequence[Path]) -> str:
    """Return the initial sidebar status for server-preloaded candidates."""
    if not discovered_files:
        return "No input catalog has been created yet."
    noun = "file" if len(discovered_files) == 1 else "files"
    return f"Registered {len(discovered_files)} candidate {noun}."


def _label_style() -> dict[str, str]:
    """Return shared label styling."""
    return {
        "display": "block",
        "margin": "0.55rem 0 0.3rem",
        "fontSize": "0.84rem",
        "fontWeight": "600",
    }


def _help_style() -> dict[str, str]:
    """Return shared explanatory-text styling."""
    return {
        "marginTop": "0.45rem",
        "fontSize": "0.75rem",
        "lineHeight": "1.35",
        "color": "#64748b",
    }


def _status_style() -> dict[str, str]:
    """Return shared status-message styling."""
    return {
        "margin": "0.55rem 0",
        "fontSize": "0.78rem",
        "lineHeight": "1.4",
        "color": "#475569",
        "overflowWrap": "anywhere",
    }


def _primary_button_style() -> dict[str, str]:
    """Return styling for the main discovery action."""
    return {
        "width": "100%",
        "padding": "0.5rem 0.7rem",
        "border": "1px solid #2563eb",
        "borderRadius": "0.35rem",
        "backgroundColor": "#2563eb",
        "color": "white",
        "cursor": "pointer",
    }


def _secondary_button_style() -> dict[str, str]:
    """Return styling for the image-load action."""
    return {
        "width": "100%",
        "marginTop": "0.65rem",
        "padding": "0.5rem 0.7rem",
        "border": "1px solid #64748b",
        "borderRadius": "0.35rem",
        "backgroundColor": "#f8fafc",
        "color": "#0f172a",
        "cursor": "pointer",
    }


def _exit_button_style() -> dict[str, str]:
    """Return styling for the desktop Exit button."""
    return {
        "width": "100%",
        "marginTop": "auto",
        "padding": "0.5rem 0.7rem",
        "border": "1px solid #ef4444",
        "borderRadius": "0.35rem",
        "backgroundColor": "#7f1d1d",
        "color": "white",
        "cursor": "pointer",
    }
