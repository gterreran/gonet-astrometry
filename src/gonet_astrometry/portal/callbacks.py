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
from gonet_astrometry.portal import ids
from gonet_astrometry.portal.discovery import parse_source_paths
from gonet_astrometry.portal.figures import channel_image_figure, empty_image_figure
from gonet_astrometry.portal.session import PortalSession

logger = logging.getLogger(__name__)


def discover_source_paths(
    source_value: str | None,
    recursive_values: list[str] | None,
    *,
    session: PortalSession,
    current_selection: str | None = None,
) -> tuple[list[dict[str, str]], str | None, str]:
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
        Existing selected path to preserve when still available.

    Returns
    -------
    tuple
        Dropdown options, selected option value, and user-facing status text.
    """
    source_paths = parse_source_paths(source_value)
    if not source_paths:
        message = "Enter at least one file or folder before discovery."
        logger.warning(message)
        session.discover(())
        return [], None, message

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

    if result.files:
        logger.info(result.summary)
    else:
        logger.warning(result.summary)
    return options, selected, result.summary


def load_channel_view(
    path_value: str | None,
    channel: GONetChannel | None,
    *,
    session: PortalSession,
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
    return (
        channel_image_figure(
            data,
            channel=channel,
            source_name=str(image_path),
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
        Output(ids.DISCOVERY_STATUS, "children"),
        Input(ids.DISCOVER_FILES, "n_clicks"),
        State(ids.SOURCE_PATHS, "value"),
        State(ids.RECURSIVE_SEARCH, "value"),
        State(ids.FILE_SELECT, "value"),
        prevent_initial_call=True,
    )
    def _discover_images(
        _n_clicks: int,
        source_value: str | None,
        recursive_values: list[str] | None,
        current_selection: str | None,
    ) -> tuple[list[dict[str, str]], str | None, str]:
        return discover_source_paths(
            source_value,
            recursive_values,
            session=session,
            current_selection=current_selection,
        )

    @app.callback(  # type: ignore[untyped-decorator]
        Output(ids.IMAGE_GRAPH, "figure"),
        Output(ids.STATUS, "children"),
        Output(ids.METADATA_BODY, "children"),
        Input(ids.LOAD_IMAGE, "n_clicks"),
        Input(ids.CHANNEL, "value"),
        State(ids.FILE_SELECT, "value"),
    )
    def _load_image(
        _n_clicks: int,
        channel: GONetChannel | None,
        path_value: str | None,
    ) -> tuple[go.Figure, str, list[html.Tr]]:
        return load_channel_view(path_value, channel, session=session)


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
