"""Plotly figures used by the Dash portal."""

from __future__ import annotations

from math import ceil

import numpy as np
import plotly.graph_objects as go
from numpy.typing import NDArray
from scipy.ndimage import binary_erosion

from gonet_astrometry.adapters.gonet_wizard import GONetChannel
from gonet_astrometry.models.detection import DetectionCatalog

_MAX_MASK_POINTS = 6000


def empty_image_figure(message: str = "Load a GONet image to begin.") -> go.Figure:
    """Create the empty image-panel figure.

    Parameters
    ----------
    message
        Text displayed in the center of the empty panel.

    Returns
    -------
    plotly.graph_objects.Figure
        Empty figure configured for the image-inspection panel.
    """
    figure = go.Figure()
    figure.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font={"size": 18, "color": "#6b7280"},
    )
    figure.update_layout(
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        paper_bgcolor="#f3f4f6",
        plot_bgcolor="#f3f4f6",
        xaxis={"visible": False},
        yaxis={"visible": False},
    )
    return figure


def channel_image_figure(
    data: NDArray[np.float64],
    *,
    channel: GONetChannel,
    source_name: str,
    detections: DetectionCatalog | None = None,
    field_mask: NDArray[np.bool_] | None = None,
    dynamic_mask: NDArray[np.bool_] | None = None,
) -> go.Figure:
    """Create an interactive grayscale figure for one native GONet channel.

    Full-sensor detections and masks are mapped to compact-channel coordinates
    only for visualization. Their scientific coordinates remain unchanged.

    Parameters
    ----------
    data
        Two-dimensional native channel array in Wizard coordinates.
    channel
        Channel represented by ``data``.
    source_name
        Source name used to preserve the user's zoom state per image/channel.
    detections
        Optional full-sensor detection catalog.
    field_mask
        Optional full-sensor mask with ``True`` inside the usable field.
    dynamic_mask
        Optional full-sensor mask with ``True`` for bright contaminants.

    Returns
    -------
    plotly.graph_objects.Figure
        Heatmap with optional compact-coordinate mask and detection overlays.
    """
    zmin, zmax = _display_limits(data)
    figure = go.Figure(
        go.Heatmap(
            z=data,
            colorscale="Gray",
            showscale=False,
            zmin=zmin,
            zmax=zmax,
            hovertemplate=("column=%{x}<br>row=%{y}<br>value=%{z:.3f}<extra></extra>"),
        )
    )
    figure.update_layout(
        dragmode="pan",
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        paper_bgcolor="#111827",
        plot_bgcolor="#111827",
        uirevision=f"{source_name}:{channel}",
    )
    figure.update_xaxes(title="Compact channel column", constrain="domain")
    figure.update_yaxes(
        title="Compact channel row",
        autorange="reversed",
        scaleanchor="x",
    )

    if field_mask is not None:
        _add_mask_boundary(
            figure,
            _compact_field_mask(field_mask),
            color="#22d3ee",
            name="Usable field boundary",
        )
    if dynamic_mask is not None:
        _add_mask_boundary(
            figure,
            _compact_dynamic_mask(dynamic_mask),
            color="#f472b6",
            name="Bright-region mask",
        )

    if detections is not None and detections.detections:
        figure.add_trace(
            go.Scattergl(
                x=[detection.x / 2.0 for detection in detections.detections],
                y=[detection.y / 2.0 for detection in detections.detections],
                mode="markers",
                marker={
                    "symbol": "circle-open",
                    "size": 10,
                    "color": "#f97316",
                    "line": {"width": 1.5},
                },
                customdata=[
                    [detection.identifier, detection.signal_to_noise]
                    for detection in detections.detections
                ],
                hovertemplate=(
                    "source=%{customdata[0]}<br>"
                    "sensor x=%{x:.2f} × 2<br>"
                    "sensor y=%{y:.2f} × 2<br>"
                    "S/N=%{customdata[1]:.2f}<extra></extra>"
                ),
                name="Detections",
                showlegend=False,
            )
        )
    return figure


def _add_mask_boundary(
    figure: go.Figure,
    mask: NDArray[np.bool_],
    *,
    color: str,
    name: str,
) -> None:
    """Add a decimated compact-coordinate boundary for a nontrivial mask."""
    if not np.any(mask) or np.all(mask):
        return
    boundary = mask & ~binary_erosion(mask, border_value=0)
    rows, columns = np.nonzero(boundary)
    if rows.size == 0:
        return
    step = max(1, ceil(rows.size / _MAX_MASK_POINTS))
    figure.add_trace(
        go.Scattergl(
            x=columns[::step],
            y=rows[::step],
            mode="markers",
            marker={"size": 3, "color": color, "opacity": 0.9},
            hoverinfo="skip",
            name=name,
            showlegend=True,
        )
    )


def _compact_field_mask(mask: NDArray[np.bool_]) -> NDArray[np.bool_]:
    """Require all four full-sensor pixels in a compact quad to be usable."""
    height = mask.shape[0] // 2
    width = mask.shape[1] // 2
    return np.asarray(
        mask[0 : 2 * height : 2, 0 : 2 * width : 2]
        & mask[0 : 2 * height : 2, 1 : 2 * width : 2]
        & mask[1 : 2 * height : 2, 0 : 2 * width : 2]
        & mask[1 : 2 * height : 2, 1 : 2 * width : 2],
        dtype=np.bool_,
    )


def _compact_dynamic_mask(mask: NDArray[np.bool_]) -> NDArray[np.bool_]:
    """Mark a compact quad when any full-sensor pixel is dynamically masked."""
    height = mask.shape[0] // 2
    width = mask.shape[1] // 2
    return np.asarray(
        mask[0 : 2 * height : 2, 0 : 2 * width : 2]
        | mask[0 : 2 * height : 2, 1 : 2 * width : 2]
        | mask[1 : 2 * height : 2, 0 : 2 * width : 2]
        | mask[1 : 2 * height : 2, 1 : 2 * width : 2],
        dtype=np.bool_,
    )


def _display_limits(data: NDArray[np.float64]) -> tuple[float, float]:
    """Return robust grayscale limits without modifying channel values."""
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return 0.0, 1.0

    lower, upper = np.percentile(finite, (1.0, 99.5))
    if lower == upper:
        upper = lower + 1.0
    return float(lower), float(upper)
