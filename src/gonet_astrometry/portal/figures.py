"""Plotly figures used by the Dash portal."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from numpy.typing import NDArray

from gonet_astrometry.adapters.gonet_wizard import GONetChannel


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
) -> go.Figure:
    """Create an interactive grayscale figure for one native GONet channel.

    Parameters
    ----------
    data
        Two-dimensional native channel array in Wizard coordinates.
    channel
        Channel represented by ``data``.
    source_name
        Source name used to preserve the user's zoom state per image/channel.

    Returns
    -------
    plotly.graph_objects.Figure
        Heatmap with native compact-channel row and column coordinates.
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
    return figure


def _display_limits(data: NDArray[np.float64]) -> tuple[float, float]:
    """Return robust grayscale limits without modifying channel values."""
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return 0.0, 1.0

    lower, upper = np.percentile(finite, (1.0, 99.5))
    if lower == upper:
        upper = lower + 1.0
    return float(lower), float(upper)
