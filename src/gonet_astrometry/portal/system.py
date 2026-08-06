"""Application-level callbacks for portal logging and window shutdown."""

from __future__ import annotations

import logging
from typing import Any

from dash import Dash, Input, Output

from gonet_astrometry.portal import ids
from gonet_astrometry.portal.logging_utils import global_log_handler

logger = logging.getLogger(__name__)

_EMPTY_LOG_MESSAGE = "Portal activity will appear here..."

_AUTOSCROLL_SCRIPT = """
function(logText) {
    const getElement = () => document.getElementById("astrometry-log-window");

    const bindScrollListener = (element) => {
        if (element.dataset.autoScrollBound === "1") return;

        const updateFlag = () => {
            const nearBottom =
                (element.scrollTop + element.clientHeight) >=
                (element.scrollHeight - 30);
            window.__gonetLogShouldAutoScroll = nearBottom;
        };

        element.addEventListener("scroll", updateFlag, {passive: true});
        element.dataset.autoScrollBound = "1";
        updateFlag();
    };

    const scrollToBottomIfNeeded = () => {
        const element = getElement();
        if (!element) return;

        bindScrollListener(element);
        const shouldScroll = window.__gonetLogShouldAutoScroll !== false;

        if (shouldScroll) {
            element.scrollTop = element.scrollHeight;
            window.__gonetLogShouldAutoScroll = true;
        }
    };

    requestAnimationFrame(() => {
        requestAnimationFrame(scrollToBottomIfNeeded);
    });
    setTimeout(scrollToBottomIfNeeded, 50);

    return "";
}
"""


def get_log_text() -> str:
    """Return the current portal log buffer or an empty-state message."""
    return global_log_handler.get_logs() or _EMPTY_LOG_MESSAGE


def close_desktop_window() -> bool:
    """Request closure of the active pywebview window.

    Returns
    -------
    bool
        ``True`` when a close request was sent. ``False`` when the portal is
        running without an active pywebview window or the request failed.
    """
    try:
        import webview
    except ImportError:
        logger.warning("Exit is unavailable because pywebview is not installed.")
        return False

    windows: list[Any] = list(getattr(webview, "windows", []))
    if not windows:
        logger.warning("Exit is unavailable in server-only mode.")
        return False

    try:
        windows[0].evaluate_js("window.pywebview.api.close_window()")
    except Exception:
        logger.exception("Unable to close the pywebview window.")
        return False

    logger.info("Portal exit requested.")
    return True


def register_system_callbacks(app: Dash) -> None:
    """Register terminal polling, auto-scroll, and exit callbacks."""

    @app.callback(  # type: ignore[untyped-decorator]
        Output(ids.LOG_WINDOW, "children"),
        Input(ids.LOG_POLL_INTERVAL, "n_intervals"),
    )
    def _update_log_window(_n_intervals: int) -> str:
        return get_log_text()

    @app.callback(  # type: ignore[untyped-decorator]
        Output(ids.BTN_EXIT, "disabled"),
        Input(ids.BTN_EXIT, "n_clicks"),
        prevent_initial_call=True,
    )
    def _exit_app(_n_clicks: int) -> bool:
        return close_desktop_window()

    app.clientside_callback(
        _AUTOSCROLL_SCRIPT,
        Output(ids.LOG_AUTOSCROLL_DUMMY, "children"),
        Input(ids.LOG_WINDOW, "children"),
    )
