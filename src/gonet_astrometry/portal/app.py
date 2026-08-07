"""Dash server and pywebview launch helpers for the astrometry portal.

The portal keeps the Dash application and desktop window concerns separate.
:func:`create_app` constructs an isolated Dash application for tests and
embedding, :func:`run_portal_server` runs only the local HTTP server, and
:func:`run_portal` starts the server in a background thread before opening a
native pywebview window on the main thread.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.request import urlopen

from dash import Dash

from gonet_astrometry.adapters.gonet_wizard import (
    load_gonet_file_raw,
    load_gonet_image,
)
from gonet_astrometry.detection.registry import create_detector
from gonet_astrometry.portal.callbacks import register_callbacks
from gonet_astrometry.portal.layout import build_layout
from gonet_astrometry.portal.logging_utils import configure_portal_logging
from gonet_astrometry.portal.session import (
    DetectorFactory,
    FrameLoader,
    PortalSession,
    RawLoader,
)
from gonet_astrometry.portal.system import register_system_callbacks

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8050
DEFAULT_WINDOW_WIDTH = 1800
DEFAULT_WINDOW_HEIGHT = 1200
DEFAULT_WINDOW_MIN_SIZE = (1000, 700)
SERVER_STARTUP_TIMEOUT = 10.0

logger = logging.getLogger(__name__)


def create_app(
    *,
    initial_path: Path | None = None,
    raw_loader: RawLoader = load_gonet_file_raw,
    frame_loader: FrameLoader = load_gonet_image,
    detector_factory: DetectorFactory = create_detector,
    log_level: int = logging.INFO,
) -> Dash:
    """Create and configure the Dash portal.

    Parameters
    ----------
    initial_path
        Optional image file or directory used to seed input discovery.
    raw_loader
        Native GONet loading function used by the image callback.
    frame_loader
        Scientific frame loader used by source detection.
    detector_factory
        Factory used to construct the selected source detector.
    log_level
        Logging level captured by the portal activity terminal.

    Returns
    -------
    dash.Dash
        Configured Dash application.
    """
    configure_portal_logging(level=log_level)
    session = PortalSession(
        loader=raw_loader,
        frame_loader=frame_loader,
        detector_factory=detector_factory,
    )
    if initial_path is not None:
        result = session.discover((initial_path,), recursive=True)
        if not result.files:
            logger.warning(
                "The initial portal input did not yield any candidate GONet files: %s",
                initial_path,
            )

    app = Dash(__name__)
    app.title = "GONet Astrometry Calibrator"
    app.server.config["astrometry_session"] = session
    app.layout = build_layout(initial_path, session.files)
    register_callbacks(app, session=session)
    register_system_callbacks(app)
    return app


def run_portal_server(
    *,
    initial_path: Path | None = None,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    debug: bool = False,
) -> None:
    """Run only the local Dash server.

    This mode is useful during browser-based development and automated tests.
    The reloader is always disabled so the function is safe to run from the
    background thread created by :func:`run_portal`.

    Parameters
    ----------
    initial_path
        Optional image file or directory used to seed input discovery.
    host
        Interface on which Dash should listen.
    port
        TCP port on which Dash should listen.
    debug
        Whether to enable Dash development features.
    """
    level = logging.DEBUG if debug else logging.INFO
    configure_portal_logging(level=level)
    logger.info("Starting Dash server at http://%s:%s.", host, port)
    app = create_app(initial_path=initial_path, log_level=level)
    app.run(
        host=host,
        port=port,
        debug=debug,
        use_reloader=False,
    )


def run_portal(
    *,
    initial_path: Path | None = None,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    debug: bool = False,
) -> None:
    """Launch the calibration portal in an independent desktop window.

    The Dash server runs in a daemon thread while pywebview owns the main-thread
    GUI event loop. The window exposes GONet Wizard's ``WebviewAPI`` so native
    file and folder dialogs can be connected to the portal in a later layout
    increment without changing the desktop launcher.

    Parameters
    ----------
    initial_path
        Optional image file or directory used to seed input discovery.
    host
        Interface on which Dash should listen.
    port
        TCP port on which Dash should listen.
    debug
        Whether to enable Dash and pywebview debugging features.

    Raises
    ------
    RuntimeError
        If pywebview, the Wizard webview API, or the local Dash server cannot be
        started.
    """
    level = logging.DEBUG if debug else logging.INFO
    configure_portal_logging(level=level, clear_buffer=True)
    logger.info("Launching GONet Astrometry desktop portal.")

    webview, api_factory = _load_webview_components()
    window_url = _local_window_url(host, port)

    _start_server_thread(
        initial_path=initial_path,
        host=host,
        port=port,
        debug=debug,
    )
    _wait_for_server(window_url)
    logger.info("Dash server is ready at %s.", window_url)

    webview.create_window(
        "GONet Astrometry Calibrator",
        window_url,
        width=DEFAULT_WINDOW_WIDTH,
        height=DEFAULT_WINDOW_HEIGHT,
        min_size=DEFAULT_WINDOW_MIN_SIZE,
        background_color="#e5e7eb",
        js_api=api_factory(),
    )
    webview.start(debug=debug)


def _load_webview_components() -> tuple[Any, Callable[[], Any]]:
    """Import pywebview and the GONet Wizard JavaScript bridge lazily."""
    try:
        import webview
        from GONet_Wizard.ui.api import WebviewAPI
    except ImportError as exc:
        raise RuntimeError(
            "The desktop portal requires pywebview and GONet Wizard's "
            "WebviewAPI. Install the GUI dependencies in the active "
            "environment."
        ) from exc

    return webview, WebviewAPI


def _start_server_thread(
    *,
    initial_path: Path | None,
    host: str,
    port: int,
    debug: bool,
) -> threading.Thread:
    """Start the Dash server in a daemon thread and return the thread."""
    thread = threading.Thread(
        target=run_portal_server,
        kwargs={
            "initial_path": initial_path,
            "host": host,
            "port": port,
            "debug": debug,
        },
        name="gonet-astrometry-dash",
        daemon=True,
    )
    thread.start()
    return thread


def _local_window_url(host: str, port: int) -> str:
    """Return a browser-reachable URL for a locally bound Dash server."""
    window_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
    return f"http://{window_host}:{port}"


def _wait_for_server(
    url: str,
    *,
    timeout: float = SERVER_STARTUP_TIMEOUT,
    poll_interval: float = 0.05,
) -> None:
    """Wait until the Dash endpoint responds or raise a startup error."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urlopen(url, timeout=min(0.5, timeout)):
                return
        except OSError:
            time.sleep(poll_interval)

    raise RuntimeError(
        f"The Dash server did not become available at {url} within "
        f"{timeout:.1f} seconds."
    )
