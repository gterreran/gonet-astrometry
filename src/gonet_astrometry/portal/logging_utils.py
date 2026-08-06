"""In-memory logging support for the Dash portal terminal."""

from __future__ import annotations

import logging
import threading
from collections import deque

DEFAULT_MAX_LOG_RECORDS = 1000


class PortalLogHandler(logging.Handler):
    """Store formatted log records in a bounded, thread-safe buffer.

    Parameters
    ----------
    max_records
        Maximum number of formatted records retained in memory. Older records
        are discarded automatically when the limit is reached.
    """

    def __init__(self, max_records: int = DEFAULT_MAX_LOG_RECORDS) -> None:
        super().__init__()
        if max_records <= 0:
            raise ValueError("max_records must be greater than zero")
        self._records: deque[str] = deque(maxlen=max_records)
        self._lock = threading.RLock()

    def emit(self, record: logging.LogRecord) -> None:
        """Format and append one logging record to the buffer."""
        try:
            message = self.format(record)
        except Exception:  # pragma: no cover - delegated logging failure path
            self.handleError(record)
            return

        with self._lock:
            self._records.append(message)

    def get_logs(self) -> str:
        """Return the buffered records as newline-separated text."""
        with self._lock:
            return "\n".join(self._records)

    def clear(self) -> None:
        """Remove all buffered records."""
        with self._lock:
            self._records.clear()


global_log_handler = PortalLogHandler()
global_log_handler.setFormatter(
    logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )
)


def configure_portal_logging(
    *,
    level: int | None = None,
    clear_buffer: bool = False,
) -> None:
    """Attach the portal log buffer to the package logger.

    The configuration is idempotent, so application factories and launchers may
    call it repeatedly without adding duplicate handlers.

    Parameters
    ----------
    level
        Optional logging level for the ``gonet_astrometry`` logger and portal
        handler. When omitted, their current levels are preserved.
    clear_buffer
        Whether to discard records accumulated by a previous portal run.
    """
    package_logger = logging.getLogger("gonet_astrometry")

    if global_log_handler not in package_logger.handlers:
        package_logger.addHandler(global_log_handler)

    if level is not None:
        package_logger.setLevel(level)
        global_log_handler.setLevel(level)

    if clear_buffer:
        global_log_handler.clear()
