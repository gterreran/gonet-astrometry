import logging

import pytest

from gonet_astrometry.portal.logging_utils import (
    PortalLogHandler,
    configure_portal_logging,
    global_log_handler,
)


def test_portal_log_handler_validates_capacity() -> None:
    with pytest.raises(ValueError, match="greater than zero"):
        PortalLogHandler(max_records=0)


def test_portal_log_handler_formats_bounds_and_clears_records() -> None:
    handler = PortalLogHandler(max_records=2)
    handler.setFormatter(logging.Formatter("%(levelname)s:%(message)s"))
    logger = logging.getLogger("tests.portal.buffer")
    logger.handlers = [handler]
    logger.propagate = False
    logger.setLevel(logging.INFO)

    logger.info("first")
    logger.warning("second")
    logger.error("third")

    assert handler.get_logs() == "WARNING:second\nERROR:third"

    handler.clear()
    assert handler.get_logs() == ""


def test_configure_portal_logging_is_idempotent_and_can_clear() -> None:
    package_logger = logging.getLogger("gonet_astrometry")
    original_handlers = package_logger.handlers.copy()
    original_level = package_logger.level
    original_handler_level = global_log_handler.level

    try:
        package_logger.handlers = [
            handler
            for handler in package_logger.handlers
            if handler is not global_log_handler
        ]
        global_log_handler.clear()
        global_log_handler.emit(
            logging.LogRecord(
                "gonet_astrometry.test",
                logging.INFO,
                __file__,
                1,
                "existing",
                (),
                None,
            )
        )

        configure_portal_logging(level=logging.DEBUG, clear_buffer=True)
        configure_portal_logging()

        assert package_logger.handlers.count(global_log_handler) == 1
        assert package_logger.level == logging.DEBUG
        assert global_log_handler.level == logging.DEBUG
        assert global_log_handler.get_logs() == ""
    finally:
        package_logger.handlers = original_handlers
        package_logger.setLevel(original_level)
        global_log_handler.setLevel(original_handler_level)
        global_log_handler.clear()
