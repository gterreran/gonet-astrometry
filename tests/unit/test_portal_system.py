import logging
import sys
from types import ModuleType

import pytest

pytest.importorskip("dash")

from dash import Dash

from gonet_astrometry.portal import ids, system
from gonet_astrometry.portal.logging_utils import global_log_handler


def test_get_log_text_returns_placeholder_and_buffered_text() -> None:
    global_log_handler.clear()
    assert "Portal activity" in system.get_log_text()

    global_log_handler.emit(
        logging.LogRecord(
            "gonet_astrometry.test",
            logging.INFO,
            __file__,
            1,
            "hello",
            (),
            None,
        )
    )
    assert "hello" in system.get_log_text()
    global_log_handler.clear()


def test_close_desktop_window_handles_missing_webview(monkeypatch) -> None:
    monkeypatch.delitem(sys.modules, "webview", raising=False)
    real_import = __import__

    def fail_webview_import(
        name: str,
        globals: object = None,
        locals: object = None,
        fromlist: object = (),
        level: int = 0,
    ) -> object:
        if name == "webview":
            raise ImportError("missing")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr("builtins.__import__", fail_webview_import)

    assert system.close_desktop_window() is False


def test_close_desktop_window_handles_server_only_mode(monkeypatch) -> None:
    fake_webview = ModuleType("webview")
    fake_webview.windows = []  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "webview", fake_webview)

    assert system.close_desktop_window() is False


def test_close_desktop_window_sends_javascript_request(monkeypatch) -> None:
    scripts: list[str] = []

    class FakeWindow:
        def evaluate_js(self, script: str) -> None:
            scripts.append(script)

    fake_webview = ModuleType("webview")
    fake_webview.windows = [FakeWindow()]  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "webview", fake_webview)

    assert system.close_desktop_window() is True
    assert scripts == ["window.pywebview.api.close_window()"]


def test_close_desktop_window_reports_webview_failure(monkeypatch) -> None:
    class FakeWindow:
        def evaluate_js(self, _script: str) -> None:
            raise RuntimeError("closed")

    fake_webview = ModuleType("webview")
    fake_webview.windows = [FakeWindow()]  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "webview", fake_webview)

    assert system.close_desktop_window() is False


def test_registered_system_callbacks_delegate_to_helpers(monkeypatch) -> None:
    app = Dash(__name__)
    system.register_system_callbacks(app)

    monkeypatch.setattr(system, "get_log_text", lambda: "current log")
    monkeypatch.setattr(system, "close_desktop_window", lambda: True)

    log_callback = app.callback_map[f"{ids.LOG_WINDOW}.children"]["callback"]
    exit_callback = app.callback_map[f"{ids.BTN_EXIT}.disabled"]["callback"]

    assert log_callback.__wrapped__(1) == "current log"
    assert exit_callback.__wrapped__(1) is True
    assert f"{ids.LOG_AUTOSCROLL_DUMMY}.children" in app.callback_map
