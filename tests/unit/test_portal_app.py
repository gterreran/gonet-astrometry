import sys
from pathlib import Path
from types import ModuleType
from urllib.error import URLError

import pytest

pytest.importorskip("dash")

from gonet_astrometry.portal import app as portal_app


def test_run_portal_server_starts_dash_without_reloader(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    class FakeApp:
        def run(self, **kwargs: object) -> None:
            calls.append(kwargs)

    monkeypatch.setattr(portal_app, "create_app", lambda **_kwargs: FakeApp())

    portal_app.run_portal_server(host="0.0.0.0", port=9000, debug=True)

    assert calls == [
        {
            "host": "0.0.0.0",
            "port": 9000,
            "debug": True,
            "use_reloader": False,
        }
    ]


def test_run_portal_opens_independent_webview_window(monkeypatch) -> None:
    calls: list[tuple[str, object]] = []

    class FakeWebview:
        def create_window(self, title: str, url: str, **kwargs: object) -> None:
            calls.append(("window", (title, url, kwargs)))

        def start(self, **kwargs: object) -> None:
            calls.append(("start", kwargs))

    class FakeAPI:
        pass

    monkeypatch.setattr(
        portal_app,
        "_load_webview_components",
        lambda: (FakeWebview(), FakeAPI),
    )
    monkeypatch.setattr(
        portal_app,
        "_start_server_thread",
        lambda **kwargs: calls.append(("thread", kwargs)),
    )
    monkeypatch.setattr(
        portal_app,
        "_wait_for_server",
        lambda url: calls.append(("wait", url)),
    )

    portal_app.run_portal(
        initial_path=Path("frame.jpg"),
        host="0.0.0.0",
        port=9000,
        debug=True,
    )

    assert calls[0] == (
        "thread",
        {
            "initial_path": Path("frame.jpg"),
            "host": "0.0.0.0",
            "port": 9000,
            "debug": True,
        },
    )
    assert calls[1] == ("wait", "http://127.0.0.1:9000")
    _, window = calls[2]
    title, url, kwargs = window
    assert title == "GONet Astrometry Calibrator"
    assert url == "http://127.0.0.1:9000"
    assert isinstance(kwargs["js_api"], FakeAPI)
    assert calls[3] == ("start", {"debug": True})


def test_load_webview_components_reports_missing_dependency(monkeypatch) -> None:
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

    with pytest.raises(RuntimeError, match="desktop portal requires"):
        portal_app._load_webview_components()


def test_load_webview_components_returns_modules(monkeypatch) -> None:
    fake_webview = ModuleType("webview")
    wizard_package = ModuleType("GONet_Wizard")
    wizard_ui = ModuleType("GONet_Wizard.ui")
    fake_api_module = ModuleType("GONet_Wizard.ui.api")

    class FakeAPI:
        pass

    wizard_package.__path__ = []  # type: ignore[attr-defined]
    wizard_ui.__path__ = []  # type: ignore[attr-defined]
    fake_api_module.WebviewAPI = FakeAPI  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "webview", fake_webview)
    monkeypatch.setitem(sys.modules, "GONet_Wizard", wizard_package)
    monkeypatch.setitem(sys.modules, "GONet_Wizard.ui", wizard_ui)
    monkeypatch.setitem(sys.modules, "GONet_Wizard.ui.api", fake_api_module)

    webview, api_factory = portal_app._load_webview_components()

    assert webview is fake_webview
    assert api_factory is FakeAPI


def test_start_server_thread_uses_daemon_thread(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    class FakeThread:
        def __init__(self, **kwargs: object) -> None:
            calls.append(kwargs)

        def start(self) -> None:
            calls.append({"started": True})

    monkeypatch.setattr(portal_app.threading, "Thread", FakeThread)

    thread = portal_app._start_server_thread(
        initial_path=Path("frame.jpg"),
        host="127.0.0.1",
        port=8050,
        debug=False,
    )

    assert isinstance(thread, FakeThread)
    assert calls[0]["target"] is portal_app.run_portal_server
    assert calls[0]["daemon"] is True
    assert calls[0]["name"] == "gonet-astrometry-dash"
    assert calls[1] == {"started": True}


def test_local_window_url_rewrites_wildcard_hosts() -> None:
    assert portal_app._local_window_url("0.0.0.0", 8050) == "http://127.0.0.1:8050"
    assert portal_app._local_window_url("::", 8050) == "http://127.0.0.1:8050"
    assert portal_app._local_window_url("localhost", 9000) == "http://localhost:9000"


def test_wait_for_server_returns_after_success(monkeypatch) -> None:
    calls: list[object] = []

    class Response:
        def __enter__(self) -> "Response":
            calls.append("enter")
            return self

        def __exit__(self, *_args: object) -> None:
            calls.append("exit")

    monkeypatch.setattr(portal_app, "urlopen", lambda *_args, **_kwargs: Response())

    portal_app._wait_for_server("http://127.0.0.1:8050", timeout=0.1)

    assert calls == ["enter", "exit"]


def test_wait_for_server_times_out(monkeypatch) -> None:
    ticks = iter([0.0, 0.0, 0.2])
    monkeypatch.setattr(portal_app.time, "monotonic", lambda: next(ticks))
    monkeypatch.setattr(portal_app.time, "sleep", lambda _seconds: None)

    def unavailable(*_args: object, **_kwargs: object) -> None:
        raise URLError("not ready")

    monkeypatch.setattr(portal_app, "urlopen", unavailable)

    with pytest.raises(RuntimeError, match="did not become available"):
        portal_app._wait_for_server("http://127.0.0.1:8050", timeout=0.1)


def test_create_app_discovers_initial_input_without_loading(tmp_path) -> None:
    image = tmp_path / "frame.jpg"
    image.touch()
    calls: list[Path] = []

    class FakeRawFile:
        filename = "frame.jpg"
        is_bayer_planes = False

        def get_channel(self, _channel_name: str):
            raise AssertionError("image should not load during app construction")

    def loader(path: Path) -> FakeRawFile:
        calls.append(path)
        return FakeRawFile()

    app = portal_app.create_app(initial_path=image, raw_loader=loader)
    session = app.server.config["astrometry_session"]

    assert session.files == (image.resolve(),)
    assert session.loaded_path is None
    assert calls == []
