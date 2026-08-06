import sys
from pathlib import Path
from types import ModuleType

from gonet_astrometry.cli import build_parser, main


def test_parser_has_expected_program_name() -> None:
    assert build_parser().prog == "gonet-astrometry"


def test_main_accepts_no_arguments() -> None:
    assert main([]) == 0


def test_portal_parser_defaults() -> None:
    arguments = build_parser().parse_args(["portal"])

    assert arguments.command == "portal"
    assert arguments.image is None
    assert arguments.host == "127.0.0.1"
    assert arguments.port == 8050
    assert arguments.debug is False
    assert arguments.server_only is False


def test_portal_parser_accepts_input_alias() -> None:
    arguments = build_parser().parse_args(["portal", "--input", "night-data"])

    assert arguments.image == Path("night-data")


def test_main_launches_desktop_portal(monkeypatch) -> None:
    calls: list[tuple[str, dict[str, object]]] = []
    module = ModuleType("gonet_astrometry.portal.app")

    def fake_run_portal(**kwargs: object) -> None:
        calls.append(("desktop", kwargs))

    def fake_run_portal_server(**kwargs: object) -> None:
        calls.append(("server", kwargs))

    module.run_portal = fake_run_portal  # type: ignore[attr-defined]
    module.run_portal_server = fake_run_portal_server  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "gonet_astrometry.portal.app", module)

    result = main(
        [
            "portal",
            "--image",
            "frame.jpg",
            "--host",
            "0.0.0.0",
            "--port",
            "9000",
            "--debug",
        ]
    )

    assert result == 0
    assert calls == [
        (
            "desktop",
            {
                "initial_path": Path("frame.jpg"),
                "host": "0.0.0.0",
                "port": 9000,
                "debug": True,
            },
        )
    ]


def test_main_runs_server_only(monkeypatch) -> None:
    calls: list[tuple[str, dict[str, object]]] = []
    module = ModuleType("gonet_astrometry.portal.app")

    def fake_run_portal(**kwargs: object) -> None:
        calls.append(("desktop", kwargs))

    def fake_run_portal_server(**kwargs: object) -> None:
        calls.append(("server", kwargs))

    module.run_portal = fake_run_portal  # type: ignore[attr-defined]
    module.run_portal_server = fake_run_portal_server  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "gonet_astrometry.portal.app", module)

    result = main(["portal", "--server-only"])

    assert result == 0
    assert calls == [
        (
            "server",
            {
                "initial_path": None,
                "host": "127.0.0.1",
                "port": 8050,
                "debug": False,
            },
        )
    ]
