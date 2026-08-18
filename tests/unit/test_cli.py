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


def test_run_parser_exposes_detection_and_tracking_settings() -> None:
    arguments = build_parser().parse_args(
        [
            "run",
            "night",
            "frame.jpg",
            "--algorithm",
            "sep",
            "--threshold-sigma",
            "4.2",
            "--no-deblend",
            "--max-sources",
            "none",
            "--bright-mask-sigma",
            "none",
            "--max-motion-px-min",
            "12",
            "--max-gap-minutes",
            "12",
            "--max-missing",
            "2",
            "--max-prediction-gap-scale",
            "2.5",
            "--min-track-length",
            "4",
            "--channel",
            "red",
            "--output-dir",
            "products",
            "--output",
            "result.pdf",
            "--overwrite-products",
        ]
    )

    assert arguments.command == "run"
    assert arguments.inputs == [Path("night"), Path("frame.jpg")]
    assert arguments.algorithm == "sep"
    assert arguments.threshold_sigma == 4.2
    assert arguments.deblend is False
    assert arguments.max_sources is None
    assert arguments.bright_mask_sigma is None
    assert arguments.max_speed_px_per_minute == 12.0
    assert arguments.max_gap_minutes == 12.0
    assert arguments.max_gap_frames == 2
    assert arguments.max_prediction_gap_scale == 2.5
    assert arguments.min_track_length == 4
    assert arguments.channel == "red"
    assert arguments.output == Path("result.pdf")
    assert arguments.output_dir == Path("products")
    assert arguments.overwrite_products is True


def test_main_runs_batch_workflow(monkeypatch, capsys) -> None:
    from types import SimpleNamespace

    calls: list[dict[str, object]] = []
    module = ModuleType("gonet_astrometry.runner")

    def fake_run_cli_workflow(*args: object, **kwargs: object) -> object:
        calls.append({"inputs": args[0], **kwargs})
        return SimpleNamespace(
            output_path=Path("report.pdf"),
            file_count=3,
            detection_count=300,
            track_count=42,
            assigned_detection_count=150,
            unassigned_detection_count=150,
            skipped_file_count=2,
            reused_detection_product=False,
            reused_tracking_product=False,
        )

    module.run_cli_workflow = fake_run_cli_workflow  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "gonet_astrometry.runner", module)

    result = main(
        [
            "run",
            "night",
            "--algorithm",
            "sep",
            "--threshold-sigma",
            "6",
            "--prediction-tolerance-px",
            "8",
        ]
    )

    assert result == 0
    assert calls[0]["inputs"] == [Path("night")]
    assert calls[0]["algorithm"] == "sep"
    assert calls[0]["detection_config"].threshold_sigma == 6.0
    assert calls[0]["tracking_config"].prediction_tolerance_px == 8.0
    assert calls[0]["output_dir"] == Path("gonet_astrometry_output")
    assert calls[0]["output_path"] == Path("gonet_astrometry_output/tracking.pdf")
    assert calls[0]["overwrite_products"] is False
    assert "300 detections" in capsys.readouterr().out
