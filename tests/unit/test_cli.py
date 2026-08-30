import sys
from pathlib import Path
from types import ModuleType

import pytest

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
            "--reference-image",
            "night/frame-350.jpg",
            "--output-dir",
            "products",
            "--output",
            "result.pdf",
            "--overwrite-products",
            "--workers",
            "4",
            "--grid-calibration",
            "camera_calibration.npz",
            "--overwrite-solution",
            "--field-edge-threshold-fraction",
            "0.25",
            "--field-edge-keep-margin-px",
            "48",
            "--field-mask-keep-margin-px",
            "24",
            "--no-use-provisional-field-mask",
            "--grid-search-radius-deg",
            "75",
            "--grid-acceptance-radius-deg",
            "70",
            "--multichannel-min-support",
            "2",
            "--spherical-prediction-tolerance-arcmin",
            "6",
            "--stellar-merge-radius-arcmin",
            "12",
            "--sidereal-min-track-points",
            "7",
            "--sidereal-consistent-rms-deg",
            "0.2",
            "--solve-orientation",
            "--catalog-cache",
            "stars.npz",
            "--overwrite-orientation",
            "--orientation-limiting-magnitude",
            "5.8",
            "--orientation-bootstrap-limiting-magnitude",
            "4.6",
            "--orientation-min-matches",
            "12",
            "--orientation-track-validation-rms-deg",
            "0.25",
            "--orientation-max-declination-robust-sigma-deg",
            "0.05",
            "--orientation-max-declination-p95-deg",
            "0.12",
            "--identify-stars",
            "--overwrite-identifications",
            "--identification-limiting-magnitude",
            "4.7",
            "--identification-bootstrap-limiting-magnitude",
            "3.1",
            "--identification-final-radius-px",
            "12",
            "--identification-bright-rescue-radius-px",
            "22",
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
    assert arguments.reference_image == Path("night/frame-350.jpg")
    assert arguments.output == Path("result.pdf")
    assert arguments.output_dir == Path("products")
    assert arguments.overwrite_products is True
    assert arguments.workers == 4
    assert arguments.grid_calibration == Path("camera_calibration.npz")
    assert arguments.tracking_mode == "legacy"
    assert arguments.overwrite_solution is True
    assert arguments.footprint_threshold_fraction == 0.25
    assert arguments.use_provisional_field_mask is False
    assert arguments.field_edge_keep_margin_px == 48.0
    assert arguments.field_mask_keep_margin_px == 24.0
    assert arguments.grid_search_radius_deg == 75.0
    assert arguments.grid_acceptance_radius_deg == 70.0
    assert arguments.multichannel_min_support == 2
    assert arguments.spherical_prediction_tolerance_arcmin == 6.0
    assert arguments.stellar_merge_radius_arcmin == 12.0
    assert arguments.sidereal_min_track_points == 7
    assert arguments.sidereal_consistent_rms_deg == 0.2
    assert arguments.solve_orientation is True
    assert arguments.catalog_cache == Path("stars.npz")
    assert arguments.overwrite_orientation is True
    assert arguments.orientation_limiting_magnitude == 5.8
    assert arguments.orientation_bootstrap_limiting_magnitude == 4.6
    assert arguments.orientation_min_matches == 12
    assert arguments.orientation_track_validation_rms_deg == 0.25
    assert arguments.orientation_max_declination_robust_sigma_deg == 0.05
    assert arguments.orientation_max_declination_p95_deg == 0.12
    assert arguments.identify_stars is True
    assert arguments.overwrite_identifications is True
    assert arguments.identification_limiting_magnitude == 4.7
    assert arguments.identification_bootstrap_limiting_magnitude == 3.1
    assert arguments.identification_final_radius_px == 12.0
    assert arguments.identification_bright_rescue_radius_px == 22.0


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
            temporal_tracking_product_path=None,
            stellar_tracking_product_path=None,
            stellar_identification_product_path=None,
            fallback_tracking_product_path=None,
            reused_temporal_tracking_product=False,
            reused_stellar_tracking_product=False,
            reused_stellar_identification_product=False,
            reused_fallback_tracking_product=False,
            tracking_mode="legacy",
            catalog_track_count=0,
            fallback_track_count=0,
            sidereal_product_path=None,
            reused_sidereal_product=False,
            orientation_product_path=None,
            reused_orientation_product=False,
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
    assert calls[0]["tracking_mode"] == "legacy"
    assert calls[0]["output_dir"] == Path("gonet_astrometry_output")
    assert calls[0]["output_path"] == Path("gonet_astrometry_output/tracking.pdf")
    assert calls[0]["reference_image_path"] is None
    assert calls[0]["overwrite_products"] is False
    assert calls[0]["grid_calibration_path"] is None
    assert calls[0]["field_mask_path"] is None
    assert calls[0]["detection_workers"] == 1
    assert calls[0]["multichannel_config"].field_edge_keep_margin_px == 0.0
    assert calls[0]["multichannel_config"].field_mask_keep_margin_px == 0.0
    assert calls[0]["multichannel_config"].grid_search_radius_deg is None
    assert calls[0]["multichannel_config"].grid_acceptance_radius_deg is None
    assert calls[0]["multichannel_config"].minimum_channel_support == 1
    assert calls[0]["spherical_tracking_config"].prediction_tolerance_arcmin == 5.0
    assert calls[0]["stellar_merge_config"].merge_radius_deg == 10.0 / 60.0
    assert calls[0]["overwrite_solution"] is False
    assert calls[0]["detection_only"] is False
    assert calls[0]["sidereal_config"].min_track_points == 5
    assert calls[0]["solve_orientation"] is False
    assert calls[0]["catalog_cache_path"] is None
    assert calls[0]["overwrite_orientation"] is False
    assert calls[0]["orientation_config"].min_matches == 8
    assert calls[0]["identify_stars"] is False
    assert calls[0]["overwrite_identifications"] is False
    assert calls[0]["stellar_identification_config"].limiting_magnitude == 4.5
    assert (
        calls[0]["stellar_identification_config"].bootstrap_limiting_magnitude
        == 3.2
    )
    assert "300 detections" in capsys.readouterr().out


def test_run_parser_accepts_hybrid_tracking_mode() -> None:
    arguments = build_parser().parse_args(
        [
            "run",
            "frame.jpg",
            "--algorithm",
            "sep",
            "--grid-calibration",
            "camera_calibration.npz",
            "--tracking-mode",
            "hybrid",
        ]
    )

    assert arguments.tracking_mode == "hybrid"


def test_run_parser_accepts_detection_only_grid_mode() -> None:
    arguments = build_parser().parse_args(
        [
            "run",
            "frame.jpg",
            "--algorithm",
            "sep",
            "--grid-calibration",
            "camera_calibration.npz",
            "--detection-only",
        ]
    )

    assert arguments.detection_only is True


def test_run_parser_rejects_nonpositive_workers() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            [
                "run",
                "frame.jpg",
                "--algorithm",
                "sep",
                "--workers",
                "0",
            ]
        )
