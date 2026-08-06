from pathlib import Path

from gonet_astrometry.portal.discovery import (
    discover_gonet_files,
    parse_source_paths,
)


def test_parse_source_paths_accepts_lines_and_semicolons() -> None:
    assert parse_source_paths(" first.jpg\nsecond folder ; third.jpg ") == (
        Path("first.jpg"),
        Path("second folder"),
        Path("third.jpg"),
    )
    assert parse_source_paths(None) == ()


def test_discovery_finds_files_without_parsing_payloads(tmp_path: Path) -> None:
    direct = tmp_path / "direct.jpg"
    nested_dir = tmp_path / "night" / "camera"
    nested_dir.mkdir(parents=True)
    nested = nested_dir / "nested.JPG"
    hidden = nested_dir / "._hidden.jpg"
    unsupported = tmp_path / "calibration.tiff"
    for path in (direct, nested, hidden, unsupported):
        path.touch()

    result = discover_gonet_files(
        (direct, tmp_path / "night", unsupported, tmp_path / "missing"),
        recursive=True,
    )

    assert result.files == tuple(sorted((direct.resolve(), nested.resolve())))
    assert result.directories == ((tmp_path / "night").resolve(),)
    assert result.missing == (tmp_path / "missing",)
    assert result.unsupported == (unsupported.resolve(),)
    assert "2 candidate GONet files" in result.summary
    assert "1 missing path" in result.summary


def test_discovery_can_limit_folder_search_to_top_level(tmp_path: Path) -> None:
    top_level = tmp_path / "top.jpg"
    nested_dir = tmp_path / "nested"
    nested_dir.mkdir()
    nested = nested_dir / "nested.jpg"
    top_level.touch()
    nested.touch()

    result = discover_gonet_files((tmp_path,), recursive=False)

    assert result.files == (top_level.resolve(),)
