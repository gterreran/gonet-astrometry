from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from gonet_astrometry.catalogs import bright_star
from gonet_astrometry.catalogs.base import CatalogStar
from gonet_astrometry.catalogs.bright_star import (
    BrightStarCatalog,
    load_bright_star_catalog,
    load_or_fetch_bright_star_catalog,
    save_bright_star_catalog,
)


def test_bright_star_catalog_cache_round_trip(tmp_path: Path) -> None:
    catalog = BrightStarCatalog(
        (
            CatalogStar("HR 1", 10.0, 20.0, 2.0),
            CatalogStar("HR 2", 30.0, -5.0, 6.8),
            CatalogStar("HR 3", 40.0, 10.0, None),
        )
    )
    path = save_bright_star_catalog(tmp_path / "catalog.npz", catalog)
    with np.load(path, allow_pickle=False) as data:
        assert str(data["format"].item()) == "gonet-astrometry-bright-star-catalog"
    loaded = load_bright_star_catalog(path)
    assert loaded == catalog
    selected = loaded.query_bright_stars(
        datetime(2026, 8, 18, tzinfo=timezone.utc), 6.5
    )
    assert [star.identifier for star in selected] == ["HR 1"]


def test_load_or_fetch_reuses_existing_cache(tmp_path: Path, monkeypatch) -> None:
    path = save_bright_star_catalog(
        tmp_path / "catalog.npz",
        BrightStarCatalog((CatalogStar("HR 1", 10.0, 20.0, 2.0),)),
    )
    monkeypatch.setattr(
        "gonet_astrometry.catalogs.bright_star._fetch_vizier_bright_star_catalog",
        lambda: (_ for _ in ()).throw(AssertionError("should not fetch")),
    )
    assert load_or_fetch_bright_star_catalog(path).stars[0].identifier == "HR 1"


def test_vizier_rows_skip_missing_or_invalid_coordinates() -> None:
    rows = [
        {
            "HR": 1,
            "RAJ2000": "00 05 09.9",
            "DEJ2000": "+45 13 45",
            "Vmag": 2.1,
        },
        {
            "HR": 2,
            "RAJ2000": "",
            "DEJ2000": "+10 00 00",
            "Vmag": 3.0,
        },
        {
            "HR": 3,
            "RAJ2000": np.ma.masked,
            "DEJ2000": "+20 00 00",
            "Vmag": 4.0,
        },
        {
            "HR": 4,
            "RAJ2000": "12 00 00",
            "DEJ2000": "--",
            "Vmag": 5.0,
        },
        {
            "HR": 5,
            "RAJ2000": "23 59 59",
            "DEJ2000": "-30 00 00",
            "Vmag": np.ma.masked,
        },
    ]

    usable = bright_star._usable_vizier_rows(rows)

    assert usable == (
        ("HR 1", "00 05 09.9", "+45 13 45", 2.1),
        ("HR 5", "23 59 59", "-30 00 00", None),
    )
