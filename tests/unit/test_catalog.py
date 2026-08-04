import pytest

from gonet_astrometry.catalogs.base import CatalogStar


def test_catalog_star_holds_coordinate() -> None:
    star = CatalogStar("star-1", 10.0, 20.0, 2.5)
    assert star.identifier == "star-1"
    assert star.magnitude == 2.5


@pytest.mark.parametrize(
    ("right_ascension", "declination", "message"),
    [
        (-1.0, 0.0, "right_ascension"),
        (360.0, 0.0, "right_ascension"),
        (0.0, 91.0, "declination"),
    ],
)
def test_catalog_star_rejects_invalid_coordinates(
    right_ascension: float,
    declination: float,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        CatalogStar("bad", right_ascension, declination)
