"""Star-catalog interfaces and implementations."""

from gonet_astrometry.catalogs.base import CatalogStar, StarCatalog
from gonet_astrometry.catalogs.bright_star import (
    BrightStarCatalog,
    load_bright_star_catalog,
    load_or_fetch_bright_star_catalog,
    save_bright_star_catalog,
)

__all__ = [
    "BrightStarCatalog",
    "CatalogStar",
    "StarCatalog",
    "load_bright_star_catalog",
    "load_or_fetch_bright_star_catalog",
    "save_bright_star_catalog",
]
