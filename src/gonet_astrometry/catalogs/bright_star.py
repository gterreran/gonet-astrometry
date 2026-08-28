"""Portable cache and VizieR loader for the Yale Bright Star Catalogue."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from gonet_astrometry.catalogs.base import CatalogStar

_CATALOG_ID = "V/50/catalog"
_FORMAT = "gonet-astrometry-bright-star-catalog"
_VERSION = 1


@dataclass(frozen=True, slots=True)
class BrightStarCatalog:
    """In-memory bright-star catalog suitable for orientation matching.

    Parameters
    ----------
    stars
        Catalog records, normally loaded from the portable local cache.
    source
        Human-readable catalog provenance identifier.
    """

    stars: tuple[CatalogStar, ...]
    source: str = _CATALOG_ID

    def query_bright_stars(
        self,
        epoch: datetime,
        limiting_magnitude: float,
    ) -> tuple[CatalogStar, ...]:
        """Return stars not fainter than ``limiting_magnitude``.

        The ``epoch`` argument is accepted to satisfy the common catalog
        protocol. Proper-motion propagation is deliberately deferred; the
        first orientation solver uses the catalog ICRS directions as stored.
        """
        del epoch
        return tuple(
            star
            for star in self.stars
            if star.magnitude is not None and star.magnitude <= limiting_magnitude
        )


def load_or_fetch_bright_star_catalog(path: Path) -> BrightStarCatalog:
    """Load the local bright-star cache, querying VizieR only when absent."""
    destination = Path(path).expanduser().resolve()
    if destination.exists():
        return load_bright_star_catalog(destination)
    catalog = _fetch_vizier_bright_star_catalog()
    save_bright_star_catalog(destination, catalog)
    return catalog


def save_bright_star_catalog(path: Path, catalog: BrightStarCatalog) -> Path:
    """Write a pickle-free portable bright-star cache."""
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, NDArray[np.generic]] = {
        "format": _scalar_string(_FORMAT),
        "version": np.asarray(_VERSION, dtype=np.int64),
        "source": _scalar_string(catalog.source),
        "identifier": _strings(star.identifier for star in catalog.stars),
        "ra_deg": np.asarray(
            [star.right_ascension_deg for star in catalog.stars], dtype=np.float64
        ),
        "dec_deg": np.asarray(
            [star.declination_deg for star in catalog.stars], dtype=np.float64
        ),
        "magnitude": np.asarray(
            [
                np.nan if star.magnitude is None else float(star.magnitude)
                for star in catalog.stars
            ],
            dtype=np.float64,
        ),
    }
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(destination)
    return destination


def load_bright_star_catalog(path: Path) -> BrightStarCatalog:
    """Load and validate a portable bright-star cache without pickle."""
    source_path = Path(path).expanduser().resolve()
    try:
        with np.load(source_path, allow_pickle=False) as data:
            if _read_scalar_string(data, "format") != _FORMAT:
                raise ValueError("File is not a GONet Astrometry bright-star cache")
            if int(np.asarray(data["version"]).item()) != _VERSION:
                raise ValueError("Unsupported bright-star cache version")
            identifiers = _read_strings(data, "identifier")
            ra = np.asarray(data["ra_deg"], dtype=np.float64)
            dec = np.asarray(data["dec_deg"], dtype=np.float64)
            magnitude = np.asarray(data["magnitude"], dtype=np.float64)
            count = len(identifiers)
            if any(len(array) != count for array in (ra, dec, magnitude)):
                raise ValueError("Bright-star cache arrays have inconsistent lengths")
            stars = tuple(
                CatalogStar(
                    identifier=identifiers[index],
                    right_ascension_deg=float(ra[index]),
                    declination_deg=float(dec[index]),
                    magnitude=(
                        None if np.isnan(magnitude[index]) else float(magnitude[index])
                    ),
                )
                for index in range(count)
            )
            source = _read_scalar_string(data, "source")
    except (KeyError, OSError, TypeError, ValueError) as exc:
        raise ValueError(
            f"Could not read bright-star catalog {source_path}: {exc}"
        ) from exc
    return BrightStarCatalog(stars=stars, source=source)


def _fetch_vizier_bright_star_catalog() -> BrightStarCatalog:
    """Query the public VizieR Bright Star Catalogue and return plain records."""
    try:
        from astropy import units as u  # type: ignore
        from astropy.coordinates import SkyCoord  # type: ignore
        from astroquery.vizier import Vizier  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on optional install
        raise RuntimeError(
            "Fetching the Bright Star Catalogue requires the optional "
            "'astroquery' dependency. Install gonet-astrometry[catalog] or "
            "provide an existing bright_star_catalog.npz cache."
        ) from exc

    query = Vizier(
        columns=["HR", "RAJ2000", "DEJ2000", "Vmag"],
        row_limit=-1,
    )
    tables = query.get_catalogs(_CATALOG_ID)
    if not tables:
        raise RuntimeError("VizieR returned no Bright Star Catalogue table")
    table = tables[0]
    rows = _usable_vizier_rows(table)
    if not rows:
        raise RuntimeError(
            "VizieR Bright Star Catalogue contains no rows with usable coordinates"
        )
    coordinates = SkyCoord(
        [row[1] for row in rows],
        [row[2] for row in rows],
        unit=(u.hourangle, u.deg),
        frame="icrs",
    )
    stars = tuple(
        CatalogStar(
            identifier=row[0],
            right_ascension_deg=float(coordinates.ra.deg[index]),
            declination_deg=float(coordinates.dec.deg[index]),
            magnitude=row[3],
        )
        for index, row in enumerate(rows)
    )
    return BrightStarCatalog(stars=stars, source=_CATALOG_ID)


def _usable_vizier_rows(
    table: Iterable[Any],
) -> tuple[tuple[str, str, str, float | None], ...]:
    """Return catalog rows that contain parseable identifiers and coordinates.

    VizieR represents missing table values with masked entries. Depending on the
    table/column formatting, converting those values directly to strings can
    also yield an empty string or ``"--"``. SkyCoord correctly rejects those
    placeholders, so they must be removed before constructing the vectorized
    coordinate object.
    """
    rows: list[tuple[str, str, str, float | None]] = []
    for row in table:
        identifier = _optional_int(row["HR"])
        right_ascension = _optional_text(row["RAJ2000"])
        declination = _optional_text(row["DEJ2000"])
        if identifier is None or right_ascension is None or declination is None:
            continue
        rows.append(
            (
                f"HR {identifier}",
                right_ascension,
                declination,
                _optional_finite_float(row["Vmag"]),
            )
        )
    return tuple(rows)


def _optional_text(value: object) -> str | None:
    """Return stripped text for an unmasked catalog value, otherwise ``None``."""
    if np.ma.is_masked(value):
        return None
    text = str(value).strip()
    if not text or text.casefold() in {"--", "nan", "none"}:
        return None
    return text


def _optional_int(value: object) -> int | None:
    """Return an integer catalog value when present and parseable."""
    if np.ma.is_masked(value):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_finite_float(value: object) -> float | None:
    """Return a finite floating-point catalog value when available."""
    if np.ma.is_masked(value):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if np.isfinite(parsed) else None


def _strings(values: Iterable[object]) -> NDArray[np.str_]:
    strings = tuple(str(item) for item in values)
    width = max((len(item) for item in strings), default=1)
    return np.asarray(strings, dtype=f"<U{width}")


def _scalar_string(value: str) -> NDArray[np.str_]:
    return np.asarray(value, dtype=f"<U{max(1, len(value))}")


def _read_strings(data: np.lib.npyio.NpzFile, name: str) -> tuple[str, ...]:
    array = np.asarray(data[name])
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    return tuple(str(item) for item in array.tolist())


def _read_scalar_string(data: np.lib.npyio.NpzFile, name: str) -> str:
    array = np.asarray(data[name])
    if array.ndim != 0:
        raise ValueError(f"{name} must be scalar")
    return str(array.item())
