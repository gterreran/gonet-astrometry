"""Photutils source-detection backends."""

from __future__ import annotations

from dataclasses import dataclass, field
from inspect import Parameter, signature
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter

from gonet_astrometry.detection.common import (
    coordinate_uncertainty,
    elongation_from_axes,
    empty_catalog,
    finalize_catalog,
    row_scalar,
    scalar,
)
from gonet_astrometry.detection.config import DetectionConfig
from gonet_astrometry.detection.errors import DetectionBackendUnavailableError
from gonet_astrometry.detection.preprocessing import (
    PreparedDetectionImage,
    prepare_bayer_detection_image,
)
from gonet_astrometry.models.detection import Detection, DetectionCatalog
from gonet_astrometry.models.frame import ImageFrame


@dataclass(frozen=True, slots=True)
class DAOStarFinderDetector:
    """Detect point-like sources using Photutils ``DAOStarFinder``."""

    config: DetectionConfig = field(default_factory=DetectionConfig)

    @property
    def name(self) -> str:
        """Return the stable detector identifier."""
        return "dao-star-finder"

    def detect(self, frame_identifier: str, frame: ImageFrame) -> DetectionCatalog:
        """Prepare a native frame and run ``DAOStarFinder``."""
        prepared = prepare_bayer_detection_image(frame, self.config)
        return self.detect_prepared(frame_identifier, prepared)

    def detect_prepared(
        self,
        frame_identifier: str,
        prepared: PreparedDetectionImage,
    ) -> DetectionCatalog:
        """Run ``DAOStarFinder`` on a shared preprocessed image."""
        daofinder = _load_daofinder()
        finder_kwargs: dict[str, object] = {
            "threshold": self.config.threshold_sigma,
            "fwhm": self.config.fwhm_px,
            "exclude_border": True,
        }
        finder_kwargs[_supported_keyword(daofinder, "n_brightest", "brightest")] = (
            self.config.max_sources
        )
        finder = daofinder(**finder_kwargs)
        table = finder(prepared.data, mask=prepared.mask)
        if table is None:
            return empty_catalog(frame_identifier, self.name)

        detections: list[Detection] = []
        for index, row in enumerate(table, start=1):
            x = row_scalar(row, ("xcentroid", "x_centroid"), default=np.nan)
            y = row_scalar(row, ("ycentroid", "y_centroid"), default=np.nan)
            if not np.isfinite(x) or not np.isfinite(y):
                continue
            flux = row_scalar(row, ("flux",), default=0.0)
            peak = row_scalar(
                row,
                ("peak",),
                default=self.config.threshold_sigma,
            )
            roundness = max(
                abs(row_scalar(row, ("roundness1",), default=0.0)),
                abs(row_scalar(row, ("roundness2",), default=0.0)),
            )
            uncertainty = coordinate_uncertainty(self.config.fwhm_px, peak)
            detections.append(
                Detection(
                    identifier=index,
                    x=x,
                    y=y,
                    flux=flux,
                    signal_to_noise=max(peak, 0.0),
                    x_uncertainty=uncertainty,
                    y_uncertainty=uncertainty,
                    elongation=1.0 + roundness,
                )
            )

        return finalize_catalog(
            frame_identifier,
            self.name,
            detections,
            config=self.config,
        )


@dataclass(frozen=True, slots=True)
class PhotutilsSegmentationDetector:
    """Detect and deblend sources using Photutils segmentation."""

    config: DetectionConfig = field(default_factory=DetectionConfig)

    @property
    def name(self) -> str:
        """Return the stable detector identifier."""
        return "photutils-segmentation"

    def detect(self, frame_identifier: str, frame: ImageFrame) -> DetectionCatalog:
        """Prepare a native frame and run Photutils segmentation."""
        prepared = prepare_bayer_detection_image(frame, self.config)
        return self.detect_prepared(frame_identifier, prepared)

    def detect_prepared(
        self,
        frame_identifier: str,
        prepared: PreparedDetectionImage,
    ) -> DetectionCatalog:
        """Run Photutils segmentation on a shared preprocessed image."""
        source_finder, source_catalog = _load_segmentation_tools()
        sigma = self.config.fwhm_px / 2.354820045
        convolved = gaussian_filter(prepared.data, sigma=sigma, mode="nearest")
        finder_kwargs: dict[str, object] = {
            "deblend": self.config.deblend,
            "progress_bar": False,
        }
        finder_kwargs[_supported_keyword(source_finder, "n_pixels", "npixels")] = (
            self.config.min_pixels
        )
        finder = source_finder(**finder_kwargs)
        segment_map = finder(
            convolved,
            self.config.threshold_sigma,
            mask=prepared.mask,
        )
        if segment_map is None:
            return empty_catalog(frame_identifier, self.name)

        catalog = source_catalog(
            prepared.data,
            segment_map,
            convolved_data=convolved,
            mask=prepared.mask,
            progress_bar=False,
        )
        labels = np.atleast_1d(catalog.labels)
        x_values = _catalog_values(catalog, ("xcentroid", "x_centroid"))
        y_values = _catalog_values(catalog, ("ycentroid", "y_centroid"))
        flux_values = _catalog_values(catalog, ("segment_flux",))
        peak_values = _catalog_values(catalog, ("max_value",))
        major_values = _catalog_values(
            catalog,
            ("semimajor_sigma", "semimajor_axis"),
        )
        minor_values = _catalog_values(
            catalog,
            ("semiminor_sigma", "semiminor_axis"),
        )

        detections: list[Detection] = []
        for index in range(len(labels)):
            x = scalar(x_values[index], np.nan)
            y = scalar(y_values[index], np.nan)
            if not np.isfinite(x) or not np.isfinite(y):
                continue
            peak = max(scalar(peak_values[index]), 0.0)
            uncertainty = coordinate_uncertainty(self.config.fwhm_px, peak)
            detections.append(
                Detection(
                    identifier=int(labels[index]),
                    x=x,
                    y=y,
                    flux=scalar(flux_values[index]),
                    signal_to_noise=peak,
                    x_uncertainty=uncertainty,
                    y_uncertainty=uncertainty,
                    elongation=elongation_from_axes(
                        scalar(major_values[index], np.nan),
                        scalar(minor_values[index], np.nan),
                    ),
                )
            )

        return finalize_catalog(
            frame_identifier,
            self.name,
            detections,
            config=self.config,
        )


def _supported_keyword(factory: Any, modern: str, legacy: str) -> str:
    """Return the keyword supported by an installed Photutils callable.

    Photutils 3 renamed several constructor keywords while retaining the old
    names temporarily as deprecated aliases.  Inspecting the callable keeps
    this backend compatible with both pre-3.0 and 3.x releases without
    relying on package-version string parsing.
    """
    try:
        parameters = signature(factory).parameters
    except (TypeError, ValueError):
        return modern

    if modern in parameters:
        return modern
    if legacy in parameters:
        return legacy
    if any(
        parameter.kind is Parameter.VAR_KEYWORD for parameter in parameters.values()
    ):
        return modern

    name = getattr(factory, "__name__", type(factory).__name__)
    raise DetectionBackendUnavailableError(
        f"Unsupported Photutils API for {name}: expected keyword "
        f"'{modern}' or '{legacy}'."
    )


def _load_daofinder() -> Any:
    """Import ``DAOStarFinder`` or raise a focused dependency error."""
    try:
        from photutils.detection import DAOStarFinder
    except ImportError as exc:
        raise DetectionBackendUnavailableError(
            "DAOStarFinder requires the optional Photutils dependency."
        ) from exc
    return DAOStarFinder


def _load_segmentation_tools() -> tuple[Any, Any]:
    """Import Photutils segmentation classes lazily."""
    try:
        from photutils.segmentation import SourceCatalog, SourceFinder
    except ImportError as exc:
        raise DetectionBackendUnavailableError(
            "Photutils segmentation requires the optional Photutils dependency."
        ) from exc
    return SourceFinder, SourceCatalog


def _catalog_values(
    catalog: object,
    names: tuple[str, ...],
) -> NDArray[Any]:
    """Return the first available vector property from a source catalog."""
    for name in names:
        if hasattr(catalog, name):
            return np.atleast_1d(getattr(catalog, name))
    raise ValueError(
        "Photutils SourceCatalog does not expose any of: " + ", ".join(names)
    )
