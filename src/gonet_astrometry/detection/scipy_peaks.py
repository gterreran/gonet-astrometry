"""Built-in SciPy local-maximum source detector."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import gaussian_filter, label, maximum_filter

from gonet_astrometry.detection.common import (
    coordinate_uncertainty,
    finalize_catalog,
    local_source_measurements,
)
from gonet_astrometry.detection.config import DetectionConfig
from gonet_astrometry.detection.diagnostics import enrich_detection_catalog
from gonet_astrometry.detection.preprocessing import (
    PreparedDetectionImage,
    prepare_bayer_detection_image,
)
from gonet_astrometry.models.detection import Detection, DetectionCatalog
from gonet_astrometry.models.frame import ImageFrame


@dataclass(frozen=True, slots=True)
class ScipyPeakDetector:
    """Detect local maxima in a smoothed Bayer-normalized image.

    This backend is intentionally simple and dependency-free beyond SciPy. It
    provides a deterministic baseline against which the astronomy-specific
    Photutils and SEP backends can be compared.
    """

    config: DetectionConfig = field(default_factory=DetectionConfig)

    @property
    def name(self) -> str:
        """Return the stable detector identifier."""
        return "scipy-local-max"

    def detect(self, frame_identifier: str, frame: ImageFrame) -> DetectionCatalog:
        """Prepare a native frame, detect local maxima, and add diagnostics."""
        prepared = prepare_bayer_detection_image(frame, self.config)
        catalog = self.detect_prepared(frame_identifier, prepared)
        return enrich_detection_catalog(catalog, prepared, self.config)

    def detect_prepared(
        self,
        frame_identifier: str,
        prepared: PreparedDetectionImage,
    ) -> DetectionCatalog:
        """Detect local maxima in a shared preprocessed image."""
        sigma = self.config.fwhm_px / 2.354820045
        smoothed = gaussian_filter(prepared.data, sigma=sigma, mode="nearest")

        separation = max(1, int(round(self.config.min_separation_px)))
        filter_size = 2 * separation + 1
        local_maximum = maximum_filter(
            smoothed,
            size=filter_size,
            mode="nearest",
        )
        candidates = (
            (smoothed >= self.config.threshold_sigma)
            & (smoothed == local_maximum)
            & ~prepared.mask
        )

        border = max(1, int(np.ceil(self.config.fwhm_px)))
        candidates[:border, :] = False
        candidates[-border:, :] = False
        candidates[:, :border] = False
        candidates[:, -border:] = False

        labels, count = label(candidates)
        radius = max(2, int(np.ceil(self.config.fwhm_px)))
        detections: list[Detection] = []
        for component in range(1, count + 1):
            yy, xx = np.nonzero(labels == component)
            if yy.size == 0:
                continue
            peaks = smoothed[yy, xx]
            peak_index = int(np.argmax(peaks))
            y_peak = int(yy[peak_index])
            x_peak = int(xx[peak_index])
            x, y, flux, elongation = local_source_measurements(
                prepared.data,
                x_peak,
                y_peak,
                radius=radius,
            )
            signal_to_noise = float(smoothed[y_peak, x_peak])
            uncertainty = coordinate_uncertainty(
                self.config.fwhm_px,
                signal_to_noise,
            )
            detections.append(
                Detection(
                    identifier=component,
                    x=x,
                    y=y,
                    flux=flux,
                    signal_to_noise=signal_to_noise,
                    x_uncertainty=uncertainty,
                    y_uncertainty=uncertainty,
                    elongation=elongation,
                )
            )

        return finalize_catalog(
            frame_identifier,
            self.name,
            detections,
            config=self.config,
        )
