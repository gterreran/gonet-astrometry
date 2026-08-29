"""Configuration shared by source-detection backends."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DetectionConfig:
    """Common source-detection and preprocessing settings.

    Parameters
    ----------
    threshold_sigma
        Detection threshold in units of the locally estimated background noise.
    fwhm_px
        Approximate stellar full width at half maximum in native sensor pixels.
    min_separation_px
        Minimum separation between local maxima in native sensor pixels.
    min_pixels
        Minimum connected-pixel area used by segmentation backends.
    max_sources
        Maximum number of detections retained, ordered by backend brightness.
        ``None`` retains every detection.
    deblend
        Whether segmentation backends should split overlapping sources.
    background_box_size_px
        Width and height of the local-background boxes in full-resolution
        sensor pixels. Each Bayer parity therefore uses boxes half this size.
    background_min_valid_fraction
        Minimum fraction of unmasked pixels required for a background box.
    background_sigma_clip
        Symmetric sigma-clipping limit used within each background box.
    background_clip_iterations
        Maximum number of sigma-clipping iterations per background box.
    use_provisional_field_mask
        Whether to infer and exclude the illuminated fisheye footprint before
        estimating the background.
    footprint_threshold_fraction
        Fraction of the center-to-corner intensity contrast used to separate
        the illuminated footprint from the dark exterior.
    footprint_smoothing_px
        Gaussian smoothing scale in full-resolution pixels used only while
        estimating the footprint.
    footprint_erosion_px
        Inward erosion applied to the estimated footprint so detector kernels
        do not straddle the uncertain optical rim.
    bright_mask_sigma
        Local-significance threshold used to seed masks for large bright
        contaminants. ``None`` disables dynamic bright-region masking.
    bright_mask_min_pixels
        Minimum connected area above ``bright_mask_sigma`` required before a
        bright region is masked. This prevents isolated stars from being
        removed merely because they are bright.
    bright_mask_dilation_px
        Radius by which accepted bright-region masks are expanded.

    Raises
    ------
    ValueError
        If a numeric setting lies outside its valid range.
    """

    threshold_sigma: float = 3.5
    fwhm_px: float = 3.0
    min_separation_px: float = 5.0
    min_pixels: int = 5
    max_sources: int | None = 2000
    deblend: bool = True
    background_box_size_px: int = 256
    background_min_valid_fraction: float = 0.2
    background_sigma_clip: float = 3.0
    background_clip_iterations: int = 4
    use_provisional_field_mask: bool = True
    footprint_threshold_fraction: float = 0.2
    footprint_smoothing_px: float = 24.0
    footprint_erosion_px: float = 12.0
    bright_mask_sigma: float | None = 40.0
    bright_mask_min_pixels: int = 64
    bright_mask_dilation_px: float = 18.0

    def __post_init__(self) -> None:
        if self.threshold_sigma <= 0:
            raise ValueError("threshold_sigma must be strictly positive")
        if self.fwhm_px <= 0:
            raise ValueError("fwhm_px must be strictly positive")
        if self.min_separation_px < 0:
            raise ValueError("min_separation_px cannot be negative")
        if self.min_pixels <= 0:
            raise ValueError("min_pixels must be strictly positive")
        if self.max_sources is not None and self.max_sources <= 0:
            raise ValueError("max_sources must be positive or None")
        if self.background_box_size_px < 8:
            raise ValueError("background_box_size_px must be at least 8")
        if not 0 < self.background_min_valid_fraction <= 1:
            raise ValueError(
                "background_min_valid_fraction must lie in the interval (0, 1]"
            )
        if self.background_sigma_clip <= 0:
            raise ValueError("background_sigma_clip must be strictly positive")
        if self.background_clip_iterations <= 0:
            raise ValueError("background_clip_iterations must be strictly positive")
        if not 0 < self.footprint_threshold_fraction < 1:
            raise ValueError(
                "footprint_threshold_fraction must lie in the interval (0, 1)"
            )
        if self.footprint_smoothing_px < 0:
            raise ValueError("footprint_smoothing_px cannot be negative")
        if self.footprint_erosion_px < 0:
            raise ValueError("footprint_erosion_px cannot be negative")
        if self.bright_mask_sigma is not None and self.bright_mask_sigma <= 0:
            raise ValueError("bright_mask_sigma must be positive or None")
        if self.bright_mask_min_pixels <= 0:
            raise ValueError("bright_mask_min_pixels must be strictly positive")
        if self.bright_mask_dilation_px < 0:
            raise ValueError("bright_mask_dilation_px cannot be negative")
