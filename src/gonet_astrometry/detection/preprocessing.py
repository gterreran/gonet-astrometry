"""Mask-aware Bayer preprocessing shared by detector backends."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import (
    binary_closing,
    binary_fill_holes,
    distance_transform_edt,
    gaussian_filter,
    label,
    zoom,
)

from gonet_astrometry.detection.config import DetectionConfig
from gonet_astrometry.detection.errors import DetectionInputError
from gonet_astrometry.models.frame import ImageFrame

_BAYER_PARITIES = ((0, 0), (0, 1), (1, 0), (1, 1))
_MAD_TO_SIGMA: float = 1.482602218505602
_FLOAT_EPSILON: float = float(np.finfo(np.float64).eps)


@dataclass(frozen=True, slots=True)
class PreparedDetectionImage:
    """Bayer-normalized image used by source-detection backends.

    Parameters
    ----------
    data
        Full-resolution native sensor array expressed approximately in local
        noise-sigma units. No demosaicing or coordinate resampling is applied.
    mask
        Combined invalid-pixel mask. It includes nonfinite input pixels, the
        input frame mask, pixels outside the estimated field of view, and large
        bright dynamic contaminants.
    field_mask
        Boolean mask with ``True`` for pixels inside the usable optical field.
    dynamic_mask
        Boolean mask with ``True`` for temporarily excluded bright regions.
    backgrounds
        Median local-background values for the four sensor parities.
    noises
        Median local-noise values for the four sensor parities.
    """

    data: NDArray[np.float64]
    mask: NDArray[np.bool_]
    field_mask: NDArray[np.bool_]
    dynamic_mask: NDArray[np.bool_]
    backgrounds: tuple[float, float, float, float]
    noises: tuple[float, float, float, float]

    @property
    def field_fraction(self) -> float:
        """Return the fraction of sensor pixels inside the usable field."""
        return float(np.mean(self.field_mask))

    @property
    def dynamic_mask_fraction(self) -> float:
        """Return the fraction of field pixels excluded dynamically."""
        field_pixels = int(np.count_nonzero(self.field_mask))
        if field_pixels == 0:
            return 0.0
        masked_pixels = np.count_nonzero(self.dynamic_mask & self.field_mask)
        return float(masked_pixels / field_pixels)


def prepare_bayer_detection_image(
    frame: ImageFrame,
    config: DetectionConfig | None = None,
) -> PreparedDetectionImage:
    """Create a locally normalized, mask-aware native Bayer image.

    The four Bayer parities are processed independently. A provisional optical
    footprint excludes the dark exterior of the fisheye image before tiled,
    sigma-clipped background and noise maps are estimated. Large bright regions
    may then be masked without removing isolated bright stars.

    Parameters
    ----------
    frame
        Full-resolution native GONet image frame.
    config
        Shared detection and preprocessing settings. Defaults are used when
        omitted.

    Returns
    -------
    PreparedDetectionImage
        Locally background-subtracted, noise-scaled detector input and masks.

    Raises
    ------
    DetectionInputError
        If the image is too small or a Bayer parity has no usable pixels.
    """
    settings = config or DetectionConfig()
    if min(frame.shape) < 2:
        raise DetectionInputError(
            "A Bayer detection image must contain at least two rows and columns."
        )
    if frame.shape[0] % 2 or frame.shape[1] % 2:
        raise DetectionInputError(
            "A native Bayer detection image must have even row and column counts."
        )

    source = np.asarray(frame.data, dtype=np.float64)
    base_invalid = ~np.isfinite(source)
    if frame.mask is not None:
        base_invalid |= np.asarray(frame.mask, dtype=np.bool_)

    field_mask = estimate_provisional_field_mask(source, base_invalid, settings)
    invalid = base_invalid | ~field_mask
    normalized = np.zeros(source.shape, dtype=np.float64)
    backgrounds: list[float] = []
    noises: list[float] = []

    parity_box_size = max(4, int(round(settings.background_box_size_px / 2)))
    for row_offset, column_offset in _BAYER_PARITIES:
        samples = source[row_offset::2, column_offset::2]
        parity_invalid = invalid[row_offset::2, column_offset::2]
        if not np.any(~parity_invalid):
            raise DetectionInputError(
                "Each Bayer parity must contain at least one unmasked finite pixel."
            )

        background_map, noise_map = _local_background_maps(
            samples,
            parity_invalid,
            box_size=parity_box_size,
            min_valid_fraction=settings.background_min_valid_fraction,
            sigma_clip=settings.background_sigma_clip,
            clip_iterations=settings.background_clip_iterations,
        )
        parity = (samples - background_map) / noise_map
        parity = np.asarray(parity, dtype=np.float64)
        parity[parity_invalid] = 0.0
        normalized[row_offset::2, column_offset::2] = parity

        valid = ~parity_invalid
        backgrounds.append(float(np.median(background_map[valid])))
        noises.append(float(np.median(noise_map[valid])))

    dynamic_mask = _bright_region_mask(normalized, field_mask, invalid, settings)
    combined_mask = invalid | dynamic_mask
    normalized[combined_mask] = 0.0

    return PreparedDetectionImage(
        data=normalized,
        mask=combined_mask,
        field_mask=field_mask,
        dynamic_mask=dynamic_mask,
        backgrounds=(
            backgrounds[0],
            backgrounds[1],
            backgrounds[2],
            backgrounds[3],
        ),
        noises=(noises[0], noises[1], noises[2], noises[3]),
    )


def estimate_provisional_field_mask(
    data: NDArray[np.float64],
    invalid_mask: NDArray[np.bool_],
    config: DetectionConfig,
) -> NDArray[np.bool_]:
    """Estimate the illuminated fisheye footprint in native sensor coordinates.

    The estimate is intentionally conservative. It compares a smoothed compact
    2-by-2 Bayer-quad image with the dark corner level, keeps the largest
    connected illuminated region, fills holes, and erodes the result inward.
    If the image does not contain enough center-to-corner contrast to infer a
    footprint safely, every otherwise valid pixel is retained.

    Parameters
    ----------
    data
        Full-resolution native Bayer mosaic.
    invalid_mask
        Boolean mask with ``True`` for pixels that cannot be used.
    config
        Detection configuration controlling footprint inference.

    Returns
    -------
    numpy.ndarray
        Boolean full-resolution mask with ``True`` inside the usable field.

    Raises
    ------
    DetectionInputError
        If no finite unmasked Bayer quads are available.
    """
    valid_input = ~invalid_mask
    if not config.use_provisional_field_mask:
        return valid_input.copy()

    compact, compact_valid = _compact_quad_mean(data, valid_input)
    if not np.any(compact_valid):
        raise DetectionInputError("The image contains no finite unmasked Bayer quads.")

    smoothing_sigma = max(config.footprint_smoothing_px / 2.0, 0.0)
    filled = compact.copy()
    fallback = float(np.median(compact[compact_valid]))
    filled[~compact_valid] = fallback
    smoothed = (
        gaussian_filter(filled, sigma=smoothing_sigma, mode="nearest")
        if smoothing_sigma > 0
        else filled
    )

    height, width = compact.shape
    patch = max(4, min(height, width) // 12)
    corner_mask = np.zeros(compact.shape, dtype=bool)
    corner_mask[:patch, :patch] = True
    corner_mask[:patch, -patch:] = True
    corner_mask[-patch:, :patch] = True
    corner_mask[-patch:, -patch:] = True
    corner_values = smoothed[corner_mask & compact_valid]

    center_height = max(4, height // 4)
    center_width = max(4, width // 4)
    row_start = max(0, (height - center_height) // 2)
    column_start = max(0, (width - center_width) // 2)
    center_mask = np.zeros(compact.shape, dtype=bool)
    center_mask[
        row_start : row_start + center_height,
        column_start : column_start + center_width,
    ] = True
    center_values = smoothed[center_mask & compact_valid]

    if corner_values.size == 0 or center_values.size == 0:
        return valid_input.copy()

    exterior = float(np.median(corner_values))
    interior = float(np.median(center_values))
    corner_mad = float(np.median(np.abs(corner_values - exterior)))
    corner_noise = max(_MAD_TO_SIGMA * corner_mad, _FLOAT_EPSILON)
    contrast = interior - exterior
    if not np.isfinite(contrast) or contrast <= 3.0 * corner_noise:
        return valid_input.copy()

    threshold = exterior + config.footprint_threshold_fraction * contrast
    candidate = (smoothed > threshold) & compact_valid
    candidate = binary_closing(candidate, iterations=2)
    candidate = binary_fill_holes(candidate)
    component = _largest_component(candidate)

    minimum_area = max(16, int(0.1 * np.count_nonzero(compact_valid)))
    if np.count_nonzero(component) < minimum_area:
        return valid_input.copy()

    erosion = config.footprint_erosion_px / 2.0
    if erosion > 0:
        component = distance_transform_edt(component) > erosion
    if not np.any(component):
        return valid_input.copy()

    expanded = np.repeat(np.repeat(component, 2, axis=0), 2, axis=1)
    expanded = expanded[: data.shape[0], : data.shape[1]]
    return np.asarray(expanded & valid_input, dtype=np.bool_)


def _compact_quad_mean(
    data: NDArray[np.float64],
    valid: NDArray[np.bool_],
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Return a compact 2-by-2-quad mean without demosaicing coordinates."""
    height = data.shape[0] // 2
    width = data.shape[1] // 2
    total = np.zeros((height, width), dtype=np.float64)
    count = np.zeros((height, width), dtype=np.int16)
    for row_offset, column_offset in _BAYER_PARITIES:
        samples = data[row_offset : 2 * height : 2, column_offset : 2 * width : 2]
        sample_valid = valid[
            row_offset : 2 * height : 2,
            column_offset : 2 * width : 2,
        ]
        total += np.where(sample_valid, samples, 0.0)
        count += sample_valid.astype(np.int16)

    compact_valid = count > 0
    compact = np.zeros(total.shape, dtype=np.float64)
    compact[compact_valid] = total[compact_valid] / count[compact_valid]
    return compact, compact_valid


def _largest_component(mask: NDArray[np.bool_]) -> NDArray[np.bool_]:
    """Return the largest eight-connected component in ``mask``."""
    labels, count = label(mask, structure=np.ones((3, 3), dtype=np.int8))
    if count == 0:
        return np.zeros(mask.shape, dtype=bool)
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    return np.asarray(labels == int(np.argmax(sizes)), dtype=np.bool_)


def _local_background_maps(
    data: NDArray[np.float64],
    invalid: NDArray[np.bool_],
    *,
    box_size: int,
    min_valid_fraction: float,
    sigma_clip: float,
    clip_iterations: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Estimate smooth local background and noise maps from robust box statistics."""
    rows, columns = data.shape
    grid_rows = max(1, ceil(rows / box_size))
    grid_columns = max(1, ceil(columns / box_size))
    background_grid: NDArray[np.float64] = np.full(
        (grid_rows, grid_columns), np.nan, dtype=np.float64
    )
    noise_grid: NDArray[np.float64] = np.full(
        (grid_rows, grid_columns), np.nan, dtype=np.float64
    )

    for grid_row in range(grid_rows):
        row_start = grid_row * box_size
        row_stop = min(rows, row_start + box_size)
        for grid_column in range(grid_columns):
            column_start = grid_column * box_size
            column_stop = min(columns, column_start + box_size)
            tile = data[row_start:row_stop, column_start:column_stop]
            tile_invalid = invalid[row_start:row_stop, column_start:column_stop]
            valid = tile[~tile_invalid]
            minimum = min(tile.size, max(1, int(ceil(tile.size * min_valid_fraction))))
            if valid.size < minimum:
                continue
            background, noise = _sigma_clipped_location_scale(
                valid,
                sigma=sigma_clip,
                iterations=clip_iterations,
            )
            background_grid[grid_row, grid_column] = background
            noise_grid[grid_row, grid_column] = noise

    if not np.any(np.isfinite(background_grid)):
        raise DetectionInputError(
            "No local background box contains enough valid pixels."
        )

    background_grid = _fill_grid_holes(background_grid)
    noise_grid = _fill_grid_holes(noise_grid)
    if background_grid.size > 1:
        background_grid = gaussian_filter(background_grid, sigma=0.75, mode="nearest")
        noise_grid = gaussian_filter(noise_grid, sigma=0.75, mode="nearest")

    target_shape = (rows, columns)
    background_map = _resize_grid(background_grid, target_shape)
    noise_map = _resize_grid(noise_grid, target_shape)
    noise_map = np.maximum(noise_map, _FLOAT_EPSILON)
    return background_map, noise_map


def _sigma_clipped_location_scale(
    values: NDArray[np.float64],
    *,
    sigma: float,
    iterations: int,
) -> tuple[float, float]:
    """Return a sigma-clipped median and robust standard deviation."""
    clipped = np.asarray(values[np.isfinite(values)], dtype=np.float64)
    if clipped.size == 0:
        raise DetectionInputError("Background statistics require finite pixels.")

    for _ in range(iterations):
        center = float(np.median(clipped))
        mad = float(np.median(np.abs(clipped - center)))
        scale = _MAD_TO_SIGMA * mad
        if not np.isfinite(scale) or scale <= _FLOAT_EPSILON:
            scale = float(np.std(clipped))
        if not np.isfinite(scale) or scale <= _FLOAT_EPSILON:
            return center, 1.0
        retained = np.abs(clipped - center) <= sigma * scale
        if retained.all() or not np.any(retained):
            return center, scale
        clipped = clipped[retained]

    center = float(np.median(clipped))
    mad = float(np.median(np.abs(clipped - center)))
    scale = _MAD_TO_SIGMA * mad
    if not np.isfinite(scale) or scale <= _FLOAT_EPSILON:
        scale = float(np.std(clipped))
    if not np.isfinite(scale) or scale <= _FLOAT_EPSILON:
        scale = 1.0
    return center, scale


def _fill_grid_holes(grid: NDArray[np.float64]) -> NDArray[np.float64]:
    """Fill missing background boxes from their nearest valid neighbor."""
    missing = ~np.isfinite(grid)
    if not np.any(missing):
        return grid
    if np.all(missing):
        raise DetectionInputError("Every local background box is invalid.")
    nearest = distance_transform_edt(
        missing,
        return_distances=False,
        return_indices=True,
    )
    return np.asarray(grid[tuple(nearest)], dtype=np.float64)


def _resize_grid(
    grid: NDArray[np.float64],
    shape: tuple[int, int],
) -> NDArray[np.float64]:
    """Bilinearly resize a coarse grid to ``shape`` with deterministic cropping."""
    factors = (shape[0] / grid.shape[0], shape[1] / grid.shape[1])
    resized = zoom(grid, factors, order=1, mode="nearest", prefilter=False)
    if resized.shape[0] < shape[0] or resized.shape[1] < shape[1]:
        resized = np.pad(
            resized,
            (
                (0, max(0, shape[0] - resized.shape[0])),
                (0, max(0, shape[1] - resized.shape[1])),
            ),
            mode="edge",
        )
    return np.asarray(resized[: shape[0], : shape[1]], dtype=np.float64)


def _bright_region_mask(
    significance: NDArray[np.float64],
    field_mask: NDArray[np.bool_],
    invalid: NDArray[np.bool_],
    config: DetectionConfig,
) -> NDArray[np.bool_]:
    """Mask only large connected high-significance contaminants."""
    if config.bright_mask_sigma is None:
        return np.zeros(significance.shape, dtype=bool)

    seeds = (significance >= config.bright_mask_sigma) & field_mask & ~invalid
    labels, count = label(seeds, structure=np.ones((3, 3), dtype=np.int8))
    if count == 0:
        return np.zeros(significance.shape, dtype=bool)

    sizes = np.bincount(labels.ravel())
    accepted_labels = np.flatnonzero(sizes >= config.bright_mask_min_pixels)
    accepted_labels = accepted_labels[accepted_labels != 0]
    if accepted_labels.size == 0:
        return np.zeros(significance.shape, dtype=bool)

    accepted = np.isin(labels, accepted_labels)
    if config.bright_mask_dilation_px > 0:
        accepted = distance_transform_edt(~accepted) <= config.bright_mask_dilation_px
    return np.asarray(accepted & field_mask, dtype=np.bool_)
