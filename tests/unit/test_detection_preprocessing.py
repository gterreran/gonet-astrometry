from datetime import datetime, timezone

import numpy as np
import pytest

from gonet_astrometry.detection.config import DetectionConfig
from gonet_astrometry.detection.errors import DetectionInputError
from gonet_astrometry.detection.preprocessing import prepare_bayer_detection_image
from gonet_astrometry.models.frame import ImageFrame, ImageMetadata, ObserverLocation


def _frame(data: np.ndarray, mask: np.ndarray | None = None) -> ImageFrame:
    metadata = ImageMetadata(
        datetime(2026, 8, 6, tzinfo=timezone.utc),
        10.0,
        ObserverLocation(0.0, 0.0),
    )
    return ImageFrame(data, metadata, mask=mask)


def test_bayer_preprocessing_normalizes_each_parity_and_preserves_shape() -> None:
    data = np.zeros((8, 10), dtype=np.float64)
    for index, (row, column) in enumerate(((0, 0), (0, 1), (1, 0), (1, 1))):
        values = np.arange(data[row::2, column::2].size, dtype=np.float64)
        data[row::2, column::2] = values.reshape(data[row::2, column::2].shape)
        data[row::2, column::2] += 100.0 * index
    mask = np.zeros(data.shape, dtype=bool)
    mask[0, 0] = True

    prepared = prepare_bayer_detection_image(_frame(data, mask))

    assert prepared.data.shape == data.shape
    assert prepared.mask[0, 0]
    assert prepared.data[0, 0] == 0.0
    assert len(prepared.backgrounds) == len(prepared.noises) == 4
    assert all(noise > 0 for noise in prepared.noises)
    assert all(
        np.isclose(np.median(prepared.data[row::2, column::2]), 0.0, atol=0.3)
        for row, column in ((0, 0), (0, 1), (1, 0), (1, 1))
    )


def test_bayer_preprocessing_uses_unit_noise_for_constant_parities() -> None:
    prepared = prepare_bayer_detection_image(_frame(np.ones((4, 4))))

    assert prepared.noises == (1.0, 1.0, 1.0, 1.0)
    assert np.count_nonzero(prepared.data) == 0


def test_bayer_preprocessing_rejects_small_or_fully_masked_input() -> None:
    with pytest.raises(DetectionInputError, match="at least two"):
        prepare_bayer_detection_image(_frame(np.ones((1, 2))))
    with pytest.raises(DetectionInputError, match="even row and column"):
        prepare_bayer_detection_image(_frame(np.ones((4, 5))))

    mask = np.zeros((4, 4), dtype=bool)
    mask[0::2, 0::2] = True
    with pytest.raises(DetectionInputError, match="Each Bayer parity"):
        prepare_bayer_detection_image(_frame(np.ones((4, 4)), mask))


def test_preprocessing_masks_dark_exterior_and_tracks_local_gradient() -> None:
    rng = np.random.default_rng(12)
    rows, columns = 128, 160
    yy, xx = np.indices((rows, columns), dtype=np.float64)
    radius = 52.0
    inside = (xx - 80.0) ** 2 + (yy - 62.0) ** 2 <= radius**2
    data = rng.normal(2.0, 0.2, size=(rows, columns))
    gradient = 80.0 + 0.15 * xx + 0.08 * yy
    data[inside] = gradient[inside] + rng.normal(0.0, 1.0, np.count_nonzero(inside))
    for index, (row, column) in enumerate(((0, 0), (0, 1), (1, 0), (1, 1))):
        data[row::2, column::2] += 15.0 * index

    prepared = prepare_bayer_detection_image(
        _frame(data),
        DetectionConfig(
            background_box_size_px=32,
            footprint_smoothing_px=6.0,
            bright_mask_sigma=None,
        ),
    )

    assert prepared.field_mask[62, 80]
    assert not prepared.field_mask[0, 0]
    assert not prepared.field_mask[-1, -1]
    assert 0.2 < prepared.field_fraction < 0.6
    valid_values = prepared.data[prepared.field_mask]
    assert abs(float(np.median(valid_values))) < 0.25
    assert float(np.std(valid_values)) < 2.0


def test_preprocessing_masks_large_bright_region_but_not_isolated_peak() -> None:
    rng = np.random.default_rng(9)
    data = rng.normal(100.0, 1.0, size=(96, 112))
    data[30:42, 55:69] += 80.0
    data[70, 20] += 80.0

    prepared = prepare_bayer_detection_image(
        _frame(data),
        DetectionConfig(
            use_provisional_field_mask=False,
            background_box_size_px=32,
            bright_mask_sigma=12.0,
            bright_mask_min_pixels=20,
            bright_mask_dilation_px=2.0,
        ),
    )

    assert prepared.dynamic_mask[35, 60]
    assert not prepared.dynamic_mask[70, 20]
    assert prepared.dynamic_mask_fraction > 0
    assert prepared.mask[35, 60]
    assert prepared.data[35, 60] == 0.0
