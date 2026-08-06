import numpy as np

from gonet_astrometry.portal.figures import channel_image_figure, empty_image_figure


def test_empty_image_figure_contains_message() -> None:
    figure = empty_image_figure("Nothing loaded")

    assert figure.layout.annotations[0].text == "Nothing loaded"


def test_channel_image_figure_contains_native_channel_data() -> None:
    data = np.arange(24, dtype=np.float64).reshape(4, 6)

    figure = channel_image_figure(
        data,
        channel="green1",
        source_name="frame.jpg",
    )

    assert len(figure.data) == 1
    assert figure.data[0].type == "heatmap"
    assert np.array_equal(figure.data[0].z, data)
    assert figure.layout.dragmode == "pan"
    assert figure.layout.yaxis.autorange == "reversed"


def test_channel_image_figure_handles_constant_and_nonfinite_data() -> None:
    constant = channel_image_figure(
        np.ones((2, 2), dtype=np.float64),
        channel="red",
        source_name="constant.jpg",
    )
    nonfinite = channel_image_figure(
        np.full((2, 2), np.nan),
        channel="blue",
        source_name="nan.jpg",
    )

    assert constant.data[0].zmax > constant.data[0].zmin
    assert (nonfinite.data[0].zmin, nonfinite.data[0].zmax) == (0.0, 1.0)
