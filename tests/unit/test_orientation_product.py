from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from gonet_astrometry.models.frame import ObserverLocation
from gonet_astrometry.products.orientation import (
    OrientationProduct,
    load_orientation_product,
    save_orientation_product,
)
from gonet_astrometry.solving.orientation import (
    AbsoluteOrientationSolution,
    OrientationFitConfig,
    OrientationMatch,
)


def test_orientation_product_round_trip_without_pickle(tmp_path: Path) -> None:
    solution = AbsoluteOrientationSolution(
        grid_to_enu=np.eye(3),
        reference_time=datetime(2026, 8, 18, tzinfo=timezone.utc),
        location=ObserverLocation(42.0, -88.0, 200.0),
        ncp_grid=np.array([0.0, np.cos(np.deg2rad(42.0)), np.sin(np.deg2rad(42.0))]),
        ncp_enu=np.array([0.0, np.cos(np.deg2rad(42.0)), np.sin(np.deg2rad(42.0))]),
        twist_deg=0.0,
        fit_rms_deg=0.01,
        fit_median_deg=0.005,
        fit_p95_deg=0.02,
        anchor_count=12,
        catalog_star_count=100,
        matches=(
            OrientationMatch(
                track_identifier=7,
                catalog_identifier="HR 1",
                residual_deg=0.005,
                observed_declination_deg=42.0,
                catalog_declination_deg=42.0,
                catalog_magnitude=2.0,
                ray_grid=np.array([0.0, 1.0, 0.0]),
            ),
        ),
    )
    product = OrientationProduct(
        product_id="orientation-id",
        sidereal_product_id="sidereal-id",
        catalog_path=tmp_path / "catalog.npz",
        fit_config=OrientationFitConfig(),
        solution=solution,
    )
    path = save_orientation_product(tmp_path / "orientation.npz", product)
    with np.load(path, allow_pickle=False) as data:
        assert str(data["format"].item()) == "gonet-astrometry-absolute-orientation"
    loaded = load_orientation_product(
        path,
        expected_product_id="orientation-id",
        expected_sidereal_product_id="sidereal-id",
    )
    assert loaded.product_id == product.product_id
    assert np.allclose(loaded.solution.grid_to_enu, np.eye(3))
    assert loaded.solution.matches[0].catalog_identifier == "HR 1"
