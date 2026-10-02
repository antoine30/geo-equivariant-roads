import json

import numpy as np
import pytest
import torch

from eqroads.data_s2 import class_mask
from eqroads.export import class_polygons_to_geojson, write_class_geotiff
from eqroads.p4conv import rotate_spatial
from eqroads.segnet import P4UNet
from eqroads.synthetic import BUILDING, CLASS_NAMES, ROAD, TileConfig, make_batch
from eqroads.train import multiclass_loss

UTM40N = (10.0, 0.0, 255000.0, 0.0, -10.0, 2700000.0)


@pytest.mark.parametrize("k", [1, 2, 3])
def test_three_class_p4unet_is_equivariant(k):
    model = P4UNet(in_channels=4, width=4, out_channels=3).eval()
    x = torch.randn(1, 4, 32, 32)
    with torch.no_grad():
        assert torch.allclose(model(rotate_spatial(x, k)), rotate_spatial(model(x), k), atol=1e-4)


def test_synthetic_tiles_contain_all_classes():
    _, masks = make_batch(0, 4, TileConfig(size=96, n_settlements=(2, 3)))
    assert set(np.unique(masks)) == {0, ROAD, BUILDING}
    _, roads_only = make_batch(0, 4, TileConfig(size=96))
    assert set(np.unique(roads_only)) <= {0, ROAD}


def test_multiclass_loss_decreases_on_perfect_logits():
    target = torch.randint(0, 3, (2, 16, 16))
    perfect = torch.nn.functional.one_hot(target, 3).permute(0, 3, 1, 2).float() * 20
    assert multiclass_loss(perfect, target) < multiclass_loss(torch.zeros_like(perfect), target)


def test_class_mask_roads_take_precedence():
    roads = np.zeros((4, 4), np.uint8)
    roads[1] = 1
    frac = np.full((4, 4), 0.5, np.float32)
    frac[3] = 0.1
    c = class_mask(roads, frac, built_up_threshold=0.3)
    assert (c[1] == ROAD).all() and (c[0] == BUILDING).all() and (c[3] == 0).all()


def test_built_up_polygons_are_georeferenced(tmp_path):
    classes = np.zeros((20, 20), np.uint8)
    classes[2:6, 2:7] = BUILDING   # 20 px = 2000 m2
    classes[15, 15] = BUILDING     # isolated pixel, below min_area_px
    probs = np.zeros((3, 20, 20), np.float32)
    probs[BUILDING] = 0.8
    fc = class_polygons_to_geojson(classes, probs, BUILDING, "built-up", UTM40N, "EPSG:32640", {"source_item": "t"})
    assert len(fc["features"]) == 1
    props = fc["features"][0]["properties"]
    assert props["area_m2"] == 2000.0 and props["confidence"] == 0.8
    lon, lat = fc["features"][0]["geometry"]["coordinates"][0][0]
    assert 54.0 < lon < 55.0 and 24.0 < lat < 25.0
    json.dumps(fc)  # serialisable

    path = tmp_path / "classes.tif"
    write_class_geotiff(path, classes, UTM40N, "EPSG:32640", CLASS_NAMES)
    import rasterio
    with rasterio.open(path) as src:
        assert src.crs.to_epsg() == 32640 and (src.read(1) == classes).all()


def test_synthetic_roads_keep_their_heading():
    """Regression test: clipping must not bend roads towards the tile corners."""
    from eqroads.synthetic import _road_polyline

    rng = np.random.default_rng(0)
    for _ in range(500):
        theta = np.deg2rad(rng.uniform(-20, 20))
        p0, p1 = _road_polyline(rng, 96, theta, rng.random() < 0.3)
        v = p1 - p0
        assert abs(np.degrees(np.arctan2(-v[1], v[0])) - np.degrees(theta)) < 1e-6
        assert (p0 >= -1e-9).all() and (p1 <= 95 + 1e-9).all()
