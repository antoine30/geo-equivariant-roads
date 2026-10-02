import numpy as np

from eqroads.hough import HoughParams, detect_roads
from eqroads.metrics import relaxed_f1


def road_image(vertical: bool):
    """Clean 4-band scene with one 3-px road (vegetation background, asphalt road)."""
    img = np.empty((4, 64, 64), np.float32)
    img[:] = np.array([0.04, 0.07, 0.05, 0.35])[:, None, None]
    img += 0.005 * np.random.default_rng(0).standard_normal(img.shape)
    gt = np.zeros((64, 64), bool)
    gt[31:34, 4:60] = True
    if vertical:
        img, gt = np.rot90(img, 1, axes=(1, 2)).copy(), np.rot90(gt).copy()
    img[:, gt] = np.array([0.08, 0.09, 0.10, 0.12])[:, None]
    return img, gt


def test_hough_finds_a_clear_road_in_any_orientation():
    p = HoughParams(percentile=94.0, threshold=20, line_length=10, line_gap=10)
    scores = []
    for vertical in (False, True):
        img, gt = road_image(vertical)
        scores.append(relaxed_f1(detect_roads(img, p), gt))
    assert all(s["recall"] > 0.95 for s in scores)            # the road is found in both orientations
    assert abs(scores[0]["f1"] - scores[1]["f1"]) < 0.05      # no preferred direction
    assert all(s["precision"] > 0.5 for s in scores)          # extra lines on the road flanks are bounded
