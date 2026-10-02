"""Classical baseline: road detection with a ridge filter and the probabilistic Hough transform.

    reflectance -> two channels (mean visible brightness, near-infrared)
    -> multi-scale Sato ridge filter on each (dark and bright ridges) -> max
    -> threshold at a percentile -> probabilistic Hough transform
    -> detected line segments rasterised as one-pixel road centrelines.

No learning: a handful of parameters, tuned by grid search on the same training tiles as the
networks. The two-channel input was selected among several variants (per-band filtering,
pre-smoothing, NDVI): it gave the best training F1. The Hough transform votes over all line directions with equal weight, so this
pipeline is (up to pixel discretisation) rotation-invariant by construction, like the
p4 network and unlike a standard CNN.
"""
from __future__ import annotations

import itertools
from dataclasses import asdict, dataclass

import numpy as np
from skimage.draw import line as draw_line
from skimage.filters import sato
from skimage.transform import probabilistic_hough_line


@dataclass(frozen=True)
class HoughParams:
    percentile: float = 93.0   # ridge-response percentile kept as candidate road pixels
    threshold: int = 10        # Hough accumulator votes needed to accept a line
    line_length: int = 20      # minimum segment length (px)
    line_gap: int = 6          # maximum gap bridged inside a segment (px)


def road_response(img: np.ndarray, sigmas=(1.0, 1.5)) -> np.ndarray:
    """Line-likeness in [0, 1]: roads are thin ridges, darker or brighter than their surroundings.

    Computed on mean visible brightness and on near-infrared: averaging the visible bands
    reduces noise, and NIR separates roads from vegetation.
    """
    resp = np.zeros(img.shape[1:], dtype=np.float32)
    for band in (img[:3].mean(axis=0), img[3]):
        r = np.maximum(sato(band, sigmas=sigmas, black_ridges=True, mode="reflect"),
                       sato(band, sigmas=sigmas, black_ridges=False, mode="reflect"))
        resp = np.maximum(resp, r / (np.percentile(r, 99.5) + 1e-9))
    return np.clip(resp, 0, 1)


def hough_lines(img: np.ndarray, p: HoughParams, response: np.ndarray | None = None):
    resp = road_response(img) if response is None else response
    edges = resp >= np.percentile(resp, p.percentile)
    return probabilistic_hough_line(edges, threshold=p.threshold, line_length=p.line_length,
                                    line_gap=p.line_gap, rng=0)


def lines_to_mask(lines, shape) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for (x0, y0), (x1, y1) in lines:
        rr, cc = draw_line(y0, x0, y1, x1)
        mask[rr, cc] = True
    return mask


def detect_roads(img: np.ndarray, p: HoughParams | None = None, response=None) -> np.ndarray:
    return lines_to_mask(hough_lines(img, p or HoughParams(), response), img.shape[1:])


def tune(images, masks, score_fn, grid: dict | None = None):
    """Grid search of HoughParams maximising the mean of score_fn(pred_mask, gt_mask)."""
    grid = grid or {"percentile": [88.0, 91.0, 94.0], "threshold": [20, 40, 60],
                    "line_length": [5, 10, 20], "line_gap": [10, 20, 30, 45]}
    responses = [road_response(i) for i in images]
    best, best_score = None, -1.0
    for values in itertools.product(*grid.values()):
        p = HoughParams(**dict(zip(grid.keys(), values)))
        score = float(np.mean([score_fn(detect_roads(i, p, r), m) for i, r, m in zip(images, responses, masks)]))
        if score > best_score:
            best, best_score = p, score
    return best, best_score, asdict(best)
