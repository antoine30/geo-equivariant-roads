"""Evaluation for experiments E (sinuous roads) and F (new tracks across dunes).

Every method has its free parameters tuned on training tiles drawn from the same distribution
as the network's training data, then is scored once on the test tiles.
"""
from __future__ import annotations

import itertools
import time

import numpy as np
from scipy import ndimage
from skimage.transform import probabilistic_hough_line

from .dunes import DuneConfig, make_pairs
from .graph import extract_graph
from .hough import detect_roads, lines_to_mask
from .hough import tune as tune_hough
from .metrics import apls_components, relaxed_f1
from .radon import change_field, image_fields, local_radon_score
from .synthetic import ROAD, TileConfig, make_batch
from .train import predict

ROAD_TUNE = TileConfig(size=96, curved_prob=0.5, dead_end_prob=0.2)
ROAD_SINUOUS = TileConfig(size=96, curved_prob=1.0, n_roads=(1, 3), dead_end_prob=0.0)
ROAD_STRAIGHT = TileConfig(size=96, curved_prob=0.0, n_roads=(1, 3), dead_end_prob=0.0)
DUNES = DuneConfig(size=96)


def f1(pred, gt):
    return relaxed_f1(pred, gt, tolerance=1)["f1"]


def auc(pos, neg):
    """Area under the ROC curve (probability that a positive tile outscores a negative one)."""
    pos, neg = np.asarray(pos)[:, None], np.asarray(neg)[None, :]
    return float((pos > neg).mean() + 0.5 * (pos == neg).mean())


def pd_at_pfa(pos, neg, pfa=0.05):
    thr = np.quantile(neg, 1 - pfa)
    return float((np.asarray(pos) > thr).mean())


def _best(candidates):
    return max(candidates, key=lambda c: c[0])


# ============================================================================ E. sinuous roads
def radon_road_score(img, window):
    return np.maximum.reduce([local_radon_score(f, window, stride=window // 2) for f in image_fields(img)])


def graph_scores(pred_masks, gts, dilate):
    comps = []
    for p, g in zip(pred_masks, gts):
        p = ndimage.binary_dilation(p) if dilate else p
        comps.append(apls_components(extract_graph(g.astype(float)), extract_graph(p.astype(float))))
    return {k: float(np.mean([c[k] for c in comps])) for k in ("apls", "completeness", "correctness")}


def evaluate_roads(model, n_tune, n_test):
    tr_img, tr_mask = make_batch(310_000, n_tune, ROAD_TUNE)
    tr_gt = tr_mask == ROAD
    # Wider grid than for straight roads: sinuous roads favour short segments and small gaps.
    hough_grid = {"percentile": [88.0, 91.0, 94.0], "threshold": [2, 3, 5, 10, 20],
                  "line_length": [3, 5, 10], "line_gap": [1, 2, 3, 6, 10]}
    hough_p, _, hough_dict = tune_hough(tr_img, tr_gt, f1, hough_grid)
    windows = (16, 24, 32)
    tr_radon = {w: [radon_road_score(i, w) for i in tr_img] for w in windows}
    _, radon_w, radon_t = _best(
        (np.mean([f1(s >= t, g) for s, g in zip(tr_radon[w], tr_gt)]), w, float(t))
        for w in windows for t in np.arange(3.0, 12.5, 0.5))
    tr_prob = predict(model, tr_img)
    _, dl_t = _best((np.mean([f1(p >= t, g) for p, g in zip(tr_prob, tr_gt)]), float(t))
                    for t in np.arange(0.3, 0.75, 0.05))
    params = {"hough": hough_dict, "radon": {"window": radon_w, "threshold": radon_t}, "p4unet": {"threshold": dl_t}}
    print("E tuned:", params)

    res = {"params": params}
    for name, cfg, seed in (("straight", ROAD_STRAIGHT, 320_000), ("sinuous", ROAD_SINUOUS, 330_000)):
        img, mask = make_batch(seed, n_test, cfg)
        gt = mask == ROAD
        preds = {
            "hough": [detect_roads(i, hough_p) for i in img],
            "radon": [radon_road_score(i, radon_w) >= radon_t for i in img],
            "p4unet": list(predict(model, img) >= dl_t),
        }
        res[name] = {}
        for m, p in preds.items():
            res[name][m] = {"f1": float(np.mean([f1(a, b) for a, b in zip(p, gt)])),
                            **graph_scores(p, gt, dilate=m != "p4unet")}
        print(f"E {name}:", {m: round(v["f1"], 3) for m, v in res[name].items()})
    res["_examples"] = (img[:3], gt[:3], {m: p[:3] for m, p in preds.items()})
    return res


# ============================================================================ F. tracks in dunes
def hough_change(t0, t1, p):
    z = np.abs(change_field(t0, t1))
    edges = z >= np.percentile(z, p["percentile"])
    lines = probabilistic_hough_line(edges, threshold=p["threshold"], line_length=p["line_length"],
                                     line_gap=p["line_gap"], rng=0)
    return lines_to_mask(lines, z.shape)


def naive_change(t0, t1):
    return ndimage.gaussian_filter(np.abs(change_field(t0, t1)), 0.7)


def change_input(t0, t1):
    return np.concatenate([t0, t1], axis=-3)


def evaluate_changes(model, n_tune, n_test):
    a0, a1, al, _ = make_pairs(410_000, n_tune, DUNES, balanced=True)
    pos = [i for i in range(n_tune) if al[i].any()]

    def tune_threshold(score_maps, grid):
        return _best((np.mean([f1(score_maps[i] >= t, al[i] > 0) for i in pos]), float(t)) for t in grid)[1]

    naive_maps = [naive_change(x, y) for x, y in zip(a0, a1)]
    naive_t = tune_threshold(naive_maps, np.arange(1.0, 6.01, 0.25))
    windows = (8, 12, 16)
    radon_maps = {w: [local_radon_score(change_field(x, y), w, stride=w // 2, min_length=max(4, w // 3))
                      for x, y in zip(a0, a1)] for w in windows}
    radon_w, radon_t = _best(
        (np.mean([f1(radon_maps[w][i] >= t, al[i] > 0) for i in pos]), (w, float(t)))
        for w in windows for t in np.arange(4.0, 16.5, 0.5))[1]
    grid = {"percentile": [97.0, 98.0, 99.0, 99.5], "threshold": [2, 3, 5, 10], "line_length": [3, 5, 10],
            "line_gap": [3, 6, 10]}
    hough_p = _best((np.mean([f1(hough_change(a0[i], a1[i], p), al[i] > 0) for i in pos]), p)
                    for p in (dict(zip(grid, v)) for v in itertools.product(*grid.values())))[1]
    dl_maps = predict(model, change_input(a0, a1))
    dl_t = tune_threshold(dl_maps, np.arange(0.2, 0.85, 0.05))
    params = {"naive": {"threshold": naive_t}, "hough": hough_p,
              "radon": {"window": radon_w, "threshold": radon_t}, "p4unet": {"threshold": dl_t}}
    print("F tuned:", params)

    t0, t1, lab, has = make_pairs(400_000, n_test, DUNES, balanced=True)
    timings = {}

    def run(name, fn):
        start = time.perf_counter()
        out = [fn(x, y) for x, y in zip(t0, t1)]
        timings[name] = (time.perf_counter() - start) / len(t0)
        return out

    maps = {
        "naive difference": run("naive difference", naive_change),
        "hough": run("hough", lambda x, y: hough_change(x, y, hough_p).astype(float)),
        "radon": run("radon", lambda x, y: local_radon_score(change_field(x, y), radon_w, stride=radon_w // 2,
                                                             min_length=max(4, radon_w // 3))),
        "p4unet": run("p4unet", lambda x, y: predict(model, change_input(x, y)[None])[0]),
    }
    thresholds = {"naive difference": naive_t, "hough": 0.5, "radon": radon_t, "p4unet": dl_t}
    res = {"params": params}
    for m, s in maps.items():
        masks = [x >= thresholds[m] for x in s]
        tile_score = [x.sum() if m == "hough" else x.max() for x in s]
        ps = [tile_score[i] for i in range(n_test) if has[i]]
        ns = [tile_score[i] for i in range(n_test) if not has[i]]
        res[m] = {
            "pixel_f1_on_new_tracks": float(np.mean([f1(masks[i], lab[i] > 0) for i in range(n_test) if has[i]])),
            "false_alert_tiles": float(np.mean([masks[i].sum() >= 10 for i in range(n_test) if not has[i]])),
            "tile_auc": auc(ps, ns),
            "pd_at_pfa_5pct": pd_at_pfa(ps, ns),
            "seconds_per_tile": timings[m],
        }
        print(f"F {m}:", {k: round(v, 3) for k, v in res[m].items()})
    order = [i for i in range(n_test) if has[i]][:2] + [i for i in range(n_test) if not has[i]][:1]
    res["_examples"] = (t0[order], t1[order], lab[order], {m: [maps[m][i] for i in order] for m in maps},
                        thresholds)
    return res


def evaluate(road_model, change_model, quick=False, figures_dir=None):
    n_tune, n_test = (6, 6) if quick else (30, 40)
    roads = evaluate_roads(road_model, n_tune, n_test)
    changes = evaluate_changes(change_model, n_tune if quick else 40, 8 if quick else 80)
    if figures_dir is not None:
        try:
            from .viz import save_tracks_figures
            save_tracks_figures(roads.pop("_examples"), changes.pop("_examples"), figures_dir)
        except ImportError as err:
            print("figures skipped:", err)
    roads.pop("_examples", None)
    changes.pop("_examples", None)
    return {"E_sinuous_roads": roads, "F_new_tracks_in_dunes": changes}
