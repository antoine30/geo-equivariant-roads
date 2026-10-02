"""Experiment D: classical Hough-transform baseline on the exact test sets of experiments A, B and C.

The Hough pipeline (ridge filter + probabilistic Hough transform) has its parameters tuned by
grid search on 40 training tiles drawn from the same east-west distribution as the networks.
Deep-learning numbers are read from results/results.json and results/results_multiclass.json,
so run run_experiment.py and run_multiclass.py first.

    python scripts/run_hough.py     # ~2 min on CPU
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage

from eqroads.graph import extract_graph
from eqroads.hough import detect_roads, hough_lines, road_response, tune
from eqroads.metrics import apls_components, n_components, relaxed_f1
from eqroads.segnet import build_model
from eqroads.synthetic import ROAD, TileConfig, make_batch
from eqroads.train import predict

ROOT = Path(__file__).resolve().parents[1]
EW = (-20.0, 20.0)


def rot(arr, k):
    return np.rot90(arr, k, axes=(-2, -1)).copy()


def random_rot(images, masks, seed):  # identical to run_experiment.py, so test sets match
    ks = np.random.default_rng(seed).integers(0, 4, len(images))
    return (np.stack([rot(i, k) for i, k in zip(images, ks)]), np.stack([rot(m, k) for m, k in zip(masks, ks)]))


def f1(pred, gt):
    return relaxed_f1(pred, gt, tolerance=1)["f1"]


def mean_f1(images, masks, params):
    return float(np.mean([f1(detect_roads(i, params), m == ROAD) for i, m in zip(images, masks)]))


def main():
    out = ROOT / "results"
    dl = json.loads((out / "results.json").read_text())
    dl_mc = json.loads((out / "results_multiclass.json").read_text())

    # ---------------------------------------------------------------- tuning
    tr_img, tr_mask = make_batch(60_000, 40, TileConfig(size=96, angle_range=EW))
    params, score, params_dict = tune(tr_img, tr_mask == ROAD, f1)
    print(f"tuned Hough parameters {params_dict} (train F1 {score:.3f})")
    res = {"params": params_dict}

    # ---------------------------------------------------------------- A: orientation
    test_cfg = TileConfig(size=96, angle_range=EW)
    test_img, test_mask = make_batch(10_000, 60, test_cfg)
    any_img, any_mask = make_batch(20_000, 60, TileConfig(size=96))
    res["A"] = {
        "f1_east_west": mean_f1(test_img, test_mask, params),
        "f1_north_south": mean_f1(rot(test_img, 1), rot(test_mask, 1), params),
        "f1_any_heading": mean_f1(any_img, any_mask, params),
    }
    print("A", {k: round(v, 3) for k, v in res["A"].items()})

    # ---------------------------------------------------------------- C: roads among built-up areas
    mc_img, mc_mask = make_batch(50_000, 60, TileConfig(size=96, angle_range=EW, n_settlements=(1, 3)))
    res["C"] = {
        "road_f1_east_west": mean_f1(mc_img, mc_mask, params),
        "road_f1_north_south": mean_f1(rot(mc_img, 1), rot(mc_mask, 1), params),
    }
    print("C", {k: round(v, 3) for k, v in res["C"].items()})

    # ---------------------------------------------------------------- B: topology on long occlusions
    hard_cfg = TileConfig(size=96, angle_range=EW, n_occlusions=(3, 6), occlusion_radius=(4, 7))
    te_img, te_mask = random_rot(*make_batch(40_000, 80, hard_cfg), seed=2)
    comps, comp_err, px = [], [], []
    for img, m in zip(te_img, te_mask):
        pred = detect_roads(img, params)
        px.append(f1(pred, m == ROAD))
        # One-pixel dilation merges the parallel duplicate segments Hough finds on wide roads.
        G_gt = extract_graph((m == ROAD).astype(float))
        G = extract_graph(ndimage.binary_dilation(pred).astype(float))
        comps.append(apls_components(G_gt, G))
        comp_err.append(abs(n_components(G) - n_components(G_gt)))
    res["B"] = {
        "pixel_f1": float(np.mean(px)),
        "apls_lite": float(np.mean([c["apls"] for c in comps])),
        "apls_completeness": float(np.mean([c["completeness"] for c in comps])),
        "apls_correctness": float(np.mean([c["correctness"] for c in comps])),
        "component_error": float(np.mean(comp_err)),
    }
    print("B", {k: round(v, 3) for k, v in res["B"].items()})

    # ---------------------------------------------------------------- CPU cost per 96 x 96 tile
    torch.set_num_threads(1)
    t = time.perf_counter()
    for img in test_img[:20]:
        detect_roads(img, params)
    res["seconds_per_tile_hough"] = (time.perf_counter() - t) / 20
    model = build_model("p4unet")  # inference cost does not depend on the weights
    predict(model, test_img[:1])
    t = time.perf_counter()
    for img in test_img[:20]:
        predict(model, img[None])
    res["seconds_per_tile_p4unet"] = (time.perf_counter() - t) / 20
    print(f"CPU time per tile: Hough {res['seconds_per_tile_hough']:.3f}s, p4 U-Net {res['seconds_per_tile_p4unet']:.3f}s")

    res["deep_learning_reference"] = {
        "A": {k: dl[k] for k in ("unet", "p4unet")},
        "B": dl["topology"],
        "C": dl_mc,
    }
    (out / "results_hough.json").write_text(json.dumps(res, indent=2))

    try:
        save_figure(params, test_img, test_mask, mc_img, mc_mask, te_img, te_mask, ROOT / "docs")
    except ImportError as err:
        print("figure skipped:", err)
    print("done ->", out / "results_hough.json")


def save_figure(params, test_img, test_mask, mc_img, mc_mask, te_img, te_mask, outdir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from eqroads.viz import rgb

    cases = [
        ("straight road, as in training", test_img[0], test_mask[0]),
        ("rotated 90°", rot(test_img[0], 1), rot(test_mask[0], 1)),
        ("roads through built-up areas", mc_img[0], mc_mask[0]),
        ("long occlusions", te_img[3], te_mask[3]),
    ]
    fig, axes = plt.subplots(len(cases), 4, figsize=(10, 2.6 * len(cases)))
    for row, (title, img, mask) in enumerate(cases):
        resp = road_response(img)
        lines = hough_lines(img, params, resp)
        axes[row, 0].imshow(rgb(img))
        axes[row, 1].imshow(resp, cmap="magma", vmin=0, vmax=1)
        axes[row, 2].imshow(rgb(img))
        for (x0, y0), (x1, y1) in lines:
            axes[row, 2].plot([x0, x1], [y0, y1], color="#00e5ff", lw=1.3)
        axes[row, 3].imshow(mask == ROAD, cmap="gray")
        for ax, t in zip(axes[row], (f"RGB: {title}", "Sato ridge response", "Hough segments", "ground-truth roads")):
            ax.set_title(t, fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlim(0, img.shape[-1] - 1)
            ax.set_ylim(img.shape[-2] - 1, 0)
    fig.tight_layout()
    fig.savefig(outdir / "hough.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
