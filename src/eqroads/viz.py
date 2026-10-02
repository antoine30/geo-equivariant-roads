"""Figures for the README (requires matplotlib)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .graph import extract_graph
from .metrics import apls_lite
from .train import gnn_scores, predict, refine


def rgb(img: np.ndarray) -> np.ndarray:
    """True-colour composite (B04, B03, B02) with a simple linear stretch."""
    return np.clip(np.stack([img[2], img[1], img[0]], -1) * 3.5, 0, 1)


def draw_graph(ax, G, colour="white", inferred_colour="#00e5ff"):
    for u, v, d in G.edges(data=True):
        p = d["path"]
        inferred = d.get("inferred", False)
        ax.plot(p[:, 1], p[:, 0], color=inferred_colour if inferred else colour, lw=2.2 if inferred else 1.2)
    ends = np.array([G.nodes[n]["rc"] for n in G if G.degree(n) == 1]).reshape(-1, 2)
    ax.scatter(ends[:, 1], ends[:, 0], s=12, c="#ff3b30", zorder=3)


def _clean(ax, title):
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])


def save_figures(models, test_img, test_mask, graph_ex, graph_img, graph_prob, graph_gt, gnn, outdir: Path,
                 threshold: float = 0.5):
    outdir.mkdir(parents=True, exist_ok=True)

    # 1. Orientation generalisation ------------------------------------------------------
    i = 0
    fig, axes = plt.subplots(2, 4, figsize=(10, 6.0))
    for row, k in enumerate((0, 1)):
        img = np.rot90(test_img[i], k, axes=(-2, -1)).copy()
        mask = np.rot90(test_mask[i], k).copy()
        probs = {n: predict(m, img[None])[0] for n, m in models.items()}
        label = "east-west roads (as in training)" if k == 0 else "north-south roads (never seen)"
        axes[row, 0].imshow(rgb(img))
        _clean(axes[row, 0], f"Sentinel-2-like RGB\n{label}")
        axes[row, 1].imshow(mask, cmap="gray")
        _clean(axes[row, 1], "ground truth")
        axes[row, 2].imshow(probs["unet"], cmap="magma", vmin=0, vmax=1)
        _clean(axes[row, 2], "standard U-Net")
        axes[row, 3].imshow(probs["p4unet"], cmap="magma", vmin=0, vmax=1)
        _clean(axes[row, 3], "p4-equivariant U-Net")
    fig.tight_layout(h_pad=2.0)
    fig.savefig(outdir / "orientation.png", dpi=130)
    plt.close(fig)

    # 2. Topology repair --------------------------------------------------------------------
    # Illustrative example: among small graphs, the tile where the GNN repair helps most.
    best, j, G_after = -1.0, 0, None
    for i, ex in enumerate(graph_ex):
        if not ex.pairs or ex.G.number_of_edges() > 12:
            continue
        G_gt = extract_graph(graph_gt[i].astype(float))
        G_ref, _ = refine(ex, gnn_scores(gnn, ex), threshold)
        gain = apls_lite(G_gt, G_ref) - apls_lite(G_gt, ex.G)
        if gain > best:
            best, j, G_after = gain, i, G_ref
    ex = graph_ex[j]
    fig, axes = plt.subplots(1, 4, figsize=(11, 3.2))
    axes[0].imshow(rgb(graph_img[j]))
    _clean(axes[0], "RGB (occlusions hide road stretches)")
    axes[1].imshow(graph_prob[j], cmap="magma", vmin=0, vmax=1)
    _clean(axes[1], "road probability")
    axes[2].imshow(np.zeros_like(graph_prob[j]), cmap="gray", vmin=0, vmax=1)
    draw_graph(axes[2], ex.G)
    _clean(axes[2], "graph before (red = dead ends)")
    axes[3].imshow(graph_gt[j] * 0.35, cmap="gray", vmin=0, vmax=1)
    draw_graph(axes[3], G_after)
    _clean(axes[3], "after GNN (cyan = inferred links)\nover ground truth")
    fig.tight_layout()
    fig.savefig(outdir / "topology.png", dpi=130)
    plt.close(fig)


CLASS_COLOURS = np.array([[0.08, 0.08, 0.10], [1.0, 0.84, 0.0], [0.90, 0.22, 0.27]])


def save_multiclass_figure(models, test_img, test_mask, outdir: Path, index: int = 0):
    """Roads (yellow) and built-up areas (red), trained east-west, tested as-is and rotated 90 degrees."""
    outdir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 4, figsize=(10, 6.0))
    for row, k in enumerate((0, 1)):
        img = np.rot90(test_img[index], k, axes=(-2, -1)).copy()
        mask = np.rot90(test_mask[index], k).copy()
        label = "as in training" if k == 0 else "rotated 90° (unseen)"
        axes[row, 0].imshow(rgb(img))
        _clean(axes[row, 0], f"Sentinel-2-like RGB\n{label}")
        axes[row, 1].imshow(CLASS_COLOURS[mask])
        _clean(axes[row, 1], "ground truth")
        for col, name, title in ((2, "unet", "standard U-Net"), (3, "p4unet", "p4-equivariant U-Net")):
            pred = predict(models[name], img[None])[0].argmax(0)
            axes[row, col].imshow(CLASS_COLOURS[pred])
            _clean(axes[row, col], title)
    fig.tight_layout(h_pad=2.0)
    fig.savefig(outdir / "multiclass.png", dpi=130)
    plt.close(fig)


def save_tracks_figures(road_examples, change_examples, outdir: Path):
    """sinuous.png (experiment E) and dunes.png (experiment F)."""
    outdir.mkdir(parents=True, exist_ok=True)

    imgs, gts, preds = road_examples
    names = (("hough", "Hough"), ("radon", "local Radon"), ("p4unet", "p4-equivariant U-Net"))
    fig, axes = plt.subplots(len(imgs), 5, figsize=(12.5, 2.7 * len(imgs)))
    for r, (img, gt) in enumerate(zip(imgs, gts)):
        axes[r, 0].imshow(rgb(img))
        _clean(axes[r, 0], "RGB: sinuous road")
        axes[r, 1].imshow(gt, cmap="gray")
        _clean(axes[r, 1], "ground truth")
        for c, (key, title) in enumerate(names, start=2):
            axes[r, c].imshow(preds[key][r], cmap="gray")
            _clean(axes[r, c], title)
    fig.tight_layout()
    fig.savefig(outdir / "sinuous.png", dpi=110)
    plt.close(fig)

    t0, t1, lab, maps, thr = change_examples
    fig, axes = plt.subplots(len(t0), 6, figsize=(15, 2.7 * len(t0)))
    for r in range(len(t0)):
        state = "new track" if lab[r].any() else "no new track"
        axes[r, 0].imshow(rgb(t0[r]))
        _clean(axes[r, 0], "date 1")
        axes[r, 1].imshow(rgb(t1[r]))
        _clean(axes[r, 1], f"date 2 ({state})")
        axes[r, 2].imshow(maps["naive difference"][r] >= thr["naive difference"], cmap="gray")
        _clean(axes[r, 2], "naive difference")
        axes[r, 3].imshow(maps["radon"][r] >= thr["radon"], cmap="gray")
        _clean(axes[r, 3], "local Radon")
        axes[r, 4].imshow(maps["p4unet"][r] >= thr["p4unet"], cmap="gray")
        _clean(axes[r, 4], "p4 change network")
        axes[r, 5].imshow(lab[r], cmap="gray")
        _clean(axes[r, 5], "ground truth (new passage)")
    fig.tight_layout()
    fig.savefig(outdir / "dunes.png", dpi=110)
    plt.close(fig)
