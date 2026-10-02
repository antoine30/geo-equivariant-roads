"""Evaluation metrics: pixel-level (relaxed F1, IoU) and topology-level (APLS-lite, components).

Pixel metrics say whether the right pixels are labelled road; they barely move when a 2-pixel
gap cuts a road in two. Topology metrics capture exactly that, which is what matters for
routing and connectivity analysis.
"""
from __future__ import annotations

import networkx as nx
import numpy as np
from scipy import ndimage


def iou(pred: np.ndarray, gt: np.ndarray) -> float:
    pred, gt = pred.astype(bool), gt.astype(bool)
    union = (pred | gt).sum()
    return float((pred & gt).sum() / union) if union else 1.0


def relaxed_f1(pred: np.ndarray, gt: np.ndarray, tolerance: int = 1) -> dict:
    """Precision/recall with a pixel tolerance (standard for thin structures such as roads)."""
    pred, gt = pred.astype(bool), gt.astype(bool)
    dist_to_gt = ndimage.distance_transform_edt(~gt)
    dist_to_pred = ndimage.distance_transform_edt(~pred)
    precision = float((dist_to_gt[pred] <= tolerance).mean()) if pred.any() else 0.0
    recall = float((dist_to_pred[gt] <= tolerance).mean()) if gt.any() else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def _graph_points(G: nx.Graph, spacing: float):
    """Nodes plus points injected every `spacing` px along edges, as (src, dst, s, L, xy)."""
    pts = []
    for u, v, d in G.edges(data=True):
        src = d["src"]
        dst = v if src == u else u
        path, L = d["path"], d["length"]
        cum = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))])
        for s in np.arange(0.0, L + 1e-9, spacing):
            i = int(np.searchsorted(cum, s))
            pts.append((src, dst, float(cum[min(i, len(cum) - 1)]), L, path[min(i, len(path) - 1)]))
        pts.append((src, dst, L, L, path[-1]))
    return pts


class _Snapper:
    """Snaps an (x, y) location to the nearest point on any edge of a graph."""

    def __init__(self, G: nx.Graph):
        self.rows = []
        for u, v, d in G.edges(data=True):
            src = d["src"]
            dst = v if src == u else u
            path = d["path"]
            cum = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))])
            for xy, s in zip(path, cum):
                self.rows.append((src, dst, float(s), d["length"], xy))
        self.xy = np.array([r[4] for r in self.rows]).reshape(-1, 2)

    def __call__(self, xy, snap):
        if not len(self.rows):
            return None
        dist = np.linalg.norm(self.xy - xy, axis=1)
        i = int(np.argmin(dist))
        return self.rows[i] if dist[i] <= snap else None


def _path_len(G, p, q, cache) -> float:
    """Shortest path length between two points lying on edges of G."""
    def sssp(n):
        if n not in cache:
            cache[n] = nx.single_source_dijkstra_path_length(G, n, weight="length")
        return cache[n]

    (u1, v1, s1, L1, _), (u2, v2, s2, L2, _) = p, q
    best = abs(s1 - s2) if (u1, v1) == (u2, v2) else np.inf
    if (u1, v1) == (v2, u2):
        best = abs(s1 - (L2 - s2))
    for a, da in ((u1, s1), (v1, L1 - s1)):
        lengths = sssp(a)
        for b, db in ((u2, s2), (v2, L2 - s2)):
            if b in lengths:
                best = min(best, da + lengths[b] + db)
    return best


def _apls_one_way(G_ref: nx.Graph, G_cmp: nx.Graph, snap: float, spacing: float, n_pairs: int, rng) -> float:
    pts = _graph_points(G_ref, spacing)
    if len(pts) < 2:
        return 1.0
    snapper = _Snapper(G_cmp)
    cache_ref, cache_cmp, penalties = {}, {}, []
    for _ in range(n_pairs):
        i, j = rng.choice(len(pts), size=2, replace=False)
        L = _path_len(G_ref, pts[i], pts[j], cache_ref)
        if not np.isfinite(L) or L < spacing:
            continue
        a, b = snapper(pts[i][4], snap), snapper(pts[j][4], snap)
        if a is None or b is None:
            penalties.append(1.0)
            continue
        L2 = _path_len(G_cmp, a, b, cache_cmp)
        penalties.append(min(1.0, abs(L - L2) / L) if np.isfinite(L2) else 1.0)
    return 1.0 - float(np.mean(penalties)) if penalties else 1.0


def apls_components(G_gt: nx.Graph, G_pred: nx.Graph, snap: float = 4.0, spacing: float = 10.0,
                    n_pairs: int = 300, seed: int = 0) -> dict:
    """Both directions of APLS and their harmonic mean.

    completeness: routes of the true network that the extracted network reproduces (GT -> pred);
    correctness : routes of the extracted network that really exist (pred -> GT). False links
                  create phantom routes and lower this term.
    """
    a = _apls_one_way(G_gt, G_pred, snap, spacing, n_pairs, np.random.default_rng(seed))
    b = _apls_one_way(G_pred, G_gt, snap, spacing, n_pairs, np.random.default_rng(seed + 1))
    return {"completeness": a, "correctness": b, "apls": 2 * a * b / (a + b) if a + b else 0.0}


def apls_lite(G_gt: nx.Graph, G_pred: nx.Graph, snap: float = 4.0, spacing: float = 10.0,
              n_pairs: int = 300, seed: int = 0) -> float:
    """Average Path Length Similarity (APLS, SpaceNet), simplified implementation.

    Control points are graph nodes plus points injected every `spacing` px (100 m at 10 m GSD)
    along edges. Pairs of control points are sampled in one graph, snapped to the nearest point
    of the other graph (within `snap` px), and their shortest-path lengths compared; a missing
    route costs the maximum penalty. Computed both ways and combined by harmonic mean, as in APLS.
    Simplification: random pair sampling instead of all pairs.
    """
    rng = np.random.default_rng(seed)
    a = _apls_one_way(G_gt, G_pred, snap, spacing, n_pairs, rng)
    b = _apls_one_way(G_pred, G_gt, snap, spacing, n_pairs, rng)
    return 2 * a * b / (a + b) if a + b else 0.0


def n_components(G: nx.Graph) -> int:
    return nx.number_connected_components(G) if len(G) else 0
