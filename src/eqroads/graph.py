"""From a road probability map to a road graph, plus gap-bridging candidates for the GNN.

Pipeline:
    probability map -> threshold -> skeleton -> pixel chains -> networkx graph
    (nodes = junctions / endpoints, edges = road segments with their pixel path)
    -> candidate links from each dead end to nearby nodes or road segments
    -> E(2)-invariant features (distances, cosines, probabilities) and labels.

Every geometric feature depends only on distances and angles between points, never on
absolute coordinates or headings, so the GNN that consumes them is invariant to rotations
and translations of the scene (checked in tests/test_graph.py).
"""
from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np
from scipy import ndimage
from skimage.morphology import remove_small_objects, skeletonize

_OFFSETS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

NODE_FEATURES = ("is_endpoint", "is_chain", "is_junction", "mean_prob", "log_component_len")
EDGE_FEATURES = ("is_candidate", "length", "p_mean", "p_min", "cos_src", "cos_dst",
                 "dst_is_endpoint", "tangent_cos", "graph_detour")


# --------------------------------------------------------------------------- skeleton graph
def skeletonize_prob(prob: np.ndarray, threshold: float = 0.5, min_size: int = 8) -> np.ndarray:
    # connectivity=2 (8-neighbourhood): 1-px diagonal roads are 8-connected, not 4-connected.
    mask = remove_small_objects(prob >= threshold, max_size=min_size - 1, connectivity=2)
    return skeletonize(mask)


def _neighbours(sk, r, c):
    H, W = sk.shape
    for dr, dc in _OFFSETS:
        rr, cc = r + dr, c + dc
        if 0 <= rr < H and 0 <= cc < W and sk[rr, cc]:
            yield rr, cc


def _path_length(path: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum()) if len(path) > 1 else 0.0


def _add_edge(G, u, v, path):
    if u == v:
        return
    length = _path_length(path)
    if G.has_edge(u, v) and G.edges[u, v]["length"] <= length:
        return
    G.add_edge(u, v, path=path, src=u, length=length)


def skeleton_to_graph(sk: np.ndarray) -> nx.Graph:
    """Convert a one-pixel-wide skeleton into a graph of junctions/endpoints and segments.

    Node attribute `rc`: (row, col) position. Edge attributes: `path` (K, 2) pixel chain
    oriented from node `src`, `length` in pixels. Isolated closed loops are ignored.
    """
    sk = sk.astype(bool)
    kernel = np.ones((3, 3), dtype=int)
    kernel[1, 1] = 0
    deg = ndimage.convolve(sk.astype(int), kernel, mode="constant") * sk
    key = sk & (deg != 2)
    labels, n = ndimage.label(key, structure=np.ones((3, 3)))

    G = nx.Graph()
    if n == 0:
        return G
    centroids = ndimage.center_of_mass(key, labels, range(1, n + 1))
    for i, rc in enumerate(centroids, start=1):
        G.add_node(i, rc=np.array(rc, dtype=float))

    visited = np.zeros_like(sk)
    for r, c in np.argwhere(key):
        u = labels[r, c]
        for q in _neighbours(sk, r, c):
            if key[q]:
                if labels[q] != u:
                    _add_edge(G, u, labels[q], np.array([(r, c), q], dtype=float))
                continue
            if visited[q]:
                continue
            path, prev, cur = [(r, c), q], (r, c), q
            visited[q] = True
            end = None
            while True:
                nbrs = [p for p in _neighbours(sk, *cur) if p != prev]
                # Ignore pixels of the start junction while still leaving it (multi-pixel junctions).
                keys = [p for p in nbrs if key[p] and not (labels[p] == u and len(path) <= 3)]
                if keys:
                    end = keys[0]
                    path.append(end)
                    break
                nxt = [p for p in nbrs if not visited[p] and not key[p]]
                if not nxt:
                    break
                prev, cur = cur, nxt[0]
                visited[cur] = True
                path.append(cur)
            if end is not None:
                _add_edge(G, u, labels[end], np.array(path, dtype=float))
    return G


def oriented_path(G: nx.Graph, u, v) -> np.ndarray:
    """Pixel path of edge (u, v) oriented from u to v."""
    data = G.edges[u, v]
    return data["path"] if data["src"] == u else data["path"][::-1]


def _merge_degree2(G: nx.Graph) -> None:
    for n in [n for n in G if G.degree(n) == 2]:
        if n not in G or G.degree(n) != 2:
            continue
        a, b = list(G.neighbors(n))
        if a == b or G.has_edge(a, b):
            continue
        path = np.concatenate([oriented_path(G, a, n), oriented_path(G, n, b)[1:]])
        G.remove_node(n)
        _add_edge(G, a, b, path)


def prune_spurs(G: nx.Graph, min_length: float = 4.0) -> nx.Graph:
    """Remove short dead-end branches created by skeletonisation, then merge chain nodes."""
    G = G.copy()
    changed = True
    while changed:
        changed = False
        for n in [n for n in G if G.degree(n) == 1]:
            (m,) = list(G.neighbors(n))
            if G.degree(m) >= 3 and G.edges[n, m]["length"] < min_length:
                G.remove_node(n)
                changed = True
        _merge_degree2(G)
    G.remove_nodes_from([n for n in list(G) if G.degree(n) == 0])
    return G


def extract_graph(prob: np.ndarray, threshold: float = 0.5, spur_length: float = 4.0) -> nx.Graph:
    return prune_spurs(skeleton_to_graph(skeletonize_prob(prob, threshold)), spur_length)


# --------------------------------------------------------------------------- candidates
@dataclass
class Candidate:
    src: int                     # dead-end node
    dst: int | None              # existing node, or None if the target lies on an edge
    edge: tuple | None           # (u, v) oriented like the stored path, if target is on an edge
    idx: int | None              # index of the target pixel in that path
    dst_rc: np.ndarray
    dist: float


def endpoint_direction(G: nx.Graph, n, lookback: int = 5) -> np.ndarray:
    """Unit vector pointing out of a dead end (from the road towards the gap)."""
    (m,) = list(G.neighbors(n))
    path = oriented_path(G, n, m)
    ref = path[min(lookback, len(path) - 1)]
    d = G.nodes[n]["rc"] - ref
    norm = np.linalg.norm(d)
    return d / norm if norm > 0 else np.zeros(2)


def find_candidates(G: nx.Graph, max_dist: float = 15.0, k: int = 4, detour_ratio: float = 2.0):
    """Candidate links from each dead end to nearby nodes or road segments.

    A target is skipped if it is already reachable along the graph with a short detour
    (graph distance < detour_ratio * straight distance): linking it would add nothing.
    """
    cands = []
    for e in [n for n in G if G.degree(n) == 1]:
        p = G.nodes[e]["rc"]
        gd = nx.single_source_dijkstra_path_length(G, e, cutoff=detour_ratio * max_dist * 2, weight="length")
        found = []
        for n in G:
            if n == e:
                continue
            d = float(np.linalg.norm(G.nodes[n]["rc"] - p))
            if 1.5 <= d <= max_dist and gd.get(n, np.inf) >= detour_ratio * d:
                found.append(Candidate(e, n, None, None, G.nodes[n]["rc"].copy(), d))
        for u, v, data in G.edges(data=True):
            if e in (u, v):
                continue
            path = data["path"]
            if len(path) < 5:
                continue
            inner = path[2:-2]
            dists = np.linalg.norm(inner - p, axis=1)
            i = int(np.argmin(dists))
            d = float(dists[i])
            if not (1.5 <= d <= max_dist):
                continue
            if min(gd.get(u, np.inf), gd.get(v, np.inf)) < detour_ratio * d:
                continue
            src = data["src"]
            dst = v if src == u else u
            found.append(Candidate(e, None, (src, dst), i + 2, inner[i].copy(), d))
        found.sort(key=lambda c: c.dist)
        cands.extend(found[:k])

    # Endpoint-to-endpoint links are found twice (once from each side): keep one.
    seen, unique = set(), []
    for c in cands:
        key = ("n", frozenset((c.src, c.dst))) if c.dst is not None else ("e", c.src, c.edge, c.idx)
        if key not in seen:
            seen.add(key)
            unique.append(c)
    return unique


def materialise_candidates(G: nx.Graph, cands: list[Candidate]):
    """Split road edges at candidate anchor pixels so that every candidate joins two nodes.

    Returns a new graph and the list of candidate node pairs (src, dst).
    """
    G = G.copy()
    next_id = max(G.nodes) + 1 if len(G) else 1
    by_edge: dict[tuple, list[Candidate]] = {}
    for c in cands:
        if c.edge is not None:
            by_edge.setdefault(c.edge, []).append(c)

    anchor_of: dict[tuple, int] = {}
    for (a, b), group in by_edge.items():
        path = oriented_path(G, a, b)
        G.remove_edge(a, b)
        prev_node, prev_idx = a, 0
        for idx in sorted({c.idx for c in group}):
            G.add_node(next_id, rc=path[idx].astype(float), anchor=True)
            _add_edge(G, prev_node, next_id, path[prev_idx: idx + 1])
            anchor_of[(a, b, idx)] = next_id
            prev_node, prev_idx = next_id, idx
            next_id += 1
        _add_edge(G, prev_node, b, path[prev_idx:])

    pairs = []
    for c in cands:
        dst = c.dst if c.dst is not None else anchor_of[(*c.edge, c.idx)]
        pairs.append((c.src, dst))
    return G, pairs


# --------------------------------------------------------------------------- features
def _sample_line(arr: np.ndarray, a: np.ndarray, b: np.ndarray, trim: float = 1.0) -> np.ndarray:
    d = float(np.linalg.norm(b - a))
    n = max(int(np.ceil(d)) + 1, 2)
    t = np.linspace(0, 1, n)
    if d > 2 * trim:  # ignore the two ends, which sit on the detected road by construction
        inner = t[(t * d >= trim) & (t * d <= d - trim)]
        t = inner if len(inner) else np.array([0.5])
    pts = a[None] + t[:, None] * (b - a)[None]
    # Bilinear sampling: unlike nearest-pixel rounding (ties!), it commutes exactly with
    # 90-degree rotations of the grid, which keeps the features rotation-invariant.
    return ndimage.map_coordinates(arr.astype(float), pts.T, order=1, mode="nearest")


def _tangent(G, n) -> np.ndarray | None:
    """Local road direction at a chain (degree-2) node."""
    nb = list(G.neighbors(n))
    if len(nb) != 2:
        return None
    pa = oriented_path(G, n, nb[0])
    pb = oriented_path(G, n, nb[1])
    t = pa[min(3, len(pa) - 1)] - pb[min(3, len(pb) - 1)]
    norm = np.linalg.norm(t)
    return t / norm if norm > 0 else None


def build_features(G: nx.Graph, pairs, prob: np.ndarray, max_dist: float = 15.0):
    """Node and edge feature arrays for the GNN (all E(2)-invariant).

    Returns nodes (list of node ids), x (N, F_node), edges (list of (u, v)), edge_attr (E, F_edge),
    is_candidate (E,) bool. Road edges come first, then candidate edges in `pairs` order.
    """
    nodes = list(G.nodes)
    comp_len = {}
    for comp in nx.connected_components(G):
        total = sum(G.edges[e]["length"] for e in G.subgraph(comp).edges)
        comp_len.update({n: total for n in comp})

    x = []
    for n in nodes:
        deg = G.degree(n)
        probs = [_sample_line_path(prob, G.edges[n, m]["path"]).mean() for m in G.neighbors(n)]
        x.append([deg == 1, deg == 2, deg >= 3, np.mean(probs) if probs else 0.0,
                  np.log1p(comp_len[n]) / 5.0])
    x = np.asarray(x, dtype=np.float32)

    edges, attrs = [], []
    for u, v, data in G.edges(data=True):
        p = _sample_line_path(prob, data["path"])
        edges.append((u, v))
        attrs.append([0.0, min(data["length"] / max_dist, 3.0), p.mean(), p.min(), 0, 0, 0, 0, 0])

    for u, v in pairs:
        a, b = G.nodes[u]["rc"], G.nodes[v]["rc"]
        d = float(np.linalg.norm(b - a))
        unit = (b - a) / d
        p = _sample_line(prob, a, b)
        cos_src = float(endpoint_direction(G, u) @ unit) if G.degree(u) == 1 else 0.0
        dst_end = G.degree(v) == 1
        cos_dst = float(endpoint_direction(G, v) @ -unit) if dst_end else 0.0
        tan = _tangent(G, v)
        tangent_cos = abs(float(tan @ unit)) if tan is not None else 0.0
        try:
            gdist = nx.shortest_path_length(G, u, v, weight="length")
            detour = min(gdist / d, 5.0) / 5.0
        except nx.NetworkXNoPath:
            detour = 1.0
        edges.append((u, v))
        attrs.append([1.0, d / max_dist, p.mean(), p.min(), cos_src, cos_dst, float(dst_end),
                      tangent_cos, detour])

    edge_attr = np.asarray(attrs, dtype=np.float32).reshape(-1, len(EDGE_FEATURES))
    is_candidate = edge_attr[:, 0] > 0.5
    return nodes, x, edges, edge_attr, is_candidate


def _sample_line_path(arr: np.ndarray, path: np.ndarray) -> np.ndarray:
    rc = np.clip(np.rint(path).astype(int), 0, np.array(arr.shape) - 1)
    return arr[rc[:, 0], rc[:, 1]]


def candidate_labels(G: nx.Graph, pairs, gt_mask: np.ndarray, min_cover: float = 0.8) -> np.ndarray:
    """A candidate is a true link if the ground-truth road covers the straight segment."""
    gt = ndimage.binary_dilation(gt_mask.astype(bool), iterations=1).astype(float)
    return np.array([
        _sample_line(gt, G.nodes[u]["rc"], G.nodes[v]["rc"], trim=0.0).mean() >= min_cover
        for u, v in pairs
    ], dtype=np.float32)


def add_links(G: nx.Graph, pairs, accept) -> nx.Graph:
    """Graph with accepted candidate links added as straight `inferred` edges."""
    G = G.copy()
    for (u, v), ok in zip(pairs, accept):
        if ok and not G.has_edge(u, v):
            path = np.stack([G.nodes[u]["rc"], G.nodes[v]["rc"]])
            G.add_edge(u, v, path=path, src=u, length=_path_length(path), inferred=True)
    return G
