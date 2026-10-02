import networkx as nx
import numpy as np
import torch

from eqroads.export import graph_to_geojson
from eqroads.gnn import GapGNN, to_tensors, undirected_logits
from eqroads.graph import (
    EDGE_FEATURES,
    NODE_FEATURES,
    build_features,
    candidate_labels,
    extract_graph,
    find_candidates,
    materialise_candidates,
    skeleton_to_graph,
)
from eqroads.metrics import apls_components, apls_lite, n_components, relaxed_f1


def cross(size=41):
    sk = np.zeros((size, size), bool)
    sk[size // 2, 5:-5] = True
    sk[5:-5, size // 2] = True
    return sk


def test_cross_skeleton_gives_one_junction_four_branches():
    G = skeleton_to_graph(cross())
    degrees = sorted(d for _, d in G.degree())
    assert degrees == [1, 1, 1, 1, 4]
    assert all(abs(d["length"] - 15) <= 1.5 for *_, d in G.edges(data=True))


def gapped_road():
    """A horizontal road with a gap, plus the full ground-truth road."""
    gt = np.zeros((48, 48), np.uint8)
    gt[24, 4:44] = 1
    prob = gt.astype(float)
    prob[24, 20:27] = 0.1  # 7-pixel gap, e.g. under tree canopy
    return prob, gt


def test_gap_produces_a_positive_candidate():
    prob, gt = gapped_road()
    G = extract_graph(prob)
    assert n_components(G) == 2
    G2, pairs = materialise_candidates(G, find_candidates(G, max_dist=15))
    labels = candidate_labels(G2, pairs, gt)
    assert labels.sum() >= 1


def rotate_graph(G, size):
    """Exact image of the graph under np.rot90 (90 degrees counter-clockwise)."""
    def rot(rc):
        rc = np.atleast_2d(rc)
        return np.stack([size - 1 - rc[:, 1], rc[:, 0]], axis=1)

    R = nx.Graph()
    for n, d in G.nodes(data=True):
        R.add_node(n, **{**d, "rc": rot(d["rc"])[0]})
    for u, v, d in G.edges(data=True):
        R.add_edge(u, v, **{**d, "path": rot(d["path"])})
    return R


def test_features_are_rotation_invariant():
    rng = np.random.default_rng(0)
    from eqroads.synthetic import TileConfig, make_tile

    _, gt = make_tile(rng, TileConfig(size=64))
    prob = gt.astype(float) * rng.uniform(0.6, 1.0, gt.shape)
    prob[20:25, :] = 0.05
    G = extract_graph(prob)
    G2, pairs = materialise_candidates(G, find_candidates(G))
    _, x, _, ea, _ = build_features(G2, pairs, prob)

    R2 = rotate_graph(G2, 64)
    _, xr, _, ear, _ = build_features(R2, pairs, np.rot90(prob).copy())
    assert x.shape[1] == len(NODE_FEATURES) and ea.shape[1] == len(EDGE_FEATURES)
    np.testing.assert_allclose(x, xr, atol=1e-5)
    np.testing.assert_allclose(ea, ear, atol=1e-5)


def test_gnn_forward_shapes_and_symmetry():
    prob, _ = gapped_road()
    G = extract_graph(prob)
    G2, pairs = materialise_candidates(G, find_candidates(G))
    x, ei, ea = to_tensors(*build_features(G2, pairs, prob)[:4])
    model = GapGNN(x.shape[1], ea.shape[1]).eval()
    with torch.no_grad():
        logits = model(x, ei, ea)
    assert logits.shape[0] == ei.shape[1]
    assert undirected_logits(logits).shape[0] == ei.shape[1] // 2


def test_metrics():
    prob, gt = gapped_road()
    G_gt = extract_graph(gt.astype(float))
    G_cut = extract_graph(prob)
    assert apls_lite(G_gt, G_gt) == 1.0
    assert apls_lite(G_gt, G_cut) < 1.0
    assert relaxed_f1(gt, gt)["f1"] == 1.0
    # A gap hurts completeness (true routes are broken), not correctness (no phantom routes).
    c = apls_components(G_gt, G_cut)
    assert c["completeness"] < c["correctness"]


def test_geojson_export_is_georeferenced():
    G = extract_graph(gapped_road()[1].astype(float))
    # 10 m UTM zone 40N grid near Abu Dhabi (Sentinel-2 tiles over the UAE use EPSG:32640).
    transform = (10.0, 0.0, 250000.0, 0.0, -10.0, 2700000.0)
    fc = graph_to_geojson(G, np.ones((48, 48)), transform, "EPSG:32640", {"source_item": "test"})
    lon, lat = fc["features"][0]["geometry"]["coordinates"][0]
    assert 54.0 < lon < 55.0 and 24.0 < lat < 25.0
    assert fc["features"][0]["properties"]["status"] == "detected"
    assert abs(fc["features"][0]["properties"]["length_m"] - 390) < 15
