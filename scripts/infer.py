"""Run the pipeline on a tile from fetch_s2.py: road network and, for multi-class models, built-up areas.

    python scripts/infer.py --tile data/abu_dhabi.npz --weights results/real --out outputs/abu_dhabi

Outputs (open them in QGIS or geojson.io):
    <out>_roads.geojson      road segments, `status` detected / inferred (gap closed by the GNN)
    <out>_built_up.geojson   built-up polygons with area and confidence (multi-class model only)
    <out>_classes.tif        class map GeoTIFF with colour table (multi-class model only)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from eqroads.data_s2 import load_tile
from eqroads.export import class_polygons_to_geojson, graph_to_geojson, write_class_geotiff, write_geojson
from eqroads.gnn import GapGNN
from eqroads.graph import EDGE_FEATURES, NODE_FEATURES, extract_graph
from eqroads.metrics import apls_lite, iou, relaxed_f1
from eqroads.segnet import build_model
from eqroads.synthetic import BUILDING, CLASS_NAMES
from eqroads.train import gnn_scores, make_example, predict_large, refine

MODEL_VERSION = "p4unet+gapgnn-0.2"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tile", type=Path, required=True)
    ap.add_argument("--weights", type=Path, required=True, help="directory with p4unet.pt and gap_gnn.pt")
    ap.add_argument("--model", default="p4unet", choices=["p4unet", "unet"])
    ap.add_argument("--threshold", type=float, default=0.5, help="GNN acceptance threshold")
    ap.add_argument("--out", type=Path, required=True, help="output prefix")
    args = ap.parse_args()

    t = load_tile(args.tile)
    state = torch.load(args.weights / f"{args.model}.pt", map_location="cpu")
    n_out = state["head.weight"].shape[0]
    seg = build_model(args.model, out_channels=n_out)
    seg.load_state_dict(state)
    transform, crs = tuple(t["transform"]), str(t["crs"])
    provenance = {"source_item": str(t["item_id"]), "model_version": MODEL_VERSION}

    probs = predict_large(seg, t["image"])
    if n_out == 1:
        road_prob = probs * t["valid"]
    else:
        road_prob = probs[1] * t["valid"]
        classes = np.argmax(probs, axis=0).astype(np.uint8)
        classes[~t["valid"].astype(bool)] = 255  # clouds / no data
        write_class_geotiff(f"{args.out}_classes.tif", classes, transform, crs, CLASS_NAMES)
        fc = class_polygons_to_geojson(classes, probs, BUILDING, "built-up", transform, crs, provenance)
        write_geojson(fc, f"{args.out}_built_up.geojson")
        print(f"built-up: {len(fc['features'])} polygons, {(classes == BUILDING).mean():.2%} of the scene")
        if "classes" in t:
            ref = t["classes"]
            ok = t["valid"].astype(bool)
            print(f"  vs OSM: built-up IoU {iou((classes == BUILDING) & ok, (ref == BUILDING) & ok):.3f}, "
                  f"road IoU {iou((classes == 1) & ok, (ref == 1) & ok):.3f}")

    ex = make_example(road_prob, t["mask"])
    if ex is None:
        raise SystemExit("no road detected in this tile")
    G, accept = road_graph(ex, args)
    if t["mask"].any():  # OpenStreetMap labels are in the tile: report quality against them
        G_osm = extract_graph(t["mask"].astype(float))
        f1 = relaxed_f1(road_prob >= 0.5, t["mask"], tolerance=1)["f1"]
        print(f"  vs OSM: road relaxed F1 {f1:.3f}, APLS-lite before repair {apls_lite(G_osm, ex.G):.3f}, "
              f"after {apls_lite(G_osm, G):.3f}")
    write_geojson(graph_to_geojson(G, road_prob, transform, crs, provenance, accept),
                  f"{args.out}_roads.geojson")
    print("wrote", f"{args.out}_*")


def road_graph(ex, args):
    """Road graph, with gaps closed by the GNN if its weights are available."""
    gnn_path = args.weights / "gap_gnn.pt"
    if not gnn_path.exists() or not ex.pairs:
        return ex.G, {}
    gnn = GapGNN(len(NODE_FEATURES), len(EDGE_FEATURES))
    gnn.load_state_dict(torch.load(gnn_path, map_location="cpu"))
    scores = gnn_scores(gnn, ex)
    G, accept = refine(ex, scores, args.threshold)
    print(f"roads: {G.number_of_edges()} segments, {int(accept.sum())} gaps closed / {len(ex.pairs)} candidates")
    return G, {pair: float(s) for pair, s, a in zip(ex.pairs, scores, accept) if a}


if __name__ == "__main__":
    main()
