"""Reproduce the two experiments of the README on synthetic Sentinel-2-like tiles (CPU, ~15 min).

A. Orientation generalisation: train a p4-equivariant U-Net and a standard U-Net (same parameter
   budget) on roads running roughly east-west only, test on east-west and on north-south roads.
B. Topology repair: close segmentation gaps with the message-passing GNN, compared with no
   repair and with a classic distance/alignment rule; scored with APLS-lite.

    python scripts/run_experiment.py            # full run
    python scripts/run_experiment.py --quick    # smoke test (~1 min)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from eqroads.export import graph_to_geojson, write_geojson
from eqroads.graph import extract_graph
from eqroads.metrics import apls_components, n_components, relaxed_f1
from eqroads.segnet import build_model, count_parameters
from eqroads.synthetic import TileConfig, make_batch
from eqroads.train import (
    gnn_scores,
    heuristic_scores,
    make_example,
    predict,
    refine,
    train_gnn,
    train_segmentation,
)

ROOT = Path(__file__).resolve().parents[1]
EW = (-20.0, 20.0)  # training roads: within 20 degrees of east-west


def seg_scores(prob, masks):
    f1 = [relaxed_f1(p >= 0.5, m)["f1"] for p, m in zip(prob, masks)]
    return float(np.mean(f1))


def rot(arr, k):
    return np.rot90(arr, k, axes=(-2, -1)).copy()


def random_rot(images, masks, seed):
    ks = np.random.default_rng(seed).integers(0, 4, len(images))
    return (np.stack([rot(i, k) for i, k in zip(images, ks)]), np.stack([rot(m, k) for m, k in zip(masks, ks)]))


def link_prf(examples, accepts):
    tp = fp = fn = 0
    for ex, acc in zip(examples, accepts):
        if not ex.pairs:
            continue
        y = ex.y.numpy().astype(bool)
        tp += int((acc & y).sum())
        fp += int((acc & ~y).sum())
        fn += int((~acc & y).sum())
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    return {"precision": p, "recall": r, "f1": 2 * p * r / max(p + r, 1e-9)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--reuse", action="store_true", help="load segmentation weights from results/ if present")
    args = ap.parse_args()
    steps = 40 if args.quick else args.steps
    n_test, n_gnn_train, n_gnn_test = (8, 20, 10) if args.quick else (60, 200, 80)
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    torch.manual_seed(0)
    torch.set_num_threads(max(1, torch.get_num_threads()))
    results = {}

    # ---------------------------------------------------------------- A. orientation
    print("== A. Orientation generalisation")
    train_cfg = TileConfig(size=64, angle_range=EW)
    test_cfg = TileConfig(size=96, angle_range=EW)
    test_img, test_mask = make_batch(10_000, n_test, test_cfg)
    any_img, any_mask = make_batch(20_000, n_test, TileConfig(size=96))
    models = {}
    for name in ("unet", "p4unet"):
        torch.manual_seed(0)
        model = build_model(name)
        ckpt = out / f"{name}.pt"
        if args.reuse and ckpt.exists():
            model.load_state_dict(torch.load(ckpt))
            models[name] = model.eval()
            print(f"loaded {name} from {ckpt}")
        else:
            print(f"training {name} ({count_parameters(model):,} parameters)")
            models[name] = train_segmentation(model, lambda s: make_batch(s, 8, train_cfg), steps)
            torch.save(model.state_dict(), ckpt)
        results[name] = {
            "parameters": count_parameters(model),
            "f1_east_west": seg_scores(predict(model, test_img), test_mask),
            "f1_north_south": seg_scores(predict(model, rot(test_img, 1)), rot(test_mask, 1)),
            "f1_any_heading": seg_scores(predict(model, any_img), any_mask),
        }
        print("  ", {k: round(v, 3) if isinstance(v, float) else v for k, v in results[name].items()})

    # ---------------------------------------------------------------- B. topology
    print("== B. Topology repair with message passing")
    # Harder occlusions than in segmentation training (rows of trees, sand drifts, shadows):
    # the network can no longer bridge them alone, so its output has gaps to repair.
    seg = models["p4unet"]
    hard_cfg = TileConfig(size=96, angle_range=EW, n_occlusions=(3, 6), occlusion_radius=(4, 7))
    tr_img, tr_mask = random_rot(*make_batch(30_000, n_gnn_train, hard_cfg), seed=1)
    te_img, te_mask = random_rot(*make_batch(40_000, n_gnn_test, hard_cfg), seed=2)
    tr_prob, te_prob = predict(seg, tr_img), predict(seg, te_img)
    train_ex = [e for e in (make_example(p, m) for p, m in zip(tr_prob, tr_mask)) if e is not None]
    test_ex, test_gt, test_idx = [], [], []
    for j, (p, m) in enumerate(zip(te_prob, te_mask)):
        e = make_example(p, m)
        if e is not None:
            test_ex.append(e)
            test_gt.append(m)
            test_idx.append(j)
    n_cand = sum(len(e.pairs) for e in train_ex)
    n_pos = int(sum(e.y.sum() for e in train_ex if e.pairs))
    print(f"  {len(train_ex)} training graphs, {n_cand} candidate links ({n_pos} true)")
    results["topology_setup"] = {
        "segmentation_f1_on_hard_tiles": seg_scores(te_prob, te_mask),
        "train_graphs": len(train_ex), "train_candidates": n_cand, "train_true_links": n_pos,
        "test_graphs": len(test_ex),
    }
    print(f"  segmentation F1 on hard-occlusion tiles: {results['topology_setup']['segmentation_f1_on_hard_tiles']:.3f}")

    gnn = train_gnn(train_ex, epochs=10 if args.quick else 40)
    torch.save(gnn.state_dict(), out / "gap_gnn.pt")

    # Both methods get their acceptance threshold tuned on the training graphs (fair comparison).
    def tune(score_fn):
        best_t, best_f1 = 0.5, -1.0
        for t in np.linspace(0.05, 0.95, 19):
            f1 = link_prf(train_ex, [refine(e, score_fn(e), t)[1] for e in train_ex])["f1"]
            if f1 > best_f1:
                best_t, best_f1 = float(t), f1
        return best_t

    best_t = tune(heuristic_scores)
    gnn_t = tune(lambda e: gnn_scores(gnn, e))

    methods = {
        "no repair": lambda e: (e.G, np.zeros(len(e.pairs), bool)),
        f"rule (dist x alignment, t={best_t:.2f})": lambda e: refine(e, heuristic_scores(e), best_t),
        f"message-passing GNN (t={gnn_t:.2f})": lambda e: refine(e, gnn_scores(gnn, e), gnn_t),
    }
    gt_graphs = [extract_graph(m.astype(float)) for m in test_gt]
    results["topology"] = {}
    for name, fn in methods.items():
        outs = [fn(e) for e in test_ex]
        comps = [apls_components(g, o[0]) for g, o in zip(gt_graphs, outs)]
        apls = [c["apls"] for c in comps]
        comp_err = [abs(n_components(o[0]) - n_components(g)) for g, o in zip(gt_graphs, outs)]
        results["topology"][name] = {
            "apls_lite": float(np.mean(apls)),
            "apls_completeness": float(np.mean([c["completeness"] for c in comps])),
            "apls_correctness": float(np.mean([c["correctness"] for c in comps])),
            "component_error": float(np.mean(comp_err)),
            **({"links": link_prf(test_ex, [o[1] for o in outs])} if name != "no repair" else {}),
        }
        r = results["topology"][name]
        print(f"  {name:40s} APLS-lite {r['apls_lite']:.3f} (completeness {r['apls_completeness']:.3f}, "
              f"correctness {r['apls_correctness']:.3f})   |components - gt| {np.mean(comp_err):.2f}")

    (out / "results.json").write_text(json.dumps(results, indent=2))

    # ---------------------------------------------------------------- figure + GeoJSON sample
    try:
        from eqroads.viz import save_figures
        save_figures(models, test_img, test_mask, test_ex, te_img[test_idx], te_prob[test_idx], test_gt, gnn,
                     ROOT / "docs", threshold=gnn_t)
    except ImportError as err:  # matplotlib is optional
        print("figures skipped:", err)

    i = int(np.argmax([len(e.pairs) for e in test_ex]))
    ex = test_ex[i]
    scores = gnn_scores(gnn, ex)
    G_ref, accept = refine(ex, scores, gnn_t)
    conf = {pair: float(s) for pair, s, a in zip(ex.pairs, scores, accept) if a}
    # Illustrative georeferencing: 10 m UTM 40N grid near Abu Dhabi (synthetic tile).
    fc = graph_to_geojson(G_ref, te_prob[test_idx[i]], (10.0, 0.0, 255000.0, 0.0, -10.0, 2700000.0), "EPSG:32640",
                      {"source_item": "synthetic-demo", "model_version": "p4unet+gapgnn-0.1"}, conf)
    write_geojson(fc, ROOT / "docs" / "sample_output.geojson")
    print("done ->", out / "results.json")


if __name__ == "__main__":
    main()
