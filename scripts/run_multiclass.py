"""Experiment C: joint segmentation of roads and built-up areas on synthetic Sentinel-2-like tiles.

Same protocol as experiment A: both networks are trained on scenes whose roads run roughly
east-west, then tested on the same scenes and on the scenes rotated by 90 degrees.

    python scripts/run_multiclass.py            # ~12 min on CPU
    python scripts/run_multiclass.py --quick
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from eqroads.export import class_polygons_to_geojson, write_class_geotiff, write_geojson
from eqroads.metrics import iou, relaxed_f1
from eqroads.segnet import build_model, count_parameters
from eqroads.synthetic import BUILDING, CLASS_NAMES, ROAD, TileConfig, make_batch
from eqroads.train import predict, train_segmentation

ROOT = Path(__file__).resolve().parents[1]
EW = (-20.0, 20.0)


def rot(arr, k):
    return np.rot90(arr, k, axes=(-2, -1)).copy()


def scores(probs, masks):
    pred = probs.argmax(1)
    return {
        "road_relaxed_f1": float(np.mean([relaxed_f1(p == ROAD, m == ROAD)["f1"] for p, m in zip(pred, masks)])),
        "built_up_iou": iou(pred == BUILDING, masks == BUILDING),
        "built_up_f1": relaxed_f1(pred == BUILDING, masks == BUILDING, tolerance=0)["f1"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--steps", type=int, default=800)
    args = ap.parse_args()
    steps, n_test = (40, 8) if args.quick else (args.steps, 60)
    out = ROOT / "results"
    out.mkdir(exist_ok=True)

    train_cfg = TileConfig(size=64, angle_range=EW, n_settlements=(1, 2))
    test_cfg = TileConfig(size=96, angle_range=EW, n_settlements=(1, 3))
    test_img, test_mask = make_batch(50_000, n_test, test_cfg)

    results, models = {}, {}
    for name in ("unet", "p4unet"):
        torch.manual_seed(0)
        model = build_model(name, out_channels=3)
        print(f"training {name} ({count_parameters(model):,} parameters), 3 classes")
        models[name] = train_segmentation(model, lambda s: make_batch(s, 8, train_cfg), steps)
        torch.save(model.state_dict(), out / f"{name}_3class.pt")
        results[name] = {
            "parameters": count_parameters(model),
            "east_west": scores(predict(model, test_img), test_mask),
            "north_south": scores(predict(model, rot(test_img, 1)), rot(test_mask, 1)),
        }
        print("  ", json.dumps(results[name]))
    (out / "results_multiclass.json").write_text(json.dumps(results, indent=2))

    try:
        from eqroads.viz import save_multiclass_figure
        save_multiclass_figure(models, test_img, test_mask, ROOT / "docs")
    except ImportError as err:
        print("figure skipped:", err)

    # Sample deliverables: class GeoTIFF + built-up polygons (illustrative UTM 40N georeferencing).
    probs = predict(models["p4unet"], test_img[:1])[0]
    classes = probs.argmax(0).astype(np.uint8)
    transform, crs = (10.0, 0.0, 255000.0, 0.0, -10.0, 2700000.0), "EPSG:32640"
    write_class_geotiff(ROOT / "docs" / "sample_classes.tif", classes, transform, crs, CLASS_NAMES)
    fc = class_polygons_to_geojson(classes, probs, BUILDING, "built-up", transform, crs,
                                   {"source_item": "synthetic-demo", "model_version": "p4unet-3class-0.2"})
    write_geojson(fc, ROOT / "docs" / "sample_built_up.geojson")
    print("done ->", out / "results_multiclass.json")


if __name__ == "__main__":
    main()
