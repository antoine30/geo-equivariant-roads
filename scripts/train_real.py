"""Train the segmentation network and the gap-closing GNN on real tiles from fetch_s2.py.

    python scripts/train_real.py --tiles data/*.npz --out results/real --steps 4000
    python scripts/train_real.py --multiclass --tiles data/*.npz --out results/real   # roads + built-up

Crops overlapping clouds / no data are down-weighted through the SCL-derived `valid` mask.
A GPU is recommended for more than a few thousand steps (the script uses it when available).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from eqroads.data_s2 import load_tile
from eqroads.segnet import build_model, count_parameters
from eqroads.train import make_example, predict, train_gnn, train_segmentation


def crop_sampler(tiles, size: int, batch: int, seed: int, augment: bool, key: str = "mask"):
    rng = np.random.default_rng(seed)

    def sample(_step):
        imgs, masks, valids = [], [], []
        while len(imgs) < batch:
            t = tiles[rng.integers(len(tiles))]
            H, W = t[key].shape
            r, c = rng.integers(0, H - size), rng.integers(0, W - size)
            valid = t["valid"][r: r + size, c: c + size]
            if valid.mean() < 0.8:
                continue
            img = t["image"][:, r: r + size, c: c + size]
            mask = t[key][r: r + size, c: c + size]
            if augment:  # redundant for the p4 model on 90-degree rotations, useful for the baseline
                k = rng.integers(4)
                img, mask, valid = (np.rot90(a, k, axes=(-2, -1)) for a in (img, mask, valid))
            imgs.append(np.ascontiguousarray(img))
            masks.append(np.ascontiguousarray(mask))
            valids.append(np.ascontiguousarray(valid))
        return np.stack(imgs), np.stack(masks), np.stack(valids)

    return sample


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles", nargs="+", type=Path, required=True)
    ap.add_argument("--model", default="p4unet", choices=["p4unet", "unet"])
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--crop", type=int, default=128)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--augment", action="store_true")
    ap.add_argument("--multiclass", action="store_true",
                    help="background / road / built-up (tiles fetched with fetch_s2.py --buildings)")
    ap.add_argument("--out", type=Path, default=Path("results/real"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    tiles = [load_tile(p) for p in args.tiles]
    key = "classes" if args.multiclass else "mask"
    if any(key not in t for t in tiles):
        raise SystemExit(f"tiles lack '{key}': fetch them with fetch_s2.py --buildings")
    torch.manual_seed(0)
    model = build_model(args.model, out_channels=3 if args.multiclass else 1)
    print(f"{args.model}: {count_parameters(model):,} parameters, {len(tiles)} tiles")
    train_segmentation(model, crop_sampler(tiles, args.crop, args.batch, 0, args.augment, key), args.steps)
    torch.save(model.state_dict(), args.out / f"{args.model}.pt")

    # Gap-closing GNN trained on the segmentation output against OSM topology.
    sample = crop_sampler(tiles, 256, 1, 1, augment=False)
    examples = []
    for i in range(300):
        img, mask, _ = sample(i)
        prob = predict(model, img)[0]
        road_prob = prob if prob.ndim == 2 else prob[1]
        ex = make_example(road_prob, (mask[0] > 0).astype(np.uint8))
        if ex is not None and ex.pairs:
            examples.append(ex)
    print(f"GNN: {len(examples)} graphs, {sum(len(e.pairs) for e in examples)} candidate links")
    gnn = train_gnn(examples)
    torch.save(gnn.state_dict(), args.out / "gap_gnn.pt")
    print("saved to", args.out)


if __name__ == "__main__":
    main()
