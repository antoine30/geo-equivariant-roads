"""Experiments E and F: sinuous roads, and new tracks across dunes (change detection).

E. Mapping sinuous roads: Hough and Radon assume straight lines; the network does not.
F. Monitoring passages: detect a faint new track between two dates across migrating dunes,
   comparing a naive difference threshold, Hough, a localised Radon transform (the classical
   ship-wake detector in SAR imagery) and a p4-equivariant change-detection network.

    python scripts/run_tracks.py           # ~25 min on CPU (trains two networks)
    python scripts/run_tracks.py --reuse   # reuse trained weights from results/
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from eqroads.dunes import DuneConfig, make_pairs
from eqroads.segnet import build_model, count_parameters
from eqroads.synthetic import TileConfig, make_batch
from eqroads.train import train_segmentation

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"

ROAD_TRAIN = TileConfig(size=64, curved_prob=0.5, dead_end_prob=0.2)
ROAD_TEST = TileConfig(size=96, curved_prob=1.0, n_roads=(1, 3), dead_end_prob=0.0)
DUNES_TRAIN = DuneConfig(size=64)
DUNES_TEST = DuneConfig(size=96)


def road_sampler(step):
    return make_batch(100_000 + step, 8, ROAD_TRAIN)


def change_sampler(step):
    t0, t1, label, _ = make_pairs(200_000 + step, 8, DUNES_TRAIN)
    return np.concatenate([t0, t1], axis=1), label


def get_model(name, in_channels, sampler, steps, reuse):
    torch.manual_seed(0)
    model = build_model("p4unet", in_channels=in_channels)
    ckpt = OUT / f"{name}.pt"
    if reuse and ckpt.exists():
        model.load_state_dict(torch.load(ckpt))
        print(f"loaded {ckpt}")
        return model.eval()
    print(f"training {name}: p4unet, {in_channels} input channels, {count_parameters(model):,} parameters")
    train_segmentation(model, sampler, steps)
    torch.save(model.state_dict(), ckpt)
    return model.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reuse", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--train-only", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    steps = 40 if args.quick else 800
    road_model = get_model("p4unet_sinuous", 4, road_sampler, steps, args.reuse)
    change_model = get_model("p4unet_change", 8, change_sampler, steps, args.reuse)
    if args.train_only:
        return
    from eqroads.tracks_eval import evaluate
    results = evaluate(road_model, change_model, quick=args.quick, figures_dir=ROOT / "docs")
    (OUT / "results_tracks.json").write_text(json.dumps(results, indent=2))
    print("done ->", OUT / "results_tracks.json")


if __name__ == "__main__":
    main()
