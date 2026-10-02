"""Training and inference helpers shared by the scripts."""
from __future__ import annotations

import time
from dataclasses import dataclass

import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F

from .gnn import GapGNN, to_tensors, undirected_logits
from .graph import (
    add_links,
    build_features,
    candidate_labels,
    extract_graph,
    find_candidates,
    materialise_candidates,
)

# Fixed normalisation for Sentinel-2 L2A reflectance (B02, B03, B04, B08).
MEAN = np.array([0.12, 0.14, 0.16, 0.26], dtype=np.float32)[:, None, None]
STD = np.array([0.06, 0.07, 0.09, 0.09], dtype=np.float32)[:, None, None]


def normalise(img: np.ndarray) -> np.ndarray:
    """Per-band standardisation; 8-channel inputs are two stacked dates (t0, t1)."""
    reps = img.shape[-3] // 4
    return (img - np.concatenate([MEAN] * reps)) / np.concatenate([STD] * reps)


# --------------------------------------------------------------------------- segmentation
def seg_loss(logits, target, valid=None, pos_weight: float = 4.0):
    """Weighted BCE + soft Dice; roads cover only a few % of pixels. `valid` masks clouds."""
    w = torch.ones_like(target) if valid is None else valid
    bce = F.binary_cross_entropy_with_logits(
        logits, target, weight=w, pos_weight=torch.tensor(pos_weight), reduction="sum") / w.sum().clamp(min=1)
    p = torch.sigmoid(logits) * w
    dice = 1 - (2 * (p * target).sum() + 1) / (p.sum() + (target * w).sum() + 1)
    return bce + dice


def multiclass_loss(logits, target, valid=None, class_weights=(1.0, 4.0, 2.0)):
    """Weighted cross-entropy + soft Dice on the foreground classes (road, built-up).

    `target` holds class indices (B, H, W); `valid` (B, 1, H, W) masks clouds / no data.
    """
    C = logits.shape[1]
    w = torch.ones_like(target, dtype=logits.dtype) if valid is None else valid[:, 0]
    ce = F.cross_entropy(logits, target, weight=torch.tensor(class_weights[:C], dtype=logits.dtype),
                         reduction="none")
    ce = (ce * w).sum() / w.sum().clamp(min=1)
    p = torch.softmax(logits, dim=1) * w[:, None]
    onehot = F.one_hot(target, C).permute(0, 3, 1, 2).to(logits.dtype) * w[:, None]
    inter = (p * onehot).sum(dim=(0, 2, 3))[1:]
    denom = (p.sum(dim=(0, 2, 3)) + onehot.sum(dim=(0, 2, 3)))[1:]
    dice = 1 - ((2 * inter + 1) / (denom + 1)).mean()
    return ce + dice


def train_segmentation(model, sample_batch, steps: int, lr: float = 2e-3, log_every: int = 100, log=print):
    """`sample_batch(step)` returns numpy (images (B,4,H,W), masks (B,H,W)[, valid (B,H,W)]).

    Binary models (1 output channel) treat any non-zero mask value as positive; multi-class
    models take the mask values as class indices.
    """
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
    model.train()
    t0 = time.time()
    for step in range(steps):
        batch = sample_batch(step)
        x = torch.from_numpy(normalise(batch[0]))
        valid = torch.from_numpy(batch[2]).float()[:, None] if len(batch) > 2 else None
        logits = model(x)
        if logits.shape[1] == 1:
            loss = seg_loss(logits, torch.from_numpy(batch[1] > 0).float()[:, None], valid)
        else:
            loss = multiclass_loss(logits, torch.from_numpy(batch[1]).long(), valid)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if (step + 1) % log_every == 0:
            log(f"  step {step + 1}/{steps}  loss {loss.item():.4f}  ({time.time() - t0:.0f}s)")
    return model.eval()


@torch.no_grad()
def predict(model, images: np.ndarray, batch: int = 8) -> np.ndarray:
    """Probabilities for images (N, 4, H, W).

    Binary model: road probability (N, H, W). Multi-class model: class probabilities (N, C, H, W).
    """
    model.eval()
    out = []
    for i in range(0, len(images), batch):
        logits = model(torch.from_numpy(normalise(images[i: i + batch])))
        out.append((torch.sigmoid(logits)[:, 0] if logits.shape[1] == 1 else torch.softmax(logits, 1)).numpy())
    return np.concatenate(out)


# --------------------------------------------------------------------------- graph stage
@dataclass
class GraphExample:
    G: nx.Graph                 # road graph with anchor nodes inserted
    pairs: list                 # candidate links (src, dst)
    x: torch.Tensor
    edge_index: torch.Tensor
    edge_attr: torch.Tensor
    cand: torch.Tensor          # bool over undirected edges
    y: torch.Tensor | None      # labels for candidate edges


def make_example(prob: np.ndarray, gt: np.ndarray | None = None, max_dist: float = 15.0) -> GraphExample | None:
    G = extract_graph(prob)
    if G.number_of_edges() == 0:
        return None
    G2, pairs = materialise_candidates(G, find_candidates(G, max_dist=max_dist))
    nodes, x, edges, ea, is_cand = build_features(G2, pairs, prob, max_dist=max_dist)
    x, ei, ea = to_tensors(nodes, x, edges, ea)
    y = torch.from_numpy(candidate_labels(G2, pairs, gt)) if gt is not None and pairs else None
    return GraphExample(G2, pairs, x, ei, ea, torch.from_numpy(is_cand), y)


def train_gnn(examples: list[GraphExample], epochs: int = 40, lr: float = 3e-3, log=print) -> GapGNN:
    examples = [e for e in examples if e.pairs]
    n_pos = sum(float(e.y.sum()) for e in examples)
    n_all = sum(len(e.y) for e in examples)
    pos_weight = torch.tensor((n_all - n_pos) / max(n_pos, 1))
    model = GapGNN(examples[0].x.shape[1], examples[0].edge_attr.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    rng = np.random.default_rng(0)
    for epoch in range(epochs):
        total = 0.0
        for i in rng.permutation(len(examples)):
            e = examples[i]
            logits = undirected_logits(model(e.x, e.edge_index, e.edge_attr))[e.cand]
            loss = F.binary_cross_entropy_with_logits(logits, e.y, pos_weight=pos_weight)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        if (epoch + 1) % 10 == 0:
            log(f"  epoch {epoch + 1}/{epochs}  loss {total / len(examples):.4f}")
    return model.eval()


def _one_per_dead_end(pairs, scores, threshold):
    """Accept at most one link per dead end: the best-scoring one above threshold."""
    best = {}
    for i, ((src, _), s) in enumerate(zip(pairs, scores)):
        if s >= threshold and (src not in best or s > scores[best[src]]):
            best[src] = i
    accept = np.zeros(len(pairs), dtype=bool)
    accept[list(best.values())] = True
    return accept


@torch.no_grad()
def gnn_scores(model: GapGNN, ex: GraphExample) -> np.ndarray:
    if not ex.pairs:
        return np.zeros(0)
    logits = undirected_logits(model(ex.x, ex.edge_index, ex.edge_attr))[ex.cand]
    return torch.sigmoid(logits).numpy()


def heuristic_scores(ex: GraphExample) -> np.ndarray:
    """Classic rule-based gap closing: short, well-aligned links score high (baseline)."""
    if not ex.pairs:
        return np.zeros(0)
    ea = ex.edge_attr[: ex.edge_attr.shape[0] // 2][ex.cand].numpy()
    dist, cos_src = ea[:, 1], ea[:, 4]
    return np.clip(cos_src, 0, 1) * (1 - dist)


def refine(ex: GraphExample, scores: np.ndarray, threshold: float = 0.5):
    accept = _one_per_dead_end(ex.pairs, scores, threshold)
    return add_links(ex.G, ex.pairs, accept), accept


@torch.no_grad()
def predict_large(model, image: np.ndarray, tile: int = 256, overlap: int = 32) -> np.ndarray:
    """Sliding-window inference on a full scene (4, H, W) with blended overlaps.

    Returns (H, W) for a binary model, (C, H, W) for a multi-class model.
    """
    _, H, W = image.shape
    step = tile - overlap
    Hp, Wp = max(H, tile), max(W, tile)
    Hp, Wp = Hp + (-(Hp - tile)) % step, Wp + (-(Wp - tile)) % step
    padded = np.zeros((4, Hp, Wp), np.float32)
    padded[:, :H, :W] = image
    acc = None
    weight = np.zeros((Hp, Wp), np.float32)
    ramp = np.minimum(np.arange(tile) + 1, np.arange(tile)[::-1] + 1).clip(max=overlap) / overlap
    w2d = np.outer(ramp, ramp).astype(np.float32)
    for r in range(0, Hp - tile + 1, step):
        for c in range(0, Wp - tile + 1, step):
            p = predict(model, padded[None, :, r: r + tile, c: c + tile])[0]
            if acc is None:
                acc = np.zeros(p.shape[:-2] + (Hp, Wp), np.float32)
            acc[..., r: r + tile, c: c + tile] += p * w2d
            weight[r: r + tile, c: c + tile] += w2d
    return (acc / np.maximum(weight, 1e-6))[..., :H, :W]
