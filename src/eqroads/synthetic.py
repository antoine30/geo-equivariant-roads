"""Synthetic Sentinel-2-like tiles (B02, B03, B04, B08 at 10 m) with road masks.

Used for unit tests, CI and the controlled experiments in the README, so the whole pipeline
runs offline in minutes. Real data comes from `data_s2.py` (STAC + COG + OpenStreetMap).

Design choices that mimic the real problem:
  * roads are 1-3 px wide (5-30 m at 10 m GSD) and only partially fill their pixels,
    so their spectral contrast is weak, as on real Sentinel-2 imagery;
  * canopy / shadow occlusions hide short stretches of road while the label (as with
    OpenStreetMap) keeps the road continuous -> the segmentation map has gaps that the
    graph stage must close;
  * `angle_range` restricts road headings, which lets us test generalisation to unseen
    orientations (the motivation for rotation equivariance).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from skimage.draw import line as draw_line

# Approximate surface reflectance in (B02, B03, B04, B08)
SPECTRA = {
    "vegetation": np.array([0.04, 0.07, 0.05, 0.35]),
    "soil": np.array([0.10, 0.14, 0.18, 0.25]),
    "sand": np.array([0.18, 0.24, 0.30, 0.34]),
    "urban": np.array([0.12, 0.13, 0.14, 0.18]),
    "asphalt": np.array([0.08, 0.09, 0.10, 0.12]),
    "concrete": np.array([0.20, 0.22, 0.24, 0.26]),
    "shadow": np.array([0.02, 0.03, 0.03, 0.08]),
    "roof_concrete": np.array([0.22, 0.24, 0.27, 0.30]),
    "roof_tile": np.array([0.10, 0.11, 0.19, 0.27]),
    "roof_dark": np.array([0.09, 0.10, 0.11, 0.15]),
}
ROOFS = ("roof_concrete", "roof_tile", "roof_dark")

# Class indices of the multi-class masks
BACKGROUND, ROAD, BUILDING = 0, 1, 2
CLASS_NAMES = ("background", "road", "built-up")
BAND_NAMES = ("B02", "B03", "B04", "B08")


@dataclass
class TileConfig:
    size: int = 64
    n_roads: tuple[int, int] = (2, 5)
    angle_range: tuple[float, float] | None = None  # degrees, None = any heading
    dead_end_prob: float = 0.3
    n_occlusions: tuple[int, int] = (1, 4)
    occlusion_radius: tuple[int, int] = (2, 4)
    noise_sigma: float = 0.01
    curved_prob: float = 0.0                  # probability that a road is sinuous
    sinuosity: tuple[float, float] = (25.0, 60.0)    # heading oscillation amplitude (degrees)
    bend_wavelength: tuple[float, float] = (25.0, 60.0)  # distance between bends (px)
    n_settlements: tuple[int, int] = (0, 0)   # > 0 adds built-up areas (class 2)
    settlement_radius: tuple[int, int] = (10, 20)


def _smooth_field(rng: np.random.Generator, size: int, sigma: float) -> np.ndarray:
    f = ndimage.gaussian_filter(rng.standard_normal((size, size)), sigma, mode="wrap")
    return (f - f.mean()) / (f.std() + 1e-8)


def _background(rng: np.random.Generator, size: int) -> np.ndarray:
    classes = ["vegetation", "soil", "sand", "urban"]
    logits = np.stack([_smooth_field(rng, size, rng.uniform(4, 10)) * 2.0 for _ in classes])
    w = np.exp(logits) / np.exp(logits).sum(0, keepdims=True)
    img = np.einsum("khw,kb->bhw", w, np.stack([SPECTRA[c] for c in classes]))
    texture = 0.02 * np.stack([_smooth_field(rng, size, 1.0) for _ in range(4)])
    return img + texture


def _road_polyline(rng, size, theta, dead_end):
    """Endpoints of a straight road through a random point, clipped to the tile.

    The segment is clipped along its own direction (Liang-Barsky), so the road keeps exactly
    the heading `theta`. Clipping x and y independently would bend roads towards the corners.
    """
    c = rng.uniform(0.15 * size, 0.85 * size, size=2)
    d = np.array([np.cos(theta), -np.sin(theta)])  # (x, y) with image y pointing down
    t_lo, t_hi = -np.inf, np.inf
    for k in range(2):
        if abs(d[k]) > 1e-12:
            a, b = (0 - c[k]) / d[k], (size - 1 - c[k]) / d[k]
            t_lo, t_hi = max(t_lo, min(a, b)), min(t_hi, max(a, b))
    t0 = max(t_lo, -rng.uniform(0.2, 0.5) * size) if dead_end else t_lo
    return c + t0 * d, c + t_hi * d


def curved_path(rng, size, theta0, amplitude_deg, wavelength, step=0.5):
    """Pixel chain (rows, cols) of a sinuous road whose heading oscillates around theta0.

    heading(s) = theta0 + A sin(2 pi s / wavelength + phase), integrated in both directions from
    a random interior point until the path leaves the tile. Sub-pixel steps keep it 8-connected.
    """
    c = rng.uniform(0.2 * size, 0.8 * size, size=2)
    phase = rng.uniform(0, 2 * np.pi)
    amp = np.deg2rad(amplitude_deg)
    halves = []
    for sign in (1.0, -1.0):
        p, s, pts = c.copy(), 0.0, []
        while 0 <= p[0] <= size - 1 and 0 <= p[1] <= size - 1 and s < 6 * size:
            pts.append(p.copy())
            th = theta0 + amp * np.sin(2 * np.pi * sign * s / wavelength + phase)
            p = p + sign * step * np.array([np.cos(th), -np.sin(th)])
            s += step
        halves.append(pts)
    xy = np.array(halves[1][::-1] + halves[0][1:])
    rc = np.rint(xy[:, ::-1]).astype(int)
    keep = np.r_[True, np.any(np.diff(rc, axis=0) != 0, axis=1)]
    return rc[keep]


def _add_settlements(rng, img, mask, cfg):
    """Built-up areas along roads: blocks of 1-3 px roofs separated by streets and gardens.

    At 10 m GSD a villa covers one or two pixels mixed with its surroundings, so the label
    is "built-up" (class 2) on pixels whose footprint is mostly roof, as with OSM-derived labels.
    """
    size = mask.shape[0]
    road_px = np.argwhere(mask == ROAD)
    keep_out = ndimage.binary_dilation(mask == ROAD, iterations=1)
    yy, xx = np.mgrid[:size, :size]
    for _ in range(rng.integers(cfg.n_settlements[0], cfg.n_settlements[1] + 1)):
        if len(road_px):
            cy, cx = road_px[rng.integers(len(road_px))] + rng.integers(-6, 7, size=2)
        else:
            cy, cx = rng.integers(0, size, size=2)
        radius = rng.uniform(*cfg.settlement_radius)
        area = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius**2
        # Ground between buildings: mix of bare urban ground and gardens.
        ground = SPECTRA["urban"] if rng.random() < 0.6 else 0.5 * (SPECTRA["urban"] + SPECTRA["vegetation"])
        img[:, area & ~keep_out] = 0.5 * img[:, area & ~keep_out] + 0.5 * ground[:, None]
        roof = SPECTRA[ROOFS[rng.integers(len(ROOFS))]]
        pitch = int(rng.choice([2, 3, 4], p=[0.4, 0.4, 0.2]))  # dense core ... villa district
        oy, ox = rng.integers(0, pitch, size=2)
        for by in range(int(cy - radius) + oy, int(cy + radius), pitch):
            for bx in range(int(cx - radius) + ox, int(cx + radius), pitch):
                h, w = rng.integers(1, 3, size=2)
                if rng.random() < 0.15:
                    continue  # empty plot
                block = np.zeros_like(area)
                block[max(by, 0): max(by + h, 0), max(bx, 0): max(bx + w, 0)] = True
                block &= area & ~keep_out
                if not block.any():
                    continue
                alpha = rng.uniform(0.7, 0.95)
                img[:, block] = (1 - alpha) * img[:, block] + alpha * roof[:, None]
                mask[block] = BUILDING


def make_tile(rng: np.random.Generator, cfg: TileConfig | None = None):
    """Return (image (4,H,W) float32 reflectance, mask (H,W) uint8).

    Mask values: 0 background, 1 road, 2 built-up (only if cfg.n_settlements > 0).
    """
    cfg = cfg or TileConfig()
    size = cfg.size
    img = _background(rng, size)
    mask = np.zeros((size, size), dtype=bool)

    for _ in range(rng.integers(cfg.n_roads[0], cfg.n_roads[1] + 1)):
        if cfg.angle_range is None:
            theta = rng.uniform(0, np.pi)
        else:
            theta = np.deg2rad(rng.uniform(*cfg.angle_range))
        canvas = np.zeros((size, size), dtype=bool)
        if cfg.curved_prob > 0 and rng.random() < cfg.curved_prob:
            rc = curved_path(rng, size, theta, rng.uniform(*cfg.sinuosity), rng.uniform(*cfg.bend_wavelength))
            canvas[rc[:, 0], rc[:, 1]] = True
        else:
            p0, p1 = _road_polyline(rng, size, theta, rng.random() < cfg.dead_end_prob)
            r0, c0, r1, c1 = (int(v) for v in np.rint([p0[1], p0[0], p1[1], p1[0]]))
            rr, cc = draw_line(r0, c0, r1, c1)
            canvas[rr, cc] = True
        if rng.random() < 0.4:  # major road: ~3 px wide
            canvas = ndimage.binary_dilation(canvas, structure=ndimage.generate_binary_structure(2, 1))
        surface = SPECTRA["asphalt"] if rng.random() < 0.6 else SPECTRA["concrete"]
        alpha = rng.uniform(0.45, 0.8)  # sub-pixel road: partial spectral mixing
        img[:, canvas] = (1 - alpha) * img[:, canvas] + alpha * surface[:, None]
        mask |= canvas

    mask = mask.astype(np.uint8)
    if cfg.n_settlements[1] > 0:
        _add_settlements(rng, img, mask, cfg)

    # Occlusions placed on roads: the label stays continuous, the imagery does not.
    road_px = np.argwhere(mask == ROAD)
    yy, xx = np.mgrid[:size, :size]
    for _ in range(rng.integers(cfg.n_occlusions[0], cfg.n_occlusions[1] + 1)):
        if len(road_px) == 0:
            break
        cy, cx = road_px[rng.integers(len(road_px))]
        r = rng.uniform(*cfg.occlusion_radius)
        blob = (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r
        cover = SPECTRA["vegetation"] if rng.random() < 0.6 else SPECTRA["shadow"]
        img[:, blob] = cover[:, None] + 0.01 * rng.standard_normal((4, int(blob.sum())))

    img = img + cfg.noise_sigma * rng.standard_normal(img.shape)
    return np.clip(img, 0, 1).astype(np.float32), mask


def make_batch(seed: int, n: int, cfg: TileConfig | None = None):
    rng = np.random.default_rng(seed)
    tiles = [make_tile(rng, cfg) for _ in range(n)]
    return np.stack([t[0] for t in tiles]), np.stack([t[1] for t in tiles])
