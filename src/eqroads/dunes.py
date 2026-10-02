"""Bi-temporal desert scenes for monitoring passages: a faint new track across dunes.

The task is change detection, not road mapping: given two acquisitions of the same desert area,
find vehicle tracks that appeared in between. What makes it hard (and realistic):

  * the signal is weak: a track on a little-used piste is narrower than a 10 m pixel and only
    slightly darkens the compacted sand, so its contrast is a few percent;
  * the clutter is linear: dune crests form a dense pattern of parallel lines, and tracks often
    run along the interdune corridors, i.e. parallel to that clutter;
  * the clutter moves: dunes migrate between the two dates, so a plain image difference shows
    crest-shaped false changes;
  * the track is sinuous and partly covered by drifting sand;
  * older tracks, visible on both dates, must NOT trigger an alert.

At Sentinel-2's 10 m resolution only repeatedly used tracks integrate enough contrast along
their length to be seen; single fresh tyre tracks need VHR optical imagery or SAR coherent
change detection (see README).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from .synthetic import SPECTRA, curved_path


@dataclass
class DuneConfig:
    size: int = 64
    dune_wavelength: tuple[float, float] = (6.0, 12.0)    # crest spacing (px)
    dune_contrast: tuple[float, float] = (0.04, 0.09)     # relative shading amplitude
    migration: tuple[float, float] = (0.3, 1.0)           # crest displacement between dates (px)
    n_old_tracks: tuple[int, int] = (0, 2)
    new_track_prob: float = 0.5
    track_contrast: tuple[float, float] = (0.04, 0.12)    # relative darkening of the track
    sinuosity: tuple[float, float] = (10.0, 45.0)
    bend_wavelength: tuple[float, float] = (30.0, 80.0)
    along_dunes_prob: float = 0.5                         # track follows the interdune corridor
    drift_gaps: tuple[int, int] = (0, 3)                  # stretches hidden by drifting sand
    noise_sigma: float = 0.006


def _smooth(rng, size, sigma):
    f = ndimage.gaussian_filter(rng.standard_normal((size, size)), sigma, mode="wrap")
    return (f - f.mean()) / (f.std() + 1e-8)


def _dune_shading(u: np.ndarray, wavelength: float) -> np.ndarray:
    """Asymmetric crest profile: long gentle stoss slope, short dark slip face. Range [-1, 1]."""
    frac = np.mod(u / wavelength, 1.0)
    profile = np.where(frac < 0.7, frac / 0.7, (1.0 - frac) / 0.3)
    return 2.0 * profile - 1.0


def _track_mask(rng, size, theta, cfg, gaps: bool):
    rc = curved_path(rng, size, theta, rng.uniform(*cfg.sinuosity), rng.uniform(*cfg.bend_wavelength))
    full = np.zeros((size, size), dtype=bool)
    full[rc[:, 0], rc[:, 1]] = True
    visible = full.copy()
    if gaps and len(rc) > 10:
        for _ in range(rng.integers(cfg.drift_gaps[0], cfg.drift_gaps[1] + 1)):
            start = rng.integers(0, len(rc))
            seg = rc[start: start + rng.integers(3, 9)]
            visible[seg[:, 0], seg[:, 1]] = False
    return full, visible


def make_pair(rng: np.random.Generator, cfg: DuneConfig | None = None, new_track: bool | None = None):
    """Return (t0 (4,H,W), t1 (4,H,W), new-track mask (H,W) uint8, has_new_track bool)."""
    cfg = cfg or DuneConfig()
    size = cfg.size
    yy, xx = np.mgrid[:size, :size].astype(float)

    phi = rng.uniform(0, np.pi)                          # crest-normal direction
    lam = rng.uniform(*cfg.dune_wavelength)
    warp = _smooth(rng, size, 8.0) * lam * 0.35          # sinuous crests (seif / barchan-like)
    u = xx * np.cos(phi) + yy * np.sin(phi) + warp
    contrast = rng.uniform(*cfg.dune_contrast)
    albedo = 1.0 + 0.04 * _smooth(rng, size, 6.0)
    sand = SPECTRA["sand"][:, None, None]
    shift = rng.uniform(*cfg.migration)

    def scene(du):
        img = sand * albedo * (1.0 + contrast * _dune_shading(u + du, lam))
        return img * (1.0 + 0.02 * rng.standard_normal((4, 1, 1)))   # per-date radiometric gain

    t0, t1 = scene(0.0), scene(shift)

    crest_dir = phi + np.pi / 2                          # direction along the crests

    def heading():
        if rng.random() < cfg.along_dunes_prob:
            return -crest_dir + rng.normal(0, np.deg2rad(8))  # image y points down
        return rng.uniform(0, np.pi)

    for _ in range(rng.integers(cfg.n_old_tracks[0], cfg.n_old_tracks[1] + 1)):
        _, vis = _track_mask(rng, size, heading(), cfg, gaps=True)
        a = rng.uniform(*cfg.track_contrast)
        t0[:, vis] *= 1 - a
        t1[:, vis] *= 1 - a * rng.uniform(0.6, 1.0)     # slightly faded on the second date

    has_new = (rng.random() < cfg.new_track_prob) if new_track is None else new_track
    label = np.zeros((size, size), dtype=np.uint8)
    if has_new:
        full, vis = _track_mask(rng, size, heading(), cfg, gaps=True)
        t1[:, vis] *= 1 - rng.uniform(*cfg.track_contrast)
        label[full] = 1

    t0 = t0 + cfg.noise_sigma * rng.standard_normal(t0.shape)
    t1 = t1 + cfg.noise_sigma * rng.standard_normal(t1.shape)
    return (np.clip(t0, 0, 1).astype(np.float32), np.clip(t1, 0, 1).astype(np.float32), label, bool(has_new))


def make_pairs(seed: int, n: int, cfg: DuneConfig | None = None, balanced: bool = False):
    """Batch of pairs. With balanced=True exactly half of the pairs contain a new track."""
    rng = np.random.default_rng(seed)
    out = [make_pair(rng, cfg, (i % 2 == 0) if balanced else None) for i in range(n)]
    return (np.stack([o[0] for o in out]), np.stack([o[1] for o in out]),
            np.stack([o[2] for o in out]), np.array([o[3] for o in out]))
