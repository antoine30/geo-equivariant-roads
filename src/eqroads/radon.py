"""Localised Radon transform detector for faint linear features.

The Radon transform integrates an image along every straight line (angle theta, offset rho).
A faint line whose pixels are individually buried in noise adds up coherently along its
length, while noise averages out: for white noise of standard deviation sigma, the sum along
a line of L pixels has standard deviation sigma * sqrt(L), so a line of per-pixel contrast c
reaches a z-score of c * sqrt(L) / sigma. This is why the Radon transform is the classical
detector of ship wakes in SAR imagery (e.g. Murphy 1986; Copeland et al., IEEE TGRS 1995,
for the localised version used here), including the faint turbulent wakes of small,
low-profile vessels.

The global transform assumes a line crossing the whole image. Applying it in overlapping
windows (localised Radon) lets a curved feature be followed as a chain of short straight
pieces, as long as it is straight enough within one window.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage
from skimage.transform import iradon, radon


def robust_zscore(field: np.ndarray) -> np.ndarray:
    med = np.median(field)
    mad = np.median(np.abs(field - med)) * 1.4826 + 1e-9
    return (field - med) / mad


def highpass(band: np.ndarray, sigma: float = 3.0) -> np.ndarray:
    """Remove the slowly varying background so that thin features stand out."""
    return band - ndimage.gaussian_filter(band, sigma)


def local_radon_score(field: np.ndarray, window: int = 24, stride: int = 12, angle_step: float = 3.0,
                      min_length: int = 8, polarity: str = "both") -> np.ndarray:
    """Per-pixel detection score: z-score of the best line through the pixel, window by window.

    `field` should be roughly zero-mean with unit noise (see robust_zscore). polarity: "dark",
    "bright" or "both" (unknown contrast sign).
    """
    H, W = field.shape
    angles = np.arange(0.0, 180.0, angle_step)
    yy, xx = np.mgrid[:window, :window]
    c = (window - 1) / 2
    disk = ((yy - c) ** 2 + (xx - c) ** 2) <= (window / 2) ** 2
    length = radon(disk.astype(float), angles, circle=True)           # pixels along each line
    valid = length >= min_length
    norm = np.sqrt(np.maximum(length, 1.0))

    signs = {"dark": (-1.0,), "bright": (1.0,), "both": (1.0, -1.0)}[polarity]
    rows = sorted(set(list(range(0, max(H - window, 0) + 1, stride)) + [max(H - window, 0)]))
    cols = sorted(set(list(range(0, max(W - window, 0) + 1, stride)) + [max(W - window, 0)]))
    score = np.zeros((H, W), dtype=np.float32)
    for r in rows:
        for c0 in cols:
            patch = field[r: r + window, c0: c0 + window] * disk
            sino = radon(patch, angles, circle=True)
            for sgn in signs:
                z = np.where(valid, sgn * sino / norm, -np.inf)
                i, j = np.unravel_index(int(np.argmax(z)), z.shape)
                zmax = float(z[i, j])
                if zmax <= 0:
                    continue
                score_patch = score[r: r + window, c0: c0 + window]
                np.maximum(score_patch, zmax * _line_pixels(i, j, angles, window, disk), out=score_patch)
    return score


def _line_pixels(i: int, j: int, angles: np.ndarray, window: int, disk: np.ndarray) -> np.ndarray:
    """Pixels of the window lying on the line (rho index i, angle index j), via backprojection."""
    delta = np.zeros((window, len(angles)))
    delta[i, j] = 1.0
    back = iradon(delta, angles, filter_name=None, circle=True, output_size=window)
    return (back >= 0.5 * back.max()) & disk


def change_field(t0: np.ndarray, t1: np.ndarray) -> np.ndarray:
    """Robust z-score of the band-averaged difference between two dates."""
    return robust_zscore(t1.mean(axis=0) - t0.mean(axis=0))


def image_fields(img: np.ndarray) -> list[np.ndarray]:
    """Single-date inputs for road mapping: high-passed visible brightness and NIR."""
    return [robust_zscore(highpass(img[:3].mean(axis=0))), robust_zscore(highpass(img[3]))]
