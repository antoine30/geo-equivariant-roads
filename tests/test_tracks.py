import numpy as np

from eqroads.dunes import DuneConfig, make_pairs
from eqroads.radon import change_field, local_radon_score, robust_zscore
from eqroads.synthetic import curved_path


def test_curved_path_is_connected_and_inside_the_tile():
    rc = curved_path(np.random.default_rng(0), 96, 0.3, 45.0, 40.0)
    assert (rc >= 0).all() and (rc <= 95).all()
    steps = np.abs(np.diff(rc, axis=0)).max(axis=1)
    assert steps.max() <= 1  # 8-connected chain
    assert len(rc) > 60      # crosses a good part of the tile


def test_radon_integrates_a_line_invisible_pixel_by_pixel():
    """Per-pixel SNR 1: the line is lost in the noise pixel-wise but stands out once integrated."""
    rng = np.random.default_rng(1)
    field = rng.standard_normal((64, 64))
    field[30, 8:56] -= 1.0
    score = local_radon_score(robust_zscore(field), window=24, stride=12, polarity="dark")
    on_line = np.median(score[30, 12:52])
    off_line = np.percentile(score[np.abs(np.arange(64) - 30) > 4], 95)
    assert on_line > off_line


def test_dune_pairs_label_only_new_tracks():
    t0, t1, label, has = make_pairs(0, 6, DuneConfig(size=64), balanced=True)
    assert t0.shape == t1.shape == (6, 4, 64, 64)
    assert list(has) == [True, False] * 3
    assert all(label[i].any() == has[i] for i in range(6))


def test_new_track_darkens_the_second_date():
    t0, t1, label, _ = make_pairs(3, 2, DuneConfig(size=64, migration=(0.0, 0.0), noise_sigma=0.0), balanced=True)
    z = change_field(t0[0], t1[0])
    assert np.median(z[label[0] > 0]) < np.median(z[label[0] == 0])
