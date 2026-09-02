"""Phase 1b: fit_probe layer sweep + Probe (synthetic activations, no model)."""

import numpy as np
import pytest

from examples._shared.probe import Probe, fit_probe, load_probe, save_probe


def _blobs(n=400, d=12, sep=2.0, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=n)
    x = rng.standard_normal((n, d))
    x[:, 0] += y * sep  # class signal on dim 0
    return x, y


def test_fit_probe_learns_separable_signal():
    x, y = _blobs(sep=3.0)
    probe = fit_probe({7: x}, y)
    assert probe.layer == 7
    assert probe.val_auc > 0.9


def test_fit_probe_picks_the_informative_layer():
    x_good, y = _blobs(sep=3.0, seed=1)
    rng = np.random.default_rng(2)
    x_noise = rng.standard_normal(x_good.shape)  # no signal
    probe = fit_probe({3: x_noise, 9: x_good}, y)
    assert probe.layer == 9


def test_score_shape_and_range():
    x, y = _blobs()
    probe = fit_probe({0: x}, y)
    s = probe.score(x)
    assert s.shape == (x.shape[0],)
    assert np.all((s >= 0) & (s <= 1))


def test_direction_is_unit_norm():
    x, y = _blobs()
    probe = fit_probe({0: x}, y)
    for standardized in (True, False):
        d = probe.direction(standardized=standardized)
        assert d.shape == (x.shape[1],)
        np.testing.assert_allclose(np.linalg.norm(d), 1.0, atol=1e-6)


def test_save_load_roundtrip(tmp_path):
    x, y = _blobs(seed=3)
    probe = fit_probe({5: x}, y, pooling="mean")
    path = tmp_path / "probe.npz"
    save_probe(probe, path)
    reloaded = load_probe(path)
    assert isinstance(reloaded, Probe)
    assert reloaded.layer == 5
    assert reloaded.pooling == "mean"
    np.testing.assert_allclose(reloaded.score(x), probe.score(x))


def test_fit_probe_rejects_single_class():
    x, _ = _blobs()
    with pytest.raises(ValueError):
        fit_probe({0: x}, np.zeros(x.shape[0], dtype=int))
