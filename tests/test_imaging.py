import numpy as np

from fast_learning import imaging


def test_lmi_sign_and_significance():
    rng = np.random.default_rng(0)
    n = 40
    pre = np.stack([rng.normal(0, 1, n), rng.normal(0, 1, n), rng.normal(2, 1, n)])
    post = np.stack([rng.normal(3, 1, n), rng.normal(0, 1, n), rng.normal(0, 1, n)])
    lmi, lmi_p = imaging.compute_roc(pre, post, nshuffles=200, n_jobs=1)

    assert lmi.shape == lmi_p.shape == (3,)
    assert np.all((lmi >= -1) & (lmi <= 1))
    # Response increases after learning: LMI > 0, above the shuffle null.
    assert lmi[0] > 0.8 and lmi_p[0] >= 0.975
    # No change: LMI near 0, not significant.
    assert abs(lmi[1]) < 0.3 and 0.025 < lmi_p[1] < 0.975
    # Response decreases: LMI < 0, below the shuffle null.
    assert lmi[2] < -0.8 and lmi_p[2] <= 0.025


def test_lmi_is_twice_auc_minus_half():
    pre = np.array([[0.0, 1.0, 2.0, 3.0]])
    post = np.array([[4.0, 5.0, 6.0, 7.0]])  # perfectly separated: AUC = 1
    lmi, _ = imaging.compute_roc(pre, post, nshuffles=0)
    assert lmi[0] == 1.0


def test_lmi_shuffles_are_reproducible():
    rng = np.random.default_rng(1)
    pre, post = rng.normal(size=(2, 30)), rng.normal(0.5, 1, size=(2, 30))
    a = imaging.compute_roc(pre, post, nshuffles=50, n_jobs=1, return_shuffles=True)
    b = imaging.compute_roc(pre, post, nshuffles=50, n_jobs=1, return_shuffles=True)
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x, y)


def test_subtract_baseline():
    arr = np.arange(2 * 3 * 10, dtype=float).reshape(2, 3, 10)
    out = imaging.subtract_baseline(arr, 2, (0, 4))
    np.testing.assert_allclose(out[..., :4].mean(axis=2), 0)
    # A constant offset is removed, the shape of the trace is unchanged.
    np.testing.assert_allclose(np.diff(out, axis=2), np.diff(arr, axis=2))


def test_filter_data_by_cell_count():
    import pandas as pd

    data = pd.DataFrame({'mouse_id': ['A'] * 3 + ['B'] * 2, 'cell_type': ['wS2'] * 5, 'roi': [1, 2, 3, 1, 2]})
    out = imaging.filter_data_by_cell_count(data, min_cells=3)
    assert set(out['mouse_id']) == {'A'}
