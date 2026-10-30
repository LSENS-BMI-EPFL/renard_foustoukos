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


def test_baseline_frames_follow_the_time_axis():
    import xarray as xr

    for start in (-1, -2):
        n = int((6 - start) * 30) + 1
        x = xr.DataArray(
            np.zeros((1, 1, n)), dims=['cell', 'trial', 'time'], coords={'time': np.linspace(start, 6, n)}
        )
        i0, i1 = imaging.baseline_frames(x)
        t = x.time.values
        assert t[i0] == -1 and t[i1 - 1] < 0 <= t[i1]  # -1 to 0 s whatever the tensor start
        assert i1 - i0 == 30


def test_time_axis_puts_the_stimulus_at_zero():
    from fast_learning import tensors

    assert tensors.window_frames((1, 6)) == (30, 180)
    t = tensors.time_axis((2, 6))
    assert len(t) == 60 + 180 + 1 and t[60] == 0 and t[0] == -2 and np.isclose(t[-1], 6)


def test_select_time_is_half_open():
    import xarray as xr
    from fast_learning import tensors

    t = tensors.time_axis((1, 6))
    x = xr.DataArray(np.arange(len(t)), dims=['time'], coords={'time': t})
    resp = imaging.select_time(x, 0, 0.3)
    base = imaging.select_time(x, -1, 0)
    assert len(resp) == 9 and resp.time.values[0] == 0  # stimulus frame to +267 ms
    assert len(base) == 30 and base.time.values[-1] < 0  # stimulus frame excluded


def test_select_time_refuses_windows_beyond_the_tensor():
    import pytest
    import xarray as xr
    from fast_learning import tensors

    t = tensors.time_axis((1, 6))
    x = xr.DataArray(np.arange(len(t)), dims=['time'], coords={'time': t})
    with pytest.raises(ValueError):
        imaging.select_time(x, -2, 0)
    assert len(imaging.select_time(x, -1, 1)) == 60
