import numpy as np
import pandas as pd
import pytest

from fast_learning import reactivations as rx


def test_template_correlation():
    rng = np.random.default_rng(0)
    template = rng.normal(size=20)
    data = rng.normal(size=(20, 100))
    data[:, 10] = 3 * template + 1  # same pattern, scaled and shifted
    data[:, 20] = -template  # opposite pattern
    corr = rx.compute_template_correlation(data, template)
    assert corr.shape == (100,)
    assert corr[10] == pytest.approx(1.0)
    assert corr[20] == pytest.approx(-1.0)
    assert np.all(np.abs(corr) <= 1 + 1e-9)


def test_template_correlation_flat_template():
    corr = rx.compute_template_correlation(np.ones((5, 10)), np.zeros(5))
    np.testing.assert_array_equal(corr, 0)


def test_detect_events_finds_planted_peaks():
    corr = np.zeros(300)
    for t in (50, 150, 250):
        corr[t - 3 : t + 4] = [0.1, 0.3, 0.6, 0.9, 0.6, 0.3, 0.1]
    corr[100] = 0.2  # below threshold
    events = rx.detect_reactivation_events(
        corr, threshold=0.5, min_distance=rx.MIN_EVENT_DISTANCE_FRAMES, prominence=rx.PROMINENCE
    )
    np.testing.assert_array_equal(events, [50, 150, 250])


def test_detect_events_min_distance():
    corr = np.zeros(100)
    corr[[40, 43]] = 1.0  # closer than the minimum distance
    events = rx.detect_reactivation_events(corr, 0.3, min_distance=10, prominence=0.1, smooth=False)
    assert len(events) == 1


def test_percentile_tag():
    assert rx.percentile_tag(99) == 'p99'
    assert rx.percentile_tag(99.5) == 'p995'
    assert rx.percentile_tag(99.9) == 'p999'


def test_load_surrogate_thresholds(tmp_path):
    pd.DataFrame({'mouse_id': ['M1', 'M2'], 'threshold_percentile_median': [0.3, 0.5]}).to_csv(
        tmp_path / 'surrogate_thresholds_per_mouse_p995.csv', index=False
    )
    thr = rx.load_surrogate_thresholds(str(tmp_path / 'surrogate_thresholds_per_mouse.csv'), percentile=99.5)
    assert thr['M1'] == {d: 0.3 for d in [-2, -1, 0, 1, 2]}
    assert rx.get_threshold_for_mouse_day(thr, 'M2', 0) == 0.5
    # Unknown mouse: fixed fallback threshold.
    assert rx.get_threshold_for_mouse_day(thr, 'M3', 0) == rx.THRESHOLD_CORR
    with pytest.raises(FileNotFoundError):
        rx.load_surrogate_thresholds(str(tmp_path / 'missing.csv'), percentile=99)


def test_surrogate_rng_is_per_mouse_day():
    a = rx.surrogate_rng('GF305', 0).integers(1e9, size=5)
    b = rx.surrogate_rng('GF305', 0).integers(1e9, size=5)
    c = rx.surrogate_rng('GF305', 1).integers(1e9, size=5)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def test_mouse_selection():
    def res(n_day0):
        return {'days': {0: {'total_events': n_day0}, 1: {'total_events': 50}}}

    results = {
        'r_plus_results': {'A': res(10), 'B': res(0)},
        'r_minus_results': {'C': res(3), 'D': {'days': {1: {'total_events': 9}}}},
    }
    sel = rx.compute_mouse_selection(results).set_index('mouse_id')
    assert sel.loc['A', 'included'] and sel.loc['C', 'included']  # >= 3 events
    assert not sel.loc['B', 'included']  # no day-0 events
    assert not sel.loc['D', 'included'] and sel.loc['D', 'n_events_day0'] == 0  # day 0 missing
    assert list(sel['reward_group']) == ['R+', 'R+', 'R-', 'R-']
