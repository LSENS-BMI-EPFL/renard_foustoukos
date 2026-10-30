import numpy as np
import pandas as pd

from fast_learning import participation as pt


def test_participation_from_3d():
    n_cells, n_trials, n_t = 3, 2, 60
    data = np.zeros((n_cells, n_trials, n_t))
    win = pt.EVENT_WINDOW_FRAMES
    events = [20, n_t + 30, 2]  # trial 0 t=20, trial 1 t=30, one too close to the edge
    data[0, 0, 20 - win : 20 + win + 1] = 0.5  # cell 0 active at event 1 only
    data[1, :, :] = 0.5  # cell 1 always active
    rates, n_valid = pt.participation_from_3d(data, events, n_t, n_trials, 0.10)
    assert n_valid == 2
    np.testing.assert_allclose(rates, [0.5, 1.0, 0.0])


def test_participation_from_3d_no_valid_event():
    rates, n_valid = pt.participation_from_3d(np.zeros((2, 1, 30)), [0, 29], 30, 1, 0.10)
    assert rates is None and n_valid == 0


def test_per_cell_threshold():
    # A per-cell threshold array applies to each cell separately.
    data = np.zeros((2, 1, 60))
    data[:, 0, 20:31] = [[0.3], [0.3]]
    rates, _ = pt.participation_from_3d(data, [25], 60, 1, np.array([0.2, 0.4]))
    np.testing.assert_allclose(rates, [1.0, 0.0])
    np.testing.assert_allclose(pt.chance_participation(data, np.array([0.2, 0.4]))[1], 0.0)


def test_all_cell_rates_reliability():
    m = pt.MIN_EVENTS_FOR_RELIABILITY
    n_t = 60
    sub = {0: np.zeros((2, 4, n_t)), 1: np.zeros((2, 4, n_t))}
    results = {'days': {0: {'events': [tr * n_t + 30 for tr in range(m)]}, 1: {'events': [30]}}}
    df = pt.all_cell_rates('M', results, np.array([7, 8]), sub, {2.5: np.array([0.1, 0.1])})
    assert set(df['roi']) == {7, 8}
    assert df.loc[df['day'] == 0, 'reliable'].all() and not df.loc[df['day'] == 1, 'reliable'].any()


def test_noise_sd_is_robust_to_transients():
    rng = np.random.default_rng(1)
    sigma = 0.1  # per-frame noise; the SD of a (2w+1)-frame mean is sigma / sqrt(2w+1)
    data = rng.normal(0, sigma, (1, 400, 60))
    expected = sigma / np.sqrt(2 * pt.EVENT_WINDOW_FRAMES + 1)
    np.testing.assert_allclose(pt.noise_sd({0: data}), expected, rtol=0.05)
    data[0, :20, 20:40] += 1.0  # transients in 5% of trials
    np.testing.assert_allclose(pt.noise_sd({0: data}), expected, rtol=0.1)
    thr = pt.cell_thresholds(np.array([0.04, 0.0]))
    assert thr[2.5][0] == 0.1 and np.isinf(thr[2.5][1])


def test_aggregate_across_days():
    per_day = pd.DataFrame(
        {
            'mouse_id': 'M',
            'roi': 1,
            'day': [-2, -1, 0, 1, 2],
            'participation_rate': [0.1, 0.3, 0.5, 0.6, 0.8],
            'reliable': True,
        }
    )
    agg = pt.aggregate_across_days(per_day).iloc[0]
    assert agg['baseline_rate'] == 0.2
    assert agg['learning_rate'] == 0.5
    assert agg['post_rate'] == 0.7
    assert np.isclose(agg['delta_post'], 0.5)


def test_lmi_category():
    df = pt.add_lmi_category(pd.DataFrame({'lmi_p': [0.99, 0.5, 0.01, 0.975, 0.025]}))
    assert list(df['lmi_category']) == ['positive', 'neutral', 'negative', 'positive', 'negative']


def test_file_names():
    assert pt.thr_tag(2.5) == '2.5sd' and pt.thr_tag(10) == '10sd'
    assert pt.rates_csv(5).endswith('cell_participation_rates_per_day_5sd.csv')


def test_file_names_cells():
    assert pt.rates_csv(2.5, cells='all').endswith('2.5sd_allcells.csv')
    assert pt.merged_csv(2.5, nolick=True, cells='insample').endswith('2.5sd_insample_nolick.csv')


def test_split_halves():
    a, b = pt.split_halves('M', 11, 3)
    assert len(a) == 5 and len(b) == 6
    assert set(a).isdisjoint(b) and set(a) | set(b) == set(range(11))
    np.testing.assert_array_equal(a, pt.split_halves('M', 11, 3)[0])  # seeded
    assert not np.array_equal(a, pt.split_halves('M', 11, 4)[0])


def test_chance_participation_is_participation_at_every_time_point():
    # Chance rate = participation in events placed at every countable time point.
    rng = np.random.default_rng(0)
    n_cells, n_trials, n_t = 4, 6, 40
    data = rng.normal(0.05, 0.1, (n_cells, n_trials, n_t))
    all_points = np.arange(n_trials * n_t)
    expected, n_valid = pt.participation_from_3d(data, all_points, n_t, n_trials, 0.10)
    assert n_valid == n_trials * (n_t - 2 * pt.EVENT_WINDOW_FRAMES)
    np.testing.assert_allclose(pt.chance_participation(data, 0.10), expected)


def test_add_excess():
    per_day = pd.DataFrame({'mouse_id': 'M', 'day': 0, 'roi': [1, 2], 'participation_rate': [0.5, 0.2]})
    chance = pd.DataFrame(
        {
            'mouse_id': 'M',
            'day': 0,
            'roi': [1, 2, 1],
            'threshold': [0.1, 0.1, 0.2],
            'chance_rate': [0.1, 0.3, 0.9],
        }
    )
    out = pt.add_excess(per_day, chance, 0.1).set_index('roi')
    np.testing.assert_allclose(out['excess_rate'], [0.4, -0.1])


def test_average_over_splits():
    m = pt.MIN_EVENTS_FOR_RELIABILITY
    split_df = pd.DataFrame(
        {
            'mouse_id': 'M',
            'day': 0,
            'roi': [1, 1, 1, 2],
            'split': [0, 1, 2, 0],
            'participation_rate': [0.2, 0.4, 0.9, 0.5],
            'n_events': [m, m + 2, m - 1, m - 1],  # split 2 of roi 1 and roi 2 unreliable
        }
    )
    out = pt.average_over_splits(split_df).set_index('roi')
    assert list(out.index) == [1]
    assert np.isclose(out.loc[1, 'participation_rate'], 0.3) and out.loc[1, 'n_splits'] == 2
