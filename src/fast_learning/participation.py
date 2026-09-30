"""Participation of individual cells in reactivation events.

For each reactivation event, a cell participates if its baseline-subtracted
dF/F averaged over +/- EVENT_WINDOW_MS around the event is at least
PARTICIPATION_THRESHOLD. A cell's participation rate on a day is the fraction
of that day's valid events it participates in (events within EVENT_WINDOW_MS
of a trial edge are skipped).

Also provides the circular-shift test of participation (Supp. 4c) and the
spontaneous transient frequency used as a covariate (Supp. 4a-b).

Only the mice in the participation mouse selection are analysed (see
fast_learning.reactivations); the pipeline step participation.py applies it.
"""

import os
import zlib

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, savgol_filter

from fast_learning import imaging, paths, reactivations


# ============================================================================
# Parameters
# ============================================================================

DAYS = [-2, -1, 0, 1, 2]
SAMPLING_RATE = 30
EVENT_WINDOW_MS = 150
EVENT_WINDOW_FRAMES = int(EVENT_WINDOW_MS / 1000 * SAMPLING_RATE)
PARTICIPATION_THRESHOLD = 0.10
MIN_EVENTS_FOR_RELIABILITY = reactivations.MIN_EVENTS_PER_DAY
MIN_NOSTIM_TRIALS = 10
LMI_POSITIVE_THRESHOLD = 0.975
LMI_NEGATIVE_THRESHOLD = 0.025

# Circular-shift test (Supp. 4c)
N_SHIFTS = 1000
SHIFT_SEED = 0  # Base seed; each (mouse, day) gets its own stream
MIN_SHIFT_FRAMES = 0
SIGNIFICANCE_PCTILE = 95  # top 5 % -> p < 0.05

# Spontaneous transient detection (Supp. 4a-b)
MIN_DISTANCE_MS = 200
MIN_DISTANCE_FRAMES = int(MIN_DISTANCE_MS / 1000 * SAMPLING_RATE)
PROMINENCE_TRANSIENT = 0.2
N_STD_THRESHOLD = 3  # Per-cell threshold: N_STD_THRESHOLD * std(trace)
SAVGOL_WINDOW = 10
SAVGOL_ORDER = 2


# ============================================================================
# Output files (written by the participation pipeline step)
# ============================================================================

RESULTS_DIR = os.path.join(paths.processed_dir, 'reactivation')
DAY0_CSV = os.path.join(RESULTS_DIR, 'supp4ab_lmi_data_day0.csv')
BINARY_CSV = os.path.join(RESULTS_DIR, 'binary_participation_with_lmi.csv')
PARTICIPATION_THRESHOLDS = [0.10, 0.20, 0.50]  # main value first, then robustness checks


def thr_tag(threshold):
    return f'thr{int(round(threshold * 100))}'


def rates_csv(threshold=PARTICIPATION_THRESHOLD, selection='allnostim'):
    """Per-cell, per-day participation rates. selection: 'allnostim' or 'nolick'."""
    return os.path.join(RESULTS_DIR, f'cell_participation_rates_per_day_{selection}_{thr_tag(threshold)}.csv')


def merged_csv(threshold=PARTICIPATION_THRESHOLD, selection='allnostim'):
    """Per-cell baseline / day-0 / post participation rates merged with LMI."""
    return os.path.join(RESULTS_DIR, f'participation_lmi_merged_{selection}_{thr_tag(threshold)}.csv')


def _read(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Run pipeline/08_participation.py first.")
    return pd.read_csv(path)


def load_participation(threshold=PARTICIPATION_THRESHOLD, selection='allnostim'):
    """(merged_df, per_day_df) at one participation threshold, with lmi_category."""
    merged_df = _read(merged_csv(threshold, selection))
    per_day_df = _read(rates_csv(threshold, selection))
    if 'lmi_category' not in merged_df.columns:
        merged_df = add_lmi_category(merged_df)
    print(
        f"Loaded {len(merged_df)} cells and {len(per_day_df)} cell-day records "
        f"({selection}, {thr_tag(threshold)})."
    )
    return merged_df, per_day_df


def load_day0():
    """Day-0 participation rate, transient frequency and LMI per cell."""
    return _read(DAY0_CSV)


def load_binary_participation():
    """Binary participation per cell-day (circular-shift test), with LMI."""
    df = _read(BINARY_CSV)
    print(f"Loaded: {BINARY_CSV}  ({len(df)} rows)")
    return df


# ============================================================================
# Participation rates
# ============================================================================


def extract_event_responses(
    mouse, day, events, participation_threshold=PARTICIPATION_THRESHOLD, no_lick_only=False, time_window=None
):
    """Per-cell dF/F responses around the reactivation events of one mouse-day.

    The trial selection (no_lick_only, time_window) must be the one used to
    detect `events`, since event indices point into that selection's
    concatenated trial x time axes.

    Returns a DataFrame (mouse_id, day, roi, event_idx, avg_response,
    participates), or None with fewer than MIN_NOSTIM_TRIALS no-stim trials
    or no valid event.
    """
    xarr = imaging.load_mouse_xarray(
        mouse, paths.tensor_dir, 'tensor_xarray_learning_data.nc', subtracted=True
    )
    xarr_day = xarr.sel(trial=xarr['day'] == day)
    nostim, _ = reactivations.select_trials_by_type(
        xarr_day, no_lick_only=no_lick_only, time_window=time_window
    )

    if len(nostim.trial) < MIN_NOSTIM_TRIALS:
        return None

    n_cells, n_trials, n_timepoints = nostim.shape
    data_3d = nostim.values
    roi_list = nostim['roi'].values
    win = EVENT_WINDOW_FRAMES

    rows = []
    for event_idx in events:
        trial_idx = event_idx // n_timepoints
        time_idx = event_idx % n_timepoints
        if time_idx < win or time_idx >= n_timepoints - win or trial_idx >= n_trials:
            continue
        window_data = data_3d[:, trial_idx, time_idx - win : time_idx + win + 1]
        avg_response = np.mean(window_data, axis=1)
        participates = avg_response >= participation_threshold
        for icell in range(n_cells):
            rows.append(
                {
                    'mouse_id': mouse,
                    'day': day,
                    'roi': roi_list[icell],
                    'event_idx': event_idx,
                    'avg_response': float(avg_response[icell]),
                    'participates': bool(participates[icell]),
                }
            )

    return pd.DataFrame(rows) if rows else None


def compute_participation_rate(responses_df):
    """Aggregate cell-event responses to per-cell, per-day participation rates."""
    grouped = (
        responses_df.groupby(['mouse_id', 'day', 'roi'])
        .agg(
            n_participations=('participates', 'sum'),
            n_events=('participates', 'count'),
        )
        .reset_index()
    )
    grouped['participation_rate'] = grouped['n_participations'] / grouped['n_events']
    grouped['reliable'] = grouped['n_events'] >= MIN_EVENTS_FOR_RELIABILITY
    return grouped


def process_mouse_participation(
    mouse,
    mouse_results,
    participation_threshold=PARTICIPATION_THRESHOLD,
    no_lick_only=False,
    time_window=None,
):
    """Participation rates across all days for one mouse.

    mouse_results is that mouse's entry of a reactivation results file.
    Returns (mouse, DataFrame or None).
    """
    all_responses = []
    for day in DAYS:
        events = mouse_results.get('days', {}).get(day, {}).get('events', None)
        if events is None or len(events) == 0:
            continue
        try:
            resp_df = extract_event_responses(
                mouse,
                day,
                events,
                participation_threshold=participation_threshold,
                no_lick_only=no_lick_only,
                time_window=time_window,
            )
            if resp_df is not None and len(resp_df) > 0:
                all_responses.append(resp_df)
        except Exception as e:
            print(f"  Warning: {mouse} day {day}: {e}")
    if not all_responses:
        return mouse, None
    all_resp_df = pd.concat(all_responses, ignore_index=True)
    return mouse, compute_participation_rate(all_resp_df)


def aggregate_across_days(participation_df_all):
    """Aggregate per-day participation rates to baseline (days -2, -1),
    learning (day 0) and post (days +1, +2) rates per cell."""
    if participation_df_all is None or len(participation_df_all) == 0:
        return None

    baseline_days = [-2, -1]
    post_days = [1, 2]
    results = []

    for (mouse_id, roi), group in participation_df_all.groupby(['mouse_id', 'roi']):
        baseline_data = group[group['day'].isin(baseline_days)]
        baseline_rate = baseline_data['participation_rate'].mean() if len(baseline_data) > 0 else np.nan
        reliable_baseline = baseline_data['reliable'].all() if len(baseline_data) > 0 else False

        learning_data = group[group['day'] == 0]
        learning_rate = learning_data['participation_rate'].iloc[0] if len(learning_data) > 0 else np.nan
        reliable_learning = learning_data['reliable'].iloc[0] if len(learning_data) > 0 else False

        post_data = group[group['day'].isin(post_days)]
        post_rate = post_data['participation_rate'].mean() if len(post_data) > 0 else np.nan
        reliable_post = post_data['reliable'].all() if len(post_data) > 0 else False

        results.append(
            {
                'mouse_id': mouse_id,
                'roi': roi,
                'baseline_rate': baseline_rate,
                'learning_rate': learning_rate,
                'post_rate': post_rate,
                'delta_learning': learning_rate - baseline_rate if not np.isnan(baseline_rate) else np.nan,
                'delta_post': post_rate - baseline_rate if not np.isnan(baseline_rate) else np.nan,
                'reliable_baseline': reliable_baseline,
                'reliable_learning': reliable_learning,
                'reliable_post': reliable_post,
            }
        )

    return pd.DataFrame(results)


def add_lmi_category(lmi_df):
    """Add an lmi_category column (positive / negative / neutral) from lmi_p."""
    lmi_df = lmi_df.copy()
    lmi_df['lmi_category'] = 'neutral'
    lmi_df.loc[lmi_df['lmi_p'] >= LMI_POSITIVE_THRESHOLD, 'lmi_category'] = 'positive'
    lmi_df.loc[lmi_df['lmi_p'] <= LMI_NEGATIVE_THRESHOLD, 'lmi_category'] = 'negative'
    return lmi_df


def merge_with_lmi(participation_df, lmi_df, reward_groups):
    """Merge per-cell participation data with LMI results.

    reward_groups maps mouse_id to 'R+' / 'R-' and is used when lmi_df has no
    reward_group column.
    """
    if 'reward_group' not in lmi_df.columns:
        lmi_df = lmi_df.copy()
        lmi_df['reward_group'] = lmi_df['mouse_id'].map(reward_groups)
    lmi_df = add_lmi_category(lmi_df)

    cols = ['mouse_id', 'roi', 'lmi', 'lmi_p', 'lmi_category', 'reward_group']
    if 'cell_type' in lmi_df.columns:
        cols.append('cell_type')

    merged_df = pd.merge(participation_df, lmi_df[cols], on=['mouse_id', 'roi'], how='inner')

    print(
        f"\n  Merged: {len(merged_df)} cells total "
        f"({(merged_df['lmi_category'] == 'positive').sum()} LMI+, "
        f"{(merged_df['lmi_category'] == 'negative').sum()} LMI-, "
        f"{(merged_df['lmi_category'] == 'neutral').sum()} neutral)"
    )
    return merged_df


# ============================================================================
# Circular-shift test of participation (Supp. 4c)
# ============================================================================


def participation_from_3d(
    data_3d, events, n_timepoints, n_trials, participation_threshold=PARTICIPATION_THRESHOLD
):
    """Vectorised participation rate per cell.

    data_3d is (n_cells, n_trials, n_timepoints); events are frame indices in
    the flattened trial x time axis. Returns (rates or None, n_valid_events).
    """
    win = EVENT_WINDOW_FRAMES
    valid = [
        ev
        for ev in events
        if (ev % n_timepoints) >= win
        and (ev % n_timepoints) < n_timepoints - win
        and (ev // n_timepoints) < n_trials
    ]
    if not valid:
        return None, 0

    t_idxs = np.array([ev % n_timepoints for ev in valid])
    tr_idxs = np.array([ev // n_timepoints for ev in valid])

    windows = np.stack(
        [data_3d[:, tr_idxs[i], t_idxs[i] - win : t_idxs[i] + win + 1] for i in range(len(valid))]
    )
    avg = np.mean(windows, axis=2)
    rates = np.mean(avg >= participation_threshold, axis=0)
    return rates, len(valid)


def participation_with_shifts(mouse, day, events, n_shifts=N_SHIFTS):
    """Binary participation of each cell on one mouse-day.

    A cell participates if its participation rate exceeds the
    SIGNIFICANCE_PCTILE-th percentile of a null distribution built from
    n_shifts circular shifts of the data (the same shift for all cells, which
    preserves correlations between cells).

    Returns a DataFrame (mouse_id, day, roi, participating, n_events), or None
    with fewer than MIN_NOSTIM_TRIALS no-stim trials or fewer than
    MIN_EVENTS_FOR_RELIABILITY valid events.
    """
    try:
        xr = imaging.load_mouse_xarray(
            mouse, paths.tensor_dir, 'tensor_xarray_learning_data.nc', subtracted=True
        )
        xr_day = xr.sel(trial=xr['day'] == day)
        nostim = xr_day.sel(trial=xr_day['no_stim'] == 1)

        n_cells, n_trials, n_timepoints = nostim.shape
        if n_trials < MIN_NOSTIM_TRIALS:
            return None

        data_3d = np.nan_to_num(nostim.values, nan=0.0)
        roi_list = nostim['roi'].values
        n_frames = n_trials * n_timepoints

        if events is None or len(events) == 0:
            return None

        real_rates, n_valid = participation_from_3d(data_3d, events, n_timepoints, n_trials)
        if real_rates is None or n_valid < MIN_EVENTS_FOR_RELIABILITY:
            return None

        data_flat = data_3d.reshape(n_cells, n_frames)
        null_rates = np.full((n_shifts, n_cells), np.nan)
        rng = np.random.default_rng([SHIFT_SEED, zlib.crc32(mouse.encode()), day + 10])
        for i_shift in range(n_shifts):
            shift = (
                rng.integers(MIN_SHIFT_FRAMES + 1, n_frames)
                if MIN_SHIFT_FRAMES > 0
                else rng.integers(1, n_frames)
            )
            shifted_3d = np.roll(data_flat, shift, axis=1).reshape(n_cells, n_trials, n_timepoints)
            null_r, _ = participation_from_3d(shifted_3d, events, n_timepoints, n_trials)
            if null_r is not None:
                null_rates[i_shift] = null_r

        threshold = np.nanpercentile(null_rates, SIGNIFICANCE_PCTILE, axis=0)
        significant = real_rates > threshold

        records = [
            {
                'mouse_id': mouse,
                'day': day,
                'roi': roi_list[icell],
                'participating': bool(significant[icell]),
                'n_events': n_valid,
            }
            for icell in range(n_cells)
            if not np.isnan(real_rates[icell])
        ]
        return pd.DataFrame(records) if records else None

    except Exception as e:
        print(f"  circular shift {mouse} day {day}: {e}")
        return None


def process_mouse_circular_shift(mouse, mouse_results, n_shifts=N_SHIFTS):
    """Binary participation for all days of one mouse. Returns (mouse, DataFrame or None)."""
    dfs = []
    for day in DAYS:
        events = None
        if mouse_results is not None:
            events = mouse_results.get('days', {}).get(day, {}).get('events', None)
        df = participation_with_shifts(mouse, day, events, n_shifts)
        if df is not None:
            dfs.append(df)
    return mouse, pd.concat(dfs, ignore_index=True) if dfs else None


# ============================================================================
# Spontaneous transient frequency (Supp. 4a-b)
# ============================================================================


def detect_transients(cell_trace):
    """Peak indices of calcium transients in one cell trace.

    The height threshold is set per cell as N_STD_THRESHOLD * std(cell_trace),
    so noisy cells do not get spuriously high transient counts.
    """
    smoothed = savgol_filter(cell_trace, SAVGOL_WINDOW, SAVGOL_ORDER)
    cell_threshold = N_STD_THRESHOLD * np.std(cell_trace)
    peaks, _ = find_peaks(
        smoothed,
        height=cell_threshold,
        distance=MIN_DISTANCE_FRAMES,
        prominence=PROMINENCE_TRANSIENT,
    )
    return peaks


def transient_freq_per_cell(mouse_id, day=0):
    """Spontaneous transient frequency (events/min) of each cell on one day,
    from no-stim trials (raw dF/F). Returns a DataFrame (mouse_id, roi,
    transient_freq)."""
    try:
        xarr = imaging.load_mouse_xarray(
            mouse_id, paths.tensor_dir, 'tensor_xarray_learning_data.nc', subtracted=False
        )
    except Exception as e:
        print(f"  Warning: Could not load data for {mouse_id}: {e}")
        return pd.DataFrame()

    xarr_day = xarr.sel(trial=(xarr['day'] == day) & (xarr['no_stim'] == 1))
    if len(xarr_day.trial) == 0:
        return pd.DataFrame()

    n_cells = len(xarr_day.cell)
    roi_ids = xarr_day['roi'].values
    data = xarr_day.values.reshape(n_cells, -1)
    data = np.nan_to_num(data, nan=0.0)
    session_duration_min = data.shape[1] / SAMPLING_RATE / 60

    rows = []
    for c in range(n_cells):
        n_peaks = len(detect_transients(data[c]))
        rows.append(
            {
                'mouse_id': mouse_id,
                'roi': roi_ids[c],
                'transient_freq': n_peaks / session_duration_min,
            }
        )
    return pd.DataFrame(rows)
