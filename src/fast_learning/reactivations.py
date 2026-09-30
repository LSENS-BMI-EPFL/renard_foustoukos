"""Reactivation analysis helpers.

Mouse selection for the participation analyses
----------------------------------------------
Participation rates are computed per reactivation event, so a mouse with
almost no events on day 0 has undefined or unreliable day-0 rates (with one
event, every cell's rate is exactly 0 or 1). Mice with fewer than
MIN_DAY0_EVENTS events on day 0, at the main detection threshold, are
therefore excluded from every participation analysis (Fig. 4i-j, Supp. 4a-c
and their revisions). Event-rate analyses (Fig. 4h and its controls) keep all
mice: there, zero events is a valid measurement, and excluding mice on it
would select on the outcome.

Within the selected mice, a mouse-day enters the per-day participation
statistics only if it has at least MIN_EVENTS_PER_DAY valid events (events
closer than 150 ms to a trial edge are not used). This is the same threshold,
so every day-0 panel uses exactly the selected mice.

pipeline/07_reactivations.py writes the selection to
<processed_dir>/reactivation/mouse_selection.csv; the participation step
reads it with load_participation_mice().

Reactivation detection
----------------------
A whisker template (mean day-specific response of each cell to the last
N_MAP_TRIALS passive whisker stimuli, 0-300 ms) is correlated frame by frame
with population activity during no-stim trials. Reactivation events are peaks
of the smoothed correlation above a per-mouse threshold, set from circular-shift
surrogates of the pre-learning days (see compute_surrogate_thresholds). The
reactivation pipeline step runs detection for all mice.
"""

import os
import zlib

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, savgol_filter
from scipy.stats import percentileofscore

from fast_learning import imaging, paths


# ============================================================================
# Parameters
# ============================================================================

# Imaging / template
SAMPLING_RATE = 30  # Hz
WIN = (0, 0.300)  # Template time window (stimulus onset → 300 ms)
N_MAP_TRIALS = 40  # Mapping trials used to build the template
THRESHOLD_DFF = None  # dF/F threshold for template cells (None = all cells)

# Event detection
THRESHOLD_CORR = 0.45  # Fixed correlation threshold, used only without surrogate thresholds
MIN_EVENT_DISTANCE_MS = 150
MIN_EVENT_DISTANCE_FRAMES = int(MIN_EVENT_DISTANCE_MS / 1000 * SAMPLING_RATE)
PROMINENCE = 0.15

# Surrogates
DAYS = [-2, -1, 0, 1, 2]
PRELEARNING_DAYS = [-2, -1]
SURROGATE_SEED = 0  # Base seed; each (mouse, day) gets its own stream

# Mouse selection for the participation analyses (see module docstring)
MIN_DAY0_EVENTS = 3
MIN_EVENTS_PER_DAY = MIN_DAY0_EVENTS
SELECTION_DAY = 0
MOUSE_SELECTION_CSV = os.path.join(paths.processed_dir, 'reactivation', 'mouse_selection.csv')


# ============================================================================
# Helpers
# ============================================================================


def percentile_tag(p):
    return f"p{int(p)}" if p == int(p) else f"p{int(p * 10)}"


def surrogate_rng(mouse, day):
    """Random generator for the surrogates of one mouse and day.

    Seeded from SURROGATE_SEED, the mouse id and the day, so results do not
    depend on how mice are distributed across parallel workers.
    """
    return np.random.default_rng([SURROGATE_SEED, zlib.crc32(mouse.encode()), day + 10])


# ============================================================================
# Core imaging functions
# ============================================================================


def create_whisker_template(mouse, day, threshold_dff=THRESHOLD_DFF, verbose=False):
    """Create whisker response template from mapping data for a specific day."""
    folder = paths.tensor_dir
    xarray_map = imaging.load_mouse_xarray(mouse, folder, 'tensor_xarray_mapping_data.nc', subtracted=True)

    xarray_day = xarray_map.sel(trial=xarray_map['day'] == day)
    xarray_day = xarray_day.groupby('day').apply(lambda x: x.isel(trial=slice(-N_MAP_TRIALS, None)))

    d = xarray_day.sel(time=slice(WIN[0], WIN[1])).mean(dim='time').fillna(0)
    template = d.mean(dim='trial').values

    if threshold_dff is None:
        cells_mask = np.ones(len(template), dtype=bool)
        template_filtered = template.copy()
    else:
        cells_mask = template >= threshold_dff
        template_filtered = template.copy()
        template_filtered[~cells_mask] = 0

    return template_filtered, cells_mask


def compute_template_correlation(data, template):
    """Compute Pearson correlation between neural activity and template (vectorized)."""
    n_cells, n_timepoints = data.shape
    template_std = np.std(template)
    if template_std == 0:
        return np.zeros(n_timepoints)

    template_centered = template - np.mean(template)
    data_centered = data - np.mean(data, axis=0, keepdims=True)
    data_stds = np.std(data, axis=0)

    correlations = np.dot(template_centered, data_centered) / (template_std * data_stds * n_cells)
    correlations[data_stds == 0] = 0
    return correlations


def select_trials_by_type(xarray_day, no_lick_only=False, time_window=None):
    """Select no-stim trials, optionally restricted to no-lick (correct
    rejection) trials and/or a time window around no-stim onset."""
    mask = xarray_day['no_stim'] == 1
    if no_lick_only:
        mask = mask & (xarray_day['lick_flag'] == 0)
    selected = xarray_day.sel(trial=mask)
    if time_window is not None:
        selected = selected.sel(time=slice(time_window[0], time_window[1]))
    return selected, len(selected.trial)


def load_selected_trials(mouse, day, no_lick_only=False, time_window=None, folder=None):
    """No-stim trials of one mouse-day (raw dF/F, cells x trials x time), as
    used for reactivation detection. Event indices of a reactivation results
    file refer to these trials' concatenated trial x time axis, so pass the
    no_lick_only / time_window of that file (its 'parameters' entry)."""
    xarray_learning = imaging.load_mouse_xarray(
        mouse, folder or paths.tensor_dir, 'tensor_xarray_learning_data.nc', subtracted=False
    )
    xarray_day = xarray_learning.sel(trial=xarray_learning['day'] == day)
    selected, _ = select_trials_by_type(xarray_day, no_lick_only=no_lick_only, time_window=time_window)
    return selected


def detect_reactivation_events(
    correlations, threshold, min_distance, prominence, smooth=True, window_length=5, polyorder=2
):
    """Detect reactivation events as peaks in correlation timeseries."""
    if smooth:
        if window_length % 2 == 0:
            window_length += 1
        if window_length > len(correlations):
            window_length = len(correlations) if len(correlations) % 2 == 1 else len(correlations) - 1
        if window_length < polyorder + 2:
            smoothed_corr = correlations
        else:
            smoothed_corr = savgol_filter(correlations, window_length, polyorder)
    else:
        smoothed_corr = correlations

    peaks, _ = find_peaks(smoothed_corr, height=threshold, distance=min_distance, prominence=prominence)
    return peaks


def compute_time_above_threshold(correlations, threshold):
    """Compute percentage of time spent above correlation threshold."""
    above = correlations > threshold
    n_above = np.sum(above)
    total = len(correlations)
    pct = (n_above / total * 100) if total > 0 else 0.0
    return pct, n_above, total


def map_events_to_blocks(event_indices, nostim_trials, n_timepoints_per_trial, sampling_rate=SAMPLING_RATE):
    """Map event indices to block IDs and count events per block."""
    block_ids = nostim_trials['block_id'].values
    event_trials = event_indices // n_timepoints_per_trial

    event_blocks = [block_ids[t] for t in event_trials if t < len(block_ids)]

    unique_blocks = np.unique(block_ids)
    events_per_block = {int(b): 0 for b in unique_blocks}
    trials_per_block = {int(b): 0 for b in unique_blocks}

    for b in block_ids:
        trials_per_block[int(b)] += 1
    for b in event_blocks:
        if b in events_per_block:
            events_per_block[int(b)] += 1

    event_frequency_per_block = {
        int(b): events_per_block[int(b)]
        / (trials_per_block[int(b)] * n_timepoints_per_trial / sampling_rate / 60)
        for b in unique_blocks
    }
    return events_per_block, event_frequency_per_block, event_blocks


def compute_time_above_per_block(correlations, threshold, trial_block_ids, n_timepoints_per_trial):
    """Compute percentage of time above threshold per block."""
    unique_blocks = np.unique(trial_block_ids)
    percent_time_per_block = {}
    for block in unique_blocks:
        trials_in_block = np.where(trial_block_ids == block)[0]
        block_corr = np.concatenate(
            [
                correlations[t * n_timepoints_per_trial : (t + 1) * n_timepoints_per_trial]
                for t in trials_in_block
            ]
        )
        pct = np.sum(block_corr > threshold) / len(block_corr) * 100 if len(block_corr) > 0 else 0.0
        percent_time_per_block[int(block)] = pct
    return percent_time_per_block


def get_block_boundaries(nostim_trials, n_timepoints_per_trial):
    """Find frame indices where block transitions occur."""
    block_ids = nostim_trials['block_id'].values
    return [i * n_timepoints_per_trial for i in range(1, len(block_ids)) if block_ids[i] != block_ids[i - 1]]


def extract_performance_per_block(nostim_trials):
    """Extract whisker hit rate per block."""
    block_ids = nostim_trials['block_id'].values
    hr_w = nostim_trials['hr_w'].values
    hr_per_block = {}
    for block in np.unique(block_ids):
        vals = hr_w[block_ids == block]
        valid = vals[~np.isnan(vals)]
        if len(valid) > 0:
            hr_per_block[int(block)] = valid[0]
    return hr_per_block


def compute_reactivation_frequency_per_trial(
    selected_trials,
    template,
    threshold,
    time_bin_ms=500,
    sampling_rate=SAMPLING_RATE,
    min_distance=MIN_EVENT_DISTANCE_FRAMES,
    prominence=PROMINENCE,
):
    """Compute average reactivation frequency as function of time within trials."""
    n_cells, n_trials, n_timepoints = selected_trials.shape
    frames_per_bin = int(time_bin_ms / 1000.0 * sampling_rate)
    n_bins = n_timepoints // frames_per_bin

    if n_bins < 1:
        return np.array([]), np.array([]), np.array([]), n_trials

    n_frames = n_bins * frames_per_bin
    events_per_trial = np.zeros((n_trials, n_bins))

    for trial_idx in range(n_trials):
        trial_data = np.nan_to_num(selected_trials[:, trial_idx, :n_frames].values, nan=0.0)
        corr = compute_template_correlation(trial_data, template)
        events = detect_reactivation_events(corr, threshold, min_distance, prominence)
        for bin_idx in range(n_bins):
            events_per_trial[trial_idx, bin_idx] = np.sum(
                (events >= bin_idx * frames_per_bin) & (events < (bin_idx + 1) * frames_per_bin)
            )

    time_bins = (np.arange(n_bins) + 0.5) * frames_per_bin / sampling_rate
    bin_dur = frames_per_bin / sampling_rate
    rate = events_per_trial / bin_dur
    return time_bins, np.mean(rate, axis=0), np.std(rate, axis=0) / np.sqrt(n_trials), n_trials


# ============================================================================
# Surrogate threshold loading
# ============================================================================


def load_surrogate_thresholds(surrogate_csv_path, percentile=99):
    """Load percentile-based thresholds from surrogate CSV.

    surrogate_csv_path may already carry the percentile suffix (e.g.
    ..._p995.csv) or not, in which case it is appended.
    """
    ps = percentile_tag(percentile)
    if surrogate_csv_path.endswith(f'_{ps}.csv'):
        final_path = surrogate_csv_path
    else:
        base = surrogate_csv_path[:-4] if surrogate_csv_path.endswith('.csv') else surrogate_csv_path
        final_path = f"{base}_{ps}.csv"

    if not os.path.exists(final_path):
        raise FileNotFoundError(f"Surrogate threshold file not found: {final_path}")

    df = pd.read_csv(final_path)
    threshold_col = 'threshold_percentile_median'
    all_days = [-2, -1, 0, 1, 2]
    threshold_dict = {}

    if 'day' in df.columns:
        for _, row in df.iterrows():
            m, d = row['mouse_id'], int(row['day'])
            threshold_dict.setdefault(m, {})[d] = row[threshold_col]
    else:
        for _, row in df.iterrows():
            threshold_dict[row['mouse_id']] = {d: row[threshold_col] for d in all_days}

    return threshold_dict


def get_threshold_for_mouse_day(threshold_dict, mouse, day, default_threshold=THRESHOLD_CORR):
    """Get per-mouse, per-day threshold with fallback to default."""
    if threshold_dict is None:
        return default_threshold
    if mouse in threshold_dict and day in threshold_dict[mouse]:
        return threshold_dict[mouse][day]
    return default_threshold


# ============================================================================
# Reactivation event detection (per mouse)
# ============================================================================


def analyze_mouse_reactivation(
    mouse, days=DAYS, verbose=False, threshold_dict=None, no_lick_only=False, time_window=None
):
    """
    Detect reactivation events for a single mouse across all days.

    Returns a nested dict: {mouse, days: {day: {correlations, events, ...}}}.
    The trial data the events refer to are not stored; load_selected_trials()
    rebuilds them from the tensors.
    """
    results = {'mouse': mouse, 'days': {}}
    folder = paths.tensor_dir

    for day in days:
        try:
            template, cells_mask = create_whisker_template(mouse, day, THRESHOLD_DFF, verbose=verbose)

            selected_trials = load_selected_trials(
                mouse, day, no_lick_only=no_lick_only, time_window=time_window, folder=folder
            )
            n_selected_trials = len(selected_trials.trial)
            if n_selected_trials == 0:
                continue

            n_cells, n_trials, n_timepoints = selected_trials.shape
            data = np.nan_to_num(selected_trials.values.reshape(n_cells, -1), nan=0.0)

            correlations = compute_template_correlation(data, template)

            current_threshold = get_threshold_for_mouse_day(threshold_dict, mouse, day, THRESHOLD_CORR)
            events = detect_reactivation_events(
                correlations, current_threshold, MIN_EVENT_DISTANCE_FRAMES, PROMINENCE
            )

            pct_above, n_above, total_frames = compute_time_above_threshold(correlations, current_threshold)
            events_per_block, event_freq_per_block, event_blocks = map_events_to_blocks(
                events, selected_trials, n_timepoints, SAMPLING_RATE
            )
            trial_block_ids = selected_trials['block_id'].values
            pct_per_block = compute_time_above_per_block(
                correlations, current_threshold, trial_block_ids, n_timepoints
            )
            block_boundaries = get_block_boundaries(selected_trials, n_timepoints)
            hr_per_block = extract_performance_per_block(selected_trials)

            session_dur_sec = (n_trials * n_timepoints) / SAMPLING_RATE
            event_frequency = len(events) / (session_dur_sec / 60)

            time_bins, event_rate, event_rate_sem, n_trials_temporal = (
                compute_reactivation_frequency_per_trial(
                    selected_trials,
                    template,
                    current_threshold,
                    sampling_rate=SAMPLING_RATE,
                    min_distance=MIN_EVENT_DISTANCE_FRAMES,
                    prominence=PROMINENCE,
                )
            )

            results['days'][day] = {
                'template': template,
                'cells_mask': cells_mask,
                'correlations': correlations,
                'events': events,
                'events_per_block': events_per_block,
                'event_frequency_per_block': event_freq_per_block,
                'percent_time_above': pct_above,
                'n_frames_above': n_above,
                'total_frames': total_frames,
                'percent_time_per_block': pct_per_block,
                'hr_per_block': hr_per_block,
                'block_boundaries': block_boundaries,
                'n_trials': n_selected_trials,
                'n_timepoints': n_timepoints,
                'total_events': len(events),
                'session_duration_sec': session_dur_sec,
                'session_duration_min': session_dur_sec / 60,
                'event_frequency': event_frequency,
                'session_hr_mean': (np.mean(list(hr_per_block.values())) if hr_per_block else np.nan),
                'threshold_used': current_threshold,
                'temporal': {
                    'time_bins': time_bins,
                    'event_rate': event_rate,
                    'event_rate_sem': event_rate_sem,
                    'n_trials': n_trials_temporal,
                },
            }

        except Exception as e:
            print(f"  WARNING {mouse} day {day} skipped: {e!r}")
            if verbose:
                import traceback

                traceback.print_exc()
            continue

    return results


# ============================================================================
# Surrogate computation functions
# ============================================================================


def create_surrogate_by_circular_shift(data, min_shift_frames=0, rng=None):
    """Create one surrogate by independently circular-shifting each cell."""
    if rng is None:
        rng = np.random.default_rng()
    n_cells, n_frames = data.shape
    surrogate = np.zeros_like(data)
    for i in range(n_cells):
        shift = rng.integers(min_shift_frames if min_shift_frames > 0 else 1, n_frames)
        surrogate[i, :] = np.roll(data[i, :], shift)
    return surrogate


def compute_surrogate_thresholds(
    data, template, n_surrogates=1000, min_shift=0, percentiles=(99,), verbose=False, rng=None
):
    """
    Compute surrogate-based thresholds for multiple percentiles via circular shift.
    Returns dict keyed by percentile value.
    """
    if rng is None:
        rng = np.random.default_rng()
    observed_corr = compute_template_correlation(data, template)
    observed_pcts = {p: np.percentile(observed_corr, p) for p in percentiles}
    surrogate_pcts = {p: np.zeros(n_surrogates) for p in percentiles}

    for i in range(n_surrogates):
        surr = create_surrogate_by_circular_shift(data, min_shift, rng=rng)
        surr_corr = compute_template_correlation(surr, template)
        for p in percentiles:
            surrogate_pcts[p][i] = np.percentile(surr_corr, p)

    results = {}
    for p in percentiles:
        med = np.median(surrogate_pcts[p])
        ci = (np.percentile(surrogate_pcts[p], 2.5), np.percentile(surrogate_pcts[p], 97.5))
        pval = percentileofscore(surrogate_pcts[p], observed_pcts[p]) / 100.0
        results[p] = {
            'threshold_percentile_median': med,
            'threshold_percentile_ci': ci,
            'surrogate_percentiles': surrogate_pcts[p],
            'observed_percentile': observed_pcts[p],
            'p_value_percentile': pval,
            'percentile_value': p,
        }
    return results


def analyze_surrogates_per_day(
    mouse,
    days=DAYS,
    threshold_dff=THRESHOLD_DFF,
    n_surrogates=1000,
    percentiles=(99,),
    verbose=False,
    no_lick_only=False,
    time_window=None,
):
    """
    Compute per-day surrogate thresholds for one mouse.
    Returns (results_dfs, all_surrogate_data) where results_dfs is {percentile: DataFrame}.
    """
    folder = paths.tensor_dir
    results_lists = {p: [] for p in percentiles}
    all_surrogate_data = {p: {} for p in percentiles}

    for day in days:
        try:
            template, cells_mask = create_whisker_template(mouse, day, threshold_dff, verbose=verbose)
            n_resp = cells_mask.sum()
            if n_resp < 3 and threshold_dff is not None:
                continue

            xarray_learning = imaging.load_mouse_xarray(
                mouse, folder, 'tensor_xarray_learning_data.nc', subtracted=False
            )
            xarray_day = xarray_learning.sel(trial=xarray_learning['day'] == day)
            nostim, n_trials = select_trials_by_type(
                xarray_day, no_lick_only=no_lick_only, time_window=time_window
            )
            if n_trials < 5:
                continue

            n_cells, _, n_timepoints = nostim.shape
            data = np.nan_to_num(nostim.values.reshape(n_cells, -1), nan=0.0)
            n_frames = data.shape[1]
            if n_frames < 60:
                continue

            surrogate_results = compute_surrogate_thresholds(
                data, template, n_surrogates, 0, percentiles, verbose, rng=surrogate_rng(mouse, day)
            )

            for p in percentiles:
                pr = surrogate_results[p]
                results_lists[p].append(
                    {
                        'mouse_id': mouse,
                        'day': day,
                        'n_cells_responsive': n_resp,
                        'n_trials': n_trials,
                        'n_frames': n_frames,
                        'n_surrogates': n_surrogates,
                        'percentile_value': p,
                        'threshold_percentile_median': pr['threshold_percentile_median'],
                        'threshold_percentile_ci_lower': pr['threshold_percentile_ci'][0],
                        'threshold_percentile_ci_upper': pr['threshold_percentile_ci'][1],
                        'observed_percentile': pr['observed_percentile'],
                        'p_value_percentile': pr['p_value_percentile'],
                    }
                )
                all_surrogate_data[p][day] = pr

        except Exception as e:
            print(f"  WARNING {mouse} day {day} skipped (per-day surrogates): {e!r}")
            continue

    if all(len(results_lists[p]) == 0 for p in percentiles):
        return None, None

    results_dfs = {p: pd.DataFrame(results_lists[p]) for p in percentiles}
    return results_dfs, all_surrogate_data


def process_mouse_surrogates_per_day(
    mouse,
    days,
    threshold_dff,
    n_surrogates,
    percentiles=(99,),
    verbose=False,
    no_lick_only=False,
    time_window=None,
):
    """Parallel wrapper for per-day surrogates."""
    results_dfs, surrogate_data = analyze_surrogates_per_day(
        mouse,
        days=days,
        threshold_dff=threshold_dff,
        n_surrogates=n_surrogates,
        percentiles=percentiles,
        verbose=verbose,
        no_lick_only=no_lick_only,
        time_window=time_window,
    )
    return (mouse, results_dfs, surrogate_data)


def analyze_surrogates_per_mouse(
    mouse,
    threshold_dff=THRESHOLD_DFF,
    n_surrogates=1000,
    percentiles=(99,),
    verbose=False,
    no_lick_only=False,
    time_window=None,
):
    """
    Compute single per-mouse surrogate threshold using pre-learning days pooled.
    Returns (results_dfs, surrogate_data) where results_dfs is {percentile: DataFrame}.
    """
    folder = paths.tensor_dir
    try:
        xarray_learning = imaging.load_mouse_xarray(
            mouse, folder, 'tensor_xarray_learning_data.nc', subtracted=False
        )
    except Exception as e:
        if verbose:
            print(f"  Error loading xarray for {mouse}: {e}")
        return None, None

    pooled_surr = {p: [] for p in percentiles}
    pooled_obs = {p: [] for p in percentiles}
    total_frames = 0
    days_processed = 0

    for day in PRELEARNING_DAYS:
        try:
            template, cells_mask = create_whisker_template(mouse, day, threshold_dff, verbose=verbose)
            n_resp = cells_mask.sum()
            if n_resp < 3 and threshold_dff is not None:
                continue

            xarray_day = xarray_learning.sel(trial=xarray_learning['day'] == day)
            nostim, n_nostim_trials = select_trials_by_type(
                xarray_day, no_lick_only=no_lick_only, time_window=time_window
            )
            if n_nostim_trials < 5:
                continue

            n_cells = nostim.shape[0]
            data = np.nan_to_num(nostim.values.reshape(n_cells, -1), nan=0.0)
            if data.shape[1] < 60:
                continue

            day_results = compute_surrogate_thresholds(
                data, template, n_surrogates, 0, percentiles, verbose, rng=surrogate_rng(mouse, day)
            )

            for p in percentiles:
                pooled_surr[p].append(day_results[p]['surrogate_percentiles'])
                pooled_obs[p].append(day_results[p]['observed_percentile'])

            total_frames += data.shape[1]
            days_processed += 1

        except Exception as e:
            print(f"  WARNING {mouse} day {day} skipped (per-mouse surrogates): {e!r}")
            continue

    if days_processed == 0:
        return None, None

    results_dfs = {}
    surrogate_data = {}
    for p in percentiles:
        if not pooled_surr[p]:
            continue
        pooled = np.concatenate(pooled_surr[p])
        observed = np.median(pooled_obs[p])
        threshold = np.median(pooled)
        ci = np.percentile(pooled, [2.5, 97.5])
        pval = np.mean(pooled > observed)

        results_dfs[p] = pd.DataFrame(
            [
                {
                    'mouse_id': mouse,
                    'n_frames': total_frames,
                    'n_days_processed': days_processed,
                    'n_surrogates_per_day': n_surrogates,
                    'percentile_value': p,
                    'threshold_percentile_median': threshold,
                    'threshold_percentile_ci_lower': ci[0],
                    'threshold_percentile_ci_upper': ci[1],
                    'observed_percentile': observed,
                    'p_value_percentile': pval,
                }
            ]
        )
        surrogate_data[p] = {
            'threshold_percentile_median': threshold,
            'threshold_percentile_ci': ci,
            'surrogate_percentiles': pooled,
            'observed_percentile': observed,
            'p_value_percentile': pval,
            'percentile_value': p,
        }

    return results_dfs, surrogate_data


def process_mouse_surrogates_per_mouse(
    mouse, threshold_dff, n_surrogates, percentiles=(99,), verbose=False, no_lick_only=False, time_window=None
):
    """Parallel wrapper for per-mouse surrogates."""
    results_dfs, surrogate_data = analyze_surrogates_per_mouse(
        mouse,
        threshold_dff=threshold_dff,
        n_surrogates=n_surrogates,
        percentiles=percentiles,
        verbose=verbose,
        no_lick_only=no_lick_only,
        time_window=time_window,
    )
    return (mouse, results_dfs, surrogate_data)


# ============================================================================
# Helpers for collecting parallel results
# ============================================================================


def collect_surrogate_results(results_list, percentiles):
    """Aggregate per-mouse DataFrames from parallel results into one per percentile."""
    all_results = {p: [] for p in percentiles}
    for _mouse, results_dfs, _surrogate_data in results_list:
        if results_dfs is not None:
            for p in percentiles:
                if p in results_dfs:
                    all_results[p].append(results_dfs[p])
    return {p: pd.concat(dfs, ignore_index=True) for p, dfs in all_results.items() if dfs}


# ============================================================================
# Mouse selection for the participation analyses
# ============================================================================


def compute_mouse_selection(results_data, min_events=MIN_DAY0_EVENTS, day=SELECTION_DAY):
    """One row per mouse: reward group, number of events on `day`, included.

    results_data is the content of a reactivation_results_p*.pkl file.
    A day missing from a mouse's results counts as zero events.
    """
    rows = []
    for key, group in [('r_plus_results', 'R+'), ('r_minus_results', 'R-')]:
        for mouse, res in results_data[key].items():
            n_events = int(res.get('days', {}).get(day, {}).get('total_events', 0))
            rows.append(
                {
                    'mouse_id': mouse,
                    'reward_group': group,
                    f'n_events_day{day}': n_events,
                    'included': n_events >= min_events,
                }
            )
    return pd.DataFrame(rows)


def load_participation_mice(path=MOUSE_SELECTION_CSV):
    """Set of mouse ids included in the participation analyses."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Mouse selection not found: {path}\n"
            "Run pipeline/07_reactivations.py (or its --selection-only option) first."
        )
    df = pd.read_csv(path)
    return set(df.loc[df['included'], 'mouse_id'])
