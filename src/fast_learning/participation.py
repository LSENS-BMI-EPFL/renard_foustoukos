"""Participation of individual cells in reactivation events.

For each reactivation event, a cell participates if its baseline-subtracted
dF/F averaged over +/- EVENT_WINDOW_MS around the event reaches its
participation threshold. A cell's participation rate on a day is the fraction
of that day's valid events it participates in (events within EVENT_WINDOW_MS
of a trial edge are skipped).

Participation threshold. Each cell's threshold is a multiple k of its noise:
k = 2.5 (main), 5 and 10 (PARTICIPATION_THRESHOLDS, in noise SD units). The
noise is the robust SD (median absolute deviation x 1.4826) of the cell's
+/- 150 ms mean dF/F over all time points where events are counted, pooled
over the days analysed (noise_sd). The median absolute deviation is barely
affected by the cell's transients, which occupy a small fraction of the time,
so the threshold reflects the cell's noise rather than its activity; a single
value per cell keeps the criterion fixed across days.

Held-out cells (main analysis). Events are moments when population activity
matches the whisker template, and every cell contributes to that match in
proportion to its template weight, so a cell's participation in the events it
helped detect is partly circular. Participation is therefore measured in cells
that took no part in the detection: for each mouse the cells are split at
random into two halves, events are detected with one half only (its template,
its activity and a detection threshold recomputed for that half from
circular-shift surrogates of the pre-learning days, as in step 07), and
participation is measured in the other half; then the halves swap. Over
N_SPLITS random splits each cell is held out N_SPLITS times; its participation
rate on a day is the mean over the splits whose detection had at least
MIN_EVENTS_FOR_RELIABILITY valid events. Participation of all cells in the
events of step 07 (cells='all') is kept for comparison.

Participation above chance (Fig. 6d-e). A cell that is often active crosses the
participation threshold at many moments, events or not. Its chance rate is
the participation expected if events occurred at random times: the fraction
of time points (those where events are counted) at which its +/- 150 ms mean
dF/F reaches the threshold. excess_rate = participation_rate - chance_rate
corrects each cell-day for the cell's activity level on that day.

Only the mice in the participation mouse selection are analysed (see
fast_learning.reactivations); the pipeline step participation.py applies it.
"""

import os
import zlib

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from fast_learning import imaging, paths, reactivations


# ============================================================================
# Parameters
# ============================================================================

DAYS = [-2, -1, 0, 1, 2]
SAMPLING_RATE = 30
EVENT_WINDOW_MS = 150
EVENT_WINDOW_FRAMES = int(EVENT_WINDOW_MS / 1000 * SAMPLING_RATE)
PARTICIPATION_THRESHOLDS = [2.5, 5, 10]  # noise SD units; main value first, then robustness checks
PARTICIPATION_THRESHOLD = PARTICIPATION_THRESHOLDS[0]
MAD_TO_SD = 1.4826  # SD of a Gaussian = 1.4826 x its median absolute deviation
MIN_EVENTS_FOR_RELIABILITY = reactivations.MIN_EVENTS_PER_DAY
MIN_NOSTIM_TRIALS = 10
LMI_POSITIVE_THRESHOLD = 0.975
LMI_NEGATIVE_THRESHOLD = 0.025

# Held-out cells (main analysis): events detected with one random half of the
# cells, participation measured in the other half (see the Held-out section)
N_SPLITS = 10
N_SURROGATES_HALF = 500  # per pre-learning day, for each half's detection threshold
DETECTION_PERCENTILE = 99  # surrogate percentile of the detection threshold (as step 07, p99)
HELDOUT_SEED = 0

# ============================================================================
# Output files (written by the participation pipeline step)
# ============================================================================

RESULTS_DIR = os.path.join(paths.processed_dir, 'reactivation')


# File names. Two independent options:
#   nolick  no-lick control (events of step 07 --nolick, -1 to +1 s), '_nolick'
#   cells   which cells' participation is measured:
#           'heldout'  cells that did not take part in detecting the events
#                      (main analysis, see Held-out cells below), no suffix
#           'insample' the cells that detected those same events, '_insample'
#                      (rates only; comparison)
#           'all'      all cells, in the events of step 07, '_allcells'
#                      (participation not cross-validated; comparison)

CELLS = ('heldout', 'insample', 'all')
_CELLS_SFX = {'heldout': '', 'insample': '_insample', 'all': '_allcells'}


def suffix(nolick=False, cells='heldout'):
    """File-name suffix of a participation output."""
    return _CELLS_SFX[cells] + ('_nolick' if nolick else '')


def heldout_events_pkl(nolick=False):
    """Events detected by each half of the cells, per split and day."""
    return os.path.join(RESULTS_DIR, f'heldout_events{suffix(nolick)}.pkl')


def thr_tag(threshold):
    """File-name tag of a participation threshold (noise SD units), e.g. '2.5sd'."""
    return f'{threshold:g}sd'


def variant_tag(threshold=PARTICIPATION_THRESHOLD, nolick=False, cells='heldout'):
    """Figure-name tag of a participation condition: '' for the main analysis
    (main threshold, whole trial, held-out cells), else e.g. '_5sd' or
    '_2.5sd_insample'."""
    if threshold == PARTICIPATION_THRESHOLD and suffix(nolick, cells) == '':
        return ''
    return f'_{thr_tag(threshold)}{suffix(nolick, cells)}'


def rates_csv(threshold=PARTICIPATION_THRESHOLD, nolick=False, cells='heldout'):
    """Per-cell, per-day participation rates."""
    return os.path.join(
        RESULTS_DIR, f'cell_participation_rates_per_day_{thr_tag(threshold)}{suffix(nolick, cells)}.csv'
    )


def merged_csv(threshold=PARTICIPATION_THRESHOLD, nolick=False, cells='heldout'):
    """Per-cell baseline / day-0 / post participation rates merged with LMI."""
    return os.path.join(
        RESULTS_DIR, f'participation_lmi_merged_{thr_tag(threshold)}{suffix(nolick, cells)}.csv'
    )


def _read(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Run pipeline/08_participation.py first.")
    return pd.read_csv(path)


def load_participation(threshold=PARTICIPATION_THRESHOLD, nolick=False, cells='heldout'):
    """(merged_df, per_day_df) at one participation threshold, with lmi_category."""
    merged_df = _read(merged_csv(threshold, nolick, cells))
    per_day_df = _read(rates_csv(threshold, nolick, cells))
    if 'lmi_category' not in merged_df.columns:
        merged_df = add_lmi_category(merged_df)
    print(
        f"Loaded {len(merged_df)} cells and {len(per_day_df)} cell-day records "
        f"({thr_tag(threshold)}{suffix(nolick, cells)})."
    )
    return merged_df, per_day_df


def excess_tables(merged_df, per_day_df):
    """(merged, per_day) with participation_rate replaced by the rate above
    chance (excess_rate), aggregated and merged with LMI as the rates."""
    per_day = per_day_df.assign(participation_rate=per_day_df['excess_rate'])
    groups = merged_df.groupby('mouse_id')['reward_group'].first().to_dict()
    lmi_df = pd.read_csv(os.path.join(paths.processed_dir, 'lmi_results.csv'))
    merged = merge_with_lmi(aggregate_across_days(per_day), lmi_df, groups)
    return merged, per_day


# ============================================================================
# Participation rates
# ============================================================================


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
# Participation from a cells x trials x time array
# ============================================================================


def participation_from_3d(data_3d, events, n_timepoints, n_trials, threshold):
    """Vectorised participation rate per cell.

    data_3d is (n_cells, n_trials, n_timepoints); events are frame indices in
    the flattened trial x time axis; threshold is a dF/F value, per cell
    (array) or common (scalar). Returns (rates or None, n_valid_events).
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
    rates = np.mean(avg >= threshold, axis=0)
    return rates, len(valid)


def window_means(data_3d):
    """cells x trials x positions: dF/F averaged over +/- EVENT_WINDOW_FRAMES
    around each time point at which events are counted (at least
    EVENT_WINDOW_FRAMES from a trial edge, as participation_from_3d)."""
    win = EVENT_WINDOW_FRAMES
    return sliding_window_view(data_3d, 2 * win + 1, axis=2).mean(axis=-1)


def noise_sd(sub_by_day):
    """Robust SD of each cell's +/- 150 ms mean dF/F: median absolute deviation
    x MAD_TO_SD, over all countable time points of all days in sub_by_day
    ({day: cells x trials x time})."""
    means = np.concatenate([window_means(sub).reshape(len(sub), -1) for sub in sub_by_day.values()], axis=1)
    deviation = np.abs(means - np.median(means, axis=1, keepdims=True))
    return MAD_TO_SD * np.median(deviation, axis=1)


def cell_thresholds(noise, ks=PARTICIPATION_THRESHOLDS):
    """{k: per-cell participation threshold k x noise SD (dF/F)}. Cells without
    measurable noise (constant trace) get an infinite threshold."""
    noise = np.where(noise > 0, noise, np.inf)
    return {k: k * noise for k in ks}


def chance_participation(data_3d, threshold):
    """Per cell: fraction of the time points at which events are counted at
    which its +/- EVENT_WINDOW_FRAMES mean dF/F reaches threshold (per cell or
    common), i.e. its participation rate in events placed at random times."""
    threshold = np.asarray(threshold, dtype=float)
    if threshold.ndim == 1:
        threshold = threshold[:, None, None]
    return (window_means(data_3d) >= threshold).mean(axis=(1, 2))


def chance_rates(mouse, rois, sub_by_day, thresholds):
    """Chance participation of each cell on each day, at each threshold.

    sub_by_day maps day to a cells x trials x time baseline-subtracted array;
    thresholds maps k to per-cell thresholds (cell_thresholds).
    Returns a DataFrame (mouse_id, day, roi, threshold, chance_rate)."""
    parts = []
    for day, sub in sub_by_day.items():
        for threshold, cell_thr in thresholds.items():
            parts.append(
                pd.DataFrame(
                    dict(
                        mouse_id=mouse,
                        day=day,
                        roi=rois,
                        threshold=threshold,
                        chance_rate=chance_participation(sub, cell_thr),
                    )
                )
            )
    return pd.concat(parts, ignore_index=True) if parts else None


def load_subtracted_by_day(mouse, window=None):
    """(rois, {day: cells x trials x time}) baseline-subtracted dF/F of the
    correct-rejection trials, for days with at least MIN_NOSTIM_TRIALS trials."""
    sub_x = imaging.load_mouse_xarray(
        mouse, paths.tensor_dir, 'tensor_xarray_learning_data.nc', subtracted=True
    )
    out = {}
    for day in DAYS:
        sub_tr, n_trials = reactivations.select_trials_by_type(sub_x.sel(trial=sub_x['day'] == day), window)
        if n_trials >= MIN_NOSTIM_TRIALS:
            out[day] = np.nan_to_num(sub_tr.values)
    return sub_x['roi'].values, out


def all_cell_rates(mouse, mouse_results, rois, sub_by_day, thresholds):
    """Participation of all cells in the events of step 07 (cells='all').

    mouse_results is the mouse's entry of a reactivation results file, whose
    event indices refer to the trials of load_subtracted_by_day. Returns a
    DataFrame (mouse_id, day, roi, threshold, participation_rate, n_events,
    reliable) or None.
    """
    parts = []
    for day, sub in sub_by_day.items():
        events = mouse_results.get('days', {}).get(day, {}).get('events', None)
        if events is None or len(events) == 0:
            continue
        _, n_trials, n_t = sub.shape
        for threshold, cell_thr in thresholds.items():
            rates, n_valid = participation_from_3d(sub, events, n_t, n_trials, cell_thr)
            if rates is None:
                continue
            parts.append(
                pd.DataFrame(
                    dict(
                        mouse_id=mouse,
                        day=day,
                        roi=rois,
                        threshold=threshold,
                        participation_rate=rates,
                        n_events=n_valid,
                        reliable=n_valid >= MIN_EVENTS_FOR_RELIABILITY,
                    )
                )
            )
    return pd.concat(parts, ignore_index=True) if parts else None


def add_excess(per_day_df, chance_df, threshold):
    """Add chance_rate and excess_rate (participation_rate - chance_rate)."""
    chance = chance_df.loc[chance_df['threshold'] == threshold, ['mouse_id', 'day', 'roi', 'chance_rate']]
    out = per_day_df.merge(chance, on=['mouse_id', 'day', 'roi'], how='left')
    out['excess_rate'] = out['participation_rate'] - out['chance_rate']
    return out


# ============================================================================
# Held-out cells (cross-validated participation; main analysis)
# ============================================================================


def _split_rng(mouse, *keys):
    return np.random.default_rng([HELDOUT_SEED, zlib.crc32(mouse.encode()), *keys])


def split_halves(mouse, n_cells, split):
    """The two halves (sorted cell indices) of one random split of a mouse's cells."""
    perm = _split_rng(mouse, split).permutation(n_cells)
    return np.sort(perm[: n_cells // 2]), np.sort(perm[n_cells // 2 :])


def load_heldout_data(mouse, window=None):
    """Data of one mouse for the held-out analysis.

    Returns (rois, days): days maps each day with at least MIN_NOSTIM_TRIALS
    correct-rejection trials to a dict with
        raw       cells x frames, raw dF/F of the concatenated trials (detection, as step 07)
        sub       cells x trials x time, baseline-subtracted dF/F (participation)
        template  whisker template of the day (one value per cell)
    """
    raw_x = imaging.load_mouse_xarray(
        mouse, paths.tensor_dir, 'tensor_xarray_learning_data.nc', subtracted=False
    )
    sub_x = imaging.load_mouse_xarray(
        mouse, paths.tensor_dir, 'tensor_xarray_learning_data.nc', subtracted=True
    )
    rois = raw_x['roi'].values
    days = {}
    for day in DAYS:
        raw_tr, n_trials = reactivations.select_trials_by_type(raw_x.sel(trial=raw_x['day'] == day), window)
        if n_trials < MIN_NOSTIM_TRIALS:
            continue
        sub_tr, _ = reactivations.select_trials_by_type(sub_x.sel(trial=sub_x['day'] == day), window)
        template, _ = reactivations.create_whisker_template(mouse, day, reactivations.THRESHOLD_DFF)
        days[day] = dict(
            raw=np.nan_to_num(raw_tr.values.reshape(len(rois), -1)),
            sub=np.nan_to_num(sub_tr.values),
            template=template,
        )
    return rois, days


def _half_threshold(mouse, days, cells, split, percentile=DETECTION_PERCENTILE):
    """Detection threshold for a subset of cells, as step 07: median over
    circular-shift surrogates of the percentile-th percentile of the template
    correlation, pooled over the pre-learning days."""
    pooled = []
    for day in reactivations.PRELEARNING_DAYS:
        if day not in days:
            continue
        res = reactivations.compute_surrogate_thresholds(
            days[day]['raw'][cells],
            days[day]['template'][cells],
            N_SURROGATES_HALF,
            0,
            (percentile,),
            rng=_split_rng(mouse, split, int(cells[0]), day + 10),
        )
        pooled.append(res[percentile]['surrogate_percentiles'])
    return float(np.median(np.concatenate(pooled))) if pooled else np.nan


def detect_heldout_events(mouse, window=None, percentile=DETECTION_PERCENTILE):
    """Reactivation events detected by each half of a mouse's cells.

    For each of N_SPLITS random splits and each half: the events detected with
    that half only (its template, its activity, its own surrogate threshold),
    within each trial as in step 07. percentile: surrogate percentile of the
    detection threshold (99 main; 99.5 / 99.9 robustness). Event indices refer
    to the concatenated trials of load_heldout_data(mouse, window).

    Returns dict(mouse, rois, detections): one detection per split and half,
    dict(split, half, detect, heldout, threshold, events={day: indices}),
    detect and heldout being cell indices.
    """
    rois, days = load_heldout_data(mouse, window)
    detections = []
    for split in range(N_SPLITS):
        halves = split_halves(mouse, len(rois), split)
        for h in (0, 1):
            detect, heldout = halves[h], halves[1 - h]
            threshold = _half_threshold(mouse, days, detect, split, percentile)
            if np.isnan(threshold):
                continue
            events = {}
            for day, d in days.items():
                corr = reactivations.compute_template_correlation(d['raw'][detect], d['template'][detect])
                events[day] = reactivations.detect_reactivation_events(
                    corr,
                    threshold,
                    reactivations.MIN_EVENT_DISTANCE_FRAMES,
                    reactivations.PROMINENCE,
                    n_timepoints=d['sub'].shape[2],
                )
            detections.append(
                dict(split=split, half=h, detect=detect, heldout=heldout, threshold=threshold, events=events)
            )
    return dict(mouse=mouse, rois=rois, detections=detections)


def heldout_split_rates(detection, days, thresholds):
    """Participation per split, cell and day, at each participation threshold
    (thresholds: {k: per-cell thresholds}, see cell_thresholds).

    role 'heldout': cells outside the detecting half; role 'insample': the
    detecting half itself (same events, for comparison). Returns a DataFrame
    (mouse_id, split, half, role, roi, day, threshold, participation_rate,
    n_events), n_events being the valid events of that detection.
    """
    rois = detection['rois']
    parts = []
    for det in detection['detections']:
        for day, events in det['events'].items():
            sub = days[day]['sub']
            _, n_trials, n_t = sub.shape
            for threshold, cell_thr in thresholds.items():
                rates, n_valid = participation_from_3d(sub, events, n_t, n_trials, cell_thr)
                if rates is None:
                    continue
                for role, idx in (('heldout', det['heldout']), ('insample', det['detect'])):
                    parts.append(
                        pd.DataFrame(
                            dict(
                                mouse_id=detection['mouse'],
                                split=det['split'],
                                half=det['half'],
                                role=role,
                                roi=rois[idx],
                                day=day,
                                threshold=threshold,
                                participation_rate=rates[idx],
                                n_events=n_valid,
                            )
                        )
                    )
    return pd.concat(parts, ignore_index=True) if parts else None


def average_over_splits(split_df):
    """Per-cell, per-day participation rate: mean over the splits whose
    detection had at least MIN_EVENTS_FOR_RELIABILITY valid events; cell-days
    with no such split are left out. Same format as the all-cell rates
    (rates_csv), plus n_splits."""
    d = split_df[split_df['n_events'] >= MIN_EVENTS_FOR_RELIABILITY]
    out = (
        d.groupby(['mouse_id', 'day', 'roi'])
        .agg(
            participation_rate=('participation_rate', 'mean'),
            n_events=('n_events', 'mean'),
            n_splits=('split', 'nunique'),
        )
        .reset_index()
    )
    out['reliable'] = True
    return out
