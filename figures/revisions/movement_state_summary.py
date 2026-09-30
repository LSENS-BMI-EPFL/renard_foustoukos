"""
Reviewer check: facial/whisker movement during the passive mapping epoch,
across days.

Addresses the concern that the brain state during the post-session passive
whisker-stimulation block (cf. figure_3c.py, figure_4) might not be matched
across training days. This complements
figures/revisions/behavior_state_summary.py (session duration, trial
count, total water), by directly quantifying movement during the passive
epoch itself.

Data source: DeepLabCut trial matrices (not NWB), one .h5 file per mouse in
DLC_DIR, produced independently by G. Foustoukos. The .h5 layout (traces,
feature_names, time, per-session norm_sessions/norm_constants, trials) and
the loading logic below were verified against his load_dlc.py in that same
directory, then reimplemented here (_load_dlc_h5) so this script has no
import-time dependency on that external file. Available for 12 mice only
(a subset of the full cohort -- mice without a DLC file are silently
skipped). Of those 12, only one (GF319) is R-, so an R+ vs R- group
comparison is not statistically meaningful from this dataset alone (n=1);
this script therefore reports R+ mice only, across days. Each file holds,
per recording day relative to the whisker learning day (-2, -1, 0, +1, +2),
one array of shape (n_trials, n_features, n_time) for the 50 passive/
"unmotivated" (UM) whisker trials, sampled at 100 Hz with the whisker
stimulus at time 0.

Features are speed and amplitude for 5 DeepLabCut-tracked body parts
(tongue, whisker end, nose, whisker base, jaw end) -- 10 features total.
Traces are used raw (NORMALISED=False; Georgios's per-session (v-mean)/
(max-min) rescaling is not applied here). For each trial, four raw numbers
are computed per feature: the pre-stimulus BASELINE_WINDOW mean (-2 to 0 s),
the post-stimulus WINDOW mean (0 to 1 s), the post-stimulus WINDOW peak
(max), and the post-stimulus WINDOW RMS deviation from that trial's own
baseline. Six summary statistics are derived from these and each plotted/
saved separately (one svg + one data csv each), since they answer slightly
different questions:
    mean        -- post-stimulus window mean, no baseline correction
    peak        -- post-stimulus window peak, no baseline correction
    baseline    -- pre-stimulus baseline mean (ongoing idle movement level,
                   not a stimulus response)
    blsub_mean  -- post-stimulus mean minus that trial's own baseline
    blsub_peak  -- post-stimulus peak minus that trial's own baseline
    energy      -- RMS of the post-stimulus trace around that trial's own
                   baseline (unsigned "motion energy": movement in both
                   directions adds up instead of cancelling, unlike
                   blsub_mean/blsub_peak)

Each panel (one per body part x metric) is also annotated with a Friedman
test (nonparametric repeated-measures ANOVA, matched by mouse) for whether
the 5 days differ, rather than a Kruskal-Wallis test across days -- the 11
R+ mice are the same across all 5 days, so treating each day as an
independent sample (as Kruskal-Wallis would) ignores that repeated-measures
structure, the same pseudoreplication issue already flagged and fixed for
the reactivation panels (see figure_4i_j_lmm.py's docstring); mice missing
any of the 5 days are dropped from that panel's test (complete-case).
"""

import os

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import t as t_dist
from scipy.stats import friedmanchisquare

from fast_learning import paths, database
from fast_learning.plotting import reward_palette, save_figure
from fast_learning.stats import significance_stars as _significance_stars

DLC_DIR = '/mnt/lsens-analysis/Anthony_Renard/DLCTrialMatrices'


# ============================================================================
# Parameters
# ============================================================================

DAYS = [-2, -1, 0, 1, 2]
WINDOW = (0.0, 1.0)  # post-stimulus window, seconds relative to stimulus onset
BASELINE_WINDOW = (-2.0, 0.0)  # pre-stimulus baseline window
NORMALISED = False  # per-session (v - mean) / (max - min), see load_dlc.py; off here
MIN_TRIALS = 5  # skip a mouse x day with fewer usable trials
OUTPUT_DIR = os.path.join(paths.results_dir, 'behavior', 'passive_epoch_movement')

# (value column, y-axis phrase, output filename suffix)
STATS = [
    ('mean', 'window mean', 'mean'),
    ('peak', 'window peak', 'peak'),
    ('baseline', 'pre-stim baseline', 'baseline'),
    ('blsub_mean', 'baseline-subtracted mean', 'blsub_mean'),
    ('blsub_peak', 'baseline-subtracted peak', 'blsub_peak'),
    ('energy', 'RMS from baseline (motion energy)', 'energy'),
]


# ============================================================================
# Data
# ============================================================================


def discover_dlc_mice(dlc_dir=DLC_DIR):
    """Mouse ids for every '<mouse_id>_dlc.h5' file found in dlc_dir."""
    suffix = '_dlc.h5'
    return sorted(f[: -len(suffix)] for f in os.listdir(dlc_dir) if f.endswith(suffix))


def _text(values):
    """Decode an array of bytes/str h5py values to plain str."""
    return np.array([v.decode() if isinstance(v, bytes) else v for v in values])


def _load_dlc_h5(path, days=DAYS, normalised=NORMALISED):
    """Load one mouse's DeepLabCut trial matrix, split by recording day.

    Reimplements Georgios Foustoukos's load_dlc.load() (see DLC_DIR) inline,
    so this script has no dependency on that external file. Traces are
    (n_trials, n_features, n_time); every trial is a baseline trial with
    usable video (no NaN/padding), so trial counts differ across days.

    The first and last time sample of every trial's trace show a
    consistent edge artifact (likely a boundary effect of whatever
    smoothing/differentiation produced the speed/amplitude traces
    upstream), so both are overwritten with their nearest interior
    neighbour before anything else is computed from these traces.

    If normalised, each session's traces are rescaled per feature as
    (v - session_mean) / (session_max - session_min), using the constants
    the original DataJoint pipeline stored (norm_sessions/norm_constants),
    computed over the full original session, not just these passive trials.

    Returns
    -------
    traces_by_day : {day_offset: ndarray (n_trials, n_features, n_time)}
    features      : list of str feature names (axis 1 of each array)
    time          : ndarray (n_time,), seconds relative to stimulus onset
    """
    with h5py.File(path, 'r') as handle:
        traces = handle['traces'][...]
        features = [str(f) for f in _text(handle['feature_names'][...])]
        time = handle['time'][...]
        sessions = list(_text(handle['norm_sessions'][...]))
        constants = handle['norm_constants'][...]

        columns = {}
        for name in handle['trials']:
            values = handle['trials'][name][...]
            if values.dtype.kind in 'SO':
                values = _text(values)
            columns[name] = values
    trials = pd.DataFrame(columns)

    # Edge artifact: pad the first/last time sample with its neighbour.
    # (traces[...] above already returned an independent, writable array.)
    traces[:, :, 0] = traces[:, :, 1]
    traces[:, :, -1] = traces[:, :, -2]

    if normalised:
        for session_id, rows in trials.groupby('session_id', sort=False):
            block = constants[sessions.index(session_id)]
            span = block[:, 2] - block[:, 1]
            span = np.where(span == 0, np.nan, span)
            index = rows.index.to_numpy()
            traces[index] = (traces[index] - block[:, 0][:, None]) / span[:, None]

    traces_by_day = {}
    for day in days:
        mask = (trials['day_offset'] == day).to_numpy()
        if mask.any():
            traces_by_day[day] = traces[mask]

    return traces_by_day, features, time


def load_trial_metrics(
    mouse_id, dlc_dir=DLC_DIR, window=WINDOW, baseline_window=BASELINE_WINDOW, normalised=NORMALISED
):
    """Per-trial baseline mean, post-stimulus mean, post-stimulus peak, and
    post-stimulus RMS-from-baseline of each DLC feature, for every day
    available for this mouse.

    RMS-from-baseline (post_energy) is unsigned "motion energy": the
    post-stimulus trace's deviation from that trial's own baseline is
    squared, averaged over the window, then sqrt'd, so movement in both
    directions (e.g. a whisker protraction followed by retraction) adds up
    instead of cancelling the way a signed mean/peak would.

    Returns a long-format DataFrame: mouse_id, day, feature, baseline_mean,
    post_mean, post_peak, post_energy (one row per trial x feature), or
    None if the file is missing/unusable.
    """
    path = os.path.join(dlc_dir, f'{mouse_id}_dlc.h5')
    if not os.path.exists(path):
        return None

    traces_by_day, features, time = _load_dlc_h5(path, normalised=normalised)
    win_mask = (time >= window[0]) & (time < window[1])
    baseline_mask = (time >= baseline_window[0]) & (time < baseline_window[1])
    if not win_mask.any():
        raise ValueError(f"Window {window} does not overlap trial time axis.")
    if not baseline_mask.any():
        raise ValueError(f"Baseline window {baseline_window} does not overlap trial time axis.")

    rows = []
    for day, traces in traces_by_day.items():
        if traces.shape[0] < MIN_TRIALS:
            continue
        baseline_mean = traces[:, :, baseline_mask].mean(axis=2)  # (n_trials, n_features)
        post_window = traces[:, :, win_mask]
        post_mean = post_window.mean(axis=2)
        post_peak = post_window.max(axis=2)
        post_energy = np.sqrt(np.mean((post_window - baseline_mean[:, :, None]) ** 2, axis=2))
        for i_feat, feature in enumerate(features):
            for b, m, p, e in zip(
                baseline_mean[:, i_feat], post_mean[:, i_feat], post_peak[:, i_feat], post_energy[:, i_feat]
            ):
                rows.append(
                    {
                        'mouse_id': mouse_id,
                        'day': int(day),
                        'feature': feature,
                        'baseline_mean': float(b),
                        'post_mean': float(m),
                        'post_peak': float(p),
                        'post_energy': float(e),
                    }
                )
    return pd.DataFrame(rows) if rows else None


def build_dataset(dlc_dir=DLC_DIR):
    """Trial-level movement dataset for every R+ mouse with a DLC file and a
    reward_group entry in the session metadata database.

    R- mice are excluded: of the 12 mice with DLC data, only one (GF319) is
    R-, which is not enough for a group comparison, so this dataset reports
    R+ mice only, across days.

    Returns
    -------
    trial_df : one row per mouse x day x feature x trial, with columns
        baseline_mean, post_mean, post_peak, post_energy (raw, per trial).
    mouse_day_df : one row per mouse x day x feature, with the trial-mean of
        each of the above plus the 6 derived STATS columns (mean, peak,
        baseline, blsub_mean, blsub_peak, energy). This is the level
        statistics and plotting operate on. Averaging (a linear operator)
        is done before deriving blsub_mean/blsub_peak, which is equivalent
        to averaging per-trial differences first; energy is the mean of
        each trial's own RMS (itself already nonlinear, like post_peak).
    """
    mice = discover_dlc_mice(dlc_dir)
    print(f"Found {len(mice)} DLC files: {mice}")

    db = database.read_excel_db(paths.db_path)
    trial_parts = []
    skipped = []
    for mouse_id in mice:
        try:
            reward_group = database.get_mouse_reward_group_from_db(paths.db_path, mouse_id, db=db)
        except (IndexError, KeyError):
            skipped.append(mouse_id)
            continue
        if reward_group != 'R+':
            skipped.append(mouse_id)
            continue
        df = load_trial_metrics(mouse_id, dlc_dir)
        if df is None or len(df) == 0:
            skipped.append(mouse_id)
            continue
        df['reward_group'] = reward_group
        trial_parts.append(df)

    if skipped:
        print(f"Skipped (not R+, no DLC data, or no reward_group in db): {skipped}")
    if not trial_parts:
        raise RuntimeError("No usable DLC data found.")

    trial_df = pd.concat(trial_parts, ignore_index=True)
    mice_used = trial_df['mouse_id'].unique()
    print(f"Usable mice: {len(mice_used)} R+ ({list(mice_used)})")

    mouse_day_df = (
        trial_df.groupby(['mouse_id', 'reward_group', 'day', 'feature'])[
            ['baseline_mean', 'post_mean', 'post_peak', 'post_energy']
        ]
        .mean()
        .reset_index()
    )
    mouse_day_df['mean'] = mouse_day_df['post_mean']
    mouse_day_df['peak'] = mouse_day_df['post_peak']
    mouse_day_df['baseline'] = mouse_day_df['baseline_mean']
    mouse_day_df['blsub_mean'] = mouse_day_df['post_mean'] - mouse_day_df['baseline_mean']
    mouse_day_df['blsub_peak'] = mouse_day_df['post_peak'] - mouse_day_df['baseline_mean']
    mouse_day_df['energy'] = mouse_day_df['post_energy']

    return trial_df, mouse_day_df


def build_trace_dataset(dlc_dir=DLC_DIR, normalised=NORMALISED, baseline_window=BASELINE_WINDOW):
    """Full time-course dataset for every R+ mouse with a DLC file and a
    reward_group entry in the session metadata database (same R+-only
    selection as build_dataset()).

    Each trial's own baseline_window mean (-2 to 0 s) is subtracted from
    its whole trace first, so every trial is baseline-corrected before
    averaging -- this removes each trial's resting offset, which otherwise
    dominates raw "amp" traces (see the flat, near-constant raw baseline
    discussed earlier). Trials are then averaged per mouse (one
    baseline-corrected trace per mouse, so each mouse contributes equally
    regardless of trial count), and plotting computes the mean +/- CI across
    mice at each time point -- the same two-stage averaging convention as
    figure_3c.py.

    Returns a long-format DataFrame: mouse_id, day, feature, time, value
    (one row per mouse x day x feature x time sample; value is baseline-
    subtracted).
    """
    mice = discover_dlc_mice(dlc_dir)
    db = database.read_excel_db(paths.db_path)

    trace_parts = []
    skipped = []
    for mouse_id in mice:
        try:
            reward_group = database.get_mouse_reward_group_from_db(paths.db_path, mouse_id, db=db)
        except (IndexError, KeyError):
            skipped.append(mouse_id)
            continue
        if reward_group != 'R+':
            skipped.append(mouse_id)
            continue
        path = os.path.join(dlc_dir, f'{mouse_id}_dlc.h5')
        if not os.path.exists(path):
            skipped.append(mouse_id)
            continue

        traces_by_day, features, time = _load_dlc_h5(path, normalised=normalised)
        baseline_mask = (time >= baseline_window[0]) & (time < baseline_window[1])
        for day, traces in traces_by_day.items():
            if traces.shape[0] < MIN_TRIALS:
                continue
            baseline = traces[:, :, baseline_mask].mean(axis=2, keepdims=True)  # (n_trials, n_features, 1)
            corrected = traces - baseline
            mean_trace = corrected.mean(axis=0)  # (n_features, n_time)
            for i_feat, feature in enumerate(features):
                trace_parts.append(
                    pd.DataFrame(
                        {
                            'mouse_id': mouse_id,
                            'day': int(day),
                            'feature': feature,
                            'time': time,
                            'value': mean_trace[i_feat],
                        }
                    )
                )

    if skipped:
        print(f"Skipped (not R+, no DLC data, or no reward_group in db): {skipped}")
    if not trace_parts:
        raise RuntimeError("No usable DLC data found.")

    trace_df = pd.concat(trace_parts, ignore_index=True)
    mice_used = trace_df['mouse_id'].unique()
    print(f"Usable mice: {len(mice_used)} R+ ({list(mice_used)})")
    return trace_df


# ============================================================================
# Plot
# ============================================================================


def _feature_labels(features):
    """Split '<part>_trace_<metric>' into (metric, part) for grid layout."""
    parsed = {}
    for feature in features:
        part, metric = feature.replace('_trace_', '|').split('|')
        parsed[feature] = (metric, part)
    return parsed


def _ordered_parts(labels):
    """Body parts from a _feature_labels() dict, in a fixed display order."""
    parts = sorted({p for _, p in labels.values()})
    order = [p for p in ['tongue', 'whiskerend', 'nose', 'whiskerbase', 'jawend'] if p in parts]
    order += [p for p in parts if p not in order]
    return order


def _friedman_p(fdata, value_col, days_sorted):
    """Friedman test (nonparametric repeated-measures ANOVA, matched by
    mouse) for whether value_col differs across days_sorted.

    Mice missing any of the days are dropped (Friedman needs a complete
    mouse x day block, no missing cells) -- used instead of Kruskal-Wallis
    across days because the same mice are measured on every day, so
    treating each day as an independent sample would ignore that
    repeated-measures structure.

    Returns (statistic, p_value, n_mice); (nan, nan, n_mice) if fewer than
    3 mice have complete data across all days.
    """
    wide = fdata.pivot(index='mouse_id', columns='day', values=value_col)
    wide = wide.reindex(columns=days_sorted).dropna()
    n_mice = len(wide)
    if n_mice < 3:
        return np.nan, np.nan, n_mice
    stat, p = friedmanchisquare(*(wide[d].values for d in days_sorted))
    return stat, p, n_mice


def plot_movement_summary(
    mouse_day_df,
    value_col,
    stat_label,
    days=DAYS,
    output_dir=OUTPUT_DIR,
    filename='movement_state_summary',
    save_format='svg',
    dpi=300,
):
    """Grid of panels (rows = speed/amp, columns = body part), each showing
    the R+ mean +/- 95% CI across mice per day, for one of the STATS value
    columns (mirrors behavior_state_summary.py's layout, minus the R+ vs R-
    comparison). Each panel is annotated with a Friedman test (see
    _friedman_p) for whether the 5 days differ.

    Saves:
        <filename>.svg        -- figure; see build_dataset()'s mouse_day_df
                                  for the underlying values (saved once, not
                                  per stat, since all 6 STATS share the same
                                  table)
        <filename>_stats.csv  -- Friedman test per body part x metric panel
                                  (this IS stat-specific, unlike the shared
                                  data table, since the test result depends
                                  on which value_col was tested)
    """
    sns.set_theme(context='paper', style='ticks', palette='deep', font='sans-serif', font_scale=1)

    days_sorted = sorted(days)
    features = sorted(mouse_day_df['feature'].unique())
    labels = _feature_labels(features)
    metrics = sorted({m for m, _ in labels.values()})  # ['amp', 'speed']
    part_order = _ordered_parts(labels)

    fig, axes = plt.subplots(
        len(metrics), len(part_order), figsize=(4 * len(part_order), 4 * len(metrics)), sharex=True
    )
    axes = np.atleast_2d(axes)

    stats_rows = []
    for i, metric in enumerate(metrics):
        for j, part in enumerate(part_order):
            ax = axes[i, j]
            feature = next((f for f, (m, p) in labels.items() if m == metric and p == part), None)
            if feature is None:
                ax.axis('off')
                continue

            fdata = mouse_day_df[mouse_day_df['feature'] == feature]
            sns.barplot(
                data=fdata,
                x='day',
                y=value_col,
                order=days_sorted,
                errorbar=('ci', 95),
                color=reward_palette[1],
                alpha=0.7,
                edgecolor='black',
                ax=ax,
            )

            stat, p, n_mice = _friedman_p(fdata, value_col, days_sorted)
            stars = _significance_stars(p) if not np.isnan(p) else 'n.a.'
            p_text = f'p={p:.3g} {stars}' if not np.isnan(p) else 'n.a.'
            stats_rows.append(
                {
                    'feature': feature,
                    'metric': metric,
                    'part': part,
                    'stat_col': value_col,
                    'test': 'Friedman',
                    'effect': 'day',
                    'statistic': stat,
                    'p_value': p,
                    'n_mice': n_mice,
                    'significance': stars,
                }
            )

            metric_label = {'amp': 'Amplitude', 'speed': 'Speed'}.get(metric, metric)
            unit = '(normalized, a.u.)' if NORMALISED else '(raw px)'
            ax.set_title(f'{part}\nFriedman {p_text}', fontsize=9, fontweight='bold')
            ax.set_xlabel('Day' if i == len(metrics) - 1 else '', fontsize=9)
            ax.set_ylabel(f'{metric_label} {stat_label} {unit}' if j == 0 else '', fontsize=9)
            sns.despine(ax=ax)

    fig.suptitle(stat_label.capitalize(), fontsize=11, fontweight='bold')
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    save_figure(plt.gcf(), os.path.join(output_dir, f'{filename}.{save_format}'))
    plt.close()
    print(f"Figure saved to: {os.path.join(output_dir, filename + '.' + save_format)}")

    pd.DataFrame(stats_rows).to_csv(os.path.join(output_dir, f'{filename}_stats.csv'), index=False)
    print(f"Stats saved to: {output_dir}")


def _mean_ci_by_time(day_data, ci_level=0.95):
    """Analytic mean +/- CI across mice at each time point.

    Equivalent in spirit to seaborn's default (bootstrapped) lineplot CI,
    but computed directly from the mean/SEM/n instead of resampling --
    seaborn bootstraps independently at every one of the ~700 time points,
    which is very slow for a full trial trace; this is a single groupby.
    """
    grouped = day_data.groupby('time')['value']
    mean = grouped.mean()
    sem = grouped.sem()
    n = grouped.count()
    tval = t_dist.ppf((1 + ci_level) / 2, np.maximum(n - 1, 1))
    half_width = tval * sem
    return mean.index.values, mean.values, (mean - half_width).values, (mean + half_width).values


def plot_body_part_traces(
    trace_df, part, days=DAYS, output_dir=OUTPUT_DIR, filename=None, save_format='svg', dpi=300
):
    """One body part's full time-course: 2 rows (amplitude, speed) x 5
    columns (days), each panel the R+ mean +/- 95% CI trace across mice
    (mirrors figure_3c.py's per-day PSTH layout). Traces are per-trial
    baseline-subtracted (see build_trace_dataset).

    Saves <filename>.svg (defaults to 'movement_trace_<part>').
    """
    sns.set_theme(context='paper', style='ticks', palette='deep', font='sans-serif', font_scale=1)

    days_sorted = sorted(days)
    labels = _feature_labels(trace_df['feature'].unique())
    available = set(labels.values())  # {(metric, part), ...}
    metrics_order = [m for m in ('amp', 'speed') if (m, part) in available]

    fig, axes = plt.subplots(
        len(metrics_order),
        len(days_sorted),
        figsize=(3.2 * len(days_sorted), 3.2 * len(metrics_order)),
        sharex=True,
        sharey='row',
    )
    axes = np.atleast_2d(axes)

    for i, metric in enumerate(metrics_order):
        feature = next(f for f, (m, p) in labels.items() if m == metric and p == part)
        metric_data = trace_df[trace_df['feature'] == feature]
        metric_label = {'amp': 'Amplitude', 'speed': 'Speed'}.get(metric, metric)
        unit = '(normalized, a.u.)' if NORMALISED else '(raw px)'

        for j, day in enumerate(days_sorted):
            ax = axes[i, j]
            day_data = metric_data[metric_data['day'] == day]
            if day_data.empty:
                ax.axis('off')
                continue

            time_vals, mean_vals, lo, hi = _mean_ci_by_time(day_data)
            ax.plot(time_vals, mean_vals, color=reward_palette[1])
            ax.fill_between(time_vals, lo, hi, color=reward_palette[1], alpha=0.3, linewidth=0)
            ax.axhline(0, color='gray', linestyle='--', linewidth=0.7, alpha=0.6)
            ax.axvline(0, color='#FF9600', linestyle='-', linewidth=1)

            if i == 0:
                ax.set_title(f'Day {day:+d}', fontsize=10, fontweight='bold')
            ax.set_xlabel('Time (s)' if i == len(metrics_order) - 1 else '', fontsize=9)
            ax.set_ylabel(f'{metric_label} (Δ baseline) {unit}' if j == 0 else '', fontsize=9)
            sns.despine(ax=ax)

    fig.suptitle(part, fontsize=12, fontweight='bold')
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    if filename is None:
        filename = f'movement_trace_{part}'
    save_figure(plt.gcf(), os.path.join(output_dir, f'{filename}.{save_format}'))
    plt.close()
    print(f"Figure saved to: {os.path.join(output_dir, filename + '.' + save_format)}")


# ============================================================================
# Main execution
# ============================================================================

if __name__ == '__main__':
    print("Loading DLC movement data...")
    trial_df, mouse_day_df = build_dataset()
    print(f"\n{len(mouse_day_df)} mouse x day x feature rows, {mouse_day_df['mouse_id'].nunique()} mice")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    mouse_day_df.to_csv(os.path.join(OUTPUT_DIR, 'movement_state_summary_data.csv'), index=False)
    print(f"Data saved to: {OUTPUT_DIR}")

    for value_col, stat_label, suffix in STATS:
        print(f"\nPlotting: {stat_label} ({value_col})")
        plot_movement_summary(
            mouse_day_df, value_col, stat_label, filename=f'movement_state_summary_{suffix}'
        )

    print("\n=== Mean per day (whisker base speed, no baseline correction) ===")
    example = mouse_day_df[mouse_day_df['feature'] == 'whiskerbase_trace_speed']
    print(example.groupby('day')['mean'].mean().to_string())

    print("\nLoading full time-course traces...")
    trace_df = build_trace_dataset()
    part_order = _ordered_parts(_feature_labels(trace_df['feature'].unique()))
    for part in part_order:
        print(f"\nPlotting trace: {part}")
        plot_body_part_traces(trace_df, part)
