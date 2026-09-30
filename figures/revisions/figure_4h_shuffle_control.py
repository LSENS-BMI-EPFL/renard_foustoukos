"""
Figure 4H specificity control: shuffled-template reactivation detection.

Addresses a second reviewer's specificity concern: template-matching with
shuffled templates would help establish that detected reactivation events
reflect whisker-specific patterns rather than generic high-coactivity
moments. (The auditory-template control the reviewer also suggested is not
implemented — there are no auditory-stimulus mapping trials anywhere in
the dataset, confirmed by inspecting create_whisker_template() in
fast_learning/reactivations.py, so it is not feasible without new data
collection.)

For each mouse and day, this script reuses the already-computed real
template and detection threshold from reactivation_results_p99.pkl
(RESULTS_DIR, original all-no-stim-trials variant) and the same no-stim
trial data (reloaded from the tensors with load_selected_trials), and detects "reactivation" events with N_SHUFFLES=1000
cell-identity-shuffled versions of that template instead — same real
neural data, same fixed detection threshold as the real template (not
recalibrated per shuffle, which would let every shuffled template earn its
own top-1% cutoff by construction and mask rather than test specificity).

Statistics are deliberately two-layered:
  - Visualization: shuffled event rates pooled across mice x shuffles per
    (reward_group, day), for an intuitive picture of the null distribution.
  - Reported statistic: one "specificity index" per mouse (real event rate
    minus that mouse's own mean shuffled rate), then a Wilcoxon signed-rank
    test of those indices against zero across mice (n = mice) per
    (reward_group, day) -- the same "one number per mouse, then test across
    mice" pattern used for Figure 4J, and nonparametric to match the
    manuscript's style elsewhere. Pooling mice x shuffles into one sample
    for a p-value would reintroduce the exact pseudoreplication problem
    this rebuttal is fixing elsewhere.

Mice: all imaging mice, as in Figure 4h (the participation mouse selection
in fast_learning.reactivations does not apply to event rates).

The shuffle detection (expensive, ~n_mice x n_days x 1000 shuffles) runs
when its cached CSVs are missing or with --recompute; otherwise the cached
CSVs are plotted:
    python figures/revisions/figure_4h_shuffle_control.py [--recompute]

Figures and CSVs are saved to
    paths.manuscript_output_dir/revisions/figure_4h_shuffle_control/output/.
"""

from functools import partial
import os
import pickle
import zlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wilcoxon
from joblib import Parallel, delayed

from fast_learning import paths
from fast_learning.plotting import reward_palette, save_figure
from fast_learning.reactivations import (
    detect_reactivation_events,
    load_selected_trials,
    MIN_EVENT_DISTANCE_FRAMES,
    PROMINENCE,
)
from fast_learning.stats import significance_stars

REACTIVATION_RESULTS_DIR = os.path.join(paths.processed_dir, 'reactivation')


# ============================================================================
# Parameters
# ============================================================================

DAYS = [-2, -1, 0, 1, 2]
N_SHUFFLES = 1000
GLOBAL_SEED = 42  # fixed for reproducibility of this stochastic control
PERCENTILE_TAG = 'p99'  # detection threshold variant to use (see plan)
N_JOBS = 35

REACTIVATION_RESULTS_FILE = os.path.join(
    REACTIVATION_RESULTS_DIR, f'reactivation_results_{PERCENTILE_TAG}.pkl'
)
OUTPUT_DIR = os.path.join(paths.manuscript_output_dir, 'revisions', 'figure_4h_shuffle_control', 'output')


# ============================================================================
# Helpers
# ============================================================================

_significance_stars = partial(significance_stars, na='n.a.')


def _seed_for(mouse, day):
    """Deterministic per-(mouse, day) seed, independent of PYTHONHASHSEED."""
    key = f"{GLOBAL_SEED}_{mouse}_{day}".encode()
    return zlib.crc32(key) % (2**32)


def _load_reactivation_results(results_file):
    """Load pre-computed reactivation results from pickle."""
    if not os.path.exists(results_file):
        raise FileNotFoundError(
            f"Reactivation results file not found: {results_file}\nRun pipeline/07_reactivations.py first."
        )
    print(f"\nLoading reactivation events from: {results_file}")
    with open(results_file, 'rb') as f:
        data = pickle.load(f)
    r_plus = data['r_plus_results']
    r_minus = data['r_minus_results']
    print(f"Loaded {len(r_plus)} R+ mice and {len(r_minus)} R- mice")
    return r_plus, r_minus


# ============================================================================
# Vectorized batch template correlation
# ============================================================================


def _compute_template_correlation_batch(data, template_matrix):
    """Vectorized version of compute_template_correlation for many templates
    at once, avoiding a Python loop over shuffles for the (expensive) matrix
    operation.

    Parameters
    ----------
    data            : (n_cells, n_timepoints)
    template_matrix : (n_shuffles, n_cells)

    Returns
    -------
    correlations : (n_shuffles, n_timepoints)
    """
    n_cells, n_timepoints = data.shape
    template_stds = np.std(template_matrix, axis=1, keepdims=True)  # (n_shuffles, 1)
    template_centered = template_matrix - np.mean(template_matrix, axis=1, keepdims=True)
    data_centered = data - np.mean(data, axis=0, keepdims=True)
    data_stds = np.std(data, axis=0)  # (n_timepoints,)

    with np.errstate(divide='ignore', invalid='ignore'):
        correlations = (template_centered @ data_centered) / (
            template_stds * data_stds[np.newaxis, :] * n_cells
        )

    correlations = np.nan_to_num(correlations, nan=0.0, posinf=0.0, neginf=0.0)
    correlations[:, data_stds == 0] = 0
    correlations[template_stds.flatten() == 0, :] = 0
    return correlations


# ============================================================================
# Per-mouse-day shuffle computation
# ============================================================================


def _compute_shuffled_rates_for_mouse_day(day_results, selected_trials, n_shuffles=N_SHUFFLES, seed=None):
    """Detect "reactivation" events with n_shuffles cell-identity-shuffled
    versions of the real template, on the same real neural data and the
    same fixed detection threshold used for the real template.

    Returns an array of n_shuffles event frequencies (events/min).
    """
    template = day_results['template']
    threshold = day_results['threshold_used']
    session_duration_min = day_results['session_duration_min']

    n_cells = selected_trials.shape[0]
    data = np.nan_to_num(selected_trials.values.reshape(n_cells, -1), nan=0.0)

    rng = np.random.default_rng(seed)
    template_matrix = np.stack([rng.permutation(template) for _ in range(n_shuffles)], axis=0)

    corr_matrix = _compute_template_correlation_batch(data, template_matrix)

    shuffled_rates = np.empty(n_shuffles)
    for i in range(n_shuffles):
        events = detect_reactivation_events(corr_matrix[i], threshold, MIN_EVENT_DISTANCE_FRAMES, PROMINENCE)
        shuffled_rates[i] = len(events) / session_duration_min

    return shuffled_rates


def _process_mouse(mouse, results, n_shuffles):
    """Compute shuffled-template event rates for all days of one mouse.

    Returns (shuffle_rows, real_rows): long-format lists of dicts.
    """
    shuffle_rows, real_rows = [], []
    for day, day_results in results.get('days', {}).items():
        if 'template' not in day_results:
            continue
        seed = _seed_for(mouse, day)
        try:
            # All no-stim trials, full window: the selection of the p99 results file.
            selected_trials = load_selected_trials(mouse, day)
            shuffled_rates = _compute_shuffled_rates_for_mouse_day(
                day_results, selected_trials, n_shuffles=n_shuffles, seed=seed
            )
        except Exception as e:
            print(f"  Warning: {mouse} day {day}: {e}")
            continue

        for i, rate in enumerate(shuffled_rates):
            shuffle_rows.append(
                {
                    'mouse_id': mouse,
                    'day': day,
                    'shuffle_idx': i,
                    'shuffled_event_frequency': rate,
                }
            )
        real_rows.append(
            {
                'mouse_id': mouse,
                'day': day,
                'real_event_frequency': day_results['event_frequency'],
            }
        )
    return shuffle_rows, real_rows


# ============================================================================
# Main compute pipeline
# ============================================================================


def compute_shuffle_control(r_plus_results, r_minus_results, n_shuffles=N_SHUFFLES, n_jobs=N_JOBS):
    """Run the shuffle-detection pipeline for all mice in parallel.

    Returns (long_df, real_df):
        long_df : one row per (mouse_id, reward_group, day, shuffle_idx,
                   shuffled_event_frequency)
        real_df : one row per (mouse_id, reward_group, day,
                   real_event_frequency)
    """
    reward_group_map = {}
    all_results = {}
    for mouse, res in r_plus_results.items():
        reward_group_map[mouse] = 'R+'
        all_results[mouse] = res
    for mouse, res in r_minus_results.items():
        reward_group_map[mouse] = 'R-'
        all_results[mouse] = res

    mice = list(all_results.keys())
    print(f"\nComputing {n_shuffles} shuffled-template detections for {len(mice)} mice...")

    outputs = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(_process_mouse)(mouse, all_results[mouse], n_shuffles) for mouse in mice
    )

    shuffle_rows, real_rows = [], []
    for s_rows, r_rows in outputs:
        shuffle_rows.extend(s_rows)
        real_rows.extend(r_rows)

    long_df = pd.DataFrame(shuffle_rows)
    long_df['reward_group'] = long_df['mouse_id'].map(reward_group_map)

    real_df = pd.DataFrame(real_rows)
    real_df['reward_group'] = real_df['mouse_id'].map(reward_group_map)

    return long_df, real_df


def _compute_per_mouse_summary(long_df, real_df):
    """One row per (mouse_id, reward_group, day): real rate, mean/sd of
    shuffled rates, specificity index, and a permutation p-value from that
    mouse's own null (kept as a descriptive detail alongside the primary
    group-level Wilcoxon test).
    """
    rows = []
    real_lookup = real_df.set_index(['mouse_id', 'day'])['real_event_frequency']

    for (mouse_id, reward_group, day), grp in long_df.groupby(['mouse_id', 'reward_group', 'day']):
        if (mouse_id, day) not in real_lookup.index:
            continue
        real = real_lookup.loc[(mouse_id, day)]
        shuffled = grp['shuffled_event_frequency'].values
        n = len(shuffled)
        mean_shuffled = float(np.mean(shuffled))
        rows.append(
            {
                'mouse_id': mouse_id,
                'reward_group': reward_group,
                'day': day,
                'real_event_frequency': float(real),
                'mean_shuffled': mean_shuffled,
                'sd_shuffled': float(np.std(shuffled, ddof=1)),
                'specificity_index': float(real) - mean_shuffled,
                'permutation_p': (1 + int(np.sum(shuffled >= real))) / (n + 1),
                'n_shuffles': n,
            }
        )
    return pd.DataFrame(rows)


def _compute_group_stats(per_mouse_df):
    """For each (reward_group, day): Wilcoxon signed-rank test of the
    per-mouse specificity index against zero (n = mice). This is the
    primary reported statistic -- not a p-value from the pooled
    visualization histogram.
    """
    rows = []
    for (reward_group, day), grp in per_mouse_df.groupby(['reward_group', 'day']):
        vals = grp['specificity_index'].values
        n_mice = len(vals)
        if n_mice < 2:
            print(f"  Skipping {reward_group} day {day}: only {n_mice} mice")
            continue
        try:
            w_stat, p_value = wilcoxon(vals)
        except ValueError:
            w_stat, p_value = np.nan, np.nan
        rows.append(
            {
                'reward_group': reward_group,
                'day': day,
                'mean_specificity_index': float(np.mean(vals)),
                'median_specificity_index': float(np.median(vals)),
                'w_stat': float(w_stat) if not np.isnan(w_stat) else np.nan,
                'p_value': float(p_value) if not np.isnan(p_value) else np.nan,
                'n_mice': n_mice,
            }
        )
        print(
            f"  {reward_group} day {day}: median specificity index="
            f"{np.median(vals):.4g}, W={w_stat:.3g}, p={p_value:.4g}, "
            f"n_mice={n_mice}"
        )
    return pd.DataFrame(rows)


# ============================================================================
# Plot
# ============================================================================


def panel_4h_shuffle_control(
    long_df,
    real_df,
    group_stats_df,
    output_dir=OUTPUT_DIR,
    filename='figure_4h_shuffle_control',
    save_format='svg',
    dpi=300,
):
    """2 (reward_group) x 5 (day) grid. Each panel: pooled distribution of
    shuffled-template event rates (all mice x n_shuffles, for visual
    smoothness only) with a vertical line at the real across-mice mean
    event rate, annotated with the group-level Wilcoxon p-value (computed
    per mouse, not from this pooled histogram).

    Saves:
        <filename>.svg -- figure
    """
    sns.set_theme(context='paper', style='ticks', palette='deep', font='sans-serif', font_scale=1)

    days_sorted = sorted(DAYS)
    reward_groups = ['R+', 'R-']
    colors = {'R+': reward_palette[1], 'R-': reward_palette[0]}
    xlim = (0, 20)
    binwidth = 1

    fig, axes = plt.subplots(2, len(days_sorted), figsize=(3.2 * len(days_sorted), 6))

    for i, rg in enumerate(reward_groups):
        for j, day in enumerate(days_sorted):
            ax = axes[i, j]
            pooled = long_df.loc[
                (long_df['reward_group'] == rg) & (long_df['day'] == day), 'shuffled_event_frequency'
            ]

            if len(pooled) == 0:
                ax.axis('off')
                continue

            sns.histplot(
                pooled,
                binwidth=binwidth,
                binrange=xlim,
                color=colors[rg],
                stat='probability',
                alpha=0.6,
                ax=ax,
            )

            real_mean = real_df.loc[
                (real_df['reward_group'] == rg) & (real_df['day'] == day), 'real_event_frequency'
            ].mean()
            ax.axvline(real_mean, color='black', linestyle='--', linewidth=1.5)

            stat_row = group_stats_df[(group_stats_df['reward_group'] == rg) & (group_stats_df['day'] == day)]
            if not stat_row.empty:
                p = stat_row.iloc[0]['p_value']
                n_mice = int(stat_row.iloc[0]['n_mice'])
                stars = _significance_stars(p)
                ax.set_title(f'{rg}, day {day}\np={p:.3g} {stars} (n={n_mice} mice)', fontsize=8)
            else:
                ax.set_title(f'{rg}, day {day}\nn.a.', fontsize=8)

            ax.set_xlim(xlim)
            ax.set_ylim(0, 1)
            ax.set_xlabel('Shuffled event rate\n(events/min)', fontsize=7)
            ax.set_ylabel('Probability' if j == 0 else '', fontsize=7)
            ax.tick_params(labelsize=6)
            sns.despine(ax=ax)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    save_figure(plt.gcf(), os.path.join(output_dir, f'{filename}.{save_format}'))
    plt.close()
    print(f"Panel saved: {os.path.join(output_dir, filename + '.' + save_format)}")


# ============================================================================
# Main execution
# ============================================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Figure 4h shuffled-template control.')
    parser.add_argument(
        '--recompute', action='store_true', help='rerun the shuffle detection even if its CSVs are cached'
    )
    args = parser.parse_args()
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Reactivation results: {REACTIVATION_RESULTS_FILE}")
    print(f"N shuffles: {N_SHUFFLES}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    long_csv = os.path.join(OUTPUT_DIR, 'shuffle_control_events_long.csv')
    per_mouse_csv = os.path.join(OUTPUT_DIR, 'shuffle_control_per_mouse.csv')
    group_csv = os.path.join(OUTPUT_DIR, 'shuffle_control_group_stats.csv')

    if args.recompute or not all(os.path.exists(p) for p in [long_csv, per_mouse_csv, group_csv]):
        r_plus_results, r_minus_results = _load_reactivation_results(REACTIVATION_RESULTS_FILE)
        long_df, real_df = compute_shuffle_control(r_plus_results, r_minus_results)
        long_df.to_csv(long_csv, index=False)
        print(f"Saved: {long_csv} ({len(long_df)} rows)")

        per_mouse_df = _compute_per_mouse_summary(long_df, real_df)
        per_mouse_df.to_csv(per_mouse_csv, index=False)
        print(f"Saved: {per_mouse_csv} ({len(per_mouse_df)} rows)")

        group_stats_df = _compute_group_stats(per_mouse_df)
        group_stats_df.to_csv(group_csv, index=False)
        print(f"Saved: {group_csv} ({len(group_stats_df)} rows)")

    else:
        long_df = pd.read_csv(long_csv)
        per_mouse_df = pd.read_csv(per_mouse_csv)
        group_stats_df = pd.read_csv(group_csv)
        real_df = per_mouse_df[['mouse_id', 'reward_group', 'day', 'real_event_frequency']]

    panel_4h_shuffle_control(long_df, real_df, group_stats_df)
    print("\nDone.")
