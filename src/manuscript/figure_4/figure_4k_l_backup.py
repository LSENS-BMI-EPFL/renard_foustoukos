"""
Figure 4k-l: Reactivation participation rate vs LMI.

Panel k: Boxplot of day-0 participation rate binned by LMI (0.2-width bins),
         separately for R+ and R- mice. Descriptive only, no statistics.

Panel l: Participation rate across days (-2 to +2) for LMI+ vs LMI- cells,
         showing per-mouse averages with individual trajectories. Stats:
         2-way repeated-measures ANOVA (day × LMI category) followed by
         Mann-Whitney U posthoc tests (positive vs negative per day).

For both panels two versions are generated:
    - 'all'  : entire cell population
    - 'sig'  : restricted to significantly-participating cells
               (from circular_shift_significant_participation.csv)

Processed data files are loaded from data_processed/reactivation/.
Figures and CSVs are saved to output/.
"""

import os
import sys
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu
from statsmodels.stats.anova import AnovaRM
from joblib import Parallel, delayed

sys.path.append('/home/aprenard/repos/fast-learning')
import src.utils.utils_io as io
import src.utils.utils_imaging as utils_imaging
from src.utils.utils_plot import reward_palette

from src.core_analysis.reactivations.reactivation_lmi_prediction import (
    load_reactivation_results,
    process_single_mouse as _process_mouse_participation,
    aggregate_across_days,
    load_and_match_lmi_data,
)
from src.core_analysis.reactivations.reactivation import create_whisker_template


# ============================================================================
# Parameters
# ============================================================================

DAYS = [-2, -1, 0, 1, 2]
LMI_POSITIVE_THRESHOLD = 0.975
LMI_NEGATIVE_THRESHOLD = 0.025
N_JOBS = 35

RESULTS_DIR = os.path.join(io.processed_dir, 'reactivation')
REACTIVATION_RESULTS_FILE = os.path.join(RESULTS_DIR, 'reactivation_results_p99.pkl')
OUTPUT_DIR = '/Volumes/Petersen-Lab/analysis/Anthony_Renard/manuscripts/outputs/figure_4/output'

# Circular shift parameters
SAMPLING_RATE = 30
N_SHIFTS = 1000
MIN_SHIFT_FRAMES = 300        # 10 s at 30 Hz — minimum decorrelation gap
SIGNIFICANCE_PCTILE = 95      # top 5 % → significant (p < 0.05)
EVENT_WINDOW_MS = 150
EVENT_WINDOW_FRAMES = int(EVENT_WINDOW_MS / 1000 * SAMPLING_RATE)
PARTICIPATION_THRESHOLD = 0.10
THRESHOLD_DFF = None
MIN_EVENTS_FOR_RELIABILITY = 5


# ============================================================================
# Helpers
# ============================================================================

def _significance_stars(p):
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    return 'n.s.'


# ============================================================================
# Circular shift significance helpers
# ============================================================================

def _participation_from_3d(data_3d, events, n_timepoints, n_trials):
    """Vectorised participation rate per cell.

    Parameters
    ----------
    data_3d  : ndarray (n_cells, n_trials, n_timepoints)
    events   : array-like of event frame indices in the flattened space

    Returns
    -------
    rates   : ndarray (n_cells,) or None
    n_valid : int
    """
    win = EVENT_WINDOW_FRAMES
    valid = [ev for ev in events
             if (ev % n_timepoints) >= win
             and (ev % n_timepoints) < n_timepoints - win
             and (ev // n_timepoints) < n_trials]
    if not valid:
        return None, 0

    t_idxs  = np.array([ev % n_timepoints  for ev in valid])
    tr_idxs = np.array([ev // n_timepoints for ev in valid])

    windows = np.stack([
        data_3d[:, tr_idxs[i], t_idxs[i] - win:t_idxs[i] + win + 1]
        for i in range(len(valid))
    ])
    avg   = np.mean(windows, axis=2)                              # (n_valid, n_cells)
    rates = np.mean(avg >= PARTICIPATION_THRESHOLD, axis=0)       # (n_cells,)
    return rates, len(valid)


def _compute_participation_with_shifts(mouse, day, n_shifts, preloaded_events):
    """Compute real participation rates and circular-shift null distribution
    for one mouse × day.

    Returns a DataFrame or None on failure.
    """
    try:
        template, _ = create_whisker_template(mouse, day, THRESHOLD_DFF,
                                              verbose=False)
        folder = os.path.join(io.solve_common_paths('processed_data'), 'mice')
        xr = utils_imaging.load_mouse_xarray(
            mouse, folder, 'tensor_xarray_learning_data.nc', substracted=True)
        xr_day = xr.sel(trial=xr['day'] == day)
        nostim  = xr_day.sel(trial=xr_day['no_stim'] == 1)

        n_cells, n_trials, n_timepoints = nostim.shape
        if n_trials < 10:
            return None

        data_3d  = np.nan_to_num(nostim.values, nan=0.0)
        roi_list = nostim['roi'].values
        n_frames = n_trials * n_timepoints

        if preloaded_events is None or len(preloaded_events) == 0:
            return None

        real_rates, n_valid = _participation_from_3d(
            data_3d, preloaded_events, n_timepoints, n_trials)
        if real_rates is None or n_valid < MIN_EVENTS_FOR_RELIABILITY:
            return None

        data_flat  = data_3d.reshape(n_cells, n_frames)
        null_rates = np.full((n_shifts, n_cells), np.nan)
        for i_shift in range(n_shifts):
            shift      = np.random.randint(MIN_SHIFT_FRAMES, n_frames)
            shifted_3d = np.roll(data_flat, shift, axis=1).reshape(
                n_cells, n_trials, n_timepoints)
            null_r, _ = _participation_from_3d(
                shifted_3d, preloaded_events, n_timepoints, n_trials)
            if null_r is not None:
                null_rates[i_shift] = null_r

        threshold_95 = np.nanpercentile(null_rates, SIGNIFICANCE_PCTILE, axis=0)
        significant  = real_rates > threshold_95

        records = [
            {'mouse_id': mouse, 'day': day, 'roi': roi_list[icell],
             'participation_rate': real_rates[icell],
             'threshold_95': threshold_95[icell],
             'n_events': n_valid,
             'significant': bool(significant[icell])}
            for icell in range(n_cells)
            if not np.isnan(real_rates[icell])
        ]
        return pd.DataFrame(records) if records else None

    except Exception as e:
        print(f"  circular shift {mouse} day {day}: {e}")
        return None


def _process_mouse_circular_shift(mouse, n_shifts, preloaded_results):
    """Process all days for one mouse. Returns (mouse, DataFrame or None)."""
    dfs = []
    for day in DAYS:
        events = None
        if preloaded_results is not None:
            day_data = preloaded_results.get('days', {}).get(day, {})
            events   = day_data.get('events', None)
        df = _compute_participation_with_shifts(mouse, day, n_shifts, events)
        if df is not None:
            dfs.append(df)
    return mouse, pd.concat(dfs, ignore_index=True) if dfs else None


def _compute_circular_shift_significance(
    reactivation_results_file=REACTIVATION_RESULTS_FILE,
    results_dir=RESULTS_DIR,
    n_jobs=N_JOBS,
    n_shifts=N_SHIFTS,
):
    """Run circular-shift control and return set of significant (mouse, roi) pairs.

    Saves circular_shift_significant_participation.csv to results_dir.
    """
    if not os.path.exists(reactivation_results_file):
        raise FileNotFoundError(
            f"Reactivation results not found: {reactivation_results_file}")

    with open(reactivation_results_file, 'rb') as f:
        data = pickle.load(f)
    r_plus_results  = data['r_plus_results']
    r_minus_results = data['r_minus_results']
    all_results = {**r_plus_results, **r_minus_results}
    all_mice    = list(all_results.keys())

    print(f"\nRunning circular-shift control for {len(all_mice)} mice "
          f"({n_shifts} shifts × {len(DAYS)} days each) ...")
    raw = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(_process_mouse_circular_shift)(
            mouse, n_shifts, all_results.get(mouse))
        for mouse in all_mice
    )

    dfs = [df for _, df in raw if df is not None]
    if not dfs:
        raise RuntimeError("Circular shift control returned no data.")
    participation_df = pd.concat(dfs, ignore_index=True)

    # Keep only day-0 significant cells
    day0_df   = participation_df[participation_df['day'] == 0].copy()
    sig_cells = day0_df[day0_df['significant']][['mouse_id', 'roi']].drop_duplicates()

    n_sig = len(sig_cells)
    n_tot = len(day0_df)
    print(f"Day-0 significant cells: {n_sig}/{n_tot} ({100*n_sig/n_tot:.1f}%)")

    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, 'circular_shift_significant_participation.csv')
    sig_cells.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")

    return set(zip(sig_cells['mouse_id'], sig_cells['roi']))


def _compute_participation_data(
    results_file=REACTIVATION_RESULTS_FILE,
    results_dir=RESULTS_DIR,
    n_jobs=N_JOBS,
):
    """
    Run the full participation-rate computation pipeline.

    1. Load pre-computed reactivation events from results_file.
    2. For each mouse, extract per-event cell responses and compute
       participation rates (calls process_single_mouse from
       reactivation_lmi_prediction.py).
    3. Aggregate participation rates across day periods.
    4. Merge with LMI data.

    Intermediate CSVs are saved to results_dir for inspection.

    Returns
    -------
    merged_df : pd.DataFrame
        Per-cell aggregated data with LMI info (learning_rate = day-0 rate).
    per_day_df : pd.DataFrame
        Per-cell, per-day participation rates.
    """
    # Load reactivation events
    r_plus_reactivations, r_minus_reactivations = load_reactivation_results(results_file)
    all_reactivation_results = {**r_plus_reactivations, **r_minus_reactivations}
    all_mice = list(all_reactivation_results.keys())

    print(f"\nComputing participation rates for {len(all_mice)} mice...")
    results_list = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(_process_mouse_participation)(
            mouse, DAYS, verbose=False,
            preloaded_results=all_reactivation_results.get(mouse),
        )
        for mouse in all_mice
    )

    all_participation_data = [df for _, df in results_list if df is not None]
    if not all_participation_data:
        raise RuntimeError("No participation data computed — check reactivation results.")
    per_day_df = pd.concat(all_participation_data, ignore_index=True)

    os.makedirs(results_dir, exist_ok=True)
    per_day_df.to_csv(
        os.path.join(results_dir, 'cell_participation_rates_per_day.csv'), index=False)

    aggregated_df = aggregate_across_days(per_day_df)
    merged_df = load_and_match_lmi_data(aggregated_df)

    merged_df.to_csv(
        os.path.join(results_dir, 'participation_lmi_merged.csv'), index=False)

    return merged_df, per_day_df


def _load_data(sig_only=False):
    """
    Compute participation + LMI data and optionally restrict to
    significantly-participating cells.

    Returns
    -------
    merged_df : pd.DataFrame
        Per-cell aggregated data with LMI info (learning_rate = day-0 participation).
    per_day_df : pd.DataFrame
        Per-cell, per-day participation rates.
    """
    merged, per_day = _compute_participation_data()

    if sig_only:
        sig_keys = _compute_circular_shift_significance()

        mask_merged = [(m, r) in sig_keys
                       for m, r in zip(merged['mouse_id'], merged['roi'])]
        mask_perday = [(m, r) in sig_keys
                       for m, r in zip(per_day['mouse_id'], per_day['roi'])]
        merged = merged[mask_merged].reset_index(drop=True)
        per_day = per_day[mask_perday].reset_index(drop=True)

    # Ensure lmi_category column exists
    if 'lmi_category' not in merged.columns:
        merged['lmi_category'] = 'neutral'
        merged.loc[merged['lmi_p'] >= LMI_POSITIVE_THRESHOLD, 'lmi_category'] = 'positive'
        merged.loc[merged['lmi_p'] <= LMI_NEGATIVE_THRESHOLD, 'lmi_category'] = 'negative'

    return merged, per_day


# ============================================================================
# Panel k: day-0 participation rate vs LMI (0.2-width bins, descriptive)
# ============================================================================

def panel_k_participation_vs_lmi(
    merged_df,
    output_dir=OUTPUT_DIR,
    filename='figure_4k',
    save_format='svg',
    dpi=300,
):
    """
    Generate Figure 4 Panel k: boxplot of day-0 participation rate binned by LMI.

    LMI is divided into 0.2-width bins from -1 to 1. Descriptive only — no stats.

    Saves:
        <filename>.svg       – figure
        <filename>_data.csv  – cell-level data with lmi_bin column
    """
    sns.set_theme(context='paper', style='ticks', palette='deep',
                  font='sans-serif', font_scale=1)

    df = merged_df.dropna(subset=['lmi', 'learning_rate']).copy()

    lmi_bins = np.arange(-1.0, 1.01, 0.2)
    bin_centers = lmi_bins[:-1] + 0.1
    bin_labels = [f'{c:.1f}' for c in bin_centers]
    df['lmi_bin'] = pd.cut(df['lmi'], bins=lmi_bins, labels=bin_labels,
                           include_lowest=True)

    reward_groups = ['R+', 'R-']
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)

    for i, rg in enumerate(reward_groups):
        ax = axes[i]
        grp = df[df['reward_group'] == rg]

        sns.boxplot(
            data=grp, x='lmi_bin', y='learning_rate',
            order=bin_labels, color='steelblue',
            width=0.6, linewidth=0.8, fliersize=2, ax=ax,
        )

        ax.axvline(x=bin_labels.index('0.1') - 0.5, color='gray',
                   linestyle='--', linewidth=0.8, alpha=0.6)
        n_cells = len(grp)
        n_mice  = grp['mouse_id'].nunique()
        ax.set_title(f'{rg}  (n = {n_cells} cells, {n_mice} mice)',
                     fontsize=10, fontweight='bold')
        ax.set_xlabel('LMI', fontsize=9)
        ax.set_ylabel('Participation rate (day 0)' if i == 0 else '', fontsize=9)
        ax.tick_params(axis='x', rotation=45, labelsize=7)
        ax.tick_params(axis='y', labelsize=8)
        sns.despine(ax=ax)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f'{filename}.{save_format}'),
                format=save_format, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"Panel k saved to: {os.path.join(output_dir, filename + '.' + save_format)}")

    df[['mouse_id', 'roi', 'reward_group', 'lmi', 'lmi_bin',
        'learning_rate']].to_csv(
        os.path.join(output_dir, f'{filename}_data.csv'), index=False)
    print(f"Panel k data saved to: {output_dir}")


# ============================================================================
# Panel l: participation rate across days (LMI+ vs LMI-)
# ============================================================================

def panel_l_participation_across_days(
    merged_df,
    per_day_df,
    output_dir=OUTPUT_DIR,
    filename='figure_4l',
    save_format='svg',
    dpi=300,
):
    """
    Generate Figure 4 Panel l: participation rate across days for LMI+ vs LMI- cells.

    Per-mouse averages with individual trajectories. Stats: 2-way repeated-measures
    ANOVA (day × LMI category) followed by Mann-Whitney U posthoc tests
    (positive vs negative at each day).

    Saves:
        <filename>.svg         – figure
        <filename>_data.csv    – per-mouse × day × LMI-category participation averages
        <filename>_stats.csv   – ANOVA table + Mann-Whitney U posthoc per day
    """
    sns.set_theme(context='paper', style='ticks', palette='deep',
                  font='sans-serif', font_scale=1)

    days_sorted = sorted(DAYS)
    lmi_categories = ['positive', 'negative']
    cat_colors = {'positive': '#d62728', 'negative': '#1f77b4'}
    reward_groups = ['R+', 'R-']

    # Merge per-day rates with LMI categories + reward group
    lmi_cells = merged_df.loc[
        merged_df['lmi_category'].isin(lmi_categories),
        ['mouse_id', 'roi', 'lmi_category', 'reward_group'],
    ]
    day_data = pd.merge(per_day_df, lmi_cells, on=['mouse_id', 'roi'], how='inner')

    # Per-mouse × day × category averages
    mouse_day_avg = (
        day_data
        .groupby(['mouse_id', 'reward_group', 'lmi_category', 'day'],
                 observed=True)['participation_rate']
        .mean()
        .reset_index()
    )

    # Cell counts per reward group × LMI category (unique cells)
    cell_counts = (
        lmi_cells.groupby(['reward_group', 'lmi_category'], observed=True)
        .size()
        .to_dict()
    )

    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    all_stats_rows = []
    plot_data_rows = []

    for i, rg in enumerate(reward_groups):
        ax = axes[i]
        grp = mouse_day_avg[mouse_day_avg['reward_group'] == rg]

        sns.barplot(
            data=grp, x='day', y='participation_rate', hue='lmi_category',
            hue_order=lmi_categories, palette=cat_colors, order=days_sorted,
            estimator=np.mean, errorbar=('ci', 95), capsize=0,
            err_kws={'linewidth': 1.5}, alpha=0.7, ax=ax,
        )
        for patch in ax.patches:
            patch.set_edgecolor('black')
            patch.set_linewidth(0.6)

        # Individual mouse trajectories
        for j, cat in enumerate(lmi_categories):
            if j >= len(ax.containers):
                continue
            cat_grp = grp[grp['lmi_category'] == cat]
            x_centers = {
                days_sorted[k]: bar.get_x() + bar.get_width() / 2
                for k, bar in enumerate(ax.containers[j])
                if k < len(days_sorted)
            }
            for mouse_id in cat_grp['mouse_id'].unique():
                mdata = cat_grp[cat_grp['mouse_id'] == mouse_id].sort_values('day')
                mx = [x_centers[d] for d in mdata['day'] if d in x_centers]
                my = mdata['participation_rate'].values
                ax.plot(mx, my, '-', color=cat_colors[cat],
                        linewidth=0.8, alpha=0.4, zorder=5)

        # ── 2-way repeated-measures ANOVA (day × lmi_category) ──────────────
        # Requires mice with data in both LMI categories and all days.
        mice_pos = set(grp.loc[grp['lmi_category'] == 'positive', 'mouse_id'])
        mice_neg = set(grp.loc[grp['lmi_category'] == 'negative', 'mouse_id'])
        mice_both = sorted(mice_pos & mice_neg)

        anova_p_interaction = np.nan
        anova_rows = []
        if len(mice_both) >= 3:
            anova_data = grp[grp['mouse_id'].isin(mice_both)].copy()
            anova_data['day'] = anova_data['day'].astype(str)
            try:
                aovrm = AnovaRM(anova_data, 'participation_rate', 'mouse_id',
                                within=['day', 'lmi_category'])
                anova_fit = aovrm.fit()
                tbl = anova_fit.anova_table
                for effect in tbl.index:
                    anova_rows.append({
                        'reward_group': rg,
                        'test': '2-way rm ANOVA',
                        'effect': effect,
                        'F': tbl.loc[effect, 'F Value'],
                        'num_df': tbl.loc[effect, 'Num DF'],
                        'den_df': tbl.loc[effect, 'Den DF'],
                        'p_value': tbl.loc[effect, 'Pr > F'],
                        'significance': _significance_stars(tbl.loc[effect, 'Pr > F']),
                    })
                anova_p_interaction = tbl.loc['day:lmi_category', 'Pr > F']
                print(f"  {rg} ANOVA: interaction p = {anova_p_interaction:.4f}")
            except Exception as e:
                print(f"  {rg} ANOVA failed: {e}")
        else:
            print(f"  {rg}: only {len(mice_both)} mice with both categories — ANOVA skipped")

        all_stats_rows.extend(anova_rows)

        # ── Mann-Whitney U posthoc: positive vs negative per day ─────────────
        posthoc_p_list = []
        for k, day in enumerate(days_sorted):
            pos_vals = grp.loc[
                (grp['lmi_category'] == 'positive') & (grp['day'] == day),
                'participation_rate',
            ].values
            neg_vals = grp.loc[
                (grp['lmi_category'] == 'negative') & (grp['day'] == day),
                'participation_rate',
            ].values
            if len(pos_vals) >= 2 and len(neg_vals) >= 2:
                stat, p = mannwhitneyu(pos_vals, neg_vals, alternative='two-sided')
            else:
                stat, p = np.nan, np.nan
            posthoc_p_list.append(p)
            all_stats_rows.append({
                'reward_group': rg,
                'test': 'Mann-Whitney U (posthoc)',
                'effect': f'day {day}: positive vs negative',
                'F': np.nan,
                'num_df': np.nan,
                'den_df': np.nan,
                'statistic': stat,
                'n_positive': len(pos_vals),
                'n_negative': len(neg_vals),
                'p_value': p,
                'significance': _significance_stars(p) if not np.isnan(p) else 'n.a.',
            })

        # Significance markers from posthoc Mann-Whitney U
        y_max = grp['participation_rate'].max() * 1.05 if len(grp) > 0 else 0.5
        for k, (day, p) in enumerate(zip(days_sorted, posthoc_p_list)):
            if not np.isnan(p) and p < 0.05:
                stars = _significance_stars(p)
                if len(ax.containers) >= 2:
                    x0 = (ax.containers[0][k].get_x()
                           + ax.containers[0][k].get_width() / 2)
                    x1 = (ax.containers[1][k].get_x()
                           + ax.containers[1][k].get_width() / 2)
                    ax.text((x0 + x1) / 2, y_max, stars,
                            ha='center', va='bottom', fontsize=10)

        n_pos = cell_counts.get((rg, 'positive'), 0)
        n_neg = cell_counts.get((rg, 'negative'), 0)
        ax.set_title(f'{rg}  (LMI+: {n_pos} cells | LMI−: {n_neg} cells)',
                     fontsize=9, fontweight='bold')
        ax.set_xlabel('Day', fontsize=9)
        ax.set_ylabel('Participation rate' if i == 0 else '', fontsize=9)
        ax.tick_params(labelsize=8)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, [f'{l.capitalize()} LMI' for l in labels], fontsize=8)
        sns.despine(ax=ax)

        plot_data_rows.append(grp)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f'{filename}.{save_format}'),
                format=save_format, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"Panel l saved to: {os.path.join(output_dir, filename + '.' + save_format)}")

    pd.concat(plot_data_rows, ignore_index=True).to_csv(
        os.path.join(output_dir, f'{filename}_data.csv'), index=False)
    pd.DataFrame(all_stats_rows).to_csv(
        os.path.join(output_dir, f'{filename}_stats.csv'), index=False)
    print(f"Panel l data/stats saved to: {output_dir}")


# ============================================================================
# Main execution
# ============================================================================

if __name__ == '__main__':
    print(f"Loading data from: {RESULTS_DIR}")

    for sig_only in [False, True]:
        suffix = '_sig' if sig_only else '_all'
        label = 'significantly-participating cells' if sig_only else 'all cells'
        print(f"\n{'='*60}")
        print(f"Population: {label}")
        print('='*60)

        merged_df, per_day_df = _load_data(sig_only=sig_only)
        print(f"  {len(merged_df)} cells, {len(per_day_df)} cell-day records")

        panel_k_participation_vs_lmi(
            merged_df,
            filename=f'figure_4k{suffix}',
        )
        panel_l_participation_across_days(
            merged_df,
            per_day_df,
            filename=f'figure_4l{suffix}',
        )
