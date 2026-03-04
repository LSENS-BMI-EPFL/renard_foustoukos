"""
Figure 4k-l: Proportion of cells participating in reactivation vs LMI.

Binary participation is determined by circular-shift control: a cell is
classified as 'participating' if its real participation rate around reactivation
events exceeds the 95th percentile of a null distribution built from 1000
circular shifts of the neural data (same shift applied to all cells
simultaneously, preserving inter-cell correlations).

Panel k: Bar chart of the proportion of participating cells per LMI bin
         (0.2-width bins), day 0 only. Per-mouse proportions averaged across
         mice (mean ± 95% CI). Descriptive only, no statistics.

Panel l: Proportion of participating cells across days (-2 to +2) separately
         for LMI+ vs LMI- cells. Per-mouse averages with individual
         trajectories. Stats: 2-way repeated-measures ANOVA (day × LMI
         category) + Mann-Whitney U posthoc tests (positive vs negative per
         day).

Intermediate data is saved to data_processed/reactivation/.
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


# ============================================================================
# Parameters
# ============================================================================

DAYS = [-2, -1, 0, 1, 2]
LMI_POSITIVE_THRESHOLD = 0.975
LMI_NEGATIVE_THRESHOLD = 0.025
N_JOBS = 35

RESULTS_DIR = os.path.join(io.processed_dir, 'reactivation')
REACTIVATION_RESULTS_FILE = os.path.join(RESULTS_DIR, 'reactivation_results_p99.pkl')
BINARY_PARTICIPATION_CSV = os.path.join(RESULTS_DIR, 'binary_participation_with_lmi.csv')
OUTPUT_DIR = '/Volumes/Petersen-Lab/analysis/Anthony_Renard/manuscripts/outputs/figure_4/output'

# Execution mode
#   'compute' : run circular-shift control, save results, then plot
#   'plot'    : load previously saved results and plot only
MODE = 'compute'

# Circular shift parameters
SAMPLING_RATE = 30
N_SHIFTS = 1000
MIN_SHIFT_FRAMES = 0       
SIGNIFICANCE_PCTILE = 95      # top 5 % → p < 0.05
EVENT_WINDOW_MS = 150
EVENT_WINDOW_FRAMES = int(EVENT_WINDOW_MS / 1000 * SAMPLING_RATE)
PARTICIPATION_THRESHOLD = 0.10
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
# Circular shift helpers
# ============================================================================

def _participation_from_3d(data_3d, events, n_timepoints, n_trials):
    """Vectorised participation rate per cell.

    Parameters
    ----------
    data_3d  : ndarray (n_cells, n_trials, n_timepoints)
    events   : array-like of event frame indices in flattened space

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

    Returns a DataFrame with columns
        mouse_id, day, roi, participating (bool), n_events
    or None on failure.
    """
    try:
        folder = os.path.join(io.solve_common_paths('processed_data'), 'mice')
        xr = utils_imaging.load_mouse_xarray(
            mouse, folder, 'tensor_xarray_learning_data.nc', substracted=False)
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
             'participating': bool(significant[icell]),
             'n_events': n_valid}
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


# ============================================================================
# Data computation
# ============================================================================

def _load_binary_participation():
    """Load pre-computed binary participation data from CSV."""
    if not os.path.exists(BINARY_PARTICIPATION_CSV):
        raise FileNotFoundError(
            f"Pre-computed data not found: {BINARY_PARTICIPATION_CSV}\n"
            "Run with MODE='compute' first.")
    df = pd.read_csv(BINARY_PARTICIPATION_CSV)
    print(f"Loaded: {BINARY_PARTICIPATION_CSV}  ({len(df)} rows)")
    return df


def _compute_binary_participation():
    """Run circular-shift control for all mice × days and merge with LMI data.

    For each cell × day, determines whether the cell participates in
    reactivation (True/False) by comparing its real participation rate to the
    95th percentile of a 1000-shift circular null distribution.

    Returns
    -------
    pd.DataFrame with columns:
        mouse_id, day, roi, participating (bool), n_events,
        reward_group, lmi, lmi_p, lmi_category
    """
    if not os.path.exists(REACTIVATION_RESULTS_FILE):
        raise FileNotFoundError(
            f"Reactivation results not found: {REACTIVATION_RESULTS_FILE}\n"
            "Run reactivation_preprocessing.py first.")

    with open(REACTIVATION_RESULTS_FILE, 'rb') as f:
        data = pickle.load(f)
    r_plus_results  = data['r_plus_results']
    r_minus_results = data['r_minus_results']
    all_results = {**r_plus_results, **r_minus_results}
    reward_group_map = {m: 'R+' for m in r_plus_results}
    reward_group_map.update({m: 'R-' for m in r_minus_results})
    all_mice = list(all_results.keys())

    print(f"\nRunning circular-shift control for {len(all_mice)} mice "
          f"({N_SHIFTS} shifts × {len(DAYS)} days each) ...")
    raw = Parallel(n_jobs=N_JOBS, verbose=5)(
        delayed(_process_mouse_circular_shift)(
            mouse, N_SHIFTS, all_results.get(mouse))
        for mouse in all_mice
    )

    dfs = [df for _, df in raw if df is not None]
    if not dfs:
        raise RuntimeError("Circular shift returned no data.")
    participation_df = pd.concat(dfs, ignore_index=True)
    participation_df['reward_group'] = participation_df['mouse_id'].map(reward_group_map)

    # Load LMI data
    lmi_df = pd.read_csv(os.path.join(io.processed_dir, 'lmi_results.csv'))
    lmi_df['lmi_category'] = 'neutral'
    lmi_df.loc[lmi_df['lmi_p'] >= LMI_POSITIVE_THRESHOLD, 'lmi_category'] = 'positive'
    lmi_df.loc[lmi_df['lmi_p'] <= LMI_NEGATIVE_THRESHOLD, 'lmi_category'] = 'negative'

    merged = pd.merge(
        participation_df,
        lmi_df[['mouse_id', 'roi', 'lmi', 'lmi_p', 'lmi_category']],
        on=['mouse_id', 'roi'], how='inner',
    )

    os.makedirs(RESULTS_DIR, exist_ok=True)
    merged.to_csv(BINARY_PARTICIPATION_CSV, index=False)
    print(f"Saved: {BINARY_PARTICIPATION_CSV}")

    return merged


# ============================================================================
# Panel k: proportion of participating cells vs LMI bins (day 0)
# ============================================================================

def panel_k_proportion_vs_lmi(
    df,
    output_dir=OUTPUT_DIR,
    filename='figure_4k',
    save_format='svg',
    dpi=300,
):
    """
    Figure 4 Panel k: proportion of cells participating per LMI bin, day 0.

    Population-level proportion (all cells pooled) per 0.2-width LMI bin.
    Single bar per bin, no error bars. Descriptive only — no statistics.

    Saves:
        <filename>.svg       – figure
        <filename>_data.csv  – per-bin proportions
    """
    sns.set_theme(context='paper', style='ticks', palette='deep',
                  font='sans-serif', font_scale=1)

    day0 = df[df['day'] == 0].dropna(subset=['lmi']).copy()

    lmi_bins = np.arange(-1.0, 1.01, 0.2)
    bin_centers = lmi_bins[:-1] + 0.1
    bin_labels = [f'{c:.1f}' for c in bin_centers]
    day0['lmi_bin'] = pd.cut(day0['lmi'], bins=lmi_bins, labels=bin_labels,
                             include_lowest=True)

    # Population-level proportion and cell count per bin
    bin_stats = (
        day0.groupby(['reward_group', 'lmi_bin'], observed=True)['participating']
        .agg(proportion='mean', n_cells='count')
        .reset_index()
    )

    reward_groups = ['R+', 'R-']
    rg_colors = {'R+': reward_palette[1], 'R-': reward_palette[0]}
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)

    for i, rg in enumerate(reward_groups):
        ax = axes[i]
        grp_cells = day0[day0['reward_group'] == rg]
        grp       = bin_stats[bin_stats['reward_group'] == rg].set_index('lmi_bin')

        n_cells = len(grp_cells)
        n_mice  = grp_cells['mouse_id'].nunique()

        proportions = [grp.loc[b, 'proportion'] if b in grp.index else np.nan
                       for b in bin_labels]
        counts      = [int(grp.loc[b, 'n_cells']) if b in grp.index else 0
                       for b in bin_labels]

        bars = ax.bar(bin_labels, proportions, color=rg_colors[rg],
                      width=0.8, linewidth=0.6, edgecolor='black')

        # Annotate each bar with cell count
        for bar, n in zip(bars, counts):
            if n > 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.01,
                        str(n), ha='center', va='bottom',
                        fontsize=5.5, color='black')

        ax.axvline(x=bin_labels.index('0.1') - 0.5, color='gray',
                   linestyle='--', linewidth=0.8, alpha=0.6)
        ax.set_title(f'{rg}  (n = {n_cells} cells, {n_mice} mice)',
                     fontsize=10, fontweight='bold')
        ax.set_xlabel('LMI', fontsize=9)
        ax.set_ylabel('Proportion of cells participating' if i == 0 else '',
                      fontsize=9)
        ax.set_ylim(0, None)
        ax.tick_params(axis='x', rotation=45, labelsize=7)
        ax.tick_params(axis='y', labelsize=8)
        sns.despine(ax=ax)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f'{filename}.{save_format}'),
                format=save_format, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"Panel k saved: {os.path.join(output_dir, filename + '.' + save_format)}")

    bin_stats.to_csv(
        os.path.join(output_dir, f'{filename}_data.csv'), index=False)
    print(f"Panel k data saved: {output_dir}")


# ============================================================================
# Panel l: proportion of participating cells across days (LMI+ vs LMI-)
# ============================================================================

def panel_l_proportion_across_days(
    df,
    output_dir=OUTPUT_DIR,
    filename='figure_4l',
    save_format='svg',
    dpi=300,
):
    """
    Figure 4 Panel l: proportion of participating cells across days for
    LMI+ vs LMI- cells.

    Per-mouse averages with individual trajectories. Stats: 2-way
    repeated-measures ANOVA (day × LMI category) followed by Mann-Whitney U
    posthoc tests (positive vs negative at each day).

    Saves:
        <filename>.svg         – figure
        <filename>_data.csv    – per-mouse × day × LMI-category proportions
        <filename>_stats.csv   – ANOVA table + Mann-Whitney U posthoc per day
    """
    sns.set_theme(context='paper', style='ticks', palette='deep',
                  font='sans-serif', font_scale=1)

    days_sorted = sorted(DAYS)
    lmi_categories = ['positive', 'negative']
    cat_colors = {'positive': '#d62728', 'negative': '#1f77b4'}
    reward_groups = ['R+', 'R-']

    # Restrict to LMI+ and LMI- cells
    lmi_df = df[df['lmi_category'].isin(lmi_categories)].copy()

    # Per-mouse × day × LMI-category proportion of participating cells
    mouse_day_prop = (
        lmi_df
        .groupby(['mouse_id', 'reward_group', 'lmi_category', 'day'],
                 observed=True)['participating']
        .mean()
        .reset_index()
        .rename(columns={'participating': 'proportion'})
    )

    # Unique cell counts per group (across all days)
    cell_counts = {
        (rg, cat): lmi_df[
            (lmi_df['reward_group'] == rg) & (lmi_df['lmi_category'] == cat)
        ][['mouse_id', 'roi']].drop_duplicates().shape[0]
        for rg in reward_groups for cat in lmi_categories
    }

    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    all_stats_rows = []
    plot_data_rows = []

    for i, rg in enumerate(reward_groups):
        ax = axes[i]
        grp = mouse_day_prop[mouse_day_prop['reward_group'] == rg]

        sns.barplot(
            data=grp, x='day', y='proportion', hue='lmi_category',
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
                my = mdata['proportion'].values
                ax.plot(mx, my, '-', color=cat_colors[cat],
                        linewidth=0.8, alpha=0.4, zorder=5)

        # ── 2-way repeated-measures ANOVA (day × lmi_category) ──────────────
        mice_pos  = set(grp.loc[grp['lmi_category'] == 'positive', 'mouse_id'])
        mice_neg  = set(grp.loc[grp['lmi_category'] == 'negative', 'mouse_id'])
        mice_both = sorted(mice_pos & mice_neg)

        anova_rows = []
        if len(mice_both) >= 3:
            anova_data = grp[grp['mouse_id'].isin(mice_both)].copy()
            anova_data['day'] = anova_data['day'].astype(str)
            try:
                aovrm = AnovaRM(anova_data, 'proportion', 'mouse_id',
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
                p_int = tbl.loc['day:lmi_category', 'Pr > F']
                print(f"  {rg} ANOVA: day × lmi_category interaction p = {p_int:.4f}")
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
                'proportion',
            ].values
            neg_vals = grp.loc[
                (grp['lmi_category'] == 'negative') & (grp['day'] == day),
                'proportion',
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

        # Significance markers
        y_max = grp['proportion'].max() * 1.05 if len(grp) > 0 else 0.5
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
        ax.set_ylabel('Proportion of cells participating' if i == 0 else '',
                      fontsize=9)
        ax.set_ylim(0, None)
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
    print(f"Panel l saved: {os.path.join(output_dir, filename + '.' + save_format)}")

    pd.concat(plot_data_rows, ignore_index=True).to_csv(
        os.path.join(output_dir, f'{filename}_data.csv'), index=False)
    pd.DataFrame(all_stats_rows).to_csv(
        os.path.join(output_dir, f'{filename}_stats.csv'), index=False)
    print(f"Panel l data/stats saved: {output_dir}")


# ============================================================================
# Main execution
# ============================================================================

if __name__ == '__main__':
    print(f"Mode:             {MODE}")
    print(f"Output directory: {OUTPUT_DIR}")

    if MODE == 'compute':
        df = _compute_binary_participation()
    elif MODE == 'plot':
        df = _load_binary_participation()
    else:
        raise ValueError(f"Unknown MODE '{MODE}'. Use 'compute' or 'plot'.")
    print(f"\nDataset: {len(df)} cell-day records, "
          f"{df['mouse_id'].nunique()} mice, "
          f"{df[['mouse_id', 'roi']].drop_duplicates().shape[0]} unique cells")

    panel_k_proportion_vs_lmi(df, filename='figure_4k')
    panel_l_proportion_across_days(df, filename='figure_4l')
