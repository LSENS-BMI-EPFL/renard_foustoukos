"""
Circular shift control for the LMI participation rate analysis.

For each cell × day, builds a null distribution of participation rates from
N_SHIFTS circular time shifts. The same shift is applied to all cells
simultaneously, preserving inter-cell correlations while breaking temporal
alignment with reactivation events.

Cells whose real participation rate exceeds the 95th percentile of their null
distribution are classified as 'significantly reactivated' (p < 0.05, day 0).

This restricted population is then used to reproduce:
  1. Participation rate vs LMI boxplot (day 0).
  2. Temporal evolution of LMI+ vs LMI- participation across days.
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, ttest_rel
from joblib import Parallel, delayed
import warnings

sys.path.append(r'/home/aprenard/repos/NWB_analysis')
sys.path.append(r'/home/aprenard/repos/fast-learning')
import src.utils.utils_imaging as utils_imaging
import src.utils.utils_io as io
from src.utils.utils_plot import *
from src.core_analysis.reactivations.reactivation import create_whisker_template


# ============================================================================
# PARAMETERS
# ============================================================================

sampling_rate = 30
days = [-2, -1, 0, 1, 2]

# Circular shift parameters
n_shifts = 1000
min_shift_frames = 300       # 10 s at 30 Hz — minimum decorrelation gap
significance_pctile = 95     # top 5% → significant (p < 0.05)

# Participation parameters — must match reactivation_lmi_prediction.py
event_window_ms = 150
event_window_frames = int(event_window_ms / 1000 * sampling_rate)
participation_threshold = 0.10   # 10% dF/F
threshold_dff = None
min_events_for_reliability = 5

# LMI thresholds — must match reactivation_lmi_prediction.py
LMI_POSITIVE_THRESHOLD = 0.975
LMI_NEGATIVE_THRESHOLD = 0.025

n_jobs = 35
percentile_to_use = 99

# Execution mode
#   'compute' : run circular shifts, save results, then plot
#   'plot'    : load previously saved results and plot only
mode = 'compute'

# Paths
save_dir = os.path.join(io.processed_dir, 'reactivation')
_p_str = (str(int(percentile_to_use))
          if percentile_to_use == int(percentile_to_use)
          else str(int(percentile_to_use * 10)))
reactivation_results_file = os.path.join(save_dir,
                                          f'reactivation_results_p{_p_str}.pkl')

# Output CSVs — referenced here so other scripts can import them directly:
#   from circular_shift_participation_control import (
#       PARTICIPATION_PER_DAY_CSV, SIGNIFICANT_PARTICIPATION_CSV)
PARTICIPATION_PER_DAY_CSV      = os.path.join(
    save_dir, 'circular_shift_participation_per_day.csv')
SIGNIFICANT_PARTICIPATION_CSV  = os.path.join(
    save_dir, 'circular_shift_significant_participation.csv')

os.makedirs(save_dir, exist_ok=True)

# Load database
_, _, all_mice, db = io.select_sessions_from_db(
    io.db_path, io.nwb_dir, two_p_imaging='yes')

r_plus_mice, r_minus_mice = [], []
for mouse in all_mice:
    try:
        rg = io.get_mouse_reward_group_from_db(io.db_path, mouse, db=db)
        if rg == 'R+':
            r_plus_mice.append(mouse)
        elif rg == 'R-':
            r_minus_mice.append(mouse)
    except Exception:
        continue

print(f"Found {len(r_plus_mice)} R+ mice: {r_plus_mice}")
print(f"Found {len(r_minus_mice)} R- mice: {r_minus_mice}")


# ============================================================================
# CORE FUNCTIONS
# ============================================================================

def _load_reactivation_results(results_file):
    if not os.path.exists(results_file):
        raise FileNotFoundError(
            f"Reactivation results not found: {results_file}\n"
            "Run reactivation.py with mode='compute' first.")
    with open(results_file, 'rb') as f:
        data = pickle.load(f)
    return data['r_plus_results'], data['r_minus_results']


def _participation_from_3d(data_3d, events, n_timepoints, n_trials):
    """
    Vectorised participation rate per cell.

    Parameters
    ----------
    data_3d : ndarray (n_cells, n_trials, n_timepoints)
    events   : array-like of event frame indices in the flattened space
               (event_idx = trial_idx * n_timepoints + time_idx)

    Returns
    -------
    rates   : ndarray (n_cells,) or None
    n_valid : int — number of valid events used
    """
    win = event_window_frames
    valid = [ev for ev in events
             if (ev % n_timepoints) >= win
             and (ev % n_timepoints) < n_timepoints - win
             and (ev // n_timepoints) < n_trials]
    if not valid:
        return None, 0

    t_idxs  = np.array([ev % n_timepoints  for ev in valid])
    tr_idxs = np.array([ev // n_timepoints for ev in valid])

    # windows: (n_valid_events, n_cells, 2*win+1)
    windows = np.stack([
        data_3d[:, tr_idxs[i], t_idxs[i] - win:t_idxs[i] + win + 1]
        for i in range(len(valid))
    ])
    avg        = np.mean(windows, axis=2)                    # (n_valid, n_cells)
    rates      = np.mean(avg >= participation_threshold, axis=0)  # (n_cells,)
    return rates, len(valid)


def compute_participation_with_shifts(mouse, day, n_shifts, preloaded_events,
                                      verbose=False):
    """
    Compute real participation rates and a circular-shift null distribution
    for one mouse × day.

    The same random shift is applied to all cells simultaneously, preserving
    inter-cell correlations while breaking alignment with reactivation events.

    Parameters
    ----------
    mouse            : str
    day              : int
    n_shifts         : int
    preloaded_events : array-like
        Event frame indices from reactivation.py (reactivation_results.pkl).
    verbose          : bool

    Returns
    -------
    pd.DataFrame with columns:
        mouse_id, day, roi,
        participation_rate  — real participation rate
        threshold_95        — 95th percentile of null distribution
        n_events            — number of valid events used
        significant         — bool, real > threshold_95
    or None on failure.
    """
    try:
        # Load neural data
        template, _ = create_whisker_template(mouse, day, threshold_dff,
                                              verbose=False)
        folder = os.path.join(io.solve_common_paths('processed_data'), 'mice')
        xr = utils_imaging.load_mouse_xarray(
            mouse, folder, 'tensor_xarray_learning_data.nc', substracted=True)
        xr_day   = xr.sel(trial=xr['day'] == day)
        nostim   = xr_day.sel(trial=xr_day['no_stim'] == 1)

        n_cells, n_trials, n_timepoints = nostim.shape
        if n_trials < 10:
            return None

        data_3d  = np.nan_to_num(nostim.values, nan=0.0)
        roi_list = nostim['roi'].values
        n_frames = n_trials * n_timepoints

        if preloaded_events is None or len(preloaded_events) == 0:
            return None

        # Real participation rates
        real_rates, n_valid = _participation_from_3d(
            data_3d, preloaded_events, n_timepoints, n_trials)

        if real_rates is None or n_valid < min_events_for_reliability:
            return None

        # Null distribution: n_shifts × n_cells (same shift for all cells)
        data_flat  = data_3d.reshape(n_cells, n_frames)
        null_rates = np.full((n_shifts, n_cells), np.nan)

        for i_shift in range(n_shifts):
            shift      = np.random.randint(min_shift_frames, n_frames)
            shifted_3d = np.roll(data_flat, shift, axis=1).reshape(
                n_cells, n_trials, n_timepoints)
            null_r, _ = _participation_from_3d(
                shifted_3d, preloaded_events, n_timepoints, n_trials)
            if null_r is not None:
                null_rates[i_shift] = null_r

        threshold_95 = np.nanpercentile(null_rates, significance_pctile, axis=0)
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
        if verbose:
            print(f"    {mouse} day {day}: {e}")
        return None


def process_mouse(mouse, n_shifts, preloaded_results):
    """Process all days for one mouse. Returns (mouse_id, DataFrame or None)."""
    dfs = []
    for day in days:
        events = None
        if preloaded_results is not None:
            day_data = preloaded_results.get('days', {}).get(day, {})
            events   = day_data.get('events', None)

        df = compute_participation_with_shifts(mouse, day, n_shifts, events)
        if df is not None:
            dfs.append(df)

    return mouse, pd.concat(dfs, ignore_index=True) if dfs else None


# ============================================================================
# PLOT 1: PARTICIPATION RATE vs LMI BOXPLOT (SIGNIFICANT CELLS, DAY 0)
# ============================================================================

def plot_participation_vs_lmi_boxplot(day0_merged_df, save_path=None):
    """
    Boxplot of day-0 participation rate per LMI bin, restricted to cells that
    passed the circular-shift significance test on day 0.

    Mirrors Analysis 1 of spontaneous_activity_controls.py.

    Parameters
    ----------
    day0_merged_df : pd.DataFrame
        Day-0 significant cells, already merged with LMI data.
        Required columns: mouse_id, roi, participation_rate, lmi, reward_group.
    save_path : str or None
    """
    merged = day0_merged_df.dropna(subset=['lmi', 'participation_rate',
                                            'reward_group'])

    lmi_bins    = np.arange(-1.0, 1.2, 0.2)
    bin_labels  = [f'{lmi_bins[k]:.1f}–{lmi_bins[k+1]:.1f}'
                   for k in range(len(lmi_bins) - 1)]
    zero_boundary = 4.5
    group_colors  = {'R+': reward_palette[1], 'R-': reward_palette[0]}

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for i, rg in enumerate(['R+', 'R-']):
        ax    = axes[i]
        gdata = merged[merged['reward_group'] == rg].copy()

        if len(gdata) < 3:
            ax.text(0.5, 0.5, f'Insufficient data (n={len(gdata)})',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title(rg, fontweight='bold', fontsize=14)
            continue

        gdata['lmi_bin'] = pd.cut(gdata['lmi'], bins=lmi_bins,
                                   labels=bin_labels, include_lowest=True)

        sns.boxplot(data=gdata, x='lmi_bin', y='participation_rate',
                    color=group_colors[rg], ax=ax,
                    showfliers=False, linewidth=0.8)

        ax.axvline(zero_boundary, color='black', linestyle='--',
                   linewidth=0.8, alpha=0.5)

        r, p = pearsonr(gdata['lmi'].values, gdata['participation_rate'].values)
        p_str = ('p < 0.001 ***' if p < 0.001 else
                 f'p = {p:.3f} **' if p < 0.01 else
                 f'p = {p:.3f} *'  if p < 0.05 else
                 f'p = {p:.3f} ns')
        n_mice = gdata['mouse_id'].nunique()
        ax.text(0.05, 0.95,
                f'n = {len(gdata)} cells, {n_mice} mice\n'
                f'Pearson r = {r:.3f}\n{p_str}',
                transform=ax.transAxes, fontsize=10, va='top',
                bbox=dict(boxstyle='round', facecolor='white',
                          alpha=0.9, edgecolor='gray'))

        ax.set_xlabel('LMI bin', fontweight='bold', fontsize=12)
        ax.set_ylabel('Participation rate (day 0)' if i == 0 else '',
                      fontweight='bold', fontsize=12)
        ax.set_title(f'{rg}', fontweight='bold', fontsize=14)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3, axis='y')
        print(f"  {rg}: r={r:.3f}, p={p:.4f}, "
              f"n={len(gdata)} cells, {n_mice} mice")

    fig.suptitle(
        'Participation Rate vs LMI — Significantly Reactivated Cells (Day 0)',
        fontsize=13, fontweight='bold')
    sns.despine()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format='svg', dpi=300, bbox_inches='tight')
        print(f"  Saved: {save_path}")

    return fig


# ============================================================================
# PLOT 2: TEMPORAL EVOLUTION (SIGNIFICANT CELLS)
# ============================================================================

def plot_temporal_evolution_significant(sig_part_df, save_path=None):
    """
    Temporal evolution of LMI+ vs LMI- participation across days, restricted
    to cells that passed the day-0 circular-shift significance test.

    Mirrors plot_temporal_evolution in reactivation_lmi_prediction.py.

    Parameters
    ----------
    sig_part_df : pd.DataFrame
        All-days participation rates for significant cells, already merged with
        LMI data. Required columns: mouse_id, roi, day, participation_rate,
        lmi_category, reward_group.
    save_path : str or None
    """
    lmi_categories = ['positive', 'negative']
    reward_groups  = ['R+', 'R-']
    colors         = {'positive': '#d62728', 'negative': '#1f77b4'}

    # Aggregate to per-mouse per-day means
    records = []
    for rg in reward_groups:
        for lmi_cat in lmi_categories:
            mask  = ((sig_part_df['reward_group']  == rg) &
                     (sig_part_df['lmi_category'] == lmi_cat))
            cells = sig_part_df[mask][['mouse_id', 'roi']].drop_duplicates()
            for mouse_id in cells['mouse_id'].unique():
                mouse_rois = cells[cells['mouse_id'] == mouse_id]['roi'].values
                for day in days:
                    day_mask = ((sig_part_df['mouse_id'] == mouse_id) &
                                (sig_part_df['roi'].isin(mouse_rois)) &
                                (sig_part_df['day'] == day))
                    vals = sig_part_df[day_mask]['participation_rate'].dropna().values
                    records.append({
                        'reward_group': rg, 'lmi_category': lmi_cat,
                        'mouse_id': mouse_id, 'day': day,
                        'participation_rate': (np.mean(vals)
                                               if len(vals) > 0 else np.nan)
                    })

    plot_df = pd.DataFrame(records).dropna(subset=['participation_rate'])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for col, rg in enumerate(reward_groups):
        ax    = axes[col]
        df_rg = plot_df[plot_df['reward_group'] == rg]

        # Per-day paired t-tests (LMI+ vs LMI-)
        print(f"\n  {rg} paired t-tests (LMI+ vs LMI-):")
        posthoc = {d: {'p': np.nan} for d in days}
        for day in days:
            pos_d = df_rg[(df_rg['lmi_category'] == 'positive') &
                          (df_rg['day'] == day)]
            neg_d = df_rg[(df_rg['lmi_category'] == 'negative') &
                          (df_rg['day'] == day)]
            mice_both = sorted(set(pos_d['mouse_id']) & set(neg_d['mouse_id']))
            if len(mice_both) >= 3:
                pos_v = [pos_d[pos_d['mouse_id'] == m]['participation_rate'].values[0]
                         for m in mice_both]
                neg_v = [neg_d[neg_d['mouse_id'] == m]['participation_rate'].values[0]
                         for m in mice_both]
                try:
                    _, p = ttest_rel(pos_v, neg_v)
                    posthoc[day] = {'p': p}
                    sig = ('***' if p < 0.001 else '**' if p < 0.01
                           else '*' if p < 0.05 else 'ns')
                    print(f"    Day {day:+2d}: n={len(mice_both)} mice, "
                          f"p={p:.4f} {sig}")
                except Exception:
                    pass
            else:
                print(f"    Day {day:+2d}: insufficient paired data "
                      f"(n={len(mice_both)})")

        # Seaborn barplot: mean ± 95% CI, no end caps
        sns.barplot(
            data=df_rg, x='day', y='participation_rate', hue='lmi_category',
            hue_order=lmi_categories, palette=colors, order=days,
            estimator=np.mean, errorbar=('ci', 95), capsize=0,
            err_kws={'linewidth': 1.5}, ax=ax
        )
        for patch in ax.patches:
            patch.set_alpha(0.7)
            patch.set_edgecolor('black')
            patch.set_linewidth(0.8)

        # Individual mouse trajectories from ax.containers
        for j, lmi_cat in enumerate(lmi_categories):
            if j >= len(ax.containers):
                continue
            cat_data = df_rg[df_rg['lmi_category'] == lmi_cat]
            x_centers = {days[k]: bar.get_x() + bar.get_width() / 2
                         for k, bar in enumerate(ax.containers[j])}
            for mouse_id in cat_data['mouse_id'].unique():
                mdata = (cat_data[cat_data['mouse_id'] == mouse_id]
                         .sort_values('day'))
                mx = [x_centers[d] for d in mdata['day'] if d in x_centers]
                my = mdata['participation_rate'].values
                ax.plot(mx, my, '-', color=colors[lmi_cat],
                        linewidth=0.8, alpha=0.4, zorder=5)

        ax.set_ylim(0, 0.6)
        y_sig = ax.get_ylim()[1] * 0.92
        for k, day in enumerate(days):
            p_val = posthoc[day]['p']
            if not np.isnan(p_val) and p_val < 0.05:
                sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*'
                x0 = (ax.containers[0][k].get_x() +
                       ax.containers[0][k].get_width() / 2)
                x1 = (ax.containers[1][k].get_x() +
                       ax.containers[1][k].get_width() / 2)
                ax.text((x0 + x1) / 2, y_sig, sig,
                        ha='center', va='bottom', fontsize=12, fontweight='bold')

        ax.set_xlabel('Day', fontweight='bold', fontsize=13)
        ax.set_ylabel('Participation Rate', fontweight='bold', fontsize=13)
        ax.set_title(f'{rg} Mice — Significant Cells', fontweight='bold',
                     fontsize=14)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, [f'{l.capitalize()} LMI' for l in labels],
                  loc='best', fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        sns.despine(ax=ax)

    fig.suptitle(
        'Temporal Evolution — Significantly Reactivated Cells\n'
        '(Circular Shift Control, p < 0.05 on Day 0)',
        fontsize=13, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format='svg', dpi=300, bbox_inches='tight')
        print(f"  Saved: {save_path}")

    return fig


# ============================================================================
# MAIN EXECUTION
# ============================================================================

print("\n" + "="*60)
print("CIRCULAR SHIFT CONTROL — PARTICIPATION RATE SIGNIFICANCE")
print("="*60)
print(f"n_shifts = {n_shifts}, significance = {significance_pctile}th "
      f"percentile, min_shift = {min_shift_frames} frames "
      f"({min_shift_frames/sampling_rate:.0f} s)")

if mode == 'compute':
    # --- Load pre-computed reactivation events ---
    print("\nLoading pre-computed reactivation events...")
    r_plus_results, r_minus_results = _load_reactivation_results(
        reactivation_results_file)
    all_results = {}
    all_results.update(r_plus_results)
    all_results.update(r_minus_results)
    print(f"Loaded events for {len(all_results)} mice")

    # --- Process all mice in parallel ---
    all_mice_to_process = r_plus_mice + r_minus_mice
    print(f"\nProcessing {len(all_mice_to_process)} mice × {len(days)} days × "
          f"{n_shifts} shifts each...")

    results_list = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(process_mouse)(mouse, n_shifts, all_results.get(mouse, None))
        for mouse in all_mice_to_process
    )

    all_dfs = [df for _, df in results_list if df is not None]
    if not all_dfs:
        print("ERROR: No valid data collected!")
        raise SystemExit(1)

    participation_df = pd.concat(all_dfs, ignore_index=True)
    print(f"\nCollected {len(participation_df)} cell-day records")

    participation_df.to_csv(PARTICIPATION_PER_DAY_CSV, index=False)
    print(f"Saved: {PARTICIPATION_PER_DAY_CSV}")

else:  # mode == 'plot'
    print(f"\nLoading pre-computed results from disk...")
    if not os.path.exists(PARTICIPATION_PER_DAY_CSV):
        raise FileNotFoundError(
            f"Results not found: {PARTICIPATION_PER_DAY_CSV}\n"
            "Run with mode='compute' first.")
    participation_df = pd.read_csv(PARTICIPATION_PER_DAY_CSV)
    print(f"Loaded {len(participation_df)} cell-day records")

# --- Identify significantly reactivated cells on day 0 ---
day0_df  = participation_df[participation_df['day'] == 0].copy()
sig_mask = day0_df['significant']
sig_cells = day0_df[sig_mask][['mouse_id', 'roi']].drop_duplicates()

print(f"\nDay-0 significant cells: {len(sig_cells)} / {len(day0_df)} "
      f"({100*len(sig_cells)/len(day0_df):.1f}%)")

# --- Load LMI data ---
lmi_df = pd.read_csv(os.path.join(io.processed_dir, 'lmi_results.csv'))
if 'reward_group' not in lmi_df.columns:
    mice_rg = db[['mouse_id', 'reward_group']].drop_duplicates()
    lmi_df['reward_group'] = lmi_df['mouse_id'].map(
        dict(mice_rg[['mouse_id', 'reward_group']].values))

lmi_df['lmi_category'] = 'neutral'
lmi_df.loc[lmi_df['lmi_p'] >= LMI_POSITIVE_THRESHOLD, 'lmi_category'] = 'positive'
lmi_df.loc[lmi_df['lmi_p'] <= LMI_NEGATIVE_THRESHOLD, 'lmi_category'] = 'negative'

# Diagnostic: fraction of cells significant per LMI category
day0_with_lmi = pd.merge(
    day0_df,
    lmi_df[['mouse_id', 'roi', 'lmi_category', 'reward_group']],
    on=['mouse_id', 'roi'], how='inner')

print("\n  Fraction significant per LMI category (day 0):")
for rg in ['R+', 'R-']:
    for cat in ['positive', 'negative']:
        sub  = day0_with_lmi[(day0_with_lmi['reward_group'] == rg) &
                              (day0_with_lmi['lmi_category'] == cat)]
        n_sig  = sub['significant'].sum()
        n_tot  = len(sub)
        pct    = 100 * n_sig / n_tot if n_tot > 0 else 0
        print(f"    {rg} {cat}: {n_sig}/{n_tot} ({pct:.1f}%)")

# --- Build restricted dataset: significant cells, all days, with LMI info ---
sig_part_df = pd.merge(participation_df, sig_cells,
                        on=['mouse_id', 'roi'], how='inner')
sig_part_df = pd.merge(
    sig_part_df,
    lmi_df[['mouse_id', 'roi', 'lmi', 'lmi_p', 'lmi_category', 'reward_group']],
    on=['mouse_id', 'roi'], how='inner')

sig_part_df.to_csv(SIGNIFICANT_PARTICIPATION_CSV, index=False)
print(f"\nSaved significant-cell participation: {SIGNIFICANT_PARTICIPATION_CSV}")

# ============================================================================
# PLOT 1
print("\n" + "="*60)
print("PLOT 1: PARTICIPATION RATE vs LMI (SIGNIFICANT CELLS, DAY 0)")
print("="*60)

day0_sig_df = sig_part_df[sig_part_df['day'] == 0]
svg_path    = os.path.join(save_dir,
                            'circular_shift_participation_vs_lmi_boxplot.svg')
plot_participation_vs_lmi_boxplot(day0_sig_df, svg_path)

# ============================================================================
# PLOT 2
print("\n" + "="*60)
print("PLOT 2: TEMPORAL EVOLUTION (SIGNIFICANT CELLS)")
print("="*60)

svg_path = os.path.join(save_dir,
                         'circular_shift_temporal_evolution_significant.svg')
plot_temporal_evolution_significant(sig_part_df, svg_path)

print("\n" + "="*60)
print("DONE")
print("="*60)
print(f"\nResults saved to: {save_dir}")
