"""
Supplementary Figure 4c: Participation rate across days for significantly-
participating cells (LMI+ vs LMI-).

Cells are classified as 'significantly participating' via circular-shift
control: circular-shift null distribution (N_SHIFTS shifts) is built per
mouse x day, and a cell is retained if its day-0 participation rate exceeds
the SIGNIFICANCE_PCTILE-th percentile of its null distribution.

Panel: Participation rate across days (-2 to +2) for LMI+ vs LMI- cells in
       the significantly-participating subpopulation. Per-mouse averages with
       individual trajectories. Stats: Kruskal-Wallis test (effect of day)
       run independently for each of the four groups (R+ positive LMI,
       R+ negative LMI, R- positive LMI, R- negative LMI).

Execution modes:
    MODE = 'compute' : run full circular-shift pipeline, save CSVs, then plot
    MODE = 'plot'    : load previously saved CSVs and plot only

Processed data files are saved/loaded from data_processed/reactivation/.
Figures and CSVs are saved to io.manuscript_output_dir/supp_4/output/.
"""

import os
import sys
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kruskal
from joblib import Parallel, delayed

sys.path.append('/home/aprenard/repos/fast-learning')
import src.utils.utils_io as io
import src.utils.utils_imaging as utils_imaging


# ============================================================================
# Parameters
# ============================================================================

DAYS = [-2, -1, 0, 1, 2]
LMI_POSITIVE_THRESHOLD = 0.975
LMI_NEGATIVE_THRESHOLD = 0.025
N_JOBS = 35

RESULTS_DIR = os.path.join(io.processed_dir, 'reactivation')
REACTIVATION_RESULTS_FILE = os.path.join(RESULTS_DIR, 'reactivation_results_p99.pkl')
LMI_RESULTS_CSV = os.path.join(io.processed_dir, 'lmi_results.csv')
SUPP4C_RATES_CSV  = os.path.join(RESULTS_DIR, 'supp4c_rates_per_day.csv')
SUPP4C_SIG_CSV    = os.path.join(RESULTS_DIR, 'supp4c_significant_cells.csv')
SUPP4C_MERGED_CSV = os.path.join(RESULTS_DIR, 'supp4c_merged.csv')
OUTPUT_DIR = os.path.join(io.manuscript_output_dir, 'supp_4', 'output')
FOLDER = os.path.join(io.solve_common_paths('processed_data'), 'mice')

# Execution mode
#   'compute' : run circular-shift control + participation pipeline, save CSVs
#   'plot'    : load previously saved CSVs and plot only
MODE = 'compute'

# Circular shift parameters
SAMPLING_RATE = 30
N_SHIFTS = 1000
MIN_SHIFT_FRAMES = 300        # 10 s at 30 Hz -- minimum decorrelation gap
SIGNIFICANCE_PCTILE = 95      # top 5 % -> p < 0.05
EVENT_WINDOW_MS = 150
EVENT_WINDOW_FRAMES = int(EVENT_WINDOW_MS / 1000 * SAMPLING_RATE)
PARTICIPATION_THRESHOLD = 0.10
THRESHOLD_DFF = None
MIN_EVENTS_FOR_RELIABILITY = 5


# ============================================================================
# Mouse loading
# ============================================================================

_, _, _all_mice, _db = io.select_sessions_from_db(
    io.db_path, io.nwb_dir, two_p_imaging='yes'
)

r_plus_mice, r_minus_mice = [], []
for _mouse in _all_mice:
    try:
        _rg = io.get_mouse_reward_group_from_db(io.db_path, _mouse, db=_db)
        if _rg == 'R+':
            r_plus_mice.append(_mouse)
        elif _rg == 'R-':
            r_minus_mice.append(_mouse)
    except:
        continue

print(f"Found {len(r_plus_mice)} R+ mice and {len(r_minus_mice)} R- mice")


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
# Participation computation
# ============================================================================

def _extract_event_responses(mouse, day, preloaded_events):
    """
    Extract per-cell dF/F responses around pre-computed reactivation events.

    Uses no-stim trials only. Responses are averaged over ±EVENT_WINDOW_FRAMES
    around each event. Cells are flagged as participating if their mean response
    exceeds PARTICIPATION_THRESHOLD.

    Returns DataFrame (mouse_id, day, roi, event_idx, avg_response, participates)
    or None if insufficient data (<10 no-stim trials or no valid events).
    """
    xarr = utils_imaging.load_mouse_xarray(
        mouse, FOLDER, 'tensor_xarray_learning_data.nc', substracted=True
    )
    xarr_day = xarr.sel(trial=xarr['day'] == day)
    nostim = xarr_day.sel(trial=xarr_day['no_stim'] == 1)

    if len(nostim.trial) < 10:
        return None

    n_cells, n_trials, n_timepoints = nostim.shape
    data_3d = nostim.values
    roi_list = nostim['roi'].values
    win = EVENT_WINDOW_FRAMES

    rows = []
    for event_idx in preloaded_events:
        trial_idx = event_idx // n_timepoints
        time_idx  = event_idx % n_timepoints
        if time_idx < win or time_idx >= n_timepoints - win or trial_idx >= n_trials:
            continue
        window_data  = data_3d[:, trial_idx, time_idx - win:time_idx + win + 1]
        avg_response = np.mean(window_data, axis=1)
        participates = avg_response >= PARTICIPATION_THRESHOLD
        for icell in range(n_cells):
            rows.append({
                'mouse_id': mouse, 'day': day, 'roi': roi_list[icell],
                'event_idx': event_idx, 'avg_response': float(avg_response[icell]),
                'participates': bool(participates[icell]),
            })

    return pd.DataFrame(rows) if rows else None


def _compute_participation_rate(responses_df):
    """Aggregate cell-event responses to per-cell participation rates."""
    grouped = responses_df.groupby(['mouse_id', 'day', 'roi']).agg(
        n_participations=('participates', 'sum'),
        n_events=('participates', 'count'),
    ).reset_index()
    grouped['participation_rate'] = grouped['n_participations'] / grouped['n_events']
    grouped['reliable'] = grouped['n_events'] >= MIN_EVENTS_FOR_RELIABILITY
    return grouped


def _process_mouse_participation(mouse, preloaded_results):
    """Compute participation rates across all days for one mouse."""
    all_responses = []
    for day in DAYS:
        events = preloaded_results.get('days', {}).get(day, {}).get('events', None)
        if events is None or len(events) == 0:
            continue
        try:
            resp_df = _extract_event_responses(mouse, day, events)
            if resp_df is not None and len(resp_df) > 0:
                all_responses.append(resp_df)
        except Exception as e:
            print(f"  Warning: {mouse} day {day}: {e}")
    if not all_responses:
        return mouse, None
    all_resp_df = pd.concat(all_responses, ignore_index=True)
    return mouse, _compute_participation_rate(all_resp_df)


# ============================================================================
# Aggregation and LMI merging
# ============================================================================

def _aggregate_across_days(participation_df_all):
    """Aggregate per-day participation rates to baseline/learning/post periods."""
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

        results.append({
            'mouse_id': mouse_id, 'roi': roi,
            'baseline_rate': baseline_rate,
            'learning_rate': learning_rate,
            'post_rate': post_rate,
            'delta_learning': learning_rate - baseline_rate if not np.isnan(baseline_rate) else np.nan,
            'delta_post': post_rate - baseline_rate if not np.isnan(baseline_rate) else np.nan,
            'reliable_baseline': reliable_baseline,
            'reliable_learning': reliable_learning,
            'reliable_post': reliable_post,
        })

    return pd.DataFrame(results)


def _load_and_match_lmi_data(participation_df):
    """Load LMI data and merge with participation data."""
    lmi_df = pd.read_csv(LMI_RESULTS_CSV)

    if 'reward_group' not in lmi_df.columns:
        group_map = {m: 'R+' for m in r_plus_mice}
        group_map.update({m: 'R-' for m in r_minus_mice})
        lmi_df['reward_group'] = lmi_df['mouse_id'].map(group_map)

    lmi_df['lmi_category'] = 'neutral'
    lmi_df.loc[lmi_df['lmi_p'] >= LMI_POSITIVE_THRESHOLD, 'lmi_category'] = 'positive'
    lmi_df.loc[lmi_df['lmi_p'] <= LMI_NEGATIVE_THRESHOLD, 'lmi_category'] = 'negative'

    cols = ['mouse_id', 'roi', 'lmi', 'lmi_p', 'lmi_category', 'reward_group']
    if 'cell_type' in lmi_df.columns:
        cols.append('cell_type')

    merged_df = pd.merge(participation_df, lmi_df[cols], on=['mouse_id', 'roi'], how='inner')

    print(f"\n  Merged: {len(merged_df)} cells total "
          f"({(merged_df['lmi_category']=='positive').sum()} LMI+, "
          f"{(merged_df['lmi_category']=='negative').sum()} LMI-, "
          f"{(merged_df['lmi_category']=='neutral').sum()} neutral)")
    return merged_df


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
    avg   = np.mean(windows, axis=2)
    rates = np.mean(avg >= PARTICIPATION_THRESHOLD, axis=0)
    return rates, len(valid)


def _compute_participation_with_shifts(mouse, day, n_shifts, preloaded_events):
    """Compute real participation rates and circular-shift null distribution
    for one mouse x day.

    Returns a DataFrame with columns
        mouse_id, day, roi, participation_rate, threshold_95, n_events, significant
    or None on failure.
    """
    try:
        xr = utils_imaging.load_mouse_xarray(
            mouse, FOLDER, 'tensor_xarray_learning_data.nc', substracted=False)
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


def _compute_circular_shift_significance(all_results):
    """Run circular-shift control and return set of significant (mouse, roi) pairs.

    Saves SUPP4C_SIG_CSV to RESULTS_DIR.
    """
    all_mice = list(all_results.keys())
    print(f"\nRunning circular-shift control for {len(all_mice)} mice "
          f"({N_SHIFTS} shifts x {len(DAYS)} days each) ...")
    raw = Parallel(n_jobs=N_JOBS, verbose=5)(
        delayed(_process_mouse_circular_shift)(
            mouse, N_SHIFTS, all_results.get(mouse))
        for mouse in all_mice
    )

    dfs = [df for _, df in raw if df is not None]
    if not dfs:
        raise RuntimeError("Circular shift control returned no data.")
    participation_df = pd.concat(dfs, ignore_index=True)

    day0_df   = participation_df[participation_df['day'] == 0].copy()
    sig_cells = day0_df[day0_df['significant']][['mouse_id', 'roi']].drop_duplicates()

    n_sig = len(sig_cells)
    n_tot = len(day0_df)
    print(f"Day-0 significant cells: {n_sig}/{n_tot} ({100*n_sig/n_tot:.1f}%)")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    sig_cells.to_csv(SUPP4C_SIG_CSV, index=False)
    print(f"Saved: {SUPP4C_SIG_CSV}")

    return set(zip(sig_cells['mouse_id'], sig_cells['roi']))


# ============================================================================
# Data loading / computation
# ============================================================================

def _compute_supp4c_data():
    """Run full pipeline: participation rates + circular-shift significance filter.

    1. Load reactivation events.
    2. Compute per-cell participation rates per day.
    3. Aggregate and merge with LMI data.
    4. Run circular-shift control and restrict to day-0 significant cells.

    Saves SUPP4C_RATES_CSV, SUPP4C_SIG_CSV, SUPP4C_MERGED_CSV.

    Returns
    -------
    merged_df  : pd.DataFrame  -- sig-cell aggregated data with LMI info
    per_day_df : pd.DataFrame  -- sig-cell per-day participation rates
    """
    if not os.path.exists(REACTIVATION_RESULTS_FILE):
        raise FileNotFoundError(
            f"Reactivation results not found: {REACTIVATION_RESULTS_FILE}")

    with open(REACTIVATION_RESULTS_FILE, 'rb') as f:
        data = pickle.load(f)
    r_plus_results  = data['r_plus_results']
    r_minus_results = data['r_minus_results']
    all_results = {**r_plus_results, **r_minus_results}
    all_mice = list(all_results.keys())

    # Participation rates
    print(f"\nComputing participation rates for {len(all_mice)} mice...")
    results_list = Parallel(n_jobs=N_JOBS, verbose=10)(
        delayed(_process_mouse_participation)(
            mouse, all_results.get(mouse, {}),
        )
        for mouse in all_mice
    )
    all_data = [df for _, df in results_list if df is not None]
    if not all_data:
        raise RuntimeError("No participation data computed.")
    per_day_df = pd.concat(all_data, ignore_index=True)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    per_day_df.to_csv(SUPP4C_RATES_CSV, index=False)

    aggregated_df = _aggregate_across_days(per_day_df)
    merged_df = _load_and_match_lmi_data(aggregated_df)

    # Circular shift significance filter
    sig_keys = _compute_circular_shift_significance(all_results)
    mask_merged = [(m, r) in sig_keys
                   for m, r in zip(merged_df['mouse_id'], merged_df['roi'])]
    mask_perday = [(m, r) in sig_keys
                   for m, r in zip(per_day_df['mouse_id'], per_day_df['roi'])]
    merged_df  = merged_df[mask_merged].reset_index(drop=True)
    per_day_df = per_day_df[mask_perday].reset_index(drop=True)

    # Add lmi_category
    merged_df['lmi_category'] = 'neutral'
    merged_df.loc[merged_df['lmi_p'] >= LMI_POSITIVE_THRESHOLD,
                  'lmi_category'] = 'positive'
    merged_df.loc[merged_df['lmi_p'] <= LMI_NEGATIVE_THRESHOLD,
                  'lmi_category'] = 'negative'

    merged_df.to_csv(SUPP4C_MERGED_CSV, index=False)
    print(f"Saved: {SUPP4C_MERGED_CSV}")

    return merged_df, per_day_df


def _load_supp4c_data():
    """Load pre-computed supp_4c data from CSVs."""
    for path in [SUPP4C_RATES_CSV, SUPP4C_MERGED_CSV]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Pre-computed data not found: {path}\n"
                "Run with MODE='compute' first.")
    merged_df  = pd.read_csv(SUPP4C_MERGED_CSV)
    per_day_df = pd.read_csv(SUPP4C_RATES_CSV)

    # Filter per_day_df to sig cells recorded in merged_df
    sig_keys = set(zip(merged_df['mouse_id'], merged_df['roi']))
    mask = [(m, r) in sig_keys
            for m, r in zip(per_day_df['mouse_id'], per_day_df['roi'])]
    per_day_df = per_day_df[mask].reset_index(drop=True)

    print(f"Loaded {len(merged_df)} sig cells, {len(per_day_df)} cell-day records.")
    return merged_df, per_day_df


# ============================================================================
# Panel: participation rate across days (LMI+ vs LMI-, sig cells only)
# ============================================================================

def panel_supp4c_participation_across_days(
    merged_df,
    per_day_df,
    output_dir=OUTPUT_DIR,
    filename='supp_4c',
    save_format='svg',
    dpi=300,
):
    """Supp Figure 4c: participation rate across days for significantly-
    participating LMI+ vs LMI- cells.

    Per-mouse averages with individual trajectories. Stats: Kruskal-Wallis
    test (effect of day) run independently for each of the four groups
    (R+ positive LMI, R+ negative LMI, R- positive LMI, R- negative LMI).

    Saves:
        <filename>.svg         -- figure
        <filename>_data.csv    -- per-mouse x day x LMI-category averages
        <filename>_stats.csv   -- Kruskal-Wallis results per group
    """
    sns.set_theme(context='paper', style='ticks', palette='deep',
                  font='sans-serif', font_scale=1)

    days_sorted = sorted(DAYS)
    lmi_categories = ['positive', 'negative']
    cat_colors = {'positive': '#d62728', 'negative': '#1f77b4'}
    reward_groups = ['R+', 'R-']

    lmi_cells = merged_df.loc[
        merged_df['lmi_category'].isin(lmi_categories),
        ['mouse_id', 'roi', 'lmi_category', 'reward_group'],
    ]
    day_data = pd.merge(per_day_df, lmi_cells, on=['mouse_id', 'roi'], how='inner')

    mouse_day_avg = (
        day_data
        .groupby(['mouse_id', 'reward_group', 'lmi_category', 'day'],
                 observed=True)['participation_rate']
        .mean()
        .reset_index()
    )

    cell_counts = (
        lmi_cells.groupby(['reward_group', 'lmi_category'], observed=True)
        .size()
        .to_dict()
    )

    # Kruskal-Wallis: effect of day within each (reward_group, lmi_category) group
    all_stats_rows = []
    kw_results = {}
    for rg in reward_groups:
        for cat in lmi_categories:
            grp_data = mouse_day_avg[
                (mouse_day_avg['reward_group'] == rg) &
                (mouse_day_avg['lmi_category'] == cat)
            ]
            day_groups = [
                grp_data[grp_data['day'] == day]['participation_rate'].values
                for day in days_sorted
            ]
            day_groups = [g for g in day_groups if len(g) > 0]
            if len(day_groups) >= 2:
                try:
                    H, p = kruskal(*day_groups)
                except Exception:
                    H, p = np.nan, np.nan
            else:
                H, p = np.nan, np.nan
            kw_results[(rg, cat)] = (H, p)
            all_stats_rows.append({
                'reward_group': rg,
                'lmi_category': cat,
                'test': 'Kruskal-Wallis',
                'effect': 'day',
                'H_statistic': H,
                'p_value': p,
                'significance': _significance_stars(p) if not np.isnan(p) else 'n.a.',
                'n_days': len(day_groups),
            })
            print(f"  KW {rg} {cat} LMI: H={H:.3f}, p={p:.4g}")

    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
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

        # Annotate Kruskal-Wallis results for each LMI group
        for j, cat in enumerate(lmi_categories):
            H, p = kw_results.get((rg, cat), (np.nan, np.nan))
            stars = _significance_stars(p) if not np.isnan(p) else 'n.a.'
            ax.text(0.02, 0.97 - j * 0.12,
                    f'{cat.capitalize()} LMI: KW p={p:.3g} {stars}',
                    transform=ax.transAxes, va='top', ha='left',
                    fontsize=7, color=cat_colors[cat])

        n_pos = cell_counts.get((rg, 'positive'), 0)
        n_neg = cell_counts.get((rg, 'negative'), 0)
        ax.set_title(f'{rg}  (LMI+: {n_pos} cells | LMI-: {n_neg} cells)',
                     fontsize=9, fontweight='bold')
        ax.set_xlabel('Day', fontsize=9)
        ax.set_ylabel('Participation rate (sig. cells)' if i == 0 else '', fontsize=9)
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
    print(f"Panel saved: {os.path.join(output_dir, filename + '.' + save_format)}")

    pd.concat(plot_data_rows, ignore_index=True).to_csv(
        os.path.join(output_dir, f'{filename}_data.csv'), index=False)
    pd.DataFrame(all_stats_rows).to_csv(
        os.path.join(output_dir, f'{filename}_stats.csv'), index=False)
    print(f"Data/stats saved: {output_dir}")


# ============================================================================
# Main execution
# ============================================================================

if __name__ == '__main__':
    print(f"Mode:             {MODE}")
    print(f"Output directory: {OUTPUT_DIR}")

    if MODE == 'compute':
        merged_df, per_day_df = _compute_supp4c_data()
    elif MODE == 'plot':
        merged_df, per_day_df = _load_supp4c_data()
    else:
        raise ValueError(f"Unknown MODE '{MODE}'. Use 'compute' or 'plot'.")

    print(f"\nDataset: {len(merged_df)} sig cells, "
          f"{len(per_day_df)} cell-day records, "
          f"{merged_df['mouse_id'].nunique()} mice")

    panel_supp4c_participation_across_days(merged_df, per_day_df, filename='supp_4c')
