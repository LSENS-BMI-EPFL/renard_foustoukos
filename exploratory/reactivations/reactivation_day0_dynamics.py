"""
Day-0 within-day reactivation dynamics (exploratory).

A reviewer suggested examining how reactivation evolves within Day 0 and
how its time course aligns with behavioral performance. This is genuinely
interesting but out of scope for the current manuscript -- this script is
a quick, curiosity-driven look, not manuscript-bound analysis. It answers
three questions:

  1. How many reactivation events are detected per trial?
  2. How does that compare before vs. after the first whisker hit?
  3. Does reactivation start low and rise with performance (a), or start
     high from the beginning of the session and get maintained by reward
     in R+ while decaying without reward in R- (b)?

Built entirely as post-processing on already-computed results -- no new
template-matching or event detection:
  - reactivation_results_p99.pkl (original all-no-stim-trials variant,
    same as used for the Figure 4H shuffle-template control) already
    stores, per mouse per day, the detected event indices and the
    no-stim-only trial data they came from (with trial_id as a
    coordinate), so per-trial event counts are a direct re-aggregation
    of already-detected events by trial instead of by whole day.
  - "First whisker hit" needs one small additional load of each mouse's
    full Day-0 tensor_xarray_learning_data.nc (coordinates only -- day,
    whisker_stim, outcome_w, early_lick, trial_id -- not the cell x time
    data), since whisker-stim trials aren't part of the stored no-stim-only
    selected_trials.

No windowing/binning: reactivation count is computed per individual trial,
not aggregated into blocks.

Outputs are saved to paths.results_dir/reactivation_day0_dynamics/.
"""

import os
import sys
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wilcoxon

from fast_learning import imaging
from fast_learning import paths
from fast_learning.plotting import reward_palette


# ============================================================================
# Parameters
# ============================================================================

DAY = 0
RESULTS_FILE = os.path.join(paths.processed_dir, 'reactivation', 'reactivation_results_p99.pkl')
OUTPUT_DIR = os.path.join(paths.results_dir, 'reactivation_day0_dynamics')
MIN_MICE_PER_POSITION = 3  # trim first-hit-aligned positions with fewer contributing mice


# ============================================================================
# Loading
# ============================================================================

def _load_reactivation_results(results_file):
    """Load pre-computed reactivation results from pickle."""
    if not os.path.exists(results_file):
        raise FileNotFoundError(
            f"Reactivation results file not found: {results_file}\n"
            "Please run reactivation_preprocessing.py with mode='compute' first.")
    print(f"Loading reactivation events from: {results_file}")
    with open(results_file, 'rb') as f:
        data = pickle.load(f)
    return data['r_plus_results'], data['r_minus_results']


def _compute_per_trial_events(mouse, day_results):
    """Per-no-stim-trial event counts for one mouse's day-0 results.

    Returns a DataFrame: mouse_id, trial_id (session-wide numbering),
    nostim_position (0-indexed position within the no-stim trial
    sequence), n_events.
    """
    events = day_results['events']
    selected_trials = day_results['selected_trials']
    n_timepoints_per_trial = selected_trials.sizes['time']
    trial_ids = selected_trials['trial_id'].values
    n_nostim_trials = len(trial_ids)

    event_trial_pos = events // n_timepoints_per_trial
    event_trial_pos = event_trial_pos[event_trial_pos < n_nostim_trials]

    counts = np.bincount(event_trial_pos, minlength=n_nostim_trials)

    return pd.DataFrame({
        'mouse_id': mouse,
        'trial_id': trial_ids,
        'nostim_position': np.arange(n_nostim_trials),
        'n_events': counts,
    })


def _find_first_hit_trial_id(mouse, day=DAY):
    """First Day-0 trial_id with a whisker hit (excluding early licks).

    Returns None if the mouse had no whisker hit on Day 0.
    """
    folder = paths.tensor_dir
    xarr = imaging.load_mouse_xarray(
        mouse, folder, 'tensor_xarray_learning_data.nc', subtracted=False)
    xarr_day = xarr.sel(trial=xarr['day'] == day)

    whisker_stim = xarr_day['whisker_stim'].values
    outcome_w = xarr_day['outcome_w'].values
    early_lick = xarr_day['early_lick'].values
    trial_id = xarr_day['trial_id'].values

    hit_mask = (whisker_stim == 1) & (outcome_w == 1) & (early_lick == 0)
    hit_trial_ids = trial_id[hit_mask]
    if len(hit_trial_ids) == 0:
        return None
    return float(np.min(hit_trial_ids))


def _add_relative_position(df_events, first_hit_trial_id):
    """Add each no-stim trial's position relative to the first whisker hit,
    in no-stim-trial steps (so a step of 1 always means "the next no-stim
    trial", regardless of how many whisker/auditory trials fall in
    between) -- not raw trial_id, and not any fixed-size block/window.
    """
    df = df_events.copy()
    if first_hit_trial_id is None:
        df['relative_position'] = np.nan
        return df

    after_mask = df['trial_id'] >= first_hit_trial_id
    if not after_mask.any():
        # First hit happens after every no-stim trial in the day.
        df['relative_position'] = df['nostim_position'] - (df['nostim_position'].max() + 1)
        return df

    hit_nostim_position = df.loc[after_mask, 'nostim_position'].min()
    df['relative_position'] = df['nostim_position'] - hit_nostim_position
    return df


def build_dataset(day=DAY):
    """Assemble the full per-trial event/first-hit dataset across mice."""
    r_plus_results, r_minus_results = _load_reactivation_results(RESULTS_FILE)

    all_dfs = []
    missing_hit = []
    for reward_group, results in [('R+', r_plus_results), ('R-', r_minus_results)]:
        for mouse, res in results.items():
            day_results = res.get('days', {}).get(day)
            if day_results is None or 'events' not in day_results:
                print(f"  Skipping {mouse}: no day-{day} results")
                continue

            df_events = _compute_per_trial_events(mouse, day_results)
            first_hit = _find_first_hit_trial_id(mouse, day=day)
            if first_hit is None:
                missing_hit.append(mouse)
            df_events = _add_relative_position(df_events, first_hit)
            df_events['reward_group'] = reward_group
            df_events['first_hit_trial_id'] = first_hit
            all_dfs.append(df_events)

    df = pd.concat(all_dfs, ignore_index=True)
    if missing_hit:
        print(f"  No Day-0 whisker hit found for: {missing_hit} "
              f"(excluded from before/after and aligned-trajectory analyses)")
    return df, missing_hit


# ============================================================================
# Plot 1: per-trial event count distribution (Q1)
# ============================================================================

def plot_per_trial_distribution(df, output_dir=OUTPUT_DIR, filename='per_trial_event_distribution'):
    sns.set_theme(context='paper', style='ticks', palette='deep', font='sans-serif', font_scale=1)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    for ax, rg in zip(axes, ['R+', 'R-']):
        d = df[df['reward_group'] == rg]
        max_n = int(d['n_events'].max())
        color = reward_palette[1] if rg == 'R+' else reward_palette[0]
        sns.histplot(d['n_events'], bins=np.arange(-0.5, max_n + 1.5, 1),
                     stat='probability', color=color, ax=ax)
        ax.set_title(f'{rg} (n={d["mouse_id"].nunique()} mice, {len(d)} trials)')
        ax.set_xlabel('Reactivation events per trial')
        ax.set_ylabel('Proportion of trials' if ax is axes[0] else '')
        sns.despine(ax=ax)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f'{filename}.svg'), format='svg', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {os.path.join(output_dir, filename + '.svg')}")

    df.to_csv(os.path.join(output_dir, f'{filename}_data.csv'), index=False)


# ============================================================================
# Plot 2: first-hit-aligned trajectory, unwindowed (Q3)
# ============================================================================

def plot_first_hit_aligned_trajectory(df, output_dir=OUTPUT_DIR,
                                       filename='first_hit_aligned_trajectory',
                                       min_mice_per_position=MIN_MICE_PER_POSITION):
    """Mean event count per relative-trial-position (no binning), R+ vs R-,
    with a vertical line at position 0 (the first no-stim trial at/after
    the first whisker hit). This is the plot that distinguishes hypothesis
    (a) (rise starting at/after position 0) from (b) (already high and
    flat/declining on both sides of 0, diverging by reward group).

    Positions with fewer than min_mice_per_position contributing mice are
    dropped rather than silently averaged over a shrinking, potentially
    biased sample (sessions have finite length and first-hit position
    varies per mouse, so N shrinks toward both edges).
    """
    sns.set_theme(context='paper', style='ticks', palette='deep', font='sans-serif', font_scale=1)

    d = df.dropna(subset=['relative_position']).copy()
    d['relative_position'] = d['relative_position'].astype(int)

    per_mouse = (
        d.groupby(['reward_group', 'mouse_id', 'relative_position'])['n_events']
        .mean()
        .reset_index()
    )

    n_mice_per_pos = per_mouse.groupby(['reward_group', 'relative_position'])['mouse_id'].nunique()
    valid_positions = n_mice_per_pos[n_mice_per_pos >= min_mice_per_position].index
    per_mouse = per_mouse.set_index(['reward_group', 'relative_position'])
    per_mouse = per_mouse.loc[per_mouse.index.isin(valid_positions)].reset_index()

    fig, ax = plt.subplots(1, 1, figsize=(9, 5))
    sns.lineplot(
        data=per_mouse, x='relative_position', y='n_events', hue='reward_group',
        palette={'R+': reward_palette[1], 'R-': reward_palette[0]},
        hue_order=['R+', 'R-'], errorbar='ci', marker='o', markersize=4, ax=ax,
    )
    ax.axvline(0, color='black', linestyle='--', linewidth=1.2, alpha=0.7,
               label='First whisker hit')
    ax.set_xlabel('No-stim trial position relative to first whisker hit')
    ax.set_ylabel('Mean reactivation events per trial')
    ax.set_title(f'Day 0: reactivation aligned to first whisker hit\n'
                 f'(positions with <{min_mice_per_position} mice trimmed)')
    ax.legend(title='')
    sns.despine()
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f'{filename}.svg'), format='svg', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {os.path.join(output_dir, filename + '.svg')}")

    per_mouse.to_csv(os.path.join(output_dir, f'{filename}_data.csv'), index=False)


# ============================================================================
# Plot 3: before/after first-hit summary (Q2)
# ============================================================================

def plot_before_after_summary(df, output_dir=OUTPUT_DIR, filename='before_after_first_hit'):
    """Per-mouse mean event rate before vs. after the first whisker hit,
    paired, R+ vs R- -- the coarse two-bin summary of the aligned
    trajectory above. Wilcoxon signed-rank per group (n=mice), matching
    the nonparametric convention used elsewhere in this project.
    """
    d = df.dropna(subset=['relative_position']).copy()
    d['period'] = np.where(d['relative_position'] < 0, 'before', 'after')

    per_mouse = (
        d.groupby(['reward_group', 'mouse_id', 'period'])['n_events']
        .mean()
        .reset_index()
    )
    pivoted = per_mouse.pivot_table(
        index=['reward_group', 'mouse_id'], columns='period', values='n_events'
    ).reset_index()
    pivoted = pivoted.dropna(subset=['before', 'after'])

    stats_rows = []
    for rg, grp in pivoted.groupby('reward_group'):
        if len(grp) < 2:
            print(f"  Skipping Wilcoxon for {rg}: only {len(grp)} mice with both periods")
            continue
        try:
            w_stat, p_value = wilcoxon(grp['before'], grp['after'])
        except ValueError:
            w_stat, p_value = np.nan, np.nan
        stats_rows.append({
            'reward_group': rg,
            'test': 'Wilcoxon signed-rank (before vs after)',
            'median_before': grp['before'].median(),
            'median_after': grp['after'].median(),
            'w_stat': w_stat, 'p_value': p_value, 'n_mice': len(grp),
        })
        print(f"  {rg}: before={grp['before'].median():.3f}, after={grp['after'].median():.3f}, "
              f"W={w_stat:.3g}, p={p_value:.4g}, n={len(grp)}")

    sns.set_theme(context='paper', style='ticks', palette='deep', font='sans-serif', font_scale=1)
    fig, axes = plt.subplots(1, 2, figsize=(7, 5), sharey=True)
    for ax, rg in zip(axes, ['R+', 'R-']):
        gd = per_mouse[per_mouse['reward_group'] == rg]
        color = reward_palette[1] if rg == 'R+' else reward_palette[0]
        sns.pointplot(data=gd, x='period', y='n_events', order=['before', 'after'],
                      color=color, errorbar='ci', ax=ax)
        for mouse_id in gd['mouse_id'].unique():
            md = gd[gd['mouse_id'] == mouse_id].set_index('period').reindex(['before', 'after'])
            if md['n_events'].isna().any():
                continue
            ax.plot([0, 1], md['n_events'].values, color=color, alpha=0.3, linewidth=0.8)
        ax.set_title(rg)
        ax.set_xlabel('')
        ax.set_ylabel('Mean reactivation events per trial' if ax is axes[0] else '')
        sns.despine(ax=ax)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f'{filename}.svg'), format='svg', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {os.path.join(output_dir, filename + '.svg')}")

    pivoted.to_csv(os.path.join(output_dir, f'{filename}_data.csv'), index=False)
    pd.DataFrame(stats_rows).to_csv(os.path.join(output_dir, f'{filename}_stats.csv'), index=False)


# ============================================================================
# Main execution
# ============================================================================

if __name__ == '__main__':
    print(f"Output directory: {OUTPUT_DIR}")

    df, missing_hit = build_dataset(day=DAY)
    print(f"\nDataset: {len(df)} trial-rows, {df['mouse_id'].nunique()} mice")

    print("\nPlotting per-trial event distribution (Q1)...")
    plot_per_trial_distribution(df)

    print("\nPlotting first-hit-aligned trajectory (Q3)...")
    plot_first_hit_aligned_trajectory(df)

    print("\nPlotting before/after first-hit summary (Q2)...")
    plot_before_after_summary(df)

    print("\nDone.")
