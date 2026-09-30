"""
Reviewer check: session-level behavioral state summary (total water reward,
session duration, total trial count) per mouse x session, across days and
reward groups.

Addresses the concern that behavioral engagement at the end of sessions --
when passive/mapping trials are presented -- might differ between R+/R- mice
and across training days.

Reward logic (5 uL per rewarded trial):
    R+ : auditory hit (auditory_stim==1 & lick_flag==1)
         + whisker hit (whisker_stim==1 & lick_flag==1)
    R- : auditory hit only (auditory_stim==1 & lick_flag==1)

Trial table is recomputed fresh via make_behavior_table() (not loaded from
a precomputed CSV) for the imaging cohort, same selection pattern as
exploratory/behavior/behavior.py.
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu

from fast_learning import paths, database
from fast_learning.behavior import make_behavior_table
from fast_learning.plotting import reward_palette, save_figure
from fast_learning.stats import significance_stars as _significance_stars


# ============================================================================
# Parameters
# ============================================================================

REWARD_UL_PER_TRIAL = 5
DAYS = [-2, -1, 0, 1, 2]
OUTPUT_DIR = os.path.join(paths.results_dir, 'behavior', 'session_state_check')


# ============================================================================
# Data
# ============================================================================


def load_behavior_table():
    """Fresh trial table for the imaging cohort (same selection pattern as
    behavior.py's mice_imaging block)."""
    db_path = paths.db_path
    nwb_dir = paths.nwb_dir

    mice_imaging = database.select_mice_from_db(
        db_path,
        nwb_dir,
        experimenters=None,
        exclude_cols=['exclude', 'two_p_exclude'],
        optogenetic=['no', np.nan],
        pharmacology=['no', np.nan],
        two_p_imaging='yes',
    )
    session_list, nwb_list, mice_list, db = database.select_sessions_from_db(
        db_path,
        nwb_dir,
        experimenters=None,
        exclude_cols=['exclude', 'two_p_exclude'],
        day=["-2", "-1", '0', '+1', '+2'],
        mouse_id=mice_imaging,
    )
    table = make_behavior_table(
        nwb_list,
        session_list,
        db_path,
        cut_session=True,
        stop_flag_yaml=paths.stop_flags_yaml,
        trial_indices_yaml=paths.trial_indices_yaml,
    )
    return table


def compute_session_summary(table):
    """One row per (mouse_id, session_id, day, reward_group):
    n_trials, session_duration_min, total_reward_uL."""
    rows = []
    for (mouse, session, day, rg), g in table.groupby(['mouse_id', 'session_id', 'day', 'reward_group']):
        n_trials = len(g)
        duration_min = (g['start_time'].max() - g['start_time'].min()) / 60

        aud_hits = int(((g['auditory_stim'] == 1) & (g['lick_flag'] == 1)).sum())
        wh_hits = 0
        if rg == 'R+':
            wh_hits = int(((g['whisker_stim'] == 1) & (g['lick_flag'] == 1)).sum())
        reward_uL = REWARD_UL_PER_TRIAL * (aud_hits + wh_hits)

        rows.append(
            {
                'mouse_id': mouse,
                'session_id': session,
                'day': day,
                'reward_group': rg,
                'n_trials': n_trials,
                'session_duration_min': duration_min,
                'total_reward_uL': reward_uL,
            }
        )

    return pd.DataFrame(rows)


# ============================================================================
# Plot
# ============================================================================

METRICS = [
    ('n_trials', 'Number of trials'),
    ('session_duration_min', 'Session duration (min)'),
    ('total_reward_uL', 'Total water reward (uL)'),
]


def plot_session_summary(
    df, days=DAYS, output_dir=OUTPUT_DIR, filename='session_state_summary', save_format='svg', dpi=300
):
    """Bar plot of each metric across days, R+ vs R-, with Mann-Whitney U
    per day (mirrors figure_4h.py's panel_h_reactivation_rate pattern).

    Saves:
        <filename>.svg        -- figure (3 panels)
        <filename>_data.csv   -- per-session summary table
        <filename>_stats.csv  -- Mann-Whitney U results per day per metric
    """
    sns.set_theme(context='paper', style='ticks', palette='deep', font='sans-serif', font_scale=1)

    days_sorted = sorted(days)
    fig, axes = plt.subplots(1, len(METRICS), figsize=(5 * len(METRICS), 4))
    all_stats_rows = []

    for ax, (metric, ylabel) in zip(axes, METRICS):
        stats_rows = []
        p_values = []
        for day in days_sorted:
            r_plus_vals = df[(df['day'] == day) & (df['reward_group'] == 'R+')][metric].dropna().values
            r_minus_vals = df[(df['day'] == day) & (df['reward_group'] == 'R-')][metric].dropna().values
            if len(r_plus_vals) > 0 and len(r_minus_vals) > 0:
                stat, p = mannwhitneyu(r_plus_vals, r_minus_vals, alternative='two-sided')
            else:
                stat, p = np.nan, 1.0
            p_values.append(p)
            stats_rows.append(
                {
                    'metric': metric,
                    'test': 'Mann-Whitney U',
                    'day': day,
                    'R+_n': len(r_plus_vals),
                    'R-_n': len(r_minus_vals),
                    'R+_mean': np.nanmean(r_plus_vals) if len(r_plus_vals) else np.nan,
                    'R-_mean': np.nanmean(r_minus_vals) if len(r_minus_vals) else np.nan,
                    'statistic': stat,
                    'p_value': p,
                    'significance': _significance_stars(p),
                }
            )
        all_stats_rows.extend(stats_rows)

        sns.barplot(
            data=df,
            x='day',
            y=metric,
            hue='reward_group',
            order=days_sorted,
            errorbar=('ci', 95),
            palette={'R+': reward_palette[1], 'R-': reward_palette[0]},
            hue_order=['R+', 'R-'],
            alpha=0.7,
            edgecolor='black',
            ax=ax,
        )

        y_max = df[metric].max()
        y_range = y_max * 0.05
        width = 0.35
        for day_idx, (day, p) in enumerate(zip(days_sorted, p_values)):
            stars = _significance_stars(p)
            if stars != 'n.s.':
                r_plus_vals = df[(df['day'] == day) & (df['reward_group'] == 'R+')][metric].dropna()
                r_minus_vals = df[(df['day'] == day) & (df['reward_group'] == 'R-')][metric].dropna()
                ci_plus = (
                    r_plus_vals.mean() + 1.96 * r_plus_vals.std() / np.sqrt(len(r_plus_vals))
                    if len(r_plus_vals) > 0
                    else 0
                )
                ci_minus = (
                    r_minus_vals.mean() + 1.96 * r_minus_vals.std() / np.sqrt(len(r_minus_vals))
                    if len(r_minus_vals) > 0
                    else 0
                )
                y1 = max(ci_plus, ci_minus)
                y2 = y1 + y_range
                x1, x2 = day_idx - width / 2, day_idx + width / 2
                ax.plot([x1, x1, x2, x2], [y1, y2, y2, y1], 'k-', linewidth=1)
                ax.text((x1 + x2) / 2, y2, stars, ha='center', va='bottom', fontsize=10)

        ax.set_xlabel('Day', fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.legend(title='', fontsize=9)

    sns.despine()
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    save_figure(plt.gcf(), os.path.join(output_dir, f'{filename}.{save_format}'))
    plt.close()
    print(f"Figure saved to: {os.path.join(output_dir, filename + '.' + save_format)}")

    df.to_csv(os.path.join(output_dir, f'{filename}_data.csv'), index=False)
    pd.DataFrame(all_stats_rows).to_csv(os.path.join(output_dir, f'{filename}_stats.csv'), index=False)
    print(f"Data/stats saved to: {output_dir}")


# ============================================================================
# Main execution
# ============================================================================

if __name__ == '__main__':
    print("Loading behavior table for imaging cohort...")
    table = load_behavior_table()
    print(
        f"Loaded {len(table)} trials across "
        f"{table['session_id'].nunique()} sessions, "
        f"{table['mouse_id'].nunique()} mice"
    )

    summary_df = compute_session_summary(table)
    print(f"\nSummary: {len(summary_df)} sessions")

    plot_session_summary(summary_df)

    print("\n=== Mean per day / reward group ===")
    print(
        summary_df.groupby(['day', 'reward_group'])[['n_trials', 'session_duration_min', 'total_reward_uL']]
        .mean()
        .to_string()
    )
