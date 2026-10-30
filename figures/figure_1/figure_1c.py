"""
Figure 1c: Behavioral performance of the imaging mice across days

One figure, three plots:
- left: lick probability on auditory, whisker and catch trials across the
  five days, R+ and R- (thin lines: mice; points: mean across mice);
- middle: whisker lick probability on Days 0, +1, +2, R+ vs R-;
- right: d' (whisker vs catch) on the same days, R+ vs R-.
Bars: mean across mice; dots: mice. Mann-Whitney U test per day (n = mice).

d' is computed per session from trial counts with the log-linear correction
(fast_learning.stats.dprime), whisker trials as signal and catch trials as
noise, then averaged per mouse and day.

Outputs: <figures_dir>/Figure1/figure_1c.pdf, figure_1c_left_data.csv,
figure_1c_right_data.csv, figure_1c_right_stats.csv,
figure_1c_dprime_data.csv, figure_1c_dprime_stats.csv.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import mannwhitneyu

from fast_learning import paths
from fast_learning.plotting import behavior_palette, panel_size, save_figure, set_style
from fast_learning.stats import dprime, format_p

OUTPUT_DIR = os.path.join(paths.manuscript_output_dir, 'Figure1')
TABLE_PATH = os.path.join(paths.processed_dir, 'behavior', 'behavior_imagingmice_table_5days_cut.csv')
DAYS_OF_INTEREST = [0, 1, 2]


# ============================================================================
# Data
# ============================================================================


def load_table(table_path=TABLE_PATH):
    table = pd.read_csv(paths.adjust_path_to_host(table_path))
    # Remove spurious whisker trials from the mapping sessions (days -2 and -1).
    table.loc[table.day.isin([-2, -1]), ['outcome_w', 'hr_w']] = np.nan
    return table


def session_performance(table):
    """Lick probability (%) per session and trial type."""
    agg = table.groupby(['mouse_id', 'session_id', 'reward_group', 'day'], as_index=False)[
        ['outcome_c', 'outcome_a', 'outcome_w']
    ].mean()
    agg[['outcome_c', 'outcome_a', 'outcome_w']] *= 100
    agg['day'] = agg['day'].astype(str)
    return agg


def session_dprime(table, days=DAYS_OF_INTEREST):
    """d' per mouse and day (sessions averaged), whisker vs catch trials."""
    t = table[table.day.isin(days)]
    c = t.groupby(['mouse_id', 'session_id', 'reward_group', 'day']).agg(
        hits=('outcome_w', 'sum'),
        n_whisker=('outcome_w', 'count'),
        false_alarms=('outcome_c', 'sum'),
        n_catch=('outcome_c', 'count'),
    )
    c['dprime'] = dprime(c['hits'], c['n_whisker'], c['false_alarms'], c['n_catch'])
    c = c.reset_index().groupby(['day', 'mouse_id', 'reward_group'], as_index=False)['dprime'].mean()
    c['day'] = c['day'].astype(str)
    return c


def group_stats(data, y, days=DAYS_OF_INTEREST):
    """Mann-Whitney U, R+ vs R-, on each day."""
    rows = []
    for day in days:
        d = data[data['day'] == str(day)]
        rp, rm = d.loc[d.reward_group == 'R+', y], d.loc[d.reward_group == 'R-', y]
        stat, p = mannwhitneyu(rp, rm, alternative='two-sided')
        rows.append({'day': day, 'n_R+': len(rp), 'n_R-': len(rm), 'statistic': stat, 'p_value': p})
    return pd.DataFrame(rows)


# ============================================================================
# Plots
# ============================================================================


def plot_across_days(ax, agg):
    """Lick probability per trial type across days: mice and group means."""
    for col, palette in [
        ('outcome_c', behavior_palette[4:6]),
        ('outcome_a', behavior_palette[0:2]),
        ('outcome_w', behavior_palette[2:4]),
    ]:
        kw = dict(
            data=agg, x='day', y=col, hue='reward_group', hue_order=['R-', 'R+'], palette=palette, ax=ax
        )
        sns.lineplot(**kw, units='mouse_id', estimator=None, alpha=0.4, legend=False, linewidth=1, seed=0)
        sns.pointplot(**kw, estimator=np.mean, legend=False, linewidth=1, seed=0)
    ax.set_xlabel('Training days')
    ax.set_ylabel('Lick probability (%)')


def plot_day_bars(ax, data, y, stats, ylabel, ylim, legend=False):
    """Bars (mean across mice) and dots (mice) per day, R+ vs R-, with the
    p-value of each day above it."""
    kw = dict(data=data, x='day', y=y, hue='reward_group', hue_order=['R+', 'R-'], ax=ax)
    sns.barplot(**kw, palette=behavior_palette[2:4][::-1], width=0.8, dodge=True, seed=0, legend=legend)
    sns.swarmplot(**kw, palette=['grey', 'grey'], dodge=True, alpha=0.6, size=2, legend=False)
    for xi, p in enumerate(stats['p_value']):
        ax.text(
            xi, 1.0, format_p(p), transform=ax.get_xaxis_transform(), ha='center', va='bottom', fontsize=5
        )
    if ylim[0] < 0:
        ax.axhline(0, color='grey', linestyle='--', linewidth=0.6)
    ax.set_ylim(*ylim)
    ax.set_xlabel('Day')
    ax.set_ylabel(ylabel)
    if legend:
        ax.legend(frameon=False, title='', loc='upper left', bbox_to_anchor=(1.0, 1.0))


def generate_figure(table=None, save_path=OUTPUT_DIR):
    table = load_table() if table is None else table
    agg = session_performance(table)
    days = [str(d) for d in DAYS_OF_INTEREST]
    hits = (
        agg[agg['day'].isin(days)]
        .groupby(['day', 'mouse_id', 'reward_group'], as_index=False)['outcome_w']
        .mean()
    )
    hit_stats = group_stats(hits, 'outcome_w')
    dp = session_dprime(table)
    dp_stats = group_stats(dp, 'dprime')

    set_style()
    fig, axes = plt.subplots(1, 3, figsize=panel_size(3, w=1.3))
    plot_across_days(axes[0], agg)
    plot_day_bars(axes[1], hits, 'outcome_w', hit_stats, 'Lick probability (%)', (0, 100))
    plot_day_bars(axes[2], dp, 'dprime', dp_stats, "d' (whisker vs catch)", (-1, 4), legend=True)
    for ax in axes:
        sns.despine(ax=ax, trim=True)
    plt.tight_layout()

    os.makedirs(save_path, exist_ok=True)
    save_figure(fig, os.path.join(save_path, 'figure_1c.pdf'))
    plt.close(fig)
    agg.to_csv(os.path.join(save_path, 'figure_1c_left_data.csv'), index=False)
    hits.to_csv(os.path.join(save_path, 'figure_1c_right_data.csv'), index=False)
    hit_stats.to_csv(os.path.join(save_path, 'figure_1c_right_stats.csv'), index=False)
    dp.to_csv(os.path.join(save_path, 'figure_1c_dprime_data.csv'), index=False)
    dp_stats.to_csv(os.path.join(save_path, 'figure_1c_dprime_stats.csv'), index=False)
    print(hit_stats.round(4).to_string(index=False))
    print(dp_stats.round(6).to_string(index=False))


if __name__ == '__main__':
    generate_figure()
