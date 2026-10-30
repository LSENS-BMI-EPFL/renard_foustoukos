"""
Figure 3—figure supplement 2a-f: Behavioural state of the imaging mice
across days

- a-c: number of trials, task duration and water consumed per session, R+ vs
  R- on each day (Mann-Whitney U, n = mice; p-values above each day). Stats
  file also: pre (Days -2/-1) vs post (Days +1/+2) within each group
  (Wilcoxon signed-rank).
- d-f: whisker angular speed, tongue speed and jaw speed in the 1 s after the
  passive whisker stimulus, last 40 passive trials per day, R+ mice with
  video (n = 11); Friedman test across days. Bars: mean and bootstrapped 95%
  CI across mice; lines: mice.
See fast_learning.state.

The session measures are read from the NWB files (several minutes) and cached
in the data file; --recompute rebuilds them.

Outputs: <figures_dir>/Figure3_SupFig2/figure3_supfig2_a_f.pdf,
figure3_supfig2_a_c_data.csv (session), figure3_supfig2_a_c_stats.csv,
figure3_supfig2_d_f_data.csv (mouse x day x speed), figure3_supfig2_d_f_stats.csv.
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from fast_learning import paths
from fast_learning import state
from fast_learning.plotting import panel_size, reward_palette, save_figure, set_style
from fast_learning.stats import format_p

OUTPUT_DIR = os.path.join(paths.manuscript_output_dir, 'Figure3_SupFig2')
SESSION_DATA = os.path.join(OUTPUT_DIR, 'figure3_supfig2_a_c_data.csv')
GROUP_COLORS = {'R+': reward_palette[1], 'R-': reward_palette[0]}
SESSION_YLIMS = {'n_trials': (0, 600), 'session_duration_min': (0, 120), 'total_reward_uL': (0, 1000)}
SPEED_YLIMS = {'whisker_angular_speed': (0, 1500), 'tongue_speed': (0, 10), 'jaw_speed': (0, 800)}
DAY_TICKS = [f'{d:+d}'.replace('+0', '0') for d in state.DAYS]


def session_panel(ax, summary, stats, measure, legend=False):
    sns.barplot(
        data=summary,
        x='day',
        y=measure,
        order=state.DAYS,
        hue='reward_group',
        hue_order=list(GROUP_COLORS),
        palette=GROUP_COLORS,
        errorbar=('ci', 95),
        seed=0,
        alpha=0.7,
        edgecolor='black',
        width=0.7,
        legend=legend,
        ax=ax,
    )
    for xi, day in enumerate(state.DAYS):
        row = (stats['measure'] == measure) & (stats['comparison'] == f'R+ vs R-, day {day}')
        p = stats.loc[row, 'p_value'].iloc[0]
        ax.text(
            xi,
            1.0 + 0.08 * (xi % 2),  # staggered so that neighbours do not overlap
            format_p(p),
            transform=ax.get_xaxis_transform(),
            ha='center',
            va='bottom',
            fontsize=5,
        )
    ax.set_xticks(range(len(state.DAYS)), DAY_TICKS)
    ax.set_ylim(*SESSION_YLIMS[measure])
    ax.set_xlabel('Day')
    ax.set_ylabel(state.SESSION_MEASURES[measure])
    if legend:
        ax.legend(frameon=False, title='', loc='upper left', bbox_to_anchor=(1.0, 1.0))


def speed_panel(ax, means, stats, speed, unit):
    d = means[means['speed'] == speed]
    for _, m in d.groupby('mouse_id'):
        m = m.set_index('day').reindex(state.DAYS)
        ax.plot(range(len(state.DAYS)), m['value'], color='grey', linewidth=0.5, alpha=0.6)
    sns.barplot(
        data=d,
        x='day',
        y='value',
        order=state.DAYS,
        color=GROUP_COLORS['R+'],
        alpha=0.7,
        errorbar=('ci', 95),
        seed=0,
        ax=ax,
    )
    p = stats.loc[stats['speed'] == speed, 'p_value'].iloc[0]
    ax.set_title(f'Friedman {format_p(p)}')
    ax.set_xticks(range(len(state.DAYS)), DAY_TICKS)
    ax.set_ylim(*SPEED_YLIMS[speed])
    ax.set_xlabel('Day')
    ax.set_ylabel(f'{state.SPEEDS[speed]}\n({unit}, {state.WINDOW[0]:g}-{state.WINDOW[1]:g} s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    parser.add_argument(
        '--recompute', action='store_true', help='rebuild the session measures from the NWB files'
    )
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if args.recompute or not os.path.exists(SESSION_DATA):
        summary = state.session_summary(state.load_imaging_trials())
        summary.to_csv(SESSION_DATA, index=False)
    else:
        summary = pd.read_csv(SESSION_DATA)
    s_stats = state.session_stats(summary)
    means, units = state.movement_means()
    m_stats = state.movement_stats(means)

    set_style()
    fig, axes = plt.subplots(2, 3, figsize=panel_size(3, 2, w=1.2))
    for ax, measure in zip(axes[0], state.SESSION_MEASURES):
        session_panel(ax, summary, s_stats, measure, legend=ax is axes[0, -1])
    for ax, speed in zip(axes[1], state.SPEEDS):
        speed_panel(ax, means, m_stats, speed, units[speed])
    sns.despine()
    plt.tight_layout()
    save_figure(fig, os.path.join(OUTPUT_DIR, 'figure3_supfig2_a_f.pdf'))
    plt.close(fig)

    s_stats.to_csv(os.path.join(OUTPUT_DIR, 'figure3_supfig2_a_c_stats.csv'), index=False)
    means.to_csv(os.path.join(OUTPUT_DIR, 'figure3_supfig2_d_f_data.csv'), index=False)
    m_stats.to_csv(os.path.join(OUTPUT_DIR, 'figure3_supfig2_d_f_stats.csv'), index=False)
    print(s_stats[['measure', 'comparison', 'mean_1', 'mean_2', 'p_value']].round(4).to_string(index=False))
    print(m_stats.drop(columns=['test']).round(3).to_string(index=False))
    print(np.round(means.groupby(['speed', 'day'])['value'].mean().unstack(), 1))
