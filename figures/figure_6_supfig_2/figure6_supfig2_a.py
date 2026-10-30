"""
Figure 6—figure supplement 2a: Reactivation rate in the lick-free and the
lick-possible parts of the catch trials

The reactivation analysis uses correct-rejection no-stim trials over the
whole trial (-1 to 6 s around no-stim onset). These
trials have no lick before onset (no-lick period) and none in the 0-1 s
response window, but a mouse can lick after 1 s. If lick-related activity
produced template matches, the reactivation rate would be higher after 1 s
than in the lick-free part of the trial.

Events are detected within each trial, so a peak cannot be confirmed in the
first and last frames of a trial: the event rate falls off within ~0.4 s of
the trial edges (-1 s and 6 s). EDGE_S is left out of both windows so that
they are compared on equal terms.

For each mouse and day, event times are taken from the reactivation results
of step 07, and rates (events/min) are computed in 0.5 s bins across the
trial (edges excluded) and in two windows: lick-free (-0.6 to 1 s) and
post-response (1 to 5.6 s). Days are pooled per mouse (each mouse's rate
averaged over its days).

  a, left: event rate across the trial, mean across mice with bootstrapped
         95% CI.
  a, right: rate in the lick-free vs post-response window, mean across mice
         with bootstrapped 95% CI and individual mice (lines); paired
         Wilcoxon signed-rank test per reward group (n = mice).
Stats file: also R+ vs R- in the lick-free window alone, per day
(Mann-Whitney U, as Fig. 6c).

Outputs: <figures_dir>/Figure6_SupFig2/figure6_supfig2_a.pdf, _data.csv
(mouse x day x window), _bins.csv (mouse x time bin), _stats.csv.
"""

import os
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu, wilcoxon

from fast_learning import paths, reactivations as rx
from fast_learning.plotting import panel_size, reward_palette, save_figure, set_style
from fast_learning.stats import format_p, significance_stars
from fast_learning.tensors import time_axis

EDGE_S = 0.4  # s at each trial edge where per-trial peak detection is depleted
LICK_FREE = (-1 + EDGE_S, 1)  # s: no-lick period before onset and response window
POST_RESPONSE = (1, 6 - EDGE_S)  # s: licks possible in correct rejections
BIN_S = 0.5
PROFILE = (-0.5, 5.5)  # s: range of the binned time course, away from the edges
GROUP_COLORS = {'R+': reward_palette[1], 'R-': reward_palette[0]}
OUTPUT_DIR = os.path.join(paths.manuscript_output_dir, 'Figure6_SupFig2')


def event_rates(results, t):
    """Per mouse x day: event rates per time bin and per window (events/min)."""
    edges = np.arange(PROFILE[0], PROFILE[1] + 1e-9, BIN_S)
    frame_s = t[1] - t[0]
    window_rows, bin_rows = [], []
    for rg, key in [('R+', 'r_plus_results'), ('R-', 'r_minus_results')]:
        for mouse, res in results[key].items():
            for day, d in res['days'].items():
                n_t, n_trials = d['n_timepoints'], d['n_trials']
                if n_t != len(t):
                    raise ValueError(f'{mouse} day {day}: {n_t} frames per trial, expected {len(t)}.')
                ev_t = t[np.asarray(d['events'], dtype=int) % n_t]
                for name, (a, b) in [('lick_free', LICK_FREE), ('post_response', POST_RESPONSE)]:
                    in_win = (t >= a) & (t < b)
                    minutes = n_trials * in_win.sum() * frame_s / 60
                    n_ev = int(((ev_t >= a) & (ev_t < b)).sum())
                    window_rows.append(
                        dict(
                            mouse_id=mouse,
                            reward_group=rg,
                            day=day,
                            window=name,
                            n_events=n_ev,
                            rate=n_ev / minutes,
                        )
                    )
                counts, _ = np.histogram(ev_t, bins=edges)
                frames_per_bin, _ = np.histogram(t[(t >= edges[0]) & (t < edges[-1])], bins=edges)
                for i, c in enumerate(counts):
                    minutes = n_trials * frames_per_bin[i] * frame_s / 60
                    bin_rows.append(
                        dict(
                            mouse_id=mouse,
                            reward_group=rg,
                            day=day,
                            time=edges[i] + BIN_S / 2,
                            rate=c / minutes,
                        )
                    )
    return pd.DataFrame(window_rows), pd.DataFrame(bin_rows)


def compute_stats(win_df):
    per_mouse = win_df.groupby(['reward_group', 'mouse_id', 'window'])['rate'].mean().unstack('window')
    rows = []
    for rg in ['R+', 'R-']:
        g = per_mouse.loc[rg]
        stat, p = wilcoxon(g['lick_free'], g['post_response'])
        rows.append(
            dict(
                reward_group=rg,
                test=(
                    f'Wilcoxon signed-rank, lick-free ({LICK_FREE[0]:g} to {LICK_FREE[1]:g} s) vs '
                    f'post-response ({POST_RESPONSE[0]:g} to {POST_RESPONSE[1]:g} s), days pooled'
                ),
                n_mice=len(g),
                median_lick_free=g['lick_free'].median(),
                median_post_response=g['post_response'].median(),
                median_ratio_post_over_free=(g['post_response'] / g['lick_free']).median(),
                statistic=stat,
                p_value=p,
                significance=significance_stars(p),
            )
        )
    # R+ vs R- with the lick-free window alone, per day (as Fig. 6c).
    free = win_df[win_df['window'] == 'lick_free']
    for day in sorted(free['day'].unique()):
        a = free.query('day == @day and reward_group == "R+"')['rate']
        b = free.query('day == @day and reward_group == "R-"')['rate']
        stat, p = mannwhitneyu(a, b)
        rows.append(
            dict(
                reward_group='R+ vs R-',
                test=f'Mann-Whitney U, lick-free window only, day {day:+d}',
                n_mice=f'{len(a)} / {len(b)}',
                median_lick_free=f'{a.median():.2f} / {b.median():.2f}',
                statistic=stat,
                p_value=p,
                significance=significance_stars(p),
            )
        )
    return pd.DataFrame(rows), per_mouse


def plot(bin_df, per_mouse, stats, filename):
    set_style()
    fig, axes = plt.subplots(1, 2, figsize=panel_size(2, w=1.1))

    ax = axes[0]
    ax.axvspan(*LICK_FREE, color='0.92', zorder=0)
    mouse_bins = bin_df.groupby(['reward_group', 'mouse_id', 'time'])['rate'].mean().reset_index()
    sns.lineplot(
        data=mouse_bins,
        x='time',
        y='rate',
        hue='reward_group',
        hue_order=['R+', 'R-'],
        palette=GROUP_COLORS,
        errorbar=('ci', 95),
        seed=0,
        ax=ax,
    )
    ax.axvline(0, color='grey', linestyle='--', linewidth=0.6)
    ax.set_ylim(bottom=0, top=15)
    ax.set_xlabel('Time from no-stim onset (s)')
    ax.set_ylabel('Reactivation rate (events/min)')
    ax.legend(frameon=False, title='')
    ax.text(0, 1.0, 'lick-free', transform=ax.get_xaxis_transform(), ha='center', va='bottom')

    ax = axes[1]
    for gi, rg in enumerate(['R+', 'R-']):
        g = per_mouse.loc[rg]
        x = np.array([gi * 3, gi * 3 + 1])
        for _, row in g.iterrows():
            ax.plot(
                x,
                [row['lick_free'], row['post_response']],
                '-',
                color='grey',
                linewidth=0.5,
                alpha=0.6,
                zorder=3,
            )
        # Bars: mean across mice with bootstrapped 95% CI (seeded); lines: mice.
        long = g[['lick_free', 'post_response']].melt(var_name='window', value_name='rate')
        sns.barplot(
            data=long.assign(x=np.where(long['window'] == 'lick_free', x[0], x[1])),
            x='x',
            y='rate',
            order=list(range(5)),
            native_scale=True,
            color=GROUP_COLORS[rg],
            alpha=0.7,
            edgecolor='black',
            errorbar=('ci', 95),
            seed=0,
            width=0.7,
            ax=ax,
            zorder=1,
        )
        p = stats.query('reward_group == @rg')['p_value'].iloc[0]  # the paired test
        top = np.nanmax(g[['lick_free', 'post_response']].values)
        ax.plot(
            [x[0], x[0], x[1], x[1]], [top * 1.04, top * 1.08, top * 1.08, top * 1.04], 'k-', linewidth=0.8
        )
        ax.text(x.mean(), top * 1.09, format_p(p), ha='center', va='bottom')
        ax.text(x.mean(), -0.3, rg, transform=ax.get_xaxis_transform(), ha='center', va='top')
    ax.set_xticks([0, 1, 3, 4], ['Lick-\nfree', 'Post', 'Lick-\nfree', 'Post'])
    ax.set_xlabel('')
    ax.set_xlim(-1, 5)
    ax.set_ylim(bottom=0, top=15)
    ax.set_ylabel('Reactivation rate (events/min)')
    sns.despine()
    plt.tight_layout()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_figure(fig, os.path.join(OUTPUT_DIR, f'{filename}.pdf'))
    plt.close(fig)


if __name__ == '__main__':
    results_file = os.path.join(rx.RESULTS_DIR, 'reactivation_results_p99.pkl')
    with open(results_file, 'rb') as f:
        results = pickle.load(f)

    t = time_axis((1, 6))  # learning tensors: -1 to 6 s, stimulus frame at 0
    win_df, bin_df = event_rates(results, t)
    stats, per_mouse = compute_stats(win_df)

    name = 'figure6_supfig2_a'
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    win_df.to_csv(os.path.join(OUTPUT_DIR, f'{name}_data.csv'), index=False)
    bin_df.to_csv(os.path.join(OUTPUT_DIR, f'{name}_bins.csv'), index=False)
    stats.to_csv(os.path.join(OUTPUT_DIR, f'{name}_stats.csv'), index=False)
    plot(bin_df, per_mouse, stats, name)
    print(
        stats[
            [
                'reward_group',
                'n_mice',
                'median_lick_free',
                'median_post_response',
                'median_ratio_post_over_free',
                'p_value',
            ]
        ].to_string(index=False)
    )
    print(f'Saved to {OUTPUT_DIR}')
