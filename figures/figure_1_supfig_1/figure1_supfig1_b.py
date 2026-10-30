"""
Figure 1—figure supplement 1b: Licking to the whisker stimulus at the start of
Day 0 (R2 public 1, R1 recommendation 1, R3 public 1).

Do mice lick to the whisker stimulus before they learn its contingency, as
expected if the auditory lick policy generalizes to the new stimulus? For
each imaging mouse on Day 0, lick probability on the first N_WHISKER whisker
trials and on the catch trials of the same period (catch trials before the
N_WHISKER-th whisker trial).

Stats (n = mice): whisker vs catch within each group (Wilcoxon signed-rank);
R+ vs R- whisker lick probability (Mann-Whitney U). Bars: mean across mice
with bootstrapped 95% CI; lines: mice.

Input: behavior_imagingmice_table_5days_cut.csv (pipeline step 03).
Outputs: <figures_dir>/Figure1_SupFig1/figure1_supfig1_b.pdf,
figure1_supfig1_b_data.csv (mouse), figure1_supfig1_b_stats.csv.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import mannwhitneyu, wilcoxon

from fast_learning import paths
from fast_learning.plotting import panel_size, reward_palette, save_figure, set_style
from fast_learning.stats import format_p

N_WHISKER = 10
GROUP_COLORS = {'R+': reward_palette[1], 'R-': reward_palette[0]}
OUTPUT_DIR = os.path.join(paths.manuscript_output_dir, 'Figure1_SupFig1')


def initial_licking(table):
    """Per mouse: lick probability on the first N_WHISKER whisker trials of
    Day 0 and on the catch trials before the N_WHISKER-th whisker trial."""
    rows = []
    day0 = table[table['day'] == 0]
    for (mouse, rg), t in day0.groupby(['mouse_id', 'reward_group']):
        t = t.sort_values('trial_id')
        whisker = t[t['whisker_stim'] == 1].head(N_WHISKER)
        end = whisker['trial_id'].iloc[-1]
        catch = t[(t['no_stim'] == 1) & (t['trial_id'] < end)]
        rows.append(
            dict(
                mouse_id=mouse,
                reward_group=rg,
                whisker=whisker['lick_flag'].mean(),
                catch=catch['lick_flag'].mean(),
                n_whisker=len(whisker),
                n_catch=len(catch),
                first_whisker_trial=whisker['lick_flag'].iloc[0],
            )
        )
    return pd.DataFrame(rows)


def compute_stats(mice):
    rows = []
    for rg in GROUP_COLORS:
        d = mice[mice['reward_group'] == rg]
        res = wilcoxon(d['whisker'], d['catch'])
        rows.append(
            dict(
                comparison='whisker vs catch',
                reward_group=rg,
                test='Wilcoxon signed-rank (n = mice)',
                n=len(d),
                mean_1=d['whisker'].mean(),
                mean_2=d['catch'].mean(),
                statistic=res.statistic,
                p_value=res.pvalue,
            )
        )
    a, b = (mice.loc[mice['reward_group'] == g, 'whisker'] for g in GROUP_COLORS)
    res = mannwhitneyu(a, b, alternative='two-sided')
    rows.append(
        dict(
            comparison='whisker, R+ vs R-',
            reward_group='R+ vs R-',
            test='Mann-Whitney U (n = mice)',
            n=f'{len(a)} / {len(b)}',
            mean_1=a.mean(),
            mean_2=b.mean(),
            statistic=res.statistic,
            p_value=res.pvalue,
        )
    )
    return pd.DataFrame(rows)


def plot(mice, stats, filename='figure1_supfig1_b'):
    set_style()
    fig, ax = plt.subplots(figsize=panel_size(1, w=1.5))
    for gi, rg in enumerate(GROUP_COLORS):
        d = mice[mice['reward_group'] == rg]
        x = np.array([gi * 3, gi * 3 + 1])
        for _, row in d.iterrows():
            ax.plot(
                x, 100 * row[['whisker', 'catch']].astype(float), '-', color='grey', linewidth=0.5, alpha=0.6
            )
        long = d.melt(id_vars='mouse_id', value_vars=['whisker', 'catch'], var_name='trial', value_name='p')
        sns.barplot(
            data=long.assign(x=np.where(long['trial'] == 'whisker', x[0], x[1]), p=100 * long['p']),
            x='x',
            y='p',
            order=list(range(5)),
            native_scale=True,
            color=GROUP_COLORS[rg],
            alpha=0.7,
            edgecolor='black',
            errorbar=('ci', 95),
            seed=0,
            width=0.7,
            ax=ax,
        )
        p = stats.query('reward_group == @rg')['p_value'].iloc[0]
        ax.text(
            x.mean(),
            1.0,
            format_p(p),
            transform=ax.get_xaxis_transform(),
            ha='center',
            va='bottom',
            fontsize=5,
        )
        ax.text(
            x.mean(),
            1.08,
            rg,
            transform=ax.get_xaxis_transform(),
            ha='center',
            va='bottom',
            color=GROUP_COLORS[rg],
        )
    ax.set_xticks([0, 1, 3, 4], ['Whisker', 'Catch'] * 2)
    ax.set_xlim(-0.7, 4.7)
    ax.set_ylim(0, 100)
    ax.set_xlabel('')
    ax.set_ylabel(f'Lick probability (%)\nfirst {N_WHISKER} whisker trials, Day 0')
    sns.despine()
    plt.tight_layout()
    save_figure(fig, os.path.join(OUTPUT_DIR, f'{filename}.pdf'))
    plt.close(fig)


if __name__ == '__main__':
    table = pd.read_csv(
        paths.adjust_path_to_host(
            os.path.join(paths.processed_dir, 'behavior', 'behavior_imagingmice_table_5days_cut.csv')
        )
    )
    mice = initial_licking(table)
    stats = compute_stats(mice)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    mice.to_csv(os.path.join(OUTPUT_DIR, 'figure1_supfig1_b_data.csv'), index=False)
    stats.to_csv(os.path.join(OUTPUT_DIR, 'figure1_supfig1_b_stats.csv'), index=False)
    plot(mice, stats)
    print(stats.round(3).to_string(index=False))
    print(
        mice.groupby('reward_group')[['whisker', 'catch', 'first_whisker_trial', 'n_catch']].mean().round(2)
    )
