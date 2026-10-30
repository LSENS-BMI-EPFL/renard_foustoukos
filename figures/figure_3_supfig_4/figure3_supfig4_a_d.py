"""
Figure 3—figure supplement 4a-d: Stability of the passive whisker responses
within the passive block, and day-to-day drift between the baseline days
(-2/-1) and across learning (-1/0)

Response vectors: all cells, last 40 passive trials of each day (block order),
mean dF/F over 0-300 ms (as Fig. 3h-i); n = mice (18 R+, 16 R-).
- a, b: population response (mean over cells) along the block, in bins of 10
  trials, per day (R+, R-).
- c: first vs last 20 trials, each mouse averaged over the five days
  (adaptation within the block).
- d, e: pre (Days -2/-1) vs post (Days +1/+2) learning, computed from the
  first (d) or the last (e) 20 trials of each day only.
- f: within each day, correlation of the mean response vectors of the first
  and the last 20 trials, and of the odd and the even trials (same number of
  trials, no separation in time); each mouse's median over the five days.
- g: pattern correlation (noise-corrected; 1 = same pattern) within the
  passive block (first vs last 20 trials, median over days) and between
  consecutive days; within-block and -1/0 vs -2/-1 within group (p-values
  above them) and R+ vs R- on -2/-1 (Mann-Whitney U).
- h-j: LMI computed between consecutive days (one day on each side, as
  pipeline step 05 otherwise): fraction of LMI+ cells (h), LMI- cells (i) and
  the net fraction (j, LMI+ minus LMI-), with the two-day LMI of Fig. 3
  (-2,-1 / +1,+2) for reference. Change from -2/-1 to -1/0 within group
  (p-values above -1/0); net fraction against zero (p-values above -2/-1 and
  the two-day LMI); R+ vs R- (Mann-Whitney U, stats file only).
Wilcoxon signed-rank tests (n = mice) unless stated. Bars and points: mean
and bootstrapped 95% CI across mice; lines: mice. See fast_learning.drift.

Outputs: <figures_dir>/Figure3_SupFig4/figure3_supfig4_a_d.pdf,
figure3_supfig4_a_b_data.csv (mouse x day x bin), figure3_supfig4_c_e_data.csv
(mouse x day x half), figure3_supfig4_c_e_stats.csv, figure3_supfig4_f_data.csv
(mouse x day), figure3_supfig4_f_stats.csv, figure3_supfig4_g_data.csv
(mouse x pair), figure3_supfig4_g_stats.csv, figure3_supfig4_h_j_data.csv
(mouse x pair), figure3_supfig4_h_j_cells.csv (cell x pair),
figure3_supfig4_h_j_stats.csv.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from fast_learning import database, paths
from fast_learning.drift import (
    BIN,
    DAY_PAIRS,
    DAYS,
    N_HALF,
    N_MAP_TRIALS,
    FRACTIONS,
    GROUP_COLORS,
    LEARNING_PAIR,
    LMI_POSITIVE_THRESHOLD,
    REFERENCE,
    WITHIN,
    amplitude_stats,
    amplitude_tables,
    compute_metrics,
    drift_lmi,
    drift_lmi_stats,
    fractions,
    load_response_vectors,
    pair_label,
    pre_post_halves,
    signal_noise_stats,
    within_block,
    within_block_means,
    within_block_stats,
)
from fast_learning.plotting import panel_size, save_figure, set_style
from fast_learning.stats import format_p

OUTPUT_DIR = os.path.join(paths.manuscript_output_dir, 'Figure3_SupFig4')
DAY_PALETTE = sns.color_palette('viridis', len(DAYS))


def pair_points(ax, data, y, order, legend=False, x='pair'):
    """Mean and bootstrapped 95% CI across mice for each day pair."""
    sns.pointplot(
        data=data,
        x=x,
        y=y,
        order=order,
        hue='reward_group',
        hue_order=['R+', 'R-'],
        palette=GROUP_COLORS,
        errorbar=('ci', 95),
        seed=0,
        dodge=0.3,
        markersize=3,
        linewidth=1,
        legend=legend,
        ax=ax,
    )
    ax.set_xlabel('Day pair')


def bins_panel(ax, bins, rg, legend=False):
    """Population response along the block, in bins of BIN trials, per day."""
    d = bins[bins['reward_group'] == rg].assign(response=lambda t: 100 * t['response'])
    sns.lineplot(
        data=d,
        x='bin',
        y='response',
        hue='day',
        hue_order=DAYS,
        palette=DAY_PALETTE,
        errorbar=('ci', 95),
        seed=0,
        err_kws=dict(alpha=0.15),
        legend=legend,
        ax=ax,
    )
    n_bins = N_MAP_TRIALS // BIN
    ax.set_xticks(range(1, n_bins + 1), [f'{(b - 1) * BIN + 1}–{b * BIN}' for b in range(1, n_bins + 1)])
    ax.set_xlabel(f'Passive trial (last {N_MAP_TRIALS})')
    ax.set_ylabel('Response (% ΔF/F0)')
    ax.set_ylim(0, 8)
    ax.set_title(rg, color=GROUP_COLORS[rg])
    if legend:
        ax.legend(frameon=False, title='Day', loc='upper right', ncol=2)


def paired_bars(ax, wide, cols, labels, stats, measure):
    """Two paired conditions per group: bars (mean, 95% CI), lines (mice)."""
    for gi, rg in enumerate(GROUP_COLORS):
        g = 100 * wide.xs(rg, level='reward_group')[list(cols)]
        x = np.array([gi * 3, gi * 3 + 1])
        for _, row in g.iterrows():
            ax.plot(x, row.values, '-', color='grey', linewidth=0.5, alpha=0.6)
        long = g.reset_index().melt(id_vars='mouse_id', value_vars=list(cols), var_name='c', value_name='v')
        sns.barplot(
            data=long.assign(x=np.where(long['c'] == cols[0], x[0], x[1])),
            x='x',
            y='v',
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
        p = stats[(stats['reward_group'] == rg) & (stats['measure'] == measure)]['p_value'].iloc[0]
        kw = dict(transform=ax.get_xaxis_transform(), ha='center', va='bottom')
        ax.text(x.mean(), 1.0, format_p(p), fontsize=5, **kw)
        ax.text(x.mean(), 1.08, rg, color=GROUP_COLORS[rg], **kw)
    ax.set_xticks([0, 1, 3, 4], list(labels) * 2)
    ax.set_xlim(-0.7, 4.7)
    ax.set_ylim(0, 12)
    ax.set_yticks(range(0, 13, 4))
    ax.set_xlabel('')
    ax.set_ylabel('Response (% ΔF/F0)')


def within_block_panel(ax, means, stats):
    """First/last vs odd/even correlation within each day."""
    splits = {'r_first_last': 'First vs\nlast 20', 'r_odd_even': 'Odd vs\neven'}
    long = means.reset_index().melt(
        id_vars=['reward_group', 'mouse_id'], value_vars=list(splits), var_name='split', value_name='r'
    )
    pair_points(ax, long, 'r', list(splits), legend=True, x='split')
    ax.set_xticks(range(len(splits)), list(splits.values()))
    ax.set_xlabel('Trials of the same day')
    pvalues(ax, 0.5, stats[stats['measure'] == 'r all days'])
    ax.set_ylim(0, 1)
    ax.set_ylabel('Pattern correlation')
    ax.legend(frameon=False, loc='lower left')


def pattern_drift_panel(ax, drift_df, stats):
    """Noise-corrected correlation within the passive block and between
    consecutive days."""
    order = [WITHIN] + [pair_label(*p) for p in DAY_PAIRS]
    pair_points(ax, drift_df, 'r_corrected', order)
    ax.set_xticks(range(len(order)), ['Within\nday', *order[1:]])
    ax.set_xlabel('')
    ax.axhline(1, color='grey', linestyle='--', linewidth=0.6)
    # p-values above each pair: -1/0 vs -2/-1 within group (colored), R+ vs R- (black).
    for gi, rg in enumerate(['R+', 'R-', 'R+ vs R-']):
        for xi, pair in enumerate(order):
            row = stats[(stats['reward_group'] == rg) & (stats['measure'] == f'drift {pair}')]
            if len(row):
                ax.text(
                    xi,
                    1.04 + 0.06 * gi if rg in GROUP_COLORS else 1.04,
                    ('' if rg in GROUP_COLORS else 'R+ vs R−\n') + format_p(row['p_value'].iloc[0]),
                    ha='center',
                    va='bottom',
                    color=GROUP_COLORS.get(rg, 'black'),
                    fontsize=5,
                )
    ax.set_ylim(0.4, 1.25)
    ax.set_ylabel('Pattern correlation\n(noise-corrected)')


def pvalues(ax, x, rows):
    for gi, rg in enumerate(['R+', 'R-']):  # bottom to top
        r = rows[rows['reward_group'] == rg]
        if len(r):
            ax.text(
                x,
                1.0 + 0.07 * gi,
                format_p(r['p_value'].iloc[0]),
                transform=ax.get_xaxis_transform(),
                ha='center',
                va='bottom',
                color=GROUP_COLORS[rg],
                fontsize=5,
            )


def lmi_panel(ax, frac, stats, m):
    order = [pair_label(*p) for p in DAY_PAIRS] + [REFERENCE]
    pair_points(ax, frac.assign(pct=100 * frac[m]), 'pct', order)
    ax.axvline(len(DAY_PAIRS) - 0.5, color='grey', linewidth=0.4)
    ax.set_xticks(range(len(order)), [*order[:-1], 'LMI\n(2 days)'])
    rows = stats[stats['measure'] == m]
    pvalues(ax, order.index(pair_label(*LEARNING_PAIR)), rows[rows['pair'].str.contains('minus')])
    if m == 'net':
        ax.axhline(0, color='grey', linestyle='--', linewidth=0.6)
        ax.set_ylim(-25, 25)
        for pair in [order[0], REFERENCE]:
            pvalues(ax, order.index(pair), rows[rows['pair'] == pair])
    else:
        ax.axhline(100 * (1 - LMI_POSITIVE_THRESHOLD), color='grey', linestyle='--', linewidth=0.6)
        ax.set_ylim(0, 30)
    ax.set_ylabel(FRACTIONS[m])


if __name__ == '__main__':
    vectors = load_response_vectors()
    day_df, drift_df = compute_metrics(vectors)
    drift_stats = signal_noise_stats(day_df, drift_df)[0]
    drift_stats = drift_stats[drift_stats['measure'].str.startswith('drift')]
    wb = within_block(vectors)
    wb_means = within_block_means(wb)
    wb_stats = within_block_stats(wb, drift_df)
    within = wb_means['r_corrected'].reset_index().assign(pair=WITHIN)
    pattern_df = pd.concat([within, drift_df], ignore_index=True)
    pattern_stats = pd.concat([wb_stats[wb_stats['measure'] == f'drift {WITHIN}'], drift_stats])

    lmi_df = pd.read_csv(os.path.join(paths.processed_dir, 'lmi_results.csv'))
    _, _, _, db = database.select_sessions_from_db(paths.db_path, paths.nwb_dir, two_p_imaging='yes')
    groups = {
        m: database.get_mouse_reward_group_from_db(paths.db_path, m, db) for m in lmi_df['mouse_id'].unique()
    }
    cells = drift_lmi(groups, lmi_df)
    reference = lmi_df.assign(pair=REFERENCE, reward_group=lmi_df['mouse_id'].map(groups))
    frac = fractions(pd.concat([cells, reference[cells.columns.intersection(reference.columns)]]))
    lmi_stats = drift_lmi_stats(frac)

    bins, halves = amplitude_tables(vectors)
    amp_stats = amplitude_stats(halves)

    set_style()
    fig, axes = plt.subplots(2, 5, figsize=panel_size(5, 2))
    for ax, rg in zip(axes[0, :2], GROUP_COLORS):
        bins_panel(ax, bins, rg, legend=rg == 'R-')
    block = halves.groupby(['reward_group', 'mouse_id', 'half'])['response'].mean().unstack()
    paired_bars(axes[0, 2], block, ('first', 'last'), ('First', 'Last'), amp_stats, 'amplitude first/last')
    axes[0, 2].set_title(f'First vs last {N_HALF} trials', pad=16)
    pp = pre_post_halves(halves)
    for ax, half in zip(axes[0, 3:], ('first', 'last')):
        measure = f'amplitude pre/post, {half} {N_HALF} trials'
        paired_bars(ax, pp.xs(half, level='half'), ('pre', 'post'), ('Pre', 'Post'), amp_stats, measure)
        ax.set_title(f'{half.capitalize()} {N_HALF} trials', pad=16)
    within_block_panel(axes[1, 0], wb_means, wb_stats)
    pattern_drift_panel(axes[1, 1], pattern_df, pattern_stats)
    for ax, m in zip(axes[1, 2:], FRACTIONS):
        lmi_panel(ax, frac, lmi_stats, m)
    sns.despine()
    plt.tight_layout()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_figure(fig, os.path.join(OUTPUT_DIR, 'figure3_supfig4_a_d.pdf'))
    plt.close(fig)

    out = {
        'a_b_data': bins,
        'c_e_data': halves,
        'c_e_stats': amp_stats,
        'f_data': wb,
        'f_stats': wb_stats,
        'g_data': pattern_df,
        'g_stats': pattern_stats,
        'h_j_data': frac,
        'h_j_cells': cells,
        'h_j_stats': lmi_stats,
    }
    for name, df in out.items():
        df.to_csv(os.path.join(OUTPUT_DIR, f'figure3_supfig4_{name}.csv'), index=False)
    print(amp_stats.drop(columns=['test']).round(4).to_string(index=False))
    print(wb_stats.drop(columns=['test']).round(4).to_string(index=False))
    print(
        drift_stats[['reward_group', 'measure', 'median_pre', 'median_post', 'p_value']].to_string(
            index=False
        )
    )
    print(
        lmi_stats[['pair', 'measure', 'reward_group', 'median_r_plus', 'median_r_minus', 'p_value']]
        .round(4)
        .to_string(index=False)
    )
