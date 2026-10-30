"""
Figure 6d-e: Participation in reactivations above chance vs LMI.

A cell that is often active crosses the participation threshold at many
moments, events or not, so its participation rate partly reflects its
activity. Its chance rate is the participation expected if events occurred
at random times: the fraction of time points (those where events are
counted) at which its +/- 150 ms mean dF/F reaches the threshold, on the
same day and trials. Participation above chance = participation rate -
chance rate, per cell-day (fast_learning.participation). The participation
rate itself is in Figure 6—figure supplement 1.

Panel d: day-0 participation above chance vs LMI (one dot per cell),
         separately for R+ and R- mice. Stats: linear mixed-effects model
         excess ~ LMI + (1 + LMI | mouse), with an intercept and a slope per
         mouse, so that the test of the slope takes the mouse, not the cell,
         as the unit. Stats file also: per-mouse Spearman correlation of LMI
         and participation, Wilcoxon signed-rank test against zero (n = mice),
         insensitive to the tails of the LMI distribution.

Panel e: participation above chance across days (-2 to +2) for LMI+ vs LMI-
         cells, per-mouse averages. Stats: linear trend across days with mice
         as the unit: one slope per mouse, then a Wilcoxon signed-rank test of
         the slopes against zero, for each (reward group, LMI category).

Participation is measured in held-out cells: events are detected with a
random half of the cells and participation is measured in the other half,
averaged over 10 random splits, so that no cell is scored on events it helped
detect (pipeline/08_participation.py, fast_learning.participation).
--cells all: participation of all cells in the events of step 07 (not
cross-validated); --cells insample: the detecting half in the same events.
Both are comparisons showing the effect of the circularity.

Cells: cell-days with >= 3 valid reactivation events (for held-out cells, in
at least one split). Mice: those in the participation mouse selection
(>= 3 reactivation events on day 0; see fast_learning.reactivations). LMI:
last 40 passive trials of each day (pipeline/05_lmi.py).

Inputs:  participation rates with chance rates from
         pipeline/08_participation.py, at each participation threshold: 2.5 x
         each cell's noise SD (main), 5 and 10 x (robustness checks; see
         fast_learning.participation).
Outputs: main analysis (2.5 noise SD, held-out cells, whole trial):
         <figures_dir>/Figure6/figure_6d.pdf, _stats.csv and figure_6e.pdf,
         _data.csv, _stats.csv. Other conditions, in Figure6/variants/:
         figure_6d_<k>sd<sfx> and figure_6e_<k>sd<sfx>; <sfx> is empty for
         held-out cells, _insample or _allcells, then _nolick for the no-lick
         control.
"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from fast_learning import paths, participation
from fast_learning.plotting import VARIANTS_DIR, reward_palette, save_figure, set_style, panel_size
from fast_learning.stats import (
    format_p,
    lmm_slope,
    per_mouse_rank_correlation,
    per_mouse_slope_test,
    significance_stars,
)


# ============================================================================
# Parameters
# ============================================================================

DAYS = [-2, -1, 0, 1, 2]


OUTPUT_DIR = os.path.join(paths.manuscript_output_dir, 'Figure6')


# ============================================================================
# Panel d: scatter of day-0 participation rate vs LMI, mixed-model stats
# ============================================================================


def panel_i_participation_vs_lmi(
    merged_df, output_dir=OUTPUT_DIR, filename='figure_6d', ylabel='Participation rate (day 0)'
):
    """Figure 6 Panel d: scatter of day-0 participation rate vs LMI.

    One dot per cell, separate subplots for R+ and R-. The line and the
    annotated statistic are the fixed-effect slope of
    learning_rate ~ lmi + (1 + lmi | mouse_id).

    Saves <filename>.pdf and <filename>_stats.csv (per reward group: slope,
    CI, p-value, share of variance of the mouse intercept, SD of the slope
    across mice, and the per-mouse Spearman correlation test).
    """
    set_style()

    df = merged_df.dropna(subset=['lmi', 'learning_rate']).copy()
    if 'reliable_learning' in df.columns:
        df = df[df['reliable_learning']]

    reward_groups = ['R+', 'R-']
    # Dots are rasterized (cannot be recoloured in Illustrator) and opaque.
    rg_colors = {'R+': reward_palette[1], 'R-': reward_palette[0]}
    fig, axes = plt.subplots(1, 2, figsize=panel_size(2), sharey=True)
    stats_rows = []

    for i, rg in enumerate(reward_groups):
        ax = axes[i]
        grp = df[df['reward_group'] == rg]
        x = grp['lmi'].values
        n_mice = grp['mouse_id'].nunique()

        ax.scatter(x, grp['learning_rate'].values, color=rg_colors[rg], s=2, linewidths=0, rasterized=True)

        if len(grp) >= 3 and n_mice >= 2:
            fit = lmm_slope(grp, 'learning_rate', 'lmi')
            rank, _ = per_mouse_rank_correlation(grp, 'learning_rate', 'lmi')
            x_line = np.linspace(x.min(), x.max(), 200)
            ax.plot(x_line, fit['slope'] * x_line + fit['intercept'], color='black', linewidth=1, zorder=5)
            ax.text(
                0.05,
                0.95,
                f"LMM slope = {fit['slope']:.3f}\n[{fit['ci_low']:.3f}, {fit['ci_high']:.3f}]\n{format_p(fit['p_value'])}",
                transform=ax.transAxes,
                va='top',
                ha='left',
            )
            stats_rows.append(
                {
                    'reward_group': rg,
                    'test': 'LMM learning_rate ~ lmi + (1 + lmi | mouse)',
                    'n_cells': len(grp),
                    'n_mice': n_mice,
                    'lmm_slope': fit['slope'],
                    'lmm_se': fit['se'],
                    'lmm_ci_low': fit['ci_low'],
                    'lmm_ci_high': fit['ci_high'],
                    'p_value': fit['p_value'],
                    'significance': significance_stars(fit['p_value']),
                    'icc_mouse': fit['icc_mouse'],
                    'slope_sd_mouse': fit['slope_sd_mouse'],
                    'method': fit['method'],
                    'spearman_median_rho': rank['median_rho'],
                    'spearman_n_positive': rank['n_positive'],
                    'spearman_p_wilcoxon': rank['p_value'],
                }
            )
            print(
                f"  LMM {rg}: slope={fit['slope']:.4g}, p={fit['p_value']:.4g} ({fit['method']}); "
                f"Spearman {rank['n_positive']}/{rank['n_mice']} mice > 0, p={rank['p_value']:.4g}"
            )

        ax.axvline(x=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
        ax.set_title(f'{rg}\n{len(grp)} cells, {n_mice} mice', fontweight='bold')
        ax.set_xlabel('LMI')
        ax.set_ylabel(ylabel if i == 0 else '')
        sns.despine(ax=ax)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    save_figure(fig, os.path.join(output_dir, f'{filename}.pdf'))
    plt.close(fig)
    pd.DataFrame(stats_rows).to_csv(os.path.join(output_dir, f'{filename}_stats.csv'), index=False)
    print(f"Panel i saved: {os.path.join(output_dir, filename)}")


# ============================================================================
# Panel e: participation rate across days (LMI+ vs LMI-)
# ============================================================================


def panel_j_participation_across_days(
    merged_df,
    per_day_df,
    output_dir=OUTPUT_DIR,
    filename='figure_6e',
    ylabel='Participation rate',
    ylim=(0, 0.4),
):
    """Figure 6 Panel e: participation rate across days for LMI+ vs LMI- cells.

    Bars are means of per-mouse averages. Stats: per-mouse day slope,
    Wilcoxon signed-rank test against zero (n = mice), for each
    (reward_group, lmi_category).

    Saves <filename>.pdf, <filename>_data.csv (per mouse x day x LMI
    category averages) and <filename>_stats.csv.
    """
    set_style()

    days_sorted = sorted(DAYS)
    lmi_categories = ['positive', 'negative']
    cat_colors = {'positive': '#d62728', 'negative': '#1f77b4'}
    cat_labels = {'positive': 'Positive LMI', 'negative': 'Negative LMI'}
    reward_groups = ['R+', 'R-']

    lmi_cells = merged_df.loc[
        merged_df['lmi_category'].isin(lmi_categories),
        ['mouse_id', 'roi', 'lmi_category', 'reward_group'],
    ]
    day_data = pd.merge(per_day_df, lmi_cells, on=['mouse_id', 'roi'], how='inner')
    if 'reliable' in day_data.columns:
        day_data = day_data[day_data['reliable']]

    mouse_day_avg = (
        day_data.groupby(['mouse_id', 'reward_group', 'lmi_category', 'day'], observed=True)[
            'participation_rate'
        ]
        .mean()
        .reset_index()
    )
    cell_counts = lmi_cells.groupby(['reward_group', 'lmi_category'], observed=True).size().to_dict()

    stats_rows = []
    tests = {}
    for rg in reward_groups:
        for cat in lmi_categories:
            grp = mouse_day_avg[
                (mouse_day_avg['reward_group'] == rg) & (mouse_day_avg['lmi_category'] == cat)
            ]
            test = per_mouse_slope_test(grp)
            if test is None:
                continue
            tests[(rg, cat)] = test
            stats_rows.append(
                {
                    'reward_group': rg,
                    'lmi_category': cat,
                    'test': 'Per-mouse day slope, Wilcoxon signed-rank (n = mice)',
                    'mean_day_slope': test['mean_slope'],
                    'median_day_slope': test['median_slope'],
                    'sd_day_slope': test['sd_slope'],
                    'w_stat': test['w_stat'],
                    'p_value': test['p_value'],
                    'significance': significance_stars(test['p_value']),
                    'n_mice': test['n_mice'],
                    'n_cells': cell_counts.get((rg, cat), 0),
                }
            )
            print(
                f"  {rg} {cat} LMI: median slope={test['median_slope']:.4g}, p={test['p_value']:.4g}, n={test['n_mice']}"
            )

    fig, axes = plt.subplots(1, 2, figsize=panel_size(2), sharey=True)
    for i, rg in enumerate(reward_groups):
        ax = axes[i]
        grp = mouse_day_avg[mouse_day_avg['reward_group'] == rg]
        sns.barplot(
            data=grp,
            x='day',
            y='participation_rate',
            hue='lmi_category',
            hue_order=lmi_categories,
            palette=cat_colors,
            order=days_sorted,
            estimator=np.mean,
            errorbar=('ci', 95),
            capsize=0,
            err_kws={'linewidth': 1.5},
            alpha=0.7,
            ax=ax,
            seed=0,
        )
        for patch in ax.patches:
            patch.set_edgecolor('black')
            patch.set_linewidth(0.6)

        short = {'positive': 'LMI+', 'negative': 'LMI−'}
        ax.text(0.02, 0.99, 'Day slope:', transform=ax.transAxes, va='top', ha='left')
        for j, cat in enumerate(lmi_categories):
            test = tests.get((rg, cat))
            text = f'{short[cat]}: n.a.' if test is None else f"{short[cat]}: {format_p(test['p_value'])}"
            ax.text(
                0.02,
                0.89 - j * 0.1,
                text,
                transform=ax.transAxes,
                va='top',
                ha='left',
                color=cat_colors[cat],
            )

        n_pos = cell_counts.get((rg, 'positive'), 0)
        n_neg = cell_counts.get((rg, 'negative'), 0)
        ax.set_title(f'{rg}\nLMI+ {n_pos}, LMI− {n_neg} cells')
        ax.set_xlabel('Day')
        ax.set_ylabel(ylabel if i == 0 else '')
        ax.set_ylim(*ylim)
        handles, labels = ax.get_legend_handles_labels()
        ax.get_legend().remove()
        sns.despine(ax=ax)

    plt.tight_layout()
    fig.legend(
        handles,
        [cat_labels[lab] for lab in labels],
        loc='upper center',
        bbox_to_anchor=(0.5, 0.0),
        ncol=2,
        frameon=False,
    )
    os.makedirs(output_dir, exist_ok=True)
    save_figure(fig, os.path.join(output_dir, f'{filename}.pdf'))
    plt.close(fig)
    mouse_day_avg.to_csv(os.path.join(output_dir, f'{filename}_data.csv'), index=False)
    pd.DataFrame(stats_rows).to_csv(os.path.join(output_dir, f'{filename}_stats.csv'), index=False)
    print(f"Panel j saved: {os.path.join(output_dir, filename)}")


# ============================================================================
# Main execution
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--cells',
        choices=participation.CELLS,
        default='heldout',
        help='held-out cells (main), in-sample cells of the same detections, or all cells (step 07 events)',
    )
    parser.add_argument(
        '--nolick', action='store_true', help='no-lick control (-1 to +1 s), written to Figure6/variants'
    )
    args = parser.parse_args()
    print(f"Output directory: {OUTPUT_DIR}")

    for threshold in participation.PARTICIPATION_THRESHOLDS:
        # Main analysis: figure_6d/e in OUTPUT_DIR; other conditions in its variants folder.
        tag = participation.variant_tag(threshold, args.nolick, args.cells)
        out_dir = os.path.join(OUTPUT_DIR, VARIANTS_DIR) if tag else OUTPUT_DIR
        print(f"\n--- participation threshold {threshold:g} x noise SD ({tag or 'main'}) ---")
        merged, per_day = participation.excess_tables(
            *participation.load_participation(threshold, args.nolick, args.cells)
        )
        print(
            f"Dataset: {len(merged)} cells, {len(per_day)} cell-day records, {merged['mouse_id'].nunique()} mice"
        )
        panel_i_participation_vs_lmi(
            merged,
            output_dir=out_dir,
            filename=f'figure_6d{tag}',
            ylabel='Participation above chance (day 0)',
        )
        panel_j_participation_across_days(
            merged,
            per_day,
            output_dir=out_dir,
            filename=f'figure_6e{tag}',
            ylabel='Participation above chance',
            ylim=(None, 0.12),
        )
