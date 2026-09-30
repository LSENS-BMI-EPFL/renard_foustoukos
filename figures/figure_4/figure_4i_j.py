"""
Figure 4i_j: Reactivation participation rate vs LMI.

Panel i: Scatter plot of day-0 participation rate vs LMI (one dot per cell),
         separately for R+ and R- mice, with a linear regression line and
         Pearson r coefficient.

Panel j: Participation rate across days (-2 to +2) for LMI+ vs LMI- cells,
         showing per-mouse averages. Stats: per-(reward_group, LMI category)
         Kruskal-Wallis test for an effect of day, run independently for
         each of the four groups.

NOTE (revision): per reviewer comment (3), these two panels' statistics
pool cells across mice as independent observations (panel i) or compare
two independently-obtained p-values to each other (panel j), rather than
accounting for within-mouse correlation or testing the day x LMI-category
interaction directly. A mixed-effects version (mouse_id as random
intercept, non-modulated cells reinstated as a third category in panel j)
is implemented in figures/revisions/figure_4i_j_lmm.py, reusing this
module's data pipeline unchanged.

Mice: those in the participation mouse selection (>= 3 reactivation events
on day 0; see fast_learning.reactivations).

Inputs:  participation rates from pipeline/08_participation.py, at each
         participation threshold (10% main, 20% and 50% robustness checks).
Outputs: <figures_dir>/figure_4/output/figure_4i_<sel>_thr<N>.svg, _stats.csv
         and figure_4j_<sel>_thr<N>.svg, _data.csv, _stats.csv.
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, linregress, kruskal

from fast_learning import paths, participation
from fast_learning.plotting import reward_palette, save_figure
from fast_learning.stats import significance_stars


# ============================================================================
# Parameters
# ============================================================================

DAYS = [-2, -1, 0, 1, 2]

# Trial selection of the reactivation events:
#   False : all no-stim trials, full window (main analysis)
#   True  : no-stim trials without licks, +/- 2 s around no-stim onset
NO_LICK_ONLY = False
SELECTION = 'nolick' if NO_LICK_ONLY else 'allnostim'

OUTPUT_DIR = os.path.join(paths.manuscript_output_dir, 'figure_4', 'output')


# ============================================================================
# Panel i: scatter of day-0 participation rate vs LMI
# ============================================================================


def panel_i_participation_vs_lmi(
    merged_df,
    output_dir=OUTPUT_DIR,
    filename='figure_4i',
    save_format='svg',
    dpi=300,
):
    """Figure 4 Panel i: scatter of day-0 participation rate vs LMI.

    One dot per cell. Separate subplots for R+ and R-.
    Linear regression line with Pearson r and p-value displayed.

    Saves:
        <filename>.svg        – figure
        <filename>_stats.csv  – per-reward-group Pearson r, p-value, regression params
    """
    sns.set_theme(context='paper', style='ticks', palette='deep', font='sans-serif', font_scale=1)

    df = merged_df.dropna(subset=['lmi', 'learning_rate']).copy()

    reward_groups = ['R+', 'R-']
    rg_colors = {'R+': reward_palette[1], 'R-': reward_palette[0]}
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    stats_rows = []

    for i, rg in enumerate(reward_groups):
        ax = axes[i]
        grp = df[df['reward_group'] == rg]
        x = grp['lmi'].values
        y = grp['learning_rate'].values

        # Scatter
        ax.scatter(x, y, color=rg_colors[rg], s=4, alpha=0.4, linewidths=0, rasterized=True)

        # Linear regression line
        if len(x) >= 3:
            slope, intercept, r_value, p_value, se = linregress(x, y)
            pearson_r, pearson_p = pearsonr(x, y)
            x_line = np.linspace(x.min(), x.max(), 200)
            ax.plot(x_line, slope * x_line + intercept, color='black', linewidth=1.2, zorder=5)
            stars = significance_stars(pearson_p)
            ax.text(
                0.05,
                0.95,
                f'r = {pearson_r:.3f}\np = {pearson_p:.3g} {stars}',
                transform=ax.transAxes,
                va='top',
                ha='left',
                fontsize=8,
            )
            stats_rows.append(
                {
                    'reward_group': rg,
                    'n_cells': len(x),
                    'n_mice': grp['mouse_id'].nunique(),
                    'pearson_r': pearson_r,
                    'p_value': pearson_p,
                    'significance': stars,
                    'slope': slope,
                    'intercept': intercept,
                    'stderr': se,
                }
            )
        else:
            pearson_r, pearson_p = np.nan, np.nan

        ax.axvline(x=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
        n_cells = len(grp)
        n_mice = grp['mouse_id'].nunique()
        ax.set_title(f'{rg}  (n = {n_cells} cells, {n_mice} mice)', fontsize=10, fontweight='bold')
        ax.set_xlabel('LMI', fontsize=9)
        ax.set_ylabel('Participation rate (day 0)' if i == 0 else '', fontsize=9)
        ax.tick_params(labelsize=8)
        sns.despine(ax=ax)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    save_figure(plt.gcf(), os.path.join(output_dir, f'{filename}.{save_format}'))
    plt.close()
    print(f"Panel i saved: {os.path.join(output_dir, filename + '.' + save_format)}")

    pd.DataFrame(stats_rows).to_csv(os.path.join(output_dir, f'{filename}_stats.csv'), index=False)
    print(f"Panel i stats saved: {output_dir}")


# ============================================================================
# Panel j: participation rate across days (LMI+ vs LMI-)
# ============================================================================


def panel_j_participation_across_days(
    merged_df,
    per_day_df,
    output_dir=OUTPUT_DIR,
    filename='figure_4j',
    save_format='svg',
    dpi=300,
):
    """Figure 4 Panel j: participation rate across days for LMI+ vs LMI- cells.

    Per-mouse averages with individual trajectories. Stats: Kruskal-Wallis
    test (effect of day) run independently for each of the four groups
    (R+ positive LMI, R+ negative LMI, R- positive LMI, R- negative LMI).

    Saves:
        <filename>.svg         – figure
        <filename>_data.csv    – per-mouse × day × LMI-category averages
        <filename>_stats.csv   – Kruskal-Wallis results per group
    """
    sns.set_theme(context='paper', style='ticks', palette='deep', font='sans-serif', font_scale=1)

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
        day_data.groupby(['mouse_id', 'reward_group', 'lmi_category', 'day'], observed=True)[
            'participation_rate'
        ]
        .mean()
        .reset_index()
    )
    # Round away summation-order noise (~1e-16) so that equal means stay tied
    # in the rank-based Kruskal-Wallis test.
    mouse_day_avg['participation_rate'] = mouse_day_avg['participation_rate'].round(12)

    cell_counts = lmi_cells.groupby(['reward_group', 'lmi_category'], observed=True).size().to_dict()

    # Kruskal-Wallis: effect of day within each (reward_group, lmi_category) group
    all_stats_rows = []
    kw_results = {}
    for rg in reward_groups:
        for cat in lmi_categories:
            grp_data = mouse_day_avg[
                (mouse_day_avg['reward_group'] == rg) & (mouse_day_avg['lmi_category'] == cat)
            ]
            day_groups = [
                grp_data[grp_data['day'] == day]['participation_rate'].values for day in days_sorted
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
            all_stats_rows.append(
                {
                    'reward_group': rg,
                    'lmi_category': cat,
                    'test': 'Kruskal-Wallis',
                    'effect': 'day',
                    'H_statistic': H,
                    'p_value': p,
                    'significance': significance_stars(p) if not np.isnan(p) else 'n.a.',
                    'n_days': len(day_groups),
                }
            )
            print(f"  KW {rg} {cat} LMI: H={H:.3f}, p={p:.4g}")

    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    plot_data_rows = []

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
        )
        for patch in ax.patches:
            patch.set_edgecolor('black')
            patch.set_linewidth(0.6)

        # # Individual mouse trajectories
        # for j, cat in enumerate(lmi_categories):
        #     if j >= len(ax.containers):
        #         continue
        #     cat_grp = grp[grp['lmi_category'] == cat]
        #     x_centers = {
        #         days_sorted[k]: bar.get_x() + bar.get_width() / 2
        #         for k, bar in enumerate(ax.containers[j])
        #         if k < len(days_sorted)
        #     }
        #     for mouse_id in cat_grp['mouse_id'].unique():
        #         mdata = cat_grp[cat_grp['mouse_id'] == mouse_id].sort_values('day')
        #         mx = [x_centers[d] for d in mdata['day'] if d in x_centers]
        #         my = mdata['participation_rate'].values
        #         ax.plot(mx, my, '-', color=cat_colors[cat],
        #                 linewidth=0.8, alpha=0.4, zorder=5)

        # Annotate Kruskal-Wallis results for each LMI group
        for j, cat in enumerate(lmi_categories):
            H, p = kw_results.get((rg, cat), (np.nan, np.nan))
            stars = significance_stars(p) if not np.isnan(p) else 'n.a.'
            ax.text(
                0.02,
                0.97 - j * 0.12,
                f'{cat.capitalize()} LMI: KW p={p:.3g} {stars}',
                transform=ax.transAxes,
                va='top',
                ha='left',
                fontsize=7,
                color=cat_colors[cat],
            )

        n_pos = cell_counts.get((rg, 'positive'), 0)
        n_neg = cell_counts.get((rg, 'negative'), 0)
        ax.set_title(f'{rg}  (LMI+: {n_pos} cells | LMI−: {n_neg} cells)', fontsize=9, fontweight='bold')
        ax.set_xlabel('Day', fontsize=9)
        ax.set_ylabel('Participation rate' if i == 0 else '', fontsize=9)
        ax.set_ylim(0, 0.4)
        ax.tick_params(labelsize=8)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, [f'{lab.capitalize()} LMI' for lab in labels], fontsize=8)
        sns.despine(ax=ax)

        plot_data_rows.append(grp)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    save_figure(plt.gcf(), os.path.join(output_dir, f'{filename}.{save_format}'))
    plt.close()
    print(f"Panel j saved: {os.path.join(output_dir, filename + '.' + save_format)}")

    pd.concat(plot_data_rows, ignore_index=True).to_csv(
        os.path.join(output_dir, f'{filename}_data.csv'), index=False
    )
    pd.DataFrame(all_stats_rows).to_csv(os.path.join(output_dir, f'{filename}_stats.csv'), index=False)
    print(f"Panel j data/stats saved: {output_dir}")


# ============================================================================
# Main execution
# ============================================================================

if __name__ == '__main__':
    print(f"Trial selection:  {SELECTION}")
    print(f"Output directory: {OUTPUT_DIR}")

    for threshold in participation.PARTICIPATION_THRESHOLDS:
        tag = participation.thr_tag(threshold)
        print(f"\n--- participation_threshold={threshold} ({tag}) ---")
        merged_df, per_day_df = participation.load_participation(threshold, SELECTION)
        print(
            f"Dataset: {len(merged_df)} cells, {len(per_day_df)} cell-day records, "
            f"{merged_df['mouse_id'].nunique()} mice"
        )

        panel_i_participation_vs_lmi(merged_df, filename=f'figure_4i_{SELECTION}_{tag}')
        panel_j_participation_across_days(merged_df, per_day_df, filename=f'figure_4j_{SELECTION}_{tag}')
