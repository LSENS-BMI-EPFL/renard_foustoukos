"""
Supplementary Figure 4c: Proportion of cells participating in reactivation
across days for LMI+ vs LMI- cells (binary participation).

Binary participation is determined by circular-shift control: a cell x day is
classified as 'participating' if the cell's real participation rate around
reactivation events exceeds the 95th percentile of a null distribution built
from N_SHIFTS circular shifts of the neural data (same shift applied to all
cells simultaneously, preserving inter-cell correlations).

Panel: Proportion of cells participating across days (-2 to +2) separately
       for LMI+ vs LMI- cells. Per-mouse averages with individual
       trajectories. Stats: Kruskal-Wallis test (effect of day) run
       independently for each of the four groups (R+ positive LMI,
       R+ negative LMI, R- positive LMI, R- negative LMI).

Mice: those in the participation mouse selection (>= 3 reactivation events
on day 0; see fast_learning.reactivations). Within those, a mouse x day is
kept only if it has at least 3 valid events (fast_learning.participation).

Inputs:  binary participation per cell-day (pipeline/08_participation.py).
Outputs: <figures_dir>/supp_4/output/supp_4c.svg, supp_4c_data.csv, supp_4c_stats.csv.

NOTE (revision): per reviewer comment (3), the per-group Kruskal-Wallis test
treats the 5 repeated days per mouse as independent cross-sections. A
corrected version (per-mouse day-slope fit, then a one-sample t-test of
those slopes across mice within each reward_group x lmi_category group) is
implemented in figures/revisions/supp_4c_lmm.py, on the same data.
(A random-intercept mixed model was tried first but gave anti-conservative
p-values with this few mice per group — see that script's docstring.)
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kruskal

from fast_learning import paths, participation
from fast_learning.stats import significance_stars
from fast_learning.plotting import save_figure


DAYS = participation.DAYS
OUTPUT_DIR = os.path.join(paths.manuscript_output_dir, 'supp_4', 'output')


# ============================================================================
# Panel: proportion of participating cells across days (LMI+ vs LMI-)
# ============================================================================


def panel_supp4c_proportion_across_days(
    df,
    output_dir=OUTPUT_DIR,
    filename='supp_4c',
    save_format='svg',
    dpi=300,
):
    """Supp Figure 4c: proportion of cells participating across days for
    LMI+ vs LMI- cells (binary participation).

    Per-mouse averages with individual trajectories. Stats: Kruskal-Wallis
    test (effect of day) run independently for each of the four groups
    (R+ positive LMI, R+ negative LMI, R- positive LMI, R- negative LMI).

    Saves:
        <filename>.svg         -- figure
        <filename>_data.csv    -- per-mouse x day x LMI-category proportions
        <filename>_stats.csv   -- Kruskal-Wallis results per group
    """
    sns.set_theme(context='paper', style='ticks', palette='deep', font='sans-serif', font_scale=1)

    days_sorted = sorted(DAYS)
    lmi_categories = ['positive', 'negative']
    cat_colors = {'positive': '#d62728', 'negative': '#1f77b4'}
    reward_groups = ['R+', 'R-']

    lmi_df = df[df['lmi_category'].isin(lmi_categories)].copy()

    mouse_day_prop = (
        lmi_df.groupby(['mouse_id', 'reward_group', 'lmi_category', 'day'], observed=True)['participating']
        .mean()
        .reset_index()
        .rename(columns={'participating': 'proportion'})
    )
    # Round away summation-order noise (~1e-16) so that equal proportions stay
    # tied in the rank-based Kruskal-Wallis test.
    mouse_day_prop['proportion'] = mouse_day_prop['proportion'].round(12)

    cell_counts = {
        (rg, cat): lmi_df[(lmi_df['reward_group'] == rg) & (lmi_df['lmi_category'] == cat)][
            ['mouse_id', 'roi']
        ]
        .drop_duplicates()
        .shape[0]
        for rg in reward_groups
        for cat in lmi_categories
    }

    # Kruskal-Wallis: effect of day within each (reward_group, lmi_category) group
    all_stats_rows = []
    kw_results = {}
    for rg in reward_groups:
        for cat in lmi_categories:
            grp_data = mouse_day_prop[
                (mouse_day_prop['reward_group'] == rg) & (mouse_day_prop['lmi_category'] == cat)
            ]
            day_groups = [grp_data[grp_data['day'] == day]['proportion'].values for day in days_sorted]
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
        grp = mouse_day_prop[mouse_day_prop['reward_group'] == rg]

        sns.barplot(
            data=grp,
            x='day',
            y='proportion',
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
                my = mdata['proportion'].values
                ax.plot(mx, my, '-', color=cat_colors[cat], linewidth=0.8, alpha=0.4, zorder=5)

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
        ax.set_title(f'{rg}  (LMI+: {n_pos} cells | LMI-: {n_neg} cells)', fontsize=9, fontweight='bold')
        ax.set_xlabel('Day', fontsize=9)
        ax.set_ylabel('Proportion of cells participating' if i == 0 else '', fontsize=9)
        ax.set_ylim(0, None)
        ax.tick_params(labelsize=8)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, [f'{lab.capitalize()} LMI' for lab in labels], fontsize=8)
        sns.despine(ax=ax)

        plot_data_rows.append(grp)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    save_figure(plt.gcf(), os.path.join(output_dir, f'{filename}.{save_format}'))
    plt.close()
    print(f"Panel saved: {os.path.join(output_dir, filename + '.' + save_format)}")

    pd.concat(plot_data_rows, ignore_index=True).to_csv(
        os.path.join(output_dir, f'{filename}_data.csv'), index=False
    )
    pd.DataFrame(all_stats_rows).to_csv(os.path.join(output_dir, f'{filename}_stats.csv'), index=False)
    print(f"Data/stats saved: {output_dir}")


# ============================================================================
# Main execution
# ============================================================================

if __name__ == '__main__':
    print(f"Output directory: {OUTPUT_DIR}")
    df = participation.load_binary_participation()

    print(
        f"\nDataset: {len(df)} cell-day records, "
        f"{df['mouse_id'].nunique()} mice, "
        f"{df[['mouse_id', 'roi']].drop_duplicates().shape[0]} unique cells"
    )

    panel_supp4c_proportion_across_days(df, filename='supp_4c')
