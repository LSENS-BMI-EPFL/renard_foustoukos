"""
Supplementary Figure 4c revision: LMM-based proportion of cells
participating in reactivation across days (LMI+ vs LMI-).

Addresses reviewer comment (3), which also applies here: the original
panel runs a Kruskal-Wallis test of the effect of day within each
(reward_group, lmi_category) group, treating the 5 repeated days from the
same mice as independent cross-sections. This mirrors the issue fixed in
figure_4i_j_lmm.py for Figure 4J — and, like there, a random-intercept LMM
(participating ~ day + (1 | mouse_id)) turned out to give anti-conservative
p-values: it assumes every mouse shares the same day-slope, so real
mouse-to-mouse differences in that slope leak into the (large, cell-level)
residual variance instead of a mouse-level term, and statsmodels.MixedLM's
large-sample Wald z-test is optimistic with only a handful of mice per
group. This panel is therefore tested directly at the mouse level instead:
one day-slope fit per mouse (on that mouse's own per-day proportions),
then a Wilcoxon signed-rank test of those slopes against zero (n = mice;
nonparametric, matching the manuscript's style elsewhere).

This script reads the same binary-participation data as supp_4c.py (written
by pipeline/08_participation.py) and only replaces the statistics and
figure-annotation logic.

Figures and CSVs are saved to
    paths.manuscript_output_dir/revisions/supp_4c_lmm/output/.
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress, wilcoxon

from fast_learning import paths, participation
from fast_learning.stats import significance_stars as _significance_stars
from fast_learning.plotting import save_figure

DAYS = participation.DAYS

OUTPUT_DIR = os.path.join(paths.manuscript_output_dir, 'revisions', 'supp_4c_lmm', 'output')


# ============================================================================
# Per-mouse day-slope test
# ============================================================================


def _fit_permouse_slope_test(mouse_day_data, value_col='proportion'):
    """Two-stage test of the day effect for one (reward_group, lmi_category)
    subset: fit one proportion-vs-day slope per mouse (on that mouse's own
    per-day proportions), then test whether those slopes differ from zero
    across mice with a Wilcoxon signed-rank test (the one-sample
    nonparametric test — not Mann-Whitney U, which compares two independent
    samples and doesn't apply to a single set of values tested against
    zero).

    Used instead of a mixed-effects model: a random-intercept-only LMM
    assumes every mouse shares the same day slope, so real mouse-to-mouse
    differences in that slope leak into the (large, cell-level) residual
    variance rather than a (small, mouse-level) random-slope term — and
    statsmodels.MixedLM reports a large-sample Wald z-test rather than a
    small-sample-corrected test, which is optimistic with only a handful
    of mice per group. Testing directly at the mouse level sidesteps both
    problems: the sample size and degrees of freedom are honestly just the
    number of mice.

    Returns None if fewer than 2 mice have at least 2 distinct days.
    """
    slopes = []
    for mouse_id, mdata in mouse_day_data.groupby('mouse_id'):
        mdata = mdata.dropna(subset=[value_col])
        if mdata['day'].nunique() < 2:
            continue
        slope, _, _, _, _ = linregress(mdata['day'], mdata[value_col])
        slopes.append(slope)

    n_mice = len(slopes)
    if n_mice < 2:
        return None

    slopes = np.array(slopes)
    try:
        w_stat, p_value = wilcoxon(slopes)
    except ValueError:
        w_stat, p_value = np.nan, np.nan
    return {
        'mean_slope': float(np.mean(slopes)),
        'median_slope': float(np.median(slopes)),
        'sd_slope': float(np.std(slopes, ddof=1)),
        'w_stat': float(w_stat) if not np.isnan(w_stat) else np.nan,
        'p_value': float(p_value) if not np.isnan(p_value) else np.nan,
        'n_mice': n_mice,
    }


# ============================================================================
# Panel: proportion of participating cells across days (LMI+ vs LMI-)
# ============================================================================


def panel_supp4c_proportion_across_days_lmm(
    df,
    output_dir=OUTPUT_DIR,
    filename='supp_4c_lmm',
    save_format='svg',
    dpi=300,
):
    """Supp Figure 4c (revision): proportion of cells participating across
    days for LMI+ vs LMI- cells (binary participation).

    Per-mouse averages with individual trajectories, as before. For each
    (reward_group, lmi_category) group, one day-slope is fit per mouse (on
    that mouse's own per-day proportions), then a Wilcoxon signed-rank test
    asks whether those slopes differ from zero across mice (n = mice) —
    replacing both the original per-group Kruskal-Wallis test and a
    random-intercept mixed model (which gave anti-conservative p-values
    here — see the module docstring).

    Saves:
        <filename>.svg         -- figure
        <filename>_data.csv    -- per-mouse x day x LMI-category proportions
        <filename>_stats.csv   -- per-mouse day-slope Wilcoxon signed-rank test per (reward_group, lmi_category)
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

    cell_counts = {
        (rg, cat): lmi_df[(lmi_df['reward_group'] == rg) & (lmi_df['lmi_category'] == cat)][
            ['mouse_id', 'roi']
        ]
        .drop_duplicates()
        .shape[0]
        for rg in reward_groups
        for cat in lmi_categories
    }

    # Per-mouse day-slope test within each (reward_group, lmi_category); n = mice
    all_stats_rows = []
    slopes_by_rg = {rg: {} for rg in reward_groups}
    for rg in reward_groups:
        for cat in lmi_categories:
            grp_mouse_day = mouse_day_prop[
                (mouse_day_prop['reward_group'] == rg) & (mouse_day_prop['lmi_category'] == cat)
            ]
            test = _fit_permouse_slope_test(grp_mouse_day, value_col='proportion')
            if test is None:
                print(f"  Skipping {rg} {cat} LMI: too few mice with >=2 days")
                continue
            slopes_by_rg[rg][cat] = test
            all_stats_rows.append(
                {
                    'reward_group': rg,
                    'lmi_category': cat,
                    'test': 'Per-mouse day slope, Wilcoxon signed-rank (n=mice)',
                    'mean_day_slope': test['mean_slope'],
                    'median_day_slope': test['median_slope'],
                    'sd_day_slope': test['sd_slope'],
                    'w_stat': test['w_stat'],
                    'p_value': test['p_value'],
                    'significance': _significance_stars(test['p_value']),
                    'n_mice': test['n_mice'],
                    'n_cells': cell_counts.get((rg, cat), 0),
                }
            )
            print(
                f"  {rg} {cat} LMI: median day slope={test['median_slope']:.4g}, "
                f"W={test['w_stat']:.3g}, "
                f"p={test['p_value']:.4g}, n_mice={test['n_mice']}"
            )

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

        for j, cat in enumerate(lmi_categories):
            s = slopes_by_rg.get(rg, {}).get(cat)
            if s is None:
                text = f'{cat.capitalize()} LMI: n.a.'
            else:
                stars = _significance_stars(s['p_value'])
                text = (
                    f"{cat.capitalize()} LMI day slope: p={s['p_value']:.3g} {stars} (n={s['n_mice']} mice)"
                )
            ax.text(
                0.02,
                0.97 - j * 0.12,
                text,
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

    panel_supp4c_proportion_across_days_lmm(df, filename='supp_4c_lmm')
