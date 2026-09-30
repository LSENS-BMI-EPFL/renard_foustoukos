"""
Figure 4i_j revision: LMM-based reactivation participation rate vs LMI.

Addresses reviewer comment (3). The two panels needed different fixes:

Panel i pools cells across mice as independent observations. This is
fixed with a linear mixed-effects model, mouse_id as a random intercept
(statsmodels MixedLM) — a single cross-sectional relationship per cell,
with no repeated-measures structure to misspecify.

Panel j's original Kruskal-Wallis test treated the 5 repeated days per
mouse as independent cross-sections. A first pass fit a random-intercept
LMM (participation_rate ~ day + (1 | mouse_id)) instead, but that gave
smaller p-values than simply averaging across mice and testing there —
because (1) a random-intercept-only model assumes every mouse shares the
same day-slope, so real mouse-to-mouse differences in that slope leak into
the (large, cell-level) residual variance instead of a mouse-level term,
and (2) statsmodels.MixedLM reports a large-sample Wald z-test, not a
small-sample-corrected test, which is optimistic with only a handful of
mice per group. Panel j is therefore tested directly at the mouse level
instead: one day-slope fit per mouse, then a Wilcoxon signed-rank test of
those slopes against zero (n = mice, correct small sample size by
construction; nonparametric, matching the manuscript's style elsewhere).

This script reads the same participation data as figure_4i_j.py (written by
pipeline/08_participation.py) and only replaces the statistics and
figure-annotation logic for panels I and J.

Panel i: Scatter plot of day-0 participation rate vs LMI (one dot per cell),
         separately for R+ and R- mice, with a linear mixed-effects model
         (participation_rate ~ lmi + (1 | mouse_id)) replacing the pooled
         Pearson correlation.

Panel j: Participation rate across days (-2 to +2) for LMI+ vs LMI- cells.
         Stats: for each (reward_group, lmi_category) group, one
         participation-rate-vs-day slope is fit per mouse (on that mouse's
         own per-day means), then a Wilcoxon signed-rank test asks whether
         those slopes differ from zero across mice — the simple-slope
         question the manuscript actually claims, tested with mice as the
         unit.

Figures and CSVs are saved to
    paths.manuscript_output_dir/revisions/figure_4i_j_lmm/output/.
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress, wilcoxon
from statsmodels.regression.mixed_linear_model import MixedLM

from fast_learning import paths, participation
from fast_learning.plotting import reward_palette, save_figure
from fast_learning.stats import significance_stars as _significance_stars

DAYS = [-2, -1, 0, 1, 2]
SELECTION = 'allnostim'  # trial selection of the reactivation events (see figure_4i_j.py)

OUTPUT_DIR = os.path.join(paths.manuscript_output_dir, 'revisions', 'figure_4i_j_lmm', 'output')


# ============================================================================
# Mixed-effects model helpers
# ============================================================================


def _fit_lmm_4i(df, group_col='mouse_id'):
    """Fit participation_rate ~ lmi + (1 | mouse_id).

    Returns the fit result plus the mouse-level intraclass correlation
    (fraction of total variance attributable to between-mouse differences),
    which quantifies the clustering the pooled Pearson correlation ignored.
    """
    model = MixedLM.from_formula('learning_rate ~ lmi', groups=group_col, data=df)
    result = model.fit()
    re_var = result.cov_re.iloc[0, 0]
    icc = re_var / (re_var + result.scale)
    return result, icc


def _fit_permouse_slope_test(mouse_day_data):
    """Two-stage test of the day effect for one (reward_group, lmi_category)
    subset: fit one participation-rate-vs-day slope per mouse (on that
    mouse's own per-day means, i.e. one row per mouse per day), then test
    whether those slopes differ from zero across mice with a Wilcoxon
    signed-rank test (the one-sample nonparametric test, matching the
    manuscript's nonparametric style elsewhere — not Mann-Whitney U, which
    compares two independent samples and doesn't apply to a single set of
    values tested against zero).

    Used instead of a mixed-effects model for this panel: a
    random-intercept-only LMM assumes every mouse shares the same day
    slope, so real mouse-to-mouse differences in that slope leak into the
    (large, cell-level) residual variance rather than a (small,
    mouse-level) random-slope term — and statsmodels.MixedLM reports a
    large-sample Wald z-test rather than a small-sample-corrected test,
    which is optimistic with only a handful of mice per group. Testing
    directly at the mouse level sidesteps both problems: the sample size
    and degrees of freedom are honestly just the number of mice.

    Returns None if fewer than 2 mice have at least 2 distinct days.
    """
    slopes = []
    for mouse_id, mdata in mouse_day_data.groupby('mouse_id'):
        mdata = mdata.dropna(subset=['participation_rate'])
        if mdata['day'].nunique() < 2:
            continue
        slope, _, _, _, _ = linregress(mdata['day'], mdata['participation_rate'])
        slopes.append(slope)

    n_mice = len(slopes)
    if n_mice < 2:
        return None

    slopes = np.array(slopes)
    try:
        w_stat, p_value = wilcoxon(slopes)
    except ValueError:
        # e.g. all slopes are exactly zero, or too few non-zero differences
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
# Panel i: scatter of day-0 participation rate vs LMI, LMM stats
# ============================================================================


def panel_i_participation_vs_lmi_lmm(
    merged_df,
    output_dir=OUTPUT_DIR,
    filename='figure_4i_lmm',
    save_format='svg',
    dpi=300,
):
    """Figure 4 Panel I (revision): scatter of day-0 participation rate vs LMI.

    One dot per cell. Separate subplots for R+ and R-. The annotated
    statistic is the fixed-effect slope of a mixed-effects model with a
    per-mouse random intercept, instead of the pooled Pearson r/p.

    Saves:
        <filename>.svg        - figure
        <filename>_stats.csv  - per-reward-group LMM slope, CI, p-value, ICC
    """
    sns.set_theme(context='paper', style='ticks', palette='deep', font='sans-serif', font_scale=1)

    df = merged_df.dropna(subset=['lmi', 'learning_rate']).copy()
    if 'reliable_learning' in df.columns:
        df = df[df['reliable_learning']]

    reward_groups = ['R+', 'R-']
    rg_colors = {'R+': reward_palette[1], 'R-': reward_palette[0]}
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    stats_rows = []

    for i, rg in enumerate(reward_groups):
        ax = axes[i]
        grp = df[df['reward_group'] == rg]
        x = grp['lmi'].values
        y = grp['learning_rate'].values
        n_mice = grp['mouse_id'].nunique()

        ax.scatter(x, y, color=rg_colors[rg], s=4, alpha=0.4, linewidths=0, rasterized=True)

        if len(grp) >= 3 and n_mice >= 2:
            result, icc = _fit_lmm_4i(grp)
            slope = result.params['lmi']
            intercept = result.params['Intercept']
            se = result.bse['lmi']
            p_value = result.pvalues['lmi']
            ci_low, ci_high = result.conf_int().loc['lmi']

            x_line = np.linspace(x.min(), x.max(), 200)
            ax.plot(x_line, slope * x_line + intercept, color='black', linewidth=1.2, zorder=5)
            stars = _significance_stars(p_value)
            ax.text(
                0.05,
                0.95,
                f'LMM slope = {slope:.3f} [{ci_low:.3f}, {ci_high:.3f}]\n'
                f'p = {p_value:.3g} {stars}  (ICC={icc:.2f})',
                transform=ax.transAxes,
                va='top',
                ha='left',
                fontsize=8,
            )
            stats_rows.append(
                {
                    'reward_group': rg,
                    'n_cells': len(x),
                    'n_mice': n_mice,
                    'lmm_slope': slope,
                    'lmm_se': se,
                    'lmm_ci_low': ci_low,
                    'lmm_ci_high': ci_high,
                    'lmm_p': p_value,
                    'significance': stars,
                    'icc_mouse': icc,
                    'converged': result.converged,
                }
            )
            print(
                f"  LMM {rg} learning_rate~lmi: slope={slope:.4g}, "
                f"p={p_value:.4g}, ICC={icc:.3f}, converged={result.converged}"
            )
        else:
            print(f"  Skipping LMM for {rg}: insufficient data (n_cells={len(grp)}, n_mice={n_mice})")

        ax.axvline(x=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
        ax.set_title(f'{rg}  (n = {len(grp)} cells, {n_mice} mice)', fontsize=10, fontweight='bold')
        ax.set_xlabel('LMI', fontsize=9)
        ax.set_ylabel('Participation rate (day 0)' if i == 0 else '', fontsize=9)
        ax.tick_params(labelsize=8)
        sns.despine(ax=ax)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    save_figure(plt.gcf(), os.path.join(output_dir, f'{filename}.{save_format}'))
    plt.close()
    print(f"Panel i (LMM) saved: {os.path.join(output_dir, filename + '.' + save_format)}")

    pd.DataFrame(stats_rows).to_csv(os.path.join(output_dir, f'{filename}_stats.csv'), index=False)
    print(f"Panel i (LMM) stats saved: {output_dir}")


# ============================================================================
# Panel j: participation rate across days (LMI+ vs LMI-)
# ============================================================================


def panel_j_participation_across_days_lmm(
    merged_df,
    per_day_df,
    output_dir=OUTPUT_DIR,
    filename='figure_4j_lmm',
    save_format='svg',
    dpi=300,
):
    """Figure 4 Panel J (revision): participation rate across days for
    LMI-positive vs LMI-negative cells.

    Per-mouse averages are still shown as bars (as before). For each
    (reward_group, lmi_category) group, one day-slope is fit per mouse
    (on that mouse's own per-day means), then a Wilcoxon signed-rank test
    asks whether those slopes differ from zero across mice — n = mice,
    nonparametric, matching the manuscript's style elsewhere. This replaces
    both the original per-group Kruskal-Wallis test (which treated the 5
    repeated days per mouse as independent cross-sections) and a
    random-intercept mixed model (which gave anti-conservative p-values
    here — see the module docstring).

    Saves:
        <filename>.svg         - figure
        <filename>_data.csv    - per-mouse x day x LMI-category averages (plotted)
        <filename>_stats.csv   - per-mouse day-slope Wilcoxon signed-rank test per (reward_group, lmi_category)
    """
    sns.set_theme(context='paper', style='ticks', palette='deep', font='sans-serif', font_scale=1)

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

    # Per-mouse day-slope test within each (reward_group, lmi_category); n = mice
    all_stats_rows = []
    slopes_by_rg = {rg: {} for rg in reward_groups}
    for rg in reward_groups:
        for cat in lmi_categories:
            grp_mouse_day = mouse_day_avg[
                (mouse_day_avg['reward_group'] == rg) & (mouse_day_avg['lmi_category'] == cat)
            ]
            test = _fit_permouse_slope_test(grp_mouse_day)
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

        for j, cat in enumerate(lmi_categories):
            s = slopes_by_rg.get(rg, {}).get(cat)
            if s is None:
                text = f'{cat_labels[cat]}: n.a.'
            else:
                stars = _significance_stars(s['p_value'])
                text = f"{cat_labels[cat]} day slope: p={s['p_value']:.3g} {stars} (n={s['n_mice']} mice)"
            ax.text(
                0.02,
                0.97 - j * 0.09,
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
        ax.set_ylabel('Participation rate' if i == 0 else '', fontsize=9)
        ax.set_ylim(0, 0.4)
        ax.tick_params(labelsize=8)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, [cat_labels[lab] for lab in labels], fontsize=8)
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
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Participation thresholds: {participation.PARTICIPATION_THRESHOLDS}")

    for participation_threshold in participation.PARTICIPATION_THRESHOLDS:
        tag = participation.thr_tag(participation_threshold)
        print(f"\n--- participation_threshold={participation_threshold} ({tag}) ---")
        merged_df, per_day_df = participation.load_participation(participation_threshold, SELECTION)
        print(
            f"Dataset: {len(merged_df)} cells, {len(per_day_df)} cell-day records, "
            f"{merged_df['mouse_id'].nunique()} mice"
        )

        panel_i_participation_vs_lmi_lmm(merged_df, filename=f'figure_4i_lmm_{SELECTION}_{tag}')
        panel_j_participation_across_days_lmm(
            merged_df, per_day_df, filename=f'figure_4j_lmm_{SELECTION}_{tag}'
        )
