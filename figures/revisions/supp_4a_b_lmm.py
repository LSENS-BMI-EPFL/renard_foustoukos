"""
Supplementary Figure 4a-b revision: LMM-based spontaneous activity controls
for the LMI-participation relationship.

Addresses reviewer comment (3), which also applies here: both panels pool
cells across mice within a reward group and run plain Pearson correlations
(raw and partial), treating cells from the same mouse as independent
observations. Both are refit here with mouse_id as a random intercept
(statsmodels MixedLM).

This script reads the same day-0 data as supp_4a_b.py (written by
pipeline/08_participation.py) and only replaces the statistics and
figure-annotation logic.

Panel 4a: Scatter plot of cell participation rate vs spontaneous transient
          frequency (Day 0), colored by LMI. Stats: mixed-effects model
          (participation_rate ~ transient_freq + (1 | mouse_id)) replacing
          the pooled Pearson correlation.

Panel 4b: LMI vs participation rate, raw and controlling for transient
          frequency. Stats: for each reward group, two mixed-effects models
          - raw:     participation_rate ~ lmi + (1 | mouse_id)
          - partial: participation_rate ~ lmi + transient_freq + (1 | mouse_id)
          The lmi coefficient from the partial model (holding transient_freq
          fixed) is the mixed-model analog of the partial correlation. The
          residual scatter in the right-hand column is kept purely as an
          added-variable-plot visualization; the reported statistic comes
          from the mixed model, not from correlating those residuals.

Figures and CSVs are saved to
    paths.manuscript_output_dir/revisions/supp_4a_b_lmm/output/.
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from scipy.stats import linregress
from statsmodels.regression.mixed_linear_model import MixedLM

from fast_learning import paths, participation
from fast_learning.plotting import reward_palette, save_figure, lmi_cmap
from fast_learning.stats import significance_stars as _significance_stars

LMI_DATA_CSV = participation.DAY0_CSV

OUTPUT_DIR = os.path.join(paths.manuscript_output_dir, 'revisions', 'supp_4a_b_lmm', 'output')


# ============================================================================
# Mixed-effects model helper
# ============================================================================


def _fit_lmm(formula, data, group_col='mouse_id'):
    """Fit a linear mixed-effects model with mouse_id as a random intercept."""
    model = MixedLM.from_formula(formula, groups=group_col, data=data)
    return model.fit()


# ============================================================================
# Panel 4a: participation rate vs transient frequency scatter (LMI color)
# ============================================================================


def panel_supp4a_scatter_lmm(
    data_csv_path,
    output_dir=OUTPUT_DIR,
    filename='supp_4a_lmm',
    save_format='svg',
    dpi=300,
):
    """Supp Figure 4a (revision): scatter of participation rate vs transient
    frequency (Day 0), colored by LMI.

    Saves:
        <filename>.svg       -- figure
        <filename>_data.csv  -- data used for the plot
        <filename>_stats.csv -- LMM slope, CI, p per reward group
    """

    sns.set_theme(context='paper', style='ticks', palette='deep', font='sans-serif', font_scale=1)

    merged = pd.read_csv(data_csv_path)
    merged = merged.dropna(subset=['lmi', 'transient_freq', 'participation_rate', 'reward_group'])

    lmi_abs_max = np.abs(merged['lmi']).max()
    norm = mcolors.TwoSlopeNorm(vmin=-lmi_abs_max, vcenter=0, vmax=lmi_abs_max)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    stats_rows = []

    for i, reward_group in enumerate(['R+', 'R-']):
        ax = axes[i]
        gdata = merged[merged['reward_group'] == reward_group]
        n_mice = gdata['mouse_id'].nunique()

        if len(gdata) < 3 or n_mice < 2:
            ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(reward_group, fontweight='bold')
            continue

        sc = ax.scatter(
            gdata['transient_freq'],
            gdata['participation_rate'],
            c=gdata['lmi'],
            cmap=lmi_cmap,
            norm=norm,
            alpha=0.6,
            s=15,
            linewidths=0,
        )

        result = _fit_lmm('participation_rate ~ transient_freq', gdata)
        slope = result.params['transient_freq']
        intercept = result.params['Intercept']
        p_value = result.pvalues['transient_freq']
        ci_low, ci_high = result.conf_int().loc['transient_freq']

        x_range = np.linspace(gdata['transient_freq'].min(), gdata['transient_freq'].max(), 100)
        ax.plot(x_range, slope * x_range + intercept, 'k-', linewidth=1.5)

        stars = _significance_stars(p_value)
        ax.text(
            0.05,
            0.95,
            f'LMM slope = {slope:.3f} [{ci_low:.3f}, {ci_high:.3f}]\n'
            f'p = {p_value:.3g} {stars}\nn = {len(gdata)} cells, {n_mice} mice',
            transform=ax.transAxes,
            fontsize=10,
            va='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'),
        )

        plt.colorbar(sc, ax=ax, label='LMI')
        ax.set_xlabel('Transient frequency (events/min)', fontweight='bold', fontsize=12)
        ax.set_ylabel('Participation rate' if i == 0 else '', fontweight='bold', fontsize=12)
        ax.set_title(f'{reward_group}  (n={n_mice} mice, {len(gdata)} cells)', fontweight='bold', fontsize=13)
        ax.grid(True, alpha=0.3)
        sns.despine(ax=ax)

        stats_rows.append(
            {
                'reward_group': reward_group,
                'test': 'LMM participation_rate ~ transient_freq + (1|mouse_id)',
                'lmm_slope': slope,
                'lmm_p': p_value,
                'lmm_ci_low': ci_low,
                'lmm_ci_high': ci_high,
                'n_cells': len(gdata),
                'n_mice': n_mice,
                'converged': result.converged,
            }
        )
        print(
            f"  {reward_group}: LMM slope={slope:.4g}, p={p_value:.4g}, "
            f"n={len(gdata)} cells, {n_mice} mice, converged={result.converged}"
        )

    fig.suptitle(
        'Participation Rate vs Transient Frequency (Day 0, colored by LMI) - LMM',
        fontsize=13,
        fontweight='bold',
    )
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    save_figure(plt.gcf(), os.path.join(output_dir, f'{filename}.{save_format}'))
    plt.close()
    print(f"Panel saved: {os.path.join(output_dir, filename + '.' + save_format)}")

    merged.to_csv(os.path.join(output_dir, f'{filename}_data.csv'), index=False)
    pd.DataFrame(stats_rows).to_csv(os.path.join(output_dir, f'{filename}_stats.csv'), index=False)
    print(f"Data/stats saved: {output_dir}")


# ============================================================================
# Panel 4b: LMI vs participation rate, raw and controlling for transient freq
# ============================================================================


def panel_supp4b_partial_corr_lmm(
    data_csv_path,
    output_dir=OUTPUT_DIR,
    filename='supp_4b_lmm',
    save_format='svg',
    dpi=300,
):
    """Supp Figure 4b (revision): LMI vs participation rate, raw and partial.

    Layout unchanged: 2 rows (R+, R-) x 2 columns (raw, partial). The
    right-hand column still plots residuals-after-regressing-out-
    transient_freq as an added-variable-plot visualization, but the
    annotated statistic is the lmi coefficient of a mixed-effects model
    that includes transient_freq as a covariate, not a correlation of those
    residuals.

    Saves:
        <filename>.svg       -- figure
        <filename>_data.csv  -- data with residuals per reward group
        <filename>_stats.csv -- raw and partial LMM slope/CI/p per reward group
    """
    sns.set_theme(context='paper', style='ticks', palette='deep', font='sans-serif', font_scale=1)

    merged = pd.read_csv(data_csv_path)
    merged = merged.dropna(subset=['lmi', 'participation_rate', 'transient_freq', 'reward_group'])

    group_colors = {'R+': reward_palette[1], 'R-': reward_palette[0]}

    def residuals(a, b):
        slope, intercept, _, _, _ = linregress(b, a)
        return a - (slope * b + intercept)

    def annotate(ax, label, slope, p_value, ci_low, ci_high):
        stars = _significance_stars(p_value)
        ax.text(
            0.05,
            0.95,
            f'{label}\nslope = {slope:.3f} [{ci_low:.3f}, {ci_high:.3f}]\np = {p_value:.3g} {stars}',
            transform=ax.transAxes,
            fontsize=9,
            va='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'),
        )

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharey=True)
    stats_rows = []
    data_rows = []

    for row, reward_group in enumerate(['R+', 'R-']):
        color = group_colors[reward_group]
        gdata = merged[merged['reward_group'] == reward_group].copy()
        n_mice = gdata['mouse_id'].nunique()

        if len(gdata) < 5 or n_mice < 2:
            for col in range(2):
                axes[row, col].text(
                    0.5,
                    0.5,
                    'Insufficient data',
                    ha='center',
                    va='center',
                    transform=axes[row, col].transAxes,
                )
            continue

        result_raw = _fit_lmm('participation_rate ~ lmi', gdata)
        slope_raw = result_raw.params['lmi']
        p_raw = result_raw.pvalues['lmi']
        ci_raw = result_raw.conf_int().loc['lmi']

        result_partial = _fit_lmm('participation_rate ~ lmi + transient_freq', gdata)
        slope_partial = result_partial.params['lmi']
        p_partial = result_partial.pvalues['lmi']
        ci_partial = result_partial.conf_int().loc['lmi']

        print(
            f"  {reward_group}  raw LMM slope={slope_raw:.3f} p={p_raw:.4f} | "
            f"partial LMM slope={slope_partial:.3f} p={p_partial:.4f}  "
            f"(n={len(gdata)} cells, {n_mice} mice)"
        )

        stats_rows.append(
            {
                'reward_group': reward_group,
                'test': 'LMM (mouse_id random intercept)',
                'raw_slope': slope_raw,
                'raw_p': p_raw,
                'raw_ci_low': ci_raw[0],
                'raw_ci_high': ci_raw[1],
                'partial_slope': slope_partial,
                'partial_p': p_partial,
                'partial_ci_low': ci_partial[0],
                'partial_ci_high': ci_partial[1],
                'n_cells': len(gdata),
                'n_mice': n_mice,
                'raw_converged': result_raw.converged,
                'partial_converged': result_partial.converged,
            }
        )

        # Residuals are kept only for the added-variable-plot visualization;
        # the reported statistics above come from the LMM fits, not from
        # correlating these residuals.
        lmi = gdata['lmi'].values
        part = gdata['participation_rate'].values
        freq = gdata['transient_freq'].values
        lmi_resid = residuals(lmi, freq)
        part_resid = residuals(part, freq)

        gdata['lmi_resid'] = lmi_resid
        gdata['part_resid'] = part_resid
        data_rows.append(gdata)

        for col, (x, y, label, slope, p_value, ci, xlabel, ylabel) in enumerate(
            [
                (lmi, part, 'Raw', slope_raw, p_raw, ci_raw, 'LMI', 'Participation rate'),
                (
                    lmi_resid,
                    part_resid,
                    'Partial (ctrl transient freq)',
                    slope_partial,
                    p_partial,
                    ci_partial,
                    'LMI  (residual | transient freq)',
                    'Participation rate  (residual | transient freq)',
                ),
            ]
        ):
            ax = axes[row, col]
            ax.scatter(x, y, color=color, alpha=0.3, s=10, linewidths=0)

            plot_slope, plot_intercept, _, _, _ = linregress(x, y)
            x_range = np.linspace(x.min(), x.max(), 100)
            ax.plot(x_range, plot_slope * x_range + plot_intercept, color='black', linewidth=1.5)

            ax.axvline(0, color='gray', linestyle='--', linewidth=0.7, alpha=0.5)
            ax.axhline(0, color='gray', linestyle='--', linewidth=0.7, alpha=0.5)
            annotate(ax, label, slope, p_value, ci[0], ci[1])

            ax.set_xlim(-1, 1)
            ax.set_xlabel(xlabel, fontsize=11)
            ax.set_ylabel(ylabel if col == 0 else '', fontsize=11)
            title = 'Raw' if col == 0 else 'Partial  (ctrl transient freq)'
            ax.set_title(
                f'{reward_group} - {title}  (n={len(lmi)} cells, {n_mice} mice)',
                fontweight='bold',
                fontsize=12,
            )
            ax.grid(True, alpha=0.3)
            sns.despine(ax=ax)

    fig.suptitle(
        'LMI vs Participation Rate: Raw and Partial (LMM, mouse random intercept)',
        fontsize=13,
        fontweight='bold',
    )
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    save_figure(plt.gcf(), os.path.join(output_dir, f'{filename}.{save_format}'))
    plt.close()
    print(f"Panel saved: {os.path.join(output_dir, filename + '.' + save_format)}")

    pd.concat(data_rows, ignore_index=True).to_csv(
        os.path.join(output_dir, f'{filename}_data.csv'), index=False
    )
    pd.DataFrame(stats_rows).to_csv(os.path.join(output_dir, f'{filename}_stats.csv'), index=False)
    print(f"Data/stats saved: {output_dir}")


# ============================================================================
# Main execution
# ============================================================================

if __name__ == '__main__':
    print(f"Input:            {LMI_DATA_CSV}")
    print(f"Output directory: {OUTPUT_DIR}")
    participation.load_day0()  # fails early if step 08 has not been run

    print("\nPlotting panel supp_4a (LMM)...")
    panel_supp4a_scatter_lmm(LMI_DATA_CSV, filename='supp_4a_lmm')

    print("\nPlotting panel supp_4b (LMM)...")
    panel_supp4b_partial_corr_lmm(LMI_DATA_CSV, filename='supp_4b_lmm')

    print(f"\nDone. Figures saved to: {OUTPUT_DIR}")
