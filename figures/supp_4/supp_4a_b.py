"""
Supplementary Figure 4a-b: Spontaneous activity controls for the
LMI-participation relationship.

Panel 4a: Scatter plot of cell participation rate vs spontaneous transient
frequency (Day 0). One dot per cell, color-coded by LMI using the coolwarm
colormap (blue = negative LMI, red = positive LMI, centered at 0).
One panel per reward group.

Panel 4b: Partial correlation — LMI vs participation rate, raw and after
controlling for spontaneous transient frequency. Layout: 2 rows (R+, R-) x
2 columns (raw, partial). Tests whether spontaneous activity explains the
LMI-participation relationship.

Mice: those in the participation mouse selection (>= 3 reactivation events
on day 0; see fast_learning.reactivations).

Inputs:  day-0 participation, transient frequency and LMI per cell
         (pipeline/08_participation.py).
Outputs: <figures_dir>/supp_4/output/supp_4a.svg, supp_4b.svg and their
         _data.csv / _stats.csv.

NOTE (revision): per reviewer comment (3), both panels' Pearson correlations
(raw and partial) pool cells across mice as independent observations. A
mixed-effects version (mouse_id as random intercept; the partial correlation
becomes the lmi coefficient of a participation_rate ~ lmi + transient_freq
model) is implemented in figures/revisions/supp_4a_b_lmm.py, on the
same data.
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from scipy.stats import pearsonr, linregress

from fast_learning import paths, participation
from fast_learning.plotting import reward_palette, save_figure, lmi_cmap


OUTPUT_DIR = os.path.join(paths.manuscript_output_dir, 'supp_4', 'output')
LMI_DATA_CSV = participation.DAY0_CSV


# ============================================================================
# Panel 4a: participation rate vs transient frequency scatter (LMI color)
# ============================================================================


def panel_supp4a_scatter(
    data_csv_path,
    output_dir=OUTPUT_DIR,
    filename='supp_4a',
    save_format='svg',
    dpi=300,
):
    """
    Supp Figure 4a: scatter of participation rate vs transient frequency (Day 0).
    One dot per cell, colored by LMI (coolwarm colormap, centered at 0).
    One panel per reward group. Regression line and Pearson r annotated.

    Saves:
        <filename>.svg       -- figure
        <filename>_data.csv  -- data used for the plot
        <filename>_stats.csv -- Pearson r and p per reward group
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

        if len(gdata) < 3:
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

        slope, intercept, _, _, _ = linregress(gdata['transient_freq'], gdata['participation_rate'])
        x_range = np.linspace(gdata['transient_freq'].min(), gdata['transient_freq'].max(), 100)
        ax.plot(x_range, slope * x_range + intercept, 'k-', linewidth=1.5)

        r, p = pearsonr(gdata['transient_freq'], gdata['participation_rate'])
        p_str = (
            'p < 0.001 ***'
            if p < 0.001
            else f'p = {p:.3f} **'
            if p < 0.01
            else f'p = {p:.3f} *'
            if p < 0.05
            else f'p = {p:.3f} ns'
        )
        n_mice = gdata['mouse_id'].nunique()
        ax.text(
            0.05,
            0.95,
            f'r = {r:.3f}\n{p_str}\nn = {len(gdata)} cells, {n_mice} mice',
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
                'pearson_r': r,
                'p_value': p,
                'n_cells': len(gdata),
                'n_mice': n_mice,
                'test': 'Pearson r (transient_freq vs participation_rate)',
            }
        )
        print(f"  {reward_group}: r={r:.3f}, p={p:.4f}, n={len(gdata)} cells, {n_mice} mice")

    fig.suptitle(
        'Participation Rate vs Transient Frequency (Day 0, colored by LMI)',
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
# Panel 4b: partial correlation — LMI vs participation | transient freq
# ============================================================================


def panel_supp4b_partial_corr(
    data_csv_path,
    output_dir=OUTPUT_DIR,
    filename='supp_4b',
    save_format='svg',
    dpi=300,
):
    """
    Supp Figure 4b: added-variable scatter plots comparing raw and partial
    correlation between LMI and participation rate.

    Layout: 2 rows (R+, R-) x 2 columns (raw, partial).
      Left column : raw scatter of LMI vs participation_rate.
      Right column: residuals of LMI and participation_rate after regressing
                    each on transient_freq (partial regression / added-variable
                    plot). Slope and r equal the partial regression coefficient
                    and partial correlation.

    If the partial r (right) remains significant and close in magnitude to the
    raw r (left), spontaneous activity does not explain the LMI-participation
    relationship.

    Saves:
        <filename>.svg       -- figure
        <filename>_data.csv  -- data with residuals per reward group
        <filename>_stats.csv -- raw and partial Pearson r / p per reward group
    """
    sns.set_theme(context='paper', style='ticks', palette='deep', font='sans-serif', font_scale=1)

    merged = pd.read_csv(data_csv_path)
    merged = merged.dropna(subset=['lmi', 'participation_rate', 'transient_freq', 'reward_group'])

    group_colors = {'R+': reward_palette[1], 'R-': reward_palette[0]}

    def residuals(a, b):
        slope, intercept, _, _, _ = linregress(b, a)
        return a - (slope * b + intercept)

    def annotate(ax, r, p):
        p_str = (
            'p < 0.001 ***'
            if p < 0.001
            else f'p = {p:.3f} **'
            if p < 0.01
            else f'p = {p:.3f} *'
            if p < 0.05
            else f'p = {p:.3f} ns'
        )
        ax.text(
            0.05,
            0.95,
            f'r = {r:.3f}\n{p_str}',
            transform=ax.transAxes,
            fontsize=10,
            va='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'),
        )

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharey=True)
    stats_rows = []
    data_rows = []

    for row, reward_group in enumerate(['R+', 'R-']):
        color = group_colors[reward_group]
        gdata = merged[merged['reward_group'] == reward_group].copy()

        if len(gdata) < 5:
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

        lmi = gdata['lmi'].values
        part = gdata['participation_rate'].values
        freq = gdata['transient_freq'].values

        lmi_resid = residuals(lmi, freq)
        part_resid = residuals(part, freq)

        r_raw, p_raw = pearsonr(lmi, part)
        r_partial, p_partial = pearsonr(lmi_resid, part_resid)

        n_mice = gdata['mouse_id'].nunique()
        print(
            f"  {reward_group}  raw r={r_raw:.3f} p={p_raw:.4f} | "
            f"partial r={r_partial:.3f} p={p_partial:.4f}  "
            f"(n={len(lmi)} cells, {n_mice} mice)"
        )

        stats_rows.append(
            {
                'reward_group': reward_group,
                'r_raw': r_raw,
                'p_raw': p_raw,
                'r_partial': r_partial,
                'p_partial': p_partial,
                'n_cells': len(lmi),
                'n_mice': n_mice,
                'test': 'Pearson r (LMI vs participation_rate)',
            }
        )

        gdata = gdata.copy()
        gdata['lmi_resid'] = lmi_resid
        gdata['part_resid'] = part_resid
        data_rows.append(gdata)

        for col, (x, y, r, p, xlabel, ylabel) in enumerate(
            [
                (lmi, part, r_raw, p_raw, 'LMI', 'Participation rate'),
                (
                    lmi_resid,
                    part_resid,
                    r_partial,
                    p_partial,
                    'LMI  (residual | transient freq)',
                    'Participation rate  (residual | transient freq)',
                ),
            ]
        ):
            ax = axes[row, col]
            ax.scatter(x, y, color=color, alpha=0.3, s=10, linewidths=0)

            slope, intercept, _, _, _ = linregress(x, y)
            x_range = np.linspace(x.min(), x.max(), 100)
            ax.plot(x_range, slope * x_range + intercept, color='black', linewidth=1.5)

            ax.axvline(0, color='gray', linestyle='--', linewidth=0.7, alpha=0.5)
            ax.axhline(0, color='gray', linestyle='--', linewidth=0.7, alpha=0.5)
            annotate(ax, r, p)

            ax.set_xlim(-1, 1)
            ax.set_xlabel(xlabel, fontsize=11)
            ax.set_ylabel(ylabel if col == 0 else '', fontsize=11)
            title = 'Raw' if col == 0 else 'Partial  (ctrl transient freq)'
            ax.set_title(
                f'{reward_group} — {title}  (n={len(lmi)} cells, {n_mice} mice)',
                fontweight='bold',
                fontsize=12,
            )
            ax.grid(True, alpha=0.3)
            sns.despine(ax=ax)

    fig.suptitle(
        'LMI vs Participation Rate: Raw and Partial Correlation (Day 0)',
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

    print("\nPlotting panel supp_4a...")
    panel_supp4a_scatter(LMI_DATA_CSV, filename='supp_4a')

    print("\nPlotting panel supp_4b...")
    panel_supp4b_partial_corr(LMI_DATA_CSV, filename='supp_4b')

    print(f"\nDone. Figures saved to: {OUTPUT_DIR}")
