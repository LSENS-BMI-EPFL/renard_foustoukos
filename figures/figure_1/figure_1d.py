"""
Figure 1d: Day 0 performance across whisker trials

This script generates Panel d for Figure 1, showing performance during
the first learning session (day 0) aligned to whisker trial number.
Includes raw trial outcomes and fitted learning curves with statistical testing.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors
import seaborn as sns
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

from fast_learning import paths
from fast_learning.plotting import reward_palette, save_figure


OUTPUT_DIR = os.path.join(paths.manuscript_output_dir, 'figure_1', 'output')


# ============================================================================
# Main Panel Generation
# ============================================================================


def generate_panel(
    table_path=os.path.join(
        paths.processed_dir, 'behavior', 'behavior_imagingmice_table_5days_cut_with_learning_curves.csv'
    ),
    n_trials=120,
    save_path=OUTPUT_DIR,
    save_format='svg',
    dpi=300,
):
    """
    Generate Figure 1 Panel d: Day 0 performance across whisker trials.

    Shows two subplots comparing R+ vs R- reward groups:
    1. Raw trial outcomes (outcome_w)
    2. Fitted learning curves (learning_curve_w)

    Both panels include statistical testing (Mann-Whitney U with FDR correction)
    displayed as grayscale rectangles at the top of each plot.

    Args:
        table_path: Path to CSV file containing behavioral data with learning curves
        n_trials: Maximum number of whisker trials to include
        save_path: Directory to save output figure and data
        save_format: Figure format ('svg', 'png', 'pdf')
        dpi: Resolution for saved figure
    """

    # Load behavioral data
    table_path = paths.adjust_path_to_host(table_path)
    table = pd.read_csv(table_path)

    # Filter for whisker trials on day 0
    df = table.loc[(table.whisker_stim == 1) & (table.day == 0)]
    df = df.loc[df.trial_w <= n_trials]

    # Prepare data for plotting
    df_single = df.copy()
    df_learning = df.copy()

    # Set plotting theme
    sns.set_theme(context='paper', style='ticks', palette='deep', font='sans-serif', font_scale=1)

    # Create two-panel figure
    fig, axes = plt.subplots(1, 2, sharey=True, figsize=(12, 5))

    # p > 0.05 (not significant) is left pure white. p <= 0.05 uses a log10
    # gradient from a visible grey right at the p=0.05 boundary down to
    # black as p keeps shrinking. A linear scale collapses almost all
    # significant p-values (which routinely span many orders of magnitude,
    # e.g. 0.04 down to 1e-8) into indistinguishable black; log gives each
    # order of magnitude equal visual weight instead. PVALUE_FLOOR is the
    # p-value at which the gradient bottoms out at pure black -- tune it to
    # the actual range of p-values in the data (anything smaller renders
    # identically black).
    PVALUE_FLOOR = 1e-3
    NONSIG_GREY = 0.8  # grey at p=0.05
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        'pval_cmap', [(0.0, 'black'), (1.0, (NONSIG_GREY,) * 3)]
    )
    cmap.set_over('white')  # p > vmax (0.05) -- not part of the gradient
    norm = matplotlib.colors.LogNorm(vmin=PVALUE_FLOOR, vmax=0.05)

    # ========================================================================
    # Panel 1: Raw trial outcomes
    # ========================================================================
    ax = axes[0]
    sns.lineplot(
        data=df_single,
        x='trial_w',
        y='outcome_w',
        palette=reward_palette[::-1],
        hue='reward_group',
        errorbar='ci',
        err_style='band',
        ax=ax,
        legend=False,
    )

    # Statistical testing for each trial
    p_values_single = []
    for trial_w in df_single['trial_w'].unique():
        group_R_plus = df_single[(df_single['trial_w'] == trial_w) & (df_single['reward_group'] == 'R+')][
            'outcome_w'
        ]
        group_R_minus = df_single[(df_single['trial_w'] == trial_w) & (df_single['reward_group'] == 'R-')][
            'outcome_w'
        ]

        if len(group_R_plus) > 0 and len(group_R_minus) > 0:
            stat, p_value = mannwhitneyu(group_R_plus, group_R_minus, alternative='two-sided')
            p_values_single.append((trial_w, p_value))

    # FDR correction
    trials_single, raw_pvals_single = zip(*p_values_single)
    _, corrected_pvals_single, _, _ = multipletests(raw_pvals_single, alpha=0.05, method='fdr_bh')
    p_values_single = list(zip(trials_single, corrected_pvals_single))

    # Plot p-value rectangles
    for trial, p_value in p_values_single:
        color = cmap(norm(max(p_value, PVALUE_FLOOR)))
        ax.add_patch(plt.Rectangle((trial - 0.4, 0.95), 0.8, 0.03, color=color, edgecolor='none'))

    ax.set_title('Raw trial outcomes')
    ax.set_xlabel('Whisker trial')
    ax.set_ylabel('Hit rate')
    ax.set_ylim([-0.1, 1])

    # ========================================================================
    # Panel 2: Fitted learning curves
    # ========================================================================
    ax = axes[1]
    sns.lineplot(
        data=df_learning,
        x='trial_w',
        y='learning_curve_w',
        palette=reward_palette[::-1],
        hue='reward_group',
        errorbar='ci',
        err_style='band',
        ax=ax,
    )

    # Statistical testing for each trial
    p_values_learning = []
    for trial_w in df_learning['trial_w'].unique():
        group_R_plus = df_learning[
            (df_learning['trial_w'] == trial_w) & (df_learning['reward_group'] == 'R+')
        ]['learning_curve_w']
        group_R_minus = df_learning[
            (df_learning['trial_w'] == trial_w) & (df_learning['reward_group'] == 'R-')
        ]['learning_curve_w']

        if len(group_R_plus) > 0 and len(group_R_minus) > 0:
            stat, p_value = mannwhitneyu(group_R_plus, group_R_minus, alternative='two-sided')
            p_values_learning.append((trial_w, p_value))

    # FDR correction
    trials_learning, raw_pvals_learning = zip(*p_values_learning)
    _, corrected_pvals_learning, _, _ = multipletests(raw_pvals_learning, alpha=0.05, method='fdr_bh')
    p_values_learning = list(zip(trials_learning, corrected_pvals_learning))

    # Plot p-value rectangles
    for trial, p_value in p_values_learning:
        color = cmap(norm(max(p_value, PVALUE_FLOOR)))
        ax.add_patch(plt.Rectangle((trial - 0.4, 0.95), 0.8, 0.03, color=color, edgecolor='none'))

    ax.set_title('Fitted learning curves')
    ax.set_xlabel('Whisker trial')
    ax.set_ylim([-0.1, 1])
    ax.legend(frameon=False, title='Reward group')

    # Shared colorbar for the p-value color code (same cmap/norm as the
    # significance rectangles above; p > 0.05 renders white in the actual
    # rectangles via cmap.set_over(), but isn't depicted on the bar itself
    # -- no extend cap). Reversed so the most significant (blackest) end
    # is on top and the p=0.05 boundary is at the bottom.
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, orientation='vertical', shrink=0.6, pad=0.02, aspect=15)
    tick_vals = [PVALUE_FLOOR, 1e-2, 0.05]
    cbar.set_ticks(tick_vals)
    cbar.set_ticklabels(['≤ {:.0e}'.format(PVALUE_FLOOR), '0.01', '0.05'])
    cbar.set_label('p-value (FDR-corrected)', fontsize=9)
    cbar.ax.invert_yaxis()

    sns.despine()
    plt.tight_layout()

    # Save figure and data
    os.makedirs(save_path, exist_ok=True)

    output_file = os.path.join(save_path, f'figure_1d.{save_format}')
    save_figure(plt.gcf(), output_file)
    # plt.close()

    # Save data and statistics
    data_file_single = os.path.join(save_path, 'figure_1d_raw_data.csv')
    stats_file_single = os.path.join(save_path, 'figure_1d_raw_stats.csv')
    df_single.to_csv(data_file_single, index=False)
    pd.DataFrame(p_values_single, columns=['trial_w', 'p_value']).to_csv(stats_file_single, index=False)

    data_file_learning = os.path.join(save_path, 'figure_1d_learning_data.csv')
    stats_file_learning = os.path.join(save_path, 'figure_1d_learning_stats.csv')
    df_learning.to_csv(data_file_learning, index=False)
    pd.DataFrame(p_values_learning, columns=['trial_w', 'p_value']).to_csv(stats_file_learning, index=False)

    print(f"Figure 1d saved to: {output_file}")
    print(f"Figure 1d data saved to: {data_file_single} and {data_file_learning}")
    print(f"Figure 1d statistics saved to: {stats_file_single} and {stats_file_learning}")


# ============================================================================
# Main execution
# ============================================================================

if __name__ == '__main__':
    generate_panel()
