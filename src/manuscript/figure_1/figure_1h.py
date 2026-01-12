"""
Figure 1h: Day 0 performance across whisker trials

This script generates Panel h for Figure 1, showing performance during
the first learning session (day 0) aligned to whisker trial number.
Includes raw trial outcomes and fitted learning curves with statistical testing.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors
import seaborn as sns
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

sys.path.append('/home/aprenard/repos/fast-learning')
import src.utils.utils_io as io
from src.utils.utils_plot import reward_palette


# ============================================================================
# Main Panel Generation
# ============================================================================

def generate_panel(
    table_path='//sv-nas1.rcp.epfl.ch/Petersen-Lab/analysis/Anthony_Renard/data_processed/behavior/behavior_imagingmice_table_5days_cut_with_learning_curves.csv',
    n_trials=120,
    save_path='/mnt/lsens-analysis/Anthony_Renard/analysis_output/fast-learning/behavior',
    save_format='svg',
    dpi=300
):
    """
    Generate Figure 1 Panel h: Day 0 performance across whisker trials.

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
    table_path = io.adjust_path_to_host(table_path)
    table = pd.read_csv(table_path)

    # Filter for whisker trials on day 0
    df = table.loc[(table.whisker_stim == 1) & (table.day == 0)]
    df = df.loc[df.trial_w <= n_trials]

    # Prepare data for plotting
    df_single = df.copy()
    df_learning = df.copy()

    # Set plotting theme
    sns.set_theme(
        context='paper',
        style='ticks',
        palette='deep',
        font='sans-serif',
        font_scale=1
    )

    # Create two-panel figure
    fig, axes = plt.subplots(1, 2, sharey=True, figsize=(12, 5))

    # Create colormap for p-value visualization
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        'pval_cmap', ['black', 'white']
    )
    norm = matplotlib.colors.Normalize(vmin=0, vmax=0.05)

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
        legend=False
    )

    # Statistical testing for each trial
    p_values_single = []
    for trial_w in df_single['trial_w'].unique():
        group_R_plus = df_single[
            (df_single['trial_w'] == trial_w) &
            (df_single['reward_group'] == 'R+')
        ]['outcome_w']
        group_R_minus = df_single[
            (df_single['trial_w'] == trial_w) &
            (df_single['reward_group'] == 'R-')
        ]['outcome_w']

        if len(group_R_plus) > 0 and len(group_R_minus) > 0:
            stat, p_value = mannwhitneyu(
                group_R_plus, group_R_minus,
                alternative='two-sided'
            )
            p_values_single.append((trial_w, p_value))

    # FDR correction
    trials_single, raw_pvals_single = zip(*p_values_single)
    _, corrected_pvals_single, _, _ = multipletests(
        raw_pvals_single, alpha=0.05, method='fdr_bh'
    )
    p_values_single = list(zip(trials_single, corrected_pvals_single))

    # Plot p-value rectangles
    for trial, p_value in p_values_single:
        color = cmap(norm(min(p_value, 0.05)))
        ax.add_patch(plt.Rectangle(
            (trial - 0.4, 0.95), 0.8, 0.03,
            color=color, edgecolor='none'
        ))

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
        ax=ax
    )

    # Statistical testing for each trial
    p_values_learning = []
    for trial_w in df_learning['trial_w'].unique():
        group_R_plus = df_learning[
            (df_learning['trial_w'] == trial_w) &
            (df_learning['reward_group'] == 'R+')
        ]['learning_curve_w']
        group_R_minus = df_learning[
            (df_learning['trial_w'] == trial_w) &
            (df_learning['reward_group'] == 'R-')
        ]['learning_curve_w']

        if len(group_R_plus) > 0 and len(group_R_minus) > 0:
            stat, p_value = mannwhitneyu(
                group_R_plus, group_R_minus,
                alternative='two-sided'
            )
            p_values_learning.append((trial_w, p_value))

    # FDR correction
    trials_learning, raw_pvals_learning = zip(*p_values_learning)
    _, corrected_pvals_learning, _, _ = multipletests(
        raw_pvals_learning, alpha=0.05, method='fdr_bh'
    )
    p_values_learning = list(zip(trials_learning, corrected_pvals_learning))

    # Plot p-value rectangles
    for trial, p_value in p_values_learning:
        color = cmap(norm(min(p_value, 0.05)))
        ax.add_patch(plt.Rectangle(
            (trial - 0.4, 0.95), 0.8, 0.03,
            color=color, edgecolor='none'
        ))

    ax.set_title('Fitted learning curves')
    ax.set_xlabel('Whisker trial')
    ax.set_ylim([-0.1, 1])
    ax.legend(frameon=False, title='Reward group')

    sns.despine()
    plt.tight_layout()

    # Save figure and data
    save_path = io.adjust_path_to_host(save_path)
    os.makedirs(save_path, exist_ok=True)

    output_file = os.path.join(save_path, f'figure_1h.{save_format}')
    plt.savefig(output_file, format=save_format, dpi=dpi, bbox_inches='tight')
    # plt.close()

    # Save data and statistics
    data_file_single = os.path.join(save_path, 'figure_1h_raw_data.csv')
    stats_file_single = os.path.join(save_path, 'figure_1h_raw_stats.csv')
    df_single.to_csv(data_file_single, index=False)
    pd.DataFrame(
        p_values_single, columns=['trial_w', 'p_value']
    ).to_csv(stats_file_single, index=False)

    data_file_learning = os.path.join(save_path, 'figure_1h_learning_data.csv')
    stats_file_learning = os.path.join(save_path, 'figure_1h_learning_stats.csv')
    df_learning.to_csv(data_file_learning, index=False)
    pd.DataFrame(
        p_values_learning, columns=['trial_w', 'p_value']
    ).to_csv(stats_file_learning, index=False)

    print(f"Figure 1h saved to: {output_file}")
    print(f"Figure 1h data saved to: {data_file_single} and {data_file_learning}")
    print(f"Figure 1h statistics saved to: {stats_file_single} and {stats_file_learning}")


# ============================================================================
# Main execution
# ============================================================================

if __name__ == '__main__':
    generate_panel()
