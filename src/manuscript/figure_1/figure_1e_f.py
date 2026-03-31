"""
Figure 1e-f: Average behavioral performance across imaging mice

This script generates Panels e and f for Figure 1:
- Panel e: Performance across 5 days for all imaging mice (line plot)
- Panel f: Performance comparison for days 0, +1, +2 (bar plot with statistics)
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu

sys.path.append('/home/aprenard/repos/fast-learning')
import src.utils.utils_io as io
from src.utils.utils_plot import behavior_palette


OUTPUT_DIR = os.path.join(io.manuscript_output_dir, 'figure_1', 'output')


# ============================================================================
# Panel e: Performance across training days
# ============================================================================

def panel_e_performance_across_days(
    table_path=os.path.join(io.processed_dir, 'behavior', 'behavior_imagingmice_table_5days_cut.csv'),
    save_path=OUTPUT_DIR,
    save_format='svg',
    dpi=300
):
    """
    Generate Figure 1 Panel e: Performance across training days.

    Shows average lick probability across 5 days of training for all imaging mice,
    separated by reward group (R+ and R-) and trial type (auditory, whisker, no-stim).

    Args:
        table_path: Path to CSV file containing behavioral data
        save_path: Directory to save output figure and data
        save_format: Figure format ('svg', 'png', 'pdf')
        dpi: Resolution for saved figure
    """

    # Load behavioral data
    table_path = io.adjust_path_to_host(table_path)
    table = pd.read_csv(table_path)

    # Remove spurious whisker trials from mapping sessions (days -2 and -1)
    table.loc[table.day.isin([-2, -1]), 'outcome_w'] = np.nan
    table.loc[table.day.isin([-2, -1]), 'hr_w'] = np.nan

    # Average performance per session
    table_agg = table.groupby(
        ['mouse_id', 'session_id', 'reward_group', 'day'],
        as_index=False
    )[['outcome_c', 'outcome_a', 'outcome_w']].agg(np.mean)

    # Convert performance to percentage
    table_agg[['outcome_c', 'outcome_a', 'outcome_w']] = \
        table_agg[['outcome_c', 'outcome_a', 'outcome_w']] * 100

    # Convert day to string for categorical plotting
    table_agg['day'] = table_agg['day'].astype(str)

    # Set plotting theme
    sns.set_theme(
        context='paper',
        style='ticks',
        palette='deep',
        font='sans-serif',
        font_scale=1,
        rc={
            'xtick.major.width': 1,
            'ytick.major.width': 1,
            'pdf.fonttype': 42,
            'ps.fonttype': 42,
            'svg.fonttype': 'none'
        }
    )

    # Create figure
    fig = plt.figure(figsize=(6, 6))
    ax = plt.gca()

    # Plot individual mouse traces (thin lines)
    sns.lineplot(
        data=table_agg, x='day', y='outcome_c', units='mouse_id',
        estimator=None, hue="reward_group", hue_order=['R-', 'R+'],
        palette=behavior_palette[4:6], alpha=0.4, legend=False,
        ax=ax, marker=None, linewidth=1
    )
    sns.lineplot(
        data=table_agg, x='day', y='outcome_a', units='mouse_id',
        estimator=None, hue="reward_group", hue_order=['R-', 'R+'],
        palette=behavior_palette[0:2], alpha=0.4, legend=False,
        ax=ax, marker=None, linewidth=1
    )
    sns.lineplot(
        data=table_agg, x='day', y='outcome_w', units='mouse_id',
        estimator=None, hue="reward_group", hue_order=['R-', 'R+'],
        palette=behavior_palette[2:4], alpha=0.4, legend=False,
        ax=ax, marker=None, linewidth=1
    )

    # Plot group averages (thick lines with markers)
    sns.pointplot(
        data=table_agg, x='day', y='outcome_c', estimator=np.mean,
        palette=behavior_palette[4:6], hue="reward_group",
        hue_order=['R-', 'R+'], alpha=1, legend=True,
        ax=ax, linewidth=2
    )
    sns.pointplot(
        data=table_agg, x='day', y='outcome_a', estimator=np.mean,
        palette=behavior_palette[0:2], hue="reward_group",
        hue_order=['R-', 'R+'], alpha=1, legend=True,
        ax=ax, linewidth=2
    )
    sns.pointplot(
        data=table_agg, x='day', y='outcome_w', estimator=np.mean,
        palette=behavior_palette[2:4], hue="reward_group",
        hue_order=['R-', 'R+'], alpha=1, legend=True,
        ax=ax, linewidth=2
    )

    # Formatting
    plt.xlabel('Training days')
    plt.ylabel('Lick probability (%)')
    plt.legend()
    sns.despine(trim=True)

    # Ensure tick thickness is set for SVG output
    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_tick_params(width=1)

    # Save figure and data
    os.makedirs(save_path, exist_ok=True)

    output_file = os.path.join(save_path, f'figure_1e.{save_format}')
    plt.savefig(output_file, format=save_format, dpi=dpi, bbox_inches='tight')
    # plt.close()

    # Save data
    data_file = os.path.join(save_path, 'figure_1e_data.csv')
    table_agg.to_csv(data_file, index=False)

    print(f"Figure 1e saved to: {output_file}")
    print(f"Figure 1e data saved to: {data_file}")


# ============================================================================
# Panel f: Performance comparison for days 0, +1, +2
# ============================================================================

def panel_f_performance_barplot(
    table_path=os.path.join(io.processed_dir, 'behavior', 'behavior_imagingmice_table_5days_cut.csv'),
    days_of_interest=[0, 1, 2],
    save_path=OUTPUT_DIR,
    save_format='svg',
    dpi=300
):
    """
    Generate Figure 1 Panel f: Performance comparison with statistics.

    Shows whisker trial performance for days 0, +1, +2 with bar plots
    and statistical comparisons between reward groups.

    Args:
        table_path: Path to CSV file containing behavioral data
        days_of_interest: List of day indices to compare
        save_path: Directory to save output figure and data
        save_format: Figure format ('svg', 'png', 'pdf')
        dpi: Resolution for saved figure
    """

    # Load behavioral data
    table_path = io.adjust_path_to_host(table_path)
    table = pd.read_csv(table_path)

    # Remove spurious whisker trials from mapping sessions
    table.loc[table.day.isin([-2, -1]), 'outcome_w'] = np.nan
    table.loc[table.day.isin([-2, -1]), 'hr_w'] = np.nan

    # Average performance per session
    table_agg = table.groupby(
        ['mouse_id', 'session_id', 'reward_group', 'day'],
        as_index=False
    )[['outcome_c', 'outcome_a', 'outcome_w']].agg(np.mean)

    # Convert performance to percentage
    table_agg[['outcome_c', 'outcome_a', 'outcome_w']] = \
        table_agg[['outcome_c', 'outcome_a', 'outcome_w']] * 100

    # Convert day to string for categorical plotting
    table_agg['day'] = table_agg['day'].astype(str)

    # Select data for days of interest
    day_data = table_agg[table_agg['day'].isin([str(d) for d in days_of_interest])]
    avg_performance = day_data.groupby(
        ['day', 'mouse_id', 'reward_group']
    )['outcome_w'].mean().reset_index()

    # Set plotting theme
    sns.set_theme(
        context='paper',
        style='ticks',
        palette='deep',
        font='sans-serif',
        font_scale=1,
        rc={
            'pdf.fonttype': 42,
            'ps.fonttype': 42,
            'svg.fonttype': 'none'
        }
    )

    # Create figure
    plt.figure(figsize=(8, 6))

    # Bar plot
    sns.barplot(
        data=avg_performance,
        x='day',
        y='outcome_w',
        hue='reward_group',
        palette=behavior_palette[2:4][::-1],
        width=0.3,
        dodge=True
    )

    # Swarm plot for individual mice
    sns.swarmplot(
        data=avg_performance,
        x='day',
        y='outcome_w',
        hue='reward_group',
        dodge=True,
        color='grey',
        alpha=0.6,
    )

    # Formatting
    plt.xlabel('Day')
    plt.ylabel('Lick probability (%)')
    plt.ylim([0, 100])
    plt.legend(title='Reward group')
    sns.despine(trim=True)

    # Statistical testing: Mann-Whitney U test for each day
    stats = []
    for day in days_of_interest:
        df_day = avg_performance[avg_performance['day'] == str(day)]
        group_R_plus = df_day[df_day['reward_group'] == 'R+']['outcome_w']
        group_R_minus = df_day[df_day['reward_group'] == 'R-']['outcome_w']

        stat, p_value = mannwhitneyu(
            group_R_plus, group_R_minus,
            alternative='two-sided'
        )
        stats.append({'day': day, 'statistic': stat, 'p_value': p_value})

        # Add significance stars to the plot
        ax = plt.gca()
        xpos = days_of_interest.index(day)
        ypos = 95

        if p_value < 0.001:
            plt.text(xpos, ypos, '***', ha='center', va='bottom',
                    color='black', fontsize=14)
        elif p_value < 0.01:
            plt.text(xpos, ypos, '**', ha='center', va='bottom',
                    color='black', fontsize=14)
        elif p_value < 0.05:
            plt.text(xpos, ypos, '*', ha='center', va='bottom',
                    color='black', fontsize=14)

    # Save figure and data
    os.makedirs(save_path, exist_ok=True)

    output_file = os.path.join(save_path, f'figure_1f.{save_format}')
    plt.savefig(output_file, format=save_format, dpi=dpi, bbox_inches='tight')
    # plt.close()

    # Save data and statistics
    data_file = os.path.join(save_path, 'figure_1f_data.csv')
    stats_file = os.path.join(save_path, 'figure_1f_stats.csv')
    avg_performance.to_csv(data_file, index=False)
    pd.DataFrame(stats).to_csv(stats_file, index=False)

    print(f"Figure 1f saved to: {output_file}")
    print(f"Figure 1f data saved to: {data_file}")
    print(f"Figure 1f statistics saved to: {stats_file}")


# ============================================================================
# Main execution
# ============================================================================

if __name__ == '__main__':
    # Generate both panels
    panel_e_performance_across_days()
    panel_f_performance_barplot()
