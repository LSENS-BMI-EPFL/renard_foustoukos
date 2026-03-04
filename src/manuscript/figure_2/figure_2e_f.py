"""
Figure 2e-f: Optogenetic inactivation during learning

This script generates Panels e and f for Figure 2:
- Panel e: Performance across optogenetic inactivation days (wS1 and fpS1)
- Panel f: Performance comparison for days 0 and +1 with statistics
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
from src.utils.utils_plot import stim_palette


OUTPUT_DIR = '/Volumes/Petersen-Lab/analysis/Anthony_Renard/manuscripts/outputs/figure_2/output'


# ============================================================================
# Panel e: Optogenetic inactivation across days
# ============================================================================

def panel_e_opto_timecourse(
    table_path='//sv-nas1.rcp.epfl.ch/Petersen-Lab/analysis/Anthony_Renard/data_processed/behavior/behavior_opto_learning.csv',
    save_path=OUTPUT_DIR,
    save_format='svg',
    dpi=300
):
    """
    Generate Figure 2 Panel e: Optogenetic inactivation timecourse.

    Shows performance across pre-injection, optogenetic inactivation, and recovery days
    for wS1 and fpS1 inactivation mice.

    Args:
        table_path: Path to CSV file containing optogenetic behavioral data
        save_path: Directory to save output figure and data
        save_format: Figure format ('svg', 'png', 'pdf')
        dpi: Resolution for saved figure
    """

    # Load behavioral data
    table_path = io.adjust_path_to_host(table_path)
    table = pd.read_csv(table_path)

    # Get mouse groups from database
    db_path = io.db_path
    nwb_dir = io.nwb_dir

    fpS1_mice = io.select_mice_from_db(
        db_path, nwb_dir, experimenters=None,
        exclude_cols=['exclude', 'opto_exclude'],
        optogenetic='yes',
        opto_inactivation_type='learning',
        opto_area='fpS1',
    )

    wS1_mice = io.select_mice_from_db(
        db_path, nwb_dir, experimenters=None,
        exclude_cols=['exclude', 'opto_exclude'],
        optogenetic='yes',
        opto_inactivation_type='learning',
        opto_area='wS1',
    )

    # Add area labels to table
    table.loc[table.mouse_id.isin(fpS1_mice), 'area'] = 'fpS1'
    table.loc[table.mouse_id.isin(wS1_mice), 'area'] = 'wS1'

    # Get opto_day info from database
    _, _, _, db = io.select_sessions_from_db(
        db_path, nwb_dir, experimenters=None,
        exclude_cols=['exclude', 'opto_exclude'],
        opto_inactivation_type=['learning'],
        opto_day=["pre_-2", "pre_-1", "opto", "recovery_1"],
    )

    table = pd.merge(
        table,
        db[['mouse_id', 'session_id', 'opto_day']],
        on=['mouse_id', 'session_id'],
        how='left'
    )

    # Define inactivation day labels
    inactivation_labels = ['pre_-2', 'pre_-1', 'opto', 'recovery_1']

    # Aggregate performance by session
    data = table.groupby(
        ['mouse_id', 'session_id', 'opto_day', 'area'],
        as_index=False
    )[['outcome_c', 'outcome_a', 'outcome_w']].agg('mean')

    # Order data by inactivation labels
    data['opto_day'] = pd.Categorical(
        data['opto_day'],
        categories=inactivation_labels,
        ordered=True
    )
    data = data.sort_values(by=['mouse_id', 'opto_day'])

    # Convert performance to percentage
    data['outcome_c'] = data['outcome_c'] * 100
    data['outcome_a'] = data['outcome_a'] * 100
    data['outcome_w'] = data['outcome_w'] * 100

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

    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, sharey=True, figsize=(12, 5))

    # ========================================================================
    # Left panel: wS1 inactivation
    # ========================================================================
    ax = axes[0]

    # Plot individual mouse traces
    for imouse in wS1_mice:
        sns.lineplot(
            data=data.loc[data.mouse_id == imouse],
            x='opto_day', y='outcome_c', estimator=np.mean,
            color=stim_palette[2], alpha=0.6, legend=False,
            ax=ax, marker=None, err_style='bars', linewidth=1
        )
        sns.lineplot(
            data=data.loc[data.mouse_id == imouse],
            x='opto_day', y='outcome_a', estimator=np.mean,
            color=stim_palette[0], alpha=0.6, legend=False,
            ax=ax, marker=None, err_style='bars', linewidth=1
        )
        sns.lineplot(
            data=data.loc[data.mouse_id == imouse],
            x='opto_day', y='outcome_w', estimator=np.mean,
            color=stim_palette[1], alpha=0.6, legend=False,
            ax=ax, marker=None, err_style='bars', linewidth=1
        )

    # Plot group averages
    sns.pointplot(
        data=data.loc[data.mouse_id.isin(wS1_mice)],
        x='opto_day', y='outcome_c', order=inactivation_labels,
        color=stim_palette[2], ax=ax, linewidth=2
    )
    sns.pointplot(
        data=data.loc[data.mouse_id.isin(wS1_mice)],
        x='opto_day', y='outcome_a', order=inactivation_labels,
        color=stim_palette[0], ax=ax, linewidth=2
    )
    sns.pointplot(
        data=data.loc[data.mouse_id.isin(wS1_mice)],
        x='opto_day', y='outcome_w', order=inactivation_labels,
        color=stim_palette[1], ax=ax, linewidth=2
    )

    # Add dots for individual points
    sns.stripplot(
        data=data.loc[data.mouse_id.isin(wS1_mice)],
        x='opto_day', y='outcome_c', order=inactivation_labels,
        color=stim_palette[2], ax=ax, jitter=False,
        dodge=True, alpha=0.5, size=4
    )
    sns.stripplot(
        data=data.loc[data.mouse_id.isin(wS1_mice)],
        x='opto_day', y='outcome_a', order=inactivation_labels,
        color=stim_palette[0], ax=ax, jitter=False,
        dodge=True, alpha=0.5, size=4
    )
    sns.stripplot(
        data=data.loc[data.mouse_id.isin(wS1_mice)],
        x='opto_day', y='outcome_w', order=inactivation_labels,
        color=stim_palette[1], ax=ax, jitter=False,
        dodge=True, alpha=0.5, size=4
    )

    ax.set_title('wS1')

    # ========================================================================
    # Right panel: fpS1 inactivation
    # ========================================================================
    ax = axes[1]

    # Plot individual mouse traces
    for imouse in fpS1_mice:
        sns.lineplot(
            data=data.loc[data.mouse_id == imouse],
            x='opto_day', y='outcome_c', estimator=np.mean,
            color=stim_palette[2], alpha=0.6, legend=False,
            ax=ax, marker=None, err_style='bars', linewidth=1
        )
        sns.lineplot(
            data=data.loc[data.mouse_id == imouse],
            x='opto_day', y='outcome_a', estimator=np.mean,
            color=stim_palette[0], alpha=0.6, legend=False,
            ax=ax, marker=None, err_style='bars', linewidth=1
        )
        sns.lineplot(
            data=data.loc[data.mouse_id == imouse],
            x='opto_day', y='outcome_w', estimator=np.mean,
            color=stim_palette[1], alpha=0.6, legend=False,
            ax=ax, marker=None, err_style='bars', linewidth=1
        )

    # Plot group averages
    sns.pointplot(
        data=data.loc[data.mouse_id.isin(fpS1_mice)],
        x='opto_day', y='outcome_c', order=inactivation_labels,
        color=stim_palette[2], ax=ax, linewidth=2
    )
    sns.pointplot(
        data=data.loc[data.mouse_id.isin(fpS1_mice)],
        x='opto_day', y='outcome_a', order=inactivation_labels,
        color=stim_palette[0], ax=ax, linewidth=2
    )
    sns.pointplot(
        data=data.loc[data.mouse_id.isin(fpS1_mice)],
        x='opto_day', y='outcome_w', order=inactivation_labels,
        color=stim_palette[1], ax=ax, linewidth=2
    )

    ax.set_title('fpS1')

    # Format both axes
    for ax in axes:
        ax.set_yticks([0, 20, 40, 60, 80, 100])
        ax.set_xticklabels(['-2', '-1', '0', '+1'])
        ax.set_xlabel('Optogenetic inactivation during learning')
        ax.set_ylabel('Lick probability (%)')

    sns.despine(trim=True)

    # Save figure and data
    os.makedirs(save_path, exist_ok=True)

    output_file = os.path.join(save_path, f'figure_2e.{save_format}')
    plt.savefig(output_file, format=save_format, dpi=dpi, bbox_inches='tight')
    plt.close()

    # Save data
    data_file = os.path.join(save_path, 'figure_2e_data.csv')
    data.to_csv(data_file, index=False)

    print(f"Figure 2e saved to: {output_file}")
    print(f"Figure 2e data saved to: {data_file}")

    return data


# ============================================================================
# Panel f: Bar plot quantification for days 0 and +1
# ============================================================================

def panel_f_opto_barplot(
    data=None,
    table_path='//sv-nas1.rcp.epfl.ch/Petersen-Lab/analysis/Anthony_Renard/data_processed/behavior/behavior_opto_learning.csv',
    days_of_interest=['opto', 'recovery_1'],
    day_labels=['D0', 'D+1'],
    save_path=OUTPUT_DIR,
    save_format='svg',
    dpi=300
):
    """
    Generate Figure 2 Panel f: Bar plot comparison of wS1 vs fpS1.

    Shows whisker trial performance for days 0 and +1 with bar plots
    and statistical comparisons between wS1 and fpS1 inactivation.

    Args:
        data: Pre-processed data from panel_e (optional, will load if None)
        table_path: Path to CSV file containing opto data (if data is None)
        days_of_interest: List of opto_day labels to compare
        day_labels: Corresponding display labels for days
        save_path: Directory to save output figure and data
        save_format: Figure format ('svg', 'png', 'pdf')
        dpi: Resolution for saved figure
    """

    # Load data if not provided
    if data is None:
        table_path = io.adjust_path_to_host(table_path)
        table = pd.read_csv(table_path)

        # Get mouse groups and process data (same as panel_e)
        db_path = io.db_path
        nwb_dir = io.nwb_dir

        fpS1_mice = io.select_mice_from_db(
            db_path, nwb_dir, experimenters=None,
            exclude_cols=['exclude', 'opto_exclude'],
            optogenetic='yes',
            opto_inactivation_type='learning',
            opto_area='fpS1',
        )

        wS1_mice = io.select_mice_from_db(
            db_path, nwb_dir, experimenters=None,
            exclude_cols=['exclude', 'opto_exclude'],
            optogenetic='yes',
            opto_inactivation_type='learning',
            opto_area='wS1',
        )

        table.loc[table.mouse_id.isin(fpS1_mice), 'area'] = 'fpS1'
        table.loc[table.mouse_id.isin(wS1_mice), 'area'] = 'wS1'

        _, _, _, db = io.select_sessions_from_db(
            db_path, nwb_dir, experimenters=None,
            exclude_cols=['exclude', 'opto_exclude'],
            opto_inactivation_type=['learning'],
            opto_day=["pre_-2", "pre_-1", "opto", "recovery_1"],
        )

        table = pd.merge(
            table,
            db[['mouse_id', 'session_id', 'opto_day']],
            on=['mouse_id', 'session_id'],
            how='left'
        )

        inactivation_labels = ['pre_-2', 'pre_-1', 'opto', 'recovery_1']

        data = table.groupby(
            ['mouse_id', 'session_id', 'opto_day', 'area'],
            as_index=False
        )[['outcome_c', 'outcome_a', 'outcome_w']].agg('mean')

        data['opto_day'] = pd.Categorical(
            data['opto_day'],
            categories=inactivation_labels,
            ordered=True
        )
        data = data.sort_values(by=['mouse_id', 'opto_day'])

        data['outcome_c'] = data['outcome_c'] * 100
        data['outcome_a'] = data['outcome_a'] * 100
        data['outcome_w'] = data['outcome_w'] * 100

    # Filter for days of interest
    day_data = data[data['opto_day'].isin(days_of_interest)].copy()
    day_data['day_label'] = day_data['opto_day'].map(
        dict(zip(days_of_interest, day_labels))
    )

    # Set plotting theme
    sns.set_theme(
        context='paper',
        style='ticks',
        palette='deep',
        font='sans-serif',
        font_scale=1
    )

    # Create figure
    plt.figure(figsize=(8, 6))

    # Bar plot
    sns.barplot(
        data=day_data,
        x='day_label',
        y='outcome_w',
        hue='area',
        palette=[stim_palette[1]],
        width=0.3,
        dodge=True,
        order=day_labels,
        hue_order=['wS1', 'fpS1']
    )

    # Swarm plot for individual mice
    sns.swarmplot(
        data=day_data,
        x='day_label',
        y='outcome_w',
        hue='area',
        dodge=True,
        color='black',
        alpha=0.6,
        order=day_labels,
        hue_order=['wS1', 'fpS1']
    )

    # Formatting
    plt.xlabel('Day')
    plt.ylabel('Whisker Performance (%)')
    plt.ylim([0, 100])
    plt.legend(title='Area')
    sns.despine()

    # Statistical testing: Mann-Whitney U test for each day
    stats = []
    for day, label in zip(days_of_interest, day_labels):
        df_day = day_data[day_data['opto_day'] == day]
        group_wS1 = df_day[df_day['area'] == 'wS1']['outcome_w']
        group_fpS1 = df_day[df_day['area'] == 'fpS1']['outcome_w']

        stat, p_value = mannwhitneyu(
            group_wS1, group_fpS1,
            alternative='two-sided'
        )
        stats.append({'day': label, 'statistic': stat, 'p_value': p_value})

        # Add significance stars to the plot
        ax = plt.gca()
        xpos = day_labels.index(label)
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

        # Add p-value text
        plt.text(xpos, 90, f'p={p_value:.3g}', ha='center', va='bottom',
                color='black', fontsize=10)

    # Save figure and data
    os.makedirs(save_path, exist_ok=True)

    output_file = os.path.join(save_path, f'figure_2f.{save_format}')
    plt.savefig(output_file, format=save_format, dpi=dpi, bbox_inches='tight')
    plt.close()

    # Save data and statistics
    data_file = os.path.join(save_path, 'figure_2f_data.csv')
    stats_file = os.path.join(save_path, 'figure_2f_stats.csv')
    day_data.to_csv(data_file, index=False)
    pd.DataFrame(stats).to_csv(stats_file, index=False)

    print(f"Figure 2f saved to: {output_file}")
    print(f"Figure 2f data saved to: {data_file}")
    print(f"Figure 2f statistics saved to: {stats_file}")


# ============================================================================
# Main execution
# ============================================================================

if __name__ == '__main__':
    # Generate panel e and get data
    data = panel_e_opto_timecourse()

    # Generate panel f using the same data
    panel_f_opto_barplot(data=data)
