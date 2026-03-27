"""
Figure 3d-e: Pre vs post learning response comparison (all cells)

Panel d: R+ mice — PSTH and response amplitude before/after learning (all cells)
Panel e: R- mice — PSTH and response amplitude before/after learning (all cells)

Variance is across mice: cells are averaged per mouse before group statistics.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wilcoxon

sys.path.append('/home/aprenard/repos/fast-learning')
import src.utils.utils_imaging as utils_imaging
import src.utils.utils_io as io

OUTPUT_DIR = os.path.join(io.manuscript_output_dir, 'figure_3', 'output')

DAYS_SELECTED = [-2, -1, 1, 2]
MIN_CELLS = 3


# ============================================================================
# Data loading
# ============================================================================

def load_and_process_response_data(
    mice=None,
    days=[-2, -1, 0, 1, 2],
    win_sec_amp=(0, 0.300),
    win_sec_psth=(-0.5, 1.5),
    baseline_win=(0, 1),
    sampling_rate=30,
    file_name='tensor_xarray_mapping_data.nc'
):
    """
    Load average response and PSTH data for pre/post learning comparison.

    Returns avg_resp and psth DataFrames, both in % dF/F0, with a
    'learning_period' column ('pre' for days -2/-1, 'post' for days +1/+2).
    """
    if mice is None:
        _, _, mice, _ = io.select_sessions_from_db(
            io.db_path, io.nwb_dir,
            two_p_imaging='yes',
            experimenters=['AR', 'GF', 'MI']
        )

    baseline_win_samples = (
        int(baseline_win[0] * sampling_rate),
        int(baseline_win[1] * sampling_rate)
    )

    avg_resp_list = []
    psth_list = []

    for mouse_id in mice:
        reward_group = io.get_mouse_reward_group_from_db(io.db_path, mouse_id)

        folder = os.path.join(io.processed_dir, 'mice')
        xarr = utils_imaging.load_mouse_xarray(mouse_id, folder, file_name)
        xarr = utils_imaging.substract_baseline(xarr, 2, baseline_win_samples)

        # Average response
        avg = xarr.sel(trial=xarr['day'].isin(days))
        avg = avg.sel(time=slice(win_sec_amp[0], win_sec_amp[1])).mean(dim='time')
        avg.name = 'average_response'
        avg_df = avg.to_dataframe().reset_index()
        avg_df['mouse_id'] = mouse_id
        avg_df['reward_group'] = reward_group
        avg_resp_list.append(avg_df)

        # PSTH
        p = xarr.sel(trial=xarr['day'].isin(days))
        p = p.sel(time=slice(win_sec_psth[0], win_sec_psth[1]))
        p = p.groupby('day').mean(dim='trial')
        p.name = 'psth'
        p_df = p.to_dataframe().reset_index()
        p_df['mouse_id'] = mouse_id
        p_df['reward_group'] = reward_group
        psth_list.append(p_df)

    avg_resp = pd.concat(avg_resp_list)
    psth = pd.concat(psth_list)

    avg_resp['average_response'] *= 100
    psth['psth'] *= 100
    psth['time'] = psth['time'].round(4)

    avg_resp['learning_period'] = avg_resp['day'].map(
        lambda x: 'pre' if x in [-2, -1] else 'post'
    )
    psth['learning_period'] = psth['day'].map(
        lambda x: 'pre' if x in [-2, -1] else 'post'
    )

    return avg_resp, psth


# ============================================================================
# Panel generation
# ============================================================================

def generate_panel(
    reward_group,
    avg_resp,
    psth,
    panel_name,
    save_path=OUTPUT_DIR,
    save_format='svg',
    dpi=300,
):
    """
    Generate one panel: PSTH (left) + response amplitude barplot (right)
    for a single reward group, averaged across mice.

    Saves <panel_name>.svg, <panel_name>_data.csv, <panel_name>_stats.csv.
    """
    color = '#1b9e77' if reward_group == 'R+' else '#c959af'

    # Select days and reward group
    data_avg = avg_resp[avg_resp['day'].isin(DAYS_SELECTED) &
                        (avg_resp['reward_group'] == reward_group)]
    data_psth = psth[psth['day'].isin(DAYS_SELECTED) &
                     (psth['reward_group'] == reward_group)]

    # Filter by minimum cell count per mouse
    data_avg  = utils_imaging.filter_data_by_cell_count(data_avg, MIN_CELLS)
    data_psth = utils_imaging.filter_data_by_cell_count(data_psth, MIN_CELLS)

    # Average across cells per mouse
    mouse_avg = (
        data_avg
        .groupby(['mouse_id', 'learning_period'])['average_response']
        .mean()
        .reset_index()
    )
    mouse_psth = (
        data_psth
        .groupby(['mouse_id', 'learning_period', 'time'])['psth']
        .mean()
        .reset_index()
    )

    sns.set_theme(context='paper', style='ticks', palette='deep',
                  font='sans-serif', font_scale=1)

    fig, axes = plt.subplots(1, 2, figsize=(6, 3))

    # PSTH
    sns.lineplot(
        data=mouse_psth,
        x='time', y='psth',
        hue='learning_period', hue_order=['pre', 'post'],
        palette=['#a3a3a3', color],
        ax=axes[0], legend=False,
    )
    axes[0].axvline(0, color='orange', linestyle='-', linewidth=0.8)
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('dF/F0 (%)')
    axes[0].set_title(f'{reward_group} PSTH')

    # Response amplitude
    sns.barplot(
        data=mouse_avg,
        x='learning_period', y='average_response',
        order=['pre', 'post'],
        color=color,
        ax=axes[1],
    )
    sns.swarmplot(
        data=mouse_avg,
        x='learning_period', y='average_response',
        order=['pre', 'post'],
        color='black', alpha=0.7, size=4,
        ax=axes[1],
    )
    axes[1].set_xlabel('Learning period')
    axes[1].set_ylabel('Avg response (dF/F0 %)')
    axes[1].set_title(f'{reward_group} response amplitude')

    sns.despine()
    plt.tight_layout()

    os.makedirs(save_path, exist_ok=True)
    fig_path = os.path.join(save_path, f'{panel_name}.{save_format}')
    plt.savefig(fig_path, format=save_format, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"Panel saved: {fig_path}")

    # Save data
    mouse_avg['reward_group'] = reward_group
    data_path = os.path.join(save_path, f'{panel_name}_data.csv')
    mouse_avg.to_csv(data_path, index=False)

    # Statistics: Wilcoxon signed-rank pre vs post
    pre  = mouse_avg[mouse_avg['learning_period'] == 'pre']['average_response'].values
    post = mouse_avg[mouse_avg['learning_period'] == 'post']['average_response'].values
    try:
        stat, p_value = wilcoxon(pre, post)
    except ValueError:
        stat, p_value = np.nan, np.nan

    stats_df = pd.DataFrame([{
        'reward_group': reward_group,
        'test': 'Wilcoxon signed-rank',
        'comparison': 'pre vs post',
        'n_mice': len(pre),
        'stat': stat,
        'p_value': p_value,
    }])
    stats_path = os.path.join(save_path, f'{panel_name}_stats.csv')
    stats_df.to_csv(stats_path, index=False)
    print(f"Stats saved: {stats_path}")
    print(stats_df.to_string(index=False))


# ============================================================================
# Main execution
# ============================================================================

if __name__ == '__main__':
    print("Loading response and PSTH data...")
    avg_resp, psth = load_and_process_response_data()

    print("\nGenerating panel d (R+)...")
    generate_panel('R+', avg_resp, psth, panel_name='figure_3d')

    print("\nGenerating panel e (R-)...")
    generate_panel('R-', avg_resp, psth, panel_name='figure_3e')

    print("\nDone!")
