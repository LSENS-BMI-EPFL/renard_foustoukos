import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

# sys.path.append(r'H:/anthony/repos/NWB_analysis')
sys.path.append(r'/home/aprenard/repos/NWB_analysis')
# from nwb_wrappers import nwb_reader_functions as nwb_read
from fast_learning import imaging
from fast_learning import paths, database
from fast_learning.plotting import *
from fast_learning.behavior import *


# ===========================================================
# Per-cell trial-by-trial responses for top LMI cells
# ===========================================================

# Parameters
n_cells_per_group = 30  # Number of top positive/negative cells per reward group
trials_per_row = 10  # Number of trials per row in grid
max_rows_per_page = 10  # Maximum number of rows per page before creating a new page
sampling_rate = 30
win_sec_stim = (-1, 5)  # Time window for stimulus-aligned data
win_sec_lick = (-1, 5)  # Time window for lick-aligned data

# Output PDF
output_dir = '/mnt/lsens-analysis/Anthony_Renard/analysis_output/fast-learning/lmi_characterisation'
output_dir = paths.adjust_path_to_host(output_dir)
output_pdf = os.path.join(output_dir, 'per_cell_trial_responses.pdf')

_, _, mice, db = database.select_sessions_from_db(paths.db_path,
                                            paths.nwb_dir,
                                            two_p_imaging='yes',
                                            experimenters=['AR', 'GF', 'MI'])
mice_count = db[['mouse_id', 'reward_group']].drop_duplicates()

# Load LMI data
lmi_df = os.path.join(paths.processed_dir, f'lmi_results.csv')
lmi_df = pd.read_csv(lmi_df)
lmi_df['reward_group'] = lmi_df['mouse_id'].map(
    dict(mice_count[['mouse_id', 'reward_group']].values)
)

# Select top n_cells_per_group positive and negative cells per reward group
selected_cells = []
for reward_group in lmi_df['reward_group'].unique():
    group_df = lmi_df[lmi_df['reward_group'] == reward_group]
    group_sorted = group_df.sort_values('lmi', ascending=False)

    # Top positive cells
    top_pos = group_sorted.head(n_cells_per_group).copy()
    top_pos['lmi_group'] = 'positive'
    selected_cells.append(top_pos)

    # Top negative cells
    top_neg = group_sorted.tail(n_cells_per_group).copy()
    top_neg['lmi_group'] = 'negative'
    selected_cells.append(top_neg)

selected_cells_df = pd.concat(selected_cells, ignore_index=True)

mice = [m for m in mice if m != 'GF305']

print(f"\nGenerating per-cell trial-by-trial analysis for {len(selected_cells_df)} cells...")
print(f"Selected cells per reward group: {n_cells_per_group} positive, {n_cells_per_group} negative")


# ===========================================================
# PSTH computation and plotting functions
# ===========================================================

def compute_cell_psths(mouse_id, roi, folder):
    """
    Compute PSTHs for a given cell across different trial types and learning phases.

    Parameters:
    -----------
    mouse_id : str
        Mouse identifier
    roi : int
        ROI identifier for the cell
    folder : str
        Path to data folder

    Returns:
    --------
    dict : Dictionary containing PSTH data for different conditions
    """
    psth_data = {}
    days_learning = [-2, -1, 0, 1, 2]

    # 1. Load mapping data for each day
    try:
        file_name_mapping = 'tensor_xarray_mapping_data.nc'
        xarr_mapping = imaging.load_mouse_xarray(mouse_id, folder, file_name_mapping, subtracted=True)
        xarr_mapping = xarr_mapping.sel(time=slice(win_sec_stim[0], win_sec_stim[1]))

        # Select this cell
        cell_idx = np.where(xarr_mapping['roi'].values == roi)[0]
        if len(cell_idx) > 0:
            xarr_whisker_mapping = xarr_mapping.isel(cell=cell_idx[0])

            # Load data for each day
            for day in days_learning:
                xarr_day = xarr_whisker_mapping.sel(trial=xarr_whisker_mapping['day'] == day)
                if len(xarr_day.trial) > 0:
                    psth_data[f'mapping_day{day}'] = xarr_day
    except Exception as e:
        print(f"    Warning: Could not load mapping data: {e}")

    # 2. Load learning data (all days)
    try:
        file_name_learning = 'tensor_xarray_learning_data.nc'
        xarr_learning = imaging.load_mouse_xarray(mouse_id, folder, file_name_learning, subtracted=True)
        xarr_learning = xarr_learning.sel(trial=xarr_learning['day'].isin(days_learning))
        xarr_learning = xarr_learning.sel(time=slice(win_sec_stim[0], win_sec_stim[1]))

        # Select this cell
        cell_idx = np.where(xarr_learning['roi'].values == roi)[0]
        if len(cell_idx) > 0:
            xarr_learning_cell = xarr_learning.isel(cell=cell_idx[0])

            # Whisker hits (exclude pre-learning days -2, -1 as these are mistaken trials)
            xarr_w_hit = xarr_learning_cell.sel(trial=(xarr_learning_cell['whisker_stim'] == 1) &
                                                       (xarr_learning_cell['lick_flag'] == 1) &
                                                       (~xarr_learning_cell['day'].isin([-2, -1])))
            if len(xarr_w_hit.trial) > 0:
                psth_data['whisker_hit'] = xarr_w_hit

            # Whisker misses (exclude pre-learning days -2, -1 as these are mistaken trials)
            xarr_w_miss = xarr_learning_cell.sel(trial=(xarr_learning_cell['whisker_stim'] == 1) &
                                                        (xarr_learning_cell['lick_flag'] == 0) &
                                                        (~xarr_learning_cell['day'].isin([-2, -1])))
            if len(xarr_w_miss.trial) > 0:
                psth_data['whisker_miss'] = xarr_w_miss

            # Correct rejections (no_stim, no lick)
            xarr_cr = xarr_learning_cell.sel(trial=(xarr_learning_cell['no_stim'] == 1) &
                                                    (xarr_learning_cell['lick_flag'] == 0))
            if len(xarr_cr.trial) > 0:
                psth_data['correct_rejection'] = xarr_cr

            # Auditory hits
            xarr_a_hit = xarr_learning_cell.sel(trial=(xarr_learning_cell['auditory_stim'] == 1) &
                                                       (xarr_learning_cell['lick_flag'] == 1))
            if len(xarr_a_hit.trial) > 0:
                psth_data['auditory_hit'] = xarr_a_hit
    except Exception as e:
        print(f"    Warning: Could not load learning data: {e}")

    # 3. Load lick-aligned data for false alarms
    try:
        file_name_lick = 'lick_aligned_xarray.nc'
        xarr_lick = imaging.load_mouse_xarray(mouse_id, folder, file_name_lick, subtracted=False)
        xarr_lick = xarr_lick.sel(trial=xarr_lick['day'].isin(days_learning))
        xarr_lick = xarr_lick.sel(time=slice(win_sec_lick[0], win_sec_lick[1]))

        # Select this cell
        cell_idx = np.where(xarr_lick['roi'].values == roi)[0]
        if len(cell_idx) > 0:
            xarr_lick_cell = xarr_lick.isel(cell=cell_idx[0])

            # False alarms (no_stim, lick)
            xarr_fa = xarr_lick_cell.sel(trial=(xarr_lick_cell['no_stim'] == 1) &
                                                (xarr_lick_cell['lick_flag'] == 1))
            if len(xarr_fa.trial) > 0:
                psth_data['false_alarm'] = xarr_fa
    except Exception as e:
        print(f"    Warning: Could not load lick-aligned data: {e}")

    return psth_data


def compute_whisker_evolution(mouse_id, roi, folder):
    """
    Compute whisker trial responses across the session to track learning evolution.

    Parameters:
    -----------
    mouse_id : str
        Mouse identifier
    roi : int
        ROI identifier for the cell
    folder : str
        Path to data folder

    Returns:
    --------
    dict : Dictionary containing whisker trial data for different session phases
    """
    evolution_data = {}

    try:
        # Load learning data for Day 0 only
        file_name_learning = 'tensor_xarray_learning_data.nc'
        xarr_learning = imaging.load_mouse_xarray(mouse_id, folder, file_name_learning, subtracted=True)
        xarr_learning = xarr_learning.sel(trial=xarr_learning['day'] == 0)
        xarr_learning = xarr_learning.sel(time=slice(win_sec_stim[0], win_sec_stim[1]))

        # Select this cell
        cell_idx = np.where(xarr_learning['roi'].values == roi)[0]
        if len(cell_idx) == 0:
            return evolution_data

        xarr_cell = xarr_learning.isel(cell=cell_idx[0])

        # Get all whisker trials (hits + misses) in chronological order
        whisker_trials = xarr_cell.sel(trial=xarr_cell['whisker_stim'] == 1)
        n_whisker = len(whisker_trials.trial)

        if n_whisker == 0:
            return evolution_data

        # --- Row 0: First trials (individual, not averaged) ---
        # Find first hit
        lick_flags = whisker_trials['lick_flag'].values
        first_hit_idx = np.where(lick_flags == 1)[0]

        if len(first_hit_idx) > 0:
            first_hit_idx = first_hit_idx[0]

            # Misses before first hit
            if first_hit_idx > 0:
                evolution_data['misses_before_first_hit'] = whisker_trials.isel(trial=slice(0, first_hit_idx))

            # First hit
            evolution_data['first_hit'] = whisker_trials.isel(trial=first_hit_idx)

            # Next 5 trials after first hit
            next_5_end = min(first_hit_idx + 6, n_whisker)  # +6 because we want 5 trials AFTER first hit
            if next_5_end > first_hit_idx + 1:
                evolution_data['next_5_after_hit'] = whisker_trials.isel(trial=slice(first_hit_idx + 1, next_5_end))

        # --- Percentile ranges (for rows 1-3) ---
        # Calculate percentile indices based on all whisker trials
        first_20_idx = max(1, int(n_whisker * 0.20))
        start_20_idx = first_20_idx
        end_80_idx = max(start_20_idx + 1, int(n_whisker * 0.80))
        last_20_idx = max(first_20_idx, int(n_whisker * 0.80))

        # All whisker trials by percentile
        evolution_data['all_first_20pct'] = whisker_trials.isel(trial=slice(0, first_20_idx))
        if end_80_idx > start_20_idx:
            evolution_data['all_middle_60pct'] = whisker_trials.isel(trial=slice(start_20_idx, end_80_idx))
        if last_20_idx < n_whisker:
            evolution_data['all_last_20pct'] = whisker_trials.isel(trial=slice(last_20_idx, n_whisker))

        # Get hits and misses separately
        hits = whisker_trials.sel(trial=whisker_trials['lick_flag'] == 1)
        misses = whisker_trials.sel(trial=whisker_trials['lick_flag'] == 0)

        # For hits: extract trials from same chronological positions
        if len(hits.trial) > 0:
            # Get trial indices of hits within the whisker_trials array
            hit_indices = np.where(lick_flags == 1)[0]

            # Hits in first 20%
            hits_in_first_20 = hit_indices[hit_indices < first_20_idx]
            if len(hits_in_first_20) > 0:
                evolution_data['hits_first_20pct'] = whisker_trials.isel(trial=hits_in_first_20)

            # Hits in middle 60%
            hits_in_middle = hit_indices[(hit_indices >= start_20_idx) & (hit_indices < end_80_idx)]
            if len(hits_in_middle) > 0:
                evolution_data['hits_middle_60pct'] = whisker_trials.isel(trial=hits_in_middle)

            # Hits in last 20%
            hits_in_last_20 = hit_indices[hit_indices >= last_20_idx]
            if len(hits_in_last_20) > 0:
                evolution_data['hits_last_20pct'] = whisker_trials.isel(trial=hits_in_last_20)

        # For misses: extract trials from same chronological positions
        if len(misses.trial) > 0:
            # Get trial indices of misses within the whisker_trials array
            miss_indices = np.where(lick_flags == 0)[0]

            # Misses in first 20%
            misses_in_first_20 = miss_indices[miss_indices < first_20_idx]
            if len(misses_in_first_20) > 0:
                evolution_data['misses_first_20pct'] = whisker_trials.isel(trial=misses_in_first_20)

            # Misses in middle 60%
            misses_in_middle = miss_indices[(miss_indices >= start_20_idx) & (miss_indices < end_80_idx)]
            if len(misses_in_middle) > 0:
                evolution_data['misses_middle_60pct'] = whisker_trials.isel(trial=misses_in_middle)

            # Misses in last 20%
            misses_in_last_20 = miss_indices[miss_indices >= last_20_idx]
            if len(misses_in_last_20) > 0:
                evolution_data['misses_last_20pct'] = whisker_trials.isel(trial=misses_in_last_20)

        print(f"    Whisker evolution: {n_whisker} total trials")
        if 'first_hit' in evolution_data:
            n_misses = len(evolution_data['misses_before_first_hit'].trial) if 'misses_before_first_hit' in evolution_data else 0
            print(f"      First hit at trial {first_hit_idx + 1}, {n_misses} misses before")

    except Exception as e:
        print(f"    Warning: Could not compute whisker evolution: {e}")
        import traceback
        traceback.print_exc()

    return evolution_data


def plot_whisker_evolution_page(evolution_data, mouse_id, roi, reward_group, lmi_group, lmi_value):
    """
    Create a page showing whisker response evolution across the session.

    Parameters:
    -----------
    evolution_data : dict
        Dictionary containing whisker trial xarrays for different session phases
    mouse_id, roi, reward_group, lmi_group, lmi_value : Cell metadata

    Returns:
    --------
    fig : matplotlib figure
    """
    import matplotlib.cm as cm

    # Create figure with 4 rows x 3 columns
    fig, axes = plt.subplots(4, 3, figsize=(20, 16), sharey=True)
    plt.subplots_adjust(hspace=0.35, wspace=0.25)  # Increase vertical spacing

    # Calculate global y-limits
    all_values = []
    for key, xarr in evolution_data.items():
        if xarr is not None:
            if hasattr(xarr, 'trial') and len(xarr.trial) > 0:
                all_values.extend(xarr.values.flatten() * 100)
            elif not hasattr(xarr, 'trial'):  # Single trial (first_hit)
                all_values.extend(xarr.values.flatten() * 100)

    if len(all_values) > 0:
        global_ymin = np.nanpercentile(all_values, 1)
        global_ymax = np.nanpercentile(all_values, 99.9)
        y_range = global_ymax - global_ymin
        global_ymin -= y_range * 0.1
        global_ymax += y_range * 0.2
    else:
        global_ymin, global_ymax = -5, 20

    # Define color palette based on reward group
    if reward_group == 'R+':
        cmap = cm._colormaps['Greens']
        trial_color_hit = trial_type_rew_palette[3]
        trial_color_miss = trial_type_rew_palette[2]
    else:  # R-
        cmap = cm._colormaps['Reds']
        trial_color_hit = trial_type_nonrew_palette[3]
        trial_color_miss = trial_type_nonrew_palette[2]

    time_vals = evolution_data.get('first_hit', evolution_data.get('all_first_20pct'))['time'].values if len(evolution_data) > 0 else np.linspace(-1, 5, 100)

    # === ROW 0: First trials (individual, not averaged) ===

    # Panel 0,0: Misses before first hit (averaged)
    ax = axes[0, 0]
    if 'misses_before_first_hit' in evolution_data and len(evolution_data['misses_before_first_hit'].trial) > 0:
        xarr = evolution_data['misses_before_first_hit']
        n_trials = len(xarr.trial)

        # Convert to dataframe and plot average with CI
        df = xarr.to_dataframe(name='activity').reset_index()
        df['activity'] = df['activity'] * 100

        sns.lineplot(data=df, x='time', y='activity', errorbar='ci',
                    ax=ax, color=trial_color_miss, linewidth=2.5, linestyle='--')

        ax.axvline(0, color=stim_palette[1], linestyle='-', linewidth=1.5)
        ax.set_title(f'Avg misses before 1st hit\n(n={n_trials})', fontsize=10, fontweight='bold')
        ax.set_ylabel('DF/F0 (%)', fontsize=10)
    else:
        ax.text(0.5, 0.5, 'No misses\nbefore 1st hit', ha='center', va='center',
               transform=ax.transAxes, fontsize=9)
    # Add row label on the left
    ax.text(-0.22, 0.5, 'Trials Around First Hit', transform=ax.transAxes,
           rotation=90, va='center', ha='center', fontsize=10, fontweight='bold')
    ax.set_ylabel('DF/F0 (%)', fontsize=10)
    ax.set_ylim(global_ymin, global_ymax)

    # Panel 0,1: First hit
    ax = axes[0, 1]
    if 'first_hit' in evolution_data:
        trial_data = evolution_data['first_hit'].values * 100
        ax.plot(time_vals, trial_data, color=cmap(0.7), linestyle='-',
               linewidth=2.5, alpha=0.9)
        ax.axvline(0, color=stim_palette[1], linestyle='-', linewidth=1.5)
        ax.set_title('First whisker hit', fontsize=10, fontweight='bold')
    else:
        ax.text(0.5, 0.5, 'No first whisker hit', ha='center', va='center',
               transform=ax.transAxes, fontsize=9)
    ax.set_ylim(global_ymin, global_ymax)

    # Panel 0,2: Next 5 trials after first hit
    ax = axes[0, 2]
    if 'next_5_after_hit' in evolution_data and len(evolution_data['next_5_after_hit'].trial) > 0:
        xarr = evolution_data['next_5_after_hit']
        n_trials = len(xarr.trial)
        lick_flags = xarr['lick_flag'].values
        for i in range(n_trials):
            trial_data = xarr.isel(trial=i).values * 100
            color_intensity = (i + 1) / (n_trials + 1)  # Light to dark
            linestyle = '-' if lick_flags[i] == 1 else '--'
            ax.plot(time_vals, trial_data, color=cmap(color_intensity), linestyle=linestyle,
                   linewidth=1.5, alpha=0.8)
        ax.axvline(0, color=stim_palette[1], linestyle='-', linewidth=1.5)
        ax.set_title(f'Next {n_trials} whisker trials\nafter 1st hit (from light to dark)', fontsize=10, fontweight='bold')
    else:
        ax.text(0.5, 0.5, 'No trials\nafter 1st hit', ha='center', va='center',
               transform=ax.transAxes, fontsize=9)
    ax.set_ylim(global_ymin, global_ymax)

    # === ROWS 1-3: Averaged PSTHs by percentile ===
    percentile_panels = [
        (1, 'all', 'All Whisker Trials', trial_color_hit, '-'),  # Use hit color for all trials
        (2, 'hits', 'Whisker Hits Only', trial_color_hit, '-'),
        (3, 'misses', 'Whisker Misses Only', trial_color_miss, '--')
    ]

    for row_idx, trial_type, row_label, color, linestyle in percentile_panels:
        # Define percentile labels based on trial type
        if trial_type == 'all':
            percentile_labels = [
                ('first_20pct', 'First 20% whisker trials'),
                ('middle_60pct', 'Middle 20-80% whisker trials'),
                ('last_20pct', 'Last 20% whisker trials')
            ]
        elif trial_type == 'hits':
            percentile_labels = [
                ('first_20pct', 'Hits among first 20% whisker trials'),
                ('middle_60pct', 'Hits among middle 20-80% whisker trials'),
                ('last_20pct', 'Hits among last 20% whisker trials')
            ]
        else:  # misses
            percentile_labels = [
                ('first_20pct', 'Misses among first 20% whisker trials'),
                ('middle_60pct', 'Misses among middle 20-80% whisker trials'),
                ('last_20pct', 'Misses among last 20% whisker trials')
            ]

        for col_idx, (percentile, pct_label) in enumerate(percentile_labels):
            ax = axes[row_idx, col_idx]

            key = f'{trial_type}_{percentile}'

            if key in evolution_data and evolution_data[key] is not None and len(evolution_data[key].trial) > 0:
                xarr = evolution_data[key]
                df = xarr.to_dataframe(name='activity').reset_index()
                df['activity'] = df['activity'] * 100

                # Plot PSTH with confidence interval
                sns.lineplot(data=df, x='time', y='activity', errorbar='ci',
                            ax=ax, color=color, linewidth=2.5, linestyle=linestyle)

                ax.axvline(0, color=stim_palette[1], linestyle='-', linewidth=1.5)
                ax.set_title(f'{pct_label} (n={len(xarr.trial)})', fontsize=9, fontweight='bold')
                if col_idx == 0:
                    ax.set_ylabel('DF/F0 (%)', fontsize=10)
                    # Add row label on the left
                    ax.text(-0.22, 0.5, row_label, transform=ax.transAxes,
                           rotation=90, va='center', ha='center', fontsize=10, fontweight='bold')
                if row_idx == 3:
                    ax.set_xlabel('Time (s)', fontsize=10)
            else:
                ax.text(0.5, 0.5, f'No data {pct_label}', ha='center', va='center',
                       transform=ax.transAxes, fontsize=9)
                if col_idx == 0:
                    ax.text(-0.22, 0.5, row_label, transform=ax.transAxes,
                           rotation=90, va='center', ha='center', fontsize=10, fontweight='bold')

            ax.set_ylim(global_ymin, global_ymax)

    # Overall title
    if reward_group == 'R-':
        fig.suptitle(f'{mouse_id} ROI {roi} ({reward_group}, {lmi_group} LMI={lmi_value:.3f})\n' +
                    'Whisker Response Evolution Across Session (Day 0)',
                    fontsize=14, fontweight='bold', y=0.995, color=reward_palette[0])
    elif reward_group == 'R+':
        fig.suptitle(f'{mouse_id} ROI {roi} ({reward_group}, {lmi_group} LMI={lmi_value:.3f})\n' +
                    'Whisker Response Evolution Across Session (Day 0)',
                    fontsize=14, fontweight='bold', y=0.995, color=reward_palette[1])

    sns.despine(fig=fig)
    return fig


def plot_psth_summary_page(psth_data, mouse_id, roi, reward_group, lmi_group, lmi_value, cell_color):
    """
    Create a summary page with PSTHs for all trial types and learning phases.

    Parameters:
    -----------
    psth_data : dict
        Dictionary containing PSTH xarrays for different conditions
    mouse_id, roi, reward_group, lmi_group, lmi_value : Cell metadata
    cell_color : Color for this cell

    Returns:
    --------
    fig : matplotlib figure
    """
    # Create figure with grid layout: 2 mapping plots + 5 rows x 5 cols for trial types by day
    fig = plt.figure(figsize=(22, 18))
    gs = fig.add_gridspec(6, 5, hspace=0.4, wspace=0.3)

    days_learning = [-2, -1, 0, 1, 2]

    # First pass: calculate global y-axis limits across all data
    all_values = []
    for key, xarr in psth_data.items():
        all_values.extend(xarr.values.flatten() * 100)  # Convert to percent

    if len(all_values) > 0:
        global_ymin = np.nanpercentile(all_values, 1)  # Use 1st percentile to avoid outliers
        global_ymax = np.nanpercentile(all_values, 99.9)  # Use 99th percentile
        # Add some padding
        y_range = global_ymax - global_ymin
        global_ymin -= y_range * 0.1
        global_ymax += y_range * 0.2
    else:
        global_ymin, global_ymax = -5, 20  # Default range

    # Collect all axes for setting shared y-limits
    all_axes = []

    # Set reward-group-specific colors for mapping trials
    if reward_group == 'R+':
        mapping_color = trial_type_rew_palette[3]  # Green
    else:  # R-
        mapping_color = trial_type_nonrew_palette[3]  # Magenta

    # --- Row 0: Mapping trials (passive whisker) by day ---
    days_learning = [-2, -1, 0, 1, 2]
    for col_idx, day in enumerate(days_learning):
        ax = fig.add_subplot(gs[0, col_idx])
        all_axes.append(ax)

        key = f'mapping_day{day}'

        if key in psth_data:
            xarr = psth_data[key]
            df = xarr.to_dataframe(name='activity').reset_index()
            df['activity'] = df['activity'] * 100

            sns.lineplot(data=df, x='time', y='activity', errorbar='ci',
                        ax=ax, color=mapping_color, linewidth=2.5, linestyle='-')

            ax.axvline(0, color=stim_palette[1], linestyle='-', linewidth=1.5)
            n_trials = len(xarr.trial)
            ax.set_title(f'Day {day}\n(n={n_trials} trials)', fontsize=9, fontweight='bold')
            ax.set_ylabel('DF/F0 (%)', fontsize=10)

            if col_idx == 0:
                # Add trial type label on the left
                ax.text(-0.3, 0.5, 'Passive Whisker\nTrials', transform=ax.transAxes,
                       rotation=90, va='center', ha='center', fontsize=10, fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_ylabel('DF/F0 (%)', fontsize=10)
            if col_idx == 0:
                ax.text(-0.3, 0.5, 'Passive Whisker\nTrials', transform=ax.transAxes,
                       rotation=90, va='center', ha='center', fontsize=10, fontweight='bold')

    # --- Rows 1-5: Trial types by day (5 panels per row, one per day) ---
    # Set whisker colors based on reward group
    if reward_group == 'R+':
        whisker_hit_color = trial_type_rew_palette[3]
        whisker_miss_color = trial_type_rew_palette[2]
    else:  # R-
        whisker_hit_color = trial_type_nonrew_palette[3]
        whisker_miss_color = trial_type_nonrew_palette[2]

    trial_types = [
        ('whisker_hit', 'Whisker Hit', whisker_hit_color, '-', 1),
        ('whisker_miss', 'Whisker Miss', whisker_miss_color, '--', 2),
        ('false_alarm', 'False Alarm (lick-aligned)', behavior_palette[5], '-', 3),
        ('correct_rejection', 'Correct Rejection', behavior_palette[4], '--', 4),
        ('auditory_hit', 'Auditory Hit', behavior_palette[1], '-', 5)
    ]

    for trial_key, trial_label, trial_color, trial_linestyle, row_idx in trial_types:
        for col_idx, day in enumerate(days_learning):
            ax = fig.add_subplot(gs[row_idx, col_idx])
            all_axes.append(ax)

            if trial_key in psth_data:
                xarr = psth_data[trial_key]

                # Filter for this specific day
                day_data = xarr.sel(trial=xarr['day'] == day)

                if len(day_data.trial) > 0:
                    # Create dataframe for this day
                    df = day_data.to_dataframe(name='activity').reset_index()
                    df['activity'] = df['activity'] * 100

                    # Plot PSTH with confidence interval
                    sns.lineplot(data=df, x='time', y='activity', errorbar='ci',
                               ax=ax, color=trial_color, linewidth=2, linestyle=trial_linestyle)

                    # Set time 0 marker color based on trial type
                    if trial_key in ['whisker_hit', 'whisker_miss']:
                        t0_color = stim_palette[1]  # Orange for whisker
                    elif trial_key in ['auditory_hit']:
                        t0_color = stim_palette[0]  # Blue for auditory
                    else:
                        t0_color = stim_palette[2]  # Grey/black for no_stim
                    ax.axvline(0, color=t0_color, linestyle='-', linewidth=1.5)
                    ax.set_title(f'Day {day:+d}' if day != 0 else 'Day 0', fontsize=9, fontweight='bold')
                    ax.set_xlabel('Time (s)' if row_idx == 5 else '', fontsize=8)
                    ax.set_ylabel('DF/F0 (%)', fontsize=10)
                    ax.tick_params(labelsize=7)

                    # Add trial type label on the left side (first column only)
                    if col_idx == 0:
                        ax.text(-0.3, 0.5, trial_label, transform=ax.transAxes,
                               rotation=90, va='center', ha='center', fontsize=10, fontweight='bold')
                else:
                    ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                           transform=ax.transAxes, fontsize=8)
                    ax.set_ylabel('DF/F0 (%)', fontsize=10)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    if col_idx == 0:
                        ax.text(-0.3, 0.5, trial_label, transform=ax.transAxes,
                               rotation=90, va='center', ha='center', fontsize=10, fontweight='bold')
            else:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                       transform=ax.transAxes, fontsize=8)
                ax.set_xticks([])
                ax.set_yticks([])
                if col_idx == 0:
                    ax.text(-0.3, 0.5, trial_label, transform=ax.transAxes,
                           rotation=90, va='center', ha='center', fontsize=10, fontweight='bold')

    # Set shared y-axis limits for all panels
    for ax in all_axes:
        ax.set_ylim(global_ymin, global_ymax)

    # Overall title
    if reward_group == 'R-':
        fig.suptitle(f'{mouse_id} ROI {roi} ({reward_group}, {lmi_group} LMI={lmi_value:.3f})\n' +
                 f'PSTH Summary Across Learning',
                 fontsize=14, fontweight='bold', y=0.995, color=reward_palette[0])
    elif reward_group == 'R+':
        fig.suptitle(f'{mouse_id} ROI {roi} ({reward_group}, {lmi_group} LMI={lmi_value:.3f})\n' +
                 f'PSTH Summary Across Learning',
                 fontsize=14, fontweight='bold', y=0.995, color=reward_palette[1])

    sns.despine(fig=fig)  # Specify figure to avoid searching all figures
    return fig

with PdfPages(output_pdf) as pdf:
    for idx, cell_row in selected_cells_df.iterrows():
        mouse_id = cell_row['mouse_id']
        roi = cell_row['roi']
        lmi_group = cell_row['lmi_group']
        reward_group = cell_row['reward_group']
        lmi_value = cell_row['lmi']

        print(f"\nProcessing cell {idx+1}/{len(selected_cells_df)}: {mouse_id} ROI {roi} ({lmi_group} LMI, {reward_group})")

        # Get color for this cell based on reward group
        cell_color = reward_palette[0] if reward_group == 'R-' else reward_palette[1]

        # Load stimulus-aligned data
        file_name = 'tensor_xarray_learning_data.nc'
        folder = paths.tensor_dir
        xarr_stim = imaging.load_mouse_xarray(mouse_id, folder, file_name, subtracted=False)
        xarr_stim = xarr_stim.sel(trial=xarr_stim['day'] == 0)
        xarr_stim = xarr_stim.sel(time=slice(win_sec_stim[0], win_sec_stim[1]))

        # Select this specific cell
        cell_idx = np.where(xarr_stim['roi'].values == roi)[0]
        if len(cell_idx) == 0:
            print(f"  Cell ROI {roi} not found in data, skipping")
            continue
        xarr_cell = xarr_stim.isel(cell=cell_idx[0])
        xarr_cell.load()  # Load into memory and close file handle

        # Load lick-aligned data for false alarms
        file_name_lick = 'lick_aligned_xarray.nc'
        xarr_lick = imaging.load_mouse_xarray(mouse_id, folder, file_name_lick, subtracted=False)
        xarr_lick = xarr_lick.sel(trial=xarr_lick['day'] == 0)
        xarr_lick = xarr_lick.sel(time=slice(win_sec_lick[0], win_sec_lick[1]))
        cell_idx_lick = np.where(xarr_lick['roi'].values == roi)[0]
        if len(cell_idx_lick) == 0:
            print(f"  Cell ROI {roi} not found in lick-aligned data")
            xarr_cell_lick = None
        else:
            xarr_cell_lick = xarr_lick.isel(cell=cell_idx_lick[0])
            xarr_cell_lick.load()  # Load into memory and close file handle

        # ===== Generate PSTH summary page =====
        print(f"  Generating PSTH summary page...")
        psth_data = compute_cell_psths(mouse_id, roi, folder)

        if len(psth_data) > 0:
            fig_psth = plot_psth_summary_page(psth_data, mouse_id, roi, reward_group,
                                             lmi_group, lmi_value, cell_color)
            pdf.savefig(fig_psth)
            plt.close(fig_psth)
            print(f"  PSTH summary page created with {len(psth_data)} trial type(s)")
        else:
            print(f"  Warning: No PSTH data available for this cell")

        # ===== Generate whisker evolution page =====
        print(f"  Generating whisker evolution page...")
        evolution_data = compute_whisker_evolution(mouse_id, roi, folder)

        if len(evolution_data) > 0:
            fig_evolution = plot_whisker_evolution_page(evolution_data, mouse_id, roi,
                                                       reward_group, lmi_group, lmi_value)
            pdf.savefig(fig_evolution)
            plt.close(fig_evolution)
            print(f"  Whisker evolution page created")
        else:
            print(f"  Warning: No whisker evolution data available for this cell")

        # ===== Collect all trials in session order =====
        # Build list of trial info: (trial_idx, data, color, linestyle, marker_color, trial_type)
        trial_info = []
        
        n_total_trials = len(xarr_cell.trial)
        time_vals = xarr_cell['time'].values
        
        for trial_idx in range(n_total_trials):
            trial_data_xarr = xarr_cell.isel(trial=trial_idx)
            trial_data = trial_data_xarr.values * 100  # Convert to DF/F0 %
            
            whisker = trial_data_xarr['whisker_stim'].values
            auditory = trial_data_xarr['auditory_stim'].values
            no_stim = trial_data_xarr['no_stim'].values
            lick = trial_data_xarr['lick_flag'].values
            
            # Determine trial type and styling
            if whisker == 1:
                if lick == 1:  # whisker hit
                    color = trial_type_rew_palette[3] if reward_group == 'R+' else trial_type_nonrew_palette[3]
                    linestyle = '-'
                    marker_color = stim_palette[1]  # Orange
                    trial_type = 'W Hit'
                    linewidth = 4
                else:  # whisker miss
                    color = trial_type_rew_palette[2] if reward_group == 'R+' else trial_type_nonrew_palette[2]
                    linestyle = '--'
                    marker_color = stim_palette[1]
                    trial_type = 'W Miss'
                    linewidth = 2.5
            elif auditory == 1:
                if lick == 1:  # auditory hit
                    color = behavior_palette[1]
                    linestyle = '-'
                    marker_color = stim_palette[0]  # Blue
                    trial_type = 'A Hit'
                    linewidth = 4
                else:  # auditory miss
                    color = behavior_palette[0]
                    linestyle = '-'
                    marker_color = stim_palette[0]
                    trial_type = 'A Miss'
                    linewidth = 2.5
            elif no_stim == 1:
                if lick == 1:  # no_stim hit (false alarm)
                    color = behavior_palette[5]
                    linestyle = '-'
                    marker_color = stim_palette[2]  # Black
                    trial_type = 'NS Hit'
                    linewidth = 4
                else:  # no_stim miss (correct rejection)
                    color = behavior_palette[4]
                    linestyle = '--'
                    marker_color = stim_palette[2]
                    trial_type = 'NS Miss'
                    linewidth = 2.5
            else:
                continue  # Skip trials that don't match any category
            
            trial_info.append({
                'index': trial_idx,
                'data': trial_data,
                'color': color,
                'linestyle': linestyle,
                'linewidth': linewidth,
                'marker_color': marker_color,
                'type': trial_type
            })
        
        n_trials = len(trial_info)
        print(f"  Total trials: {n_trials}")
        
        if n_trials == 0:
            print(f"  No trials to plot")
            continue
        
        # ===== Plot all trials in session order with pagination =====
        trials_per_page = trials_per_row * max_rows_per_page
        n_pages = int(np.ceil(n_trials / trials_per_page))

        # Track if we've seen the first whisker hit
        first_whisker_hit_found = False

        for page_num in range(n_pages):
            start_trial = page_num * trials_per_page
            end_trial = min(start_trial + trials_per_page, n_trials)
            trials_on_page = end_trial - start_trial

            n_rows = int(np.ceil(trials_on_page / trials_per_row))
            n_cols = trials_per_row

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.5 * n_cols, 3 * n_rows),
                                    sharey=True, sharex=True)
            if n_rows == 1:
                axes = axes.reshape(1, -1)

            for idx_in_page in range(trials_on_page):
                trial_idx = start_trial + idx_in_page
                trial = trial_info[trial_idx]

                row_idx = idx_in_page // trials_per_row
                col_idx = idx_in_page % trials_per_row
                ax = axes[row_idx, col_idx]

                # Plot trial data with appropriate styling
                ax.plot(time_vals, trial['data'],
                       color=trial['color'],
                       linestyle=trial['linestyle'],
                       linewidth=trial['linewidth'])
                ax.axvline(0, color=trial['marker_color'], linestyle='-', linewidth=2)

                # Title with trial number and type
                ax.set_title(f"Trial {trial['index'] + 1}\n{trial['type']}", fontsize=11, fontweight='bold')
                ax.set_ylabel('DF/F0 (%)' if col_idx == 0 else '')
                ax.set_xlabel('Time (s)' if row_idx == n_rows - 1 else '')
                # Set xticks only on bottom row (sharex=True makes this redundant elsewhere)
                if row_idx == n_rows - 1:
                    ax.set_xticks([-1, 0, 1, 2, 3, 4, 5])
                ax.tick_params(labelsize=7)

                # Add "FIRST HIT!" label to the first whisker hit trial
                if trial['type'] == 'W Hit' and not first_whisker_hit_found:
                    ax.text(0.98, 0.95, 'FIRST HIT!', transform=ax.transAxes,
                           fontsize=11, fontweight='bold', color='red',
                           ha='right', va='top')
                    first_whisker_hit_found = True
            
            # Hide unused subplots
            for idx_in_page in range(trials_on_page, n_rows * n_cols):
                row_idx = idx_in_page // trials_per_row
                col_idx = idx_in_page % trials_per_row
                axes[row_idx, col_idx].axis('off')
            
            # Page title
            if n_pages > 1:
                page_info = f' - Page {page_num + 1}/{n_pages}'
            else:
                page_info = ''
            
            plt.suptitle(f'{mouse_id} ROI {roi} ({reward_group}, {lmi_group} LMI={lmi_value:.3f})\n' +
                        f'All Trials in Session Order (Day 0){page_info}',
                        fontsize=12, y=0.998)
            plt.tight_layout()
            sns.despine(fig=fig)  # Specify figure to avoid searching all figures
            pdf.savefig(fig)
            plt.close(fig)
        
        print(f"  Plotted {n_trials} trials across {n_pages} page(s)")

        
        # ===== Lick-aligned False Alarms (separate page) =====
        if xarr_cell_lick is not None:
            false_alarms = xarr_cell_lick.sel(trial=(xarr_cell_lick['no_stim'] == 1) & (xarr_cell_lick['lick_flag'] == 1))
            n_fa_trials = len(false_alarms.trial)

            if n_fa_trials > 0:
                print(f"  Lick-aligned false alarms: {n_fa_trials} trials")
                
                # Plot false alarms with pagination
                trials_per_page = trials_per_row * max_rows_per_page
                n_pages = int(np.ceil(n_fa_trials / trials_per_page))
                
                for page_num in range(n_pages):
                    start_trial = page_num * trials_per_page
                    end_trial = min(start_trial + trials_per_page, n_fa_trials)
                    trials_on_page = end_trial - start_trial
                    
                    n_rows = int(np.ceil(trials_on_page / trials_per_row))
                    n_cols = trials_per_row
                    
                    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.5 * n_cols, 3 * n_rows),
                                            sharey=True, sharex=True, )
                    if n_rows == 1:
                        axes = axes.reshape(1, -1)
                    
                    time_vals_lick = false_alarms['time'].values
                    
                    for idx_in_page in range(trials_on_page):
                        trial_idx = start_trial + idx_in_page
                        trial_data = false_alarms.isel(trial=trial_idx).values * 100
                        
                        row_idx = idx_in_page // trials_per_row
                        col_idx = idx_in_page % trials_per_row
                        ax = axes[row_idx, col_idx]
                        
                        ax.plot(time_vals_lick, trial_data, color=behavior_palette[5], linewidth=1.5)
                        ax.axvline(0, color=stim_palette[2], linestyle='-', linewidth=1.5)
                        
                        ax.set_title(f'FA Trial {trial_idx + 1}', fontsize=8)
                        ax.set_ylabel('DF/F0 (%)' if col_idx == 0 else '')
                        ax.set_xlabel('Time from lick (s)' if row_idx == n_rows - 1 else '')
                        # Set xticks only on bottom row (sharex=True makes this redundant elsewhere)
                        if row_idx == n_rows - 1:
                            ax.set_xticks([-1, 0, 1, 2, 3, 4, 5])
                        ax.tick_params(labelsize=7)
                    
                    # Hide unused subplots
                    for idx_in_page in range(trials_on_page, n_rows * n_cols):
                        row_idx = idx_in_page // trials_per_row
                        col_idx = idx_in_page % trials_per_row
                        axes[row_idx, col_idx].axis('off')
                    
                    # Page title
                    if n_pages > 1:
                        page_info = f' - Page {page_num + 1}/{n_pages}'
                    else:
                        page_info = ''
                    
                    plt.suptitle(f'{mouse_id} ROI {roi} ({reward_group}, {lmi_group} LMI={lmi_value:.3f})\n' +
                                f'False Alarms - Lick Aligned (Day 0){page_info}',
                                fontsize=12, y=0.998)
                    plt.tight_layout()
                    sns.despine(fig=fig)  # Specify figure to avoid searching all figures
                    pdf.savefig(fig)
                    plt.close(fig)
            else:
                print(f"  No lick-aligned false alarms")

print(f"\nPer-cell analysis complete! PDF saved to: {output_pdf}")