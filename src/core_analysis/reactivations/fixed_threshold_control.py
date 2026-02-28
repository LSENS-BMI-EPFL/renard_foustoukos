"""
Fixed-threshold reactivation control: circular shift event frequency comparison.

When using a fixed correlation threshold (use_surrogate_thresholds = False), this
script validates that the threshold is not generating spurious events by comparing
the frequency of detected events in real data against the frequency detected in
circular-shift surrogates, across all 5 days.

For each mouse and each day (no-stim trials):
  1. Compute template correlation on real data, detect events → real event frequency
  2. Run N circular shift iterations, detect events in each → surrogate frequencies
  3. Average surrogate frequencies per mouse

Figure: two panels (R+, R-), x-axis = days, grouped bars (Real vs Surrogate) per
day, with per-mouse lines connecting the two bars within each day.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
from joblib import Parallel, delayed

sys.path.append(r'/home/aprenard/repos/NWB_analysis')
sys.path.append(r'/home/aprenard/repos/fast-learning')
import src.utils.utils_imaging as utils_imaging
import src.utils.utils_io as io
from src.utils.utils_plot import *
from src.core_analysis.reactivations.reactivation import (
    create_whisker_template,
    compute_template_correlation,
    detect_reactivation_events,
)
from src.core_analysis.reactivations.reactivation_surrogates_per_day import (
    create_surrogate_by_circular_shift,
)


# =============================================================================
# PARAMETERS
# =============================================================================

sampling_rate = 30          # Hz
days = [-2, -1, 0, 1, 2]
days_str = ['-2', '-1', '0', '+1', '+2']
threshold_dff = None        # Template cell threshold (None = all cells)
threshold_corr = 0.45       # Fixed correlation threshold for event detection
min_event_distance_ms = 150
min_event_distance_frames = int(min_event_distance_ms / 1000 * sampling_rate)
prominence = 0.15
n_surrogates = 20           # Circular shift iterations per mouse per day
n_jobs = 35                 # Parallel workers

# Visualisation
sns.set_theme(context='paper', style='ticks', palette='deep',
              font='sans-serif', font_scale=1)

# Output
save_dir = os.path.join(io.results_dir, 'reactivation')
os.makedirs(save_dir, exist_ok=True)

# Load database and split mice by reward group
_, _, all_mice, db = io.select_sessions_from_db(
    io.db_path, io.nwb_dir, two_p_imaging='yes'
)

r_plus_mice, r_minus_mice = [], []
for mouse in all_mice:
    try:
        rg = io.get_mouse_reward_group_from_db(io.db_path, mouse, db=db)
        if rg == 'R+':
            r_plus_mice.append(mouse)
        elif rg == 'R-':
            r_minus_mice.append(mouse)
    except:
        continue

print(f"R+ mice: {r_plus_mice}")
print(f"R- mice: {r_minus_mice}")


# =============================================================================
# CORE COMPUTATION
# =============================================================================

def process_single_mouse(mouse, days, threshold_corr, n_surrogates):
    """
    Compute real and surrogate event frequencies for one mouse across all days.

    Loads the xarray once, then loops over days. For each day: computes template
    correlation on real data and on N circular-shift surrogates, detects events
    with the fixed threshold, and returns events/min for each condition.

    Parameters
    ----------
    mouse : str
    days : list of int
    threshold_corr : float
    n_surrogates : int

    Returns
    -------
    records : list of dict
        Keys: mouse_id, day, condition ('Real' | 'Surrogate'), event_freq
    """
    records = []

    # Load xarray once for all days
    folder = os.path.join(io.solve_common_paths('processed_data'), 'mice')
    xarray = utils_imaging.load_mouse_xarray(
        mouse, folder, 'tensor_xarray_learning_data.nc', substracted=True
    )

    for day in days:
        try:
            template, _ = create_whisker_template(mouse, day, threshold_dff, verbose=False)

            xarray_day = xarray.sel(trial=xarray['day'] == day)
            nostim = xarray_day.sel(trial=xarray_day['no_stim'] == 1)

            if len(nostim.trial) < 5:
                continue

            n_cells, n_trials, n_timepoints = nostim.shape
            data = nostim.values.reshape(n_cells, -1)
            data = np.nan_to_num(data, nan=0.0)
            n_frames = data.shape[1]
            duration_min = n_frames / sampling_rate / 60.0

            # Real data
            real_corr = compute_template_correlation(data, template)
            real_events = detect_reactivation_events(
                real_corr, threshold=threshold_corr,
                min_distance=min_event_distance_frames, prominence=prominence
            )
            real_freq = len(real_events) / duration_min

            # Circular shift surrogates
            surrogate_freqs = np.zeros(n_surrogates)
            for i in range(n_surrogates):
                surrogate_data = create_surrogate_by_circular_shift(data, min_shift_frames=0)
                surrogate_corr = compute_template_correlation(surrogate_data, template)
                surrogate_events = detect_reactivation_events(
                    surrogate_corr, threshold=threshold_corr,
                    min_distance=min_event_distance_frames, prominence=prominence
                )
                surrogate_freqs[i] = len(surrogate_events) / duration_min

            records.append({'mouse_id': mouse, 'day': day,
                            'condition': 'Real', 'event_freq': real_freq})
            records.append({'mouse_id': mouse, 'day': day,
                            'condition': 'Surrogate', 'event_freq': surrogate_freqs.mean()})

        except Exception as e:
            continue

    return records


# =============================================================================
# VISUALISATION
# =============================================================================

def plot_real_vs_surrogate(df, save_path):
    """
    Two-panel figure (R+ | R-).

    Each panel: for each day, two side-by-side bars (Real vs Surrogate, mean ± SEM)
    with per-mouse lines connecting paired values within that day.
    """
    bar_width = 0.35
    gap = 1.0
    x_centres = np.arange(len(days)) * gap

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

    for ax, group, group_color in [
        (axes[0], 'R+', reward_palette[1]),
        (axes[1], 'R-', reward_palette[0]),
    ]:
        group_df = df[df['reward_group'] == group]

        if group_df.empty:
            ax.set_title(f'{group} (no data)')
            continue

        for i, day in enumerate(days):
            day_df = group_df[group_df['day'] == day]
            x_real = x_centres[i] - bar_width / 2
            x_surr = x_centres[i] + bar_width / 2

            for x_pos, cond, color in [
                (x_real, 'Real', group_color),
                (x_surr, 'Surrogate', 'lightgray'),
            ]:
                cond_df = day_df[day_df['condition'] == cond]
                if cond_df.empty:
                    continue
                mean_val = cond_df['event_freq'].mean()
                sem_val = cond_df['event_freq'].sem()
                ax.bar(x_pos, mean_val, width=bar_width, color=color,
                       edgecolor='black', linewidth=0.8)
                ax.errorbar(x_pos, mean_val, yerr=sem_val, fmt='none',
                            color='black', capsize=3, linewidth=1.0)

            # Per-mouse lines connecting Real → Surrogate
            for _, mouse_df in day_df.groupby('mouse_id'):
                pivot = mouse_df.set_index('condition').reindex(['Real', 'Surrogate'])
                if pivot['event_freq'].isna().any():
                    continue
                ax.plot([x_real, x_surr], pivot['event_freq'].values,
                        color='black', alpha=0.4, linewidth=0.8,
                        marker='o', markersize=2)

        n_mice = group_df['mouse_id'].nunique()
        ax.set_xticks(x_centres)
        ax.set_xticklabels(days_str)
        ax.set_xlabel('Day')
        ax.set_ylabel('Event frequency (events / min)')
        ax.set_title(f'{group} (n={n_mice} mice)')
        sns.despine(ax=ax)

    # Shared legend
    handles = [Patch(color=reward_palette[1], label='Real (R+)'),
               Patch(color=reward_palette[0], label='Real (R-)'),
               Patch(color='lightgray', edgecolor='black', label='Circular shift')]
    fig.legend(handles=handles, loc='upper right', frameon=False, fontsize=8)

    fig.suptitle(
        f'Fixed threshold (corr={threshold_corr}) — Real vs circular-shift event frequency',
        fontsize=10
    )
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved figure: {save_path}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    print(f"\nFixed-threshold control: real vs circular-shift event frequency")
    print(f"  Days: {days}, threshold_corr: {threshold_corr}, n_surrogates: {n_surrogates}")
    print(f"  Parallel jobs: {n_jobs}\n")

    all_mice_to_process = r_plus_mice + r_minus_mice
    print(f"Processing {len(all_mice_to_process)} mice in parallel...")

    results = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(process_single_mouse)(mouse, days, threshold_corr, n_surrogates)
        for mouse in all_mice_to_process
    )

    # Flatten list of lists
    all_records = [rec for mouse_records in results for rec in mouse_records]

    if not all_records:
        print("ERROR: No valid results collected.")
        import sys; sys.exit(1)

    df = pd.DataFrame(all_records)
    df['reward_group'] = df['mouse_id'].apply(
        lambda m: io.get_mouse_reward_group_from_db(io.db_path, m, db=db)
    )

    # Save data
    csv_path = os.path.join(save_dir, 'fixed_threshold_control_event_freq.csv')
    df.to_csv(csv_path, index=False)
    print(f"Saved data: {csv_path}")

    # Plot
    svg_path = os.path.join(save_dir, 'fixed_threshold_control_event_freq.svg')
    plot_real_vs_surrogate(df, svg_path)
