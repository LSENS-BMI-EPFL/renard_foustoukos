"""
Calcium transient illustration figure.

Shows concatenated single-trial traces for cells with the best stimulus-evoked
responses, selected by SNR (mean response [0, 0.3s] / std baseline [-1, 0s]).
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(r'/home/aprenard/repos/fast-learning')
sys.path.append(r'/home/aprenard/repos/NWB_analysis')
import src.utils.utils_io as io
import src.utils.utils_imaging as utils_imaging


# #############################################################################
# Parameters.
# #############################################################################

sampling_rate = 30

mouse_id = 'GF314'
day_for_trials = 2
trials_range = (10, 16)  # (start, stop) indices into the day's trials, e.g. (10, 20)
n_cells_to_plot = 20

file_name = 'tensor_xarray_mapping_data.nc'
folder = os.path.join(io.processed_dir, 'mice')


# #############################################################################
# Load data and compute SNR for each cell.
# #############################################################################

xarr = utils_imaging.load_mouse_xarray(mouse_id, folder, file_name, substracted=False)

# Select trials from target day
xarr_day = xarr.sel(trial=xarr['day'] == day_for_trials)

if xarr_day.sizes['trial'] < trials_range[1]:
    raise ValueError(f"Not enough trials: {xarr_day.sizes['trial']} < {trials_range[1]}")

trial_indices = xarr_day['trial'][trials_range[0]:trials_range[1]].values
n_trials_plot = len(trial_indices)
xarr_trials = xarr_day.sel(trial=trial_indices)

time_vals = xarr_trials['time'].values
response_mask = (time_vals >= 0) & (time_vals <= 0.3)
baseline_mask = (time_vals >= -1) & (time_vals < 0)

cell_metrics = []
n_cells = xarr_trials.sizes['cell']
for cell_idx in range(n_cells):
    cell_data = xarr_trials.isel(cell=cell_idx).values  # trials x time

    # Signal: mean dF/F in response window, averaged across trials
    signal = np.mean(cell_data[:, response_mask])

    # Noise: mean std of baseline across trials
    noise = np.mean(np.std(cell_data[:, baseline_mask], axis=1))

    snr = signal / noise if noise > 0 else 0.0

    cell_metrics.append({
        'roi': xarr_trials['cell'].values[cell_idx],
        'cell_idx': cell_idx,
        'snr': snr
    })

cell_metrics_df = pd.DataFrame(cell_metrics)
top_cells = cell_metrics_df.nlargest(n_cells_to_plot, 'snr')

print(f"Selected {len(top_cells)} cells by SNR (day {day_for_trials}):")
for i, (_, cell) in enumerate(top_cells.iterrows()):
    print(f"  {i+1}. ROI {cell['roi']}, SNR={cell['snr']:.2f}")


# #############################################################################
# Build concatenated traces and plot.
# #############################################################################

# Load full traces for selected cells
concatenated_traces = []
time_vec = None

for _, cell in top_cells.iterrows():
    cell_idx = int(cell['cell_idx'])
    xarr_cell = xarr_day.isel(cell=cell_idx).sel(trial=trial_indices)
    traces = xarr_cell.values * 100  # trials x time, convert to % dF/F

    n_trials, n_timepoints_per_trial = traces.shape
    nan_gap = 60  # frames between trials

    parts = []
    for t in range(n_trials):
        parts.append(traces[t, :])
        if t < n_trials - 1:
            parts.append(np.full(nan_gap, np.nan))
    concatenated_traces.append(np.concatenate(parts))

    if time_vec is None:
        time_vec = xarr_cell['time'].values
        n_timepoints = n_timepoints_per_trial

# Build time vector with gaps
trial_duration = n_timepoints / sampling_rate
gap_duration = nan_gap / sampling_rate

time_parts = []
for t in range(n_trials_plot):
    time_parts.append(time_vec + t * (trial_duration + gap_duration))
    if t < n_trials_plot - 1:
        time_parts.append(np.full(nan_gap, np.nan))
t_full = np.concatenate(time_parts)

# Plot
offset_step = 400  # % dF/F between cells

fig, ax = plt.subplots(figsize=(3, 6))

for i, trace in enumerate(concatenated_traces):
    ax.plot(t_full, trace + i * offset_step, color='black', linewidth=0.5)

# Stimulus onset lines
for t in range(n_trials_plot):
    stim_time = t * (trial_duration + gap_duration)
    ax.axvline(stim_time, color='#FF9600', linestyle='-', linewidth=0.8, alpha=0.7)

ax.set_xlabel('Time (s)', fontsize=9)
ax.set_ylabel('DF/F0 (%)', fontsize=9)
ax.tick_params(labelsize=8)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()


# #############################################################################
# Save.
# #############################################################################

output_dir = '/mnt/lsens-analysis/Anthony_Renard/analysis_output/fast-learning/illustrations'
output_dir = io.adjust_path_to_host(output_dir)

svg_file = f'calcium_transient_traces_{mouse_id}.svg'
fig.savefig(os.path.join(output_dir, svg_file), format='svg', dpi=300)
print(f"\nSaved: {os.path.join(output_dir, svg_file)}")

plt.show()
