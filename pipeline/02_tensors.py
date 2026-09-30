"""Pipeline step 02: per-mouse dF/F tensors.

For each imaging mouse, dF/F aligned to trial onsets (cells x trials x time)
as xarrays with the trial table as coordinates:
    tensor_xarray_learning_data.nc   task trials of days -2..+2 (cut at the stop flag)
    tensor_xarray_mapping_data.nc    passive whisker mapping trials
    *_baselinesubstracted.nc         the same, minus the pre-stimulus baseline
    lick_aligned_xarray.nc           learning trials aligned to the first lick
                                     (exploratory analyses only)

Inputs:  NWB files, session metadata, stop-flag YAMLs.
Outputs: <processed_dir>/mice/<mouse>/*.nc. The other steps read tensors from
         tensor_dir (config.yaml); set it to <processed_dir>/mice to use these.

Usage:
    python pipeline/02_tensors.py
"""

import os

import numpy as np
import pandas as pd
import yaml
import xarray as xr

from fast_learning import paths, database
from cicada_nwb import NWBSession
from fast_learning import imaging
from fast_learning.tensors import make_events_aligned_array_3d
from fast_learning.behavior import make_behavior_table


# =============================================================================
# Create mice tensors with xarrays for session data (not mapping trials).
# =============================================================================

# Get directories and files.
db_path = paths.db_path
nwb_path = paths.nwb_dir
trial_indices_yaml = paths.trial_indices_yaml
stop_flag_yaml = paths.stop_flags_yaml
trial_indices_sensory_map_yaml = paths.trial_indices_sensory_map_yaml
stop_flag_sensory_map_yaml = paths.stop_flags_sensory_map_yaml
processed_data_dir = paths.processed_dir

days = ['-2', '-1', '0', '+1', '+2']
_, nwb_list, mice_list, _ = database.select_sessions_from_db(
    db_path,
    nwb_path,
    exclude_cols=['exclude', 'two_p_exclude'],
    experimenters=['AR', 'GF', 'MI'],
    day=days,
    two_p_imaging='yes',
)

with open(trial_indices_yaml, 'r') as stream:
    trial_indices = yaml.load(stream, yaml.Loader)
trial_indices = pd.DataFrame(trial_indices.items(), columns=['session_id', 'trial_idx'])
# # For "non motivated" sensory mapping trials at the end of the session.
# with open(trial_indices_sensory_map_yaml, 'r') as stream:
#     trial_indices_sensory_map = yaml.load(stream, yaml.Loader)
# trial_indices_sensory_map = pd.DataFrame(trial_indices_sensory_map.items(), columns=['session_id', 'trial_idx'])


for mouse in mice_list:
    save_dir = os.path.join(processed_data_dir, 'mice', mouse)
    # Check if dataset is already created.
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'tensor_xarray_learning_data.nc')
    # if os.path.exists(save_path_data):
    #     continue
    session_nwb = [nwb for nwb in nwb_list if mouse in nwb]

    # Get data and metadata for each session.
    sessions = []
    data = []
    metadatas = []
    kept_trial_ids = []  # (session_id, trial_id) kept per session, in tensor order

    for nwb_file in session_nwb:
        session_id = nwb_file[-25:-4]
        sessions.append(session_id)

        # Parameters for tensor array.
        cell_types = ['na', 'wM1', 'wS2']
        rrs_keys = ['ophys', 'fluorescence_all_cells', 'dff']
        time_range = (2, 7)
        epoch_name = None
        trial_selection = None

        idx_selection = trial_indices.loc[trial_indices.session_id == session_id, 'trial_idx'].values[0]
        # idx_sensory_map = trial_indices_sensory_map.loc[trial_indices_sensory_map.session_id==session_id, 'trial_idx'].values[0]

        # Generate a 3d array containing all trial types.
        print(f'Processing {session_id} {trial_selection}')
        traces, metadata = make_events_aligned_array_3d(
            nwb_file, rrs_keys, time_range, trial_selection, epoch_name, cell_types, idx_selection
        )

        # Drop trials whose event window extends beyond the recording
        # boundary (align_array_to_events leaves these entirely NaN across
        # all cells and timepoints rather than dropping them). Keep
        # metadata['trials'] (trial IDs) in sync so behav_table can be
        # filtered/reordered to match below -- behav_table is built
        # independently (via stop_flag_yaml) and assigned onto the tensor
        # by position, so it must exactly match which trials survive here.
        all_nan_trials = np.isnan(traces).all(axis=(0, 2))
        if all_nan_trials.any():
            print(f'  Dropping {all_nan_trials.sum()} trial(s) with event window beyond recording boundary.')
            traces = traces[:, ~all_nan_trials, :]
            metadata['trials'] = metadata['trials'][~all_nan_trials]

        data.append(traces)
        metadatas.append(metadata)
        kept_trial_ids.append(pd.DataFrame({'session_id': session_id, 'trial_id': metadata['trials']}))

    # Sessions are concatenated on the trial dim. Different sessions can
    # yield a slightly different number of timepoints (per-session sampling
    # rate jitter rounds differently in align_array_to_timestamps), so
    # truncate all sessions to the shortest common time length first.
    n_t_per_session = [d.shape[2] for d in data]
    min_n_t = min(n_t_per_session)
    if len(set(n_t_per_session)) > 1:
        print(
            f'  Truncating sessions to common length: {min_n_t} timepoints '
            f'(session lengths: {n_t_per_session})'
        )
        data = [d[:, :, :min_n_t] for d in data]
    tensor = np.concatenate(data, axis=1)
    kept_trial_ids = pd.concat(kept_trial_ids, ignore_index=True)

    # Load trial table and compute performance for those sessions, then
    # filter+reorder it to exactly the trials kept in the tensor above
    # (same session_id/trial_id, same order -- required since it gets
    # assigned onto the 'trial' dim by position, not by an explicit join).
    print('Make behavior table')
    behav_table = make_behavior_table(
        session_nwb,
        sessions,
        db_path,
        cut_session=True,
        stop_flag_yaml=stop_flag_yaml,
        trial_indices_yaml=trial_indices_yaml,
    )
    behav_table = behav_table.set_index(['session_id', 'trial_id'])
    behav_table = behav_table.loc[list(kept_trial_ids.itertuples(index=False, name=None))].reset_index()

    time = np.linspace(-time_range[0], time_range[1], tensor.shape[2])
    # Create xarray.
    ds = xr.DataArray(
        tensor,
        dims=['cell', 'trial', 'time'],
        coords={
            'roi': ('cell', metadata['rois']),
            'cell_type': ('cell', metadata['cell_types']),
            'time': time,
        },
    )
    for col in behav_table.columns:
        ds[col] = ('trial', behav_table[col].values)
    ds.attrs['session_ids'] = sessions
    ds.attrs['mouse_id'] = mouse

    # Save dataset.
    print(f'Saving {mouse}')
    ds.to_netcdf(save_path)


# =============================================================================
# Create mice tensors with xarrays for mapping trials.
# =============================================================================

# Get directories and files.
db_path = paths.db_path
nwb_path = paths.nwb_dir
trial_indices_sensory_map_yaml = paths.trial_indices_sensory_map_yaml
stop_flag_sensory_map_yaml = paths.stop_flags_sensory_map_yaml
processed_data_dir = paths.processed_dir

days = ['-2', '-1', '0', '+1', '+2']
_, nwb_list, mice_list, _ = database.select_sessions_from_db(
    db_path,
    nwb_path,
    exclude_cols=['exclude', 'two_p_exclude'],
    experimenters=['AR', 'GF', 'MI'],
    day=days,
    two_p_imaging='yes',
)

# For "non motivated" sensory mapping trials at the end of the session.
with open(trial_indices_sensory_map_yaml, 'r') as stream:
    trial_indices = yaml.load(stream, yaml.Loader)
trial_indices = pd.DataFrame(trial_indices.items(), columns=['session_id', 'trial_idx'])

# mice_list = [mouse for mouse in mice_list if 'AR163'==mouse]
# mice_list = ['AR144']

for mouse in mice_list:
    save_dir = os.path.join(processed_data_dir, 'mice', mouse)
    # Check if dataset is already created.
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'tensor_xarray_mapping_data.nc')
    # if os.path.exists(save_path_data):
    #     continue
    session_nwb = [nwb for nwb in nwb_list if mouse in nwb]

    # Get data and metadata for each session.
    sessions = []
    data = []
    metadatas = []
    behavior_days = []

    for nwb_file in session_nwb:
        session_id = nwb_file[-25:-4]
        sessions.append(session_id)

        # Parameters for tensor array.
        cell_types = ['na', 'wM1', 'wS2']
        rrs_keys = ['ophys', 'fluorescence_all_cells', 'dff']
        time_range = (2, 7)
        epoch_name = None
        trial_selection = None

        idx_selection = trial_indices.loc[trial_indices.session_id == session_id, 'trial_idx'].values[0]

        # Generate a 3d array containing all trial types.
        print(f'Processing {session_id} {trial_selection}')
        traces, metadata = make_events_aligned_array_3d(
            nwb_file, rrs_keys, time_range, trial_selection, epoch_name, cell_types, idx_selection
        )

        print(f'{np.isnan(traces).sum()} nan values in tensor.')

        # Drop trials whose event window extends beyond the recording
        # boundary: align_array_to_events leaves these entirely NaN (all
        # cells, all timepoints) rather than dropping them, which happens
        # occasionally for mapping trials since they are collected at the
        # end of the session. Same idiom as imaging.extract_trials().
        all_nan_trials = np.isnan(traces).all(axis=(0, 2))
        if all_nan_trials.any():
            print(f'  Dropping {all_nan_trials.sum()} trial(s) with event window beyond recording boundary.')
            traces = traces[:, ~all_nan_trials, :]

        data.append(traces)
        metadatas.append(metadata)
        # Get session day.
        # I don't load the trial table with the mapping trials, so get it for
        # nwb file.
        with NWBSession(nwb_file) as nwb_session:
            _, d = nwb_session.petersen.get_bhv_type_and_training_day_index()
        behavior_days.extend([d for _ in range(traces.shape[1])])

    # Sessions are concatenated on the trial dim. Different sessions can
    # yield a slightly different number of timepoints (per-session sampling
    # rate jitter rounds differently in align_array_to_timestamps), so
    # truncate all sessions to the shortest common time length first.
    n_t_per_session = [d.shape[2] for d in data]
    min_n_t = min(n_t_per_session)
    if len(set(n_t_per_session)) > 1:
        print(
            f'  Truncating sessions to common length: {min_n_t} timepoints '
            f'(session lengths: {n_t_per_session})'
        )
        data = [d[:, :, :min_n_t] for d in data]
    tensor = np.concatenate(data, axis=1)

    time = np.linspace(-time_range[0], time_range[1], tensor.shape[2])
    # Create xarray.
    ds = xr.DataArray(
        tensor,
        dims=['cell', 'trial', 'time'],
        coords={
            'roi': ('cell', metadata['rois']),
            'cell_type': ('cell', metadata['cell_types']),
            'time': time,
            'day': ('trial', behavior_days),
        },
    )
    ds.attrs['session_ids'] = sessions
    ds.attrs['mouse_id'] = mouse

    # Save dataset.
    print(f'Saving {mouse}')
    ds.to_netcdf(save_path)


# #############################################################################
#  Load xarrays and substract baseline.
# #############################################################################

# Get directories and files.
db_path = paths.db_path
nwb_path = paths.nwb_dir
trial_indices_sensory_map_yaml = paths.trial_indices_sensory_map_yaml
stop_flag_sensory_map_yaml = paths.stop_flags_sensory_map_yaml
processed_data_dir = paths.processed_dir
days = ['-2', '-1', '0', '+1', '+2']
sampling_rate = 30  # Hz, for imaging data.
baseline_win = (0, 1)
baseline_win = (int(baseline_win[0] * sampling_rate), int(baseline_win[1] * sampling_rate))

_, nwb_list, mice_list, _ = database.select_sessions_from_db(
    db_path,
    nwb_path,
    exclude_cols=['exclude', 'two_p_exclude'],
    experimenters=['AR', 'GF', 'MI'],
    day=days,
    two_p_imaging='yes',
)

# mice_list = ['GF305',]
for mouse_id in mice_list:
    reward_group = database.get_mouse_reward_group_from_db(paths.db_path, mouse_id)

    file_name = 'tensor_xarray_learning_data.nc'
    folder = os.path.join(paths.processed_dir, 'mice')
    xarr = imaging.load_mouse_xarray(mouse_id, folder, file_name, subtracted=False)
    xarr = imaging.subtract_baseline(xarr, 2, baseline_win)

    # Save the xarray.
    save_path = os.path.join(folder, mouse_id, 'tensor_xarray_learning_data_baselinesubstracted.nc')
    xarr.to_netcdf(save_path)

    file_name = 'tensor_xarray_mapping_data.nc'
    folder = os.path.join(paths.processed_dir, 'mice')

    xarr = imaging.load_mouse_xarray(mouse_id, folder, file_name, subtracted=False)
    xarr = imaging.subtract_baseline(xarr, 2, baseline_win)
    # Save the xarray.
    save_path = os.path.join(folder, mouse_id, 'tensor_xarray_mapping_data_baselinesubstracted.nc')
    xarr.to_netcdf(save_path)


# #############################################################################
# Lick-aligned xarrays.
# #############################################################################

db_path = paths.db_path
db = database.read_excel_db(db_path)
nwb_path = paths.nwb_dir
processed_dir = os.path.join(paths.processed_dir, 'mice')

days = ['-2', '-1', '0', '+1', '+2']
_, _, mice_list, _ = database.select_sessions_from_db(
    db_path,
    nwb_path,
    exclude_cols=['exclude', 'two_p_exclude'],
    experimenters=['AR', 'GF', 'MI'],
    day=days,
    two_p_imaging='yes',
)

# mouse = 'GF305'
# mice_list = mice_list[-8:]
# mice_list = mice_list[-8:]

for mouse in mice_list:
    print(f'Processing lick aligned array for {mouse}')
    file_name = 'tensor_xarray_learning_data.nc'
    xarray = imaging.load_mouse_xarray(mouse, processed_dir, file_name, subtracted=True)
    rew_gp = database.get_mouse_reward_group_from_db(db_path, mouse, db)

    lick_times = xarray.coords['lick_time'].values
    stim_onset = xarray.coords['stim_onset'].values
    reaction_time = lick_times - stim_onset
    # GF333 and GF334 have a few strange reaction times.
    reaction_time[reaction_time >= 1.25] = np.nan
    # plt.plot(reaction_time)
    # plt.show()

    # Create a new xarray for lick-aligned traces
    time = xarray.coords['time'].values
    aligned_time = np.linspace(-1, 5, 180)
    aligned_traces = []

    for itrial in range(xarray.shape[1]):
        lick_onset = reaction_time[itrial]
        # Trials with no lick are set to nan.
        if np.isnan(lick_onset):
            aligned_traces.append(np.full((xarray.shape[0], 180), np.nan))
            continue
        lick_onset_idx = (np.abs(time - lick_onset)).argmin()
        data = xarray[:, itrial, :].values
        aligned_data = data[:, lick_onset_idx - 30 : lick_onset_idx + 150]

        # Two samples missing sometimes for late licks.
        if aligned_data.shape[1] < 180:
            pad_width = 180 - aligned_data.shape[1]
            pad_values = np.repeat(aligned_data[:, -1][:, np.newaxis], pad_width, axis=1)
            aligned_data = np.concatenate([aligned_data, pad_values], axis=1)
        aligned_traces.append(aligned_data)

    aligned_traces = np.stack(aligned_traces, axis=1)
    aligned_traces.shape
    aligned_xarray = xr.DataArray(
        aligned_traces,
        dims=['cell', 'trial', 'time'],
        coords={
            'time': ('time', aligned_time),
            'reaction_time': ('trial', reaction_time),
        },
    )

    # Add all other coordinates from the original xarray
    for coord in xarray.coords:
        if coord not in aligned_xarray.coords:
            aligned_xarray.coords[coord] = xarray.coords[coord]

    # Save the aligned xarray
    save_path = os.path.join(processed_dir, mouse, 'lick_aligned_xarray.nc')
    aligned_xarray.to_netcdf(save_path)
    print(f'Saved {save_path}')
