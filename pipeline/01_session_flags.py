"""Pipeline step 01: session stop flags and trial indices.

For each session, the (start, stop) trial range to analyse and the matching
trial ids:
    - end of session: stop three trials before the first run of three auditory
      misses followed by at most two auditory hits (disengagement);
    - sensory mapping: the last 50 unrewarded whisker trials of the passive
      mapping block at the end of imaging sessions.
Session-specific exceptions are listed in the code.

The published YAML files are part of the dataset (config: stop_flags_dir);
this step regenerates them for checking, it is not run by default.

Inputs:  NWB files, session metadata.
Outputs: <processed_dir>/stop_flags/{stop_flags,trial_indices}_{end_session,sensory_map}.yaml

Usage:
    python pipeline/01_session_flags.py
"""

import os

import yaml
import matplotlib.pyplot as plt
import pandas as pd
from cicada_nwb import NWBSession

from fast_learning import paths, database


# =============================================================================
# Path configuation.
# =============================================================================

nwb_dir = paths.nwb_dir
db_path = paths.db_path


def get_trial_table(nwb_file):
    """Trial table of one NWB file (index 'id', one row per trial)."""
    with NWBSession(nwb_file) as session:
        return session.behavior.get_trial_table()


# =============================================================================
# Generate stop flags for the whisker sensory mapping block.
# =============================================================================
# Imaging sessions have a block of 50 whisker trials at the end of the session.
# Find the trial index of the first whisker trial of this block.

# List nwb files.
session_list, nwb_list, mice, db_filtered = database.select_sessions_from_db(
    db_path, nwb_dir, two_p_imaging='yes', sensory_mapping='yes', exclude_cols=['exclude']
)

stop_flags = {}
trial_indices = {}
trial_count = []
# mouse_ids = ['AR176', 'AR177', 'AR178', 'AR179', 'AR180', ]

for nwb_file in nwb_list:
    mouse_id = nwb_file[-25:-20]
    # if mouse_id not in mouse_ids:
    #     continue
    session_id = nwb_file[-25:-4]
    table = get_trial_table(nwb_file)
    print(f"Processing {session_id} ({nwb_list.index(nwb_file) + 1}/{len(nwb_list)})")

    # Beginning of the whisker sensory mapping block.
    # Find final segment of one's in the whisker_stim column.

    start = table.whisker_stim != table.whisker_stim.shift()
    start = int(start.cumsum().idxmax())

    # Exceptions.
    if session_id == 'AR132_20240427_122605':
        start = 571
    if session_id == 'AR139_20240428_180459':
        start = 406
    if session_id == 'GF350_28052021_145113':
        start = 224
    if session_id == 'GF340_29012021_091839':
        start = 364
    if session_id == 'GF340_30012021_125418':
        start = 389
    if session_id == 'GF340_31012021_124052':
        start = 519
    if session_id == 'GF340_01022021_131205':
        start = 677
    if session_id == 'GF340_02022021_100218':
        start = 717
    if session_id == 'GF307_19112020_083908':
        start = 220
    if session_id == 'GF333_25012021_141608':
        start = 254
    if session_id == 'GF350_28052021_145113':
        start = 214
    if session_id == 'MI075_21122021_151949':
        start = 189
    if session_id == 'MI075_23122021_150004':
        start = 414
    if session_id == 'AR133_20240428_134911':
        start = 333
    if session_id == 'AR144_20240522_190834':
        start = 341

    # Select the index that leaves 50 whisker misses.
    try:
        start = (
            table.loc[(table.whisker_stim == 1) & (table.lick_flag == 0) & (table.index >= start)]
            .iloc[-50:]
            .index[0]
        )
        stop = start + 49  # Because the last trial is included with .loc.

    # Cases with no whisker stim at the end (few excluded mice).
    except IndexError:
        start = table.index.max()
        stop = table.index.max()

    stop_flags[session_id] = (int(start), int(stop))

    # Further exceptions where I included a trial with a lick to avoid
    # having less than 50 whisker trials.
    if session_id == 'AR133_20240425_115233':
        stop_flags[session_id] = (int(338), int(table.index.max()))
    if session_id == 'AR137_20240425_170755':
        stop_flags[session_id] = (int(207), int(table.index.max()))
    if 'AR127' in session_id:
        # For 4 out of 5 sessions, I manual stopped the sessions too early
        # from last trial onset.
        start = start - 1
        stop = stop - 1
        stop_flags[session_id] = (int(start), int(stop))

    # Get the indices of those trials.
    trial_ids = table.loc[start:stop, 'trial_id'].to_list()
    trial_indices[session_id] = trial_ids

    # Count trial for sanity check.
    n_wh_miss = table.loc[
        (table.trial_id >= start)
        & (table.trial_id <= stop)
        & (table.whisker_stim == 1)
        & (table.lick_flag == 0)
    ].shape[0]
    n_wh_hit = table.loc[
        (table.trial_id >= start)
        & (table.trial_id <= stop)
        & (table.whisker_stim == 1)
        & (table.lick_flag == 1)
    ].shape[0]
    trial_count.append([mouse_id, session_id, start, stop, n_wh_miss, n_wh_hit])

# Save yaml files. Written under the output root; the yaml files in
# mice_info/stop_flags are read-only inputs.
yaml_dir = os.path.join(paths.processed_dir, 'stop_flags')
os.makedirs(yaml_dir, exist_ok=True)
yaml_save = os.path.join(yaml_dir, 'stop_flags_sensory_map.yaml')
with open(yaml_save, 'w') as stream:
    yaml.safe_dump(stop_flags, stream)
yaml_save = os.path.join(yaml_dir, 'trial_indices_sensory_map.yaml')
with open(yaml_save, 'w') as stream:
    yaml.safe_dump(trial_indices, stream)

# Sanity check.
trial_count = pd.DataFrame(
    trial_count, columns=['mouse_id', 'session_id', 'start', 'stop', 'n_wh_miss', 'n_wh_hit']
)

nwb_file = '//sv-nas1.rcp.epfl.ch/Petersen-Lab/analysis/Anthony_Renard/NWB/AR127_20240221_133407.nwb'
nwb_file = paths.adjust_path_to_host(nwb_file)
table = get_trial_table(nwb_file)
table = table.reset_index()
table = table.loc[(table.trial_id >= 250)]
plt.figure()
plt.scatter(table.loc[table.auditory_stim == 1, 'trial_id'], table.loc[table.auditory_stim == 1, 'lick_flag'])
plt.scatter(table.loc[table.whisker_stim == 1, 'trial_id'], table.loc[table.whisker_stim == 1, 'lick_flag'])
plt.scatter(table.loc[table.no_stim == 1, 'trial_id'], table.loc[table.no_stim == 1, 'lick_flag'])


# =============================================================================
# Generate session stop flags.
# =============================================================================
# The criteria to cut the session is the mouse is not performing the auditory
# defined as three auditory misses in a row and no more than two auditory
# hits in the rest.

# List nwb files.
nwb_list = database.read_excel_db(db_path)
nwb_list = nwb_list.loc[(nwb_list['exclude'] != 'exclude')]
nwb_list = list(nwb_list.session_id)
nwb_list = [os.path.join(nwb_dir, f + '.nwb') for f in nwb_list]
# The database also lists sessions without an NWB file; skip them.
missing = [f for f in nwb_list if not os.path.exists(f)]
nwb_list = [f for f in nwb_list if os.path.exists(f)]
print(f"{len(missing)} sessions in the database have no NWB file: skipped.")

stop_flags = {}
trial_indices = {}
# mouse_ids = ['AR176', 'AR177', 'AR178', 'AR179', 'AR180', ]

for nwb_file in nwb_list:
    mouse_id = nwb_file[-25:-20]
    # if mouse_id not in mouse_ids:
    #     continue
    session_id = nwb_file[-25:-4]
    table = get_trial_table(nwb_file)
    print(f"\rProcessing {session_id} ({nwb_list.index(nwb_file) + 1}/{len(nwb_list)})", end="")

    # If a session does not contain auditory trials, keep whole session.
    # This happens for some test sessions of the GF mice.
    if table.auditory_stim.sum() == 0:
        stop = table.trial_id.idxmax()
        stop_flags[session_id] = (0, int(stop))
        trial_indices[session_id] = table['trial_id'].to_list()
        continue

    # Get index after which no more than two hits until session end.
    two_hits_left = table.loc[table.auditory_stim == 1, 'lick_flag']
    two_hits_left = (two_hits_left.cumsum() >= (two_hits_left.sum() - 2)).idxmax()
    # Find three auditory misses in a row.
    three_aud_misses = table.loc[table.auditory_stim == 1, 'lick_flag']
    three_aud_misses = three_aud_misses.rolling(window=3).sum() == 0
    # Stop flag is the first occurence of three auditory misses in a row
    # followed by no more than three hits in the rest of the session.
    # In case no three auditory misses keep the whole session.
    if three_aud_misses.sum() == 0:
        stop = table.loc[table.auditory_stim == 1, 'trial_id'].idxmax()
    else:
        stop = three_aud_misses.loc[three_aud_misses.index > two_hits_left].idxmax()
        # Stop three trials before the three auditory misses.
        position = three_aud_misses.index.get_loc(stop)
        stop = three_aud_misses.index[position - 3]

    # Exceptions.
    if session_id == 'AR115_20231116_142507':
        stop = 35
    stop_flags[session_id] = (0, int(stop))

    # Get the indices of those trials.
    trial_ids = table.loc[(table.index <= stop), 'trial_id'].to_list()
    trial_indices[session_id] = trial_ids

# Save yaml files.
yaml_save = os.path.join(yaml_dir, 'stop_flags_end_session.yaml')
with open(yaml_save, 'w') as stream:
    yaml.dump(stop_flags, stream)
yaml_save = os.path.join(yaml_dir, 'trial_indices_end_session.yaml')
with open(yaml_save, 'w') as stream:
    yaml.dump(trial_indices, stream)
