"""Event-aligned two-photon tensor construction, built on cicada_nwb/cicada_analysis.

Replaces the equivalent functionality previously provided by the (now
unmaintained) NWB_analysis package (analysis.psth_analysis.make_events_aligned_array_3d).
"""

import numpy as np

from cicada_nwb import NWBSession
from cicada_analysis.cicada_tools.core import (
    align_array_to_events,
    filter_events_based_on_epochs,
    find_nearest,
)

# Nominal frame rate of all recordings (imaging_rate in the NWB files). Event
# windows are converted to frames with it, rather than with each session's
# median frame interval: the AR sessions' timestamps are quantised to 0.2 ms,
# so that median is 33.2 or 33.4 ms (30.12 or 29.94 Hz) at random, which put
# the stimulus one frame apart between sessions.
FRAME_RATE = 30.0


def window_frames(time_range):
    """(frames before, frames after) the stimulus for time_range = (s before, s after)."""
    return int(round(time_range[0] * FRAME_RATE)), int(round(time_range[1] * FRAME_RATE))


def time_axis(time_range):
    """Time of each frame (s) relative to the stimulus frame, which is at exactly 0."""
    n_pre, n_post = window_frames(time_range)
    return np.arange(-n_pre, n_post + 1) / FRAME_RATE


def _select_events(session, trial_selection, epoch_name, trial_idx):
    df = session.behavior.get_trial_table().reset_index()
    if trial_idx is not None:
        df = df.loc[df['id'].isin(trial_idx)]
    if trial_selection:
        for col, allowed in trial_selection.items():
            col_type = type(df[col].values[0])
            df = df.loc[df[col].isin([col_type(v) for v in allowed])]
    if df.empty:
        return None, None

    trial_ids = df['trial_id'].values
    col0 = 'stim_onset' if 'stim_onset' in df.columns else 'start_time'
    events = df[col0].values

    if epoch_name:
        epochs = session.behavior.get_behavioral_epochs_times(epoch_name)
        if epochs is not None and len(epochs) > 0:
            events = filter_events_based_on_epochs(events, epochs)

    return events, trial_ids


def make_events_aligned_array_3d(
    nwb_path, rrs_keys, time_range, trial_selection, epoch_name, cell_types, trial_idx_table=None
):
    """Generate, for a single nwb file, a 3d array of activity aligned on
    trial-table events. Cell types are stacked along the first dimension.

    Returns:
        (numpy.ndarray, dict): (n_cells, n_events, n_t) array of aligned
        activity, and a metadata dict with 'mice', 'rois', 'cell_types', 'trials'.
    """

    metadata = {'mice': [], 'rois': [], 'cell_types': [], 'trials': []}

    mouse_id = nwb_path[-25:-20]
    session_id = nwb_path[-25:-4]
    print(f"\rProcessing {session_id}")

    if trial_idx_table is None:
        trial_idx = None
    elif isinstance(trial_idx_table, list):
        trial_idx = trial_idx_table
    else:
        trial_idx = trial_idx_table.loc[trial_idx_table.session_id == session_id, 'trial_idx'].values[0]

    with NWBSession(nwb_path) as session:
        events, trial_ids = _select_events(session, trial_selection, epoch_name, trial_idx)
        if events is None:
            print(f'Session {session_id} has no events in this trial type.')
            return None, None

        activity = session.calcium_imaging.get_roi_response_serie_data(rrs_keys)
        activity_ts = session.calcium_imaging.get_roi_response_serie_timestamps(rrs_keys)
        cell_type_dict = session.calcium_imaging.get_cell_indices_by_cell_type(rrs_keys)
        if not cell_type_dict:
            cell_type_dict = {'na': np.arange(activity.shape[0])}

        # Stimulus frame of each event: the frame nearest to it in time.
        event_frames = [find_nearest(activity_ts, t) for t in events]

        ct_arrays = []
        for cell_type in cell_types:
            if cell_type in cell_type_dict:
                rois = cell_type_dict[cell_type]
                activity_aligned = align_array_to_events(
                    activity[rois], event_frames, window_frames(time_range)
                )
                ct_arrays.append(activity_aligned)
                metadata['mice'].extend([mouse_id] * activity_aligned.shape[0])
                metadata['rois'].extend(rois)
                metadata['cell_types'].extend([cell_type] * activity_aligned.shape[0])

    data = np.concatenate(ct_arrays, axis=0)
    metadata['trials'].extend(trial_ids)
    for key, val in metadata.items():
        metadata[key] = np.array(val)

    return data, metadata
