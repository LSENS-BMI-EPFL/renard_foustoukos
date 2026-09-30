"""Event-aligned two-photon tensor construction, built on cicada_nwb/cicada_analysis.

Replaces the equivalent functionality previously provided by the (now
unmaintained) NWB_analysis package (analysis.psth_analysis.make_events_aligned_array_3d).
"""

import numpy as np

from cicada_nwb import NWBSession
from cicada_analysis.cicada_tools.core import align_array_to_timestamps, filter_events_based_on_epochs


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

        ct_arrays = []
        for cell_type in cell_types:
            if cell_type in cell_type_dict:
                rois = cell_type_dict[cell_type]
                activity_aligned = align_array_to_timestamps(
                    activity[rois], events, activity_ts, window_s=time_range
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
