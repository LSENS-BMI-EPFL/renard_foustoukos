"""Session metadata table: selecting sessions and mice.

The table has one row per recording session with at least the columns
``mouse_id``, ``session_id``, ``day`` and ``reward_group``, plus the
exclusion and experiment-type columns used as filters (``exclude``,
``two_p_exclude``, ``two_p_imaging``, ``optogenetic``, ``pharmacology``, ...).
It is read from ``paths.db_path``, as CSV or Excel.
"""

import os
import warnings

import pandas as pd
import yaml


def read_excel_db(db_path):
    """Read the session metadata table (.csv or .xlsx) with ``day`` as str.

    Fully empty rows are dropped.
    """
    if str(db_path).endswith('.csv'):
        database = pd.read_csv(db_path, dtype={'day': str})
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            database = pd.read_excel(db_path, converters={'day': str})
    return database.loc[~database.isna().all(axis=1)]


def _filter(db, exclude_cols, filters):
    for key, val in filters.items():
        if isinstance(val, list):
            db = db.loc[db[key].isin(val)]
        else:
            db = db.loc[db[key] == val]
    for col in exclude_cols:
        db = db.loc[db[col] != 'exclude']
    return db


def select_sessions_from_db(
    db_path, nwb_path, experimenters=None, exclude_cols=['exclude', 'two_p_exclude'], **filters
):
    """Select sessions matching column filters.

    Args:
        db_path: Path to the session metadata table.
        nwb_path: Folder containing the NWB files.
        experimenters: Optional list of experimenter prefixes (first two
            characters of the mouse id) to keep.
        exclude_cols: Columns in which the value 'exclude' drops a session.
        **filters: column=value or column=[values] conditions.

    Returns:
        (session_ids, nwb_paths, mouse_ids, filtered_table)
    """
    db = _filter(read_excel_db(db_path), exclude_cols, filters)
    mice_list = list(db.mouse_id.unique())
    session_list = list(db.session_id)
    if experimenters:
        session_list = [s for s in session_list if s[:2] in experimenters]
        mice_list = [m for m in mice_list if m[:2] in experimenters]
    nwb_paths = [os.path.join(nwb_path, f + '.nwb') for f in session_list]
    return session_list, nwb_paths, mice_list, db


def select_mice_from_db(
    db_path, nwb_path, experimenters=None, exclude_cols=['exclude', 'two_p_exclude'], **filters
):
    """Select mice with at least one session matching the filters.

    Same arguments as select_sessions_from_db; returns a list of mouse ids.
    """
    db = _filter(read_excel_db(db_path), exclude_cols, filters)
    mice_list = list(db.mouse_id.unique())
    if experimenters:
        mice_list = [m for m in mice_list if m[:2] in experimenters]
    return mice_list


def get_reward_group_from_db(db_path, session_id):
    """Reward group ('R+' or 'R-') of a session."""
    db = read_excel_db(db_path)
    return db.loc[db['session_id'] == session_id, 'reward_group'].values[0]


def get_mouse_reward_group_from_db(db_path, mouse_id, db=None):
    """Reward group ('R+' or 'R-') of a mouse. Pass db to avoid re-reading."""
    if db is None:
        db = read_excel_db(db_path)
    return db.loc[db['mouse_id'] == mouse_id, 'reward_group'].values[0]


def read_stop_flags_and_indices_yaml(stop_flag_yaml_path, trial_indices_path):
    """Read the stop-flag and trial-index YAML files.

    Returns:
        (stop_flags, trial_indices): dict session_id -> (start, stop), and a
        DataFrame with columns session_id, trial_idx.
    """
    with open(stop_flag_yaml_path, 'r') as file:
        stop_flags = yaml.load(file, Loader=yaml.FullLoader)
    with open(trial_indices_path, 'r') as file:
        trial_indices = yaml.load(file, Loader=yaml.FullLoader)
    trial_indices = pd.DataFrame(trial_indices.items(), columns=['session_id', 'trial_idx'])
    return stop_flags, trial_indices
