"""Pipeline step 03: trial-level behaviour tables.

Same session selections and make_behavior_table() calls as the original
analysis. Each table is written to <processed_dir>/behavior/:

    behavior_imagingmice_table_5days_cut.csv   Imaging mice, days -2..+2, sessions cut
                                               at the stop flag (Figs 1, 4, S1).
    behavior_particle_test.csv                 Particle-test sessions of imaging mice,
                                               uncut (S1c).
    behavior_opto_learning.csv                 Optogenetic inactivation during learning
                                               (Fig 2e-f).
    behavior_muscimol.csv                      Muscimol inactivation during learning
                                               (Fig 2b-c).
    behavior_muscimol_execution.csv            Muscimol inactivation in expert mice
                                               (execution revision of Fig 2b-c).

Learning curves are fitted separately by 04_learning_curves.py, which reads
the first table and writes behavior_imagingmice_table_5days_cut_with_learning_curves.csv.

Reads NWB files, the session database and the stop-flag YAML files; never
writes to them.

Usage:
    python pipeline/03_behavior_tables.py [--tables imaging particle opto muscimol muscimol_execution]
"""

import argparse
import os

import numpy as np

from fast_learning import paths, database
from fast_learning.behavior import make_behavior_table


OUTPUT_DIR = os.path.join(paths.processed_dir, 'behavior')
DAYS = ['-2', '-1', '0', '+1', '+2']
PARTICLE_TEST_DAYS = ['whisker_on_1', 'whisker_off', 'whisker_on_2']
MUSCIMOL_DAYS = [
    'pre_-2',
    'pre_-1',
    'muscimol_1',
    'muscimol_2',
    'muscimol_3',
    'recovery_1',
    'recovery_2',
    'recovery_3',
]
OPTO_DAYS = ['pre_-2', 'pre_-1', 'opto', 'recovery_1']


def _build(session_list, nwb_list, cut_session):
    return make_behavior_table(
        nwb_list,
        session_list,
        paths.db_path,
        cut_session=cut_session,
        stop_flag_yaml=paths.stop_flags_yaml,
        trial_indices_yaml=paths.trial_indices_yaml,
    )


def _save(table, file_name):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, file_name)
    table.to_csv(path, index=False)
    print(f'Saved {path} ({table.session_id.nunique()} sessions, {table.mouse_id.nunique()} mice)')


def imaging_mice_table():
    mice = database.select_mice_from_db(
        paths.db_path,
        paths.nwb_dir,
        experimenters=None,
        exclude_cols=['exclude', 'two_p_exclude'],
        optogenetic=['no', np.nan],
        pharmacology=['no', np.nan],
        two_p_imaging='yes',
    )
    session_list, nwb_list, _, _ = database.select_sessions_from_db(
        paths.db_path,
        paths.nwb_dir,
        experimenters=None,
        exclude_cols=['exclude', 'two_p_exclude'],
        day=DAYS,
        mouse_id=mice,
    )
    _save(_build(session_list, nwb_list, cut_session=True), 'behavior_imagingmice_table_5days_cut.csv')


def particle_test_table():
    mice = database.select_mice_from_db(
        paths.db_path,
        paths.nwb_dir,
        experimenters=None,
        exclude_cols=['exclude'],
        day=PARTICLE_TEST_DAYS,
        optogenetic=['no', np.nan],
        pharmacology=['no', np.nan],
        two_p_imaging='yes',
    )
    session_list, nwb_list, _, _ = database.select_sessions_from_db(
        paths.db_path,
        paths.nwb_dir,
        experimenters=None,
        exclude_cols=['exclude', 'two_p_exclude'],
        day=PARTICLE_TEST_DAYS,
        mouse_id=mice,
    )
    _save(_build(session_list, nwb_list, cut_session=False), 'behavior_particle_test.csv')


def opto_learning_table():
    mice = database.select_mice_from_db(
        paths.db_path,
        paths.nwb_dir,
        experimenters=None,
        exclude_cols=['exclude', 'opto_exclude'],
        opto_inactivation_type=['learning'],
        optogenetic='yes',
    )
    session_list, nwb_list, _, _ = database.select_sessions_from_db(
        paths.db_path,
        paths.nwb_dir,
        experimenters=None,
        exclude_cols=['exclude', 'opto_exclude'],
        opto_inactivation_type=['learning'],
        opto_day=OPTO_DAYS,
        mouse_id=mice,
    )
    _save(_build(session_list, nwb_list, cut_session=True), 'behavior_opto_learning.csv')


def muscimol_learning_table():
    # The original generation code was commented out in behavior.py; this
    # restores it with the same selection.
    mice = database.select_mice_from_db(
        paths.db_path, paths.nwb_dir, experimenters=None, exclude_cols=['exclude'], pharmacology='yes'
    )
    session_list, nwb_list, _, _ = database.select_sessions_from_db(
        paths.db_path,
        paths.nwb_dir,
        experimenters=None,
        exclude_cols=['exclude'],
        pharma_inactivation_type=['learning'],
        pharma_day=MUSCIMOL_DAYS,
        mouse_id=mice,
    )
    _save(_build(session_list, nwb_list, cut_session=True), 'behavior_muscimol.csv')


EXECUTION_DAYS = ['pre_-2', 'pre_-1', 'muscimol_1', 'ringer_1', 'muscimol_2']


def muscimol_execution_table():
    # Moved from revisions/figure_2b_c_execution.py, same selection.
    session_list, nwb_list, mice_list, _ = database.select_sessions_from_db(
        paths.db_path,
        paths.nwb_dir,
        experimenters=None,
        exclude_cols=['exclude'],
        pharma_inactivation_type=['execution'],
        pharma_day=EXECUTION_DAYS,
    )
    print(f"Execution sessions: {len(session_list)}, {len(mice_list)} mice")
    _save(_build(session_list, nwb_list, cut_session=True), 'behavior_muscimol_execution.csv')


TABLES = {
    'imaging': imaging_mice_table,
    'particle': particle_test_table,
    'opto': opto_learning_table,
    'muscimol': muscimol_learning_table,
    'muscimol_execution': muscimol_execution_table,
}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        '--tables',
        nargs='+',
        choices=list(TABLES),
        default=list(TABLES),
        help='tables to build (default: all)',
    )
    args = parser.parse_args()
    for name in args.tables:
        print(f'\n=== {name} ===')
        TABLES[name]()
