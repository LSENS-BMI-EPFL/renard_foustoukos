"""Check that the reward group recorded in each NWB file (wh_reward in the
session metadata) matches the reward group in the session database.

Prints the sessions where the two disagree (an empty table means none).
"""

import os

import pandas as pd
from cicada_nwb import NWBSession

from fast_learning import paths, database


# #############################################################################
# Check that the NWB reward group matches the session database.
# #############################################################################

sessions, nwb_files, mice, db = database.select_sessions_from_db(
    paths.db_path, paths.nwb_dir, exclude_cols=['exclude']
)

# Read groups from db.
db_reward_groups = db[['mouse_id', 'session_id', 'reward_group']].drop_duplicates()

# Read groups from nwb files.
nwb_reward_groups = []
missing = []

for session, nwb_file in zip(sessions, nwb_files):
    if not os.path.exists(nwb_file):
        missing.append(session)
        continue
    with NWBSession(nwb_file) as nwb_session:
        metadata = nwb_session.petersen.get_session_metadata()
    g = 'R+' if metadata['wh_reward'] == 1 else 'R-'
    nwb_reward_groups.append([session[:5], session, g])

nwb_reward_groups = pd.DataFrame(nwb_reward_groups, columns=['mouse_id', 'session_id', 'reward_group'])

df = pd.merge(db_reward_groups, nwb_reward_groups, on=['mouse_id', 'session_id'], suffixes=('_db', '_nwb'))

print(f'{len(nwb_reward_groups)} sessions checked, {len(missing)} without an NWB file.')
print(df.loc[df.reward_group_db != df.reward_group_nwb, :])
