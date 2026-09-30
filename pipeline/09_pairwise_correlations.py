"""Pipeline step 09: pairwise correlations between projection neurons.

For each imaging mouse, the pre-stimulus correlation of every same-type
projector pair (wS2-wS2, wM1-wM1), before and after learning
(fast_learning.correlations).

Inputs:  mapping tensors (paths.tensor_dir), session metadata.
Outputs: <processed_dir>/pairwise_correlations/<R+|R->/mapping/pairwise_correlations_prepost.csv

Usage:
    python pipeline/09_pairwise_correlations.py
"""

import os
from multiprocessing import Pool

import pandas as pd

from fast_learning import paths, database, correlations

N_CORES = 35


if __name__ == '__main__':
    _, _, mice, db = database.select_sessions_from_db(
        paths.db_path, paths.nwb_dir, two_p_imaging='yes', experimenters=['AR', 'GF', 'MI']
    )
    mice_by_group = {}
    for mouse_id in mice:
        rg = database.get_mouse_reward_group_from_db(paths.db_path, mouse_id)
        mice_by_group.setdefault(rg, []).append(mouse_id)

    for reward_group in ['R+', 'R-']:
        group_mice = mice_by_group.get(reward_group, [])
        print(f"\n{reward_group}: {len(group_mice)} mice, {N_CORES} cores")
        with Pool(processes=N_CORES) as pool:
            results_list = pool.map(correlations.pairwise_correlations_mouse, group_mice)
        corr_df = pd.DataFrame([item for sublist in results_list for item in sublist])
        out = correlations.correlations_csv(reward_group)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        corr_df.to_csv(out, index=False)
        print(f"Saved: {out} ({len(corr_df)} pair-periods)")
