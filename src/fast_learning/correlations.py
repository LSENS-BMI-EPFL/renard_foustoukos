"""Pairwise correlations between projection neurons (Supp. 3m).

Correlations are computed in a quiet window before stimulus onset (WIN_SEC)
of mapping trials, for same-type projector pairs, before (PRE_DAYS) and after
(POST_DAYS) learning. pipeline/09_pairwise_correlations.py runs it for all
imaging mice.
"""

import os
from itertools import combinations

import numpy as np
from scipy.stats import pearsonr

from fast_learning import imaging, paths, database


WIN_SEC = (-2, 0)  # quiet window before stimulus onset
PRE_DAYS = [-2, -1]
POST_DAYS = [1, 2]
PAIR_TYPES = ['wS2-wS2', 'wM1-wM1']

RESULTS_DIR = os.path.join(paths.processed_dir, 'pairwise_correlations')


def correlations_csv(reward_group):
    """Pair-level correlations of one reward group."""
    return os.path.join(RESULTS_DIR, reward_group, 'mapping', 'pairwise_correlations_prepost.csv')


def pairwise_correlations_mouse(mouse_id):
    """Mean pre-stimulus correlation of every same-type projector pair
    (wS2-wS2, wM1-wM1) of one mouse, pre and post learning.

    For each pair and period, the Pearson correlation of the two dF/F traces
    over WIN_SEC is computed on every mapping trial and averaged over trials.
    Returns a list of dicts (one per pair and period).
    """
    reward_group = database.get_mouse_reward_group_from_db(paths.db_path, mouse_id)
    folder = paths.tensor_dir
    xarr = imaging.load_mouse_xarray(mouse_id, folder, 'tensor_xarray_mapping_data.nc', subtracted=False)
    xarr.name = 'dff'
    xarr = xarr.sel(trial=xarr['day'].isin(PRE_DAYS + POST_DAYS))
    xarr = xarr.sel(time=slice(WIN_SEC[0], WIN_SEC[1]))

    mouse_results = []
    for period, days in [('pre', PRE_DAYS), ('post', POST_DAYS)]:
        xarr_period = xarr.sel(trial=xarr['day'].isin(days))
        all_cells_data = xarr_period.values  # (n_cells, n_trials, n_time)
        cell_types = xarr_period.coords['cell_type'].values
        rois = xarr_period.coords['roi'].values
        n_cells, n_trials, _ = all_cells_data.shape

        if n_trials == 0:
            continue

        for i, j in combinations(range(n_cells), 2):
            if cell_types[i] != cell_types[j]:
                continue
            if cell_types[i] not in ['wS2', 'wM1']:
                continue

            trial_corrs = []
            for t in range(n_trials):
                ci, cj = all_cells_data[i, t, :], all_cells_data[j, t, :]
                valid = ~(np.isnan(ci) | np.isnan(cj))
                if valid.sum() > 1 and np.std(ci[valid]) > 0 and np.std(cj[valid]) > 0:
                    trial_corrs.append(pearsonr(ci[valid], cj[valid])[0])

            if trial_corrs:
                mouse_results.append(
                    {
                        'mouse_id': mouse_id,
                        'reward_group': reward_group,
                        'period': period,
                        'pair_type': f'{cell_types[i]}-{cell_types[i]}',
                        'roi_i': rois[i],
                        'roi_j': rois[j],
                        'correlation': np.mean(trial_corrs),
                        'n_trials': len(trial_corrs),
                    }
                )

    print(f"  {mouse_id}: {len(mouse_results)} pairs")
    return mouse_results
