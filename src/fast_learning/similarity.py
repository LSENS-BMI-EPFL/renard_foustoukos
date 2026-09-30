"""Trial-by-trial similarity of mapping responses (Fig. 3h-j).

Responses are, per cell, the mean dF/F over WIN after each of the last
N_MAP_TRIALS mapping trials of each day in DAYS, so a similarity matrix has
len(DAYS) * N_MAP_TRIALS rows, ordered by day.
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


WIN = (0, 0.300)  # stimulus onset to 300 ms after
DAYS = [-2, -1, 0, 1, 2]
N_MAP_TRIALS = 40


def compute_similarity_matrix(vector, similarity_metric):
    """Compute a trial-by-trial similarity matrix for one mouse."""
    if similarity_metric == 'pearson':
        cm = np.corrcoef(vector.values.T)
    elif similarity_metric == 'spearman':
        cm, _ = spearmanr(vector.values.T, axis=1)
    elif similarity_metric == 'cosine':
        data = vector.values.T  # (trials, cells)
        data = np.nan_to_num(data, nan=0.0)
        norms = np.linalg.norm(data, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        normalized = data / norms
        cm = normalized @ normalized.T
    np.fill_diagonal(cm, np.nan)
    return cm


def compute_within_day_metrics(corr_matrices, mice_ids, reward_group):
    """Compute average within-day correlation per mouse for each day."""
    results = []
    for cm in corr_matrices:
        row = {}
        for i, day in enumerate(DAYS):
            day_idx = np.arange(i * N_MAP_TRIALS, (i + 1) * N_MAP_TRIALS)
            row[f'within_day{day:+d}'] = np.nanmean(cm[np.ix_(day_idx, day_idx)])
        results.append(row)
    df = pd.DataFrame(results)
    df['reward_group'] = reward_group
    df['mouse_id'] = mice_ids
    return df


def compute_day0_metrics(corr_matrices, mice_ids, reward_group):
    """Compute average correlation between day 0 and each other day, per mouse."""
    day0_idx = np.arange(2 * N_MAP_TRIALS, 3 * N_MAP_TRIALS)
    results = []
    for cm in corr_matrices:
        row = {}
        for i, day in enumerate(DAYS):
            day_idx = np.arange(i * N_MAP_TRIALS, (i + 1) * N_MAP_TRIALS)
            row[f'corr_day0_vs_day{day:+d}'] = np.nanmean(cm[np.ix_(day0_idx, day_idx)])
        results.append(row)
    df = pd.DataFrame(results)
    df['reward_group'] = reward_group
    df['mouse_id'] = mice_ids
    return df


def compute_reorganization_metrics(corr_matrices, mice_ids, reward_group):
    """Compute network reorganization index per mouse."""
    pre_idx = np.arange(0, 2 * N_MAP_TRIALS)
    post_idx = np.arange(3 * N_MAP_TRIALS, 5 * N_MAP_TRIALS)
    results = []
    for cm in corr_matrices:
        within_pre = np.nanmean(cm[np.ix_(pre_idx, pre_idx)])
        within_post = np.nanmean(cm[np.ix_(post_idx, post_idx)])
        between = np.nanmean(cm[np.ix_(pre_idx, post_idx)])
        results.append(
            {
                'within_pre': within_pre,
                'within_post': within_post,
                'between_pre_post': between,
                'reorganization_index': (within_pre + within_post) / 2 - between,
            }
        )
    df = pd.DataFrame(results)
    df['reward_group'] = reward_group
    df['mouse_id'] = mice_ids
    return df
