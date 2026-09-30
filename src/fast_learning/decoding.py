"""Population decoding of pre- vs post-learning mapping responses.

Each mouse's population response to a passive whisker stimulus is the mean
baseline-subtracted dF/F of every cell over WIN after stimulus onset, for the
last N_MAP_TRIALS mapping trials of each day. A logistic-regression decoder
trained on days -2/-1 vs +1/+2 separates pre- from post-learning responses;
pipeline/06_decoder.py trains it once per mouse and saves the weights, which
Fig. 4b-c, Supp. 2c, Supp. 3k-l and the d-prime revision reuse.
"""

import os
import pickle

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from fast_learning import imaging, paths, database


# ============================================================================
# Parameters
# ============================================================================

SAMPLING_RATE = 30
WIN = (0, 0.300)
BASELINE_WIN = (-1, 0)
DAYS = [-2, -1, 0, 1, 2]
N_MAP_TRIALS = 40

RESULTS_DIR = os.path.join(paths.processed_dir, 'decoding')
DECODER_WEIGHTS_PKL = os.path.join(RESULTS_DIR, 'decoder_weights.pkl')
CLASSIFIER_WEIGHTS_CSV = os.path.join(RESULTS_DIR, 'classifier_weights.csv')


# ============================================================================
# Data loading and processing
# ============================================================================


def load_and_process_data(
    select_lmi=False,
    projection_type=None,
    n_min_proj=5,
):
    """
    Load imaging data for decoding analyses.

    Args:
        select_lmi: If True, restrict to LMI-significant cells
        projection_type: Cell type filter ('wS2', 'wM1', or None for all cells)
        n_min_proj: Minimum number of projection cells required to include a mouse

    Returns:
        vectors_rew: List of xarrays (one per R+ mouse), shape (cells, trials)
        vectors_nonrew: List of xarrays (one per R- mouse), shape (cells, trials)
        mice_rew: List of R+ mouse IDs
        mice_nonrew: List of R- mouse IDs
    """
    _, _, mice, db = database.select_sessions_from_db(paths.db_path, paths.nwb_dir, two_p_imaging='yes')
    print(mice)

    selected_cells = None
    if select_lmi:
        processed_folder = paths.processed_dir
        lmi_df = pd.read_csv(os.path.join(processed_folder, 'lmi_results.csv'))
        selected_cells = lmi_df.loc[(lmi_df['lmi_p'] <= 0.025) | (lmi_df['lmi_p'] >= 0.975)]

    vectors_rew, vectors_nonrew = [], []
    mice_rew, mice_nonrew = [], []

    for mouse in mice:
        print(f"Processing mouse: {mouse}")
        folder = paths.tensor_dir
        xarray = imaging.load_mouse_xarray(mouse, folder, 'tensor_xarray_mapping_data.nc')
        # Manual baseline subtraction
        xarray = xarray - np.nanmean(
            xarray.sel(time=slice(BASELINE_WIN[0], BASELINE_WIN[1])).values, axis=2, keepdims=True
        )
        rew_gp = database.get_mouse_reward_group_from_db(paths.db_path, mouse, db)

        xarray = xarray.sel(trial=xarray['day'].isin(DAYS))

        if select_lmi and selected_cells is not None:
            selected_for_mouse = selected_cells.loc[selected_cells['mouse_id'] == mouse]['roi']
            xarray = xarray.sel(cell=xarray['roi'].isin(selected_for_mouse))

        if projection_type is not None:
            xarray = xarray.sel(cell=xarray['cell_type'] == projection_type)
            if xarray.sizes['cell'] < n_min_proj:
                print(f"Not enough cells of type {projection_type} for mouse {mouse}.")
                continue

        n_trials = xarray[0, :, 0].groupby('day').count(dim='trial').values
        if np.any(n_trials < N_MAP_TRIALS):
            print(f'Not enough mapping trials for {mouse}.')
            continue

        d = xarray.groupby('day').apply(lambda x: x.isel(trial=slice(-N_MAP_TRIALS, None)))
        d = d.sel(time=slice(WIN[0], WIN[1])).mean(dim='time')
        d = d.fillna(0)

        if rew_gp == 'R-':
            vectors_nonrew.append(d)
            mice_nonrew.append(mouse)
        elif rew_gp == 'R+':
            vectors_rew.append(d)
            mice_rew.append(mouse)

    print(f"Loaded {len(vectors_rew)} R+ mice and {len(vectors_nonrew)} R- mice")
    return vectors_rew, vectors_nonrew, mice_rew, mice_nonrew


# ============================================================================
# Fixed pre/post decoder
# ============================================================================


def train_and_save_decoder_weights(
    vectors_rew,
    vectors_nonrew,
    mice_rew,
    mice_nonrew,
    pre_days=[-2, -1],
    post_days=[1, 2],
    seed=42,
    results_dir=RESULTS_DIR,
):
    """
    Train a fixed pre/post logistic regression decoder per mouse on mapping
    trials (Days -2/-1 vs +1/+2) and save the weights to results_dir.

    Saved file: decoder_weights.pkl
    Structure: {mouse_id: {'scaler': StandardScaler, 'clf': LogisticRegression,
                            'sign_flip': int, 'reward_group': str}}

    The sign_flip ensures that higher decision values always correspond to
    post-learning activity.
    """
    weights = {}

    for vectors, mice_list, reward_group in [
        (vectors_rew, mice_rew, 'R+'),
        (vectors_nonrew, mice_nonrew, 'R-'),
    ]:
        for d, mouse in zip(vectors, mice_list):
            day_per_trial = d['day'].values
            train_mask = np.isin(day_per_trial, pre_days + post_days)
            if np.sum(train_mask) < 4:
                print(f'  {mouse}: not enough training trials, skipping.')
                continue

            X_train = d.values[:, train_mask].T
            y_train = np.array([0 if day in pre_days else 1 for day in day_per_trial[train_mask]])

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            clf = LogisticRegression(max_iter=5000, random_state=seed)
            clf.fit(X_train_scaled, y_train)

            # Ensure post > pre in decision value
            pre_mask = np.isin(day_per_trial, pre_days)
            post_mask = np.isin(day_per_trial, post_days)
            mean_dec_pre = np.mean(clf.decision_function(scaler.transform(d.values[:, pre_mask].T)))
            mean_dec_post = np.mean(clf.decision_function(scaler.transform(d.values[:, post_mask].T)))
            sign_flip = -1 if mean_dec_pre > mean_dec_post else 1

            roi_ids = d['roi'].values if 'roi' in d.coords else d['cell'].values
            weights[mouse] = {
                'scaler': scaler,
                'clf': clf,
                'sign_flip': sign_flip,
                'reward_group': reward_group,
                'roi': roi_ids,
            }
            print(
                f'  {mouse} ({reward_group}): decoder trained '
                f'({X_train.shape[1]} cells, sign_flip={sign_flip})'
            )

    os.makedirs(results_dir, exist_ok=True)

    # Save full sklearn objects as pickle
    out_path = os.path.join(results_dir, 'decoder_weights.pkl')
    with open(out_path, 'wb') as f:
        pickle.dump(weights, f)
    print(f'Decoder weights saved: {out_path}  ({len(weights)} mice)')

    # Save per-cell weights as CSV for downstream analyses
    rows = []
    for mouse, w in weights.items():
        coefs = w['clf'].coef_.flatten()
        for roi, coef in zip(w['roi'], coefs):
            rows.append(
                {
                    'mouse_id': mouse,
                    'roi': roi,
                    'reward_group': w['reward_group'],
                    'classifier_weight_raw': coef,
                    'classifier_weight': coef * w['sign_flip'],
                    'sign_flip': w['sign_flip'],
                }
            )
    csv_path = os.path.join(results_dir, 'classifier_weights.csv')
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f'Classifier weights CSV saved: {csv_path}')

    return weights


def load_decoder_weights(path=DECODER_WEIGHTS_PKL):
    """{mouse_id: {'scaler', 'clf', 'sign_flip', 'reward_group', 'roi'}}."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Run pipeline/06_decoder.py first.")
    with open(path, 'rb') as f:
        return pickle.load(f)


def load_classifier_weights(path=CLASSIFIER_WEIGHTS_CSV):
    """Per-cell classifier weights (mouse_id, roi, reward_group, classifier_weight, ...)."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Run pipeline/06_decoder.py first.")
    return pd.read_csv(path)
