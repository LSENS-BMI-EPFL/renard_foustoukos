"""
Figure 4c: Progressive learning during Day 0 — population average behaviour
and decoder value, R+ vs R-.

Two-row × two-column layout:
  Row 1: Behavioural learning curve averaged across mice (Day 0 whisker trials).
  Row 2: Mean decoder decision value averaged across mice (sliding window on
         Day 0 whisker trials).

Decoder weights (trained on Days -2/-1 vs +1/+2 mapping trials) are loaded
from RESULTS_DIR/decoder_weights.pkl, produced by figure_3m_o.py.
"""

import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.append(r'/home/aprenard/repos/NWB_analysis')
sys.path.append(r'/home/aprenard/repos/fast-learning')

import src.utils.utils_imaging as utils_imaging
import src.utils.utils_io as io
from src.utils.utils_plot import reward_palette


# ============================================================================
# Parameters
# ============================================================================

win = (0, 0.300)          # response window from stimulus onset (seconds)
window_size = 10
step_size = 1
cut_n_trials = 100

RESULTS_DIR = os.path.join(io.processed_dir, 'decoding')
OUTPUT_DIR = os.path.join(io.manuscript_output_dir, 'figure_4', 'output')


# ============================================================================
# Load decoder weights
# ============================================================================

weights_path = os.path.join(RESULTS_DIR, 'decoder_weights.pkl')
with open(weights_path, 'rb') as f:
    weights = pickle.load(f)
print(f"Loaded decoder weights for {len(weights)} mice.")


# ============================================================================
# Load behaviour and Day-0 learning data
# ============================================================================

bh_path = os.path.join(io.processed_dir, 'behavior',
                        'behavior_imagingmice_table_5days_cut_with_learning_curves.csv')
table = pd.read_csv(bh_path)
bh_df = table.loc[(table['day'] == 0) & (table['whisker_stim'] == 1)]

folder = os.path.join(io.processed_dir, 'mice')
xarrays_learning = {}
for mouse in weights:
    xarr = utils_imaging.load_mouse_xarray(mouse, folder, 'tensor_xarray_learning_data.nc')
    xarr = xarr.sel(trial=xarr['day'].isin([0]))
    xarr = xarr.sel(trial=xarr['whisker_stim'] == 1)
    xarr = xarr.sel(time=slice(win[0], win[1])).mean(dim='time')
    xarr = xarr.fillna(0)
    xarrays_learning[mouse] = xarr


# ============================================================================
# Apply decoder (sliding window)
# ============================================================================

results = []
for mouse, w in weights.items():
    xarr = xarrays_learning[mouse]
    scaler, clf, sign_flip = w['scaler'], w['clf'], w['sign_flip']
    n_trials = xarr.sizes['trial']
    for start_idx in range(0, max(0, n_trials - window_size + 1), step_size):
        end_idx = start_idx + window_size
        X_win = xarr.values[:, start_idx:end_idx].T
        if X_win.shape[0] == 0:
            continue
        dec_vals = clf.decision_function(scaler.transform(X_win))
        results.append({
            'mouse_id': mouse,
            'reward_group': w['reward_group'],
            'trial_center': start_idx + window_size // 2,
            'mean_decision_value': np.mean(dec_vals) * sign_flip,
        })

results_df = pd.concat([pd.DataFrame(results)], ignore_index=True)
mice_rew = [m for m, w in weights.items() if w['reward_group'] == 'R+']
mice_nonrew = [m for m, w in weights.items() if w['reward_group'] == 'R-']


# ============================================================================
# Figure
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(10, 6))

data_rew = bh_df.loc[bh_df['mouse_id'].isin(mice_rew)]
data_nonrew = bh_df.loc[bh_df['mouse_id'].isin(mice_nonrew)]
results_rew = results_df[results_df['reward_group'] == 'R+']
results_nonrew = results_df[results_df['reward_group'] == 'R-']

# Row 1: behaviour
for ax, data, color, title in [
    (axes[0, 0], data_rew,    reward_palette[1], 'R+ mice'),
    (axes[0, 1], data_nonrew, reward_palette[0], 'R- mice'),
]:
    sns.lineplot(data=data, x='trial_w', y='learning_curve_w',
                 color=color, errorbar='ci', ax=ax)
    ax.set_xlabel('Trial within Day 0')
    ax.set_ylabel('Learning curve (w)')
    ax.set_title(title)
    ax.set_ylim(0, 1)
    ax.set_xlim(0, cut_n_trials)

# Row 2: decision values
for ax, data, color, title in [
    (axes[1, 0], results_rew,    reward_palette[1], 'R+ decoder value'),
    (axes[1, 1], results_nonrew, reward_palette[0], 'R- decoder value'),
]:
    sns.lineplot(data=data, x='trial_center', y='mean_decision_value',
                 estimator=np.mean, errorbar='ci', color=color, ax=ax)
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax.set_xlabel('Trial within Day 0')
    ax.set_ylabel('Mean decision value')
    ax.set_title(title)
    ax.set_xlim(0, cut_n_trials)

plt.tight_layout()
sns.despine()


# ============================================================================
# Save
# ============================================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)
out_path = os.path.join(OUTPUT_DIR, 'figure_4c.svg')
fig.savefig(out_path, format='svg', dpi=300, bbox_inches='tight')
print(f"\nSaved: {out_path}")

plt.show()
