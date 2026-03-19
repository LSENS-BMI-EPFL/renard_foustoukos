"""
Supplementary Figure 1g: Example fitted learning curves for two mice (one R+,
one R-) on Day 0.

Each panel shows the fitted lick probability curve with 80% CI, the false
alarm rate, and a vertical line marking the learning trial.
"""

import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sys.path.append(r'/home/aprenard/repos/fast-learning')
import src.utils.utils_io as io
from src.utils.utils_plot import reward_palette, stim_palette


# ============================================================================
# Parameters
# ============================================================================

EXAMPLE_MICE = {'R+': 'GF305', 'R-': 'AR180'}
MAX_TRIALS = 100
OUTPUT_DIR = os.path.join(io.manuscript_output_dir, 'supp_1', 'output')


# ============================================================================
# Load data
# ============================================================================

bh_path = os.path.join(io.processed_dir, 'behavior',
                       'behavior_imagingmice_table_5days_cut_with_learning_curves.csv')
table = pd.read_csv(bh_path)


# ============================================================================
# Figure
# ============================================================================

sns.set_theme(context='paper', style='ticks', font='sans-serif', font_scale=1,
              rc={'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none'})

fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))

for ax, (rg, mouse_id) in zip(axes, EXAMPLE_MICE.items()):
    color = reward_palette[1] if rg == 'R+' else reward_palette[0]

    session_row = table[(table['mouse_id'] == mouse_id) & (table['day'] == 0)]
    d = session_row[session_row['whisker_stim'] == 1].reset_index(drop=True)
    d = d.loc[d['trial_w'] < MAX_TRIALS]

    learning_curve_w = d['learning_curve_w'].values.astype(float)
    learning_ci_low  = d['learning_curve_w_ci_low'].values.astype(float)
    learning_ci_high = d['learning_curve_w_ci_high'].values.astype(float)
    learning_chance  = d['learning_curve_chance'].values.astype(float)

    ax.plot(d['trial_w'], learning_curve_w, color=color, linewidth=2, label='Fitted curve')
    ax.fill_between(d['trial_w'], learning_ci_low, learning_ci_high,
                    color=color, alpha=0.2, label='80% CI')
    ax.plot(d['trial_w'], learning_chance, color=stim_palette[2], linewidth=1.5,
            label='False alarm rate')

    learning_trial = session_row['learning_trial'].values[0]
    if not pd.isna(learning_trial):
        ax.axvline(x=learning_trial, color='black', linestyle='--', linewidth=1,
                   label=f'Learning trial ({int(learning_trial)})')

    ax.set_ylim(0, 1)
    ax.set_xlabel('Whisker trial')
    ax.set_ylabel('Lick probability')
    ax.set_title(f'{mouse_id} – {rg}')
    ax.legend(frameon=False, fontsize=7)
    sns.despine(ax=ax)

fig.tight_layout()


# ============================================================================
# Save
# ============================================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

fig.savefig(os.path.join(OUTPUT_DIR, 'supp_1g.svg'), format='svg', dpi=300, bbox_inches='tight')
print("Saved: supp_1g.svg")

# Data: example mice learning curve data
data_out = []
for rg, mouse_id in EXAMPLE_MICE.items():
    session_row = table[(table['mouse_id'] == mouse_id) & (table['day'] == 0)]
    d = session_row[session_row['whisker_stim'] == 1].reset_index(drop=True)
    d = d.loc[d['trial_w'] < MAX_TRIALS].copy()
    d['reward_group'] = rg
    data_out.append(d[['mouse_id', 'reward_group', 'trial_w',
                        'learning_curve_w', 'learning_curve_w_ci_low',
                        'learning_curve_w_ci_high', 'learning_curve_chance',
                        'learning_trial']])

pd.concat(data_out, ignore_index=True).to_csv(
    os.path.join(OUTPUT_DIR, 'supp_1g_data.csv'), index=False)
print("Saved: supp_1g_data.csv")

plt.show()
