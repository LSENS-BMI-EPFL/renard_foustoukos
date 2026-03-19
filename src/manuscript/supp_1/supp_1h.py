"""
Supplementary Figure 1h: Distribution of learning trials on Day 0 for R+ and R-.

Histogram of the trial number at which each mouse crossed the learning
threshold, one panel per reward group.
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.append(r'/home/aprenard/repos/fast-learning')
import src.utils.utils_io as io
from src.utils.utils_plot import reward_palette


# ============================================================================
# Parameters
# ============================================================================

OUTPUT_DIR = os.path.join(io.manuscript_output_dir, 'supp_1', 'output')


# ============================================================================
# Load data
# ============================================================================

bh_path = os.path.join(io.processed_dir, 'behavior',
                       'behavior_imagingmice_table_5days_cut_with_learning_curves.csv')
table = pd.read_csv(bh_path)

day0 = (table[table['day'] == 0]
        .groupby('session_id', as_index=False)
        .first()[['session_id', 'mouse_id', 'reward_group', 'learning_trial']])


# ============================================================================
# Figure
# ============================================================================

sns.set_theme(context='paper', style='ticks', font='sans-serif', font_scale=1,
              rc={'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none'})

fig, axes = plt.subplots(1, 2, figsize=(7, 3), sharey=True)

xlims = {'R+': (0, 70), 'R-': (0, 120)}

for ax, rg, color in zip(axes, ['R+', 'R-'], [reward_palette[1], reward_palette[0]]):
    data_rg = day0[day0['reward_group'] == rg]['learning_trial'].dropna()
    bin_edges = np.arange(0, data_rg.max() + 5, 5) if len(data_rg) > 0 else 15
    ax.hist(data_rg, bins=bin_edges, color=color, edgecolor='white')
    n_learned = len(data_rg)
    n_total = (day0['reward_group'] == rg).sum()
    ax.set_title(f'{rg}  (n learned = {n_learned} / {n_total})')
    ax.set_xlabel('Learning trial (whisker)')
    ax.set_ylabel('Number of mice')
    ax.set_ylim(0, 6)
    ax.set_xlim(xlims[rg])
    sns.despine(ax=ax)

fig.tight_layout()


# ============================================================================
# Save
# ============================================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

fig.savefig(os.path.join(OUTPUT_DIR, 'supp_1h.svg'), format='svg', dpi=300, bbox_inches='tight')
print("Saved: supp_1h.svg")

day0.to_csv(os.path.join(OUTPUT_DIR, 'supp_1h_data.csv'), index=False)
print("Saved: supp_1h_data.csv")

plt.show()
