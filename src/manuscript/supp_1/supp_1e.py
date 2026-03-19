"""
Supplementary Figure 1e: Fitted learning curve across Day 0 trials, R+ vs R-,
realigned to each mouse's first hit trial.

Per-trial Mann-Whitney U tests with FDR correction; significant trials shown
as a colour bar at the top of the panel.
"""

import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

sys.path.append(r'/home/aprenard/repos/fast-learning')
import src.utils.utils_io as io
from src.utils.utils_plot import reward_palette


# ============================================================================
# Parameters
# ============================================================================

DAY = 0
MAX_TRIALS = 100
OUTPUT_DIR = os.path.join(io.manuscript_output_dir, 'supp_1', 'output')


# ============================================================================
# Load and prepare data
# ============================================================================

bh_path = os.path.join(io.processed_dir, 'behavior',
                       'behavior_imagingmice_table_5days_cut_with_learning_curves.csv')
table = pd.read_csv(bh_path)

df_base = table.loc[(table['whisker_stim'] == 1) & (table['day'] == DAY)]
df_base = df_base.loc[df_base['trial_w'] < MAX_TRIALS]


# ============================================================================
# Realign to first hit per mouse
# ============================================================================

def realign_to_first_hit(data):
    realigned = []
    for mouse in data['mouse_id'].unique():
        md = data[data['mouse_id'] == mouse].copy().sort_values('trial_w')
        first_hit_idx = md[md['outcome_w'] == 1].index
        if len(first_hit_idx) > 0:
            first_hit_trial = md.loc[first_hit_idx[0], 'trial_w']
            md['trial_w_realigned'] = md['trial_w'] - first_hit_trial
            realigned.append(md)
    return pd.concat(realigned, ignore_index=True)

df = realign_to_first_hit(df_base)
df = df.loc[(df['trial_w_realigned'] >= 0) & (df['trial_w_realigned'] < MAX_TRIALS)]


# ============================================================================
# Statistics: per-trial Mann-Whitney U with FDR correction
# ============================================================================

p_values_raw = []
for trial_w in sorted(df['trial_w_realigned'].unique()):
    rp = df[(df['trial_w_realigned'] == trial_w) & (df['reward_group'] == 'R+')]['learning_curve_w']
    rm = df[(df['trial_w_realigned'] == trial_w) & (df['reward_group'] == 'R-')]['learning_curve_w']
    if len(rp) > 0 and len(rm) > 0:
        stat, p = mannwhitneyu(rp, rm, alternative='two-sided')
        p_values_raw.append({'trial_w_realigned': trial_w, 'statistic': stat, 'p_value_raw': p})

stats_df = pd.DataFrame(p_values_raw)
_, corrected, _, _ = multipletests(stats_df['p_value_raw'], alpha=0.05, method='fdr_bh')
stats_df['p_value_fdr'] = corrected


# ============================================================================
# Figure
# ============================================================================

sns.set_theme(context='paper', style='ticks', font='sans-serif', font_scale=1,
              rc={'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none'})

cmap = matplotlib.colors.LinearSegmentedColormap.from_list('pval_cmap', ['black', 'white'])
norm = matplotlib.colors.Normalize(vmin=0, vmax=0.05)

fig, ax = plt.subplots(figsize=(7, 4))

sns.lineplot(data=df, x='trial_w_realigned', y='learning_curve_w',
             hue='reward_group', hue_order=['R+', 'R-'],
             palette=reward_palette[::-1],
             errorbar='ci', err_style='band', ax=ax)

for _, row in stats_df.iterrows():
    color = cmap(norm(min(row['p_value_fdr'], 0.05)))
    ax.add_patch(plt.Rectangle((row['trial_w_realigned'] - 0.4, 0.95), 0.8, 0.03,
                                color=color, edgecolor='none'))

ax.set_xlabel('Whisker trial (from first hit)')
ax.set_ylabel('Lick probability')
ax.set_ylim(-0.1, 1)
ax.legend(frameon=False, title='Reward group')
sns.despine()


# ============================================================================
# Save
# ============================================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

fig.savefig(os.path.join(OUTPUT_DIR, 'supp_1e.svg'), format='svg', dpi=300, bbox_inches='tight')
print("Saved: supp_1e.svg")

df.to_csv(os.path.join(OUTPUT_DIR, 'supp_1e_data.csv'), index=False)
print("Saved: supp_1e_data.csv")

stats_df.to_csv(os.path.join(OUTPUT_DIR, 'supp_1e_stats.csv'), index=False)
print("Saved: supp_1e_stats.csv")

plt.show()
