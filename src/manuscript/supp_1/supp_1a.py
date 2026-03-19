"""
Supplementary Figure 1a: First hit trial on Day 0 for R+ vs R-.

Bar plot (mean) with individual mouse dots showing the trial number of the
first whisker hit on Day 0, comparing reward groups.
Statistics: Mann-Whitney U test (two-sided).
"""

import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import mannwhitneyu

sys.path.append(r'/home/aprenard/repos/fast-learning')
import src.utils.utils_io as io
from src.utils.utils_plot import reward_palette


# ============================================================================
# Parameters
# ============================================================================

OUTPUT_DIR = os.path.join(io.manuscript_output_dir, 'supp_1', 'output')


# ============================================================================
# Load behaviour table
# ============================================================================

bh_path = os.path.join(io.processed_dir, 'behavior',
                       'behavior_imagingmice_table_5days_cut_with_learning_curves.csv')
table = pd.read_csv(bh_path)


# ============================================================================
# Compute first hit trial per mouse
# ============================================================================

day_0_data = table[(table['day'] == 0) & (table['whisker_stim'] == 1)]
f = lambda x: x.reset_index(drop=True).idxmax() + 1
fh = day_0_data.groupby(['mouse_id', 'reward_group'], as_index=False)[['outcome_w']].agg(f)
fh = fh.rename(columns={'outcome_w': 'first_hit_trial'})


# ============================================================================
# Statistics: Mann-Whitney U test
# ============================================================================

stat, p_value = mannwhitneyu(
    fh[fh['reward_group'] == 'R+']['first_hit_trial'],
    fh[fh['reward_group'] == 'R-']['first_hit_trial'],
    alternative='two-sided'
)
print(f"Mann-Whitney U test: U={stat:.3f}, p={p_value:.4f}")


# ============================================================================
# Figure
# ============================================================================

sns.set_theme(context='paper', style='ticks', font='sans-serif', font_scale=1,
              rc={'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none'})

fig, ax = plt.subplots(figsize=(4, 6))
sns.barplot(data=fh, x='reward_group', y='first_hit_trial',
            order=['R+', 'R-'], palette=reward_palette[::-1], width=0.3, ax=ax)
sns.stripplot(data=fh, x='reward_group', y='first_hit_trial',
              order=['R+', 'R-'], color='grey', jitter=False, dodge=True, alpha=0.4, ax=ax)
ax.set_ylabel('First hit trial')
ax.set_xlabel('')
sns.despine()


# ============================================================================
# Save
# ============================================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

fig.savefig(os.path.join(OUTPUT_DIR, 'supp_1a.svg'), format='svg', dpi=300, bbox_inches='tight')
print(f"Saved: supp_1a.svg")

fh.to_csv(os.path.join(OUTPUT_DIR, 'supp_1a_data.csv'), index=False)
print(f"Saved: supp_1a_data.csv")

stats_df = pd.DataFrame([{
    'test': 'Mann-Whitney U',
    'group_1': 'R+',
    'group_2': 'R-',
    'statistic': stat,
    'p_value': p_value,
    'alternative': 'two-sided',
}])
stats_df.to_csv(os.path.join(OUTPUT_DIR, 'supp_1a_stats.csv'), index=False)
print(f"Saved: supp_1a_stats.csv")

plt.show()
