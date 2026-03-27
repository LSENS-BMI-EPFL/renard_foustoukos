"""
Figure 3f, g: Learning Modulation Index (LMI) for all cells.

  Panel f: Distribution of LMI values (R+ vs R-, KS test)
  Panel g: Proportion of cells with significant positive/negative LMI (R+ vs R-)

Statistics: KS test (R+ vs R-) for distributions.
            Mann-Whitney U test (R+ vs R-) for proportions.
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import mannwhitneyu, ks_2samp

sys.path.append('/home/aprenard/repos/fast-learning')
import src.utils.utils_io as io
from src.utils.utils_plot import reward_palette


# ============================================================================
# Parameters
# ============================================================================

LMI_POS_THRESHOLD = 0.975
LMI_NEG_THRESHOLD = 0.025
OUTPUT_DIR = os.path.join(io.manuscript_output_dir, 'figure_3', 'output')


# ============================================================================
# Load LMI data and assign reward groups
# ============================================================================

lmi_df = pd.read_csv(os.path.join(io.processed_dir, 'lmi_results.csv'))

_, _, mice, _ = io.select_sessions_from_db(io.db_path, io.nwb_dir,
                                            two_p_imaging='yes')

for mouse in lmi_df['mouse_id'].unique():
    lmi_df.loc[lmi_df['mouse_id'] == mouse, 'reward_group'] = \
        io.get_mouse_reward_group_from_db(io.db_path, mouse)

lmi_df = lmi_df.loc[lmi_df['mouse_id'].isin(mice)]

lmi_df['lmi_pos'] = lmi_df['lmi_p'] >= LMI_POS_THRESHOLD
lmi_df['lmi_neg'] = lmi_df['lmi_p'] <= LMI_NEG_THRESHOLD


# ============================================================================
# Compute proportions per mouse
# ============================================================================

lmi_prop = (lmi_df.groupby(['mouse_id', 'reward_group'])
            [['lmi_pos', 'lmi_neg']]
            .apply(lambda x: x.sum() / x.count())
            .reset_index())


# ============================================================================
# Statistics
# ============================================================================

def get_star(p):
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    return 'n.s.'


# KS test: R+ vs R- on LMI distribution
rp_lmi = lmi_df[lmi_df['reward_group'] == 'R+']['lmi'].values
rm_lmi = lmi_df[lmi_df['reward_group'] == 'R-']['lmi'].values
ks_stat, ks_p = ks_2samp(rp_lmi, rm_lmi, alternative='two-sided')
print(f"LMI distribution KS: D={ks_stat:.3f}, p={ks_p:.4f}")

# Mann-Whitney U: R+ vs R- on proportions
stats_rows = []
for sign in ['lmi_pos', 'lmi_neg']:
    rp = lmi_prop[lmi_prop['reward_group'] == 'R+'][sign]
    rm = lmi_prop[lmi_prop['reward_group'] == 'R-'][sign]
    stat, p = mannwhitneyu(rp, rm, alternative='two-sided')
    stats_rows.append({'lmi_sign': sign, 'test': 'Mann-Whitney U',
                       'statistic': stat, 'p_value': p})
    print(f"{sign}: U={stat:.3f}, p={p:.4f}")

stats_df = pd.DataFrame(stats_rows)


# ============================================================================
# Figure f: LMI distribution
# ============================================================================

sns.set_theme(context='paper', style='ticks', font='sans-serif', font_scale=1,
              rc={'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none'})

fig_f, ax_f = plt.subplots(1, 1, figsize=(4, 4))

bin_edges = np.linspace(-1, 1, 31)
for rg, color in zip(['R-', 'R+'], reward_palette):
    sns.histplot(lmi_df[lmi_df['reward_group'] == rg]['lmi'],
                 bins=bin_edges, kde=True, stat='probability',
                 color=color, label=rg, alpha=0.5, ax=ax_f)

ax_f.text(0.98, 0.98, get_star(ks_p), ha='right', va='top', fontsize=10,
          transform=ax_f.transAxes)
ax_f.set_xlim(-1, 1)
ax_f.set_xlabel('LMI')
ax_f.set_ylabel('Probability')
ax_f.legend(frameon=False)
sns.despine(trim=True)
plt.tight_layout()


# ============================================================================
# Figure g: proportion bar plots
# ============================================================================

fig_g, axes_g = plt.subplots(1, 2, figsize=(5, 4), sharey=True)

for ax, sign, title in [
    (axes_g[0], 'lmi_pos', 'Positive LMI'),
    (axes_g[1], 'lmi_neg', 'Negative LMI'),
]:
    sns.barplot(data=lmi_prop, x='reward_group', order=['R+', 'R-'],
                hue='reward_group', hue_order=['R-', 'R+'],
                y=sign, palette=reward_palette, legend=False, ax=ax)
    sns.swarmplot(data=lmi_prop, x='reward_group', order=['R+', 'R-'],
                  y=sign, color='k', size=4, alpha=0.7, ax=ax)
    ax.set_title(title)
    ax.set_xlabel('')
    ax.set_ylabel('Proportion of cells')

    stat_row = stats_df[stats_df['lmi_sign'] == sign]
    if not stat_row.empty:
        star = get_star(stat_row.iloc[0]['p_value'])
        ax.annotate(star, xy=(0.5, 0.95), xycoords='axes fraction',
                    ha='center', va='top', fontsize=14, color='black')

sns.despine(trim=True)
plt.tight_layout()


# ============================================================================
# Save
# ============================================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

fig_f.savefig(os.path.join(OUTPUT_DIR, 'figure_3f.svg'), format='svg', dpi=300, bbox_inches='tight')
print("Saved: figure_3f.svg")

fig_g.savefig(os.path.join(OUTPUT_DIR, 'figure_3g.svg'), format='svg', dpi=300, bbox_inches='tight')
print("Saved: figure_3g.svg")

stats_df.to_csv(os.path.join(OUTPUT_DIR, 'figure_3f_g_stats.csv'), index=False)
print("Saved: figure_3f_g_stats.csv")

plt.show()
