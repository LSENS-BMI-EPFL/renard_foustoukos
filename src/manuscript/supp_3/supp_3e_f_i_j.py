"""
Supplementary Figure 3e, f, i, j: Proportions and distributions of LMI for
projection neurons.

  Panel e: wS2 — distribution of LMI values (R+ vs R-, KS test)
  Panel f: wS2 — proportion of cells with significant positive/negative LMI
  Panel i: wM1 — distribution of LMI values (R+ vs R-, KS test)
  Panel j: wM1 — proportion of cells with significant positive/negative LMI

Statistics: Mann-Whitney U test (R+ vs R-) per cell type × LMI sign.
           Kolmogorov-Smirnov test (R+ vs R-) per cell type for distributions.
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import mannwhitneyu, ks_2samp

sys.path.append(r'/home/aprenard/repos/fast-learning')
import src.utils.utils_io as io
from src.utils.utils_plot import reward_palette


# ============================================================================
# Parameters
# ============================================================================

LMI_POS_THRESHOLD = 0.975
LMI_NEG_THRESHOLD = 0.025
OUTPUT_DIR = os.path.join(io.manuscript_output_dir, 'supp_3', 'output')


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
# Compute proportions
# ============================================================================

lmi_prop_ct = (lmi_df.groupby(['mouse_id', 'reward_group', 'cell_type'])
               [['lmi_pos', 'lmi_neg']]
               .apply(lambda x: x.sum() / x.count())
               .reset_index())


# ============================================================================
# Statistics: Mann-Whitney U (R+ vs R-) per cell type × sign
# ============================================================================

stats_rows = []
for ct in ['wS2', 'wM1']:
    for sign in ['lmi_pos', 'lmi_neg']:
        sub = lmi_prop_ct[lmi_prop_ct['cell_type'] == ct]
        rp = sub[sub['reward_group'] == 'R+'][sign]
        rm = sub[sub['reward_group'] == 'R-'][sign]
        stat, p = mannwhitneyu(rp, rm, alternative='two-sided')
        stats_rows.append({'cell_type': ct, 'lmi_sign': sign,
                           'test': 'Mann-Whitney U', 'statistic': stat, 'p_value': p})
        print(f"{ct} {sign}: U={stat:.3f}, p={p:.4f}")

stats_df = pd.DataFrame(stats_rows)


# ============================================================================
# Statistics: KS test (R+ vs R-) per cell type for distributions
# ============================================================================

ks_rows = []
for ct in ['wS2', 'wM1']:
    sub = lmi_df[lmi_df['cell_type'] == ct]
    rp = sub[sub['reward_group'] == 'R+']['lmi'].values
    rm = sub[sub['reward_group'] == 'R-']['lmi'].values
    stat, p = ks_2samp(rp, rm, alternative='two-sided')
    ks_rows.append({'cell_type': ct, 'test': 'KS', 'statistic': stat, 'p_value': p})
    print(f"{ct} LMI distribution KS: D={stat:.3f}, p={p:.4f}")

ks_df = pd.DataFrame(ks_rows)


# ============================================================================
# Helper
# ============================================================================

def get_star(p):
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    return 'n.s.'


def plot_distribution(cell_type, ax):
    data = lmi_df[lmi_df['cell_type'] == cell_type]
    bin_edges = np.linspace(-1, 1, 31)
    for rg, color in zip(['R-', 'R+'], reward_palette):
        sns.histplot(data[data['reward_group'] == rg]['lmi'],
                     bins=bin_edges, kde=True, stat='probability',
                     color=color, label=rg, alpha=0.5, ax=ax)
    ks_row = ks_df[ks_df['cell_type'] == cell_type]
    if not ks_row.empty:
        star = get_star(ks_row.iloc[0]['p_value'])
        ax.text(0.98, 0.98, star, ha='right', va='top', fontsize=10,
                transform=ax.transAxes)
    ax.set_xlim(-1, 1)
    ax.set_xlabel('LMI')
    ax.set_ylabel('Probability')
    ax.legend(frameon=False)


def plot_proportion_pair(cell_type, ax_pos, ax_neg):
    data = lmi_prop_ct[lmi_prop_ct['cell_type'] == cell_type]
    for ax, sign, title in [
        (ax_pos, 'lmi_pos', f'{cell_type} — positive LMI'),
        (ax_neg, 'lmi_neg', f'{cell_type} — negative LMI'),
    ]:
        sns.barplot(data=data, x='reward_group', order=['R+', 'R-'],
                    hue='reward_group', hue_order=['R-', 'R+'],
                    y=sign, palette=reward_palette, legend=False, ax=ax)
        sns.swarmplot(data=data, x='reward_group', order=['R+', 'R-'],
                      y=sign, color='k', size=4, alpha=0.7, ax=ax)
        ax.set_title(title)
        ax.set_xlabel('')
        ax.set_ylabel('Proportion of cells')

        # Significance annotation
        stat_row = stats_df[(stats_df['cell_type'] == cell_type) &
                             (stats_df['lmi_sign'] == sign)]
        if not stat_row.empty:
            star = get_star(stat_row.iloc[0]['p_value'])
            if star:
                ax.annotate(star, xy=(0.5, 0.95), xycoords='axes fraction',
                            ha='center', va='top', fontsize=14, color='black')


# ============================================================================
# Figures
# ============================================================================

sns.set_theme(context='paper', style='ticks', font='sans-serif', font_scale=1,
              rc={'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none'})

# wS2 — panel e: distribution
fig_wS2_dist, ax_wS2_dist = plt.subplots(1, 1, figsize=(3, 4))
plot_distribution('wS2', ax_wS2_dist)
sns.despine(trim=True)
plt.tight_layout()

# wS2 — panel f: proportions
fig_wS2, axes_wS2 = plt.subplots(1, 2, figsize=(5, 4), sharey=True)
plot_proportion_pair('wS2', axes_wS2[0], axes_wS2[1])
sns.despine(trim=True)
plt.tight_layout()

# wM1 — panel i: distribution
fig_wM1_dist, ax_wM1_dist = plt.subplots(1, 1, figsize=(3, 4))
plot_distribution('wM1', ax_wM1_dist)
sns.despine(trim=True)
plt.tight_layout()

# wM1 — panel j: proportions
fig_wM1, axes_wM1 = plt.subplots(1, 2, figsize=(5, 4), sharey=True)
plot_proportion_pair('wM1', axes_wM1[0], axes_wM1[1])
sns.despine(trim=True)
plt.tight_layout()


# ============================================================================
# Save
# ============================================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

fig_wS2_dist.savefig(os.path.join(OUTPUT_DIR, 'supp_3e.svg'), format='svg',
                     dpi=300, bbox_inches='tight')
print("Saved: supp_3e.svg")

fig_wS2.savefig(os.path.join(OUTPUT_DIR, 'supp_3f.svg'), format='svg',
                dpi=300, bbox_inches='tight')
print("Saved: supp_3f.svg")

fig_wM1_dist.savefig(os.path.join(OUTPUT_DIR, 'supp_3i.svg'), format='svg',
                     dpi=300, bbox_inches='tight')
print("Saved: supp_3i.svg")

fig_wM1.savefig(os.path.join(OUTPUT_DIR, 'supp_3j.svg'), format='svg',
                dpi=300, bbox_inches='tight')
print("Saved: supp_3j.svg")

lmi_prop_ct[lmi_prop_ct['cell_type'].isin(['wS2', 'wM1'])].to_csv(
    os.path.join(OUTPUT_DIR, 'supp_3e_f_i_j_data.csv'), index=False)
print("Saved: supp_3e_f_i_j_data.csv")

stats_df.to_csv(os.path.join(OUTPUT_DIR, 'supp_3e_f_i_j_stats.csv'), index=False)
print("Saved: supp_3e_f_i_j_stats.csv")

ks_df.to_csv(os.path.join(OUTPUT_DIR, 'supp_3e_i_ks_stats.csv'), index=False)
print("Saved: supp_3e_i_ks_stats.csv")

plt.show()
