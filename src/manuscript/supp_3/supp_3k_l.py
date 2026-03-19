"""
Supplementary Figure 3k, l: CDF comparison of wS2 vs wM1 projector neurons.

  Panel k: CDF of LMI values (positive and negative, 2 reward groups)
  Panel l: CDF of classifier weights (positive and negative, 2 reward groups)

Statistics: Kolmogorov-Smirnov test (two-sided, wS2 vs wM1) per reward group
and sign.

Classifier weights are loaded from io.processed_dir/decoding (saved by
figure_3m_o.py). Cell-type labels for the weight file are retrieved from the
mapping xarrays.
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import ks_2samp

sys.path.append(r'/home/aprenard/repos/NWB_analysis')
sys.path.append(r'/home/aprenard/repos/fast-learning')
import src.utils.utils_imaging as utils_imaging
import src.utils.utils_io as io
from src.utils.utils_plot import s2_m1_palette


# ============================================================================
# Parameters
# ============================================================================

CELL_TYPES = ['wS2', 'wM1']
CELL_TYPE_COLORS = {'wS2': s2_m1_palette[0], 'wM1': s2_m1_palette[1]}
RESULTS_DIR = os.path.join(io.processed_dir, 'decoding')
OUTPUT_DIR  = os.path.join(io.manuscript_output_dir, 'supp_3', 'output')


# ============================================================================
# Helper
# ============================================================================

def pvalue_to_stars(p):
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    return 'ns'


def compute_ks_test(df, value_col, reward_groups, positive_only=False, negative_only=False):
    rows = []
    for rg in reward_groups:
        sub = df[df['reward_group'] == rg]
        if positive_only:
            sub = sub[sub[value_col] > 0]
        elif negative_only:
            sub = sub[sub[value_col] < 0]
        ws2 = sub[sub['cell_type_group'] == 'wS2'][value_col].values
        wm1 = sub[sub['cell_type_group'] == 'wM1'][value_col].values
        if len(ws2) > 0 and len(wm1) > 0:
            stat, p = ks_2samp(ws2, wm1, alternative='two-sided')
            rows.append({
                'reward_group': rg,
                'value_type': 'positive' if positive_only else ('negative' if negative_only else 'all'),
                'n_wS2': len(ws2), 'n_wM1': len(wm1),
                'ks_statistic': stat, 'ks_pvalue': p,
                'ks_stars': pvalue_to_stars(p),
            })
    return pd.DataFrame(rows)


def plot_cdf_panel(ax, data, value_col, reward_group, positive_only, ks_df, xlabel):
    sub = data[data['reward_group'] == reward_group]
    if positive_only:
        sub = sub[sub[value_col] > 0]
        vals_fn = lambda v: v
    else:
        sub = sub[sub[value_col] < 0]
        vals_fn = np.abs   # plot absolute values for negative

    for ct in CELL_TYPES:
        values = vals_fn(sub[sub['cell_type_group'] == ct][value_col].values)
        if len(values) > 0:
            sv = np.sort(values)
            cdf = np.arange(1, len(sv) + 1) / len(sv)
            ax.plot(sv, cdf, label=ct, color=CELL_TYPE_COLORS[ct], linewidth=2, alpha=0.8)

    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel('Cumulative probability', fontsize=11)
    sign_label = 'Positive' if positive_only else 'Negative (abs)'
    ax.set_title(f'{reward_group} — {sign_label}', fontsize=11, fontweight='bold')
    ax.legend(frameon=False)
    ax.set_xlim(left=0, right=0.8)
    ax.set_ylim(0, 1)

    vtype = 'positive' if positive_only else 'negative'
    ks_row = ks_df[(ks_df['reward_group'] == reward_group) & (ks_df['value_type'] == vtype)]
    if not ks_row.empty:
        stars = ks_row.iloc[0]['ks_stars']
        pval  = ks_row.iloc[0]['ks_pvalue']
        ax.text(0.98, 0.02, f'KS: {stars} (p={pval:.4f})',
                ha='right', va='bottom', fontsize=9, transform=ax.transAxes,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))


# ============================================================================
# Load LMI data
# ============================================================================

lmi_df = pd.read_csv(os.path.join(io.processed_dir, 'lmi_results.csv'))
for mouse in lmi_df['mouse_id'].unique():
    lmi_df.loc[lmi_df['mouse_id'] == mouse, 'reward_group'] = \
        io.get_mouse_reward_group_from_db(io.db_path, mouse)
lmi_df['cell_type_group'] = lmi_df['cell_type'].replace({'na': 'non_projector'})

print(f"LMI: {len(lmi_df)} cells, {lmi_df['mouse_id'].nunique()} mice")


# ============================================================================
# Load classifier weights + cell-type labels
# ============================================================================

weights_df = pd.read_csv(os.path.join(RESULTS_DIR, 'classifier_weights.csv'))

cell_type_info = []
for mouse_id in weights_df['mouse_id'].unique():
    try:
        xarr = utils_imaging.load_mouse_xarray(
            mouse_id, os.path.join(io.processed_dir, 'mice'),
            'tensor_xarray_mapping_data.nc')
        rois = xarr.coords['roi'].values
        cts  = (xarr.coords['cell_type'].values if 'cell_type' in xarr.coords
                else [None] * len(rois))
        for roi, ct in zip(rois, cts):
            cell_type_info.append({'mouse_id': mouse_id, 'roi': roi, 'cell_type_xr': ct})
    except Exception as e:
        print(f"Warning: could not load cell types for {mouse_id}: {e}")

if cell_type_info:
    weights_df = weights_df.merge(pd.DataFrame(cell_type_info), on=['mouse_id', 'roi'], how='left')
else:
    weights_df['cell_type_xr'] = None

weights_df['cell_type_group'] = weights_df['cell_type_xr'].copy()
weights_df.loc[weights_df['cell_type_xr'].isna() |
               (weights_df['cell_type_xr'] == 'na'), 'cell_type_group'] = 'non_projector'

print(f"Weights: {len(weights_df)} cells, {weights_df['mouse_id'].nunique()} mice")


# ============================================================================
# KS tests
# ============================================================================

ks_lmi_pos    = compute_ks_test(lmi_df,     'lmi',                ['R+', 'R-'], positive_only=True)
ks_lmi_neg    = compute_ks_test(lmi_df,     'lmi',                ['R+', 'R-'], negative_only=True)
ks_weight_pos = compute_ks_test(weights_df, 'classifier_weight',  ['R+', 'R-'], positive_only=True)
ks_weight_neg = compute_ks_test(weights_df, 'classifier_weight',  ['R+', 'R-'], negative_only=True)

stats_lmi    = pd.concat([ks_lmi_pos,    ks_lmi_neg],    ignore_index=True)
stats_weight = pd.concat([ks_weight_pos, ks_weight_neg], ignore_index=True)
print(stats_lmi)
print(stats_weight)


# ============================================================================
# Figure k: CDF of LMI
# ============================================================================

sns.set_theme(context='paper', style='ticks', font='sans-serif', font_scale=1,
              rc={'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none'})

fig_k = plt.figure(figsize=(6, 6))
gs_k  = fig_k.add_gridspec(2, 2, hspace=0.4, wspace=0.4)

for row_idx, rg in enumerate(['R+', 'R-']):
    ax_pos = fig_k.add_subplot(gs_k[row_idx, 0])
    ax_neg = fig_k.add_subplot(gs_k[row_idx, 1])
    plot_cdf_panel(ax_pos, lmi_df, 'lmi', rg, positive_only=True,
                   ks_df=stats_lmi, xlabel='LMI' if row_idx == 1 else '')
    plot_cdf_panel(ax_neg, lmi_df, 'lmi', rg, positive_only=False,
                   ks_df=stats_lmi, xlabel='|LMI|' if row_idx == 1 else '')

sns.despine()


# ============================================================================
# Figure l: CDF of classifier weights
# ============================================================================

fig_l = plt.figure(figsize=(6, 6))
gs_l  = fig_l.add_gridspec(2, 2, hspace=0.4, wspace=0.4)

for row_idx, rg in enumerate(['R+', 'R-']):
    ax_pos = fig_l.add_subplot(gs_l[row_idx, 0])
    ax_neg = fig_l.add_subplot(gs_l[row_idx, 1])
    plot_cdf_panel(ax_pos, weights_df, 'classifier_weight', rg, positive_only=True,
                   ks_df=stats_weight, xlabel='Classifier weight' if row_idx == 1 else '')
    plot_cdf_panel(ax_neg, weights_df, 'classifier_weight', rg, positive_only=False,
                   ks_df=stats_weight, xlabel='|Classifier weight|' if row_idx == 1 else '')

sns.despine()


# ============================================================================
# Save
# ============================================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

fig_k.savefig(os.path.join(OUTPUT_DIR, 'supp_3k.svg'), format='svg', dpi=300, bbox_inches='tight')
print("Saved: supp_3k.svg")

fig_l.savefig(os.path.join(OUTPUT_DIR, 'supp_3l.svg'), format='svg', dpi=300, bbox_inches='tight')
print("Saved: supp_3l.svg")

# Data CSVs (projection-type cells only)
lmi_df[lmi_df['cell_type_group'].isin(CELL_TYPES)][
    ['mouse_id', 'roi', 'reward_group', 'cell_type_group', 'lmi', 'lmi_p']
].to_csv(os.path.join(OUTPUT_DIR, 'supp_3k_data.csv'), index=False)
print("Saved: supp_3k_data.csv")

weights_df[weights_df['cell_type_group'].isin(CELL_TYPES)][
    ['mouse_id', 'roi', 'reward_group', 'cell_type_group', 'classifier_weight']
].to_csv(os.path.join(OUTPUT_DIR, 'supp_3l_data.csv'), index=False)
print("Saved: supp_3l_data.csv")

stats_lmi.to_csv(os.path.join(OUTPUT_DIR, 'supp_3k_stats.csv'), index=False)
print("Saved: supp_3k_stats.csv")

stats_weight.to_csv(os.path.join(OUTPUT_DIR, 'supp_3l_stats.csv'), index=False)
print("Saved: supp_3l_stats.csv")

plt.show()
