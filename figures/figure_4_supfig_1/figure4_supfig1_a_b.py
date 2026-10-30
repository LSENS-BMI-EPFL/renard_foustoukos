"""
Figure 4—figure supplement 1a, b: CDF comparison of wS2 vs wM1 projector neurons.

  Panel a: CDF of LMI values (positive and negative, 2 reward groups)
  Panel b: CDF of classifier weights (positive and negative, 2 reward groups),
           normalised within each mouse: each mouse has its own decoder, and
           the scale of its weights varies about 3-fold across mice (smaller
           with more cells), so weights are divided by the mouse's median
           |weight| over all its cells. Pooled raw weights mix this scale with
           the mice that contribute each cell type.

Mice: for each cell type, mice with at least MIN_CELLS cells of that type, as
in Figure 4.

Statistics: KS distance between pooled wS2 and wM1 cells per reward group and
sign, with a permutation p-value that respects mice (wS2/wM1 labels shuffled
among the cells of each mouse, 10,000 shuffles).

Classifier weights are loaded from paths.processed_dir/decoding (saved by
pipeline/06_decoder.py). Cell-type labels for the weight file are retrieved from the
mapping xarrays.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from fast_learning import imaging
from fast_learning import paths, database
from fast_learning.plotting import s2_m1_palette, save_figure, set_style, panel_size
from fast_learning.stats import format_p, ks_permutation_test, significance_stars


# ============================================================================
# Parameters
# ============================================================================

CELL_TYPES = ['wS2', 'wM1']
CELL_TYPE_COLORS = {'wS2': s2_m1_palette[0], 'wM1': s2_m1_palette[1]}
RESULTS_DIR = os.path.join(paths.processed_dir, 'decoding')
MIN_CELLS = 3  # minimum cells of a type per mouse to include that mouse
XMAX_NORM = 5  # x-axis limit of the normalised weights (multiples of the median |weight|)
OUTPUT_DIR = os.path.join(paths.manuscript_output_dir, 'Figure4_SupFig1')


# ============================================================================
# Helper
# ============================================================================


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
            ks = ks_permutation_test(sub, value_col, 'cell_type_group', ('wS2', 'wM1'), shuffle='within_mice')
            stat, p = ks['ks_statistic'], ks['p_value']
            rows.append(
                {
                    'reward_group': rg,
                    'value_type': 'positive' if positive_only else ('negative' if negative_only else 'all'),
                    'n_wS2': len(ws2),
                    'n_wM1': len(wm1),
                    'ks_statistic': stat,
                    'ks_pvalue': p,
                    'test': 'KS, permutation within mice',
                    'n_mice': ks['n_mice'],
                    'ks_stars': significance_stars(p, ns='ns'),
                }
            )
    return pd.DataFrame(rows)


def plot_cdf_panel(ax, data, value_col, reward_group, positive_only, ks_df, xlabel, xmax=0.8):
    sub = data[data['reward_group'] == reward_group]
    if positive_only:
        sub = sub[sub[value_col] > 0]
        vals_fn = lambda v: v
    else:
        sub = sub[sub[value_col] < 0]
        vals_fn = np.abs  # plot absolute values for negative

    for ct in CELL_TYPES:
        values = vals_fn(sub[sub['cell_type_group'] == ct][value_col].values)
        if len(values) > 0:
            sv = np.sort(values)
            cdf = np.arange(1, len(sv) + 1) / len(sv)
            ax.plot(sv, cdf, label=ct, color=CELL_TYPE_COLORS[ct], linewidth=1, alpha=0.8)

    ax.set_xlabel(xlabel)
    ax.set_ylabel('Cumulative probability')
    sign_label = 'Positive' if positive_only else 'Negative (abs)'
    ax.set_title(f'{reward_group} — {sign_label}', fontweight='bold')
    ax.legend(frameon=False)
    ax.set_xlim(left=0, right=xmax)
    ax.set_ylim(0, 1)

    vtype = 'positive' if positive_only else 'negative'
    ks_row = ks_df[(ks_df['reward_group'] == reward_group) & (ks_df['value_type'] == vtype)]
    if not ks_row.empty:
        pval = ks_row.iloc[0]['ks_pvalue']
        ax.text(
            0.98,
            0.02,
            f'KS {format_p(pval)}',
            ha='right',
            va='bottom',
            transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
        )


if __name__ == '__main__':
    # ============================================================================
    # Load LMI data
    # ============================================================================

    lmi_df = pd.read_csv(os.path.join(paths.processed_dir, 'lmi_results.csv'))
    for mouse in lmi_df['mouse_id'].unique():
        lmi_df.loc[lmi_df['mouse_id'] == mouse, 'reward_group'] = database.get_mouse_reward_group_from_db(
            paths.db_path, mouse
        )
    lmi_df = imaging.filter_data_by_cell_count(lmi_df, MIN_CELLS)
    lmi_df['cell_type_group'] = lmi_df['cell_type'].replace({'na': 'non_projector'})

    print(f"LMI: {len(lmi_df)} cells, {lmi_df['mouse_id'].nunique()} mice")

    # ============================================================================
    # Load classifier weights + cell-type labels
    # ============================================================================

    weights_df = pd.read_csv(os.path.join(RESULTS_DIR, 'classifier_weights.csv'))

    cell_type_info = []
    for mouse_id in weights_df['mouse_id'].unique():
        try:
            xarr = imaging.load_mouse_xarray(mouse_id, paths.tensor_dir, 'tensor_xarray_mapping_data.nc')
            rois = xarr.coords['roi'].values
            cts = xarr.coords['cell_type'].values if 'cell_type' in xarr.coords else [None] * len(rois)
            for roi, ct in zip(rois, cts):
                cell_type_info.append({'mouse_id': mouse_id, 'roi': roi, 'cell_type_xr': ct})
        except Exception as e:
            print(f"Warning: could not load cell types for {mouse_id}: {e}")

    if cell_type_info:
        weights_df = weights_df.merge(pd.DataFrame(cell_type_info), on=['mouse_id', 'roi'], how='left')
    else:
        weights_df['cell_type_xr'] = None

    weights_df['cell_type_group'] = weights_df['cell_type_xr'].copy()
    weights_df.loc[
        weights_df['cell_type_xr'].isna() | (weights_df['cell_type_xr'] == 'na'), 'cell_type_group'
    ] = 'non_projector'

    scale = weights_df.groupby('mouse_id')['classifier_weight'].transform(lambda w: w.abs().median())
    weights_df['weight_norm'] = weights_df['classifier_weight'] / scale
    # Normalised by all of the mouse's cells (above), then filtered by cell type.
    weights_df = imaging.filter_data_by_cell_count(
        weights_df.assign(cell_type=weights_df['cell_type_group']), MIN_CELLS
    )
    print(f"Weights: {len(weights_df)} cells, {weights_df['mouse_id'].nunique()} mice")

    # ============================================================================
    # KS tests
    # ============================================================================

    ks_lmi_pos = compute_ks_test(lmi_df, 'lmi', ['R+', 'R-'], positive_only=True)
    ks_lmi_neg = compute_ks_test(lmi_df, 'lmi', ['R+', 'R-'], negative_only=True)
    ks_weight_pos = compute_ks_test(weights_df, 'weight_norm', ['R+', 'R-'], positive_only=True)
    ks_weight_neg = compute_ks_test(weights_df, 'weight_norm', ['R+', 'R-'], negative_only=True)

    stats_lmi = pd.concat([ks_lmi_pos, ks_lmi_neg], ignore_index=True)
    stats_weight = pd.concat([ks_weight_pos, ks_weight_neg], ignore_index=True)
    print(stats_lmi)
    print(stats_weight)

    # ============================================================================
    # Figure a: CDF of LMI
    # ============================================================================

    set_style()

    fig_k = plt.figure(figsize=panel_size(2, 2))
    gs_k = fig_k.add_gridspec(2, 2, hspace=0.4, wspace=0.4)

    for row_idx, rg in enumerate(['R+', 'R-']):
        ax_pos = fig_k.add_subplot(gs_k[row_idx, 0])
        ax_neg = fig_k.add_subplot(gs_k[row_idx, 1])
        plot_cdf_panel(
            ax_pos,
            lmi_df,
            'lmi',
            rg,
            positive_only=True,
            ks_df=stats_lmi,
            xlabel='LMI' if row_idx == 1 else '',
        )
        plot_cdf_panel(
            ax_neg,
            lmi_df,
            'lmi',
            rg,
            positive_only=False,
            ks_df=stats_lmi,
            xlabel='|LMI|' if row_idx == 1 else '',
        )

    sns.despine()

    # ============================================================================
    # Figure b: CDF of classifier weights
    # ============================================================================

    fig_l = plt.figure(figsize=panel_size(2, 2))
    gs_l = fig_l.add_gridspec(2, 2, hspace=0.4, wspace=0.4)

    for row_idx, rg in enumerate(['R+', 'R-']):
        ax_pos = fig_l.add_subplot(gs_l[row_idx, 0])
        ax_neg = fig_l.add_subplot(gs_l[row_idx, 1])
        plot_cdf_panel(
            ax_pos,
            weights_df,
            'weight_norm',
            rg,
            positive_only=True,
            ks_df=stats_weight,
            xlabel='Classifier weight (normalised)' if row_idx == 1 else '',
            xmax=XMAX_NORM,
        )
        plot_cdf_panel(
            ax_neg,
            weights_df,
            'weight_norm',
            rg,
            positive_only=False,
            ks_df=stats_weight,
            xlabel='|Classifier weight| (normalised)' if row_idx == 1 else '',
            xmax=XMAX_NORM,
        )

    sns.despine()

    # ============================================================================
    # Save
    # ============================================================================

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    save_figure(fig_k, os.path.join(OUTPUT_DIR, 'figure4_supfig1_a.svg'))
    print("Saved: figure4_supfig1_a.svg")

    save_figure(fig_l, os.path.join(OUTPUT_DIR, 'figure4_supfig1_b.svg'))
    print("Saved: figure4_supfig1_b.svg")

    # Data CSVs (projection-type cells only)
    lmi_df[lmi_df['cell_type_group'].isin(CELL_TYPES)][
        ['mouse_id', 'roi', 'reward_group', 'cell_type_group', 'lmi', 'lmi_p']
    ].to_csv(os.path.join(OUTPUT_DIR, 'figure4_supfig1_a_data.csv'), index=False)
    print("Saved: figure4_supfig1_a_data.csv")

    weights_df[weights_df['cell_type_group'].isin(CELL_TYPES)][
        ['mouse_id', 'roi', 'reward_group', 'cell_type_group', 'classifier_weight', 'weight_norm']
    ].to_csv(os.path.join(OUTPUT_DIR, 'figure4_supfig1_b_data.csv'), index=False)
    print("Saved: figure4_supfig1_b_data.csv")

    stats_lmi.to_csv(os.path.join(OUTPUT_DIR, 'figure4_supfig1_a_stats.csv'), index=False)
    print("Saved: figure4_supfig1_a_stats.csv")

    stats_weight.to_csv(os.path.join(OUTPUT_DIR, 'figure4_supfig1_b_stats.csv'), index=False)
    print("Saved: figure4_supfig1_b_stats.csv")

    plt.close()
