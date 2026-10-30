"""
Figure 4e, f, i, j: Proportions and distributions of LMI for
projection neurons.

  Panel e: wS2 — distribution of LMI values (R+ vs R-)
  Panel f: wS2 — proportion of cells with significant positive/negative LMI
  Panel i: wM1 — distribution of LMI values (R+ vs R-)
  Panel j: wM1 — proportion of cells with significant positive/negative LMI

Statistics: f, j: Mann-Whitney U test (R+ vs R-, one proportion per mouse)
           per cell type x LMI sign.
           e, i: KS distance between pooled cells (R+ vs R-) per cell type,
           with a permutation p-value that respects mice (whole mice
           reassigned to R+/R-, 10,000 shuffles).

Mice: for each cell type, mice with at least MIN_CELLS cells of that type, as
in Fig. 4b-d, g-h (a proportion from one or two cells is noise).
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import mannwhitneyu

from fast_learning import database, imaging, paths
from fast_learning.plotting import reward_palette, save_figure, set_style, panel_size
from fast_learning.stats import format_p, ks_permutation_test


# ============================================================================
# Parameters
# ============================================================================

LMI_POS_THRESHOLD = 0.975
LMI_NEG_THRESHOLD = 0.025
MIN_CELLS = 3  # minimum cells of a type per mouse to include that mouse
OUTPUT_DIR = os.path.join(paths.manuscript_output_dir, 'Figure4')


if __name__ == '__main__':
    # ============================================================================
    # Load LMI data and assign reward groups
    # ============================================================================

    lmi_df = pd.read_csv(os.path.join(paths.processed_dir, 'lmi_results.csv'))

    _, _, mice, _ = database.select_sessions_from_db(paths.db_path, paths.nwb_dir, two_p_imaging='yes')

    for mouse in lmi_df['mouse_id'].unique():
        lmi_df.loc[lmi_df['mouse_id'] == mouse, 'reward_group'] = database.get_mouse_reward_group_from_db(
            paths.db_path, mouse
        )

    lmi_df = lmi_df.loc[lmi_df['mouse_id'].isin(mice)]
    lmi_df = imaging.filter_data_by_cell_count(lmi_df, MIN_CELLS)

    lmi_df['lmi_pos'] = lmi_df['lmi_p'] >= LMI_POS_THRESHOLD
    lmi_df['lmi_neg'] = lmi_df['lmi_p'] <= LMI_NEG_THRESHOLD

    # ============================================================================
    # Compute proportions
    # ============================================================================

    lmi_prop_ct = (
        lmi_df.groupby(['mouse_id', 'reward_group', 'cell_type'])[['lmi_pos', 'lmi_neg']]
        .apply(lambda x: x.sum() / x.count())
        .reset_index()
    )

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
            stats_rows.append(
                {'cell_type': ct, 'lmi_sign': sign, 'test': 'Mann-Whitney U', 'statistic': stat, 'p_value': p}
            )
            print(f"{ct} {sign}: U={stat:.3f}, p={p:.4f}")

    stats_df = pd.DataFrame(stats_rows)

    # ============================================================================
    # Statistics: KS test (R+ vs R-) per cell type for distributions
    # ============================================================================

    ks_rows = []
    for ct in ['wS2', 'wM1']:
        sub = lmi_df[lmi_df['cell_type'] == ct]
        ks = ks_permutation_test(sub, 'lmi', 'reward_group', ('R+', 'R-'), shuffle='between_mice')
        stat, p = ks['ks_statistic'], ks['p_value']
        ks_rows.append(
            {
                'cell_type': ct,
                'test': 'KS, permutation across mice',
                'statistic': stat,
                'p_value': p,
                'n_mice': ks['n_mice'],
            }
        )
        print(f"{ct} LMI distribution KS: D={stat:.3f}, permutation p={p:.4f}")

    ks_df = pd.DataFrame(ks_rows)

    # ============================================================================
    # Helper
    # ============================================================================

    def plot_distribution(cell_type, ax):
        data = lmi_df[lmi_df['cell_type'] == cell_type]
        bin_edges = np.linspace(-1, 1, 31)
        for rg, color in zip(['R-', 'R+'], reward_palette):
            sns.histplot(
                data[data['reward_group'] == rg]['lmi'],
                bins=bin_edges,
                kde=True,
                stat='probability',
                color=color,
                label=rg,
                alpha=0.5,
                ax=ax,
            )
        ks_row = ks_df[ks_df['cell_type'] == cell_type]
        if not ks_row.empty:
            p_text = f"KS {format_p(ks_row.iloc[0]['p_value'])}"
            ax.text(0.98, 0.98, p_text, ha='right', va='top', transform=ax.transAxes)
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
            sns.barplot(
                data=data,
                x='reward_group',
                order=['R+', 'R-'],
                hue='reward_group',
                hue_order=['R-', 'R+'],
                y=sign,
                palette=reward_palette,
                legend=False,
                ax=ax,
                seed=0,
            )
            sns.swarmplot(
                data=data, x='reward_group', order=['R+', 'R-'], y=sign, color='k', size=2, alpha=0.7, ax=ax
            )
            ax.set_title(title)
            ax.set_xlabel('')
            ax.set_ylabel('Proportion of cells')

            # Significance annotation
            stat_row = stats_df[(stats_df['cell_type'] == cell_type) & (stats_df['lmi_sign'] == sign)]
            if not stat_row.empty:
                ax.annotate(
                    format_p(stat_row.iloc[0]['p_value']),
                    xy=(0.5, 0.95),
                    xycoords='axes fraction',
                    ha='center',
                    va='top',
                    color='black',
                )

    # ============================================================================
    # Figures
    # ============================================================================

    set_style()

    # wS2 — panel e: distribution
    fig_wS2_dist, ax_wS2_dist = plt.subplots(1, 1, figsize=panel_size())
    plot_distribution('wS2', ax_wS2_dist)
    sns.despine(trim=True)
    plt.tight_layout()

    # wS2 — panel f: proportions
    fig_wS2, axes_wS2 = plt.subplots(1, 2, figsize=panel_size(2, w=0.7), sharey=True)
    plot_proportion_pair('wS2', axes_wS2[0], axes_wS2[1])
    sns.despine(trim=True)
    plt.tight_layout()

    # wM1 — panel i: distribution
    fig_wM1_dist, ax_wM1_dist = plt.subplots(1, 1, figsize=panel_size())
    plot_distribution('wM1', ax_wM1_dist)
    sns.despine(trim=True)
    plt.tight_layout()

    # wM1 — panel j: proportions
    fig_wM1, axes_wM1 = plt.subplots(1, 2, figsize=panel_size(2, w=0.7), sharey=True)
    plot_proportion_pair('wM1', axes_wM1[0], axes_wM1[1])
    sns.despine(trim=True)
    plt.tight_layout()

    # ============================================================================
    # Save
    # ============================================================================

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    save_figure(fig_wS2_dist, os.path.join(OUTPUT_DIR, 'figure_4e.svg'))
    print("Saved: figure_4e.svg")

    save_figure(fig_wS2, os.path.join(OUTPUT_DIR, 'figure_4f.svg'))
    print("Saved: figure_4f.svg")

    save_figure(fig_wM1_dist, os.path.join(OUTPUT_DIR, 'figure_4i.svg'))
    print("Saved: figure_4i.svg")

    save_figure(fig_wM1, os.path.join(OUTPUT_DIR, 'figure_4j.svg'))
    print("Saved: figure_4j.svg")

    lmi_prop_ct[lmi_prop_ct['cell_type'].isin(['wS2', 'wM1'])].to_csv(
        os.path.join(OUTPUT_DIR, 'figure_4e_f_i_j_data.csv'), index=False
    )
    print("Saved: figure_4e_f_i_j_data.csv")

    stats_df.to_csv(os.path.join(OUTPUT_DIR, 'figure_4e_f_i_j_stats.csv'), index=False)
    print("Saved: figure_4e_f_i_j_stats.csv")

    ks_df.to_csv(os.path.join(OUTPUT_DIR, 'figure_4e_i_ks_stats.csv'), index=False)
    print("Saved: figure_4e_i_ks_stats.csv")

    plt.close()
