"""
Figure 3f, g: Learning Modulation Index (LMI) for all cells.

  Panel f: Distribution of LMI values (R+ vs R-)
  Panel g: Proportion of cells with significant positive/negative LMI (R+ vs R-)

Statistics: f: KS distance between pooled cells, with a permutation p-value that
            respects mice (whole mice reassigned to R+/R-, 10,000 shuffles).
            g: Mann-Whitney U test (R+ vs R-, one proportion per mouse).
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import mannwhitneyu

from fast_learning import paths, database
from fast_learning.plotting import reward_palette, save_figure, set_style, panel_size
from fast_learning.stats import format_p, ks_permutation_test


# ============================================================================
# Parameters
# ============================================================================

LMI_POS_THRESHOLD = 0.975
LMI_NEG_THRESHOLD = 0.025
OUTPUT_DIR = os.path.join(paths.manuscript_output_dir, 'Figure3')


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

    lmi_df['lmi_pos'] = lmi_df['lmi_p'] >= LMI_POS_THRESHOLD
    lmi_df['lmi_neg'] = lmi_df['lmi_p'] <= LMI_NEG_THRESHOLD

    # ============================================================================
    # Compute proportions per mouse
    # ============================================================================

    lmi_prop = (
        lmi_df.groupby(['mouse_id', 'reward_group'])[['lmi_pos', 'lmi_neg']]
        .apply(lambda x: x.sum() / x.count())
        .reset_index()
    )

    # ============================================================================
    # Statistics
    # ============================================================================

    # KS distance R+ vs R-, permutation p-value with whole mice reassigned.
    ks = ks_permutation_test(lmi_df, 'lmi', 'reward_group', ('R+', 'R-'), shuffle='between_mice')
    ks_stat, ks_p = ks['ks_statistic'], ks['p_value']
    print(f"LMI distribution KS: D={ks_stat:.3f}, permutation p={ks_p:.4f}")

    # Mann-Whitney U: R+ vs R- on proportions
    stats_rows = []
    for sign in ['lmi_pos', 'lmi_neg']:
        rp = lmi_prop[lmi_prop['reward_group'] == 'R+'][sign]
        rm = lmi_prop[lmi_prop['reward_group'] == 'R-'][sign]
        stat, p = mannwhitneyu(rp, rm, alternative='two-sided')
        stats_rows.append({'lmi_sign': sign, 'test': 'Mann-Whitney U', 'statistic': stat, 'p_value': p})
        print(f"{sign}: U={stat:.3f}, p={p:.4f}")

    stats_rows.append(
        {
            'lmi_sign': 'all (distribution)',
            'test': 'KS, permutation across mice',
            'statistic': ks_stat,
            'p_value': ks_p,
        }
    )
    stats_df = pd.DataFrame(stats_rows)

    # ============================================================================
    # Figure f: LMI distribution
    # ============================================================================

    set_style()

    fig_f, ax_f = plt.subplots(1, 1, figsize=panel_size())

    bin_edges = np.linspace(-1, 1, 31)
    for rg, color in zip(['R-', 'R+'], reward_palette):
        sns.histplot(
            lmi_df[lmi_df['reward_group'] == rg]['lmi'],
            bins=bin_edges,
            kde=True,
            stat='probability',
            color=color,
            label=rg,
            alpha=0.5,
            ax=ax_f,
        )

    ax_f.text(0.98, 0.98, f'KS {format_p(ks_p)}', ha='right', va='top', transform=ax_f.transAxes)
    ax_f.set_xlim(-1, 1)
    ax_f.set_xlabel('LMI')
    ax_f.set_ylabel('Probability')
    ax_f.legend(frameon=False)
    sns.despine(trim=True)
    plt.tight_layout()

    # ============================================================================
    # Figure g: proportion bar plots
    # ============================================================================

    fig_g, axes_g = plt.subplots(1, 2, figsize=panel_size(2, w=0.7), sharey=True)

    for ax, sign, title in [
        (axes_g[0], 'lmi_pos', 'Positive LMI'),
        (axes_g[1], 'lmi_neg', 'Negative LMI'),
    ]:
        sns.barplot(
            data=lmi_prop,
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
            data=lmi_prop, x='reward_group', order=['R+', 'R-'], y=sign, color='k', size=2, alpha=0.7, ax=ax
        )
        ax.set_title(title)
        ax.set_xlabel('')
        ax.set_ylabel('Proportion of cells')

        stat_row = stats_df[stats_df['lmi_sign'] == sign]
        if not stat_row.empty:
            ax.annotate(
                format_p(stat_row.iloc[0]['p_value']),
                xy=(0.5, 0.95),
                xycoords='axes fraction',
                ha='center',
                va='top',
                color='black',
            )

    sns.despine(trim=True)
    plt.tight_layout()

    # ============================================================================
    # Save
    # ============================================================================

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    save_figure(fig_f, os.path.join(OUTPUT_DIR, 'figure_3f.svg'))
    print("Saved: figure_3f.svg")

    save_figure(fig_g, os.path.join(OUTPUT_DIR, 'figure_3g.svg'))
    print("Saved: figure_3g.svg")

    stats_df.to_csv(os.path.join(OUTPUT_DIR, 'figure_3f_g_stats.csv'), index=False)
    print("Saved: figure_3f_g_stats.csv")

    plt.close()
