"""
Figure 1—figure supplement 1a: First hit trial on Day 0 for R+ vs R-.

Bar plot (mean) with individual mouse dots showing the trial number of the
first whisker hit on Day 0, comparing reward groups.
Statistics: Mann-Whitney U test (two-sided).
"""

import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import mannwhitneyu

from fast_learning import paths
from fast_learning.plotting import reward_palette, save_figure, set_style, panel_size


# ============================================================================
# Parameters
# ============================================================================

OUTPUT_DIR = os.path.join(paths.manuscript_output_dir, 'Figure1_SupFig1')


# ============================================================================
# Load behaviour table
# ============================================================================

bh_path = os.path.join(
    paths.processed_dir, 'behavior', 'behavior_imagingmice_table_5days_cut_with_learning_curves.csv'
)
if __name__ == '__main__':
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
        alternative='two-sided',
    )
    print(f"Mann-Whitney U test: U={stat:.3f}, p={p_value:.4f}")

    # ============================================================================
    # Figure
    # ============================================================================

    set_style()

    fig, ax = plt.subplots(figsize=panel_size(w=0.7))
    sns.barplot(
        data=fh,
        x='reward_group',
        y='first_hit_trial',
        order=['R+', 'R-'],
        palette=reward_palette[::-1],
        width=0.3,
        ax=ax,
        seed=0,
    )
    sns.swarmplot(
        data=fh,
        x='reward_group',
        y='first_hit_trial',
        order=['R+', 'R-'],
        color='grey',
        dodge=True,
        alpha=0.4,
        ax=ax,
        size=2,
    )
    ax.set_ylabel('First hit trial')
    ax.set_xlabel('')
    sns.despine()

    # ============================================================================
    # Save
    # ============================================================================

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    save_figure(fig, os.path.join(OUTPUT_DIR, 'figure1_supfig1_a.svg'))
    print("Saved: figure1_supfig1_a.svg")

    fh.to_csv(os.path.join(OUTPUT_DIR, 'figure1_supfig1_a_data.csv'), index=False)
    print("Saved: figure1_supfig1_a_data.csv")

    stats_df = pd.DataFrame(
        [
            {
                'test': 'Mann-Whitney U',
                'group_1': 'R+',
                'group_2': 'R-',
                'statistic': stat,
                'p_value': p_value,
                'alternative': 'two-sided',
            }
        ]
    )
    stats_df.to_csv(os.path.join(OUTPUT_DIR, 'figure1_supfig1_a_stats.csv'), index=False)
    print("Saved: figure1_supfig1_a_stats.csv")

    plt.close()
