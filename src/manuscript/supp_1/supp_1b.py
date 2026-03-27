"""
Supplementary Figure 1b: Whisker lick probability on Days 0, +1, +2 for R+ vs R-.

Bar plot (mean) with individual mouse dots per day, comparing reward groups.
Significance stars from per-day Mann-Whitney U tests (two-sided).
"""

import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import mannwhitneyu

sys.path.append(r'/home/aprenard/repos/fast-learning')
import src.utils.utils_io as io
from src.utils.utils_plot import behavior_palette


# ============================================================================
# Parameters
# ============================================================================

DAYS = [0, 1, 2]
OUTPUT_DIR = os.path.join(io.manuscript_output_dir, 'supp_1', 'output')


# ============================================================================
# Load behaviour table
# ============================================================================

bh_path = os.path.join(io.processed_dir, 'behavior',
                       'behavior_imagingmice_table_5days_cut_with_learning_curves.csv')
table = pd.read_csv(bh_path)


# ============================================================================
# Compute per-mouse average per day
# ============================================================================

day_data = table[table['day'].isin(DAYS)]
avg_performance = (day_data
                   .groupby(['day', 'mouse_id', 'reward_group'])['outcome_w']
                   .mean()
                   .reset_index())
avg_performance['outcome_w'] *= 100  # convert to percentage


# ============================================================================
# Statistics: Mann-Whitney U test per day
# ============================================================================

stats_rows = []
for day in DAYS:
    df_day = avg_performance[avg_performance['day'] == day]
    group_rp = df_day[df_day['reward_group'] == 'R+']['outcome_w']
    group_rm = df_day[df_day['reward_group'] == 'R-']['outcome_w']
    stat, p_value = mannwhitneyu(group_rp, group_rm, alternative='two-sided')
    stats_rows.append({'day': day, 'group_1': 'R+', 'group_2': 'R-',
                       'test': 'Mann-Whitney U', 'statistic': stat, 'p_value': p_value})
    print(f"Day {day:+d}: U={stat:.3f}, p={p_value:.4f}")

stats_df = pd.DataFrame(stats_rows)


# ============================================================================
# Figure
# ============================================================================

sns.set_theme(context='paper', style='ticks', font='sans-serif', font_scale=1,
              rc={'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none'})

fig, ax = plt.subplots(figsize=(8, 6))

sns.barplot(data=avg_performance, x='day', y='outcome_w', hue='reward_group',
            hue_order=['R+', 'R-'], palette=behavior_palette[2:4][::-1],
            width=0.3, dodge=True, ax=ax)
sns.swarmplot(data=avg_performance, x='day', y='outcome_w', hue='reward_group',
              hue_order=['R+', 'R-'], dodge=True, color='grey', alpha=0.6, ax=ax)

# Significance stars
for row in stats_rows:
    xpos = DAYS.index(row['day'])
    p = row['p_value']
    stars = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
    if stars:
        ax.text(xpos, 95, stars, ha='center', va='bottom', color='black', fontsize=14)

ax.set_xlabel('Day')
ax.set_ylabel('Lick probability (%)')
ax.set_ylim([0, 100])
ax.legend(title='Reward group')
sns.despine(trim=True)


# ============================================================================
# Save
# ============================================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

fig.savefig(os.path.join(OUTPUT_DIR, 'supp_1b.svg'), format='svg', dpi=300, bbox_inches='tight')
print(f"Saved: supp_1b.svg")

avg_performance.to_csv(os.path.join(OUTPUT_DIR, 'supp_1b_data.csv'), index=False)
print(f"Saved: supp_1b_data.csv")

stats_df.to_csv(os.path.join(OUTPUT_DIR, 'supp_1b_stats.csv'), index=False)
print(f"Saved: supp_1b_stats.csv")

plt.show()
