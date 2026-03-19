"""
Supplementary Figure 1c: Particle test — whisker hit rate and false alarm rate
across ON / OFF / ON periods (R+ mice only).

Three panels:
  1. Whisker hit rate across ON / OFF / ON epochs.
  2. No-stim false alarm rate across ON / OFF / ON epochs.
  3. OFF period only — whisker hit vs false alarm rate.

Statistics: paired Wilcoxon signed-rank tests (two-sided).
"""

import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import wilcoxon

sys.path.append(r'/home/aprenard/repos/fast-learning')
import src.utils.utils_io as io
from src.utils.utils_plot import trial_type_rew_palette


# ============================================================================
# Parameters
# ============================================================================

ON_OFF_ORDER = ['whisker_on_1', 'whisker_off', 'whisker_on_2']
OUTPUT_DIR = os.path.join(io.manuscript_output_dir, 'supp_1', 'output')


# ============================================================================
# Load particle-test behaviour table
# ============================================================================

table_path = io.adjust_path_to_host(
    r'//sv-nas1.rcp.epfl.ch/Petersen-Lab/analysis/Anthony_Renard/data_processed/behavior/behavior_particle_test.csv'
)
table_particle_test = pd.read_csv(table_path)


# ============================================================================
# Prepare data
# ============================================================================

df_w = table_particle_test.loc[table_particle_test['reward_group'] == 'R+'].copy()
df_w['outcome_w'] = df_w['outcome_w'] * 100
df_w = df_w.groupby(['mouse_id', 'behavior_type'])['outcome_w'].mean().reset_index()

df_ns = table_particle_test.loc[table_particle_test['reward_group'] == 'R+'].copy()
df_ns['outcome_c'] = df_ns['outcome_c'] * 100
df_ns = df_ns.groupby(['mouse_id', 'behavior_type'])['outcome_c'].mean().reset_index()

df_off = table_particle_test.loc[
    (table_particle_test['reward_group'] == 'R+') &
    (table_particle_test['behavior_type'] == 'whisker_off')
].copy()
df_off['outcome_w'] = df_off['outcome_w'] * 100
df_off['outcome_c'] = df_off['outcome_c'] * 100
df_off = df_off.groupby('mouse_id')[['outcome_w', 'outcome_c']].mean().reset_index()

df_off_long = df_off.melt(id_vars='mouse_id', value_vars=['outcome_w', 'outcome_c'],
                           var_name='trial_type', value_name='lick_rate')
df_off_long['trial_type'] = df_off_long['trial_type'].map(
    {'outcome_w': 'Whisker hit', 'outcome_c': 'False alarm'})


# ============================================================================
# Helper functions
# ============================================================================

def pval_to_stars(p):
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    return 'n.s.'


def draw_stat_bracket(ax, x1, x2, y, p_value, h=3):
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=0.8, color='black')
    ax.text((x1 + x2) / 2, y + h, pval_to_stars(p_value),
            ha='center', va='bottom', fontsize=8)


def paired_wilcoxon(df, group_col, value_col, group1, group2):
    g1 = df[df[group_col] == group1].sort_values('mouse_id')[value_col].values
    g2 = df[df[group_col] == group2].sort_values('mouse_id')[value_col].values
    return wilcoxon(g1, g2, alternative='two-sided')


# ============================================================================
# Statistics
# ============================================================================

stat1, p1 = paired_wilcoxon(df_w,  'behavior_type', 'outcome_w', 'whisker_on_1', 'whisker_off')
stat2, p2 = paired_wilcoxon(df_w,  'behavior_type', 'outcome_w', 'whisker_off',  'whisker_on_2')
stat3, p3 = paired_wilcoxon(df_ns, 'behavior_type', 'outcome_c', 'whisker_on_1', 'whisker_off')
stat4, p4 = paired_wilcoxon(df_ns, 'behavior_type', 'outcome_c', 'whisker_off',  'whisker_on_2')
stat5, p5 = wilcoxon(df_off['outcome_w'].values, df_off['outcome_c'].values, alternative='two-sided')

for label, stat, p in [
    ('whisker ON1 vs OFF', stat1, p1),
    ('whisker OFF vs ON2', stat2, p2),
    ('no-stim ON1 vs OFF', stat3, p3),
    ('no-stim OFF vs ON2', stat4, p4),
    ('OFF: whisker hit vs FA', stat5, p5),
]:
    print(f"{label}: W={stat:.3f}, p={p:.4f}")


# ============================================================================
# Figure
# ============================================================================

sns.set_theme(context='paper', style='ticks', font='sans-serif', font_scale=1,
              rc={'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none'})

fig, axes = plt.subplots(1, 3, figsize=(10, 5), sharey=True)

# Panel 1: whisker hit rate ON / OFF / ON
ax = axes[0]
sns.barplot(data=df_w, x='behavior_type', y='outcome_w', order=ON_OFF_ORDER,
            color=trial_type_rew_palette[3], ax=ax)
for mouse_id in df_w['mouse_id'].unique():
    v = {bt: df_w.loc[(df_w['mouse_id'] == mouse_id) & (df_w['behavior_type'] == bt), 'outcome_w'].to_numpy()
         for bt in ON_OFF_ORDER}
    if all(len(x) > 0 for x in v.values()):
        ax.plot([0, 1], [v['whisker_on_1'][0], v['whisker_off'][0]], color='grey', linewidth=1, alpha=0.8)
        ax.plot([1, 2], [v['whisker_off'][0], v['whisker_on_2'][0]], color='grey', linewidth=1, alpha=0.8)
draw_stat_bracket(ax, 0, 1, 88, p1)
draw_stat_bracket(ax, 1, 2, 88, p2)
ax.set_xticks([0, 1, 2])
ax.set_xticklabels(['ON', 'OFF', 'ON'])
ax.set_ylim(0, 100)
ax.set_ylabel('Lick rate (%)')
ax.set_xlabel('Particle test')
ax.set_title('Whisker hit rate')

# Panel 2: no-stim false alarm rate ON / OFF / ON
ax = axes[1]
sns.barplot(data=df_ns, x='behavior_type', y='outcome_c', order=ON_OFF_ORDER,
            color=trial_type_rew_palette[5], ax=ax)
for mouse_id in df_ns['mouse_id'].unique():
    v = {bt: df_ns.loc[(df_ns['mouse_id'] == mouse_id) & (df_ns['behavior_type'] == bt), 'outcome_c'].to_numpy()
         for bt in ON_OFF_ORDER}
    if all(len(x) > 0 for x in v.values()):
        ax.plot([0, 1], [v['whisker_on_1'][0], v['whisker_off'][0]], color='grey', linewidth=1, alpha=0.8)
        ax.plot([1, 2], [v['whisker_off'][0], v['whisker_on_2'][0]], color='grey', linewidth=1, alpha=0.8)
draw_stat_bracket(ax, 0, 1, 88, p3)
draw_stat_bracket(ax, 1, 2, 88, p4)
ax.set_xticks([0, 1, 2])
ax.set_xticklabels(['ON', 'OFF', 'ON'])
ax.set_ylim(0, 100)
ax.set_xlabel('Particle test')
ax.set_title('False alarm rate')

# Panel 3: OFF only — whisker hit vs false alarm
ax = axes[2]
sns.barplot(data=df_off_long, x='trial_type', y='lick_rate',
            order=['Whisker hit', 'False alarm'],
            palette={'Whisker hit': trial_type_rew_palette[3],
                     'False alarm': trial_type_rew_palette[5]},
            ax=ax)
for mouse_id in df_off['mouse_id'].unique():
    row = df_off[df_off['mouse_id'] == mouse_id]
    ax.plot([0, 1], [row['outcome_w'].values[0], row['outcome_c'].values[0]],
            color='grey', linewidth=1, alpha=0.8)
draw_stat_bracket(ax, 0, 1, 88, p5)
ax.set_ylim(0, 100)
ax.set_xlabel('OFF period')
ax.set_title('OFF: whisker vs FA')

sns.despine()
plt.tight_layout()


# ============================================================================
# Save
# ============================================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

fig.savefig(os.path.join(OUTPUT_DIR, 'supp_1c.svg'), format='svg', dpi=300, bbox_inches='tight')
print(f"Saved: supp_1c.svg")

df_w.to_csv(os.path.join(OUTPUT_DIR, 'supp_1c_data_whisker.csv'), index=False)
df_ns.to_csv(os.path.join(OUTPUT_DIR, 'supp_1c_data_nostim.csv'), index=False)
df_off.to_csv(os.path.join(OUTPUT_DIR, 'supp_1c_data_off.csv'), index=False)
print(f"Saved: data CSVs")

pd.DataFrame([
    {'comparison': 'whisker ON1 vs OFF',     'test': 'Wilcoxon', 'statistic': stat1, 'p_value': p1},
    {'comparison': 'whisker OFF vs ON2',     'test': 'Wilcoxon', 'statistic': stat2, 'p_value': p2},
    {'comparison': 'no-stim ON1 vs OFF',     'test': 'Wilcoxon', 'statistic': stat3, 'p_value': p3},
    {'comparison': 'no-stim OFF vs ON2',     'test': 'Wilcoxon', 'statistic': stat4, 'p_value': p4},
    {'comparison': 'OFF: whisker hit vs FA', 'test': 'Wilcoxon', 'statistic': stat5, 'p_value': p5},
]).to_csv(os.path.join(OUTPUT_DIR, 'supp_1c_stats.csv'), index=False)
print(f"Saved: supp_1c_stats.csv")

plt.show()
