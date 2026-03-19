"""
Supplementary Figure 1i-j: Reaction times for auditory, whisker, and no-stim
hit trials.

Panel i: Mean reaction time per stimulus type across days (bar plot).
Panel j: Mean reaction time across trials within Day 0 (line plot).

Only hit trials (lick_flag == 1 and outcome == 1) are included.
"""

import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sys.path.append(r'/home/aprenard/repos/fast-learning')
import src.utils.utils_io as io
from src.utils.utils_plot import trial_type_rew_palette, trial_type_nonrew_palette


# ============================================================================
# Parameters
# ============================================================================

DAYS = [-2, -1, 0, 1, 2]
MAX_TRIALS_RT = 100
MIN_MICE_PER_TRIAL = 5   # minimum mice required to plot a trial bin in panel j
OUTPUT_DIR = os.path.join(io.manuscript_output_dir, 'supp_1', 'output')

# (filter_col, outcome_col, trial_col, label, rp_palette_idx, rm_palette_idx)
STIM_DEFS = [
    ('auditory_stim', 'outcome_a', 'trial_a', 'Auditory', 1, 0),
    ('whisker_stim',  'outcome_w', 'trial_w', 'Whisker',  3, 3),
    ('no_stim',       'outcome_c', 'trial_c', 'No stim',  5, 4),
]


# ============================================================================
# Load data
# ============================================================================

bh_path = os.path.join(io.processed_dir, 'behavior',
                       'behavior_imagingmice_table_5days_cut.csv')
table = pd.read_csv(bh_path)

table = table[table['lick_flag'] == 1]
table['reaction_time'] = table['lick_time'] - table['stim_onset']

# Re-index per stim type within session
for trial_col, stim_col in [('trial_w', 'whisker_stim'),
                              ('trial_a', 'auditory_stim'),
                              ('trial_c', 'no_stim')]:
    table[trial_col] = table.groupby(['mouse_id', 'session_id']).cumcount()


# ============================================================================
# Panel i: per-mouse mean RT per stim type × day
# ============================================================================

rt_rows = []
for stim_col, outcome_col, trial_col, stim_label, rpi, rmi in STIM_DEFS:
    df_stim = table.loc[(table[stim_col] == 1) & (table[outcome_col] == 1)]
    for (mouse_id, reward_group, day), grp in df_stim.groupby(['mouse_id', 'reward_group', 'day']):
        if day not in DAYS:
            continue
        rt_rows.append({
            'mouse_id':      mouse_id,
            'reward_group':  reward_group,
            'day':           day,
            'stim_type':     stim_label,
            'reaction_time': grp['reaction_time'].mean(),
        })
rt_df = pd.DataFrame(rt_rows)

sns.set_theme(context='paper', style='ticks', font='sans-serif', font_scale=1,
              rc={'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none'})

fig_i, axes_i = plt.subplots(1, 3, figsize=(14, 4), sharey=True)

for ax, (stim_col, outcome_col, trial_col, stim_label, rpi, rmi) in zip(axes_i, STIM_DEFS):
    color_rp = trial_type_rew_palette[rpi]
    color_rm = trial_type_nonrew_palette[rmi]

    df_plot = rt_df[rt_df['stim_type'] == stim_label]

    sns.barplot(data=df_plot, x='day', y='reaction_time',
                hue='reward_group', hue_order=['R+', 'R-'],
                palette={'R+': color_rp, 'R-': color_rm},
                order=DAYS, errorbar='ci', capsize=0.05, alpha=0.8, ax=ax)

    day_positions = {d: i for i, d in enumerate(DAYS)}
    bar_width = 0.35
    group_offsets = {'R+': -bar_width / 2, 'R-': bar_width / 2}

    for mouse_id in df_plot['mouse_id'].unique():
        mouse_data = df_plot[
            (df_plot['mouse_id'] == mouse_id) & (df_plot['day'].isin(DAYS))
        ].sort_values('day')
        rg = mouse_data['reward_group'].iloc[0]
        xs = [day_positions[d] + group_offsets[rg] for d in mouse_data['day']]
        ax.scatter(xs, mouse_data['reaction_time'].values,
                   color='grey', s=8, alpha=0.5, zorder=5, linewidths=0)

    ax.set_title(stim_label)
    ax.set_xlabel('Day')
    ax.set_ylabel('Reaction time (s)' if ax is axes_i[0] else '')
    ax.legend(frameon=False)

sns.despine()
plt.tight_layout()


# ============================================================================
# Panel j: mean RT across trials within Day 0
# ============================================================================

fig_j, axes_j = plt.subplots(1, 3, figsize=(14, 4), sharey=True)

rt_day0_rows = []
for ax, (stim_col, outcome_col, trial_col, stim_label, rpi, rmi) in zip(axes_j, STIM_DEFS):
    color_rp = trial_type_rew_palette[rpi]
    color_rm = trial_type_nonrew_palette[rmi]

    df_stim = table.loc[
        (table[stim_col] == 1) &
        (table[outcome_col] == 1) &
        (table['day'] == 0) &
        (table[trial_col] < MAX_TRIALS_RT)
    ]

    for rg, color in [('R+', color_rp), ('R-', color_rm)]:
        df_rg = df_stim[df_stim['reward_group'] == rg]
        mouse_counts = df_rg.groupby(trial_col)['mouse_id'].nunique()
        valid_trials = mouse_counts[mouse_counts >= MIN_MICE_PER_TRIAL].index
        df_rg = df_rg[df_rg[trial_col].isin(valid_trials)]
        sns.lineplot(data=df_rg, x=trial_col, y='reaction_time',
                     color=color, errorbar='ci', err_style='band',
                     label=rg, ax=ax)

        for _, row in df_rg.groupby(trial_col)['reaction_time'].mean().reset_index().iterrows():
            rt_day0_rows.append({
                'stim_type': stim_label,
                'reward_group': rg,
                trial_col: row[trial_col],
                'reaction_time_mean': row['reaction_time'],
            })

    ax.set_title(stim_label)
    ax.set_xlabel('Hit trials')
    ax.set_ylabel('Reaction time (s)' if ax is axes_j[0] else '')
    ax.set_ylim(bottom=0, top=1)
    ax.legend(frameon=False)

sns.despine()
plt.tight_layout()


# ============================================================================
# Save
# ============================================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

fig_i.savefig(os.path.join(OUTPUT_DIR, 'supp_1i.svg'), format='svg', dpi=300, bbox_inches='tight')
print("Saved: supp_1i.svg")

fig_j.savefig(os.path.join(OUTPUT_DIR, 'supp_1j.svg'), format='svg', dpi=300, bbox_inches='tight')
print("Saved: supp_1j.svg")

rt_df.to_csv(os.path.join(OUTPUT_DIR, 'supp_1i_data.csv'), index=False)
print("Saved: supp_1i_data.csv")

pd.DataFrame(rt_day0_rows).to_csv(os.path.join(OUTPUT_DIR, 'supp_1j_data.csv'), index=False)
print("Saved: supp_1j_data.csv")

plt.show()
