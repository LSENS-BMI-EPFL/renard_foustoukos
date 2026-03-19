"""
Supplementary Figure 1f: Day 0 lick probability for whisker, auditory, and
no-stim trial types on a common time axis (minutes from session start).

Traces are interpolated onto a regular time grid per mouse and cut at the
100th whisker trial. R+ and R- share the same axis; colour encodes stimulus
type × reward group.
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.append(r'/home/aprenard/repos/fast-learning')
import src.utils.utils_io as io
from src.utils.utils_plot import behavior_palette


# ============================================================================
# Parameters
# ============================================================================

DAY = 0
TIME_RESOLUTION = 5    # seconds between interpolation grid points
MAX_WHISKER_TRIALS = 100
OUTPUT_DIR = os.path.join(io.manuscript_output_dir, 'supp_1', 'output')


# ============================================================================
# Load data
# ============================================================================

bh_path = os.path.join(io.processed_dir, 'behavior',
                       'behavior_imagingmice_table_5days_cut_with_learning_curves.csv')
table = pd.read_csv(bh_path)
df_day0 = table.loc[table['day'] == DAY].copy()


# ============================================================================
# Interpolate learning curves onto common time grid
# ============================================================================

stim_type_map = {
    'whisker':  ('whisker_stim',  'learning_curve_w'),
    'auditory': ('auditory_stim', 'learning_curve_a'),
    'no_stim':  ('no_stim',       'learning_curve_ns'),
}

rg_stim_colors = {
    'R+': {'whisker': behavior_palette[3], 'auditory': behavior_palette[1], 'no_stim': behavior_palette[5]},
    'R-': {'whisker': behavior_palette[2], 'auditory': behavior_palette[0], 'no_stim': behavior_palette[4]},
}
stim_labels = {'whisker': 'Whisker', 'auditory': 'Auditory', 'no_stim': 'No stim'}

max_time = df_day0['stim_onset'].max()
time_grid = np.arange(0, max_time + TIME_RESOLUTION, TIME_RESOLUTION)

interp_records = []
for mouse in df_day0['mouse_id'].unique():
    mouse_data = df_day0[df_day0['mouse_id'] == mouse]
    reward_group = mouse_data['reward_group'].iloc[0]

    whisker_onsets = (mouse_data.loc[mouse_data['whisker_stim'] == 1, 'stim_onset']
                      .dropna().sort_values().reset_index(drop=True))
    if len(whisker_onsets) == 0:
        continue
    cutoff_time = whisker_onsets.iloc[min(MAX_WHISKER_TRIALS, len(whisker_onsets)) - 1]

    for stim_name, (stim_col, lc_col) in stim_type_map.items():
        if lc_col not in mouse_data.columns:
            continue
        stim_data = mouse_data.loc[mouse_data[stim_col] == 1, ['stim_onset', lc_col]].dropna()
        if len(stim_data) < 2:
            continue

        t_mouse = time_grid[time_grid <= cutoff_time]
        lc_interp = np.interp(t_mouse, stim_data['stim_onset'].values, stim_data[lc_col].values)

        for t, lc in zip(t_mouse, lc_interp):
            interp_records.append({
                'mouse_id': mouse,
                'reward_group': reward_group,
                'stim_type': stim_name,
                'time': t / 60,
                'lick_probability': lc,
            })

df_interp = pd.DataFrame(interp_records)


# ============================================================================
# Figure
# ============================================================================

sns.set_theme(context='paper', style='ticks', font='sans-serif', font_scale=1,
              rc={'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none'})

fig, ax = plt.subplots(figsize=(4, 4))

for rg in ['R+', 'R-']:
    df_rg = df_interp[df_interp['reward_group'] == rg]
    for stim_name in ['whisker', 'auditory', 'no_stim']:
        df_stim = df_rg[df_rg['stim_type'] == stim_name]
        sns.lineplot(data=df_stim, x='time', y='lick_probability',
                     color=rg_stim_colors[rg][stim_name],
                     errorbar='ci', err_style='band',
                     label=f'{stim_labels[stim_name]} {rg}', ax=ax)

ax.set_xlabel('Time from session start (min)')
ax.set_ylabel('Lick probability')
ax.set_ylim(-0.1, 1.05)
ax.set_xlim(right=50)
sns.despine()
plt.tight_layout()


# ============================================================================
# Save
# ============================================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

fig.savefig(os.path.join(OUTPUT_DIR, 'supp_1f.svg'), format='svg', dpi=300, bbox_inches='tight')
print("Saved: supp_1f.svg")

df_interp.to_csv(os.path.join(OUTPUT_DIR, 'supp_1f_data.csv'), index=False)
print("Saved: supp_1f_data.csv")

plt.show()
