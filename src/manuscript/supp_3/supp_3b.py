"""
Supplementary Figure 3b: Grand-average mapping-trial PSTHs for projection
neurons (wS2 and wM1) across learning days, R+ vs R-.

2 rows (wS2, wM1) × 5 columns (Days -2 to +2). Each panel shows the mean ±
CI across mice (variance = mice; minimum 3 cells per mouse to include).
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.append(r'/home/aprenard/repos/NWB_analysis')
sys.path.append(r'/home/aprenard/repos/fast-learning')

import src.utils.utils_imaging as utils_imaging
import src.utils.utils_io as io
from src.utils.utils_plot import reward_palette


# ============================================================================
# Parameters
# ============================================================================

SAMPLING_RATE = 30
WIN_SEC       = (-0.5, 1.5)
BASELINE_WIN  = (0, 1)
BASELINE_WIN  = (int(BASELINE_WIN[0] * SAMPLING_RATE), int(BASELINE_WIN[1] * SAMPLING_RATE))
DAYS          = [-2, -1, 0, 1, 2]
CELL_TYPES    = ['wS2', 'wM1']
MIN_CELLS     = 3   # minimum cells per mouse to include that mouse

OUTPUT_DIR = os.path.join(io.manuscript_output_dir, 'supp_3', 'output')


# ============================================================================
# Load and process imaging data
# ============================================================================

_, _, mice, db = io.select_sessions_from_db(io.db_path, io.nwb_dir,
                                             two_p_imaging='yes',
                                             experimenters=['AR', 'GF', 'MI'])

psth_records = []
for mouse_id in mice:
    reward_group = io.get_mouse_reward_group_from_db(io.db_path, mouse_id)
    folder = os.path.join(io.processed_dir, 'mice')
    xarr = utils_imaging.load_mouse_xarray(mouse_id, folder, 'tensor_xarray_mapping_data.nc')
    xarr = utils_imaging.substract_baseline(xarr, 2, BASELINE_WIN)
    xarr = xarr.sel(trial=xarr['day'].isin(DAYS))
    xarr = xarr.sel(time=slice(WIN_SEC[0], WIN_SEC[1]))
    xarr = xarr.groupby('day').mean(dim='trial')

    xarr.name = 'psth'
    df = xarr.to_dataframe().reset_index()
    df['mouse_id']     = mouse_id
    df['reward_group'] = reward_group
    psth_records.append(df)

psth = pd.concat(psth_records, ignore_index=True)


# ============================================================================
# Aggregate: per-mouse mean per cell type, then convert to % dF/F0
# ============================================================================

psth_filtered = utils_imaging.filter_data_by_cell_count(psth, MIN_CELLS)

data_ctype = psth_filtered[psth_filtered['cell_type'].isin(CELL_TYPES)].copy()
data_ctype = (data_ctype
              .groupby(['mouse_id', 'day', 'reward_group', 'time', 'cell_type'])['psth']
              .mean()
              .reset_index())
data_ctype['psth'] = data_ctype['psth'] * 100


# ============================================================================
# Figure
# ============================================================================

sns.set_theme(context='paper', style='ticks', font='sans-serif', font_scale=1,
              rc={'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none'})

fig, axes = plt.subplots(len(CELL_TYPES), len(DAYS),
                          figsize=(18, 10), sharey=True)

for i, cell_type in enumerate(CELL_TYPES):
    for j, day in enumerate(DAYS):
        ax = axes[i, j]
        d = data_ctype[(data_ctype['cell_type'] == cell_type) &
                       (data_ctype['day'] == day)]
        sns.lineplot(data=d, x='time', y='psth', errorbar='ci',
                     hue='reward_group', hue_order=['R-', 'R+'],
                     palette=reward_palette, estimator='mean',
                     ax=ax, legend=False)
        ax.axvline(0, color='#FF9600', linestyle='-')
        ax.set_title(f'{cell_type} — Day {day:+d}')
        ax.set_ylabel('DF/F0 (%)' if j == 0 else '')
        ax.set_xlabel('Time (s)')

plt.ylim(-1, 16)
plt.tight_layout()
sns.despine()


# ============================================================================
# Save
# ============================================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

fig.savefig(os.path.join(OUTPUT_DIR, 'supp_3b.svg'), format='svg', dpi=300, bbox_inches='tight')
print("Saved: supp_3b.svg")

data_ctype.to_csv(os.path.join(OUTPUT_DIR, 'supp_3b_data.csv'), index=False)
print("Saved: supp_3b_data.csv")

plt.show()
