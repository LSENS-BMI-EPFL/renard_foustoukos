"""
Illustration of mapping trial PSTHs for individual LMI example cells.

Plots a single figure (3 rows × 5 days) showing one example cell per row:
  - Row 0: best negative LMI cell
  - Row 1: best positive LMI cell
  - Row 2: average positive LMI cell

Each cell is selected as the one whose lmi_p is closest to the corresponding
PVAL_* parameter. Adjust those parameters to browse the LMI population and
pick illustrative examples.
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.append(r'/home/aprenard/repos/NWB_analysis')
sys.path.append(r'/home/aprenard/repos/fast-learning')

import src.utils.utils_io as io
import src.utils.utils_imaging as utils_imaging
from src.utils.utils_plot import *
from src.utils.utils_behavior import *


# =============================================================================
# PARAMETERS
# =============================================================================

# Target lmi_p values — each selects the significant cell whose lmi_p is
# closest to this value. Adjust to navigate the LMI distribution.
PVAL_NEG = 0.0     # best negative LMI (search within lmi_p <= 0.025)
PVAL_POS = 1.0     # best positive LMI (search within lmi_p >= 0.975)
PVAL_AVG = 0.98    # average positive LMI (search within lmi_p >= 0.975)

LMI_POSITIVE_THRESHOLD = 0.975
LMI_NEGATIVE_THRESHOLD = 0.025

WIN_SEC = (-0.5, 1.5)
DAYS = [-2, -1, 0, 1, 2]


# =============================================================================
# LOAD DATA
# =============================================================================

lmi_df = pd.read_csv(os.path.join(io.processed_dir, 'lmi_results.csv'))

pos_lmi = lmi_df.loc[lmi_df['lmi_p'] >= LMI_POSITIVE_THRESHOLD]
neg_lmi = lmi_df.loc[lmi_df['lmi_p'] <= LMI_NEGATIVE_THRESHOLD]


# =============================================================================
# CELL SELECTION
# =============================================================================

def select_cell(pool, target_pval):
    """Return the row in pool whose lmi_p is closest to target_pval."""
    idx = (pool['lmi_p'] - target_pval).abs().idxmin()
    return pool.loc[idx]


cell_neg = select_cell(neg_lmi, PVAL_NEG)
cell_pos = select_cell(pos_lmi, PVAL_POS)
cell_avg = select_cell(pos_lmi, PVAL_AVG)

cells = [
    (cell_neg, 'Negative LMI'),
    (cell_pos, 'Best positive LMI'),
    (cell_avg, 'Average positive LMI'),
]

print("Selected cells:")
for cell, label in cells:
    print(f"  {label}: {cell['mouse_id']} ROI {int(cell['roi'])} "
          f"| LMI = {cell['lmi']:.2f} (p = {cell['lmi_p']:.3f})")


# =============================================================================
# PLOT
# =============================================================================

folder = os.path.join(io.processed_dir, 'mice')

fig, axes = plt.subplots(len(cells), len(DAYS), figsize=(15, 9))

for i, (cell, label) in enumerate(cells):
    mouse_id = cell['mouse_id']
    roi = int(cell['roi'])

    xarr = utils_imaging.load_mouse_xarray(
        mouse_id, folder, 'tensor_xarray_mapping_data.nc', substracted=True
    )
    xarr = xarr.sel(cell=xarr['roi'].isin([roi])).sel(time=slice(*WIN_SEC))

    for j, day in enumerate(DAYS):
        ax = axes[i, j]
        day_data = xarr.sel(trial=xarr['day'] == day)

        if day_data.sizes['trial'] == 0:
            ax.set_visible(False)
            continue

        time = day_data.time.values
        # Individual trials.
        for t in range(day_data.sizes['trial']):
            ax.plot(time, day_data.isel(trial=t).squeeze().values * 100,
                    color='gray', alpha=0.2, linewidth=0.5)
        # Mean trace.
        mean_trace = day_data.mean(dim='trial').squeeze().values * 100
        ax.plot(time, mean_trace, color='k', linewidth=1.5)
        ax.axvline(0, color='#FF9600', linestyle='-', linewidth=1)
        ax.set_xlabel('Time (s)')

        if i == 0:
            ax.set_title(f'Day {day:+d}')

    row_label = (f'{label}\n{mouse_id} ROI {roi}\n'
                 f'LMI = {cell["lmi"]:.2f}  p = {cell["lmi_p"]:.3f}')
    axes[i, 0].set_ylabel(row_label, fontsize=8)

plt.tight_layout()
sns.despine()
plt.show()
