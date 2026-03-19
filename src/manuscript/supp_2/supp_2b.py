"""
Supplementary Figure 2b: Example mapping-trial PSTHs for three illustrative
cells across learning days.

3 rows × 5 days:
  Row 0: negative LMI cell
  Row 1: best positive LMI cell
  Row 2: average positive LMI cell

Individual trials are shown in grey; the mean trace in black; stimulus onset
as a vertical orange line.
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


# ============================================================================
# Parameters
# ============================================================================

# Example cells: (mouse_id, roi, label) — adjust to browse the LMI population.
EXAMPLE_CELLS = [
    ('GF306', 94,  'Negative LMI'),
    ('GF334', 77,  'Best positive LMI'),
    ('GF313', 137, 'Average positive LMI'),
]

# Y-axis limits per mouse (% dF/F).
YLIMS = {
    'GF306': (-50,  250),
    'GF334': (-100, 600),
    'GF313': (-50,  400),
}

WIN_SEC = (-0.5, 1.5)
DAYS    = [-2, -1, 0, 1, 2]

OUTPUT_DIR = os.path.join(io.manuscript_output_dir, 'supp_2', 'output')


# ============================================================================
# Load LMI results
# ============================================================================

lmi_df = pd.read_csv(os.path.join(io.processed_dir, 'lmi_results.csv'))

def select_cell(mouse_id, roi):
    mask = (lmi_df['mouse_id'] == mouse_id) & (lmi_df['roi'] == roi)
    return lmi_df.loc[mask].iloc[0]

cells = [(select_cell(m, r), label) for m, r, label in EXAMPLE_CELLS]

print("Selected cells:")
for cell, label in cells:
    print(f"  {label}: {cell['mouse_id']} ROI {int(cell['roi'])} "
          f"| LMI = {cell['lmi']:.2f} (p = {cell['lmi_p']:.3f})")


# ============================================================================
# Figure
# ============================================================================

folder = os.path.join(io.processed_dir, 'mice')

fig, axes = plt.subplots(len(cells), len(DAYS), figsize=(15, 9))

for i, (cell, label) in enumerate(cells):
    mouse_id = cell['mouse_id']
    roi      = int(cell['roi'])

    xarr = utils_imaging.load_mouse_xarray(
        mouse_id, folder, 'tensor_xarray_mapping_data.nc', substracted=True
    )
    xarr = xarr.sel(cell=xarr['roi'].isin([roi])).sel(time=slice(*WIN_SEC))

    y_min, y_max = YLIMS[mouse_id]

    for j, day in enumerate(DAYS):
        ax = axes[i, j]
        day_data = xarr.sel(trial=xarr['day'] == day)

        if day_data.sizes['trial'] == 0:
            ax.set_visible(False)
            continue

        time = day_data.time.values
        for t in range(day_data.sizes['trial']):
            ax.plot(time, day_data.isel(trial=t).squeeze().values * 100,
                    color='gray', alpha=0.2, linewidth=0.5)

        mean_trace = day_data.mean(dim='trial').squeeze().values * 100
        ax.plot(time, mean_trace, color='k', linewidth=1.5)
        ax.axvline(0, color='#FF9600', linestyle='-', linewidth=1)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel('Time (s)')

        if i == 0:
            ax.set_title(f'Day {day:+d}')

    row_label = (f'{label}\n{mouse_id} ROI {roi}\n'
                 f'LMI = {cell["lmi"]:.2f}  p = {cell["lmi_p"]:.3f}')
    axes[i, 0].set_ylabel(row_label, fontsize=8)

plt.tight_layout()
sns.despine()


# ============================================================================
# Save
# ============================================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

fig.savefig(os.path.join(OUTPUT_DIR, 'supp_2b.svg'), format='svg', dpi=300, bbox_inches='tight')
print(f"\nSaved: supp_2b.svg")

# Data: LMI values for the selected example cells
pd.DataFrame([
    {'mouse_id': cell['mouse_id'], 'roi': int(cell['roi']),
     'label': label, 'lmi': cell['lmi'], 'lmi_p': cell['lmi_p']}
    for cell, label in cells
]).to_csv(os.path.join(OUTPUT_DIR, 'supp_2b_data.csv'), index=False)
print("Saved: supp_2b_data.csv")

plt.show()
