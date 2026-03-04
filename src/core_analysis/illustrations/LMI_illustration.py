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

# Example cells to display — identified from the browser PDFs.
# Each entry: (mouse_id, roi, label)
EXAMPLE_CELLS = [
    ('GF306', 94, 'Negative LMI'),
    ('GF334', 77, 'Best positive LMI'),
    ('GF313', 137, 'Average positive LMI'),
]

YLIMS = {
    'GF306': (-50, 250),
    'GF334': (-100, 600),
    'GF313': (-50, 400),
}

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

def select_cell(mouse_id, roi):
    """Return the lmi_df row matching (mouse_id, roi)."""
    mask = (lmi_df['mouse_id'] == mouse_id) & (lmi_df['roi'] == roi)
    return lmi_df.loc[mask].iloc[0]


cells = [
    (select_cell(mouse_id, roi), label)
    for mouse_id, roi, label in EXAMPLE_CELLS
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

    y_min, y_max = YLIMS[mouse_id]

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
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel('Time (s)')

        if i == 0:
            ax.set_title(f'Day {day:+d}')

    row_label = (f'{label}\n{mouse_id} ROI {roi}\n'
                 f'LMI = {cell["lmi"]:.2f}  p = {cell["lmi_p"]:.3f}')
    axes[i, 0].set_ylabel(row_label, fontsize=8)

plt.tight_layout()
sns.despine()
plt.savefig(os.path.join(io.results_dir, 'illustrations', 'lmi_example_cells.svg'))


# # =============================================================================
# # BROWSER PDFs: top-50 positive and top-50 negative cells, 10 per page
# # =============================================================================

# from matplotlib.backends.backend_pdf import PdfPages

# N_CELLS = 50
# CELLS_PER_PAGE = 10

# top_pos = pos_lmi.nlargest(N_CELLS, 'lmi').reset_index(drop=True)
# top_neg = neg_lmi.nsmallest(N_CELLS, 'lmi').reset_index(drop=True)

# output_dir = os.path.join(io.results_dir, 'illustrations', 'lmi_browser')
# os.makedirs(output_dir, exist_ok=True)

# def plot_cell_row(axes_row, cell, label):
#     mouse_id = cell['mouse_id']
#     roi = int(cell['roi'])

#     xarr = utils_imaging.load_mouse_xarray(
#         mouse_id, folder, 'tensor_xarray_mapping_data.nc', substracted=True
#     )
#     xarr = xarr.sel(cell=xarr['roi'].isin([roi])).sel(time=slice(*WIN_SEC))

#     y_min, y_max = np.inf, -np.inf
#     for day in DAYS:
#         day_data = xarr.sel(trial=xarr['day'] == day)
#         if day_data.sizes['trial'] > 0:
#             y_vals = day_data.values * 100
#             y_min = min(y_min, np.percentile(y_vals, 1))
#             y_max = max(y_max, np.percentile(y_vals, 99))

#     for j, day in enumerate(DAYS):
#         ax = axes_row[j]
#         day_data = xarr.sel(trial=xarr['day'] == day)

#         if day_data.sizes['trial'] == 0:
#             ax.set_visible(False)
#             continue

#         time = day_data.time.values
#         for t in range(day_data.sizes['trial']):
#             ax.plot(time, day_data.isel(trial=t).squeeze().values * 100,
#                     color='gray', alpha=0.2, linewidth=0.5)
#         mean_trace = day_data.mean(dim='trial').squeeze().values * 100
#         ax.plot(time, mean_trace, color='k', linewidth=1.5)
#         ax.axvline(0, color='#FF9600', linestyle='-', linewidth=1)
#         ax.set_ylim(y_min, y_max)
#         ax.set_xlabel('Time (s)')

#         if j == 0:
#             row_label = (f'{label}\n{mouse_id} ROI {roi}\n'
#                          f'LMI = {cell["lmi"]:.2f}  p = {cell["lmi_p"]:.3f}')
#             ax.set_ylabel(row_label, fontsize=7)
#         else:
#             ax.set_yticklabels([])


# def write_browser_pdf(cell_pool, pdf_path, group_label):
#     n_pages = int(np.ceil(len(cell_pool) / CELLS_PER_PAGE))
#     with PdfPages(pdf_path) as pdf:
#         for page in range(n_pages):
#             page_cells = cell_pool.iloc[page * CELLS_PER_PAGE:(page + 1) * CELLS_PER_PAGE]
#             n_rows = len(page_cells)
#             fig, axes = plt.subplots(n_rows, len(DAYS), figsize=(15, 2 * n_rows))
#             if n_rows == 1:
#                 axes = axes[np.newaxis, :]
#             fig.suptitle(f'{group_label} — page {page + 1}/{n_pages}', fontsize=10)

#             for i, (_, cell) in enumerate(page_cells.iterrows()):
#                 rank = page * CELLS_PER_PAGE + i + 1
#                 label = f'#{rank}'
#                 plot_cell_row(axes[i], cell, label)

#                 if i == 0:
#                     for j, day in enumerate(DAYS):
#                         axes[i, j].set_title(f'Day {day:+d}')

#             plt.tight_layout()
#             sns.despine()
#             pdf.savefig(fig)
#             plt.close(fig)
#             print(f'  {group_label}: saved page {page + 1}/{n_pages}')

#     print(f'Saved: {pdf_path}')


# print("\nGenerating browser PDFs...")
# write_browser_pdf(top_pos, os.path.join(output_dir, 'top50_positive_lmi.pdf'), 'Top-50 Positive LMI')
# write_browser_pdf(top_neg, os.path.join(output_dir, 'top50_negative_lmi.pdf'), 'Top-50 Negative LMI')
