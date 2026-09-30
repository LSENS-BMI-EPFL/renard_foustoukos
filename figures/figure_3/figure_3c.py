"""
Figure 3c: Grand-average mapping-trial PSTHs for all cells across learning
days, R+ vs R-.

1 row × 5 columns (Days -2 to +2). Each panel shows the mean ± CI across
mice (minimum 3 cells per mouse to include).
"""

import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from fast_learning import imaging
from fast_learning import paths, database
from fast_learning.plotting import reward_palette, save_figure


# ============================================================================
# Parameters
# ============================================================================

SAMPLING_RATE = 30
WIN_SEC = (-0.5, 1.5)
BASELINE_WIN = (0, 1)
if __name__ == '__main__':
    BASELINE_WIN = (int(BASELINE_WIN[0] * SAMPLING_RATE), int(BASELINE_WIN[1] * SAMPLING_RATE))
    DAYS = [-2, -1, 0, 1, 2]
    MIN_CELLS = 3

    OUTPUT_DIR = os.path.join(paths.manuscript_output_dir, 'figure_3', 'output')

    # ============================================================================
    # Load and process imaging data
    # ============================================================================

    _, _, mice, _ = database.select_sessions_from_db(
        paths.db_path, paths.nwb_dir, two_p_imaging='yes', experimenters=['AR', 'GF', 'MI']
    )

    psth_records = []
    for mouse_id in mice:
        reward_group = database.get_mouse_reward_group_from_db(paths.db_path, mouse_id)
        folder = paths.tensor_dir
        xarr = imaging.load_mouse_xarray(mouse_id, folder, 'tensor_xarray_mapping_data.nc')
        xarr = imaging.subtract_baseline(xarr, 2, BASELINE_WIN)
        xarr = xarr.sel(trial=xarr['day'].isin(DAYS))
        xarr = xarr.sel(time=slice(WIN_SEC[0], WIN_SEC[1]))
        xarr = xarr.groupby('day').mean(dim='trial')

        xarr.name = 'psth'
        df = xarr.to_dataframe().reset_index()
        df['mouse_id'] = mouse_id
        df['reward_group'] = reward_group
        psth_records.append(df)

    psth = pd.concat(psth_records, ignore_index=True)

    # ============================================================================
    # Aggregate: per-mouse mean across all cells, convert to % dF/F0
    # ============================================================================

    psth_filtered = imaging.filter_data_by_cell_count(psth, MIN_CELLS)

    data = psth_filtered.groupby(['mouse_id', 'day', 'reward_group', 'time'])['psth'].mean().reset_index()
    data['psth'] = data['psth'] * 100
    data['time'] = data['time'].round(4)

    # ============================================================================
    # Figure
    # ============================================================================

    sns.set_theme(
        context='paper',
        style='ticks',
        font='sans-serif',
        font_scale=1,
        rc={'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none'},
    )

    fig, axes = plt.subplots(1, len(DAYS), figsize=(18, 4), sharey=True)

    for j, day in enumerate(DAYS):
        d = data[data['day'] == day]
        sns.lineplot(
            data=d,
            x='time',
            y='psth',
            errorbar='ci',
            hue='reward_group',
            hue_order=['R-', 'R+'],
            palette=reward_palette,
            estimator='mean',
            ax=axes[j],
            legend=False,
        )
        axes[j].axvline(0, color='#FF9600', linestyle='-')
        axes[j].set_title(f'Day {day:+d}')
        axes[j].set_ylabel('DF/F0 (%)' if j == 0 else '')
        axes[j].set_xlabel('Time (s)')

    plt.ylim(-1, 12)
    plt.tight_layout()
    sns.despine()

    # ============================================================================
    # Save
    # ============================================================================

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    save_figure(fig, os.path.join(OUTPUT_DIR, 'figure_3c.svg'))
    print("Saved: figure_3c.svg")

    plt.close()
