"""
Figure 4c, d, g, h: Pre vs post learning responses for
projection neurons.

  Panel c: wS2 PSTH (pre vs post, R+ top / R- bottom)
  Panel d: wS2 response amplitude bar plots (pre vs post)
  Panel g: wM1 PSTH
  Panel h: wM1 response amplitude bar plots

Data: mapping trials, Days -2/-1 (pre) vs +1/+2 (post), per-mouse mean.
Statistics: Wilcoxon signed-rank test (paired, pre vs post) per reward group
and cell type.
"""

import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import wilcoxon


from fast_learning import imaging
from fast_learning import paths, database
from fast_learning.plotting import save_figure, panel_size, reward_palette


# ============================================================================
# Parameters
# ============================================================================

WIN_SEC_AMP = (0, 0.300)
WIN_SEC_PSTH = (-0.5, 1.5)
if __name__ == '__main__':
    DAYS = [-2, -1, 0, 1, 2]
    DAYS_SELECTED = [-2, -1, 1, 2]
    MIN_CELLS = 3
    CELL_TYPES = ['wS2', 'wM1']

    # Colours: pre=grey, post per reward group
    COLORS = {
        'R+': {'pre': '#a3a3a3', 'post': reward_palette[1]},
        'R-': {'pre': '#a3a3a3', 'post': reward_palette[0]},
    }
    BAR_COLORS = {'R+': reward_palette[1], 'R-': reward_palette[0]}

    OUTPUT_DIR = os.path.join(paths.manuscript_output_dir, 'Figure4')

    # ============================================================================
    # Load imaging data
    # ============================================================================

    _, _, mice, db = database.select_sessions_from_db(
        paths.db_path, paths.nwb_dir, two_p_imaging='yes', experimenters=['AR', 'GF', 'MI']
    )

    avg_resp_list, psth_list = [], []

    for mouse_id in mice:
        reward_group = database.get_mouse_reward_group_from_db(paths.db_path, mouse_id)
        folder = paths.tensor_dir
        xarr = imaging.load_mouse_xarray(mouse_id, folder, 'tensor_xarray_mapping_data.nc')
        xarr = imaging.subtract_baseline(xarr, 2, imaging.baseline_frames(xarr))

        # Average response (amplitude window)
        avg = xarr.sel(trial=xarr['day'].isin(DAYS))
        avg = imaging.select_time(avg, WIN_SEC_AMP[0], WIN_SEC_AMP[1]).mean(dim='time')
        avg.name = 'average_response'
        avg = avg.to_dataframe().reset_index()
        avg['mouse_id'] = mouse_id
        avg['reward_group'] = reward_group
        avg_resp_list.append(avg)

        # PSTH
        p = xarr.sel(trial=xarr['day'].isin(DAYS))
        p = imaging.select_time(p, WIN_SEC_PSTH[0], WIN_SEC_PSTH[1])
        p = p.groupby('day').mean(dim='trial')
        p.name = 'psth'
        p = p.to_dataframe().reset_index()
        p['mouse_id'] = mouse_id
        p['reward_group'] = reward_group
        psth_list.append(p)

    avg_resp = pd.concat(avg_resp_list, ignore_index=True)
    psth = pd.concat(psth_list, ignore_index=True)

    # Convert to % dF/F0 and tag learning period
    avg_resp['average_response'] *= 100
    psth['psth'] *= 100
    psth['time'] = psth['time'].round(4)
    avg_resp['learning_period'] = avg_resp['day'].map(lambda x: 'pre' if x in [-2, -1] else 'post')
    psth['learning_period'] = psth['day'].map(lambda x: 'pre' if x in [-2, -1] else 'post')

    # ============================================================================
    # Aggregate per mouse, filter to projection cell types
    # ============================================================================

    avg_resp_filt = imaging.filter_data_by_cell_count(
        avg_resp[avg_resp['day'].isin(DAYS_SELECTED)], MIN_CELLS
    )
    psth_filt = imaging.filter_data_by_cell_count(psth[psth['day'].isin(DAYS_SELECTED)], MIN_CELLS)

    data_avg_proj = (
        avg_resp_filt[avg_resp_filt['cell_type'].isin(CELL_TYPES)]
        .groupby(['mouse_id', 'learning_period', 'reward_group', 'cell_type'])['average_response']
        .mean()
        .reset_index()
    )
    data_psth_proj = (
        psth_filt[psth_filt['cell_type'].isin(CELL_TYPES)]
        .groupby(['mouse_id', 'learning_period', 'reward_group', 'time', 'cell_type'])['psth']
        .mean()
        .reset_index()
    )

    # ============================================================================
    # Statistics: Wilcoxon signed-rank (pre vs post) per group × cell type
    # ============================================================================

    stats_rows = []
    for rg in ['R+', 'R-']:
        for ct in CELL_TYPES:
            sub = data_avg_proj[(data_avg_proj['reward_group'] == rg) & (data_avg_proj['cell_type'] == ct)]
            pre = sub[sub['learning_period'] == 'pre'].sort_values('mouse_id')['average_response']
            post = sub[sub['learning_period'] == 'post'].sort_values('mouse_id')['average_response']
            stat, p = wilcoxon(pre.values, post.values)
            stats_rows.append(
                {'reward_group': rg, 'cell_type': ct, 'test': 'Wilcoxon', 'statistic': stat, 'p_value': p}
            )
            print(f"{rg} {ct}: W={stat:.3f}, p={p:.4f}")

    stats_df = pd.DataFrame(stats_rows)

    # ============================================================================
    # Helper to plot one cell-type block (PSTH + bar, 2 reward groups)
    # ============================================================================

    def plot_cell_type(cell_type, fig_tag):
        fig, axes = plt.subplots(2, 2, figsize=panel_size(2, 2), sharex=False, sharey=False)

        for row, rg in enumerate(['R+', 'R-']):
            pal = [COLORS[rg]['pre'], COLORS[rg]['post']]

            # PSTH panel
            ax_psth = axes[row, 0]
            d = data_psth_proj[
                (data_psth_proj['reward_group'] == rg) & (data_psth_proj['cell_type'] == cell_type)
            ]
            sns.lineplot(
                data=d,
                x='time',
                y='psth',
                hue='learning_period',
                hue_order=['pre', 'post'],
                palette=sns.color_palette(pal),
                ax=ax_psth,
                legend=False,
                seed=0,
            )
            ax_psth.axvline(0, color='#FF9600', linestyle='-')
            ax_psth.set_ylabel('DF/F0 (%)')
            ax_psth.set_xlabel('Time (s)')
            ax_psth.set_title(f'{cell_type} PSTH — {rg}')

            # Bar panel
            ax_bar = axes[row, 1]
            d = data_avg_proj[
                (data_avg_proj['reward_group'] == rg) & (data_avg_proj['cell_type'] == cell_type)
            ]
            sns.barplot(
                data=d,
                x='learning_period',
                y='average_response',
                order=['pre', 'post'],
                color=BAR_COLORS[rg],
                ax=ax_bar,
                seed=0,
            )
            sns.swarmplot(
                data=d,
                x='learning_period',
                y='average_response',
                order=['pre', 'post'],
                color='black',
                alpha=0.5,
                size=2,
                ax=ax_bar,
            )
            ax_bar.set_ylim(-2, 15)
            ax_bar.set_ylabel('Average response (% dF/F0)')
            ax_bar.set_xlabel('')
            ax_bar.set_title(f'{cell_type} amplitude — {rg}')

        sns.despine()
        plt.tight_layout()
        return fig

    # ============================================================================
    # Produce figures
    # ============================================================================

    fig_wS2 = plot_cell_type('wS2', 'wS2')
    fig_wM1 = plot_cell_type('wM1', 'wM1')

    # ============================================================================
    # Save
    # ============================================================================

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    save_figure(fig_wS2, os.path.join(OUTPUT_DIR, 'figure_4c_d.svg'))
    print("Saved: figure_4c_d.svg")

    save_figure(fig_wM1, os.path.join(OUTPUT_DIR, 'figure_4g_h.svg'))
    print("Saved: figure_4g_h.svg")

    data_avg_proj.to_csv(os.path.join(OUTPUT_DIR, 'figure_4c_d_g_h_data.csv'), index=False)
    print("Saved: figure_4c_d_g_h_data.csv")

    stats_df.to_csv(os.path.join(OUTPUT_DIR, 'figure_4c_d_g_h_stats.csv'), index=False)
    print("Saved: figure_4c_d_g_h_stats.csv")

    plt.close()
