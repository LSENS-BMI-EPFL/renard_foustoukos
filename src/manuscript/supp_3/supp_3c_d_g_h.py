"""
Supplementary Figure 3c, d, g, h: Pre vs post learning responses for
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
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import wilcoxon

sys.path.append(r'/home/aprenard/repos/NWB_analysis')
sys.path.append(r'/home/aprenard/repos/fast-learning')

import src.utils.utils_imaging as utils_imaging
import src.utils.utils_io as io


# ============================================================================
# Parameters
# ============================================================================

SAMPLING_RATE  = 30
WIN_SEC_AMP    = (0, 0.300)
WIN_SEC_PSTH   = (-0.5, 1.5)
BASELINE_WIN   = (0, 1)
BASELINE_WIN   = (int(BASELINE_WIN[0] * SAMPLING_RATE),
                  int(BASELINE_WIN[1] * SAMPLING_RATE))
DAYS           = [-2, -1, 0, 1, 2]
DAYS_SELECTED  = [-2, -1, 1, 2]
MIN_CELLS      = 3
CELL_TYPES     = ['wS2', 'wM1']

# Colours: pre=grey, post per reward group
COLORS = {
    'R+': {'pre': '#a3a3a3', 'post': '#1b9e77'},
    'R-': {'pre': '#a3a3a3', 'post': '#c959affe'},
}
BAR_COLORS = {'R+': '#1b9e77', 'R-': '#c959affe'}

OUTPUT_DIR = os.path.join(io.manuscript_output_dir, 'supp_3', 'output')


# ============================================================================
# Load imaging data
# ============================================================================

_, _, mice, db = io.select_sessions_from_db(io.db_path, io.nwb_dir,
                                             two_p_imaging='yes',
                                             experimenters=['AR', 'GF', 'MI'])

avg_resp_list, psth_list = [], []

for mouse_id in mice:
    reward_group = io.get_mouse_reward_group_from_db(io.db_path, mouse_id)
    folder = os.path.join(io.processed_dir, 'mice')
    xarr = utils_imaging.load_mouse_xarray(mouse_id, folder,
                                            'tensor_xarray_mapping_data.nc')
    xarr = utils_imaging.substract_baseline(xarr, 2, BASELINE_WIN)

    # Average response (amplitude window)
    avg = xarr.sel(trial=xarr['day'].isin(DAYS))
    avg = avg.sel(time=slice(WIN_SEC_AMP[0], WIN_SEC_AMP[1])).mean(dim='time')
    avg.name = 'average_response'
    avg = avg.to_dataframe().reset_index()
    avg['mouse_id']     = mouse_id
    avg['reward_group'] = reward_group
    avg_resp_list.append(avg)

    # PSTH
    p = xarr.sel(trial=xarr['day'].isin(DAYS))
    p = p.sel(time=slice(WIN_SEC_PSTH[0], WIN_SEC_PSTH[1]))
    p = p.groupby('day').mean(dim='trial')
    p.name = 'psth'
    p = p.to_dataframe().reset_index()
    p['mouse_id']     = mouse_id
    p['reward_group'] = reward_group
    psth_list.append(p)

avg_resp = pd.concat(avg_resp_list, ignore_index=True)
psth     = pd.concat(psth_list,     ignore_index=True)

# Convert to % dF/F0 and tag learning period
avg_resp['average_response'] *= 100
psth['psth']                 *= 100
avg_resp['learning_period'] = avg_resp['day'].map(lambda x: 'pre' if x in [-2, -1] else 'post')
psth['learning_period']     = psth['day'].map(lambda x: 'pre' if x in [-2, -1] else 'post')


# ============================================================================
# Aggregate per mouse, filter to projection cell types
# ============================================================================

avg_resp_filt = utils_imaging.filter_data_by_cell_count(
    avg_resp[avg_resp['day'].isin(DAYS_SELECTED)], MIN_CELLS)
psth_filt = utils_imaging.filter_data_by_cell_count(
    psth[psth['day'].isin(DAYS_SELECTED)], MIN_CELLS)

data_avg_proj  = (avg_resp_filt[avg_resp_filt['cell_type'].isin(CELL_TYPES)]
                  .groupby(['mouse_id', 'learning_period', 'reward_group', 'cell_type'])
                  ['average_response'].mean().reset_index())
data_psth_proj = (psth_filt[psth_filt['cell_type'].isin(CELL_TYPES)]
                  .groupby(['mouse_id', 'learning_period', 'reward_group', 'time', 'cell_type'])
                  ['psth'].mean().reset_index())


# ============================================================================
# Statistics: Wilcoxon signed-rank (pre vs post) per group × cell type
# ============================================================================

stats_rows = []
for rg in ['R+', 'R-']:
    for ct in CELL_TYPES:
        sub = data_avg_proj[(data_avg_proj['reward_group'] == rg) &
                             (data_avg_proj['cell_type']   == ct)]
        pre  = sub[sub['learning_period'] == 'pre'] .sort_values('mouse_id')['average_response']
        post = sub[sub['learning_period'] == 'post'].sort_values('mouse_id')['average_response']
        stat, p = wilcoxon(pre.values, post.values)
        stats_rows.append({'reward_group': rg, 'cell_type': ct,
                           'test': 'Wilcoxon', 'statistic': stat, 'p_value': p})
        print(f"{rg} {ct}: W={stat:.3f}, p={p:.4f}")

stats_df = pd.DataFrame(stats_rows)


# ============================================================================
# Helper to plot one cell-type block (PSTH + bar, 2 reward groups)
# ============================================================================

def plot_cell_type(cell_type, fig_tag):
    fig, axes = plt.subplots(2, 2, figsize=(6, 5), sharex=False, sharey=False)

    for row, rg in enumerate(['R+', 'R-']):
        pal = [COLORS[rg]['pre'], COLORS[rg]['post']]

        # PSTH panel
        ax_psth = axes[row, 0]
        d = data_psth_proj[(data_psth_proj['reward_group'] == rg) &
                            (data_psth_proj['cell_type']   == cell_type)]
        sns.lineplot(data=d, x='time', y='psth', hue='learning_period',
                     hue_order=['pre', 'post'], palette=sns.color_palette(pal),
                     ax=ax_psth, legend=False)
        ax_psth.axvline(0, color='#FF9600', linestyle='-')
        ax_psth.set_ylabel('DF/F0 (%)')
        ax_psth.set_xlabel('Time (s)')
        ax_psth.set_title(f'{cell_type} PSTH — {rg}')

        # Bar panel
        ax_bar = axes[row, 1]
        d = data_avg_proj[(data_avg_proj['reward_group'] == rg) &
                           (data_avg_proj['cell_type']   == cell_type)]
        sns.barplot(data=d, x='learning_period', y='average_response',
                    order=['pre', 'post'], color=BAR_COLORS[rg], ax=ax_bar)
        sns.swarmplot(data=d, x='learning_period', y='average_response',
                      order=['pre', 'post'], color='black', alpha=0.5,
                      size=4, ax=ax_bar)
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

fig_wS2.savefig(os.path.join(OUTPUT_DIR, 'supp_3c_d.svg'), format='svg',
                dpi=300, bbox_inches='tight')
print("Saved: supp_3c_d.svg")

fig_wM1.savefig(os.path.join(OUTPUT_DIR, 'supp_3g_h.svg'), format='svg',
                dpi=300, bbox_inches='tight')
print("Saved: supp_3g_h.svg")

data_avg_proj.to_csv(os.path.join(OUTPUT_DIR, 'supp_3c_d_g_h_data.csv'), index=False)
print("Saved: supp_3c_d_g_h_data.csv")

stats_df.to_csv(os.path.join(OUTPUT_DIR, 'supp_3c_d_g_h_stats.csv'), index=False)
print("Saved: supp_3c_d_g_h_stats.csv")

plt.show()
