"""
Supplementary Figure 3m: Pairwise correlations between projection neurons
(wS2-wS2 and wM1-wM1 pairs) during a 2 s pre-stimulus quiet window,
compared pre vs post learning. Mapping trials only.

Stats at the cell-pair level (Mann-Whitney U, pre vs post).

ANALYSIS_MODE:
  'compute' — run the full computation and save intermediate CSVs
  'analyze' — load previously saved CSVs and plot only
"""

import os
import sys
from itertools import combinations
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import mannwhitneyu, pearsonr

sys.path.append(r'/home/aprenard/repos/NWB_analysis')
sys.path.append(r'/home/aprenard/repos/fast-learning')

import src.utils.utils_imaging as utils_imaging
import src.utils.utils_io as io


# ============================================================================
# Parameters
# ============================================================================

ANALYSIS_MODE = 'analyze'   # 'compute' or 'analyze'
WIN_SEC       = (-2, 0)     # quiet window before stimulus onset
PRE_DAYS      = [-2, -1]
POST_DAYS     = [1, 2]
N_CORES       = 35
PAIR_TYPES    = ['wS2-wS2', 'wM1-wM1']

# Intermediate results (heavy CSVs): kept in processed_dir
RESULTS_DIR = os.path.join(io.processed_dir, 'pairwise_correlations')

# Final figures + stats CSVs
OUTPUT_DIR = os.path.join(io.manuscript_output_dir, 'supp_3', 'output')


# ============================================================================
# Session / mouse setup
# ============================================================================

_, _, mice, db = io.select_sessions_from_db(io.db_path, io.nwb_dir,
                                             two_p_imaging='yes',
                                             experimenters=['AR', 'GF', 'MI'])

mice_by_group = {}
for mouse_id in mice:
    rg = io.get_mouse_reward_group_from_db(io.db_path, mouse_id)
    mice_by_group.setdefault(rg, []).append(mouse_id)


# ============================================================================
# Per-mouse computation
# ============================================================================

def process_mouse(mouse_id):
    reward_group = io.get_mouse_reward_group_from_db(io.db_path, mouse_id)
    folder = os.path.join(io.processed_dir, 'mice')
    xarr = utils_imaging.load_mouse_xarray(
        mouse_id, folder, 'tensor_xarray_mapping_data.nc', substracted=False)
    xarr.name = 'dff'
    xarr = xarr.sel(trial=xarr['day'].isin(PRE_DAYS + POST_DAYS))
    xarr = xarr.sel(time=slice(WIN_SEC[0], WIN_SEC[1]))

    mouse_results = []
    for period, days in [('pre', PRE_DAYS), ('post', POST_DAYS)]:
        xarr_period    = xarr.sel(trial=xarr['day'].isin(days))
        all_cells_data = xarr_period.values        # (n_cells, n_trials, n_time)
        cell_types     = xarr_period.coords['cell_type'].values
        rois           = xarr_period.coords['roi'].values
        n_cells, n_trials, _ = all_cells_data.shape

        if n_trials == 0:
            continue

        for i, j in combinations(range(n_cells), 2):
            if cell_types[i] != cell_types[j]:
                continue
            if cell_types[i] not in ['wS2', 'wM1']:
                continue

            trial_corrs = []
            for t in range(n_trials):
                ci, cj = all_cells_data[i, t, :], all_cells_data[j, t, :]
                valid  = ~(np.isnan(ci) | np.isnan(cj))
                if valid.sum() > 1 and np.std(ci[valid]) > 0 and np.std(cj[valid]) > 0:
                    trial_corrs.append(pearsonr(ci[valid], cj[valid])[0])

            if trial_corrs:
                mouse_results.append({
                    'mouse_id':     mouse_id,
                    'reward_group': reward_group,
                    'period':       period,
                    'pair_type':    f'{cell_types[i]}-{cell_types[i]}',
                    'roi_i':        rois[i],
                    'roi_j':        rois[j],
                    'correlation':  np.mean(trial_corrs),
                    'n_trials':     len(trial_corrs),
                })

    print(f"  {mouse_id}: {len(mouse_results)} pairs")
    return mouse_results


# ============================================================================
# Significance annotation helper
# ============================================================================

def add_significance_stars(ax, x1, x2, y, p_value):
    if p_value < 0.001:
        stars = '***'
    elif p_value < 0.01:
        stars = '**'
    elif p_value < 0.05:
        stars = '*'
    else:
        return
    h = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.02
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, c='black')
    pval_text = f'{stars}\np<0.001' if p_value < 0.001 else f'{stars}\np={p_value:.3f}'
    ax.text((x1 + x2) / 2, y + h, pval_text,
            ha='center', va='bottom', fontsize=10, fontweight='bold')


# ============================================================================
# Main loop per reward group
# ============================================================================

if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    sns.set_theme(context='paper', style='ticks', font='sans-serif', font_scale=1,
                  rc={'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none'})

    for reward_group in ['R+', 'R-']:
        if reward_group not in mice_by_group:
            print(f"No mice for {reward_group}, skipping.")
            continue

        group_mice = mice_by_group[reward_group]
        inter_dir  = os.path.join(RESULTS_DIR, reward_group, 'mapping')
        os.makedirs(inter_dir, exist_ok=True)
        corr_csv   = os.path.join(inter_dir, 'pairwise_correlations_prepost.csv')

        # ── Compute or load ──────────────────────────────────────────────────
        if ANALYSIS_MODE == 'compute' or not os.path.exists(corr_csv):
            print(f"\n[COMPUTE] {reward_group} — {len(group_mice)} mice, {N_CORES} cores")
            with Pool(processes=N_CORES) as pool:
                results_list = pool.map(process_mouse, group_mice)
            corr_df = pd.DataFrame(
                [item for sublist in results_list for item in sublist])
            corr_df.to_csv(corr_csv, index=False)
            print(f"Saved: {corr_csv}")
        else:
            print(f"\n[ANALYZE] Loading {corr_csv}")
            corr_df = pd.read_csv(corr_csv)

        print(f"Pairs: {corr_df['pair_type'].value_counts().to_dict()}")

        # ── Pair-level stats ─────────────────────────────────────────────────
        stats_pair = []
        for pt in PAIR_TYPES:
            sub  = corr_df[corr_df['pair_type'] == pt]
            pre  = sub[sub['period'] == 'pre']['correlation'].values
            post = sub[sub['period'] == 'post']['correlation'].values
            if len(pre) > 0 and len(post) > 0:
                stat, p = mannwhitneyu(pre, post, alternative='two-sided')
                stats_pair.append({
                    'pair_type':  pt, 'test': 'Mann-Whitney U',
                    'mean_pre':   np.mean(pre),  'sem_pre':  np.std(pre)  / np.sqrt(len(pre)),
                    'mean_post':  np.mean(post), 'sem_post': np.std(post) / np.sqrt(len(post)),
                    'n_pre':      len(pre), 'n_post': len(post),
                    'statistic':  stat, 'p_value': p,
                })
        stats_pair_df = pd.DataFrame(stats_pair)

        stats_pair_df.to_csv(
            os.path.join(OUTPUT_DIR, f'supp_3m_{reward_group}_stats.csv'), index=False)
        corr_df.to_csv(
            os.path.join(OUTPUT_DIR, f'supp_3m_{reward_group}_data.csv'), index=False)

        # ── Pair-level figure ─────────────────────────────────────────────────
        fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
        for idx, pt in enumerate(PAIR_TYPES):
            ax  = axes[idx]
            sub = corr_df[corr_df['pair_type'] == pt]
            sns.barplot(data=sub, x='period', y='correlation',
                        order=['pre', 'post'], ax=ax, errorbar='se', capsize=0.1)
            ax.axhline(0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
            ax.set_title(pt, fontsize=14, fontweight='bold')
            ax.set_xlabel('Period', fontsize=12)
            ax.set_ylabel('Pearson correlation' if idx == 0 else '', fontsize=12)
            ax.set_ylim(0, 0.02)

            row = stats_pair_df[stats_pair_df['pair_type'] == pt]
            if not row.empty:
                p_val = row.iloc[0]['p_value']
                if p_val < 0.05:
                    pre_top  = sub[sub['period'] == 'pre']['correlation'].agg(['mean', 'sem']).sum()
                    post_top = sub[sub['period'] == 'post']['correlation'].agg(['mean', 'sem']).sum()
                    add_significance_stars(ax, 0, 1, max(pre_top, post_top) +
                                           (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.06, p_val)
                else:
                    ax.text(0.95, 0.95, f'p={p_val:.3f}', transform=ax.transAxes,
                            fontsize=10, va='top', ha='right',
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.suptitle(f'Pre vs Post — Pair Level ({reward_group})', fontsize=14, y=1.02)
        plt.tight_layout()
        sns.despine()
        fig.savefig(os.path.join(OUTPUT_DIR, f'supp_3m_{reward_group}.svg'),
                    format='svg', dpi=300, bbox_inches='tight')
        print(f"Saved: supp_3m_{reward_group}.svg")
        plt.close()

    print("\nDone.")
