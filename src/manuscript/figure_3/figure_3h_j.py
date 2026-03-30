"""
Figure 3h-j: Trial-by-trial correlation matrices and network reorganization metrics

This script generates Panels h-j for Figure 3:
- Panel h: Average trial-by-trial correlation matrices for R+ and R- groups
- Panel i: Within-day correlation trajectory across days
- Panel j: Network reorganization index

For each panel, two CSV files are saved alongside this script:
- figure_3X_data.csv: data points displayed in the panel
- figure_3X_stats.csv: statistical test results
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, mannwhitneyu, wilcoxon, kruskal
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

sys.path.append('/home/aprenard/repos/fast-learning')
import src.utils.utils_imaging as utils_imaging
import src.utils.utils_io as io
from src.utils.utils_plot import reward_palette


# ============================================================================
# Parameters
# ============================================================================

SAMPLING_RATE = 30
WIN = (0, 0.300)       # stimulus onset to 300 ms after
BASELINE_WIN = (-1, 0)
DAYS = [-2, -1, 0, 1, 2]
N_MAP_TRIALS = 40

OUTPUT_DIR = os.path.join(io.manuscript_output_dir, 'figure_3', 'output')


# ============================================================================
# Data loading and processing
# ============================================================================

def load_and_process_data(
    similarity_metric='cosine',
    select_lmi=False,
    zscore=False,
    projection_type=None,
    n_min_proj=5,
    substract_baseline=True,
):
    """
    Load imaging data and compute trial-by-trial similarity matrices.

    Args:
        similarity_metric: 'pearson', 'spearman', or 'cosine'
        select_lmi: If True, restrict to LMI-significant cells
        zscore: If True, z-score responses within each day to remove recording drift
        projection_type: Cell type filter ('wS2', 'wM1', or None for all cells)
        n_min_proj: Minimum number of projection cells required to include a mouse
        substract_baseline: If True, subtract baseline from traces

    Returns:
        corr_matrices_rew: List of similarity matrices (one per R+ mouse)
        corr_matrices_nonrew: List of similarity matrices (one per R- mouse)
        mice_rew: List of R+ mouse IDs
        mice_nonrew: List of R- mouse IDs
    """
    if similarity_metric not in ('pearson', 'spearman', 'cosine'):
        raise ValueError(
            f"similarity_metric must be 'pearson', 'spearman', or 'cosine', got '{similarity_metric}'"
        )

    _, _, mice, db = io.select_sessions_from_db(io.db_path, io.nwb_dir, two_p_imaging='yes')
    print(mice)

    selected_cells = None
    if select_lmi:
        processed_folder = io.solve_common_paths('processed_data')
        lmi_df = pd.read_csv(os.path.join(processed_folder, 'lmi_results.csv'))
        selected_cells = lmi_df.loc[(lmi_df['lmi_p'] <= 0.025) | (lmi_df['lmi_p'] >= 0.975)]

    vectors_rew, vectors_nonrew = [], []
    mice_rew, mice_nonrew = [], []

    for mouse in mice:
        print(f"Processing mouse: {mouse}")
        folder = os.path.join(io.solve_common_paths('processed_data'), 'mice')
        xarray = utils_imaging.load_mouse_xarray(
            mouse, folder, 'tensor_xarray_mapping_data.nc', substracted=substract_baseline
        )
        rew_gp = io.get_mouse_reward_group_from_db(io.db_path, mouse, db)

        xarray = xarray.sel(trial=xarray['day'].isin(DAYS))

        if select_lmi and selected_cells is not None:
            selected_for_mouse = selected_cells.loc[selected_cells['mouse_id'] == mouse]['roi']
            xarray = xarray.sel(cell=xarray['roi'].isin(selected_for_mouse))

        if projection_type is not None:
            xarray = xarray.sel(cell=xarray['cell_type'] == projection_type)
            if xarray.sizes['cell'] < n_min_proj:
                print(f"Not enough cells of type {projection_type} for mouse {mouse}.")
                continue

        n_trials = xarray[0, :, 0].groupby('day').count(dim='trial').values
        if np.any(n_trials < N_MAP_TRIALS):
            print(f'Not enough mapping trials for {mouse}.')
            continue

        # Select last N_MAP_TRIALS mapping trials per day and average over time window
        d = xarray.groupby('day').apply(lambda x: x.isel(trial=slice(-N_MAP_TRIALS, None)))
        d = d.sel(time=slice(WIN[0], WIN[1])).mean(dim='time')

        # Optionally z-score within each day to remove recording drift
        if zscore:
            d_normalized = d.copy()
            for day in DAYS:
                day_mask = d['day'] == day
                day_data = d.sel(trial=day_mask)
                day_mean = day_data.mean(dim='trial')
                day_std = day_data.std(dim='trial')
                day_std = day_std.where(day_std > 0, 1)
                d_normalized.loc[dict(trial=day_mask)] = ((day_data - day_mean) / day_std).values
            d = d_normalized

        if rew_gp == 'R-':
            vectors_nonrew.append(d)
            mice_nonrew.append(mouse)
        elif rew_gp == 'R+':
            vectors_rew.append(d)
            mice_rew.append(mouse)

    print(f"Loaded {len(vectors_rew)} R+ mice and {len(vectors_nonrew)} R- mice")

    corr_matrices_rew = [_compute_similarity_matrix(v, similarity_metric) for v in vectors_rew]
    corr_matrices_nonrew = [_compute_similarity_matrix(v, similarity_metric) for v in vectors_nonrew]

    return corr_matrices_rew, corr_matrices_nonrew, mice_rew, mice_nonrew


def _compute_similarity_matrix(vector, similarity_metric):
    """Compute a trial-by-trial similarity matrix for one mouse."""
    if similarity_metric == 'pearson':
        cm = np.corrcoef(vector.values.T)
    elif similarity_metric == 'spearman':
        cm, _ = spearmanr(vector.values.T, axis=1)
    elif similarity_metric == 'cosine':
        data = vector.values.T  # (trials, cells)
        data = np.nan_to_num(data, nan=0.0)
        norms = np.linalg.norm(data, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        normalized = data / norms
        cm = normalized @ normalized.T
    np.fill_diagonal(cm, np.nan)
    return cm


def _significance_stars(p):
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    return 'n.s.'


# ============================================================================
# Metric computation
# ============================================================================

def _compute_within_day_metrics(corr_matrices, mice_ids, reward_group):
    """Compute average within-day correlation per mouse for each day."""
    results = []
    for cm in corr_matrices:
        row = {}
        for i, day in enumerate(DAYS):
            day_idx = np.arange(i * N_MAP_TRIALS, (i + 1) * N_MAP_TRIALS)
            row[f'within_day{day:+d}'] = np.nanmean(cm[np.ix_(day_idx, day_idx)])
        results.append(row)
    df = pd.DataFrame(results)
    df['reward_group'] = reward_group
    df['mouse_id'] = mice_ids
    return df


def _compute_day0_metrics(corr_matrices, mice_ids, reward_group):
    """Compute average correlation between day 0 and each other day, per mouse."""
    day0_idx = np.arange(2 * N_MAP_TRIALS, 3 * N_MAP_TRIALS)
    results = []
    for cm in corr_matrices:
        row = {}
        for i, day in enumerate(DAYS):
            day_idx = np.arange(i * N_MAP_TRIALS, (i + 1) * N_MAP_TRIALS)
            row[f'corr_day0_vs_day{day:+d}'] = np.nanmean(cm[np.ix_(day0_idx, day_idx)])
        results.append(row)
    df = pd.DataFrame(results)
    df['reward_group'] = reward_group
    df['mouse_id'] = mice_ids
    return df


def _compute_reorganization_metrics(corr_matrices, mice_ids, reward_group):
    """Compute network reorganization index per mouse."""
    pre_idx = np.arange(0, 2 * N_MAP_TRIALS)
    post_idx = np.arange(3 * N_MAP_TRIALS, 5 * N_MAP_TRIALS)
    results = []
    for cm in corr_matrices:
        within_pre = np.nanmean(cm[np.ix_(pre_idx, pre_idx)])
        within_post = np.nanmean(cm[np.ix_(post_idx, post_idx)])
        between = np.nanmean(cm[np.ix_(pre_idx, post_idx)])
        results.append({
            'within_pre': within_pre,
            'within_post': within_post,
            'between_pre_post': between,
            'reorganization_index': (within_pre + within_post) / 2 - between,
        })
    df = pd.DataFrame(results)
    df['reward_group'] = reward_group
    df['mouse_id'] = mice_ids
    return df


# ============================================================================
# Panel h: Average trial-by-trial correlation matrices
# ============================================================================

def panel_h_correlation_matrices(
    corr_matrices_rew=None,
    corr_matrices_nonrew=None,
    mice_rew=None,
    mice_nonrew=None,
    similarity_metric='cosine',
    select_lmi=False,
    zscore=False,
    projection_type=None,
    output_dir=OUTPUT_DIR,
    save_format='svg',
    dpi=300,
):
    """
    Generate Figure 3 Panel h: Average trial-by-trial correlation matrices.

    Shows the 200x200 (5 days x 40 trials) similarity matrices averaged across
    mice, separately for R+ and R- reward groups.

    Saves:
        figure_3h_data.csv: day-pair block-averaged correlations for both groups
    """
    if corr_matrices_rew is None or corr_matrices_nonrew is None:
        corr_matrices_rew, corr_matrices_nonrew, mice_rew, mice_nonrew = load_and_process_data(
            similarity_metric=similarity_metric,
            select_lmi=select_lmi,
            zscore=zscore,
            projection_type=projection_type,
        )

    sns.set_theme(context='paper', style='ticks', palette='deep', font='sans-serif', font_scale=1)

    avg_corr_rew = np.nanmean(corr_matrices_rew, axis=0)
    avg_corr_nonrew = np.nanmean(corr_matrices_nonrew, axis=0)

    vmax = np.nanpercentile(avg_corr_rew, 99)
    vmin = 0

    fig = plt.figure(figsize=(16, 6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.05], wspace=0.25)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax_cbar = fig.add_subplot(gs[0, 2])

    edges = np.cumsum([N_MAP_TRIALS] * len(DAYS))

    # R+ group
    ax0.imshow(avg_corr_rew, cmap='viridis', vmax=vmax, vmin=vmin, aspect='auto')
    for edge in edges[:-1] - 0.5:
        ax0.axvline(x=edge, color='white', linestyle='-', linewidth=1.5)
        ax0.axhline(y=edge, color='white', linestyle='-', linewidth=1.5)
    ax0.set_xticks(edges - N_MAP_TRIALS / 2)
    ax0.set_xticklabels(DAYS)
    ax0.set_yticks(edges - N_MAP_TRIALS / 2)
    ax0.set_yticklabels(DAYS)
    ax0.set_xlabel('Day')
    ax0.set_ylabel('Day')
    ax0.set_title(f'R+ Group (N={len(mice_rew)} mice)')

    # R- group
    im1 = ax1.imshow(avg_corr_nonrew, cmap='viridis', vmax=vmax, vmin=vmin, aspect='auto')
    for edge in edges[:-1] - 0.5:
        ax1.axvline(x=edge, color='white', linestyle='-', linewidth=1.5)
        ax1.axhline(y=edge, color='white', linestyle='-', linewidth=1.5)
    ax1.set_xticks(edges - N_MAP_TRIALS / 2)
    ax1.set_xticklabels(DAYS)
    ax1.set_yticks(edges - N_MAP_TRIALS / 2)
    ax1.set_yticklabels(DAYS)
    ax1.set_xlabel('Day')
    ax1.set_title(f'R- Group (N={len(mice_nonrew)} mice)')

    metric_label = {
        'pearson': 'Pearson Correlation',
        'spearman': 'Spearman Correlation',
        'cosine': 'Cosine Similarity',
    }[similarity_metric]
    cbar = fig.colorbar(im1, cax=ax_cbar, label=metric_label)
    cbar.set_ticks([vmin, vmax])
    plt.tight_layout()

    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f'figure_3h.{save_format}'), format=save_format, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"Figure 3h saved to: {os.path.join(output_dir, 'figure_3h.' + save_format)}")

    # Save data CSV: block-averaged (day x day) correlations for each group
    records = []
    for i, day_row in enumerate(DAYS):
        for j, day_col in enumerate(DAYS):
            row_idx = np.arange(i * N_MAP_TRIALS, (i + 1) * N_MAP_TRIALS)
            col_idx = np.arange(j * N_MAP_TRIALS, (j + 1) * N_MAP_TRIALS)
            records.append({
                'reward_group': 'R+',
                'day_row': day_row,
                'day_col': day_col,
                'correlation': np.nanmean(avg_corr_rew[np.ix_(row_idx, col_idx)]),
            })
            records.append({
                'reward_group': 'R-',
                'day_row': day_row,
                'day_col': day_col,
                'correlation': np.nanmean(avg_corr_nonrew[np.ix_(row_idx, col_idx)]),
            })
    data_df = pd.DataFrame(records)
    data_df.to_csv(os.path.join(OUTPUT_DIR, 'figure_3h_data.csv'), index=False)
    print(f"Figure 3h data saved to: {os.path.join(OUTPUT_DIR, 'figure_3h_data.csv')}")


# ============================================================================
# Panel i: Within-day correlation trajectory
# ============================================================================

def panel_i_within_day_correlations(
    corr_matrices_rew=None,
    corr_matrices_nonrew=None,
    mice_rew=None,
    mice_nonrew=None,
    similarity_metric='cosine',
    select_lmi=False,
    zscore=False,
    projection_type=None,
    output_dir=OUTPUT_DIR,
    save_format='svg',
    dpi=300,
):
    """
    Generate Figure 3 Panel i: Within-day network similarity trajectory.

    Shows the average within-day correlation across learning days for R+ and R-
    groups, with individual mouse lines and between-group statistics.

    Saves:
        figure_3i_data.csv: per-mouse within-day correlations for each day
        figure_3i_stats.csv: 2-way ANOVA (day x reward_group) + Mann-Whitney U post-hoc per day
    """
    if corr_matrices_rew is None or corr_matrices_nonrew is None:
        corr_matrices_rew, corr_matrices_nonrew, mice_rew, mice_nonrew = load_and_process_data(
            similarity_metric=similarity_metric,
            select_lmi=select_lmi,
            zscore=zscore,
            projection_type=projection_type,
        )

    sns.set_theme(context='paper', style='ticks', palette='deep', font='sans-serif', font_scale=1)

    metrics_rew = _compute_within_day_metrics(corr_matrices_rew, mice_rew, 'R+')
    metrics_nonrew = _compute_within_day_metrics(corr_matrices_nonrew, mice_nonrew, 'R-')
    metrics_combined = pd.concat([metrics_rew, metrics_nonrew], ignore_index=True)

    # Reshape to long format for ANOVA
    day_cols = [f'within_day{d:+d}' for d in DAYS]
    long_df = metrics_combined.melt(
        id_vars=['mouse_id', 'reward_group'],
        value_vars=day_cols,
        var_name='day_label', value_name='correlation',
    )
    long_df['day'] = long_df['day_label'].str.extract(r'day([+-]?\d+)').astype(int)
    long_df['day_label'] = pd.Categorical(long_df['day_label'], categories=day_cols, ordered=True)

    # Statistics: 2-way ANOVA (day x reward_group)
    model = ols('correlation ~ C(reward_group) * C(day)', data=long_df).fit()
    anova_table = anova_lm(model, typ=2)
    anova_rows = []
    for term, row in anova_table.iterrows():
        anova_rows.append({
            'test': '2-way ANOVA',
            'term': term,
            'F': row.get('F', np.nan),
            'p_value': row['PR(>F)'],
            'significance': _significance_stars(row['PR(>F)']) if not np.isnan(row['PR(>F)']) else '',
        })

    # Post-hoc: Mann-Whitney U between groups for each day (no correction)
    stats_rows = anova_rows
    stats_dict = {}
    for day in DAYS:
        col = f'within_day{day:+d}'
        r_plus = metrics_rew[col].dropna()
        r_minus = metrics_nonrew[col].dropna()
        stat, p = mannwhitneyu(r_plus, r_minus, alternative='two-sided')
        stats_dict[day] = p
        stats_rows.append({
            'test': 'Mann-Whitney U (post-hoc)',
            'term': f'R+ vs R- day {day:+d}',
            'F': np.nan,
            'p_value': p,
            'significance': _significance_stars(p),
        })

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))

    sns.pointplot(
        data=long_df, x='day_label', y='correlation', hue='reward_group',
        palette=reward_palette[::-1], ax=ax, errorbar='ci',
        markers='o', linestyles='-', markersize=8, linewidth=2,
    )

    for mouse_id in metrics_combined['mouse_id'].unique():
        mouse_data = long_df[long_df['mouse_id'] == mouse_id].sort_values('day')
        rg = mouse_data['reward_group'].iloc[0]
        color = reward_palette[1] if rg == 'R+' else reward_palette[0]
        ax.plot(range(len(DAYS)), mouse_data['correlation'].values, color=color, alpha=0.3, linewidth=0.8, zorder=1)

    ylim_top = 0.8 if similarity_metric in ('pearson', 'cosine') else 0.3
    for day in DAYS:
        ax.text(DAYS.index(day), ylim_top * 0.95, _significance_stars(stats_dict[day]), ha='center', va='bottom', fontsize=9)

    ax.set_ylim(0, ylim_top)
    ax.set_xlabel('Day')
    ax.set_ylabel('Within-Day Correlation')
    ax.set_title('Within-Day Network Similarity')
    ax.set_xticklabels(DAYS)
    ax.legend(title='Group')
    sns.despine()
    plt.tight_layout()

    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f'figure_3i.{save_format}'), format=save_format, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"Figure 3i saved to: {os.path.join(output_dir, 'figure_3i.' + save_format)}")

    # Save CSVs
    metrics_combined.to_csv(os.path.join(OUTPUT_DIR, 'figure_3i_data.csv'), index=False)
    pd.DataFrame(stats_rows).to_csv(os.path.join(OUTPUT_DIR, 'figure_3i_stats.csv'), index=False)
    print(f"Figure 3i data/stats saved to: {OUTPUT_DIR}")


# ============================================================================
# Panel j: Network reorganization index
# ============================================================================

def panel_j_reorganization_index(
    corr_matrices_rew=None,
    corr_matrices_nonrew=None,
    mice_rew=None,
    mice_nonrew=None,
    similarity_metric='cosine',
    select_lmi=False,
    zscore=False,
    projection_type=None,
    output_dir=OUTPUT_DIR,
    save_format='svg',
    dpi=300,
):
    """
    Generate Figure 3 Panel j: Network reorganization index.

    Shows bar + swarm plot of the reorganization index ((within_pre + within_post) / 2
    - between_pre_post) comparing R+ and R- groups.

    Saves:
        figure_3j_data.csv: per-mouse reorganization index values
        figure_3j_stats.csv: Mann-Whitney U test result (R+ vs R-)
    """
    if corr_matrices_rew is None or corr_matrices_nonrew is None:
        corr_matrices_rew, corr_matrices_nonrew, mice_rew, mice_nonrew = load_and_process_data(
            similarity_metric=similarity_metric,
            select_lmi=select_lmi,
            zscore=zscore,
            projection_type=projection_type,
        )

    sns.set_theme(context='paper', style='ticks', palette='deep', font='sans-serif', font_scale=1)

    metrics_rew = _compute_reorganization_metrics(corr_matrices_rew, mice_rew, 'R+')
    metrics_nonrew = _compute_reorganization_metrics(corr_matrices_nonrew, mice_nonrew, 'R-')
    metrics_combined = pd.concat([metrics_rew, metrics_nonrew], ignore_index=True)

    # Statistics: Mann-Whitney U between groups
    stat, p = mannwhitneyu(
        metrics_rew['reorganization_index'].dropna(),
        metrics_nonrew['reorganization_index'].dropna(),
        alternative='two-sided',
    )
    stats_rows = [{'metric': 'reorganization_index', 'test': 'Mann-Whitney U', 'U_statistic': stat, 'p_value': p, 'significance': _significance_stars(p)}]

    long_df = metrics_combined[['mouse_id', 'reward_group', 'reorganization_index']].copy()
    long_df['metric'] = 'reorganization_index'

    fig, ax = plt.subplots(1, 1, figsize=(4, 5))

    sns.barplot(
        data=long_df, x='metric', y='reorganization_index', hue='reward_group',
        palette=reward_palette[::-1], ax=ax, errorbar='ci',
    )
    sns.swarmplot(
        data=long_df, x='metric', y='reorganization_index', hue='reward_group',
        dodge=True, ax=ax, size=4, color='grey', legend=False,
    )

    ylim_top = 0.3 if similarity_metric in ('pearson', 'cosine') else 0.15
    p_text = 'p<0.001' if p < 0.001 else f'p={p:.3f}' if p < 0.01 else f'p={p:.2f}'
    ax.text(0, ylim_top * 0.95, p_text, ha='center', va='bottom', fontsize=9)

    ax.set_ylim(0, ylim_top)
    ax.set_xlabel('')
    ax.set_ylabel('Reorganization Index')
    ax.set_title('Network Reorganization')
    ax.set_xticklabels(['Reorganization\nIndex'])
    ax.legend(title='Group')
    sns.despine()
    plt.tight_layout()

    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f'figure_3j.{save_format}'), format=save_format, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"Figure 3j saved to: {os.path.join(output_dir, 'figure_3j.' + save_format)}")

    # Save CSVs
    metrics_combined[['mouse_id', 'reward_group', 'within_pre', 'within_post', 'between_pre_post', 'reorganization_index']].to_csv(
        os.path.join(OUTPUT_DIR, 'figure_3j_data.csv'), index=False
    )
    pd.DataFrame(stats_rows).to_csv(os.path.join(OUTPUT_DIR, 'figure_3j_stats.csv'), index=False)
    print(f"Figure 3j data/stats saved to: {OUTPUT_DIR}")


# ============================================================================
# Main execution
# ============================================================================

if __name__ == '__main__':
    CORRELATION_METHOD = 'cosine'  # 'pearson', 'spearman', or 'cosine'
    SELECT_LMI = False
    ZSCORE = False
    PROJECTION_TYPE = None  # 'wS2', 'wM1', or None

    print("Loading data and computing similarity matrices...")
    corr_matrices_rew, corr_matrices_nonrew, mice_rew, mice_nonrew = load_and_process_data(
        similarity_metric=CORRELATION_METHOD,
        select_lmi=SELECT_LMI,
        zscore=ZSCORE,
        projection_type=PROJECTION_TYPE,
    )

    shared = dict(
        corr_matrices_rew=corr_matrices_rew,
        corr_matrices_nonrew=corr_matrices_nonrew,
        mice_rew=mice_rew,
        mice_nonrew=mice_nonrew,
        similarity_metric=CORRELATION_METHOD,
        select_lmi=SELECT_LMI,
        zscore=ZSCORE,
        projection_type=PROJECTION_TYPE,
    )

    print("\nGenerating panel h (correlation matrices)...")
    panel_h_correlation_matrices(**shared)

    print("\nGenerating panel i (within-day correlations)...")
    panel_i_within_day_correlations(**shared)

    print("\nGenerating panel j (reorganization index)...")
    panel_j_reorganization_index(**shared)

    print("\nDone!")
