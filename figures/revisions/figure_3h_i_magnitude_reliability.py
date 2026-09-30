"""
Figure 3H-I revision: response magnitude and split-half reliability controls.

Addresses a reviewer comment on the correlation analysis in Figure 3H-J:
cosine similarity is scale-invariant, so a reported decrease in R- animals
cannot be attributed to uniform response suppression on its own, and needs
clarification (heterogeneous suppression vs. reduced SNR / increased
variability). Scoped to panels 3H/3I only, per the user's choice (not yet
extended to 3J's reorganization index, which is built from the same raw
similarity matrix and would need an analogous fix later).

Two companion metrics are added, both computed on the exact same trial
response vectors figure_3h_j.py's cosine-similarity matrices are built
from:

  - Mean response magnitude (L2 norm): a genuinely scale-sensitive metric
    -- literally the ||v|| that cosine similarity's v/||v|| normalization
    discards -- used to directly check whether uniform suppression is
    occurring, which the existing cosine-similarity panels cannot detect
    by construction.

  - Within-day split-half reliability: for each day, repeatedly split the
    40 trials into two random halves, average the raw (un-normalized)
    trial vectors within each half, and cosine-compare the two
    noise-reduced half-averages, instead of comparing raw single-trial
    pairs. Comparing this trajectory against the existing raw within-day
    cosine-similarity trajectory is the diagnostic: if split-half
    reliability declines about as much as the raw metric, the decline
    reflects genuine pattern reorganization (heterogeneous suppression);
    if split-half reliability stays comparatively flat/high, trial-to-trial
    noise/SNR is the dominant driver instead.

This script does not modify figure_3h_j.py. load_and_process_data() there
does not expose the raw per-trial response vectors (only the derived
similarity matrices), so _load_vectors_and_matrices() below mirrors that
function's data-loading loop to additionally keep the raw vectors, while
reusing _compute_similarity_matrix(), _compute_within_day_metrics(), and
_significance_stars() from figure_3h_j.py unchanged for everything else.

Figures and CSVs are saved to
    paths.manuscript_output_dir/revisions/figure_3h_i_magnitude_reliability/output/.
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

from fast_learning import imaging
from fast_learning import paths, database
from fast_learning.plotting import reward_palette, save_figure
from fast_learning.similarity import (
    DAYS,
    N_MAP_TRIALS,
    WIN,
    compute_similarity_matrix as _compute_similarity_matrix,
    compute_within_day_metrics as _compute_within_day_metrics,
)
from fast_learning.stats import significance_stars as _significance_stars


# ============================================================================
# Parameters
# ============================================================================

N_SPLITS = 100  # random splits averaged per (mouse, day) split-half estimate
GLOBAL_SEED = 42  # fixed for reproducibility of this stochastic control

OUTPUT_DIR = os.path.join(
    paths.manuscript_output_dir, 'revisions', 'figure_3h_i_magnitude_reliability', 'output'
)


# ============================================================================
# Data loading (mirrors figure_3h_j.py's load_and_process_data loop, but
# additionally keeps the raw response vectors)
# ============================================================================


def _load_vectors_and_matrices(
    similarity_metric='cosine',
    select_lmi=False,
    zscore=False,
    projection_type=None,
    n_min_proj=5,
    subtract_baseline=True,
):
    """Load imaging data and return both the raw (cell x trial) response
    vectors and their similarity matrices, per mouse.

    Mirrors figure_3h_j.py's load_and_process_data() loop exactly (same
    mouse/day/trial selection, same window-averaging and optional
    z-scoring), reusing _compute_similarity_matrix() from figure_3h_j.py
    unchanged for the similarity-matrix computation, so the matrices this
    returns are identical to what the original pipeline produces.
    """
    _, _, mice, db = database.select_sessions_from_db(paths.db_path, paths.nwb_dir, two_p_imaging='yes')

    selected_cells = None
    if select_lmi:
        processed_folder = paths.processed_dir
        lmi_df = pd.read_csv(os.path.join(processed_folder, 'lmi_results.csv'))
        selected_cells = lmi_df.loc[(lmi_df['lmi_p'] <= 0.025) | (lmi_df['lmi_p'] >= 0.975)]

    vectors_rew, vectors_nonrew = [], []
    mice_rew, mice_nonrew = [], []

    for mouse in mice:
        print(f"Processing mouse: {mouse}")
        folder = paths.tensor_dir
        xarray = imaging.load_mouse_xarray(
            mouse, folder, 'tensor_xarray_mapping_data.nc', subtracted=subtract_baseline
        )
        rew_gp = database.get_mouse_reward_group_from_db(paths.db_path, mouse, db)

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

    return corr_matrices_rew, corr_matrices_nonrew, mice_rew, mice_nonrew, vectors_rew, vectors_nonrew


# ============================================================================
# New metric: mean response magnitude (L2 norm)
# ============================================================================


def _compute_response_magnitude_metrics(vectors, mice_ids, reward_group):
    """Mean L2 norm of the trial response vector, per day, per mouse.

    This is the magnitude information cosine similarity discards (cosine
    similarity operates on v / ||v||; this tracks ||v|| itself), used to
    directly check for uniform response suppression across days/groups.
    """
    results = []
    for v in vectors:
        data = np.nan_to_num(v.values, nan=0.0)  # (n_cells, n_trials)
        row = {}
        for i, day in enumerate(DAYS):
            day_idx = np.arange(i * N_MAP_TRIALS, (i + 1) * N_MAP_TRIALS)
            trial_norms = np.linalg.norm(data[:, day_idx], axis=0)
            row[f'magnitude_day{day:+d}'] = np.mean(trial_norms)
        results.append(row)
    df = pd.DataFrame(results)
    df['reward_group'] = reward_group
    df['mouse_id'] = mice_ids
    return df


# ============================================================================
# New metric: within-day split-half reliability
# ============================================================================


def _split_half_cosine(day_data, n_splits, rng):
    """Split-half reliability for one day's trials: repeatedly split into two
    random halves, average the raw (un-normalized) trial vectors within each
    half, cosine-compare the two half-averages, and return the mean over
    n_splits random partitions.

    day_data: (n_cells, n_trials_day) raw response vectors for one day.
    """
    n_trials_day = day_data.shape[1]
    half = n_trials_day // 2

    sims = np.full(n_splits, np.nan)
    for s in range(n_splits):
        perm = rng.permutation(n_trials_day)
        idx_a, idx_b = perm[:half], perm[half : 2 * half]
        mean_a = day_data[:, idx_a].mean(axis=1)
        mean_b = day_data[:, idx_b].mean(axis=1)
        norm_a, norm_b = np.linalg.norm(mean_a), np.linalg.norm(mean_b)
        if norm_a > 0 and norm_b > 0:
            sims[s] = np.dot(mean_a, mean_b) / (norm_a * norm_b)
    return np.nanmean(sims)


def _compute_split_half_reliability_metrics(
    vectors, mice_ids, reward_group, n_splits=N_SPLITS, seed=GLOBAL_SEED
):
    """Within-day split-half reliability, per day, per mouse.

    Compares two noise-reduced (20-trial-averaged) estimates of the same
    day's response pattern, rather than raw single-trial pairs -- unlike
    the existing within-day cosine metric, this is much less sensitive to
    pure trial-to-trial noise, so comparing the two trajectories directly
    distinguishes genuine pattern reorganization from reduced SNR.
    """
    results = []
    for m_idx, v in enumerate(vectors):
        data = np.nan_to_num(v.values, nan=0.0)
        rng = np.random.default_rng(seed + m_idx)
        row = {}
        for i, day in enumerate(DAYS):
            day_idx = np.arange(i * N_MAP_TRIALS, (i + 1) * N_MAP_TRIALS)
            row[f'splithalf_day{day:+d}'] = _split_half_cosine(data[:, day_idx], n_splits=n_splits, rng=rng)
        results.append(row)
    df = pd.DataFrame(results)
    df['reward_group'] = reward_group
    df['mouse_id'] = mice_ids
    return df


# ============================================================================
# Generic day x reward_group trajectory panel (shared by both new metrics,
# and reused to regenerate the existing within-day cosine metric for
# direct side-by-side comparison)
# ============================================================================


def _panel_metric_across_days(
    metrics_combined,
    value_prefix,
    ylabel,
    title,
    filename,
    output_dir=OUTPUT_DIR,
    save_format='svg',
    dpi=300,
):
    """Day x reward_group trajectory panel: pointplot with individual mouse
    lines, 2-way ANOVA (day x reward_group) + per-day Mann-Whitney U
    post-hoc -- mirrors figure_3h_j.py's panel_i_within_day_correlations
    layout and statistics exactly, so these new metrics are directly
    comparable to the existing within-day cosine-similarity trajectory.

    Saves:
        <filename>.svg
        <filename>_data.csv
        <filename>_stats.csv
    """
    sns.set_theme(context='paper', style='ticks', palette='deep', font='sans-serif', font_scale=1)

    day_cols = [f'{value_prefix}day{d:+d}' for d in DAYS]
    long_df = metrics_combined.melt(
        id_vars=['mouse_id', 'reward_group'],
        value_vars=day_cols,
        var_name='day_label',
        value_name='value',
    )
    long_df['day'] = long_df['day_label'].str.extract(r'day([+-]?\d+)').astype(int)
    long_df['day_label'] = pd.Categorical(long_df['day_label'], categories=day_cols, ordered=True)

    model = ols('value ~ C(reward_group) * C(day)', data=long_df).fit()
    anova_table = anova_lm(model, typ=2)
    stats_rows = []
    for term, row in anova_table.iterrows():
        stats_rows.append(
            {
                'test': '2-way ANOVA',
                'term': term,
                'F': row.get('F', np.nan),
                'p_value': row['PR(>F)'],
                'significance': _significance_stars(row['PR(>F)']) if not np.isnan(row['PR(>F)']) else '',
            }
        )

    stats_dict = {}
    for day in DAYS:
        col = f'{value_prefix}day{day:+d}'
        r_plus = metrics_combined.loc[metrics_combined['reward_group'] == 'R+', col].dropna()
        r_minus = metrics_combined.loc[metrics_combined['reward_group'] == 'R-', col].dropna()
        stat, p = mannwhitneyu(r_plus, r_minus, alternative='two-sided')
        stats_dict[day] = p
        stats_rows.append(
            {
                'test': 'Mann-Whitney U (post-hoc)',
                'term': f'R+ vs R- day {day:+d}',
                'F': np.nan,
                'p_value': p,
                'significance': _significance_stars(p),
            }
        )

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    sns.pointplot(
        data=long_df,
        x='day_label',
        y='value',
        hue='reward_group',
        palette=reward_palette[::-1],
        ax=ax,
        errorbar='ci',
        markers='o',
        linestyles='-',
        markersize=8,
        linewidth=2,
    )
    for mouse_id in metrics_combined['mouse_id'].unique():
        mouse_data = long_df[long_df['mouse_id'] == mouse_id].sort_values('day')
        rg = mouse_data['reward_group'].iloc[0]
        color = reward_palette[1] if rg == 'R+' else reward_palette[0]
        ax.plot(range(len(DAYS)), mouse_data['value'].values, color=color, alpha=0.3, linewidth=0.8, zorder=1)

    ylim_top = float(long_df['value'].max()) * 1.2
    for day in DAYS:
        ax.text(
            DAYS.index(day),
            ylim_top * 0.95,
            _significance_stars(stats_dict[day]),
            ha='center',
            va='bottom',
            fontsize=9,
        )

    ax.set_ylim(0, ylim_top)
    ax.set_xlabel('Day')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticklabels(DAYS)
    ax.legend(title='Group')
    sns.despine()
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    save_figure(plt.gcf(), os.path.join(output_dir, f'{filename}.{save_format}'))
    plt.close()
    print(f"{filename} saved to: {os.path.join(output_dir, filename + '.' + save_format)}")

    metrics_combined.to_csv(os.path.join(output_dir, f'{filename}_data.csv'), index=False)
    pd.DataFrame(stats_rows).to_csv(os.path.join(output_dir, f'{filename}_stats.csv'), index=False)
    print(f"{filename} data/stats saved to: {output_dir}")


# ============================================================================
# Main execution
# ============================================================================

if __name__ == '__main__':
    CORRELATION_METHOD = 'cosine'
    SELECT_LMI = False
    ZSCORE = False
    PROJECTION_TYPE = None

    print("Loading data and computing similarity matrices + raw response vectors...")
    (corr_matrices_rew, corr_matrices_nonrew, mice_rew, mice_nonrew, vectors_rew, vectors_nonrew) = (
        _load_vectors_and_matrices(
            similarity_metric=CORRELATION_METHOD,
            select_lmi=SELECT_LMI,
            zscore=ZSCORE,
            projection_type=PROJECTION_TYPE,
        )
    )

    print(
        "\nRegenerating existing within-day cosine similarity (reference, unchanged) for direct comparison..."
    )
    within_day_rew = _compute_within_day_metrics(corr_matrices_rew, mice_rew, 'R+')
    within_day_nonrew = _compute_within_day_metrics(corr_matrices_nonrew, mice_nonrew, 'R-')
    within_day_combined = pd.concat([within_day_rew, within_day_nonrew], ignore_index=True)
    _panel_metric_across_days(
        within_day_combined,
        value_prefix='within_',
        ylabel='Within-Day Cosine Similarity',
        title='Within-Day Similarity (reference, same as existing panel i)',
        filename='figure_3i_within_day_reference',
    )

    print("\nComputing mean response magnitude (L2 norm)...")
    mag_rew = _compute_response_magnitude_metrics(vectors_rew, mice_rew, 'R+')
    mag_nonrew = _compute_response_magnitude_metrics(vectors_nonrew, mice_nonrew, 'R-')
    mag_combined = pd.concat([mag_rew, mag_nonrew], ignore_index=True)
    _panel_metric_across_days(
        mag_combined,
        value_prefix='magnitude_',
        ylabel='Mean Response L2 Norm',
        title='Response Magnitude Across Days',
        filename='figure_3h_response_magnitude',
    )

    print("\nComputing within-day split-half reliability...")
    sh_rew = _compute_split_half_reliability_metrics(vectors_rew, mice_rew, 'R+')
    sh_nonrew = _compute_split_half_reliability_metrics(vectors_nonrew, mice_nonrew, 'R-')
    sh_combined = pd.concat([sh_rew, sh_nonrew], ignore_index=True)
    _panel_metric_across_days(
        sh_combined,
        value_prefix='splithalf_',
        ylabel='Split-Half Reliability (cosine)',
        title='Within-Day Split-Half Reliability',
        filename='figure_3i_split_half_reliability',
    )

    print("\nDone!")
