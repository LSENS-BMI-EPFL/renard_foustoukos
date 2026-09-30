"""
Supplementary figure (panel index TBD): LMI distribution vs. shuffled null.

Reviewer request: for R+ and R- independently, overlay the real LMI
distribution against a null LMI distribution built by chance (label
shuffling), to show the real distribution is more spread than chance alone
would produce.

LMI computation (data source, response/baseline windows, pre=[-2,-1] vs
post=[+1,+2] day pooling) exactly mirrors the "Compute LMI" section of
src/preprocessing/processing_tensor_data/stats_on_tensors.py. The null
distribution reuses the same shuffle procedure already used there for LMI
significance testing (imaging.compute_roc), via the new
return_shuffles=True option, which keeps every per-cell per-shuffle null LMI
value instead of collapsing them to a single percentile. All shuffle x cell
null values are pooled per reward group (not averaged per cell, which would
artificially shrink the null's spread, and not one shuffle per cell, which
would give too few null points for a stable histogram).

The null distribution (slow) is computed when its cache is missing or with
--recompute, and loaded from the cache otherwise:
    python figures/revisions/figure_3f_LMIshuffles.py [--recompute]

Real LMI values are loaded as-is from lmi_results.csv (already computed).
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp, levene

from fast_learning import paths, database
from fast_learning import imaging
from fast_learning.plotting import reward_palette, save_figure
from fast_learning.stats import significance_stars as _significance_stars


# ============================================================================
# Parameters
# ============================================================================

RESPONSE_WIN = (0, 0.300)
BASELINE_WIN = (-1, 0)
N_SHUFFLES = 100
DAYS = ['-2', '-1', '0', '+1', '+2']

PROCESSED_DATA_DIR = paths.processed_dir
NULL_LMI_CSV = os.path.join(PROCESSED_DATA_DIR, 'lmi_null_shuffles.csv')
LMI_RESULTS_CSV = os.path.join(paths.processed_dir, 'lmi_results.csv')
OUTPUT_DIR = os.path.join(paths.manuscript_output_dir, 'revisions', 'figure_3f_LMIshuffles')


# ============================================================================
# Helpers
# ============================================================================


# ============================================================================
# Null distribution computation
# ============================================================================


def compute_null_lmi_distribution(n_shuffles=N_SHUFFLES):
    """Recompute LMI shuffles for every mouse, pooling per-cell per-shuffle
    null LMI values into one long-format DataFrame.

    Mirrors stats_on_tensors.py's "Compute LMI" section exactly (same data
    source, windows, pre/post day pooling); the only difference is
    return_shuffles=True to keep the full per-shuffle null LMI values instead
    of only the significance percentile.

    Saves NULL_LMI_CSV.
    """
    db_path = paths.db_path
    nwb_path = paths.nwb_dir

    _, _, mice_list, _ = database.select_sessions_from_db(
        db_path,
        nwb_path,
        exclude_cols=['exclude', 'two_p_exclude'],
        experimenters=['AR', 'GF', 'MI'],
        day=DAYS,
        two_p_imaging='yes',
    )

    rows = []
    for mouse_id in mice_list:
        print(f'Processing {mouse_id}')
        reward_group = database.get_mouse_reward_group_from_db(db_path, mouse_id)

        # Same loader as 05_lmi.py, so artefact cells are excluded.
        data_mapping = imaging.load_mouse_xarray(
            mouse_id, paths.tensor_dir, 'tensor_xarray_mapping_data.nc', subtracted=False
        )
        data_mapping = data_mapping - np.nanmean(
            data_mapping.sel(time=slice(*BASELINE_WIN)), axis=2, keepdims=True
        )

        data_pre = data_mapping.sel(trial=data_mapping.coords['day'].isin([-2, -1]))
        data_pre = data_pre.sel(time=slice(*RESPONSE_WIN)).mean(dim='time')
        data_post = data_mapping.sel(trial=data_mapping.coords['day'].isin([1, 2]))
        data_post = data_post.sel(time=slice(*RESPONSE_WIN)).mean(dim='time')

        _, _, lmi_shuffles = imaging.compute_roc(
            data_pre, data_post, nshuffles=n_shuffles, return_shuffles=True
        )

        # lmi_shuffles shape: (n_cells, n_shuffles) -- pool to long format.
        rows.append(
            pd.DataFrame(
                {
                    'mouse_id': mouse_id,
                    'reward_group': reward_group,
                    'null_lmi': lmi_shuffles.ravel(),
                }
            )
        )

    null_df = pd.concat(rows, ignore_index=True)
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    null_df.to_csv(NULL_LMI_CSV, index=False)
    print(f"Saved: {NULL_LMI_CSV}  ({len(null_df)} rows)")
    return null_df


def load_null_lmi_distribution():
    if not os.path.exists(NULL_LMI_CSV):
        raise FileNotFoundError(
            f"Pre-computed null LMI data not found: {NULL_LMI_CSV}\nRun with --recompute first."
        )
    df = pd.read_csv(NULL_LMI_CSV)
    print(f"Loaded: {NULL_LMI_CSV}  ({len(df)} rows)")
    return df


def load_real_lmi():
    """Load real LMI values and assign reward groups (mirrors figure_3f_g.py)."""
    lmi_df = pd.read_csv(LMI_RESULTS_CSV)
    lmi_mice = set(lmi_df['mouse_id'].unique())
    _, _, mice, _ = database.select_sessions_from_db(paths.db_path, paths.nwb_dir, two_p_imaging='yes')
    print(f"  lmi_results.csv mice ({len(lmi_mice)}): {sorted(lmi_mice)}")
    print(f"  DB two_p_imaging=='yes' mice ({len(mice)}): {sorted(mice)}")
    print(f"  Overlap: {len(lmi_mice & set(mice))} mice")
    for mouse in lmi_df['mouse_id'].unique():
        lmi_df.loc[lmi_df['mouse_id'] == mouse, 'reward_group'] = database.get_mouse_reward_group_from_db(
            paths.db_path, mouse
        )
    lmi_df = lmi_df.loc[lmi_df['mouse_id'].isin(mice)]
    print(
        f"  lmi_df after DB filter: {len(lmi_df)} rows, "
        f"reward_group counts: {lmi_df['reward_group'].value_counts(dropna=False).to_dict()}"
    )
    return lmi_df


# ============================================================================
# Plot
# ============================================================================


def plot_lmi_vs_null(
    lmi_df, null_df, output_dir=OUTPUT_DIR, filename='supp_3_LMI_shuffles', save_format='svg', dpi=300
):
    """Real LMI distribution overlaid with pooled shuffled-null LMI
    distribution, one panel per reward group.

    Saves:
        <filename>.svg        -- figure
        <filename>_stats.csv  -- per-group real vs. null spread comparison
    """
    sns.set_theme(
        context='paper',
        style='ticks',
        font='sans-serif',
        font_scale=1,
        rc={'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none'},
    )

    reward_groups = ['R+', 'R-']
    rg_colors = {'R+': reward_palette[1], 'R-': reward_palette[0]}
    bin_edges = np.linspace(-1, 1, 31)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    stats_rows = []

    for ax, rg in zip(axes, reward_groups):
        real_vals = lmi_df.loc[lmi_df['reward_group'] == rg, 'lmi'].dropna().values
        null_vals = null_df.loc[null_df['reward_group'] == rg, 'null_lmi'].dropna().values

        sns.histplot(
            null_vals,
            bins=bin_edges,
            stat='probability',
            element='step',
            fill=False,
            color='dimgray',
            linewidth=1.2,
            label='Shuffled null',
            ax=ax,
        )
        sns.histplot(
            real_vals,
            bins=bin_edges,
            stat='probability',
            kde=True,
            color=rg_colors[rg],
            alpha=0.5,
            label='Real LMI',
            ax=ax,
        )

        ks_stat, ks_p = ks_2samp(real_vals, null_vals, alternative='two-sided')
        lev_stat, lev_p = levene(real_vals, null_vals)
        std_real, std_null = float(np.std(real_vals)), float(np.std(null_vals))

        stats_rows.append(
            {
                'reward_group': rg,
                'n_real': len(real_vals),
                'n_null': len(null_vals),
                'std_real': std_real,
                'std_null': std_null,
                'ks_statistic': ks_stat,
                'ks_p_value': ks_p,
                'levene_statistic': lev_stat,
                'levene_p_value': lev_p,
            }
        )

        ax.text(
            0.02,
            0.98,
            f'std real = {std_real:.3f}\nstd null = {std_null:.3f}\n'
            f"Levene p = {lev_p:.3g} {_significance_stars(lev_p)}",
            transform=ax.transAxes,
            va='top',
            ha='left',
            fontsize=8,
        )
        ax.set_title(rg, fontsize=10, fontweight='bold')
        ax.set_xlim(-1, 1)
        ax.set_xlabel('LMI')
        ax.set_ylabel('Probability' if ax is axes[0] else '')
        ax.legend(frameon=False, fontsize=8)

    sns.despine(trim=True)
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    save_figure(fig, os.path.join(output_dir, f'{filename}.{save_format}'))
    print(f"Saved: {os.path.join(output_dir, filename + '.' + save_format)}")

    stats_df = pd.DataFrame(stats_rows)
    stats_df.to_csv(os.path.join(output_dir, f'{filename}_stats.csv'), index=False)
    print(f"Saved: {os.path.join(output_dir, filename + '_stats.csv')}")

    return stats_df


# ============================================================================
# Main execution
# ============================================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Real vs shuffled-null LMI distributions.')
    parser.add_argument(
        '--recompute', action='store_true', help='recompute the null LMI distribution even if it is cached'
    )
    args = parser.parse_args()
    print(f"Output directory: {OUTPUT_DIR}")

    if args.recompute or not os.path.exists(NULL_LMI_CSV):
        null_df = compute_null_lmi_distribution()
    else:
        null_df = load_null_lmi_distribution()

    lmi_df = load_real_lmi()
    stats_df = plot_lmi_vs_null(lmi_df, null_df)
    print("\n=== Real vs. shuffled-null LMI spread, per reward group ===")
    print(stats_df.to_string(index=False))
