"""
Reactivation Surrogate Analysis - Per-Mouse Threshold Determination

Computes a single surrogate-based correlation threshold per mouse, using
pre-learning days (-2 and -1) independently with day-matched templates.

Differences from reactivation_surrogates_per_day.py:
- Uses only pre-learning days (-2 and -1) to build the null distribution
- Produces ONE threshold per mouse (no day column in output CSV)
- Template for each day is built from that day's own mapping trials

Approach:
1. For each pre-learning day (-2, -1):
   a. Build whisker response template from that day's mapping trials
   b. Load no-stim trials for that day
   c. Generate N surrogate datasets by circular time-shifting each cell independently
   d. Compute template correlation for each surrogate
   e. Extract percentile from each surrogate's correlation distribution → N values
2. Pool all surrogate percentile values across days (2N total)
3. Take median of the pooled distribution → single threshold per mouse

Output:
- CSV file with one row per mouse (surrogate_thresholds_per_mouse_p99.csv, etc.)
- Per-mouse PDF showing the pooled surrogate distribution
- Summary PDF comparing thresholds across mice and reward groups
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import percentileofscore
from joblib import Parallel, delayed

sys.path.append(r'/home/aprenard/repos/NWB_analysis')
sys.path.append(r'/home/aprenard/repos/fast-learning')
import src.utils.utils_imaging as utils_imaging
import src.utils.utils_io as io
from src.utils.utils_plot import *
from src.core_analysis.reactivations.reactivation import (
    create_whisker_template,
    compute_template_correlation
)
from src.core_analysis.reactivations.reactivation_surrogates_per_day import (
    create_surrogate_by_circular_shift,
    compute_surrogate_thresholds,
)


# =============================================================================
# PARAMETERS
# =============================================================================

sampling_rate = 30  # Hz
prelearning_days = [-2, -1]       # Days used to build the null distribution (each with its own template)
days_str = ['-2', '-1', '0', '+1', '+2']

# Template parameters
threshold_dff = None  # dF/F threshold for template cells (None = all cells)

# Surrogate parameters
n_surrogates = 1000
min_shift_frames = 0
percentiles_to_compute = [99]
np.random.seed(42)

# Parallel processing
n_jobs = 35

# Visualization
sns.set_theme(context='paper', style='ticks', palette='deep', font='sans-serif', font_scale=1)

# Load database
_, _, all_mice, db = io.select_sessions_from_db(
    io.db_path, io.nwb_dir, two_p_imaging='yes'
)

r_plus_mice = []
r_minus_mice = []
for mouse in all_mice:
    try:
        rg = io.get_mouse_reward_group_from_db(io.db_path, mouse, db=db)
        if rg == 'R+':
            r_plus_mice.append(mouse)
        elif rg == 'R-':
            r_minus_mice.append(mouse)
    except:
        continue

print(f"Found {len(r_plus_mice)} R+ mice: {r_plus_mice}")
print(f"Found {len(r_minus_mice)} R- mice: {r_minus_mice}")


# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def analyze_mouse_surrogates(mouse, threshold_dff=None, n_surrogates=1000,
                              percentiles=[95, 99, 99.9], verbose=True):
    """
    Compute surrogate threshold for one mouse from pre-learning data.

    For each pre-learning day (-2 and -1), builds a day-matched template from
    mapping trials and runs circular-shift surrogates on no-stim data. The
    raw surrogate percentile values from both days are pooled, and the median
    of the pooled distribution becomes the single per-mouse threshold.

    Parameters
    ----------
    mouse : str
    threshold_dff : float or None
    n_surrogates : int
    percentiles : list of float
    verbose : bool

    Returns
    -------
    results_dfs : dict
        {percentile: DataFrame with one row for this mouse}
    surrogate_data : dict
        {percentile: surrogate_results dict} for plotting
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"ANALYZING MOUSE: {mouse}")
        print(f"{'='*60}")

    # Load xarray once for all pre-learning days
    try:
        folder = os.path.join(io.solve_common_paths('processed_data'), 'mice')
        xarray_learning = utils_imaging.load_mouse_xarray(
            mouse, folder, 'tensor_xarray_learning_data.nc', substracted=False
        )
    except Exception as e:
        if verbose:
            print(f"  Error loading xarray for {mouse}: {e}")
        return None, None

    # Accumulate surrogate percentile values and observed percentiles per day
    # {percentile: [array_from_day_m2, array_from_day_m1]}
    pooled_surr_percentiles = {p: [] for p in percentiles}
    pooled_obs_percentiles = {p: [] for p in percentiles}
    total_frames = 0
    days_processed = 0

    for day in prelearning_days:
        # Build template from this day's mapping trials
        try:
            template, cells_mask = create_whisker_template(
                mouse, day, threshold_dff, verbose=verbose
            )
            n_cells_responsive = cells_mask.sum()
        except Exception as e:
            if verbose:
                print(f"  Warning: Could not build template for day {day}: {e}")
            continue

        if n_cells_responsive < 3 and threshold_dff is not None:
            if verbose:
                print(f"  Warning: Only {n_cells_responsive} responsive cells on day {day}, skipping.")
            continue

        # Load no-stim data for this day
        try:
            xarray_day = xarray_learning.sel(trial=xarray_learning['day'] == day)
            nostim = xarray_day.sel(trial=xarray_day['no_stim'] == 1)

            if len(nostim.trial) < 5:
                if verbose:
                    print(f"  Warning: Only {len(nostim.trial)} no-stim trials on day {day}, skipping.")
                continue

            n_cells = nostim.shape[0]
            data_day = nostim.values.reshape(n_cells, -1)
            data_day = np.nan_to_num(data_day, nan=0.0)
            n_frames_day = data_day.shape[1]

            if n_frames_day < 60:
                if verbose:
                    print(f"  Warning: Day {day} data too short ({n_frames_day} frames), skipping.")
                continue

        except Exception as e:
            if verbose:
                print(f"  Error loading no-stim data for day {day}: {e}")
            continue

        if verbose:
            print(f"  Day {day}: {len(nostim.trial)} trials, {n_frames_day} frames, "
                  f"{n_cells_responsive} responsive cells in template")

        # Run circular-shift surrogates for this day with its own template
        day_results = compute_surrogate_thresholds(
            data_day, template, n_surrogates, min_shift_frames,
            percentiles=percentiles, verbose=verbose
        )

        for p in percentiles:
            pooled_surr_percentiles[p].append(day_results[p]['surrogate_percentiles'])
            pooled_obs_percentiles[p].append(day_results[p]['observed_percentile'])

        total_frames += n_frames_day
        days_processed += 1

    if days_processed == 0:
        if verbose:
            print(f"  No usable pre-learning data for {mouse}")
        return None, None

    # Pool surrogate percentile arrays across days and compute final threshold
    results_dfs = {}
    surrogate_data = {}

    for p in percentiles:
        if len(pooled_surr_percentiles[p]) == 0:
            continue

        pooled = np.concatenate(pooled_surr_percentiles[p])
        observed = np.median(pooled_obs_percentiles[p])

        threshold = np.median(pooled)
        ci = np.percentile(pooled, [2.5, 97.5])
        p_value = np.mean(pooled > observed)

        p_results = {
            'threshold_percentile_median': threshold,
            'threshold_percentile_ci': ci,
            'surrogate_percentiles': pooled,
            'observed_percentile': observed,
            'p_value_percentile': p_value,
            'percentile_value': p,
        }

        results_dfs[p] = pd.DataFrame([{
            'mouse_id': mouse,
            'n_frames': total_frames,
            'n_days_processed': days_processed,
            'n_surrogates_per_day': n_surrogates,
            'percentile_value': p,
            'threshold_percentile_median': threshold,
            'threshold_percentile_ci_lower': ci[0],
            'threshold_percentile_ci_upper': ci[1],
            'observed_percentile': observed,
            'p_value_percentile': p_value,
        }])
        surrogate_data[p] = p_results

    if verbose:
        print(f"\n  Completed mouse {mouse} ({days_processed} days processed, "
              f"{total_frames} total frames)")

    return results_dfs, surrogate_data


def process_single_mouse(mouse, threshold_dff, n_surrogates,
                         percentiles=[95, 99, 99.9], verbose=False):
    """Wrapper for parallel processing."""
    results_dfs, surrogate_data = analyze_mouse_surrogates(
        mouse, threshold_dff=threshold_dff,
        n_surrogates=n_surrogates, percentiles=percentiles, verbose=verbose
    )
    return (mouse, results_dfs, surrogate_data)


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def plot_surrogate_distributions(mouse, surrogate_data, save_path):
    """
    Single-page PDF: surrogate distribution for this mouse.

    Parameters
    ----------
    mouse : str
    surrogate_data : dict
        {percentile: surrogate_results}
    save_path : str
    """
    if surrogate_data is None or len(surrogate_data) == 0:
        print(f"  Warning: No surrogate data for {mouse}, skipping.")
        return

    with PdfPages(save_path) as pdf:
        for p, results in sorted(surrogate_data.items()):
            percentile_val = int(results.get('percentile_value', p))

            fig, ax = plt.subplots(figsize=(7, 5))
            ax.hist(results['surrogate_percentiles'], bins=50, alpha=0.7,
                    color='steelblue', edgecolor='black', linewidth=0.5)
            ax.axvline(results['threshold_percentile_median'], color='blue',
                       linewidth=2, label=f"Median: {results['threshold_percentile_median']:.4f}")
            ax.axvspan(results['threshold_percentile_ci'][0],
                       results['threshold_percentile_ci'][1],
                       alpha=0.2, color='blue', label='95% CI')
            ax.axvline(results['observed_percentile'], color='red', linestyle='--',
                       linewidth=2, label=f"Observed: {results['observed_percentile']:.4f}")

            stats_text = (f"n_surrogates = {len(results['surrogate_percentiles'])}\n"
                          f"p-value = {results['p_value_percentile']:.3f}")
            ax.text(0.97, 0.97, stats_text, transform=ax.transAxes, fontsize=10,
                    va='top', ha='right',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))

            ax.set_xlabel(f'{percentile_val}th Percentile Correlation', fontweight='bold')
            ax.set_ylabel('Count', fontweight='bold')
            ax.set_title(f'{mouse} — pre-learning surrogate: {percentile_val}th percentile',
                         fontweight='bold')
            ax.legend(loc='upper left')
            ax.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()

    print(f"    Saved PDF: {save_path}")


def plot_threshold_summary_across_mice(all_results_df, save_path):
    """
    Summary PDF: per-mouse thresholds split by reward group.

    Parameters
    ----------
    all_results_df : pd.DataFrame
        One row per mouse (all percentiles concatenated).
    save_path : str
    """
    all_results_df = all_results_df.copy()
    all_results_df['reward_group'] = all_results_df['mouse_id'].apply(
        lambda m: io.get_mouse_reward_group_from_db(io.db_path, m, db=db)
    )
    percentile_val = int(all_results_df['percentile_value'].iloc[0])

    with PdfPages(save_path) as pdf:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Left: bar per mouse coloured by reward group
        ax = axes[0]
        sorted_df = all_results_df.sort_values('threshold_percentile_median', ascending=False)
        colors = [('steelblue' if rg == 'R+' else 'coral')
                  for rg in sorted_df['reward_group']]
        x_pos = np.arange(len(sorted_df))
        ax.bar(x_pos, sorted_df['threshold_percentile_median'], color=colors,
               edgecolor='black', linewidth=0.8)
        ax.errorbar(x_pos, sorted_df['threshold_percentile_median'],
                    yerr=[sorted_df['threshold_percentile_median'] - sorted_df['threshold_percentile_ci_lower'],
                          sorted_df['threshold_percentile_ci_upper'] - sorted_df['threshold_percentile_median']],
                    fmt='none', color='black', capsize=3, linewidth=1)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(sorted_df['mouse_id'], rotation=45, ha='right', fontsize=8)
        ax.set_ylabel(f'{percentile_val}th Percentile Threshold', fontweight='bold')
        ax.set_title('Per-Mouse Threshold (sorted)', fontweight='bold')
        from matplotlib.patches import Patch
        ax.legend(handles=[Patch(color='steelblue', label='R+'),
                            Patch(color='coral', label='R-')], loc='upper right')
        ax.grid(True, alpha=0.3, axis='y')

        # Right: boxplot R+ vs R-
        ax = axes[1]
        for rg, color in [('R+', 'steelblue'), ('R-', 'coral')]:
            vals = all_results_df[all_results_df['reward_group'] == rg]['threshold_percentile_median']
            x = 0 if rg == 'R+' else 1
            bp = ax.boxplot(vals, positions=[x], widths=0.4, patch_artist=True,
                            showmeans=True, showfliers=True)
            bp['boxes'][0].set_facecolor(color)
            bp['boxes'][0].set_alpha(0.7)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(['R+', 'R-'])
        ax.set_ylabel(f'{percentile_val}th Percentile Threshold', fontweight='bold')
        ax.set_title('R+ vs R- Threshold Distribution', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        fig.suptitle('Per-Mouse Surrogate Thresholds (days -2 & -1, day-matched templates, pooled)',
                     fontsize=14, fontweight='bold', y=1.01)
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

    print(f"  Saved summary PDF: {save_path}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("REACTIVATION SURROGATE ANALYSIS - PER-MOUSE THRESHOLDS")
    print("="*60)

    print(f"\nParameters:")
    print(f"  Pre-learning days: {prelearning_days} (each day uses its own template)")
    print(f"  Responsiveness threshold: {threshold_dff}")
    print(f"  Number of surrogates: {n_surrogates}")
    print(f"  Percentiles: {percentiles_to_compute}")
    print(f"  Parallel jobs: {n_jobs}")

    output_dir = '/mnt/lsens-analysis/Anthony_Renard/analysis_output/fast-learning/reactivation_surrogates_per_mouse'
    output_dir = io.adjust_path_to_host(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nResults will be saved to: {output_dir}")

    # Process all mice in parallel
    all_mice_to_process = r_plus_mice + r_minus_mice
    print(f"\nProcessing {len(all_mice_to_process)} mice in parallel...")

    results_list = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(process_single_mouse)(mouse, threshold_dff, n_surrogates,
                                      percentiles=percentiles_to_compute, verbose=False)
        for mouse in all_mice_to_process
    )

    # Collect results
    all_results = {p: [] for p in percentiles_to_compute}
    all_surrogate_data = {p: {} for p in percentiles_to_compute}

    for mouse, results_dfs, surrogate_data in results_list:
        if results_dfs is not None:
            for p in percentiles_to_compute:
                all_results[p].append(results_dfs[p])
                all_surrogate_data[p][mouse] = surrogate_data[p]

    if all(len(all_results[p]) == 0 for p in percentiles_to_compute):
        print("\nERROR: No valid results collected!")
        sys.exit(1)

    all_results_dfs = {p: pd.concat(all_results[p], ignore_index=True)
                       for p in percentiles_to_compute}

    print(f"\nCollected results: {len(all_results_dfs[percentiles_to_compute[0]])} mice")

    # Save CSV files
    print("\nSaving CSV files...")
    for p in percentiles_to_compute:
        p_str = f"p{int(p)}" if p == int(p) else f"p{int(p*10)}"
        csv_path = os.path.join(output_dir, f'surrogate_thresholds_per_mouse_{p_str}.csv')
        all_results_dfs[p].to_csv(csv_path, index=False)
        print(f"  Saved {p}th percentile thresholds → {csv_path}")

    # Per-mouse PDFs
    print("\nGenerating per-mouse PDFs...")
    for p in percentiles_to_compute:
        p_str = f"p{int(p)}" if p == int(p) else f"p{int(p*10)}"
        pdf_dir = os.path.join(output_dir, f'per_mouse_pdfs_{p_str}')
        os.makedirs(pdf_dir, exist_ok=True)
        for mouse, surrogate_data in all_surrogate_data[p].items():
            if surrogate_data is not None:
                pdf_path = os.path.join(pdf_dir, f'{mouse}_surrogate_by_mouse_{p_str}.pdf')
                plot_surrogate_distributions(mouse, {p: surrogate_data}, pdf_path)

    # Summary PDFs
    print("\nGenerating summary PDFs...")
    for p in percentiles_to_compute:
        p_str = f"p{int(p)}" if p == int(p) else f"p{int(p*10)}"
        summary_path = os.path.join(output_dir, f'surrogate_threshold_summary_per_mouse_{p_str}.pdf')
        plot_threshold_summary_across_mice(all_results_dfs[p], summary_path)

    # Summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    for p in percentiles_to_compute:
        df = all_results_dfs[p]
        print(f"\n{p}th percentile:")
        for rg, mice in [('R+', r_plus_mice), ('R-', r_minus_mice)]:
            group = df[df['mouse_id'].isin(mice)]
            if len(group):
                print(f"  {rg} (n={len(group)}): "
                      f"{group['threshold_percentile_median'].mean():.4f} ± "
                      f"{group['threshold_percentile_median'].std():.4f}")

    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"\nOutput files in: {output_dir}")
    for p in percentiles_to_compute:
        p_str = f"p{int(p)}" if p == int(p) else f"p{int(p*10)}"
        print(f"  - CSV: surrogate_thresholds_per_mouse_{p_str}.csv")
