"""
Reactivation Surrogate Analysis - Per-Day Threshold Determination

This script computes statistically principled correlation thresholds for reactivation
event detection via circular time-shift surrogate analysis. It generates per-mouse,
per-day thresholds, where each day's threshold is computed using only that day's data.

Differences from reactivation_surrogates.py:
- Computes separate threshold for EACH day using only that day's data
- Output CSV includes 'day' column with one row per mouse-day combination
- Each day (−2, −1, 0, +1, +2) gets its own threshold based on its own no-stim trials

Approach:
1. For each mouse and each day, compute a day-specific threshold
2. Create whisker response template from that day's mapping trials
3. Load no-stim trial data from that day only
4. Generate N surrogate datasets by circular time-shifting each cell independently
5. Compute template correlation for each surrogate
6. Extract both percentile (pointwise threshold) and maximum (FWER) from each
7. Average across surrogates to get final thresholds with confidence intervals
8. Each day gets its own threshold for event detection on that day

Output:
- CSV file with one threshold per mouse-day combination (includes day column)
- Multi-page PDFs showing surrogate distributions for each mouse (one page per day)
- Summary plots comparing thresholds across mice, days, and reward groups

Note: This allows thresholds to adapt to day-specific changes in neural activity patterns.
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
import warnings

sys.path.append(r'/home/aprenard/repos/NWB_analysis')
sys.path.append(r'/home/aprenard/repos/fast-learning')
import src.utils.utils_imaging as utils_imaging
import src.utils.utils_io as io
from src.utils.utils_plot import *
from src.core_analysis.reactivations.reactivation import (
    create_whisker_template,
    compute_template_correlation
)

# =============================================================================
# PARAMETERS
# =============================================================================

sampling_rate = 30  # Hz
days = [-2, -1, 0, 1, 2]
days_str = ['-2', '-1', '0', '+1', '+2']
n_map_trials = 40  # Number of mapping t/home/aprenard/repos/fast-learning/src/core_analysis/reactivations/reactivation_surrogates_per_day.pyrials for template

# Template parameters
threshold_dff = None  # 5% dF/F threshold for template cells (use None for all cells)

# Surrogate parameters
n_surrogates = 1000  # Number of surrogate iterations
min_shift_frames = 0  # Minimum shift: 1 second at 30Hz
percentiles_to_compute = [95, 99, 99.9]  # Percentiles for pointwise thresholds (computed in one go!)
np.random.seed(42)  # For reproducibility

# Parallel processing
n_jobs = 35

# Visualization
sns.set_theme(context='paper', style='ticks', palette='deep', font='sans-serif', font_scale=1)

# Load database
_, _, all_mice, db = io.select_sessions_from_db(
    io.db_path,
    io.nwb_dir,
    two_p_imaging='yes'
)

# Separate mice by reward group
r_plus_mice = []
r_minus_mice = []
for mouse in all_mice:
    try:
        reward_group = io.get_mouse_reward_group_from_db(io.db_path, mouse, db=db)
        if reward_group == 'R+':
            r_plus_mice.append(mouse)
        elif reward_group == 'R-':
            r_minus_mice.append(mouse)
    except:
        continue

# # Testing
# r_plus_mice = ['AR127']
# r_minus_mice = []

print(f"Found {len(r_plus_mice)} R+ mice: {r_plus_mice}")
print(f"Found {len(r_minus_mice)} R- mice: {r_minus_mice}")


# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def create_surrogate_by_circular_shift(data, min_shift_frames=0):
    """
    Create one surrogate by independently shifting each cell in time.

    Uses circular (roll) shifts to preserve autocorrelation structure while
    breaking cross-neuron correlations.

    Parameters
    ----------
    data : np.ndarray
        (n_cells, n_frames) neural activity
    min_shift_frames : int
        Minimum shift amount (default: 30 = 1 sec at 30Hz)

    Returns
    -------
    surrogate : np.ndarray
        (n_cells, n_frames) time-shifted data
    """
    n_cells, n_frames = data.shape
    surrogate = np.zeros_like(data)

    for icell in range(n_cells):
        # Random shift in range [min_shift_frames, n_frames)
        shift_frames = np.random.randint(min_shift_frames, n_frames)
        surrogate[icell, :] = np.roll(data[icell, :], shift_frames)

    return surrogate


def compute_surrogate_thresholds(data, template, n_surrogates=10000, min_shift=0,
                                 percentiles=[95, 99, 99.9], verbose=True):
    """
    Compute surrogate-based thresholds via circular time shifts for MULTIPLE percentiles.

    Generates n_surrogates time-shifted versions of the data, computes template
    correlation for each, and extracts statistics to define thresholds. Efficiently
    computes multiple percentiles from the same set of surrogate correlations.

    Parameters
    ----------
    data : np.ndarray
        (n_cells, n_frames) neural activity
    template : np.ndarray
        (n_cells,) response template
    n_surrogates : int
        Number of surrogate iterations (default: 1000)
    min_shift : int
        Minimum shift in frames (default: 30)
    percentiles : list of float
        Percentiles for pointwise thresholds (default: [95, 99, 99.9])
    verbose : bool
        Print progress

    Returns
    -------
    results : dict
        Dictionary keyed by percentile value, where each value is a dict with:
        {
            percentile_value: {
                'threshold_percentile_mean': float,
                'threshold_percentile_ci': (lower, upper),
                'threshold_max_mean': float,
                'threshold_max_ci': (lower, upper),
                'surrogate_percentiles': np.ndarray of percentiles,
                'surrogate_maxs': np.ndarray of maxima,
                'observed_percentile': float,
                'observed_max': float,
                'p_value_percentile': float (percentile of observed in null),
                'p_value_max': float,
                'percentile_value': float (the percentile used)
            }
        }
    """
    n_cells, n_frames = data.shape

    if verbose:
        print(f"    Computing {n_surrogates} surrogates for percentiles {percentiles}...")
        print(f"    Data shape: {n_cells} cells × {n_frames} frames")

    # First compute observed statistics for all percentiles
    observed_corr = compute_template_correlation(data, template)
    observed_percentiles = {p: np.percentile(observed_corr, p) for p in percentiles}
    observed_max = np.max(observed_corr)

    if verbose:
        for p in percentiles:
            print(f"    Observed: {p}th percentile = {observed_percentiles[p]:.4f}")
        print(f"    Observed max = {observed_max:.4f}")

    # Initialize storage for each percentile
    surrogate_percentiles = {p: np.zeros(n_surrogates) for p in percentiles}
    surrogate_maxs = np.zeros(n_surrogates)

    # Generate surrogates and compute statistics for ALL percentiles at once
    for i in range(n_surrogates):
        if verbose and (i+1) % 100 == 0:
            print(f"      Surrogate {i+1}/{n_surrogates}")

        # Generate surrogate
        surrogate_data = create_surrogate_by_circular_shift(data, min_shift)

        # Compute correlation with template (FAST with vectorized version!)
        surrogate_corr = compute_template_correlation(surrogate_data, template)

        # Extract ALL percentiles from the same surrogate correlation
        for p in percentiles:
            surrogate_percentiles[p][i] = np.percentile(surrogate_corr, p)

        # Max is shared across all percentiles
        surrogate_maxs[i] = np.max(surrogate_corr)

    # Compute statistics separately for each percentile
    results = {}
    for p in percentiles:
        # Compute threshold means and confidence intervals
        threshold_percentile_mean = np.mean(surrogate_percentiles[p])
        threshold_percentile_ci = (np.percentile(surrogate_percentiles[p], 2.5),
                                    np.percentile(surrogate_percentiles[p], 97.5))

        threshold_max_mean = np.mean(surrogate_maxs)
        threshold_max_ci = (np.percentile(surrogate_maxs, 2.5),
                            np.percentile(surrogate_maxs, 97.5))

        # Compute p-values: where does observed fall in surrogate distribution?
        p_value_percentile = percentileofscore(surrogate_percentiles[p],
                                                observed_percentiles[p]) / 100.0
        p_value_max = percentileofscore(surrogate_maxs, observed_max) / 100.0

        if verbose:
            print(f"    Threshold ({p}th): {threshold_percentile_mean:.4f} "
                  f"[{threshold_percentile_ci[0]:.4f}, {threshold_percentile_ci[1]:.4f}]")

        # Store results for this percentile
        results[p] = {
            'threshold_percentile_mean': threshold_percentile_mean,
            'threshold_percentile_ci': threshold_percentile_ci,
            'threshold_max_mean': threshold_max_mean,
            'threshold_max_ci': threshold_max_ci,
            'surrogate_percentiles': surrogate_percentiles[p],
            'surrogate_maxs': surrogate_maxs,
            'observed_percentile': observed_percentiles[p],
            'observed_max': observed_max,
            'p_value_percentile': p_value_percentile,
            'p_value_max': p_value_max,
            'percentile_value': p
        }

    if verbose:
        print(f"    Max threshold (FWER): {threshold_max_mean:.4f} "
              f"[{threshold_max_ci[0]:.4f}, {threshold_max_ci[1]:.4f}]")

    return results


def analyze_mouse_surrogates(mouse, days=[-2, -1, 0, 1, 2], threshold_dff=0.05,
                             n_surrogates=10000, percentiles=[95, 99, 99.9], verbose=True):
    """
    Compute surrogate thresholds for one mouse across multiple days and percentiles.

    Each day gets its own threshold computed from that day's data only. Multiple
    percentiles are computed efficiently from the same set of surrogates.

    Parameters
    ----------
    mouse : str
        Mouse ID
    days : list
        Days to process (default: [-2, -1, 0, 1, 2])
    threshold_dff : float or None
        Responsiveness threshold for template cells (default: 0.05 = 5%).
        If None, all cells are used.
    n_surrogates : int
        Number of surrogate iterations (default: 10000)
    percentiles : list of float
        Percentiles for pointwise thresholds (default: [95, 99, 99.9])
    verbose : bool
        Print progress

    Returns
    -------
    results_dfs : dict
        Dictionary keyed by percentile value, each containing a DataFrame with
        threshold results for this mouse (one row per day)
    all_surrogate_data : dict
        Nested dict {percentile: {day: surrogate_results}} for plotting
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"ANALYZING MOUSE: {mouse}")
        print(f"{'='*60}")

    # Initialize separate result lists for each percentile
    results_lists = {p: [] for p in percentiles}
    all_surrogate_data = {p: {} for p in percentiles}

    for day in days:
        try:
            if verbose:
                print(f"\n  Processing Day {day}...")

            # Step 1: Create template from this day's mapping trials
            template, cells_mask = create_whisker_template(mouse, day, threshold_dff, verbose=verbose)
            n_cells_responsive = cells_mask.sum()

            if n_cells_responsive < 3 and threshold_dff is not None:
                if verbose:
                    print(f"    Warning: Only {n_cells_responsive} responsive cells, skipping...")
                continue

            # Step 2: Load no-stim trial data from this day only
            folder = os.path.join(io.solve_common_paths('processed_data'), 'mice')
            file_name = 'tensor_xarray_learning_data.nc'
            xarray_learning = utils_imaging.load_mouse_xarray(mouse, folder, file_name, substracted=True)

            # Select this day and no-stim trials
            xarray_day = xarray_learning.sel(trial=xarray_learning['day'] == day)
            nostim_trials = xarray_day.sel(trial=xarray_day['no_stim'] == 1)

            n_nostim_trials = len(nostim_trials.trial)
            if n_nostim_trials < 5:
                if verbose:
                    print(f"    Warning: Only {n_nostim_trials} no-stim trials on day {day}")
                continue

            # Reshape to 2D (concatenate trials for this day)
            n_cells, n_trials, n_timepoints = nostim_trials.shape
            data = nostim_trials.values.reshape(n_cells, -1)
            data = np.nan_to_num(data, nan=0.0)
            n_frames = data.shape[1]

            if n_frames < 60:  # Less than 2 seconds
                if verbose:
                    print(f"    Warning: Session too short ({n_frames} frames), skipping...")
                continue

            if verbose:
                print(f"    Data: {n_trials} trials × {n_timepoints} frames = {n_frames} total frames")

            # Step 3: Compute surrogate thresholds for ALL percentiles at once
            surrogate_results = compute_surrogate_thresholds(
                data, template, n_surrogates, min_shift_frames, percentiles=percentiles, verbose=verbose
            )

            # Store results separately for each percentile
            for p in percentiles:
                p_results = surrogate_results[p]
                results_lists[p].append({
                    'mouse_id': mouse,
                    'day': day,
                    'n_cells_responsive': n_cells_responsive,
                    'n_trials': n_trials,
                    'n_timepoints': n_timepoints,
                    'n_frames': n_frames,
                    'n_surrogates': n_surrogates,
                    'percentile_value': p,
                    'threshold_percentile_mean': p_results['threshold_percentile_mean'],
                    'threshold_percentile_ci_lower': p_results['threshold_percentile_ci'][0],
                    'threshold_percentile_ci_upper': p_results['threshold_percentile_ci'][1],
                    'threshold_max_mean': p_results['threshold_max_mean'],
                    'threshold_max_ci_lower': p_results['threshold_max_ci'][0],
                    'threshold_max_ci_upper': p_results['threshold_max_ci'][1],
                    'observed_percentile': p_results['observed_percentile'],
                    'observed_max': p_results['observed_max'],
                    'p_value_percentile': p_results['p_value_percentile'],
                    'p_value_max': p_results['p_value_max']
                })

                # Store detailed data for plotting
                all_surrogate_data[p][day] = p_results

        except Exception as e:
            if verbose:
                print(f"    Error processing day {day}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    # Check if any data was collected
    if all(len(results_lists[p]) == 0 for p in percentiles):
        if verbose:
            print(f"\n  No valid data for mouse {mouse}")
        return None, None

    # Create DataFrames for each percentile
    results_dfs = {p: pd.DataFrame(results_lists[p]) for p in percentiles}

    if verbose:
        print(f"\n  Completed mouse {mouse}: {len(results_lists[percentiles[0]])} days processed")
        print(f"  Generated results for {len(percentiles)} percentiles: {percentiles}")

    return results_dfs, all_surrogate_data


def process_single_mouse(mouse, days, threshold_dff, n_surrogates, percentiles=[95, 99, 99.9], verbose=False):
    """
    Wrapper for parallel processing.
    """
    results_dfs, surrogate_data = analyze_mouse_surrogates(
        mouse, days=days, threshold_dff=threshold_dff,
        n_surrogates=n_surrogates, percentiles=percentiles, verbose=verbose
    )
    return (mouse, results_dfs, surrogate_data)


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def plot_surrogate_distributions(mouse, surrogate_data, save_path):
    """
    Generate multi-page PDF showing surrogate distributions for each day.

    Parameters
    ----------
    mouse : str
        Mouse ID
    surrogate_data : dict
        {day: surrogate_results} from analyze_mouse_surrogates
    save_path : str
        Path to save PDF
    """
    if surrogate_data is None or len(surrogate_data) == 0:
        print(f"    Warning: No surrogate data for {mouse}, skipping plot")
        return

    with PdfPages(save_path) as pdf:
        for day in sorted(surrogate_data.keys()):
            results = surrogate_data[day]

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

            # Get percentile value from results
            percentile_val = int(results.get('percentile_value', 95))

            # Left panel: Percentile distribution
            ax1.hist(results['surrogate_percentiles'], bins=50, alpha=0.7, color='steelblue',
                    edgecolor='black', linewidth=0.5)

            # Add mean threshold line
            ax1.axvline(results['threshold_percentile_mean'], color='blue', linestyle='-',
                       linewidth=2, label=f"Mean: {results['threshold_percentile_mean']:.4f}")

            # Add confidence interval
            ax1.axvspan(results['threshold_percentile_ci'][0], results['threshold_percentile_ci'][1],
                       alpha=0.2, color='blue', label='95% CI')

            # Add observed value
            ax1.axvline(results['observed_percentile'], color='red', linestyle='--',
                       linewidth=2, label=f"Observed: {results['observed_percentile']:.4f}")

            # Statistics text
            stats_text = f"n_surrogates = {len(results['surrogate_percentiles'])}\n"
            stats_text += f"p-value = {results['p_value_percentile']:.3f}"
            ax1.text(0.97, 0.97, stats_text, transform=ax1.transAxes,
                    fontsize=10, verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))

            ax1.set_xlabel(f'{percentile_val}th Percentile Correlation', fontweight='bold')
            ax1.set_ylabel('Count', fontweight='bold')
            ax1.set_title(f'{percentile_val}th Percentile Threshold (Pointwise)', fontweight='bold')
            ax1.legend(loc='upper left')
            ax1.grid(True, alpha=0.3, axis='y')

            # Right panel: Max correlation distribution
            ax2.hist(results['surrogate_maxs'], bins=50, alpha=0.7, color='darkorange',
                    edgecolor='black', linewidth=0.5)

            # Add mean threshold line
            ax2.axvline(results['threshold_max_mean'], color='orange', linestyle='-',
                       linewidth=2, label=f"Mean: {results['threshold_max_mean']:.4f}")

            # Add confidence interval
            ax2.axvspan(results['threshold_max_ci'][0], results['threshold_max_ci'][1],
                       alpha=0.2, color='orange', label='95% CI')

            # Add observed value
            ax2.axvline(results['observed_max'], color='red', linestyle='--',
                       linewidth=2, label=f"Observed: {results['observed_max']:.4f}")

            # Statistics text
            stats_text = f"n_surrogates = {len(results['surrogate_maxs'])}\n"
            stats_text += f"p-value = {results['p_value_max']:.3f}"
            ax2.text(0.97, 0.97, stats_text, transform=ax2.transAxes,
                    fontsize=10, verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))

            ax2.set_xlabel('Maximum Correlation', fontweight='bold')
            ax2.set_ylabel('Count', fontweight='bold')
            ax2.set_title(f'Maximum Threshold (FWER)', fontweight='bold')
            ax2.legend(loc='upper left')
            ax2.grid(True, alpha=0.3, axis='y')

            # Overall title
            fig.suptitle(f'{mouse} - Day {day} Surrogate Analysis',
                        fontsize=14, fontweight='bold')
            plt.tight_layout()

            pdf.savefig(fig, bbox_inches='tight')
            plt.close()

    print(f"    Saved PDF: {save_path}")


def plot_threshold_summary_across_mice(all_results_df, save_path):
    """
    Summary plots comparing per-day thresholds across mice.

    Parameters
    ----------
    all_results_df : pd.DataFrame
        Combined results from all mice (one row per mouse-day)
    save_path : str
        Path to save PDF
    """
    # Add reward group information
    all_results_df['reward_group'] = all_results_df['mouse_id'].apply(
        lambda m: io.get_mouse_reward_group_from_db(io.db_path, m, db=db)
    )

    # Get percentile value from data
    percentile_val = int(all_results_df['percentile_value'].iloc[0]) if len(all_results_df) > 0 else 95

    with PdfPages(save_path) as pdf:
        # Page 1: Threshold distributions by day and reward group
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        # Percentile thresholds across days - line plot
        ax = axes[0, 0]
        for reward_group, color in zip(['R+', 'R-'], ['steelblue', 'coral']):
            group_data = all_results_df[all_results_df['reward_group'] == reward_group]
            day_means = group_data.groupby('day')['threshold_percentile_mean'].mean()
            day_sems = group_data.groupby('day')['threshold_percentile_mean'].sem()
            ax.errorbar(day_means.index, day_means.values, yerr=day_sems.values,
                       marker='o', linewidth=2, markersize=8, capsize=5,
                       label=f'{reward_group} (n={group_data["mouse_id"].nunique()})',
                       color=color)
        ax.set_xlabel('Day', fontweight='bold')
        ax.set_ylabel(f'{percentile_val}th Percentile Threshold', fontweight='bold')
        ax.set_title(f'{percentile_val}th Percentile Threshold Across Days', fontweight='bold')
        ax.set_xticks(days)
        ax.set_xticklabels(days_str)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Percentile thresholds - box plot by day
        ax = axes[0, 1]
        data_to_plot = [all_results_df[all_results_df['day'] == d]['threshold_percentile_mean'].values
                       for d in days]
        bp = ax.boxplot(data_to_plot, labels=days_str, patch_artist=True, showmeans=True, widths=0.5)
        for patch in bp['boxes']:
            patch.set_facecolor('steelblue')
            patch.set_alpha(0.7)
        ax.set_xlabel('Day', fontweight='bold')
        ax.set_ylabel(f'{percentile_val}th Percentile Threshold', fontweight='bold')
        ax.set_title(f'{percentile_val}th Percentile by Day (All Mice)', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        # Max thresholds across days - line plot
        ax = axes[1, 0]
        for reward_group, color in zip(['R+', 'R-'], ['darkorange', 'orchid']):
            group_data = all_results_df[all_results_df['reward_group'] == reward_group]
            day_means = group_data.groupby('day')['threshold_max_mean'].mean()
            day_sems = group_data.groupby('day')['threshold_max_mean'].sem()
            ax.errorbar(day_means.index, day_means.values, yerr=day_sems.values,
                       marker='o', linewidth=2, markersize=8, capsize=5,
                       label=f'{reward_group} (n={group_data["mouse_id"].nunique()})',
                       color=color)
        ax.set_xlabel('Day', fontweight='bold')
        ax.set_ylabel('Maximum Threshold (FWER)', fontweight='bold')
        ax.set_title('Maximum (FWER) Threshold Across Days', fontweight='bold')
        ax.set_xticks(days)
        ax.set_xticklabels(days_str)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Max thresholds - box plot by day
        ax = axes[1, 1]
        data_to_plot = [all_results_df[all_results_df['day'] == d]['threshold_max_mean'].values
                       for d in days]
        bp = ax.boxplot(data_to_plot, labels=days_str, patch_artist=True, showmeans=True, widths=0.5)
        for patch in bp['boxes']:
            patch.set_facecolor('darkorange')
            patch.set_alpha(0.7)
        ax.set_xlabel('Day', fontweight='bold')
        ax.set_ylabel('Maximum Threshold (FWER)', fontweight='bold')
        ax.set_title('Maximum (FWER) by Day (All Mice)', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        fig.suptitle('Per-Day Threshold Summary Across All Mice',
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

        # Page 2: Reward group comparison
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        for i, (reward_group, color_p, color_m) in enumerate(zip(['R+', 'R-'],
                                                                  [('steelblue', 'darkorange'),
                                                                   ('coral', 'orchid')])):
            group_data = all_results_df[all_results_df['reward_group'] == reward_group]

            # Percentile thresholds
            ax = axes[i, 0]
            data_to_plot = [group_data[group_data['day'] == d]['threshold_percentile_mean'].values
                           for d in days]
            bp = ax.boxplot(data_to_plot, labels=days_str, patch_artist=True, showmeans=True, widths=0.5)
            for patch in bp['boxes']:
                patch.set_facecolor(color_p[0])
                patch.set_alpha(0.7)
            ax.set_xlabel('Day', fontweight='bold')
            ax.set_ylabel(f'{percentile_val}th Percentile Threshold', fontweight='bold')
            ax.set_title(f'{reward_group}: {percentile_val}th Percentile by Day (n={group_data["mouse_id"].nunique()} mice)',
                        fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')

            # Max thresholds
            ax = axes[i, 1]
            data_to_plot = [group_data[group_data['day'] == d]['threshold_max_mean'].values
                           for d in days]
            bp = ax.boxplot(data_to_plot, labels=days_str, patch_artist=True, showmeans=True, widths=0.5)
            for patch in bp['boxes']:
                patch.set_facecolor(color_p[1])
                patch.set_alpha(0.7)
            ax.set_xlabel('Day', fontweight='bold')
            ax.set_ylabel('Maximum Threshold (FWER)', fontweight='bold')
            ax.set_title(f'{reward_group}: Maximum (FWER) by Day (n={group_data["mouse_id"].nunique()} mice)',
                        fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

    print(f"  Saved summary PDF: {save_path}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("REACTIVATION SURROGATE ANALYSIS - PER-DAY THRESHOLDS")
    print("="*60)
    print(f"\nParameters:")
    print(f"  Responsiveness threshold: {threshold_dff*100 if threshold_dff is not None else 'None (all cells)'}% dF/F")
    print(f"  Number of surrogates: {n_surrogates}")
    print(f"  Percentile thresholds: {percentiles_to_compute} (computed simultaneously!)")
    print(f"  Minimum time shift: {min_shift_frames} frames ({min_shift_frames/sampling_rate:.1f} sec)")
    print(f"  Days: {days} (separate threshold per day)")
    print(f"  Parallel jobs: {n_jobs}")

    # Create output directory
    output_dir = '/mnt/lsens-analysis/Anthony_Renard/analysis_output/fast-learning/reactivation_surrogates_per_day'
    output_dir = io.adjust_path_to_host(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nResults will be saved to: {output_dir}")

    # Process all mice in parallel
    print("\n" + "="*60)
    print("PROCESSING ALL MICE")
    print("="*60)

    all_mice_to_process = r_plus_mice + r_minus_mice
    print(f"Processing {len(all_mice_to_process)} mice in parallel (per-day thresholds)...")
    print(f"Computing {len(percentiles_to_compute)} percentiles from same surrogates for efficiency!")

    results_list = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(process_single_mouse)(mouse, days, threshold_dff, n_surrogates,
                                     percentiles=percentiles_to_compute, verbose=False)
        for mouse in all_mice_to_process
    )

    # Collect results separately for each percentile
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

    # Create DataFrames for each percentile
    all_results_dfs = {p: pd.concat(all_results[p], ignore_index=True) for p in percentiles_to_compute}

    # Print collection summary
    first_percentile = percentiles_to_compute[0]
    print(f"\nCollected results: {len(all_results_dfs[first_percentile])} mouse-day combinations")
    print(f"  {all_results_dfs[first_percentile]['mouse_id'].nunique()} unique mice")
    print(f"  {all_results_dfs[first_percentile].groupby('mouse_id')['day'].count().mean():.1f} days per mouse (average)")
    print(f"  {len(percentiles_to_compute)} percentiles computed: {percentiles_to_compute}")

    # Save separate CSV files for each percentile
    print("\nSaving CSV files...")
    for p in percentiles_to_compute:
        # Create filename: p95 for 95%, p99 for 99%, p999 for 99.9%
        p_str = f"p{int(p)}" if p == int(p) else f"p{int(p*10)}"
        csv_path = os.path.join(output_dir, f'surrogate_thresholds_per_day_{p_str}.csv')
        all_results_dfs[p].to_csv(csv_path, index=False)
        print(f"  Saved {p}th percentile thresholds to: {csv_path}")

    # Generate per-mouse PDFs for each percentile
    print("\n" + "="*60)
    print("GENERATING PER-MOUSE VISUALIZATIONS")
    print("="*60)

    for p in percentiles_to_compute:
        print(f"\n  Generating visualizations for {p}th percentile...")
        p_str = f"p{int(p)}" if p == int(p) else f"p{int(p*10)}"
        pdf_dir = os.path.join(output_dir, f'per_mouse_pdfs_{p_str}')
        os.makedirs(pdf_dir, exist_ok=True)

        for mouse, surrogate_data in all_surrogate_data[p].items():
            if surrogate_data is not None:
                pdf_path = os.path.join(pdf_dir, f'{mouse}_surrogate_analysis_per_day_{p_str}.pdf')
                print(f"    Generating PDF for {mouse}...")
                plot_surrogate_distributions(mouse, surrogate_data, pdf_path)

    # Generate summary plots for each percentile
    print("\n" + "="*60)
    print("GENERATING SUMMARY VISUALIZATIONS")
    print("="*60)

    for p in percentiles_to_compute:
        p_str = f"p{int(p)}" if p == int(p) else f"p{int(p*10)}"
        summary_pdf_path = os.path.join(output_dir, f'surrogate_threshold_summary_per_day_{p_str}.pdf')
        print(f"  Generating summary for {p}th percentile...")
        plot_threshold_summary_across_mice(all_results_dfs[p], summary_pdf_path)

    # Print summary statistics for each percentile
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)

    for p in percentiles_to_compute:
        print(f"\n{'='*60}")
        print(f"PERCENTILE: {p}th")
        print(f"{'='*60}")

        all_results_df = all_results_dfs[p]

        for reward_group in ['R+', 'R-']:
            group_mice = r_plus_mice if reward_group == 'R+' else r_minus_mice
            group_data = all_results_df[all_results_df['mouse_id'].isin(group_mice)]

            print(f"\n{reward_group} Group (n={group_data['mouse_id'].nunique()} mice):")
            for day in days:
                day_data = group_data[group_data['day'] == day]
                if len(day_data) > 0:
                    print(f"\n  Day {day}:")
                    print(f"    {p}th percentile: {day_data['threshold_percentile_mean'].mean():.4f} ± {day_data['threshold_percentile_mean'].std():.4f}")
                    print(f"    Max (FWER):       {day_data['threshold_max_mean'].mean():.4f} ± {day_data['threshold_max_mean'].std():.4f}")

    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"\nProcessed {len(all_mice_to_process)} mice")
    print(f"Total mouse-day combinations per percentile: {len(all_results_dfs[percentiles_to_compute[0]])}")
    print(f"Percentiles computed: {percentiles_to_compute}")
    print(f"Results saved to: {output_dir}")
    print(f"\nKey points:")
    print(f"  - Each mouse-day combination has its own threshold")
    print(f"  - Each day's threshold is computed using only that day's data")
    print(f"  - Multiple percentiles ({percentiles_to_compute}) computed from same surrogates!")
    print(f"  - Thresholds can adapt to day-specific changes in neural activity")
    print(f"\nOutput files:")
    for p in percentiles_to_compute:
        p_str = f"p{int(p)}" if p == int(p) else f"p{int(p*10)}"
        print(f"  - CSV: surrogate_thresholds_per_day_{p_str}.csv")
        print(f"  - PDFs: per_mouse_pdfs_{p_str}/ and surrogate_threshold_summary_per_day_{p_str}.pdf")
    print(f"\nNext steps:")
    print(f"1. Review per-mouse PDFs for each percentile")
    print(f"2. Examine summary plots for each percentile")
    print(f"3. Choose which percentile to use based on your conservative/liberal preference")
    print(f"4. Use chosen threshold CSV with reactivation.py (threshold_mode='day')")
    print(f"4. Use reactivation.py and reactivation_lmi_prediction.py with threshold_mode='day'")
