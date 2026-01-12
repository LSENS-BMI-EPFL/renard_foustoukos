"""
Single-cell plasticity analysis during day 0 whisker learning.

This script fits sigmoid models to trial-by-trial responses of LMI-significant cells
and identifies cells showing online plasticity. Cells are ranked by response amplitude
and filtered by statistical significance (p < 0.05 via likelihood ratio test vs flat model).

Output: CSV with amplitude metrics, distribution plots, and 4 stratified PDF reports
        showing top 50 significant cells per group (R+/-, LMI+/-) sorted by amplitude.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import chi2, mannwhitneyu, kruskal
from scipy.optimize import curve_fit
from scipy.special import expit
from joblib import Parallel, delayed
import warnings

sys.path.append(r'/home/aprenard/repos/NWB_analysis')
sys.path.append(r'/home/aprenard/repos/fast-learning')

import src.utils.utils_imaging as utils_imaging
import src.utils.utils_io as io
from src.utils.utils_plot import *

# =============================================================================
# PARAMETERS
# =============================================================================

# Analysis parameters
RUN_FITTING = False
GENERATE_PDFS = False
SAMPLING_RATE = 30  # Hz
RESPONSE_WIN = (0, 0.300)  # 0-300ms response window
RESPONSE_TYPE = 'mean'  # 'mean' or 'peak' within response window
AMPLITUDE_TYPE = 'absolute'  # 'absolute' or 'relative' - how to compute amplitude
MIN_TRIALS = 20  # Minimum whisker trials required for fitting
ALPHA = 0.05  # Significance threshold
N_CORES = 35  # Number of cores for parallel processing (one per mouse)
DAYS_LEARNING = [-2, -1, 0, 1, 2]  # Days for mapping PSTH visualization

# LMI thresholds for cell selection
LMI_POSITIVE_THRESHOLD = 0.975  # Top 2.5% LMI cells
LMI_NEGATIVE_THRESHOLD = 0.025  # Bottom 2.5% LMI cells

# Output directory
OUTPUT_DIR = io.adjust_path_to_host(
    '/mnt/lsens-analysis/Anthony_Renard/analysis_output/fast-learning/day0_learning/plasticity'
)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================================================================
# DATA LOADING
# =============================================================================

def load_day0_data(mouse_id, response_type='mean', response_win=(0, 0.300)):
    """
    Load and prepare day 0 whisker trial data for a single mouse.

    Parameters
    ----------
    mouse_id : str
        Mouse identifier
    response_type : str
        'mean' or 'peak' - how to compute response from window
    response_win : tuple
        (start, end) time window in seconds

    Returns
    -------
    responses : np.ndarray
        Shape (n_cells, n_trials) - response values
    trial_indices : np.ndarray
        Shape (n_trials,) - trial_w values (whisker trial numbers)
    roi_ids : np.ndarray
        Shape (n_cells,) - ROI identifiers
    """
    # Load xarray
    folder = os.path.join(io.processed_dir, 'mice')
    file_name = 'tensor_xarray_learning_data.nc'
    xarray = utils_imaging.load_mouse_xarray(mouse_id, folder, file_name, substracted=False)

    # Select day 0 whisker trials
    xarray = xarray.sel(trial=xarray['day'] == 0)
    xarray = xarray.sel(trial=xarray['whisker_stim'] == 1)

    # Extract trial indices
    trial_indices = xarray['trial_w'].values

    # Compute response per trial
    xarray_win = xarray.sel(time=slice(*response_win))

    if response_type == 'mean':
        responses_xarr = xarray_win.mean(dim='time')
    elif response_type == 'peak':
        responses_xarr = xarray_win.max(dim='time')
    else:
        raise ValueError(f"response_type must be 'mean' or 'peak', got {response_type}")

    # Extract ROI identifiers
    roi_ids = xarray['roi'].values

    # Convert to numpy array: (n_cells, n_trials)
    responses = responses_xarr.values

    return responses, trial_indices, roi_ids


# =============================================================================
# MODEL FITTING FUNCTIONS
# =============================================================================

def sigmoid_4pl(x, baseline, max_val, inflection, slope_param):
    """
    4-parameter logistic (sigmoid) function using scipy's expit.

    Parameters
    ----------
    x : array-like
        Independent variable (trial numbers)
    baseline : float
        Lower asymptote (response level at early trials, x → -∞)
    max_val : float
        Upper asymptote (response level at late trials, x → +∞)
    inflection : float
        Inflection point (trial number where change is steepest)
    slope_param : float
        Slope parameter (controls steepness)

    Returns
    -------
    y : array-like
        Sigmoid function output
    """
    # Use scipy's expit (logistic sigmoid) for numerical stability
    # expit(x) = 1 / (1 + exp(-x))
    return baseline + (max_val - baseline) * expit((x - inflection) / slope_param)


def fit_sigmoid_model(x, y):
    """
    Fit 4-parameter logistic sigmoid model to data.

    Parameters
    ----------
    x : np.ndarray
        Trial indices (1D)
    y : np.ndarray
        Response values (1D)

    Returns
    -------
    results : dict or None
        Returns None if fitting fails, otherwise dict with:
        {
            'baseline': float,
            'max_val': float,
            'inflection': float,
            'slope_param': float,
            'predictions': np.ndarray,
            'residuals': np.ndarray,
            'n_params': int,  # Always 4
            'fit_success': bool
        }
    """
    # Remove NaN values
    mask = ~np.isnan(y)
    x_clean, y_clean = x[mask], y[mask]

    if len(x_clean) < 5:  # Need more points for 4 parameters
        return None

    # Compute initial parameter estimates
    y_min, y_max = np.nanmin(y_clean), np.nanmax(y_clean)
    y_range = y_max - y_min

    # Initial guesses
    p0 = [
        y_min,  # baseline
        y_max,  # max_val
        np.median(x_clean),  # inflection (middle trial)
        (x_clean[-1] - x_clean[0]) / 4  # slope_param (quarter of trial range)
    ]

    # Bounds to ensure numerical stability
    bounds = (
        [y_min - y_range, y_min - y_range, x_clean[0], 0.1],  # Lower bounds
        [y_max + y_range, y_max + y_range, x_clean[-1], (x_clean[-1] - x_clean[0]) * 2]  # Upper bounds
    )

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            popt, pcov = curve_fit(
                sigmoid_4pl, x_clean, y_clean,
                p0=p0, bounds=bounds, maxfev=5000
            )

        predictions = sigmoid_4pl(x_clean, *popt)
        residuals = y_clean - predictions

        # Compute robust amplitude by evaluating fitted curve over trial range
        # This is more robust than |max_val - baseline| which can be affected by outlier parameters
        trial_range = np.linspace(x_clean[0], x_clean[-1], 100)
        predictions_range = sigmoid_4pl(trial_range, *popt)

        # Signed amplitude: max_val - baseline (can be positive or negative)
        # Positive: cell increases response during learning
        # Negative: cell decreases response during learning
        baseline_val = popt[0]
        max_val = popt[1]
        amplitude_absolute = max_val - baseline_val

        # Relative amplitude: normalized by baseline (avoid division by zero)
        if abs(baseline_val) > .1:  # Avoid division by very small values
            amplitude_relative = amplitude_absolute / abs(baseline_val)
        else:
            amplitude_relative = amplitude_absolute  # Fallback to absolute if baseline ~0

        # Compute parameter standard errors from covariance matrix
        try:
            perr = np.sqrt(np.diag(pcov))
        except:
            perr = None

        return {
            'baseline': popt[0],
            'max_val': popt[1],
            'inflection': popt[2],
            'slope_param': popt[3],
            'predictions': predictions,
            'residuals': residuals,
            'n_params': 4,
            'fit_success': True,
            'x_clean': x_clean,
            'y_clean': y_clean,
            'amplitude_absolute': amplitude_absolute,
            'amplitude_relative': amplitude_relative,
            'pcov': pcov,
            'perr': perr
        }

    except (RuntimeError, ValueError):
        # Fitting failed
        return None


# =============================================================================
# STATISTICAL TESTS
# =============================================================================

def compute_pseudo_r_squared(residuals, y):
    """
    Compute pseudo-R² for sigmoid model.

    Pseudo-R² = 1 - (RSS / TSS)
    where TSS = total sum of squares

    Parameters
    ----------
    residuals : np.ndarray
        Model residuals
    y : np.ndarray
        Observed values

    Returns
    -------
    pseudo_r2 : float
    """
    tss = np.sum((y - np.mean(y)) ** 2)
    rss = np.sum(residuals ** 2)

    if tss == 0:
        return 0.0

    return 1 - (rss / tss)


def fit_flat_model(y):
    """
    Fit flat (constant mean) model for null hypothesis comparison.

    Parameters
    ----------
    y : np.ndarray
        Response values (1D)

    Returns
    -------
    results : dict
    """
    mask = ~np.isnan(y)
    y_clean = y[mask]

    mean_val = np.nanmean(y_clean)
    predictions = np.full_like(y_clean, mean_val)
    residuals = y_clean - predictions

    return {
        'mean': mean_val,
        'predictions': predictions,
        'residuals': residuals,
        'n_params': 1,
        'y_clean': y_clean
    }


def likelihood_ratio_test(residuals_null, residuals_alt, df_diff):
    """
    Perform likelihood ratio test between nested models.

    Parameters
    ----------
    residuals_null : np.ndarray
        Residuals from null (flat) model
    residuals_alt : np.ndarray
        Residuals from alternative (sigmoid) model
    df_diff : int
        Difference in degrees of freedom

    Returns
    -------
    p_value : float
    """
    n = len(residuals_null)
    rss_null = np.sum(residuals_null ** 2)
    rss_alt = np.sum(residuals_alt ** 2)

    if rss_alt <= 0:
        return 0.0

    lr_stat = n * np.log(rss_null / rss_alt)
    p_value = 1 - chi2.cdf(lr_stat, df_diff)

    return p_value


# =============================================================================
# SINGLE-CELL ANALYSIS
# =============================================================================

def analyze_single_cell(x, y, min_trials=20, amplitude_type='absolute'):
    """
    Fit sigmoid model to single cell's trial-by-trial responses.

    Parameters
    ----------
    x : np.ndarray
        Trial indices (1D)
    y : np.ndarray
        Response values (1D)
    min_trials : int
        Minimum number of trials required
    amplitude_type : str
        'absolute' or 'relative' - how to compute amplitude

    Returns
    -------
    results : dict or None
        Returns None if insufficient data or fitting failed
        Otherwise returns dict with sigmoid fit results
    """
    # Check data quality
    mask = ~np.isnan(y)
    n_valid = np.sum(mask)

    if n_valid < min_trials:
        return None

    # Fit sigmoid model
    sigmoid_fit = fit_sigmoid_model(x, y)
    if sigmoid_fit is None or not sigmoid_fit.get('fit_success', False):
        return None

    # Fit flat model for significance test
    flat_fit = fit_flat_model(y)

    # Compute pseudo-R²
    pseudo_r2 = compute_pseudo_r_squared(sigmoid_fit['residuals'], sigmoid_fit['y_clean'])

    # Perform likelihood ratio test
    p_value = likelihood_ratio_test(
        flat_fit['residuals'],
        sigmoid_fit['residuals'],
        df_diff=sigmoid_fit['n_params'] - flat_fit['n_params']
    )

    # Select amplitude type (absolute or relative)
    if amplitude_type == 'relative':
        amplitude = sigmoid_fit['amplitude_relative']
    else:  # Default to 'absolute'
        amplitude = sigmoid_fit['amplitude_absolute']

    return {
        'p_value': p_value,
        'pseudo_r2': pseudo_r2,
        'amplitude': amplitude,
        'amplitude_absolute': sigmoid_fit['amplitude_absolute'],
        'amplitude_relative': sigmoid_fit['amplitude_relative'],
        'baseline': sigmoid_fit['baseline'],
        'max_val': sigmoid_fit['max_val'],
        'inflection': sigmoid_fit['inflection'],
        'slope_param': sigmoid_fit['slope_param'],
        'predictions': sigmoid_fit['predictions'],
        'residuals': sigmoid_fit['residuals'],
        'x_clean': sigmoid_fit['x_clean'],
        'y_clean': sigmoid_fit['y_clean']
    }


# =============================================================================
# MOUSE-LEVEL PROCESSING
# =============================================================================

def process_mouse(mouse_id, response_type='mean', response_win=(0, 0.300),
                  min_trials=20, amplitude_type='absolute'):
    """
    Process all cells for a single mouse using sigmoid model.

    Parameters
    ----------
    mouse_id : str
        Mouse identifier
    response_type : str
        'mean' or 'peak'
    response_win : tuple
        (start, end) time window
    min_trials : int
        Minimum trials required for fitting
    amplitude_type : str
        'absolute' or 'relative' - how to compute amplitude

    Returns
    -------
    results_df : pd.DataFrame
        Columns: mouse_id, roi, reward_group, p_value, pseudo_r2, amplitude,
                 amplitude_absolute, amplitude_relative, baseline, max_val,
                 inflection, slope_param, n_trials
    """
    print(f"Processing {mouse_id}...")

    # Load data
    responses, trial_indices, roi_ids = load_day0_data(mouse_id, response_type, response_win)

    # Get reward group
    reward_group = io.get_mouse_reward_group_from_db(io.db_path, mouse_id)

    # Process each cell
    results = []
    n_cells = responses.shape[0]

    for i, roi in enumerate(roi_ids):
        y = responses[i, :]
        x = trial_indices

        cell_results = analyze_single_cell(x, y, min_trials, amplitude_type)

        if cell_results is not None:
            results.append({
                'mouse_id': mouse_id,
                'roi': roi,
                'reward_group': reward_group,
                'p_value': cell_results['p_value'],
                'pseudo_r2': cell_results['pseudo_r2'],
                'amplitude': cell_results['amplitude'],
                'amplitude_absolute': cell_results['amplitude_absolute'],
                'amplitude_relative': cell_results['amplitude_relative'],
                'baseline': cell_results['baseline'],
                'max_val': cell_results['max_val'],
                'inflection': cell_results['inflection'],
                'slope_param': cell_results['slope_param'],
                'n_trials': len(x)
            })

    results_df = pd.DataFrame(results)
    print(f"  Processed {len(results_df)} cells for {mouse_id}")

    return results_df


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def plot_distributions(results_df, output_dir):
    """
    Plot distributions of key plasticity metrics by reward group.
    Uses only cells with significant sigmoid fits (p_value < ALPHA).

    Parameters
    ----------
    results_df : pd.DataFrame
        Results dataframe with all cells
    output_dir : str
        Output directory for plots
    """
    sns.set_theme(context='paper', style='ticks', palette='deep', font='sans-serif', font_scale=1.5)

    # Filter for significant cells only
    results_df = results_df[results_df['p_value'] < ALPHA].copy()

    # Convert amplitude to percentage and clip to ±200% ΔF/F
    results_df['amplitude_pct'] = results_df['amplitude'].clip(lower=-2, upper=2) * 100

    # Compute shared x-axis ranges for inflection plots
    inflection_min = results_df['inflection'].min()
    inflection_max = results_df['inflection'].max()
    inflection_range = (inflection_min - 5, inflection_max + 5)

    # Compute shared x-axis range for inflection_relative plots (all cells with learning trial)
    df_with_learning = results_df.dropna(subset=['learning_trial'])
    if len(df_with_learning) > 0:
        inflection_rel_min = df_with_learning['inflection_relative'].min()
        inflection_rel_max = df_with_learning['inflection_relative'].max()
        inflection_rel_range = (inflection_rel_min - 5, inflection_rel_max + 5)
    else:
        inflection_rel_range = (-50, 50)  # Default range if no data

    # Distributions by reward group
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), dpi=150)

    for idx, group in enumerate(['R+', 'R-']):
        df_group = results_df[results_df['reward_group'] == group]
        color = reward_palette[1] if group == 'R+' else reward_palette[0]

        # Amplitude distribution - proportions with -200 to +200% range
        ax = axes[idx, 0]
        sns.histplot(data=df_group, x='amplitude_pct', ax=ax, color=color,
                     binwidth=10, binrange=(-200, 200), stat='proportion')
        ax.set_xlabel('Amplitude (% ΔF/F)')
        ax.set_ylabel('Proportion')
        ax.set_title(f'{group}: Amplitude (n={len(df_group)})')
        ax.set_xlim([-200, 200])

        # Inflection points - shared x-axis
        ax = axes[idx, 1]
        sns.histplot(data=df_group, x='inflection', binwidth=10, ax=ax, color=color,
                     stat='proportion')
        ax.set_xlabel('Inflection Point (trial number)')
        ax.set_ylabel('Proportion')
        ax.set_title(f'{group}: Inflection Points (n={len(df_group)})')
        ax.set_xlim(inflection_range)

        # Inflection relative to learning trial - ALL cells with learning data, no KDE
        ax = axes[idx, 2]
        df_group_with_learning = df_group.dropna(subset=['learning_trial'])
        if len(df_group_with_learning) > 0:
            sns.histplot(
                data=df_group_with_learning, x='inflection_relative',
                color=color, ax=ax, binwidth=10, stat='proportion'
            )
            ax.axvline(0, color='black', linestyle='--', linewidth=1.5,
                      label='Behavioral learning trial')
            ax.legend(frameon=False)
        ax.set_xlabel('Cellular inflection - Learning trial (trials)')
        ax.set_ylabel('Proportion')
        ax.set_title(f'{group}: Inflection Timing (n={len(df_group_with_learning)})')
        ax.set_xlim(inflection_rel_range)

    plt.tight_layout()
    sns.despine()
    plt.savefig(os.path.join(output_dir, 'distributions_by_reward.svg'), format='svg', dpi=150)
    plt.close()

    print("  ✓ Distribution plots saved")


def plot_amplitude_distributions_by_lmi(results_df, output_dir, alpha=0.05):
    """
    Plot amplitude distributions comparing R+ vs R- for all cells.

    Creates single-panel figure with overlaid histograms showing amplitude
    distributions for R+ and R- mice, using ALL cells with significant fits
    (irrespective of LMI classification).

    Parameters
    ----------
    results_df : pd.DataFrame
        Results dataframe with all cells
    output_dir : str
        Output directory for saving figure
    alpha : float
        Significance threshold for filtering cells (default: 0.05)
    """
    sns.set_theme(context='paper', style='ticks', palette='deep', font='sans-serif', font_scale=1.5)

    # Filter for significant cells only
    sig_cells = results_df[results_df['p_value'] < alpha].copy()

    # Convert amplitude to percentage (x100) and cap at ±200% ΔF/F (±2.0 in original units)
    sig_cells['amplitude_pct'] = sig_cells['amplitude'].clip(lower=-2.0, upper=2.0) * 100

    # Split by reward group
    rplus_cells = sig_cells[sig_cells['reward_group'] == 'R+']
    rminus_cells = sig_cells[sig_cells['reward_group'] == 'R-']

    # Create single panel figure
    fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=150)

    # Histogram for R- with KDE
    sns.histplot(data=rminus_cells, x='amplitude_pct', ax=ax, color=reward_palette[0],
                 alpha=0.4, label=f'R- (n={len(rminus_cells)})', stat='density',
                 binwidth=10, binrange=(-200, 200), kde=True, line_kws={'linewidth': 2})

    # Overlaid histogram for R+ with KDE
    sns.histplot(data=rplus_cells, x='amplitude_pct', ax=ax, color=reward_palette[1],
                 alpha=0.4, label=f'R+ (n={len(rplus_cells)})', stat='density',
                 binwidth=10, binrange=(-200, 200), kde=True, line_kws={'linewidth': 2})

    ax.set_xlabel('Amplitude (%ΔF/F)', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_xlim([-200, 200])
    ax.set_title(f'Amplitude Distribution by Reward Group\n(All cells with significant fits, n={len(sig_cells)})',
                 fontsize=14, fontweight='bold')
    ax.axvline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.legend(frameon=False, title='Reward Group')

    plt.tight_layout()
    sns.despine()
    plt.savefig(os.path.join(output_dir, 'amplitude_distributions_by_reward_lmi.svg'), format='svg', dpi=150)
    plt.close()

    print("  ✓ Amplitude distribution plots saved")


def plot_lmi_groups_amplitude_barplot_general(results_df, output_dir, stat_level='mice',
                                               amplitude_filter='all', filename_suffix=''):
    """
    Create bar plot comparing amplitude across LMI groups for R+ and R- groups.

    Two panels (R+ and R-), each with three bars showing average amplitude for:
    - LMI positive cells (lmi_p >= 0.975)
    - Non-significant cells (0.025 < lmi_p < 0.975)
    - LMI negative cells (lmi_p <= 0.025)

    Uses ALL cells (no significance filtering).

    Parameters
    ----------
    results_df : pd.DataFrame
        Results dataframe with all cells
    output_dir : str
        Output directory for saving figure
    stat_level : str
        'mice' = aggregate per mouse then across mice (default)
        'cells' = compute statistics directly over cells
    amplitude_filter : str
        'all' = all cells (default)
        'positive' = only cells with amplitude > 0
        'negative' = only cells with amplitude < 0
    filename_suffix : str
        Suffix to add to filename (default: '')
    """
    print(f"\n  Computing LMI group amplitude comparison ({stat_level} level, {amplitude_filter} amplitude)...")

    # Use ALL cells (no significance filtering)
    results_df = results_df.copy()

    # Filter by amplitude sign if requested
    if amplitude_filter == 'positive':
        results_df = results_df[results_df['amplitude'] > 0]
        filter_label = 'positive amplitude only'
    elif amplitude_filter == 'negative':
        results_df = results_df[results_df['amplitude'] < 0]
        filter_label = 'negative amplitude only'
    else:
        filter_label = 'all cells'

    if len(results_df) == 0:
        print(f"  No cells found with {amplitude_filter} amplitude. Skipping.")
        return

    # Define LMI categories
    def assign_lmi_category(lmi_p):
        if lmi_p >= LMI_POSITIVE_THRESHOLD:
            return 'LMI+'
        elif lmi_p <= LMI_NEGATIVE_THRESHOLD:
            return 'LMI-'
        else:
            return 'Non-sig'

    results_df['lmi_category'] = results_df['lmi_p'].apply(assign_lmi_category)

    # Prepare data based on stat_level
    if stat_level == 'mice':
        # Aggregate per mouse first
        plot_data = results_df.groupby(['mouse_id', 'reward_group', 'lmi_category'])['amplitude'].mean().reset_index()
        plot_data.columns = ['mouse_id', 'reward_group', 'lmi_category', 'mean_amplitude']
        stat_label = 'Stats over mice'
    else:  # stat_level == 'cells'
        # Use cells directly
        plot_data = results_df[['reward_group', 'lmi_category', 'amplitude']].copy()
        plot_data.columns = ['reward_group', 'lmi_category', 'mean_amplitude']
        stat_label = 'Stats over cells'

    # Create figure
    sns.set_theme(context='paper', style='ticks', palette='deep', font='sans-serif', font_scale=1.3)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=150, sharey=True)

    # Define colors for LMI categories
    lmi_colors = {
        'LMI+': '#d62728',      # Red
        'Non-sig': '#7f7f7f',   # Gray
        'LMI-': '#1f77b4'       # Blue
    }

    # Order of categories for plotting
    category_order = ['LMI+', 'Non-sig', 'LMI-']

    # Plot for each reward group
    for idx, reward_group in enumerate(['R+', 'R-']):
        ax = axes[idx]

        # Filter data for this reward group
        data = plot_data[plot_data['reward_group'] == reward_group]

        # Ensure all categories are present
        data = data[data['lmi_category'].isin(category_order)]

        # Create bar plot with seaborn
        sns.barplot(data=data, x='lmi_category', y='mean_amplitude',
                   order=category_order, palette=lmi_colors,
                   errorbar='ci', ax=ax, alpha=0.7, edgecolor='black', linewidth=1.5)

        # Add individual points
        if stat_level == 'mice':
            # Show individual mouse points
            sns.stripplot(data=data, x='lmi_category', y='mean_amplitude',
                         order=category_order, color='black', size=4, alpha=0.4, ax=ax)
        # For cells, don't plot individual points (too many)

        # Statistical testing
        # Prepare data for each category
        lmi_pos_data = data[data['lmi_category'] == 'LMI+']['mean_amplitude'].values
        non_sig_data = data[data['lmi_category'] == 'Non-sig']['mean_amplitude'].values
        lmi_neg_data = data[data['lmi_category'] == 'LMI-']['mean_amplitude'].values

        # Kruskal-Wallis test (overall)
        if len(lmi_pos_data) > 0 and len(non_sig_data) > 0 and len(lmi_neg_data) > 0:
            h_stat, p_kruskal = kruskal(lmi_pos_data, non_sig_data, lmi_neg_data)

            # Post-hoc pairwise Mann-Whitney U tests
            comparisons = [
                ('LMI+', 'Non-sig', lmi_pos_data, non_sig_data, 0, 1),
                ('LMI+', 'LMI-', lmi_pos_data, lmi_neg_data, 0, 2),
                ('Non-sig', 'LMI-', non_sig_data, lmi_neg_data, 1, 2)
            ]

            # Bonferroni correction
            alpha_corrected = 0.05 / len(comparisons)

            # Compute y_max and y_min based on actual bar heights (mean ± error), not raw data
            # This prevents outliers from making the y-axis too large
            bar_heights_max = []
            bar_heights_min = []
            for cat in category_order:
                cat_values = data[data['lmi_category'] == cat]['mean_amplitude'].values
                if len(cat_values) > 0:
                    cat_mean = np.mean(cat_values)
                    cat_sem = np.std(cat_values) / np.sqrt(len(cat_values))
                    # Use mean ± 1.96*SEM (approximate 95% CI)
                    bar_heights_max.append(cat_mean + 1.96 * cat_sem)
                    bar_heights_min.append(cat_mean - 1.96 * cat_sem)

            if len(bar_heights_max) > 0:
                y_max = max(bar_heights_max)
                y_min = min(bar_heights_min)
                # Ensure y_min doesn't go below 0 if all bars are positive
                # or y_max doesn't go above 0 if all bars are negative
                if y_max > 0 and y_min > 0:
                    y_min = 0
                elif y_max < 0 and y_min < 0:
                    y_max = 0
                y_range = y_max - y_min
            else:
                y_max = data['mean_amplitude'].max()
                y_min = data['mean_amplitude'].min()
                y_range = y_max - y_min

            for comp_idx, (cat1, cat2, data1, data2, x1, x2) in enumerate(comparisons):
                if len(data1) > 0 and len(data2) > 0:
                    stat, p_val = mannwhitneyu(data1, data2, alternative='two-sided')

                    # Determine significance stars
                    if p_val < alpha_corrected:
                        if p_val < 0.001:
                            stars = '***'
                        elif p_val < 0.01:
                            stars = '**'
                        else:
                            stars = '*'

                        # Draw significance bracket
                        y = y_max + y_range * (0.05 + 0.08 * comp_idx)
                        ax.plot([x1, x1, x2, x2], [y, y + y_range*0.02, y + y_range*0.02, y],
                               'k-', linewidth=1.5)
                        ax.text((x1 + x2) / 2, y + y_range*0.025, stars,
                               ha='center', va='bottom', fontsize=14, fontweight='bold')

            # Set y-axis limits based on bar heights with margin for brackets
            # This prevents outliers from making the plot too large
            y_upper = y_max + y_range * 0.30  # 30% margin above for brackets and padding
            y_lower = y_min - abs(y_range) * 0.05  # 5% margin below
            ax.set_ylim(y_lower, y_upper)

        # Styling
        ax.set_xlabel('LMI Category', fontsize=12)
        ax.set_ylabel('Amplitude (ΔF/F)', fontsize=12)
        ax.set_title(f'{reward_group} {"Mice" if stat_level == "mice" else "Cells"}',
                    fontsize=14, fontweight='bold')

        # Add sample sizes
        for cat_idx, cat in enumerate(category_order):
            cat_data = data[data['lmi_category'] == cat]
            if stat_level == 'mice':
                n = len(cat_data)
                n_mice = cat_data['mouse_id'].nunique() if 'mouse_id' in cat_data.columns else n
                ax.text(cat_idx, -0.02, f'n={n} mice',
                       ha='center', va='top', fontsize=9, transform=ax.get_xaxis_transform())
            else:
                n = len(cat_data)
                ax.text(cat_idx, -0.02, f'n={n} cells',
                       ha='center', va='top', fontsize=9, transform=ax.get_xaxis_transform())

    # Add overall title
    fig.suptitle(f'LMI Groups Amplitude Comparison\n{stat_label}, {filter_label}',
                fontsize=16, fontweight='bold', y=1.02)

    plt.tight_layout()
    sns.despine()

    # Save figure
    filename = f'lmi_groups_amplitude_comparison{filename_suffix}.svg'
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, format='svg', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  ✓ LMI groups amplitude comparison saved to: {save_path}")


def plot_lmi_groups_amplitude_barplot(results_df, output_dir):
    """
    Wrapper function that calls the general version with default parameters
    and generates all 6 versions of the plot.
    """
    # 1. All cells, stats over mice
    plot_lmi_groups_amplitude_barplot_general(results_df, output_dir,
                                              stat_level='mice', amplitude_filter='all',
                                              filename_suffix='')

    # 2. All cells, stats over cells
    plot_lmi_groups_amplitude_barplot_general(results_df, output_dir,
                                              stat_level='cells', amplitude_filter='all',
                                              filename_suffix='_stats_cells')

    # 3. Positive amplitude only, stats over mice
    plot_lmi_groups_amplitude_barplot_general(results_df, output_dir,
                                              stat_level='mice', amplitude_filter='positive',
                                              filename_suffix='_positive_mice')

    # 4. Positive amplitude only, stats over cells
    plot_lmi_groups_amplitude_barplot_general(results_df, output_dir,
                                              stat_level='cells', amplitude_filter='positive',
                                              filename_suffix='_positive_cells')

    # 5. Negative amplitude only, stats over mice
    plot_lmi_groups_amplitude_barplot_general(results_df, output_dir,
                                              stat_level='mice', amplitude_filter='negative',
                                              filename_suffix='_negative_mice')

    # 6. Negative amplitude only, stats over cells
    plot_lmi_groups_amplitude_barplot_general(results_df, output_dir,
                                              stat_level='cells', amplitude_filter='negative',
                                              filename_suffix='_negative_cells')


def plot_lmi_groups_amplitude_histograms(results_df, output_dir):
    """
    Create histogram plots showing amplitude distributions for LMI groups.

    Two panels (R+ and R-), each with three overlaid histograms showing
    amplitude distributions for LMI+, Non-sig, and LMI- cells.

    Uses ALL cells (no significance filtering).

    Parameters
    ----------
    results_df : pd.DataFrame
        Results dataframe with all cells
    output_dir : str
        Output directory for saving figure
    """
    print("\n  Computing LMI groups amplitude histograms...")

    # Use ALL cells (no significance filtering)
    results_df = results_df.copy()

    # Define LMI categories
    def assign_lmi_category(lmi_p):
        if lmi_p >= LMI_POSITIVE_THRESHOLD:
            return 'LMI+'
        elif lmi_p <= LMI_NEGATIVE_THRESHOLD:
            return 'LMI-'
        else:
            return 'Non-sig'

    results_df['lmi_category'] = results_df['lmi_p'].apply(assign_lmi_category)

    # Convert amplitude to percentage (x100) and cap at ±150% ΔF/F (±1.5 in original units)
    results_df['amplitude_pct'] = results_df['amplitude'].clip(lower=-1.5, upper=1.5) * 100

    # Create figure
    sns.set_theme(context='paper', style='ticks', palette='deep', font='sans-serf', font_scale=1.3)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=150, sharey=True)

    # Define colors for LMI categories (matching bar plot)
    lmi_colors = {
        'LMI+': '#d62728',      # Red
        'Non-sig': '#7f7f7f',   # Gray
        'LMI-': '#1f77b4'       # Blue
    }

    # Order of categories for plotting
    category_order = ['LMI+', 'Non-sig', 'LMI-']

    # Plot for each reward group
    for idx, reward_group in enumerate(['R+', 'R-']):
        ax = axes[idx]

        # Filter data for this reward group
        data = results_df[results_df['reward_group'] == reward_group]

        # Plot each LMI category as overlaid histogram
        for category in category_order:
            cat_data = data[data['lmi_category'] == category]

            if len(cat_data) > 0:
                sns.histplot(
                    data=cat_data, x='amplitude_pct', ax=ax,
                    binwidth=10, binrange=(-150, 150),
                    color=lmi_colors[category],
                    alpha=0.4,
                    label=f'{category} (n={len(cat_data)})',
                    stat='density',
                    kde=True,
                    line_kws={'linewidth': 2}
                )

        # Styling
        ax.set_xlabel('Amplitude (%ΔF/F)', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_xlim([-150, 150])
        ax.set_title(f'{reward_group} Mice', fontsize=14, fontweight='bold')
        ax.legend(loc='best', frameon=False, fontsize=10)

    plt.tight_layout()
    sns.despine()

    # Save figure
    save_path = os.path.join(output_dir, 'lmi_groups_amplitude_histograms.svg')
    plt.savefig(save_path, format='svg', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  ✓ LMI groups amplitude histograms saved to: {save_path}")


def plot_amplitude_sign_across_reward_groups(results_df, output_dir):
    """
    Compare average amplitude for positive vs negative amplitude cells across reward groups.

    Creates a bar plot showing the average amplitude (in % ΔF/F) for cells with
    positive and negative amplitude, comparing R+ and R- mice. Uses ALL cells with
    significant fits, irrespective of LMI classification.

    Parameters
    ----------
    results_df : pd.DataFrame
        Results dataframe with all cells (must have 'amplitude' and 'reward_group' columns)
    output_dir : str
        Output directory for saving figure
    """
    print("\n  Computing average amplitude across reward groups...")

    # Filter for significant cells only
    # sig_cells = results_df[results_df['p_value'] < ALPHA].copy()
    sig_cells = results_df.copy()

    # Classify cells by amplitude sign
    sig_cells['amplitude_sign'] = sig_cells['amplitude'].apply(
        lambda x: 'Positive' if x > 0 else 'Negative'
    )

    # Compute average amplitude per mouse first (for each sign category)
    mouse_averages = []
    for mouse_id in sig_cells['mouse_id'].unique():
        mouse_data = sig_cells[sig_cells['mouse_id'] == mouse_id]
        reward_group = mouse_data['reward_group'].iloc[0]

        # Positive amplitude cells
        pos_cells = mouse_data[mouse_data['amplitude_sign'] == 'Positive']
        if len(pos_cells) > 0:
            mouse_averages.append({
                'mouse_id': mouse_id,
                'reward_group': reward_group,
                'amplitude_sign': 'Positive',
                'mean_amplitude': pos_cells['amplitude'].mean() * 100,  # Convert to %
                'n_cells': len(pos_cells)
            })

        # Negative amplitude cells
        neg_cells = mouse_data[mouse_data['amplitude_sign'] == 'Negative']
        if len(neg_cells) > 0:
            mouse_averages.append({
                'mouse_id': mouse_id,
                'reward_group': reward_group,
                'amplitude_sign': 'Negative',
                'mean_amplitude': neg_cells['amplitude'].mean() * 100,  # Convert to %
                'n_cells': len(neg_cells)
            })

    avg_df = pd.DataFrame(mouse_averages)

    # Create figure
    sns.set_theme(context='paper', style='ticks', palette='deep', font='sans-serif', font_scale=1.3)
    fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=150)

    # Plot grouped bar plot - using reward_palette
    x_positions = np.array([0, 1])  # Positive, Negative
    width = 0.35
    amplitude_signs = ['Positive', 'Negative']

    for idx, reward in enumerate(['R-', 'R+']):
        reward_data = avg_df[avg_df['reward_group'] == reward]

        means = []
        sems = []
        for sign in amplitude_signs:
            sign_data = reward_data[reward_data['amplitude_sign'] == sign]['mean_amplitude']
            means.append(sign_data.mean() if len(sign_data) > 0 else 0)
            sems.append(sign_data.sem() if len(sign_data) > 0 else 0)

        offset = width * (idx - 0.5)
        ax.bar(x_positions + offset, means, width, label=reward,
               color=reward_palette[idx], yerr=sems, capsize=5, alpha=0.8)

    ax.set_ylabel('Average Amplitude (% ΔF/F)', fontsize=12)
    ax.set_xlabel('Amplitude Sign', fontsize=12)
    ax.set_title('Average Amplitude Across Reward Groups\n(All cells with significant fits)',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x_positions)
    ax.set_xticklabels(amplitude_signs)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.legend(frameon=False, title='Reward Group')
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()
    sns.despine()

    # Save figure
    save_path = os.path.join(output_dir, 'all_cells_amplitude_across_reward_groups.svg')
    plt.savefig(save_path, format='svg', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Average amplitude comparison saved to: {save_path}")

    # Print summary statistics
    print("\n  Summary statistics (average amplitude in % ΔF/F):")
    for reward in ['R+', 'R-']:
        for sign in amplitude_signs:
            data = avg_df[(avg_df['reward_group'] == reward) &
                         (avg_df['amplitude_sign'] == sign)]
            if len(data) > 0:
                mean_amp = data['mean_amplitude'].mean()
                sem_amp = data['mean_amplitude'].sem()
                n_mice = len(data)
                total_cells = data['n_cells'].sum()
                print(f"    {reward} {sign}: {mean_amp:.2f} ± {sem_amp:.2f} % ΔF/F "
                      f"(n={n_mice} mice, {total_cells} cells)")


def plot_cell_psth_split_by_inflection(ax, mouse_id, roi, inflection_trial, reward_group='R+'):
    """
    Plot PSTH of whisker stimulus responses split by inflection point.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to plot on
    mouse_id : str
        Mouse identifier§
    roi : int
        Cell ROI number
    inflection_trial : int
        Trial index of sigmoid inflection
    reward_group : str, optional
        Reward group ('R+' or 'R-') for color coding (default: 'R+')
    """

    # Load xarray data (baseline-subtracted)¨
    folder = os.path.join(io.processed_dir, 'mice')
    xarr = utils_imaging.load_mouse_xarray(
        mouse_id, folder, 'tensor_xarray_learning_data.nc', substracted=True)

    # Filter for this cell
    xarr_cell = xarr.sel(cell=xarr['roi'] == roi)
    xarr_cell = xarr_cell.sel(trial=xarr_cell['day'] == 0)

    # Filter for whisker stimulus trials (whisker_stim == 1)
    whisker_trials = xarr_cell.sel(trial=xarr_cell['whisker_stim'] == 1)

    # Get trial_w indices
    trial_w_indices = whisker_trials['trial_w'].values

    # Split trials by inflection point
    before_mask = trial_w_indices <= inflection_trial
    after_mask = trial_w_indices > inflection_trial

    trials_before = whisker_trials.isel(trial=before_mask)
    trials_after = whisker_trials.isel(trial=after_mask)
    
    # Convert to DataFrame without computing mean 
    df_before = trials_before.to_dataframe(name='activity').reset_index()
    df_before['activity'] = df_before['activity'] * 100  # Convert to %ΔF/F
    df_before['period'] = f'Before inflection (n={before_mask.sum()})'

    df_after = trials_after.to_dataframe(name='activity').reset_index()
    df_after['activity'] = df_after['activity'] * 100
    df_after['period'] = f'After inflection (n={after_mask.sum()})'

    # Combine
    df_combined = pd.concat([df_before, df_after], ignore_index=True)

    # Use reward-based colors for after-inflection trace
    if reward_group == 'R+':
        colors = ['gray', reward_palette[1]]  # Gray before, green after (R+)
    else:  # R-
        colors = ['gray', reward_palette[0]]  # Light gray before, magenta after (R-)

    # Plot with seaborn - let it compute mean and CI across trials
    sns.lineplot(
        data=df_combined, x='time', y='activity', hue='period',
        errorbar='ci', ax=ax, palette=colors
    )

    # Styling
    ax.axhline(0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.axvline(0, color='red', linestyle='--', linewidth=0.5, alpha=0.5, label='Stimulus onset')
    ax.set_xlabel('Time from stimulus (s)', fontsize=9)
    ax.set_ylabel('Activity (%ΔF/F)', fontsize=9)
    ax.set_title('Whisker Stimulus Response', fontsize=10)
    ax.legend(loc='best', frameon=False, fontsize=8)
    sns.despine(ax=ax)


def plot_behavior_learning_curves(ax, mouse_id, behavior_table, reward_group):
    """
    Plot behavioral learning curves for a specific mouse.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to plot on
    mouse_id : str
        Mouse identifier
    behavior_table : pd.DataFrame
        Behavior table with learning curves
    reward_group : str
        Reward group ('R+' or 'R-')
    """
    # Filter for this mouse, day 0, whisker trials
    mouse_data = behavior_table[
        (behavior_table['mouse_id'] == mouse_id) &
        (behavior_table['day'] == 0) &
        (behavior_table['whisker_stim'] == 1)
    ].reset_index(drop=True)

    if len(mouse_data) == 0:
        ax.text(0.5, 0.5, 'No behavior data', ha='center', va='center', transform=ax.transAxes)
        ax.axis('off')
        return

    # Extract learning curves
    trial_w = mouse_data['trial_w'].values
    learning_curve_w = mouse_data['learning_curve_w'].values.astype(float)
    learning_curve_w_ci_low = mouse_data['learning_curve_w_ci_low'].values.astype(float)
    learning_curve_w_ci_high = mouse_data['learning_curve_w_ci_high'].values.astype(float)
    learning_curve_chance = mouse_data['learning_curve_chance'].values.astype(float)

    # Colors based on reward group
    w_color = behavior_palette[3] if reward_group == 'R+' else behavior_palette[2]
    ns_color = behavior_palette[5]

    # Plot learning curves
    ax.plot(trial_w, learning_curve_w, color=w_color, linewidth=2, label='Whisker')
    ax.fill_between(trial_w, learning_curve_w_ci_low, learning_curve_w_ci_high,
                     color=w_color, alpha=0.2)
    ax.plot(trial_w, learning_curve_chance, color=ns_color, linewidth=2, label='No stim')

    # Add learning trial vertical line
    learning_trial = mouse_data['learning_trial'].values[0]
    if not pd.isna(learning_trial):
        ax.axvline(learning_trial, color='black', linestyle='-', linewidth=1.5, alpha=0.8)

    # Styling
    ax.set_ylim([0, 1])
    ax.set_xlabel('Whisker Trial', fontsize=9)
    ax.set_ylabel('Lick Probability', fontsize=9)
    ax.set_title('Behavioral Learning Curve', fontsize=10)
    ax.legend(loc='best', frameon=False, fontsize=8)
    sns.despine(ax=ax)


def plot_5day_mapping_psth(axes, mouse_id, roi):
    """
    Plot mapping trial PSTH across 5 days for a single cell.

    Parameters
    ----------
    axes : list of matplotlib.axes.Axes
        List of 5 axes (one per day)
    mouse_id : str
        Mouse identifier
    roi : int
        Cell ROI number
    """
    # Load mapping data
    folder = os.path.join(io.processed_dir, 'mice')
    file_name_mapping = 'tensor_xarray_mapping_data.nc'
 
    xarr_mapping = utils_imaging.load_mouse_xarray(mouse_id, folder, file_name_mapping, substracted=True)
    xarr_mapping = xarr_mapping.sel(cell=xarr_mapping['roi'] == roi)
    xarr_mapping.load()

    # Plot each day
    for idx, day in enumerate(DAYS_LEARNING):
        ax = axes[idx]

        # Select day data
        day_data = xarr_mapping.sel(trial=xarr_mapping['day'] == day)

        if len(day_data.trial) > 0:
            # Convert to DataFrame for seaborn
            df = day_data.to_dataframe(name='activity').reset_index()
            df['activity'] = df['activity'] * 100  # Convert to %

            # Plot with CI
            sns.lineplot(data=df, x='time', y='activity', errorbar='ci',
                        ax=ax, color='darkorange', linewidth=1.5)

            # Styling
            ax.axvline(0, color='darkorange', linestyle='-', linewidth=1, alpha=0.5)
            ax.set_title(f'Day {day}\n(n={len(day_data.trial)})', fontsize=9)
            ax.set_ylabel('ΔF/F (%)', fontsize=8)
            ax.set_xlabel('Time (s)', fontsize=8)
            ax.tick_params(labelsize=7)
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'Day {day}', fontsize=9)

        sns.despine(ax=ax)



def create_cell_pdf_report(results_df, output_dir, pdf_name, n_cells=50, sort_by='lmi'):
    """
    Create PDF report with individual cell plots showing raw data and sigmoid fits.

    Parameters
    ----------
    results_df : pd.DataFrame
        Results dataframe with all cells
    output_dir : str
        Output directory
    pdf_name : str
        Name of PDF file to create
    n_cells : int
        Number of top cells to include in report (default: 50)
    sort_by : str
        Column name to sort by (default: 'lmi'). Use 'combined_score' for
        quality-based sorting.
    """
    print(f"\n  Generating {pdf_name} for top {n_cells} cells (sorted by {sort_by})...")

    # Filter for significant cells and sort by specified column
    results_significant = results_df[results_df['p_value'] < ALPHA]
    results_sorted = results_significant.sort_values(sort_by, ascending=False).head(n_cells)

    # Load behavior table for learning curves
    behavior_path = io.adjust_path_to_host(
        r'/mnt/lsens-analysis/Anthony_Renard/data_processed/behavior/'
        r'behavior_imagingmice_table_5days_cut_with_learning_curves.csv'
    )
    behavior_table = pd.read_csv(behavior_path)

    # Create PDF
    pdf_path = os.path.join(output_dir, pdf_name)

    with PdfPages(pdf_path) as pdf:
        for idx, (_, row) in enumerate(results_sorted.iterrows()):
            print(f"  Plotting cell {idx+1}/{len(results_sorted)}: {row['mouse_id']}_{row['roi']}")

            # Load raw data for this cell
            responses, trial_indices, roi_ids = load_day0_data(
                row['mouse_id'], RESPONSE_TYPE, RESPONSE_WIN
            )

            # Find this cell's index
            cell_idx = np.where(roi_ids == row['roi'])[0][0]
            y = responses[cell_idx, :]
            x = trial_indices

            # Create figure with expanded layout
            fig = plt.figure(figsize=(16, 12), dpi=150)
            gs = fig.add_gridspec(
                4, 5,
                hspace=0.45,
                wspace=0.35,
                height_ratios=[1, 0.8, 1, 0.9],
                top=0.96,   # shift the grid up on the page
                bottom=0.06
            )

            # Row 0: Sigmoid fit (full width)
            ax_main = fig.add_subplot(gs[1, :])

            # Plot raw data
            color = reward_palette[1] if row['reward_group'] == 'R+' else reward_palette[0]
            ax_main.scatter(x, y * 100, alpha=0.5, s=30, color=color, label='Raw data')

            # Fit sigmoid model
            sigmoid_fit = fit_sigmoid_model(x, y)

            if sigmoid_fit is not None and sigmoid_fit.get('fit_success', False):
                x_fit = sigmoid_fit['x_clean']
                y_fit = sigmoid_fit['predictions']
                ax_main.plot(x_fit, y_fit * 100, 'darkorange', linewidth=3, label='Sigmoid fit')

                inflexion = row['inflection']
                ax_main.axvline(inflexion, color='darkorange', linestyle='-',
                                linewidth=2, alpha=0.8, label='Inflexion point')

            ax_main.set_xlabel('Whisker Trial Number (trial_w)', fontsize=12)
            ax_main.set_ylabel('Response (ΔF/F0 %)', fontsize=12)
            ax_main.legend(loc='best', fontsize=10)
            ax_main.grid(True, alpha=0.3)
            sns.despine(ax=ax_main)

            # Row 1: Behavior learning curves (full width)
            ax_behavior = fig.add_subplot(gs[0, :])
            plot_behavior_learning_curves(ax_behavior, row['mouse_id'], behavior_table,
                                         row['reward_group'])

            # Row 2: Info panel (left) and Day 0 PSTH (right)
            ax_info = fig.add_subplot(gs[2, 0:2])
            ax_info.axis('off')

            # Format learning trial info
            learning_trial_text = ""
            if 'learning_trial' in row and not pd.isna(row['learning_trial']):
                learning_trial_text = f"""

Learning trial: {row['learning_trial']:.0f}
Inflection rel. to learning: {row['inflection_relative']:.1f} trials
"""

            info_text = f"""
LMI: {row['lmi']:.3f}, p: {row['lmi_p']:.3f}

p-value: {row['p_value']:.4e}
Pseudo-R²: {row['pseudo_r2']:.4f}
Significant: {'YES' if row['p_value'] < ALPHA else 'NO'}

Amplitude: {row['amplitude']*100:.2f}
Inflection: {row['inflection']:.1f} trials
Baseline: {row['baseline']*100:.2f}
Max: {row['max_val']*100:.2f}
            """

            ax_info.text(0.05, 0.95, info_text, transform=ax_info.transAxes,
                    fontsize=10, verticalalignment='top', fontfamily='monospace')

            # Day 0 PSTH (before/after inflection)
            ax_psth = fig.add_subplot(gs[2, 2:])
            inflection_trial = row['inflection']
            plot_cell_psth_split_by_inflection(ax_psth, row['mouse_id'], row['roi'],
                                                inflection_trial,
                                                reward_group=row['reward_group'])

            # Row 3: 5-day mapping PSTH (one panel per day)
            # Create 5 axes that share the same y-axis
            ax0 = fig.add_subplot(gs[3, 0])
            axes_5day = [ax0] + [fig.add_subplot(gs[3, i], sharey=ax0) for i in range(1, 5)]
            plot_5day_mapping_psth(axes_5day, row['mouse_id'], row['roi'])

            # Overall title
            fig.suptitle(f'Cell Plasticity Report #{idx+1} - {row["mouse_id"]}_{row["roi"]} ({row["reward_group"]})',
                        fontsize=16, fontweight='bold', y=0.995)

            # Save to PDF
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

    print(f"  ✓ PDF report saved to: {pdf_path}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main(run_fitting=RUN_FITTING, generate_pdfs=GENERATE_PDFS):
    """
    Main execution function.

    Parameters
    ----------
    run_fitting : bool, optional
        If True, run sigmoid fitting and amplitude computation (default: True).
        If False, load existing results from CSV files.
    generate_pdfs : bool, optional
        If True, generate single-cell PDF reports (default: True).
        If False, skip PDF generation.
    """
    print("="*70)
    print("SINGLE-CELL PLASTICITY ANALYSIS - DAY 0")
    print("="*70)

    if run_fitting:
        print("\nMode: Running sigmoid fitting and amplitude computation")

        # Load mice list
        _, _, mice, db = io.select_sessions_from_db(
            io.db_path, io.nwb_dir, two_p_imaging='yes'
        )

        print(f"\nProcessing {len(mice)} mice in parallel using {N_CORES} cores...")

        # Process all mice in parallel
        all_results = Parallel(n_jobs=N_CORES, verbose=10)(
            delayed(process_mouse)(
                mouse_id,
                response_type=RESPONSE_TYPE,
                response_win=RESPONSE_WIN,
                min_trials=MIN_TRIALS,
                amplitude_type=AMPLITUDE_TYPE
            )
            for mouse_id in mice
        )

        # Filter out None results and empty dataframes
        all_results = [r for r in all_results if r is not None and len(r) > 0]

        # Combine results
        results_df = pd.concat(all_results, ignore_index=True)

        # Add LMI information
        lmi_df = pd.read_csv(os.path.join(io.processed_dir, 'lmi_results.csv'))
        results_df = results_df.merge(
            lmi_df[['mouse_id', 'roi', 'lmi', 'lmi_p']],
            on=['mouse_id', 'roi'],
            how='inner'
        )

        print(f"\nTotal cells before LMI filtering: {len(results_df)}")

        # Filter for LMI-significant cells only
        lmi_positive = results_df[results_df['lmi_p'] >= LMI_POSITIVE_THRESHOLD].copy()
        lmi_negative = results_df[results_df['lmi_p'] <= LMI_NEGATIVE_THRESHOLD].copy()

        print(f"LMI+ cells (p >= {LMI_POSITIVE_THRESHOLD}): {len(lmi_positive)}")
        print(f"LMI- cells (p <= {LMI_NEGATIVE_THRESHOLD}): {len(lmi_negative)}")

        # Add LMI sign column
        lmi_positive['lmi_sign'] = 'Positive'
        lmi_negative['lmi_sign'] = 'Negative'

        # Combine for saving
        results_lmi = pd.concat([lmi_positive, lmi_negative], ignore_index=True)

        # Load and merge behavioral learning trial data
        learning_path = io.adjust_path_to_host(
            r'/mnt/lsens-analysis/Anthony_Renard/data_processed/behavior/'
            r'behavior_imagingmice_table_5days_cut_with_learning_curves.csv'
        )
        learning_df = pd.read_csv(learning_path)
        learning_df = learning_df[['mouse_id', 'learning_trial']].dropna(subset=['learning_trial']).drop_duplicates()

        # Merge learning_trial into ALL cells (for distribution plots)
        results_df = results_df.merge(learning_df, on='mouse_id', how='left')
        results_df['inflection_relative'] = results_df['inflection'] - results_df['learning_trial']

        # Merge learning_trial into LMI-filtered results
        results_lmi = results_lmi.merge(learning_df, on='mouse_id', how='left')

        # Compute inflection relative to learning trial
        results_lmi['inflection_relative'] = results_lmi['inflection'] - results_lmi['learning_trial']

        # Save results
        csv_path_all = os.path.join(OUTPUT_DIR, 'plasticity_results_all_cells.csv')
        results_df.to_csv(csv_path_all, index=False)
        print(f"\nSaved all cells results to {csv_path_all}")

        csv_path_lmi = os.path.join(OUTPUT_DIR, 'plasticity_results_lmi_cells.csv')
        results_lmi.to_csv(csv_path_lmi, index=False)
        print(f"Saved LMI-filtered results to {csv_path_lmi}")

    else:
        print("\nMode: Loading existing results from CSV files")

        # Load existing results
        csv_path_all = os.path.join(OUTPUT_DIR, 'plasticity_results_all_cells.csv')
        csv_path_lmi = os.path.join(OUTPUT_DIR, 'plasticity_results_lmi_cells.csv')

        if not os.path.exists(csv_path_all) or not os.path.exists(csv_path_lmi):
            raise FileNotFoundError(
                f"Results files not found. Please run with run_fitting=True first.\n"
                f"Expected files:\n  {csv_path_all}\n  {csv_path_lmi}"
            )

        results_df = pd.read_csv(csv_path_all)
        results_lmi = pd.read_csv(csv_path_lmi)

        print(f"\nLoaded all cells results from {csv_path_all}")
        print(f"Loaded LMI-filtered results from {csv_path_lmi}")
        print(f"Total cells: {len(results_df)}")
        print(f"LMI-filtered cells: {len(results_lmi)}")

    # Extract LMI+ and LMI- subsets from results_lmi for summary statistics
    lmi_positive = results_lmi[results_lmi['lmi_sign'] == 'Positive']
    lmi_negative = results_lmi[results_lmi['lmi_sign'] == 'Negative']

    # Quantify proportions
    print("\n" + "="*70)
    print("SUMMARY STATISTICS (LMI-SIGNIFICANT CELLS)")
    print("="*70)

    n_total = len(results_lmi)
    n_significant = np.sum(results_lmi['p_value'] < ALPHA)

    print(f"\nTotal LMI-significant cells: {n_total}")
    print(f"  - LMI+ cells: {len(lmi_positive)}")
    print(f"  - LMI- cells: {len(lmi_negative)}")
    print(f"\nCells with significant plasticity (p < {ALPHA}): {n_significant} ({100*n_significant/n_total:.1f}%)")

    # By reward group and LMI sign
    print("\nBy group:")
    for lmi_sign in ['Positive', 'Negative']:
        for group in ['R+', 'R-']:
            df_subset = results_lmi[(results_lmi['lmi_sign'] == lmi_sign) &
                                    (results_lmi['reward_group'] == group)]
            n_subset = len(df_subset)
            n_sig_subset = np.sum(df_subset['p_value'] < ALPHA)
            if n_subset > 0:
                print(f"  LMI{lmi_sign[0]}, {group}: {n_sig_subset}/{n_subset} ({100*n_sig_subset/n_subset:.1f}%) significant")

    # Mean amplitude (significant cells only)
    sig_cells = results_lmi[results_lmi['p_value'] < ALPHA]
    if len(sig_cells) > 0:
        print("\nMean amplitude (significant cells only):")
        for lmi_sign in ['Positive', 'Negative']:
            for group in ['R+', 'R-']:
                df_subset = sig_cells[(sig_cells['lmi_sign'] == lmi_sign) &
                                      (sig_cells['reward_group'] == group)]
                if len(df_subset) > 0:
                    mean_amp = df_subset['amplitude'].mean()
                    std_amp = df_subset['amplitude'].std()
                    print(f"  LMI{lmi_sign[0]}, {group}: {mean_amp:.4f} ± {std_amp:.4f}")

    # Inflection timing analysis
    print("\n" + "="*70)
    print("INFLECTION TIMING RELATIVE TO BEHAVIORAL LEARNING")
    print("="*70)
    sig_with_learning = sig_cells.dropna(subset=['learning_trial'])
    if len(sig_with_learning) > 0:
        for reward in ['R+', 'R-']:
            subset = sig_with_learning[sig_with_learning['reward_group'] == reward]
            if len(subset) > 0:
                mean_rel = subset['inflection_relative'].mean()
                std_rel = subset['inflection_relative'].std()
                median_rel = subset['inflection_relative'].median()
                print(f"\n{reward}: mean={mean_rel:.2f} ± {std_rel:.2f} trials, "
                      f"median={median_rel:.2f} trials (n={len(subset)})")

                # Report how many cells have inflection before/after learning
                before = (subset['inflection_relative'] < 0).sum()
                after = (subset['inflection_relative'] >= 0).sum()
                print(f"  Before learning: {before} ({100*before/len(subset):.1f}%)")
                print(f"  After learning: {after} ({100*after/len(subset):.1f}%)")
    else:
        print("\nNo cells with learning trial data available.")

    # Generate visualizations
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)

    # Plot distributions using ALL cells
    plot_distributions(results_df, OUTPUT_DIR)
    plot_amplitude_distributions_by_lmi(results_df, OUTPUT_DIR, alpha=ALPHA)

    # Plot LMI groups amplitude comparison (using ALL cells)
    plot_lmi_groups_amplitude_barplot(results_df, OUTPUT_DIR)

    # Plot LMI groups amplitude histograms (using ALL cells)
    plot_lmi_groups_amplitude_histograms(results_df, OUTPUT_DIR)

    # Plot amplitude sign comparison across reward groups (using ALL cells)
    plot_amplitude_sign_across_reward_groups(results_df, OUTPUT_DIR)

    # Generate 4 separate PDF reports
    if generate_pdfs:
        print("\n" + "="*70)
        print("GENERATING PDF REPORTS (4 SEPARATE FILES)")
        print("="*70)

        # R+ LMI+ (sorted by LMI value)
        subset = results_lmi[(results_lmi['reward_group'] == 'R+') &
                             (results_lmi['lmi_sign'] == 'Positive')]
        
        if len(subset) > 0:
            create_cell_pdf_report(subset, OUTPUT_DIR,
                                   'plasticity_R+_LMI_positive.pdf',
                                   n_cells=min(100, len(subset)))
        else:
            print("  No R+ LMI+ cells found")

        # R+ LMI+ with significant fits - sorted by combined quality score
        subset_sig = results_lmi[(results_lmi['reward_group'] == 'R+') &
                                 (results_lmi['lmi_sign'] == 'Positive') &
                                 (results_lmi['p_value'] < ALPHA)].copy()
        if len(subset_sig) > 0:
            # Compute combined score: pseudo_r2 * absolute_amplitude
            subset_sig['combined_score'] = (
                subset_sig['pseudo_r2'] *
                subset_sig['amplitude'].abs()
            )

            # Sort by combined score
            subset_sig_sorted = subset_sig.sort_values('combined_score', ascending=False)

            print(f"\n  Generating R+ LMI+ best quality report...")
            print(f"    Combined score = pseudo_r² × |amplitude|")
            print(f"    Top cell: pseudo_r²={subset_sig_sorted.iloc[0]['pseudo_r2']:.3f}, "
                  f"|amplitude|={abs(subset_sig_sorted.iloc[0]['amplitude']):.3f}, "
                  f"score={subset_sig_sorted.iloc[0]['combined_score']:.4f}")

            create_cell_pdf_report(subset_sig_sorted, OUTPUT_DIR,
                                   'plasticity_R+_LMI_positive_best_quality.pdf',
                                   n_cells=min(100, len(subset_sig_sorted)),
                                   sort_by='combined_score')
        else:
            print("  No R+ LMI+ cells with significant fits found")

        # # R+ LMI-
        # subset = results_lmi[(results_lmi['reward_group'] == 'R+') &
        #                      (results_lmi['lmi_sign'] == 'Negative')]
        # if len(subset) > 0:
        #     create_cell_pdf_report(subset, OUTPUT_DIR,
        #                            'plasticity_R+_LMI_negative.pdf',
        #                            n_cells=min(50, len(subset)))
        # else:
        #     print("  No R+ LMI- cells found")

        # # R- LMI+
        # subset = results_lmi[(results_lmi['reward_group'] == 'R-') &
        #                      (results_lmi['lmi_sign'] == 'Positive')]
        # if len(subset) > 0:
        #     create_cell_pdf_report(subset, OUTPUT_DIR,
        #                            'plasticity_R-_LMI_positive.pdf',
        #                            n_cells=min(50, len(subset)))
        # else:
        #     print("  No R- LMI+ cells found")

        # # R- LMI-
        # subset = results_lmi[(results_lmi['reward_group'] == 'R-') &
        #                      (results_lmi['lmi_sign'] == 'Negative')]
        # if len(subset) > 0:
        #     create_cell_pdf_report(subset, OUTPUT_DIR,
        #                            'plasticity_R-_LMI_negative.pdf',
        #                            n_cells=min(50, len(subset)))
        # else:
        #     print("  No R- LMI- cells found")
    else:
        print("\n" + "="*70)
        print("SKIPPING PDF REPORTS (generate_pdfs=False)")
        print("="*70)

    print(f"\nAll outputs saved to: {OUTPUT_DIR}")
    print("\nAnalysis complete!")


if __name__ == '__main__':
    main()
