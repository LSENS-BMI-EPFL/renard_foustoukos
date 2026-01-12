import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import colors
from scipy.stats import mannwhitneyu, wilcoxon
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import spearmanr
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import mannwhitneyu
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from scipy.stats import bootstrap
from joblib import Parallel, delayed
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.utils import resample

# sys.path.append(r'H:/anthony/repos/NWB_analysis')
sys.path.append(r'/home/aprenard/repos/NWB_analysis')
sys.path.append(r'/home/aprenard/repos/fast-learning')
# from nwb_wrappers import nwb_reader_functions as nwb_read
import src.utils.utils_imaging as utils_imaging
import src.utils.utils_io as io
from src.utils.utils_plot import *
from src.utils.utils_behavior import *



# #########################################
# Trial-by-trial correlation matrices averaged across mice.
# #########################################

# Compute a 200x200 similarity matrix (5 days × 40 trials) for each mouse,
# then average across mice to quantify network reorganization across learning.
#
# SOLUTIONS TO DAY-TO-DAY DRIFT ISSUE:
# 1. Z-scoring within each day (set zscore=True):
#    - Removes day-specific baseline shifts (e.g., recording drift)
#    - Preserves trial-to-trial variability within each day
#    - Makes pre-training days (-2, -1) properly cluster together
#
# 2. Alternative similarity metrics (set similarity_metric):
#    - 'pearson': Standard Pearson correlation (default)
#    - 'cosine': Cosine similarity (less sensitive to magnitude)
#    - 'spearman': Rank correlation (robust to monotonic transforms)


sampling_rate = 30
win = (0, 0.300)  # from stimulus onset to 180 ms after.
win_length = f'{int(np.round((win[1]-win[0]) * 1000))}'  # for file naming.
baseline_win = (-1, 0)
baseline_win = (int(baseline_win[0] * sampling_rate), int(baseline_win[1] * sampling_rate))
days_str = ['-2', '-1', '0', '+1', '+2']
days = [-2, -1, 0, 1, 2]
n_map_trials = 40
substract_baseline = True
select_responsive_cells = False
select_lmi = False
zscore = False
projection_type = None  # 'wS2', 'wM1' or None
n_min_proj = 5
similarity_metric = 'pearson'
sns.set_theme(context='paper', style='ticks', palette='deep', font='sans-serif', font_scale=1)

_, _, mice, db = io.select_sessions_from_db(io.db_path,
                                            io.nwb_dir,
                                            two_p_imaging='yes',)
print(mice)

# Load data.
vectors_rew = []
vectors_nonrew = []
mice_rew = []
mice_nonrew = []

# Load responsive cells.
if select_responsive_cells:
    test_df = os.path.join(io.processed_dir, f'response_test_results_win_300ms.csv')
    test_df = pd.read_csv(test_df)
    test_df = test_df.loc[test_df['day'].isin(days)]
    # Select cells as responsive if they pass the test on at least one day.
    selected_cells = test_df.groupby(['mouse_id', 'roi', 'cell_type'])['pval_mapping'].min().reset_index()
    selected_cells = selected_cells.loc[selected_cells['pval_mapping'] <= 0.05/5]

if select_lmi:
    lmi_df = os.path.join(io.processed_dir, f'lmi_results.csv')
    lmi_df = pd.read_csv(lmi_df)
    selected_cells = lmi_df.loc[(lmi_df['lmi_p'] <= 0.025) | (lmi_df['lmi_p'] >= 0.975)]

# Load and prepare data for each mouse.
for mouse in mice:
    print(f"Processing mouse: {mouse}")
    folder = os.path.join(io.solve_common_paths('processed_data'), 'mice')
    file_name = 'tensor_xarray_mapping_data.nc'
    xarray = utils_imaging.load_mouse_xarray(mouse, folder, file_name, substracted=substract_baseline)
    rew_gp = io.get_mouse_reward_group_from_db(io.db_path, mouse, db)

    # Select days.
    xarray = xarray.sel(trial=xarray['day'].isin(days))
    
    # Select responsive cells.
    if select_responsive_cells or select_lmi:
        selected_cells_for_mouse = selected_cells.loc[selected_cells['mouse_id'] == mouse]['roi']
        xarray = xarray.sel(cell=xarray['roi'].isin(selected_cells_for_mouse))
    
    # Option to select a specific projection type or all cells
    if projection_type is not None:
        xarray = xarray.sel(cell=xarray['cell_type'] == projection_type)
        if xarray.sizes['cell'] < n_min_proj:
            print(f"No cells of type {projection_type} for mouse {mouse}.")
            continue
        
    # Check that each day has at least n_map_trials mapping trials
    n_trials = xarray[0, :, 0].groupby('day').count(dim='trial').values
    if np.any(n_trials < n_map_trials):
        print(f'Not enough mapping trials for {mouse}.')
        continue

    # Select last n_map_trials mapping trials for each day.
    d = xarray.groupby('day').apply(lambda x: x.isel(trial=slice(-n_map_trials, None)))    
    d = d.sel(time=slice(win[0], win[1])).mean(dim='time')
    
    # Normalize to remove day-specific baseline shifts
    # This addresses recording drift across days while preserving within-day structure
    if zscore:
        # Option 1: Z-score within each day (recommended to fix day-to-day drift)
        # This removes mean and scales variance independently for each day
        d_normalized = d.copy()
        for day in days:
            day_mask = d['day'] == day
            day_data = d.sel(trial=day_mask)
            # Z-score across trials within this day, for each cell
            day_mean = day_data.mean(dim='trial')
            day_std = day_data.std(dim='trial')
            # Avoid division by zero
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


# Compute trial-by-trial correlation matrices for each mouse and average.
# -----------------------------------------------------------------------

def compute_correlation_matrix(vector):
    """Compute Pearson correlation matrix for a single mouse (200x200)."""
    cm = np.corrcoef(vector.values.T)
    np.fill_diagonal(cm, np.nan)  # Exclude diagonal
    return cm

def compute_cosine_similarity_matrix(vector):
    """
    Compute cosine similarity matrix for a single mouse (200x200).
    Cosine similarity is less sensitive to magnitude differences than correlation.
    """
    from sklearn.metrics.pairwise import cosine_similarity
    cm = cosine_similarity(vector.values.T)
    np.fill_diagonal(cm, np.nan)  # Exclude diagonal
    return cm

def compute_spearman_correlation_matrix(vector):
    """
    Compute Spearman rank correlation matrix for a single mouse (200x200).
    Robust to monotonic transformations and outliers.
    """
    from scipy.stats import spearmanr
    cm, _ = spearmanr(vector.values.T, axis=1)
    np.fill_diagonal(cm, np.nan)  # Exclude diagonal
    return cm

# Choose similarity metric
# Options: 'pearson', 'cosine', 'spearman'


if similarity_metric == 'pearson':
    compute_similarity_matrix = compute_correlation_matrix
elif similarity_metric == 'cosine':
    compute_similarity_matrix = compute_cosine_similarity_matrix
elif similarity_metric == 'spearman':
    compute_similarity_matrix = compute_spearman_correlation_matrix
else:
    raise ValueError(f"Unknown similarity metric: {similarity_metric}")

print(f"Using similarity metric: {similarity_metric}")

# Compute similarity matrices for all mice
corr_matrices_rew = []
corr_matrices_nonrew = []

for vector in vectors_rew:
    corr_matrices_rew.append(compute_similarity_matrix(vector))

for vector in vectors_nonrew:
    corr_matrices_nonrew.append(compute_similarity_matrix(vector))

# Average across mice
avg_corr_rew = np.nanmean(corr_matrices_rew, axis=0)
avg_corr_nonrew = np.nanmean(corr_matrices_nonrew, axis=0)

# Set consistent color scale
vmax_rew = np.nanpercentile(avg_corr_rew, 99)
vmin = 0

# Use select_lmi to indicate cell selection in filenames
celltype_str = 'lmi_cells' if select_lmi else 'all_cells'

# Plot average correlation matrices
fig = plt.figure(figsize=(16, 6))
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.05], wspace=0.25)

ax0 = fig.add_subplot(gs[0, 0])
ax1 = fig.add_subplot(gs[0, 1])
ax_cbar = fig.add_subplot(gs[0, 2])

# R+ group
im0 = ax0.imshow(avg_corr_rew, cmap='viridis', vmax=vmax_rew, vmin=vmin, aspect='auto')
edges = np.cumsum([n_map_trials for _ in range(len(days))])
for edge in edges[:-1] - 0.5:
    ax0.axvline(x=edge, color='white', linestyle='-', linewidth=1.5)
    ax0.axhline(y=edge, color='white', linestyle='-', linewidth=1.5)
ax0.set_xticks(edges - n_map_trials / 2)
ax0.set_xticklabels(days)
ax0.set_yticks(edges - n_map_trials / 2)
ax0.set_yticklabels(days)
ax0.set_xlabel('Day')
ax0.set_ylabel('Day')
ax0.set_title(f'R+ Group (N={len(vectors_rew)} mice)')

# R- group
im1 = ax1.imshow(avg_corr_nonrew, cmap='viridis', vmax=vmax_rew, vmin=vmin, aspect='auto')
for edge in edges[:-1] - 0.5:
    ax1.axvline(x=edge, color='white', linestyle='-', linewidth=1.5)
    ax1.axhline(y=edge, color='white', linestyle='-', linewidth=1.5)
ax1.set_xticks(edges - n_map_trials / 2)
ax1.set_xticklabels(days)
ax1.set_yticks(edges - n_map_trials / 2)
ax1.set_yticklabels(days)
ax1.set_xlabel('Day')
ax1.set_title(f'R- Group (N={len(vectors_nonrew)} mice)')

# Add colorbar on third axis
metric_label = {'pearson': 'Pearson Correlation', 
                'cosine': 'Cosine Similarity', 
                'spearman': 'Spearman Correlation'}[similarity_metric]
cbar = fig.colorbar(im1, cax=ax_cbar, label=metric_label)
plt.tight_layout()

# Save figure
output_dir = '/mnt/lsens-analysis/Anthony_Renard/analysis_output/fast-learning/correlation_matrices/mapping'
output_dir = io.adjust_path_to_host(output_dir)
zscore_str = '_zscore' if zscore else ''
svg_file = f'trialwise_{similarity_metric}_matrices_ctype_{projection_type}_{celltype_str}{zscore_str}.svg'
plt.savefig(os.path.join(output_dir, svg_file), format='svg', dpi=300)
plt.savefig(os.path.join(output_dir, svg_file.replace('.svg', '.png')), format='png', dpi=300)


# Quantification: Network Reorganization Metrics
# ----------------------------------------------

def compute_network_metrics(corr_matrices, days, n_map_trials):
    """
    Compute metrics to quantify network reorganization.
    
    Returns per-mouse:
    - within_pre: average correlation within pre-training days
    - within_post: average correlation within post-training days
    - between_pre_post: average correlation between pre and post days
    - stability_index: (within_pre + within_post) / 2 - between_pre_post
    """
    
    results = []
    
    for cm in corr_matrices:
        # Define trial indices for each period
        # Days: -2, -1, 0, 1, 2
        # Pre: days -2, -1 (indices 0-79)
        # Post: days 1, 2 (indices 120-199)
        
        pre_idx = np.arange(0, 2 * n_map_trials)  # Days -2, -1
        post_idx = np.arange(3 * n_map_trials, 5 * n_map_trials)  # Days 1, 2
        
        # Within-pre correlations
        pre_corr = cm[np.ix_(pre_idx, pre_idx)]
        within_pre = np.nanmean(pre_corr)
        
        # Within-post correlations
        post_corr = cm[np.ix_(post_idx, post_idx)]
        within_post = np.nanmean(post_corr)
        
        # Between pre-post correlations
        between_corr = cm[np.ix_(pre_idx, post_idx)]
        between_pre_post = np.nanmean(between_corr)
        
        # Reorganization index: higher means more reorganization (less similarity pre/post)
        reorganization_index = (within_pre + within_post) / 2 - between_pre_post
        
        results.append({
            'within_pre': within_pre,
            'within_post': within_post,
            'between_pre_post': between_pre_post,
            'reorganization_index': reorganization_index,
        })
    
    return pd.DataFrame(results)

# Compute metrics for both groups
metrics_rew = compute_network_metrics(corr_matrices_rew, days, n_map_trials)
metrics_rew['reward_group'] = 'R+'
metrics_rew['mouse_id'] = mice_rew

metrics_nonrew = compute_network_metrics(corr_matrices_nonrew, days, n_map_trials)
metrics_nonrew['reward_group'] = 'R-'
metrics_nonrew['mouse_id'] = mice_nonrew

metrics_combined = pd.concat([metrics_rew, metrics_nonrew], ignore_index=True)

# Print summary statistics
print("\n" + "="*60)
print("NETWORK REORGANIZATION METRICS")
print("="*60)
for group in ['R+', 'R-']:
    data = metrics_combined[metrics_combined['reward_group'] == group]
    print(f"\n{group} Group (N={len(data)}):")
    print(f"  Within-pre correlation:     {data['within_pre'].mean():.3f} ± {data['within_pre'].std():.3f}")
    print(f"  Within-post correlation:    {data['within_post'].mean():.3f} ± {data['within_post'].std():.3f}")
    print(f"  Between pre-post:           {data['between_pre_post'].mean():.3f} ± {data['between_pre_post'].std():.3f}")
    print(f"  Reorganization index:       {data['reorganization_index'].mean():.3f} ± {data['reorganization_index'].std():.3f}")

# Statistical comparison between groups
print("\n" + "-"*60)
print("STATISTICAL COMPARISONS (Mann-Whitney U test)")
print("-"*60)

stats_dict = {}
for metric in ['within_pre', 'within_post', 'between_pre_post', 'reorganization_index']:
    r_plus = metrics_rew[metric].dropna()
    r_minus = metrics_nonrew[metric].dropna()
    stat, p_value = mannwhitneyu(r_plus, r_minus, alternative='two-sided')
    stats_dict[metric] = p_value
    print(f"{metric:25s}: U={stat:.1f}, p={p_value:.4f} {'***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'n.s.'}")

# Visualization of metrics
fig, ax = plt.subplots(1, 1, figsize=(8, 5))

# Combine Pre, Post, and Reorganization Index in one panel
metrics_long = metrics_combined.melt(
    id_vars=['mouse_id', 'reward_group'],
    value_vars=['within_pre', 'within_post', 'reorganization_index'],
    var_name='metric', value_name='value'
)
sns.swarmplot(data=metrics_long, x='metric', y='value', hue='reward_group',
              palette=reward_palette[::-1], dodge=True, alpha=0.5, ax=ax, size=4)
sns.pointplot(data=metrics_long, x='metric', y='value', hue='reward_group',
              palette=reward_palette[::-1], ax=ax, linestyles='none',
              errorbar='ci', dodge=True)
ax.set_ylim(0, .3)
ax.set_xlabel('')
ax.set_ylabel('Value')
ax.set_title('Network Reorganization Metrics')
ax.legend(title='Group')
ax.set_xticklabels(['Within Pre', 'Within Post', 'Reorganization\nIndex'])
ax.set_ylim(0, None)

# Add p-values
metric_order = ['within_pre', 'within_post', 'reorganization_index']
for i, metric in enumerate(metric_order):
    p_val = stats_dict[metric]
    y_max = metrics_long[metrics_long['metric'] == metric]['value'].max()
    y_pos = y_max * 1.05

    # Format p-value text
    if p_val < 0.001:
        p_text = 'p<0.001'
    elif p_val < 0.01:
        p_text = f'p={p_val:.3f}'
    else:
        p_text = f'p={p_val:.2f}'

    ax.text(i, y_pos*0.98, p_text, ha='center', va='bottom', fontsize=9)

sns.despine()
plt.tight_layout()

# Save figure
zscore_str = '_zscore' if zscore else ''
svg_file = f'network_reorganization_metrics_{similarity_metric}_ctype_{projection_type}_{celltype_str}{zscore_str}.svg'
plt.savefig(os.path.join(output_dir, svg_file), format='svg', dpi=300)
plt.savefig(os.path.join(output_dir, svg_file.replace('.svg', '.png')), format='png', dpi=300)

# Save data
metrics_combined.to_csv(os.path.join(output_dir, f'network_reorganization_metrics_{similarity_metric}_ctype_{projection_type}_{celltype_str}{zscore_str}.csv'), index=False)

# Save statistical results
stats_results = []
for metric in ['within_pre', 'within_post', 'between_pre_post', 'reorganization_index']:
    r_plus = metrics_rew[metric].dropna()
    r_minus = metrics_nonrew[metric].dropna()
    stat, p_value = mannwhitneyu(r_plus, r_minus, alternative='two-sided')
    stats_results.append({
        'metric': metric,
        'U_statistic': stat,
        'p_value': p_value
    })
stats_df = pd.DataFrame(stats_results)
stats_df.to_csv(os.path.join(output_dir, f'network_reorganization_stats_{similarity_metric}_ctype_{projection_type}_{celltype_str}{zscore_str}.csv'), index=False)





# ###########################################################
# Correlation matrices including mapping and learning trials.
# ###########################################################
# Focus on Day 0 learning with variable session lengths handled properly

sampling_rate = 30
win = (0, 0.300)  # from stimulus onset to 180 ms after
baseline_win = (-1, 0)
baseline_win = (int(baseline_win[0] * sampling_rate), int(baseline_win[1] * sampling_rate))
days = [-2, -1, 0, 1, 2]
n_map_trials = 40  # Fixed number of mapping trials per day
n_learning_trials = 80  # Number of learning trials to take (last 80)
substract_baseline = True
select_responsive_cells = False
select_lmi = False  # Use LMI-selected cells
similarity_metric = 'spearman'  # 'pearson', 'spearman', or 'cosine'
sns.set_theme(context='paper', style='ticks', palette='deep', font='sans-serif', font_scale=1)

_, _, mice, db = io.select_sessions_from_db(io.db_path,
                                            io.nwb_dir,
                                            two_p_imaging='yes',)
print(f"Total mice: {len(mice)}")

# Load cell selection criteria
if select_lmi:
    lmi_df = os.path.join(io.processed_dir, f'lmi_results.csv')
    lmi_df = pd.read_csv(lmi_df)
    selected_cells = lmi_df.loc[(lmi_df['lmi_p'] <= 0.025) | (lmi_df['lmi_p'] >= 0.975)]

# Storage for per-mouse data
mouse_data_rew = []
mouse_data_nonrew = []
mice_rew = []
mice_nonrew = []

for mouse in mice:
    print(f"\nProcessing mouse: {mouse}")
    folder = os.path.join(io.solve_common_paths('processed_data'), 'mice')
    
    # Load mapping data
    file_name = 'tensor_xarray_mapping_data.nc'
    xarray_map = utils_imaging.load_mouse_xarray(mouse, folder, file_name, substracted=substract_baseline)
    
    # Load learning data
    file_name = 'tensor_xarray_learning_data.nc'
    xarray_learning = utils_imaging.load_mouse_xarray(mouse, folder, file_name, substracted=substract_baseline)
    
    # Get reward group
    rew_gp = io.get_mouse_reward_group_from_db(io.db_path, mouse, db)
    
    # Select cells
    if select_lmi:
        selected_cells_mouse = selected_cells.loc[selected_cells['mouse_id'] == mouse]['roi']
        xarray_map = xarray_map.sel(cell=xarray_map['roi'].isin(selected_cells_mouse))
        xarray_learning = xarray_learning.sel(cell=xarray_learning['roi'].isin(selected_cells_mouse))
    
    # Check minimum number of cells
    if xarray_map.sizes['cell'] < 5:
        print(f"  Skipping {mouse}: insufficient cells ({xarray_map.sizes['cell']})")
        continue
    
    # Process mapping trials per day: days -2, -1, 0, +1, +2
    # Structure will be: map_day-2 | map_day-1 | learning_day0 | map_day0 | learning_day+1 | map_day+1 | learning_day+2 | map_day+2
    
    mapping_days_data = {}
    for day in [-2, -1, 0, 1, 2]:
        map_day = xarray_map.sel(trial=xarray_map['day'] == day)
        n_trials_day = map_day.sizes['trial']
        
        if n_trials_day < n_map_trials:
            print(f"  Skipping {mouse}: insufficient mapping trials on day {day} ({n_trials_day})")
            break
        
        # Take last n_map_trials from this day
        map_day = map_day.isel(trial=slice(-n_map_trials, None))
        map_day = map_day.sel(time=slice(win[0], win[1])).mean(dim='time')
        mapping_days_data[day] = map_day.values  # (n_cells, n_map_trials)
    else:
        # All days have sufficient mapping trials, continue processing
        pass
    
    # If we broke out of the loop early, skip this mouse
    if len(mapping_days_data) < 5:
        continue
    
    # Process learning trials (whisker trials only) for days 0, +1, +2
    # Take last n_learning_trials from each day (no binning)
    
    learning_days_data = {}
    for day in [0, 1, 2]:
        learning_day = xarray_learning.sel(trial=(xarray_learning['day'] == day) & 
                                                  (xarray_learning['trial_type'] == 'whisker_trial'))
        learning_day = learning_day.sel(time=slice(win[0], win[1])).mean(dim='time')
        
        n_trials_day = learning_day.sizes['trial']
        
        if n_trials_day < n_learning_trials:
            print(f"  Skipping {mouse}: insufficient learning trials on day {day} ({n_trials_day} < {n_learning_trials})")
            break
        
        # Take last n_learning_trials from this day
        learning_day = learning_day.isel(trial=slice(0, n_learning_trials))
        learning_days_data[day] = learning_day.values  # (n_cells, n_learning_trials)
    else:
        # All days have sufficient learning trials
        print(f"  Reward group: {rew_gp}, Cells: {learning_days_data[0].shape[0]}, " + 
              f"Day 0: {learning_days_data[0].shape[1]} trials, Day +1: {learning_days_data[1].shape[1]}, Day +2: {learning_days_data[2].shape[1]}")
    
    # If we broke out of the loop early, skip this mouse
    if len(learning_days_data) < 3:
        continue
    
    # Store data for this mouse
    # Structure: map_day-2 (40) | map_day-1 (40) | learning_day0 (80) | map_day0 (40) | 
    #            learning_day+1 (80) | map_day+1 (40) | learning_day+2 (80) | map_day+2 (40)
    mouse_data = {
        'mouse_id': mouse,
        'reward_group': rew_gp,
        'map_day-2': mapping_days_data[-2],  # (n_cells, 40)
        'map_day-1': mapping_days_data[-1],  # (n_cells, 40)
        'learning_day0': learning_days_data[0],  # (n_cells, 80)
        'map_day0': mapping_days_data[0],  # (n_cells, 40)
        'learning_day1': learning_days_data[1],  # (n_cells, 80)
        'map_day1': mapping_days_data[1],  # (n_cells, 40)
        'learning_day2': learning_days_data[2],  # (n_cells, 80)
        'map_day2': mapping_days_data[2],  # (n_cells, 40)
    }
    
    # Verify dimensions
    # Total: 40 + 40 + 80 + 40 + 80 + 40 + 80 + 40 = 440 trials
    expected_trials = 5 * n_map_trials + 3 * n_learning_trials  # 200 + 240 = 440
    actual_trials = sum([data.shape[1] for data in [
        mapping_days_data[-2], mapping_days_data[-1],
        learning_days_data[0], mapping_days_data[0],
        learning_days_data[1], mapping_days_data[1],
        learning_days_data[2], mapping_days_data[2]
    ]])
    
    print(f"  Matrix size will be: {expected_trials} × {expected_trials}")
    
    if actual_trials != expected_trials:
        print(f"  WARNING: Dimension mismatch! Expected {expected_trials}, got {actual_trials}")
        continue
    
    if rew_gp == 'R+':
        mouse_data_rew.append(mouse_data)
        mice_rew.append(mouse)
    elif rew_gp == 'R-':
        mouse_data_nonrew.append(mouse_data)
        mice_nonrew.append(mouse)

print(f"\nLoaded {len(mouse_data_rew)} R+ mice and {len(mouse_data_nonrew)} R- mice")


# Compute correlation matrices for each mouse
# --------------------------------------------

def compute_correlation_matrix(mouse_data, similarity_metric='spearman'):
    """
    Compute trial-by-trial correlation matrix for one mouse.
    
    Structure: Map Day-2 (40) | Map Day-1 (40) | Learn Day 0 (80) | Map Day 0 (40) | 
               Learn Day+1 (80) | Map Day+1 (40) | Learn Day+2 (80) | Map Day+2 (40)
    Total: 440 × 440
    """
    # Concatenate all data in the specified order
    combined = np.concatenate([
        mouse_data['map_day-2'],      # 40 trials
        mouse_data['map_day-1'],      # 40 trials
        mouse_data['learning_day0'],   # 80 trials
        mouse_data['map_day0'],        # 40 trials
        mouse_data['learning_day1'],   # 80 trials
        mouse_data['map_day1'],        # 40 trials
        mouse_data['learning_day2'],   # 80 trials
        mouse_data['map_day2'],        # 40 trials
    ], axis=1)  # Shape: (n_cells, 440)
    
    # Compute correlation matrix
    if similarity_metric == 'spearman':
        from scipy.stats import spearmanr
        corr_matrix, _ = spearmanr(combined.T, axis=1)
    elif similarity_metric == 'pearson':
        corr_matrix = np.corrcoef(combined.T)
    elif similarity_metric == 'cosine':
        from sklearn.metrics.pairwise import cosine_similarity
        corr_matrix = cosine_similarity(combined.T)
    else:
        raise ValueError(f"Unknown similarity metric: {similarity_metric}")
    
    # Set diagonal to NaN
    np.fill_diagonal(corr_matrix, np.nan)
    
    return corr_matrix


# Compute matrices for all mice
print(f"\nComputing correlation matrices using {similarity_metric} similarity...")

matrices_rew = []
for mouse_data in mouse_data_rew:
    matrix = compute_correlation_matrix(mouse_data, similarity_metric)
    matrices_rew.append(matrix)
    print(f"  R+ {mouse_data['mouse_id']}: matrix shape {matrix.shape}")

matrices_nonrew = []
for mouse_data in mouse_data_nonrew:
    matrix = compute_correlation_matrix(mouse_data, similarity_metric)
    matrices_nonrew.append(matrix)
    print(f"  R- {mouse_data['mouse_id']}: matrix shape {matrix.shape}")

# Average across mice within each reward group
avg_corr_rew = np.nanmean(matrices_rew, axis=0)
avg_corr_nonrew = np.nanmean(matrices_nonrew, axis=0)

print(f"\nAverage correlation matrix shapes:")
print(f"  R+ group: {avg_corr_rew.shape}")
print(f"  R- group: {avg_corr_nonrew.shape}")


# Visualize average correlation matrices
# ---------------------------------------

# Define epoch boundaries for visualization
# Structure: Map-2 (40) | Map-1 (40) | Learn0 (80) | Map0 (40) | Learn1 (80) | Map1 (40) | Learn2 (80) | Map2 (40)
epoch_boundaries = [40, 80, 160, 200, 280, 320, 400, 440]
epoch_labels = ['Map\nDay -2', 'Map\nDay -1', 'Learn\nDay 0', 'Map\nDay 0', 
                'Learn\nDay +1', 'Map\nDay +1', 'Learn\nDay +2', 'Map\nDay +2']
epoch_centers = [20, 60, 120, 180, 240, 300, 360, 420]

# Set color scale
vmax = np.nanpercentile(avg_corr_rew, 99)
vmin = 0

# Create figure with separate colorbar axis
fig = plt.figure(figsize=(16, 6))
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.05], wspace=0.3)

ax0 = fig.add_subplot(gs[0, 0])
ax1 = fig.add_subplot(gs[0, 1])
ax_cbar = fig.add_subplot(gs[0, 2])

# R+ group
im0 = ax0.imshow(avg_corr_rew, cmap='viridis', vmax=vmax, vmin=vmin, aspect='auto')
for edge in epoch_boundaries[:-1]:
    ax0.axvline(x=edge - 0.5, color='white', linestyle='-', linewidth=1.5)
    ax0.axhline(y=edge - 0.5, color='white', linestyle='-', linewidth=1.5)
ax0.set_xticks(epoch_centers)
ax0.set_xticklabels(epoch_labels, fontsize=8)
ax0.set_yticks(epoch_centers)
ax0.set_yticklabels(epoch_labels, fontsize=8)
ax0.set_title(f'R+ Group (N={len(mouse_data_rew)} mice)')

# R- group
im1 = ax1.imshow(avg_corr_nonrew, cmap='viridis', vmax=vmax, vmin=vmin, aspect='auto')
for edge in epoch_boundaries[:-1]:
    ax1.axvline(x=edge - 0.5, color='white', linestyle='-', linewidth=1.5)
    ax1.axhline(y=edge - 0.5, color='white', linestyle='-', linewidth=1.5)
ax1.set_xticks(epoch_centers)
ax1.set_xticklabels(epoch_labels, fontsize=8)
ax1.set_yticks(epoch_centers)
ax1.set_yticklabels(epoch_labels, fontsize=8)
ax1.set_title(f'R- Group (N={len(mouse_data_nonrew)} mice)')

# Add colorbar
metric_label = {'pearson': 'Pearson Correlation', 
                'cosine': 'Cosine Similarity', 
                'spearman': 'Spearman Correlation'}[similarity_metric]
cbar = fig.colorbar(im1, cax=ax_cbar, label=metric_label)
plt.tight_layout()

# Save figure
output_dir = '/mnt/lsens-analysis/Anthony_Renard/analysis_output/fast-learning/correlation_matrices/learning_mapping'
output_dir = io.adjust_path_to_host(output_dir)
os.makedirs(output_dir, exist_ok=True)

celltype_str = 'lmi_cells' if select_lmi else 'all_cells'
svg_file = f'avg_correlation_matrices_{similarity_metric}_{celltype_str}.svg'
plt.savefig(os.path.join(output_dir, svg_file), format='svg', dpi=300)
plt.savefig(os.path.join(output_dir, svg_file.replace('.svg', '.png')), format='png', dpi=300)

print(f"\nSaved figure to: {os.path.join(output_dir, svg_file)}")



# #############################################################################
# Day 0 specific correlation matrix analysis for different time windows.
# #############################################################################

# Define time windows (in seconds)
time_windows = [
    (0, 0.300),
    (0.300, 1.000),
    (1.000, 2.000),
    (2.000, 3.000),
    (3.000, 4.000),
    (4.000, 5.000)
]

# Analysis parameters
sampling_rate = 30
n_learning_trials = 80  # Number of trials to analyze
substract_baseline = True
select_lmi = False  # Use LMI-selected cells
similarity_metric = 'spearman'  # 'pearson', 'spearman', or 'cosine'

print("\n" + "="*80)
print("DAY 0 TIME WINDOW ANALYSIS")
print("="*80)
print(f"Time windows: {[(int(w[0]*1000), int(w[1]*1000)) for w in time_windows]} ms")
print(f"Similarity metric: {similarity_metric}")
print(f"Number of trials per mouse: {n_learning_trials}")

# Load mice
_, _, mice, db = io.select_sessions_from_db(io.db_path,
                                            io.nwb_dir,
                                            two_p_imaging='yes',)

# Load cell selection criteria
if select_lmi:
    lmi_df = os.path.join(io.processed_dir, f'lmi_results.csv')
    lmi_df = pd.read_csv(lmi_df)
    selected_cells = lmi_df.loc[(lmi_df['lmi_p'] <= 0.025) | (lmi_df['lmi_p'] >= 0.975)]

# Storage for per-mouse, per-window data
window_data_rew = {i: [] for i in range(len(time_windows))}
window_data_nonrew = {i: [] for i in range(len(time_windows))}
mice_rew_tw = []
mice_nonrew_tw = []

# Load and process data for each mouse
for mouse in mice:
    print(f"\nProcessing mouse: {mouse}")
    folder = os.path.join(io.solve_common_paths('processed_data'), 'mice')

    # Load learning data
    file_name = 'tensor_xarray_learning_data.nc'
    xarray_learning = utils_imaging.load_mouse_xarray(mouse, folder, file_name, substracted=substract_baseline)

    # Get reward group
    rew_gp = io.get_mouse_reward_group_from_db(io.db_path, mouse, db)

    # Select cells
    if select_lmi:
        selected_cells_mouse = selected_cells.loc[selected_cells['mouse_id'] == mouse]['roi']
        xarray_learning = xarray_learning.sel(cell=xarray_learning['roi'].isin(selected_cells_mouse))

    # Check minimum number of cells
    if xarray_learning.sizes['cell'] < 5:
        print(f"  Skipping {mouse}: insufficient cells ({xarray_learning.sizes['cell']})")
        continue

    # Select day 0 whisker trials
    day0_whisker = xarray_learning.sel(trial=(xarray_learning['day'] == 0) &
                                              (xarray_learning['trial_type'] == 'whisker_trial'))

    n_trials_day0 = day0_whisker.sizes['trial']

    if n_trials_day0 < n_learning_trials:
        print(f"  Skipping {mouse}: insufficient day 0 whisker trials ({n_trials_day0} < {n_learning_trials})")
        continue

    # Take first n_learning_trials
    day0_whisker = day0_whisker.isel(trial=slice(0, n_learning_trials))

    print(f"  Reward group: {rew_gp}, Cells: {day0_whisker.sizes['cell']}, Trials: {day0_whisker.sizes['trial']}")

    # Process each time window
    window_arrays = []
    for win_idx, (win_start, win_end) in enumerate(time_windows):
        # Select time window and average over time
        win_data = day0_whisker.sel(time=slice(win_start, win_end)).mean(dim='time')
        window_arrays.append(win_data.values)  # (n_cells, n_trials)
        print(f"    Window {int(win_start*1000)}-{int(win_end*1000)}ms: shape {win_data.values.shape}")

    # Store data by reward group
    if rew_gp == 'R+':
        mice_rew_tw.append(mouse)
        for win_idx, win_array in enumerate(window_arrays):
            window_data_rew[win_idx].append(win_array)
    elif rew_gp == 'R-':
        mice_nonrew_tw.append(mouse)
        for win_idx, win_array in enumerate(window_arrays):
            window_data_nonrew[win_idx].append(win_array)

print(f"\nLoaded {len(mice_rew_tw)} R+ mice and {len(mice_nonrew_tw)} R- mice")


# Compute correlation matrices for each time window
# --------------------------------------------------

def compute_corr_matrix_from_array(data_array, similarity_metric='spearman'):
    """
    Compute trial-by-trial correlation matrix.

    Args:
        data_array: (n_cells, n_trials) array
        similarity_metric: 'spearman', 'pearson', or 'cosine'

    Returns:
        corr_matrix: (n_trials, n_trials) correlation matrix
    """
    if similarity_metric == 'spearman':
        from scipy.stats import spearmanr
        corr_matrix, _ = spearmanr(data_array.T, axis=1)
    elif similarity_metric == 'pearson':
        corr_matrix = np.corrcoef(data_array.T)
    elif similarity_metric == 'cosine':
        from sklearn.metrics.pairwise import cosine_similarity
        corr_matrix = cosine_similarity(data_array.T)
    else:
        raise ValueError(f"Unknown similarity metric: {similarity_metric}")

    # Set diagonal to NaN
    np.fill_diagonal(corr_matrix, np.nan)

    return corr_matrix


# Compute matrices for each time window
print(f"\nComputing correlation matrices for each time window...")

window_matrices_rew = []
window_matrices_nonrew = []

for win_idx in range(len(time_windows)):
    print(f"\nTime window {time_windows[win_idx][0]*1000:.0f}-{time_windows[win_idx][1]*1000:.0f}ms:")

    # R+ group
    matrices_rew_win = []
    for mouse_data in window_data_rew[win_idx]:
        matrix = compute_corr_matrix_from_array(mouse_data, similarity_metric)
        matrices_rew_win.append(matrix)

    avg_matrix_rew = np.nanmean(matrices_rew_win, axis=0)
    window_matrices_rew.append(avg_matrix_rew)
    print(f"  R+ group: {len(matrices_rew_win)} mice, avg matrix shape {avg_matrix_rew.shape}")

    # R- group
    matrices_nonrew_win = []
    for mouse_data in window_data_nonrew[win_idx]:
        matrix = compute_corr_matrix_from_array(mouse_data, similarity_metric)
        matrices_nonrew_win.append(matrix)

    avg_matrix_nonrew = np.nanmean(matrices_nonrew_win, axis=0)
    window_matrices_nonrew.append(avg_matrix_nonrew)
    print(f"  R- group: {len(matrices_nonrew_win)} mice, avg matrix shape {avg_matrix_nonrew.shape}")


# Visualize correlation matrices for all time windows
# ----------------------------------------------------

# Create figure with all time windows
n_windows = len(time_windows)
fig, axes = plt.subplots(2, n_windows, figsize=(4*n_windows, 8))

# Set consistent color scale across all windows
all_matrices = window_matrices_rew + window_matrices_nonrew
vmax = np.nanpercentile(np.concatenate([m.flatten() for m in all_matrices]), 99)
vmin = 0

for win_idx in range(n_windows):
    win_start, win_end = time_windows[win_idx]
    win_label = f"{int(win_start*1000)}-{int(win_end*1000)}ms"

    # R+ group
    ax = axes[0, win_idx]
    im = ax.imshow(window_matrices_rew[win_idx], cmap='viridis', vmax=vmax, vmin=vmin, aspect='auto')
    ax.set_title(f'{win_label}\nR+ (N={len(mice_rew_tw)})', fontsize=10)
    ax.set_xlabel('Trial')
    if win_idx == 0:
        ax.set_ylabel('Trial')

    # R- group
    ax = axes[1, win_idx]
    im = ax.imshow(window_matrices_nonrew[win_idx], cmap='viridis', vmax=vmax, vmin=vmin, aspect='auto')
    ax.set_title(f'{win_label}\nR- (N={len(mice_nonrew_tw)})', fontsize=10)
    ax.set_xlabel('Trial')
    if win_idx == 0:
        ax.set_ylabel('Trial')

# Add colorbar
metric_label = {'pearson': 'Pearson Correlation',
                'cosine': 'Cosine Similarity',
                'spearman': 'Spearman Correlation'}[similarity_metric]
fig.colorbar(im, ax=axes.ravel().tolist(), label=metric_label, fraction=0.046, pad=0.04)

plt.tight_layout()

# Save figure
output_dir = '/mnt/lsens-analysis/Anthony_Renard/analysis_output/fast-learning/correlation_matrices/day0_time_windows'
output_dir = io.adjust_path_to_host(output_dir)
os.makedirs(output_dir, exist_ok=True)

celltype_str = 'lmi_cells' if select_lmi else 'all_cells'
svg_file = f'day0_whisker_correlation_matrices_{similarity_metric}_{celltype_str}_all_windows.svg'
plt.savefig(os.path.join(output_dir, svg_file), format='svg', dpi=300)
plt.savefig(os.path.join(output_dir, svg_file.replace('.svg', '.png')), format='png', dpi=300)

print(f"\nSaved figure to: {os.path.join(output_dir, svg_file)}")


# Quantify correlation strength across time windows
# --------------------------------------------------

def compute_mean_correlation(corr_matrix):
    """Compute mean of off-diagonal correlation values."""
    return np.nanmean(corr_matrix)

# Compute mean correlation for each window and reward group
mean_corr_rew = [compute_mean_correlation(m) for m in window_matrices_rew]
mean_corr_nonrew = [compute_mean_correlation(m) for m in window_matrices_nonrew]

# Create summary dataframe
window_labels = [f"{int(w[0]*1000)}-{int(w[1]*1000)}" for w in time_windows]
summary_df = pd.DataFrame({
    'time_window': window_labels + window_labels,
    'mean_correlation': mean_corr_rew + mean_corr_nonrew,
    'reward_group': ['R+'] * n_windows + ['R-'] * n_windows
})

print("\n" + "="*60)
print("MEAN CORRELATION BY TIME WINDOW")
print("="*60)
print(summary_df.to_string(index=False))

# Plot mean correlation across time windows
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

x_positions = np.arange(n_windows)
width = 0.35

bars1 = ax.bar(x_positions - width/2, mean_corr_rew, width,
               label='R+', color=reward_palette[1], alpha=0.8)
bars2 = ax.bar(x_positions + width/2, mean_corr_nonrew, width,
               label='R-', color=reward_palette[0], alpha=0.8)

ax.set_xlabel('Time Window (ms)')
ax.set_ylabel(f'Mean {metric_label}')
ax.set_title('Day 0 Whisker Trial Correlation Across Time Windows')
ax.set_xticks(x_positions)
ax.set_xticklabels(window_labels, rotation=45, ha='right')
ax.legend()
ax.set_ylim(0, None)

sns.despine()
plt.tight_layout()

# Save figure
svg_file = f'day0_whisker_mean_correlation_by_window_{similarity_metric}_{celltype_str}.svg'
plt.savefig(os.path.join(output_dir, svg_file), format='svg', dpi=300)
plt.savefig(os.path.join(output_dir, svg_file.replace('.svg', '.png')), format='png', dpi=300)

print(f"\nSaved figure to: {os.path.join(output_dir, svg_file)}")

# Save summary data
summary_df.to_csv(os.path.join(output_dir, f'day0_whisker_correlation_summary_{similarity_metric}_{celltype_str}.csv'), index=False)

print("\n" + "="*80)
print("DAY 0 TIME WINDOW ANALYSIS COMPLETE")
print("="*80)

