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
import xarray as xr

# sys.path.append(r'H:/anthony/repos/NWB_analysis')
sys.path.append(r'/home/aprenard/repos/NWB_analysis')
# from nwb_wrappers import nwb_reader_functions as nwb_read
from fast_learning import imaging
from fast_learning import paths, database
from fast_learning.plotting import *
from fast_learning.behavior import *
from statannotations.Annotator import Annotator
from scipy.stats import mannwhitneyu
from scipy.stats import wilcoxon
from scipy.stats import spearmanr, pearsonr




# #############################################################################
# Gradual learning during Day 0 realigned to "learning trial".
# #############################################################################

# keep plot before meeting with the most gradual four mice (from GF)

sampling_rate = 30
win = (0, .300)  # from stimulus onset to 300 ms after.
win_length = f'{int(np.round((win[1]-win[0]) * 1000))}'  # for file naming.
# win = (int(win[0] * sampling_rate), int(win[1] * sampling_rate))
days_str = ['-2', '-1', '0', '+1', '+2']
days = [-2, -1, 0, 1, 2]
subtract_baseline = True
realigned_to_learning = False
sns.set_theme(context='paper', style='ticks', palette='deep', font='sans-serif', font_scale=1)

processed_folder = paths.processed_dir  

_, _, mice, db = database.select_sessions_from_db(paths.db_path,
                                            paths.nwb_dir,
                                            two_p_imaging='yes',)


# excluded_mice = ['GF307', 'GF310', 'GF333', 'MI075', 'AR144', 'AR135', 'AR163']
# mice = [m for m in mice if m not in excluded_mice]
# mice = ['GF305', 'GF306', 'GF317', 'GF323', 'GF318', 'GF313']
# mice = ['GF305', 'GF306', 'GF317', ]

# Load LMI dataframe.
# -------------------

lmi_df = pd.read_csv(os.path.join(processed_folder, 'lmi_results.csv'))

# learning_trials = {'GF305':138, 'GF306': 200, 'GF317': 96, 'GF323': 211, 'GF318': 208, 'GF313': 141}

# Load behaviour table with learning trials.
path = paths.adjust_path_to_host(r'/mnt/lsens-analysis/Anthony_Renard/data_processed/behavior/behavior_imagingmice_table_5days_cut_with_learning_curves.csv')
table = pd.read_csv(path)


# Load data.
# ----------

dfs = []
for mouse in mice:
    print(mouse)
    processed_dir = paths.tensor_dir
    file_name = 'tensor_xarray_learning_data.nc'
    xarray = utils_imaging.load_mouse_xarray(mouse, processed_dir, file_name, subtracted=True)
    xarray = utils_imaging.load_mouse_xarray(mouse, processed_dir, file_name, subtracted=True)
    rew_gp = database.get_mouse_reward_group_from_db(paths.db_path, mouse, db)
    
    # Select day 0. 
    xarray = xarray.sel(trial=xarray['day'] == 0)
    # Select whisker trials.
    xarray = xarray.sel(trial=xarray['whisker_stim']==1)
    
    if realigned_to_learning:
        eureka = table.loc[(table['mouse_id']==mouse) & (table['day']==0), 'learning_trial'].values[0]
        if np.isnan(eureka):
            print(f'No learning trial for mouse {mouse}. Skipping.')
            continue
        # Select trials around the learning trial.
        xarray = xarray.sel(trial=xarray['trial_w']>=eureka-30)
        xarray = xarray.sel(trial=xarray['trial_w']<=eureka+30)
        xarray.coords['trial_w'] = xarray['trial_w'] - eureka
    
    # Average time bin.
    xarray = xarray.sel(time=slice(win[0], win[1])).mean(dim='time')
    
    # # Select positive LMI cells.
    # lmi_pos = lmi_df.loc[(lmi_df.mouse_id==mouse) & (lmi_df.lmi_p>=0.095), 'roi']
    # xa_pos = xarray.sel(cell=xarray['roi'].isin(lmi_pos))
    # print(xa_pos.shape)
    
    # # Select negative LMI cells.
    # lmi_neg = lmi_df.loc[(lmi_df.mouse_id==mouse) & (lmi_df.lmi_p<=0.025), 'roi']
    # xa_neg = xarray.sel(cell=xarray['roi'].isin(lmi_neg))
    # print(xa_neg.shape)
    
    # xa_pos.name = 'activity'
    # xa_neg.name = 'activity'
    # df_pos = xa_pos.to_dataframe().reset_index()
    # df_pos['lmi'] = 'positive'
    # df_neg = xa_neg.to_dataframe().reset_index()
    # df_neg['lmi'] = 'negative'
    # df = pd.concat([df_pos, df_neg])
    # df['mouse_id'] = mouse
    # df['reward_group'] = rew_gp
    # dfs.append(df)
    # Select all cells.
    
    # Don't select cells.
    xarray.name = 'activity'
    df = xarray.to_dataframe().reset_index()
    df['mouse_id'] = mouse
    df['reward_group'] = rew_gp
    df['lmi'] = 'positive'
    dfs.append(df)
        
dfs = pd.concat(dfs)


# # Select specific group of cells.
# # -------------------------------

# roc_df = os.path.join(paths.processed_dir, 'roc_stimvsbaseline_results.csv')
# roc_df = pd.read_csv(roc_df)

# # Create a dataframe with columns 'mouse_id' and 'roi' for ROIs significant at day 0 but not at day -2 or -1

# # Find ROIs significant at day 0 (roc_p >= 0.95)
# day0_sig = roc_df[(roc_df['day'] == 0) & (roc_df['roc_p'] >= 0.95)][['mouse_id', 'roi']]

# # Find ROIs significant at day -2 or -1 (roc_p <= 0.05)
# neg_sig = roc_df[(roc_df['day'].isin([-2, -1])) & (roc_df['roc_p'] >= 0.95)][['mouse_id', 'roi']]

# # Merge to find ROIs in day0_sig but not in neg_sig for each mouse
# merged = pd.merge(
#     day0_sig,
#     neg_sig,
#     on=['mouse_id', 'roi'],
#     how='left',
#     indicator=True
# )
# day0_only_df = merged[merged['_merge'] == 'left_only'][['mouse_id', 'roi']].reset_index(drop=True)

# dfs = dfs.merge(day0_sig, on=['mouse_id', 'roi'], how='inner')



# Plot behavior and activity during day 0.
# ----------------------------------------


def preprocess_activity_and_behavior(df, table, mice_gp, cluster_id, realigned_to_learning, rolling_window=10):
    """
    Preprocess activity and behavior data for plotting.
    Returns:
        data: activity dataframe (smoothed, filtered by cluster and mice)
        data_behav: behavior dataframe (filtered by mice and optionally realigned)
    """
    
    # Select mice group.
    data = df.loc[df['mouse_id'].isin(mice_gp)]
    # Select roi in cluster_id.
    if cluster_id is not None:
        data = data[data['cluster_id'] == cluster_id]
    data = data.groupby(['mouse_id', 'lmi', 'reward_group', 'trial_w', 'cell_type', 'roi'])[['activity']].agg('mean').reset_index()
    data = data.groupby(['mouse_id', 'lmi', 'reward_group', 'trial_w'])[['activity']].agg('mean').reset_index()
    # Smooth activity with a rolling window.
    data['activity_smoothed'] = data.groupby(['mouse_id', 'lmi', 'reward_group'])['activity'].transform(
        lambda x: x.rolling(rolling_window, center=True).mean()
    )

    data_behav = table.loc[table['mouse_id'].isin(mice_gp) & (table['day'] == 0)].copy()
    if realigned_to_learning:
        # Reindex trial_w with learning trial for each session.
        data_behav['trial_w'] = data_behav['trial_w'] - data_behav['learning_trial']
        # Select trials around learning trial.
        data_behav = data_behav.loc[(data_behav['trial_w'] >= -30) & (data_behav['trial_w'] <= 60)]

    # Cut data to 100 trials.
    if not realigned_to_learning:
        data_behav = data_behav.loc[data_behav['trial_w'] <= 60]
        data = data.loc[data['trial_w'] <= 60]

    return data, data_behav



def plot_gradual_learning(data, data_behav, reward_palette, stim_palette, realigned_to_learning, mice_gp_label, output_dir, cluster_id=None):
    """
    Plot gradual learning: behavioral performance and activity for LMI positive/negative cells.
    """
    sns.set_theme(context='paper', style='ticks', palette='deep', font='sans-serif', font_scale=2)
    
    fig, axes = plt.subplots(2, 2, dpi=300, figsize=(15, 15))

    # Plot behavioral performance and chance
    for i, group in enumerate(['R+', 'R-']):
        sns.lineplot(
            data=data_behav.loc[data_behav.reward_group == group],
            x='trial_w', y='learning_curve_w',
            color=reward_palette[1 if group == 'R+' else 0],
            ax=axes[0, i]
        )
        sns.lineplot(
            data=data_behav.loc[data_behav.reward_group == group],
            x='trial_w', y='learning_curve_chance',
            color=stim_palette[2],
            ax=axes[0, i]
        )

    # Plot activity for LMI positive and negative cells
    for i, group in enumerate(['R+', 'R-']):
        sns.lineplot(
            data=data.loc[(data.reward_group == group)],
            x='trial_w', y='activity_smoothed',
            color=reward_palette[1 if group == 'R+' else 0],
            ax=axes[1, i]
        )

    axes[0, 0].set_title('R+')
    axes[0, 1].set_title('R-')


    axes[0, 0].set_ylabel('Day 0 performance')
    for ax in axes.flatten()[2:]:
        ax.set_ylabel('Activity (dF/F0)')
    for ax in axes.flatten()[:4]:
        ax.set_xlabel('')
    for ax in axes.flatten()[4:]:
        ax.set_xlabel('Whisker trials')

    axes[0, 0].set_ylim(0, 1)
    axes[0, 1].set_ylim(0, 1)
    for ax in axes.flatten()[2:]:
        ax.set_ylim(0, 0.12)
    sns.despine()

    # Save plot
    if cluster_id is not None:
        cluster_str = f"_cluster{cluster_id}"
    else:
        cluster_str = "_allcells"
    svg_file = (
        f'gradual_change{cluster_str}_realigned_{mice_gp_label}.svg'
        if realigned_to_learning else
        f'gradual_change{cluster_str}_{mice_gp_label}.svg'
    )
    plt.savefig(os.path.join(output_dir, svg_file), format='svg', dpi=300)



# Load clustering results.
clustering_df = "/mnt/lsens-analysis/Anthony_Renard/data_processed/clustering/cluster_id_day0_learning_nclust_3.csv"
clustering_df = paths.adjust_path_to_host(clustering_df)
clust_df = pd.read_csv(clustering_df)
dfs = dfs.merge(clust_df[['mouse_id', 'roi', 'cluster_id']], on=['mouse_id', 'roi'], how='inner')

# # drop columns
# dfs = dfs.drop(columns=['cluster_id_x', 'cluster_id_y'])

# mice_gp = mice_groups['good_day0']  # Select mice group for plotting.
# mice_gp = mice_groups['meh_day0'] + mice_groups['bad_day0']  # Select mice group for plotting.
mice_gp = mice
mice_gp_label = 'allmice'
# mice_gp_label = 'goodmice'
# mice_gp_label = 'badmice'
cluster_id=2
realigned_to_learning = False
output_dir = '/mnt/lsens-analysis/Anthony_Renard/analysis_output/fast-learning/day0_learning/gradual_learning'

data, data_behav = preprocess_activity_and_behavior(
    dfs, table, mice_gp, cluster_id, realigned_to_learning, rolling_window=10)
plot_gradual_learning(data, data_behav, reward_palette, stim_palette, realigned_to_learning, mice_gp_label, output_dir, cluster_id=cluster_id)





# correlation_results = []

# # Loop through each mouse
# for mouse in mice_gp:

#     # Get behavioral performance for this mouse (trial_w, learning_curve_w)
#     perf = data_behav.loc[(data_behav['mouse_id'] == mouse) & (data_behav['whisker_stim'] == 1)][['trial_w', 'learning_curve_w']]
#     act = data.loc[(data['mouse_id'] == mouse)][['trial_w', 'activity_smoothed']]
#     # Merge with behavior
#     merged = pd.merge(perf, act, on='trial_w', how='inner')
#     if len(merged) > 5:
#         corr, pval = pearsonr(merged['learning_curve_w'], merged['activity_smoothed'])
#     else:
#         corr, pval = np.nan, np.nan
#     correlation_results.append({'mouse_id': mouse, 'spearman_r': corr, 'p_value': pval})

# correlation_df = pd.DataFrame(correlation_results)
# print(correlation_df)

# # Test if the correlation coefficients are significantly different from zero across mice,
# # split between rewarded (R+) and non-rewarded (R-) mice

# # First, add reward group info to correlation_df
# correlation_df['reward_group'] = correlation_df['mouse_id'].map(
#     lambda m: database.get_mouse_reward_group_from_db(paths.db_path, m, db)
# )

# for group in ['R+', 'R-']:
#     group_corrs = correlation_df.loc[correlation_df['reward_group'] == group, 'spearman_r'].dropna()
#     print(f"\nReward group: {group}")
#     if len(group_corrs) > 1:
#         stat, p = wilcoxon(group_corrs)
#         print(f"Wilcoxon test for correlation coefficients: stat={stat}, p={p}")
#     else:
#         print("Not enough mice for group-level significance test.")

# plt.figure(figsize=(4, 4), dpi=150)
# sns.pointplot(
#     data=correlation_df,
#     x='reward_group',
#     y='spearman_r',
#     order=['R+', 'R-'],
#     join=False,
#     color='k',
#     errorbar='sd'
# )
# sns.stripplot(
#     data=correlation_df,
#     x='reward_group',
#     y='spearman_r',
#     order=['R+', 'R-'],
#     color='gray',
#     alpha=0.7,
#     jitter=True
# )
# plt.ylabel('Spearman correlation (activity vs. performance)')
# plt.xlabel('Reward group')
# plt.title('Average correlation across mice')
# sns.despine()
# plt.tight_layout()
# plt.show()







# plt.plot(data.loc[data.mouse_id=='GF305'].outcome_w)

# d = dfs.loc[dfs.mouse_id=='GF306']
# plt.scatter(x=d.loc[d.lick_flag==0]['trial_w'],y=d.loc[d.lick_flag==0]['outcome_w']-0.15)
# plt.scatter(x=d.loc[d.lick_flag==1]['trial_w'],y=d.loc[d.lick_flag==1]['outcome_w']-1.15)


# # Remove mice that licked to first whisker stim.
# mice_to_keep = dfs.loc[(dfs.trial_w==1.0) & (dfs.outcome_w==0.0), 'mouse_id'].unique()
# # mice_to_keep = mice_to_keep[-6:]
# data = dfs.loc[(dfs.mouse_id.isin(mice_to_keep))]
# data = data.loc[(data.outcome_w==1.0)]
# data = data.groupby(['mouse_id', 'lmi', 'cell', 'reward_group', 'trial'])['activity'].agg('mean').reset_index()
# data = data.loc[(data.reward_group=='R+') & (data.lmi=='positive')]
# data = data.loc[data.trial<100]
# sns.lineplot(data=data, x='trial', y='activity', palette=cell_types_palette)

# dfs.mouse_id.unique()












# #############################################################################
# PSTH during learning.
# #############################################################################

# Parameters.
sampling_rate = 30
win_psth = (-0.5, 1.5)
win_bin = (0, 0.180)
baseline_win = (0, 1)
baseline_win = (int(baseline_win[0] * sampling_rate), int(baseline_win[1] * sampling_rate))
days = [0]
days_str = ['0']

_, _, mice, db = database.select_sessions_from_db(paths.db_path,
                                            paths.nwb_dir,
                                            two_p_imaging='yes',
                                            experimenters=['AR', 'GF', 'MI'])


# Load the data.
# --------------

psth = []
for mouse_id in mice:
    reward_group = database.get_mouse_reward_group_from_db(paths.db_path, mouse_id)

    file_name = 'tensor_xarray_learning_data.nc'
    folder = paths.tensor_dir
    xarr = utils_imaging.load_mouse_xarray(mouse_id, folder, file_name, subtracted=True)
    xarr = utils_imaging.load_mouse_xarray(mouse_id, folder, file_name, subtracted=True)

    # Select days.
    xarr = xarr.sel(trial=xarr['day'].isin(days))
    # Select PSTH trace length.
    xarr = xarr.sel(time=slice(win_psth[0], win_psth[1]))
    # Select whisker trials.
    xarr_wh = xarr.sel(trial=xarr['whisker_stim']==1)
    # Average trials per days.
    xarr_wh = xarr_wh.groupby('day').mean(dim='trial')
    
    # Select auditory trials.
    xarr_aud = xarr.sel(trial=xarr['auditory_stim']==1)
    # Average trials per days.
    xarr_aud = xarr_aud.groupby('day').mean(dim='trial')
    
    # Select no stim trials.
    xarr_ns = xarr.sel(trial=xarr['no_stim']==1)
    # Average trials per days.
    xarr_ns = xarr_ns.groupby('day').mean(dim='trial')
    # 
    xarr_wh.name = 'psth'
    xarr_wh = xarr_wh.to_dataframe().reset_index()
    xarr_wh['mouse_id'] = mouse_id
    xarr_wh['reward_group'] = reward_group
    xarr_wh['stim'] = 'whisker'
    psth.append(xarr_wh)
    
    xarr_aud.name = 'psth'
    xarr_aud = xarr_aud.to_dataframe().reset_index()
    xarr_aud['mouse_id'] = mouse_id
    xarr_aud['reward_group'] = reward_group
    xarr_aud['stim'] = 'auditory'
    psth.append(xarr_aud)
    
    xarr_ns.name = 'psth'
    xarr_ns = xarr_ns.to_dataframe().reset_index()
    xarr_ns['mouse_id'] = mouse_id
    xarr_ns['reward_group'] = reward_group
    xarr_ns['stim'] = 'no_stim'
    psth.append(xarr_ns)

psth = pd.concat(psth)


# PSTH for the three stimuli.
# ###########################

variance = 'mice'  # 'mice' or 'cells'

if variance == "mice":
    min_cells = 3
    data = utils_imaging.filter_data_by_cell_count(psth, min_cells)
    data = utils_imaging.filter_data_by_cell_count(psth, min_cells)
    data = data.groupby(['mouse_id', 'day', 'reward_group', 'time', 'cell_type', 'stim'])['psth'].agg('mean').reset_index()
    data_bin = data.loc[(data.time>=win_bin[0]) & (data.time<=win_bin[1])]
    data_bin = data_bin.groupby(['mouse_id', 'day', 'reward_group', 'cell_type', 'stim'])['psth'].agg('mean').reset_index()
else:
    data = psth.groupby(['mouse_id', 'day', 'reward_group', 'time', 'cell_type', 'roi', 'stim'])['psth'].agg('mean').reset_index()
    data_bin = data.loc[(data.time>=win_bin[0]) & (data.time<=win_bin[1])]
    data_bin = data_bin.groupby(['mouse_id', 'day', 'reward_group', 'cell_type', 'roi', 'stim'])['psth'].agg('mean').reset_index()
# Convert data to percent dF/F0.
data['psth'] = data['psth'] * 100


# Plot for all cells.
# -------------------

fig, axes = plt.subplots(3, figsize=(3, 8), sharey=True)

# Whisker stim.
d = data.loc[(data['stim'] == 'whisker')]    
sns.lineplot(data=d, x='time', y='psth', errorbar='ci', hue='reward_group',
            hue_order=['R-', 'R+'], palette=reward_palette, estimator='mean', ax=axes[0], legend=False)
axes[0].axvline(0, color='#FF9600', linestyle='--')

# Auditory stim.
d = data.loc[(data['stim'] == 'auditory')]    
sns.lineplot(data=d, x='time', y='psth', errorbar='ci', hue='reward_group',
            hue_order=['R-', 'R+'], palette=reward_palette, estimator='mean', ax=axes[1], legend=False)
axes[1].axvline(0, color='#1f77b4', linestyle='--')

# No stim.
d = data.loc[(data['stim'] == 'no_stim')]    
sns.lineplot(data=d, x='time', y='psth', errorbar='ci', hue='reward_group',
            hue_order=['R-', 'R+'], palette=reward_palette, estimator='mean', ax=axes[2], legend=False)
axes[2].axvline(0, color='#333333', linestyle='--')

axes[0].set_title('All cells')
axes[0].set_ylabel('Whisker stim')
axes[1].set_ylabel('Auditory stim')
axes[2].set_ylabel('No stim')
plt.ylim(-1, 12)
sns.despine()

# Save figure for all cells.
output_dir = '/mnt/lsens-analysis/Anthony_Renard/analysis_output/fast-learning/day0_learning/psth'
output_dir = paths.adjust_path_to_host(output_dir)
svg_file = f'psth_day0_three_stim_on_{variance}.svg'
plt.savefig(os.path.join(output_dir, svg_file), format='svg', dpi=300)


# Plot for projectors.
# --------------------

cell_types = ['wM1', 'wS2']
fig, axes = plt.subplots(3, 2, figsize=(6, 8), sharey=True)

# Whisker stim.
for j, ct in enumerate(cell_types):
    if ct == 'all':
        d = data.loc[(data['stim'] == 'whisker')]    
    else:
        d = data.loc[(data['cell_type'] == ct) & (data['stim'] == 'whisker')]    
    sns.lineplot(data=d, x='time', y='psth', errorbar='ci', hue='reward_group',
                hue_order=['R-', 'R+'], palette=reward_palette, estimator='mean', ax=axes[0, j], legend=False)
    axes[0, j].axvline(0, color='#FF9600', linestyle='--')

# Auditory stim.
for j, ct in enumerate(cell_types):
    if ct == 'all':
        d = data.loc[(data['stim'] == 'auditory')]    
    else:
        d = data.loc[(data['cell_type'] == ct) & (data['stim'] == 'auditory')]    
    sns.lineplot(data=d, x='time', y='psth', errorbar='ci', hue='reward_group',
                hue_order=['R-', 'R+'], palette=reward_palette, estimator='mean', ax=axes[1, j], legend=False)
    axes[1, j].axvline(0, color='#1f77b4', linestyle='--')

# No stim.
for j, ct in enumerate(cell_types):
    if ct == 'all':
        d = data.loc[(data['stim'] == 'no_stim')]    
    else:
        d = data.loc[(data['cell_type'] == ct) & (data['stim'] == 'no_stim')]    
    sns.lineplot(data=d, x='time', y='psth', errorbar='ci', hue='reward_group',
                hue_order=['R-', 'R+'], palette=reward_palette, estimator='mean', ax=axes[2, j], legend=False)
    axes[2, j].axvline(0, color='#333333', linestyle='--')

axes[0, 0].set_title('wM1 projectors')
axes[0, 1].set_title('wS2 projectors')
axes[0, 0].set_ylabel('Whisker stim')
axes[1, 0].set_ylabel('Auditory stim')
axes[2, 0].set_ylabel('No stim')
plt.ylim(-1, 20)
sns.despine()

# Save figure for all cells.
output_dir = '/mnt/lsens-analysis/Anthony_Renard/analysis_output/fast-learning/day0_learning/psth'
output_dir = paths.adjust_path_to_host(output_dir)
svg_file = f'psth_day0_three_stim_projectors_on_{variance}.svg'
plt.savefig(os.path.join(output_dir, svg_file), format='svg', dpi=300)


















# #############################################################################
# PSTH during learning. Early vs last trials.
# #############################################################################

# Parameters.
sampling_rate = 30
win_psth = (-1, 1.5)
win_bin = (0, 0.300)
baseline_win = (0, .5)
baseline_win = (int(baseline_win[0] * sampling_rate), int(baseline_win[1] * sampling_rate))
days = [0]
days_str = ['0']
early_trials = range(0, 20)
late_trials = range(30, 100)  # Last trials of the session.
variance = "mice"
_, _, mice, db = database.select_sessions_from_db(paths.db_path,
                                            paths.nwb_dir,
                                            two_p_imaging='yes',
                                            experimenters=['AR', 'GF', 'MI'])


# Load the data.
# --------------

psth = []
for mouse_id in mice:
    reward_group = database.get_mouse_reward_group_from_db(paths.db_path, mouse_id)

    file_name = 'tensor_xarray_learning_data.nc'
    folder = paths.tensor_dir
    xarr = utils_imaging.load_mouse_xarray(mouse_id, folder, file_name)
    xarr = utils_imaging.load_mouse_xarray(mouse_id, folder, file_name)

    # Select days.
    xarr = xarr.sel(trial=xarr['day'].isin(days))
    # Select PSTH trace length.
    xarr = xarr.sel(time=slice(win_psth[0], win_psth[1]))
    
    # Select whisker trials.
    xarr_wh = xarr.sel(trial=xarr['whisker_stim']==1)
    # Average trials per days.
    xarr_wh_early = xarr_wh.sel(trial=xarr_wh['trial_w'].isin(early_trials))
    xarr_wh_late = xarr_wh.sel(trial=xarr_wh['trial_w'].isin(late_trials))
    xarr_wh_early = xarr_wh_early.groupby('day').mean(dim='trial')
    xarr_wh_late = xarr_wh_late.groupby('day').mean(dim='trial')
    
    xarr_wh_early.name = 'activity'
    xarr_wh_early = xarr_wh_early.to_dataframe().reset_index()
    xarr_wh_early['mouse_id'] = mouse_id
    xarr_wh_early['reward_group'] = reward_group
    xarr_wh_early['stim'] = 'whisker'
    xarr_wh_early['epoch'] = 'early'
    psth.append(xarr_wh_early)
    
    xarr_wh_late.name = 'activity'
    xarr_wh_late = xarr_wh_late.to_dataframe().reset_index()
    xarr_wh_late['mouse_id'] = mouse_id
    xarr_wh_late['reward_group'] = reward_group
    xarr_wh_late['stim'] = 'whisker'
    xarr_wh_late['epoch'] = 'late'
    psth.append(xarr_wh_late)

psth = pd.concat(psth)


# Select specific group of cells.
# -------------------------------

# roc_df = os.path.join(paths.processed_dir, 'roc_stimvsbaseline_results.csv')
# roc_df = pd.read_csv(roc_df)

# # Create a dataframe with columns 'mouse_id' and 'roi' for ROIs significant at day 0 but not at day -2 or -1

# # Find ROIs significant at day 0 (roc_p >= 0.95)
# day0_sig = roc_df[(roc_df['day'] == 0) & (roc_df['roc_p'] >= 0.95)][['mouse_id', 'roi']]

# # Find ROIs significant at day -2 or -1 (roc_p <= 0.05)
# neg_sig = roc_df[(roc_df['day'].isin([-2, -1])) & (roc_df['roc_p'] >= 0.95)][['mouse_id', 'roi']]

# # Merge to find ROIs in day0_sig but not in neg_sig for each mouse
# merged = pd.merge(
#     day0_sig,
#     neg_sig,
#     on=['mouse_id', 'roi'],
#     how='left',
#     indicator=True
# )
# day0_only_df = merged[merged['_merge'] == 'left_only'][['mouse_id', 'roi']].reset_index(drop=True)

# psth_copy = psth.merge(day0_only_df, on=['mouse_id', 'roi'], how='inner')


# Alternatively, select clusters of interest.
clustering_df = "/mnt/lsens-analysis/Anthony_Renard/data_processed/clustering/cluster_id_day0_learning_nclust_3.csv"
clustering_df = paths.adjust_path_to_host(clustering_df)
clust_df = pd.read_csv(clustering_df)

# Select LMI

lmi_df = pd.read_csv(os.path.join(paths.processed_dir, 'lmi_results.csv'))
# Select significant LMI cells (positive and negative)
lmi_sig = lmi_df.loc[(lmi_df['lmi_p'] <= 0.025) | (lmi_df['lmi_p'] >= 0.975), ['mouse_id', 'roi']]
psth = psth.merge(lmi_sig, on=['mouse_id', 'roi'], how='inner')



def process_psth_data(psth_df, variance, min_cells=3, win_bin=(0, 0.180), cluster_id=None, select_lmi=False, lmi='both'):
    """
    Process PSTH data for early vs late trials, supporting both 'mice' and 'cell' variance.
    Returns: data, data_bin, data_proj, data_bin_proj (all with activity in percent dF/F0)
    """
    
    if cluster_id is not None:
        # Filter psth_df to include only ROIs in the specified cluster_id
        psth_df = psth_df.merge(clust_df[['mouse_id', 'roi', 'cluster_id']], on=['mouse_id', 'roi'], how='inner')
        psth_df_copy = psth_df.loc[psth_df['cluster_id'].isin([cluster_id])]  # Select clusters of interest.
    else:
        psth_df_copy = psth_df.copy()
    
    if select_lmi:
        if lmi == 'positive':
            lmi_sel = lmi_df.loc[lmi_df['lmi_p'] >= 0.95, ['mouse_id', 'roi']]
        elif lmi == 'negative':
            lmi_sel = lmi_df.loc[lmi_df['lmi_p'] <= 0.05, ['mouse_id', 'roi']]
        elif lmi == 'both':
            lmi_sel = lmi_df.loc[(lmi_df['lmi_p'] <= 0.025) | (lmi_df['lmi_p'] >= 0.975), ['mouse_id', 'roi']]
        psth_df_copy = psth_df_copy.merge(lmi_sel, on=['mouse_id', 'roi'], how='inner')

    
    if variance == "mice":
        # Filter by minimum cell count per mouse
        data = utils_imaging.filter_data_by_cell_count(psth_df_copy, min_cells)
        data = utils_imaging.filter_data_by_cell_count(psth_df_copy, min_cells)
        # Aggregate by ROI, then by mouse (mean across ROIs)
        data = data.groupby(['mouse_id', 'day', 'reward_group', 'time', 'stim', 'epoch', 'roi'])['activity'].agg('mean').reset_index()
        data = data.groupby(['mouse_id', 'day', 'reward_group', 'time', 'stim', 'epoch'])['activity'].agg('mean').reset_index()
        data_bin = data.loc[(data.time >= win_bin[0]) & (data.time <= win_bin[1])]
        data_bin = data_bin.groupby(['mouse_id', 'day', 'reward_group', 'stim', 'epoch'])['activity'].agg('mean').reset_index()

        # Projector neurons (with cell_type)
        data_proj = utils_imaging.filter_data_by_cell_count(psth_df_copy, min_cells)
        data_proj = utils_imaging.filter_data_by_cell_count(psth_df_copy, min_cells)
        data_proj = data_proj.groupby(['mouse_id', 'day', 'reward_group', 'time', 'cell_type', 'stim', 'epoch', 'roi'])['activity'].agg('mean').reset_index()
        data_proj = data_proj.groupby(['mouse_id', 'day', 'reward_group', 'time', 'cell_type', 'stim', 'epoch'])['activity'].agg('mean').reset_index()
        data_bin_proj = data_proj.loc[(data_proj.time >= win_bin[0]) & (data_proj.time <= win_bin[1])]
        data_bin_proj = data_bin_proj.groupby(['mouse_id', 'day', 'reward_group', 'cell_type', 'stim', 'epoch'])['activity'].agg('mean').reset_index()
    else:
        # No filtering, keep per-ROI
        data = psth_df_copy.groupby(['mouse_id', 'day', 'reward_group', 'time', 'roi', 'stim', 'epoch'])['activity'].agg('mean').reset_index()
        data_bin = data.loc[(data.time >= win_bin[0]) & (data.time <= win_bin[1])]
        data_bin = data_bin.groupby(['mouse_id', 'day', 'reward_group', 'roi', 'stim', 'epoch'])['activity'].agg('mean').reset_index()

        data_proj = psth_df_copy.groupby(['mouse_id', 'day', 'reward_group', 'time', 'cell_type', 'roi', 'stim', 'epoch'])['activity'].agg('mean').reset_index()
        data_bin_proj = data_proj.loc[(data_proj.time >= win_bin[0]) & (data_proj.time <= win_bin[1])]
        data_bin_proj = data_bin_proj.groupby(['mouse_id', 'day', 'reward_group', 'cell_type', 'roi', 'stim', 'epoch'])['activity'].agg('mean').reset_index()

    # Convert to percent dF/F0
    for df in [data, data_bin, data_proj, data_bin_proj]:
        df['activity'] = df['activity'] * 100

    return data, data_bin, data_proj, data_bin_proj



# # Select subgroup of mice.
# mice_gp = mice_groups['gradual_day0']
# data = data.loc[data['mouse_id'].isin(mice_gp)]
# data_bin = data_bin.loc[data_bin['mouse_id'].isin(mice_gp)]
# data_proj = data_proj.loc[data_proj['mouse_id'].isin(mice_gp)]
# data_bin_proj = data_bin_proj.loc[data_bin_proj['mouse_id'].isin(mice_gp)]


# Plot for psth all cells.
# ------------------------

def plot_psth_early_vs_late(data, early_trials, late_trials, reward_palette, output_dir, variance, cluster_id, select_lmi=False, lmi='both'):
    """
    Plot PSTH for early vs late trials for all cells, split by reward group.
    """
    fig, axes = plt.subplots(2, figsize=(3, 8), sharey=True)

    # Whisker stim.
    for idx, group in enumerate(['R+', 'R-']):
        d = data.loc[data.reward_group == group]
        sns.lineplot(
            data=d, x='time', y='activity',
            errorbar='ci', style='epoch', style_order=['late', 'early'],
            color=reward_palette[1 if group == 'R+' else 0],
            estimator='mean', legend=False, ax=axes[idx]
        )
        axes[idx].axvline(0, color='#FF9600', linestyle='--')
        axes[idx].set_ylabel(group)

    axes[0].set_title(f'All cells - early vs late trials {early_trials}-{late_trials}')
    plt.ylim(-1, 35)
    sns.despine()

    # Save figure for all cells.
    output_dir = paths.adjust_path_to_host(output_dir)
    if cluster_id is not None:
        cluster_str = f'_cluster{cluster_id}'
    else:
        cluster_str = ''
        
    if select_lmi:
        lmi_str = f'_lmi{lmi}'
    else:
        lmi_str = ''
    svg_file = f'psth_day0_earlyvslate_on_{variance}_{early_trials}_{late_trials}{cluster_str}{lmi_str}.svg'
    plt.savefig(os.path.join(output_dir, svg_file), format='svg', dpi=300)
    plt.show()


# PLot and save for each cluster
for cl_id in [None, 0, 1, 2]:
    data, data_bin, data_proj, data_bin_proj = process_psth_data(psth, variance, min_cells=3, win_bin=win_bin, cluster_id=cl_id)
    plot_psth_early_vs_late(
        data=data,
        early_trials=early_trials,
        late_trials=late_trials,
    reward_palette=reward_palette,
    output_dir='/mnt/lsens-analysis/Anthony_Renard/analysis_output/fast-learning/day0_learning/psth/early_vs_late/',
    variance=variance,
    cluster_id=cl_id
)

for lmi_type in ['positive', 'negative']:
    data, data_bin, data_proj, data_bin_proj = process_psth_data(psth, variance, min_cells=3, win_bin=win_bin, cluster_id=cl_id, select_lmi=True, lmi=lmi_type)
    plot_psth_early_vs_late(
        data=data,
        early_trials=early_trials,
        late_trials=late_trials,
    reward_palette=reward_palette,
    output_dir='/mnt/lsens-analysis/Anthony_Renard/analysis_output/fast-learning/day0_learning/psth/early_vs_late/',
    variance=variance,
    cluster_id=None,
    select_lmi=True, lmi=lmi_type
    )




# Plot for projectors.
# --------------------

cell_types = ['wM1', 'wS2']

fig, axes = plt.subplots(2,2, figsize=(6, 8), sharey=True)

for j, ct in enumerate(cell_types):
    d = data_proj.loc[(data_proj.reward_group=='R+') & (data_proj.cell_type==ct)]
    sns.lineplot(data=d, x='time', y='activity', errorbar='ci', style='epoch', style_order=['late', 'early'],
                color=reward_palette[1], estimator='mean', legend=False, ax=axes[0, j])
    d = data_proj.loc[(data_proj.reward_group=='R-') & (data_proj.cell_type==ct)]
    sns.lineplot(data=d, x='time', y='activity', errorbar='ci', style='epoch', style_order=['late', 'early'],
                color=reward_palette[0], estimator='mean', legend=False, ax=axes[1, j])
    axes[0, j].axvline(0, color='#FF9600', linestyle='--')
    axes[1, j].axvline(0, color='#FF9600', linestyle='--')

axes[0,0].set_title(f'wM1 - early vs late trials {early_trials}-{late_trials}')
axes[0,1].set_title(f'wS2 - early vs late trials {early_trials}-{late_trials}')
axes[0,0].set_ylabel('R+')
axes[1,0].set_ylabel('R-')
plt.ylim(-2, 20)
sns.despine()

# Save figure for all cells.
output_dir = '/mnt/lsens-analysis/Anthony_Renard/analysis_output/fast-learning/day0_learning/psth/early_vs_late/'
output_dir = paths.adjust_path_to_host(output_dir)
svg_file = f'psth_day0_earlyvslate_projectors_on_{variance}_{early_trials}_{late_trials}.svg'
plt.savefig(os.path.join(output_dir, svg_file), format='svg', dpi=300)


# Bar plot and stats for early vs late, for both all cells and projectors.
# ------------------------------------------------------------------------

def plot_and_save_barplot_early_vs_late(data_bin, data_bin_proj, variance, early_trials, late_trials, reward_palette, io, output_dir, cluster_id=None):
    """
    Plot and save barplots for early vs late trials for all cells and projectors.
    Also computes and saves Wilcoxon stats and exports data to CSV.
    Generates two figures: one for all cells, one for projectors.
    """
    output_dir = paths.adjust_path_to_host(output_dir)
    if cluster_id is not None:
        cluster_str = f'_cluster{cluster_id}'
        cluster_title = f" (cluster {cluster_id})"
    else:
        cluster_str = ''
        cluster_title = ''
    # --- Figure 1: All cells, split by reward group ---
    fig1, axes1 = plt.subplots(1, 2, figsize=(8, 6), sharey=True)
    for idx, group in enumerate(['R+', 'R-']):
        sns.barplot(
            data=data_bin[data_bin['reward_group'] == group],
            x='epoch', y='activity',
            errorbar='ci', order=['early', 'late'],
            color=reward_palette[1 if group == 'R+' else 0],
            estimator='mean', ax=axes1[idx]
        )
        axes1[idx].set_title(f'All cells{cluster_title}\n{group}\nEarly: {early_trials}, Late: {late_trials}')
        axes1[idx].set_ylabel('Activity (% dF/F0)')
        axes1[idx].set_xlabel('')
    plt.ylim(0, 12)
    sns.despine()
    plt.tight_layout()
    plt.show()
    svg_file1 = f'barplot_day0_earlyvslate_on_{variance}_{early_trials}_{late_trials}_allcells{cluster_str}.svg'
    plt.savefig(os.path.join(output_dir, svg_file1), format='svg', dpi=300)

    # --- Figure 2: Projectors (wS2 and wM1), split by reward group ---
    fig2, axes2 = plt.subplots(2, 2, figsize=(10, 10), sharey=True)
    for row_idx, group in enumerate(['R+', 'R-']):
        for col_idx, cell_type in enumerate(['wS2', 'wM1']):
            sns.barplot(
                data=data_bin_proj[(data_bin_proj['cell_type'] == cell_type) & (data_bin_proj['reward_group'] == group)],
                x='epoch', y='activity',
                errorbar='ci', order=['early', 'late'],
                color=reward_palette[1 if group == 'R+' else 0],
                estimator='mean', ax=axes2[row_idx, col_idx]
            )
            axes2[row_idx, col_idx].set_title(f'{cell_type} projectors{cluster_title}\n{group}\nEarly: {early_trials}, Late: {late_trials}')
            axes2[row_idx, col_idx].set_ylabel('Activity (% dF/F0)')
            axes2[row_idx, col_idx].set_xlabel('')
    plt.ylim(0, 16)
    sns.despine()
    plt.tight_layout()
    plt.show()
    svg_file2 = f'barplot_day0_earlyvslate_on_{variance}_{early_trials}_{late_trials}_projectors{cluster_str}.svg'
    plt.savefig(os.path.join(output_dir, svg_file2), format='svg', dpi=300)

    # Stats.
    stats_results = []

    # 1. All cells (ignore cell_type)
    barplot_data = data_bin.groupby(['mouse_id', 'reward_group', 'epoch'])['activity'].mean().reset_index()
    for group in barplot_data['reward_group'].unique():
        d = barplot_data[barplot_data['reward_group'] == group]
        d_all = d.groupby(['mouse_id', 'epoch'])['activity'].mean().reset_index()
        early = d_all[d_all['epoch'] == 'early'].set_index('mouse_id')['activity']
        late = d_all[d_all['epoch'] == 'late'].set_index('mouse_id')['activity']
        common = early.index.intersection(late.index)
        if len(common) > 0:
            stat, p = wilcoxon(early.loc[common], late.loc[common])
        else:
            stat, p = np.nan, np.nan
        stats_results.append({'reward_group': group, 'cell_type': 'all', 'wilcoxon_stat': stat, 'p_value': p})

    # 2. wS2 and 3. wM1
    barplot_data_proj = data_bin_proj.groupby(['mouse_id', 'reward_group', 'epoch', 'cell_type'])['activity'].mean().reset_index()
    for cell_type in ['wS2', 'wM1']:
        for group in barplot_data_proj['reward_group'].unique():
            d = barplot_data_proj[(barplot_data_proj['reward_group'] == group) & (barplot_data_proj['cell_type'] == cell_type)]
            early = d[d['epoch'] == 'early'].set_index('mouse_id')['activity']
            late = d[d['epoch'] == 'late'].set_index('mouse_id')['activity']
            common = early.index.intersection(late.index)
            if len(common) > 0:
                stat, p = wilcoxon(early.loc[common], late.loc[common])
            else:
                stat, p = np.nan, np.nan
            stats_results.append({'reward_group': group, 'cell_type': cell_type, 'wilcoxon_stat': stat, 'p_value': p})

    stats_df = pd.DataFrame(stats_results)
    print(stats_df)

    # Save barplot data and stats to CSV
    barplot_data.to_csv(os.path.join(output_dir, f'barplot_day0_earlyvslate_on_{variance}_{early_trials}_{late_trials}_data{cluster_str}.csv'), index=False)
    stats_df.to_csv(os.path.join(output_dir, f'barplot_day0_earlyvslate_on_{variance}_{early_trials}_{late_trials}_stats{cluster_str}.csv'), index=False)















# #############################################################################
# PSTH during learning. Hit VS miss trials.
# #############################################################################

# Parameters.
sampling_rate = 30
win_psth = (-0.5, 4)
win_bin = (0, 0.180)
days = [0]
days_str = ['0']
_, _, mice, db = database.select_sessions_from_db(paths.db_path,
                                            paths.nwb_dir,
                                            two_p_imaging='yes',
                                            experimenters=['AR', 'GF', 'MI'])


# Load the data.
# --------------

psth = []
for mouse_id in mice:
    reward_group = database.get_mouse_reward_group_from_db(paths.db_path, mouse_id)

    file_name = 'tensor_xarray_learning_data.nc'
    folder = paths.tensor_dir
    xarr = utils_imaging.load_mouse_xarray(mouse_id, folder, file_name)
    xarr = utils_imaging.load_mouse_xarray(mouse_id, folder, file_name)

    # Select days.
    xarr = xarr.sel(trial=xarr['day'].isin(days))
    # Select PSTH trace length.
    xarr = xarr.sel(time=slice(win_psth[0], win_psth[1]))
    
    # Select whisker trials.
    xarr_wh = xarr.sel(trial=xarr['whisker_stim']==1)
    # Average trials per days.
    xarr_wh_hit = xarr_wh.sel(trial=xarr_wh['outcome_w']==1)
    xarr_wh_miss = xarr_wh.sel(trial=xarr_wh['outcome_w']==0)
    xarr_wh_hit = xarr_wh_hit.groupby('day').mean(dim='trial')
    xarr_wh_miss = xarr_wh_miss.groupby('day').mean(dim='trial')
    
    xarr_wh_hit.name = 'activity'
    xarr_wh_hit = xarr_wh_hit.to_dataframe().reset_index()
    xarr_wh_hit['mouse_id'] = mouse_id
    xarr_wh_hit['reward_group'] = reward_group
    xarr_wh_hit['stim'] = 'whisker'
    xarr_wh_hit['outcome'] = 'hit'
    psth.append(xarr_wh_hit)
    
    xarr_wh_miss.name = 'activity'
    xarr_wh_miss = xarr_wh_miss.to_dataframe().reset_index()
    xarr_wh_miss['mouse_id'] = mouse_id
    xarr_wh_miss['reward_group'] = reward_group
    xarr_wh_miss['stim'] = 'whisker'
    xarr_wh_miss['outcome'] = 'miss'
    psth.append(xarr_wh_miss)
   
    # Select no stim trials.
    xarr_nt = xarr.sel(trial=xarr['no_stim']==1)
    
    # Check if that nouse has FA trials.
    if xarr_nt.sel(trial=xarr_nt['outcome_c']==1).size == 0:
        # If no FA trials, we can skip this mouse.
        continue
    # Average trials per days.
    xarr_nt_hit = xarr_nt.sel(trial=xarr_nt['outcome_c']==1)
    xarr_nt_miss = xarr_nt.sel(trial=xarr_nt['outcome_c']==0)
    xarr_nt_hit = xarr_nt_hit.groupby('day').mean(dim='trial')
    xarr_nt_miss = xarr_nt_miss.groupby('day').mean(dim='trial')
    
    xarr_nt_hit.name = 'activity'
    xarr_nt_hit = xarr_nt_hit.to_dataframe().reset_index()
    xarr_nt_hit['mouse_id'] = mouse_id
    xarr_nt_hit['reward_group'] = reward_group
    xarr_nt_hit['stim'] = 'no_stim'
    xarr_nt_hit['outcome'] = 'hit'
    psth.append(xarr_nt_hit)
    
    xarr_nt_miss.name = 'activity'
    xarr_nt_miss = xarr_nt_miss.to_dataframe().reset_index()
    xarr_nt_miss['mouse_id'] = mouse_id
    xarr_nt_miss['reward_group'] = reward_group
    xarr_nt_miss['stim'] = 'no_stim'
    xarr_nt_miss['outcome'] = 'miss'
    psth.append(xarr_nt_miss)

psth = pd.concat(psth)


variance = 'cell'  # 'mice' or 'cells'
if variance == "mice":
    min_cells = 3
    data = utils_imaging.filter_data_by_cell_count(psth, min_cells)
    data = utils_imaging.filter_data_by_cell_count(psth, min_cells)
    data = data.groupby(['mouse_id', 'day', 'reward_group', 'time', 'cell_type', 'stim', 'outcome'])['activity'].agg('mean').reset_index()
    data_bin = data.loc[(data.time>=win_bin[0]) & (data.time<=win_bin[1])]
    data_bin = data_bin.groupby(['mouse_id', 'day', 'reward_group', 'cell_type', 'stim', 'outcome'])['activity'].agg('mean').reset_index()
else:
    data = psth.groupby(['mouse_id', 'day', 'reward_group', 'time', 'cell_type', 'roi', 'stim', 'outcome'])['activity'].agg('mean').reset_index()
    data_bin = data.loc[(data.time>=win_bin[0]) & (data.time<=win_bin[1])]
    data_bin = data_bin.groupby(['mouse_id', 'day', 'reward_group', 'cell_type', 'roi', 'stim', 'outcome'])['activity'].agg('mean').reset_index()
# Convert data to percent dF/F0.
data['activity'] = data['activity'] * 100


# Plot for psth all cells.
# ------------------------

fig, axes = plt.subplots(2, 2, figsize=(6, 8), sharey=True)

# Whisker stim.
d = data.loc[(data.reward_group=='R+') & (data.stim=='whisker')]
sns.lineplot(data=d, x='time', y='activity',
            errorbar='ci', style='outcome', style_order=['hit', 'miss'],
            color=reward_palette[1], estimator='mean', legend=False, ax=axes[0,0])
d = data.loc[(data.reward_group=='R-') & (data.stim=='whisker')]
sns.lineplot(data=d, x='time', y='activity',
            errorbar='ci', style='outcome', style_order=['hit', 'miss'],
            color=reward_palette[0], estimator='mean', legend=False, ax=axes[1,0])

d = data.loc[(data.reward_group=='R+') & (data.stim=='no_stim')]
sns.lineplot(data=d, x='time', y='activity',
            errorbar='ci', style='outcome', style_order=['hit', 'miss'],
            color=reward_palette[1], estimator='mean', legend=False, ax=axes[0,1])
d = data.loc[(data.reward_group=='R-') & (data.stim=='no_stim')]
sns.lineplot(data=d, x='time', y='activity',
            errorbar='ci', style='outcome', style_order=['hit', 'miss'],
            color=reward_palette[0], estimator='mean', legend=False, ax=axes[1,1])

for ax in axes.flat:
    ax.axvline(0, color='#FF9600', linestyle='--')

axes[0,0].set_title('Whisker stim')
axes[0,1].set_title('No stim')
axes[0,0].set_ylabel('R+')
axes[1,0].set_ylabel('R-')
plt.ylim(-1, 12)
sns.despine()

# Save figure for all cells.
output_dir = '/mnt/lsens-analysis/Anthony_Renard/analysis_output/fast-learning/day0_learning/psth/hit_vs_miss/'
output_dir = paths.adjust_path_to_host(output_dir)
svg_file = f'psth_day0_hitvsmiss_on_{variance}.svg'
plt.savefig(os.path.join(output_dir, svg_file), format='svg', dpi=300)


# Plot for projectors.
# --------------------
cell_types = ['wM1', 'wS2']
stims = ['whisker', 'no_stim']
stim_titles = {'whisker': 'Whisker stim', 'no_stim': 'No stim'}

fig, axes = plt.subplots(2, 4, figsize=(12, 8), sharey=True)

for j, ct in enumerate(cell_types):
    for k, stim in enumerate(stims):
        # R+
        d = data.loc[
            (data.reward_group == 'R+') &
            (data.cell_type == ct) &
            (data.stim == stim)
        ]
        sns.lineplot(
            data=d, x='time', y='activity',
            errorbar='ci', style='outcome', style_order=['hit', 'miss'],
            color=reward_palette[1], estimator='mean', legend=False, ax=axes[0, j*2 + k]
        )
        axes[0, j*2 + k].axvline(0, color='#FF9600', linestyle='--')
        axes[0, j*2 + k].set_title(f"{ct} - {stim_titles[stim]}")
        # R-
        d = data.loc[
            (data.reward_group == 'R-') &
            (data.cell_type == ct) &
            (data.stim == stim)
        ]
        sns.lineplot(
            data=d, x='time', y='activity',
            errorbar='ci', style='outcome', style_order=['hit', 'miss'],
            color=reward_palette[0], estimator='mean', legend=False, ax=axes[1, j*2 + k]
        )
        axes[1, j*2 + k].axvline(0, color='#FF9600', linestyle='--')

axes[0, 0].set_ylabel('R+')
axes[1, 0].set_ylabel('R-')
plt.ylim(-2, 20)
sns.despine()

# Save figure for projector neurons.
output_dir = '/mnt/lsens-analysis/Anthony_Renard/analysis_output/fast-learning/day0_learning/psth/hit_vs_miss/'
output_dir = paths.adjust_path_to_host(output_dir)
svg_file = f'psth_day0_hitvsmiss_projectors_on_{variance}.svg'
plt.savefig(os.path.join(output_dir, svg_file), format='svg', dpi=300)


# Bar plot and stats for hit vs miss, for both whisker and no_stim.
# -----------------------------------------------------------------
for stim in ['whisker', 'no_stim']:
    stim_title = 'Whisker stim' if stim == 'whisker' else 'No stim'
    fig, axes = plt.subplots(2, 3, figsize=(9, 8), sharey=True)

    # All cells
    sns.barplot(
        data=data.loc[(data.reward_group == 'R+') & (data.stim == stim)],
        x='outcome', y='activity',
        errorbar='ci', order=['hit', 'miss'],
        color=reward_palette[1], estimator='mean', ax=axes[0, 0]
    )
    sns.barplot(
        data=data.loc[(data.reward_group == 'R-') & (data.stim == stim)],
        x='outcome', y='activity',
        errorbar='ci', order=['hit', 'miss'],
        color=reward_palette[0], estimator='mean', ax=axes[1, 0]
    )
    axes[0, 0].set_title('All cells')
    axes[0, 0].set_ylabel('R+')
    axes[1, 0].set_ylabel('R-')

    # wS2 projectors
    sns.barplot(
        data=data.loc[(data.reward_group == 'R+') & (data.cell_type == 'wS2') & (data.stim == stim)],
        x='outcome', y='activity',
        errorbar='ci', order=['hit', 'miss'],
        color=reward_palette[1], estimator='mean', ax=axes[0, 1]
    )
    sns.barplot(
        data=data.loc[(data.reward_group == 'R-') & (data.cell_type == 'wS2') & (data.stim == stim)],
        x='outcome', y='activity',
        errorbar='ci', order=['hit', 'miss'],
        color=reward_palette[0], estimator='mean', ax=axes[1, 1]
    )
    axes[0, 1].set_title('wS2 projectors')

    # wM1 projectors
    sns.barplot(
        data=data.loc[(data.reward_group == 'R+') & (data.cell_type == 'wM1') & (data.stim == stim)],
        x='outcome', y='activity',
        errorbar='ci', order=['hit', 'miss'],
        color=reward_palette[1], estimator='mean', ax=axes[0, 2]
    )
    sns.barplot(
        data=data.loc[(data.reward_group == 'R-') & (data.cell_type == 'wM1') & (data.stim == stim)],
        x='outcome', y='activity',
        errorbar='ci', order=['hit', 'miss'],
        color=reward_palette[0], estimator='mean', ax=axes[1, 2]
    )
    axes[0, 2].set_title('wM1 projectors')

    for ax in axes.flat:
        ax.set_xlabel('')

    plt.suptitle(f'Hit vs Miss: {stim_title}')
    plt.ylim(0, 8)
    sns.despine()
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    # Save figure for all cells and projectors.
    output_dir = '/mnt/lsens-analysis/Anthony_Renard/analysis_output/fast-learning/day0_learning/psth/hit_vs_miss/'
    output_dir = paths.adjust_path_to_host(output_dir)
    svg_file = f'barplot_day0_hitvsmiss_{stim}_on_{variance}_all_wS2_wM1.svg'
    plt.savefig(os.path.join(output_dir, svg_file), format='svg', dpi=300)
    plt.close(fig)

    # Stats.
    # Prepare data for stats and export
    barplot_data = data.loc[data.stim == stim].groupby(['mouse_id', 'reward_group', 'cell_type', 'outcome'])['activity'].mean().reset_index()

    stats_results = []

    # 1. All cells (ignore cell_type)
    for group in barplot_data['reward_group'].unique():
        d = barplot_data[barplot_data['reward_group'] == group]
        # Average across cell types for each mouse and outcome
        d_all = d.groupby(['mouse_id', 'outcome'])['activity'].mean().reset_index()
        hit = d_all[d_all['outcome'] == 'hit'].set_index('mouse_id')['activity']
        miss = d_all[d_all['outcome'] == 'miss'].set_index('mouse_id')['activity']
        common = hit.index.intersection(miss.index)
        if len(common) > 0:
            stat, p = wilcoxon(hit.loc[common], miss.loc[common])
        else:
            stat, p = np.nan, np.nan
        stats_results.append({'stim': stim, 'reward_group': group, 'cell_type': 'all', 'wilcoxon_stat': stat, 'p_value': p})

    # 2. wS2 and 3. wM1
    for cell_type in ['wS2', 'wM1']:
        for group in barplot_data['reward_group'].unique():
            d = barplot_data[(barplot_data['reward_group'] == group) & (barplot_data['cell_type'] == cell_type)]
            hit = d[d['outcome'] == 'hit'].set_index('mouse_id')['activity']
            miss = d[d['outcome'] == 'miss'].set_index('mouse_id')['activity']
            common = hit.index.intersection(miss.index)
            if len(common) > 0:
                stat, p = wilcoxon(hit.loc[common], miss.loc[common])
            else:
                stat, p = np.nan, np.nan
            stats_results.append({'stim': stim, 'reward_group': group, 'cell_type': cell_type, 'wilcoxon_stat': stat, 'p_value': p})

    stats_df = pd.DataFrame(stats_results)

    # Save barplot data and stats to CSV
    barplot_data.to_csv(os.path.join(output_dir, f'barplot_day0_hitvsmiss_{stim}_on_{variance}_all_wS2_wM1_data.csv'), index=False)
    stats_df.to_csv(os.path.join(output_dir, f'barplot_day0_hitvsmiss_{stim}_on_{variance}_all_wS2_wM1_stats.csv'), index=False)








# #############################################################################
# Day 0 modulation index.
# #############################################################################

# Look for cells active during the first whisk trials of the 
# session but stop firing at some point and cells initially silent that became
# active by the end.
# I expect the rewarded group to have less cells that stop firing which are
# consolidated by reward.


sampling_rate = 30
win = (0, 0.300)  # from stimulus onset to 300 ms after.
win_length = f'{int(np.round((win[1]-win[0]) * 1000))}'  # for file naming.
# win = (int(win[0] * sampling_rate), int(win[1] * sampling_rate))
baseline_win = (-1, 0)
baseline_win = (int(baseline_win[0] * sampling_rate), int(baseline_win[1] * sampling_rate))
days_str = ['-2', '-1', '0', '+1', '+2']
subtract_baseline = True
n_first = 15
n_last = 15

sns.set_theme(context='paper', style='ticks', palette='deep', font='sans-serif', font_scale=1)

processed_folder = paths.processed_dir  

_, _, mice, db = database.select_sessions_from_db(paths.db_path,
                                            paths.nwb_dir,
                                            two_p_imaging='yes',)

# Load day 0 LMI.
lmi_df = os.path.join(paths.processed_dir, f'lmi_day0_results.csv')
lmi_df = pd.read_csv(lmi_df)
selected_cells = lmi_df.loc[(lmi_df['lmi_p'] <= 0.025) | (lmi_df['lmi_p'] >= 0.975)]

psth = []
for mouse_id in mice:
    # Disregard these mice as the number of trials is too low.
    # if mouse_id in ['GF307', 'GF310', 'GF333', 'AR144', 'AR135']:
    #     continue
    reward_group = database.get_mouse_reward_group_from_db(paths.db_path, mouse_id)

    file_name = 'tensor_xarray_learning_data.nc'
    folder = paths.tensor_dir
    xarr = utils_imaging.load_mouse_xarray(mouse_id, folder, file_name)
    xarr = utils_imaging.subtract_baseline(xarr, 2, baseline_win)
    xarr = utils_imaging.load_mouse_xarray(mouse_id, folder, file_name)
    xarr = utils_imaging.subtract_baseline(xarr, 2, baseline_win)
    
    
    # Keep days of interest.
    xarr = xarr.sel(trial=xarr['day'].isin([0]))
    # Select whisker trials.
    xarr = xarr.sel(trial=xarr['whisker_stim']==1)
    # Select PSTH trace length.
    xarr = xarr.sel(time=slice(win[0], win[1]))
    # Average trials per days.
    xarr = xarr.groupby('day').mean(dim='trial')
    
    xarr.name = 'psth'
    xarr = xarr.to_dataframe().reset_index()
    xarr['mouse_id'] = mouse_id
    xarr['reward_group'] = reward_group
    psth.append(xarr)
psth = pd.concat(psth)
    
# Plot proportion of pre vs post modulated cells for both reward groups during
# day 0.

# Compute and plot proportion across mice for both reward groups
lmi_df['reward_group'] = lmi_df['mouse_id'].map(lambda m: database.get_mouse_reward_group_from_db(paths.db_path, m, db))
mouse_props = []
for mouse, group in lmi_df.groupby('mouse_id'):
    reward_group = group['reward_group'].iloc[0]
    n_cells = group['roi'].nunique()
    n_pos = group.loc[group['lmi_p'] >= 0.975, 'roi'].nunique()
    n_neg = group.loc[group['lmi_p'] <= 0.025, 'roi'].nunique()
    mouse_props.append({
        'mouse_id': mouse,
        'reward_group': reward_group,
        'prop_positive': n_pos / n_cells if n_cells > 0 else np.nan,
        'prop_negative': n_neg / n_cells if n_cells > 0 else np.nan
    })
mouse_props_df = pd.DataFrame(mouse_props)

plt.figure(figsize=(6,4), dpi=150)
sns.barplot(data=mouse_props_df.melt(id_vars=['mouse_id', 'reward_group'], 
                                     value_vars=['prop_positive', 'prop_negative'],
                                     var_name='modulation', value_name='proportion'),
            x='reward_group', y='proportion', hue='modulation', ci='sd')
sns.stripplot(data=mouse_props_df.melt(id_vars=['mouse_id', 'reward_group'], 
                                       value_vars=['prop_positive', 'prop_negative'],
                                       var_name='modulation', value_name='proportion'),
              x='reward_group', y='proportion', hue='modulation', 
              dodge=True, marker='o', alpha=0.7, linewidth=0.5, edgecolor='k')
plt.title('Proportion of significantly modulated ROIs per mouse')
plt.ylabel('Proportion')
plt.xlabel('Reward group')
plt.legend(title='Modulation', bbox_to_anchor=(1.05, 1), loc='upper left')
sns.despine()
plt.tight_layout()
plt.show()


# ##############################
# Reinforcement during learning.
# ##############################

# Test the idea that a successful trial reinforces the association i.e.
# increase performance on the subsequent trial.

# Performance at whisker trial n-1 vs n+1 depending on performance at trial n.

# Load behavior table for all imaging mice, day 0
behavior_path = paths.adjust_path_to_host(
    '/mnt/lsens-analysis/Anthony_Renard/data_processed/behavior/behavior_imagingmice_table_5days_cut_with_learning_curves.csv'
)
behavior_df = pd.read_csv(behavior_path)
behavior_df = behavior_df[behavior_df['day'] == 0]
# Only keep whisker trials
behavior_df = behavior_df[behavior_df['whisker_stim'] == 1]


# For each mouse, compute performance at trial n-1 and n+1, grouped by outcome at trial n
reinforcement_results = []
for mouse_id in behavior_df['mouse_id'].unique():
    mouse_trials = behavior_df[behavior_df['mouse_id'] == mouse_id].sort_values('trial_w')
    # Only keep trials with valid neighbors
    for idx in range(1, len(mouse_trials) - 1):
        trial_n_minus_1 = mouse_trials.iloc[idx - 1]
        trial_n = mouse_trials.iloc[idx]
        trial_n_plus_1 = mouse_trials.iloc[idx + 1]
        reinforcement_results.append({
            'mouse_id': mouse_id,
            'reward_group': trial_n['reward_group'],
            'trial_n': trial_n['trial_w'],
            'outcome_n': trial_n['outcome_w'],
            'outcome_n_minus_1': trial_n_minus_1['learning_curve_w'],
            'outcome_n_plus_1': trial_n_plus_1['learning_curve_w'],
        })
reinforcement_df = pd.DataFrame(reinforcement_results)

# Group by outcome at trial n (hit=1, miss=0), compare perf at n-1 and n+1
summary = (
    reinforcement_df
    .groupby(['reward_group', 'outcome_n', 'mouse_id'])
    .agg({'outcome_n_minus_1': 'mean', 'outcome_n_plus_1': 'mean'})
    .reset_index()
)
data = summary.loc[summary.reward_group=='R+']
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
sns.barplot(data=data, x='outcome_n', y='outcome_n_minus_1', ax=axes[0])
sns.barplot(data=data, x='outcome_n', y='outcome_n_plus_1', ax=axes[1])


axes[0].set_xticklabels(['Miss', 'Hit'])
axes[1].set_xticklabels(['Miss', 'Hit'])


# Same the first days of auditory pretraining.
# ############################################


# Load behavior table for all imaging mice, day 0
behavior_path = paths.adjust_path_to_host(
    '//sv-nas1.rcp.epfl.ch/Petersen-Lab/analysis/Anthony_Renard/data_processed/behavior/behavior_imagingmice_table_pretraining_cut.csv'
)
behavior_df = pd.read_csv(behavior_path)


# Reindex 'day' for each mouse to create a 'pretraining_day' column starting from 0
behavior_df['pretraining_day'] = behavior_df.groupby('mouse_id')['day'].transform(lambda x: x.rank(method='dense').astype(int) - 1)
# Only keep whisker trials
behavior_df = behavior_df[behavior_df['auditory_stim'] == 1]
# Keep first three days.
behavior_df = behavior_df[behavior_df['pretraining_day'] < 3]
behavior_df = behavior_df.sort_values(['mouse_id', 'pretraining_day', 'trial_a'])
# Reindex 'trial_a' from 0 to n across the three days for each mouse
behavior_df['trial_a_reindexed'] = behavior_df.groupby('mouse_id').cumcount()




# For each mouse, compute performance at trial n-1 and n+1, grouped by outcome at trial n
reinforcement_results = []
for mouse_id in behavior_df['mouse_id'].unique():
    mouse_trials = behavior_df[behavior_df['mouse_id'] == mouse_id].sort_values('trial_a_reindexed')
    # Only keep trials with valid neighbors
    for idx in range(1, len(mouse_trials) - 1):
        trial_n_minus_1 = mouse_trials.iloc[idx - 1]
        trial_n = mouse_trials.iloc[idx]
        trial_n_plus_1 = mouse_trials.iloc[idx + 1]
        reinforcement_results.append({
            'mouse_id': mouse_id,
            'reward_group': trial_n['reward_group'],
            'trial_n': trial_n['trial_a_reindexed'],
            'outcome_n': trial_n['outcome_a'],
            'outcome_n_minus_1': trial_n_minus_1['outcome_a'],
            'outcome_n_plus_1': trial_n_plus_1['outcome_a'],
        })
reinforcement_df = pd.DataFrame(reinforcement_results)

# Group by outcome at trial n (hit=1, miss=0), compare perf at n-1 and n+1
summary = (
    reinforcement_df
    .groupby(['reward_group', 'outcome_n', 'mouse_id'])
    .agg({'outcome_n_minus_1': 'mean', 'outcome_n_plus_1': 'mean'})
    .reset_index()
)
data = summary.loc[summary.reward_group=='R+']
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
sns.barplot(data=data, x='outcome_n', y='outcome_n_minus_1', ax=axes[0])
sns.barplot(data=data, x='outcome_n', y='outcome_n_plus_1', ax=axes[1])

