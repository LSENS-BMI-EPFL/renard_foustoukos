"""This script contains functions to analyze behavior data.
It assumes that a excel sheet contains sessions metadata
and that data is stored on the server with the usual folder structure.
This code will be refactored to read behavioral data from NWB files.
"""

import os
import sys
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import pymc as pm 
import scipy as sp

sys.path.append(r'/home/aprenard/repos/fast-learning')
sys.path.append(r'/home/aprenard/repos/NWB_analysis')
import src.utils.utils_io as io
from src.utils.utils_plot import *
from src.utils.utils_behavior import *
from nwb_wrappers import nwb_reader_functions as nwb_read
from scipy.stats import mannwhitneyu, wilcoxon
from matplotlib.colors import Normalize
from scipy.ndimage import gaussian_filter1d
from statsmodels.stats.multitest import multipletests
import matplotlib
from scipy.signal import find_peaks, hilbert
import matplotlib.cm as cm



# nwb_dir = r'//sv-nas1.rcp.epfl.ch/Petersen-Lab/analysis/Anthony_Renard/NWB'
# session_list = ['MS039_20240305_085825.nwb',
#                 'MS041_20240305_123141.nwb',
#                 'MS065_20240421_150811.nwb',
#                 'MS067_20240422_152536.nwb',
#                 'MS069_20240422_154117.nwb',
#                 'MS128_20240926_111219.nwb',
#                 'MS129_20240926_112052.nwb',
#                 'MS130_20240925_112915.nwb',
#                 'MS131_20240926_120457.nwb',
#                 'MS135_20240925_133312.nwb']

# # session_list = ['MS061_20240421_144950.nwb',
# #                 'MS066_20240422_143733.nwb',
# #                 'MS066_20240422_143733.nwb',
# #                 'MS127_20240926_105210.nwb',
# #                 'MS132_20240925_115907.nwb',
# #                 'MS133_20240925_121916.nwb',
# #                 'MS134_20240925_114817.nwb']

# nwb_list = [os.path.join(nwb_dir, nwb) for nwb in session_list]
# reward_group = 'R+'

# table = []
# for nwb, session in zip(nwb_list, session_list):
#     df = nwb_read.get_trial_table(nwb)
#     df = df.reset_index()
#     df.rename(columns={'id': 'trial_id'}, inplace=True)
#     df = compute_performance(df, session, reward_group)
#     table.append(df)
# table = pd.concat(table)
# table['day'] = '0'
# table = table.astype({'mouse_id':str,
#                       'session_id':str,
#                       'day':str})

# table = table.loc[table.block_id<=17]
# sns.set_theme(context='talk', style='ticks', palette='deep', font='sans-serif', font_scale=1)
# plt.figure(figsize=(15,6))
# plot_perf_across_blocks(table, reward_group, palette, nmax_trials=300, ax=None)
# sns.despine()


# Create and save data tables.
# ############################

# Read behavior results.
db_path = io.db_path
# db_path = 'C://Users//aprenard//recherches//fast-learning//docs//sessions_muscimol_GF.xlsx'
nwb_dir = io.nwb_dir
stop_flag_yaml = io.stop_flags_yaml
trial_indices_yaml = io.trial_indices_yaml

mice_behavior = io.select_mice_from_db(db_path, nwb_dir, experimenters=None,
                                    exclude_cols = ['exclude'],
                                    optogenetic = ['no', np.nan],
                                    pharmacology = ['no',np.nan],
                                    )
# Add mice with inactivation done after D0 learning.
mice_pharma_execution = io.select_mice_from_db(db_path, nwb_dir, experimenters=None,
                                    exclude_cols = ['exclude'],
                                    optogenetic = ['no', np.nan],
                                    pharmacology = ['yes'],
                                    pharma_inactivation_type = ['execution'],
                                    )
mice_opto_execution = io.select_mice_from_db(db_path, nwb_dir, experimenters=None,
                                    exclude_cols = ['exclude'],
                                    optogenetic = ['yes', np.nan],
                                    pharmacology = ['no',np.nan],
                                    opto_inactivation_type = ['execution'],
                                    )
mice_opto_learning = io.select_mice_from_db(db_path, nwb_dir, experimenters=None,
                                    exclude_cols = ['exclude'],
                                    optogenetic = ['yes', np.nan],
                                    pharmacology = ['no',np.nan],
                                    opto_inactivation_type = ['learning'],
                                    )
# Because some opto mice had both execution and learning inactivation.
mice_opto = list(set(mice_opto_execution) - set(mice_opto_learning))
# Combine and deduplicate mouse lists
mice_behavior = list(set(mice_behavior + mice_pharma_execution + mice_opto))
mice_behavior = sorted(mice_behavior)

len(mice_behavior)

session_list, nwb_list, mice_list, db = io.select_sessions_from_db(
    db_path, nwb_dir, experimenters=None,
    exclude_cols = ['exclude',],
    day = ["-2", "-1", '0', '+1', '+2'],
    mouse_id = mice_behavior,
    )
table = make_behavior_table(nwb_list, session_list, db_path, cut_session=True, stop_flag_yaml=stop_flag_yaml, trial_indices_yaml=trial_indices_yaml)
# Save the table to a CSV file
save_path = r'//sv-nas1.rcp.epfl.ch/Petersen-Lab/analysis/Anthony_Renard/data_processed/behavior/behavior_behaviormice_table_5days_cut.csv'
save_path = io.adjust_path_to_host(save_path)
table.to_csv(save_path, index=False)


mice_imaging = io.select_mice_from_db(db_path, nwb_dir, experimenters=None,
                                    exclude_cols = ['exclude',  'two_p_exclude'],
                                    optogenetic = ['no', np.nan],
                                    pharmacology = ['no',np.nan],
                                    two_p_imaging = 'yes',
                                    )

session_list, nwb_list, mice_list, db = io.select_sessions_from_db(
    db_path, nwb_dir, experimenters=None,
    exclude_cols = ['exclude',  'two_p_exclude'],
    day = ["-2", "-1", '0', '+1', '+2'],
    # day = ['whisker_on_1', 'whisker_off', 'whisker_on_2'],
    mouse_id = mice_imaging,
    )
table = make_behavior_table(nwb_list, session_list, db_path, cut_session=True, stop_flag_yaml=stop_flag_yaml, trial_indices_yaml=trial_indices_yaml)
# Save the table to a CSV file
save_path = r'//sv-nas1.rcp.epfl.ch/Petersen-Lab/analysis/Anthony_Renard/data_processed/behavior/behavior_imagingmice_table_5days_cut.csv'
save_path = io.adjust_path_to_host(save_path)
table.to_csv(save_path, index=False)

# Auditory days for imaging mice.
mice_imaging = io.select_mice_from_db(db_path, nwb_dir, experimenters=None,
                                    exclude_cols = ['exclude',  'two_p_exclude'],
                                    optogenetic = ['no', np.nan],
                                    pharmacology = ['no',np.nan],
                                    two_p_imaging = 'yes',
                                    )

session_list, nwb_list, mice_list, db = io.select_sessions_from_db(
    db_path, nwb_dir, experimenters=None,
    exclude_cols = ['exclude',  'two_p_exclude'],
    day = [f"-{i}" for i in range(8,1,-1)],
    # day = ['whisker_on_1', 'whisker_off', 'whisker_on_2'],
    mouse_id = mice_imaging,
    )
table = make_behavior_table(nwb_list, session_list, db_path, cut_session=True, stop_flag_yaml=stop_flag_yaml, trial_indices_yaml=trial_indices_yaml)
# Save the table to a CSV file
save_path = r'//sv-nas1.rcp.epfl.ch/Petersen-Lab/analysis/Anthony_Renard/data_processed/behavior/behavior_imagingmice_table_pretraining_cut.csv'
save_path = io.adjust_path_to_host(save_path)
table.to_csv(save_path, index=False)

mice_opto = io.select_mice_from_db(db_path, nwb_dir, experimenters=None,
                                    exclude_cols = ['exclude'],
                                    optogenetic = ['yes'],
                                    pharmacology = ['no',np.nan],
                                    )

# Read behavior results.

particle_test_mice = io.select_mice_from_db(db_path, nwb_dir, experimenters=None,
                                    exclude_cols = ['exclude'],
                                    day = ['whisker_on_1', 'whisker_off', 'whisker_on_2'],
                                    optogenetic = ['no', np.nan],
                                    pharmacology = ['no',np.nan],
                                    two_p_imaging = 'yes',)

session_list, nwb_list, mice_list, db = io.select_sessions_from_db(
    db_path, nwb_dir, experimenters=None,
    exclude_cols=['exclude', 'two_p_exclude'],
    day = ['whisker_on_1', 'whisker_off', 'whisker_on_2'],
    mouse_id = particle_test_mice,
    )

table_particle_test = make_behavior_table(nwb_list, session_list, db_path, cut_session=False, stop_flag_yaml=stop_flag_yaml, trial_indices_yaml=trial_indices_yaml)
# Save the table to a CSV file
save_path = r'//sv-nas1.rcp.epfl.ch/Petersen-Lab/analysis/Anthony_Renard/data_processed/behavior/behavior_particle_test.csv'
save_path = io.adjust_path_to_host(save_path)
table_particle_test.to_csv(save_path, index=False)


mice_behavior = io.select_mice_from_db(db_path, nwb_dir, experimenters=None,
                                    exclude_cols = ['exclude'],
                                    optogenetic = ['no', np.nan],
                                    pharmacology = ['no',np.nan],
                                    )

mice_imaging = io.select_mice_from_db(db_path, nwb_dir, experimenters=None,
                                    exclude_cols = ['exclude',  'two_p_exclude'],
                                    optogenetic = ['no', np.nan],
                                    pharmacology = ['no',np.nan],
                                    two_p_imaging = 'yes',
                                    )

mice_opto = io.select_mice_from_db(db_path, nwb_dir, experimenters=None,
                                    exclude_cols = ['exclude'],
                                    optogenetic = ['yes'],
                                    pharmacology = ['no',np.nan],
                                    )

# # plot_single_session(table, 'GF311_19112020_160412')
# # table.loc[(table.reward_group=='R+') & (table.day==0), 'session_id'].unique()
# # sns.despine()    

#     # Plot lick trace for a trial.
#     # ------------------------

session_list, nwb_list, mice_list, db = io.select_sessions_from_db(
    db_path, nwb_dir, experimenters=None,
    exclude_cols=['exclude'],
    # day = ["-2", "-1", '0', '+1', '+2'],
    day = ['whisker_on_1', 'whisker_off', 'whisker_on_2'],
    subject_id = mice_imaging,
    )


#     bin_file = "//sv-nas1.rcp.epfl.ch/Petersen-Lab/data/AR184/Training/AR184_20250319_175559/log_continuous.bin"
#     bin_file = io.adjust_path_to_host(bin_file)
#     # Read binary file using numpy
#     bin_data = np.fromfile(bin_file)
#     ttl = bin_data[2::6]
#     lick_trace = bin_data[1::6]

table = make_behavior_table(nwb_list, session_list, db_path= db_path, cut_session=True, stop_flag_yaml=stop_flag_yaml, trial_indices_yaml=trial_indices_yaml)
# Save the table to a CSV file
save_path = r'//sv-nas1.rcp.epfl.ch/Petersen-Lab/analysis/Anthony_Renard/data_processed/behavior/behavior_imagingmice_table_5days_cut.csv'
save_path = io.adjust_path_to_host(save_path)
table.to_csv(save_path, index=False)


# ############################################# 
# Day 0 for each mouse.
# #############################################

# table = pd.read_csv(r'//sv-nas1.rcp.epfl.ch/Petersen-Lab/analysis/Anthony_Renard/data_processed/behavior/behavior_table_muscimol.csv')
# table = table.loc[table.pharma_inactivation_type=='learning']

# session_list, nwb_list, mice_list, db = io.select_sessions_from_db(
#     db_path, nwb_dir, experimenters=['AR'],
#     exclude_cols=['exclude'],
#     pharma_day = ['pre_-1', 'pre_-2', 'muscimol_1', 'muscimol_2', 'muscimol_3', 'recovery_1', 'recovery_2', 'recovery_3'],
#     subject_id = ['AR181', 'AR182', 'AR183', 'AR184',],
#     )

# Load the table from the CSV file.
table_file = io.adjust_path_to_host(r'//sv-nas1.rcp.epfl.ch/Petersen-Lab/analysis/Anthony_Renard/data_processed/behavior/behavior_table_all_mice_5days.csv')
table = pd.read_csv(table_file)

session_list = table.loc[table.day==0].session_id.drop_duplicates().to_list()
pdf_path = '//sv-nas1.rcp.epfl.ch/Petersen-Lab/analysis/Anthony_Renard/analysis_output/behaviorsingle_sessions_allmice.pdf'
pdf_path = io.adjust_path_to_host(pdf_path)

with PdfPages(pdf_path) as pdf:
    for session_id in session_list:
        print(session_id)
        plot_single_session(table, session_id, ax=None)
        pdf.savefig()
        plt.close()

# Average performance across days.
# ################################

pdf_path = '//sv-nas1.rcp.epfl.ch/Petersen-Lab/analysis/Anthony_Renard/analysis_output/behaviorsingle_sessions_muscimol.pdf'
pdf_path = io.adjust_path_to_host(pdf_path)

with PdfPages(pdf_path) as pdf:
    for session_id in session_list:
        plot_single_session(table, session_id, ax=None)
        pdf.savefig()
        plt.close()


# All five days for a single mouse.
# ################################

table_path = r'//sv-nas1.rcp.epfl.ch/Petersen-Lab/analysis/Anthony_Renard/data_processed/behavior/behavior_imagingmice_table_5days_cut.csv'
table_path = io.adjust_path_to_host(table_path)
table = pd.read_csv(table_path)

# mouse_id = 'AR180'
mouse_id = 'GF305'
# for  mouse_id in mice_imaging:
data = table.loc[table.mouse_id==mouse_id]
data  = data.loc[data.day.isin([-2, -1, 0, 1, 2])]
sessions = data.session_id.drop_duplicates().to_list()
sns.set_theme(context='paper', style='ticks', palette='deep', font='sans-serif', font_scale=1,
              rc={'xtick.major.width': .8,'ytick.major.width': .8,})

data = data.loc[data.trial_id<=180]

fig, axes = plt.subplots(1, 5, figsize=(10, 2))
for i, session in enumerate(sessions):
    ax = axes[i]
    plot_single_session(
        data, session, ax=ax, palette=behavior_palette, 
         do_scatter=False, linewidth=1.5,
    )
    
# Save the figure.
output_dir = r'/mnt/lsens-analysis/Anthony_Renard/analysis_output/fast-learning/behavior/exemples'
plt.savefig(os.path.join(output_dir, f'behavior_single_mouse_{mouse_id}.svg'), dpi=300)



# ################################
# Average performance across days.
# ################################


# table = make_behavior_table(nwb_list, session_list, db_path, cut_session=True, stop_flag_yaml=stop_flag_yaml, trial_indices_yaml=trial_indices_yaml)
# # Save the table to a CSV file
# save_path = r'//sv-nas1.rcp.epfl.ch/Petersen-Lab/analysis/Anthony_Renard/data_processed/behavior/behavior_imagingmice_table_5days_cut.csv'
# table.to_csv(save_path, index=False)

# Load table.
table_file = io.adjust_path_to_host(r'//sv-nas1.rcp.epfl.ch/Petersen-Lab/analysis/Anthony_Renard/data_processed/behavior/behavior_imagingmice_table_5days_cut.csv')
table = pd.read_csv(table_file)

# Remove spurious whisker trials coming mapping session.
table.loc[table.day.isin([-2, -1]), 'outcome_w'] = np.nan
table.loc[table.day.isin([-2, -1]), 'hr_w'] = np.nan

# Average performance.
table = table.groupby(['mouse_id','session_id','reward_group','day'], as_index=False)[['outcome_c','outcome_a','outcome_w']].agg(np.mean)
# Convert performance to percentage
table[['outcome_c', 'outcome_a', 'outcome_w']] = table[['outcome_c', 'outcome_a', 'outcome_w']] * 100

# Plot.
sns.set_theme(context='paper', style='ticks', palette='deep', font='sans-serif', font_scale=1,
              rc={'xtick.major.width': 1, 'ytick.major.width': 1, 'pdf.fonttype':42, 'ps.fonttype':42, 'svg.fonttype':'none'})
fig = plt.figure(figsize=(6,6))
ax = plt.gca()
table['day'] = table['day'].astype(str)  # Convert 'day' column to string for categorical alignment

sns.lineplot(data=table, x='day', y='outcome_c', units='mouse_id',
            estimator=None, hue="reward_group", hue_order=['R-', 'R+'],
            palette=behavior_palette[4:6], alpha=.4, legend=False, ax=ax, marker=None, linewidth=1)
sns.lineplot(data=table, x='day', y='outcome_a', units='mouse_id',
            estimator=None, hue="reward_group", hue_order=['R-', 'R+'],
            palette=behavior_palette[0:2], alpha=.4, legend=False, ax=ax, marker=None, linewidth=1)
sns.lineplot(data=table, x='day', y='outcome_w', units='mouse_id',
            estimator=None, hue="reward_group", hue_order=['R-', 'R+'],
            palette=behavior_palette[2:4], alpha=.4, legend=False, ax=ax, marker=None, linewidth=1)

sns.pointplot(data=table, x='day', y='outcome_c', estimator=np.mean, palette=behavior_palette[4:6], hue="reward_group", 
                hue_order=['R-', 'R+'], alpha=1, legend=True, ax=ax, linewidth=2)
sns.pointplot(data=table, x='day', y='outcome_a', estimator=np.mean, palette=behavior_palette[0:2], hue="reward_group", 
                hue_order=['R-', 'R+'], alpha=1, legend=True, ax=ax, linewidth=2)
sns.pointplot(data=table, x='day', y='outcome_w', estimator=np.mean, palette=behavior_palette[2:4], hue="reward_group", 
                hue_order=['R-', 'R+'], alpha=1, legend=True, ax=ax, linewidth=2)

plt.xlabel('Training days')
plt.ylabel('Lick probability (%)')
# plt.ylim([-0.2, 1.05])
plt.legend()
sns.despine(trim=True)

# Ensure tick thickness is set for SVG output
for axis in [ax.xaxis, ax.yaxis]:
    axis.set_tick_params(width=1)

# Save plot.
output_dir = r'/mnt/lsens-analysis/Anthony_Renard/analysis_output/fast-learning/behavior'
plt.savefig(os.path.join(output_dir, 'performance_across_days_imagingmice.svg'), dpi=300)
# Save the dataframe to CSV
table.to_csv(os.path.join(output_dir, 'performance_across_days_imagingmice_data.csv'), index=False)


# Histogram quantifying D0.
# -------------------------
# Select data for days 0, +1, and +2
days_of_interest = [0, 1, 2]
day_data = table[table['day'].isin(days_of_interest)]
avg_performance = day_data.groupby(['day', 'mouse_id', 'reward_group'])['outcome_w'].mean().reset_index()

sns.set_theme(context='paper', style='ticks', palette='deep', font='sans-serif', font_scale=1,
              rc={'pdf.fonttype':42, 'ps.fonttype':42, 'svg.fonttype':'none'})

# Plot barplot for each day
plt.figure(figsize=(8, 6))
sns.barplot(
    data=avg_performance,
    x='day',
    y='outcome_w',
    hue='reward_group',
    palette=behavior_palette[2:4][::-1],
    width=0.3,
    dodge=True
)
sns.swarmplot(
    data=avg_performance,
    x='day',
    y='outcome_w',
    hue='reward_group',
    dodge=True,
    color='grey',
    alpha=0.6,
)
plt.xlabel('Day')
plt.ylabel('Lick probability (%)')
plt.ylim([0, 100])
plt.legend(title='Reward group')
sns.despine(trim=True)

# Test significance with Mann-Whitney U test for each day
stats = []
for day in days_of_interest:
    df_day = avg_performance[avg_performance['day'] == day]
    group_R_plus = df_day[df_day['reward_group'] == 'R+']['outcome_w']
    group_R_minus = df_day[df_day['reward_group'] == 'R-']['outcome_w']
    stat, p_value = mannwhitneyu(group_R_plus, group_R_minus, alternative='two-sided')
    stats.append({'day': day, 'statistic': stat, 'p_value': p_value})
    # Add stars to the plot to indicate significance
    ax = plt.gca()
    xpos = days_of_interest.index(day)
    ypos = 95
    if p_value < 0.001:
        plt.text(xpos, ypos, '***', ha='center', va='bottom', color='black', fontsize=14)
    elif p_value < 0.01:
        plt.text(xpos, ypos, '**', ha='center', va='bottom', color='black', fontsize=14)
    elif p_value < 0.05:
        plt.text(xpos, ypos, '*', ha='center', va='bottom', color='black', fontsize=14)

# Save.
output_dir = r'/mnt/lsens-analysis/Anthony_Renard/analysis_output/fast-learning/behavior'
plt.savefig(os.path.join(output_dir, 'performance_D0_D1_D2_barplot_imagingmice.svg'), dpi=300)
avg_performance.to_csv(os.path.join(output_dir, 'performance_D0_D1_D2_barplot_imagingmice_data.csv'), index=False)
pd.DataFrame(stats).to_csv(os.path.join(output_dir, 'performance_D0_D1_D2_barplot_imagingmice_stats.csv'), index=False)


# Plot first whisker hit for both reward group (sanity check).
# ------------------------------------------------------------
 
# Select data of day 0
day_0_data = table[(table['day'] == 0) & (table.whisker_stim == 1)]
f = lambda x: x.reset_index(drop=True).idxmax()+1
fh = day_0_data.groupby(['mouse_id', 'reward_group'], as_index=False)[['outcome_w']].agg(f)

sns.set_theme(context='paper', style='ticks', palette='deep', font='sans-serif', font_scale=1,
                rc={'pdf.fonttype':42, 'ps.fonttype':42, 'svg.fonttype':'none'})
plt.figure(figsize=(4, 6))
sns.barplot(data=fh, x='reward_group', y='outcome_w', palette=reward_palette[::-1], width=0.3)
sns.stripplot(data=fh, x='reward_group', y='outcome_w', color='grey', jitter=False, dodge=True, alpha=.4)
plt.ylabel('First hit trial')
sns.despine()
output_dir = r'/mnt/lsens-analysis/Anthony_Renard/analysis_output/fast-learning/behavior'
plt.savefig(os.path.join(output_dir, 'firsthit_D0.svg'), dpi=300)

# Save the first hit data to CSV
output_dir = io.adjust_path_to_host(r'/mnt/lsens-analysis/Anthony_Renard/analysis_output/fast-learning/behavior')
fh.to_csv(os.path.join(output_dir, 'firsthit_D0_data.csv'), index=False)

# Mann-Whitney U test for first hit trial between reward groups
stat, p_value = mannwhitneyu(
    fh[fh['reward_group'] == 'R+']['outcome_w'],
    fh[fh['reward_group'] == 'R-']['outcome_w'],
    alternative='two-sided'   
)
# Save stats to file
with open(os.path.join(output_dir, 'firsthit_D0_stats.csv'), 'w') as f:
    f.write(f'Mann-Whitney U Test: Statistic={stat}, P-value={p_value}')
 
# Particle test plot.
# -------------------

table_path = r'//sv-nas1.rcp.epfl.ch/Petersen-Lab/analysis/Anthony_Renard/data_processed/behavior/behavior_particle_test.csv'
table_path = io.adjust_path_to_host(table_path)
table_particle_test = pd.read_csv(table_path)

df = table_particle_test.loc[table_particle_test.reward_group=='R+']
df['outcome_w'] = df['outcome_w'] * 100
df = df.groupby(['mouse_id', 'behavior_type'])['outcome_w'].mean().reset_index()

plt.figure(figsize=(4, 8))
sns.barplot(data=df, x='behavior_type', y='outcome_w', order=['whisker_on_1', 'whisker_off', 'whisker_on_2'], color='#1b9e77')
ax = plt.gca()
   
# Draw lines to connect each individual mouse across the three behavior types
for mouse_id in df.mouse_id.unique():
    a = df.loc[(df.mouse_id==mouse_id) & (df.behavior_type == 'whisker_on_1'), 'outcome_w'].to_numpy()[0]
    b = df.loc[(df.mouse_id==mouse_id) & (df.behavior_type == 'whisker_off'), 'outcome_w'].to_numpy()[0]
    plt.plot(['whisker_on_1', 'whisker_off'], [a,b], alpha=.8, color='grey', marker='', linewidth=1)
    a = df.loc[(df.mouse_id==mouse_id) & (df.behavior_type == 'whisker_off'), 'outcome_w'].to_numpy()[0]
    b = df.loc[(df.mouse_id==mouse_id) & (df.behavior_type == 'whisker_on_2'), 'outcome_w'].to_numpy()[0]
    plt.plot(['whisker_off', 'whisker_on_2'], [a,b], alpha=.8, color='grey', marker='', linewidth=1)
plt.xticks([0, 1, 2], ['ON', 'OFF', 'ON'])
plt.ylabel('Lick probability (%)')
plt.xlabel('Particle test')
sns.despine()

# Save figure.
output_dir = '/mnt/lsens-analysis/Anthony_Renard/analysis_output/fast-learning/behavior'
plt.savefig(os.path.join(output_dir, 'particle_test_imagingmice.svg'), dpi=300)


def test_significance(data, group_col, value_col, group1, group2):
    group1_data = data[data[group_col] == group1][value_col]
    group2_data = data[data[group_col] == group2][value_col]
    stat, p_value = wilcoxon(group1_data, group2_data, alternative='two-sided')
    return stat, p_value

# Perform the test for whisker_on_1 vs whisker_off
stat1, p_value1 = test_significance(df, 'behavior_type', 'outcome_w', 'whisker_on_1', 'whisker_off')
# Perform the test for whisker_off vs whisker_on_2
stat2, p_value2 = test_significance(df, 'behavior_type', 'outcome_w', 'whisker_off', 'whisker_on_2')

# Save the dataframe and stats to CSV files
save_path = '/mnt/lsens-analysis/Anthony_Renard/analysis_output/fast-learning/behavior'
save_path = io.adjust_path_to_host(save_path)
table_particle_test.to_csv(os.path.join(save_path, 'particle_test_imagningmice_data.csv'), index=False)
with open(os.path.join(save_path, 'particle_test_imagningmice_stats.csv'), 'w') as f:
    f.write(f'whisker_on_1 vs whisker_off: Statistic: {stat1}, P-value: {p_value1}\n')
    f.write(f'whisker_off vs whisker_on_2: Statistic: {stat2}, P-value: {p_value2}')
    # Test significance with Wilcoxon signed-rank test



# #####################################################################
# Performance over blocks during whisker learning sessions across mice.
# #####################################################################

# Read behavior results.
db_path = io.db_path
# db_path = 'C://Users//aprenard//recherches//fast-learning//docs//sessions_muscimol_GF.xlsx'
nwb_dir = io.nwb_dir
stop_flag_yaml = io.stop_flags_yaml
trial_indices_yaml = io.trial_indices_yaml

# Load the table from the CSV file.
table_file = io.adjust_path_to_host(r'/mnt/lsens-analysis/Anthony_Renard/data_processed/behavior/behavior_imagingmice_table_5days_cut.csv')
# table_file = io.adjust_path_to_host(r'//sv-nas1.rcp.epfl.ch/Petersen-Lab/analysis/Anthony_Renard/data_processed/behavior/behavior_behaviormice_table_5days_cut.csv')
table = pd.read_csv(table_file)
table.mouse_id.unique().size


# Performance over blocks.
# ------------------------


data = table.copy()    
data['block_id'] = data.block_id + 1  # Start block index at 1 for plot.

# Performance over blocks
fig, axes = plt.subplots(1, 5, sharey=True, figsize=(20, 4))
plot_perf_across_blocks(data, "R-", -2, behavior_palette, nmax_trials=240, ax=axes[0])
plot_perf_across_blocks(data, "R-", -1, behavior_palette, nmax_trials=240, ax=axes[1])
plot_perf_across_blocks(data, "R-", 0, behavior_palette, nmax_trials=240, ax=axes[2])
plot_perf_across_blocks(data, "R-", 1, behavior_palette, nmax_trials=240, ax=axes[3])
plot_perf_across_blocks(data, "R-", 2, behavior_palette, nmax_trials=240, ax=axes[4])

plot_perf_across_blocks(data, "R+", -2, behavior_palette, nmax_trials=240, ax=axes[0])
plot_perf_across_blocks(data, "R+", -1, behavior_palette, nmax_trials=240, ax=axes[1])
plot_perf_across_blocks(data, "R+", 0, behavior_palette, nmax_trials=240, ax=axes[2])
plot_perf_across_blocks(data, "R+", 1, behavior_palette, nmax_trials=240, ax=axes[3])
plot_perf_across_blocks(data, "R+", 2, behavior_palette, nmax_trials=240, ax=axes[4])


# Save the figure
output_dir = io.adjust_path_to_host(r'/mnt/lsens-analysis/Anthony_Renard/analysis_output/fast-learning/behavior')
output_file = os.path.join(output_dir, 'performance_over_blocks_imagingmice.svg')
plt.savefig(output_file, format='svg', dpi=300)



# Performance during sessions across mice with fitted learning curves.
# --------------------------------------------------------------------

# Load the table from the CSV file.
table_file = io.adjust_path_to_host(r'/mnt/lsens-analysis/Anthony_Renard/data_processed/behavior/behavior_imagingmice_table_5days_cut_with_learning_curves.csv')
table = pd.read_csv(table_file)

table.columns

# Parameter: maximum number of trials per type to plot
max_trials_per_type = 120

# Performance over blocks
fig, axes = plt.subplots(1, 1, sharey=True, figsize=(8, 4)); axes = [axes]


for i, day in enumerate([0]):
    ax = axes[i]
    for reward_group, color_w, color_a, color_ns in zip(
        ['R-', 'R+'],
        behavior_palette[2:4],  # whisker
        behavior_palette[0:2],  # auditory
        behavior_palette[4:6],  # no_stim
    ):
        d = table[(table.day == day) & (table.reward_group == reward_group)]
        # Cut to max_trials_per_type for each trial type
        d_whisker = d.loc[d.trial_w<max_trials_per_type]
        d_auditory = d.loc[d.trial_a<max_trials_per_type]
        d_nostim = d.loc[d.trial_c<max_trials_per_type]
        
        # # Plot no_stim learning curve
        # sns.lineplot(
        #     data=d_nostim,
        #     x='trial_c', y='learning_curve_ns',
        #     errorbar='ci', ax=ax, color=color_ns, label=f'{reward_group} no_stim', linewidth=2,
        #     err_kws={'edgecolor': 'none'}
        # )
        
        # Plot whisker learning curve
        sns.lineplot(
            data=d_whisker,
            x='trial_w', y='learning_curve_w',
            errorbar='ci', ax=ax, color=color_w, label=f'{reward_group} whisker', linewidth=2,
            err_kws={'edgecolor': 'none'}
        )
        # # Plot auditory learning curve
        # sns.lineplot(
        #     data=d_auditory,
        #     x='trial_a', y='learning_curve_a',
        #     errorbar='ci', ax=ax, color=color_a, label=f'{reward_group} auditory', linewidth=2
        # )
    ax.set_title(f'Day {day}')
    ax.set_xlabel('Whisker trial')
axes[0].set_ylabel('Lick probability')
sns.despine()

# Save the figure
output_dir = io.adjust_path_to_host(r'/mnt/lsens-analysis/Anthony_Renard/analysis_output/fast-learning/behavior')
output_file = os.path.join(output_dir, 'performance_over_blocks_imagingmice.svg')
plt.savefig(output_file, format='svg', dpi=300)




# Performance during sessions across mice during day 0.
# -----------------------------------------------------

# Load the table from the CSV file.
table_file = io.adjust_path_to_host(r'/mnt/lsens-analysis/Anthony_Renard/data_processed/behavior/behavior_imagingmice_table_5days_cut_with_learning_curves.csv')
table = pd.read_csv(table_file)

table.columns

# Parameter: maximum number of trials per type to plot
max_trials_per_type = 120

# Performance over blocks - three representations
fig, axes = plt.subplots(1, 3, sharey=True, figsize=(15, 4))

day = 0

for i, (metric, ylabel, title) in enumerate([
    ('outcome_w', 'Hit rate (raw)', 'Raw trial outcomes'),
    ('hr_w', 'Hit rate (block avg)', 'Block-averaged performance'),
    ('learning_curve_w', 'Lick probability', 'Fitted learning curves')
]):
    ax = axes[i]

    for reward_group, color_w in zip(
        ['R-', 'R+'],
        behavior_palette[2:4],  # whisker colors
    ): 
        d = table[(table.day == day) & (table.reward_group == reward_group)]
        # Cut to max_trials_per_type for each trial type
        d_whisker = d.loc[d.trial_w < max_trials_per_type]

        # Plot the specified metric
        sns.lineplot(
            data=d_whisker,
            x='trial_w', y=metric,
            errorbar='ci', ax=ax, color=color_w, label=f'{reward_group}', linewidth=2,
            err_kws={'edgecolor': 'none'}
        )

    ax.set_title(title)
    ax.set_xlabel('Whisker trial')
    if i == 0:
        ax.set_ylabel(ylabel)
    ax.legend(frameon=False)

sns.despine()
plt.tight_layout()

# Save the figure
output_dir = io.adjust_path_to_host(r'/mnt/lsens-analysis/Anthony_Renard/analysis_output/fast-learning/behavior')
output_file = os.path.join(output_dir, 'performance_comparison_day0.svg')
plt.savefig(output_file, format='svg', dpi=300)







# Performance over trials aligned to first whisker trial.
# -------------------------------------------------------
import matplotlib

# Prepare data - simply filter for whisker trials on day 0
n_trials = 120
df = table.loc[(table.whisker_stim==1) & (table.day==0)]
df = df.loc[df.trial_w <= n_trials]

# Prepare data for single trial plot
df_single = df.copy()

# Prepare data for learning curve plot
df_learning = df.copy()

# Create three-panel figure
fig, axes = plt.subplots(1, 2, sharey=True, figsize=(12, 5))

# Create colormap for p-values
cmap = matplotlib.colors.LinearSegmentedColormap.from_list('pval_cmap', ['black', 'white'])
norm = matplotlib.colors.Normalize(vmin=0, vmax=0.05)

# Panel 1: Single trial performance
ax = axes[0]
sns.lineplot(data=df_single, x='trial_w', y='outcome_w',
            palette=reward_palette[::-1], hue='reward_group',
            errorbar='ci', err_style='band', ax=ax, legend=False)

# Statistical test for panel 1
p_values_single = []
for trial_w in df_single['trial_w'].unique():
    group_R_plus = df_single[(df_single['trial_w'] == trial_w) & (df_single['reward_group'] == 'R+')]['outcome_w']
    group_R_minus = df_single[(df_single['trial_w'] == trial_w) & (df_single['reward_group'] == 'R-')]['outcome_w']
    if len(group_R_plus) > 0 and len(group_R_minus) > 0:
        stat, p_value = mannwhitneyu(group_R_plus, group_R_minus, alternative='two-sided')
        p_values_single.append((trial_w, p_value))

# FDR correction
trials_single, raw_pvals_single = zip(*p_values_single)
_, corrected_pvals_single, _, _ = multipletests(raw_pvals_single, alpha=0.05, method='fdr_bh')
p_values_single = list(zip(trials_single, corrected_pvals_single))

# Plot p-value rectangles
for trial, p_value in p_values_single:
    color = cmap(norm(min(p_value, 0.05)))
    ax.add_patch(plt.Rectangle((trial - 0.4, .95), 0.8, 0.03, color=color, edgecolor='none'))

ax.set_title('Raw trial outcomes')
ax.set_xlabel('Whisker trial')
ax.set_ylabel('Hit rate')
ax.set_ylim([-0.1, 1])

# Panel 2: Fitted learning curves
ax = axes[1]
sns.lineplot(data=df_learning, x='trial_w', y='learning_curve_w',
            palette=reward_palette[::-1], hue='reward_group',
            errorbar='ci', err_style='band', ax=ax)

# Statistical test for panel 3
p_values_learning = []
for trial_w in df_learning['trial_w'].unique():
    group_R_plus = df_learning[(df_learning['trial_w'] == trial_w) & (df_learning['reward_group'] == 'R+')]['learning_curve_w']
    group_R_minus = df_learning[(df_learning['trial_w'] == trial_w) & (df_learning['reward_group'] == 'R-')]['learning_curve_w']
    if len(group_R_plus) > 0 and len(group_R_minus) > 0:
        stat, p_value = mannwhitneyu(group_R_plus, group_R_minus, alternative='two-sided')
        p_values_learning.append((trial_w, p_value))

# FDR correction
trials_learning, raw_pvals_learning = zip(*p_values_learning)
_, corrected_pvals_learning, _, _ = multipletests(raw_pvals_learning, alpha=0.05, method='fdr_bh')
p_values_learning = list(zip(trials_learning, corrected_pvals_learning))

# Plot p-value rectangles
for trial, p_value in p_values_learning:
    color = cmap(norm(min(p_value, 0.05)))
    ax.add_patch(plt.Rectangle((trial - 0.4, .95), 0.8, 0.03, color=color, edgecolor='none'))

ax.set_title('Fitted learning curves')
ax.set_xlabel('Whisker trial')
ax.set_ylim([-0.1, 1])
ax.legend(frameon=False, title='Reward group')

sns.despine()
plt.tight_layout()

# Save the figure
output_dir = io.adjust_path_to_host(r'/mnt/lsens-analysis/Anthony_Renard/analysis_output/fast-learning/behavior')
output_file = os.path.join(output_dir, f'performance_D0_comparison_aligned_whisker_trial.svg')
plt.savefig(output_file, format='svg', dpi=300)

# Save all data and p-values
df_single.to_csv(os.path.join(output_dir, 'performance_D0_single_trial_data.csv'), index=False)
pd.DataFrame(p_values_single, columns=['trial_w', 'p_value']).to_csv(
    os.path.join(output_dir, 'performance_D0_single_trial_stats.csv'), index=False)

df_learning.to_csv(os.path.join(output_dir, 'performance_D0_learning_curve_data.csv'), index=False)
pd.DataFrame(p_values_learning, columns=['trial_w', 'p_value']).to_csv(
    os.path.join(output_dir, 'performance_D0_learning_curve_stats.csv'), index=False)


# ############################################################
# Muscimol inactivation.
# ############################################################

# Read behavior results.
db_path = io.db_path
# db_path = 'C://Users//aprenard//recherches//fast-learning//docs//sessions_muscimol_GF.xlsx'
nwb_dir = io.nwb_dir
stop_flag_yaml = io.stop_flags_yaml
trial_indices_yaml = io.trial_indices_yaml

mice_muscimol = io.select_mice_from_db(db_path, nwb_dir, experimenters=None,
                                    exclude_cols = ['exclude'],
                                    pharmacology = 'yes',
                                    )

# Read behavior results.
session_list, nwb_list, mice_list, db = io.select_sessions_from_db(
    db_path, nwb_dir, experimenters=None,
    exclude_cols=['exclude'],
    pharma_inactivation_type = ['learning'],
    pharma_day = ["pre_-2", "pre_-1",
            "muscimol_1", "muscimol_2", "muscimol_3",
            "recovery_1", "recovery_2", "recovery_3"],
    # day = ['whisker_on_1', 'whisker_off', 'whisker_on_2'],
    mouse_id = mice_muscimol,
    )

# table_muscimol_learning = make_behavior_table(nwb_list, session_list, db_path, cut_session=True, stop_flag_yaml=io.stop_flags_yaml, trial_indices_yaml=io.trial_indices_yaml)
# # Save the table to a CSV file
# save_path = r'//sv-nas1.rcp.epfl.ch/Petersen-Lab/analysis/Anthony_Renard/data_processed/behavior/behavior_muscimol.csv'
# save_path = io.adjust_path_to_host(save_path)
# table_muscimol_learning.to_csv(save_path, index=False)

# Load table.
table_path = r'//sv-nas1.rcp.epfl.ch/Petersen-Lab/analysis/Anthony_Renard/data_processed/behavior/behavior_muscimol.csv'
table_path = io.adjust_path_to_host(table_path)
table = pd.read_csv(table_path)

fpS1_mice = io.select_mice_from_db(db_path, nwb_dir, experimenters=None,
                                    exclude_cols = ['exclude'],
                                    pharmacology = 'yes',
                                    pharma_inactivation_type = 'learning',
                                    pharma_area = 'fpS1',
                                    )

wS1_mice = io.select_mice_from_db(db_path, nwb_dir, experimenters=None,
                                    exclude_cols = ['exclude'],
                                    pharmacology = 'yes',
                                    pharma_inactivation_type = 'learning',
                                    pharma_area = 'wS1',
                                    )

table.loc[table.mouse_id.isin(fpS1_mice), 'area'] = 'fpS1'
table.loc[table.mouse_id.isin(wS1_mice), 'area'] = 'wS1' 
table = pd.merge(
    table,
    db[['mouse_id', 'session_id', 'pharma_day']],
    on=['mouse_id', 'session_id'],
    how='left'
)
inactivation_labels = ['pre_-2', 'pre_-1', 'muscimol_1', 'muscimol_2', 'muscimol_3', 'recovery_1', 'recovery_2', 'recovery_3']

data = table.groupby(['mouse_id', 'session_id', 'pharma_day', 'area'], as_index=False)[['outcome_c','outcome_a','outcome_w']].agg('mean')
# Order the data by inactivation labels for each session
data['pharma_day'] = pd.Categorical(data['pharma_day'], categories=inactivation_labels, ordered=True)
data = data.sort_values(by=['mouse_id', 'pharma_day'])
# Convert performance to percentage.
data['outcome_c'] = data['outcome_c'] * 100
data['outcome_a'] = data['outcome_a'] * 100
data['outcome_w'] = data['outcome_w'] * 100


# Plot inactivation for wS1 and fpS1.
# -----------------------------------

sns.set_theme(context='paper', style='ticks', palette='deep', font='sans-serif', font_scale=1,
              rc={'pdf.fonttype':42, 'ps.fonttype':42, 'svg.fonttype':'none'})
fig, axes = plt.subplots(1, 2, sharey=True, figsize=(12,5))

ax = axes[0]

for imouse in wS1_mice:
    sns.lineplot(data=data.loc[data.mouse_id==imouse], x='pharma_day', y='outcome_c', estimator=np.mean, color=stim_palette[2],
                alpha=.6, legend=False, ax=ax, marker=None, err_style='bars', linewidth=1)
    sns.lineplot(data=data.loc[data.mouse_id==imouse], x='pharma_day', y='outcome_a', estimator=np.mean, color=stim_palette[0],
                alpha=.6, legend=False, ax=ax, marker=None, err_style='bars', linewidth=1)
    sns.lineplot(data=data.loc[data.mouse_id==imouse], x='pharma_day', y='outcome_w', estimator=np.mean, color=stim_palette[1],
                alpha=.6, legend=False, ax=ax, marker=None, err_style='bars', linewidth=1)

sns.pointplot(data=data.loc[data.mouse_id.isin(wS1_mice)], x='pharma_day',
            y='outcome_c', order=inactivation_labels,
            color=stim_palette[2], ax=ax, linewidth=2)
sns.pointplot(data=data.loc[data.mouse_id.isin(wS1_mice)], x='pharma_day',
            y='outcome_a', order=inactivation_labels,
            color=stim_palette[0], ax=ax, linewidth=2)
sns.pointplot(data=data.loc[data.mouse_id.isin(wS1_mice)], x='pharma_day',
            y='outcome_w', order=inactivation_labels,
            color=stim_palette[1], ax=ax, linewidth=2)

ax = axes[1]

for imouse in fpS1_mice:
    sns.lineplot(data=data.loc[data.mouse_id==imouse], x='pharma_day', y='outcome_c', estimator=np.mean, color=stim_palette[2],
                alpha=.6, legend=False, ax=ax, marker=None, err_style='bars', linewidth=1)
    sns.lineplot(data=data.loc[data.mouse_id==imouse], x='pharma_day', y='outcome_a', estimator=np.mean, color=stim_palette[0],
                alpha=.6, legend=False, ax=ax, marker=None, err_style='bars', linewidth=1)
    sns.lineplot(data=data.loc[data.mouse_id==imouse], x='pharma_day', y='outcome_w', estimator=np.mean, color=stim_palette[1],
                alpha=.6, legend=False, ax=ax, marker=None, err_style='bars', linewidth=1)

sns.pointplot(data=data.loc[data.mouse_id.isin(fpS1_mice)], x='pharma_day',
            y='outcome_c', order=inactivation_labels,
            color=stim_palette[2], ax=ax, linewidth=2)
sns.pointplot(data=data.loc[data.mouse_id.isin(fpS1_mice)], x='pharma_day',
            y='outcome_a', order=inactivation_labels,
            color=stim_palette[0], ax=ax, linewidth=2)
sns.pointplot(data=data.loc[data.mouse_id.isin(fpS1_mice)], x='pharma_day',
            y='outcome_w', order=inactivation_labels,
            color=stim_palette[1], ax=ax, linewidth=2)

for ax in axes:
    ax.set_yticks([0,20, 40, 60, 80, 100])
    ax.set_xticklabels(['-2', '-1', 'M 1', 'M 2', 'M 3', 'R 1', 'R 2', 'R 3'])
    ax.set_xlabel('Muscimol inactivation during learning')
    ax.set_ylabel('Lick probability (%)')
sns.despine(trim=True)

# Save figure
output_dir = fr'/mnt/lsens-analysis/Anthony_Renard/analysis_output/fast-learning/behavior'
output_dir = io.adjust_path_to_host(output_dir)
svg_file = f'muscimol_learning.svg'
plt.savefig(os.path.join(output_dir, svg_file), format='svg', dpi=300)
# Save data.
data.to_csv(os.path.join(output_dir, 'muscimol_learning_data.csv'), index=False)

# Bar plot and stats for Day 0 (muscimol_1), Day +1 (muscimol_2), and Day +2 (muscimol_3)
# ----------------------------------------------------------------------------------------

days_of_interest = ['muscimol_1', 'muscimol_2', 'muscimol_3']
day_labels = ['D0', 'D+1', 'D+2']

day_data = data[data['pharma_day'].isin(days_of_interest)].copy()
day_data['day_label'] = day_data['pharma_day'].map(dict(zip(days_of_interest, day_labels)))

plt.figure(figsize=(8, 6))
sns.barplot(
    data=day_data,
    x='day_label',
    y='outcome_w',
    hue='area',
    palette=[reward_palette[1]],
    width=0.3,
    dodge=True
)
sns.swarmplot(
    data=day_data,
    x='day_label',
    y='outcome_w',
    hue='area',
    dodge=True,
    color=stim_palette[2],
    alpha=0.6
)
plt.xlabel('Day')
plt.ylabel('Whisker Performance (%)')
plt.ylim([0, 100])
plt.legend(title='Area')
sns.despine()

# Perform Mann-Whitney U test for each day between wS1 and fpS1
stats = []
for day, label in zip(days_of_interest, day_labels):
    df_day = day_data[day_data['pharma_day'] == day]
    group_wS1 = df_day[df_day['area'] == 'wS1']['outcome_w']
    group_fpS1 = df_day[df_day['area'] == 'fpS1']['outcome_w']
    stat, p_value = mannwhitneyu(group_wS1, group_fpS1, alternative='two-sided')
    stats.append({'day': label, 'statistic': stat, 'p_value': p_value})
    # Add significance stars to the plot
    ax = plt.gca()
    xpos = day_labels.index(label)
    ypos = 95
    if p_value < 0.001:
        plt.text(xpos, ypos, '***', ha='center', va='bottom', color='black', fontsize=14)
    elif p_value < 0.01:
        plt.text(xpos, ypos, '**', ha='center', va='bottom', color='black', fontsize=14)
    elif p_value < 0.05:
        plt.text(xpos, ypos, '*', ha='center', va='bottom', color='black', fontsize=14)
    # Add p-value text below stars
    plt.text(xpos, 90, f'p={p_value:.3g}', ha='center', va='bottom', color='black', fontsize=10)

# Save the results to CSV files
output_dir = io.adjust_path_to_host(r'/mnt/lsens-analysis/Anthony_Renard/analysis_output/fast-learning/behavior')
plt.savefig(os.path.join(output_dir, 'muscimol_learning_day0_day1_day2.svg'), format='svg', dpi=300)
day_data.to_csv(os.path.join(output_dir, 'muscimol_learning_day0_day1_day2_data.csv'), index=False)
pd.DataFrame(stats).to_csv(os.path.join(output_dir, 'muscimol_learning_day0_day1_day2_stats.csv'), index=False)


# ###################
# Reaction time plot.
# ###################

# Read behavior results.
db_path = io.db_path
# db_path = 'C://Users//aprenard//recherches//fast-learning//docs//sessions_muscimol_GF.xlsx'
nwb_dir = io.nwb_dir
stop_flag_yaml = io.stop_flags_yaml
trial_indices_yaml = io.trial_indices_yaml

experimenters = ['AR', 'GF', 'MI']
mice_imaging = io.select_mice_from_db(db_path, nwb_dir,
                                    experimenters = experimenters,
                                    exclude_cols = ['exclude',  'two_p_exclude'],
                                    optogenetic = ['no', np.nan],
                                    pharmacology = ['no',np.nan],
                                    )
# Load the table from the CSV file.
table_file = io.adjust_path_to_host(r'/mnt/lsens-analysis/Anthony_Renard/data_processed/behavior/behavior_imagingmice_table_5days_cut.csv')
table = pd.read_csv(table_file)

# Compute reaction time.
table['reaction_time'] = table['lick_time'] - table['stim_onset']
table.loc[table.reaction_time>=1.25] = np.nan

table = table.loc[table.mouse_id.isin(mice_groups['gradual_day0'])]


# Plot reaction time histogram together with scatter
# plot of reaction time for whisker trials across
# the three days with trial_w on the x-axis and reaction_time on the y-axis.
# --------------------------------------------------------------------------

binwidth = 0.05  # 50 ms bin width
nwhiskertrials = 60

fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
data = table.loc[(table.reward_group=='R+') & (table.auditory_stim==1)] 
for day in [0, 1, 2]:
    day_data = data[data['day'] == day]
    sns.histplot(day_data, x='reaction_time',
                binwidth=binwidth, ax=axes[day], stat='probability', color='steelblue', alpha=0.7)
data = table.loc[(table.reward_group=='R+') & (table.whisker_stim==1)] 
for day in [0, 1, 2]:
    day_data = data[data['day'] == day]
    sns.histplot(day_data, x='reaction_time',
                binwidth=binwidth, ax=axes[day], stat='probability', color='#1b9e77', alpha=0.7)
# Plot reaction time for whisker trials as scatter, color-coded by trial_w (order in session)
for day in [0, 1, 2]:
    day_data = table[(table['reward_group'] == 'R+') & (table['whisker_stim'] == 1) & (table['day'] == day) & (table.trial_w <= nwhiskertrials)] 
    if not day_data.empty:
        cmap = cm.get_cmap('coolwarm', nwhiskertrials)
        norm = Normalize(vmin=day_data['trial_w'].min(), vmax=day_data['trial_w'].max())
        colors = cmap(norm(day_data['trial_w']))
        axes[day].scatter(day_data['reaction_time'], np.random.normal(0.4, 0.03, size=len(day_data)), 
                            c=colors, s=20, alpha=0.7, label='Whisker trials')
        # Add colorbar
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=axes[day], orientation='vertical', pad=0.01)
        cbar.set_label('Whisker trial order (trial_w)')
        axes[day].set_title(f'Day {day}')
# Save figure.
output_dir = io.adjust_path_to_host(r'/mnt/lsens-analysis/Anthony_Renard/analysis_output/fast-learning/behavior')
plt.savefig(os.path.join(output_dir, f'reaction_time_histogram_{nwhiskertrials}.svg'), format='svg', dpi=300)


# Plot reaction time to whisker trials across the three days with trial_w on the x-axis and reaction_time on the y-axis.
mwhiskertrials = 40
fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
for i, day in enumerate([0, 1, 2]):
    day_data = table[(table['reward_group'] == 'R+') & (table['whisker_stim'] == 1)
                        & (table['day'] == day) & (table.trial_w <= nwhiskertrials)
                        & (table.lick_flag == 1)]
    day_data = day_data.assign(trial_whit_id=day_data.groupby('session_id').cumcount() + 1)
    sns.pointplot(
        data=day_data,
        x='trial_whit_id',
        y='reaction_time',
        ax=axes[i],
        dodge=True,
        errorbar='ci',
        markers='o',
        linestyles='-',
    )
    axes[i].set_title(f'Day {day}')
sns.despine()
# Save figure.
output_dir = io.adjust_path_to_host(r'/mnt/lsens-analysis/Anthony_Renard/analysis_output/fast-learning/behavior')
plt.savefig(os.path.join(output_dir, f'reaction_time_pointplot.svg'), format='svg', dpi=300)




# ############################################################
# Opto inactivation.
# ############################################################

# Read behavior results.
db_path = io.db_path
# db_path = 'C://Users//aprenard//recherches//fast-learning//docs//sessions_muscimol_GF.xlsx'
nwb_dir = io.nwb_dir
stop_flag_yaml = io.stop_flags_yaml
trial_indices_yaml = io.trial_indices_yaml

mice_opto = io.select_mice_from_db(db_path, nwb_dir, experimenters=None,
                                    exclude_cols=['exclude', 'opto_exclude'],
                                    opto_inactivation_type = ['learning'],
                                    optogenetic = 'yes',
                                    )

# Read behavior results.
session_list, nwb_list, mice_list, db = io.select_sessions_from_db(
    db_path, nwb_dir, experimenters=None,
    exclude_cols=['exclude', 'opto_exclude'],
    opto_inactivation_type = ['learning'],
    opto_day = ["pre_-2", "pre_-1",
            "opto", "recovery_1",],
    # day = ['whisker_on_1', 'whisker_off', 'whisker_on_2'],
    mouse_id = mice_opto,
    )

table_opto_learning = make_behavior_table(nwb_list, session_list, db_path, cut_session=True, stop_flag_yaml=io.stop_flags_yaml, trial_indices_yaml=io.trial_indices_yaml)
# Save the table to a CSV file
save_path = r'//sv-nas1.rcp.epfl.ch/Petersen-Lab/analysis/Anthony_Renard/data_processed/behavior/behavior_opto_learning.csv'
save_path = io.adjust_path_to_host(save_path)
table_opto_learning.to_csv(save_path, index=False)

# Load table.
table_path = r'//sv-nas1.rcp.epfl.ch/Petersen-Lab/analysis/Anthony_Renard/data_processed/behavior/behavior_opto_learning.csv'
table_path = io.adjust_path_to_host(table_path)
table = pd.read_csv(table_path)

fpS1_mice = io.select_mice_from_db(db_path, nwb_dir, experimenters=None,
                                    exclude_cols = ['exclude', 'opto_exclude'],
                                    optogenetic = 'yes',
                                    opto_inactivation_type = 'learning',
                                    opto_area = 'fpS1',
                                    )

wS1_mice = io.select_mice_from_db(db_path, nwb_dir, experimenters=None,
                                    exclude_cols = ['exclude', 'opto_exclude'],
                                    optogenetic = 'yes',
                                    opto_inactivation_type = 'learning',
                                    opto_area = 'wS1',
                                    )

table.loc[table.mouse_id.isin(fpS1_mice), 'area'] = 'fpS1'
table.loc[table.mouse_id.isin(wS1_mice), 'area'] = 'wS1' 
table = pd.merge(
    table,
    db[['mouse_id', 'session_id', 'opto_day']],
    on=['mouse_id', 'session_id'],
    how='left'
)
inactivation_labels = ['pre_-2', 'pre_-1', 'opto', 'recovery_1']

data = table.groupby(['mouse_id', 'session_id', 'opto_day', 'area'], as_index=False)[['outcome_c','outcome_a','outcome_w']].agg('mean')
# Order the data by inactivation labels for each session
data['opto_day'] = pd.Categorical(data['opto_day'], categories=inactivation_labels, ordered=True)
data = data.sort_values(by=['mouse_id', 'opto_day'])
# Convert performance to percentage.
data['outcome_c'] = data['outcome_c'] * 100
data['outcome_a'] = data['outcome_a'] * 100
data['outcome_w'] = data['outcome_w'] * 100


# Plot inactivation for wS1 and fpS1.
# -----------------------------------

fig, axes = plt.subplots(1, 2, sharey=True, figsize=(12,5))

ax = axes[0]

for imouse in wS1_mice:
    sns.lineplot(data=data.loc[data.mouse_id==imouse], x='opto_day', y='outcome_c', estimator=np.mean, color=stim_palette[2],
                alpha=.6, legend=False, ax=ax, marker=None, err_style='bars', linewidth=1)
    sns.lineplot(data=data.loc[data.mouse_id==imouse], x='opto_day', y='outcome_a', estimator=np.mean, color=stim_palette[0],
                alpha=.6, legend=False, ax=ax, marker=None, err_style='bars', linewidth=1)
    sns.lineplot(data=data.loc[data.mouse_id==imouse], x='opto_day', y='outcome_w', estimator=np.mean, color=stim_palette[1],
                alpha=.6, legend=False, ax=ax, marker=None, err_style='bars', linewidth=1)

sns.pointplot(data=data.loc[data.mouse_id.isin(wS1_mice)], x='opto_day',
            y='outcome_c', order=inactivation_labels,
            color=stim_palette[2], ax=ax, linewidth=2)
sns.pointplot(data=data.loc[data.mouse_id.isin(wS1_mice)], x='opto_day',
            y='outcome_a', order=inactivation_labels,
            color=stim_palette[0], ax=ax, linewidth=2)
sns.pointplot(data=data.loc[data.mouse_id.isin(wS1_mice)], x='opto_day',
            y='outcome_w', order=inactivation_labels,
            color=stim_palette[1], ax=ax, linewidth=2)

# Add dots for individual points
sns.stripplot(data=data.loc[data.mouse_id.isin(wS1_mice)], x='opto_day',
              y='outcome_c', order=inactivation_labels,
              color=stim_palette[2], ax=ax, jitter=False, dodge=True, alpha=0.5, size=4)
sns.stripplot(data=data.loc[data.mouse_id.isin(wS1_mice)], x='opto_day',
              y='outcome_a', order=inactivation_labels,
              color=stim_palette[0], ax=ax, jitter=False, dodge=True, alpha=0.5, size=4)
sns.stripplot(data=data.loc[data.mouse_id.isin(wS1_mice)], x='opto_day',
              y='outcome_w', order=inactivation_labels,
              color=stim_palette[1], ax=ax, jitter=False, dodge=True, alpha=0.5, size=4)

ax = axes[1]

for imouse in fpS1_mice:
    sns.lineplot(data=data.loc[data.mouse_id==imouse], x='opto_day', y='outcome_c', estimator=np.mean, color=stim_palette[2],
                alpha=.6, legend=False, ax=ax, marker=None, err_style='bars', linewidth=1)
    sns.lineplot(data=data.loc[data.mouse_id==imouse], x='opto_day', y='outcome_a', estimator=np.mean, color=stim_palette[0],
                alpha=.6, legend=False, ax=ax, marker=None, err_style='bars', linewidth=1)
    sns.lineplot(data=data.loc[data.mouse_id==imouse], x='opto_day', y='outcome_w', estimator=np.mean, color=stim_palette[1],
                alpha=.6, legend=False, ax=ax, marker=None, err_style='bars', linewidth=1)

sns.pointplot(data=data.loc[data.mouse_id.isin(fpS1_mice)], x='opto_day',
            y='outcome_c', order=inactivation_labels,
            color=stim_palette[2], ax=ax, linewidth=2)
sns.pointplot(data=data.loc[data.mouse_id.isin(fpS1_mice)], x='opto_day',
            y='outcome_a', order=inactivation_labels,
            color=stim_palette[0], ax=ax, linewidth=2)
sns.pointplot(data=data.loc[data.mouse_id.isin(fpS1_mice)], x='opto_day',
            y='outcome_w', order=inactivation_labels,
            color=stim_palette[1], ax=ax, linewidth=2)

for ax in axes:
    ax.set_yticks([0,20, 40, 60, 80, 100])
    ax.set_xticklabels(['-2', '-1', '0', '+1'])
    ax.set_xlabel('Optogenetic inactivation during learning')
    ax.set_ylabel('Lick probability (%)')
sns.despine(trim=True)

# Save figure
output_dir = fr'/mnt/lsens-analysis/Anthony_Renard/analysis_output/fast-learning/behavior'
output_dir = io.adjust_path_to_host(output_dir)
svg_file = f'opto_learning.svg'
plt.savefig(os.path.join(output_dir, svg_file), format='svg', dpi=300)
# Save data.
data.to_csv(os.path.join(output_dir, 'opto_learning_data.csv'), index=False)

# Bar plot and stats for Day 0 (opto), Day +1 (recovery_1)
# ----------------------------------------------------------

days_of_interest = ['opto', 'recovery_1']
day_labels = ['D0', 'D+1']

day_data = data[data['opto_day'].isin(days_of_interest)].copy()
day_data['day_label'] = day_data['opto_day'].map(dict(zip(days_of_interest, day_labels)))

plt.figure(figsize=(8, 6))
sns.barplot(
    data=day_data,
    x='day_label',
    y='outcome_w',
    hue='area',
    palette=[stim_palette[1]],
    width=0.3,
    dodge=True,
    order=day_labels,
    hue_order=['wS1', 'fpS1']
)
sns.swarmplot(
    data=day_data,
    x='day_label',
    y='outcome_w',
    hue='area',
    dodge=True,
    color='black',
    alpha=0.6,
    order=day_labels,
    hue_order=['wS1', 'fpS1']
)
plt.xlabel('Day')
plt.ylabel('Whisker Performance (%)')
plt.ylim([0, 100])
plt.legend(title='Area')
sns.despine()

# Perform Mann-Whitney U test for each day between wS1 and fpS1
stats = []
for day, label in zip(days_of_interest, day_labels):
    df_day = day_data[day_data['opto_day'] == day]
    group_wS1 = df_day[df_day['area'] == 'wS1']['outcome_w']
    group_fpS1 = df_day[df_day['area'] == 'fpS1']['outcome_w']
    stat, p_value = mannwhitneyu(group_wS1, group_fpS1, alternative='two-sided')
    stats.append({'day': label, 'statistic': stat, 'p_value': p_value})
    # Add significance stars to the plot
    ax = plt.gca()
    xpos = day_labels.index(label)
    ypos = 95
    if p_value < 0.001:
        plt.text(xpos, ypos, '***', ha='center', va='bottom', color='black', fontsize=14)
    elif p_value < 0.01:
        plt.text(xpos, ypos, '**', ha='center', va='bottom', color='black', fontsize=14)
    elif p_value < 0.05:
        plt.text(xpos, ypos, '*', ha='center', va='bottom', color='black', fontsize=14)
    # Add p-value text below stars
    plt.text(xpos, 90, f'p={p_value:.3g}', ha='center', va='bottom', color='black', fontsize=10)

# Save the results to CSV files
output_dir = io.adjust_path_to_host(r'/mnt/lsens-analysis/Anthony_Renard/analysis_output/fast-learning/behavior')
plt.savefig(os.path.join(output_dir, 'opto_learning_day0_day1.svg'), format='svg', dpi=300)
day_data.to_csv(os.path.join(output_dir, 'opto_learning_day0_day1_data.csv'), index=False)
pd.DataFrame(stats).to_csv(os.path.join(output_dir, 'opto_learning_day0_day1_stats.csv'), index=False)


# ##############################################
# Fit learning curves and define learning trial.
# ##############################################§

# Load behavior results.
# ----------------------

db_path = io.db_path
# db_path = 'C://Users//aprenard//recherches//fast-learning//docs//sessions_muscimol_GF.xlsx'
nwb_dir = io.nwb_dir
stop_flag_yaml = io.stop_flags_yaml
trial_indices_yaml = io.trial_indices_yaml

experimenters = ['AR', 'GF', 'MI']
mice_imaging = io.select_mice_from_db(db_path, nwb_dir,
                                    experimenters = experimenters,
                                    exclude_cols = ['exclude',  'two_p_exclude'],
                                    optogenetic = ['no', np.nan],
                                    pharmacology = ['no',np.nan],
                                    two_p_imaging = 'yes'
                                    )
# Load the table from the CSV file.
# table_file = io.adjust_path_to_host(r'/mnt/lsens-analysis/Anthony_Renard/data_processed/behavior/behavior_imagingmice_table_5days_cut.csv')
table_file = io.adjust_path_to_host(r'/mnt/lsens-analysis/Anthony_Renard/data_processed/behavior/behavior_imagingmice_table_5days_cut_with_learning_curves.csv')
table = pd.read_csv(table_file)

# Fit learning curves and define learning trial.
table = compute_learning_curves(table)
table = compute_learning_trial(table, n_consecutive_trials=10)

# Save updated table.
save_path = io.adjust_path_to_host(r'/mnt/lsens-analysis/Anthony_Renard/data_processed/behavior/behavior_imagingmice_table_5days_cut_with_learning_curves.csv')
table.to_csv(save_path, index=False)

# table = pd.read_csv(save_path)


# d = table.loc[(table.day==0) & (table.reward_group=='R+') & (table.whisker_stim==1) & (table.mouse_id=='GF305')]

# dd = fit_learning_curve(d.outcome_w.values, alpha=1, beta=1)
# plt.plot(d.hr_w.reset_index(drop=True))
# plt.plot(dd[1])


# # Convert learning curve columns to float type to avoid dtype issues
# learning_curve_cols = [
#     'learning_curve_w', 'learning_curve_w_ci_low', 'learning_curve_w_ci_high',
#     'learning_curve_a', 'learning_curve_a_ci_low', 'learning_curve_a_ci_high',
#     'learning_curve_ns', 'learning_curve_ns_ci_low', 'learning_curve_ns_ci_high'
# ]
# for col in learning_curve_cols:
#     if col in table.columns:
#         table[col] = pd.to_numeric(table[col], errors='coerce')

# Day 0 learning curves pdf
pdf_path = '/mnt/lsens-analysis/Anthony_Renard/analysis_output/fast-learning/day0_learning/behavior/learning_curves_day0.pdf'
pdf_path = io.adjust_path_to_host(pdf_path)
session_list = table.session_id.unique()
sns.set_theme(context='paper', style='ticks', palette='deep', font='sans-serif', font_scale=1)

def plot_learning_curves_pdf(table, session_list, pdf_path):
    with PdfPages(pdf_path) as pdf:
        for session_id in session_list:
            data = table.loc[table.session_id == session_id]
            if data.day.iloc[0] != 0:
                continue
            reward_group = data.reward_group.values[0]
            color = reward_palette[0] if reward_group == 'R-' else reward_palette[1]

            d = data.loc[data.whisker_stim == 1].reset_index(drop=True)


            learning_curve_w = d.learning_curve_w.values.astype(float)
            learning_ci_low = d.learning_curve_w_ci_low.values.astype(float)
            learning_ci_high = d.learning_curve_w_ci_high.values.astype(float)
            learning_chance = d.learning_curve_chance.astype(float)

            plt.plot(d.trial_w, learning_curve_w, label='Whisker (learning curve)', color=color)
            plt.fill_between(d.trial_w, learning_ci_low, learning_ci_high, color=color, alpha=0.2)
            sns.lineplot(data=d, x='trial_w', y='hr_w', color=color, legend=False, linestyle='--')
            plt.plot(d.trial_w, learning_chance, label='No_stim', color=stim_palette[2])

            learning_trial = data.learning_trial.values[0]
            if not pd.isna(learning_trial):
                plt.axvline(x=learning_trial, color='black', linestyle='--', label='Learning trial')
            plt.title(f'Session {session_id} - {reward_group}')
            plt.ylim([0, 1])
            plt.xlabel('Whisker trial')
            plt.ylabel('Lick probability')
            plt.legend()
            sns.despine()

            pdf.savefig()
            plt.close()

plot_learning_curves_pdf(table, session_list, pdf_path)
