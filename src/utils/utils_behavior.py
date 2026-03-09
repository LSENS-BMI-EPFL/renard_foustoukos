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
from nwb_wrappers import nwb_reader_functions as nwb_read
from scipy.stats import mannwhitneyu, wilcoxon
from matplotlib.colors import Normalize
from scipy.ndimage import gaussian_filter1d
import matplotlib.cm as cm


def make_behavior_table(nwb_list, session_list, db_path, cut_session, stop_flag_yaml, trial_indices_yaml):
    if cut_session:
        start_stop, trial_indices = io.read_stop_flags_and_indices_yaml(stop_flag_yaml, trial_indices_yaml)
    table = []
    for nwb, session in zip(nwb_list, session_list):
        df = nwb_read.get_trial_table(nwb)
        df = df.reset_index()
        if 'trial_id' not in df.columns:
            df.rename(columns={'id': 'trial_id'}, inplace=True)
        reward_group = io.get_reward_group_from_db(db_path, session)
        metadata = nwb_read.get_session_metadata(nwb)
        df['day'] = metadata['day']
        df['behavior_type'] = metadata['behavior_type']
        df['session_id'] = session
        df['mouse_id'] = session[:5]
        df['reward_group'] = reward_group

        # Cut session.
        if cut_session:
            flags = start_stop[session]
            df = df.loc[df['trial_id']<=flags[1]]
        df = compute_performance(df)
        table.append(df)
    table = pd.concat(table)
    
    return table

def add_db_metadata_to_table(table, db_path, session_list):
    db = io.read_excel_db(db_path)
    db = db.loc[db['session_id'].isin(session_list)]
    db = db[['session_id', 'reward_group', 'pharmacology', 'pharma_day', 'pharma_inactivation_type', 'pharma_area']]
    table = pd.merge(table, db, on='session_id', how='left')
    return table
    
def cut_sessions(table, stop_flag_yaml, trial_indices_yaml):
    start_stop, trial_indices = io.read_stop_flags_and_indices_yaml(stop_flag_yaml, trial_indices_yaml)
    
    temp = []
    for session, flags in start_stop.items():
        session_data = table.loc[table['session_id'] == session]
        session_data = session_data.loc[session_data['trial_id']<=flags[1]]
        temp.append(session_data)
    filtered_table = pd.concat(temp)
    filtered_table = filtered_table.reset_index(drop=True)
    
    return filtered_table


def compute_performance(table, block_size=20):
    
    # Add trial number for each stimulus.
    table['trial_w'] = table.loc[(table.early_lick==0) & (table.whisker_stim==1.0)]\
                                    .groupby('session_id').cumcount()+1
    table['trial_a'] = table.loc[(table.early_lick==0) & (table.auditory_stim==1.0)]\
                                    .groupby('session_id').cumcount()+1
    table['trial_c'] = table.loc[(table.early_lick==0) & (table.no_stim==1.0)]\
                                    .groupby('session_id').cumcount()+1

    # Add performance outcome column for each stimulus.
    table['outcome_w'] = table.loc[(table.whisker_stim==1.0)]['lick_flag']
    table['outcome_a'] = table.loc[(table.auditory_stim==1.0)]['lick_flag']
    table['outcome_c'] = table.loc[(table.no_stim==1.0)]['lick_flag']

    # Cumulative sum performance representation.
    table['cumsum_w'] = table.loc[(table.whisker_stim==1.0)]['outcome_w']\
                                    .map({0:-1,1:1,np.nan:0})
    table['cumsum_w'] = table.groupby('session_id')['cumsum_w'].transform('cumsum')
    table['cumsum_a'] = table.loc[(table.auditory_stim==1.0)]['outcome_a']\
                                    .map({0:-1,1:1,np.nan:0})
    table['cumsum_a'] = table.groupby('session_id')['cumsum_a'].transform('cumsum')
    table['cumsum_c'] = table.loc[(table.no_stim==1.0)]['outcome_c']\
                                    .map({0:-1,1:1,np.nan:0})
    table['cumsum_c'] = table.groupby('session_id')['cumsum_c'].transform('cumsum')


    # Add block index of n=BLOCK_SIZE trials and compute hit rate on them.
    table['block_id'] = table.loc[table.early_lick==0, 'trial_id'].transform(lambda x: x // block_size)
    # Compute hit rates. Use transform to propagate hit rate to all entries.
    table['hr_w'] = table.groupby(['session_id', 'block_id'], as_index=False)['outcome_w']\
                                .transform('mean')
    table['hr_a'] = table.groupby(['session_id', 'block_id'], as_index=False)['outcome_a']\
                                .transform('mean')
    table['hr_c'] = table.groupby(['session_id', 'block_id'], as_index=False)['outcome_c']\
                                .transform('mean')

    # # Also store performance in new data frame with block as time steps for convenience.
    # cols = ['session_id','mouse_id','session_day','reward_group','block_id']
    # df_block_perf = table.loc[table.early_lick==0, cols].drop_duplicates()
    # df_block_perf = df_block_perf.reset_index()
    # # Compute hit rates.
    # df_block_perf['hr_w'] = table.groupby(['session_id', 'block_id'], as_index=False)\
    #                                 .agg(np.nanmean)['outcome_w']
    # df_block_perf['hr_a'] = table.groupby(['session_id', 'block_id'], as_index=False)\
    #                                 .agg(np.nanmean)['outcome_a']
    # df_block_perf['hr_c'] = table.groupby(['session_id', 'block_id'], as_index=False)\
    #                                 .agg(np.nanmean)['outcome_c']

    return table


def plot_single_mouse_performance_across_days(df, mouse_id, color_palette):
    """ Plots performance of a single mouse across all auditory and whisker sessions.

    Args:
        df (_type_): _description_
        mouse_id (_type_): _description_
        color_palette (_type_): _description_
    """

    # Set plot parameters.
    raster_marker = 2
    marker_width = 2
    block_size = 20
    reward_group = table.loc[table.mouse_id==mouse_id, 'reward_group'].iloc[0]
    color_wh = color_palette[2]
    if reward_group == 'R-':  # green for whisker R+, red for R-.
        color_wh = color_palette[3]

    # Find ID of sessions to plot.
    # Start with more days than needed in the right order and filter out.
    # days = [f'-{i}' for i in range(30,0,-1)] + ['0'] + [f'+{i}' for i in range(1,30)]
    days  = [iday for iday in table.day.drop_duplicates()]
    # Order dataframe by days.
    f = lambda x: x.map({iday: iorder for iorder, iday in enumerate(days)})
    table = table.sort_values(by='day', key=f)
    sessions = table.loc[(table.mouse_id==mouse_id) & (table.day.isin(days)),'session_id']
    sessions = sessions.drop_duplicates().to_list()
    
    # Initialize figure.
    # f, axes = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1, 5]}, figsize=(20,6))
    fig = plt.figure(figsize=(4,6))
    ax = plt.gca()

    # Plot performance.
    d = table.loc[(table.session_id.isin(sessions)), ['session_id','day','hr_c','hr_a','hr_w']]
    d = d.groupby(['session_id','day'], as_index=False).agg("mean")
    d = d.sort_values(by='day', key=f)
    sns.lineplot(data=d, x='day', y='hr_c', color=color_palette[5], ax=ax,
                 marker='o')
    sns.lineplot(data=d, x='day', y='hr_a', color=color_palette[0], ax=ax,
                 marker='o')
    if reward_group in ['R+', 'R-']:
        sns.lineplot(data=d, x='day', y='hr_w', color=color_wh, ax=ax,
                     marker='o')

    ax.set_ylim([-0.2,1.05])
    ax.set_xlabel('Performance across training days')
    ax.set_ylabel('Lick probability')
    ax.set_title('Training days')
    fig.suptitle(f'{mouse_id}')
    sns.despine()


def plot_single_session(table, session_id, palette, ax=None, do_scatter=True, linewidth=2):
        """ Plots performance of a single session with average per block and single trial raster plot.

        Args:
            table (_type_): _description_
            session_id (_type_): _description_
            palette (_type_): _description_
            ax: matplotlib axis (optional)
            do_scatter: bool, whether to plot scatter raster
            n_trial_xticks: int, x-tick interval
            linewidth: int or float, line width for line plots
        """
        
        # Set plot parameters.
        raster_marker = 2
        marker_width = 2
        figsize = (15,6)
        block_size = 20
        reward_group = table.loc[table.session_id==session_id, 'reward_group'].iloc[0]
        color_wh = palette[3]
        if reward_group == 'R-':  # #238443 for whisker R+, #d51a1c for R-.
            color_wh = palette[2]    

        # Initialize figure.
        if ax is None:
            f = plt.figure(figsize=figsize)
            ax = plt.gca()

        # Plot performance of the session across time.
        d = table.loc[table.session_id==session_id]
        # Select entries with trial numbers at middle of each block for alignment with
        # raster plot.
        d = d.sort_values(['block_id'])
        d = d.loc[d.early_lick==0][int(block_size/2)::block_size]
        if not d.hr_c.isna().all():
            sns.lineplot(data=d, x='block_id', y='hr_c', color=palette[5], ax=ax,
                         marker='o', linewidth=linewidth)
        if not d.hr_a.isna().all():
            sns.lineplot(data=d, x='block_id', y='hr_a', color=palette[1], ax=ax,
                         marker='o', linewidth=linewidth)
        if not d.hr_w.isna().all():
            sns.lineplot(data=d, x='block_id', y='hr_w', color=color_wh, ax=ax,
                         marker='o', linewidth=linewidth)

        if do_scatter:
            # Plot single trial raster plot.
            d = table.loc[table.session_id==session_id]
            if not d.hr_c.isna().all():
                ax.scatter(x=d.loc[d.lick_flag==0]['trial_id'], y=d.loc[d.lick_flag==0]['outcome_c']-0.1,
                           color=palette[4], marker=raster_marker, linewidths=marker_width)
                ax.scatter(x=d.loc[d.lick_flag==1]['trial_id'], y=d.loc[d.lick_flag==1]['outcome_c']-1.1,
                           color=palette[5], marker=raster_marker, linewidths=marker_width)

            if not d.hr_a.isna().all():
                ax.scatter(x=d.loc[d.lick_flag==0]['trial_id'], y=d.loc[d.lick_flag==0]['outcome_a']-0.15,
                           color=palette[0], marker=raster_marker, linewidths=marker_width)
                ax.scatter(x=d.loc[d.lick_flag==1]['trial_id'], y=d.loc[d.lick_flag==1]['outcome_a']-1.15,
                           color=palette[1], marker=raster_marker, linewidths=marker_width)

            if not d.hr_w.isna().all():
                ax.scatter(x=d.loc[d.lick_flag==0]['trial_id'], y=d.loc[d.lick_flag==0]['outcome_w']-0.2,
                           color=palette[2], marker=raster_marker, linewidths=marker_width)
                ax.scatter(x=d.loc[d.lick_flag==1]['trial_id'], y=d.loc[d.lick_flag==1]['outcome_w']-1.2,
                           color=palette[3], marker=raster_marker, linewidths=marker_width)

        nmax_blocks = d.block_id.max()
        if do_scatter:
            ax.set_ylim([-0.2,1.05])
        else:
            ax.set_ylim([-0.1,1.1])
        # Set x-ticks and labels for each block
        if nmax_blocks is not None and not np.isnan(nmax_blocks):
            xticks = np.arange(0, nmax_blocks + 1)
            ax.set_xticks(xticks)
            ax.set_xticklabels([str(x) for x in xticks])
        ax.set_xlabel('Block (20 trials)')
        ax.set_ylabel('Lick probability')
        ax.set_title(f'{session_id}')
        sns.despine()
    

def plot_perf_across_blocks(data, reward_group, day, palette, nmax_trials=None, ax=None):
    
    if nmax_trials:
        data = data.loc[(data.trial_id<=nmax_trials)]
    data = data.loc[(data.reward_group==reward_group) & (data.day==day),
                ['mouse_id','session_id','block_id','hr_c','hr_a','hr_w']]
    data = data.groupby(['mouse_id','session_id', 'block_id'], as_index=False).agg("mean")

    color_c = palette[5]
    if reward_group=='R-':
        color_c = palette[4]
    sns.lineplot(
        data=data, x='block_id', y='hr_c', estimator='mean',
        color=color_c, alpha=1, legend=True, marker='o',
        markerfacecolor=color_c, markeredgecolor=color_c,  # No border, same as fill
        errorbar='ci', err_style='band', ax=ax
    )
    
    color_a = palette[1]
    if reward_group=='R-':
        color_a = palette[0]
    sns.lineplot(data=data, x='block_id', y='hr_a', estimator='mean',
                 color=color_a, alpha=1, legend=True, marker='o',
                 markerfacecolor=color_a, markeredgecolor=color_a,  # No border, same as fill
                 errorbar='ci', err_style='band', ax=ax)
    
    color_wh = palette[3]
    if reward_group=='R-':
        color_wh = palette[2]
    sns.lineplot(data=data, x='block_id', y='hr_w', estimator='mean',
                 color=color_wh ,alpha=1, legend=True, marker='o',
                 markerfacecolor=color_wh, markeredgecolor=color_wh,  # No border, same as fill
                 errorbar='ci', err_style='band', ax=ax)

    nblocks = int(data.block_id.max())
    if not ax:
        ax = plt.gca()
    ax.set_xticks(range(1, 14))
    ax.set_xticklabels([str(i) for i in range(1, 14)])
    ax.set_ylim([0,1.1])
    ax.set_yticks([i*0.2 for i in range(6)])
    ax.set_xlabel('Block (20 trials)')
    ax.set_ylabel('Lick probability')
    sns.despine(trim=True)


def fit_learning_curve(outcomes, alpha=1, beta=1):
    n_trials = len(outcomes)
    conf_int = 80  # Confidence interval percentage
    if n_trials == 0:
        return None, None, None, None

    with pm.Model() as model:

        # Precision (Inverse Variance)
        tau = pm.Gamma("tau", alpha=alpha, beta=beta)
        sigma = pm.Deterministic("sigma", 1 / pm.math.sqrt(tau))  # Convert to std dev

        # Define latent states as a Gaussian Random Walk (GRW)
        x = pm.GaussianRandomWalk("x", mu=0, sigma=sigma, shape=n_trials) #mu here is for the gaussian noise, different from mu

        # Logit transformation: logit(p[t]) = x[t]
        p = pm.Deterministic("p", pm.math.invlogit(x))

        # # Bernoulli likelihood for observed behavior
        obs = pm.Bernoulli("obs", p=p, observed=outcomes)

        # MCMC Sampling
        trace = pm.sample(1000, tune=10, cores=8, chains=4)

    # Extract posterior mean and credible intervals for p (Pr(correct))
    p_samples = trace.posterior["p"].values.reshape(-1, n_trials)
    p_mean = np.mean(p_samples, axis=0)
    p_low, p_high = np.percentile(p_samples, [int((100 - conf_int) / 2), int(100 - (100 - conf_int) / 2)], axis=0)
    
    return p_samples, p_mean, p_low, p_high


def compute_learning_curves(table):
        
    session_list = table.session_id.unique()
    
    for session in session_list:
        data = table.loc[table.day.isin([-2,-1,0,1,2])].reset_index(drop=True).copy()
        # if data.day[0] != 0:
        #     continue
        print(f'Processing session {session}...')

        data_w = data[(data.session_id == session) & (data.whisker_stim==1)].reset_index(drop=True)
        outcomes = data_w.outcome_w.values
        p_samples_w, p_mean_w, p_low_w, p_high_w = fit_learning_curve(outcomes)
        
        data_a = data[(data.session_id == session) & (data.auditory_stim==1)].reset_index(drop=True)
        outcomes = data_a.outcome_a.values
        p_samples_a, p_mean_a, p_low_a, p_high_a = fit_learning_curve(outcomes)
        
        data_ns = data[(data.session_id == session) & (data.no_stim==1)].reset_index(drop=True)
        outcomes = data_ns.outcome_c.values
        p_samples_ns, p_mean_ns, p_low_ns, p_high_ns = fit_learning_curve(outcomes)

        if p_mean_w is not None:
            table.loc[(table.session_id==session) & (table.whisker_stim==1), 'learning_curve_w'] = p_mean_w.astype(float)
            table.loc[(table.session_id==session) & (table.whisker_stim==1), 'learning_curve_w_ci_low'] = p_low_w.astype(float)
            table.loc[(table.session_id==session) & (table.whisker_stim==1), 'learning_curve_w_ci_high'] = p_high_w.astype(float)
        
        if p_mean_a is not None:
            table.loc[(table.session_id==session) & (table.auditory_stim==1), 'learning_curve_a'] = p_mean_a.astype(float)
            table.loc[(table.session_id==session) & (table.auditory_stim==1), 'learning_curve_a_ci_low'] = p_low_a.astype(float)
            table.loc[(table.session_id==session) & (table.auditory_stim==1), 'learning_curve_a_ci_high'] = p_high_a.astype(float)
        
        if p_mean_ns is not None:
            table.loc[(table.session_id==session) & (table.no_stim==1), 'learning_curve_ns'] = p_mean_ns.astype(float)
            table.loc[(table.session_id==session) & (table.no_stim==1), 'learning_curve_ns_ci_low'] = p_low_ns.astype(float)
            table.loc[(table.session_id==session) & (table.no_stim==1), 'learning_curve_ns_ci_high'] = p_high_ns.astype(float)

    return table


def compute_learning_trial(table, n_consecutive_trials=10):

    for session in table.session_id.unique():
        if table.loc[(table.session_id == session), 'day'].values[0] != 0:
            continue
        reward_group = table.loc[(table.session_id == session), 'reward_group'].values[0]
        print(f'Processing session {session}...')

        print(f'Defining learning trial for session {session}...')
        data_w = table[(table.session_id == session) & (table.whisker_stim==1)].reset_index(drop=True)
        data_ns = table[(table.session_id == session) & (table.no_stim==1)].reset_index(drop=True)
        # Get performance at whisker stim times
        p_mean_w = data_w.learning_curve_w.values
        p_low_w = data_w.learning_curve_w_ci_low.values
        p_high_w = data_w.learning_curve_w_ci_high.values
        p_mean_ns = data_ns.learning_curve_ns.values

        # Define bounds and get no stim timestamps in range
        timestamps_no_stim = data_ns.start_time.values
        timestamps_whisker = data_w['start_time'].values
        bounds_w = (min(timestamps_whisker), max(timestamps_whisker))
        bounds_n = (min(timestamps_no_stim), max(timestamps_no_stim))
        interp_range = np.linspace(min(bounds_w[0], bounds_n[0]), max(bounds_w[1], bounds_n[1]), len(p_mean_ns))
        # Interpolation - cubic (smooth) and bounded
        interp_func = sp.interpolate.CubicSpline(x=interp_range,
                                                    y=p_mean_ns,
                                                    extrapolate=False)
        # Interpolate false alarm rate at whisker stim times
        interp_p_far = interp_func(timestamps_whisker)

        # Identify learning trial
        learning_trial = np.nan
        if reward_group == 'R+':
            # R+: learning when whisker perf lower CI bound exceeds false alarm rate
            trials_above_chance = np.where(p_low_w > interp_p_far)[0]
            trials_above_chance = data_w.loc[trials_above_chance, 'trial_w'].values
            for idx in range(len(trials_above_chance)):
                if idx + n_consecutive_trials-1 < len(trials_above_chance):
                    if np.all(np.diff(trials_above_chance[idx:idx+n_consecutive_trials]) == 1):
                        learning_trial = trials_above_chance[idx]
                        break
        elif reward_group == 'R-':
            # R-: learning when whisker perf is no longer different from false alarm rate,
            # i.e. the false alarm rate falls within the whisker CI [p_low_w, p_high_w]
            trials_at_chance = np.where((interp_p_far >= p_low_w) & (interp_p_far <= p_high_w))[0]
            trials_at_chance = data_w.loc[trials_at_chance, 'trial_w'].values
            for idx in range(len(trials_at_chance)):
                if idx + n_consecutive_trials-1 < len(trials_at_chance):
                    if np.all(np.diff(trials_at_chance[idx:idx+n_consecutive_trials]) == 1):
                        learning_trial = trials_at_chance[idx]
                        break

        table.loc[(table.session_id==session) & (table.whisker_stim==1), 'learning_curve_chance'] = interp_p_far
        table.loc[(table.session_id==session), 'learning_trial'] = learning_trial
    return table
