"""
Behavior quantified via d' (signal detection theory) instead of whisker hit
rate.

Addresses a reviewer comment asking that behavioral performance be
quantified with d' rather than raw whisker hit rate, for:
    - Figure 1C (right)
    - Figure 1D
    - Figure 2C and 2F
    - Figure 4C and E
    - The execution-inactivation panel (figure_2b_c_execution.py)

------------------------------------------------------------------------
How d' is computed
------------------------------------------------------------------------
d' = Z(hit rate) - Z(false-alarm rate), where Z = Phi^-1 is the inverse
standard normal CDF (scipy.stats.norm.ppf).

  - "Hit rate"        = probability of licking on whisker-stimulus trials
                         (the signal).
  - "False-alarm rate"= probability of licking on no-stim/catch trials
                         (the noise). Every session already records catch
                         trials interleaved with whisker/auditory trials
                         (outcome_c in the trial tables), so no new data
                         collection is needed.

Two computation modes are used, matching the two aggregation granularities
already present in the original hit-rate panels:

1. Session-level d' (count-based, log-linear corrected). Panels that
   average outcome_w/outcome_c into one hit-rate/FA-rate value per session
   (Fig 1C-right, 2C, 2F, execution panel c) instead compute d' directly
   from that session's trial counts:
       hits = sum(outcome_w),  n_hit = # whisker trials
       fas  = sum(outcome_c),  n_fa  = # catch trials
   A session can have as few as ~10-20 catch trials, so 0% or 100% rates
   are common and would give +/-inf after the z-transform. This is
   corrected with the standard log-linear correction (Hautus, 1995):
       hit_rate' = (hits + 0.5) / (n_hit + 1)
       fa_rate'  = (fas  + 0.5) / (n_fa  + 1)
       d' = Z(hit_rate') - Z(fa_rate')
   This is the standard, minimal-bias correction used throughout the
   signal-detection literature for exactly this small-N/extreme-rate
   problem. See `_dprime_from_counts`.

2. Trial-resolved d' (from the existing Bayesian learning curves). Panels
   that plot a per-trial trajectory across Day-0 whisker trials (Fig 1D,
   4C, 4E) already have, in the processed data
   (behavior_imagingmice_table_5days_cut_with_learning_curves.csv), a
   smoothed per-trial lick-probability curve fit separately to whisker
   trials (learning_curve_w) and catch trials (learning_curve_ns), plus
   learning_curve_chance: the catch curve cubic-spline-interpolated onto
   whisker-trial indices (utils_behavior.compute_learning_trial). These
   are Bayesian posterior means that only asymptotically approach 0/1, so
   rather than the count-based correction, they are just clipped a small
   epsilon away from the boundary before the z-transform:
       d'(trial) = Z(clip(learning_curve_w, eps, 1-eps))
                 - Z(clip(learning_curve_chance, eps, 1-eps))
   with eps = 1e-3. See `_dprime_from_rates` / `_load_trial_resolved_dprime`.

Everything else (grouping, day/mouse filters, bar/swarm plots, Mann-Whitney
U or Wilcoxon tests, output file naming) is kept identical to the original
panel -- only the plotted/tested quantity changes.

------------------------------------------------------------------------
Panel-specific notes / scope decisions
------------------------------------------------------------------------
- Figure 1D: the original has two subplots -- raw single-trial outcomes
  (left) and fitted learning curves (right). A per-trial d' isn't
  meaningful at single-trial (binary, N=1) resolution, so only the
  right-subplot's trajectory is reproduced here, as a single-panel plot.

- Figure 4C: only the behavioral row (row 1: learning_curve_w vs trial) is
  reproduced. Row 2 (decoder decision value) is not a hit-rate-based
  quantity and is unchanged/out of scope.

- Figure 4D is out of scope: the original figure_4d.py doesn't plot a
  hit-rate-based quantity at all -- it plots the slope of a neural
  decoder's decision value (the behavior table it loads is unused, dead
  code), so there's no whisker-hit-rate metric there for d' to replace.

- Figure 4E: correlates a neural decoder's decision value against
  behavior. Only the behavioral variable correlated against it changes,
  from learning_curve_w to the trial-resolved d' -- the decoder itself
  isn't a hit-rate-based quantity, so the reviewer's request doesn't apply
  to it. Rather than duplicating figure_4e.py's inline decoder retraining,
  this reuses the pre-trained decoder cached in decoder_weights.pkl (built
  by pipeline/06_decoder.py) that figure_4b.py/figure_4c.py already reuse, applied
  via the same sliding window -- keeping one decoder consistent across
  Figure 4 instead of two subtly different ones. Still needs
  NWB/network-drive access to load the Day-0 imaging tensors.

- Execution panel c: reuses the trial-level table of the execution sessions
  (behavior_muscimol_execution.csv, built by 03_behavior_tables.py).
"""

import os
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors
import seaborn as sns
from scipy.stats import norm, mannwhitneyu, wilcoxon, ttest_1samp, pearsonr
from statsmodels.stats.multitest import multipletests

from fast_learning import paths, database
from fast_learning import imaging
from fast_learning.plotting import stim_palette, reward_palette, behavior_palette, save_figure

# Muscimol execution sessions, as in figure_2b_c_execution.py (keep in sync).
EXECUTION_TABLE_PATH = os.path.join(paths.processed_dir, 'behavior', 'behavior_muscimol_execution.csv')
EXECUTION_DAYS_OF_INTEREST = ['muscimol_1', 'ringer_1', 'muscimol_2']
EXECUTION_DAY_LABELS = ['M1', 'Ringer', 'M2']


OUTPUT_DIR = os.path.join(paths.manuscript_output_dir, 'revisions', 'behavior_dprime', 'output')
EPS = 1e-3  # clipping epsilon for already-smoothed probability curves


# ============================================================================
# Shared d' helpers
# ============================================================================


def _dprime_from_counts(hits, n_hit, fas, n_fa):
    """d' from raw trial counts, with the log-linear correction (Hautus,
    1995) that avoids +/-inf when a session's hit or FA rate is exactly 0
    or 1 (common with the ~10-30 trial/session counts here)."""
    hit_rate = (hits + 0.5) / (n_hit + 1)
    fa_rate = (fas + 0.5) / (n_fa + 1)
    return norm.ppf(hit_rate) - norm.ppf(fa_rate)


def _dprime_from_rates(p_hit, p_fa, eps=EPS):
    """d' from already-smoothed probability curves (the Bayesian learning
    curves), which only asymptote toward 0/1 -- a light clip is enough to
    keep the z-transform finite."""
    p_hit = np.clip(p_hit, eps, 1 - eps)
    p_fa = np.clip(p_fa, eps, 1 - eps)
    return norm.ppf(p_hit) - norm.ppf(p_fa)


def _session_dprime(table, group_cols):
    """Collapse a trial-level table to one d' value per group (typically
    one row per session), using whisker trials as signal and no-stim/catch
    trials as noise."""
    counts = (
        table.groupby(group_cols)
        .agg(
            n_hit=('outcome_w', 'count'),
            hits=('outcome_w', 'sum'),
            n_fa=('outcome_c', 'count'),
            fas=('outcome_c', 'sum'),
        )
        .reset_index()
    )
    counts['dprime_w'] = _dprime_from_counts(counts['hits'], counts['n_hit'], counts['fas'], counts['n_fa'])
    return counts


def _load_trial_resolved_dprime(
    table_path=os.path.join(
        paths.processed_dir, 'behavior', 'behavior_imagingmice_table_5days_cut_with_learning_curves.csv'
    ),
    n_trials=120,
):
    """Load the Day-0 whisker-trial table (used by figure_1d.py and
    figure_4c/d/e.py) and add a trial-resolved d' column, computed from the
    existing Bayesian learning-curve fits:
        dprime_w(trial) = Z(learning_curve_w) - Z(learning_curve_chance)
    where learning_curve_w is the smoothed whisker hit-probability curve
    and learning_curve_chance is the smoothed no-stim/catch false-alarm
    probability curve, already interpolated onto whisker-trial indices
    (utils_behavior.compute_learning_trial)."""
    table_path = paths.adjust_path_to_host(table_path)
    table = pd.read_csv(table_path)

    df = table.loc[(table.whisker_stim == 1) & (table.day == 0)].copy()
    df = df.loc[df.trial_w <= n_trials]

    df['dprime_w'] = _dprime_from_rates(df['learning_curve_w'], df['learning_curve_chance'])

    return df


# ============================================================================
# Shared bar+swarm+Mann-Whitney plotting helper
# (Figure 1C-right, 2C, 2F, execution panel c)
# ============================================================================


def _dprime_barplot_by_group(
    data,
    day_col,
    days_of_interest,
    day_labels,
    group_col,
    group_order,
    bar_palette,
    filename,
    output_dir=OUTPUT_DIR,
    save_format='svg',
    dpi=300,
    xlabel='Day',
    ylabel="d' (whisker vs. catch)",
    legend_title=None,
    swarm_color='grey',
):
    """Bar + swarm plot of dprime_w across days_of_interest, split into the
    two levels of group_col, with a Mann-Whitney U test between them for
    each day. Mirrors the bar+swarm+MWU scaffold shared by the original
    whisker-hit-rate panels (figure_1c.py right, figure_2b_c.py panel c,
    figure_2e_f.py panel f, figure_2b_c_execution.py panel c), with
    dprime_w substituted for outcome_w, and an unbounded y-axis instead of
    the original's [0, 100] percentage range."""

    day_data = data[data[day_col].astype(str).isin([str(d) for d in days_of_interest])].copy()
    day_data['day_label'] = (
        day_data[day_col].astype(str).map(dict(zip([str(d) for d in days_of_interest], day_labels)))
    )

    sns.set_theme(context='paper', style='ticks', palette='deep', font='sans-serif', font_scale=1)
    plt.figure(figsize=(8, 6))

    sns.barplot(
        data=day_data,
        x='day_label',
        y='dprime_w',
        hue=group_col,
        order=day_labels,
        hue_order=group_order,
        palette=bar_palette,
        width=0.3,
        dodge=True,
    )
    sns.swarmplot(
        data=day_data,
        x='day_label',
        y='dprime_w',
        hue=group_col,
        order=day_labels,
        hue_order=group_order,
        dodge=True,
        color=swarm_color,
        alpha=0.6,
    )

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.axhline(0, color='black', linestyle='--', alpha=0.4, linewidth=1)
    if legend_title:
        plt.legend(title=legend_title)
    sns.despine()

    y_top = day_data['dprime_w'].max()
    y_range = day_data['dprime_w'].max() - day_data['dprime_w'].min()
    ypos_star = y_top + 0.12 * y_range
    ypos_p = y_top + 0.04 * y_range

    stats = []
    for day, label in zip(days_of_interest, day_labels):
        df_day = day_data[day_data['day_label'] == label]
        group_a = df_day[df_day[group_col] == group_order[0]]['dprime_w']
        group_b = df_day[df_day[group_col] == group_order[1]]['dprime_w']

        stat, p_value = mannwhitneyu(group_a, group_b, alternative='two-sided')
        stats.append({'day': label, 'statistic': stat, 'p_value': p_value})

        xpos = day_labels.index(label)
        if p_value < 0.001:
            plt.text(xpos, ypos_star, '***', ha='center', va='bottom', color='black', fontsize=14)
        elif p_value < 0.01:
            plt.text(xpos, ypos_star, '**', ha='center', va='bottom', color='black', fontsize=14)
        elif p_value < 0.05:
            plt.text(xpos, ypos_star, '*', ha='center', va='bottom', color='black', fontsize=14)
        plt.text(xpos, ypos_p, f'p={p_value:.3g}', ha='center', va='bottom', color='black', fontsize=10)

    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f'{filename}.{save_format}')
    save_figure(plt.gcf(), output_file)
    plt.close()

    data_file = os.path.join(output_dir, f'{filename}_data.csv')
    stats_file = os.path.join(output_dir, f'{filename}_stats.csv')
    day_data.to_csv(data_file, index=False)
    pd.DataFrame(stats).to_csv(stats_file, index=False)

    print(f"{filename} saved to: {output_file}")
    return day_data, pd.DataFrame(stats)


# ============================================================================
# Figure 1C (right): session-level d', R+ vs R-, days 0/+1/+2
# ============================================================================


def panel_1c_right_dprime(
    table_path=os.path.join(paths.processed_dir, 'behavior', 'behavior_imagingmice_table_5days_cut.csv'),
    days_of_interest=[0, 1, 2],
    save_path=OUTPUT_DIR,
    save_format='svg',
    dpi=300,
):
    """Figure 1 Panel c (right) revision: session-level d' bar plot,
    R+ vs R-, days 0/+1/+2. Mirrors figure_1c.py's
    panel_c_right_performance_barplot, with dprime_w (whisker vs. catch)
    substituted for outcome_w (whisker hit rate)."""

    table_path = paths.adjust_path_to_host(table_path)
    table = pd.read_csv(table_path)

    table = table.loc[table.day.isin(days_of_interest)].copy()
    table['day'] = table['day'].astype(str)

    session_dprime = _session_dprime(table, ['mouse_id', 'session_id', 'reward_group', 'day'])
    avg_performance = session_dprime.groupby(['day', 'mouse_id', 'reward_group'], as_index=False)[
        'dprime_w'
    ].mean()

    _dprime_barplot_by_group(
        avg_performance,
        day_col='day',
        days_of_interest=[str(d) for d in days_of_interest],
        day_labels=[str(d) for d in days_of_interest],
        group_col='reward_group',
        group_order=['R+', 'R-'],
        bar_palette=behavior_palette[2:4][::-1],
        filename='figure_1c_right_dprime',
        output_dir=save_path,
        save_format=save_format,
        dpi=dpi,
        xlabel='Day',
        legend_title='Reward group',
        swarm_color='grey',
    )


# ============================================================================
# Figure 1D: trial-resolved d' across Day-0 whisker trials, R+ vs R-
# ============================================================================


def panel_1d_dprime(
    table_path=os.path.join(
        paths.processed_dir, 'behavior', 'behavior_imagingmice_table_5days_cut_with_learning_curves.csv'
    ),
    n_trials=120,
    save_path=OUTPUT_DIR,
    save_format='svg',
    dpi=300,
):
    """Figure 1 Panel d revision: trial-resolved d' across Day-0 whisker
    trials, R+ vs R-. Single panel -- see module docstring for why the
    original's raw-outcome subplot has no per-trial d' counterpart."""

    df = _load_trial_resolved_dprime(table_path=table_path, n_trials=n_trials)

    sns.set_theme(context='paper', style='ticks', palette='deep', font='sans-serif', font_scale=1)
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))

    sns.lineplot(
        data=df,
        x='trial_w',
        y='dprime_w',
        palette=reward_palette[::-1],
        hue='reward_group',
        errorbar='ci',
        err_style='band',
        ax=ax,
    )

    # Same p-value color code as figure_1d.py: p > 0.05 (not significant)
    # is pure white (via cmap.set_over(), not part of the gradient); p <=
    # 0.05 uses a log10 gradient from a visible grey right at the p=0.05
    # boundary down to black as p keeps shrinking. PVALUE_FLOOR is where
    # the gradient bottoms out at pure black -- tuned to 1e-3 here (vs.
    # 1e-6 in figure_1d.py) to match the smaller dynamic range of p-values
    # this panel's d' test actually produces.
    PVALUE_FLOOR = 1e-3
    NONSIG_GREY = 0.8  # grey at p=0.05
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        'pval_cmap', [(0.0, 'black'), (1.0, (NONSIG_GREY,) * 3)]
    )
    cmap.set_over('white')  # p > vmax (0.05) -- not part of the gradient
    cnorm = matplotlib.colors.LogNorm(vmin=PVALUE_FLOOR, vmax=0.05)

    p_values = []
    for trial_w in df['trial_w'].unique():
        group_R_plus = df[(df.trial_w == trial_w) & (df.reward_group == 'R+')]['dprime_w']
        group_R_minus = df[(df.trial_w == trial_w) & (df.reward_group == 'R-')]['dprime_w']
        if len(group_R_plus) > 0 and len(group_R_minus) > 0:
            _, p_value = mannwhitneyu(group_R_plus, group_R_minus, alternative='two-sided')
            p_values.append((trial_w, p_value))

    trials, raw_pvals = zip(*p_values)
    _, corrected_pvals, _, _ = multipletests(raw_pvals, alpha=0.05, method='fdr_bh')
    p_values = list(zip(trials, corrected_pvals))

    rect_height = 0.1
    rect_y = 3 - rect_height - 0.05  # just below the ylim=3 ceiling
    for trial, p_value in p_values:
        color = cmap(cnorm(max(p_value, PVALUE_FLOOR)))
        ax.add_patch(plt.Rectangle((trial - 0.4, rect_y), 0.8, rect_height, color=color, edgecolor='none'))

    ax.axhline(0, color='black', linestyle='--', alpha=0.4, linewidth=1)
    ax.set_title("d' (whisker vs. catch)")
    ax.set_xlabel('Whisker trial')
    ax.set_ylabel("d'")
    ax.legend(frameon=False, title='Reward group')
    ax.set_ylim(0, 3)

    # Colorbar for the p-value color code (same cmap/cnorm as the rectangles
    # above; p > 0.05 renders white via cmap.set_over(), not depicted on the
    # bar -- no extend cap). Reversed so the most significant (blackest)
    # end is on top and the p=0.05 boundary is at the bottom.
    sm = plt.cm.ScalarMappable(norm=cnorm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation='vertical', shrink=0.6, pad=0.02, aspect=15)
    tick_vals = [PVALUE_FLOOR, 1e-2, 0.05]
    cbar.set_ticks(tick_vals)
    cbar.set_ticklabels(['≤ {:.0e}'.format(PVALUE_FLOOR), '0.01', '0.05'])
    cbar.set_label('p-value (FDR-corrected)', fontsize=9)
    cbar.ax.invert_yaxis()

    sns.despine()
    plt.tight_layout()

    os.makedirs(save_path, exist_ok=True)
    output_file = os.path.join(save_path, f'figure_1d_dprime.{save_format}')
    save_figure(plt.gcf(), output_file)
    plt.close()

    data_file = os.path.join(save_path, 'figure_1d_dprime_data.csv')
    stats_file = os.path.join(save_path, 'figure_1d_dprime_stats.csv')
    df.to_csv(data_file, index=False)
    pd.DataFrame(p_values, columns=['trial_w', 'p_value']).to_csv(stats_file, index=False)

    print(f"Figure 1d (d') saved to: {output_file}")


# ============================================================================
# Figure 2C: session-level d', wS1 vs fpS1 (muscimol, learning)
# ============================================================================


def panel_2c_dprime(
    table_path=os.path.join(paths.processed_dir, 'behavior', 'behavior_muscimol.csv'),
    days_of_interest=['muscimol_1', 'muscimol_2', 'muscimol_3'],
    day_labels=['D0', 'D+1', 'D+2'],
    save_path=OUTPUT_DIR,
    save_format='svg',
    dpi=300,
):
    """Figure 2 Panel c revision: session-level d' bar plot, wS1 vs fpS1
    (muscimol inactivation during learning). Mirrors
    figure_2b_c.py:panel_c_muscimol_barplot, with dprime_w substituted for
    outcome_w."""

    table_path = paths.adjust_path_to_host(table_path)
    table = pd.read_csv(table_path)

    db_path = paths.db_path
    nwb_dir = paths.nwb_dir

    fpS1_mice = database.select_mice_from_db(
        db_path,
        nwb_dir,
        experimenters=None,
        exclude_cols=['exclude'],
        pharmacology='yes',
        pharma_inactivation_type='learning',
        pharma_area='fpS1',
    )
    wS1_mice = database.select_mice_from_db(
        db_path,
        nwb_dir,
        experimenters=None,
        exclude_cols=['exclude'],
        pharmacology='yes',
        pharma_inactivation_type='learning',
        pharma_area='wS1',
    )
    table.loc[table.mouse_id.isin(fpS1_mice), 'area'] = 'fpS1'
    table.loc[table.mouse_id.isin(wS1_mice), 'area'] = 'wS1'

    _, _, _, db = database.select_sessions_from_db(
        db_path,
        nwb_dir,
        experimenters=None,
        exclude_cols=['exclude'],
        pharma_inactivation_type=['learning'],
        pharma_day=[
            "pre_-2",
            "pre_-1",
            "muscimol_1",
            "muscimol_2",
            "muscimol_3",
            "recovery_1",
            "recovery_2",
            "recovery_3",
        ],
    )
    table = pd.merge(
        table, db[['mouse_id', 'session_id', 'pharma_day']], on=['mouse_id', 'session_id'], how='left'
    )

    data = _session_dprime(table, ['mouse_id', 'session_id', 'pharma_day', 'area'])

    _dprime_barplot_by_group(
        data,
        day_col='pharma_day',
        days_of_interest=days_of_interest,
        day_labels=day_labels,
        group_col='area',
        group_order=['wS1', 'fpS1'],
        bar_palette=[reward_palette[1]],
        filename='figure_2c_dprime',
        output_dir=save_path,
        save_format=save_format,
        dpi=dpi,
        xlabel='Day',
        legend_title='Area',
        swarm_color=stim_palette[2],
    )


# ============================================================================
# Figure 2F: session-level d', wS1 vs fpS1 (optogenetics, learning)
# ============================================================================


def panel_2f_dprime(
    table_path=os.path.join(paths.processed_dir, 'behavior', 'behavior_opto_learning.csv'),
    days_of_interest=['opto', 'recovery_1'],
    day_labels=['D0', 'D+1'],
    save_path=OUTPUT_DIR,
    save_format='svg',
    dpi=300,
):
    """Figure 2 Panel f revision: session-level d' bar plot, wS1 vs fpS1
    (optogenetic inactivation during learning). Mirrors
    figure_2e_f.py:panel_f_opto_barplot, with dprime_w substituted for
    outcome_w."""

    table_path = paths.adjust_path_to_host(table_path)
    table = pd.read_csv(table_path)

    db_path = paths.db_path
    nwb_dir = paths.nwb_dir

    fpS1_mice = database.select_mice_from_db(
        db_path,
        nwb_dir,
        experimenters=None,
        exclude_cols=['exclude', 'opto_exclude'],
        optogenetic='yes',
        opto_inactivation_type='learning',
        opto_area='fpS1',
    )
    wS1_mice = database.select_mice_from_db(
        db_path,
        nwb_dir,
        experimenters=None,
        exclude_cols=['exclude', 'opto_exclude'],
        optogenetic='yes',
        opto_inactivation_type='learning',
        opto_area='wS1',
    )
    table.loc[table.mouse_id.isin(fpS1_mice), 'area'] = 'fpS1'
    table.loc[table.mouse_id.isin(wS1_mice), 'area'] = 'wS1'

    _, _, _, db = database.select_sessions_from_db(
        db_path,
        nwb_dir,
        experimenters=None,
        exclude_cols=['exclude', 'opto_exclude'],
        opto_inactivation_type=['learning'],
        opto_day=["pre_-2", "pre_-1", "opto", "recovery_1"],
    )
    table = pd.merge(
        table, db[['mouse_id', 'session_id', 'opto_day']], on=['mouse_id', 'session_id'], how='left'
    )

    data = _session_dprime(table, ['mouse_id', 'session_id', 'opto_day', 'area'])

    _dprime_barplot_by_group(
        data,
        day_col='opto_day',
        days_of_interest=days_of_interest,
        day_labels=day_labels,
        group_col='area',
        group_order=['wS1', 'fpS1'],
        bar_palette=[reward_palette[1]],
        filename='figure_2f_dprime',
        output_dir=save_path,
        save_format=save_format,
        dpi=dpi,
        xlabel='Day',
        legend_title='Area',
        swarm_color='black',
    )


# ============================================================================
# Execution panel c: session-level d', wS1 vs fpS1 (muscimol, execution)
# ============================================================================


def panel_2c_execution_dprime(
    table_path=EXECUTION_TABLE_PATH,
    days_of_interest=EXECUTION_DAYS_OF_INTEREST,
    day_labels=EXECUTION_DAY_LABELS,
    save_path=OUTPUT_DIR,
    save_format='svg',
    dpi=300,
):
    """Execution-inactivation Panel c revision: session-level d' bar plot,
    wS1 vs fpS1 (muscimol inactivation during execution). Mirrors
    figure_2b_c_execution.py:panel_c_muscimol_barplot_execution, with
    dprime_w substituted for outcome_w. Requires
    behavior_muscimol_execution.csv (03_behavior_tables.py
    --tables muscimol_execution)."""

    table_path = paths.adjust_path_to_host(table_path)
    table = pd.read_csv(table_path)

    db_path = paths.db_path
    nwb_dir = paths.nwb_dir

    fpS1_mice = database.select_mice_from_db(
        db_path,
        nwb_dir,
        experimenters=None,
        exclude_cols=['exclude'],
        pharmacology='yes',
        pharma_inactivation_type='execution',
        pharma_area='fpS1',
    )
    wS1_mice = database.select_mice_from_db(
        db_path,
        nwb_dir,
        experimenters=None,
        exclude_cols=['exclude'],
        pharmacology='yes',
        pharma_inactivation_type='execution',
        pharma_area='wS1',
    )
    table.loc[table.mouse_id.isin(fpS1_mice), 'area'] = 'fpS1'
    table.loc[table.mouse_id.isin(wS1_mice), 'area'] = 'wS1'

    _, _, _, db = database.select_sessions_from_db(
        db_path,
        nwb_dir,
        experimenters=None,
        exclude_cols=['exclude'],
        pharma_inactivation_type=['execution'],
        pharma_day=["pre_-2", "pre_-1", "muscimol_1", "ringer_1", "muscimol_2"],
    )
    table = pd.merge(
        table, db[['mouse_id', 'session_id', 'pharma_day']], on=['mouse_id', 'session_id'], how='left'
    )

    data = _session_dprime(table, ['mouse_id', 'session_id', 'pharma_day', 'area'])

    _dprime_barplot_by_group(
        data,
        day_col='pharma_day',
        days_of_interest=days_of_interest,
        day_labels=day_labels,
        group_col='area',
        group_order=['wS1', 'fpS1'],
        bar_palette=[reward_palette[1]],
        filename='figure_2c_execution_dprime',
        output_dir=save_path,
        save_format=save_format,
        dpi=dpi,
        xlabel='Day',
        legend_title='Area',
        swarm_color=stim_palette[2],
    )


# ============================================================================
# Figure 4C (behavior row only): trial-resolved d', R+ vs R-
# ============================================================================


def panel_4c_dprime(
    table_path=os.path.join(
        paths.processed_dir, 'behavior', 'behavior_imagingmice_table_5days_cut_with_learning_curves.csv'
    ),
    cut_n_trials=100,
    save_path=OUTPUT_DIR,
    save_format='svg',
    dpi=300,
):
    """Figure 4 Panel c revision (behavior row only): trial-resolved d'
    across Day-0 whisker trials, R+ vs R- (in place of learning_curve_w).
    The decoder row of the original figure is unchanged/not reproduced --
    it isn't a hit-rate-based quantity."""

    df = _load_trial_resolved_dprime(table_path=table_path, n_trials=cut_n_trials)

    data_rew = df.loc[df.reward_group == 'R+']
    data_nonrew = df.loc[df.reward_group == 'R-']

    sns.set_theme(context='paper', style='ticks', palette='deep', font='sans-serif', font_scale=1)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)

    for ax, data, color, title in [
        (axes[0], data_rew, reward_palette[1], 'R+ mice'),
        (axes[1], data_nonrew, reward_palette[0], 'R- mice'),
    ]:
        sns.lineplot(data=data, x='trial_w', y='dprime_w', color=color, errorbar='ci', ax=ax)
        ax.axhline(0, color='black', linestyle='--', alpha=0.4)
        ax.set_xlabel('Trial within Day 0')
        ax.set_ylabel("d' (whisker vs. catch)")
        ax.set_title(title)
        ax.set_xlim(0, cut_n_trials)

    sns.despine()
    plt.tight_layout()

    os.makedirs(save_path, exist_ok=True)
    out_path = os.path.join(save_path, f'figure_4c_dprime.{save_format}')
    save_figure(fig, out_path)
    plt.close()

    df.to_csv(os.path.join(save_path, 'figure_4c_dprime_data.csv'), index=False)

    print(f"Figure 4c (d', behavior row) saved to: {out_path}")


# ============================================================================
# Figure 4E: per-mouse correlation between decoder value and d'
# ============================================================================


def panel_4e_dprime_correlation(
    table_path=os.path.join(
        paths.processed_dir, 'behavior', 'behavior_imagingmice_table_5days_cut_with_learning_curves.csv'
    ),
    cut_n_trials=100,
    weights_path=os.path.join(paths.processed_dir, 'decoding', 'decoder_weights.pkl'),
    window_size=10,
    step_size=1,
    save_path=OUTPUT_DIR,
    save_format='svg',
    dpi=300,
):
    """Figure 4 Panel e revision: per-mouse Pearson correlation between the
    neural decoder's decision value and trial-resolved d' (in place of
    learning_curve_w).

    Uses the same pre-trained decoder cached in decoder_weights.pkl (built
    by pipeline/06_decoder.py) that figure_4b.py/figure_4c.py already reuse,
    applied via the identical 10-trial sliding window -- rather than
    figure_4e.py's original approach of retraining a fresh per-mouse
    decoder inline. This keeps the decoder consistent with the rest of
    Figure 4 (per user decision) and avoids re-running classifier
    training; only the sliding-window application to Day-0 trials is
    (unavoidably) redone here, since no script caches per-trial decision
    values to disk. Still needs NWB/network-drive access to load the
    Day-0 imaging tensors."""

    win = (0, 0.300)

    df = _load_trial_resolved_dprime(table_path=table_path, n_trials=cut_n_trials)

    with open(weights_path, 'rb') as f:
        weights = pickle.load(f)
    print(f"Loaded decoder weights for {len(weights)} mice.")

    folder = paths.tensor_dir
    results = []
    for mouse, w in weights.items():
        xarr = imaging.load_mouse_xarray(mouse, folder, 'tensor_xarray_learning_data.nc')
        xarr = xarr.sel(trial=xarr['day'].isin([0]))
        xarr = xarr.sel(trial=xarr['whisker_stim'] == 1)
        xarr = xarr.sel(time=slice(win[0], win[1])).mean(dim='time')
        xarr = xarr.fillna(0)

        scaler, clf, sign_flip = w['scaler'], w['clf'], w['sign_flip']
        n_trials = xarr.sizes['trial']
        for start_idx in range(0, max(0, n_trials - window_size + 1), step_size):
            end_idx = start_idx + window_size
            X_win = xarr.values[:, start_idx:end_idx].T
            if X_win.shape[0] == 0:
                continue
            dec_vals = clf.decision_function(scaler.transform(X_win))
            results.append(
                {
                    'mouse_id': mouse,
                    'reward_group': w['reward_group'],
                    'trial_start': start_idx,
                    'mean_decision_value': np.mean(dec_vals) * sign_flip,
                }
            )
    results_combined = pd.DataFrame(results)

    corr_real, corr_mice, corr_groups = [], [], []
    for mouse in results_combined['mouse_id'].unique():
        group = results_combined.loc[results_combined['mouse_id'] == mouse, 'reward_group'].iloc[0]
        dec_mouse = results_combined[results_combined['mouse_id'] == mouse]
        bh_mouse = df[df['mouse_id'] == mouse]

        common_trials = np.intersect1d(dec_mouse['trial_start'], bh_mouse['trial_w'])
        if len(common_trials) < 10:
            continue

        dec_vals = dec_mouse.set_index('trial_start').loc[common_trials]['mean_decision_value'].values
        perf_vals = bh_mouse.set_index('trial_w').loc[common_trials]['dprime_w'].values

        corr = pearsonr(perf_vals, dec_vals)[0]
        corr_real.append(corr)
        corr_mice.append(mouse)
        corr_groups.append(group)
        print(f"{mouse} ({group}): r={corr:.3f}")

    df_corr = pd.DataFrame({'mouse_id': corr_mice, 'reward_group': corr_groups, 'correlation': corr_real})

    pop_stats, pop_stats_rows = {}, []
    for group in ['R+', 'R-']:
        sub = df_corr[df_corr['reward_group'] == group]
        if len(sub) >= 3:
            _, p_wilcox = wilcoxon(sub['correlation'].values, alternative='greater')
            _, p_ttest = ttest_1samp(sub['correlation'].values, 0, alternative='greater')
            pop_stats[group] = p_wilcox
            pop_stats_rows.append(
                {
                    'reward_group': group,
                    'n': len(sub),
                    'mean_correlation': np.mean(sub['correlation'].values),
                    'std_correlation': np.std(sub['correlation'].values),
                    'p_wilcoxon': p_wilcox,
                    'p_ttest': p_ttest,
                }
            )
            print(
                f"{group} (N={len(sub)}): mean r={np.mean(sub['correlation'].values):.3f}, "
                f"Wilcoxon p={p_wilcox:.4f}"
            )
    df_pop_stats = pd.DataFrame(pop_stats_rows)

    sns.set_theme(context='paper', style='ticks', palette='deep', font='sans-serif', font_scale=1)
    fig, ax = plt.subplots(1, 1, figsize=(4, 5))

    sns.swarmplot(
        data=df_corr,
        x='reward_group',
        y='correlation',
        palette=reward_palette[::-1],
        size=8,
        alpha=0.6,
        ax=ax,
    )
    sns.pointplot(
        data=df_corr,
        x='reward_group',
        y='correlation',
        palette=reward_palette[::-1],
        errorbar='ci',
        markersize=10,
        join=False,
        ax=ax,
    )
    ax.axhline(0, color='black', linestyle='--', alpha=0.5, linewidth=1)

    for i, group in enumerate(['R+', 'R-']):
        if group in pop_stats:
            p = pop_stats[group]
            p_text = f'p={p:.4f}' if p >= 0.001 else 'p<0.001'
            sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'n.s.'))
            y_pos = ax.get_ylim()[1] - 0.1 * (ax.get_ylim()[1] - ax.get_ylim()[0])
            ax.text(i, y_pos, f'{p_text}\n{sig}', ha='center', va='top', fontsize=9, fontweight='bold')

    ax.set_ylim(-1, 1)
    ax.set_xlabel('Reward group')
    ax.set_ylabel("Pearson r\n(Decision value vs d')")
    sns.despine()
    plt.tight_layout()

    os.makedirs(save_path, exist_ok=True)
    save_figure(fig, os.path.join(save_path, f'figure_4e_dprime.{save_format}'))
    plt.close()

    df_corr.to_csv(os.path.join(save_path, 'figure_4e_dprime_correlations.csv'), index=False)
    df_pop_stats.to_csv(os.path.join(save_path, 'figure_4e_dprime_stats.csv'), index=False)

    print(f"Figure 4e (d' correlation) saved to: {save_path}")


# ============================================================================
# Main execution
# ============================================================================

if __name__ == '__main__':
    panel_1c_right_dprime()
    panel_1d_dprime()
    panel_2c_dprime()
    panel_2f_dprime()
    panel_2c_execution_dprime()
    panel_4c_dprime()
    panel_4e_dprime_correlation()
