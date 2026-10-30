"""Inactivation cohorts (Figure 2 and its supplements): session tables, d',
comparison with the naive R+ imaging mice, and shared plots.

A cohort is the trial table of one inactivation experiment, with the
inactivated area of each mouse (wS1, or fpS1 as control) and the day label of
each session from the session metadata. Per session: lick probability (%) on
whisker, auditory and catch trials, d' = Z(whisker hit rate) - Z(catch
false-alarm rate) and auditory d' = Z(auditory hit rate) - Z(catch
false-alarm rate), from trial counts with the log-linear correction
(fast_learning.stats.dprime). n = mice throughout; group comparisons are
Mann-Whitney U tests.
"""

import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import mannwhitneyu

from fast_learning import database, paths
from fast_learning.plotting import reward_palette, stim_palette
from fast_learning.stats import dprime, format_p

BEHAVIOR_DIR = os.path.join(paths.processed_dir, 'behavior')
AREAS = ['wS1', 'fpS1']
AREA_COLORS = {'wS1': reward_palette[1], 'fpS1': '#8c8c8c'}
OUTCOME_COLORS = {'p_whisker': reward_palette[1], 'p_auditory': stim_palette[0], 'p_catch': stim_palette[2]}
MEASURES = {'p_whisker': 'Whisker lick probability (%)', 'dprime': "d' (whisker vs catch)"}
AUDITORY_MEASURES = {
    'p_auditory': 'Auditory lick probability (%)',
    'dprime_auditory': "d' (auditory vs catch)",
}
LABELS = MEASURES | AUDITORY_MEASURES
YLIMS = {'p_whisker': (0, 100), 'dprime': (-1, 4), 'p_auditory': (0, 100), 'dprime_auditory': (0, 6)}


@dataclass(frozen=True)
class Cohort:
    table: str  # trial table in BEHAVIOR_DIR
    prefix: str  # 'pharma' or 'opto' (session metadata columns)
    inactivation_type: str  # 'learning' or 'execution'
    days: tuple  # day labels in the session metadata, in order
    labels: tuple  # tick labels of these days


MUSCIMOL_LEARNING = Cohort(
    'behavior_muscimol.csv',
    'pharma',
    'learning',
    ('pre_-2', 'pre_-1', 'muscimol_1', 'muscimol_2', 'muscimol_3', 'recovery_1', 'recovery_2', 'recovery_3'),
    ('-2', '-1', 'M 1', 'M 2', 'M 3', 'R 1', 'R 2', 'R 3'),
)
OPTO_LEARNING = Cohort(
    'behavior_opto_learning.csv',
    'opto',
    'learning',
    ('pre_-2', 'pre_-1', 'opto', 'recovery_1'),
    ('-2', '-1', '0', '+1'),
)
MUSCIMOL_EXECUTION = Cohort(
    'behavior_muscimol_execution.csv',
    'pharma',
    'execution',
    ('pre_-2', 'pre_-1', 'muscimol_1', 'ringer_1', 'muscimol_2'),
    ('-2', '-1', 'M 1', 'Ringer', 'M 2'),
)


# ============================================================================
# Data
# ============================================================================


def load_trials(cohort):
    """Trial table of a cohort with 'area' and 'inactivation_day' columns
    (sessions outside cohort.days and mice of neither area are dropped)."""
    trials = pd.read_csv(paths.adjust_path_to_host(os.path.join(BEHAVIOR_DIR, cohort.table)))
    prefix = cohort.prefix
    exclude = ['exclude', 'opto_exclude'] if prefix == 'opto' else ['exclude']
    flag = {'opto': 'optogenetic', 'pharma': 'pharmacology'}[prefix]
    for area in AREAS:
        mice = database.select_mice_from_db(
            paths.db_path,
            paths.nwb_dir,
            experimenters=None,
            exclude_cols=exclude,
            **{flag: 'yes', f'{prefix}_inactivation_type': cohort.inactivation_type, f'{prefix}_area': area},
        )
        trials.loc[trials.mouse_id.isin(mice), 'area'] = area
    _, _, _, db = database.select_sessions_from_db(
        paths.db_path,
        paths.nwb_dir,
        experimenters=None,
        exclude_cols=exclude,
        **{f'{prefix}_inactivation_type': [cohort.inactivation_type], f'{prefix}_day': list(cohort.days)},
    )
    db = db[['mouse_id', 'session_id', f'{prefix}_day']].rename(columns={f'{prefix}_day': 'inactivation_day'})
    return trials.merge(db, on=['mouse_id', 'session_id'], how='inner').dropna(subset=['area'])


def counts(trials, by):
    """Trial counts, lick probabilities (%) and d' per group of trials."""
    out = trials.groupby(by, observed=True).agg(
        n_whisker=('outcome_w', 'count'),
        hits=('outcome_w', 'sum'),
        n_auditory=('outcome_a', 'count'),
        auditory_hits=('outcome_a', 'sum'),
        n_catch=('outcome_c', 'count'),
        false_alarms=('outcome_c', 'sum'),
        p_whisker=('outcome_w', 'mean'),
        p_auditory=('outcome_a', 'mean'),
        p_catch=('outcome_c', 'mean'),
    )
    out[['p_whisker', 'p_auditory', 'p_catch']] *= 100
    out['dprime'] = dprime(out['hits'], out['n_whisker'], out['false_alarms'], out['n_catch'])
    out['dprime_auditory'] = dprime(
        out['auditory_hits'], out['n_auditory'], out['false_alarms'], out['n_catch']
    )
    return out.reset_index()


def session_table(cohort, trials=None):
    """One row per session, with 'day_label' ordered as the cohort's days."""
    trials = load_trials(cohort) if trials is None else trials
    sessions = counts(trials, ['mouse_id', 'session_id', 'area', 'inactivation_day'])
    sessions['day_label'] = pd.Categorical(
        sessions['inactivation_day'].map(dict(zip(cohort.days, cohort.labels))),
        categories=list(cohort.labels),
        ordered=True,
    )
    return sessions.sort_values(['mouse_id', 'day_label'])


def naive_day0_trials():
    """Trials of the naive R+ imaging mice on Day 0 (first whisker session)."""
    t = pd.read_csv(
        paths.adjust_path_to_host(os.path.join(BEHAVIOR_DIR, 'behavior_imagingmice_table_5days_cut.csv'))
    )
    return t[(t.day == 0) & (t.reward_group == 'R+')]


# ============================================================================
# Statistics
# ============================================================================


def mwu_row(label, measure, a, b, names):
    a, b = np.asarray(a, float), np.asarray(b, float)
    res = mannwhitneyu(a, b, alternative='two-sided')
    return {
        'measure': measure,
        'day': label,
        'comparison': f'{names[0]} vs {names[1]}',
        'n_1': len(a),
        'n_2': len(b),
        'median_1': np.median(a),
        'median_2': np.median(b),
        'test': 'Mann-Whitney U',
        'statistic': res.statistic,
        'p_value': res.pvalue,
    }


def area_stats(sessions, measure, labels):
    """wS1 vs fpS1 on each day label."""
    rows = []
    for label in labels:
        d = sessions[sessions.day_label == label]
        rows.append(mwu_row(label, measure, *(d.loc[d.area == a, measure] for a in AREAS), AREAS))
    return pd.DataFrame(rows)


# ============================================================================
# Plots
# ============================================================================


def bars(ax, data, x, order, hue, hue_order, colors, y, pvalues, legend=False):
    """Bars (mean, bootstrapped 95% CI) and dots (mice), with the p-value of
    each x category above it. Bars are split by hue within each x category
    unless hue is x."""
    kw = dict(data=data, x=x, y=y, order=order, hue=hue, hue_order=hue_order, ax=ax)
    sns.barplot(
        **kw, palette=colors, errorbar=('ci', 95), seed=0, width=0.6, alpha=0.7, legend=legend, dodge=hue != x
    )
    if hue == x:
        sns.swarmplot(data=data, x=x, y=y, order=order, color='black', size=2, ax=ax)
    else:
        sns.swarmplot(**kw, palette=dict.fromkeys(hue_order, 'black'), dodge=True, size=2, legend=False)
    for xi, p in enumerate(pvalues):
        ax.text(
            xi, 1.0, format_p(p), transform=ax.get_xaxis_transform(), ha='center', va='bottom', fontsize=5
        )
    if y == 'dprime':
        ax.axhline(0, color='grey', linestyle='--', linewidth=0.6)
    ax.set_ylim(*YLIMS[y])
    ax.set_xlabel('')
    ax.set_ylabel(LABELS[y])
    if legend:
        ax.legend(frameon=False, title='', loc='upper left', bbox_to_anchor=(1.0, 1.0))


def area_bars(ax, sessions, stats, labels, y, title='', legend=False):
    """y per day label, wS1 vs fpS1, with the p-value of each day."""
    d = sessions[sessions.day_label.isin(labels)]
    pvalues = [stats.query('measure == @y and day == @label')['p_value'].iloc[0] for label in labels]
    bars(ax, d, 'day_label', list(labels), 'area', AREAS, AREA_COLORS, y, pvalues, legend)
    ax.set_title(title, pad=8)


def timecourse(ax, sessions, area, labels, xlabel=''):
    """Lick probability on whisker, auditory and catch trials across days for
    the mice of one area: lines = mice, dots = mean and bootstrapped 95% CI."""
    d = sessions[sessions.area == area].assign(x=lambda t: t.day_label.cat.codes)
    for col, color in OUTCOME_COLORS.items():
        for _, m in d.groupby('mouse_id'):
            m = m.dropna(subset=[col])
            # A mouse with a single day for this trial type (e.g. whisker trials
            # on Day 0 only) has no line to draw: show it as a dot.
            marker = 'o' if len(m) == 1 else None
            ax.plot(m['x'], m[col], color=color, alpha=0.4, linewidth=0.5, marker=marker, markersize=2)
        sns.pointplot(
            data=d,
            x='x',
            y=col,
            native_scale=True,
            color=color,
            errorbar=('ci', 95),
            seed=0,
            markersize=3,
            linewidth=1,
            ax=ax,
        )
    ax.set_xticks(range(len(labels)), labels)
    ax.set_ylim(0, 100)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Lick probability (%)')
    ax.set_title(f'{area}, n = {d.mouse_id.nunique()}')


def save(fig, path):
    """Despine, tighten and save a figure (formats from config.yaml)."""
    from fast_learning.plotting import save_figure

    sns.despine(fig=fig)
    fig.tight_layout()
    save_figure(fig, path)
    plt.close(fig)
