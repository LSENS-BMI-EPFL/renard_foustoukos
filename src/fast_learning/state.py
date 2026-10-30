"""Behavioural state of the imaging mice across days (Figure 3—figure
supplement 2): session measures from the trial tables, and whisker and
orofacial movements during the passive trials (DeepLabCut).

Session measures, per session (task trials up to the stop flag):
- number of trials;
- duration of the task (first to last trial start), in minutes;
- water consumed: 5 uL per rewarded trial (auditory hits in both groups,
  whisker hits in R+ mice).

Movements: DeepLabCut export of G. Foustoukos, one '<mouse>_dlc.h5' file per
mouse in DLC_DIR (11 R+ mice and one R- mouse, so R+ mice only). Each file
holds, for the whisker trials after disengagement with usable video (no lick),
three speeds at 100 Hz, stimulus at 0 s: whisker angular speed (|d theta/dt|
of the base-to-tip whisker angle, deg/s), tongue and jaw speed (pixels/s).
Per mouse and day, the last N_TRIALS trials (in session order) are averaged,
as the passive-response analyses, and the speed is averaged over WINDOW.
"""

import os

import h5py
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, mannwhitneyu, wilcoxon

from fast_learning import database, paths

DAYS = [-2, -1, 0, 1, 2]
PRE_DAYS, POST_DAYS = [-2, -1], [1, 2]
REWARD_UL_PER_TRIAL = 5
SESSION_MEASURES = {
    'n_trials': 'Number of trials',
    'session_duration_min': 'Session duration (min)',
    'total_reward_uL': 'Total water reward (µL)',
}
DLC_DIR = os.path.join(paths.dlc_dir, 'Speeds', 'DLCTrialMatrices')
WINDOW = (0.0, 1.0)  # s after stimulus onset
N_TRIALS = 40
MIN_TRIALS = 5
SPEEDS = {
    'whisker_angular_speed': 'Whisker angular speed',
    'tongue_speed': 'Tongue speed',
    'jaw_speed': 'Jaw speed',
}


# ============================================================================
# Session measures
# ============================================================================


def load_imaging_trials():
    """Task trials of the imaging mice on Days -2 to +2, cut at the stop flags
    (read from the NWB files)."""
    from fast_learning.behavior import make_behavior_table  # heavy imports, only here

    mice = database.select_mice_from_db(
        paths.db_path,
        paths.nwb_dir,
        experimenters=None,
        exclude_cols=['exclude', 'two_p_exclude'],
        optogenetic=['no', np.nan],
        pharmacology=['no', np.nan],
        two_p_imaging='yes',
    )
    sessions, nwbs, _, _ = database.select_sessions_from_db(
        paths.db_path,
        paths.nwb_dir,
        experimenters=None,
        exclude_cols=['exclude', 'two_p_exclude'],
        day=['-2', '-1', '0', '+1', '+2'],
        mouse_id=mice,
    )
    return make_behavior_table(
        nwbs,
        sessions,
        paths.db_path,
        cut_session=True,
        stop_flag_yaml=paths.stop_flags_yaml,
        trial_indices_yaml=paths.trial_indices_yaml,
    )


def session_summary(trials):
    """One row per session: number of trials, task duration, water."""
    rows = []
    for (mouse, session, day, rg), g in trials.groupby(['mouse_id', 'session_id', 'day', 'reward_group']):
        rewarded = (g['auditory_stim'] == 1) & (g['lick_flag'] == 1)
        if rg == 'R+':
            rewarded |= (g['whisker_stim'] == 1) & (g['lick_flag'] == 1)
        rows.append(
            dict(
                mouse_id=mouse,
                session_id=session,
                day=day,
                reward_group=rg,
                n_trials=len(g),
                session_duration_min=(g['start_time'].max() - g['start_time'].min()) / 60,
                total_reward_uL=REWARD_UL_PER_TRIAL * int(rewarded.sum()),
            )
        )
    return pd.DataFrame(rows)


def session_stats(summary):
    """R+ vs R- on each day (Mann-Whitney U), and pre vs post learning within
    each group (each mouse's mean over Days -2/-1 and +1/+2, Wilcoxon)."""
    rows = []
    for measure in SESSION_MEASURES:
        for day in DAYS:
            d = summary[summary['day'] == day]
            a, b = (d.loc[d['reward_group'] == g, measure] for g in ('R+', 'R-'))
            res = mannwhitneyu(a, b, alternative='two-sided')
            rows.append(
                dict(
                    measure=measure,
                    comparison=f'R+ vs R-, day {day}',
                    test='Mann-Whitney U (n = mice)',
                    n=f'{len(a)} / {len(b)}',
                    mean_1=a.mean(),
                    mean_2=b.mean(),
                    statistic=res.statistic,
                    p_value=res.pvalue,
                )
            )
        period = summary['day'].map(dict.fromkeys(PRE_DAYS, 'pre') | dict.fromkeys(POST_DAYS, 'post'))
        pp = summary.assign(period=period).dropna(subset=['period'])
        pp = pp.groupby(['reward_group', 'mouse_id', 'period'])[measure].mean().unstack().dropna()
        for rg in ('R+', 'R-'):
            g = pp.loc[rg]
            res = wilcoxon(g['pre'], g['post'])
            rows.append(
                dict(
                    measure=measure,
                    comparison=f'{rg}: pre (days -2, -1) vs post (days +1, +2)',
                    test='Wilcoxon signed-rank (n = mice)',
                    n=str(len(g)),
                    mean_1=g['pre'].mean(),
                    mean_2=g['post'].mean(),
                    statistic=res.statistic,
                    p_value=res.pvalue,
                )
            )
    return pd.DataFrame(rows)


# ============================================================================
# Movements
# ============================================================================


def _text(values):
    return [v.decode() if isinstance(v, bytes) else str(v) for v in values]


def load_mouse_speeds(path):
    """({day: (n_trials, n_speeds, n_time)} for the last N_TRIALS trials,
    speed names, units, time) of one mouse."""
    with h5py.File(path, 'r') as h:
        traces = h['traces'][...]
        names = _text(h['feature_names'][...])
        units = _text(h['feature_units'][...])
        time = h['time'][...]
        days = h['trials/day_offset'][...]
        order = h['trials/trial_number'][...]
    # Edge artefact of the speed traces: first and last samples take their
    # neighbour's value.
    traces[:, :, 0] = traces[:, :, 1]
    traces[:, :, -1] = traces[:, :, -2]
    by_day = {}
    for d in DAYS:
        idx = np.flatnonzero(days == d)
        idx = idx[np.argsort(order[idx], kind='stable')][-N_TRIALS:]
        if len(idx) >= MIN_TRIALS:
            by_day[d] = traces[idx]
    return by_day, names, units, time


def movement_means():
    """Per R+ mouse x day x speed: mean speed in WINDOW. Returns (DataFrame,
    {speed: unit})."""
    db = database.read_excel_db(paths.db_path)
    rows, units = [], {}
    for f in sorted(f for f in os.listdir(DLC_DIR) if f.endswith('_dlc.h5')):
        mouse = f[: -len('_dlc.h5')]
        if database.get_mouse_reward_group_from_db(paths.db_path, mouse, db=db) != 'R+':
            continue
        by_day, names, unit_list, time = load_mouse_speeds(os.path.join(DLC_DIR, f))
        units.update(zip(names, unit_list))
        window = (time >= WINDOW[0]) & (time < WINDOW[1])
        for day, tr in by_day.items():
            mean_trace = tr.mean(axis=0)
            for i, name in enumerate(names):
                rows.append(
                    dict(
                        mouse_id=mouse,
                        day=day,
                        speed=name,
                        n_trials=len(tr),
                        value=mean_trace[i, window].mean(),
                    )
                )
    return pd.DataFrame(rows), units


def movement_stats(means):
    """Friedman test across the five days (mice with all days), per speed."""
    rows = []
    for speed in SPEEDS:
        wide = means[means['speed'] == speed].pivot(index='mouse_id', columns='day', values='value')
        wide = wide.reindex(columns=DAYS).dropna()
        stat, p = friedmanchisquare(*(wide[d] for d in DAYS)) if len(wide) >= 3 else (np.nan, np.nan)
        rows.append(
            dict(
                speed=speed,
                window=f'{WINDOW[0]:g}-{WINDOW[1]:g} s',
                test='Friedman across days (n = mice with all days)',
                n_mice=len(wide),
                **{f'mean_day_{d:+d}': wide[d].mean() for d in DAYS},
                statistic=stat,
                p_value=p,
            )
        )
    return pd.DataFrame(rows)
