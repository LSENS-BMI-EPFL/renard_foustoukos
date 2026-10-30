"""Response strength, trial-to-trial noise and day-to-day drift of passive
whisker responses (Figure 3—figure supplements 3 and 4).

Response vectors: all cells, last N_MAP_TRIALS passive trials per day, mean
over WIN (as Fig. 3h-i). Days -2 and -1 are the only consecutive days without
learning, so -2/-1 measures drift; Day-0 passive responses are recorded after
the learning session, so -1/0 spans learning. n = mice throughout.

- Signal power: squared norm of the trial-averaged response vector, corrected
  for noise (sum of squared means minus noise / n trials). Noise power: summed
  trial-to-trial variance over cells.
- Day-to-day pattern drift: correlation of the mean response vectors of two
  consecutive days, divided by the geometric mean of their split-half
  reliabilities (Spearman-Brown corrected), so that 1 means the same pattern
  on both days.
- Adaptation within the passive block: population response of each trial
  (mean over cells), in bins of BIN trials along the block and in the first
  and the last N_HALF trials; first vs last (each mouse averaged over days),
  and pre vs post learning computed from the first or the last N_HALF trials
  only.
- Stability within the passive block: in each day, correlation of the mean
  response vectors of the first and the last N_HALF of the N_MAP_TRIALS
  trials (block order), compared with that of the odd and the even trials
  (same number of trials, no separation in time), and noise-corrected as the
  day-to-day drift (reliability of each half), to be compared with it. Each
  mouse is summarised by its median over the five days (with 20-trial halves
  the reliability is noisy in a few low-signal mouse-days).
- LMI between two consecutive days: as pipeline step 05 (ROC of each cell's
  response, 0-300 ms after a -1-0 s baseline, last 40 passive trials, 1000
  label shuffles; LMI+ at p >= 0.975, LMI- at p <= 0.025), one day on each
  side. Fractions of LMI+ and LMI- cells per mouse and pair, and the net
  fraction (LMI+ minus LMI-).
"""

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, pearsonr, rankdata, wilcoxon

from fast_learning import database, imaging, paths
from fast_learning.participation import LMI_NEGATIVE_THRESHOLD, LMI_POSITIVE_THRESHOLD
from fast_learning.plotting import reward_palette
from fast_learning.similarity import (
    DAYS,
    N_MAP_TRIALS,
    WIN,
    compute_similarity_matrix,
    compute_within_day_metrics,
)
from fast_learning.stats import significance_stars

PRE_DAYS = [-2, -1]
POST_DAYS = [1, 2]
BASELINE_PAIR = (-2, -1)  # the only consecutive days without learning
LEARNING_PAIR = (-1, 0)  # spans the first whisker learning session
DAY_PAIRS = [BASELINE_PAIR, LEARNING_PAIR]
N_SPLITS = 100
N_HALF = N_MAP_TRIALS // 2
BIN = 10  # trials per bin along the passive block
WITHIN = 'within'  # pair label of the first/last comparison within a day
RESPONSE_WIN = (0, 0.300)  # LMI, as pipeline step 05
BASELINE_WIN = (-1, 0)
N_SHUFFLES = 1000
SEED = 0
REFERENCE = 'LMI (-2,-1/+1,+2)'
FRACTIONS = {
    'positive': 'LMI+ cells (%)',
    'negative': 'LMI− cells (%)',
    'net': 'LMI+ minus LMI− cells (%)',
}
GROUP_COLORS = {'R+': reward_palette[1], 'R-': reward_palette[0]}


def pair_label(a, b):
    return f'{a:+d}/{b:+d}'.replace('+0', '0')


# ============================================================================
# Signal, noise and pattern drift
# ============================================================================


def load_response_vectors():
    """{mouse: (reward_group, cells x trials array, day of each trial)},
    with the selection of figure_3h_j.load_and_process_data (all cells,
    last N_MAP_TRIALS mapping trials per day, mean over WIN)."""
    _, _, mice, db = database.select_sessions_from_db(paths.db_path, paths.nwb_dir, two_p_imaging='yes')
    out = {}
    for mouse in mice:
        xarr = imaging.load_mouse_xarray(mouse, paths.tensor_dir, 'tensor_xarray_mapping_data.nc')
        xarr = xarr.sel(trial=xarr['day'].isin(DAYS))
        n_trials = xarr[0, :, 0].groupby('day').count(dim='trial').values
        if np.any(n_trials < N_MAP_TRIALS):
            print(f'Not enough mapping trials for {mouse}, skipped.')
            continue
        d = xarr.groupby('day').apply(lambda x: x.isel(trial=slice(-N_MAP_TRIALS, None)))
        d = imaging.select_time(d, WIN[0], WIN[1]).mean(dim='time')
        out[mouse] = (database.get_mouse_reward_group_from_db(paths.db_path, mouse, db), d)
    return out


def signal_noise(trials):
    """(signal power, noise power) of a cells x trials array."""
    n = trials.shape[1]
    noise = trials.var(axis=1, ddof=1).sum()
    mu = trials.mean(axis=1)
    return mu @ mu - noise / n, noise


def split_half_reliability(trials, rng, n_splits=N_SPLITS):
    """Spearman-Brown corrected split-half reliability of the mean pattern."""
    n, h = trials.shape[1], trials.shape[1] // 2
    r = []
    for _ in range(n_splits):
        idx = rng.permutation(n)
        r.append(np.corrcoef(trials[:, idx[:h]].mean(1), trials[:, idx[h:]].mean(1))[0, 1])
    r = np.mean(r)
    return 2 * r / (1 + r)


def compute_metrics(vectors):
    day_rows, drift_rows = [], []
    rng = np.random.default_rng(SEED)
    for mouse, (rg, d) in vectors.items():
        X, day_of = d.values, d['day'].values
        cosine = compute_within_day_metrics([compute_similarity_matrix(d, 'cosine')], [mouse], rg).iloc[0]
        for day in DAYS:
            s, n = signal_noise(X[:, day_of == day])
            day_rows.append(
                {
                    'mouse_id': mouse,
                    'reward_group': rg,
                    'day': day,
                    'signal_power': s,
                    'noise_power': n,
                    'snr': s / n,
                    'cosine_observed': cosine[f'within_day{day:+d}'],
                    'cosine_predicted': s / (s + n),
                }
            )
        mean = {day: X[:, day_of == day].mean(1) for day in DAYS}
        rel = {day: split_half_reliability(X[:, day_of == day], rng) for day in DAYS}
        for a, b in DAY_PAIRS:
            r = pearsonr(mean[a], mean[b])[0]
            drift_rows.append(
                {
                    'mouse_id': mouse,
                    'reward_group': rg,
                    'pair': pair_label(a, b),
                    'r': r,
                    'reliability_a': rel[a],
                    'reliability_b': rel[b],
                    'r_corrected': r / np.sqrt(rel[a] * rel[b]),
                }
            )
    return pd.DataFrame(day_rows), pd.DataFrame(drift_rows)


def within_block(vectors):
    """Per mouse and day: correlation of the mean response vectors of the
    first and the last N_HALF trials and of the odd and the even trials, and
    the first/last correlation corrected by the split-half reliabilities of
    the two halves."""
    rows = []
    rng = np.random.default_rng(SEED)
    for mouse, (rg, d) in vectors.items():
        X, day_of = d.values, d['day'].values
        for day in DAYS:
            t = X[:, day_of == day]  # block order
            first, last = t[:, :N_HALF], t[:, -N_HALF:]
            r = pearsonr(first.mean(1), last.mean(1))[0]
            rel_first, rel_last = split_half_reliability(first, rng), split_half_reliability(last, rng)
            rows.append(
                {
                    'mouse_id': mouse,
                    'reward_group': rg,
                    'day': day,
                    'r_first_last': r,
                    'r_odd_even': pearsonr(t[:, 0::2].mean(1), t[:, 1::2].mean(1))[0],
                    'reliability_first': rel_first,
                    'reliability_last': rel_last,
                    'r_corrected': r / np.sqrt(rel_first * rel_last),
                }
            )
    return pd.DataFrame(rows)


def _comparison_row(rg, measure, test, x, y, names, paired=True):
    res = wilcoxon(x, y) if paired else mannwhitneyu(x, y)
    return {
        'reward_group': rg,
        'measure': measure,
        'comparison': f'{names[0]} vs {names[1]}',
        'test': test,
        'n_mice': len(x) if paired else f'{len(x)} / {len(y)}',
        'mean_1': np.mean(x),
        'mean_2': np.mean(y),
        'median_1': np.median(x),
        'median_2': np.median(y),
        'statistic': res.statistic,
        'p_value': res.pvalue,
    }


def amplitude_tables(vectors):
    """Population response (mean over cells) per mouse, day and bin of BIN
    trials, and per mouse, day and half ('first' / 'last' N_HALF trials)."""
    bins, halves = [], []
    for mouse, (rg, d) in vectors.items():
        X, day_of = d.values, d['day'].values
        for day in DAYS:
            r = np.nanmean(X[:, day_of == day], axis=0)  # block order
            for b in range(int(np.ceil(len(r) / BIN))):
                bins.append(
                    dict(
                        mouse_id=mouse,
                        reward_group=rg,
                        day=day,
                        bin=b + 1,
                        response=r[b * BIN : (b + 1) * BIN].mean(),
                    )
                )
            for half, sl in [('first', slice(0, N_HALF)), ('last', slice(-N_HALF, None))]:
                halves.append(
                    dict(mouse_id=mouse, reward_group=rg, day=day, half=half, response=r[sl].mean())
                )
    return pd.DataFrame(bins), pd.DataFrame(halves)


def pre_post_halves(halves):
    """Per mouse and half: mean response over the pre and the post days."""
    period = halves['day'].map({d: 'pre' for d in PRE_DAYS} | {d: 'post' for d in POST_DAYS})
    return (
        halves.assign(period=period)
        .dropna(subset=['period'])
        .groupby(['reward_group', 'mouse_id', 'half', 'period'])['response']
        .mean()
        .unstack()
    )


def amplitude_stats(halves):
    """First vs last N_HALF trials (each mouse averaged over days), and pre vs
    post computed from either half, per group (Wilcoxon signed-rank, n = mice)."""
    block = halves.groupby(['reward_group', 'mouse_id', 'half'])['response'].mean().unstack()
    pp = pre_post_halves(halves)
    test = 'Wilcoxon signed-rank (n = mice)'
    rows = []
    for rg in ['R+', 'R-']:
        b = block.loc[rg]
        rows.append(
            _comparison_row(rg, 'amplitude first/last', test, b['first'], b['last'], ('first', 'last'))
        )
        for half in ('first', 'last'):
            g = pp.loc[(rg, slice(None), half), :]
            measure = f'amplitude pre/post, {half} {N_HALF} trials'
            rows.append(_comparison_row(rg, measure, test, g['pre'], g['post'], ('pre', 'post')))
    return pd.DataFrame(rows)


def within_block_means(wb):
    """Each mouse's median over days of the within-block measures."""
    cols = ['r_first_last', 'r_odd_even', 'r_corrected']
    return wb.groupby(['reward_group', 'mouse_id'])[cols].median()


def within_block_stats(wb, drift_df):
    """First/last vs odd/even correlation (each mouse's median over days, and
    per day), and the noise-corrected first/last correlation (median over
    days) vs the baseline drift (-2/-1), per group (Wilcoxon signed-rank,
    n = mice)."""
    base = pair_label(*BASELINE_PAIR)
    drift = drift_df.pivot(index=['reward_group', 'mouse_id'], columns='pair', values='r_corrected')
    mean = within_block_means(wb)
    rows = []
    for rg in ['R+', 'R-']:
        names = ('first/last', 'odd/even')
        g = mean.loc[rg]
        test = 'Wilcoxon signed-rank (n = mice), median over days'
        rows.append(_comparison_row(rg, 'r all days', test, g['r_first_last'], g['r_odd_even'], names))
        for day in DAYS:
            d = wb[(wb['reward_group'] == rg) & (wb['day'] == day)]
            test = 'Wilcoxon signed-rank (n = mice)'
            rows.append(_comparison_row(rg, f'r day {day}', test, d['r_first_last'], d['r_odd_even'], names))
        x = g['r_corrected']
        y = drift.loc[rg][base].reindex(x.index)
        test = 'Wilcoxon signed-rank (n = mice), noise-corrected correlation, within: median over days'
        rows.append(_comparison_row(rg, f'drift {WITHIN}', test, x, y, (WITHIN, base)))
    return pd.DataFrame(rows)


def pre_post_means(day_df, metrics):
    period = day_df['day'].map({d: 'pre' for d in PRE_DAYS} | {d: 'post' for d in POST_DAYS})
    return (
        day_df.assign(period=period)
        .dropna(subset=['period'])
        .groupby(['reward_group', 'mouse_id', 'period'])[metrics]
        .mean()
    )


def signal_noise_row(rg, measure, test, n, median_pre, median_post, stat, p):
    return {
        'reward_group': rg,
        'measure': measure,
        'test': test,
        'n_mice': n,
        'median_pre': median_pre,
        'median_post': median_post,
        'statistic': stat,
        'p_value': p,
        'significance': significance_stars(p) if np.isfinite(p) else '',
    }


def signal_noise_stats(day_df, drift_df):
    metrics = ['signal_power', 'noise_power', 'snr', 'cosine_observed', 'cosine_predicted']
    pp = pre_post_means(day_df, metrics).unstack('period')
    base, learn = pair_label(*BASELINE_PAIR), pair_label(*LEARNING_PAIR)
    drift = drift_df.pivot(index=['reward_group', 'mouse_id'], columns='pair', values='r_corrected')
    rows = []
    for rg in ['R+', 'R-']:
        g = pp.loc[rg]
        for m in metrics:
            rows.append(
                signal_noise_row(
                    rg,
                    m,
                    'Wilcoxon signed-rank, pre (days -2, -1) vs post (days +1, +2)',
                    len(g),
                    g[(m, 'pre')].median(),
                    g[(m, 'post')].median(),
                    *wilcoxon(g[(m, 'pre')], g[(m, 'post')]),
                )
            )
        g = drift.loc[rg]
        rows.append(
            signal_noise_row(
                rg,
                f'drift {learn}',
                f'Wilcoxon signed-rank, noise-corrected correlation {learn} vs {base}',
                len(g),
                g[base].median(),
                g[learn].median(),
                *wilcoxon(g[learn] - g[base]),
            )
        )
    x, y = drift.loc['R+'][base], drift.loc['R-'][base]
    rows.append(
        signal_noise_row(
            'R+ vs R-',
            f'drift {base}',
            'Mann-Whitney U, noise-corrected correlation (median_pre: R+, median_post: R-)',
            len(x) + len(y),
            x.median(),
            y.median(),
            *mannwhitneyu(x, y),
        )
    )
    r, _ = pearsonr(day_df['cosine_observed'], day_df['cosine_predicted'])
    rows.append(
        signal_noise_row(
            'all',
            'observed vs predicted within-day cosine',
            'Pearson r across mouse-days (descriptive)',
            day_df['mouse_id'].nunique(),
            np.nan,
            r,
            r,
            np.nan,
        )
    )
    return pd.DataFrame(rows), pp


# ============================================================================
# LMI between two consecutive days
# ============================================================================


def auc_rows(ranks, post_idx, n_post):
    """AUC of each row for the trials in post_idx against the others, from ranks."""
    n_pre = ranks.shape[-1] - n_post
    return (ranks[..., post_idx].sum(-1) - n_post * (n_post + 1) / 2) / (n_pre * n_post)


def lmi_two_days(pre, post, rng, n_shuffles=N_SHUFFLES):
    """LMI = 2 (AUC - 0.5) of each cell (rows) and its shuffle percentile, as
    imaging.compute_roc (scipy percentileofscore, kind='rank'), vectorised
    over cells with the same label shuffles for all cells."""
    X = np.concatenate([pre, post], axis=1)
    n, n_post = X.shape[1], post.shape[1]
    ranks = rankdata(X, axis=1)
    auc = auc_rows(ranks, np.arange(n - n_post, n), n_post)
    null = np.stack([auc_rows(ranks, rng.permutation(n)[:n_post], n_post) for _ in range(n_shuffles)], 1)
    below = (null < auc[:, None]).sum(1)
    at_or_below = (null <= auc[:, None]).sum(1)
    p = (below + at_or_below + (at_or_below > below)) * 0.5 / n_shuffles
    return 2 * (auc - 0.5), p


def mouse_drift_lmi(mouse, rng):
    """Cell x day pair LMI and p of one mouse, preprocessed as pipeline step 05."""
    data = imaging.load_mouse_xarray(
        mouse, paths.tensor_dir, 'tensor_xarray_mapping_data.nc', subtracted=False
    )
    data = data - np.nanmean(imaging.select_time(data, *BASELINE_WIN), axis=2, keepdims=True)
    data = imaging.last_trials(data)
    resp = imaging.select_time(data, *RESPONSE_WIN).mean(dim='time')
    days = resp['day'].values
    rows = []
    for a, b in DAY_PAIRS:
        if not (np.any(days == a) and np.any(days == b)):
            continue
        lmi, p = lmi_two_days(resp.values[:, days == a], resp.values[:, days == b], rng)
        rows.append(
            pd.DataFrame(
                dict(
                    mouse_id=mouse,
                    roi=resp['roi'].values,
                    pair=pair_label(a, b),
                    lmi=lmi,
                    lmi_p=p,
                    n_trials_a=int((days == a).sum()),
                    n_trials_b=int((days == b).sum()),
                )
            )
        )
    return pd.concat(rows, ignore_index=True)


def drift_lmi(groups, lmi_df):
    """Cell x pair LMIs of all mice, seeded per mouse."""
    mice = sorted(lmi_df['mouse_id'].unique())
    seeds = np.random.SeedSequence(SEED).spawn(len(mice))
    cells = pd.concat(
        [mouse_drift_lmi(m, np.random.default_rng(s)) for m, s in zip(mice, seeds)], ignore_index=True
    )
    cells['reward_group'] = cells['mouse_id'].map(groups)
    return cells


def fractions(cells):
    """Per mouse and pair, from cell counts (so that equal counts give exactly equal fractions)."""
    pos = (cells['lmi_p'] >= LMI_POSITIVE_THRESHOLD).astype(int)
    neg = (cells['lmi_p'] <= LMI_NEGATIVE_THRESHOLD).astype(int)
    cells = cells.assign(positive=pos, negative=neg, net=pos - neg)
    return (
        cells.groupby(['mouse_id', 'reward_group', 'pair'])
        .agg(**{m: (m, 'mean') for m in FRACTIONS}, n_cells=('roi', 'size'))
        .reset_index()
    )


def lmi_row(pair, measure, rg, test, x_plus, x_minus, stat, p):
    return dict(
        pair=pair,
        measure=measure,
        reward_group=rg,
        test=test,
        median_r_plus=x_plus.median(),
        median_r_minus=x_minus.median(),
        n_r_plus=len(x_plus),
        n_r_minus=len(x_minus),
        statistic=stat,
        p_value=p,
        significance=significance_stars(p),
    )


def drift_lmi_stats(frac):
    """Change from -2/-1 to -1/0 of each fraction; net fraction against zero and R+ vs R-."""
    base, learn = pair_label(*BASELINE_PAIR), pair_label(*LEARNING_PAIR)
    rows = []
    for m in FRACTIONS:
        w = frac.pivot(index=['mouse_id', 'reward_group'], columns='pair', values=m).reset_index()
        d = {rg: (w[learn] - w[base])[w['reward_group'] == rg].dropna() for rg in ['R+', 'R-']}
        for rg, x in d.items():
            rows.append(
                lmi_row(
                    f'{learn} minus {base}', m, rg, 'Wilcoxon signed-rank', d['R+'], d['R-'], *wilcoxon(x)
                )
            )
    for pair in [base, learn, REFERENCE]:
        g = {rg: frac[(frac['pair'] == pair) & (frac['reward_group'] == rg)]['net'] for rg in ['R+', 'R-']}
        for rg, x in g.items():
            rows.append(lmi_row(pair, 'net', rg, 'Wilcoxon signed-rank vs 0', g['R+'], g['R-'], *wilcoxon(x)))
        rows.append(
            lmi_row(
                pair, 'net', 'R+ vs R-', 'Mann-Whitney U', g['R+'], g['R-'], *mannwhitneyu(g['R+'], g['R-'])
            )
        )
    return pd.DataFrame(rows)
