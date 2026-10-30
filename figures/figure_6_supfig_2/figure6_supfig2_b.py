"""
Figure 6—figure supplement 2b: Reactivation rate and the outcome of the
previous trial

Reactivations are detected in correct-rejection no-stim trials (step 07). For
each of these trials, the trial just before it in the session is classified
as rewarded (lick on a rewarded trial: auditory hits, and whisker hits in R+
mice) or unrewarded (misses, catch trials, and whisker hits in R- mice), and
the last whisker trial before it as a hit (lick) or a miss. The event rate of
a trial is its number of events per minute of trial (-1 to 6 s).

Per mouse and day, rates are averaged over the trials of each category
(a category needs at least MIN_TRIALS trials).
  b, left: after a rewarded vs an unrewarded trial;
  b, middle: after a whisker hit vs a whisker miss (last whisker trial);
  paired Wilcoxon signed-rank test per reward group (n = mice).
  b, right: R+ vs R- after unrewarded trials only, so that the group
     difference of Fig. 6c cannot come from the rewards collected by R+ mice
     (Mann-Whitney U).
Figure: Day 0. Stats file: Days 0, +1 and +2. Bars: mean across mice with
bootstrapped 95% CI; lines and dots: mice.

Outputs: <figures_dir>/Figure6_SupFig2/figure6_supfig2_b.pdf,
figure6_supfig2_b_trials.csv (correct-rejection trial),
figure6_supfig2_b_data.csv (mouse x day x category),
figure6_supfig2_b_stats.csv.
"""

import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import mannwhitneyu, wilcoxon

from fast_learning import imaging, paths, reactivations as rx
from fast_learning.plotting import panel_size, reward_palette, save_figure, set_style
from fast_learning.stats import format_p

DAYS = [0, 1, 2]
PLOT_DAY = 0
MIN_TRIALS = 3
SAMPLING_RATE = rx.SAMPLING_RATE
GROUP_COLORS = {'R+': reward_palette[1], 'R-': reward_palette[0]}
COMPARISONS = {
    'previous_trial': ('rewarded', 'unrewarded'),
    'last_whisker_trial': ('hit', 'miss'),
}
TICK_LABELS = {'rewarded': 'Rew.', 'unrewarded': 'Unrew.'}
YLABEL = 'Reactivations\n(events/min)'
OUTPUT_DIR = os.path.join(paths.manuscript_output_dir, 'Figure6_SupFig2')


# ============================================================================
# Data
# ============================================================================


def trial_table(mouse, day):
    """Trials of one mouse-day in session order (tensor trial coordinates)."""
    x = imaging.load_mouse_xarray(mouse, paths.tensor_dir, 'tensor_xarray_learning_data.nc', subtracted=False)
    x = x.sel(trial=x['day'] == day)
    cols = ['trial_id', 'whisker_stim', 'auditory_stim', 'no_stim', 'lick_flag', 'reward_available']
    return pd.DataFrame({c: x.coords[c].values for c in cols})


def label_trials(trials):
    """Outcome of the previous trial and of the last whisker trial, for every
    trial (NaN when there is none)."""
    t = trials.reset_index(drop=True)
    rewarded = np.where((t['reward_available'] == 1) & (t['lick_flag'] == 1), 'rewarded', 'unrewarded')
    previous = pd.Series(rewarded).shift(1)
    previous[t['trial_id'].diff() != 1] = np.nan  # previous trial missing from the tensor
    whisker = pd.Series(np.where(t['lick_flag'] == 1, 'hit', 'miss')).where(t['whisker_stim'] == 1)
    t['previous_trial'] = previous
    t['last_whisker_trial'] = whisker.ffill().shift(1)
    return t


def correct_rejection_rates(results):
    """One row per correct-rejection trial: events, rate and labels."""
    rows = []
    for rg, key in [('R+', 'r_plus_results'), ('R-', 'r_minus_results')]:
        for mouse, res in results[key].items():
            for day in DAYS:
                d = res['days'].get(day)
                if d is None:
                    continue
                t = label_trials(trial_table(mouse, day))
                cr = t[(t['no_stim'] == 1) & (t['lick_flag'] == 0)].copy()
                if len(cr) != d['n_trials']:
                    raise ValueError(f'{mouse} day {day}: {len(cr)} trials, results have {d["n_trials"]}.')
                n_t = d['n_timepoints']
                events = np.asarray(d['events'], dtype=int) // n_t
                cr['n_events'] = np.bincount(events, minlength=len(cr))
                cr['rate'] = cr['n_events'] / (n_t / SAMPLING_RATE / 60)
                rows.append(cr.assign(mouse_id=mouse, reward_group=rg, day=day))
    return pd.concat(rows, ignore_index=True)


def mouse_means(trials):
    """Mean rate per mouse x day x comparison x category (>= MIN_TRIALS trials)."""
    parts = []
    for comparison in COMPARISONS:
        g = trials.dropna(subset=[comparison]).groupby(['reward_group', 'mouse_id', 'day', comparison])[
            'rate'
        ]
        m = g.agg(['mean', 'size']).reset_index().rename(columns={comparison: 'category', 'size': 'n_trials'})
        parts.append(m[m['n_trials'] >= MIN_TRIALS].assign(comparison=comparison))
    return pd.concat(parts, ignore_index=True).rename(columns={'mean': 'rate'})


# ============================================================================
# Statistics
# ============================================================================


def compute_stats(means):
    rows = []
    for day in DAYS:
        for comparison, (a, b) in COMPARISONS.items():
            for rg in GROUP_COLORS:
                d = means.query('day == @day and comparison == @comparison and reward_group == @rg')
                paired = d.pivot(index='mouse_id', columns='category', values='rate').dropna()
                if len(paired) < 3 or not {a, b} <= set(paired.columns):
                    continue
                res = wilcoxon(paired[a], paired[b])
                rows.append(
                    dict(
                        day=day,
                        comparison=f'{comparison}: {a} vs {b}',
                        reward_group=rg,
                        test='Wilcoxon signed-rank (n = mice)',
                        n_mice=len(paired),
                        median_1=paired[a].median(),
                        median_2=paired[b].median(),
                        n_mice_higher_1=int((paired[a] > paired[b]).sum()),
                        statistic=res.statistic,
                        p_value=res.pvalue,
                    )
                )
        d = means.query('day == @day and comparison == "previous_trial" and category == "unrewarded"')
        rp, rm = (d.loc[d['reward_group'] == g, 'rate'] for g in GROUP_COLORS)
        res = mannwhitneyu(rp, rm, alternative='two-sided')
        rows.append(
            dict(
                day=day,
                comparison='after unrewarded trials: R+ vs R-',
                reward_group='R+ vs R-',
                test='Mann-Whitney U (n = mice)',
                n_mice=f'{len(rp)} / {len(rm)}',
                median_1=rp.median(),
                median_2=rm.median(),
                statistic=res.statistic,
                p_value=res.pvalue,
            )
        )
    return pd.DataFrame(rows)


# ============================================================================
# Plotting
# ============================================================================


def paired_panel(ax, means, stats, comparison, xlabel):
    a, b = COMPARISONS[comparison]
    d = means.query('day == @PLOT_DAY and comparison == @comparison')
    for gi, rg in enumerate(GROUP_COLORS):
        g = d[d['reward_group'] == rg]
        paired = g.pivot(index='mouse_id', columns='category', values='rate').dropna()
        x = np.array([gi * 3, gi * 3 + 1])
        for _, row in paired.iterrows():
            ax.plot(x, [row[a], row[b]], '-', color='grey', linewidth=0.5, alpha=0.6)
        long = paired.reset_index().melt(id_vars='mouse_id', value_vars=[a, b], value_name='v')
        sns.barplot(
            data=long.assign(x=np.where(long['category'] == a, x[0], x[1])),
            x='x',
            y='v',
            order=list(range(5)),
            native_scale=True,
            color=GROUP_COLORS[rg],
            alpha=0.7,
            errorbar=('ci', 95),
            seed=0,
            width=0.7,
            ax=ax,
        )
        p = stats.query('day == @PLOT_DAY and reward_group == @rg and comparison.str.startswith(@comparison)')
        if len(p):
            ax.text(
                x.mean(),
                1.0,
                format_p(p['p_value'].iloc[0]),
                transform=ax.get_xaxis_transform(),
                ha='center',
                va='bottom',
                fontsize=5,
            )
        ax.text(
            x.mean(),
            1.08,
            rg,
            transform=ax.get_xaxis_transform(),
            ha='center',
            va='bottom',
            color=GROUP_COLORS[rg],
        )
    ax.set_xticks([0, 1, 3, 4], [TICK_LABELS.get(c, c.capitalize()) for c in (a, b)] * 2)
    ax.set_xlim(-0.7, 4.7)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(YLABEL)


def group_panel(ax, means, stats):
    d = means.query('day == @PLOT_DAY and comparison == "previous_trial" and category == "unrewarded"')
    kw = dict(data=d, x='reward_group', y='rate', order=list(GROUP_COLORS), ax=ax)
    sns.barplot(
        **kw,
        hue='reward_group',
        palette=GROUP_COLORS,
        alpha=0.7,
        errorbar=('ci', 95),
        seed=0,
        width=0.6,
        legend=False,
    )
    sns.swarmplot(**kw, color='black', size=2)
    p = stats.query('day == @PLOT_DAY and reward_group == "R+ vs R-"')['p_value'].iloc[0]
    ax.text(0.5, 1.0, format_p(p), transform=ax.get_xaxis_transform(), ha='center', va='bottom', fontsize=5)
    ax.set_xlabel('After unrewarded trials')
    ax.set_ylabel(YLABEL)


def plot(means, stats, filename='figure6_supfig2_b'):
    set_style()
    fig, axes = plt.subplots(1, 3, figsize=panel_size(3, w=1.15))
    paired_panel(axes[0], means, stats, 'previous_trial', 'Previous trial')
    paired_panel(axes[1], means, stats, 'last_whisker_trial', 'Last whisker trial')
    group_panel(axes[2], means, stats)
    axes[0].set_ylim(0, 30)
    axes[1].set_ylim(0, 25)
    axes[2].set_ylim(bottom=0)
    sns.despine()
    plt.tight_layout()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_figure(fig, os.path.join(OUTPUT_DIR, f'{filename}.pdf'))
    plt.close(fig)


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    with open(os.path.join(rx.RESULTS_DIR, 'reactivation_results_p99.pkl'), 'rb') as f:
        results = pickle.load(f)
    trials = correct_rejection_rates(results)
    means = mouse_means(trials)
    stats = compute_stats(means)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    trials.to_csv(os.path.join(OUTPUT_DIR, 'figure6_supfig2_b_trials.csv'), index=False)
    means.to_csv(os.path.join(OUTPUT_DIR, 'figure6_supfig2_b_data.csv'), index=False)
    stats.to_csv(os.path.join(OUTPUT_DIR, 'figure6_supfig2_b_stats.csv'), index=False)
    plot(means, stats)
    cols = [
        'day',
        'comparison',
        'reward_group',
        'n_mice',
        'median_1',
        'median_2',
        'n_mice_higher_1',
        'p_value',
    ]
    print(stats[cols].round(3).to_string())
