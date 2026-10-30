"""
Figure 6—figure supplement 3a-c: Robustness of the reactivation results to
the detection threshold, the participation threshold and licking

Each column is one analysis condition; the main analysis (filled grey band)
is the 99th surrogate percentile for detection, a participation threshold of
2.5 x each cell's noise SD and the whole catch trial (-1 to 6 s).
  - Detection threshold: 99.5th and 99.9th surrogate percentiles. Events are
    re-detected at each percentile (step 07 for the event rates; held-out
    detection, for each half of the cells and each split, for participation;
    cached in <processed_dir>/reactivation/heldout_events_p<...>.pkl, about
    20 min per percentile on 35 cores, rerun with --recompute).
  - Participation threshold: 5 and 10 x the cell's noise SD.
  - Lick-free control: events detected in the part of the catch trial without
    licks (-1 to +1 s; step 07 --nolick and step 08 --nolick).

- a: Day-0 reactivation rate, R+ vs R- (Fig. 6c); mean and bootstrapped 95%
  CI across mice; Mann-Whitney U (n = mice). The participation threshold does
  not change the events, so its conditions are not repeated here.
- b: slope of day-0 participation above chance vs LMI (Fig. 6d): linear
  mixed-effects model with a random intercept and slope per mouse; slope and
  95% CI.
- c: day trend of participation above chance for LMI+ and LMI- cells
  (Fig. 6e): per-mouse slope across days; mean and bootstrapped 95% CI across
  mice; Wilcoxon signed-rank test against zero (n = mice).
In b and c, filled markers: p < 0.05; open: p >= 0.05. Participation in
held-out cells, mice of the participation selection (at p99).

Outputs: <figures_dir>/Figure6_SupFig3/figure6_supfig3_a_c.pdf,
figure6_supfig3_a_data.csv (mouse x condition), figure6_supfig3_c_data.csv
(per-mouse slopes), figure6_supfig3_stats.csv (all conditions, also the
participation rate without chance correction).
"""

import argparse
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed
from scipy.stats import mannwhitneyu

from fast_learning import paths, participation as pt, reactivations as rx
from fast_learning.plotting import panel_size, reward_palette, save_figure, set_style
from fast_learning.stats import bootstrap_ci, format_p, lmm_slope, per_mouse_slope_test

MAIN = pt.PARTICIPATION_THRESHOLD
N_JOBS = 35
# (label, detection percentile, participation threshold, lick-free)
CONDITIONS = [
    ('Main', 99, MAIN, False),
    ('p99.5', 99.5, MAIN, False),
    ('p99.9', 99.9, MAIN, False),
    ('5 SD', 99, 5, False),
    ('10 SD', 99, 10, False),
    ('Lick-free', 99, MAIN, True),
]
EVENT_CONDITIONS = [
    c for c in CONDITIONS if c[2] == MAIN
]  # the participation threshold leaves events unchanged
GROUPS = [('R+', 'positive'), ('R+', 'negative'), ('R-', 'positive'), ('R-', 'negative')]
GROUP_LABELS = {
    ('R+', 'positive'): 'R+ LMI+',
    ('R+', 'negative'): 'R+ LMI−',
    ('R-', 'positive'): 'R− LMI+',
    ('R-', 'negative'): 'R− LMI−',
}
RG_COLORS = {'R+': reward_palette[1], 'R-': reward_palette[0]}
LMI_MARKERS = {'positive': 'o', 'negative': 's'}
MEASURES = ('excess', 'rate')  # excess is plotted; rate in the stats file
OUTPUT_DIR = os.path.join(paths.manuscript_output_dir, 'Figure6_SupFig3')


# ============================================================================
# Data
# ============================================================================


def _reward_groups():
    with open(os.path.join(rx.RESULTS_DIR, 'reactivation_results_p99.pkl'), 'rb') as f:
        res = pickle.load(f)
    return {m: 'R+' for m in res['r_plus_results']} | {m: 'R-' for m in res['r_minus_results']}


def heldout_tables(percentile, threshold, nolick, recompute=False):
    """(merged, per_day) of held-out cells for one condition. The main
    percentile is read from step 08; other percentiles (main participation
    threshold) are re-detected here and cached."""
    if percentile == pt.DETECTION_PERCENTILE:
        return pt.load_participation(threshold, nolick)

    cache = os.path.join(pt.RESULTS_DIR, f'heldout_events_{rx.percentile_tag(percentile)}.pkl')
    mice = sorted(rx.load_participation_mice())
    if recompute or not os.path.exists(cache):
        print(f'Held-out detection at p{percentile:g} for {len(mice)} mice ...')
        detections = Parallel(n_jobs=N_JOBS, verbose=5)(
            delayed(pt.detect_heldout_events)(m, None, percentile) for m in mice
        )
        with open(cache, 'wb') as f:
            pickle.dump({d['mouse']: d for d in detections}, f)
    with open(cache, 'rb') as f:
        detections = pickle.load(f)

    def one(det):
        rois, days = pt.load_heldout_data(det['mouse'])
        sub_by_day = {d: v['sub'] for d, v in days.items()}
        noise = pt.noise_sd(sub_by_day)
        thresholds = {threshold: pt.cell_thresholds(noise, [threshold])[threshold]}
        cells = pd.DataFrame(dict(mouse_id=det['mouse'], roi=rois[noise > 0]))
        rates = pt.heldout_split_rates(det, days, thresholds).merge(cells)
        return rates, pt.chance_rates(det['mouse'], rois, sub_by_day, thresholds)

    out = Parallel(n_jobs=N_JOBS)(delayed(one)(d) for d in detections.values())
    split_df = pd.concat([s for s, _ in out if s is not None], ignore_index=True)
    chance_df = pd.concat([c for _, c in out if c is not None], ignore_index=True)
    per_day = pt.add_excess(
        pt.average_over_splits(split_df[split_df['role'] == 'heldout']), chance_df, threshold
    )
    lmi_df = pd.read_csv(os.path.join(paths.processed_dir, 'lmi_results.csv'))
    merged = pt.merge_with_lmi(pt.aggregate_across_days(per_day), lmi_df, _reward_groups())
    return merged, per_day


def day0_event_rates(percentile, nolick):
    """Per mouse: Day-0 reactivation rate (events/min) of step 07 (Fig. 6c)."""
    path = os.path.join(rx.results_dir(nolick), f'reactivation_results_{rx.percentile_tag(percentile)}.pkl')
    with open(path, 'rb') as f:
        res = pickle.load(f)
    return pd.DataFrame(
        [
            dict(mouse_id=m, reward_group=rg, rate=r['days'][0]['event_frequency'])
            for rg, key in [('R+', 'r_plus_results'), ('R-', 'r_minus_results')]
            for m, r in res[key].items()
            if 0 in r['days']
        ]
    )


# ============================================================================
# Statistics
# ============================================================================


def condition_stats(merged, per_day, labels):
    """Fig. 6d LMM slopes and Fig. 6e per-mouse day slopes for one condition."""
    rows, slopes = [], []
    day0 = merged.dropna(subset=['lmi', 'learning_rate'])
    day0 = day0[day0['reliable_learning']]
    for rg in ['R+', 'R-']:
        fit = lmm_slope(day0[day0['reward_group'] == rg], 'learning_rate', 'lmi')
        rows.append(
            dict(
                **labels,
                panel='6d',
                reward_group=rg,
                lmi_category='',
                estimate=fit['slope'],
                ci_low=fit['ci_low'],
                ci_high=fit['ci_high'],
                p_value=fit['p_value'],
                n=f"{(day0['reward_group'] == rg).sum()} cells, {day0.loc[day0['reward_group'] == rg, 'mouse_id'].nunique()} mice",
            )
        )
    cells = merged.loc[
        merged['lmi_category'].isin(['positive', 'negative']),
        ['mouse_id', 'roi', 'lmi_category', 'reward_group'],
    ]
    d = per_day.merge(cells, on=['mouse_id', 'roi'])
    d = d[d['reliable']]
    mouse_day = (
        d.groupby(['mouse_id', 'reward_group', 'lmi_category', 'day'])['participation_rate']
        .mean()
        .reset_index()
    )
    for rg, cat in GROUPS:
        test = per_mouse_slope_test(
            mouse_day[(mouse_day['reward_group'] == rg) & (mouse_day['lmi_category'] == cat)]
        )
        s = pd.Series(test['slopes'])
        low, high = bootstrap_ci(s.values)
        rows.append(
            dict(
                **labels,
                panel='6e',
                reward_group=rg,
                lmi_category=cat,
                estimate=s.mean(),
                ci_low=low,
                ci_high=high,
                p_value=test['p_value'],
                n=f"{test['n_mice']} mice",
            )
        )
        slopes += [
            dict(**labels, reward_group=rg, lmi_category=cat, mouse_id=m, slope=v) for m, v in s.items()
        ]
    return rows, slopes


def event_rate_stats(rates):
    rows = []
    for label, g in rates.groupby('condition', sort=False):
        a, b = (g.loc[g['reward_group'] == rg, 'rate'] for rg in ('R+', 'R-'))
        res = mannwhitneyu(a, b, alternative='two-sided')
        rows.append(
            dict(
                condition=label,
                measure='day-0 event rate',
                panel='6c',
                reward_group='R+ vs R-',
                lmi_category='',
                estimate=a.mean() - b.mean(),
                ci_low=np.nan,
                ci_high=np.nan,
                p_value=res.pvalue,
                n=f'{len(a)} / {len(b)} mice',
            )
        )
    return pd.DataFrame(rows)


# ============================================================================
# Figure
# ============================================================================


def marker_kw(color, p):
    return dict(color=color, markerfacecolor=color if p < 0.05 else 'white', markeredgecolor=color)


def plot_event_rates(ax, rates, stats):
    order = [c[0] for c in EVENT_CONDITIONS]
    sns.barplot(
        data=rates,
        x='condition',
        y='rate',
        order=order,
        hue='reward_group',
        hue_order=['R+', 'R-'],
        palette=RG_COLORS,
        errorbar=('ci', 95),
        seed=0,
        alpha=0.7,
        edgecolor='black',
        width=0.7,
        ax=ax,
    )
    for xi, label in enumerate(order):
        p = stats.loc[stats['condition'] == label, 'p_value'].iloc[0]
        y = 1.0 + 0.07 * (xi % 2)  # staggered so that neighbours do not overlap
        ax.text(xi, y, format_p(p), transform=ax.get_xaxis_transform(), ha='center', va='bottom', fontsize=5)
    ax.set_xlabel('')
    ax.set_ylabel('Day-0 events / min')
    ax.legend(frameon=False, title='', loc='upper right')


def plot_lmm(ax, stats):
    s = stats[stats['panel'] == '6d']
    order = [c[0] for c in CONDITIONS]
    for i, rg in enumerate(['R+', 'R-']):
        g = s[s['reward_group'] == rg].set_index('condition').loc[order]
        for xi, (_, row) in enumerate(g.iterrows()):
            ax.errorbar(
                xi + (i - 0.5) * 0.25,
                row['estimate'],
                yerr=[[row['estimate'] - row['ci_low']], [row['ci_high'] - row['estimate']]],
                fmt='o',
                markersize=3,
                linewidth=1,
                **marker_kw(RG_COLORS[rg], row['p_value']),
            )
    ax.axhline(0, color='grey', linestyle='--', linewidth=0.6)
    ax.set_xticks(range(len(order)), order)
    ax.set_ylabel('Slope vs LMI, day 0\n(above chance, LMM)')
    legend(ax, [(rg, 'o', RG_COLORS[rg]) for rg in ['R+', 'R-']], loc='upper right')


def plot_day_slopes(ax, stats):
    s = stats[stats['panel'] == '6e']
    order = [c[0] for c in CONDITIONS]
    offsets = np.linspace(-0.3, 0.3, len(GROUPS))
    for (rg, cat), off in zip(GROUPS, offsets):
        g = s[(s['reward_group'] == rg) & (s['lmi_category'] == cat)].set_index('condition').loc[order]
        for xi, (_, row) in enumerate(g.iterrows()):
            ax.errorbar(
                xi + off,
                row['estimate'],
                yerr=[[row['estimate'] - row['ci_low']], [row['ci_high'] - row['estimate']]],
                fmt=LMI_MARKERS[cat],
                markersize=3,
                linewidth=1,
                **marker_kw(RG_COLORS[rg], row['p_value']),
            )
    ax.axhline(0, color='grey', linestyle='--', linewidth=0.6)
    ax.set_xticks(range(len(order)), order)
    ax.set_ylabel('Slope across days\n(above chance, per day)')
    handles = [(GROUP_LABELS[g], LMI_MARKERS[g[1]], RG_COLORS[g[0]]) for g in GROUPS]
    legend(ax, handles, loc='upper left', bbox_to_anchor=(1.0, 1.0))


def legend(ax, entries, **kw):
    """Legend with filled markers, and open = p >= 0.05."""
    from matplotlib.lines import Line2D

    handles = [
        Line2D([], [], marker=m, color=c, linestyle='', markersize=3, label=lab) for lab, m, c in entries
    ]
    handles.append(
        Line2D(
            [],
            [],
            marker='o',
            color='grey',
            markerfacecolor='white',
            linestyle='',
            markersize=3,
            label='p ≥ 0.05',
        )
    )
    ax.legend(handles=handles, frameon=False, **kw)


def figure(rates, stats, ev_stats):
    set_style()
    widths = [len(EVENT_CONDITIONS), len(CONDITIONS), len(CONDITIONS)]
    fig, axes = plt.subplots(1, 3, figsize=panel_size(3, w=1.4), gridspec_kw={'width_ratios': widths})
    plot_event_rates(axes[0], rates, ev_stats)
    plot_lmm(axes[1], stats)
    plot_day_slopes(axes[2], stats)
    for ax in axes:
        ax.axvspan(-0.5, 0.5, color='grey', alpha=0.08, linewidth=0, zorder=0)
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    sns.despine()
    plt.tight_layout()
    save_figure(fig, os.path.join(OUTPUT_DIR, 'figure6_supfig3_a_c.pdf'))
    plt.close(fig)


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    parser.add_argument(
        '--recompute', action='store_true', help='rerun the held-out detections at p99.5 / p99.9'
    )
    args = parser.parse_args()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    stat_rows, slope_rows = [], []
    for label, perc, thr, nolick in CONDITIONS:
        merged, per_day = heldout_tables(perc, thr, nolick, args.recompute)
        for measure in MEASURES:
            m, d = pt.excess_tables(merged, per_day) if measure == 'excess' else (merged, per_day)
            r, s = condition_stats(m, d, dict(condition=label, measure=measure))
            stat_rows += r
            slope_rows += s
    stats, slopes = pd.DataFrame(stat_rows), pd.DataFrame(slope_rows)
    rates = pd.concat(
        [
            day0_event_rates(perc, nolick).assign(condition=label)
            for label, perc, _, nolick in EVENT_CONDITIONS
        ],
        ignore_index=True,
    )
    ev_stats = event_rate_stats(rates)

    excess = stats[stats['measure'] == 'excess']
    figure(rates, excess, ev_stats)
    pd.concat([ev_stats, stats], ignore_index=True).to_csv(
        os.path.join(OUTPUT_DIR, 'figure6_supfig3_stats.csv'), index=False
    )
    rates.to_csv(os.path.join(OUTPUT_DIR, 'figure6_supfig3_a_data.csv'), index=False)
    slopes[slopes['measure'] == 'excess'].to_csv(
        os.path.join(OUTPUT_DIR, 'figure6_supfig3_c_data.csv'), index=False
    )

    noise = pt.load_participation(MAIN)[1].drop_duplicates(['mouse_id', 'roi'])['noise_sd']
    thr = MAIN * noise
    print(
        f'Noise SD median {noise.median():.3f} (IQR {noise.quantile(0.25):.3f}-{noise.quantile(0.75):.3f}); '
        f'{MAIN:g}-SD threshold median {thr.median():.3f} (IQR {thr.quantile(0.25):.3f}-{thr.quantile(0.75):.3f}) dF/F0'
    )
    pd.set_option('display.width', 200)
    print(ev_stats[['condition', 'estimate', 'p_value', 'n']].round(4).to_string(index=False))
    print(excess.drop(columns=['measure']).round(4).to_string(index=False))
