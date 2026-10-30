"""
Figure 2—figure supplement 2a-b: Learning on the first day without
inactivation, compared with naive mice learning on Day 0

Mice inactivated in wS1 during learning are compared, on their first session
without inactivation, with the naive R+ imaging mice on Day 0:
- a: muscimol, Day +3 (first recovery day, after muscimol on Days 0 to +2);
- b: optogenetics, Day +1 (light off; 4 of the 6 wS1 mice were trained on
  Day +1).
For each cohort, top row whisker lick probability, bottom row d' (whisker vs
catch): whole session; blocks of 20 trials (all trial types, early-lick
trials excluded; block_id of Fig. 1B), first N_BLOCKS blocks; first block.
Bars: mean and bootstrapped 95% CI across mice; dots: mice; curves: mean and
95% CI of the naive mice, thin lines: inactivated mice, thick line: their
mean. Mann-Whitney U, inactivated vs naive (n = mice).

Outputs: <figures_dir>/Figure2_SupFig2/figure2_supfig2_a_b.pdf,
figure2_supfig2_sessions_data.csv (session), figure2_supfig2_blocks_data.csv
(session x block), figure2_supfig2_stats.csv.
"""

import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from fast_learning import inactivation as inact
from fast_learning import paths
from fast_learning.plotting import panel_size, reward_palette, set_style
from fast_learning.stats import format_p

OUTPUT_DIR = os.path.join(paths.manuscript_output_dir, 'Figure2_SupFig2')
N_BLOCKS = 10  # blocks of 20 trials shown
NAIVE = 'Naive, Day 0'
# (name, cohort, day in the session metadata, label of the inactivated mice)
COMPARISONS = [
    ('muscimol', inact.MUSCIMOL_LEARNING, 'recovery_1', 'wS1 muscimol, Day +3'),
    ('optogenetics', inact.OPTO_LEARNING, 'recovery_1', 'wS1 light, Day +1'),
]


# ============================================================================
# Data and statistics
# ============================================================================


def tables(cohort, day, label, naive):
    """Sessions and blocks of 20 trials of the wS1 mice on `day` and of the
    naive mice on Day 0."""
    trials = inact.load_trials(cohort)
    after = trials[(trials.area == 'wS1') & (trials.inactivation_day == day)]
    both = pd.concat([naive.assign(group=NAIVE), after.assign(group=label)], ignore_index=True)
    sessions = inact.counts(both, ['group', 'mouse_id', 'session_id'])
    blocks = inact.counts(both[both.block_id < N_BLOCKS], ['group', 'mouse_id', 'session_id', 'block_id'])
    blocks['block'] = blocks['block_id'].astype(int) + 1
    return sessions, blocks


def compare(sessions, blocks, label):
    rows = []
    for measure in inact.MEASURES:
        for part, d in [('session', sessions), ('first block', blocks[blocks.block == 1])]:
            a, b = (d.loc[d.group == g, measure] for g in (label, NAIVE))
            rows.append(inact.mwu_row(part, measure, a, b, (label, NAIVE)))
    return pd.DataFrame(rows)


# ============================================================================
# Plots
# ============================================================================


def group_bars(ax, data, stats, y, part, label, colors, title):
    order = [NAIVE, label]
    p = stats.query('measure == @y and day == @part')['p_value'].iloc[0]
    inact.bars(ax, data, 'group', order, 'group', order, colors, y, [])
    ax.text(0.5, 1.0, format_p(p), transform=ax.get_xaxis_transform(), ha='center', va='bottom', fontsize=5)
    ax.set_xticks([0, 1], ['Naive\nDay 0', label.replace(', ', '\n')])
    ax.set_title(title, pad=8)


def block_curves(ax, blocks, y, label, colors):
    for group, color in colors.items():
        g = blocks[blocks.group == group]
        if group == label:
            for _, m in g.groupby('mouse_id'):
                ax.plot(m['block'], m[y], color=color, alpha=0.4, linewidth=0.5)
        sns.lineplot(
            data=g,
            x='block',
            y=y,
            color=color,
            errorbar=('ci', 95) if group == NAIVE else None,
            seed=0,
            ax=ax,
            label=f'{group}, n = {g.mouse_id.nunique()}',
        )
    if y == 'dprime':
        ax.axhline(0, color='grey', linestyle='--', linewidth=0.6)
    ax.set_xticks(range(1, N_BLOCKS + 1))
    ax.set_xlabel('Block (20 trials)')
    ax.set_ylabel(inact.MEASURES[y])
    ax.set_ylim(*inact.YLIMS[y])
    ax.legend(frameon=False, loc='lower center', bbox_to_anchor=(0.5, 1.0))


def plot(results, path):
    set_style()
    fig, axes = plt.subplots(4, 3, figsize=panel_size(3, 4), gridspec_kw={'width_ratios': [2, 3, 2]})
    for i, (name, label, sessions, blocks, stats) in enumerate(results):
        colors = {NAIVE: reward_palette[1], label: 'black'}
        for row, y in zip(axes[2 * i : 2 * i + 2], inact.MEASURES):
            group_bars(row[0], sessions, stats, y, 'session', label, colors, 'Whole session')
            block_curves(row[1], blocks, y, label, colors)
            group_bars(
                row[2], blocks[blocks.block == 1], stats, y, 'first block', label, colors, 'First block'
            )
    inact.save(fig, path)


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    naive = inact.naive_day0_trials()
    results, all_sessions, all_blocks, all_stats = [], [], [], []
    for name, cohort, day, label in COMPARISONS:
        sessions, blocks = tables(cohort, day, label, naive)
        stats = compare(sessions, blocks, label).assign(cohort=name)
        results.append((name, label, sessions, blocks, stats))
        all_sessions.append(sessions.assign(cohort=name))
        all_blocks.append(blocks.assign(cohort=name))
        all_stats.append(stats)
        n_blocks = blocks[blocks.group == label].groupby('mouse_id')['block'].max()
        print(f"{name}: blocks per inactivated mouse {n_blocks.min()}-{n_blocks.max()}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plot(results, os.path.join(OUTPUT_DIR, 'figure2_supfig2_a_b.pdf'))
    pd.concat(all_sessions).to_csv(os.path.join(OUTPUT_DIR, 'figure2_supfig2_sessions_data.csv'), index=False)
    pd.concat(all_blocks).to_csv(os.path.join(OUTPUT_DIR, 'figure2_supfig2_blocks_data.csv'), index=False)
    stats = pd.concat(all_stats)
    stats.to_csv(os.path.join(OUTPUT_DIR, 'figure2_supfig2_stats.csv'), index=False)
    cols = ['cohort', 'measure', 'day', 'n_1', 'n_2', 'median_1', 'median_2', 'p_value']
    print(stats[cols].round(3).to_string(index=False))
