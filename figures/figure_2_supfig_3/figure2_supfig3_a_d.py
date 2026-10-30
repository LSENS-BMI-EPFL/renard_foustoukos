"""
Figure 2—figure supplement 3a-d: Auditory performance during the
inactivation of wS1 or fpS1 in learning

The same quantification as the whisker performance of Figure 2c and 2f,
for the auditory trials of the same sessions:
- a, b: muscimol (Days 0, +1, +2): auditory lick probability and d'
  (auditory vs catch), wS1 vs fpS1;
- c, d: optogenetics (light on Day 0, no light on Day +1): the same.
Auditory d' = Z(auditory hit rate) - Z(catch false-alarm rate), from trial
counts with the log-linear correction. Bars: mean and bootstrapped 95% CI
across mice; dots: mice. Mann-Whitney U per day (n = mice). See
fast_learning.inactivation.

Outputs: <figures_dir>/Figure2_SupFig3/figure2_supfig3_a_d.pdf,
figure2_supfig3_a_d_data.csv (session), figure2_supfig3_a_d_stats.csv.
"""

import os

import matplotlib.pyplot as plt
import pandas as pd

from fast_learning import inactivation as inact
from fast_learning import paths
from fast_learning.plotting import panel_size, set_style

OUTPUT_DIR = os.path.join(paths.manuscript_output_dir, 'Figure2_SupFig3')
# (name, cohort, tested day labels, their tick labels, title), as Figure 2.
COHORTS = [
    ('muscimol', inact.MUSCIMOL_LEARNING, ['M 1', 'M 2', 'M 3'], ['0', '+1', '+2'], 'Muscimol'),
    ('optogenetics', inact.OPTO_LEARNING, ['0', '+1'], ['0', '+1'], 'Optogenetics'),
]


if __name__ == '__main__':
    set_style()
    widths = [2 + len(c[2]) for c in COHORTS for _ in inact.AUDITORY_MEASURES]
    fig, axes = plt.subplots(1, 4, figsize=panel_size(4, w=1.2), gridspec_kw={'width_ratios': widths})
    all_sessions, all_stats = [], []
    for i, (name, cohort, tested, tested_labels, title) in enumerate(COHORTS):
        sessions = inact.session_table(cohort)
        shown = sessions.assign(
            day_label=sessions['day_label'].cat.rename_categories(dict(zip(tested, tested_labels)))
        )
        for j, y in enumerate(inact.AUDITORY_MEASURES):
            stats = inact.area_stats(sessions, y, tested).assign(cohort=name)
            ax = axes[2 * i + j]
            s = stats.replace({'day': dict(zip(tested, tested_labels))})
            inact.area_bars(ax, shown, s, tested_labels, y, title=title, legend=ax is axes[-1])
            if y == 'p_auditory':  # most mice at 100%: room for the p-values
                ax.set_ylim(0, 110)
                ax.set_yticks(range(0, 101, 20))
            ax.set_xlabel('Day')
            all_stats.append(stats)
        all_sessions.append(sessions.assign(cohort=name))
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    inact.save(fig, os.path.join(OUTPUT_DIR, 'figure2_supfig3_a_d.pdf'))

    pd.concat(all_sessions).to_csv(os.path.join(OUTPUT_DIR, 'figure2_supfig3_a_d_data.csv'), index=False)
    stats = pd.concat(all_stats)
    stats.to_csv(os.path.join(OUTPUT_DIR, 'figure2_supfig3_a_d_stats.csv'), index=False)
    cols = ['cohort', 'measure', 'day', 'n_1', 'n_2', 'median_1', 'median_2', 'p_value']
    print(stats[cols].round(4).to_string(index=False))
