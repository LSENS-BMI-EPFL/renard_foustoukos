"""
Figure 6c: Reactivation rate across days, R+ vs R-

Grouped bar plot (one bar per group per day) showing reactivation event
frequency (events/min) across the 5 experimental days. Individual mouse
trajectories are overlaid. Mann-Whitney U test per day (R+ vs R-).

Mice: all imaging mice. The participation mouse selection
(fast_learning.reactivations) is deliberately not applied here: a day with
zero events is a valid event-rate measurement.

Inputs:  <processed_dir>/reactivation/reactivation_results_p*.pkl (step 07).
Outputs: main analysis (99th surrogate percentile, whole trial):
         <figures_dir>/Figure6/figure_6c.pdf, _data.csv, _stats.csv. Other
         conditions (p995, p999; --nolick: lick-free control), in
         Figure6/variants/: figure_6c_<p><_nolick>.
"""

import argparse
import os
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu

from fast_learning import paths, reactivations as rx
from fast_learning.plotting import VARIANTS_DIR, reward_palette, save_figure, set_style, panel_size
from fast_learning.stats import significance_stars as _significance_stars
from fast_learning.stats import format_p


# ============================================================================
# Parameters
# ============================================================================

DAYS = [-2, -1, 0, 1, 2]

OUTPUT_DIR = os.path.join(paths.manuscript_output_dir, 'Figure6')


RESULTS_DIR = os.path.join(paths.processed_dir, 'reactivation')
PERCENTILES = ['p99', 'p995', 'p999']


# ============================================================================
# Helpers
# ============================================================================


# ============================================================================
# Panel h
# ============================================================================


def panel_h_reactivation_rate(
    r_plus_results,
    r_minus_results,
    days=DAYS,
    output_dir=OUTPUT_DIR,
    filename='figure_6c',
    save_format='svg',
    dpi=300,
):
    """
    Generate Figure 6 Panel c: reactivation frequency per day, R+ vs R-.

    Args:
        filename: Output filename stem (without extension).

    Saves:
        <filename>_data.csv: mouse_id, reward_group, day, event_frequency
        <filename>_stats.csv: Mann-Whitney U results per day
    """
    set_style()

    days_sorted = sorted(days)

    # Build long-format DataFrame
    data_list = []
    for mouse, results in r_plus_results.items():
        for day in days_sorted:
            if day in results['days']:
                data_list.append(
                    {
                        'mouse_id': mouse,
                        'reward_group': 'R+',
                        'Day': day,
                        'Frequency': results['days'][day]['event_frequency'],
                    }
                )
    for mouse, results in r_minus_results.items():
        for day in days_sorted:
            if day in results['days']:
                data_list.append(
                    {
                        'mouse_id': mouse,
                        'reward_group': 'R-',
                        'Day': day,
                        'Frequency': results['days'][day]['event_frequency'],
                    }
                )

    df = pd.DataFrame(data_list)

    # Statistics: Mann-Whitney U per day
    stats_rows = []
    p_values = []
    for day in days_sorted:
        r_plus_vals = df[(df['Day'] == day) & (df['reward_group'] == 'R+')]['Frequency'].values
        r_minus_vals = df[(df['Day'] == day) & (df['reward_group'] == 'R-')]['Frequency'].values
        if len(r_plus_vals) > 0 and len(r_minus_vals) > 0:
            stat, p = mannwhitneyu(r_plus_vals, r_minus_vals, alternative='two-sided')
        else:
            stat, p = np.nan, 1.0
        p_values.append(p)
        stats_rows.append(
            {
                'test': 'Mann-Whitney U',
                'day': day,
                'R+_n': len(r_plus_vals),
                'R-_n': len(r_minus_vals),
                'R+_mean': np.nanmean(r_plus_vals),
                'R-_mean': np.nanmean(r_minus_vals),
                'statistic': stat,
                'p_value': p,
                'significance': _significance_stars(p),
            }
        )

    # Add dummy rows so all days appear even if data is missing
    for day in days_sorted:
        for group in ['R+', 'R-']:
            if day not in df[df['reward_group'] == group]['Day'].values:
                df = pd.concat(
                    [
                        df,
                        pd.DataFrame(
                            {'mouse_id': [''], 'reward_group': [group], 'Day': [day], 'Frequency': [np.nan]}
                        ),
                    ],
                    ignore_index=True,
                )

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=panel_size(w=1.6))

    sns.barplot(
        data=df,
        x='Day',
        y='Frequency',
        hue='reward_group',
        errorbar=('ci', 95),
        palette={'R+': reward_palette[1], 'R-': reward_palette[0]},
        hue_order=['R+', 'R-'],
        alpha=0.7,
        edgecolor='black',
        ax=ax,
        seed=0,
    )

    # # Individual mouse trajectories
    # bar_width = 0.35
    # group_offsets = {'R+': -bar_width / 2, 'R-': bar_width / 2}
    # x_positions = {day: idx for idx, day in enumerate(days_sorted)}

    # for mouse in df['mouse_id'].unique():
    #     if mouse == '':
    #         continue
    #     mouse_data = df[df['mouse_id'] == mouse].sort_values('Day')
    #     group = mouse_data['reward_group'].iloc[0]
    #     mouse_x = [x_positions[d] + group_offsets[group] for d in mouse_data['Day']]
    #     mouse_y = mouse_data['Frequency'].values
    #     color = reward_palette[1] if group == 'R+' else reward_palette[0]
    #     ax.plot(mouse_x, mouse_y, '-', color=color, linewidth=0.8, alpha=0.5, zorder=5)

    # p-value above each day
    y_max = df['Frequency'].max()
    y_range = y_max * 0.05
    width = 0.35

    for day_idx, (day, p) in enumerate(zip(days_sorted, p_values)):
        if not np.isnan(p):
            r_plus_vals = df[(df['Day'] == day) & (df['reward_group'] == 'R+') & (df['mouse_id'] != '')][
                'Frequency'
            ]
            r_minus_vals = df[(df['Day'] == day) & (df['reward_group'] == 'R-') & (df['mouse_id'] != '')][
                'Frequency'
            ]
            ci_plus = (
                r_plus_vals.mean() + 1.96 * r_plus_vals.std() / np.sqrt(len(r_plus_vals))
                if len(r_plus_vals) > 0
                else 0
            )
            ci_minus = (
                r_minus_vals.mean() + 1.96 * r_minus_vals.std() / np.sqrt(len(r_minus_vals))
                if len(r_minus_vals) > 0
                else 0
            )
            y1 = max(ci_plus, ci_minus)
            y2 = y1 + y_range
            x1 = day_idx - width / 2
            x2 = day_idx + width / 2
            ax.plot([x1, x1, x2, x2], [y1, y2, y2, y1], 'k-', linewidth=1)
            ax.text((x1 + x2) / 2, y2, format_p(p), ha='center', va='bottom')

    ax.set_xlabel('Day')
    ax.set_ylabel('Reactivation rate (events/min)')
    ax.legend(title='')
    ax.set_ylim(0, 20)
    sns.despine()
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    save_figure(plt.gcf(), os.path.join(output_dir, f'{filename}.{save_format}'))
    plt.close()
    print(f"Figure saved to: {os.path.join(output_dir, filename + '.' + save_format)}")

    # Save CSVs
    df[df['mouse_id'] != ''].to_csv(os.path.join(output_dir, f'{filename}_data.csv'), index=False)
    pd.DataFrame(stats_rows).to_csv(os.path.join(output_dir, f'{filename}_stats.csv'), index=False)
    print(f"Data/stats saved to: {output_dir}")


# ============================================================================
# Main execution
# ============================================================================


def _load_and_plot(results_file, filename, output_dir=OUTPUT_DIR):
    print(f"Loading reactivation results from: {results_file}")
    if not os.path.exists(results_file):
        raise FileNotFoundError(
            f"Results file not found: {results_file}\nRun pipeline/07_reactivations.py first."
        )

    with open(results_file, 'rb') as f:
        results_data = pickle.load(f)

    r_plus_results = results_data['r_plus_results']
    r_minus_results = results_data['r_minus_results']
    print(f"Loaded results for {len(r_plus_results)} R+ mice and {len(r_minus_results)} R- mice")

    panel_h_reactivation_rate(r_plus_results, r_minus_results, output_dir=output_dir, filename=filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--nolick', action='store_true', help='no-lick control (-1 to +1 s), written to Figure6/variants'
    )
    nolick = parser.parse_args().nolick
    sfx = '_nolick' if nolick else ''
    for pstr in PERCENTILES:
        results_file = os.path.join(rx.results_dir(nolick), f'reactivation_results_{pstr}.pkl')
        if not os.path.exists(results_file) and pstr != 'p99':
            print(f'{results_file} not found, skipped.')
            continue
        # p99 on the whole trial is the main analysis (Figure6/figure_6c); the
        # other percentiles and the no-lick control go to Figure6/variants.
        if pstr == 'p99' and not nolick:
            _load_and_plot(results_file, filename='figure_6c')
        else:
            filename = ('figure_6c' if pstr == 'p99' else f'figure_6c_{pstr}') + sfx
            _load_and_plot(results_file, filename=filename, output_dir=os.path.join(OUTPUT_DIR, VARIANTS_DIR))
