"""
Diagnostic: correlate each cell's day-N whisker-response template value
(the per-cell vector used by reactivation_preprocessing.py's
create_whisker_template() to detect reactivation events) against that
cell's LMI.

Motivation: template construction (reactivation_preprocessing.py,
create_whisker_template) and LMI (stats_on_tensors.py, "Compute LMI"
section) are both derived from mapping-trial dF/F magnitude. For days
-2/-1/+1/+2, the last-40-trial subset used to build that day's template
is literally contained in LMI's pre ([-2,-1]) or post ([+1,+2]) trial
pool. Day 0 mapping trials are excluded from LMI entirely, so day 0 has
no trial overlap and serves as a partial negative control: if template
vs LMI correlation is much weaker on day 0 than on +/-1/+/-2, that
directly implicates the trial-overlap mechanism.

This is a QC/methods-check script, not a manuscript figure panel.
Outputs go to a separate 'diagnostics' subfolder.
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, linregress

sys.path.append('/home/aprenard/repos/fast-learning')
from fast_learning import paths
from fast_learning import imaging
from fast_learning.plotting import reward_palette
from fast_learning import database
from fast_learning.reactivations import create_whisker_template

_, _, _mice, _db = database.select_sessions_from_db(paths.db_path, paths.nwb_dir, two_p_imaging='yes')
_groups = {m: database.get_mouse_reward_group_from_db(paths.db_path, m, db=_db) for m in _mice}
r_plus_mice = [m for m in _mice if _groups[m] == 'R+']
r_minus_mice = [m for m in _mice if _groups[m] == 'R-']


# ============================================================================
# Parameters
# ============================================================================

DAYS_TO_CHECK = [-2, -1, 0, 1, 2]
LMI_RESULTS_CSV = os.path.join(paths.processed_dir, 'lmi_results.csv')
FOLDER = paths.tensor_dir
OUTPUT_DIR = os.path.join(paths.manuscript_output_dir, 'figure_4', 'diagnostics')


def _significance_stars(p):
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    return 'n.s.'


def _day_tag(day):
    return f'day{"p" if day >= 0 else "m"}{abs(day)}'


# ============================================================================
# Data collection
# ============================================================================

def collect_template_lmi_data(days, lmi_df):
    """Build a per-cell, per-day DataFrame of (template_value, lmi).

    Reuses create_whisker_template() unmodified so the correlation is
    against the exact template used by the real detection pipeline.
    """
    reward_map = {m: 'R+' for m in r_plus_mice}
    reward_map.update({m: 'R-' for m in r_minus_mice})

    rows = []
    for mouse in r_plus_mice + r_minus_mice:
        try:
            xarr = imaging.load_mouse_xarray(
                mouse, FOLDER, 'tensor_xarray_mapping_data.nc', subtracted=True)
        except Exception as e:
            print(f"  Skipping {mouse}: could not load mapping xarray ({e})")
            continue
        roi_ids = xarr['roi'].values
        mouse_lmi = lmi_df[lmi_df['mouse_id'] == mouse].set_index('roi')

        for day in days:
            try:
                template, _ = create_whisker_template(mouse, day, verbose=False)
            except Exception as e:
                print(f"  Skipping {mouse} day {day}: {e}")
                continue
            for roi, tmpl_val in zip(roi_ids, template):
                if roi in mouse_lmi.index:
                    rows.append({
                        'mouse_id': mouse,
                        'reward_group': reward_map.get(mouse, 'unknown'),
                        'roi': roi,
                        'day': day,
                        'template_value': float(tmpl_val),
                        'lmi': mouse_lmi.loc[roi, 'lmi'],
                    })

    return pd.DataFrame(rows)


# ============================================================================
# Plot + stats
# ============================================================================

def plot_template_vs_lmi(df, day, output_dir=OUTPUT_DIR, save_format='svg', dpi=300):
    """Scatter template_value vs lmi for one day, one subplot per reward group."""
    sns.set_theme(context='paper', style='ticks', palette='deep',
                  font='sans-serif', font_scale=1)

    reward_groups = ['R+', 'R-']
    rg_colors = {'R+': reward_palette[1], 'R-': reward_palette[0]}
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    stats_rows = []

    day_df = df[df['day'] == day]

    for i, rg in enumerate(reward_groups):
        ax = axes[i]
        grp = day_df[day_df['reward_group'] == rg].dropna(subset=['lmi', 'template_value'])
        x = grp['lmi'].values
        y = grp['template_value'].values

        ax.scatter(x, y, color=rg_colors[rg], s=4, alpha=0.4, linewidths=0,
                   rasterized=True)

        if len(x) >= 3:
            slope, intercept, _, _, se = linregress(x, y)
            pearson_r, pearson_p = pearsonr(x, y)
            x_line = np.linspace(x.min(), x.max(), 200)
            ax.plot(x_line, slope * x_line + intercept, color='black',
                    linewidth=1.2, zorder=5)
            stars = _significance_stars(pearson_p)
            ax.text(0.05, 0.95, f'r = {pearson_r:.3f}\np = {pearson_p:.3g} {stars}',
                    transform=ax.transAxes, va='top', ha='left', fontsize=8)
            stats_rows.append({
                'day': day, 'reward_group': rg, 'n_cells': len(x),
                'n_mice': grp['mouse_id'].nunique(),
                'pearson_r': pearson_r, 'p_value': pearson_p,
                'significance': stars, 'slope': slope, 'intercept': intercept,
                'stderr': se,
            })
        else:
            stats_rows.append({
                'day': day, 'reward_group': rg, 'n_cells': len(x),
                'n_mice': grp['mouse_id'].nunique(),
                'pearson_r': np.nan, 'p_value': np.nan,
                'significance': 'n.a.', 'slope': np.nan, 'intercept': np.nan,
                'stderr': np.nan,
            })

        ax.axvline(x=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
        ax.set_title(f'{rg}  (n = {len(grp)} cells, {grp["mouse_id"].nunique()} mice)',
                     fontsize=10, fontweight='bold')
        ax.set_xlabel('LMI', fontsize=9)
        ax.set_ylabel(f'Day {day:+d} template value (dF/F)' if i == 0 else '', fontsize=9)
        ax.tick_params(labelsize=8)
        sns.despine(ax=ax)

    fig.suptitle(f'Reactivation template (day {day:+d}) vs LMI', fontsize=11)
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    filename = f'template_vs_lmi_{_day_tag(day)}'
    plt.savefig(os.path.join(output_dir, f'{filename}.{save_format}'),
                format=save_format, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"Saved: {os.path.join(output_dir, filename + '.' + save_format)}")

    return pd.DataFrame(stats_rows)


# ============================================================================
# Main execution
# ============================================================================

if __name__ == '__main__':
    print(f"Days checked: {DAYS_TO_CHECK}")
    print(f"LMI results: {LMI_RESULTS_CSV}")
    print(f"Output directory: {OUTPUT_DIR}")

    lmi_df = pd.read_csv(LMI_RESULTS_CSV)
    data_df = collect_template_lmi_data(DAYS_TO_CHECK, lmi_df)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    data_df.to_csv(os.path.join(OUTPUT_DIR, 'template_vs_lmi_data.csv'), index=False)

    all_stats = []
    for day in DAYS_TO_CHECK:
        stats_df = plot_template_vs_lmi(data_df, day)
        all_stats.append(stats_df)

    summary = pd.concat(all_stats, ignore_index=True)
    summary.to_csv(os.path.join(OUTPUT_DIR, 'template_vs_lmi_summary_stats.csv'), index=False)
    print("\n=== Summary: template value vs LMI, per day and reward group ===")
    print(summary.to_string(index=False))
    print("\nIf |r| on days -2/-1/+1/+2 is clearly larger than on day 0, that "
          "directly implicates the shared-trial overlap between template "
          "construction and LMI as a driver of the panel J result.")
