"""
Figure 1d: Day 0 performance across whisker trials, R+ vs R-

One figure, two plots, aligned on whisker trial number:
- left: fitted learning curve (probability of licking on whisker trials);
- right: d' across trials, Z(whisker learning curve) - Z(catch learning curve
  interpolated onto whisker trials), from the same fits
  (fast_learning.stats.dprime_from_rates).
Lines: mean across mice with 95% CI. Per-trial Mann-Whitney U test R+ vs R-,
FDR-corrected (Benjamini-Hochberg) over trials; p-values shown as a grey
scale above the curves (white: p > 0.05), with a shared colour bar.

The raw single-trial outcomes are in Figure 1—figure supplement 1d.

Input: behavior_imagingmice_table_5days_cut_with_learning_curves.csv.
Outputs: <figures_dir>/Figure1/figure_1d.pdf, figure_1d_learning_data.csv,
figure_1d_learning_stats.csv, figure_1d_dprime_data.csv, figure_1d_dprime_stats.csv.
"""

import os

import matplotlib.colors
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

from fast_learning import paths
from fast_learning.plotting import panel_size, reward_palette, save_figure, set_style
from fast_learning.stats import dprime_from_rates

OUTPUT_DIR = os.path.join(paths.manuscript_output_dir, 'Figure1')
TABLE_PATH = os.path.join(
    paths.processed_dir, 'behavior', 'behavior_imagingmice_table_5days_cut_with_learning_curves.csv'
)
N_TRIALS = 120
XLIM = (0, 100)  # whisker trials shown

# p-value colour code: p > 0.05 white; p <= 0.05 a log10 gradient from a
# visible grey at p = 0.05 down to black at PVALUE_FLOOR and below.
PVALUE_FLOOR = 1e-3
NONSIG_GREY = 0.8
CMAP = matplotlib.colors.LinearSegmentedColormap.from_list(
    'pval_cmap', [(0.0, 'black'), (1.0, (NONSIG_GREY,) * 3)]
)
CMAP.set_over('white')
NORM = matplotlib.colors.LogNorm(vmin=PVALUE_FLOOR, vmax=0.05)


def load_day0(table_path=TABLE_PATH, n_trials=N_TRIALS):
    table = pd.read_csv(paths.adjust_path_to_host(table_path))
    df = table.loc[(table.whisker_stim == 1) & (table.day == 0) & (table.trial_w <= n_trials)].copy()
    df['dprime'] = dprime_from_rates(df['learning_curve_w'], df['learning_curve_chance'])
    return df


def trial_pvalues(df, y):
    """Per-trial Mann-Whitney U, R+ vs R-, FDR-corrected over trials."""
    rows = []
    for trial_w in sorted(df['trial_w'].unique()):
        d = df[df['trial_w'] == trial_w]
        rp, rm = d.loc[d.reward_group == 'R+', y], d.loc[d.reward_group == 'R-', y]
        if len(rp) and len(rm):
            rows.append((trial_w, mannwhitneyu(rp, rm, alternative='two-sided').pvalue))
    trials, p = zip(*rows)
    _, p_fdr, _, _ = multipletests(p, alpha=0.05, method='fdr_bh')
    return pd.DataFrame({'trial_w': trials, 'p_value_raw': p, 'p_value': p_fdr})


def curve_panel(ax, df, y, stats, ylabel, ylim, legend=False, zero_line=False):
    sns.lineplot(
        data=df,
        x='trial_w',
        y=y,
        hue='reward_group',
        hue_order=['R+', 'R-'],
        palette=reward_palette[::-1],
        errorbar='ci',
        err_style='band',
        legend=legend,
        ax=ax,
        seed=0,
    )
    height = 0.03 * (ylim[1] - ylim[0])
    for trial, p in zip(stats['trial_w'], stats['p_value']):
        ax.add_patch(
            plt.Rectangle(
                (trial - 0.4, ylim[1] - 1.5 * height),
                0.8,
                height,
                facecolor=CMAP(NORM(max(p, PVALUE_FLOOR))),
                edgecolor='none',
            )
        )
    if zero_line:
        ax.axhline(0, color='grey', linestyle='--', linewidth=0.6)
    ax.set_ylim(*ylim)
    ax.set_xlim(*XLIM)
    ax.set_xlabel('Whisker trial')
    ax.set_ylabel(ylabel)
    if legend:
        ax.legend(frameon=False, title='', loc='lower left')


def generate_figure(save_path=OUTPUT_DIR):
    df = load_day0()
    learning_stats = trial_pvalues(df, 'learning_curve_w')
    dprime_stats = trial_pvalues(df, 'dprime')

    set_style()
    fig, axes = plt.subplots(1, 2, figsize=panel_size(2, w=1.2))
    fig.subplots_adjust(wspace=0.45)
    curve_panel(axes[0], df, 'learning_curve_w', learning_stats, 'Lick probability', (-0.1, 1))
    curve_panel(
        axes[1], df, 'dprime', dprime_stats, "d' (whisker vs catch)", (-1, 3), legend=True, zero_line=True
    )

    sm = plt.cm.ScalarMappable(norm=NORM, cmap=CMAP)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, orientation='vertical', shrink=0.6, pad=0.02, aspect=15)
    cbar.set_ticks([PVALUE_FLOOR, 1e-2, 0.05])
    cbar.set_ticklabels([f'≤ {PVALUE_FLOOR:.0e}', '0.01', '0.05'])
    cbar.set_label('p-value (FDR-corrected)')
    cbar.ax.invert_yaxis()
    sns.despine()

    os.makedirs(save_path, exist_ok=True)
    save_figure(fig, os.path.join(save_path, 'figure_1d.pdf'))
    plt.close(fig)
    cols = ['mouse_id', 'session_id', 'reward_group', 'trial_w']
    df[cols + ['learning_curve_w']].to_csv(
        os.path.join(save_path, 'figure_1d_learning_data.csv'), index=False
    )
    learning_stats.to_csv(os.path.join(save_path, 'figure_1d_learning_stats.csv'), index=False)
    df[cols + ['learning_curve_w', 'learning_curve_chance', 'dprime']].to_csv(
        os.path.join(save_path, 'figure_1d_dprime_data.csv'), index=False
    )
    dprime_stats.to_csv(os.path.join(save_path, 'figure_1d_dprime_stats.csv'), index=False)
    for name, s in [('learning curve', learning_stats), ("d'", dprime_stats)]:
        sig = s.loc[s['p_value'] <= 0.05, 'trial_w']
        print(
            f"{name}: first FDR-significant trial {sig.min() if len(sig) else None}, {len(sig)} of {len(s)} trials"
        )


if __name__ == '__main__':
    generate_figure()
