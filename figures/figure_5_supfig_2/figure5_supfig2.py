"""
Figure 5—figure supplement 2: Relation of the decoder to Day-0 behaviour,
with performance quantified by d' instead of whisker lick probability

Per-mouse correlation between the decoder decision value and d' across
Day-0 whisker trials (Fig. 5i with d' in place of lick probability);
Wilcoxon signed-rank test against zero (n = mice).

Outputs: <figures_dir>/Figure5_SupFig2/figure5_supfig2.pdf,
figure5_supfig2_correlations.csv, figure5_supfig2_stats.csv.

------------------------------------------------------------------------
How d' is computed
------------------------------------------------------------------------
d' = Z(hit rate) - Z(false-alarm rate), where Z = Phi^-1 is the inverse
standard normal CDF (scipy.stats.norm.ppf).

  - "Hit rate"        = probability of licking on whisker-stimulus trials
                         (the signal).
  - "False-alarm rate"= probability of licking on no-stim/catch trials
                         (the noise). Every session already records catch
                         trials interleaved with whisker/auditory trials
                         (outcome_c in the trial tables), so no new data
                         collection is needed.

d' here is trial-resolved, from the existing Bayesian learning curves
(behavior_imagingmice_table_5days_cut_with_learning_curves.csv): the
smoothed whisker lick-probability curve (learning_curve_w) and the catch curve
interpolated onto whisker-trial indices (learning_curve_chance). These are
posterior means that only approach 0/1, so they are clipped eps = 1e-3 away
from the boundary before the z-transform:
    d'(trial) = Z(clip(learning_curve_w)) - Z(clip(learning_curve_chance))
(fast_learning.stats.dprime_from_rates). Session-level d' from trial counts,
with the log-linear correction, is fast_learning.stats.dprime.

Everything else (grouping, day/mouse filters, bar/swarm plots, Mann-Whitney
U or Wilcoxon tests, output file naming) is kept identical to the original
panel -- only the plotted/tested quantity changes.

Scope: Fig. 5h (slopes of the decoder value) has no behavioural measure, so
it has no d' version. Only the behavioural variable changes; the
decoder is the one of Fig. 5f-i (pipeline/06_decoder.py, decoder_weights.pkl),
applied with the same 10-trial sliding window.
"""

import os
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wilcoxon, ttest_1samp, pearsonr

from fast_learning import paths, decoding
from fast_learning import imaging
from fast_learning.plotting import reward_palette, save_figure
from fast_learning.stats import dprime_from_rates, format_p


OUTPUT_DIR = os.path.join(paths.manuscript_output_dir, 'Figure5_SupFig2')
EPS = 1e-3  # clipping epsilon for already-smoothed probability curves


# ============================================================================
# Shared d' helpers
# ============================================================================


def _load_trial_resolved_dprime(
    table_path=os.path.join(
        paths.processed_dir, 'behavior', 'behavior_imagingmice_table_5days_cut_with_learning_curves.csv'
    ),
    n_trials=120,
):
    """Load the Day-0 whisker-trial table (as figure_1d.py and
    figure_5g.py) and add a trial-resolved d' column, computed from the
    existing Bayesian learning-curve fits:
        dprime_w(trial) = Z(learning_curve_w) - Z(learning_curve_chance)
    where learning_curve_w is the smoothed whisker hit-probability curve
    and learning_curve_chance is the smoothed no-stim/catch false-alarm
    probability curve, already interpolated onto whisker-trial indices
    (utils_behavior.compute_learning_trial)."""
    table_path = paths.adjust_path_to_host(table_path)
    table = pd.read_csv(table_path)

    df = table.loc[(table.whisker_stim == 1) & (table.day == 0)].copy()
    df = df.loc[df.trial_w <= n_trials]

    df['dprime_w'] = dprime_from_rates(df['learning_curve_w'], df['learning_curve_chance'], eps=EPS)

    return df


# ============================================================================
# Panel b: per-mouse correlation between decoder value and d' (Fig. 5i)
# ============================================================================


def panel_b_dprime_correlation(
    table_path=os.path.join(
        paths.processed_dir, 'behavior', 'behavior_imagingmice_table_5days_cut_with_learning_curves.csv'
    ),
    cut_n_trials=100,
    weights_path=os.path.join(paths.processed_dir, 'decoding', 'decoder_weights.pkl'),
    window_size=10,
    step_size=1,
    save_path=OUTPUT_DIR,
    save_format='svg',
    dpi=300,
):
    """Panel b: per-mouse Pearson correlation between the decoder decision
    value and trial-resolved d' (in place of learning_curve_w), with the
    decoder of Fig. 5f-i and the same 10-trial sliding window."""

    win = (0, 0.300)

    df = _load_trial_resolved_dprime(table_path=table_path, n_trials=cut_n_trials)

    with open(weights_path, 'rb') as f:
        weights = pickle.load(f)
    print(f"Loaded decoder weights for {len(weights)} mice.")

    folder = paths.tensor_dir
    results = []
    for mouse, w in weights.items():
        xarr = imaging.load_mouse_xarray(mouse, folder, 'tensor_xarray_learning_data.nc')
        xarr = xarr.sel(trial=xarr['day'].isin([0]))
        xarr = xarr.sel(trial=xarr['whisker_stim'] == 1)
        xarr = imaging.select_time(xarr, *decoding.ACTIVE_WIN).mean(dim='time')
        xarr = xarr.fillna(0)

        scaler, clf, sign_flip = w['scaler'], w['clf'], w['sign_flip']
        n_trials = xarr.sizes['trial']
        for start_idx in range(0, max(0, n_trials - window_size + 1), step_size):
            end_idx = start_idx + window_size
            X_win = xarr.values[:, start_idx:end_idx].T
            if X_win.shape[0] == 0:
                continue
            dec_vals = clf.decision_function(scaler.transform(X_win))
            results.append(
                {
                    'mouse_id': mouse,
                    'reward_group': w['reward_group'],
                    'trial_start': start_idx,
                    'mean_decision_value': np.mean(dec_vals) * sign_flip,
                }
            )
    results_combined = pd.DataFrame(results)

    corr_real, corr_mice, corr_groups = [], [], []
    for mouse in results_combined['mouse_id'].unique():
        group = results_combined.loc[results_combined['mouse_id'] == mouse, 'reward_group'].iloc[0]
        dec_mouse = results_combined[results_combined['mouse_id'] == mouse]
        bh_mouse = df[df['mouse_id'] == mouse]

        common_trials = np.intersect1d(dec_mouse['trial_start'], bh_mouse['trial_w'])
        if len(common_trials) < 10:
            continue

        dec_vals = dec_mouse.set_index('trial_start').loc[common_trials]['mean_decision_value'].values
        perf_vals = bh_mouse.set_index('trial_w').loc[common_trials]['dprime_w'].values

        corr = pearsonr(perf_vals, dec_vals)[0]
        corr_real.append(corr)
        corr_mice.append(mouse)
        corr_groups.append(group)
        print(f"{mouse} ({group}): r={corr:.3f}")

    df_corr = pd.DataFrame({'mouse_id': corr_mice, 'reward_group': corr_groups, 'correlation': corr_real})

    pop_stats, pop_stats_rows = {}, []
    for group in ['R+', 'R-']:
        sub = df_corr[df_corr['reward_group'] == group]
        if len(sub) >= 3:
            _, p_wilcox = wilcoxon(sub['correlation'].values, alternative='greater')
            _, p_ttest = ttest_1samp(sub['correlation'].values, 0, alternative='greater')
            pop_stats[group] = p_wilcox
            pop_stats_rows.append(
                {
                    'reward_group': group,
                    'n': len(sub),
                    'mean_correlation': np.mean(sub['correlation'].values),
                    'std_correlation': np.std(sub['correlation'].values),
                    'p_wilcoxon': p_wilcox,
                    'p_ttest': p_ttest,
                }
            )
            print(
                f"{group} (N={len(sub)}): mean r={np.mean(sub['correlation'].values):.3f}, "
                f"Wilcoxon p={p_wilcox:.4f}"
            )
    df_pop_stats = pd.DataFrame(pop_stats_rows)

    sns.set_theme(context='paper', style='ticks', palette='deep', font='sans-serif', font_scale=1)
    fig, ax = plt.subplots(1, 1, figsize=(4, 5))

    sns.swarmplot(
        data=df_corr,
        x='reward_group',
        y='correlation',
        palette=reward_palette[::-1],
        size=8,
        alpha=0.6,
        ax=ax,
    )
    sns.pointplot(
        data=df_corr,
        x='reward_group',
        y='correlation',
        palette=reward_palette[::-1],
        errorbar='ci',
        markersize=10,
        join=False,
        ax=ax,
        seed=0,
    )
    ax.axhline(0, color='black', linestyle='--', alpha=0.5, linewidth=1)

    for i, group in enumerate(['R+', 'R-']):
        if group in pop_stats:
            p = pop_stats[group]
            y_pos = ax.get_ylim()[1] - 0.1 * (ax.get_ylim()[1] - ax.get_ylim()[0])
            ax.text(i, y_pos, format_p(p), ha='center', va='top', fontsize=9, fontweight='bold')

    ax.set_ylim(-1, 1)
    ax.set_xlabel('Reward group')
    ax.set_ylabel("Pearson r\n(Decision value vs d')")
    sns.despine()
    plt.tight_layout()

    os.makedirs(save_path, exist_ok=True)
    save_figure(fig, os.path.join(save_path, f'figure5_supfig2.{save_format}'))
    plt.close()

    df_corr.to_csv(os.path.join(save_path, 'figure5_supfig2_correlations.csv'), index=False)
    df_pop_stats.to_csv(os.path.join(save_path, 'figure5_supfig2_stats.csv'), index=False)

    print(f"Figure 5i (d' correlation) saved to: {save_path}")


# ============================================================================
# Main execution
# ============================================================================

if __name__ == '__main__':
    panel_b_dprime_correlation()
