"""
Figure 5—figure supplement 1a: Relationship between classifier weights and the
Learning Modulation Index (LMI).

Scatter plot (one dot per cell, all mice pooled) with the pooled linear
regression line and bootstrapped 95% CI, for illustration. Stats with mice
as the unit: one regression slope of classifier weight vs LMI per mouse,
then a Wilcoxon signed-rank test of the slopes against zero. Classifier
weights are loaded from the CSV produced by pipeline/06_decoder.py.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.utils import resample

from fast_learning import paths
from fast_learning.plotting import save_figure, set_style, panel_size
from fast_learning.stats import format_p, per_mouse_slope_test


# ============================================================================
# Parameters
# ============================================================================

N_BOOT = 1000
SEED = 42
RESULTS_DIR = os.path.join(paths.processed_dir, 'decoding')
OUTPUT_DIR = os.path.join(paths.manuscript_output_dir, 'Figure5_SupFig1')


if __name__ == '__main__':
    # ============================================================================
    # Load and merge data
    # ============================================================================

    weights_df = pd.read_csv(os.path.join(RESULTS_DIR, 'classifier_weights.csv'))
    lmi_df = pd.read_csv(os.path.join(paths.processed_dir, 'lmi_results.csv'))

    merged = weights_df.merge(lmi_df[['mouse_id', 'roi', 'lmi']], on=['mouse_id', 'roi'], how='inner')
    merged = merged.dropna(subset=['lmi', 'classifier_weight'])

    print(f"Cells for analysis: {len(merged)} across {merged['mouse_id'].nunique()} mice")

    # ============================================================================
    # Linear regression + bootstrap CI
    # ============================================================================

    lmi_flat = merged['lmi'].values
    w_flat = merged['classifier_weight'].values

    X = lmi_flat.reshape(-1, 1)
    reg = LinearRegression().fit(X, w_flat)
    r2 = reg.score(X, w_flat)
    r_pearson, p_pearson = pearsonr(lmi_flat, w_flat)
    print(
        f"R² = {r2:.3f}  |  r = {r_pearson:.3f}  |  p = {p_pearson:.2e}"
        f"  |  slope = {reg.coef_[0]:.4f}  |  intercept = {reg.intercept_:.4f}"
    )

    # Stats: per-mouse slopes tested against zero (n = mice).
    slope_test = per_mouse_slope_test(merged, y='classifier_weight', x='lmi')
    print(
        f"Per-mouse slopes: median = {slope_test['median_slope']:.4f}, "
        f"Wilcoxon p = {slope_test['p_value']:.3g}, n = {slope_test['n_mice']} mice"
    )

    x_vals = np.linspace(lmi_flat.min(), lmi_flat.max(), 200)
    y_pred = reg.predict(x_vals.reshape(-1, 1))

    # Bootstrap CI for regression line
    y_boot = np.zeros((N_BOOT, len(x_vals)))
    r2_boot = []
    for i in range(N_BOOT):
        Xb, yb = resample(lmi_flat, w_flat, random_state=SEED + i)
        rb = LinearRegression().fit(Xb.reshape(-1, 1), yb)
        y_boot[i] = rb.predict(x_vals.reshape(-1, 1))
        r2_boot.append(rb.score(Xb.reshape(-1, 1), yb))

    ci_low = np.percentile(y_boot, 2.5, axis=0)
    ci_high = np.percentile(y_boot, 97.5, axis=0)
    r2_ci = np.percentile(r2_boot, [2.5, 97.5])
    print(f"Bootstrapped R² 95% CI: [{r2_ci[0]:.3f}, {r2_ci[1]:.3f}]")

    # ============================================================================
    # Figure
    # ============================================================================

    set_style()

    fig, ax = plt.subplots(figsize=panel_size())

    for mouse in merged['mouse_id'].unique():
        sub = merged[merged['mouse_id'] == mouse]
        ax.scatter(sub['lmi'], sub['classifier_weight'], alpha=0.3, s=2, linewidths=0)

    ax.plot(x_vals, y_pred, color='#2d2d2d', linewidth=1)
    ax.fill_between(x_vals, ci_low, ci_high, color='black', alpha=0.2, label='95% CI')

    ax.text(
        0.05,
        0.95,
        f"Per-mouse slopes vs 0: {format_p(slope_test['p_value'])}\n(n = {slope_test['n_mice']} mice)",
        transform=ax.transAxes,
        va='top',
        ha='left',
    )

    ax.set_xlabel('Learning Modulation Index (LMI)')
    ax.set_ylabel('Classifier weight')
    ax.set_ylim(-2.5, 2)
    sns.despine()
    plt.tight_layout()

    # ============================================================================
    # Save
    # ============================================================================

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    save_figure(fig, os.path.join(OUTPUT_DIR, 'figure5_supfig1_a.svg'))
    print("Saved: figure5_supfig1_a.svg")

    merged[['mouse_id', 'roi', 'reward_group', 'lmi', 'classifier_weight']].to_csv(
        os.path.join(OUTPUT_DIR, 'figure5_supfig1_a_data.csv'), index=False
    )
    print("Saved: figure5_supfig1_a_data.csv")

    pd.DataFrame(
        [
            {
                'test': 'Per-mouse slope of weight vs LMI, Wilcoxon signed-rank (n = mice)',
                'median_slope': slope_test['median_slope'],
                'mean_slope': slope_test['mean_slope'],
                'w_stat': slope_test['w_stat'],
                'p_value': slope_test['p_value'],
                'pooled_r': r_pearson,
                'pooled_p_cells': p_pearson,
                'r2': r2,
                'r2_ci_low': r2_ci[0],
                'r2_ci_high': r2_ci[1],
                'slope': reg.coef_[0],
                'intercept': reg.intercept_,
                'n_cells': len(merged),
                'n_mice': merged['mouse_id'].nunique(),
                'n_boot': N_BOOT,
            }
        ]
    ).to_csv(os.path.join(OUTPUT_DIR, 'figure5_supfig1_a_stats.csv'), index=False)
    print("Saved: figure5_supfig1_a_stats.csv")

    plt.close()
