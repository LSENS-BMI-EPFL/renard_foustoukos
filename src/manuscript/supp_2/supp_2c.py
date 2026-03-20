"""
Supplementary Figure 2c: Relationship between classifier weights and the
Learning Modulation Index (LMI).

Scatter plot (one dot per cell, all mice pooled) with a global linear
regression line and bootstrapped 95% CI. Classifier weights are loaded from
the CSV produced by figure_3m_o.py.
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.utils import resample

sys.path.append(r'/home/aprenard/repos/fast-learning')
import src.utils.utils_io as io


# ============================================================================
# Parameters
# ============================================================================

N_BOOT = 1000
RESULTS_DIR = os.path.join(io.processed_dir, 'decoding')
OUTPUT_DIR  = os.path.join(io.manuscript_output_dir, 'supp_2', 'output')


# ============================================================================
# Load and merge data
# ============================================================================

weights_df = pd.read_csv(os.path.join(RESULTS_DIR, 'classifier_weights.csv'))
lmi_df     = pd.read_csv(os.path.join(io.processed_dir, 'lmi_results.csv'))

merged = weights_df.merge(lmi_df[['mouse_id', 'roi', 'lmi']], on=['mouse_id', 'roi'], how='inner')
merged = merged.dropna(subset=['lmi', 'classifier_weight'])

print(f"Cells for analysis: {len(merged)} across {merged['mouse_id'].nunique()} mice")


# ============================================================================
# Linear regression + bootstrap CI
# ============================================================================

lmi_flat = merged['lmi'].values
w_flat   = merged['classifier_weight'].values

X = lmi_flat.reshape(-1, 1)
reg = LinearRegression().fit(X, w_flat)
r2  = reg.score(X, w_flat)
r_pearson, p_pearson = pearsonr(lmi_flat, w_flat)
print(f"R² = {r2:.3f}  |  r = {r_pearson:.3f}  |  p = {p_pearson:.2e}"
      f"  |  slope = {reg.coef_[0]:.4f}  |  intercept = {reg.intercept_:.4f}")

x_vals = np.linspace(lmi_flat.min(), lmi_flat.max(), 200)
y_pred = reg.predict(x_vals.reshape(-1, 1))

# Bootstrap CI for regression line
y_boot = np.zeros((N_BOOT, len(x_vals)))
r2_boot = []
for i in range(N_BOOT):
    Xb, yb = resample(lmi_flat, w_flat)
    rb = LinearRegression().fit(Xb.reshape(-1, 1), yb)
    y_boot[i] = rb.predict(x_vals.reshape(-1, 1))
    r2_boot.append(rb.score(Xb.reshape(-1, 1), yb))

ci_low  = np.percentile(y_boot, 2.5,  axis=0)
ci_high = np.percentile(y_boot, 97.5, axis=0)
r2_ci   = np.percentile(r2_boot, [2.5, 97.5])
print(f"Bootstrapped R² 95% CI: [{r2_ci[0]:.3f}, {r2_ci[1]:.3f}]")


# ============================================================================
# Figure
# ============================================================================

sns.set_theme(context='paper', style='ticks', font='sans-serif', font_scale=1,
              rc={'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none'})

fig, ax = plt.subplots(figsize=(4, 4))

for mouse in merged['mouse_id'].unique():
    sub = merged[merged['mouse_id'] == mouse]
    ax.scatter(sub['lmi'], sub['classifier_weight'], alpha=0.3, s=8, linewidths=0)

ax.plot(x_vals, y_pred, color='#2d2d2d', linewidth=2)
ax.fill_between(x_vals, ci_low, ci_high, color='black', alpha=0.2, label='95% CI')

p_str = f'p = {p_pearson:.2e}' if p_pearson >= 1e-4 else f'p < 0.0001'
ax.text(0.05, 0.95, f'r = {r_pearson:.3f}\n{p_str}',
        transform=ax.transAxes, va='top', ha='left', fontsize=9)

ax.set_xlabel('Learning Modulation Index (LMI)')
ax.set_ylabel('Classifier weight')
ax.set_ylim(-2.5, 2)
sns.despine()
plt.tight_layout()


# ============================================================================
# Save
# ============================================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

fig.savefig(os.path.join(OUTPUT_DIR, 'supp_2c.svg'), format='svg', dpi=300, bbox_inches='tight')
print("Saved: supp_2c.svg")

merged[['mouse_id', 'roi', 'reward_group', 'lmi', 'classifier_weight']].to_csv(
    os.path.join(OUTPUT_DIR, 'supp_2c_data.csv'), index=False)
print("Saved: supp_2c_data.csv")

pd.DataFrame([{
    'r': r_pearson,
    'p_value': p_pearson,
    'r2': r2,
    'r2_ci_low': r2_ci[0],
    'r2_ci_high': r2_ci[1],
    'slope': reg.coef_[0],
    'intercept': reg.intercept_,
    'n_cells': len(merged),
    'n_mice': merged['mouse_id'].nunique(),
    'n_boot': N_BOOT,
}]).to_csv(os.path.join(OUTPUT_DIR, 'supp_2c_stats.csv'), index=False)
print("Saved: supp_2c_stats.csv")

plt.show()
