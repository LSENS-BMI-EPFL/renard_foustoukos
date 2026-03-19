"""
Supplementary Figure 2d: Decoding accuracy vs percentage of most-modulated
cells retained.

Cells are ranked by |LMI| in descending order and progressively removed from
the top. At each step, a 10-fold stratified cross-validated logistic regression
is re-run to classify pre vs post mapping trials. The curve shows that
decoding relies on LMI-modulated cells.
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import bootstrap
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

sys.path.append(r'/home/aprenard/repos/fast-learning')
sys.path.append(r'/home/aprenard/repos/NWB_analysis')
import src.utils.utils_io as io
from src.utils.utils_plot import reward_palette

# Import shared data-loading function from figure_3m_o
sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                                '..', '..', 'manuscript', 'figure_3'))
from figure_3m_o import load_and_process_data


# ============================================================================
# Parameters
# ============================================================================

PERCENTILES  = np.arange(0, 91, 5)   # % of most-modulated cells to remove
N_SPLITS     = 10
SEED         = 42
OUTPUT_DIR   = os.path.join(io.manuscript_output_dir, 'supp_2', 'output')


# ============================================================================
# Load mapping vectors and LMI
# ============================================================================

vectors_rew, vectors_nonrew, mice_rew, mice_nonrew = load_and_process_data()

lmi_df = pd.read_csv(os.path.join(io.processed_dir, 'lmi_results.csv'))

le = LabelEncoder()
le.fit(['pre', 'post'])


# ============================================================================
# Compute accuracy curve
# ============================================================================

accs_rew_curve    = []
accs_nonrew_curve = []

for perc in PERCENTILES:
    accs_rew_perc    = []
    accs_nonrew_perc = []

    for group, vectors, mice_group in [
        ('R+', vectors_rew,    mice_rew),
        ('R-', vectors_nonrew, mice_nonrew),
    ]:
        for i, mouse in enumerate(mice_group):
            d = vectors[i]
            cell_ids = d['roi'].values if 'roi' in d.coords else d['cell'].values

            lmi_mouse = (lmi_df[(lmi_df['mouse_id'] == mouse) &
                                (lmi_df['roi'].isin(cell_ids))]
                         .set_index('roi')
                         .reindex(cell_ids))
            abs_lmi = np.abs(lmi_mouse['lmi'].values)

            sorted_idx = np.argsort(-abs_lmi)   # descending |LMI|
            n_remove   = int(np.round(len(cell_ids) * perc / 100))
            keep_idx   = sorted_idx[n_remove:]

            if len(keep_idx) < 2:
                continue

            d_sub = d.isel({d.dims[0]: keep_idx})
            days  = d_sub['day'].values
            mask  = np.isin(days, [-2, -1, 1, 2])
            X_raw = d_sub.values[:, mask].T
            labels = np.array(['pre' if day in [-2, -1] else 'post'
                                for day in days[mask]])
            y_enc  = le.transform(labels)

            X = StandardScaler().fit_transform(X_raw)
            cv  = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
            acc = np.mean(cross_val_score(
                LogisticRegression(max_iter=50000), X, y_enc, cv=cv, n_jobs=1))

            if group == 'R+':
                accs_rew_perc.append(acc)
            else:
                accs_nonrew_perc.append(acc)

    accs_rew_curve.append(accs_rew_perc)
    accs_nonrew_curve.append(accs_nonrew_perc)
    print(f"Removed {perc:2d}%: R+ mean={np.nanmean(accs_rew_perc):.3f}, "
          f"R- mean={np.nanmean(accs_nonrew_perc):.3f}")


# ============================================================================
# Aggregate with bootstrap CI
# ============================================================================

x_vals = 100 - PERCENTILES   # percent retained (100% → 10%)

def agg_with_ci(accs_list):
    means, ci_lows, ci_highs = [], [], []
    for a in accs_list:
        a = np.array(a)
        means.append(np.nanmean(a))
        if len(a) > 1:
            res = bootstrap((a,), np.nanmean, confidence_level=0.95,
                            n_resamples=1000, method='basic')
            ci_lows.append(res.confidence_interval.low)
            ci_highs.append(res.confidence_interval.high)
        else:
            ci_lows.append(np.nan)
            ci_highs.append(np.nan)
    return np.array(means), np.array(ci_lows), np.array(ci_highs)

mean_rew,    ci_low_rew,    ci_high_rew    = agg_with_ci(accs_rew_curve)
mean_nonrew, ci_low_nonrew, ci_high_nonrew = agg_with_ci(accs_nonrew_curve)

df_plot = pd.DataFrame({
    'percent_cells_retained': np.tile(x_vals, 2),
    'accuracy':      np.concatenate([mean_rew,    mean_nonrew]),
    'ci_low':        np.concatenate([ci_low_rew,  ci_low_nonrew]),
    'ci_high':       np.concatenate([ci_high_rew, ci_high_nonrew]),
    'reward_group':  ['R+'] * len(x_vals) + ['R-'] * len(x_vals),
})


# ============================================================================
# Figure
# ============================================================================

sns.set_theme(context='paper', style='ticks', font='sans-serif', font_scale=1,
              rc={'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none'})

fig, ax = plt.subplots(figsize=(6, 5))

sns.lineplot(data=df_plot, x='percent_cells_retained', y='accuracy',
             hue='reward_group', hue_order=['R+', 'R-'],
             palette=reward_palette[::-1], ax=ax)

for group, color in zip(['R+', 'R-'], reward_palette[::-1]):
    sub = df_plot[df_plot['reward_group'] == group]
    ax.fill_between(sub['percent_cells_retained'], sub['ci_low'], sub['ci_high'],
                    color=color, alpha=0.3)

ax.set_xlabel('Percent of cells retained')
ax.set_ylabel('Classification accuracy')
ax.set_ylim(0, 1)
ax.set_xlim(100, x_vals.min())   # flip x-axis
ax.legend(frameon=False)
sns.despine()
plt.tight_layout()


# ============================================================================
# Save
# ============================================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

fig.savefig(os.path.join(OUTPUT_DIR, 'supp_2d.svg'), format='svg', dpi=300, bbox_inches='tight')
print("Saved: supp_2d.svg")

df_plot.to_csv(os.path.join(OUTPUT_DIR, 'supp_2d_data.csv'), index=False)
print("Saved: supp_2d_data.csv")

plt.show()
