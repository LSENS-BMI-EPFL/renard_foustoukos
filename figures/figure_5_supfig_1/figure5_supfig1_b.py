"""
Figure 5—figure supplement 1b: Decoding accuracy vs percentage of most-modulated
cells retained.

The most modulated cells are progressively removed and the pre vs post
decoder (10-fold stratified cross-validated logistic regression, passive
mapping trials) is re-run. Cells are ranked within each fold, by the LMI
computed on the training trials only (|AUC - 0.5|, pre vs post): ranking by
the LMI of lmi_results.csv, which is computed on the same trials, leaves cells
whose pre/post difference is ~0 over all trials, so that the training and the
held-out trials differ in opposite directions and accuracy falls below chance.
Control: the same number of cells removed at random (one draw per fold).

Lines: mean and bootstrapped 95% CI across mice. Dashed line: chance (50%).

Outputs: <figures_dir>/Figure5_SupFig1/figure5_supfig1_b.pdf,
figure5_supfig1_b_data.csv (group x condition x percentage),
figure5_supfig1_b_mice.csv (mouse x condition x percentage).
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import bootstrap, rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from fast_learning import paths
from fast_learning.decoding import load_and_process_data
from fast_learning.drift import auc_rows
from fast_learning.plotting import panel_size, reward_palette, save_figure, set_style


# ============================================================================
# Parameters
# ============================================================================

PERCENTILES = np.arange(0, 91, 5)  # % of most-modulated cells to remove
N_SPLITS = 10
SEED = 42
CONDITIONS = {'lmi': 'Most modulated removed', 'random': 'Random cells removed'}
OUTPUT_DIR = os.path.join(paths.manuscript_output_dir, 'Figure5_SupFig1')


def removal_curves(d, rng):
    """Cross-validated accuracy of one mouse at each percentage of cells removed,
    removing the cells with the largest training-fold |LMI| or random cells."""
    days = d['day'].values
    mask = np.isin(days, [-2, -1, 1, 2])
    X = d.values[:, mask].T  # trials x cells
    y = np.isin(days[mask], [1, 2]).astype(int)  # 1 = post
    n_cells = X.shape[1]
    acc = {c: np.zeros((N_SPLITS, len(PERCENTILES))) for c in CONDITIONS}
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    for fold, (tr, te) in enumerate(cv.split(X, y)):
        ranks = rankdata(X[tr].T, axis=1)
        abs_lmi = np.abs(auc_rows(ranks, np.flatnonzero(y[tr]), y[tr].sum()) - 0.5)
        order = {'lmi': np.argsort(-abs_lmi), 'random': rng.permutation(n_cells)}
        for j, perc in enumerate(PERCENTILES):
            n_remove = int(np.round(n_cells * perc / 100))
            for c in CONDITIONS:
                keep = order[c][n_remove:]
                # Scaling is fitted on the training fold only.
                model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=50000))
                model.fit(X[tr][:, keep], y[tr])
                acc[c][fold, j] = model.score(X[te][:, keep], y[te])
    return {c: a.mean(0) for c, a in acc.items()}


def mean_ci(a):
    a = np.asarray(a)
    res = bootstrap((a,), np.mean, confidence_level=0.95, n_resamples=1000, method='basic', random_state=SEED)
    return a.mean(), res.confidence_interval.low, res.confidence_interval.high


if __name__ == '__main__':
    vectors_rew, vectors_nonrew, mice_rew, mice_nonrew = load_and_process_data()
    rng = np.random.default_rng(SEED)

    rows = []
    for group, vectors, mice in [('R+', vectors_rew, mice_rew), ('R-', vectors_nonrew, mice_nonrew)]:
        for d, mouse in zip(vectors, mice):
            curves = removal_curves(d, rng)
            for c, accs in curves.items():
                for perc, a in zip(PERCENTILES, accs):
                    rows.append(
                        {
                            'mouse_id': mouse,
                            'reward_group': group,
                            'condition': c,
                            'percent_cells_retained': 100 - perc,
                            'accuracy': a,
                        }
                    )
            print(f"{mouse} ({group}): {curves['lmi'][-1]:.3f} / {curves['random'][-1]:.3f} at 10% retained")
    mice_df = pd.DataFrame(rows)

    summary = []
    for (group, c, x), sub in mice_df.groupby(['reward_group', 'condition', 'percent_cells_retained']):
        m, lo, hi = mean_ci(sub['accuracy'])
        summary.append(
            {
                'reward_group': group,
                'condition': c,
                'percent_cells_retained': x,
                'accuracy': m,
                'ci_low': lo,
                'ci_high': hi,
                'n_mice': len(sub),
            }
        )
    df_plot = pd.DataFrame(summary)

    # ============================================================================
    # Figure
    # ============================================================================

    set_style()
    fig, ax = plt.subplots(figsize=panel_size())
    for group, color in zip(['R+', 'R-'], reward_palette[::-1]):
        for c, ls in [('lmi', '-'), ('random', '--')]:
            sub = df_plot[(df_plot.reward_group == group) & (df_plot.condition == c)]
            ax.plot(
                sub['percent_cells_retained'],
                sub['accuracy'],
                color=color,
                linestyle=ls,
                label=f'{group}, {CONDITIONS[c].lower()}',
            )
            ax.fill_between(
                sub['percent_cells_retained'], sub['ci_low'], sub['ci_high'], color=color, alpha=0.3
            )
    ax.axhline(0.5, color='grey', linestyle=':', linewidth=0.8)
    ax.set_xlabel('Percent of cells retained')
    ax.set_ylabel('Classification accuracy')
    ax.set_ylim(0, 1)
    ax.set_xlim(100, 100 - PERCENTILES.max())  # flip x-axis
    ax.legend(frameon=False, fontsize=5)
    sns.despine()
    plt.tight_layout()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_figure(fig, os.path.join(OUTPUT_DIR, 'figure5_supfig1_b.pdf'))
    plt.close()
    df_plot.to_csv(os.path.join(OUTPUT_DIR, 'figure5_supfig1_b_data.csv'), index=False)
    mice_df.to_csv(os.path.join(OUTPUT_DIR, 'figure5_supfig1_b_mice.csv'), index=False)
    print(df_plot[df_plot.percent_cells_retained.isin([100, 50, 10])].round(3).to_string(index=False))
