"""
Supplementary Figure 3m: Pairwise correlations between projection neurons
(wS2-wS2 and wM1-wM1 pairs) during a 2 s pre-stimulus quiet window,
compared pre vs post learning. Mapping trials only.

Stats at the cell-pair level (Mann-Whitney U, pre vs post).

Inputs:  pair-level correlations (pipeline/09_pairwise_correlations.py).
Outputs: <figures_dir>/supp_3/output/supp_3m_<group>.svg, _data.csv, _stats.csv.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import mannwhitneyu


from fast_learning import paths, correlations
from fast_learning.plotting import save_figure


# ============================================================================
# Parameters
# ============================================================================

PAIR_TYPES = correlations.PAIR_TYPES

OUTPUT_DIR = os.path.join(paths.manuscript_output_dir, 'supp_3', 'output')


# ============================================================================
# Significance annotation helper
# ============================================================================


def add_significance_stars(ax, x1, x2, y, p_value):
    if p_value < 0.001:
        stars = '***'
    elif p_value < 0.01:
        stars = '**'
    elif p_value < 0.05:
        stars = '*'
    else:
        return
    h = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.02
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, c='black')
    pval_text = f'{stars}\np<0.001' if p_value < 0.001 else f'{stars}\np={p_value:.3f}'
    ax.text((x1 + x2) / 2, y + h, pval_text, ha='center', va='bottom', fontsize=10, fontweight='bold')


# ============================================================================
# Main loop per reward group
# ============================================================================

if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    sns.set_theme(
        context='paper',
        style='ticks',
        font='sans-serif',
        font_scale=1,
        rc={'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none'},
    )

    for reward_group in ['R+', 'R-']:
        corr_csv = correlations.correlations_csv(reward_group)
        if not os.path.exists(corr_csv):
            raise FileNotFoundError(f"{corr_csv} not found. Run pipeline/09_pairwise_correlations.py first.")
        print(f"\nLoading {corr_csv}")
        corr_df = pd.read_csv(corr_csv)

        print(f"Pairs: {corr_df['pair_type'].value_counts().to_dict()}")

        # ── Pair-level stats ─────────────────────────────────────────────────
        stats_pair = []
        for pt in PAIR_TYPES:
            sub = corr_df[corr_df['pair_type'] == pt]
            pre = sub[sub['period'] == 'pre']['correlation'].values
            post = sub[sub['period'] == 'post']['correlation'].values
            if len(pre) > 0 and len(post) > 0:
                stat, p = mannwhitneyu(pre, post, alternative='two-sided')
                stats_pair.append(
                    {
                        'pair_type': pt,
                        'test': 'Mann-Whitney U',
                        'mean_pre': np.mean(pre),
                        'sem_pre': np.std(pre) / np.sqrt(len(pre)),
                        'mean_post': np.mean(post),
                        'sem_post': np.std(post) / np.sqrt(len(post)),
                        'n_pre': len(pre),
                        'n_post': len(post),
                        'statistic': stat,
                        'p_value': p,
                    }
                )
        stats_pair_df = pd.DataFrame(stats_pair)

        stats_pair_df.to_csv(os.path.join(OUTPUT_DIR, f'supp_3m_{reward_group}_stats.csv'), index=False)
        corr_df.to_csv(os.path.join(OUTPUT_DIR, f'supp_3m_{reward_group}_data.csv'), index=False)

        # ── Pair-level figure ─────────────────────────────────────────────────
        fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
        for idx, pt in enumerate(PAIR_TYPES):
            ax = axes[idx]
            sub = corr_df[corr_df['pair_type'] == pt]
            sns.barplot(
                data=sub,
                x='period',
                y='correlation',
                order=['pre', 'post'],
                ax=ax,
                errorbar='se',
                capsize=0.1,
            )
            ax.axhline(0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
            ax.set_title(pt, fontsize=14, fontweight='bold')
            ax.set_xlabel('Period', fontsize=12)
            ax.set_ylabel('Pearson correlation' if idx == 0 else '', fontsize=12)
            ax.set_ylim(0, 0.02)

            row = stats_pair_df[stats_pair_df['pair_type'] == pt]
            if not row.empty:
                p_val = row.iloc[0]['p_value']
                if p_val < 0.05:
                    pre_top = sub[sub['period'] == 'pre']['correlation'].agg(['mean', 'sem']).sum()
                    post_top = sub[sub['period'] == 'post']['correlation'].agg(['mean', 'sem']).sum()
                    add_significance_stars(
                        ax, 0, 1, max(pre_top, post_top) + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.06, p_val
                    )
                else:
                    ax.text(
                        0.95,
                        0.95,
                        f'p={p_val:.3f}',
                        transform=ax.transAxes,
                        fontsize=10,
                        va='top',
                        ha='right',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                    )

        plt.suptitle(f'Pre vs Post — Pair Level ({reward_group})', fontsize=14, y=1.02)
        plt.tight_layout()
        sns.despine()
        save_figure(fig, os.path.join(OUTPUT_DIR, f'supp_3m_{reward_group}.svg'))
        print(f"Saved: supp_3m_{reward_group}.svg")
        plt.close()

    print("\nDone.")
