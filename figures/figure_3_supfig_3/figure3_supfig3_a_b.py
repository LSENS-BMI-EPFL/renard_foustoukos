"""
Figure 3—figure supplement 3a-b: Signal and noise power of the passive whisker
responses, before vs after learning

Response vectors as Fig. 3h-i (all cells, last 40 passive trials per day,
0-300 ms). Signal power: squared norm of the trial-averaged vector minus the
contribution of noise; noise power: summed trial-to-trial variance (see
fast_learning.drift). Each mouse averaged over Days -2/-1 (pre) and +1/+2
(post); Wilcoxon signed-rank test per group (n = mice). Bars: mean and
bootstrapped 95% CI across mice; lines: mice.
Stats file also: SNR, observed and predicted within-day cosine.

Outputs: <figures_dir>/Figure3_SupFig3/figure3_supfig3_a_b.pdf,
figure3_supfig3_a_b_data.csv (mouse x day), figure3_supfig3_a_b_stats.csv.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from fast_learning import paths
from fast_learning.drift import GROUP_COLORS, compute_metrics, load_response_vectors, signal_noise_stats
from fast_learning.plotting import panel_size, save_figure, set_style
from fast_learning.stats import format_p

OUTPUT_DIR = os.path.join(paths.manuscript_output_dir, 'Figure3_SupFig3')


def paired_panel(ax, pp, metric, ylabel, stats, ylim):
    for gi, rg in enumerate(['R+', 'R-']):
        g = pp.loc[rg][metric]
        x = np.array([gi * 3, gi * 3 + 1])
        for _, row in g.iterrows():
            ax.plot(x, [row['pre'], row['post']], '-', color='grey', linewidth=0.5, alpha=0.6)
        # Bars: mean across mice with bootstrapped 95% CI (seeded); lines: mice.
        long = g.reset_index().melt(id_vars='mouse_id', value_vars=['pre', 'post'], value_name='v')
        sns.barplot(
            data=long.assign(x=np.where(long['period'] == 'pre', x[0], x[1])),
            x='x',
            y='v',
            order=list(range(5)),
            native_scale=True,
            color=GROUP_COLORS[rg],
            alpha=0.7,
            edgecolor='black',
            errorbar=('ci', 95),
            seed=0,
            width=0.7,
            ax=ax,
            zorder=1,
        )
        p = stats.query('reward_group == @rg and measure == @metric')['p_value'].iloc[0]
        top = np.nanmax(g.values)
        ax.plot(
            [x[0], x[0], x[1], x[1]], [top * 1.04, top * 1.08, top * 1.08, top * 1.04], 'k-', linewidth=0.8
        )
        ax.text(x.mean(), top * 1.09, format_p(p), ha='center', va='bottom')
        ax.text(x.mean(), -0.2, rg, transform=ax.get_xaxis_transform(), ha='center', va='top')
    ax.set_xticks([0, 1, 3, 4], ['Pre', 'Post', 'Pre', 'Post'])
    ax.set_xlim(-0.7, 4.7)
    ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel)
    ax.set_xlabel('')


if __name__ == '__main__':
    vectors = load_response_vectors()
    day_df, drift_df = compute_metrics(vectors)
    stats, pp = signal_noise_stats(day_df, drift_df)
    stats = stats[~stats['measure'].str.startswith('drift')]

    set_style()
    fig, axes = plt.subplots(1, 2, figsize=panel_size(2))
    paired_panel(axes[0], pp, 'signal_power', 'Signal power', stats, ylim=(0, 15))
    paired_panel(axes[1], pp, 'noise_power', 'Noise power', stats, ylim=(0, 20))
    sns.despine()
    plt.tight_layout()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_figure(fig, os.path.join(OUTPUT_DIR, 'figure3_supfig3_a_b.pdf'))
    plt.close(fig)
    day_df.to_csv(os.path.join(OUTPUT_DIR, 'figure3_supfig3_a_b_data.csv'), index=False)
    stats.to_csv(os.path.join(OUTPUT_DIR, 'figure3_supfig3_a_b_stats.csv'), index=False)
    print(
        stats[['reward_group', 'measure', 'n_mice', 'median_pre', 'median_post', 'p_value']].to_string(
            index=False
        )
    )
