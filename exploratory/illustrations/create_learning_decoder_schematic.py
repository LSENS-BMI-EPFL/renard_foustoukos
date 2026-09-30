import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import FancyArrowPatch
from matplotlib import patches

sns.set_theme(context='paper', style='ticks', palette='deep', font='sans-serif', font_scale=1.2,
            rc={'pdf.fonttype':42, 'ps.fonttype':42, 'svg.fonttype':'none'})

# Create figure
fig, ax = plt.subplots(1, 1, figsize=(10, 8))

# Set up the 2D neural activity space
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
ax.set_aspect('equal')

# Decision boundary (hyperplane) - diagonal line
x_line = np.array([-5, 5])
y_line = np.array([5, -5])
ax.plot(x_line, y_line, 'k--', linewidth=2, label='Decision boundary', zorder=1)

# Generate pre-learning trials (Day -2, -1) - clustered on upper left
np.random.seed(42)
n_pre = 40
pre_x = np.random.normal(-2.5, 0.6, n_pre)
pre_y = np.random.normal(2.5, 0.6, n_pre)

# Generate post-learning trials (Day +1, +2) - clustered on lower right
n_post = 40
post_x = np.random.normal(2.5, 0.6, n_post)
post_y = np.random.normal(-2.5, 0.6, n_post)

# Plot pre and post mapping trials
ax.scatter(pre_x, pre_y, c='#800026', s=80, alpha=0.6,
          edgecolors='black', linewidth=0.5, label='Pre-learning (Day -2/-1)', zorder=3)
ax.scatter(post_x, post_y, c='#3f3483', s=80, alpha=0.6,
          edgecolors='black', linewidth=0.5, label='Post-learning (Day +1/+2)', zorder=3)

# Generate Day 0 learning trials showing gradual transition
n_learning = 20
# Create a trajectory from pre to post region
t = np.linspace(0, 1, n_learning)
learning_x = -2.5 + 5.0 * t + np.random.normal(0, 0.3, n_learning)
learning_y = 2.5 - 5.0 * t + np.random.normal(0, 0.3, n_learning)

# Plot Day 0 learning trials with gradient color
colors_gradient = plt.cm.Spectral(t)
for i in range(n_learning):
    ax.scatter(learning_x[i], learning_y[i], c=[colors_gradient[i]],
              s=100, alpha=0.8, edgecolors='black', linewidth=1.0,
              marker='D', zorder=4)

# Add trajectory line through Day 0 trials
ax.plot(learning_x, learning_y, 'gray', linewidth=1.5, alpha=0.5,
       linestyle='--', zorder=2, label='Learning trajectory')

# Add arrow to show direction of learning
arrow_start_idx = 3
arrow_end_idx = n_learning - 3
arrow = FancyArrowPatch((learning_x[arrow_start_idx], learning_y[arrow_start_idx]),
                       (learning_x[arrow_end_idx], learning_y[arrow_end_idx]),
                       arrowstyle='->', mutation_scale=30, linewidth=2.5,
                       color='black', alpha=0.7, zorder=5)
ax.add_patch(arrow)

# Add text annotations
ax.text(-3, 4, 'Pre-learning\nstate', fontsize=14, fontweight='bold',
       ha='center', va='center', color='#4C72B0')
ax.text(3, -4, 'Post-learning\nstate', fontsize=14, fontweight='bold',
       ha='center', va='center', color='#DD8452')
ax.text(0.5, 2, 'Day 0\nlearning trials', fontsize=12, fontweight='bold',
       ha='center', va='center', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Add colorbar for Day 0 trials
sm = plt.cm.ScalarMappable(cmap='Spectral', norm=plt.Normalize(vmin=0, vmax=n_learning-1))
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('Trial number (Day 0)', fontsize=12)
cbar.set_ticks([0, n_learning//2, n_learning-1])
cbar.set_ticklabels(['Early', 'Mid', 'Late'])

# Labels and title
ax.set_xlabel('Neural activity dimension 1', fontsize=14)
ax.set_ylabel('Neural activity dimension 2', fontsize=14)
ax.set_title('Gradual neural transition during Day 0 learning', fontsize=16, fontweight='bold', pad=20)

# Legend
handles, labels = ax.get_legend_handles_labels()
# Add custom legend entry for Day 0 trials
from matplotlib.lines import Line2D
day0_handle = Line2D([0], [0], marker='D', color='w', markerfacecolor='gray',
                    markersize=10, markeredgecolor='black', label='Day 0 learning trials')
handles.append(day0_handle)
labels.append('Day 0 learning trials')
ax.legend(handles=handles, labels=labels, loc='upper right', fontsize=11, framealpha=0.9)

# Remove top and right spines
sns.despine()

plt.tight_layout()

# Save figure
output_dir = '/mnt/lsens-analysis/Anthony_Renard/analysis_output/fast-learning/day0_learning/gradual_learning'
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, 'decoder_learning_schematic.svg')
plt.savefig(output_path, format='svg', dpi=300, bbox_inches='tight')
print(f"Schematic saved to: {output_path}")

plt.show()
