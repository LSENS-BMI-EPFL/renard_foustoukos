"""
Figure 1e: Example behavioral performance across learning days

This script generates Panel e for Figure 1, showing behavioral performance
across 5 days (days -2, -1, 0, +1, +2) for two example mice (GF305 and AR180).
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append('/home/aprenard/repos/fast-learning')
import src.utils.utils_io as io
from src.utils.utils_plot import behavior_palette
from src.utils.utils_behavior import plot_single_session


OUTPUT_DIR = os.path.join(io.manuscript_output_dir, 'figure_1', 'output')


# ============================================================================
# Main Panel Generation
# ============================================================================

def generate_panel(
    mouse_ids=['GF305', 'AR180'],
    days=[-2, -1, 0, 1, 2],
    max_trials=180,
    table_path=os.path.join(io.processed_dir, 'behavior', 'behavior_imagingmice_table_5days_cut.csv'),
    save_path=OUTPUT_DIR,
    save_format='svg',
    dpi=300
):
    """
    Generate Figure 1 Panel e: Behavioral performance across learning days.

    Shows behavioral performance across 5 days for example mice to illustrate
    the learning trajectory. Each subplot represents one day of training.

    Args:
        mouse_ids: List of mouse identifiers to plot
        days: List of day indices relative to learning day 0
        max_trials: Maximum trial number to include per session
        table_path: Path to CSV file containing behavioral data
        save_path: Directory to save output figures
        save_format: Figure format ('svg', 'png', 'pdf')
        dpi: Resolution for saved figure
    """

    # Load behavioral data
    table_path = io.adjust_path_to_host(table_path)
    table = pd.read_csv(table_path)

    # Set plotting theme
    sns.set_theme(
        context='paper',
        style='ticks',
        palette='deep',
        font='sans-serif',
        font_scale=1,
        rc={'xtick.major.width': 0.8, 'ytick.major.width': 0.8}
    )

    # Generate figure for each mouse
    os.makedirs(save_path, exist_ok=True)

    for mouse_id in mouse_ids:
        # Filter data for this mouse and days of interest
        data = table.loc[table.mouse_id == mouse_id]
        data = data.loc[data.day.isin(days)]
        data = data.loc[data.trial_id <= max_trials]

        # Get sessions for each day
        sessions = data.session_id.drop_duplicates().to_list()

        # Create figure with 5 subplots (one per day)
        fig, axes = plt.subplots(1, 5, figsize=(10, 2))

        # Plot each session
        for i, session in enumerate(sessions):
            ax = axes[i]
            plot_single_session(
                data,
                session,
                ax=ax,
                palette=behavior_palette,
                do_scatter=False,
                linewidth=1.5,
            )

        # Save figure
        output_file = os.path.join(save_path, f'figure_1e_{mouse_id}.{save_format}')
        plt.savefig(output_file, format=save_format, dpi=dpi, bbox_inches='tight')
        # plt.close()

        print(f"Figure 1e ({mouse_id}) saved to: {output_file}")

    # Save data: behavioral table filtered for these mice and days
    all_data = []
    for mouse_id in mouse_ids:
        d = table.loc[table.mouse_id == mouse_id]
        d = d.loc[d.day.isin(days)]
        d = d.loc[d.trial_id <= max_trials]
        all_data.append(d)
    pd.concat(all_data, ignore_index=True).to_csv(
        os.path.join(save_path, 'figure_1e_data.csv'), index=False
    )
    print(f"Figure 1e data saved to: {os.path.join(save_path, 'figure_1e_data.csv')}")


# ============================================================================
# Main execution
# ============================================================================

if __name__ == '__main__':
    generate_panel()
