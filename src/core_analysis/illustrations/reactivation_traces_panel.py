"""
Standalone script to generate a 5-column panel showing correlation traces
across days for an example mouse (AR127).

This creates a publication-ready figure panel showing reactivation events
detected across all 5 days of the experiment.
"""

import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(r'/home/aprenard/repos/NWB_analysis')
sys.path.append(r'/home/aprenard/repos/fast-learning')
import src.utils.utils_io as io

# ============================================================================
# PARAMETERS
# ============================================================================

# Mouse to plot
MOUSE = 'AR127'

# Days to include
DAYS = [-2, -1, 0, 1, 2]

# Sampling rate (Hz)
SAMPLING_RATE = 30

# Time duration to display per day (in seconds)
# Set to None to show entire trace, or specify duration (e.g., 180 for 3 minutes)
TIME_WINDOW_PER_DAY = 150 

# Number of horizontal lines/rows to display per subplot
# Each line will show TIME_WINDOW_PER_DAY / N_LINES_PER_SUBPLOT seconds
N_LINES_PER_SUBPLOT = 5  # 5 lines = 30 seconds per line

# Path to saved reactivation results
RESULTS_DIR = os.path.join(io.results_dir, 'reactivation')
RESULTS_FILE = os.path.join(RESULTS_DIR, 'reactivation_results.pkl')

# Output path
OUTPUT_DIR = os.path.join(io.results_dir, 'reactivation', 'illustrations')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================================
# PLOTTING FUNCTION
# ============================================================================

def plot_example_mouse_correlation_traces_panel(r_plus_results, r_minus_results,
                                                 mouse='AR127',
                                                 days=[-2, -1, 0, 1, 2],
                                                 sampling_rate=30,
                                                 time_window=None,
                                                 n_lines=5,
                                                 save_path=None):
    """
    Create a 5-column panel showing correlation traces across days for an example mouse.
    Each column shows one day with correlation trace split into multiple horizontal lines,
    similar to an oscilloscope display.
    Sized to fit as a panel in a row of an A4 figure.

    Parameters
    ----------
    r_plus_results : dict
        Results dictionary for R+ mice
    r_minus_results : dict
        Results dictionary for R- mice
    mouse : str, optional
        Mouse ID (default: 'AR127')
    days : list, optional
        List of days to plot (default: [-2, -1, 0, 1, 2])
    sampling_rate : float, optional
        Sampling rate (Hz, default: 30)
    time_window : float, optional
        Time duration to display per day in seconds (default: None shows entire trace)
    n_lines : int, optional
        Number of horizontal lines to display per subplot (default: 5)
    save_path : str, optional
        Path to save figure

    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure
    """
    # Find mouse in results
    if mouse in r_plus_results:
        results = r_plus_results[mouse]
        reward_group = 'R+'
    elif mouse in r_minus_results:
        results = r_minus_results[mouse]
        reward_group = 'R-'
    else:
        print(f"Warning: Mouse {mouse} not found in results")
        return None

    # Calculate global y-limits across all days
    all_correlations = []
    for day in days:
        if day in results['days']:
            all_correlations.extend(results['days'][day]['correlations'])

    if len(all_correlations) == 0:
        print(f"Warning: No correlation data found for {mouse}")
        return None

    ylim = (np.min(all_correlations), np.max(all_correlations))

    # Create figure with 5 columns (one per day)
    # Size: ~8.27 inches wide (A4 width), ~2.5 inches tall (suitable for a figure panel)
    fig, axes = plt.subplots(1, 5, figsize=(8.27, 2.5), sharey=False)

    for i, day in enumerate(days):
        ax = axes[i]

        if day not in results['days']:
            # No data for this day
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                   transform=ax.transAxes, fontsize=10, color='gray')
            ax.set_xlabel(f'Day {day}', fontsize=9, fontweight='bold')
            ax.set_ylim(ylim)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            continue

        # Get data for this day
        day_data = results['days'][day]
        correlations = day_data['correlations']
        events = day_data['events']
        threshold_used = day_data.get('threshold_used', None)

        # Apply time window if specified
        if time_window is not None:
            max_frames = int(time_window * sampling_rate)
            correlations = correlations[:max_frames]
            # Filter events to only those within the time window
            events = events[events < max_frames]

        # Calculate frames per line
        total_frames = len(correlations)
        frames_per_line = int(total_frames / n_lines) if n_lines > 0 else total_frames

        # Calculate amplitude range for normalization
        corr_range = ylim[1] - ylim[0]
        # Vertical spacing between lines (as fraction of correlation range)
        line_spacing = corr_range * 1.2  # 20% gap between lines

        # Plot each line
        total_events = 0
        for line_idx in range(n_lines):
            # Get data segment for this line
            start_idx = line_idx * frames_per_line
            end_idx = min((line_idx + 1) * frames_per_line, total_frames)

            if start_idx >= total_frames:
                break

            corr_segment = correlations[start_idx:end_idx]

            # Time axis for this segment (starting from 0 for each line)
            time_segment = np.arange(len(corr_segment)) / sampling_rate

            # Vertical offset for this line (top to bottom)
            offset = -line_idx * line_spacing

            # Plot correlation trace with offset
            ax.plot(time_segment, corr_segment + offset, 'k-', linewidth=0.7, alpha=0.9)

            # Plot reference lines (0 and 0.45) as grey dotted lines
            threshold = 0.45
            ax.axhline(threshold + offset, color='gray', linestyle=':',
                      linewidth=0.6, alpha=0.5)
            ax.axhline(0 + offset, color='gray', linestyle=':',
                      linewidth=0.6, alpha=0.5)

            # Add vertical lines for reactivation events in this segment (from findpeak detection)
            segment_events = events[(events >= start_idx) & (events < end_idx)]
            for event_idx in segment_events:
                # Convert to time within this segment
                event_time = (event_idx - start_idx) / sampling_rate
                # Calculate y-range for this specific trace line
                y_min = offset - corr_range * 0.6
                y_max = offset + corr_range * 0.6
                ax.plot([event_time, event_time], [y_min, y_max],
                       color='red', linewidth=0.8, alpha=0.7, clip_on=False)
                total_events += 1

            # Add time label at the start of each line
            start_time_sec = start_idx / sampling_rate
            ax.text(-0.02, offset, f'{int(start_time_sec)}s',
                   transform=ax.get_yaxis_transform(),
                   fontsize=6, ha='right', va='center', color='gray')

        # Formatting
        # Set y-limits to accommodate all lines
        y_bottom = -(n_lines - 1) * line_spacing - corr_range
        y_top = corr_range * 0.5
        ax.set_ylim(y_bottom, y_top)

        # Set x-limits (time per line)
        time_per_line = frames_per_line / sampling_rate
        ax.set_xlim(0, time_per_line)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.set_yticks([])  # Hide y-ticks since we have multiple offset traces

        # Only show y-label on first subplot
        if i == 0:
            ax.set_ylabel('Correlation', fontsize=9, fontweight='bold')

        # Add event count annotation
        ax.text(0.98, 0.98, f'n={total_events}',
               transform=ax.transAxes, fontsize=7,
               ha='right', va='top',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                        edgecolor='gray', alpha=0.8))

        # Set x-axis to show time in seconds
        ax.tick_params(axis='both', labelsize=7)
        # Show ticks every 10-30 seconds depending on time per line
        if time_per_line > 60:
            tick_interval = 30
        elif time_per_line > 30:
            tick_interval = 15
        else:
            tick_interval = 10
        tick_locs = np.arange(0, time_per_line + 1, tick_interval)
        ax.set_xticks(tick_locs)
        ax.set_xticklabels([f'{int(t)}' for t in tick_locs])
        ax.set_xlabel(f'Day {day}\nTime (s)', fontsize=8, fontweight='bold')

    # Overall title
    fig.suptitle(f'{mouse} ({reward_group}) - Correlation Traces Across Days',
                fontsize=11, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved example mouse panel: {save_path}")

    return fig


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("GENERATING EXAMPLE MOUSE CORRELATION TRACES PANEL")
    print("="*70)
    print(f"\nMouse: {MOUSE}")
    print(f"Days: {DAYS}")
    print(f"Sampling rate: {SAMPLING_RATE} Hz")
    print(f"Lines per subplot: {N_LINES_PER_SUBPLOT}")
    if TIME_WINDOW_PER_DAY is not None:
        print(f"Time window per day: {TIME_WINDOW_PER_DAY}s ({TIME_WINDOW_PER_DAY/60:.1f} min)")
        print(f"Time per line: {TIME_WINDOW_PER_DAY/N_LINES_PER_SUBPLOT:.1f}s")
    else:
        print(f"Time window per day: Full trace")

    # Load saved reactivation results
    print(f"\nLoading results from: {RESULTS_FILE}")
    if not os.path.exists(RESULTS_FILE):
        raise FileNotFoundError(
            f"Results file not found: {RESULTS_FILE}\n"
            f"Please run reactivation.py with mode='compute' first."
        )

    with open(RESULTS_FILE, 'rb') as f:
        results_data = pickle.load(f)

    r_plus_results = results_data['r_plus_results']
    r_minus_results = results_data['r_minus_results']

    print(f"✓ Loaded results for {len(r_plus_results)} R+ mice and {len(r_minus_results)} R- mice")

    # Check if mouse exists
    if MOUSE in r_plus_results:
        print(f"✓ Found {MOUSE} in R+ group")
    elif MOUSE in r_minus_results:
        print(f"✓ Found {MOUSE} in R- group")
    else:
        available_mice = list(r_plus_results.keys()) + list(r_minus_results.keys())
        print(f"\n✗ Error: Mouse {MOUSE} not found in results")
        print(f"Available mice: {available_mice}")
        sys.exit(1)

    # Generate panel figure
    print(f"\nGenerating correlation traces panel...")

    # Save as both SVG and PNG
    svg_path = os.path.join(OUTPUT_DIR, f'{MOUSE}_correlation_traces_panel.svg')
    png_path = os.path.join(OUTPUT_DIR, f'{MOUSE}_correlation_traces_panel.png')

    fig = plot_example_mouse_correlation_traces_panel(
        r_plus_results,
        r_minus_results,
        mouse=MOUSE,
        days=DAYS,
        sampling_rate=SAMPLING_RATE,
        time_window=TIME_WINDOW_PER_DAY,
        n_lines=N_LINES_PER_SUBPLOT,
        save_path=svg_path
    )

    # Also save as PNG
    if fig is not None:
        fig.savefig(png_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved PNG version: {png_path}")

    print("\n" + "="*70)
    print("DONE!")
    print("="*70)
    print(f"\nOutput files:")
    print(f"  • {svg_path}")
    print(f"  • {png_path}")
    print()
