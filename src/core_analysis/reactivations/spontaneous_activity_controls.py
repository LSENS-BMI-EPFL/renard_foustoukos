"""
Spontaneous activity controls for the LMI–participation relationship.

Three steps:
0. Computation: build lmi_data_csv by merging per-cell participation rates,
   spontaneous transient frequencies, and LMI values (day 0).
1. Scatter: participation rate vs transient frequency (Day 0), one dot per cell,
   color-coded by LMI using the coolwarm colormap.
2. Partial correlation: LMI vs participation rate, raw and after controlling
   for transient frequency.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from scipy.signal import find_peaks, savgol_filter
from scipy.stats import pearsonr, linregress

sys.path.append(r'/home/aprenard/repos/NWB_analysis')
sys.path.append(r'/home/aprenard/repos/fast-learning')
import src.utils.utils_imaging as utils_imaging
import src.utils.utils_io as io
from src.utils.utils_plot import *


# ============================================================================
# PARAMETERS
# ============================================================================

sampling_rate = 30                # Hz
min_distance_ms = 200             # Minimum distance between transient peaks (ms)
min_distance_frames = int(min_distance_ms / 1000 * sampling_rate)
prominence_transient = 0.2        # Prominence for transient detection (dF/F)
n_std_threshold = 3               # Per-cell threshold: n_std * std(trace)
savgol_window = 10                # Savitzky-Golay smoothing window (frames)
savgol_order = 2                  # Savitzky-Golay polynomial order

days = [-2, -1, 0, 1, 2]

# Paths
save_dir = os.path.join(io.results_dir, 'reactivation')
lmi_data_csv = os.path.join(save_dir, 'participation_vs_transient_lmi_day0_data.csv')
participation_csv = os.path.join(io.processed_dir, 'reactivation', 'cell_participation_rates_per_day.csv')
reactivation_results_pkl = os.path.join(io.processed_dir, 'reactivation', 'reactivation_results_p99.pkl')
lmi_results_csv = os.path.join(io.processed_dir, 'lmi_results.csv')
folder = os.path.join(io.solve_common_paths('processed_data'), 'mice')

os.makedirs(save_dir, exist_ok=True)


# ============================================================================
# MOUSE LOADING
# ============================================================================

_, _, all_mice, db = io.select_sessions_from_db(
    io.db_path,
    io.nwb_dir,
    two_p_imaging='yes'
)

r_plus_mice = []
r_minus_mice = []

for mouse in all_mice:
    try:
        reward_group = io.get_mouse_reward_group_from_db(io.db_path, mouse, db=db)
        if reward_group == 'R+':
            r_plus_mice.append(mouse)
        elif reward_group == 'R-':
            r_minus_mice.append(mouse)
    except:
        continue

print(f"Found {len(r_plus_mice)} R+ mice and {len(r_minus_mice)} R- mice")


# ============================================================================
# TRANSIENT DETECTION HELPERS
# ============================================================================

def detect_transients(cell_trace):
    """Detect calcium transients in a single cell trace. Returns peak frame indices.

    The height threshold is set per cell as n_std_threshold * std(cell_trace),
    preventing noisy cells from generating spuriously high transient counts.
    """
    smoothed = savgol_filter(cell_trace, savgol_window, savgol_order)
    cell_threshold = n_std_threshold * np.std(cell_trace)
    peaks, _ = find_peaks(
        smoothed,
        height=cell_threshold,
        distance=min_distance_frames,
        prominence=prominence_transient
    )
    return peaks


def compute_transient_freq_per_cell(mouse_id, day=0):
    """
    Compute transient frequency (events/min) per cell for a given mouse and day.
    Uses no-stim trials only. Returns DataFrame: mouse_id, roi, transient_freq.
    """
    try:
        xarr = utils_imaging.load_mouse_xarray(
            mouse_id, folder, 'tensor_xarray_learning_data.nc', substracted=True
        )
    except Exception as e:
        print(f"  Warning: Could not load data for {mouse_id}: {e}")
        return pd.DataFrame()

    xarr_day = xarr.sel(trial=(xarr['day'] == day) & (xarr['no_stim'] == 1))
    if len(xarr_day.trial) == 0:
        return pd.DataFrame()

    n_cells = len(xarr_day.cell)
    roi_ids = xarr_day['roi'].values
    data = xarr_day.values.reshape(n_cells, -1)
    data = np.nan_to_num(data, nan=0.0)
    session_duration_min = data.shape[1] / sampling_rate / 60

    rows = []
    for c in range(n_cells):
        n_peaks = len(detect_transients(data[c]))
        rows.append({
            'mouse_id': mouse_id,
            'roi': roi_ids[c],
            'transient_freq': n_peaks / session_duration_min,
        })
    return pd.DataFrame(rows)


# ============================================================================
# COMPUTE PARTICIPATION CSV
# ============================================================================

def compute_participation_csv(save_path):
    """
    Compute per-cell participation rates across all days from pre-computed
    reactivation events and save to save_path.

    Requires reactivation_results_p99.pkl generated by reactivation.py.
    Uses analyze_mouse_participation from reactivation_lmi_prediction.py.
    """
    import src.core_analysis.reactivations.reactivation_lmi_prediction as rlp

    r_plus_reactivations, r_minus_reactivations = rlp.load_reactivation_results(
        reactivation_results_pkl
    )
    all_reactivation_results = {**r_plus_reactivations, **r_minus_reactivations}

    all_participation_data = []
    for mouse in r_plus_mice + r_minus_mice:
        print(f"  Computing participation for {mouse}...")
        participation_df = rlp.analyze_mouse_participation(
            mouse, days=days, verbose=False,
            preloaded_results=all_reactivation_results.get(mouse)
        )
        if participation_df is not None:
            all_participation_data.append(participation_df)

    if not all_participation_data:
        print("  Warning: No participation data computed.")
        return None

    participation_df_all = pd.concat(all_participation_data, ignore_index=True)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    participation_df_all.to_csv(save_path, index=False)
    print(f"  Saved: {save_path}  ({len(participation_df_all)} cell-day records)")
    return participation_df_all


# ============================================================================
# BUILD LMI DATA CSV
# ============================================================================

def build_lmi_data_csv(save_path):
    """
    Assemble the per-cell day-0 data CSV used by both analyses.

    Merges:
      - cell_participation_rates_per_day.csv  (mouse_id, roi, participation_rate, day)
      - spontaneous transient frequencies     (computed here, day 0, no-stim trials)
      - lmi_results.csv                       (mouse_id, roi, lmi, lmi_p)

    Saves the merged DataFrame to save_path.
    Computes participation_csv first if it does not exist.
    """
    # Compute participation rates if needed
    if not os.path.exists(participation_csv):
        print(f"  Participation CSV not found, computing it...")
        compute_participation_csv(participation_csv)

    # Load participation rates (day 0 only)
    if not os.path.exists(participation_csv):
        print(f"  Warning: Participation CSV not found: {participation_csv}")
        return None
    part_df = pd.read_csv(participation_csv)
    part_df = part_df[part_df['day'] == 0][['mouse_id', 'roi', 'participation_rate']].copy()
    if len(part_df) == 0:
        print("  Warning: No day-0 participation data.")
        return None

    # Compute transient frequencies per cell
    mice = part_df['mouse_id'].unique()
    transient_parts = []
    for mouse_id in mice:
        print(f"  Computing transient freq for {mouse_id}...")
        transient_parts.append(compute_transient_freq_per_cell(mouse_id, day=0))
    transient_df = pd.concat([d for d in transient_parts if len(d) > 0], ignore_index=True)
    if len(transient_df) == 0:
        print("  Warning: No transient data computed.")
        return None

    # Load LMI results
    if not os.path.exists(lmi_results_csv):
        print(f"  Warning: LMI results CSV not found: {lmi_results_csv}")
        return None
    lmi_df = pd.read_csv(lmi_results_csv)[['mouse_id', 'roi', 'lmi', 'lmi_p']]

    # Merge
    merged = part_df.merge(transient_df, on=['mouse_id', 'roi'], how='inner')
    merged = merged.merge(lmi_df, on=['mouse_id', 'roi'], how='inner')

    # Add reward group
    group_map = {m: 'R+' for m in r_plus_mice}
    group_map.update({m: 'R-' for m in r_minus_mice})
    merged['reward_group'] = merged['mouse_id'].map(group_map)
    merged = merged.dropna(subset=['reward_group', 'transient_freq', 'participation_rate', 'lmi'])

    merged.to_csv(save_path, index=False)
    print(f"  Saved: {save_path}  ({len(merged)} cells, {merged['mouse_id'].nunique()} mice)")
    return merged


# ============================================================================
# ANALYSIS 1: PARTICIPATION RATE VS TRANSIENT FREQUENCY (scatter, LMI color)
# ============================================================================

def plot_participation_vs_transient_freq_scatter(data_csv_path, save_path):
    """
    Scatter plot: participation rate vs transient frequency (Day 0).
    One dot per cell, color-coded by LMI using the coolwarm colormap
    (blue = negative LMI, red = positive LMI, centered at 0).
    One panel per reward group (R+ and R-).

    Parameters
    ----------
    data_csv_path : str
        Path to lmi_data_csv.
    save_path : str
        Path to save SVG file.
    """
    if not os.path.exists(data_csv_path):
        print(f"  Warning: Data CSV not found: {data_csv_path}")
        return None

    merged = pd.read_csv(data_csv_path)
    merged = merged.dropna(subset=['lmi', 'transient_freq', 'participation_rate', 'reward_group'])

    # Symmetric colormap centered at LMI = 0
    lmi_abs_max = np.abs(merged['lmi']).max()
    norm = mcolors.TwoSlopeNorm(vmin=-lmi_abs_max, vcenter=0, vmax=lmi_abs_max)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for i, reward_group in enumerate(['R+', 'R-']):
        ax = axes[i]
        gdata = merged[merged['reward_group'] == reward_group]

        if len(gdata) < 3:
            ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center',
                    transform=ax.transAxes)
            ax.set_title(reward_group, fontweight='bold')
            continue

        sc = ax.scatter(
            gdata['transient_freq'], gdata['participation_rate'],
            c=gdata['lmi'], cmap='coolwarm', norm=norm,
            alpha=0.6, s=15, linewidths=0
        )

        # Regression line
        slope, intercept, _, _, _ = linregress(
            gdata['transient_freq'], gdata['participation_rate']
        )
        x_range = np.linspace(gdata['transient_freq'].min(), gdata['transient_freq'].max(), 100)
        ax.plot(x_range, slope * x_range + intercept, 'k-', linewidth=1.5)

        r, p = pearsonr(gdata['transient_freq'], gdata['participation_rate'])
        p_str = ('p < 0.001 ***' if p < 0.001 else
                 f'p = {p:.3f} **' if p < 0.01 else
                 f'p = {p:.3f} *' if p < 0.05 else
                 f'p = {p:.3f} ns')
        n_mice = gdata['mouse_id'].nunique()
        ax.text(0.05, 0.95,
                f'r = {r:.3f}\n{p_str}\nn = {len(gdata)} cells, {n_mice} mice',
                transform=ax.transAxes, fontsize=10, va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))

        plt.colorbar(sc, ax=ax, label='LMI')
        ax.set_xlabel('Transient frequency (events/min)', fontweight='bold', fontsize=12)
        ax.set_ylabel('Participation rate' if i == 0 else '', fontweight='bold', fontsize=12)
        ax.set_title(f'{reward_group}  (n={n_mice} mice, {len(gdata)} cells)',
                     fontweight='bold', fontsize=13)
        ax.grid(True, alpha=0.3)
        print(f"  {reward_group}: r={r:.3f}, p={p:.4f}, n={len(gdata)} cells, {n_mice} mice")

    fig.suptitle('Participation Rate vs Transient Frequency (Day 0, colored by LMI)',
                 fontsize=13, fontweight='bold')
    sns.despine()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format='svg', dpi=300, bbox_inches='tight')
        print(f"  Saved: {save_path}")
        plt.close()

    return fig


# ============================================================================
# ANALYSIS 2: PARTIAL CORRELATION — LMI vs PARTICIPATION CONTROLLING FOR
#             TRANSIENT FREQUENCY
# ============================================================================

def plot_partial_correlation_lmi_participation(data_csv_path, save_path):
    """
    Added-variable (partial regression) scatter plots comparing the raw and
    partial relationship between LMI and participation rate.

    Layout: 2 rows (R+, R-) × 2 columns (raw, partial).
    - Left column: raw scatter of LMI vs participation_rate.
    - Right column: partial scatter — residuals of LMI and participation_rate
      after each is regressed on transient_freq. The slope and r of these
      residuals equal the partial regression coefficient and partial correlation.

    If the partial r (right column) remains significant and close in magnitude
    to the raw r (left column), spontaneous activity does not explain the
    LMI–participation relationship.

    Parameters
    ----------
    data_csv_path : str
        Path to lmi_data_csv.
    save_path : str
        Path to save SVG file.
    """
    if not os.path.exists(data_csv_path):
        print(f"  Warning: Data CSV not found: {data_csv_path}")
        return None

    merged = pd.read_csv(data_csv_path)
    merged = merged.dropna(subset=['lmi', 'participation_rate', 'transient_freq', 'reward_group'])

    def residuals(a, b):
        slope, intercept, _, _, _ = linregress(b, a)
        return a - (slope * b + intercept)

    def annotate(ax, r, p):
        p_str = ('p < 0.001 ***' if p < 0.001 else
                 f'p = {p:.3f} **' if p < 0.01 else
                 f'p = {p:.3f} *' if p < 0.05 else
                 f'p = {p:.3f} ns')
        ax.text(0.05, 0.95, f'r = {r:.3f}\n{p_str}',
                transform=ax.transAxes, fontsize=10, va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for row, reward_group in enumerate(['R+', 'R-']):
        color = reward_palette[1] if reward_group == 'R+' else reward_palette[0]
        gdata = merged[merged['reward_group'] == reward_group]

        if len(gdata) < 5:
            for col in range(2):
                axes[row, col].text(0.5, 0.5, 'Insufficient data',
                                    ha='center', va='center',
                                    transform=axes[row, col].transAxes)
            continue

        lmi = gdata['lmi'].values
        part = gdata['participation_rate'].values
        freq = gdata['transient_freq'].values

        lmi_resid = residuals(lmi, freq)
        part_resid = residuals(part, freq)

        r_raw, p_raw = pearsonr(lmi, part)
        r_partial, p_partial = pearsonr(lmi_resid, part_resid)

        print(f"  {reward_group}  raw r={r_raw:.3f} p={p_raw:.4f} | "
              f"partial r={r_partial:.3f} p={p_partial:.4f}  (n={len(lmi)} cells)")

        for col, (x, y, r, p, xlabel, ylabel) in enumerate([
            (lmi,       part,       r_raw,     p_raw,
             'LMI', 'Participation rate'),
            (lmi_resid, part_resid, r_partial, p_partial,
             'LMI  (residual | transient freq)',
             'Participation rate  (residual | transient freq)'),
        ]):
            ax = axes[row, col]
            ax.scatter(x, y, color=color, alpha=0.3, s=10, linewidths=0)

            slope, intercept, _, _, _ = linregress(x, y)
            x_range = np.linspace(x.min(), x.max(), 100)
            ax.plot(x_range, slope * x_range + intercept,
                    color='black', linewidth=1.5)

            ax.axvline(0, color='gray', linestyle='--', linewidth=0.7, alpha=0.5)
            ax.axhline(0, color='gray', linestyle='--', linewidth=0.7, alpha=0.5)
            annotate(ax, r, p)

            ax.set_xlabel(xlabel, fontsize=11)
            ax.set_ylabel(ylabel if col == 0 else '', fontsize=11)
            title = ('Raw' if col == 0 else 'Partial  (ctrl transient freq)')
            n_mice = gdata['mouse_id'].nunique()
            ax.set_title(f'{reward_group} — {title}  (n={len(lmi)} cells, {n_mice} mice)',
                         fontweight='bold', fontsize=12)
            ax.grid(True, alpha=0.3)

    fig.suptitle('LMI vs Participation Rate: Raw and Partial Correlation (Day 0)',
                 fontsize=13, fontweight='bold')
    sns.despine()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format='svg', dpi=300, bbox_inches='tight')
        print(f"  Saved: {save_path}")
        plt.close()

    return fig


# ============================================================================
# MAIN EXECUTION
# ============================================================================

print("\n" + "="*60)
print("STEP 0: BUILD LMI DATA CSV")
print("(computes participation rates if needed, then merges with transient freq and LMI)")
print("="*60)

print(f"\nBuilding {lmi_data_csv}...")
build_lmi_data_csv(lmi_data_csv)

print("\n" + "="*60)
print("ANALYSIS 1: PARTICIPATION RATE VS TRANSIENT FREQUENCY")
print("="*60)

svg_path = os.path.join(save_dir, 'participation_vs_transient_freq_scatter_day0.svg')
print(f"\nScatter: participation rate vs transient frequency (Day 0), colored by LMI...")
plot_participation_vs_transient_freq_scatter(lmi_data_csv, svg_path)

print("\n" + "="*60)
print("ANALYSIS 2: PARTIAL CORRELATION — LMI vs PARTICIPATION | TRANSIENT FREQ")
print("="*60)

svg_path = os.path.join(save_dir, 'partial_corr_lmi_participation_day0.svg')
print(f"\nPartial correlation: LMI vs participation rate, controlling for transient freq...")
plot_partial_correlation_lmi_participation(lmi_data_csv, svg_path)

print("\n" + "="*60)
print("DONE")
print("="*60)
print(f"\nResults saved to: {save_dir}")
