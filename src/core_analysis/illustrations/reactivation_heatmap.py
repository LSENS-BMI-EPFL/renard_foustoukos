"""
Standalone script to generate a heatmap-based illustration of reactivation
for an example mouse (AR127) on a single day.

Layout:
  - Left column (narrow): whisker template as a horizontal bar per cell (y = cells)
  - Right column (wide):  neural activity heatmap (y = cells, x = time)
  - Top of heatmap:       tick marks at detected reactivation events
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import AutoLocator

sys.path.append(r'/home/aprenard/repos/NWB_analysis')
sys.path.append(r'/home/aprenard/repos/fast-learning')
import src.utils.utils_io as io

# ============================================================================
# PARAMETERS
# ============================================================================

MOUSE = 'AR127'
DAY = 0              # Which day to display
SAMPLING_RATE = 30   # Hz
TIME_WINDOW = 180    # seconds to show (3 minutes)

RESULTS_DIR = os.path.join(io.results_dir, 'reactivation')
RESULTS_FILE = os.path.join(RESULTS_DIR, 'reactivation_results.pkl')

OUTPUT_DIR = os.path.join(io.results_dir, 'reactivation', 'illustrations')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Heatmap colormap
HEATMAP_CMAP = 'RdBu_r'

# ============================================================================
# PLOTTING FUNCTION
# ============================================================================

def plot_reactivation_heatmap(r_plus_results, r_minus_results,
                               mouse='AR127',
                               day=0,
                               sampling_rate=30,
                               time_window=180,
                               sort_by='template',
                               sig_only=False,
                               save_path=None):
    """
    Create a heatmap illustration of reactivation for a single mouse / day.

    Parameters
    ----------
    r_plus_results, r_minus_results : dict
        Results dictionaries from reactivation.py
    mouse : str
    day : int
    sampling_rate : float
    time_window : float
        Seconds to display.
    sort_by : str
        'template' (default) or 'participation' — cell ordering.
    save_path : str or None

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    # ------------------------------------------------------------------
    # 1. Locate mouse and day
    # ------------------------------------------------------------------
    if mouse in r_plus_results:
        results = r_plus_results[mouse]
        reward_group = 'R+'
    elif mouse in r_minus_results:
        results = r_minus_results[mouse]
        reward_group = 'R-'
    else:
        raise ValueError(f"Mouse {mouse} not found in results")

    if day not in results['days']:
        raise ValueError(f"Day {day} not found for mouse {mouse}")

    day_data = results['days'][day]

    # ------------------------------------------------------------------
    # 2. Extract data
    # ------------------------------------------------------------------
    template = np.array(day_data['template'])            # (n_cells,)
    correlations = np.array(day_data['correlations'])    # (n_frames_total,)
    events = np.array(day_data['events'])                # event frame indices
    threshold = day_data.get('threshold_used', 0.45)
    n_timepoints = day_data['n_timepoints']              # frames per trial

    # selected_trials is an xarray (n_cells, n_trials, n_timepoints)
    selected_trials = day_data['selected_trials']
    n_cells, n_trials, n_tp = selected_trials.shape
    neural_data = selected_trials.values.reshape(n_cells, -1)  # (n_cells, n_frames)
    neural_data = np.nan_to_num(neural_data, nan=0.0)

    # Truncate to requested time window
    max_frames = int(time_window * sampling_rate)
    neural_data = neural_data[:, :max_frames]
    correlations = correlations[:max_frames]
    events = events[events < max_frames]
    n_frames = neural_data.shape[1]

    # ------------------------------------------------------------------
    # 3. Load per-cell metrics in original cell order, then sort
    # ------------------------------------------------------------------
    roi_ids_orig = selected_trials.coords['roi'].values  # (n_cells,)

    # LMI
    lmi_df = pd.read_csv(os.path.join(io.processed_dir, 'lmi_results.csv'))
    mouse_lmi = lmi_df[lmi_df['mouse_id'] == mouse].set_index('roi')
    lmi_orig = np.array([
        mouse_lmi.loc[r, 'lmi'] if r in mouse_lmi.index else np.nan
        for r in roi_ids_orig
    ])

    # Participation rate for this day
    part_csv = os.path.join(io.results_dir, 'reactivation_lmi',
                            'cell_participation_rates_per_day.csv')
    part_df = pd.read_csv(part_csv)
    mouse_part = (part_df[(part_df['mouse_id'] == mouse) & (part_df['day'] == day)]
                  .set_index('roi'))
    part_orig = np.array([
        mouse_part.loc[r, 'participation_rate'] if r in mouse_part.index else np.nan
        for r in roi_ids_orig
    ])

    # ------------------------------------------------------------------
    # 3d. Restrict to significantly-participating cells (optional)
    # ------------------------------------------------------------------
    if sig_only:
        sig_csv = os.path.join(io.results_dir, 'reactivation',
                               'circular_shift_significant_participation.csv')
        sig_df  = pd.read_csv(sig_csv)
        sig_rois = set(sig_df[sig_df['mouse_id'] == mouse]['roi'].values)
        cell_mask = np.array([r in sig_rois for r in roi_ids_orig])
        neural_data   = neural_data[cell_mask, :]
        template      = template[cell_mask]
        lmi_orig      = lmi_orig[cell_mask]
        part_orig     = part_orig[cell_mask]
        roi_ids_orig  = roi_ids_orig[cell_mask]
        n_cells       = neural_data.shape[0]

    # Sort index — NaN values placed at the bottom in both modes
    if sort_by == 'participation':
        key = np.where(np.isnan(part_orig), -np.inf, part_orig)
        sort_idx = np.argsort(key)[::-1]
    else:  # 'template'
        sort_idx = np.argsort(template)[::-1]

    neural_data_sorted = neural_data[sort_idx, :]
    template_sorted    = template[sort_idx]
    lmi_values         = lmi_orig[sort_idx]
    part_values        = part_orig[sort_idx]

    # ------------------------------------------------------------------
    # 4. Colour ranges — symmetric around 0 so white = 0 in RdBu_r
    # ------------------------------------------------------------------
    # Clip display range: percentile 2–99 of actual data
    act_data_min = float(np.percentile(neural_data_sorted, 2))
    act_data_max = float(np.percentile(neural_data_sorted, 99))
    # Symmetric vmin/vmax so 0 maps to white; data is clipped to display range
    vmax_act = max(abs(act_data_min), abs(act_data_max))
    vmin_act = -vmax_act
    # Clip data so no pixel renders a colour beyond the colorbar limits
    neural_data_display = np.clip(neural_data_sorted, act_data_min, act_data_max)

    t_abs_max = max(abs(float(np.nanmin(template_sorted))),
                    abs(float(np.nanmax(template_sorted))), 1e-6)
    vmin_tmpl, vmax_tmpl = -t_abs_max, t_abs_max

    # ------------------------------------------------------------------
    # 5. Build figure with GridSpec
    #
    #   cols: [lmi | part | template | heatmap | cbar_heatmap]
    #   rows: [blanks × 3 | events   | blank   ]   ← row 0 (thin)
    #         [lmi | part | template | heatmap | cbar_heatmap]   ← row 1
    #         [cbar_lmi|cbar_part|cbar_tmpl|blank|blank]   ← row 2 (thin)
    #
    #   Dedicated colorbar row/column keeps all data axes the same
    #   physical size → event ticks align correctly with the heatmap.
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=(11, 6))

    gs = gridspec.GridSpec(
        3, 5,
        figure=fig,
        width_ratios=[1, 1, 1, 14, 0.4],
        height_ratios=[1, 14, 0.8],
        wspace=0.03,
        hspace=0.04,
    )

    # Main data axes
    ax_lmi      = fig.add_subplot(gs[1, 0])
    ax_part     = fig.add_subplot(gs[1, 1])
    ax_template = fig.add_subplot(gs[1, 2])
    ax_heatmap  = fig.add_subplot(gs[1, 3])
    ax_cbar     = fig.add_subplot(gs[1, 4])

    # Events panel shares x-axis with heatmap
    ax_events = fig.add_subplot(gs[0, 3], sharex=ax_heatmap)

    # Colorbar axes for the three strips (bottom row)
    ax_cbar_lmi  = fig.add_subplot(gs[2, 0])
    ax_cbar_part = fig.add_subplot(gs[2, 1])
    ax_cbar_tmpl = fig.add_subplot(gs[2, 2])

    # Hide unused cells
    for _ax in [fig.add_subplot(gs[0, 0]),
                fig.add_subplot(gs[0, 1]),
                fig.add_subplot(gs[0, 2]),
                fig.add_subplot(gs[0, 4]),
                fig.add_subplot(gs[2, 3]),
                fig.add_subplot(gs[2, 4])]:
        _ax.set_visible(False)

    # ------------------------------------------------------------------
    # 6. Heatmap
    # ------------------------------------------------------------------
    ax_heatmap.imshow(
        neural_data_display,
        aspect='auto',
        cmap=HEATMAP_CMAP,
        vmin=vmin_act, vmax=vmax_act,
        interpolation='none',
        origin='upper',
        extent=[0, n_frames / sampling_rate, n_cells - 0.5, -0.5],
    )
    ax_heatmap.set_xlabel('Time (s)', fontsize=9)
    ax_heatmap.set_yticks([])
    ax_heatmap.tick_params(axis='x', labelsize=7)
    ax_heatmap.spines['top'].set_visible(False)
    ax_heatmap.spines['right'].set_visible(False)
    ax_heatmap.spines['left'].set_visible(False)

    # ------------------------------------------------------------------
    # 7. Events panel (shares x-axis with heatmap)
    # ------------------------------------------------------------------
    ax_events.set_ylim(0, 1)
    for ev in events:
        ax_events.axvline(ev / sampling_rate, color='#e84040', linewidth=0.8, alpha=0.9)
    ax_events.set_xticks([])
    ax_events.set_yticks([])
    for sp in ax_events.spines.values():
        sp.set_visible(False)
    ax_events.set_title(
        f'{mouse} ({reward_group})  |  Day {day}  |  {len(events)} reactivations',
        fontsize=10, fontweight='bold', pad=4
    )
    # Restore auto x-ticks on heatmap (sharex may have inherited the empty locator)
    ax_heatmap.xaxis.set_major_locator(AutoLocator())

    # ------------------------------------------------------------------
    # Helper: render a (n_cells × 1) strip, masking NaN as light grey
    # ------------------------------------------------------------------
    def _strip(ax, values, cmap, xlabel, norm=None, vmin=None, vmax=None):
        img = np.ma.masked_invalid(values.reshape(-1, 1))
        cmap_obj = plt.get_cmap(cmap).copy()
        cmap_obj.set_bad(color='lightgrey')
        kwargs = dict(aspect='auto', cmap=cmap_obj,
                      interpolation='none', origin='upper')
        if norm is not None:
            kwargs['norm'] = norm
        else:
            kwargs['vmin'] = vmin
            kwargs['vmax'] = vmax
        im = ax.imshow(img, **kwargs)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(xlabel, fontsize=8)
        for sp in ax.spines.values():
            sp.set_visible(False)
        return im

    # ------------------------------------------------------------------
    # 8. Template strip  (RdBu_r, TwoSlopeNorm centred at 0)
    # ------------------------------------------------------------------
    im_tmpl = _strip(ax_template, template_sorted, HEATMAP_CMAP,
                     'Template', vmin=vmin_tmpl, vmax=vmax_tmpl)

    # ------------------------------------------------------------------
    # 9. LMI strip  (RdBu_r, fixed ±1 scale)
    # ------------------------------------------------------------------
    im_lmi = _strip(ax_lmi, lmi_values, HEATMAP_CMAP, 'LMI', vmin=-1, vmax=1)

    # ------------------------------------------------------------------
    # 10. Participation rate strip  (sequential, 0–1)
    # ------------------------------------------------------------------
    im_part = _strip(ax_part, part_values, 'Reds', 'Particip.\nrate',
                     vmin=0, vmax=1)

    # ------------------------------------------------------------------
    # 11. Colourbars
    # ------------------------------------------------------------------
    tmpl_data_min = float(np.nanmin(template_sorted))
    tmpl_data_max = float(np.nanmax(template_sorted))

    # Main heatmap colorbar — clip display to actual data range
    cb_act = fig.colorbar(ax_heatmap.images[0], cax=ax_cbar)
    cb_act.ax.set_ylim(act_data_min, act_data_max)
    cb_act.set_ticks([act_data_min, 0, act_data_max])
    cb_act.set_ticklabels([f'{act_data_min:.2f}', '0', f'{act_data_max:.2f}'])
    ax_cbar.set_ylabel('dF/F', fontsize=8)
    ax_cbar.tick_params(labelsize=7)

    # Strip colorbars (bottom row, horizontal)
    for im, cax, label, dmin, dmax in [
        (im_lmi,  ax_cbar_lmi,  'LMI',   -1,           1          ),
        (im_part, ax_cbar_part, 'Rate',   0,            1          ),
        (im_tmpl, ax_cbar_tmpl, 'dF/F',  tmpl_data_min, tmpl_data_max),
    ]:
        cb = fig.colorbar(im, cax=cax, orientation='horizontal')
        cb.ax.set_xlim(dmin, dmax)
        cb.set_ticks([dmin, dmax])
        cb.set_ticklabels([f'{dmin:.2f}', f'{dmax:.2f}'])
        cb.set_label(label, fontsize=7)
        cb.ax.tick_params(labelsize=6)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {save_path}")

    return fig


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("GENERATING REACTIVATION HEATMAP")
    print("=" * 70)
    print(f"  Mouse : {MOUSE}")
    print(f"  Day   : {DAY}")
    print(f"  Window: {TIME_WINDOW}s ({TIME_WINDOW/60:.1f} min)")

    if not os.path.exists(RESULTS_FILE):
        raise FileNotFoundError(
            f"Results file not found: {RESULTS_FILE}\n"
            "Please run reactivation.py with mode='compute' first."
        )

    with open(RESULTS_FILE, 'rb') as f:
        results_data = pickle.load(f)

    r_plus_results = results_data['r_plus_results']
    r_minus_results = results_data['r_minus_results']
    print(f"  Loaded {len(r_plus_results)} R+ and {len(r_minus_results)} R- mice")

    for sort_by, sig_only in [('template',      False),
                               ('participation', False),
                               ('template',      True),
                               ('participation', True)]:
        suffix = f'sort_{sort_by}' + ('_sig' if sig_only else '')
        svg_path = os.path.join(
            OUTPUT_DIR,
            f'{MOUSE}_day{DAY}_reactivation_heatmap_{suffix}.svg'
        )
        pop_label = 'significant cells' if sig_only else 'all cells'
        print(f"\n  Generating figure sorted by {sort_by}, {pop_label}...")
        plot_reactivation_heatmap(
            r_plus_results,
            r_minus_results,
            mouse=MOUSE,
            day=DAY,
            sampling_rate=SAMPLING_RATE,
            time_window=TIME_WINDOW,
            sort_by=sort_by,
            sig_only=sig_only,
            save_path=svg_path,
        )
        plt.close()

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
