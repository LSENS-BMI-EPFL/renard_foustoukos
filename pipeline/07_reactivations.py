"""Pipeline step 07: reactivation events.

Part 1 - Surrogate thresholds: per-mouse detection thresholds from circular-
         shift surrogates of the pre-learning days (per-day thresholds are
         also available with SURROGATE_MODE).
Part 2 - Event detection: template-correlation reactivation events for all
         R+ and R- mice, at each percentile in PERCENTILES.

Trials: correct-rejection no-stim trials (no stimulus, no lick in the
response window), whole trial (-1 to 6 s); see
fast_learning.reactivations.select_trials_by_type.
Part 3 - Mouse selection for the participation analyses, from the detection
         at PERCENTILE_TO_USE (see fast_learning.reactivations).

Inputs:  per-mouse tensors (paths.tensor_dir), session metadata.
Outputs: <processed_dir>/reactivation/
             surrogate_thresholds_per_mouse_p<N>.csv
             reactivation_results_p<N>.pkl      (all mice)
             mouse_selection.csv
         With --nolick (no-lick control: the same trials, -1 to +1 s around
         no-stim onset), the same files except mouse_selection.csv in
         reactivation/nolick/.

Usage:
    python pipeline/07_reactivations.py
    python pipeline/07_reactivations.py --nolick           # no-lick control
    python pipeline/07_reactivations.py --selection-only   # rebuild mouse_selection.csv
"""

import argparse
import os
import pickle
import sys

from joblib import Parallel, delayed

from fast_learning import paths, database
from fast_learning import reactivations as rx


# ============================================================================
# Parameters
# ============================================================================

# Part 1: Surrogates
RUN_SURROGATES = True
SURROGATE_MODE = 'mouse'  # 'day' | 'mouse' | 'both'
N_SURROGATES = 1000
PERCENTILES = [99, 99.5, 99.9]
N_JOBS = 35

# Part 2: Detection
USE_SURROGATE_THRESHOLDS = 'mouse'  # 'day' | 'mouse' | None (fixed threshold)
PERCENTILE_TO_USE = 99  # main threshold; defines the mouse selection

OUTPUT_DIR = os.path.join(paths.processed_dir, 'reactivation')


# ============================================================================
# Mouse lists
# ============================================================================


def get_mouse_lists():
    """R+ and R- imaging mice from the session database."""
    _, _, all_mice, db = database.select_sessions_from_db(paths.db_path, paths.nwb_dir, two_p_imaging='yes')
    r_plus_mice, r_minus_mice = [], []
    for mouse in all_mice:
        try:
            rg = database.get_mouse_reward_group_from_db(paths.db_path, mouse, db=db)
            if rg == 'R+':
                r_plus_mice.append(mouse)
            elif rg == 'R-':
                r_minus_mice.append(mouse)
        except Exception:
            continue
    print(f"R+ mice ({len(r_plus_mice)}): {r_plus_mice}")
    print(f"R- mice ({len(r_minus_mice)}): {r_minus_mice}")
    return r_plus_mice, r_minus_mice


# ============================================================================
# Part 1a: Per-day surrogate thresholds
# ============================================================================


def run_surrogates_per_day(
    mice,
    output_dir=OUTPUT_DIR,
    days=rx.DAYS,
    threshold_dff=rx.THRESHOLD_DFF,
    n_surrogates=N_SURROGATES,
    percentiles=PERCENTILES,
    n_jobs=N_JOBS,
    window=None,
):
    """
    Compute per-day surrogate thresholds for all mice in parallel and save CSVs.

    Saves:
        surrogate_thresholds_per_day_p<N>.csv  (one row per mouse × day)
    """
    print("\n" + "=" * 60)
    print("PART 1a — PER-DAY SURROGATE THRESHOLDS")
    print("=" * 60)
    print(
        f"  Mice: {len(mice)}, Days: {days}, Surrogates: {n_surrogates}, "
        f"Percentiles: {percentiles}, Jobs: {n_jobs}"
    )

    results_list = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(rx.process_mouse_surrogates_per_day)(
            mouse,
            days,
            threshold_dff,
            n_surrogates,
            percentiles=percentiles,
            verbose=False,
            window=window,
        )
        for mouse in mice
    )

    all_dfs = rx.collect_surrogate_results(results_list, percentiles)

    if not all_dfs:
        print("ERROR: No valid per-day surrogate results collected.")
        return {}

    os.makedirs(output_dir, exist_ok=True)
    for p, df in all_dfs.items():
        path = os.path.join(output_dir, f'surrogate_thresholds_per_day_{rx.percentile_tag(p)}.csv')
        df.to_csv(path, index=False)
        print(f"  Saved: {path}  ({len(df)} rows)")

    return all_dfs


# ============================================================================
# Part 1b: Per-mouse surrogate thresholds
# ============================================================================


def run_surrogates_per_mouse(
    mice,
    output_dir=OUTPUT_DIR,
    threshold_dff=rx.THRESHOLD_DFF,
    n_surrogates=N_SURROGATES,
    percentiles=PERCENTILES,
    n_jobs=N_JOBS,
    window=None,
):
    """
    Compute per-mouse surrogate thresholds using pre-learning days, in parallel.

    Saves:
        surrogate_thresholds_per_mouse_p<N>.csv  (one row per mouse)
    """
    print("\n" + "=" * 60)
    print("PART 1b — PER-MOUSE SURROGATE THRESHOLDS")
    print("=" * 60)
    print(f"  Mice: {len(mice)}, Surrogates: {n_surrogates}, Percentiles: {percentiles}, Jobs: {n_jobs}")

    results_list = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(rx.process_mouse_surrogates_per_mouse)(
            mouse,
            threshold_dff,
            n_surrogates,
            percentiles=percentiles,
            verbose=False,
            window=window,
        )
        for mouse in mice
    )

    all_dfs = rx.collect_surrogate_results(results_list, percentiles)

    if not all_dfs:
        print("ERROR: No valid per-mouse surrogate results collected.")
        return {}

    os.makedirs(output_dir, exist_ok=True)
    for p, df in all_dfs.items():
        path = os.path.join(output_dir, f'surrogate_thresholds_per_mouse_{rx.percentile_tag(p)}.csv')
        df.to_csv(path, index=False)
        print(f"  Saved: {path}  ({len(df)} rows)")

    return all_dfs


# ============================================================================
# Part 2: Reactivation event detection
# ============================================================================


def run_reactivation_detection(
    r_plus_mice,
    r_minus_mice,
    output_dir=OUTPUT_DIR,
    use_surrogate_thresholds=USE_SURROGATE_THRESHOLDS,
    percentile=PERCENTILE_TO_USE,
    threshold_corr=rx.THRESHOLD_CORR,
    n_jobs=N_JOBS,
    window=None,
):
    """
    Detect reactivation events for all mice and save results.

    Loads surrogate thresholds from output_dir if use_surrogate_thresholds is
    set (raises if the CSV is missing); uses the fixed threshold_corr only when
    use_surrogate_thresholds is None.

    Saves:
        reactivation_results_p<N>.pkl
    """
    print("\n" + "=" * 60)
    print("PART 2 — REACTIVATION EVENT DETECTION")
    print("=" * 60)

    # Load surrogate thresholds
    threshold_dict = None
    if use_surrogate_thresholds is not None:
        csv_name = (
            f'surrogate_thresholds_per_day_{rx.percentile_tag(percentile)}.csv'
            if use_surrogate_thresholds == 'day'
            else f'surrogate_thresholds_per_mouse_{rx.percentile_tag(percentile)}.csv'
        )
        csv_path = os.path.join(output_dir, csv_name)
        threshold_dict = rx.load_surrogate_thresholds(csv_path, percentile=percentile)
        print(f"  Loaded thresholds: {csv_path} ({len(threshold_dict)} mice)")
    else:
        print(f"  Using fixed threshold: {threshold_corr}")

    # Results filename — always suffixed with the percentile
    results_file = os.path.join(output_dir, f'reactivation_results_{rx.percentile_tag(percentile)}.pkl')
    os.makedirs(output_dir, exist_ok=True)

    def _run_group(mice_list, group_name):
        print(f"\n  Processing {group_name} mice ({len(mice_list)})...")
        results_list = Parallel(n_jobs=n_jobs, verbose=10)(
            delayed(rx.analyze_mouse_reactivation)(
                mouse,
                verbose=False,
                threshold_dict=threshold_dict,
                window=window,
            )
            for mouse in mice_list
        )
        return dict(zip(mice_list, results_list))

    r_plus_results = _run_group(r_plus_mice, 'R+')
    r_minus_results = _run_group(r_minus_mice, 'R-')

    results_data = {
        'r_plus_results': r_plus_results,
        'r_minus_results': r_minus_results,
        'parameters': {
            'trial_type': 'no_stim_correct_rejection',
            'window': window,
            'use_surrogate_thresholds': use_surrogate_thresholds,
            'percentile': percentile,
            'threshold_corr': threshold_corr,
            'min_event_distance_ms': rx.MIN_EVENT_DISTANCE_MS,
            'prominence': rx.PROMINENCE,
            'days': rx.DAYS,
        },
    }

    with open(results_file, 'wb') as f:
        pickle.dump(results_data, f)
    print(f"\n  Saved: {results_file}")

    return results_data


# ============================================================================
# Part 3: Mouse selection for the participation analyses
# ============================================================================


def save_mouse_selection(results_data, path=rx.MOUSE_SELECTION_CSV):
    """Write the participation mouse selection computed from results_data."""
    selection = rx.compute_mouse_selection(results_data)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    selection.to_csv(path, index=False)
    excluded = selection.loc[~selection['included']]
    print(
        f"\n  Mouse selection (>= {rx.MIN_DAY0_EVENTS} day-0 events): "
        f"{selection['included'].sum()} of {len(selection)} mice included"
    )
    print(excluded.to_string(index=False) if len(excluded) else "  No mouse excluded.")
    print(f"  Saved: {path}")
    return selection


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pipeline step 07: reactivation events.')
    parser.add_argument(
        '--selection-only',
        action='store_true',
        help='only rebuild mouse_selection.csv from the existing results file',
    )
    parser.add_argument(
        '--nolick', action='store_true', help='no-lick control: trials restricted to -1 to +1 s around onset'
    )
    args = parser.parse_args()

    if args.selection_only:
        results_file = os.path.join(
            OUTPUT_DIR, f'reactivation_results_{rx.percentile_tag(PERCENTILE_TO_USE)}.pkl'
        )
        print(f"Loading {results_file}")
        with open(results_file, 'rb') as f:
            save_mouse_selection(pickle.load(f))
        sys.exit(0)

    window = rx.trial_window(args.nolick)
    output_dir = rx.results_dir(args.nolick)

    print("\n" + "=" * 60)
    print("REACTIVATION PIPELINE" + (" - NO-LICK CONTROL (-1 to +1 s)" if args.nolick else ""))
    print("=" * 60)
    print(f"  Output directory: {output_dir}")
    print(f"  Run surrogates: {RUN_SURROGATES}  (mode: {SURROGATE_MODE})")
    print(f"  Detection threshold mode: {USE_SURROGATE_THRESHOLDS}")

    r_plus_mice, r_minus_mice = get_mouse_lists()
    all_mice_to_process = r_plus_mice + r_minus_mice

    # ------------------------------------------------------------------
    # Part 1: Surrogate computation
    # ------------------------------------------------------------------
    if RUN_SURROGATES:
        if SURROGATE_MODE in ('day', 'both'):
            run_surrogates_per_day(all_mice_to_process, output_dir=output_dir, window=window)
        if SURROGATE_MODE in ('mouse', 'both'):
            run_surrogates_per_mouse(all_mice_to_process, output_dir=output_dir, window=window)
    else:
        print("\nSkipping surrogate computation (RUN_SURROGATES=False).")

    # ------------------------------------------------------------------
    # Part 2: Reactivation event detection, one run per percentile
    # ------------------------------------------------------------------
    for percentile in PERCENTILES:
        results_data = run_reactivation_detection(
            r_plus_mice,
            r_minus_mice,
            output_dir=output_dir,
            percentile=percentile,
            window=window,
        )

        # --------------------------------------------------------------
        # Part 3: Mouse selection, from the main detection threshold of the
        # main analysis (the no-lick control uses the same mice)
        # --------------------------------------------------------------
        if percentile == PERCENTILE_TO_USE and not args.nolick:
            save_mouse_selection(results_data)
