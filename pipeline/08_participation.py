"""Pipeline step 08: participation of cells in reactivation events.

Participation is measured in held-out cells by default (--cells heldout): the
events are detected with one random half of the cells and participation is
measured in the other half, over N_SPLITS splits (see
fast_learning.participation). This removes the circularity of measuring a
cell's participation in events it helped detect. --cells all measures
participation of all cells in the events of step 07, for comparison.

Uses the participation mouse selection of step 07 (mice with >= 3 day-0
events; see fast_learning.reactivations).

Parts (both by default, or one with --only):
    events  held-out only: events detected by each half of the cells, per split
            and day (surrogate threshold per half; the slow part)
    rates   per-cell participation rates per day, at each participation
            threshold (2.5, 5 and 10 x the cell's noise SD; see
            fast_learning.participation), with the cell's noise SD and
            threshold, the chance rate and the rate above chance
            (excess_rate), and per-cell baseline / day-0 / post rates merged
            with LMI                         (Fig. 6d-e and its supplement 1)
            Held-out: also the in-sample rates (cells of the detecting half,
            same events), for comparison.
    merge   only the merge of the saved per-day rates with LMI (after the LMI
            of step 05 changes; not run by default)

Inputs:  <processed_dir>/reactivation/reactivation_results_p99.pkl,
         mouse_selection.csv, <processed_dir>/lmi_results.csv, tensors.
Outputs: <processed_dir>/reactivation/
             heldout_events.pkl                               (held-out)
             cell_participation_rates_per_day_<k>sd<sfx>.csv
             participation_lmi_merged_<k>sd<sfx>.csv
         <sfx>: none for held-out cells, _insample, _allcells; with --nolick
         (no-lick control) followed by _nolick.

Usage:
    python pipeline/08_participation.py [--cells heldout|all] [--only events rates merge] [--nolick]
"""

import argparse
import os
import pickle

import pandas as pd
from joblib import Parallel, delayed

from fast_learning import paths, participation as pt, reactivations as rx


# ============================================================================
# Parameters
# ============================================================================

N_JOBS = 35

RESULTS_DIR = pt.RESULTS_DIR
LMI_RESULTS_CSV = os.path.join(paths.processed_dir, 'lmi_results.csv')


# ============================================================================
# Inputs
# ============================================================================


def load_selected_results(results_file):
    """Reactivation results of the selected mice, and their reward groups."""
    print(f"\nLoading reactivation events from: {results_file}")
    with open(results_file, 'rb') as f:
        data = pickle.load(f)
    included = rx.load_participation_mice()
    r_plus = {m: r for m, r in data['r_plus_results'].items() if m in included}
    r_minus = {m: r for m, r in data['r_minus_results'].items() if m in included}
    excluded = sorted((set(data['r_plus_results']) | set(data['r_minus_results'])) - included)
    print(f"Selected {len(r_plus)} R+ and {len(r_minus)} R- mice; excluded: {excluded or 'none'}")
    reward_groups = {m: 'R+' for m in data['r_plus_results']}
    reward_groups.update({m: 'R-' for m in data['r_minus_results']})
    return {**r_plus, **r_minus}, reward_groups


def load_heldout_events(nolick=False):
    path = pt.heldout_events_pkl(nolick)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Run this step with --only events first.")
    with open(path, 'rb') as f:
        return pickle.load(f)


def _save_rates(per_day_df, noise_df, reward_groups, threshold, nolick, cells):
    """Add each cell's noise SD and participation threshold, merge with LMI, save."""
    per_day_df = per_day_df.merge(noise_df, on=['mouse_id', 'roi'], how='inner')
    per_day_df['threshold_dff'] = threshold * per_day_df['noise_sd']
    lmi_df = pd.read_csv(LMI_RESULTS_CSV)
    merged = pt.merge_with_lmi(pt.aggregate_across_days(per_day_df), lmi_df, reward_groups)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    per_day_df.to_csv(pt.rates_csv(threshold, nolick, cells), index=False)
    merged.to_csv(pt.merged_csv(threshold, nolick, cells), index=False)
    print(f"Saved: {pt.rates_csv(threshold, nolick, cells)} ({len(per_day_df)} cell-days)")
    print(f"Saved: {pt.merged_csv(threshold, nolick, cells)} ({len(merged)} cells)")


def _noise_and_thresholds(mouse, rois, sub_by_day):
    """Per-cell noise SD (DataFrame) and per-cell thresholds {k: dF/F}.
    Cells without measurable noise (constant trace) are left out."""
    noise = pt.noise_sd(sub_by_day)
    noise_df = pd.DataFrame(dict(mouse_id=mouse, roi=rois, noise_sd=noise))
    return noise_df[noise_df['noise_sd'] > 0], pt.cell_thresholds(noise)


# ============================================================================
# Parts: held-out cells
# ============================================================================


def run_events(results, nolick=False):
    """Events detected by each half of the cells, per split and day."""
    print(
        f"\nHeld-out detection for {len(results)} mice: {pt.N_SPLITS} splits x 2 halves, "
        f"{pt.N_SURROGATES_HALF} surrogates per pre-learning day and half"
    )
    detections = Parallel(n_jobs=N_JOBS, verbose=5)(
        delayed(pt.detect_heldout_events)(mouse, rx.trial_window(nolick)) for mouse in results
    )
    out = {
        'parameters': dict(
            n_splits=pt.N_SPLITS,
            n_surrogates=pt.N_SURROGATES_HALF,
            percentile=pt.DETECTION_PERCENTILE,
            seed=pt.HELDOUT_SEED,
            window=rx.trial_window(nolick),
        ),
        'mice': {d['mouse']: d for d in detections},
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(pt.heldout_events_pkl(nolick), 'wb') as f:
        pickle.dump(out, f)
    print(f"Saved: {pt.heldout_events_pkl(nolick)}")


def _heldout_mouse(detection, window):
    """Participation per split, chance rates and noise SD of one mouse."""
    mouse = detection['mouse']
    rois, days = pt.load_heldout_data(mouse, window)
    sub_by_day = {d: v['sub'] for d, v in days.items()}
    noise_df, thresholds = _noise_and_thresholds(mouse, rois, sub_by_day)
    split = pt.heldout_split_rates(detection, days, thresholds)
    return split, pt.chance_rates(mouse, rois, sub_by_day, thresholds), noise_df


def _all_mouse(mouse, mouse_results, window):
    """Participation of all cells, chance rates and noise SD of one mouse."""
    rois, sub_by_day = pt.load_subtracted_by_day(mouse, window)
    noise_df, thresholds = _noise_and_thresholds(mouse, rois, sub_by_day)
    rates = pt.all_cell_rates(mouse, mouse_results, rois, sub_by_day, thresholds)
    return rates, pt.chance_rates(mouse, rois, sub_by_day, thresholds), noise_df


def _concat(out, i):
    return pd.concat([o[i] for o in out if o[i] is not None], ignore_index=True)


def run_rates_heldout(reward_groups, nolick=False):
    """Held-out (and in-sample) participation rates, at each threshold."""
    events = load_heldout_events(nolick)
    out = Parallel(n_jobs=N_JOBS, verbose=5)(
        delayed(_heldout_mouse)(det, rx.trial_window(nolick)) for det in events['mice'].values()
    )
    split_df, chance_df, noise_df = _concat(out, 0), _concat(out, 1), _concat(out, 2)
    for threshold in pt.PARTICIPATION_THRESHOLDS:
        for role in ('heldout', 'insample'):
            print(f"\n--- {role} participation rates, threshold {threshold:g} noise SD ---")
            sel = split_df[(split_df['role'] == role) & (split_df['threshold'] == threshold)]
            per_day = pt.add_excess(pt.average_over_splits(sel), chance_df, threshold)
            _save_rates(per_day, noise_df, reward_groups, threshold, nolick, role)


# ============================================================================
# Parts: all cells (events of step 07)
# ============================================================================


def run_rates_all(results, reward_groups, nolick=False):
    """Participation rates of all cells per day and merged with LMI, at each threshold."""
    out = Parallel(n_jobs=N_JOBS, verbose=5)(
        delayed(_all_mouse)(mouse, results[mouse], rx.trial_window(nolick)) for mouse in results
    )
    rates_df, chance_df, noise_df = _concat(out, 0), _concat(out, 1), _concat(out, 2)
    for threshold in pt.PARTICIPATION_THRESHOLDS:
        print(f"\n--- participation rates (all cells), threshold {threshold:g} noise SD ---")
        per_day = rates_df[rates_df['threshold'] == threshold].drop(columns='threshold')
        _save_rates(
            pt.add_excess(per_day, chance_df, threshold), noise_df, reward_groups, threshold, nolick, 'all'
        )


def run_merge(reward_groups, cells, nolick=False):
    """Merge the saved per-day participation rates with the current LMI."""
    lmi_df = pd.read_csv(LMI_RESULTS_CSV)
    for threshold in pt.PARTICIPATION_THRESHOLDS:
        for role in cells:
            per_day_df = pd.read_csv(pt.rates_csv(threshold, nolick, role))
            merged = pt.merge_with_lmi(pt.aggregate_across_days(per_day_df), lmi_df, reward_groups)
            merged.to_csv(pt.merged_csv(threshold, nolick, role), index=False)
            print(f"Saved: {pt.merged_csv(threshold, nolick, role)} ({len(merged)} cells)")


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    parts = ['events', 'rates']
    parser = argparse.ArgumentParser(description='Pipeline step 08: participation in reactivations.')
    parser.add_argument('--only', nargs='+', choices=parts + ['merge'], default=parts)
    parser.add_argument(
        '--cells',
        choices=['heldout', 'all'],
        default='heldout',
        help='held-out cells (main analysis) or all cells in the events of step 07',
    )
    parser.add_argument(
        '--nolick', action='store_true', help='no-lick control (events of step 07 --nolick, -1 to +1 s)'
    )
    args = parser.parse_args()

    results, reward_groups = load_selected_results(
        os.path.join(rx.results_dir(args.nolick), 'reactivation_results_p99.pkl')
    )
    if 'merge' in args.only:
        run_merge(
            reward_groups, ('heldout', 'insample') if args.cells == 'heldout' else ('all',), args.nolick
        )
    if args.cells == 'heldout':
        if 'events' in args.only:
            run_events(results, args.nolick)
        if 'rates' in args.only:
            run_rates_heldout(reward_groups, args.nolick)
    elif 'rates' in args.only:
        run_rates_all(results, reward_groups, args.nolick)
