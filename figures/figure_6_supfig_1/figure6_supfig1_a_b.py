"""
Figure 6—figure supplement 1a-b: Participation rate in reactivations vs LMI,
without correction for chance

The panels of Figure 6d-e on the participation rate itself, before the chance
rate (the participation expected from the cell's activity alone) is
subtracted.

Panel a: day-0 participation rate vs LMI, as Fig. 6d. Stats: linear
         mixed-effects model participation_rate ~ LMI + (1 + LMI | mouse), per
         reward group; per-mouse Spearman correlation (stats file).
Panel b: participation rate across days for LMI+ vs LMI- cells, as Fig. 6e.
         Stats: per-mouse day slope, Wilcoxon signed-rank test against zero
         (n = mice).

Participation is measured in held-out cells (events detected with the other
half of the cells, 10 random splits; as Fig. 6d-e). --cells all / insample:
all cells in the events of step 07 / the detecting half, for comparison.

Inputs:  participation rates (pipeline/08_participation.py).
Outputs: main analysis (2.5 noise SD, held-out cells, whole trial):
         <figures_dir>/Figure6_SupFig1/figure6_supfig1_a.pdf, _stats.csv and
         figure6_supfig1_b.pdf, _data.csv, _stats.csv. Other conditions, in
         Figure6_SupFig1/variants/: figure6_supfig1_a_<k>sd<sfx> and
         figure6_supfig1_b_<k>sd<sfx>; <sfx> is empty for held-out cells,
         _insample or _allcells, then _nolick for the no-lick control.
"""

import argparse
import importlib.util
import os

from fast_learning import paths, participation
from fast_learning.plotting import VARIANTS_DIR

OUTPUT_DIR = os.path.join(paths.manuscript_output_dir, 'Figure6_SupFig1')

# Panel functions of Figure 6d-e (figure scripts are not a package).
_spec = importlib.util.spec_from_file_location(
    'figure_6d_e', os.path.join(os.path.dirname(__file__), '..', 'figure_6', 'figure_6d_e.py')
)
fig6 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fig6)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--cells',
        choices=participation.CELLS,
        default='heldout',
        help='held-out cells (main), in-sample cells of the same detections, or all cells (step 07 events)',
    )
    parser.add_argument(
        '--nolick', action='store_true', help='no-lick control (-1 to +1 s), written to the variants folder'
    )
    args = parser.parse_args()
    print(f"Output directory: {OUTPUT_DIR}")

    for threshold in participation.PARTICIPATION_THRESHOLDS:
        # Main analysis in OUTPUT_DIR; other conditions in its variants folder.
        tag = participation.variant_tag(threshold, args.nolick, args.cells)
        out_dir = os.path.join(OUTPUT_DIR, VARIANTS_DIR) if tag else OUTPUT_DIR
        print(f"\n--- participation threshold {threshold:g} x noise SD ({tag or 'main'}) ---")
        merged, per_day = participation.load_participation(threshold, args.nolick, args.cells)
        fig6.panel_i_participation_vs_lmi(merged, output_dir=out_dir, filename=f'figure6_supfig1_a{tag}')
        fig6.panel_j_participation_across_days(
            merged, per_day, output_dir=out_dir, filename=f'figure6_supfig1_b{tag}'
        )
