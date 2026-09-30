"""Pipeline step 04: learning curves and learning trial of the imaging mice.

Reads  paths.processed_dir/behavior/behavior_imagingmice_table_5days_cut.csv
       (written by 03_behavior_tables.py).
Writes paths.processed_dir/behavior/behavior_imagingmice_table_5days_cut_with_learning_curves.csv

Uses fast_learning.behavior.compute_learning_curves / compute_learning_trial, the
implementation that produced the published learning trials: for R+ mice
learning is the first of 10 consecutive whisker trials whose lower 80%
credible bound exceeds the interpolated false-alarm rate; for R- mice it is
the first of 10 consecutive trials where the false-alarm rate lies within the
whisker credible interval.

Each (session, stimulus) fit is seeded from RANDOM_SEED so results are
reproducible. The pre-refactor fits were unseeded, so curves differ from the
reference table by MCMC sampling noise.
"""

import os

import pandas as pd

from fast_learning import paths
from fast_learning.behavior import compute_learning_curves, compute_learning_trial


RANDOM_SEED = 0
N_CONSECUTIVE_TRIALS = 10
BEHAVIOR_DIR = os.path.join(paths.processed_dir, 'behavior')
INPUT_CSV = os.path.join(BEHAVIOR_DIR, 'behavior_imagingmice_table_5days_cut.csv')
OUTPUT_CSV = os.path.join(BEHAVIOR_DIR, 'behavior_imagingmice_table_5days_cut_with_learning_curves.csv')


if __name__ == '__main__':
    table = pd.read_csv(INPUT_CSV)
    table = compute_learning_curves(table, random_seed=RANDOM_SEED)
    table = compute_learning_trial(table, n_consecutive_trials=N_CONSECUTIVE_TRIALS)
    table.to_csv(OUTPUT_CSV, index=False)
    print(f'Saved {OUTPUT_CSV}')
