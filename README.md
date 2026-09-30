# fast-learning

[![CI](https://github.com/LSENS-BMI-EPFL/renard_foustoukos/actions/workflows/ci.yml/badge.svg)](https://github.com/LSENS-BMI-EPFL/renard_foustoukos/actions/workflows/ci.yml)
[![eLife](https://img.shields.io/badge/eLife-10.7554%2FeLife.111818.1-087acc)](https://doi.org/10.7554/eLife.111818.1)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)

Analysis code for

> Renard A.\*, Foustoukos G.\*, Iuga M., Bech P., Bisi A., Dard R.F., Crochet S.\*, Petersen C.C.H.\*
> **Rapid cortical reorganization tracks goal-directed sensorimotor learning in real time.**
> *eLife* (2026), reviewed preprint. [doi:10.7554/eLife.111818.1](https://doi.org/10.7554/eLife.111818.1)
>
> \* equal contribution

It goes from the NWB files of the published dataset to every figure panel,
with the source data and statistics of each panel.

---

## The study

Head-fixed mice learned to lick in response to a brief deflection of one
whisker. After auditory pre-training, the whisker stimulus was introduced on
Day 0 and kept for two more days; for one group of mice (R+) whisker licks
were rewarded, for the other (R−) they were not. R+ mice learned within a
single session, and inactivating the whisker primary somatosensory cortex
(wS1) prevented this learning.

The same layer 2/3 wS1 neurons were imaged with two-photon calcium imaging
over five days (days −2 to +2; 6,056 neurons in 35 mice), including neurons
projecting to secondary somatosensory (wS2) and motor (wM1) cortex. The
analyses show that:

- **the population response to the whisker stimulus reorganizes in R+ mice**,
  measured per neuron by a learning modulation index (LMI) and at the
  population level by trial-by-trial similarity, with the largest changes in
  wS2-projecting neurons;
- **a decoder of pre- vs post-learning activity tracks learning in real time**
  on Day 0, in step with behaviour;
- **the whisker-evoked ensemble reactivates spontaneously between trials**,
  more often in R+ mice, and neurons that become more responsive during
  learning take part in these reactivations more.

---

## Installation

Python ≥ 3.11. With conda:

```bash
git clone https://github.com/LSENS-BMI-EPFL/renard_foustoukos.git fast-learning
cd fast-learning
conda env create -f environment.yml      # creates the 'fast-learning' env and installs the package
conda activate fast-learning
```

or in any virtual environment: `pip install -e .` The exact package versions
used for the paper are in [`requirements-lock.txt`](requirements-lock.txt).
Optional extras: `.[preprocessing]` (suite2p, for the upstream steps) and
`.[gui]` (projection-neuron labelling GUI).

## Data and configuration

The dataset (NWB files, session metadata, processed tensors) will be
deposited on Zenodo. Copy [`config.example.yaml`](config.example.yaml) to
`config.yaml` and set `data_root` (the downloaded dataset) and `output_root`
(where processed data and figures are written). Input folders are only read.

## Reproducing the figures

```bash
python run_all.py --list                  # the steps, in order
python run_all.py --stage figures         # all figures from the processed data (~15 min)
python run_all.py --only figure_4 supp_4  # some figures
python run_all.py --stage all             # everything from the NWB files (several hours)
python figures/figure_3/figure_3l_n.py    # or any script on its own
```

Each figure script writes, in `<output_root>/figures/<figure>/output/`, the
panel (PDF by default; set `figure_formats: [pdf, svg]` in `config.yaml` for
SVG too) with its plotted data (`*_data.csv`) and statistics (`*_stats.csv`).
Each step's log goes to `<output_root>/logs/`. Computations are seeded, so
reruns give identical results.

## How the code is organised

```
src/fast_learning/   the analysis library: paths and metadata, tensors, LMI,
                     decoding, reactivation detection, participation,
                     similarity, statistics, plotting
pipeline/            NWB files -> processed data, one numbered step per product
  upstream/          raw imaging -> NWB inputs (suite2p, dF/F), for reference
figures/             one script per figure panel (or group of panels)
run_all.py           runs the pipeline and the figures in order
tests/               unit tests of the core computations (pytest)
exploratory/         earlier and exploratory analyses, not needed for the paper
```

| Step | Produces |
|---|---|
| `01_session_flags` | trial ranges analysed per session (end of engagement, passive mapping block); shipped with the data, off by default |
| `02_tensors` | per-mouse dF/F tensors (cells × trials × time) with the trial table |
| `03_behavior_tables` | trial-level behaviour tables (imaging, particle test, muscimol, optogenetics) |
| `04_learning_curves` | Bayesian learning curves and learning trial per session |
| `05_lmi` | learning modulation index of each neuron, with shuffle significance |
| `06_decoder` | per-mouse pre/post-learning decoder (weights reused by Fig. 4b–e) |
| `07_reactivations` | reactivation events (surrogate thresholds, template correlation), mouse selection |
| `08_participation` | participation of each neuron in reactivations |
| `09_pairwise_correlations` | pre-stimulus correlations between projection neurons |

### Figures

<!-- figure-table:start -->
| Panel(s) | Script | Content |
|---|---|---|
| **Figure 1** | | |
| 1b | [`figures/figure_1/figure_1b.py`](figures/figure_1/figure_1b.py) | Example behavioral performance across learning days |
| 1c | [`figures/figure_1/figure_1c.py`](figures/figure_1/figure_1c.py) | Average behavioral performance across imaging mice |
| 1d | [`figures/figure_1/figure_1d.py`](figures/figure_1/figure_1d.py) | Day 0 performance across whisker trials |
| **Figure 2** | | |
| 2b, c | [`figures/figure_2/figure_2b_c.py`](figures/figure_2/figure_2b_c.py) | Muscimol inactivation during learning |
| 2e, f | [`figures/figure_2/figure_2e_f.py`](figures/figure_2/figure_2e_f.py) | Optogenetic inactivation during learning |
| **Figure 3** | | |
| 3a | [`figures/figure_3/figure_3a.py`](figures/figure_3/figure_3a.py) | FOV with LMI overlay + calcium transient illustration for the same mouse |
| 3b | [`figures/figure_3/figure_3b.py`](figures/figure_3/figure_3b.py) | Raster plot of single-cell PSTH activity across learning days for GF314 |
| 3c | [`figures/figure_3/figure_3c.py`](figures/figure_3/figure_3c.py) | Grand-average mapping-trial PSTHs for all cells across learning days, R+ vs R- |
| 3d, e | [`figures/figure_3/figure_3d_e.py`](figures/figure_3/figure_3d_e.py) | Pre vs post learning response comparison (all cells) |
| 3f, g | [`figures/figure_3/figure_3f_g.py`](figures/figure_3/figure_3f_g.py) | Learning Modulation Index (LMI) for all cells |
| 3h–j | [`figures/figure_3/figure_3h_j.py`](figures/figure_3/figure_3h_j.py) | Trial-by-trial correlation matrices and network reorganization metrics |
| 3l–n | [`figures/figure_3/figure_3l_n.py`](figures/figure_3/figure_3l_n.py) | Decoding analyses |
| **Figure 4** | | |
| 4b | [`figures/figure_4/figure_4b.py`](figures/figure_4/figure_4b.py) | Example mice — behaviour and decoder decision value during Day 0 |
| 4c | [`figures/figure_4/figure_4c.py`](figures/figure_4/figure_4c.py) | Progressive learning during Day 0 — population average behaviour and decoder value, R+ vs R- |
| 4d | [`figures/figure_4/figure_4d.py`](figures/figure_4/figure_4d.py) | Slope analysis of the progressive learning decoder — R+ vs R- |
| 4e | [`figures/figure_4/figure_4e.py`](figures/figure_4/figure_4e.py) | Correlation between decoder decision value and behavioural performance across Day-0 whisker trials — R+ vs R- |
| 4f | [`figures/figure_4/figure_4f.py`](figures/figure_4/figure_4f.py) | Reactivation heatmap illustration (mouse AR127, single day) |
| 4g | [`figures/figure_4/figure_4g.py`](figures/figure_4/figure_4g.py) | Example correlation traces across days (mouse AR127) |
| 4h | [`figures/figure_4/figure_4h.py`](figures/figure_4/figure_4h.py) | Reactivation rate across days, R+ vs R- |
| 4i, j | [`figures/figure_4/figure_4i_j.py`](figures/figure_4/figure_4i_j.py) | Reactivation participation rate vs LMI |
| **Figure 1 – supplement** | | |
| S1a | [`figures/supp_1/supp_1a.py`](figures/supp_1/supp_1a.py) | First hit trial on Day 0 for R+ vs R- |
| S1b | [`figures/supp_1/supp_1b.py`](figures/supp_1/supp_1b.py) | Whisker lick probability on Days 0, +1, +2 for R+ vs R- |
| S1c | [`figures/supp_1/supp_1c.py`](figures/supp_1/supp_1c.py) | Particle test — whisker hit rate and false alarm rate across ON / OFF / ON periods (R+ mice only) |
| S1d | [`figures/supp_1/supp_1d.py`](figures/supp_1/supp_1d.py) | Single-trial whisker hit rate across Day 0 trials, R+ vs R- (non-realigned) |
| S1e | [`figures/supp_1/supp_1e.py`](figures/supp_1/supp_1e.py) | Fitted learning curve across Day 0 trials, R+ vs R-, realigned to each mouse's first hit trial |
| S1f | [`figures/supp_1/supp_1f.py`](figures/supp_1/supp_1f.py) | Day 0 lick probability for whisker, auditory, and no-stim trial types on a common time axis (minutes from session start) |
| S1g, h | [`figures/supp_1/supp_1g_h.py`](figures/supp_1/supp_1g_h.py) | Reaction times for auditory, whisker, and no-stim hit trials |
| **Figure 2 – supplement** | | |
| S2b | [`figures/supp_2/supp_2b.py`](figures/supp_2/supp_2b.py) | Example mapping-trial PSTHs for three illustrative cells across learning days |
| S2c | [`figures/supp_2/supp_2c.py`](figures/supp_2/supp_2c.py) | Relationship between classifier weights and the Learning Modulation Index (LMI) |
| S2d | [`figures/supp_2/supp_2d.py`](figures/supp_2/supp_2d.py) | Decoding accuracy vs percentage of most-modulated cells retained |
| **Figure 3 – supplement** | | |
| S3b | [`figures/supp_3/supp_3b.py`](figures/supp_3/supp_3b.py) | Grand-average mapping-trial PSTHs for projection neurons (wS2 and wM1) across learning days, R+ vs R- |
| S3c, d, g, h | [`figures/supp_3/supp_3c_d_g_h.py`](figures/supp_3/supp_3c_d_g_h.py) | Pre vs post learning responses for projection neurons |
| S3e, f, i, j | [`figures/supp_3/supp_3e_f_i_j.py`](figures/supp_3/supp_3e_f_i_j.py) | Proportions and distributions of LMI for projection neurons |
| S3k, l | [`figures/supp_3/supp_3k_l.py`](figures/supp_3/supp_3k_l.py) | CDF comparison of wS2 vs wM1 projector neurons |
| S3m | [`figures/supp_3/supp_3m.py`](figures/supp_3/supp_3m.py) | Pairwise correlations between projection neurons (wS2-wS2 and wM1-wM1 pairs) during a 2 s pre-stimulus quiet window, compared pre vs post learning. Mapping trials only |
| **Figure 4 – supplement** | | |
| S4a, b | [`figures/supp_4/supp_4a_b.py`](figures/supp_4/supp_4a_b.py) | Spontaneous activity controls for the LMI-participation relationship |
| S4c | [`figures/supp_4/supp_4c.py`](figures/supp_4/supp_4c.py) | Proportion of cells participating in reactivation across days for LMI+ vs LMI- cells (binary participation) |
| **Revision analyses (in progress)** | | |
| — | [`figures/revisions/behavior_dprime.py`](figures/revisions/behavior_dprime.py) | Behavior quantified via d' (signal detection theory) instead of whisker hit rate |
| — | [`figures/revisions/behavior_state_summary.py`](figures/revisions/behavior_state_summary.py) | Session-level behavioral state summary (total water reward, session duration, total trial count) per mouse x session, across days and reward groups |
| — | [`figures/revisions/figure_2b_c_execution.py`](figures/revisions/figure_2b_c_execution.py) | Muscimol inactivation during execution |
| — | [`figures/revisions/figure_3f_LMIshuffles.py`](figures/revisions/figure_3f_LMIshuffles.py) | LMI distribution vs. shuffled null |
| — | [`figures/revisions/figure_3h_i_magnitude_reliability.py`](figures/revisions/figure_3h_i_magnitude_reliability.py) | Response magnitude and split-half reliability controls |
| — | [`figures/revisions/figure_4h_shuffle_control.py`](figures/revisions/figure_4h_shuffle_control.py) | Shuffled-template reactivation detection |
| — | [`figures/revisions/figure_4i_j_lmm.py`](figures/revisions/figure_4i_j_lmm.py) | LMM-based reactivation participation rate vs LMI |
| — | [`figures/revisions/movement_state_summary.py`](figures/revisions/movement_state_summary.py) | Facial/whisker movement during the passive mapping epoch, across days |
| — | [`figures/revisions/supp_4a_b_lmm.py`](figures/revisions/supp_4a_b_lmm.py) | LMM-based spontaneous activity controls for the LMI-participation relationship |
| — | [`figures/revisions/supp_4c_lmm.py`](figures/revisions/supp_4c_lmm.py) | LMM-based proportion of cells participating in reactivation across days (LMI+ vs LMI-) |
<!-- figure-table:end -->

The table is generated from the scripts' docstrings: `python tools/update_figure_table.py`.

## Development

```bash
pip install -e ".[dev]"
pytest          # unit tests on synthetic data
ruff check .    # lint
ruff format .   # format
```

## Citation and license

Please cite the article above ([`CITATION.cff`](CITATION.cff)). The code is
released under the [MIT license](LICENSE).
