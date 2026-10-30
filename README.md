# fast-learning

[![CI](https://github.com/aprrenard/renard_foustoukos_elife_2026/actions/workflows/ci.yml/badge.svg)](https://github.com/aprrenard/renard_foustoukos_elife_2026/actions/workflows/ci.yml)
[![eLife](https://img.shields.io/badge/eLife-10.7554%2FeLife.111818.1-087acc)](https://doi.org/10.7554/eLife.111818.1)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)

Analysis code for

> Renard A.\*, Foustoukos G.\*, Iuga M., Bech P., Bisi A., Dard R.F., Crochet S.\*, Petersen C.C.H.\*
> **Rapid cortical reorganization tracks goal-directed sensorimotor learning in real time.**
> *eLife* (2026), reviewed preprint. [doi:10.7554/eLife.111818.1](https://doi.org/10.7554/eLife.111818.1)
>
> \* equal contribution

It goes from the NWB files of the dataset to every figure panel, with the
source data and statistics of each panel. Development, with the full commit
history, happens at
[aprrenard/renard_foustoukos_elife_2026](https://github.com/aprrenard/renard_foustoukos_elife_2026).

## Data

| What | Where |
|---|---|
| NWB files (behaviour and two-photon imaging, one file per session) | DANDI Archive: [DANDI:002039](https://dandiarchive.org/dandiset/002039) |
| This code and the preprocessed data (session metadata, per-mouse imaging tensors, derived data used by the figures) | Zenodo: [doi:10.5281/zenodo.XXXXXXX](https://doi.org/10.5281/zenodo.XXXXXXX) |

You can reproduce the figures in two ways:

- **from the preprocessed data** (Zenodo): fast, about 15 minutes, a few GB to download;
- **from the NWB files** (DANDI): reruns the whole analysis, several hours.

## Step by step

### 1. Install

You need Python 3.11 or later and [conda](https://docs.conda.io/en/latest/miniconda.html)
(or any other virtual environment).

```bash
git clone https://github.com/LSENS-BMI-EPFL/renard_foustoukos.git fast-learning
cd fast-learning
conda env create -f environment.yml      # creates the 'fast-learning' environment and installs the code
conda activate fast-learning
```

Without conda: `pip install -e .` in a fresh virtual environment. The exact
package versions used for the paper are in
[`requirements-lock.txt`](requirements-lock.txt).

### 2. Download the data

Make a folder for the data, for example `/home/me/fast-learning-data` (use absolute paths throughout).

**Preprocessed data.** Download the archive from
[Zenodo](https://doi.org/10.5281/zenodo.XXXXXXX) and extract it into that
folder. It contains:

```
fast-learning-data/
    metadata/sessions.csv      session metadata
    metadata/stop_flags/       trials analysed in each session
    processed/mice/            per-mouse imaging tensors
    derived/                   behaviour tables, LMI, decoders, reactivations, ...
```

**NWB files** (only to rerun the whole analysis; 1470 files, about 275 GB).
Download the dandiset into `fast-learning-data/nwb`:

```bash
pip install dandi
cd /home/me/fast-learning-data
dandi download DANDI:002039      # creates the folder 002039/
mv 002039 nwb
```

### 3. Configure

Copy [`config.example.yaml`](config.example.yaml) to `config.yaml` and set
where the data are and where the results should go:

```yaml
data_root: /home/me/fast-learning-data        # the folder from step 2
output_root: /home/me/fast-learning-outputs   # created if needed
```

The data folder is only read; everything is written under `output_root`.

### 4. Reproduce the figures

**From the preprocessed data** (about 15 minutes). Add to `config.yaml`

```yaml
processed_dir: /home/me/fast-learning-data/derived
```

then run

```bash
python run_all.py --stage figures
```

**From the NWB files** (several hours). Leave `processed_dir` unset, so the
derived data are recomputed into `<output_root>/processed`, and run

```bash
python run_all.py --stage all
```

To also rebuild the imaging tensors from the NWB files, add
`tensor_dir: /home/me/fast-learning-outputs/processed/mice` to `config.yaml`.

### 5. Find the results

Each figure folder of `<output_root>/figures/` (`Figure1/`, `Figure1_SupFig1/`,
...) holds the panels (PDF; add `figure_formats: [pdf, svg]` to
`config.yaml` for SVG too), the data plotted in each panel (`*_data.csv`) and
its statistics (`*_stats.csv`). The log of each step is in
`<output_root>/logs/`. All computations are seeded, so reruns give identical
results.

### Running only some figures

```bash
python run_all.py --list                              # all steps, in order
python run_all.py --only figure_6 figure_6_supfig_1   # some figure folders
python figures/figure_5/figure_5h.py                  # a single script
```

Each script in `figures/` starts with a docstring describing its panels.

## How the code is organised

```
src/fast_learning/   the analysis library: paths and metadata, tensors, LMI,
                     decoding, reactivation detection, participation,
                     similarity, statistics, plotting
pipeline/            NWB files -> derived data, one numbered step per product
  upstream/          raw imaging -> NWB inputs (suite2p, dF/F), for reference
figures/             one script per figure panel (or group of panels)
run_all.py           runs the pipeline and the figures in order
tests/               unit tests of the core computations (pytest)
exploratory/         earlier and exploratory analyses, not needed for the paper
```

## Development

```bash
pip install -e ".[dev]"
pytest          # unit tests on synthetic data
ruff check .    # lint
ruff format .   # format
```

Optional extras: `.[preprocessing]` (suite2p, for the upstream steps) and
`.[gui]` (projection-neuron labelling GUI).

## Citation and license

Please cite the article above ([`CITATION.cff`](CITATION.cff)). The code is
released under the [MIT license](LICENSE).
