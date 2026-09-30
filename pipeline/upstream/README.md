# Upstream processing: raw imaging data → NWB

These scripts turn the raw two-photon recordings into the inputs of the NWB
files. They are included for transparency: they need the raw data (ScanImage
TIFF files, suite2p output), which is **not** part of the published dataset,
and paths to the lab servers. They are not run by `run_all.py`. The published
NWB files are the starting point of the reproducible pipeline (`pipeline/01…09`).

| Script | What it does |
|---|---|
| `run_suite2p.py` | Motion correction, ROI detection and trace extraction with suite2p, on the sessions of one mouse concatenated. |
| `split_sessions.py` | Splits the concatenated suite2p traces back into individual sessions. |
| `compute_dff.py` | Neuropil-corrected fluorescence to dF/F0, with a running-percentile baseline. |
| `projection_gui.py` | Qt GUI to label projection neurons (wS2- and wM1-projecting) from the retrograde CTB injections, by registering the CTB images onto the two-photon field of view. |

Extra dependencies: `pip install -e .[preprocessing]` (suite2p, tifffile,
ScanImageTiffReader) and `pip install -e .[gui]` for the projection GUI.
