"""Locations of input data and outputs.

Paths are read from a YAML configuration file: ``config.yaml`` at the
repository root, or the file named by the ``FAST_LEARNING_CONFIG``
environment variable. Copy ``config.example.yaml`` to ``config.yaml`` and
edit it for your machine.

Only ``data_root`` and ``output_root`` are required. Every other location
defaults to the layout of the published dataset under ``data_root``:

    data_root/
        nwb/                       NWB files, one per session
        metadata/sessions.csv      session metadata table
        metadata/stop_flags/       trial selection YAML files
        processed/mice/<mouse>/    per-mouse xarray tensors (.nc)

Outputs are written under ``output_root``:

    output_root/processed/         derived data (behaviour tables, LMI, decoders, ...)
    output_root/figures/           figure panels with their source data and statistics
    output_root/results/           other analysis outputs

Input locations are only ever read.
"""

import os
from pathlib import Path

import yaml

CONFIG_ENV = 'FAST_LEARNING_CONFIG'
REPO_ROOT = Path(__file__).resolve().parents[2]


def _config_file():
    path = os.environ.get(CONFIG_ENV)
    path = Path(path) if path else REPO_ROOT / 'config.yaml'
    if not path.exists():
        raise FileNotFoundError(
            f'Configuration file not found: {path}\n'
            f'Copy {REPO_ROOT / "config.example.yaml"} to {REPO_ROOT / "config.yaml"} '
            f'and set data_root and output_root, or point {CONFIG_ENV} to your own file.'
        )
    return path


def _load_config():
    path = _config_file()
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    missing = [k for k in ('data_root', 'output_root') if not cfg.get(k)]
    if missing:
        raise KeyError(f'{path}: missing required key(s): {", ".join(missing)}')
    return cfg


_cfg = _load_config()
_data_root = Path(_cfg['data_root'])
_output_root = Path(_cfg['output_root'])
_stop_flags_dir = Path(_cfg.get('stop_flags_dir', _data_root / 'metadata' / 'stop_flags'))


def _get(key, default):
    return str(Path(_cfg.get(key, default)))


# Inputs (read only).
nwb_dir = _get('nwb_dir', _data_root / 'nwb')
db_path = _get('session_metadata', _data_root / 'metadata' / 'sessions.csv')
tensor_dir = _get('tensor_dir', _data_root / 'processed' / 'mice')
# Per-trial DeepLabCut movement matrices (movement revision analysis).
dlc_dir = _get('dlc_dir', _data_root / 'DLCTrialMatrices')
# suite2p ops file of the example mouse of Figure 3a (for its mean image).
fov_ops = _get('fov_ops', _data_root / 'metadata' / 'GF314_ops.npy')
trial_indices_yaml = str(_stop_flags_dir / 'trial_indices_end_session.yaml')
stop_flags_yaml = str(_stop_flags_dir / 'stop_flags_end_session.yaml')
trial_indices_sensory_map_yaml = str(_stop_flags_dir / 'trial_indices_sensory_map.yaml')
stop_flags_sensory_map_yaml = str(_stop_flags_dir / 'stop_flags_sensory_map.yaml')

# Outputs.
output_root = str(_output_root)
processed_dir = _get('processed_dir', _output_root / 'processed')
figures_dir = _get('figures_dir', _output_root / 'figures')

# Figure file formats written by plotting.save_figure (e.g. [pdf, svg]).
figure_formats = list(_cfg.get('figure_formats', ['pdf']))

# Name used by the figure scripts.
manuscript_output_dir = figures_dir


def adjust_path_to_host(path):
    """Map EPFL server paths between their Linux, macOS and Windows forms.

    Only needed by exploratory scripts that still contain hardcoded server
    paths; code that uses the paths above does not need it.
    """
    import platform

    unc = '//sv-nas1.rcp.epfl.ch/Petersen-Lab'
    mounts = {
        'Linux': {'analysis': '/mnt/lsens-analysis', 'data': '/mnt/lsens-data'},
        'Darwin': {'analysis': '/Volumes/Petersen-Lab/analysis', 'data': '/Volumes/Petersen-Lab/data'},
    }
    forms = {
        share: [f'{unc}/{share}'] + [m[share] for m in mounts.values()] for share in ('analysis', 'data')
    }
    target = mounts.get(platform.system())
    for share, variants in forms.items():
        dest = target[share] if target else f'{unc}/{share}'
        for v in variants:
            if path.startswith(v):
                return dest + path[len(v) :]
    return path
