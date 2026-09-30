"""Analysis library for Renard, Foustoukos et al., eLife (2026).

Modules:
    paths      locations of input data and outputs (from config.yaml)
    database   session metadata table: selecting sessions and mice
    tensors    event-aligned activity arrays from NWB files
    imaging    loading per-mouse xarray tensors, baseline subtraction, ROC/LMI
    behavior   trial tables, performance and learning curves
    plotting   figure style and colour palettes

Modules are imported explicitly (e.g. ``from fast_learning import paths``) so
that importing one does not read the configuration or pull in heavy
dependencies needed by another.
"""

__version__ = '0.1.0'
