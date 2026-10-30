import numpy as np
import xarray as xr

from fast_learning.drift import N_HALF, within_block
from fast_learning.similarity import DAYS, N_MAP_TRIALS


def _vectors(switch, n_cells=200, seed=0):
    """One mouse, DAYS x N_MAP_TRIALS trials of a fixed pattern plus noise;
    with `switch`, the second half of each day has a different pattern."""
    rng = np.random.default_rng(seed)
    a, b = rng.normal(size=(2, n_cells))
    trials = []
    for _ in DAYS:
        pattern = np.repeat(a[:, None], N_MAP_TRIALS, axis=1)
        if switch:
            pattern[:, N_HALF:] = b[:, None]
        trials.append(pattern + rng.normal(scale=2, size=pattern.shape))
    day = np.repeat(DAYS, N_MAP_TRIALS)
    d = xr.DataArray(np.concatenate(trials, axis=1), dims=('cell', 'trial'), coords={'day': ('trial', day)})
    return {'M1': ('R+', d)}


def test_within_block_stable_pattern():
    wb = within_block(_vectors(switch=False))
    assert np.allclose(wb['r_corrected'], 1, atol=0.1)
    assert np.allclose(wb['r_first_last'], wb['r_odd_even'], atol=0.05)


def test_within_block_pattern_switch():
    wb = within_block(_vectors(switch=True))
    assert (wb['r_corrected'] < 0.3).all()
    assert (wb['r_first_last'] < wb['r_odd_even']).all()
