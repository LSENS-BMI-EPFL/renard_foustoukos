import numpy as np
import xarray as xr

from fast_learning import similarity
from fast_learning.stats import significance_stars


def test_significance_stars():
    assert [significance_stars(p) for p in (0.0001, 0.005, 0.03, 0.2)] == ['***', '**', '*', 'n.s.']
    assert significance_stars(0.05) == 'n.s.'
    assert significance_stars(0.2, ns='ns') == 'ns'
    assert significance_stars(float('nan'), na='n.a.') == 'n.a.'
    assert significance_stars(float('nan')) == 'n.s.'


def _vectors(n_cells=10):
    """Responses (cells x trials) for len(DAYS) days of N_MAP_TRIALS trials,
    with one fixed population pattern per day."""
    rng = np.random.default_rng(0)
    n = similarity.N_MAP_TRIALS
    patterns = rng.normal(size=(len(similarity.DAYS), n_cells))
    values = np.concatenate([np.repeat(p[:, None], n, axis=1) for p in patterns], axis=1)
    days = np.repeat(similarity.DAYS, n)
    return xr.DataArray(values, dims=['cell', 'trial'], coords={'day': ('trial', days)})


def test_similarity_matrix():
    v = _vectors()
    cm = similarity.compute_similarity_matrix(v, 'cosine')
    n = len(similarity.DAYS) * similarity.N_MAP_TRIALS
    assert cm.shape == (n, n)
    assert np.all(np.isnan(np.diag(cm)))
    # Identical trials within a day have cosine similarity 1.
    assert np.allclose(cm[0, 1 : similarity.N_MAP_TRIALS], 1)


def test_within_day_metrics():
    cm = similarity.compute_similarity_matrix(_vectors(), 'pearson')
    df = similarity.compute_within_day_metrics([cm], ['M'], 'R+')
    assert list(df['mouse_id']) == ['M']
    for day in similarity.DAYS:
        assert np.isclose(df[f'within_day{day:+d}'].iloc[0], 1)
