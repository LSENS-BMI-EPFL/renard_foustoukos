"""Statistics helpers shared by the figure scripts."""

import math


def format_p(p, prefix='p='):
    """P-value label for figures: 'p=0.051' (three decimals) from 0.001 up,
    'p=5×10$^{-4}$' (one significant digit, rendered as a superscript) below,
    and 'p<10$^{-300}$' when the p-value underflows to 0.

    A NaN p-value gives 'n.a.'.
    """
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return 'n.a.'
    if p >= 0.001:
        return f'{prefix}{p:.3f}'
    if p == 0:  # underflow: below the smallest representable double
        return prefix.replace('=', '<') + '10$^{-300}$'
    mantissa, exponent = f'{p:.0e}'.split('e')
    return f'{prefix}{mantissa}×10$^{{{int(exponent)}}}$'


def bootstrap_ci(values, ci=95, n_boot=1000, seed=0):
    """Bootstrapped CI of the mean, computed as seaborn computes the error
    bars of the bar plots (errorbar=('ci', 95), seed=0), so that stats files
    and plotted bars agree. Returns (low, high)."""
    import numpy as np
    from seaborn.algorithms import bootstrap
    from seaborn.utils import ci as percentile_interval

    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    low, high = percentile_interval(bootstrap(values, func='mean', n_boot=n_boot, seed=seed), ci)
    return float(low), float(high)


def significance_stars(p, ns='n.s.', na=None):
    """'***' (p < 0.001), '**' (p < 0.01), '*' (p < 0.05), else `ns`.

    A NaN p-value gives `na` when it is set, and `ns` otherwise.
    """
    if na is not None and isinstance(p, float) and math.isnan(p):
        return na
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    return ns


def dprime(hits, n_hit, false_alarms, n_fa):
    """d' = Z(hit rate) - Z(false-alarm rate) from trial counts, with the
    log-linear correction (Hautus, 1995): rates are (count + 0.5) / (n + 1),
    so that 0% and 100% rates stay finite. Works on scalars and arrays."""
    from scipy.stats import norm

    hit_rate = (hits + 0.5) / (n_hit + 1)
    fa_rate = (false_alarms + 0.5) / (n_fa + 1)
    return norm.ppf(hit_rate) - norm.ppf(fa_rate)


def dprime_from_rates(p_hit, p_fa, eps=1e-3):
    """d' from smoothed hit and false-alarm probabilities (e.g. fitted
    learning curves), clipped to [eps, 1 - eps] so that the z-transform stays
    finite. Works on scalars and arrays."""
    import numpy as np
    from scipy.stats import norm

    p_hit = np.clip(p_hit, eps, 1 - eps)
    p_fa = np.clip(p_fa, eps, 1 - eps)
    return norm.ppf(p_hit) - norm.ppf(p_fa)


OPTIMIZERS = ['bfgs', 'lbfgs', 'powell', 'cg']


def _mixed_fit(formula, df, group, re_formula=None):
    """Fit a mixed model with a random intercept per group, and the random
    slopes of re_formula (e.g. '~x') when given.

    Optimizers can stop at different likelihoods on random-slope models, so
    the model is fitted with each of OPTIMIZERS and the converged fit with
    finite standard errors and the highest likelihood is kept. When there is
    none (typically a variance estimated at zero, where standard errors are
    undefined), the fit falls back to ordinary least squares with standard
    errors clustered by mouse, which still takes the mouse as the unit.
    Returns (result, mouse intercept variance, residual variance, method)."""
    import warnings

    import numpy as np
    import statsmodels.formula.api as smf
    from statsmodels.regression.mixed_linear_model import MixedLM

    model = MixedLM.from_formula(formula, groups=group, re_formula=re_formula, data=df)
    best = None
    for optimizer in OPTIMIZERS:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            try:
                result = model.fit(method=optimizer)
            except (ValueError, np.linalg.LinAlgError):
                continue
        if result.converged and np.all(np.isfinite(result.bse)) and (best is None or result.llf > best.llf):
            best = result
    if best is not None:
        terms = '1 + ' + re_formula.lstrip('~').replace(' ', '') if re_formula else '1'
        return best, float(best.cov_re.iloc[0, 0]), best.scale, f'LMM ({terms} | mouse)'
    groups = df[group].astype('category').cat.codes
    ols = smf.ols(formula, data=df).fit(cov_type='cluster', cov_kwds={'groups': groups})
    return ols, 0.0, ols.scale, 'OLS, mouse-clustered SE (mixed model at the boundary)'


def lmm_slope(df, y, x, group='mouse_id', covariates=(), random_slope=True):
    """Mixed model y ~ x [+ covariates] + (1 + x | group): the fixed-effect
    slope of x with cells (or cell pairs) grouped by mouse. Each mouse has its
    own intercept and its own slope of x (random_slope=False: intercept
    only), so that the test of the slope takes the mouse, not the cell, as the
    unit. With covariates, the slope of x is the one holding them fixed (the
    mixed-model analogue of a partial correlation).

    Returns a dict with slope, se, CI, p-value (Wald), intercept, the share of
    variance of the mouse intercept (at x = 0), the SD of the slope across
    mice and the method used.
    """
    import numpy as np

    formula = ' + '.join([f'{y} ~ {x}', *covariates])
    result, re_var, resid_var, method = _mixed_fit(formula, df, group, f'~{x}' if random_slope else None)
    ci_low, ci_high = result.conf_int().loc[x]
    cov_re = getattr(result, 'cov_re', None)
    slope_sd = float(np.sqrt(cov_re.loc[x, x])) if cov_re is not None and x in cov_re.index else 0.0
    return {
        'slope': result.params[x],
        'se': result.bse[x],
        'ci_low': ci_low,
        'ci_high': ci_high,
        'p_value': result.pvalues[x],
        'intercept': result.params['Intercept'],
        'icc_mouse': re_var / (re_var + resid_var),
        'slope_sd_mouse': slope_sd,
        'method': method,
    }


def per_mouse_rank_correlation(df, y, x, group='mouse_id'):
    """Spearman correlation of x and y within each mouse, then a Wilcoxon
    signed-rank test of the per-mouse correlations against zero (n = mice).
    Ranks make the test insensitive to the magnitude of extreme values.
    Returns (dict of summary statistics, per-mouse correlations)."""
    import numpy as np
    from scipy.stats import spearmanr, wilcoxon

    rho = df.groupby(group).apply(lambda m: spearmanr(m[x], m[y])[0], include_groups=False).dropna()
    return {
        'median_rho': float(np.median(rho)),
        'n_positive': int((rho > 0).sum()),
        'n_mice': len(rho),
        'p_value': float(wilcoxon(rho).pvalue) if len(rho) >= 3 else np.nan,
    }, rho


def per_mouse_slope_test(mouse_day_data, y='participation_rate', x='day'):
    """Linear trend across days tested with mice as the unit: one slope of y
    vs x per mouse (on that mouse's per-day means), then a Wilcoxon
    signed-rank test of the slopes against zero.

    Returns None with fewer than 2 mice having at least 2 days; otherwise
    the summary statistics and 'slopes' ({mouse: slope}).
    """
    import numpy as np
    from scipy.stats import linregress, wilcoxon

    slopes, mice = [], []
    for mouse, mdata in mouse_day_data.groupby('mouse_id'):
        mdata = mdata.dropna(subset=[y])
        if mdata[x].nunique() >= 2:
            slopes.append(linregress(mdata[x], mdata[y]).slope)
            mice.append(mouse)
    if len(slopes) < 2:
        return None
    slopes = np.array(slopes)
    try:
        w_stat, p_value = wilcoxon(slopes)
    except ValueError:  # e.g. all slopes zero
        w_stat, p_value = np.nan, np.nan
    return {
        'mean_slope': float(np.mean(slopes)),
        'median_slope': float(np.median(slopes)),
        'sd_slope': float(np.std(slopes, ddof=1)),
        'w_stat': float(w_stat),
        'p_value': float(p_value),
        'n_mice': len(slopes),
        'slopes': dict(zip(mice, slopes)),
    }


def ks_permutation_test(df, value, label, groups, shuffle, n_perm=10000, seed=0, mouse='mouse_id'):
    """Kolmogorov-Smirnov comparison of two groups of cells whose p-value
    respects that cells are recorded in mice.

    The statistic is the usual two-sample KS distance between the pooled cell
    values of groups[0] and groups[1]. Its null distribution is built by
    shuffling labels at the level where they are exchangeable under the null:
      shuffle='between_mice': the label is a property of the mouse (e.g. R+ vs
          R-); whole mice are reassigned to the groups, each keeping all its
          cells, with the number of mice per group fixed.
      shuffle='within_mice': the label is a property of the cell within mice
          (e.g. wS2 vs wM1); labels are shuffled among the cells of each
          mouse, so each mouse keeps its number of cells of each label.
    p = (1 + number of shuffles with a distance >= observed) / (1 + n_perm).
    """
    import numpy as np
    from scipy.stats import ks_2samp

    df = df[df[label].isin(groups)].dropna(subset=[value])
    values = df[value].to_numpy()
    labels = (df[label] == groups[0]).to_numpy()
    mice = df[mouse].to_numpy()

    def distance(is_first):
        return ks_2samp(values[is_first], values[~is_first]).statistic

    observed = distance(labels)
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    if shuffle == 'between_mice':
        mouse_ids, inverse = np.unique(mice, return_inverse=True)
        mouse_label = np.zeros(len(mouse_ids), dtype=bool)
        mouse_label[inverse[labels]] = True
        if np.any(mouse_label[inverse] != labels):
            raise ValueError(f'{label} is not constant within mice.')
        for k in range(n_perm):
            null[k] = distance(rng.permutation(mouse_label)[inverse])
    elif shuffle == 'within_mice':
        blocks = [np.flatnonzero(mice == m) for m in np.unique(mice)]
        shuffled = labels.copy()
        for k in range(n_perm):
            for idx in blocks:
                shuffled[idx] = rng.permutation(labels[idx])
            null[k] = distance(shuffled)
    else:
        raise ValueError(shuffle)
    return {
        'ks_statistic': observed,
        'p_value': (1 + np.sum(null >= observed - 1e-12)) / (1 + n_perm),
        'n_perm': n_perm,
        'shuffle': shuffle,
        'n_cells': (int(labels.sum()), int((~labels).sum())),
        'n_mice': len(np.unique(mice)),
    }
