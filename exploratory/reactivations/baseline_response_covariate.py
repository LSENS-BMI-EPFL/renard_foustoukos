"""
Reply to R2 public 4 (not in the paper): does the LMI-participation relation
of Fig. 6d hold when each cell's whisker response before learning is a
covariate?

Pre-learning response: mean dF/F 0-300 ms (baseline-subtracted tensors) over
the last 40 passive mapping trials of days -2 and -1 (as the LMI), z-scored
within reward group. Model: day-0 held-out participation above chance (main,
as Fig. 6d) or participation rate (2.5 x noise SD) ~ LMI + pre-learning
response + (1 + LMI | mouse). Prints the results.
"""

import warnings

import pandas as pd
from joblib import Parallel, delayed

from fast_learning import imaging, paths, participation
from fast_learning.stats import lmm_slope


def responses(mouse):
    """Mean response (0-300 ms) of each cell, before learning and on day 0."""
    x = imaging.last_trials(
        imaging.load_mouse_xarray(mouse, paths.tensor_dir, 'tensor_xarray_mapping_data.nc')
    )
    out = {}
    for name, days in [('pre_resp', [-2, -1]), ('day0_resp', [0])]:
        v = imaging.select_time(x.sel(trial=x['day'].isin(days)), 0, 0.3).mean(dim=['time', 'trial'])
        out[name] = v.values
    return pd.DataFrame({'mouse_id': mouse, 'roi': x['roi'].values, **out})


if __name__ == '__main__':
    warnings.simplefilter('ignore')
    rate = participation.load_participation(2.5, False, 'heldout')
    tables = {'above chance': participation.excess_tables(*rate)[0], 'rate': rate[0]}
    resp = pd.concat(Parallel(n_jobs=20)(delayed(responses)(m) for m in rate[0].mouse_id.unique()))
    for measure, merged in tables.items():
        print(f'--- participation {measure} ---')
        d = merged.merge(resp, on=['mouse_id', 'roi']).dropna(subset=['lmi', 'learning_rate', 'pre_resp'])
        d = d[d['reliable_learning']]
        for group, s in d.groupby('reward_group'):
            s = s.assign(pre_z=(s.pre_resp - s.pre_resp.mean()) / s.pre_resp.std())
            print(
                f"{group}: {len(s)} cells, corr(LMI, pre-learning response) = {s['lmi'].corr(s['pre_resp']):.2f}"
            )
            for label, cov in [('LMI alone', ()), ('LMI + pre-learning response', ('pre_z',))]:
                f = lmm_slope(s, 'learning_rate', 'lmi', covariates=cov)
                print(
                    f"  {label:30s} slope {f['slope']:.3f} [{f['ci_low']:.3f}, {f['ci_high']:.3f}], "
                    f"p={f['p_value']:.1e} ({f['method']})"
                )
            f = lmm_slope(s, 'learning_rate', 'pre_z')
            print(f"  pre-learning response alone: slope {f['slope']:.4f}, p={f['p_value']:.1e}")
