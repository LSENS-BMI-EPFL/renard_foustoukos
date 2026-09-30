"""Statistics helpers shared by the figure scripts."""

import math


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
