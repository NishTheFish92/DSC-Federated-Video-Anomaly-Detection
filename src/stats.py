"""Paired comparisons over seeds: t-test, confidence interval and equivalence test (TOST)."""
import numpy as np
from scipy import stats

EQUIVALENCE_MARGIN = 0.02  # AUC difference we treat as practically negligible


def paired_comparison(a, b, margin=EQUIVALENCE_MARGIN):
    """Compare b against a, run by run (same seed = same pair).

    Returns the mean difference b - a, its 95% confidence interval, the paired t-test
    p-value, and the TOST p-value for "the difference lies within +-margin". A small TOST
    p-value is evidence of equivalence; a large t-test p-value on its own is not.
    """
    d = np.asarray(b) - np.asarray(a)
    n, mean = len(d), d.mean()
    se = d.std(ddof=1) / np.sqrt(n)
    half = stats.t.ppf(0.975, n - 1) * se
    p_lower = stats.t.sf((mean + margin) / se, n - 1)  # H0: difference <= -margin
    p_upper = stats.t.cdf((mean - margin) / se, n - 1)  # H0: difference >= +margin
    return {
        "mean_diff": float(mean),
        "ci95": (float(mean - half), float(mean + half)),
        "p_ttest": float(stats.ttest_rel(b, a).pvalue),
        "p_tost": float(max(p_lower, p_upper)),
    }
