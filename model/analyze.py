"""
Statistical analysis routines for model comparison and diagnostics.
Includes paired bootstrap significance testing for Kendall's tau and other metrics.
"""
import numpy as np


def paired_bootstrap(a, b, n_boot=2000, seed=0, ci=0.95):
    """
    Performs paired bootstrap on two paired vectors of evaluation metrics (e.g. Kendall's taus
    across the same disruptions).
    Returns:
        d: float, observed mean difference (mean(a - b))
        lo: float, lower bound of bootstrap confidence interval
        hi: float, upper bound of bootstrap confidence interval
        p: float, two-sided empirical p-value for H0: mean(a - b) == 0
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    diff = a - b
    d = float(np.mean(diff))
    n = len(diff)
    if n == 0:
        return 0.0, 0.0, 0.0, 1.0

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, n, size=(n_boot, n))
    boot_means = diff[indices].mean(axis=1)

    alpha = (1.0 - ci) / 2.0
    lo = float(np.percentile(boot_means, 100.0 * alpha))
    hi = float(np.percentile(boot_means, 100.0 * (1.0 - alpha)))

    # Two-sided empirical p-value
    p_le = np.mean(boot_means <= 0)
    p_ge = np.mean(boot_means >= 0)
    p = float(2.0 * min(p_le, p_ge))
    p = min(1.0, max(0.0, p))

    return d, lo, hi, p


def compare_models(taus_dict, ref_name="ARDN_v2_full", n_boot=2000):
    """Compare a dictionary of model taus against a reference model."""
    if ref_name not in taus_dict:
        return {}
    ref = taus_dict[ref_name]
    results = {}
    for name, taus in taus_dict.items():
        if name == ref_name:
            continue
        d, lo, hi, p = paired_bootstrap(ref, taus, n_boot=n_boot)
        sig = bool(lo > 0 or hi < 0)
        results[name] = {"dtau": d, "ci": (lo, hi), "p_value": p, "significant": sig}
    return results
