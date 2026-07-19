"""
deletion_conrad.py

Conrad sampling (Deletion, 1次: 平均のみ保持)
論文Appendix E: p(x) = b*exp(-bx+b)/(exp(b)-1)
"""
import numpy as np
from scipy.optimize import brentq


def _conrad_mean_given_b(b):
    if abs(b) < 1e-8:
        return 0.5
    return -(b - np.exp(b) + 1) / (b * (np.exp(b) - 1))


def _solve_b_for_mean(target_mean, tol=1e-10, b_range=(-50, 50)):
    if abs(target_mean - 0.5) < 1e-6:
        return 0.0

    def f(b):
        return _conrad_mean_given_b(b) - target_mean

    lo, hi = b_range
    try:
        b = brentq(f, lo, hi, xtol=tol)
    except ValueError:
        b = lo if f(lo) * f((lo + hi) / 2) < 0 else hi
    return b


def conrad_sample(n_samples, target_means, rng):
    """
    各次元ごとに独立に Conrad分布からサンプルする。
    target_means: (D,) 各次元の目標平均
    戻り値: (n_samples, D), bs: (D,) 各次元で解かれたb
    """
    D = len(target_means)
    samples = np.empty((n_samples, D), dtype=np.float32)
    bs = np.empty(D, dtype=np.float64)

    for d in range(D):
        b = _solve_b_for_mean(float(target_means[d]))
        bs[d] = b

        u = rng.uniform(0, 1, size=n_samples)
        if abs(b) < 1e-8:
            x = u
        else:
            eb = np.exp(b)
            x = 1.0 - np.log(eb - u * (eb - 1.0)) / b
        samples[:, d] = np.clip(x, 0.0, 1.0)

    return samples, bs
