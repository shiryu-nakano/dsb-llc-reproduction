"""
deletion_gaussian.py

Gaussian sampling (Deletion, 2次: 平均+共分散を保持)
最大エントロピー分布 = ガウス分布。
"""
import numpy as np


def gaussian_sample(n_samples, mean, cov, rng, clip=True):
    """
    平均・共分散が既知の多変量正規分布からサンプルする。
    mean: (D,)
    cov:  (D, D)
    """
    D = len(mean)
    try:
        L = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        eigvals, eigvecs = np.linalg.eigh(cov)
        eigvals = np.clip(eigvals, 1e-8, None)
        cov_fixed = eigvecs @ np.diag(eigvals) @ eigvecs.T
        L = np.linalg.cholesky(cov_fixed)

    z = rng.standard_normal(size=(n_samples, D)).astype(np.float32)
    samples = mean[None, :] + z @ L.T

    if clip:
        samples = np.clip(samples, 0.0, 1.0)
    return samples
