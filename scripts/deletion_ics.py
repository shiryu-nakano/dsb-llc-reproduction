"""
deletion_ics.py

Independent Coordinate Sampling (Deletion, 1次別法: 周辺分布のみ保持)
Sklarの定理: コピュラを積(独立)に置き換える。
"""
import numpy as np


def ics_sample(n_samples, class_data, rng):
    """
    各次元ごとに、そのクラスの経験分布(周辺分布)から独立にサンプルする。
    class_data: (N_c, D) あるクラスの実データ
    """
    N_c, D = class_data.shape
    samples = np.empty((n_samples, D), dtype=np.float32)

    for d in range(D):
        idx = rng.integers(0, N_c, size=n_samples)
        samples[:, d] = class_data[idx, d]

    return samples
