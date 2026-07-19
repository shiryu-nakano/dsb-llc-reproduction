"""
test_truncated_normal_scaling.py

Truncated normal samplingのサンプル数を変えて、
所要時間のスケーリングを確認する。
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import time
import pickle
import numpy as np
import torch

from deletion_truncated_normal import truncated_normal_sample


def main():
    STATS_PATH = Path("~/cuda_test/cifar10_class_stats.pkl").expanduser()

    print("Loading precomputed class statistics...")
    with open(STATS_PATH, "rb") as f:
        stats = pickle.load(f)
    means = stats["means"]
    shrunk_covs = stats["shrunk_covs"]

    target_mean = means[0]
    target_cov = shrunk_covs[0]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")

    rng = np.random.default_rng(42)

    for n_samples in [20, 100, 500, 1000]:
        print(f"--- n_samples={n_samples} ---")
        t0 = time.time()
        samples = truncated_normal_sample(
            n_samples, target_mean, target_cov, rng,
            n_iter=500,  # 本番設定
            device=device,
        )
        elapsed = time.time() - t0
        mean_diff = np.max(np.abs(samples.mean(axis=0) - target_mean))
        print(f"  elapsed: {elapsed:.2f}s | mean_diff: {mean_diff:.4f}")

        # 5000サンプル・10クラス分の見積もり
        est_per_class = elapsed * (5000 / n_samples) if n_samples < 5000 else elapsed
        est_total = est_per_class * 10
        print(f"  estimated time for 5000 samples/class: {est_per_class:.1f}s")
        print(f"  estimated total (10 classes): {est_total:.1f}s ({est_total/60:.1f} min)")
        print()


if __name__ == "__main__":
    main()
