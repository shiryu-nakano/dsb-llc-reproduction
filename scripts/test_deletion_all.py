"""
test_deletion_all.py

Deletion手法(Conrad / Gaussian / ICS / Truncated normal)を
まとめて小規模テストするスクリプト。

各手法について、クラス0の統計量から少数サンプルを生成し、
以下を確認する:
  - エラーなく実行できるか
  - 生成サンプルの平均が目標(そのクラスの実際の平均)に近いか
  - 値域が[0,1]に収まっているか
  - 所要時間(本番規模への見積もり)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import time
import pickle
import numpy as np

from cifar10_loader import load_cifar10
from deletion_conrad import conrad_sample
from deletion_gaussian import gaussian_sample
from deletion_ics import ics_sample
from deletion_truncated_normal import truncated_normal_sample


def report(name, samples, target_mean, elapsed):
    mean_diff = np.max(np.abs(samples.mean(axis=0) - target_mean))
    print(f"  shape: {samples.shape}")
    print(f"  value range: [{samples.min():.4f}, {samples.max():.4f}]")
    print(f"  max |mean diff from target|: {mean_diff:.4f}")
    print(f"  elapsed: {elapsed:.3f}s")
    print()


def main():
    STATS_PATH = Path("~/cuda_test/cifar10_class_stats.pkl").expanduser()
    N_TEST = 20        # テスト用の少数サンプル数
    TEST_CLASS = 0

    print("Loading precomputed class statistics...")
    with open(STATS_PATH, "rb") as f:
        stats = pickle.load(f)
    means = stats["means"]
    shrunk_covs = stats["shrunk_covs"]
    H, W, C = stats["image_shape"]
    D = H * W * C

    print("Loading CIFAR-10 (for ICS raw data)...")
    x_train, y_train, x_test, y_test = load_cifar10()
    x_class = x_train[y_train == TEST_CLASS].reshape(-1, D)

    target_mean = means[TEST_CLASS]
    target_cov = shrunk_covs[TEST_CLASS]

    rng = np.random.default_rng(42)

    # ── 1. Conrad sampling ──
    print("=" * 60)
    print("1. Conrad sampling (1st order)")
    print("=" * 60)
    t0 = time.time()
    samples, bs = conrad_sample(N_TEST, target_mean, rng)
    elapsed = time.time() - t0
    report("conrad", samples, target_mean, elapsed)
    print(f"  b range: [{bs.min():.4f}, {bs.max():.4f}]")
    print()

    # ── 2. Gaussian sampling ──
    print("=" * 60)
    print("2. Gaussian sampling (2nd order)")
    print("=" * 60)
    t0 = time.time()
    samples = gaussian_sample(N_TEST, target_mean, target_cov, rng)
    elapsed = time.time() - t0
    report("gaussian", samples, target_mean, elapsed)

    # ── 3. ICS ──
    print("=" * 60)
    print("3. ICS (Independent Coordinate Sampling, 1st order alt.)")
    print("=" * 60)
    t0 = time.time()
    samples = ics_sample(N_TEST, x_class, rng)
    elapsed = time.time() - t0
    report("ics", samples, target_mean, elapsed)

    # ── 4. Truncated normal ──
    print("=" * 60)
    print("4. Truncated normal (2nd order, hypercube constrained)")
    print("=" * 60)
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  using device: {device}")
    t0 = time.time()
    samples = truncated_normal_sample(
        N_TEST, target_mean, target_cov, rng,
        n_iter=100,  # テスト用に少なめ
        device=device,
    )
    elapsed = time.time() - t0
    report("truncated_normal", samples, target_mean, elapsed)

    print("All tests completed.")


if __name__ == "__main__":
    main()
