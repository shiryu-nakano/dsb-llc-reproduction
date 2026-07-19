"""
moment_edit.py

CIFAR-10の各クラスの低次モーメント(平均・共分散)を計算し、
モーメント編集データ(grafting/deletion)を生成するスクリプト。

Step 1: クラスごとの平均・共分散の計算 (全手法共通の前処理)
"""
import numpy as np
from pathlib import Path
import pickle

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from train_and_save import load_cifar10

def compute_class_statistics(x, y, num_classes=10):
    """
    各クラスの平均・共分散を計算する。

    x: (N, 32, 32, 3) float32, [0,1]に正規化済み
    y: (N,) int, クラスラベル

    共分散は画像をフラット化した (32*32*3 = 3072,) ベクトルに対して計算する。
    3072次元の共分散行列は 3072x3072 = 約940万要素。
    """
    D = x.shape[1] * x.shape[2] * x.shape[3]  # 3072
    x_flat = x.reshape(x.shape[0], D)  # (N, 3072)

    means = np.zeros((num_classes, D), dtype=np.float64)
    covs  = np.zeros((num_classes, D, D), dtype=np.float64)

    for c in range(num_classes):
        mask = (y == c)
        x_c = x_flat[mask]  # (N_c, 3072)
        n_c = x_c.shape[0]
        print(f"Class {c}: {n_c} samples")

        mean_c = x_c.mean(axis=0)
        # 共分散: (N_c, D) -> (D, D)
        centered = x_c - mean_c
        cov_c = (centered.T @ centered) / (n_c - 1)

        means[c] = mean_c
        covs[c]  = cov_c

    return means, covs


def shrinkage_covariance(cov, n_samples, alpha=None):
    """
    Bodnar et al. (2014) の線形shrinkage推定。
    高次元(D=3072)で標本数(N_c=5000程度)が同程度かそれ以下の場合、
    標本共分散行列は数値的に不安定(ほぼ特異)になるため、
    単位行列方向に縮小することで安定化する。

    cov: (D, D) 標本共分散行列
    n_samples: 標本数 N_c
    alpha: shrinkage強度 (Noneなら簡易的な自動選択)
    """
    D = cov.shape[0]
    trace_cov = np.trace(cov)
    target = (trace_cov / D) * np.eye(D)  # 等方的な行列をターゲットにする

    if alpha is None:
        # 簡易的な自動選択: 標本数が少ないほど強く縮小する
        alpha = min(1.0, D / (n_samples + D))

    shrunk = (1 - alpha) * cov + alpha * target
    return shrunk, alpha


def main():
    OUT_PATH = Path("~/cuda_test/cifar10_class_stats.pkl").expanduser()

    print("Loading CIFAR-10...")
    x_train, y_train, x_test, y_test = load_cifar10()
    print(f"x_train shape: {x_train.shape}, dtype: {x_train.dtype}")
    print(f"value range: [{x_train.min():.3f}, {x_train.max():.3f}]")

    print("\nComputing per-class mean and covariance...")
    means, covs = compute_class_statistics(x_train, y_train, num_classes=10)

    print("\nApplying shrinkage to covariance matrices...")
    n_per_class = len(x_train) // 10  # CIFAR-10は各クラス5000枚
    shrunk_covs = np.zeros_like(covs)
    for c in range(10):
        shrunk_covs[c], alpha = shrinkage_covariance(covs[c], n_per_class)
        print(f"Class {c}: shrinkage alpha = {alpha:.4f}")

    # 保存
    with open(OUT_PATH, "wb") as f:
        pickle.dump({
            "means": means,
            "covs": covs,
            "shrunk_covs": shrunk_covs,
            "image_shape": x_train.shape[1:],  # (32, 32, 3)
        }, f)
    print(f"\nSaved to {OUT_PATH}")


if __name__ == "__main__":
    main()