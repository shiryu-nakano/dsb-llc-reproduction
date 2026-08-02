"""
test_third_order.py

third_order_sample の最小動作確認。
クラス0のみ、少数サンプル・少ないステップ数で試す。
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import time
import numpy as np
import torch

from cifar10_loader import load_cifar10
from deletion_third_order_fast import third_order_sample


def main():
    print("Loading CIFAR-10...")
    x_train, y_train, x_test, y_test = load_cifar10()

    H, W, C = x_train.shape[1:]
    D = H * W * C

    TEST_CLASS = 0
    N_FAKE = 20

    x_class = x_train[y_train == TEST_CLASS].reshape(-1, D)
    print(f"Class {TEST_CLASS}: {x_class.shape[0]} real samples, D={D}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    t0 = time.time()
    samples = third_order_sample(
        x_class, N_FAKE,
        n_steps=50,       # テスト用に少なめ
        slice_size=8,     # メモリ制御版スモーク
        device=device,
        verbose=True,
    )
    elapsed = time.time() - t0

    print(f"\nElapsed: {elapsed:.2f}s")
    print(f"shape: {samples.shape}")
    print(f"value range: [{samples.min():.4f}, {samples.max():.4f}]")

    target_mean = x_class.mean(axis=0)
    mean_diff = np.max(np.abs(samples.mean(axis=0) - target_mean))
    print(f"max |mean diff from target|: {mean_diff:.4f}")

    # 本番規模の見積もり (n_steps=1500, slice_size=1, D=3072)
    # スライス数 = D / slice_size なので、時間はほぼ (1500/50) * (D/4) 倍
    scale_steps = 1500 / 50
    scale_slices = (D / 1) / (D / 4)  # slice_size=1にすると4倍のスライス数
    est_total = elapsed * scale_steps * scale_slices
    print(f"\nEstimated time for full run (n_steps=1500, slice_size=1): "
          f"{est_total:.1f}s ({est_total/60:.1f} min) per class")
    print(f"Estimated total (10 classes): {est_total*10/60:.1f} min")


if __name__ == "__main__":
    main()
