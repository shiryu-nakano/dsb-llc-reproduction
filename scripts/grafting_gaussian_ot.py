"""
grafting_gaussian_ot.py

CIFAR-10の各クラスを、他の全9クラスへ Gaussian Optimal Transport で
移植したデータを生成する。
Grafting・2次(Gaussian OT)。

concept_erasure.QuadraticFitter / QuadraticEditor をそのまま使用し、
features-across-time (train_vision.py) と数学的に完全に一致した
実装にする。

CIFAR10読み込みは jax非依存の cifar10_loader.py を使用する
(PyTorchとJAXのCUDAコンテキスト競合を避けるため)。

使用例:
  CUDA_VISIBLE_DEVICES=1 python scripts/grafting_gaussian_ot.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))  # cifar10_loader.py用

import time
import numpy as np
import torch

from concept_erasure import QuadraticFitter

from cifar10_loader import load_cifar10  # ← jax非依存版に変更


def main():
    OUT_PATH = (
        Path(__file__).resolve().parent.parent
        / "moment_data" / "grafting" / "gaussian_ot" / "gaussian_ot_cifar10.npz"
    )
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Loading CIFAR-10...")
    x_train, y_train, x_test, y_test = load_cifar10()

    num_classes = 10
    H, W, C = x_train.shape[1:]
    D = H * W * C

    X = torch.from_numpy(x_train.reshape(-1, D)).float().to(device)
    Y = torch.from_numpy(y_train.astype(np.int64)).to(device)

    print("Computing statistics (QuadraticFitter.fit)...")
    t0 = time.time()
    fitter = QuadraticFitter.fit(X, Y)
    print(f"  done in {time.time()-t0:.1f}s")

    print("Computing optimal transport maps (fitter.editor)...")
    t0 = time.time()
    editor = fitter.editor("cpu")
    print(f"  done in {time.time()-t0:.1f}s")

    all_pixel_values = []
    all_original_label = []
    all_target_label = []
    all_sample_index = []

    t_start = time.time()
    for source_class in range(num_classes):
        mask_np = (y_train == source_class)
        mask = torch.from_numpy(mask_np).to(device)
        x_source = X[mask]
        n_source = x_source.shape[0]
        source_indices = np.where(mask_np)[0]

        for target_class in range(num_classes):
            if target_class == source_class:
                continue

            t0 = time.time()
            transformed = editor.transport(
                x_source.cpu(), source_class, target_class
            )
            transformed = transformed.clamp(0.0, 1.0)

            all_pixel_values.append(transformed.numpy())
            all_original_label.append(np.full(n_source, source_class, dtype=np.int64))
            all_target_label.append(np.full(n_source, target_class, dtype=np.int64))
            all_sample_index.append(source_indices)

            print(f"Gaussian OT class {source_class} -> class {target_class} "
                  f"done in {time.time()-t0:.2f}s")

    print(f"\nTotal transform time: {time.time()-t_start:.1f}s")

    pixel_values   = np.concatenate(all_pixel_values, axis=0).reshape(-1, H, W, C)
    original_label = np.concatenate(all_original_label, axis=0)
    target_label   = np.concatenate(all_target_label, axis=0)
    sample_index   = np.concatenate(all_sample_index, axis=0)

    print(f"\nTotal samples generated: {pixel_values.shape[0]}")
    print(f"  (expected: {num_classes} classes x {num_classes-1} targets x 5000 samples/class = {num_classes * (num_classes-1) * 5000})")
    print(f"Value range: [{pixel_values.min():.3f}, {pixel_values.max():.3f}]")

    np.savez_compressed(
        OUT_PATH,
        pixel_values=pixel_values.astype(np.float32),
        original_label=original_label,
        target_label=target_label,
        sample_index=sample_index,
    )
    print(f"Saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
