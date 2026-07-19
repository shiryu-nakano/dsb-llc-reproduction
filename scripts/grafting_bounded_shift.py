"""
grafting_bounded_shift.py

CIFAR-10の各クラスを、他の全9クラスの平均へ bounded_shift で移植したデータを生成する。
Grafting・1次(bounded_shift)。
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from train_and_save import load_cifar10


def bounded_shift(x, target, bounds=(0.0, 1.0), max_iter=10_000, tol=1e-4, verbose=False):
    """
    x: (N, D) あるクラスの画像群(フラット化済み)
    target: (D,) 目標平均
    """
    mu = x.mean(axis=0)
    for i in range(max_iter):
        x = x - (mu - target)
        x = np.clip(x, *bounds)
        mu = x.mean(axis=0)
        if np.max(np.abs(mu - target)) < tol:
            if verbose:
                print(f"    Converged in {i} iterations")
            break
    return x


def main():
    OUT_PATH = (
        Path(__file__).resolve().parent.parent
        / "moment_data" / "grafting" / "bounded_shift" / "shifted_cifar10.npz"
    )
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("Loading CIFAR-10...")
    x_train, y_train, x_test, y_test = load_cifar10()

    num_classes = 10
    H, W, C = x_train.shape[1:]
    D = H * W * C

    # クラスごとにデータと平均を用意
    class_data = {}
    class_means = {}
    for c in range(num_classes):
        mask = (y_train == c)
        x_c = x_train[mask].reshape(-1, D)
        class_data[c] = x_c
        class_means[c] = x_c.mean(axis=0)
        print(f"Class {c}: {x_c.shape[0]} samples")

    all_pixel_values = []
    all_original_label = []
    all_target_label = []
    all_sample_index = []

    for source_class in range(num_classes):
        x_source = class_data[source_class]
        n_source = x_source.shape[0]
        source_indices = np.where(y_train == source_class)[0]

        for target_class in range(num_classes):
            if target_class == source_class:
                continue  # 自分自身への移植はスキップ

            print(f"Grafting class {source_class} -> class {target_class} ...")
            target_mu = class_means[target_class]

            shifted = bounded_shift(x_source.copy(), target_mu, verbose=False)

            all_pixel_values.append(shifted)
            all_original_label.append(np.full(n_source, source_class, dtype=np.int64))
            all_target_label.append(np.full(n_source, target_class, dtype=np.int64))
            all_sample_index.append(source_indices)

    pixel_values   = np.concatenate(all_pixel_values, axis=0).reshape(-1, H, W, C)
    original_label = np.concatenate(all_original_label, axis=0)
    target_label   = np.concatenate(all_target_label, axis=0)
    sample_index   = np.concatenate(all_sample_index, axis=0)

    print(f"\nTotal samples generated: {pixel_values.shape[0]}")
    print(f"  (expected: {num_classes} classes x {num_classes-1} targets x 5000 samples/class = {num_classes * (num_classes-1) * 5000})")

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