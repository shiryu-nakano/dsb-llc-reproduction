"""
generate_deletion_data.py

Deletion(criterion 2)の4手法をまとめて実行し、
それぞれ moment_data/deletion/<method>/ に保存するスクリプト。
(3rd order hypercube は計算コストが高いため別スクリプトで独立実装する)

各手法とも、10クラスそれぞれについて5000サンプルを生成する。
Grafting とは異なり target_label = original_label
(そのクラスの低次統計だけ残したデータを、元クラスとして分類できるか)

保存形式:
  pixel_values:    (N, H, W, C)
  original_label:  (N,)
  target_label:    (N,)  = original_label と同一
  sample_index:    (N,)  元データセット内でのインデックス(実データ由来の手法のみ有効)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import time
import pickle
import numpy as np
import torch

from cifar10_loader import load_cifar10
from deletion_conrad import conrad_sample
from deletion_gaussian import gaussian_sample
from deletion_ics import ics_sample
from deletion_truncated_normal import truncated_normal_sample


def save_deletion_data(method_name, pixel_values, original_label, sample_index, H, W, C):
    """指定された手法の結果を moment_data/deletion/<method_name>/ に保存する"""
    out_path = (
        Path(__file__).resolve().parent.parent
        / "moment_data" / "deletion" / method_name / f"{method_name}_cifar10.npz"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pixel_values = pixel_values.reshape(-1, H, W, C).astype(np.float32)
    target_label = original_label.copy()  # deletionでは target = original

    np.savez_compressed(
        out_path,
        pixel_values=pixel_values,
        original_label=original_label,
        target_label=target_label,
        sample_index=sample_index,
    )
    print(f"  Saved to {out_path}")
    print(f"  shape: {pixel_values.shape}, value range: [{pixel_values.min():.3f}, {pixel_values.max():.3f}]")


def run_conrad(means, x_train, y_train, num_classes, H, W, C, rng):
    print("\n" + "=" * 60)
    print("1. Conrad sampling (1st order)")
    print("=" * 60)
    t_start = time.time()

    all_samples, all_labels, all_indices = [], [], []
    for c in range(num_classes):
        n_c = int((y_train == c).sum())
        source_indices = np.where(y_train == c)[0]

        t0 = time.time()
        samples, bs = conrad_sample(n_c, means[c], rng)
        print(f"  class {c}: {n_c} samples, done in {time.time()-t0:.2f}s "
              f"(b range: [{bs.min():.3f}, {bs.max():.3f}])")

        all_samples.append(samples)
        all_labels.append(np.full(n_c, c, dtype=np.int64))
        all_indices.append(source_indices)

    print(f"Total time: {time.time()-t_start:.1f}s")

    pixel_values = np.concatenate(all_samples, axis=0)
    original_label = np.concatenate(all_labels, axis=0)
    sample_index = np.concatenate(all_indices, axis=0)
    save_deletion_data("conrad", pixel_values, original_label, sample_index, H, W, C)


def run_gaussian(means, shrunk_covs, y_train, num_classes, H, W, C, rng):
    print("\n" + "=" * 60)
    print("2. Gaussian sampling (2nd order)")
    print("=" * 60)
    t_start = time.time()

    all_samples, all_labels, all_indices = [], [], []
    for c in range(num_classes):
        n_c = int((y_train == c).sum())
        source_indices = np.where(y_train == c)[0]

        t0 = time.time()
        samples = gaussian_sample(n_c, means[c], shrunk_covs[c], rng)
        print(f"  class {c}: {n_c} samples, done in {time.time()-t0:.2f}s")

        all_samples.append(samples)
        all_labels.append(np.full(n_c, c, dtype=np.int64))
        all_indices.append(source_indices)

    print(f"Total time: {time.time()-t_start:.1f}s")

    pixel_values = np.concatenate(all_samples, axis=0)
    original_label = np.concatenate(all_labels, axis=0)
    sample_index = np.concatenate(all_indices, axis=0)
    save_deletion_data("gaussian", pixel_values, original_label, sample_index, H, W, C)


def run_ics(x_train, y_train, num_classes, H, W, C, rng):
    print("\n" + "=" * 60)
    print("3. ICS (Independent Coordinate Sampling, 1st order alt.)")
    print("=" * 60)
    t_start = time.time()

    D = H * W * C
    all_samples, all_labels, all_indices = [], [], []
    for c in range(num_classes):
        mask = (y_train == c)
        x_class = x_train[mask].reshape(-1, D)
        n_c = x_class.shape[0]
        source_indices = np.where(mask)[0]

        t0 = time.time()
        samples = ics_sample(n_c, x_class, rng)
        print(f"  class {c}: {n_c} samples, done in {time.time()-t0:.2f}s")

        all_samples.append(samples)
        all_labels.append(np.full(n_c, c, dtype=np.int64))
        all_indices.append(source_indices)

    print(f"Total time: {time.time()-t_start:.1f}s")

    pixel_values = np.concatenate(all_samples, axis=0)
    original_label = np.concatenate(all_labels, axis=0)
    sample_index = np.concatenate(all_indices, axis=0)
    save_deletion_data("ics", pixel_values, original_label, sample_index, H, W, C)


def run_truncated_normal(means, shrunk_covs, y_train, num_classes, H, W, C, rng, device):
    print("\n" + "=" * 60)
    print("4. Truncated normal (2nd order, hypercube constrained)")
    print("=" * 60)
    print(f"  using device: {device}")
    t_start = time.time()

    all_samples, all_labels, all_indices = [], [], []
    for c in range(num_classes):
        n_c = int((y_train == c).sum())
        source_indices = np.where(y_train == c)[0]

        t0 = time.time()
        samples = truncated_normal_sample(
            n_c, means[c], shrunk_covs[c], rng,
            n_iter=500, device=device,
        )
        print(f"  class {c}: {n_c} samples, done in {time.time()-t0:.2f}s")

        all_samples.append(samples)
        all_labels.append(np.full(n_c, c, dtype=np.int64))
        all_indices.append(source_indices)

    print(f"Total time: {time.time()-t_start:.1f}s")

    pixel_values = np.concatenate(all_samples, axis=0)
    original_label = np.concatenate(all_labels, axis=0)
    sample_index = np.concatenate(all_indices, axis=0)
    save_deletion_data("truncated_normal", pixel_values, original_label, sample_index, H, W, C)


def main():
    STATS_PATH = Path("~/cuda_test/cifar10_class_stats.pkl").expanduser()

    print("Loading precomputed class statistics...")
    with open(STATS_PATH, "rb") as f:
        stats = pickle.load(f)
    means = stats["means"]
    shrunk_covs = stats["shrunk_covs"]
    H, W, C = stats["image_shape"]
    num_classes = means.shape[0]

    print("Loading CIFAR-10...")
    x_train, y_train, x_test, y_test = load_cifar10()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(42)

    # ── 4手法を順番に実行 ──
    run_conrad(means, x_train, y_train, num_classes, H, W, C, rng)
    run_gaussian(means, shrunk_covs, y_train, num_classes, H, W, C, rng)
    run_ics(x_train, y_train, num_classes, H, W, C, rng)
    run_truncated_normal(means, shrunk_covs, y_train, num_classes, H, W, C, rng, device)

    print("\n" + "=" * 60)
    print("All deletion methods completed.")
    print("=" * 60)


if __name__ == "__main__":
    main()
