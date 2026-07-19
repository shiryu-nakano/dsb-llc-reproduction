"""
grafting_cqn.py

CIFAR-10の各クラスを、他の全9クラスへ CQN (Coordinatewise Quantile
Normalization) で移植したデータを生成する。
Grafting・1次(CQN、bounded_shiftとは別法)。

各ピクセル位置ごとに独立して、元クラスの経験分布上の分位点を求め、
その分位点を目標クラスの経験分布の値に変換する(ヒストグラムマッチング)。
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import time
import numpy as np
from train_and_save import load_cifar10


def build_lut(x_class):
    """
    あるクラスの全サンプルから、各ピクセル位置ごとの
    ソート済みルックアップテーブル(LUT)を作る。

    x_class: (N_c, D) あるクラスの画像(フラット化済み)
    戻り値:  (D, N_c) 各次元ごとにソートされた値の配列
    """
    lut = np.sort(x_class, axis=0).T  # (D, N_c)
    return lut


def quantile_normalize(x, lut_source, lut_target):
    """
    x を lut_source の分位点構造から lut_target の分位点構造へ変換する。

    x:          (N, D) 変換したいサンプル群
    lut_source: (D, n_bins) 元クラスのLUT
    lut_target: (D, n_bins) 目標クラスのLUT
    """
    N, D = x.shape
    n_bins = lut_source.shape[1]
    x_transformed = np.empty_like(x)

    for d in range(D):
        indices = np.searchsorted(lut_source[d], x[:, d])
        indices = np.clip(indices, 0, n_bins - 1)
        x_transformed[:, d] = lut_target[d, indices]

    return x_transformed


def main():
    OUT_PATH = (
        Path(__file__).resolve().parent.parent
        / "moment_data" / "grafting" / "cqn" / "cqn_cifar10.npz"
    )
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("Loading CIFAR-10...")
    x_train, y_train, x_test, y_test = load_cifar10()

    num_classes = 10
    H, W, C = x_train.shape[1:]
    D = H * W * C

    # クラスごとにデータとLUTを用意 (LUT構築は1クラスにつき1回だけ)
    class_data = {}
    class_lut = {}
    for c in range(num_classes):
        mask = (y_train == c)
        x_c = x_train[mask].reshape(-1, D)
        class_data[c] = x_c
        t0 = time.time()
        class_lut[c] = build_lut(x_c)
        print(f"Class {c}: {x_c.shape[0]} samples, "
              f"LUT shape {class_lut[c].shape}, built in {time.time()-t0:.2f}s")

    all_pixel_values = []
    all_original_label = []
    all_target_label = []
    all_sample_index = []

    t_start = time.time()
    for source_class in range(num_classes):
        x_source = class_data[source_class]
        n_source = x_source.shape[0]
        source_indices = np.where(y_train == source_class)[0]
        lut_source = class_lut[source_class]

        for target_class in range(num_classes):
            if target_class == source_class:
                continue

            t0 = time.time()
            lut_target = class_lut[target_class]

            transformed = quantile_normalize(x_source, lut_source, lut_target)

            all_pixel_values.append(transformed)
            all_original_label.append(np.full(n_source, source_class, dtype=np.int64))
            all_target_label.append(np.full(n_source, target_class, dtype=np.int64))
            all_sample_index.append(source_indices)

            print(f"CQN class {source_class} -> class {target_class} "
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