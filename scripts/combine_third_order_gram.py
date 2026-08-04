"""
combine_third_order_gram.py

moment_data/deletion/third_order_gram/_partial/class{0..9}.npz を1本に結合し、
moment_data/deletion/third_order_gram/third_order_cifar10.npz に保存する。

Gram trick版(近似なし、coskew_weight=100)の最終出力。
既存の combine_third_order.py (旧ミニバッチ近似版用) とは別ファイル。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

BASE = Path("/home/nakano/server/moment_data/deletion/third_order_gram")
PARTIAL_DIR = BASE / "_partial"
OUT_PATH = BASE / "third_order_cifar10.npz"

NUM_CLASSES = 10


def main():
    missing = [c for c in range(NUM_CLASSES) if not (PARTIAL_DIR / f"class{c}.npz").exists()]
    if missing:
        print(f"ERROR: missing classes {missing} in {PARTIAL_DIR}.")
        sys.exit(1)

    all_pixels, all_orig, all_target, all_idx = [], [], [], []
    for c in range(NUM_CLASSES):
        d = np.load(PARTIAL_DIR / f"class{c}.npz")
        all_pixels.append(d["pixel_values"])
        all_orig.append(d["original_label"])
        all_target.append(d["target_label"])
        all_idx.append(d["sample_index"])
        print(f"  class {c}: {d['pixel_values'].shape[0]} samples, "
              f"range=[{d['pixel_values'].min():.4f},{d['pixel_values'].max():.4f}]")

    pixel_values = np.concatenate(all_pixels, axis=0)
    original_label = np.concatenate(all_orig, axis=0)
    target_label = np.concatenate(all_target, axis=0)
    sample_index = np.concatenate(all_idx, axis=0)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT_PATH,
        pixel_values=pixel_values,
        original_label=original_label,
        target_label=target_label,
        sample_index=sample_index,
    )
    print(f"\nSaved combined third_order (Gram, coskew_weight=100) data to {OUT_PATH}")
    print(f"total shape: {pixel_values.shape}, "
          f"range=[{pixel_values.min():.4f},{pixel_values.max():.4f}]")


if __name__ == "__main__":
    main()