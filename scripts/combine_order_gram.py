"""
combine_order_gram.py

moment_data/deletion/{first_order_gram,second_order_gram}/_partial/class{0..9}.npz
を1本に結合する。

使い方:
  python combine_order_gram.py --order 1
  python combine_order_gram.py --order 2
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

NUM_CLASSES = 10


def method_name(order):
    return {1: "first_order_gram", 2: "second_order_gram"}[order]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--order", type=int, required=True, choices=[1, 2])
    ap.add_argument("--classes", type=int, nargs="+", default=None,
                     help="結合対象クラスID。省略時は0-9全部(本番用)。"
                          "テストで一部クラスしか生成していない場合はここで指定する。")
    args = ap.parse_args()

    method = method_name(args.order)
    base = Path(f"/home/nakano/server/moment_data/deletion/{method}")
    partial_dir = base / "_partial"
    target_classes = args.classes if args.classes is not None else list(range(NUM_CLASSES))
    out_path = base / f"{method}_cifar10.npz"

    missing = [c for c in target_classes if not (partial_dir / f"class{c}.npz").exists()]
    if missing:
        print(f"ERROR: missing classes {missing} in {partial_dir}.")
        sys.exit(1)

    all_pixels, all_orig, all_target, all_idx = [], [], [], []
    for c in target_classes:
        d = np.load(partial_dir / f"class{c}.npz")
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

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        pixel_values=pixel_values,
        original_label=original_label,
        target_label=target_label,
        sample_index=sample_index,
    )
    print(f"\nSaved combined {method} data to {out_path}")
    print(f"total shape: {pixel_values.shape}, "
          f"range=[{pixel_values.min():.4f},{pixel_values.max():.4f}]")


if __name__ == "__main__":
    main()