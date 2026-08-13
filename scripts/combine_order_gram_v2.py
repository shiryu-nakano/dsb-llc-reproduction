"""
combine_order_gram_v2.py

moment_data/deletion/{method}/_partial/class{0..9}.npz を1本に結合する。
method名は run_order_gram_one_class.py と同じ命名規則
({first,second,third,fourth,fifth,sixth}_order_{gram,koleo})。

使い方:
  python combine_order_gram_v2.py --order 5
  python combine_order_gram_v2.py --order 2 --entropy
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

ORDER_NAMES = {1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth", 6: "sixth"}


def method_name(order, entropy):
    base = f"{ORDER_NAMES[order]}_order"
    return f"{base}_koleo" if entropy else f"{base}_gram"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--order", type=int, required=True, choices=[1, 2, 3, 4, 5, 6])
    ap.add_argument("--entropy", action="store_true")
    ap.add_argument("--classes", type=int, nargs="+", default=list(range(10)))
    args = ap.parse_args()

    method = method_name(args.order, args.entropy)
    base = Path(f"/home/nakano/server/moment_data/deletion/{method}")
    partial_dir = base / "_partial"
    out_path = base / f"{method}_cifar10.npz"

    missing = [c for c in args.classes if not (partial_dir / f"class{c}.npz").exists()]
    if missing:
        print(f"ERROR: missing classes {missing} in {partial_dir}.")
        sys.exit(1)

    all_pixels, all_orig, all_target, all_idx = [], [], [], []
    for c in args.classes:
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