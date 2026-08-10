"""
run_order_one_class_gram.py

1次(first_order_gram)・2次(second_order_gram)制約deletionデータを、
Gram trick(勾配降下、third_order_gramと同じ生成方式)で1クラスぶん
生成して保存するワーカースクリプト。

目的: 既存4手法(解析的サンプリング)とthird_order_gram(勾配降下)の
「生成方法の違い」がLLC比較に交絡していないか確認するため、
1次・2次も勾配降下方式(third_order_gramと同枠組み)で作る。

保存先:
  moment_data/deletion/first_order_gram/_partial/class{c}.npz
  moment_data/deletion/second_order_gram/_partial/class{c}.npz

使い方:
  CUDA_VISIBLE_DEVICES=0 python run_order_one_class_gram.py --order 1 --class_id 0
  CUDA_VISIBLE_DEVICES=0 python run_order_one_class_gram.py --order 2 --class_id 0
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from cifar10_loader import load_cifar10
from deletion_first_second_order_gram import order_sample_gram


def method_name(order):
    return {1: "first_order_gram", 2: "second_order_gram"}[order]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--order", type=int, required=True, choices=[1, 2])
    ap.add_argument("--class_id", type=int, required=True)
    ap.add_argument("--n_steps", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--rng_seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--log_every", type=int, default=300)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()

    method = method_name(args.order)
    default_dir = Path(f"/home/nakano/server/moment_data/deletion/{method}/_partial")
    PARTIAL_DIR = args.out_dir if args.out_dir is not None else default_dir
    PARTIAL_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PARTIAL_DIR / f"class{args.class_id}.npz"

    if out_path.exists() and not args.force:
        print(f"[{method} class {args.class_id}] already exists at {out_path}, skip (--force to overwrite)")
        return

    print(f"[{method} class {args.class_id}] === START {time.strftime('%H:%M:%S')} "
          f"n_steps={args.n_steps} order={args.order} (Gram trick, gradient-based) ===")

    t_load0 = time.time()
    x_train, y_train, _, _ = load_cifar10()
    H, W, C = x_train.shape[1:]
    D = H * W * C
    print(f"[{method} class {args.class_id}] data loaded in {time.time()-t_load0:.1f}s")

    mask = (y_train == args.class_id)
    x_class = x_train[mask].reshape(-1, D)
    n_c = x_class.shape[0]
    source_indices = np.where(mask)[0]
    print(f"[{method} class {args.class_id}] {n_c} real samples, D={D}, device={args.device}")

    t0 = time.time()
    samples = order_sample_gram(
        x_class, n_c,
        order=args.order,
        rng_seed=args.rng_seed,
        n_steps=args.n_steps,
        lr=args.lr,
        device=args.device,
        verbose=True,
        log_every=args.log_every,
        final_eval=True,
        desc=f"{method}_class{args.class_id}",
    )
    elapsed = time.time() - t0
    print(f"[{method} class {args.class_id}] optimization done in {elapsed:.1f}s ({elapsed/60:.1f} min)")

    samples = np.clip(samples, 0.0, 1.0)
    pixel_values = samples.reshape(-1, H, W, C).astype(np.float32)
    original_label = np.full(n_c, args.class_id, dtype=np.int64)
    target_label = original_label.copy()

    np.savez_compressed(
        out_path,
        pixel_values=pixel_values,
        original_label=original_label,
        target_label=target_label,
        sample_index=source_indices,
    )
    print(f"[{method} class {args.class_id}] saved to {out_path}")
    print(f"[{method} class {args.class_id}] shape={pixel_values.shape} "
          f"range=[{pixel_values.min():.4f},{pixel_values.max():.4f}]")
    print(f"[{method} class {args.class_id}] === END {time.strftime('%H:%M:%S')} "
          f"total_wall={time.time()-t_load0:.1f}s ({(time.time()-t_load0)/60:.1f} min) ===")


if __name__ == "__main__":
    main()