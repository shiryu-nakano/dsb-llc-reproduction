"""
run_order_gram_one_class.py

任意次数(1〜6)・エントロピー項の有無を選べる汎用ワーカースクリプト。
deletion_order_gram.py の moment_order_sample_gram を使用。

保存先命名規則:
  no-entropy: moment_data/deletion/{order}_order_gram/_partial/class{c}.npz
              (order: first/second/third/fourth/fifth/sixth)
  entropy:    moment_data/deletion/{order}_order_koleo/_partial/class{c}.npz

使い方:
  CUDA_VISIBLE_DEVICES=0 python run_order_gram_one_class.py --order 5 --class_id 0
  CUDA_VISIBLE_DEVICES=0 python run_order_gram_one_class.py --order 2 --class_id 0 --entropy
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from cifar10_loader import load_cifar10
from deletion_order_gram import moment_order_sample_gram


ORDER_NAMES = {1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth", 6: "sixth"}


def method_name(order, entropy):
    base = f"{ORDER_NAMES[order]}_order"
    return f"{base}_koleo" if entropy else f"{base}_gram"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--order", type=int, required=True, choices=[1, 2, 3, 4, 5, 6])
    ap.add_argument("--class_id", type=int, required=True)
    ap.add_argument("--entropy", action="store_true", help="KoLeoエントロピー項を追加する")
    ap.add_argument("--n_steps", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--moment_weight", type=float, default=100.0,
                     help="3次以上の各moment項に掛ける重み(全次数共通)")
    ap.add_argument("--koleo_weight", type=float, default=1e-3)
    ap.add_argument("--rng_seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--log_every", type=int, default=300)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()

    method = method_name(args.order, args.entropy)
    default_dir = Path(f"/home/nakano/server/moment_data/deletion/{method}/_partial")
    PARTIAL_DIR = args.out_dir if args.out_dir is not None else default_dir
    PARTIAL_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PARTIAL_DIR / f"class{args.class_id}.npz"

    if out_path.exists() and not args.force:
        print(f"[{method} class {args.class_id}] already exists at {out_path}, skip (--force to overwrite)")
        return

    print(f"[{method} class {args.class_id}] === START {time.strftime('%H:%M:%S')} "
          f"order={args.order} entropy={args.entropy} n_steps={args.n_steps} ===")

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

    moment_weights = {k: args.moment_weight for k in range(3, args.order + 1)}

    t0 = time.time()
    samples = moment_order_sample_gram(
        x_class, n_c,
        order=args.order,
        rng_seed=args.rng_seed,
        n_steps=args.n_steps,
        lr=args.lr,
        moment_weights=moment_weights,
        use_entropy=args.entropy,
        koleo_weight=args.koleo_weight,
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