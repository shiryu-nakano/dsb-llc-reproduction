"""
run_third_order_one_class.py

3次(coskewness制約)deletionデータを、指定した1クラスぶんだけ生成して
個別ファイルに保存するワーカースクリプト。
GPUごとにプロセスを分けて並列実行するための単位。

保存先: moment_data/deletion/third_order/_partial/class{c}.npz
  (他手法と同じ (N,H,W,C) float32 レイアウトに揃えた最終形で保存する)

使い方:
  CUDA_VISIBLE_DEVICES=4 python run_third_order_one_class.py --class_id 0
  CUDA_VISIBLE_DEVICES=5 python run_third_order_one_class.py --class_id 1
  ... (GPUごとに並列起動)

再開性: 既に対象クラスの _partial/class{c}.npz が存在すれば、
  --force を付けない限りスキップする(誤って上書き・再計算しないため)。
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from cifar10_loader import load_cifar10
from deletion_third_order_minibatch import third_order_sample


# 他の4手法(conrad/gaussian/ics/truncated_normal)と同じ NFS 共有先に固定する。
# __file__.parent.parent は ~/cuda_test/scripts/ の2つ上 = ~/cuda_test/ になって
# しまい、意図した /home/nakano/server/moment_data/ とは別物になるため、絶対パスで書く。
PARTIAL_DIR = Path("/home/nakano/server/moment_data/deletion/third_order/_partial")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--class_id", type=int, required=True, help="CIFAR-10クラスID (0-9)")
    ap.add_argument("--n_steps", type=int, default=1500)
    ap.add_argument("--slice_size", type=int, default=16)
    ap.add_argument("--coskew_batch", type=int, default=1000,
                     help="毎ステップ coskewness に使う fake サンプル数(ミニバッチ近似)")
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--rng_seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--force", action="store_true", help="既存の結果があっても上書き再計算する")
    args = ap.parse_args()

    PARTIAL_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PARTIAL_DIR / f"class{args.class_id}.npz"

    if out_path.exists() and not args.force:
        print(f"[class {args.class_id}] already exists at {out_path}, skip (--force to overwrite)")
        return

    print(f"[class {args.class_id}] === START {time.strftime('%H:%M:%S')} "
          f"n_steps={args.n_steps} coskew_batch={args.coskew_batch} "
          f"slice_size={args.slice_size} ===")

    t_load0 = time.time()
    print(f"[class {args.class_id}] loading CIFAR-10 ...")
    x_train, y_train, _, _ = load_cifar10()
    H, W, C = x_train.shape[1:]
    D = H * W * C
    print(f"[class {args.class_id}] data loaded in {time.time()-t_load0:.1f}s")

    mask = (y_train == args.class_id)
    x_class = x_train[mask].reshape(-1, D)
    n_c = x_class.shape[0]
    source_indices = np.where(mask)[0]
    print(f"[class {args.class_id}] {n_c} real samples, D={D}, device={args.device}")

    t0 = time.time()
    samples = third_order_sample(
        x_class, n_c,
        rng_seed=args.rng_seed,
        n_steps=args.n_steps,
        lr=args.lr,
        slice_size=args.slice_size,
        coskew_batch=args.coskew_batch,
        device=args.device,
        verbose=True,
        log_every=args.log_every,
        final_eval=True,
        desc=f"class{args.class_id}",
    )
    elapsed = time.time() - t0
    print(f"[class {args.class_id}] optimization done in {elapsed:.1f}s ({elapsed/60:.1f} min)")

    # bounds_loss はほぼ0だが、hypercube制約はソフト制約(ReLUペナルティ)なので
    # 数値誤差でわずかに [0,1] を外れることがある(実測: -0.0001 など)。
    # 他手法と同じく保存前に明示的にクリップしておく。
    samples = np.clip(samples, 0.0, 1.0)
    pixel_values = samples.reshape(-1, H, W, C).astype(np.float32)
    original_label = np.full(n_c, args.class_id, dtype=np.int64)
    target_label = original_label.copy()  # deletion: target == original

    np.savez_compressed(
        out_path,
        pixel_values=pixel_values,
        original_label=original_label,
        target_label=target_label,
        sample_index=source_indices,
    )
    print(f"[class {args.class_id}] saved to {out_path}")
    print(f"[class {args.class_id}] shape={pixel_values.shape} "
          f"range=[{pixel_values.min():.4f},{pixel_values.max():.4f}]")
    print(f"[class {args.class_id}] === END {time.strftime('%H:%M:%S')} "
          f"total_wall={time.time()-t_load0:.1f}s ({(time.time()-t_load0)/60:.1f} min) ===")


if __name__ == "__main__":
    main()