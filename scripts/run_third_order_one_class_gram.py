"""
run_third_order_one_class_gram.py

3次(coskewness制約)deletionデータを、Gram trick(近似なし・高速)で
1クラスぶん生成して保存するワーカースクリプト。

これまでの run_third_order_one_class.py (ミニバッチ近似版、
deletion_third_order_minibatch.py 使用) とは別の、確定版パイプライン。
Gram trick により d^3 テンソルを構成せず厳密計算するため、近似は一切ない。
sweep_coskew_weight.py での検証の結果、coskew_weight=100 を標準値として採用。

保存先: moment_data/deletion/third_order_gram/_partial/class{c}.npz
  (旧ミニバッチ版の出力 moment_data/deletion/third_order/ とは別ディレクトリに
   保存し、混同を避ける。最終的にこちらを正式な third_order データとして
   moment_data/deletion/third_order/ に置き換える想定。)

使い方:
  CUDA_VISIBLE_DEVICES=0 python run_third_order_one_class_gram.py --class_id 0
  CUDA_VISIBLE_DEVICES=1 python run_third_order_one_class_gram.py --class_id 1
  ... (GPUごとに並列起動。Gram trick は1クラス約1分なので直列でも問題ない)

再開性: 既に対象クラスの _partial/class{c}.npz が存在すれば、
  --force を付けない限りスキップする。
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from cifar10_loader import load_cifar10
from deletion_third_order_gram import third_order_sample_gram


PARTIAL_DIR_DEFAULT = Path(
    "/home/nakano/server/moment_data/deletion/third_order_gram/_partial"
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--class_id", type=int, required=True, help="CIFAR-10クラスID (0-9)")
    ap.add_argument("--n_steps", type=int, default=1500)
    ap.add_argument("--coskew_weight", type=float, default=100.0,
                     help="sweep_coskew_weight.py での検証結果に基づく標準値=100")
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--rng_seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--force", action="store_true", help="既存の結果があっても上書き再計算する")
    ap.add_argument("--out-dir", type=Path, default=None,
                     help="保存先ディレクトリを上書き。指定しなければ既定の"
                          "_partial/ (third_order_gram) に保存する。")
    args = ap.parse_args()

    PARTIAL_DIR = args.out_dir if args.out_dir is not None else PARTIAL_DIR_DEFAULT
    PARTIAL_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PARTIAL_DIR / f"class{args.class_id}.npz"

    if out_path.exists() and not args.force:
        print(f"[class {args.class_id}] already exists at {out_path}, skip (--force to overwrite)")
        return

    print(f"[class {args.class_id}] === START {time.strftime('%H:%M:%S')} "
          f"n_steps={args.n_steps} coskew_weight={args.coskew_weight} (Gram trick, no approx) ===")

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
    samples = third_order_sample_gram(
        x_class, n_c,
        rng_seed=args.rng_seed,
        n_steps=args.n_steps,
        lr=args.lr,
        coskew_weight=args.coskew_weight,
        device=args.device,
        verbose=True,
        log_every=args.log_every,
        final_eval=True,
        desc=f"class{args.class_id}",
    )
    elapsed = time.time() - t0
    print(f"[class {args.class_id}] optimization done in {elapsed:.1f}s ({elapsed/60:.1f} min)")

    # hypercube制約はソフトペナルティなので、保存前に明示的にクリップする
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